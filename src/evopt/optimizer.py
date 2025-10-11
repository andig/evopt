from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pulp


@dataclass
class OptimizationStrategy:
    charging_strategy: str
    discharging_strategy: str


@dataclass
class GridConfig:
    p_max_imp: float
    p_max_exp: float
    prc_p_exc_imp: float


@dataclass
class BatteryConfig:
    charge_from_grid: bool
    discharge_to_grid: bool
    s_min: float
    s_max: float
    s_initial: float
    c_min: float
    c_max: float
    d_max: float
    p_a: float
    p_demand: Optional[List[float]] = None  # Minimum charge demand (Wh)
    s_goal: Optional[List[float]] = None  # Goal state of charge (Wh)


@dataclass
class TimeSeriesData:
    dt: List[int]  # time step length [s]
    gt: List[float]  # Required total energy [Wh]
    ft: List[float]  # Forecasted production [Wh]
    p_N: List[float]  # Import prices [currency unit/Wh]
    p_E: List[float]  # Export prices [currency unit/Wh]


class Optimizer:
    """
    Optimizer class building the MILP model from the input data, and provides
    solve() function to run optimization and return the results
    """

    def __init__(self, strategy: OptimizationStrategy, grid: GridConfig, batteries: List[BatteryConfig], time_series: TimeSeriesData,
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6):
        """
        Constructor
        """
        self.strategy = strategy
        self.grid = grid
        self.batteries = batteries
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        # number of time steps
        self.T = len(time_series.gt)
        # time step range
        self.time_steps = range(self.T)
        # the optimization problem
        self.problem = None
        # dictionary of optimizer variables
        self.variables = {}

        # Compute scaling for strategy control parameters
        self.min_import_price = np.min(self.time_series.p_N)
        self.max_import_price = np.max(self.time_series.p_N)

        # scaling for penalty parameters. Make sure goal_penalty is always positive
        self.goal_penalty_energy = np.min([self.max_import_price, 0.1e-3]) * 10e1
        self.goal_penalty_power = np.min([self.max_import_price, 0.1e-3]) * np.max(self.time_series.dt) / 3600 * 10e1

        # penalty for exceeding grid import limit. Result shall not become infeasible but report the violation
        # with helpful information
        self.grid_p_imp_pen = np.min([self.max_import_price, 0.1e-3]) * np.max(self.time_series.dt) / 3600 * 10e1
        # penalty for exceeding the grid export limit. Result shall not become infeasible but report the 'lost'
        # solar power
        self.grid_p_exp_pen = np.min([self.max_import_price, 0.1e-3]) * np.max(self.time_series.dt) / 3600 * 10e1

        # if there is a demand rate given in the input, the grid import limit will be interpreted as the
        # threshold beyond wich the demand rate is to be applied. Compute a demand rate flag for use in the
        # build constraint and build objective methods.
        self.is_grid_demand_rate_active = False
        if self.grid.p_max_imp is not None and self.grid.prc_p_exc_imp is not None:
            self.is_grid_demand_rate_active = True

    def create_model(self):
        """
        Create and initialize the MILP model
        """

        # Create problem
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)

        self._setup_variables()
        self._setup_target_function()
        self._add_constraints()

    def _setup_variables(self):
        """
        Set up the variables of the MILP optimizer
        """

        # Charging power variables [Wh]
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                pulp.LpVariable(f"c_{i}_{t}", lowBound=0, upBound=bat.c_max * self.time_series.dt[t] / 3600.)
                for t in self.time_steps
            ]

        # Discharging power variables [Wh]
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max * self.time_series.dt[t] / 3600.)
                for t in self.time_steps
            ]

        # State of charge variables [Wh]
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                pulp.LpVariable(f"s_{i}_{t}", lowBound=bat.s_min, upBound=bat.s_max)
                for t in self.time_steps
            ]

        # penalty variable for not reaching given charge goals
        # variables are kept in a matrix Batteries X time steps, only those elements will have an
        # entry != None that have a SOC goal > 0 defined in the input data
        self.variables['s_goal_pen'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        self.variables['s_goal_pen'][i][t] = pulp.LpVariable(f"s_goal_pen_{i}_{t}", lowBound=0)

        # penalty variable for not being able to charge with the required power
        self.variables['p_demand_pen'] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if bat.p_demand is not None:
                for t in self.time_steps:
                    self.variables['p_demand_pen'][i][t] = pulp.LpVariable(f"p_demand_pen_{i}_{t}", lowBound=0)

        # Grid import/export variables [Wh]
        self.variables['n'] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in self.time_steps]
        self.variables['e'] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in self.time_steps]

        # penalty variables for exceeding grid power limits
        # for grid import
        if self.grid.p_max_imp is not None:
            self.variables['p_imp_pen'] = [pulp.LpVariable(f"p_imp_pen_{t}", lowBound=0) for t in self.time_steps]

        # for grid export
        if self.grid.p_max_exp is not None:
            self.variables['p_exp_pen'] = [pulp.LpVariable(f"p_exp_pen_{t}", lowBound=0) for t in self.time_steps]

        # for demand rate calculation, we need to track the maximum import power
        if self.is_grid_demand_rate_active:
            self.variables['p_max_imp_exc'] = pulp.LpVariable("p_max_imp_exc", lowBound=0)

        # Binary variable: power flow direction to / from grid variables
        # these variables
        # 1. avoid direct export from import if export remuneration is greater than import cost
        # 2. control grid charging to batteries and grid export from batteries acc. to configuration
        self.variables['y'] = []
        for t in self.time_steps:
            self.variables['y'].append(pulp.LpVariable(f"y_{t}", cat='Binary'))

        # Binary variable for charging activation
        self.variables['z_c'] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables['z_c'][i] = [
                    pulp.LpVariable(f"z_c_{i}_{t}", cat='Binary')
                    for t in self.time_steps
                ]
            else:
                self.variables['z_c'][i] = None

        # Binary variable to lock charging against discharging
        self.variables['z_cd'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['z_cd'][i] = [
                pulp.LpVariable(f"z_cd_{i}_{t}", cat='Binary')
                for t in self.time_steps
            ]

    def _setup_target_function(self):
        """
        Gather all target function contributions and instantiate the objective
        """

        # Objective function (1): Maximize economic benefit
        objective = 0

        # Grid import cost (negative because we want to minimize cost) [currency unit]
        for t in self.time_steps:
            objective -= self.variables['n'][t] * self.time_series.p_N[t]

        # Grid export revenue [currency unit]
        for t in self.time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t]

        # Final state of charge value [currency unit]
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a

        # charge for import power demand rate
        if self.is_grid_demand_rate_active:
            objective += - self.grid.prc_p_exc_imp * self.variables['p_max_imp_exc']

        ############################################################################
        # Penalties for goals that cannot be met
        for i, bat in enumerate(self.batteries):
            # unmet battery charging goals
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        # negative target function contribution in a maximizing optimization
                        objective += - self.goal_penalty_energy * self.variables['s_goal_pen'][i][t]
            # unmet charging demand due to battery reaching maximum SOC
            if bat.p_demand is not None:
                for t in self.time_steps:
                    objective += - self.goal_penalty_power \
                        * self.variables['p_demand_pen'][i][t] \
                        * (1 + (self.T - t)/self.T)

        # penalties for grid power limits that cannot be met.
        for t in self.time_steps:

            # penalty for exceeding the given import limit
            if self.grid.p_max_imp is not None:
                # negative target function contribution in a maximizing optimization
                objective += - self.grid_p_imp_pen * self.variables['p_imp_pen'][t]

            # penalty for exceeding the grid export limit
            if self.grid.p_max_exp is not None:
                # negative target function contribution in a maximizing optimization
                objective += - self.grid_p_exp_pen * self.variables['p_exp_pen'][t]

        #############################################################################
        # Secondary strategies to implement preferences without impact to actual cost

        # prefer charging first, then grid export
        if self.strategy.charging_strategy == 'charge_before_export':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables['c'][i][t] * self.min_import_price * 1.5e-5 * (self.T - t)

        # prefer charging at high solar production times to unload public grid from peaks
        if self.strategy.charging_strategy == 'attenuate_grid_peaks':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables['c'][i][t] * self.time_series.ft[t] * self.min_import_price * 1e-6

        # prefer discharging batteries completely before importing from grid
        if self.strategy.discharging_strategy == 'discharge_before_import':
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables['d'][i][t] * self.min_import_price * 5e-6 * (self.T - t)

        self.problem += objective

    def _add_constraints(self):
        """
        Add all constraints to the model
        """

        self.time_steps = range(self.T)

        # Constraint (2): Power balance for each time step
        for t in self.time_steps:
            # battery charge + discharge balance
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += (- self.variables['c'][i][t]
                                          + self.variables['d'][i][t])

            # grid import: if there is an import power limit, the power exceeding the limit
            # is going to the penalty variable.
            p_grid_imp = self.variables['n'][t]
            if self.grid.p_max_imp is not None:
                if self.is_grid_demand_rate_active:
                    # demand rate calculation
                    p_grid_imp = self.variables['n'][t]+self.variables['p_imp_pen'][t]
                else:
                    # grid import power limit
                    p_grid_imp = self.variables['n'][t]+self.variables['p_imp_pen'][t]

            # grid export: if there is a limit, the power exceeding the limit
            # is going to the penalty variable
            p_grid_exp = self.variables['e'][t]
            if self.grid.p_max_exp is not None:
                p_grid_exp = self.variables['e'][t]+self.variables['p_exp_pen'][t]

            self.problem += (battery_net_discharge
                             + self.time_series.ft[t]
                             + p_grid_imp
                             == p_grid_exp
                             + self.time_series.gt[t])

        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(self.time_steps) > 0:
                self.problem += (self.variables['s'][i][0]
                                 == bat.s_initial
                                 + self.eta_c * self.variables['c'][i][0]
                                 - (1 / self.eta_d) * self.variables['d'][i][0])

            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (self.variables['s'][i][t]
                                 == self.variables['s'][i][t - 1]
                                 + self.eta_c * self.variables['c'][i][t]
                                 - (1 / self.eta_d) * self.variables['d'][i][t])

            # Constraint (6): Battery SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += (self.variables['s'][i][t]
                                         + self.variables['s_goal_pen'][i][t] >= bat.s_goal[t])

            # Constraint: Minimum battery charge demand (for t > 0)
            if bat.p_demand is not None:
                for t in self.time_steps:
                    if bat.p_demand[t] > 0:
                        # clip required charge to max charging power if needed
                        # and leave some air to breathe for the optimizer
                        p_demand = bat.p_demand[t]
                        if p_demand >= bat.c_max * self.time_series.dt[t] / 3600.:
                            p_demand = bat.c_max * self.time_series.dt[t] / 3600. * 0.999
                        self.problem += (self.variables['c'][i][t] + self.variables['p_demand_pen'][i][t] >= p_demand)
                    elif bat.c_min > 0:
                        # in time steps without given charging demand, apply normal lower bound:
                        # Lower bound: either 0 or at least c_min
                        self.problem += (self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.
                                         * self.variables['z_c'][i][t])
                        self.problem += (self.variables['c'][i][t] <= self.M * self.variables['z_c'][i][t])

            # Constraint (7): Minimum charge power limits if there is not charge demand
            elif bat.c_min > 0:
                for t in self.time_steps:
                    # Lower bound: either 0 or at least c_min
                    self.problem += (self.variables['c'][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.
                                     * self.variables['z_c'][i][t])
                    self.problem += (self.variables['c'][i][t] <= self.M * self.variables['z_c'][i][t])

            # control battery charging from grid
            if not bat.charge_from_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['c'][i][t] <= self.M * self.variables['y'][t])

            # control battery discharging to grid
            if not bat.discharge_to_grid:
                for t in self.time_steps:
                    self.problem += (self.variables['d'][i][t] <= self.M * (1 - self.variables['y'][t]))

            # lock charging against discharging
            for t in self.time_steps:
                # Discharge constraint
                self.problem += self.variables['d'][i][t] <= self.M * self.variables['z_cd'][i][t]
                # Charge constraint
                self.problem += self.variables['c'][i][t] <= self.M * (1 - self.variables['z_cd'][i][t])

        # Constraints (4)-(5): Grid flow direction
        for t in self.time_steps:
            # Export constraint
            self.problem += self.variables['e'][t] <= self.M * self.variables['y'][t]
            # Import constraint
            self.problem += self.variables['n'][t] <= self.M * (1 - self.variables['y'][t])

        # limit regular grid import power
        if self.grid.p_max_imp is not None:
            if self.is_grid_demand_rate_active:
                # limit the demand rate free portion of the power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600
            else:
                # limit the actual import power
                for t in self.time_steps:
                    self.problem += self.variables['n'][t] <= self.grid.p_max_imp * self.time_series.dt[t] / 3600

        # limit regular grid export power
        if self.grid.p_max_exp is not None:
            for t in self.time_steps:
                self.problem += self.variables['e'][t] <= self.grid.p_max_exp * self.time_series.dt[t] / 3600

        # if demand rate is applied, the maximum grid import power value
        # of all time steps drives the demand rate charge
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                self.problem += self.variables['p_imp_pen'][t] <= self.variables['p_max_imp_exc']

    def solve(self) -> Dict:
        """
        Creates the MILP model if none exists and solves the optimization problem.
        Returns a dictionary with the optimization results
        """

        if self.problem is None:
            self.create_model()

        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        self.problem.solve(solver)

        # Extract results
        status = pulp.LpStatus[self.problem.status]

        # grid import and export if no demand rate is active
        # if a limit is set and exceeded, this is the part that is actually imported / exported.
        # the exceeding portion is captured in 'p_imp_pen' and / or 'p_exp_pen'
        grid_import = [pulp.value(var) for var in self.variables['n']]
        grid_export = [pulp.value(var) for var in self.variables['e']]
        # if a demand rate is active, the actual import power is both parts, 'n' and 'p_imp_pen'
        if self.is_grid_demand_rate_active:
            for t in self.time_steps:
                grid_import[t] += pulp.value(self.variables['p_imp_pen'][t])

        # get limit violations
        # grid import limit
        grid_imp_limit_violated = False
        grid_imp_overshoot = []
        if self.grid.p_max_imp is not None and not self.is_grid_demand_rate_active:
            grid_imp_limit_violated = (np.max([pulp.value(var) for var in self.variables['p_imp_pen']]) > 0)
            grid_imp_overshoot = [pulp.value(var) for var in self.variables['p_imp_pen']]
        # grid export limit
        grid_exp_limit_hit = False
        grid_exp_overshoot = []
        if self.grid.p_max_exp is not None:
            grid_exp_limit_hit = (np.max([pulp.value(var) for var in self.variables['p_exp_pen']]) > 0)
            grid_exp_overshoot = [pulp.value(var) for var in self.variables['p_exp_pen']]

        if status == 'Optimal':
            result = {
                'status': status,
                'objective_value': pulp.value(self.problem.objective),
                'limit_violations': {
                    'grid_import_limit_exceeded': grid_imp_limit_violated,
                    'grid_export_limit_hit': grid_exp_limit_hit
                },
                'batteries': [],
                'grid_import': grid_import,
                'grid_export': grid_export,
                'flow_direction': [],
                'grid_import_overshoot': grid_imp_overshoot,
                'grid_export_overshoot': grid_exp_overshoot
            }

            # Extract battery results
            for i, bat in enumerate(self.batteries):
                battery_result = {
                    'charging_power': [pulp.value(var) for var in self.variables['c'][i]],
                    'discharging_power': [pulp.value(var) for var in self.variables['d'][i]],
                    'state_of_charge': [pulp.value(var) for var in self.variables['s'][i]]
                }
                result['batteries'].append(battery_result)

            # Extract flow direction
            for y_var in self.variables['y']:
                if y_var is not None:
                    result['flow_direction'].append(int(pulp.value(y_var)))
                else:
                    result['flow_direction'].append(0)  # Default to import when constraint not active

            return result
        else:
            return {
                'status': status,
                'objective_value': None,
                'limit_violations': {
                    'grid_import_limit_exceeded': False,
                    'grid_export_limit_hit': False
                },
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': [],
                'grid_import_overshoot': [],
                'grid_export_overshoot': []
            }
