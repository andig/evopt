from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pulp


@dataclass
class OptimizationStrategy:
    charging_strategy: str
    discharging_strategy: str


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

    def __init__(
        self,
        strategy: OptimizationStrategy,
        batteries: List[BatteryConfig],
        time_series: TimeSeriesData,
        eta_c: float = 0.95,
        eta_d: float = 0.95,
        M: float = 1e6,
    ):
        """
        Constructor
        """

        self.strategy = strategy
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
        # Compute scaling parameters
        self.min_import_price = np.min(self.time_series.p_N)
        self.max_import_price = np.max(self.time_series.p_N)
        # make sure goal_penalty is always positive
        self.goal_penalty_energy = np.min([self.max_import_price, 0.1e-3]) * 10e1
        self.goal_penalty_power = np.min([self.max_import_price, 0.1e-3]) * np.max(self.time_series.dt) / 3600 * 10e1

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
        Set up the variables of the milp optimizer
        """

        # Charging power variables [Wh]
        self.variables["c"] = {}
        for i, bat in enumerate(self.batteries):
            self.variables["c"][i] = [pulp.LpVariable(f"c_{i}_{t}", lowBound=0, upBound=bat.c_max * self.time_series.dt[t] / 3600.0) for t in self.time_steps]

        # Discharging power variables [Wh]
        self.variables["d"] = {}
        for i, bat in enumerate(self.batteries):
            self.variables["d"][i] = [pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max * self.time_series.dt[t] / 3600.0) for t in self.time_steps]

        # State of charge variables [Wh]
        self.variables["s"] = {}
        for i, bat in enumerate(self.batteries):
            self.variables["s"][i] = [pulp.LpVariable(f"s_{i}_{t}", lowBound=bat.s_min, upBound=bat.s_max) for t in self.time_steps]

        # penalty variable for not reaching given charge goals
        # variables are kept in a matrix Batteries X Timesteps, only those elements will have an
        # entry != None that have a SOC goal > 0 defined in the input data
        self.variables["s_goal_pen"] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        self.variables["s_goal_pen"][i][t] = pulp.LpVariable(f"s_goal_pen_{i}_{t}", lowBound=0)

        # penalty variable for not being able to charge with the required power
        self.variables["p_demand_pen"] = [[None for t in self.time_steps] for i in range(len(self.batteries))]
        for i, bat in enumerate(self.batteries):
            if bat.p_demand is not None:
                for t in self.time_steps:
                    self.variables["p_demand_pen"][i][t] = pulp.LpVariable(f"p_demand_pen_{i}_{t}", lowBound=0)

        # Grid import/export variables [Wh]
        self.variables["n"] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in self.time_steps]
        self.variables["e"] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in self.time_steps]

        # Binary variable: power flow direction to / from grid variables
        # these variables
        # 1. avoid direct export from import if export remuneration is greater than import cost
        # 2. control grid charging to batteries and grid export from batteries acc. to configuration
        self.variables["y"] = []
        for t in self.time_steps:
            self.variables["y"].append(pulp.LpVariable(f"y_{t}", cat="Binary"))

        # Binary variable for charging activation
        self.variables["z_c"] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables["z_c"][i] = [pulp.LpVariable(f"z_c_{i}_{t}", cat="Binary") for t in self.time_steps]
            else:
                self.variables["z_c"][i] = None

        # Binary variable to lock charging against discharging
        self.variables["z_cd"] = {}
        for i, bat in enumerate(self.batteries):
            self.variables["z_cd"][i] = [pulp.LpVariable(f"z_cd_{i}_{t}", cat="Binary") for t in self.time_steps]

    def _setup_target_function(self):
        """
        Gather all target function contributions and instantiate the objective
        """

        # Objective function (1): Maximize economic benefit
        objective = 0

        # Grid import cost (negative because we want to minimize cost) [currency unit]
        for t in self.time_steps:
            objective -= self.variables["n"][t] * self.time_series.p_N[t]

        # Grid export revenue [currency unit]
        for t in self.time_steps:
            objective += self.variables["e"][t] * self.time_series.p_E[t]

        # Final state of charge value [currency unit]
        for i, bat in enumerate(self.batteries):
            objective += self.variables["s"][i][-1] * bat.p_a

        # Penalites for goals that cannot be met
        for i, bat in enumerate(self.batteries):
            # unmet battery charging goals
            if self.batteries[i].s_goal is not None:
                for t in self.time_steps:
                    if self.batteries[i].s_goal[t] > 0:
                        # negative target function contribution in a maximizing optimization
                        objective += -self.goal_penalty_energy * self.variables["s_goal_pen"][i][t]
            # unmet charging demand due to battery reaching maximum SOC
            if bat.p_demand is not None:
                for t in self.time_steps:
                    objective += -self.goal_penalty_power * self.variables["p_demand_pen"][i][t] * (1 + (self.T - t) / self.T)

        # Secondary strategies to implement preferences without impact to actual cost
        # prefer charging first, then grid export
        if self.strategy.charging_strategy == "charge_before_export":
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables["c"][i][t] * self.min_import_price * 1.5e-5 * (self.T - t)

        # prefer charging at high solar production times to unload public grid from peaks
        if self.strategy.charging_strategy == "attenuate_grid_peaks":
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables["c"][i][t] * self.time_series.ft[t] * self.min_import_price * 1e-6

        # prefer discharging batteries completely before importing from grid
        if self.strategy.discharging_strategy == "discharge_before_import":
            for i, bat in enumerate(self.batteries):
                for t in self.time_steps:
                    objective += self.variables["d"][i][t] * self.min_import_price * 5e-6 * (self.T - t)

        self.problem += objective

    def _add_constraints(self):
        """
        Add all constraints to the model
        """

        self.time_steps = range(self.T)

        # Constraint (2): Power balance
        for t in self.time_steps:
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += -self.variables["c"][i][t] + self.variables["d"][i][t]

            self.problem += battery_net_discharge + self.time_series.ft[t] + self.variables["n"][t] == self.variables["e"][t] + self.time_series.gt[t]

        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(self.time_steps) > 0:
                self.problem += (
                    self.variables["s"][i][0] == bat.s_initial + self.eta_c * self.variables["c"][i][0] - (1 / self.eta_d) * self.variables["d"][i][0]
                )

            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (
                    self.variables["s"][i][t]
                    == self.variables["s"][i][t - 1] + self.eta_c * self.variables["c"][i][t] - (1 / self.eta_d) * self.variables["d"][i][t]
                )

            # Constraint (6): Battery SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += self.variables["s"][i][t] + self.variables["s_goal_pen"][i][t] >= bat.s_goal[t]

            # Constraint: Minimum battery charge demand (for t > 0)
            if bat.p_demand is not None:
                for t in self.time_steps:
                    if bat.p_demand[t] > 0:
                        # clip required charge to max charging power if needed
                        # and leave some air to breathe for the optimizer
                        p_demand = bat.p_demand[t]
                        if p_demand >= bat.c_max * self.time_series.dt[t] / 3600.0:
                            p_demand = bat.c_max * self.time_series.dt[t] / 3600.0 * 0.999
                        self.problem += self.variables["c"][i][t] + self.variables["p_demand_pen"][i][t] >= p_demand
                    elif bat.c_min > 0:
                        # in time steps without given charging demand, apply normal lower bound:
                        # Lower bound: either 0 or at least c_min
                        self.problem += self.variables["c"][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.0 * self.variables["z_c"][i][t]
                        self.problem += self.variables["c"][i][t] <= self.M * self.variables["z_c"][i][t]

            else:
                # Constraint (7): Minimum charge power limits if there is not charge demand
                if bat.c_min > 0:
                    for t in self.time_steps:
                        # Lower bound: either 0 or at least c_min
                        self.problem += self.variables["c"][i][t] >= bat.c_min * self.time_series.dt[t] / 3600.0 * self.variables["z_c"][i][t]
                        self.problem += self.variables["c"][i][t] <= self.M * self.variables["z_c"][i][t]

            # control battery charging from grid
            if not bat.charge_from_grid:
                for t in self.time_steps:
                    self.problem += self.variables["c"][i][t] <= self.M * self.variables["y"][t]

            # control battery discharging to grid
            if not bat.discharge_to_grid:
                for t in self.time_steps:
                    self.problem += self.variables["d"][i][t] <= self.M * (1 - self.variables["y"][t])

            # lock charging against discharging
            for t in self.time_steps:
                # Discharge constraint
                self.problem += self.variables["d"][i][t] <= self.M * self.variables["z_cd"][i][t]
                # Charge constraint
                self.problem += self.variables["c"][i][t] <= self.M * (1 - self.variables["z_cd"][i][t])

        # Constraints (4)-(5): Grid flow direction
        for t in self.time_steps:
            # Export constraint
            self.problem += self.variables["e"][t] <= self.M * self.variables["y"][t]
            # Import constraint
            self.problem += self.variables["n"][t] <= self.M * (1 - self.variables["y"][t])

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

        if status == "Optimal":
            result = {
                "status": status,
                "objective_value": pulp.value(self.problem.objective),
                "batteries": [],
                "grid_import": [pulp.value(var) for var in self.variables["n"]],
                "grid_export": [pulp.value(var) for var in self.variables["e"]],
                "flow_direction": [],
            }

            # Extract battery results
            for i, bat in enumerate(self.batteries):
                battery_result = {
                    "charging_power": [pulp.value(var) for var in self.variables["c"][i]],
                    "discharging_power": [pulp.value(var) for var in self.variables["d"][i]],
                    "state_of_charge": [pulp.value(var) for var in self.variables["s"][i]],
                }
                result["batteries"].append(battery_result)

            # Extract flow direction
            for y_var in self.variables["y"]:
                if y_var is not None:
                    result["flow_direction"].append(int(pulp.value(y_var)))
                else:
                    result["flow_direction"].append(0)  # Default to import when constraint not active

            return result
        else:
            return {"status": status, "objective_value": None, "batteries": [], "grid_import": [], "grid_export": [], "flow_direction": []}
