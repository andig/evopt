from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
import pulp
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
import os
import jwt

app = Flask(__name__)

@app.before_request
def before_request_func():
    secret_key = os.environ.get('JWT_TOKEN_SECRET')
    if secret_key:
        auth_header = request.headers.get('Authorization')
        if not auth_header:
            return jsonify({"message": "Missing authorization header"}), 401
        
        try:
            token_type, token = auth_header.split(' ')
            if token_type.lower() != 'bearer':
                return jsonify({"message": "Invalid token type"}), 401
            
            jwt.decode(token, secret_key, algorithms=["HS256"])
        except jwt.ExpiredSignatureError:
            return jsonify({"message": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"message": "Invalid token"}), 401
        except Exception as e:
            return jsonify({"message": str(e)}), 401

api = Api(app, version='1.0', title='EV Charging Optimization API', 
          description='Mixed Integer Linear Programming model for EV charging optimization')

# Namespace for the API
ns = api.namespace('optimize', description='EV Charging Optimization Operations')

# Input models for API documentation
battery_config_model = api.model('BatteryConfig', {
    's_min': fields.Float(required=True, description='Minimum state of charge (Wh)'),
    's_max': fields.Float(required=True, description='Maximum state of charge (Wh)'),
    's_initial': fields.Float(required=True, description='Initial state of charge (Wh)'),
    's_goal': fields.List(fields.Float, required=False, description='Goal state of charge at each time step (Wh)'),
    'c_min': fields.Float(required=True, description='Minimum charge power (W)'),
    'c_max': fields.Float(required=True, description='Maximum charge power (W)'),
    'd_max': fields.Float(required=True, description='Maximum discharge power (W)'),
    'p_a': fields.Float(required=True, description='Value per Wh at end of horizon')
})

time_series_model = api.model('TimeSeries', {
    'dt': fields.List(fields.Float, required=True, description='duration in seconds for each time step (s)'),
    'gt': fields.List(fields.Float, required=True, description='Required average power at each time step (W)'),
    'ft': fields.List(fields.Float, required=True, description='Forecasted average power production at each time step (W)'),
    'p_N': fields.List(fields.Float, required=True, description='Price per kWh taken from grid at each time step'),
    'p_E': fields.List(fields.Float, required=True, description='Price per kWh fed into grid at each time step'),
})

optimization_input_model = api.model('OptimizationInput', {
    'batteries': fields.List(fields.Nested(battery_config_model), required=True, description='Battery configurations'),
    'time_series': fields.Nested(time_series_model, required=True, description='Time series data'),
    'eta_c': fields.Float(required=False, default=0.95, description='Charging efficiency'),
    'eta_d': fields.Float(required=False, default=0.95, description='Discharging efficiency'),
})

# Output models
battery_result_model = api.model('BatteryResult', {
    'charging_power': fields.List(fields.Float, description='Charging power at each time step (W)'),
    'discharging_power': fields.List(fields.Float, description='Discharging power at each time step (W)'),
    'state_of_charge': fields.List(fields.Float, description='State of charge at each time step (Wh)')
})

optimization_result_model = api.model('OptimizationResult', {
    'status': fields.String(description='Optimization status'),
    'objective_value': fields.Float(description='Optimal objective function value'),
    'batteries': fields.List(fields.Nested(battery_result_model), description='Battery optimization results'),
    'grid_import': fields.List(fields.Float, description='Power imported from grid at each time step (W)'),
    'grid_export': fields.List(fields.Float, description='Power exported to grid at each time step (W)'),
    'flow_direction': fields.List(fields.Integer, description='Binary flow direction (1=export, 0=import)')
})

@dataclass
class BatteryConfig:
    s_min: float
    s_max: float
    s_initial: float
    c_min: float
    c_max: float
    d_max: float
    p_a: float
    s_goal: Optional[List[float]] = None  # Goal state of charge (Wh)

@dataclass
class TimeSeriesData:
    dt: List[int] # time step length
    gt: List[float]  # Required avg. power
    ft: List[float]  # Forecasted production power
    p_N: List[float]  # Import prices
    p_E: List[float]  # Export prices

class EVChargingOptimizer:
    def __init__(self, batteries: List[BatteryConfig], time_series: TimeSeriesData, 
                 eta_c: float = 0.95, eta_d: float = 0.95, M: float = 1e6):
        self.batteries = batteries
        self.time_series = time_series
        self.eta_c = eta_c
        self.eta_d = eta_d
        self.M = M
        self.T = len(time_series.gt)
        self.problem = None
        self.variables = {}
        
    def create_model(self):
        """Create the MILP model"""
        # Create problem
        self.problem = pulp.LpProblem("EV_Charging_Optimization", pulp.LpMaximize)
        
        # Time steps
        time_steps = range(self.T)
        
        # Decision variables
        # Charging power variables [W]
        self.variables['c'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['c'][i] = [
                pulp.LpVariable(f"c_{i}_{t}", lowBound=0, upBound=bat.c_max)
                for t in time_steps
            ]
        
        # Discharging power variables [W]
        self.variables['d'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['d'][i] = [
                pulp.LpVariable(f"d_{i}_{t}", lowBound=0, upBound=bat.d_max)
                for t in time_steps
            ]
        
        # State of charge variables [Wh]
        self.variables['s'] = {}
        for i, bat in enumerate(self.batteries):
            self.variables['s'][i] = [
                pulp.LpVariable(f"s_{i}_{t}", lowBound=bat.s_min, upBound=bat.s_max)
                for t in time_steps
            ]
        
        # Grid import/export variables [W]
        self.variables['n'] = [pulp.LpVariable(f"n_{t}", lowBound=0) for t in time_steps]
        self.variables['e'] = [pulp.LpVariable(f"e_{t}", lowBound=0) for t in time_steps]
        
        # Binary flow direction variables (only when p_N <= p_E)
        self.variables['y'] = []
        for t in time_steps:
            if self.time_series.p_N[t] <= self.time_series.p_E[t]:
                self.variables['y'].append(pulp.LpVariable(f"y_{t}", cat='Binary'))
            else:
                self.variables['y'].append(None)
        
        # Binary variable for charging activation
        self.variables['z_c'] = {}
        for i, bat in enumerate(self.batteries):
            if bat.c_min > 0:
                self.variables['z_c'][i] = [
                    pulp.LpVariable(f"z_c_{i}_{t}", cat='Binary')
                    for t in time_steps
                ]
            else:
                self.variables['z_c'][i] = None
            
        # Objective function (1): Maximize economic benefit
        objective = 0
        
        # Grid import cost (negative because we want to minimize cost) [EUR]
        for t in time_steps:
            objective -= self.variables['n'][t] * self.time_series.p_N[t] * self.time_series.dt[t] / 3600 / 1000
        
        # Grid export revenue [EUR]
        for t in time_steps:
            objective += self.variables['e'][t] * self.time_series.p_E[t] * self.time_series.dt[t] / 3600 / 1000
        
        # Final state of charge value [EUR]
        for i, bat in enumerate(self.batteries):
            objective += self.variables['s'][i][-1] * bat.p_a / 1000
        
        self.problem += objective
        
        # Constraints
        self._add_constraints()
        
    def _add_constraints(self):
        """Add all constraints to the model"""
        time_steps = range(self.T)
        
        # Constraint (2): Power balance
        for t in time_steps:
            battery_net_discharge = 0
            for i, bat in enumerate(self.batteries):
                battery_net_discharge += (- self.time_series.dt[t] * self.variables['c'][i][t] + 
                                    self.time_series.dt[t] * self.variables['d'][i][t])
            
            self.problem += (battery_net_discharge 
                             + self.time_series.dt[t] * self.time_series.ft[t] 
                             + self.time_series.dt[t] * self.variables['n'][t] 
                             == self.time_series.dt[t] * self.variables['e'][t] 
                             + self.time_series.dt[t] * self.time_series.gt[t])
        
        # Constraint (3): Battery dynamics
        for i, bat in enumerate(self.batteries):
            # Initial state of charge
            if len(time_steps) > 0:
                self.problem += (self.variables['s'][i][0] 
                                == bat.s_initial 
                                + self.eta_c * self.time_series.dt[t] / 3600 * self.variables['c'][i][0] 
                                - (1/self.eta_d) * self.time_series.dt[t] / 3600 * self.variables['d'][i][0])
            
            # State of charge evolution
            for t in range(1, self.T):
                self.problem += (self.variables['s'][i][t] 
                                == self.variables['s'][i][t-1] 
                                + self.eta_c * self.time_series.dt[t] / 3600 * self.variables['c'][i][t] 
                                - (1/self.eta_d) * self.time_series.dt[t] / 3600 * self.variables['d'][i][t])

            # Constraint (6): Battery SOC goal constraints (for t > 0)
            if bat.s_goal is not None:
                for t in range(1, self.T):
                    if bat.s_goal[t] > 0:
                        self.problem += (self.variables['s'][i][t] >= bat.s_goal[t])

            # Constraint (7): Minimum charge power limits
            if bat.c_min > 0:
                for t in time_steps:
                    # Lower bound: either 0 or at least c_min
                    self.problem += self.variables['c'][i][t] >= bat.c_min * self.variables['z_c'][i][t]

        # Constraints (4)-(5): Grid flow direction (only when p_N <= p_E)
        for t in time_steps:
            if self.variables['y'][t] is not None:  # Only when p_N <= p_E
                # Export constraint
                self.problem += self.variables['e'][t] <= self.M * self.variables['y'][t]
                # Import constraint
                self.problem += self.variables['n'][t] <= self.M * (1 - self.variables['y'][t])
            
    def solve(self) -> Dict:
        """Solve the optimization problem and return results"""
        if self.problem is None:
            self.create_model()
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)  # Silent solver
        self.problem.solve(solver)
        
        # Extract results
        status = pulp.LpStatus[self.problem.status]
        
        if status == 'Optimal':
            result = {
                'status': status,
                'objective_value': pulp.value(self.problem.objective),
                'batteries': [],
                'grid_import': [pulp.value(var) for var in self.variables['n']],
                'grid_export': [pulp.value(var) for var in self.variables['e']],
                'flow_direction': []
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
                'batteries': [],
                'grid_import': [],
                'grid_export': [],
                'flow_direction': []
            }

@ns.route('/charge-schedule')
class OptimizeCharging(Resource):
    @api.expect(optimization_input_model)
    @api.marshal_with(optimization_result_model)
    def post(self):
        """
        Optimize EV charging schedule using MILP
        
        This endpoint solves a Mixed Integer Linear Programming problem to optimize
        EV charging schedules considering battery constraints, grid prices, and energy demands.
        """
        try:
            data = request.get_json()
            
            # Validate input data
            if not data:
                api.abort(400, "No input data provided")
            
            # Parse battery configurations
            batteries = []
            for bat_data in data['batteries']:
                batteries.append(BatteryConfig(
                    s_min=bat_data['s_min'],
                    s_max=bat_data['s_max'],
                    s_initial=bat_data['s_initial'],
                    s_goal=bat_data.get('s_goal'),
                    c_min=bat_data['c_min'],
                    c_max=bat_data['c_max'],
                    d_max=bat_data['d_max'],
                    p_a=bat_data['p_a'],
                ))
            
            # Parse time series data
            time_series = TimeSeriesData(
                dt=data['time_series']['dt'],
                gt=data['time_series']['gt'],
                ft=data['time_series']['ft'],
                p_N=data['time_series']['p_N'],
                p_E=data['time_series']['p_E'],
            )

            # Validate time series lengths
            lengths = [len(time_series.gt), len(time_series.ft), 
                      len(time_series.p_N), len(time_series.p_E)]

            # Validate s_goal if provided
            for i, bat in enumerate(batteries):
                if bat.s_goal is not None:
                    lengths.append(len(bat.s_goal))

            if len(set(lengths)) > 1:
                api.abort(400, "All time series must have the same length")
            
        except KeyError as e:
            api.abort(400, f"Missing required field: {str(e)}")
        except (TypeError, ValueError) as e:
            api.abort(400, f"Invalid data format: {str(e)}")
        
        try:
            # Create and solve optimizer
            optimizer = EVChargingOptimizer(
                batteries=batteries,
                time_series=time_series,
                eta_c=data.get('eta_c', 0.95),
                eta_d=data.get('eta_d', 0.95),
                M=1e6
            )
            
            result = optimizer.solve()
            return result
            
        except Exception as e:
            api.abort(500, f"Optimization failed: {str(e)}")

@ns.route('/health')
class Health(Resource):
    def get(self):
        """Health check endpoint"""
        return {'status': 'healthy', 'message': 'EV Charging MILP API is running'}

# Example data endpoint for testing
@ns.route('/example')
class ExampleData(Resource):
    def get(self):
        """Get example input data for testing the optimization"""
        example_data = {
            "batteries": [
                {
                    "s_min": 2000,
                    "s_max": 36000,
                    "s_initial": 15000,
                    "s_goal": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 30000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "c_min": 1380, 
                    "c_max": 3380, 
                    "d_max": 0,
                    "p_a": 0.25
                },
                {
                    "s_min": 2500,
                    "s_max": 16200,
                    "s_initial": 5000,
                    "c_min": 0,
                    "c_max": 6000, 
                    "d_max": 6000, 
                    "p_a": 0.15
                }
            ],
            "time_series": {
                "dt": [900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900, 900],
                "gt": [213, 77.1, 112, 109, 101, 78.7, 90.3, 166, 72.7, 59, 155, 177, 206, 224, 251, 86.5, 73.6, 69.8, 55.6, 283, 254, 886, 179, 185, 174, 165, 202, 198, 224, 104, 82.3, 135, 138, 98.5, 80.3, 121, 90.2, 64, 166, 279, 324, 303, 349, 235, 168, 191, 152, 153, 183, 220, 241, 201, 300, 285, 232, 240, 284, 196, 165, 147, 208, 198, 244, 266, 221, 218, 245, 234, 235, 287, 330, 237, 171, 160, 143, 143, 179, 199, 148, 173, 166, 237, 213, 456, 235, 162, 72.5, 138, 137, 143, 177, 146, 92, 92.9, 128, 204, 181, 260, 269, 188, 110, 90.6, 91, 91.8, 148, 130, 124, 132, 154, 196, 150, 199, 185, 113, 56.9, 58.8, 84.6, 132, 176, 215, 72.4, 67.6, 69.1, 195, 157, 258, 276, 189, 148, 94.9, 91.5, 91.4, 110, 189, 196, 192, 279, 347, 38.4, 53, 207, 7.44, 26.5, 7.04, 68.3, 36.7, 18.5, 8.26, 6.44, 1.5, 53.7, 1.3, 95.9, 93.2, 279, 127, 93.6, 12.5, 21.6, 19.9, 43.2, 160, 169, 199, 254, 267, 238, 221, 139, 110, 64.2, 41.3, 67.6, 45.7, 141, 197, 181, 229, 182, 161, 209, 154, 147, 129, 204, 190, 179, 134, 174], 
                "ft": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2.26, 7.6, 61.2, 95.8, 125, 146, 165, 184, 203, 224, 241, 264, 343, 1085, 1773, 2126, 2393, 2665, 2955, 3246, 3363, 3500, 3584, 3742, 3750, 3945, 3091, 4097, 4101, 4062, 4081, 4134, 4137, 4098, 4042, 3986, 3953, 3932, 3852, 3752, 3723, 3545, 1454, 2122, 2884, 2968, 2362, 2756, 2512, 2285, 2099, 1844, 1596, 1378, 1197, 998, 809, 584, 309, 210, 168, 129, 92.3, 56.3, 14, 721, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.24, 10.4, 72.9, 112, 150, 181, 207, 233, 258, 287, 312, 345, 394, 1002, 1663, 2028, 2268, 2509, 2726, 2978, 3132, 3283, 3276, 2442, 2279, 3235, 2639, 1182, 923, 1037, 1382, 3163, 3286, 3497, 3550, 2304, 3600, 3125, 2576, 1550, 1097, 1461, 2616, 3156, 1713, 1235, 378, 150, 42.6, 0, 0, 0, 0, 15, 84.5, 165, 305, 608, 595, 301, 166, 116, 78.4, 22.8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                "p_N": [0.2201, 0.2176, 0.1985, 0.1972, 0.20508, 0.19509, 0.19687, 0.1851, 0.19509, 0.19577, 0.1966, 0.1801, 0.18509, 0.1977, 0.18811, 0.1926, 0.17509, 0.18509, 0.19753, 0.2057, 0.1867, 0.1935, 0.208, 0.21564, 0.2048, 0.2166, 0.2217, 0.21162, 0.2553, 0.235, 0.2109, 0.19, 0.28394, 0.2181, 0.19495, 0.1606, 0.26737, 0.2, 0.17497, 0.142, 0.25956, 0.17994, 0.14496, 0.1179, 0.20344, 0.1521, 0.15993, 0.14493, 0.1583, 0.14531, 0.13994, 0.13493, 0.1417, 0.1424, 0.13249, 0.1241, 0.1326, 0.1418, 0.1497, 0.16812, 0.13994, 0.1594, 0.1744, 0.21084, 0.15479, 0.18491, 0.1911, 0.2123, 0.1405, 0.1872, 0.2211, 0.25493, 0.13508, 0.20506, 0.2482, 0.2987, 0.2496, 0.27494, 0.3013, 0.36355, 0.3513, 0.3431, 0.67743, 0.3627, 0.3868, 0.3334, 0.35, 0.40659, 0.2912, 0.2707, 0.26491, 0.24934, 0.2351, 0.2275, 0.2295, 0.2176, 0.23, 0.22, 0.21473, 0.18508, 0.21502, 0.21002, 0.19176, 0.19001, 0.20627, 0.20009, 0.19009, 0.18509, 0.19502, 0.19501, 0.18957, 0.20002, 0.18007, 0.19008, 0.19509, 0.22277, 0.18509, 0.22706, 0.20009, 0.20507, 0.23492, 0.21077, 0.22005, 0.22502, 0.25993, 0.25494, 0.22018, 0.17008, 0.25899, 0.25, 0.19006, 0.15501, 0.25, 0.22995, 0.18333, 0.125, 0.22, 0.19107, 0.165, 0.14994, 0.2, 0.16997, 0.15676, 0.13998, 0.1599, 0.15491, 0.11073, 0.10637, 0.125, 0.1499, 0.15491, 0.1301, 0.125, 0.125, 0.15992, 0.18398, 0.1399, 0.14994, 0.1849, 0.19646, 0.14495, 0.16496, 0.18734, 0.22494, 0.125, 0.2157, 0.18508, 0.24997, 0.15005, 0.20008, 0.25993, 0.25161, 0.22008, 0.26497, 0.28129, 0.2949, 0.29499, 0.30495, 0.29498, 0.46605, 0.28515, 0.29993, 0.28992, 0.27995, 0.26906, 0.25991, 0.24492, 0.23491, 0.24991],
                "p_E": [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.1]
            },
            "eta_c": 0.95,
            "eta_d": 0.95
        }
        return example_data

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7050)