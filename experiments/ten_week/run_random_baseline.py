import sys
import os

os.environ['EPLUS_PATH'] = "/Applications/EnergyPlus-24-1-0"
sys.path.append("/Applications/EnergyPlus-24-1-0")

import logging
import math
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import gymnasium as gym
import sinergym
from sinergym.utils.logger import TerminalLogger
from sinergym.utils.wrappers import (
    CSVLogger,
    LoggerWrapper,
    NormalizeAction,
    NormalizeObservation,
)

from exploration_mppi.args import parse_args
from exploration_mppi.gp import HVACGaussianProcess
from exploration_mppi.mppi_baseline import MPPIController

# ============================================================================
# HYPERPARAMETER SPACE - MODIFY THESE FOR TUNING
# ============================================================================

# Environment parameters
CALIBRATION_EPISODES = 5
CALIBRATION_STEPS_PER_EPISODE = 100

# GP hyperparameters
GP_PREDICT_DELTA = True
GP_SAFETY_THRESHOLD = 0.0

# MPPI hyperparameters
MPPI_HORIZON = 4
MPPI_NUM_SAMPLES = 100
MPPI_GAMMA = 0.85
MPPI_LAMBDA_UNCERTAINTY = 1e-2
MPPI_ETA = 1.0
MPPI_UNCERTAINTY_THRESHOLD = 999

# Updated experiment parameters for 1-week simulation periods
TOTAL_EXPERIMENT_WEEKS = 10   # Week 1: rule-based, Weeks 2-4: control+exploration
WEEKDAY_CONTROL_DAYS = 5     # Monday-Friday CLUE control
WEEKEND_EXPLORATION_DAYS = 2 # Saturday-Sunday rule-based exploration
DAYS_PER_WEEK = 7           # Each week runs 7.1 to 7.7

OCCUPIED_START_HOUR = 8
OCCUPIED_END_HOUR = 17
OCCUPIED_ACTION = 9
UNOCCUPIED_ACTION = 0
DEFAULT_CONTROLLER = False

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def step_env(env, *args, **kwargs):
    """
    Wrapper to extract correct observation variables
    Original: ['month', 'day_of_month', 'hour', 'outdoor_temperature', 'outdoor_humidity', 'wind_speed', 
               'wind_direction', 'diffuse_solar_radiation', 'direct_solar_radiation', 'htg_setpoint', 'clg_setpoint', 
               'air_temperature', 'air_humidity', 'people_occupant', 'co2_emission', 'HVAC_electricity_demand_rate', 'total_electricity_HVAC']
    
    Expected: ['hour', 'outdoor_temp', 'outdoor_humidity', 'wind_speed', 'wind_direction', 'direct_solar_radiation', 
               'air_temperature', 'air_humidity', 'people_occupant']
    """
    obs, reward, terminated, truncated, info = env.step(*args, **kwargs)
    # Extract: [hour, outdoor_temp, outdoor_humidity, wind_speed, wind_direction, 
    #          direct_solar_radiation, air_temperature, air_humidity, people_occupant]
    obs = [obs[2], obs[3], obs[4], obs[5], obs[6], obs[8], obs[11], obs[12], obs[13]]
    return obs, reward, terminated, truncated, info

def reset_env(env):
    """Reset environment and extract correct observation variables"""
    obs, info = env.reset()
    # Extract same variables as step_env
    obs = [obs[2], obs[3], obs[4], obs[5], obs[6], obs[8], obs[11], obs[12], obs[13]]
    return obs, info

def new_action_mapping(action: int) -> np.ndarray:
    """Map discrete action to [heating_setpoint, cooling_setpoint]"""
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: np.array([12, 30], dtype=np.float32),
        1: np.array([13, 29], dtype=np.float32),
        2: np.array([14, 28], dtype=np.float32),
        3: np.array([15, 27], dtype=np.float32),
        4: np.array([16, 26], dtype=np.float32),
        5: np.array([17, 25], dtype=np.float32),
        6: np.array([18, 24], dtype=np.float32),
        7: np.array([19, 23.25], dtype=np.float32),
        8: np.array([20, 23.25], dtype=np.float32),
        9: np.array([21, 23.25], dtype=np.float32),
    }
    return mapping[action]

def inverse_action_mapping(continuous_action: np.ndarray) -> int:
    """Map [heating_setpoint, cooling_setpoint] back to discrete action (0-9)"""
    if isinstance(continuous_action, list):
        continuous_action = np.array(continuous_action)
    
    heating_setpoint = continuous_action[0]
    cooling_setpoint = continuous_action[1]
    
    # Create mapping from continuous to discrete
    action_mapping = {
        0: (12, 30),
        1: (13, 29),
        2: (14, 28),
        3: (15, 27),
        4: (16, 26),
        5: (17, 25),
        6: (18, 24),
        7: (19, 23.25),
        8: (20, 23.25),
        9: (21, 23.25),
    }
    
    # Find closest match
    min_distance = float('inf')
    best_action = 0

    for action, (heat_sp, cool_sp) in action_mapping.items():
        distance = abs(cooling_setpoint - cool_sp)
        if distance < 1:
            return action

def get_day_of_week(day_in_week):
    """
    Get day of week (0=Monday, 6=Sunday) for a given day in week
    Since each week is 7.1-7.7, day_in_week is 0-6
    """
    return day_in_week % 7

def is_weekday(day_in_week):
    """Check if day is a weekday (Monday-Friday)"""
    return get_day_of_week(day_in_week) < 5

def is_weekend(day_in_week):
    """Check if day is a weekend (Saturday-Sunday)"""
    return get_day_of_week(day_in_week) >= 5

# ============================================================================
# UTILITY FUNCTIONS FOR ORGANIZED RESULTS
# ============================================================================

def create_organized_results_folder(method_name, timestamp, weather_file=None):
    """
    Create organized folder structure for experiment results
    
    Args:
        method_name: Name of the method (e.g., 'clue_1week', 'exploration')
        timestamp: Timestamp string for the run
        weather_file: Optional weather file name to include in folder name
        
    Returns:
        Dictionary with folder paths
    """
    # Create main run folder with weather identifier if provided
    if weather_file:
        # Extract first 6 characters of weather file name
        weather_id = weather_file[:6] if len(weather_file) >= 6 else weather_file
        run_folder = f"results/{method_name}_{weather_id}_{timestamp}"
    else:
        run_folder = f"results/{method_name}_{timestamp}"
    
    os.makedirs(run_folder, exist_ok=True)
    
    # Create subfolders
    subfolders = {
        'collected_exploration_data': f"{run_folder}/collected_exploration_data",
        'data': f"{run_folder}/data", 
        'figures': f"{run_folder}/figures"
    }
    
    for folder in subfolders.values():
        os.makedirs(folder, exist_ok=True)
        
    return {
        'run_folder': run_folder,
        **subfolders
    }

def calculate_energy_consumption(data_list, obs_mean, obs_var):
    """
    Calculate total energy consumption from data
    
    Args:
        data_list: List of data dictionaries with total_power_demand
        obs_mean: Observation means for denormalization (not used for power)
        obs_var: Observation variances for denormalization (not used for power)
        
    Returns:
        Total energy consumption in kWh
    """
    if not data_list:
        return 0.0
    
    total_power = 0.0
    for data in data_list:
        # Power consumption is directly available in total_power_demand (already in Watts)
        power_watts = data.get('total_power_demand', 0)
        total_power += power_watts
    
    # Convert to kWh (assuming 15-minute timesteps = 0.25 hours)
    timestep_hours = 0.25  # 15 minutes
    total_energy_kwh = total_power * timestep_hours / 1000  # Convert W to kW
    
    return total_energy_kwh

def calculate_comfort_violations(data_list, obs_mean, obs_var, comfort_bounds=(23, 26)):
    """
    Calculate comfort violation statistics during occupied hours only (8 AM to 6 PM)
    
    Args:
        data_list: List of data dictionaries with indoor_temp and hour
        obs_mean: Observation means for denormalization
        obs_var: Observation variances for denormalization  
        comfort_bounds: Tuple of (lower, upper) comfort bounds in actual temperature
        
    Returns:
        Dictionary with violation statistics
    """
    if not data_list:
        return {'violation_rate': 0.0, 'violation_hours': 0.0, 'total_hours': 0.0}
    
    temp_mean = obs_mean[6]
    temp_std = math.sqrt(obs_var[6])
    
    occupied_start = 8
    occupied_end = 18
    violation_count = 0
    occupied_count = 0
    
    for data in data_list:
        hour = data.get('hour', 0)
        if occupied_start <= hour <= occupied_end:
            occupied_count += 1
            normalized_temp = data.get('indoor_temp', 0)
            actual_temp = normalized_temp * temp_std + temp_mean
            if actual_temp < comfort_bounds[0] or actual_temp > comfort_bounds[1]:
                violation_count += 1
    
    violation_rate = violation_count / occupied_count if occupied_count > 0 else 0
    timestep_hours = 0.25  # 15 minutes
    violation_hours = violation_count * timestep_hours
    total_hours = occupied_count * timestep_hours
    
    return {
        'violation_rate': violation_rate,
        'violation_hours': violation_hours, 
        'total_hours': total_hours,
        'violation_count': violation_count,
        'total_count': occupied_count
    }

# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
# ============================================================================

def create_environment(args, use_default_controller=False):
    """
    Create and calibrate environment with normalization parameters for 1-week simulation
    
    Args:
        args: Command line arguments
        use_default_controller: If True, use empty action space for default controller
    
    Returns:
        env: Calibrated environment
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
    """
    print("="*60)
    print("Step 1: Creating and calibrating environment (1-week period: 7.1-7.7)")
    print("="*60)
    
    # Set run period for 1 week only (July 1-7 for summer, January 1-7 for winter)
    extra_params = {'timesteps_per_hour': 4}
    if args.winter:
        extra_params['runperiod'] = (1,1,1997,7,1,1997)  # January 1-7 (winter)
    else:
        extra_params['runperiod'] = (1,7,1997,7,7,1997)  # July 1-7 (summer)
    
    # Configure action space based on use_default_controller
    if use_default_controller:
        # Empty action space to use default rule-based controller
        extra_params['action_space'] = gym.spaces.Box(low=0, high=0, shape=(0,))
        print("Using default rule-based controller (empty action space)")
    
    # Create environment with weather file if specified
    if args.weather:
        env = gym.make(args.environment, weather_files=args.weather, config_params=extra_params)
        print(f"Weather file: {args.weather}")
    else:
        env = gym.make(args.environment, config_params=extra_params)
    
    # Only set action mapping if not using default controller
    if not use_default_controller:
        env.action_mapping = new_action_mapping
    
    print(f"Environment: {args.environment}")
    print(f"Action space: {env.action_space}")
    print(f"Run period: {extra_params['runperiod']} (1 week)")
    
    # Apply normalization wrapper
    env = NormalizeObservation(env)
    env.activate_update()

    print("Calibrating normalization parameters...")
    
    # Calibration phase - run several episodes to calibrate normalization
    for episode in range(CALIBRATION_EPISODES):
        print(f"  Calibration episode {episode + 1}/{CALIBRATION_EPISODES}")
        obs, info = reset_env(env)
        
        for step in range(CALIBRATION_STEPS_PER_EPISODE):
            if use_default_controller:
                # For default controller, use empty action
                action = env.action_space.sample()
            else:
                action = env.action_space.sample()
            obs, reward, terminated, truncated, info = step_env(env, action)
            
            if terminated or truncated:
                break
                
    # Extract normalization parameters for our 9 variables
    obs_var = env.var
    obs_var = [obs_var[2], obs_var[3], obs_var[4], obs_var[5], obs_var[6], obs_var[8], obs_var[11], obs_var[12], obs_var[13]]
    obs_mean = env.mean
    obs_mean = [obs_mean[2], obs_mean[3], obs_mean[4], obs_mean[5], obs_mean[6], obs_mean[8], obs_mean[11], obs_mean[12], obs_mean[13]]
    
    print("Normalization parameters calibrated:")
    print(f"  Mean: {[f'{x:.3f}' for x in obs_mean]}")
    print(f"  Var:  {[f'{x:.3f}' for x in obs_var]}")
    
    # Deactivate normalization updates for actual experiment
    env.deactivate_update()
    
    return env, obs_mean, obs_var

def collect_truth_table(env, args):
    """
    Collect outdoor temperature and environmental data for the 1-week simulation period
    
    Returns:
        truth_table: DataFrame with environmental data for MPPI planning
    """
    print("="*60)
    print("Step 2: Collecting truth table (1-week environmental data: 7.1-7.7)")
    print("="*60)
    
    truth_data = []
    obs, info = reset_env(env)
    step_count = 0
    
    # Run environment from reset to end to collect all environmental data (1 week)
    terminated = truncated = False
    while not (terminated or truncated):
        # Store current state information
        truth_data.append({
            'step': step_count,
            'hour': info['hour'],
            'day': info['day'],
            'month': info['month'],
            'obs': obs.copy(),
            'outdoor_temp': obs[1],  # Index 1 is outdoor temperature
            'indoor_temp': obs[6],   # Index 6 is indoor temperature
            'occupancy': obs[8],     # Index 8 is occupancy
        })
        
        # Take a dummy action to advance the simulation
        action = 5  # Neutral action
        obs, reward, terminated, truncated, info = step_env(env, action)
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Day {day + 1} collected, total steps: {step_count}")
    
    truth_table = pd.DataFrame(truth_data)
    print(f"Truth table collected: {len(truth_table)} timesteps (1 week)")
    
    return truth_table

def random_controller(obs, info, args):
    """
    Random controller: select random action from 0-9
    
    Args:
        obs: Normalized observation [9]
        info: Environment info dict
        args: Arguments
        
    Returns:
        action: Random discrete action (0-9)
    """
    return np.random.randint(0, 10)

def collect_initial_week_data(env, truth_table, args, week_num=1):
    """
    Collect 1 week of random action data (7.1 to 7.7)
    
    Args:
        env: Environment
        truth_table: DataFrame with environmental data
        args: Arguments
        week_num: Week number for labeling
        
    Returns:
        week_data: List of data examples for the week
    """
    print("="*60)
    print(f"Step 3: Collecting Week {week_num} random action data (7.1-7.7)")
    print("="*60)
    
    week_data = []
    obs, info = reset_env(env)
    step_count = 0
    
    # Calculate total steps for 1 week
    total_week_steps = DAYS_PER_WEEK * 24 * args.timestep
    
    while step_count < total_week_steps and not (info.get('terminated', False) or info.get('truncated', False)):
        # Use random controller
        action = random_controller(obs, info, args)
        
        # Take step
        next_obs, reward, terminated, truncated, info = step_env(env, action)
        
        # Convert discrete action to continuous for consistency
        discrete_action = action
        continuous_action = new_action_mapping(discrete_action)
        
        # Store data
        week_data.append({
            'obs': obs.copy(),
            'action': discrete_action,
            'continuous_action': continuous_action,
            'next_obs': next_obs.copy(),
            'reward': reward,
            'step': step_count,
            'hour': info['hour'],
            'day': info['day'],
            'month': info['month'],
            'indoor_temp': obs[6],
            'next_indoor_temp': next_obs[6],
            'temp_change': next_obs[6] - obs[6],
            'total_power_demand': info.get('total_power_demand', 0),
            'phase': 'random_data' if week_num == 1 else 'weekend_exploration',
            'week': week_num,
            'day_in_week': step_count // (24 * args.timestep)
        })
        
        obs = next_obs
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Day {day + 1}/7 completed, total steps: {step_count}")
            
        if terminated or truncated:
            print(f"Environment terminated during week {week_num} data collection")
            break
    
    print(f"Week {week_num} data collected: {len(week_data)} examples")
    # Print some statistics about the actions used
    actions_used = [d['action'] for d in week_data]
    print(f"Action distribution: {np.bincount(actions_used, minlength=10)}")
    return week_data

def train_gp_model(exploration_data, obs_mean, obs_var):
    """
    Train GP model with exploration data only, predicting temperature changes
    
    Args:
        exploration_data: List of exploration examples (no control data)
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        
    Returns:
        gp_model: Trained GP model
        gp_wrapper: Wrapper function for MPPI controller
    """
    print("="*60)
    print(f"Training GP model with {len(exploration_data)} exploration examples")
    print("="*60)
    
    # Extract data from exploration examples
    observations = []
    actions = []
    temperature_changes = []
    
    for example in exploration_data:
        obs = example['obs']
        action = example['action']
        temp_change = example['temp_change']
        
        observations.append(obs)
        actions.append(action)
        temperature_changes.append(temp_change)
    
    observations = np.array(observations)
    actions = np.array(actions)
    temperature_changes = np.array(temperature_changes)
    
    print(f"Temperature change stats: mean={np.mean(temperature_changes):.4f}, std={np.std(temperature_changes):.4f}")
    
    # Create and train GP model
    gp_model = HVACGaussianProcess(
        input_dim=10,  # 9 observations + 1 action
        predict_delta=GP_PREDICT_DELTA,
        safety_threshold=GP_SAFETY_THRESHOLD
    )
    
    # Normalize actions for GP (actions are 0-9, normalize to [-1, 1])
    normalized_actions = (actions / 4.5) - 1.0
    
    # Train GP model
    gp_model.fit(observations, normalized_actions, temperature_changes)
    
    print("GP model trained successfully")
    
    # Create wrapper function for MPPI controller
    def gp_wrapper(state_action):
        """
        Wrapper function for MPPI controller
        
        Args:
            state_action: [state(9) + action(1)] normalized
            
        Returns:
            predicted_temp_change: Predicted temperature change
            uncertainty: Prediction uncertainty
        """
        state_action = np.array(state_action).reshape(1, -1)
        
        # Extract state and action
        state = state_action[0, :9]
        action = state_action[0, 9]
        
        # Get prediction
        temp_change, uncertainty = gp_model.predict(state, action, return_std=True)
        
        return temp_change, uncertainty
    
    return gp_model, gp_wrapper

def create_future_env_data(current_obs, current_info, truth_table, step_count, horizon):
    """
    Create future environmental data for MPPI planning using truth table
    
    Args:
        current_obs: Current observation [9]
        current_info: Current info dict
        truth_table: DataFrame with environmental data
        step_count: Current step in simulation
        horizon: Planning horizon
        
    Returns:
        future_data: [horizon, 9] array of future environmental data
    """
    future_data = np.zeros((horizon, 9))
    
    for t in range(horizon):
        future_step = step_count + t + 1
        
        # If we have truth table data, use it
        if future_step < len(truth_table):
            future_obs = truth_table.iloc[future_step]['obs']
            future_data[t] = future_obs
        else:
            # If beyond truth table, use last known values with small variations
            if future_step - 1 < len(truth_table):
                last_obs = truth_table.iloc[-1]['obs']
                future_data[t] = last_obs
                # Add small random variations to weather variables
                future_data[t, 1:6] += np.random.normal(0, 0.02, 5)
            else:
                # Use current observation as fallback
                future_data[t] = current_obs
                future_data[t, 1:6] += np.random.normal(0, 0.02, 5)
    
    return future_data

def run_control_exploration_week(env, gp_wrapper, truth_table, all_exploration_data, obs_mean, obs_var, args, week_num):
    """
    Run 1 week of mixed control/exploration (5 weekdays control + 2 weekends exploration)
    Environment is reset to 7.1 at the beginning of each week
    
    Args:
        env: Environment
        gp_wrapper: GP wrapper function
        truth_table: DataFrame with environmental data
        all_exploration_data: All accumulated exploration data
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        args: Arguments
        week_num: Week number (2, 3, 4)
        
    Returns:
        control_data: Control data from weekdays
        exploration_data: Exploration data from weekends
    """
    print("="*60)
    print(f"Step 4.{week_num-1}: Running Week {week_num} Control+Exploration (7.1-7.7)")
    print("="*60)
    
    # Reset environment to start of week (7.1)
    obs, info = reset_env(env)
    
    # Extract normalization parameters for MPPI
    temp_mean = obs_mean[6]  # Index 6 is indoor temperature
    temp_std = math.sqrt(obs_var[6])
    hour_mean = obs_mean[0]  # Index 0 is hour
    hour_std = math.sqrt(obs_var[0])
    
    # Create MPPI controller with current exploration data
    mppi_controller = MPPIController(
        gp_model=gp_wrapper,
        horizon=MPPI_HORIZON,
        num_samples=MPPI_NUM_SAMPLES,
        gamma=MPPI_GAMMA,
        lambda_uncertainty=MPPI_LAMBDA_UNCERTAINTY,
        eta=MPPI_ETA,
        uncertainty_threshold=MPPI_UNCERTAINTY_THRESHOLD,
        temp_norm_params=(temp_mean, temp_std),
        hour_norm_params=(hour_mean, hour_std)
    )
    
    control_data = []
    exploration_data = []
    step_count = 0
    all_fallback_flags = []
    
    # Run through 1 week (7 days)
    total_week_steps = DAYS_PER_WEEK * 24 * args.timestep
    
    while step_count < total_week_steps:
        day_in_week = step_count // (24 * args.timestep)  # 0-6 for Mon-Sun
        
        # Determine if this is a weekday or weekend
        if is_weekday(day_in_week):
            # WEEKDAY: CLUE Control
            # Create future environmental data for planning
            future_env_data = create_future_env_data(
                obs, info, truth_table, step_count, MPPI_HORIZON
            )
            
            # Plan action using MPPI
            try:
                dropped_pairs, action, is_fallback = mppi_controller.plan(
                    np.array(obs), future_env_data
                )
                all_fallback_flags.append(is_fallback)
                
            except Exception as e:
                print(f"MPPI planning failed at step {step_count}: {e}")
                # Fallback to random action
                action = random_controller(obs, info, args)
                is_fallback = True
                dropped_pairs = []
                all_fallback_flags.append(is_fallback)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = step_env(env, action)
            
            # Store control data
            control_datum = {
                'step': step_count,
                'hour': info['hour'],
                'day': info['day'],
                'month': info['month'],
                'action': action,
                'reward': reward,
                'is_fallback': is_fallback,
                'dropped_pairs_count': len(dropped_pairs),
                'indoor_temp': obs[6],
                'outdoor_temp': obs[1],
                'occupancy': obs[8],
                'next_indoor_temp': next_obs[6],
                'total_power_demand': info.get('total_power_demand', 0),
                'phase': 'clue_control',
                'week': week_num,
                'day_in_week': day_in_week,
                'temp_change': next_obs[6] - obs[6],
                'obs': obs.copy(),
                'next_obs': next_obs.copy(),
                'continuous_action': new_action_mapping(action)
            }
            control_data.append(control_datum)
            
        else:
            # WEEKEND: Random Exploration
            action = random_controller(obs, info, args)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = step_env(env, action)
            
            # Store exploration data
            exploration_datum = {
                'step': step_count,
                'hour': info['hour'],
                'day': info['day'],
                'month': info['month'],
                'action': action,
                'reward': reward,
                'indoor_temp': obs[6],
                'outdoor_temp': obs[1],
                'occupancy': obs[8],
                'next_indoor_temp': next_obs[6],
                'total_power_demand': info.get('total_power_demand', 0),
                'phase': 'weekend_exploration',
                'week': week_num,
                'day_in_week': day_in_week,
                'temp_change': next_obs[6] - obs[6],
                'obs': obs.copy(),
                'next_obs': next_obs.copy(),
                'continuous_action': new_action_mapping(action)
            }
            exploration_data.append(exploration_datum)
        
        obs = next_obs
        step_count += 1
        
        # Print daily progress
        if step_count % (24 * args.timestep) == 0:
            day_completed = step_count // (24 * args.timestep)
            day_type = "Weekday (Control)" if is_weekday(day_completed - 1) else "Weekend (Exploration)"
            print(f"  Week {week_num}, Day {day_completed}/7 ({day_type}) completed")
            
        if terminated or truncated:
            print(f"Environment terminated during week {week_num}")
            break
    
    # Calculate weekly fallback rate for control days only
    week_fallback_rate = np.mean(all_fallback_flags) if all_fallback_flags else 0.0
    
    print(f"Week {week_num} completed:")
    print(f"  Control data points: {len(control_data)} (weekdays)")
    print(f"  Exploration data points: {len(exploration_data)} (weekends)")
    print(f"  Weekly fallback rate: {week_fallback_rate:.3f}")
    
    return control_data, exploration_data

def plot_weekly_results(all_week_data, weekly_metrics, obs_mean, obs_var, args, folders, mode, timestamp, total_weeks=4):
    """
    Create comprehensive visualization of multi-week experiment results
    
    Args:
        all_week_data: Dictionary with data for each week
        weekly_metrics: List of weekly metrics
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
        folders: Dictionary with folder paths
        mode: 'winter' or 'summer'
        timestamp: Timestamp string
        total_weeks: Total number of weeks
    """
    print("="*60)
    print("Step 5: Creating comprehensive multi-week visualization")
    print("="*60)
    
    # Denormalize temperatures for plotting
    temp_mean = obs_mean[6]
    temp_std = math.sqrt(obs_var[6])
    
    # Create figure with 6 subplots
    fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, 1, figsize=(20, 24))
    
    # Colors for different weeks and phases - generate enough colors for any number of weeks
    base_colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    week_colors = (base_colors * ((total_weeks // len(base_colors)) + 1))[:total_weeks]
    
    # Add comfort zone shading for occupied hours (8 AM to 6 PM)
    comfort_lower = 23  # Lower bound of comfort zone
    comfort_upper = 26  # Upper bound of comfort zone
    occupied_start = 8  # 8 AM
    occupied_end = 18   # 6 PM
    
    # Plot data for each week
    for week_num in range(1, total_weeks + 1):
        week_data = all_week_data.get(week_num, [])
        if not week_data:
            continue
            
        # Extract data for plotting
        indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in week_data]
        outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in week_data]
        actions = [d['action'] for d in week_data]
        power = [d['total_power_demand'] for d in week_data]
        
        # Create time indices (each week is 0-7 days, offset by week number)
        time_offset = (week_num - 1) * 8  # 1 day gap between weeks for clarity
        time = np.arange(len(week_data)) / (args.timestep * 24) + time_offset
        
        # Determine phase colors and labels
        if week_num == 1:
            color = week_colors[0]
            label = f'Week {week_num}: Random Data'
            alpha = 0.8
        else:
            # Separate control and exploration data
            control_data = [d for d in week_data if d.get('phase') == 'clue_control']
            exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
            
            if control_data:
                control_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in control_data]
                control_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in control_data]
                control_actions = [d['action'] for d in control_data]
                control_power = [d['total_power_demand'] for d in control_data]
                control_time = np.array([d['step'] for d in control_data]) / (args.timestep * 24) + time_offset
                
                # Plot control data
                ax1.plot(control_time, control_indoor_temps, color=week_colors[week_num-1], 
                        label=f'Week {week_num}: CLUE Control', alpha=0.8, linewidth=1.5, linestyle='-')
                ax1.plot(control_time, control_outdoor_temps, color='orange', alpha=0.4, linewidth=1)
                ax2.plot(control_time, control_actions, color=week_colors[week_num-1], alpha=0.8, linewidth=1.5, linestyle='-')
                ax3.plot(control_time, control_power, color=week_colors[week_num-1], alpha=0.8, linewidth=1.5, linestyle='-')
            
            if exploration_data:
                exploration_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in exploration_data]
                exploration_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in exploration_data]
                exploration_actions = [d['action'] for d in exploration_data]
                exploration_power = [d['total_power_demand'] for d in exploration_data]
                exploration_time = np.array([d['step'] for d in exploration_data]) / (args.timestep * 24) + time_offset
                
                # Plot exploration data
                ax1.plot(exploration_time, exploration_indoor_temps, color=week_colors[week_num-1], 
                        label=f'Week {week_num}: Weekend Random Exploration', alpha=0.6, linewidth=1.5, linestyle='--')
                ax1.plot(exploration_time, exploration_outdoor_temps, color='orange', alpha=0.4, linewidth=1)
                ax2.plot(exploration_time, exploration_actions, color=week_colors[week_num-1], alpha=0.6, linewidth=1.5, linestyle='--')
                ax3.plot(exploration_time, exploration_power, color=week_colors[week_num-1], alpha=0.6, linewidth=1.5, linestyle='--')
            
            continue  # Skip the general plotting below for weeks 2+
        
        # Plot week 1 data
        ax1.plot(time, indoor_temps, color=color, label=label, alpha=alpha, linewidth=2)
        ax1.plot(time, outdoor_temps, color='orange', alpha=0.6, linewidth=1)
        ax2.plot(time, actions, color=color, label=label, alpha=alpha, linewidth=2)
        ax3.plot(time, power, color=color, label=label, alpha=alpha, linewidth=2)
    
    # Add comfort zone shading for each week
    comfort_zone_labeled = False
    for week_num in range(1, total_weeks + 1):
        time_offset = (week_num - 1) * 8
        for day in range(7):
            occupied_day_start = time_offset + day + occupied_start / 24
            occupied_day_end = time_offset + day + occupied_end / 24
            
            label = 'Comfort Zone (8AM-6PM)' if not comfort_zone_labeled else ""
            ax1.fill_between([occupied_day_start, occupied_day_end], 
                           comfort_lower, comfort_upper, 
                           color='lightgreen', alpha=0.3, 
                           label=label)
            comfort_zone_labeled = True
    
    # Add vertical lines to separate weeks
    for week in range(1, total_weeks):
        week_start = week * 8
        for ax in [ax1, ax2, ax3, ax4, ax5, ax6]:
            ax.axvline(x=week_start, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    # Configure axes
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title(f'{total_weeks}-Week RANDOM HVAC Control Baseline Results - Weekly Reset Strategy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_ylabel('Action (0-9)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 9.5)
    
    ax3.set_ylabel('Total Power Demand (W)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot weekly average power comparison
    weeks = [w['week'] for w in weekly_metrics]
    energies = [w['energy_kwh'] for w in weekly_metrics]
    
    bars = ax4.bar(weeks, energies, alpha=0.7, color=week_colors[:len(weeks)])
    ax4.set_ylabel('Weekly Energy (kWh)')
    ax4.set_xlabel('Week Number')
    ax4.set_xticks(weeks)
    ax4.set_xticklabels([f'Week {w}' for w in weeks])
    ax4.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars, energies):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Plot comfort violation rates comparison (weekday control vs weekend exploration)
    comfort_bounds = (20, 24) if args.winter else (23, 26)
    
    weekday_violations = []
    weekend_violations = []
    weeks_with_control = []
    
    for week_num in range(1, total_weeks + 1):
        week_data = all_week_data.get(week_num, [])
        if not week_data:
            continue
            
        if week_num == 1:
            # Week 1 is all rule-based, calculate as baseline
            week_violations = calculate_comfort_violations(week_data, obs_mean, obs_var, comfort_bounds)
            weekday_violations.append(week_violations['violation_rate'] * 100)
            weekend_violations.append(week_violations['violation_rate'] * 100)
            weeks_with_control.append(week_num)
        else:
            # Separate weekday control and weekend exploration
            control_data = [d for d in week_data if d.get('phase') == 'clue_control']
            exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
            
            if control_data:
                control_violations = calculate_comfort_violations(control_data, obs_mean, obs_var, comfort_bounds)
                weekday_violations.append(control_violations['violation_rate'] * 100)
            else:
                weekday_violations.append(0)
                
            if exploration_data:
                exploration_violations = calculate_comfort_violations(exploration_data, obs_mean, obs_var, comfort_bounds)
                weekend_violations.append(exploration_violations['violation_rate'] * 100)
            else:
                weekend_violations.append(0)
                
            weeks_with_control.append(week_num)
    
    # Create grouped bar chart for comfort violations
    x = np.arange(len(weeks_with_control))
    width = 0.35
    
    bars1 = ax5.bar(x - width/2, weekday_violations, width, label='Weekday Control', alpha=0.7, color='red')
    bars2 = ax5.bar(x + width/2, weekend_violations, width, label='Weekend Random Exploration', alpha=0.7, color='green')
    
    ax5.set_ylabel('Comfort Violation Rate (%)')
    ax5.set_xlabel('Week Number')
    ax5.set_title('Weekday Control vs Weekend Random Exploration Comfort Violations')
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'Week {w}' for w in weeks_with_control])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add value labels on violation bars
    for bar, value in zip(bars1, weekday_violations):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
                
    for bar, value in zip(bars2, weekend_violations):
        height = bar.get_height()
        ax5.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Plot fallback rates comparison (weekday control vs weekend exploration)
    weekday_fallback_rates = []
    weekend_fallback_rates = []
    
    for week_num in range(1, total_weeks + 1):
        week_data = all_week_data.get(week_num, [])
        if not week_data:
            continue
            
        if week_num == 1:
            # Week 1 is all random actions, so fallback rate is 100% (since it's using random controller)
            weekday_fallback_rates.append(100.0)
            weekend_fallback_rates.append(100.0)
        else:
            # Separate weekday control and weekend exploration
            control_data = [d for d in week_data if d.get('phase') == 'clue_control']
            exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
            
            # Calculate weekday control fallback rate
            if control_data:
                control_fallbacks = sum(1 for d in control_data if d.get('is_fallback', False))
                control_fallback_rate = (control_fallbacks / len(control_data)) * 100
                weekday_fallback_rates.append(control_fallback_rate)
            else:
                weekday_fallback_rates.append(0)
                
            # Calculate weekend exploration fallback rate
            if exploration_data:
                exploration_fallbacks = sum(1 for d in exploration_data if d.get('is_fallback', False))
                exploration_fallback_rate = (exploration_fallbacks / len(exploration_data)) * 100
                weekend_fallback_rates.append(exploration_fallback_rate)
            else:
                weekend_fallback_rates.append(0)
    
    # Create grouped bar chart for fallback rates
    bars3 = ax6.bar(x - width/2, weekday_fallback_rates, width, label='Weekday Control', alpha=0.7, color='purple')
    bars4 = ax6.bar(x + width/2, weekend_fallback_rates, width, label='Weekend Exploration', alpha=0.7, color='orange')
    
    ax6.set_ylabel('Fallback Rate (%)')
    ax6.set_xlabel('Week Number')
    ax6.set_title('Weekday Control vs Weekend Exploration Fallback Rates')
    ax6.set_xticks(x)
    ax6.set_xticklabels([f'Week {w}' for w in weeks_with_control])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # Add value labels on fallback bars
    for bar, value in zip(bars3, weekday_fallback_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
                
    for bar, value in zip(bars4, weekend_fallback_rates):
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{folders['figures']}/{total_weeks}week_random_results_{mode}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"{total_weeks}-week random baseline results plot saved: {filename}")

def save_multi_week_results(all_week_data, all_exploration_data, weekly_metrics, truth_table, obs_mean, obs_var, args, folders, mode, timestamp, total_weeks=4):
    """
    Save all multi-week experiment results to files with organized structure
    
    Args: all_week_data: Dictionary with data for each week
        all_exploration_data: All exploration data across weeks
        weekly_metrics: Weekly performance metrics
        truth_table: Truth table
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
        folders: Dictionary with organized folder paths
        mode: 'winter' or 'summer'
        timestamp: Timestamp string
        total_weeks: Total number of weeks
    """
    print("="*60)
    print(f"Step 6: Saving {total_weeks}-week CLUE experiment results")
    print("="*60)
    
    # Determine comfort bounds based on season
    comfort_bounds = (20, 24) if args.winter else (23, 26)
    
    # Print weekly performance progression
    print("\n📊 WEEKLY PERFORMANCE PROGRESSION:")
    print("="*50)
    for week_metrics in weekly_metrics:
        week = week_metrics['week']
        energy = week_metrics['energy_kwh']
        violation_rate = week_metrics['comfort_violations']['violation_rate'] * 100
        violation_hours = week_metrics['comfort_violations']['violation_hours']
        
        if week == 1:
            phase_desc = "Random Action Data Collection"
            print(f"Week {week} ({phase_desc}):")
            print(f"  🔋 Energy: {energy:.2f} kWh")
            print(f"  🌡️  Comfort violations: {violation_rate:.1f}% ({violation_hours:.1f} hours)")
        else:
            phase_desc = "CLUE Control + Weekend Random Exploration"
            week_data = all_week_data[week]
            
            # Calculate separate violations for control vs exploration
            control_data = [d for d in week_data if d.get('phase') == 'clue_control']
            exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
            
            control_violations = calculate_comfort_violations(control_data, obs_mean, obs_var, comfort_bounds) if control_data else {'violation_rate': 0, 'violation_hours': 0}
            exploration_violations = calculate_comfort_violations(exploration_data, obs_mean, obs_var, comfort_bounds) if exploration_data else {'violation_rate': 0, 'violation_hours': 0}
            
            print(f"Week {week} ({phase_desc}):")
            print(f"  🔋 Energy: {energy:.2f} kWh")
            print(f"  🌡️  Overall comfort violations: {violation_rate:.1f}% ({violation_hours:.1f} hours)")
            print(f"      📈 Weekday Control: {control_violations['violation_rate']*100:.1f}% ({control_violations['violation_hours']:.1f} hours)")
            print(f"      📉 Weekend Random Exploration: {exploration_violations['violation_rate']*100:.1f}% ({exploration_violations['violation_hours']:.1f} hours)")
        print()
    
    # Calculate overall improvement
    if len(weekly_metrics) >= 2:
        week1_energy = weekly_metrics[0]['energy_kwh']
        week_last_energy = weekly_metrics[-1]['energy_kwh']
        week1_violations = weekly_metrics[0]['comfort_violations']['violation_rate'] * 100
        week_last_violations = weekly_metrics[-1]['comfort_violations']['violation_rate'] * 100
        
        total_energy_change = ((week_last_energy - week1_energy) / week1_energy * 100)
        total_violation_change = week_last_violations - week1_violations
        
        print(f"🎯 OVERALL IMPROVEMENT (Week 1 → Week {total_weeks}):")
        print(f"  Energy consumption: {total_energy_change:+.1f}%")
        print(f"  Comfort violations: {total_violation_change:+.1f}%")
        print()
    
    # Save exploration data (collected exploration data folder)
    exploration_df = pd.DataFrame(all_exploration_data)
    exploration_file = f"{folders['collected_exploration_data']}/{total_weeks}week_random_exploration_data_{mode}_{timestamp}.csv"
    exploration_df.to_csv(exploration_file, index=False)
    
    # Save control and exploration data separately for each week
    all_data = []
    for week_num, week_data in all_week_data.items():
        all_data.extend(week_data)
        
        if week_num == 1:
            # Week 1 is all random action data
            week1_df = pd.DataFrame(week_data)
            week1_file = f"{folders['data']}/week_{week_num}_random_data_{mode}_{timestamp}.csv"
            week1_df.to_csv(week1_file, index=False)
        else:
            # Weeks 2+ have control and exploration phases
            control_data = [d for d in week_data if d.get('phase') == 'clue_control']
            exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
            
            if control_data:
                control_df = pd.DataFrame(control_data)
                control_file = f"{folders['data']}/week_{week_num}_control_data_{mode}_{timestamp}.csv"
                control_df.to_csv(control_file, index=False)
                
            if exploration_data:
                exploration_df = pd.DataFrame(exploration_data)
                exploration_file = f"{folders['data']}/week_{week_num}_exploration_data_{mode}_{timestamp}.csv"
                exploration_df.to_csv(exploration_file, index=False)
    
    # Save all weekly data combined (data folder)
    all_data_df = pd.DataFrame(all_data)
    all_data_file = f"{folders['data']}/{total_weeks}week_random_all_data_{mode}_{timestamp}.csv"
    all_data_df.to_csv(all_data_file, index=False)
    
    # Save weekly metrics (data folder)
    weekly_df = pd.DataFrame(weekly_metrics)
    weekly_file = f"{folders['data']}/{total_weeks}week_random_weekly_metrics_{mode}_{timestamp}.csv"
    weekly_df.to_csv(weekly_file, index=False)
    
    # Save truth table (data folder)
    truth_file = f"{folders['data']}/truth_table_{mode}_{timestamp}.csv"
    truth_table.to_csv(truth_file, index=False)
    
    # Save normalization parameters (data folder)
    norm_params = {
        'obs_mean': obs_mean,
        'obs_var': obs_var,
        'temperature_mean': obs_mean[6],
        'temperature_var': obs_var[6]
    }
    norm_file = f"{folders['data']}/{total_weeks}week_random_normalization_params_{mode}_{timestamp}.csv"
    pd.DataFrame([norm_params]).to_csv(norm_file, index=False)
    
    # Save comprehensive metrics (data folder)
    metrics_file = f"{folders['data']}/{total_weeks}week_random_metrics_{mode}_{timestamp}.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"{total_weeks}-Week RANDOM HVAC Control Baseline Experiment Results (v0.2)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Experiment structure (weekly resets to 7.1-7.7):\n")
        f.write(f"  Week 1: 7 days random action data collection\n")
        f.write(f"  Weeks 2-{total_weeks}: 5 weekdays CLUE control + 2 weekends random exploration + retrain GP\n")
        f.write(f"Environment: {args.environment}\n")
        f.write(f"Total experiment weeks: {total_weeks}\n")
        f.write(f"GP training: Only exploration data (random actions)\n")
        f.write(f"\n📊 WEEKLY PERFORMANCE PROGRESSION:\n")
        f.write(f"{'='*50}\n")
        
        for week_metrics in weekly_metrics:
            week = week_metrics['week']
            energy = week_metrics['energy_kwh']
            violation_rate = week_metrics['comfort_violations']['violation_rate'] * 100
            violation_hours = week_metrics['comfort_violations']['violation_hours']
            
            if week == 1:
                phase_desc = "Random Action Data Collection"
                f.write(f"Week {week} ({phase_desc}):\n")
                f.write(f"  Energy: {energy:.2f} kWh\n")
                f.write(f"  Comfort violations: {violation_rate:.1f}% ({violation_hours:.1f} hours)\n")
            else:
                phase_desc = "CLUE Control + Weekend Random Exploration"
                week_data = all_week_data[week]
                
                # Calculate separate violations for control vs exploration
                control_data = [d for d in week_data if d.get('phase') == 'clue_control']
                exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
                
                control_violations = calculate_comfort_violations(control_data, obs_mean, obs_var, comfort_bounds) if control_data else {'violation_rate': 0, 'violation_hours': 0}
                exploration_violations = calculate_comfort_violations(exploration_data, obs_mean, obs_var, comfort_bounds) if exploration_data else {'violation_rate': 0, 'violation_hours': 0}
                
                f.write(f"Week {week} ({phase_desc}):\n")
                f.write(f"  Energy: {energy:.2f} kWh\n")
                f.write(f"  Overall comfort violations: {violation_rate:.1f}% ({violation_hours:.1f} hours)\n")
                f.write(f"    Weekday Control: {control_violations['violation_rate']*100:.1f}% ({control_violations['violation_hours']:.1f} hours)\n")
                f.write(f"    Weekend Random Exploration: {exploration_violations['violation_rate']*100:.1f}% ({exploration_violations['violation_hours']:.1f} hours)\n")
            f.write(f"\n")
        
        # Overall improvement
        if len(weekly_metrics) >= 2:
            week1_energy = weekly_metrics[0]['energy_kwh']
            week_last_energy = weekly_metrics[-1]['energy_kwh']
            week1_violations = weekly_metrics[0]['comfort_violations']['violation_rate'] * 100
            week_last_violations = weekly_metrics[-1]['comfort_violations']['violation_rate'] * 100
            
            total_energy_change = ((week_last_energy - week1_energy) / week1_energy * 100)
            total_violation_change = week_last_violations - week1_violations
            
            f.write(f"OVERALL IMPROVEMENT (Week 1 → Week {total_weeks}):\n")  
            f.write(f"  Energy consumption: {total_energy_change:+.1f}%\n")
            f.write(f"  Comfort violations: {total_violation_change:+.1f}%\n")
        
        f.write(f"\nData Collection Summary:\n")
        f.write(f"  Total exploration data points: {len(all_exploration_data)}\n")
        f.write(f"  Total data points: {len(all_data)}\n")
        f.write(f"  Weekly breakdown:\n")
        for week_num in range(1, total_weeks + 1):
            week_data = all_week_data[week_num]
            if week_num == 1:
                f.write(f"    Week {week_num}: {len(week_data)} random action data points\n")
            else:
                control_data = [d for d in week_data if d.get('phase') == 'clue_control']
                exploration_data = [d for d in week_data if d.get('phase') == 'weekend_exploration']
                f.write(f"    Week {week_num}: {len(control_data)} control + {len(exploration_data)} random exploration = {len(week_data)} total\n")
        
        f.write(f"\nHyperparameters:\n")
        f.write(f"  MPPI Horizon: {MPPI_HORIZON}\n")
        f.write(f"  MPPI Samples: {MPPI_NUM_SAMPLES}\n")
        f.write(f"  MPPI Gamma: {MPPI_GAMMA}\n")
        f.write(f"  MPPI Lambda: {MPPI_LAMBDA_UNCERTAINTY}\n")
        f.write(f"  MPPI Eta: {MPPI_ETA}\n")
        f.write(f"  MPPI Uncertainty Threshold: {MPPI_UNCERTAINTY_THRESHOLD}\n")
    
    print(f"Results saved:")
    print(f"  All exploration data: {exploration_file}")
    print(f"  Individual weekly files:")
    for week_num in range(1, total_weeks + 1):
        if week_num == 1:
            print(f"    Week {week_num}: random action data")
        else:
            print(f"    Week {week_num}: control + random exploration data (separate files)")
    print(f"  All data combined: {all_data_file}")
    print(f"  Weekly metrics: {weekly_file}")
    print(f"  Truth table: {truth_file}")
    print(f"  Normalization parameters: {norm_file}")
    print(f"  Comprehensive metrics: {metrics_file}")

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_multi_week_experiment():
    """Main multi-week experiment function with weekly resets"""
    args = parse_args()
    
    # Create organized folder structure 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'winter' if args.winter else 'summer'
    folders = create_organized_results_folder('random_weekly_reset', timestamp, args.weather)
    
    print("="*60)
    print(f"{TOTAL_EXPERIMENT_WEEKS}-WEEK RANDOM HVAC Control Baseline Experiment v0.2")
    print("="*60)
    print(f"Mode: {'Winter' if args.winter else 'Summer'}")
    print(f"Experiment structure (weekly resets to 7.1-7.7):")
    print(f"  Week 1: 7 days random action data collection")
    print(f"  Weeks 2-{TOTAL_EXPERIMENT_WEEKS}: 5 weekdays CLUE control + 2 weekends random exploration + retrain GP")
    print(f"Environment: {args.environment}")
    if args.weather:
        print(f"Weather file: {args.weather}")
    print(f"Total experiment weeks: {TOTAL_EXPERIMENT_WEEKS}")
    print(f"GP training: Only exploration data (random actions)")
    print(f"Results folder: {folders['run_folder']}")
    print("="*60)
    
    try:
        # Step 1: Create and calibrate environment
        env, obs_mean, obs_var = create_environment(args, use_default_controller=False)
        
        # Step 2: Collect truth table (environmental data for 1 week)
        truth_table = collect_truth_table(env, args)
        
        # Step 3: Collect Week 1 rule-based data
        week1_data = collect_initial_week_data(env, truth_table, args, week_num=1)
        
        # Initialize data storage
        all_week_data = {1: week1_data}
        all_exploration_data = week1_data.copy()  # Week 1 is all exploration data
        
        # Step 4: Loop through remaining weeks (2 to TOTAL_EXPERIMENT_WEEKS)
        for week_num in range(2, TOTAL_EXPERIMENT_WEEKS + 1):
            print(f"\n{'='*60}")
            print(f"STARTING WEEK {week_num}")
            print(f"{'='*60}")
            
            # Train GP model with all accumulated exploration data
            print(f"Training GP with {len(all_exploration_data)} exploration examples...")
            gp_model, gp_wrapper = train_gp_model(all_exploration_data, obs_mean, obs_var)
            
            # Run control+exploration week
            control_data, exploration_data = run_control_exploration_week(
                env, gp_wrapper, truth_table, all_exploration_data, obs_mean, obs_var, args, week_num
            )
            
            # Store week data
            week_data = control_data + exploration_data
            all_week_data[week_num] = week_data
            
            # Add exploration data to accumulation (for next GP training)
            all_exploration_data.extend(exploration_data)
            
            print(f"Week {week_num} completed. Total exploration data: {len(all_exploration_data)}")
        
        # Calculate weekly metrics for all weeks
        comfort_bounds = (20, 24) if args.winter else (23, 26)
        weekly_metrics = []
        
        for week_num in range(1, TOTAL_EXPERIMENT_WEEKS + 1):
            week_data = all_week_data[week_num]
            energy = calculate_energy_consumption(week_data, obs_mean, obs_var)
            comfort = calculate_comfort_violations(week_data, obs_mean, obs_var, comfort_bounds)
            
            weekly_metrics.append({
                'week': week_num,
                'energy_kwh': energy,
                'comfort_violations': comfort,
                'num_data_points': len(week_data)
            })
        
        # Step 5: Create visualizations
        plot_weekly_results(all_week_data, weekly_metrics, obs_mean, obs_var, args, folders, mode, timestamp, TOTAL_EXPERIMENT_WEEKS)
        
        # Step 6: Save results
        save_multi_week_results(all_week_data, all_exploration_data, weekly_metrics, truth_table, obs_mean, obs_var, args, folders, mode, timestamp, TOTAL_EXPERIMENT_WEEKS)
        
        print("="*60)
        print(f"{TOTAL_EXPERIMENT_WEEKS}-WEEK RANDOM BASELINE EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Week 1 data: {len(week1_data)} random action samples")
        for week_num in range(2, TOTAL_EXPERIMENT_WEEKS + 1):
            week_data = all_week_data[week_num]
            control_count = len([d for d in week_data if d.get('phase') == 'clue_control'])
            exploration_count = len([d for d in week_data if d.get('phase') == 'weekend_exploration'])
            print(f"Week {week_num}: {control_count} control + {exploration_count} random exploration = {len(week_data)} samples")
        print(f"Total exploration data: {len(all_exploration_data)} random action samples")
        print("="*60)
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    run_multi_week_experiment() 