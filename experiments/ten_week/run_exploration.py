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
from exploration_mppi.controller import ExplorationMPPIController
from exploration_mppi.zdataset import ZDataset, cold_start_populate_dataset, accept_operational_z_points, purge_type2_by_uncertainty

# ============================================================================
# HYPERPARAMETER SPACE - MODIFY THESE FOR TUNING
# ============================================================================

# Environment parameters
CALIBRATION_EPISODES = 5
CALIBRATION_STEPS_PER_EPISODE = 100

# GP hyperparameters
GP_PREDICT_DELTA = True
GP_SAFETY_THRESHOLD = 0.0

# MPPI hyperparameters (for control phase)
MPPI_HORIZON = 4
MPPI_NUM_SAMPLES = 100
MPPI_GAMMA = 0.85
MPPI_LAMBDA_UNCERTAINTY = 1e-2
MPPI_ETA = 1.0
MPPI_UNCERTAINTY_THRESHOLD = 0.6

# Exploration MPPI hyperparameters (for exploration phase)
EXPLORATION_MPPI_HORIZON = 2
EXPLORATION_MPPI_NUM_SAMPLES = 50
EXPLORATION_MPPI_GAMMA = 0.9
EXPLORATION_MPPI_LAMBDA_UNCERTAINTY = 1e-2
EXPLORATION_MPPI_ETA = 1.0
EXPLORATION_MPPI_UNCERTAINTY_THRESHOLD = None  # No filtering during exploration

# Z-Dataset parameters
Z_DATASET_MAX_SIZE_TYPE1 = 100
Z_DATASET_MAX_SIZE_TYPE2 = 1000000
# Note: get_z_targets() uses balanced 1:1 sampling by default
# This means exploration will use 2 * min(Type1_count, Type2_count) points total
# With 100 Type 1 points, exploration will use 200 points (100 Type 1 + 100 Type 2)

# Updated experiment parameters for 1-week simulation periods
TOTAL_EXPERIMENT_WEEKS = 10   # Week 1: rule-based, Weeks 2-4: control+exploration
WEEKDAY_CONTROL_DAYS = 5     # Monday-Friday CLUE control
WEEKEND_EXPLORATION_DAYS = 2 # Saturday-Sunday information gain exploration
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
    
    return best_action

def rule_based_controller(obs, info, args):
    """
    Simple rule-based controller that returns discrete actions based on occupancy
    
    Returns:
        int: Discrete action (0-9)
            - Action 9 during occupied hours (8-17)
            - Action 0 during unoccupied hours
    """
    hour = info.get('hour', 12)  # Default to noon if hour not available
    
    if OCCUPIED_START_HOUR <= hour <= OCCUPIED_END_HOUR:
        return OCCUPIED_ACTION  # Action 9 during occupied hours
    else:
        return UNOCCUPIED_ACTION  # Action 0 during unoccupied hours

def convert_dropped_pairs_to_z_points(dropped_pairs):
    """
    Convert MPPI dropped pairs to Z-dataset compatible format
    
    MPPI dropped pairs format: (observation_array, action_int) tuples
    Z-dataset format: numpy arrays of shape (10,) with [obs(9) + normalized_action(1)]
    
    Args:
        dropped_pairs: List of (observation_array, action_int) tuples from MPPI
        
    Returns:
        z_points: List of numpy arrays compatible with Z-dataset
    """
    z_points = []
    
    for i, dropped_pair in enumerate(dropped_pairs):
        try:
            # Handle tuple format: (observation_array, action_int)
            if isinstance(dropped_pair, (tuple, list)) and len(dropped_pair) == 2:
                obs_array, action_value = dropped_pair
                
                # Convert observation to numpy array
                obs = np.asarray(obs_array, dtype=np.float32)
                
                # Extract scalar action value
                if isinstance(action_value, (np.integer, int)):
                    action_int = int(action_value)
                elif isinstance(action_value, (np.floating, float)):
                    action_int = int(action_value)
                else:
                    action_int = int(action_value.item()) if hasattr(action_value, 'item') else int(action_value)
                
                # Validate observation shape (should be 9)
                if obs.shape != (9,):
                    print(f"Warning: Dropped pair {i} observation has shape {obs.shape}, expected (9,). Skipping.")
                    continue
                
                # Validate action range (should be 0-9)
                if not (0 <= action_int <= 9):
                    print(f"Warning: Dropped pair {i} action {action_int} out of range [0, 9]. Skipping.")
                    continue
                
                # Normalize action to [-1, 1] range: (action / 4.5) - 1
                normalized_action = (action_int / 4.5) - 1.0
                
                # Create z-point by concatenating obs + normalized_action
                z_point = np.concatenate([obs, [normalized_action]]).astype(np.float32)
                
                # Final validation
                if z_point.shape == (10,) and np.all(np.isfinite(z_point)):
                    z_points.append(z_point)
                else:
                    print(f"Warning: Dropped pair {i} resulted in invalid z_point (shape: {z_point.shape}, finite: {np.all(np.isfinite(z_point))}). Skipping.")
                    
            else:
                print(f"Warning: Dropped pair {i} has unexpected format {type(dropped_pair)} with length {len(dropped_pair) if hasattr(dropped_pair, '__len__') else 'unknown'}. Expected (obs, action) tuple. Skipping.")
                
        except Exception as e:
            print(f"Warning: Failed to convert dropped pair {i}: {e}")
    
    return z_points

def dummy_information_gain_function(state_action_pair, z_list):
    """Information gain function for exploration MPPI"""
    if not z_list:
        return 0.0
    
    # Simple distance-based information gain
    min_distance = float('inf')
    for z_point in z_list:
        distance = np.linalg.norm(state_action_pair - z_point)
        min_distance = min(min_distance, distance)
    
    # Higher gain for points farther from existing z points
    return max(0.0, 1.0 - min_distance)

# ============================================================================
# ENVIRONMENT AND DATA COLLECTION
# ============================================================================

def create_environment(args, use_default_controller=False):
    """
    Create and calibrate environment with normalization parameters
    
    Args:
        args: Command line arguments
        use_default_controller: If True, use empty action space for default controller
    
    Returns:
        env: Calibrated environment
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
    """
    print("="*60)
    print("Step 1: Creating and calibrating environment")
    print("="*60)
    
    # Set run period for one week (July 1-7 for summer, January 1-7 for winter)
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
    
        # Add weather file if specified
    if args.weather:
        env = gym.make(args.environment, weather_files=args.weather, config_params=extra_params)
    else:
        env = gym.make(args.environment, config_params=extra_params)
    
    # Only set action mapping if not using default controller
    if not use_default_controller:
        env.action_mapping = new_action_mapping
    
    print(f"Environment: {args.environment}")
    print(f"Action space: {env.action_space}")
    print(f"Run period: {extra_params['runperiod']}")
    
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
    Collect truth table (environmental data for one week)
    
    Returns:
        truth_table: DataFrame with environmental data for MPPI planning
    """
    print("="*60)
    print("Step 2: Collecting truth table (1-week environmental data)")
    print("="*60)
    
    truth_data = []
    obs, info = reset_env(env)
    step_count = 0
    
    # Run environment from reset to end to collect all environmental data
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
            print(f"  Day {day} collected, total steps: {step_count}")
    
    truth_table = pd.DataFrame(truth_data)
    print(f"Truth table collected: {len(truth_table)} timesteps (1 week)")
    
    return truth_table

def collect_initial_week_data(env, truth_table, args, week_num=1):
    """
    Collect Week 1 rule-based data (7 days, 7.1-7.7)
    Environment is reset to 7.1 at the beginning
    
    Args:
        env: Environment
        truth_table: DataFrame with environmental data
        args: Arguments
        week_num: Week number (should be 1)
        
    Returns:
        week_data: List of data from rule-based control
    """
    print("="*60)
    print(f"Step 3: Collecting Week {week_num} Rule-based Data (7.1-7.7)")
    print("="*60)
    
    # Reset environment to start of week (7.1)
    obs, info = reset_env(env)
    
    week_data = []
    step_count = 0
    total_week_steps = DAYS_PER_WEEK * 24 * args.timestep
    
    print(f"Running {DAYS_PER_WEEK} days of rule-based control...")
    
    while step_count < total_week_steps:
        # Use rule-based controller
        action = rule_based_controller(obs, info, args)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = step_env(env, action)
        
        # Store data
        week_data.append({
            'step': step_count,
            'week': week_num,
            'day_in_week': step_count // (24 * args.timestep) + 1,
            'hour': info['hour'],
            'day': info['day'],
            'month': info['month'],
            'action': action,
            'obs': obs.copy(),
            'next_obs': next_obs.copy(),
            'reward': reward,
            'indoor_temp': obs[6],
            'next_indoor_temp': next_obs[6],
            'temp_change': next_obs[6] - obs[6],
            'total_power_demand': info.get('total_power_demand', 0),
            'phase': 'rule_based_data',
            'is_fallback': False,
        })
        
        obs = next_obs
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Day {day}/7 completed")
            
        if terminated or truncated:
            print("Environment terminated during Week 1 data collection")
            break
    
    print(f"Week 1 rule-based data collected: {len(week_data)} examples")
    
    # Print action distribution
    actions_used = [d['action'] for d in week_data]
    print(f"Action distribution: {np.bincount(actions_used, minlength=10)}")
    
    return week_data

# ============================================================================
# GP MODEL TRAINING
# ============================================================================

def train_gp_model(training_data, obs_mean, obs_var):
    """
    Train GP model with normalized data, predicting temperature changes
    
    Args:
        training_data: List of training examples
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        
    Returns:
        gp_model: Trained GP model
        gp_wrapper: Wrapper function for MPPI controller
    """
    print("="*60)
    print("Step 4: Training GP model")
    print("="*60)
    
    # Extract data from training examples
    observations = []
    actions = []
    temperature_changes = []
    
    for example in training_data:
        obs = example['obs']
        action = example['action']
        temp_change = example['temp_change']
        
        # Normalize action to [-1, 1] range
        normalized_action = (action - 4.5) / 4.5
        
        observations.append(obs)
        actions.append(normalized_action)
        temperature_changes.append(temp_change)
    
    observations = np.array(observations)
    actions = np.array(actions).reshape(-1, 1)
    temperature_changes = np.array(temperature_changes)
    
    print(f"Training data shape: obs={observations.shape}, actions={actions.shape}, temp_changes={temperature_changes.shape}")
    print(f"Temperature change range: [{np.min(temperature_changes):.4f}, {np.max(temperature_changes):.4f}]")

    # Create and train GP model
    gp_model = HVACGaussianProcess(
        input_dim=10,  # 9 observations + 1 action
        predict_delta=GP_PREDICT_DELTA,
        safety_threshold=GP_SAFETY_THRESHOLD
    )

    # Train GP model
    gp_model.fit(observations, actions, temperature_changes)
    
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
    
    print("GP model training completed")
    return gp_model, gp_wrapper

# ============================================================================
# WEEKLY CONTROL AND EXPLORATION
# ============================================================================

def get_future_env_data(truth_table, current_step, horizon, args):
    """Get future environmental data for MPPI planning"""
    future_data = np.zeros((horizon, 9))  # 9 observation dimensions
    
    for t in range(horizon):
        step_idx = current_step + t
        if step_idx < len(truth_table):
            # Use observation data from truth table
            future_data[t] = truth_table.iloc[step_idx]['obs']
        else:
            # Repeat last available data
            future_data[t] = future_data[t-1] if t > 0 else truth_table.iloc[-1]['obs']
    
    return future_data

def run_control_exploration_week(env, gp_wrapper, z_dataset, truth_table, all_exploration_data, obs_mean, obs_var, args, week_num):
    """
    Run 1 week of mixed control/exploration with Z-dataset management
    Environment is reset to 7.1 at the beginning of each week
    
    Args:
        env: Environment
        gp_wrapper: GP wrapper function
        z_dataset: Z-dataset for information gain exploration
        truth_table: DataFrame with environmental data
        all_exploration_data: All accumulated exploration data
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        args: Arguments
        week_num: Week number (2, 3, 4)
        
    Returns:
        control_data: Control data from weekdays
        exploration_data: Exploration data from weekends
        updated_z_dataset: Updated Z-dataset after week
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
    all_dropped_pairs = []
    
    # Run through 1 week (7 days)
    total_week_steps = DAYS_PER_WEEK * 24 * args.timestep
    steps_per_day = 24 * args.timestep
    
    print(f"Running week {week_num}: {WEEKDAY_CONTROL_DAYS} weekdays control + {WEEKEND_EXPLORATION_DAYS} weekends exploration")
    
    while step_count < total_week_steps:
        current_day = step_count // steps_per_day + 1  # 1-7
        is_weekend = current_day > WEEKDAY_CONTROL_DAYS  # Days 6-7 are weekends
        
        if is_weekend:
            # WEEKEND: Information Gain Exploration using Exploration MPPI
            print(f"Day {current_day}: Running information gain exploration...")
            
            # Create exploration MPPI controller
            exploration_controller = ExplorationMPPIController(
                gp_model=gp_wrapper,
                z_dataset=z_dataset,
                information_gain_fn=dummy_information_gain_function,
                action_dim=1,
                horizon=EXPLORATION_MPPI_HORIZON,
                num_samples=EXPLORATION_MPPI_NUM_SAMPLES,
                gamma=EXPLORATION_MPPI_GAMMA,
                lambda_uncertainty=EXPLORATION_MPPI_LAMBDA_UNCERTAINTY,
                eta=EXPLORATION_MPPI_ETA,
                num_discrete_actions=10,
                uncertainty_threshold=EXPLORATION_MPPI_UNCERTAINTY_THRESHOLD,
                temp_norm_params=(temp_mean, temp_std),
                hour_norm_params=(hour_mean, hour_std)
            )
            
            # Get future environmental data for MPPI horizon
            future_env_data = get_future_env_data(truth_table, step_count, EXPLORATION_MPPI_HORIZON, args)
            
            # Get z_list for information gain computation with balanced 1:1 sampling
            # This will return 2 * min(Type1_count, Type2_count) points total
            z_targets, _ = z_dataset.get_z_targets(balanced_sampling=True)
            print(f"Exploration z_targets: {len(z_targets)} points (balanced 1:1 sampling)")
            
            # Plan action using exploration MPPI
            exploration_flag = 1.0  # Full exploration
            
            try:
                dropped_pairs, discrete_action, is_fallback = exploration_controller.plan(
                    np.array(obs), future_env_data, exploration_flag, z_targets
                )
                
                if is_fallback:
                    print(f"Step {step_count}: Using fallback action in exploration")
                    
            except Exception as e:
                print(f"Step {step_count}: Error in exploration MPPI, using rule-based fallback: {e}")
                discrete_action = rule_based_controller(obs, info, args)
                is_fallback = True
                dropped_pairs = []
                
            phase = 'weekend_exploration'
            
        else:
            # WEEKDAY: CLUE Control using standard MPPI
            # Get future environmental data for MPPI planning
            future_env_data = get_future_env_data(truth_table, step_count, MPPI_HORIZON, args)
            
            # Plan action using MPPI
            try:
                dropped_pairs, discrete_action, is_fallback = mppi_controller.plan(
                    np.array(obs), future_env_data
                )
                
                # Add dropped pairs to Z-dataset as Type 2 points
                if dropped_pairs:
                    # Debug: Check format of first dropped pair
                    if len(dropped_pairs) > 0:
                        sample_pair = dropped_pairs[0]
                        if isinstance(sample_pair, (tuple, list)) and len(sample_pair) == 2:
                            obs, action = sample_pair
                            print(f"Debug: First dropped pair format: (obs_shape={getattr(obs, 'shape', 'no shape')}, action_type={type(action)}, action_value={action})")
                        else:
                            print(f"Debug: First dropped pair type: {type(sample_pair)}, length: {len(sample_pair) if hasattr(sample_pair, '__len__') else 'unknown'}")
                    
                    # Convert dropped pairs to Z-dataset compatible format
                    z_points = convert_dropped_pairs_to_z_points(dropped_pairs)
                    
                    # Add to Z-dataset Type 2
                    if z_points:
                        points_added = accept_operational_z_points(z_dataset, z_points)
                        print(f"Added {points_added}/{len(dropped_pairs)} dropped pairs to Z-dataset Type 2")
                    else:
                        print(f"Warning: No valid z-points could be created from {len(dropped_pairs)} dropped pairs")
                
                if is_fallback:
                    print(f"Step {step_count}: Using fallback action in control")
                    
            except Exception as e:
                print(f"Step {step_count}: Error in MPPI planning, using rule-based fallback: {e}")
                discrete_action = rule_based_controller(obs, info, args)
                is_fallback = True
                dropped_pairs = []
                
            phase = 'clue_control'
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = step_env(env, discrete_action)
        
        # Store data in appropriate list
        data_entry = {
            'step': step_count,
            'week': week_num,
            'day_in_week': current_day,
            'hour': info['hour'],
            'day': info['day'],
            'month': info['month'],
            'action': discrete_action,
            'obs': obs.copy(),
            'next_obs': next_obs.copy(),
            'reward': reward,
            'indoor_temp': obs[6],
            'next_indoor_temp': next_obs[6],
            'temp_change': next_obs[6] - obs[6],
            'total_power_demand': info.get('total_power_demand', 0),
            'phase': phase,
            'is_fallback': is_fallback,
            'dropped_pairs_count': len(dropped_pairs),
        }
        
        if is_weekend:
            exploration_data.append(data_entry)
        else:
            control_data.append(data_entry)
        
        all_fallback_flags.append(is_fallback)
        all_dropped_pairs.extend(dropped_pairs)
        
        obs = next_obs
        step_count += 1
        
        # Daily progress and Z-dataset management
        if step_count % steps_per_day == 0:
            current_day_completed = step_count // steps_per_day
            print(f"  Day {current_day_completed}/7 completed")
            
            # If we just finished an exploration day, retrain GP and purge Z-dataset
            if current_day_completed > WEEKDAY_CONTROL_DAYS:  # Finished weekend exploration day
                print(f"  Day {current_day_completed} exploration completed. Retraining GP and purging Z-dataset...")
                
                # Retrain GP with all exploration data (including new exploration data from today)
                current_exploration_data = all_exploration_data + exploration_data
                print(f"  Retraining GP with {len(current_exploration_data)} exploration examples")
                
                try:
                    retrained_gp_model, retrained_gp_wrapper = train_gp_model(current_exploration_data, obs_mean, obs_var)
                    gp_wrapper = retrained_gp_wrapper  # Update for next day
                    print("  ✅ GP retrained successfully")
                    
                    # Purge Z-dataset Type 2 using GP uncertainty
                    removed_count = purge_type2_by_uncertainty(z_dataset, retrained_gp_model, MPPI_UNCERTAINTY_THRESHOLD)
                    print(f"  🗑️ Purged {removed_count} low-uncertainty points from Z-dataset Type 2")
                    
                except Exception as e:
                    print(f"  ❌ GP retraining or Z-dataset purging failed: {e}")
        
        if terminated or truncated:
            print("Environment terminated during week execution")
            break
    
    print(f"Week {week_num} completed:")
    print(f"  Control data: {len(control_data)} samples")
    print(f"  Exploration data: {len(exploration_data)} samples")
    print(f"  Total dropped pairs collected: {len(all_dropped_pairs)}")
    print(f"  Final Z-dataset state: {z_dataset}")
    
    # Show balanced sampling info
    balanced_targets, _ = z_dataset.get_z_targets(balanced_sampling=True)
    all_targets, _ = z_dataset.get_z_targets(balanced_sampling=False)
    print(f"  Z-dataset sampling: {len(balanced_targets)} balanced vs {len(all_targets)} total points")
    print(f"  Balanced ratio: {len(balanced_targets)} = 2 * min({z_dataset.num_points_type1}, {z_dataset.num_points_type2})")
    
    return control_data, exploration_data, z_dataset

# ============================================================================
# VISUALIZATION AND RESULTS
# ============================================================================

def plot_weekly_results(all_week_data, weekly_metrics, obs_mean, obs_var, args, folders, mode, timestamp, total_weeks):
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
    comfort_lower = 23 if not args.winter else 20  # Lower bound of comfort zone
    comfort_upper = 26 if not args.winter else 24  # Upper bound of comfort zone
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
            label = f'Week {week_num}: Rule-based Data'
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
                        label=f'Week {week_num}: Information Gain Control', alpha=0.8, linewidth=1.5, linestyle='-')
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
                        label=f'Week {week_num}: Weekend Information Gain Exploration', alpha=0.6, linewidth=1.5, linestyle='--')
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
    ax1.set_title(f'{total_weeks}-Week Information Gain Exploration HVAC Control Results - Weekly Reset Strategy')
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
    energies = [w['total_energy_kwh'] for w in weekly_metrics]
    
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
    bars2 = ax5.bar(x + width/2, weekend_violations, width, label='Weekend Exploration', alpha=0.7, color='green')
    
    ax5.set_ylabel('Comfort Violation Rate (%)')
    ax5.set_xlabel('Week Number')
    ax5.set_title('Weekday Control vs Weekend Exploration Comfort Violations')
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
            # Week 1 is all rule-based, so fallback rate is 100% (since it's using rule-based controller)
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
    weather_id = ""
    if args.weather:
        weather_id = f"_{args.weather[:6]}" if len(args.weather) >= 6 else f"_{args.weather}"
    
    filename = f"{folders['figures']}/{total_weeks}week_exploration_results_{mode}{weather_id}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Weekly results plot saved to {filename}")
    
    plt.close()

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
    
    # Temperature is at index 6 in observations
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

def save_multi_week_results(all_week_data, obs_mean, obs_var, args, folders, mode, timestamp, total_weeks):
    """
    Save all experiment data with organized structure and comprehensive metrics
    """
    print("="*60)
    print("Step 7: Saving multi-week results with comprehensive metrics")
    print("="*60)
    
    # Calculate comfort bounds
    comfort_bounds = (20, 24) if args.winter else (23, 26)
    
    # Save individual week data files
    for week_num in range(1, total_weeks + 1):
        week_data = all_week_data[week_num]
        
        # Create weather identifier for filenames
        weather_id = ""
        if args.weather:
            weather_id = f"_{args.weather[:6]}" if len(args.weather) >= 6 else f"_{args.weather}"
        
        if week_num == 1:
            # Week 1: rule-based data
            filename = f"{folders['collected_exploration_data']}/{total_weeks}week_exploration_rule_data_week{week_num}_{mode}{weather_id}_{timestamp}.csv"
            phase_description = "rule_based_data"
        else:
            # Week 2+: separate control and exploration data
            control_data = [d for d in week_data if d['phase'] == 'clue_control']
            exploration_data = [d for d in week_data if d['phase'] == 'weekend_exploration']
            
            # Save control data
            if control_data:
                control_filename = f"{folders['data']}/{total_weeks}week_exploration_control_data_week{week_num}_{mode}{weather_id}_{timestamp}.csv"
                control_df = pd.DataFrame(control_data)
                control_df.to_csv(control_filename, index=False)
                print(f"Week {week_num} control data saved to {control_filename}")
            
            # Save exploration data
            if exploration_data:
                exploration_filename = f"{folders['collected_exploration_data']}/{total_weeks}week_exploration_exploration_data_week{week_num}_{mode}{weather_id}_{timestamp}.csv"
                exploration_df = pd.DataFrame(exploration_data)
                exploration_df.to_csv(exploration_filename, index=False)
                print(f"Week {week_num} exploration data saved to {exploration_filename}")
            
            continue  # Skip the general week data saving for weeks 2+
        
        # Save week 1 data
        week_df = pd.DataFrame(week_data)
        week_df.to_csv(filename, index=False)
        print(f"Week {week_num} data saved to {filename}")
    
    # Calculate and save comprehensive metrics
    print("\n" + "="*50)
    print("COMPREHENSIVE EXPERIMENT METRICS")
    print("="*50)
    
    weekly_metrics = []
    total_energy = 0
    total_violations = 0
    total_occupied_hours = 0
    total_control_violations = 0
    total_control_occupied = 0
    total_exploration_violations = 0
    total_exploration_occupied = 0
    
    for week_num in range(1, total_weeks + 1):
        week_data = all_week_data[week_num]
        
        # Energy calculation
        week_energy = sum(d['total_power_demand'] for d in week_data) * 0.25 / 1000  # kWh
        total_energy += week_energy
        
        # Comfort violations during occupied hours
        week_violations = 0
        week_occupied = 0
        
        for d in week_data:
            hour = d['hour']
            if OCCUPIED_START_HOUR <= hour <= OCCUPIED_END_HOUR:
                week_occupied += 1
                temp = d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6]
                if temp < comfort_bounds[0] or temp > comfort_bounds[1]:
                    week_violations += 1
        
        week_violation_rate = (week_violations / week_occupied * 100) if week_occupied > 0 else 0
        total_violations += week_violations
        total_occupied_hours += week_occupied
        
        # Phase-specific metrics for weeks 2+
        control_violation_rate = 0
        exploration_violation_rate = 0
        
        if week_num > 1:
            # Control phase violations
            control_data = [d for d in week_data if d['phase'] == 'clue_control']
            control_violations = 0
            control_occupied = 0
            
            for d in control_data:
                hour = d['hour']
                if OCCUPIED_START_HOUR <= hour <= OCCUPIED_END_HOUR:
                    control_occupied += 1
                    temp = d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6]
                    if temp < comfort_bounds[0] or temp > comfort_bounds[1]:
                        control_violations += 1
            
            control_violation_rate = (control_violations / control_occupied * 100) if control_occupied > 0 else 0
            total_control_violations += control_violations
            total_control_occupied += control_occupied
            
            # Exploration phase violations
            exploration_data = [d for d in week_data if d['phase'] == 'weekend_exploration']
            exploration_violations = 0
            exploration_occupied = 0
            
            for d in exploration_data:
                hour = d['hour']
                if OCCUPIED_START_HOUR <= hour <= OCCUPIED_END_HOUR:
                    exploration_occupied += 1
                    temp = d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6]
                    if temp < comfort_bounds[0] or temp > comfort_bounds[1]:
                        exploration_violations += 1
            
            exploration_violation_rate = (exploration_violations / exploration_occupied * 100) if exploration_occupied > 0 else 0
            total_exploration_violations += exploration_violations
            total_exploration_occupied += exploration_occupied
        
        # Data point counts
        total_points = len(week_data)
        control_points = len([d for d in week_data if d.get('phase') == 'clue_control'])
        exploration_points = len([d for d in week_data if d.get('phase') in ['rule_based_data', 'weekend_exploration']])
        
        week_metrics = {
            'week': week_num,
            'total_energy_kwh': week_energy,
            'comfort_violation_rate_percent': week_violation_rate,
            'comfort_violations_count': week_violations,
            'occupied_hours': week_occupied * 0.25,  # Convert to hours
            'control_violation_rate_percent': control_violation_rate,
            'exploration_violation_rate_percent': exploration_violation_rate,
            'total_data_points': total_points,
            'control_data_points': control_points,
            'exploration_data_points': exploration_points
        }
        
        weekly_metrics.append(week_metrics)
        
        # Print week summary
        if week_num == 1:
            print(f"\n📊 WEEK {week_num} (Rule-based):")
        else:
            print(f"\n📊 WEEK {week_num} (Control + Exploration):")
        
        print(f"   Energy: {week_energy:.2f} kWh")
        print(f"   Comfort violations: {week_violation_rate:.1f}% ({week_violations}/{week_occupied} occupied hours)")
        print(f"   Data points: {total_points} total")
        
        if week_num > 1:
            print(f"   Control violations: {control_violation_rate:.1f}% ({control_points} points)")
            print(f"   Exploration violations: {exploration_violation_rate:.1f}% ({exploration_points} points)")
    
    # Overall experiment summary
    overall_violation_rate = (total_violations / total_occupied_hours * 100) if total_occupied_hours > 0 else 0
    overall_control_rate = (total_control_violations / total_control_occupied * 100) if total_control_occupied > 0 else 0
    overall_exploration_rate = (total_exploration_violations / total_exploration_occupied * 100) if total_exploration_occupied > 0 else 0
    
    print(f"\n🎯 OVERALL EXPERIMENT SUMMARY ({total_weeks} weeks):")
    print(f"   Total energy: {total_energy:.2f} kWh")
    print(f"   Overall comfort violations: {overall_violation_rate:.1f}% ({total_violations}/{total_occupied_hours} occupied hours)")
    print(f"   Control phase violations: {overall_control_rate:.1f}% ({total_control_violations}/{total_control_occupied} hours)")
    print(f"   Exploration phase violations: {overall_exploration_rate:.1f}% ({total_exploration_violations}/{total_exploration_occupied} hours)")
    print(f"   Total occupied hours: {total_occupied_hours * 0.25:.1f} hours")
    
    # Save weekly metrics to CSV
    metrics_df = pd.DataFrame(weekly_metrics)
    weather_id = ""
    if args.weather:
        weather_id = f"_{args.weather[:6]}" if len(args.weather) >= 6 else f"_{args.weather}"
    
    metrics_filename = f"{folders['data']}/{total_weeks}week_exploration_weekly_metrics_{mode}{weather_id}_{timestamp}.csv"
    metrics_df.to_csv(metrics_filename, index=False)
    print(f"\nWeekly metrics saved to {metrics_filename}")
    
    # Save comprehensive summary file
    weather_id = ""
    if args.weather:
        weather_id = f"_{args.weather[:6]}" if len(args.weather) >= 6 else f"_{args.weather}"
    
    summary_filename = f"{folders['data']}/{total_weeks}week_exploration_comprehensive_metrics_{mode}{weather_id}_{timestamp}.txt"
    
    with open(summary_filename, 'w') as f:
        f.write(f"{total_weeks}-WEEK INFORMATION GAIN EXPLORATION HVAC EXPERIMENT RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Experiment: Information Gain Exploration with Z-Dataset Management\n")
        f.write(f"Season: {'Winter' if args.winter else 'Summer'}\n")
        f.write(f"Total weeks: {total_weeks}\n")
        f.write(f"Environment: {args.environment}\n")
        if args.weather:
            f.write(f"Weather file: {args.weather}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Comfort bounds: {comfort_bounds[0]}-{comfort_bounds[1]}°C\n")
        f.write("\n")
        
        f.write("EXPERIMENT STRUCTURE:\n")
        f.write("  Week 1: 7 days rule-based data collection\n")
        f.write(f"  Weeks 2-{total_weeks}: 5 weekdays CLUE control + 2 weekends information gain exploration\n")
        f.write("  GP training: Only exploration data (no control data)\n")
        f.write("  Z-dataset: Information gain targets with uncertainty-based purging\n")
        f.write("\n")
        
        f.write("OVERALL PERFORMANCE:\n")
        f.write(f"  Total energy consumption: {total_energy:.2f} kWh\n")
        f.write(f"  Overall comfort violation rate: {overall_violation_rate:.2f}%\n")
        f.write(f"  Control phase violations: {overall_control_rate:.2f}%\n")
        f.write(f"  Exploration phase violations: {overall_exploration_rate:.2f}%\n")
        f.write(f"  Total experiment hours: {total_occupied_hours * 0.25:.1f} occupied hours\n")
        f.write("\n")
        
        f.write("WEEKLY BREAKDOWN:\n")
        for metrics in weekly_metrics:
            f.write(f"  Week {metrics['week']}:\n")
            f.write(f"    Energy: {metrics['total_energy_kwh']:.2f} kWh\n")
            f.write(f"    Comfort violations: {metrics['comfort_violation_rate_percent']:.1f}%\n")
            f.write(f"    Data points: {metrics['total_data_points']} total\n")
            if metrics['week'] > 1:
                f.write(f"    Control violations: {metrics['control_violation_rate_percent']:.1f}%\n")
                f.write(f"    Exploration violations: {metrics['exploration_violation_rate_percent']:.1f}%\n")
            f.write("\n")
        
        f.write("HYPERPARAMETERS:\n")
        f.write(f"  MPPI horizon: {MPPI_HORIZON}\n")
        f.write(f"  MPPI samples: {MPPI_NUM_SAMPLES}\n")
        f.write(f"  MPPI uncertainty threshold: {MPPI_UNCERTAINTY_THRESHOLD}\n")
        f.write(f"  Exploration MPPI horizon: {EXPLORATION_MPPI_HORIZON}\n")
        f.write(f"  Exploration MPPI samples: {EXPLORATION_MPPI_NUM_SAMPLES}\n")
        f.write(f"  Z-dataset Type 1 size: {Z_DATASET_MAX_SIZE_TYPE1}\n")
        f.write(f"  Z-dataset Type 2 size: {Z_DATASET_MAX_SIZE_TYPE2}\n")
    
    print(f"Comprehensive metrics saved to {summary_filename}")
    print("All results saved successfully!")

# ============================================================================
# FOLDER MANAGEMENT
# ============================================================================

def create_organized_results_folder(method_name, timestamp, weather_file=None):
    """
    Create organized folder structure for experiment results
    
    Args:
        method_name: Name of the method (e.g., 'exploration_weekly_reset')
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

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_multi_week_experiment():
    """Main multi-week experiment function with weekly resets and Z-dataset management"""
    args = parse_args()
    
    # Create organized folder structure 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'winter' if args.winter else 'summer'
    folders = create_organized_results_folder('exploration_weekly_reset', timestamp, args.weather)
    
    print("="*60)
    print(f"{TOTAL_EXPERIMENT_WEEKS}-WEEK INFORMATION GAIN EXPLORATION HVAC EXPERIMENT v0.2")
    print("="*60)
    print(f"Mode: {'Winter' if args.winter else 'Summer'}")
    print(f"Experiment structure (weekly resets to 7.1-7.7):")
    print(f"  Week 1: 7 days rule-based data collection")
    print(f"  Weeks 2-{TOTAL_EXPERIMENT_WEEKS}: 5 weekdays CLUE control + 2 weekends information gain exploration")
    print(f"Environment: {args.environment}")
    if args.weather:
        print(f"Weather file: {args.weather}")
    print(f"Total experiment weeks: {TOTAL_EXPERIMENT_WEEKS}")
    print(f"GP training: Only exploration data (no control data)")
    print(f"Z-dataset: Information gain targets with balanced 1:1 sampling + uncertainty-based purging")
    print(f"  Balanced sampling: 2 * min(Type1, Type2) points for exploration (target: 200 points)")
    print(f"Results folder: {folders['run_folder']}")
    print("="*60)
    
    try:
        # Step 1: Create and calibrate environment
        env, obs_mean, obs_var = create_environment(args, use_default_controller=False)
        
        # Step 2: Collect truth table (environmental data for 1 week)
        truth_table = collect_truth_table(env, args)
        
        # Step 3: Collect Week 1 rule-based data
        week1_data = collect_initial_week_data(env, truth_table, args, week_num=1)
        
        # Initialize Z-dataset with Type 1 (cold start) data
        z_dataset = ZDataset(
            max_size_type1=Z_DATASET_MAX_SIZE_TYPE1,
            max_size_type2=Z_DATASET_MAX_SIZE_TYPE2
        )
        
        # Populate Z-dataset with cold start data
        comfort_bounds = (20, 24) if args.winter else (23, 26)
        cold_start_populate_dataset(z_dataset, truth_table, obs_mean, obs_var, comfort_bounds)
        print(f"Z-dataset initialized with {z_dataset.num_points_type1} Type 1 points")
        print(f"Expected balanced sampling: {2 * z_dataset.num_points_type1} points (1:1 Type1:Type2 ratio)")
        
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
            
            # Run control+exploration week with Z-dataset management
            control_data, exploration_data, updated_z_dataset = run_control_exploration_week(
                env, gp_wrapper, z_dataset, truth_table, all_exploration_data, obs_mean, obs_var, args, week_num
            )
            
            # Update Z-dataset reference
            z_dataset = updated_z_dataset
            
            # Store week data
            week_data = control_data + exploration_data
            all_week_data[week_num] = week_data
            
            # Add exploration data to accumulation (for next GP training)
            all_exploration_data.extend(exploration_data)
            
            print(f"Week {week_num} completed. Total exploration data: {len(all_exploration_data)}")
            print(f"Final Z-dataset state: {z_dataset}")
        
        # Calculate weekly metrics for plotting
        comfort_bounds = (20, 24) if args.winter else (23, 26)
        weekly_metrics = []
        
        for week_num in range(1, TOTAL_EXPERIMENT_WEEKS + 1):
            week_data = all_week_data.get(week_num, [])
            
            # Energy calculation
            week_energy = sum(d['total_power_demand'] for d in week_data) * 0.25 / 1000  # kWh
            
            # Comfort violations
            week_violations = calculate_comfort_violations(week_data, obs_mean, obs_var, comfort_bounds)
            
            week_metrics = {
                'week': week_num,
                'total_energy_kwh': week_energy,
                'comfort_violation_rate_percent': week_violations['violation_rate'] * 100,
                'comfort_violations_count': week_violations['violation_count'],
                'occupied_hours': week_violations['total_hours'],
            }
            
            weekly_metrics.append(week_metrics)
        
        # Step 5: Plot weekly results
        plot_weekly_results(all_week_data, weekly_metrics, obs_mean, obs_var, args, folders, mode, timestamp, TOTAL_EXPERIMENT_WEEKS)
        
        # Step 6: Save comprehensive results
        save_multi_week_results(all_week_data, obs_mean, obs_var, args, folders, mode, timestamp, TOTAL_EXPERIMENT_WEEKS)
        
        print("\n" + "="*60)
        print(f"{TOTAL_EXPERIMENT_WEEKS}-WEEK INFORMATION GAIN EXPLORATION EXPERIMENT COMPLETED!")
        print("="*60)
        
        # Final summary
        total_control_points = sum(len([d for d in all_week_data[w] if d.get('phase') == 'clue_control']) 
                                  for w in range(2, TOTAL_EXPERIMENT_WEEKS + 1))
        total_exploration_points = sum(len([d for d in all_week_data[w] if d.get('phase') in ['rule_based_data', 'weekend_exploration']]) 
                                      for w in range(1, TOTAL_EXPERIMENT_WEEKS + 1))
        
        print(f"Final experiment summary:")
        if args.weather:
            print(f"  Weather file: {args.weather}")
        print(f"  Total control data points: {total_control_points}")
        print(f"  Total exploration data points: {total_exploration_points}")
        print(f"  GP training data: {len(all_exploration_data)} exploration examples")
        print(f"  Final Z-dataset: {z_dataset}")
        
        # Show final balanced sampling info
        balanced_targets, _ = z_dataset.get_z_targets(balanced_sampling=True)
        all_targets, _ = z_dataset.get_z_targets(balanced_sampling=False)
        print(f"  Final Z-dataset sampling: {len(balanced_targets)} balanced vs {len(all_targets)} total points")
        print(f"  Balanced sampling ratio: 1:1 (Type1:Type2) = {z_dataset.num_points_type1}:{z_dataset.num_points_type2}")
        print(f"  Results saved to: {folders['run_folder']}")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if 'env' in locals():
            env.close()

if __name__ == "__main__":
    run_multi_week_experiment() 