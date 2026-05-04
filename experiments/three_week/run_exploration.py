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
from exploration_mppi.zdataset import ZDataset, cold_start_populate_dataset

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
MPPI_HORIZON = 5
MPPI_NUM_SAMPLES = 100
MPPI_GAMMA = 0.85
MPPI_LAMBDA_UNCERTAINTY = 1e-2
MPPI_ETA = 1.0
MPPI_UNCERTAINTY_THRESHOLD = 0.6

# Exploration MPPI hyperparameters
EXPLORATION_MPPI_HORIZON = 2
EXPLORATION_MPPI_NUM_SAMPLES = 50
EXPLORATION_MPPI_GAMMA = 0.9
EXPLORATION_MPPI_LAMBDA_UNCERTAINTY = 1e-2
EXPLORATION_MPPI_ETA = 1.0
EXPLORATION_MPPI_UNCERTAINTY_THRESHOLD = None  # No filtering during exploration

# ZDataset parameters
Z_DATASET_MAX_SIZE_TYPE1 = 100
Z_DATASET_MAX_SIZE_TYPE2 = 50

# Control parameters
TRAINING_DAYS = 7      # Week 1: Rule-based training data collection
EXPLORATION_DAYS = 7   # Week 2: Exploration MPPI data collection
CONTROL_DAYS = 7       # Week 3: Standard MPPI evaluation
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

# ============================================================================
# MAIN EXPERIMENT FUNCTIONS
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
    
    # Set run period for full month (July for summer, January for winter)
    extra_params = {'timesteps_per_hour': 4}
    if args.winter:
        extra_params['runperiod'] = (1,1,1997,31,1,1997)  # January (winter)
    else:
        extra_params['runperiod'] = (1,7,1997,31,7,1997)  # July (summer)
    
    # Configure action space based on use_default_controller
    if use_default_controller:
        # Empty action space to use default rule-based controller
        extra_params['action_space'] = gym.spaces.Box(low=0, high=0, shape=(0,))
        print("Using default rule-based controller (empty action space)")
    
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
    Collect outdoor temperature and environmental data for the entire simulation period
    
    Returns:
        truth_table: DataFrame with environmental data for MPPI planning
    """
    print("="*60)
    print("Step 2: Collecting truth table (full simulation environmental data)")
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
    print(f"Truth table collected: {len(truth_table)} timesteps")
    
    return truth_table

# ============================================================================
# EXPLORATION TRAINING DATA COLLECTION
# ============================================================================

def collect_training_data(env, truth_table, args, use_default_controller=False):
    """
    Collect training data using either default controller or rule-based controller
    
    Args:
        env: Environment
        truth_table: DataFrame with environmental data
        args: Arguments
        use_default_controller: If True, use default controller (requires empty action space)
                               If False, use rule-based controller (requires discrete action space)
        
    Returns:
        training_data: List of training examples
        final_obs: Final observation after training period
        final_info: Final info after training period
    """
    print("="*60)
    controller_type = "default" if use_default_controller else "custom rule-based"
    print(f"Step 3: Collecting {TRAINING_DAYS} days of training data with {controller_type} controller")
    print("="*60)
    
    training_data = []
    obs, info = reset_env(env)
    step_count = 0
    
    # Calculate total steps for training period
    total_training_steps = TRAINING_DAYS * 24 * args.timestep
    
    while step_count < total_training_steps:
        # Use empty action to trigger default controller
        if use_default_controller:
            action = env.action_space.sample()
        else:
            action = rule_based_controller(obs, info, args)
        
        # Take step
        next_obs, reward, terminated, truncated, info = step_env(env, action)

        if use_default_controller:
            # Extract the actual action used by the default controller
            continuous_action = info['action']  # This should be [heating_setpoint, cooling_setpoint]
            
            # Convert continuous action to discrete action for GP training
            discrete_action = inverse_action_mapping(continuous_action)
        else:
            discrete_action = action
            # Convert discrete action back to continuous for consistency
            continuous_action = new_action_mapping(discrete_action)
        
        # Store training data
        training_data.append({
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
        })
        
        obs = next_obs
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Training day {day} completed, total steps: {step_count}")
            
        if terminated or truncated:
            print("Environment terminated during training collection")
            break
    
    print(f"Training data collected: {len(training_data)} examples")
    # Print some statistics about the actions used
    actions_used = [d['action'] for d in training_data]
    print(f"Action distribution: {np.bincount(actions_used, minlength=10)}")
    return training_data, obs, info


def collect_exploration_data(env, gp_wrapper, truth_table, training_data, obs_mean, obs_var, args):
    """
    Collect exploration data using exploration MPPI controller with daily GP retraining
    
    Args:
        env: Environment
        gp_wrapper: Trained GP model wrapper (will be updated daily)
        truth_table: DataFrame with environmental data
        training_data: Training data (for environment state continuation)
        obs_mean: Observation normalization mean
        obs_var: Observation normalization variance
        args: Arguments
        
    Returns:
        exploration_data: List of exploration examples
        final_obs: Final observation after exploration period
        final_info: Final info after exploration period
        z_dataset: ZDataset used for exploration
    """
    print("="*60)
    print(f"Step 5: Collecting {EXPLORATION_DAYS} days of exploration data with daily GP retraining")
    print("="*60)
    
    # Reset environment to continue from end of training period
    obs, info = reset_env(env)
    
    # Skip to end of training period
    training_steps = len(training_data)
    for i in range(training_steps):
        action = training_data[i]['action']
        obs, reward, terminated, truncated, info = step_env(env, action)
        if terminated or truncated:
            print("Environment terminated during skip to exploration period")
            break
    
    # Create and populate Z dataset for exploration
    z_dataset = ZDataset(
        max_size_type1=Z_DATASET_MAX_SIZE_TYPE1,
        max_size_type2=Z_DATASET_MAX_SIZE_TYPE2
    )
    
    # Populate with Type 1 (cold start) data
    # Determine comfort bounds based on season
    if args.winter:
        comfort_bounds = (20, 24)  # Winter comfort bounds
    else:
        comfort_bounds = (23, 26)  # Summer comfort bounds
    
    cold_start_populate_dataset(z_dataset, truth_table, obs_mean, obs_var, comfort_bounds)
    print(f"Created Z dataset with {z_dataset.num_points_type1} Type 1 exploration points")
    
    exploration_data = []
    step_count = training_steps
    
    # Calculate total steps for exploration period
    total_exploration_steps = EXPLORATION_DAYS * 24 * args.timestep
    steps_per_day = 24 * args.timestep
    
    print(f"Starting exploration data collection for {total_exploration_steps} steps")
    print(f"Will retrain GP every {steps_per_day} steps (1 day)")
    
    # Initialize current GP wrapper (will be updated daily)
    current_gp_wrapper = gp_wrapper
    
    exploration_step = 0
    while exploration_step < total_exploration_steps and not (terminated or truncated):
        # Check if we need to retrain GP (every day)
        if exploration_step > 0 and exploration_step % steps_per_day == 0:
            day_number = exploration_step // steps_per_day
            print(f"\n{'='*40}")
            print(f"Day {day_number}: Retraining GP with all historical data")
            print(f"{'='*40}")
            # Collect all historical data (training + exploration so far)
            all_historical_data = training_data + exploration_data
            print(f"Training data: {len(training_data)} samples")
            print(f"Exploration data so far: {len(exploration_data)} samples")
            print(f"Total historical data: {len(all_historical_data)} samples")
            
            # Retrain GP with all historical data
            try:
                retrained_gp_model, retrained_gp_wrapper = train_gp_model(all_historical_data, obs_mean, obs_var)
                current_gp_wrapper = retrained_gp_wrapper
                print(f"✅ GP retrained successfully with {len(all_historical_data)} samples")
            except Exception as e:
                print(f"❌ GP retraining failed: {e}")
                print("Continuing with previous GP model")
        
        # Create exploration MPPI controller with current GP
        exploration_controller = ExplorationMPPIController(
            gp_model=current_gp_wrapper,
            z_dataset=z_dataset,
            information_gain_fn=dummy_information_gain_function,
            action_dim=1,
            horizon=EXPLORATION_MPPI_HORIZON,
            num_samples=EXPLORATION_MPPI_NUM_SAMPLES,
            gamma=EXPLORATION_MPPI_GAMMA,
            lambda_uncertainty=EXPLORATION_MPPI_LAMBDA_UNCERTAINTY,
            eta=EXPLORATION_MPPI_ETA,
            num_discrete_actions=10,  # Standard 10 discrete actions
            uncertainty_threshold=EXPLORATION_MPPI_UNCERTAINTY_THRESHOLD,
            temp_norm_params=(obs_mean[6], np.sqrt(obs_var[6])),
            hour_norm_params=(obs_mean[0], np.sqrt(obs_var[0]))  # Fixed index for hour
        )
        
        # Get future environmental data for MPPI horizon from truth table
        future_env_data = get_future_env_data(truth_table, step_count, EXPLORATION_MPPI_HORIZON, args)
        
        # Get z_list for information gain computation
        z_targets, _ = z_dataset.get_z_targets()  # Extract just the z_targets list
        
        # Plan action using exploration MPPI with full exploration flag
        exploration_flag = 1.0  # Full exploration during exploration phase
        
        try:
            dropped_pairs, discrete_action, is_fallback = exploration_controller.plan(
                np.array(obs), future_env_data, exploration_flag, z_targets
            )
            
            if is_fallback:
                print(f"Exploration step {exploration_step}: Using fallback action")
                
        except Exception as e:
            print(f"Exploration step {exploration_step}: Error in exploration MPPI, using rule-based fallback: {e}")
            discrete_action = rule_based_controller(obs, info, args)
            is_fallback = True
            dropped_pairs = []
        
        # Take step with discrete action (environment expects discrete actions 0-9)
        next_obs, reward, terminated, truncated, info = step_env(env, discrete_action)
        
        # Convert discrete action to continuous for data storage consistency
        continuous_action = new_action_mapping(discrete_action)
        
        # Store exploration data in format consistent with training data
        exploration_data.append({
            'obs': obs.copy(),
            'action': discrete_action,
            'continuous_action': continuous_action,
            'next_obs': next_obs.copy(),
            'reward': reward,
            'step': step_count,
            'exploration_step': exploration_step,
            'hour': info['hour'],
            'day': info['day'],
            'month': info['month'],
            'indoor_temp': obs[6],
            'next_indoor_temp': next_obs[6],
            'temp_change': next_obs[6] - obs[6],
            'total_power_demand': info.get('total_power_demand', 0),
            'is_fallback': is_fallback,
            'dropped_pairs_count': len(dropped_pairs),
        })
        
        obs = next_obs
        step_count += 1
        exploration_step += 1
        
        # Progress reporting
        if exploration_step % (24 * args.timestep) == 0:
            day = exploration_step // (24 * args.timestep)
            print(f"Exploration day {day} completed, total steps: {exploration_step}")
            
        if terminated or truncated:
            print("Environment terminated during exploration collection")
            break
    
    print(f"\nExploration data collected: {len(exploration_data)} examples")
    print(f"Final GP training data size: {len(training_data) + len(exploration_data)} samples")
    
    # Print some statistics about the actions used
    actions_used = [d['action'] for d in exploration_data]
    print(f"Action distribution: {np.bincount(actions_used, minlength=10)}")
    return exploration_data, obs, info, z_dataset

def dummy_information_gain_function(state_action_pair, z_list):
    """Dummy information gain function for exploration"""
    if not z_list:
        return 0.0
    
    # Simple distance-based information gain
    min_distance = float('inf')
    for z_point in z_list:
        distance = np.linalg.norm(state_action_pair - z_point)
        min_distance = min(min_distance, distance)
    
    # Higher gain for points farther from existing z points
    return max(0.0, 1.0 - min_distance)

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

    # Normalize actions for GP (actions are 0-9, normalize to [-1, 1])
    normalized_actions = (actions / 4.5) - 1.0
    
    # Train GP model
    gp_model.fit(observations, normalized_actions, temperature_changes)
    
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
# MPPI CONTROL PHASE
# ============================================================================

def run_mppi_control(env, gp_wrapper, truth_table, training_data, exploration_data, obs_mean, obs_var, args):
    """
    Run MPPI controller for the control phase
    
    Args:
        env: Environment
        gp_wrapper: Trained GP model wrapper (will be retrained with exploration data)
        truth_table: DataFrame with environmental data
        training_data: Training data
        exploration_data: Exploration data
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
        
    Returns:
        control_data: List of control results
        metrics: Control metrics
    """
    print("="*60)
    print(f"Step 7: Retraining GP with combined training + exploration data")
    print("="*60)
    
    # Combine training and exploration data for GP retraining
    combined_data = training_data + exploration_data
    print(f"Training samples: {len(training_data)} (Week 1: Rule-based)")
    print(f"Exploration samples: {len(exploration_data)} (Week 2: Exploration MPPI)")
    print(f"Combined dataset: {len(combined_data)} samples for GP retraining")
    
    # Retrain GP model with combined dataset
    gp_model, gp_wrapper = train_gp_model(combined_data, obs_mean, obs_var)
    
    print("="*60)
    print(f"Step 8: Running MPPI controller for {CONTROL_DAYS} days with retrained GP")
    print("="*60)
    
    # Reset environment to start of control period
    obs, info = reset_env(env)
    
    # Skip to end of training period
    training_steps = len(training_data)
    for i in range(training_steps):
        action = training_data[i]['action']
        obs, reward, terminated, truncated, info = step_env(env, action)
        if terminated or truncated:
            print("Environment terminated during skip to control period")
            break
    
    # Skip to end of exploration period
    exploration_steps = len(exploration_data)
    for i in range(exploration_steps):
        action = exploration_data[i]['action']
        obs, reward, terminated, truncated, info = step_env(env, action)
        if terminated or truncated:
            print("Environment terminated during skip to control period")
            break
    
    # Extract normalization parameters for MPPI
    temp_mean = obs_mean[6]  # Index 6 is indoor temperature
    temp_std = math.sqrt(obs_var[6])
    hour_mean = obs_mean[0]  # Index 0 is hour
    hour_std = math.sqrt(obs_var[0])
    
    # Create MPPI controller
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
    step_count = training_steps + exploration_steps
    total_control_steps = CONTROL_DAYS * 24 * args.timestep
    all_dropped_pairs = []
    all_fallback_flags = []
    
    print(f"Starting MPPI control from step {step_count}")
    
    control_step = 0
    while control_step < total_control_steps and not (terminated or truncated):
        # Create future environmental data for planning
        future_env_data = create_future_env_data(
            obs, info, truth_table, step_count, MPPI_HORIZON
        )
        
        # Plan action using MPPI
        try:
            dropped_pairs, action, is_fallback = mppi_controller.plan(
                np.array(obs), future_env_data
            )
            all_dropped_pairs.append(dropped_pairs)
            all_fallback_flags.append(is_fallback)
            
        except Exception as e:
            print(f"MPPI planning failed at step {step_count}: {e}")
            # Fallback to rule-based action
            action = rule_based_controller(obs, info, args)
            is_fallback = True
            dropped_pairs = []
            all_dropped_pairs.append(dropped_pairs)
            all_fallback_flags.append(is_fallback)
        
        # Take step in environment
        next_obs, reward, terminated, truncated, info = step_env(env, action)
        
        # Store control data
        control_data.append({
            'step': step_count,
            'control_step': control_step,
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
        })
        
        obs = next_obs
        step_count += 1
        control_step += 1
        
        # Print progress
        if control_step % (24 * args.timestep) == 0:
            day = control_step // (24 * args.timestep)
            recent_fallback_rate = np.mean(all_fallback_flags[-24*args.timestep:])
            print(f"  Control day {day} completed - Fallback rate: {recent_fallback_rate:.3f}")
    
    # Compute metrics
    metrics = mppi_controller.get_evaluation_metrics(
        all_dropped_pairs, all_fallback_flags, len(control_data)
    )
    
    print(f"MPPI control completed: {len(control_data)} steps")
    print(f"Final metrics: {metrics}")
    
    return control_data, metrics



# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_training_data(training_data, obs_mean, obs_var, args, folders, season, timestamp):
    """Plot training data visualization"""
    print("="*60)
    print("Step 3.5: Creating training data visualization")
    print("="*60)
    
    # Extract data for plotting
    steps = [d['step'] for d in training_data]
    hours = [d['hour'] for d in training_data]
    indoor_temps = [d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6] for d in training_data]
    actions = [d['action'] for d in training_data]
    # Training data doesn't have fallback information (rule-based controller doesn't have fallbacks)
    fallbacks = [d.get('is_fallback', False) for d in training_data]
    
    # Create time axis (convert steps to hours)
    time_hours = np.array(steps) / args.timestep
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Indoor temperature
    axes[0].plot(time_hours, indoor_temps, 'b-', linewidth=1, alpha=0.7)
    # Comfort bounds in actual temperature values (since indoor_temps are denormalized)
    if args.winter:
        comfort_bounds = (20, 24)  # Winter comfort bounds
    else:
        comfort_bounds = (23, 26)  # Summer comfort bounds
    axes[0].axhline(y=comfort_bounds[0], color='r', linestyle='--', alpha=0.5, label='Comfort bounds')
    axes[0].axhline(y=comfort_bounds[1], color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Indoor Temperature (°C)')
    axes[0].set_title('Exploration Phase: Indoor Temperature')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Actions
    axes[1].plot(time_hours, actions, 'g-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Discrete Action')
    axes[1].set_title('Exploration Phase: Actions')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 9.5)
    
    # Plot 3: Fallback indicator
    fallback_indicators = [1 if f else 0 for f in fallbacks]
    axes[2].plot(time_hours, fallback_indicators, 'r-', linewidth=1, alpha=0.7)
    axes[2].set_ylabel('Fallback Action')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_title('Exploration Phase: Fallback Actions')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{folders['figures']}/exploration_training_data_{season}_{TRAINING_DAYS}days_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Training data plot saved to {filename}")
    
    plt.close()

def plot_results(training_data, exploration_data, control_data, obs_mean, obs_var, args, folders, season, timestamp):
    """Plot final results comparing all three phases"""
    print("="*60)
    print("Step 9: Creating comprehensive results visualization")
    print("="*60)
    
    # Extract training data
    train_steps = [d['step'] for d in training_data]
    train_temps = [d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6] for d in training_data]
    train_actions = [d['action'] for d in training_data]
    
    # Extract exploration data
    exploration_steps = [d['step'] for d in exploration_data]
    exploration_temps = [d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6] for d in exploration_data]
    exploration_actions = [d['action'] for d in exploration_data]
    
    # Extract control data
    control_steps = [d['step'] for d in control_data]
    control_temps = [d['indoor_temp'] * np.sqrt(obs_var[6]) + obs_mean[6] for d in control_data]  # Use 'indoor_temp' key
    control_actions = [d['action'] for d in control_data]
    
    # Convert to time hours
    train_hours = np.array(train_steps) / args.timestep
    exploration_hours = np.array(exploration_steps) / args.timestep
    control_hours = np.array(control_steps) / args.timestep
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(20, 10))
    
    # Plot 1: Temperature comparison
    axes[0].plot(train_hours, train_temps, 'b-', linewidth=1, alpha=0.7, label='Week 1: Training (Rule-based)')
    axes[0].plot(exploration_hours, exploration_temps, 'g-', linewidth=1, alpha=0.7, label='Week 2: Exploration (Exploration MPPI)')
    axes[0].plot(control_hours, control_temps, 'r-', linewidth=1, alpha=0.7, label='Week 3: Control (Standard MPPI)')
    
    # Comfort bounds in actual temperature values (since temps are denormalized)
    if args.winter:
        comfort_bounds = (20, 24)  # Winter comfort bounds
    else:
        comfort_bounds = (23, 26)  # Summer comfort bounds
    axes[0].axhline(y=comfort_bounds[0], color='gray', linestyle='--', alpha=0.5, label='Comfort bounds')
    axes[0].axhline(y=comfort_bounds[1], color='gray', linestyle='--', alpha=0.5)
    
    # Phase transition lines
    if len(train_hours) > 0:
        axes[0].axvline(x=train_hours[-1], color='black', linestyle='-', alpha=0.5, label='Phase transitions')
    if len(exploration_hours) > 0:
        axes[0].axvline(x=exploration_hours[-1], color='black', linestyle='-', alpha=0.5)
    
    axes[0].set_ylabel('Indoor Temperature (°C)')
    axes[0].set_title('3-Week HVAC Control Results: Training → Exploration → Control')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Actions comparison
    axes[1].plot(train_hours, train_actions, 'b-', linewidth=1, alpha=0.7, label='Week 1: Training (Rule-based)')
    axes[1].plot(exploration_hours, exploration_actions, 'g-', linewidth=1, alpha=0.7, label='Week 2: Exploration (Exploration MPPI)')
    axes[1].plot(control_hours, control_actions, 'r-', linewidth=1, alpha=0.7, label='Week 3: Control (Standard MPPI)')
    
    # Phase transition lines
    if len(train_hours) > 0:
        axes[1].axvline(x=train_hours[-1], color='black', linestyle='-', alpha=0.5, label='Phase transitions')
    if len(exploration_hours) > 0:
        axes[1].axvline(x=exploration_hours[-1], color='black', linestyle='-', alpha=0.5)
    
    axes[1].set_ylabel('Discrete Action')
    axes[1].set_xlabel('Time (hours)')
    axes[1].set_title('3-Week Action Sequence: Training → Exploration → Control')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 9.5)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{folders['figures']}/3week_results_{season}_{TRAINING_DAYS}days_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"3-week results plot saved to {filename}")
    
    plt.close()

def plot_exploration_data(exploration_data, obs_mean, obs_var, args, folders, season, timestamp):
    """Plot exploration data visualization"""
    print("="*60)
    print("Step 5.5: Creating exploration data visualization")
    print("="*60)
    
    # Extract data for plotting
    steps = [d['exploration_step'] for d in exploration_data]
    hours = [d['hour'] for d in exploration_data]
    indoor_temps = [d['obs'][6] * np.sqrt(obs_var[6]) + obs_mean[6] for d in exploration_data]
    actions = [d['action'] for d in exploration_data]
    fallbacks = [d.get('is_fallback', False) for d in exploration_data]
    
    # Create time axis (convert steps to hours)
    time_hours = np.array(steps) / args.timestep
    
    # Create figure
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Plot 1: Indoor temperature
    axes[0].plot(time_hours, indoor_temps, 'g-', linewidth=1, alpha=0.7)
    # Comfort bounds in actual temperature values (since indoor_temps are denormalized)
    if args.winter:
        comfort_bounds = (20, 24)  # Winter comfort bounds
    else:
        comfort_bounds = (23, 26)  # Summer comfort bounds
    axes[0].axhline(y=comfort_bounds[0], color='r', linestyle='--', alpha=0.5, label='Comfort bounds')
    axes[0].axhline(y=comfort_bounds[1], color='r', linestyle='--', alpha=0.5)
    axes[0].set_ylabel('Indoor Temperature (°C)')
    axes[0].set_title('Exploration Phase: Indoor Temperature')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    
    # Plot 2: Actions
    axes[1].plot(time_hours, actions, 'g-', linewidth=1, alpha=0.7)
    axes[1].set_ylabel('Discrete Action')
    axes[1].set_title('Exploration Phase: Actions')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim(-0.5, 9.5)
    
    # Plot 3: Fallback indicator
    fallback_indicators = [1 if f else 0 for f in fallbacks]
    axes[2].plot(time_hours, fallback_indicators, 'r-', linewidth=1, alpha=0.7)
    axes[2].set_ylabel('Fallback Action')
    axes[2].set_xlabel('Time (hours)')
    axes[2].set_title('Exploration Phase: Fallback Actions')
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-0.1, 1.1)
    
    plt.tight_layout()
    
    # Save plot
    filename = f"{folders['figures']}/exploration_data_{season}_{EXPLORATION_DAYS}days_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Exploration data plot saved to {filename}")
    
    plt.close()

# ============================================================================
# UTILITY FUNCTIONS FOR ORGANIZED RESULTS  
# ============================================================================

def create_organized_results_folder(method_name, timestamp):
    """
    Create organized folder structure for experiment results
    
    Args:
        method_name: Name of the method (e.g., 'clue', 'exploration')
        timestamp: Timestamp string for the run
        
    Returns:
        Dictionary with folder paths
    """
    # Create main run folder
    run_folder = f"results/{method_name}_daily_retrain_{timestamp}"
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

def create_energy_comfort_summary_figure(training_metrics, exploration_metrics, control_metrics, folders, timestamp, mode):
    """
    Create summary figure showing energy consumption and comfort violations for 3-week experiment
    
    Args:
        training_metrics: Dictionary with training phase metrics
        exploration_metrics: Dictionary with exploration phase metrics
        control_metrics: Dictionary with control phase metrics  
        folders: Dictionary with folder paths
        timestamp: Timestamp string
        mode: 'winter' or 'summer'
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Energy consumption comparison
    phases = ['Week 1\n(Training)', 'Week 2\n(Exploration)', 'Week 3\n(Control)']
    energy_values = [training_metrics['energy_kwh'], 
                    exploration_metrics['energy_kwh'],
                    control_metrics['energy_kwh']]
    
    bars1 = ax1.bar(phases, energy_values, color=['lightblue', 'lightcoral', 'lightgreen'], alpha=0.7)
    ax1.set_ylabel('Energy Consumption (kWh)')
    ax1.set_title('Energy Consumption Across 3 Weeks')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, energy_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Comfort violation rate comparison  
    violation_rates = [training_metrics['comfort_violations']['violation_rate'] * 100,
                      exploration_metrics['comfort_violations']['violation_rate'] * 100,
                      control_metrics['comfort_violations']['violation_rate'] * 100]
    
    bars2 = ax2.bar(phases, violation_rates, color=['salmon', 'orange', 'gold'], alpha=0.7)
    ax2.set_ylabel('Comfort Violation Rate (%)')
    ax2.set_title('Comfort Violation Rate Across 3 Weeks')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, violation_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Summary table as text
    ax3.axis('off')
    table_data = [
        ['Metric', 'Week 1 (Training)', 'Week 2 (Exploration)', 'Week 3 (Control)'],
        ['Energy (kWh)', f'{training_metrics["energy_kwh"]:.1f}', 
         f'{exploration_metrics["energy_kwh"]:.1f}',
         f'{control_metrics["energy_kwh"]:.1f}'],
        ['Comfort Violations (%)', f'{training_metrics["comfort_violations"]["violation_rate"]*100:.1f}',
         f'{exploration_metrics["comfort_violations"]["violation_rate"]*100:.1f}',
         f'{control_metrics["comfort_violations"]["violation_rate"]*100:.1f}'],
        ['Total Hours', f'{training_metrics["comfort_violations"]["total_hours"]:.1f}',
         f'{exploration_metrics["comfort_violations"]["total_hours"]:.1f}',
         f'{control_metrics["comfort_violations"]["total_hours"]:.1f}']
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.25, 0.25, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('3-Week Performance Summary Table', pad=20)
    
    # Performance summary with key findings
    ax4.axis('off')
    total_training_energy = training_metrics['energy_kwh']
    total_control_energy = control_metrics['energy_kwh']
    energy_improvement = ((total_training_energy - total_control_energy) / total_training_energy * 100)
    
    training_violations = training_metrics['comfort_violations']['violation_rate'] * 100
    control_violations = control_metrics['comfort_violations']['violation_rate'] * 100
    comfort_improvement = training_violations - control_violations
    
    fallback_info = [
        f"EXPLORATION MPPI HVAC Control Results (Daily Retraining)",
        f"Mode: {mode.title()}",
        f"Run timestamp: {timestamp}",
        f"",
        f"Key Performance Metrics:",
        f"• Energy efficiency: {energy_improvement:+.1f}% vs baseline",
        f"• Comfort improvement: {comfort_improvement:+.1f}% less violations",
        f"• Total control duration: {control_metrics['comfort_violations']['total_hours']:.1f} hours",
        f"• Total exploration data: {sum([m['comfort_violations']['total_count'] for m in [training_metrics, exploration_metrics]])} samples"
    ]
    
    for i, text in enumerate(fallback_info):
        weight = 'bold' if i == 0 or text.startswith('•') else 'normal'
        ax4.text(0.05, 0.9 - i*0.1, text, transform=ax4.transAxes, 
                fontsize=11, weight=weight, verticalalignment='top')
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{folders['figures']}/energy_comfort_summary_{mode}_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Energy & comfort summary saved: {filename}")

# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Main experiment function"""
    args = parse_args()
    
    # Create organized folder structure
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'winter' if args.winter else 'summer'
    folders = create_organized_results_folder('exploration', timestamp)
    
    print("="*60)
    print("EXPLORATION MPPI HVAC Control Experiment v0.1 (3-Week) - Daily GP Retraining")
    print("="*60)
    print(f"Mode: {'Winter' if args.winter else 'Summer'}")
    print(f"Week 1 - Training days: {TRAINING_DAYS} (Rule-based)")
    print(f"Week 2 - Exploration days: {EXPLORATION_DAYS} (Exploration MPPI with daily GP retraining)")
    print(f"Week 3 - Control days: {CONTROL_DAYS} (Standard MPPI with final retrained GP)")
    print(f"Environment: {args.environment}")
    print(f"Results folder: {folders['run_folder']}")
    print("="*60)
    
    try:
        # Step 1: Create and calibrate environment with default controller for training data collection
        training_env, obs_mean, obs_var = create_environment(args, use_default_controller=DEFAULT_CONTROLLER)
        
        # Step 2: Collect truth table (environmental data)
        truth_table = collect_truth_table(training_env, args)
        
        # Reset environment for actual experiment
        training_env.reset()
        
        # Step 3: Collect training data with rule-based controller
        training_data, final_obs, final_info = collect_training_data(
            training_env, truth_table, args, use_default_controller=DEFAULT_CONTROLLER
        )
        
        # Step 3.5: Create training data visualization
        plot_training_data(training_data, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Step 4: Train GP model
        gp_model, gp_wrapper = train_gp_model(training_data, obs_mean, obs_var)
        
        # Close training environment
        training_env.close()
        
        # Step 5: Create new environment with discrete action space for exploration
        print("="*60)
        print("Creating exploration environment with discrete action space")
        print("="*60)
        exploration_env, _, _ = create_environment(args, use_default_controller=False)
        
        # Step 6: Collect exploration data with exploration MPPI controller
        exploration_data, final_obs, final_info, z_dataset = collect_exploration_data(
            exploration_env, gp_wrapper, truth_table, training_data, obs_mean, obs_var, args
        )
        
        # Step 6.5: Create exploration data visualization
        plot_exploration_data(exploration_data, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Step 7-8: Run MPPI controller (includes GP retraining with combined data)
        control_data, metrics = run_mppi_control(
            exploration_env, gp_wrapper, truth_table, training_data, exploration_data, obs_mean, obs_var, args
        )
        
        # Step 9: Create comprehensive results visualization
        plot_results(training_data, exploration_data, control_data, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Step 10: Save all data for reproducibility
        save_results(training_data, exploration_data, control_data, truth_table, metrics, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Step 11: Save metrics
        save_metrics(metrics, args, folders, mode, timestamp)
        
        print("="*60)
        print("3-WEEK EXPLORATION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Week 1 - Training samples: {len(training_data)} (Rule-based)")
        print(f"Week 2 - Exploration samples: {len(exploration_data)} (Exploration MPPI)")
        print(f"Week 3 - Control samples: {len(control_data)} (Standard MPPI with retrained GP)")
        print(f"Total GP training data: {len(training_data) + len(exploration_data)} samples")
        print(f"Control metrics: {metrics}")
        print(f"Z dataset size: {z_dataset.num_points} (Type1: {z_dataset.num_points_type1}, Type2: {z_dataset.num_points_type2})")
        
    except Exception as e:
        print(f"Error during experiment: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if 'training_env' in locals():
            training_env.close()
        if 'exploration_env' in locals():
            exploration_env.close()

def save_results(training_data, exploration_data, control_data, truth_table, metrics, obs_mean, obs_var, args, folders, mode, timestamp):
    """
    Save all experiment data to CSV files for reproducibility with organized structure
    
    Args:
        training_data: Week 1 rule-based training data
        exploration_data: Week 2 exploration MPPI data
        control_data: Week 3 standard MPPI control data
        truth_table: Environmental truth table
        metrics: Control metrics
        obs_mean: Observation normalization mean
        obs_var: Observation normalization variance
        args: Command line arguments
        folders: Dictionary with organized folder paths
        mode: 'winter' or 'summer'
        timestamp: Timestamp string
    """
    print("="*60)
    print("Step 10: Saving all data for reproducibility with organized structure")
    print("="*60)
    
    # Calculate energy consumption and comfort violations for all phases
    print("Calculating energy consumption and comfort violations for all phases...")
    
    # Determine comfort bounds based on season
    comfort_bounds = (20, 24) if args.winter else (23, 26)
    
    # Calculate training phase metrics (Week 1)
    training_energy = calculate_energy_consumption(training_data, obs_mean, obs_var)
    training_comfort_violations = calculate_comfort_violations(training_data, obs_mean, obs_var, comfort_bounds)
    
    # Calculate exploration phase metrics (Week 2)
    exploration_energy = calculate_energy_consumption(exploration_data, obs_mean, obs_var)
    exploration_comfort_violations = calculate_comfort_violations(exploration_data, obs_mean, obs_var, comfort_bounds)
    
    # Calculate control phase metrics (Week 3)
    control_energy = calculate_energy_consumption(control_data, obs_mean, obs_var)
    control_comfort_violations = calculate_comfort_violations(control_data, obs_mean, obs_var, comfort_bounds)
    
    # Create metrics dictionaries
    training_metrics = {
        'energy_kwh': training_energy,
        'comfort_violations': training_comfort_violations
    }
    
    exploration_metrics = {
        'energy_kwh': exploration_energy,
        'comfort_violations': exploration_comfort_violations
    }
    
    control_metrics = {
        'energy_kwh': control_energy,
        'comfort_violations': control_comfort_violations
    }
    
    # Print comprehensive metrics
    print(f"\n🔋 ENERGY CONSUMPTION (3-Week Analysis):")
    print(f"  Week 1 (Training): {training_energy:.2f} kWh")
    print(f"  Week 2 (Exploration): {exploration_energy:.2f} kWh")
    print(f"  Week 3 (Control): {control_energy:.2f} kWh")
    print(f"  Control vs Training: {((control_energy - training_energy) / training_energy * 100):+.1f}%")
    
    print(f"\n🌡️  COMFORT VIOLATIONS (3-Week Analysis):")
    print(f"  Week 1 (Training): {training_comfort_violations['violation_rate']*100:.1f}% ({training_comfort_violations['violation_hours']:.1f} hours)")
    print(f"  Week 2 (Exploration): {exploration_comfort_violations['violation_rate']*100:.1f}% ({exploration_comfort_violations['violation_hours']:.1f} hours)")
    print(f"  Week 3 (Control): {control_comfort_violations['violation_rate']*100:.1f}% ({control_comfort_violations['violation_hours']:.1f} hours)")
    print(f"  Control vs Training: {(control_comfort_violations['violation_rate'] - training_comfort_violations['violation_rate'])*100:+.1f}%")
    
    # Create energy & comfort summary figure
    create_energy_comfort_summary_figure(training_metrics, exploration_metrics, control_metrics, folders, timestamp, mode)
    
    # Save training data (Week 1) in collected_exploration_data folder
    training_df = pd.DataFrame(training_data)
    training_file = f"{folders['collected_exploration_data']}/3week_exploration_training_data_{mode}_{TRAINING_DAYS}days_{timestamp}.csv"
    training_df.to_csv(training_file, index=False)
    print(f"Training data saved to {training_file}")
    
    # Save exploration data (Week 2) in collected_exploration_data folder
    exploration_df = pd.DataFrame(exploration_data)
    exploration_file = f"{folders['collected_exploration_data']}/3week_exploration_exploration_data_{mode}_{EXPLORATION_DAYS}days_{timestamp}.csv"
    exploration_df.to_csv(exploration_file, index=False)
    print(f"Exploration data saved to {exploration_file}")
    
    # Save control data (Week 3) in data folder
    control_df = pd.DataFrame(control_data)
    control_file = f"{folders['data']}/3week_exploration_control_data_{mode}_{CONTROL_DAYS}days_{timestamp}.csv"
    control_df.to_csv(control_file, index=False)
    print(f"Control data saved to {control_file}")
    
    # Save truth table (data folder)
    truth_file = f"{folders['data']}/3week_exploration_truth_table_{mode}_{timestamp}.csv"
    truth_table.to_csv(truth_file, index=False)
    print(f"Truth table saved to {truth_file}")
    
    # Save normalization parameters (data folder)
    norm_params = {
        'obs_mean': obs_mean,
        'obs_var': obs_var,
        'temperature_mean': obs_mean[6],
        'temperature_var': obs_var[6]
    }
    norm_file = f"{folders['data']}/3week_exploration_normalization_params_{mode}_{timestamp}.csv"
    pd.DataFrame([norm_params]).to_csv(norm_file, index=False)
    print(f"Normalization parameters saved to {norm_file}")
    
    # Save combined dataset info (data folder)
    combined_file = f"{folders['data']}/3week_exploration_combined_summary_{mode}_{timestamp}.csv"
    combined_data = []
    
    # Add training data with phase label
    for i, d in enumerate(training_data):
        row = d.copy()
        row['phase'] = 'training'
        row['phase_step'] = i
        combined_data.append(row)
    
    # Add exploration data with phase label
    for i, d in enumerate(exploration_data):
        row = d.copy()
        row['phase'] = 'exploration'
        row['phase_step'] = i
        combined_data.append(row)
    
    # Add control data with phase label
    for i, d in enumerate(control_data):
        row = d.copy()
        row['phase'] = 'control'
        row['phase_step'] = i
        combined_data.append(row)
    
    combined_df = pd.DataFrame(combined_data)
    combined_df.to_csv(combined_file, index=False)
    print(f"Combined data saved to {combined_file}")
    
    print(f"All data saved successfully!")
    print(f"Total files: training ({len(training_data)} rows), exploration ({len(exploration_data)} rows), control ({len(control_data)} rows)")

def save_metrics(metrics, args, folders, mode, timestamp):
    """Save metrics to file"""
    print("="*60)
    print("Step 11: Saving experiment metrics")
    print("="*60)
    
    filename = f"{folders['data']}/3week_exploration_metrics_{mode}_{TRAINING_DAYS}days_{timestamp}.txt"
    
    with open(filename, 'w') as f:
        f.write("3-WEEK EXPLORATION MPPI HVAC Control Experiment Metrics\n")
        f.write("="*60 + "\n")
        f.write(f"Season: {'Winter' if args.winter else 'Summer'}\n")
        f.write(f"Week 1 - Training days: {TRAINING_DAYS} (Rule-based)\n")
        f.write(f"Week 2 - Exploration days: {EXPLORATION_DAYS} (Exploration MPPI)\n")
        f.write(f"Week 3 - Control days: {CONTROL_DAYS} (Standard MPPI with retrained GP)\n")
        f.write(f"Environment: {args.environment}\n")
        f.write(f"Timestep: {args.timestep}\n")
        f.write(f"GP trained on: Week 1 + Week 2 combined data\n")
        f.write("\n")
        f.write("Control Phase Metrics (Week 3):\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value}\n")
        f.write("\n")
        f.write("Hyperparameters:\n")
        f.write(f"Exploration MPPI horizon: {EXPLORATION_MPPI_HORIZON}\n")
        f.write(f"Exploration MPPI samples: {EXPLORATION_MPPI_NUM_SAMPLES}\n")
        f.write(f"Control MPPI horizon: {MPPI_HORIZON}\n")
        f.write(f"Control MPPI samples: {MPPI_NUM_SAMPLES}\n")
        f.write(f"Z dataset Type1 size: {Z_DATASET_MAX_SIZE_TYPE1}\n")
        f.write(f"Z dataset Type2 size: {Z_DATASET_MAX_SIZE_TYPE2}\n")
    
    print(f"Metrics saved to {filename}")

if __name__ == "__main__":
    run_experiment() 