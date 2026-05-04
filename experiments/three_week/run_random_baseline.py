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
MPPI_HORIZON = 5
MPPI_NUM_SAMPLES = 100
MPPI_GAMMA = 0.85
MPPI_LAMBDA_UNCERTAINTY = 1e-2
MPPI_ETA = 1.0
MPPI_UNCERTAINTY_THRESHOLD = 0.6

# Control parameters
TRAINING_DAYS = 14     # Total training days (7 rule-based + 7 random)
RULE_BASED_DAYS = 7    # First 7 days with rule-based controller
RANDOM_DAYS = 7        # Next 7 days with random actions
CONTROL_DAYS = 7       # 1 week of control evaluation
OCCUPIED_START_HOUR = 8
OCCUPIED_END_HOUR = 17
OCCUPIED_ACTION = 9
UNOCCUPIED_ACTION = 0
DEFAULT_CONTROLLER = False

# ============================================================================
# UTILITY FUNCTIONSß
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
    
    # for action, (heat_sp, cool_sp) in action_mapping.items():
    #     distance = abs(heating_setpoint - heat_sp) + abs(cooling_setpoint - cool_sp)
    #     if distance < min_distance:
    #         min_distance = distance
    #         best_action = action
    
    # return best_action

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

def create_energy_comfort_summary_figure(training_metrics, control_metrics, folders, timestamp, mode):
    """
    Create summary figure showing energy consumption and comfort violations
    
    Args:
        training_metrics: Dictionary with training phase metrics
        control_metrics: Dictionary with control phase metrics  
        folders: Dictionary with folder paths
        timestamp: Timestamp string
        mode: 'winter' or 'summer'
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Energy consumption comparison
    phases = ['Training\n(Rule-based)', 'Control\n(MPPI)']
    energy_values = [training_metrics['energy_kwh'], control_metrics['energy_kwh']]
    
    bars1 = ax1.bar(phases, energy_values, color=['lightblue', 'lightgreen'], alpha=0.7)
    ax1.set_ylabel('Energy Consumption (kWh)')
    ax1.set_title('Energy Consumption Comparison')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, energy_values):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}', ha='center', va='bottom')
    
    # Comfort violation rate comparison  
    violation_rates = [training_metrics['comfort_violations']['violation_rate'] * 100,
                      control_metrics['comfort_violations']['violation_rate'] * 100]
    
    bars2 = ax2.bar(phases, violation_rates, color=['salmon', 'orange'], alpha=0.7)
    ax2.set_ylabel('Comfort Violation Rate (%)')
    ax2.set_title('Comfort Violation Rate Comparison')
    ax2.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars2, violation_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{value:.1f}%', ha='center', va='bottom')
    
    # Summary table as text
    ax3.axis('off')
    table_data = [
        ['Metric', 'Training Phase', 'Control Phase', 'Change'],
        ['Energy (kWh)', f'{training_metrics["energy_kwh"]:.1f}', 
         f'{control_metrics["energy_kwh"]:.1f}',
         f'{((control_metrics["energy_kwh"] - training_metrics["energy_kwh"]) / training_metrics["energy_kwh"] * 100):+.1f}%'],
        ['Comfort Violations (%)', f'{training_metrics["comfort_violations"]["violation_rate"]*100:.1f}',
         f'{control_metrics["comfort_violations"]["violation_rate"]*100:.1f}',
         f'{(control_metrics["comfort_violations"]["violation_rate"] - training_metrics["comfort_violations"]["violation_rate"])*100:+.1f}%'],
        ['Total Hours', f'{training_metrics["comfort_violations"]["total_hours"]:.1f}',
         f'{control_metrics["comfort_violations"]["total_hours"]:.1f}', '-']
    ]
    
    table = ax3.table(cellText=table_data, cellLoc='center', loc='center',
                     colWidths=[0.3, 0.2, 0.2, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style header row
    for i in range(4):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    ax3.set_title('Performance Summary Table', pad=20)
    
    # Fall back information (existing fallback tracking)
    ax4.axis('off')
    fallback_info = [
        f"CLUE MPPI HVAC Control Results",
        f"Mode: {mode.title()}",
        f"Run timestamp: {timestamp}",
        f"",
        f"Key Findings:",
        f"• Energy consumption change: {((control_metrics['energy_kwh'] - training_metrics['energy_kwh']) / training_metrics['energy_kwh'] * 100):+.1f}%",
        f"• Comfort violation change: {(control_metrics['comfort_violations']['violation_rate'] - training_metrics['comfort_violations']['violation_rate'])*100:+.1f}%",
        f"• Control phase duration: {control_metrics['comfort_violations']['total_hours']:.1f} hours"
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

def rule_based_controller(obs, info, args):
    """
    Rule-based controller: action 9 if occupied, action 0 otherwise
    
    Args:
        obs: Normalized observation [9]
        info: Environment info dict
        args: Arguments
        
    Returns:
        action: Discrete action (0-9)
    """
    actual_hour = info['hour']
    
    # During occupied hours, use aggressive HVAC; otherwise minimal
    if OCCUPIED_START_HOUR <= actual_hour <= OCCUPIED_END_HOUR:
        return OCCUPIED_ACTION
    else:
        return UNOCCUPIED_ACTION

def collect_training_data(env, truth_table, args, use_default_controller=False):
    """
    Collect training data using two-phase approach:
    1. First 7 days with rule-based controller
    2. Next 7 days with random actions
    3. Combine both datasets
    
    Args:
        env: Environment
        truth_table: DataFrame with environmental data
        args: Arguments
        use_default_controller: If True, use default controller (requires empty action space)
                               If False, use rule-based controller (requires discrete action space)
        
    Returns:
        training_data: List of training examples (combined from both phases)
        final_obs: Final observation after training period
        final_info: Final info after training period
    """
    print("="*60)
    print(f"Step 3: Collecting {TRAINING_DAYS} days of training data (7 rule-based + 7 random)")
    print("="*60)
    
    training_data = []
    obs, info = reset_env(env)
    step_count = 0
    
    # Phase 1: Collect 7 days of rule-based controller data
    print(f"Phase 1: Collecting {RULE_BASED_DAYS} days of rule-based controller data")
    print("-" * 40)
    
    rule_based_steps = RULE_BASED_DAYS * 24 * args.timestep
    phase1_data = []
    
    while step_count < rule_based_steps:
        # Use rule-based controller
        action = rule_based_controller(obs, info, args)
        
        # Take step
        next_obs, reward, terminated, truncated, info = step_env(env, action)
        
        # Convert discrete action to continuous for consistency
        discrete_action = action
        continuous_action = new_action_mapping(discrete_action)
        
        # Store training data
        phase1_data.append({
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
            'phase': 'rule_based'  # Mark data source
        })
        
        obs = next_obs
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Rule-based day {day} completed, total steps: {step_count}")
            
        if terminated or truncated:
            print("Environment terminated during rule-based phase")
            break
    
    print(f"Phase 1 completed: {len(phase1_data)} rule-based examples")
    
    # Phase 2: Collect 7 days of random action data
    print(f"\nPhase 2: Collecting {RANDOM_DAYS} days of random action data")
    print("-" * 40)
    
    random_steps = RANDOM_DAYS * 24 * args.timestep
    phase2_data = []
    
    while step_count < rule_based_steps + random_steps:
        # Use random actions
        action = env.action_space.sample()
        
        # Take step
        next_obs, reward, terminated, truncated, info = step_env(env, action)
        
        # Convert discrete action to continuous for consistency
        discrete_action = action
        continuous_action = new_action_mapping(discrete_action)
        
        # Store training data
        phase2_data.append({
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
            'phase': 'random'  # Mark data source
        })
        
        obs = next_obs
        step_count += 1
        
        # Print progress
        if step_count % (24 * args.timestep) == 0:
            day = step_count // (24 * args.timestep)
            print(f"  Random day {day} completed, total steps: {step_count}")
            
        if terminated or truncated:
            print("Environment terminated during random phase")
            break
    
    print(f"Phase 2 completed: {len(phase2_data)} random examples")
    
    # Combine both phases
    training_data = phase1_data + phase2_data
    
    print(f"\nCombined training data: {len(training_data)} total examples")
    print(f"  Rule-based phase: {len(phase1_data)} examples")
    print(f"  Random phase: {len(phase2_data)} examples")
    
    # Print action distribution statistics for each phase
    rule_based_actions = [d['action'] for d in phase1_data]
    random_actions = [d['action'] for d in phase2_data]
    
    print(f"Rule-based action distribution: {np.bincount(rule_based_actions, minlength=10)}")
    print(f"Random action distribution: {np.bincount(random_actions, minlength=10)}")
    
    return training_data, obs, info

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
        
        observations.append(obs)
        actions.append(action)
        temperature_changes.append(temp_change)
    
    observations = np.array(observations)
    actions = np.array(actions)
    temperature_changes = np.array(temperature_changes)
    
    print(f"Training GP with {len(observations)} examples")
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

def run_mppi_control(env, gp_wrapper, truth_table, training_data, obs_mean, obs_var, args):
    """
    Run MPPI controller for specified control period
    
    Args:
        env: Environment
        gp_wrapper: GP wrapper function
        truth_table: DataFrame with environmental data
        training_data: Training data for determining start point
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        args: Arguments
        
    Returns:
        control_data: List of control results
        metrics: Control metrics
    """
    print("="*60)
    print(f"Step 5: Running MPPI controller for {CONTROL_DAYS} days")
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
    step_count = training_steps
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

def plot_results(training_data, control_data, obs_mean, obs_var, args, folders, mode, timestamp):
    """
    Create comprehensive visualization of training and control results
    
    Args:
        training_data: Training data
        control_data: Control data
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
    """
    print("="*60)
    print("Step 7: Creating comprehensive visualization (Training + Control)")
    print("="*60)
    
    # Denormalize temperatures for plotting
    temp_mean = obs_mean[6]
    temp_std = math.sqrt(obs_var[6])
    
    # Separate training data by phase
    rule_based_data = [d for d in training_data if d.get('phase') == 'rule_based']
    random_data = [d for d in training_data if d.get('phase') == 'random']
    
    # Extract training data for plotting - rule-based phase
    rule_based_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in rule_based_data]
    rule_based_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in rule_based_data]
    rule_based_actions = [d['action'] for d in rule_based_data]
    rule_based_power = [d['total_power_demand'] for d in rule_based_data]
    
    # Extract training data for plotting - random phase
    random_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in random_data]
    random_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in random_data]
    random_actions = [d['action'] for d in random_data]
    random_power = [d['total_power_demand'] for d in random_data]
    
    # Extract control data for plotting
    control_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in control_data]
    control_outdoor_temps = [d['outdoor_temp'] * math.sqrt(obs_var[1]) + obs_mean[1] for d in control_data]
    control_actions = [d['action'] for d in control_data]
    control_power = [d['total_power_demand'] for d in control_data]
    
    # Create time indices
    rule_based_time = np.arange(len(rule_based_data)) / (args.timestep * 24)  # Days
    random_time = np.arange(len(random_data)) / (args.timestep * 24) + RULE_BASED_DAYS  # Days, offset by rule-based period
    control_time = np.arange(len(control_data)) / (args.timestep * 24) + TRAINING_DAYS  # Days, offset by training period
    
    # Create figure with 4 subplots
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
    
    # Add comfort zone shading for occupied hours (8 AM to 6 PM)
    comfort_lower = 23  # Lower bound of comfort zone
    comfort_upper = 26  # Upper bound of comfort zone
    occupied_start = 8  # 8 AM
    occupied_end = 18   # 6 PM
    
    # Calculate total days
    total_days = TRAINING_DAYS + CONTROL_DAYS
    
    # Add comfort zone shading for each day
    comfort_zone_labeled = False
    for day in range(total_days):
        # Calculate start and end times for occupied hours each day
        occupied_day_start = day + occupied_start / 24  # 8 AM
        occupied_day_end = day + occupied_end / 24      # 6 PM
        
        label = 'Comfort Zone (8AM-6PM)' if not comfort_zone_labeled else ""
        ax1.fill_between([occupied_day_start, occupied_day_end], 
                       comfort_lower, comfort_upper, 
                       color='lightgreen', alpha=0.3, 
                       label=label)
        comfort_zone_labeled = True
    
    # Add vertical lines to separate phases
    rule_based_separation = RULE_BASED_DAYS
    training_control_separation = TRAINING_DAYS
    for ax in [ax1, ax2, ax3, ax4]:
        ax.axvline(x=rule_based_separation, color='blue', linestyle='--', alpha=0.8, linewidth=1)
        ax.axvline(x=training_control_separation, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Plot temperatures
    ax1.plot(rule_based_time, rule_based_indoor_temps, 'b-', label='Training (Rule-based)', alpha=0.7)
    ax1.plot(random_time, random_indoor_temps, 'g-', label='Training (Random)', alpha=0.7)
    ax1.plot(control_time, control_indoor_temps, 'r-', label='Control (MPPI)', alpha=0.7)
    ax1.plot(rule_based_time, rule_based_outdoor_temps, 'orange', label='Outdoor Temperature', alpha=0.5)
    ax1.plot(random_time, random_outdoor_temps, 'orange', alpha=0.5)
    ax1.plot(control_time, control_outdoor_temps, 'orange', alpha=0.5)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('HVAC Control Results - Two-Phase Training (Rule-based + Random) vs Control')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot actions
    ax2.plot(rule_based_time, rule_based_actions, 'b-', label='Training (Rule-based)', alpha=0.7)
    ax2.plot(random_time, random_actions, 'g-', label='Training (Random)', alpha=0.7)
    ax2.plot(control_time, control_actions, 'r-', label='Control Actions', alpha=0.7)
    ax2.set_ylabel('Action (0-9)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 9.5)
    
    # Plot power demand
    ax3.plot(rule_based_time, rule_based_power, 'b-', label='Training (Rule-based)', alpha=0.7)
    ax3.plot(random_time, random_power, 'g-', label='Training (Random)', alpha=0.7)
    ax3.plot(control_time, control_power, 'r-', label='Control Power', alpha=0.7)
    ax3.set_ylabel('Total Power Demand (W)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot power demand comparison (daily averages)
    # Calculate daily averages
    timesteps_per_day = args.timestep * 24
    
    # Rule-based phase daily averages
    rule_based_daily_power = []
    for day in range(RULE_BASED_DAYS):
        day_start = day * timesteps_per_day
        day_end = min((day + 1) * timesteps_per_day, len(rule_based_power))
        if day_end > day_start:
            rule_based_daily_power.append(np.mean(rule_based_power[day_start:day_end]))
    
    # Random phase daily averages
    random_daily_power = []
    for day in range(RANDOM_DAYS):
        day_start = day * timesteps_per_day
        day_end = min((day + 1) * timesteps_per_day, len(random_power))
        if day_end > day_start:
            random_daily_power.append(np.mean(random_power[day_start:day_end]))
    
    # Control phase daily averages
    control_daily_power = []
    for day in range(CONTROL_DAYS):
        day_start = day * timesteps_per_day
        day_end = min((day + 1) * timesteps_per_day, len(control_power))
        if day_end > day_start:
            control_daily_power.append(np.mean(control_power[day_start:day_end]))
    
    rule_based_days_x = np.arange(len(rule_based_daily_power))
    random_days_x = np.arange(len(random_daily_power)) + RULE_BASED_DAYS
    control_days_x = np.arange(len(control_daily_power)) + TRAINING_DAYS
    
    ax4.bar(rule_based_days_x, rule_based_daily_power, alpha=0.7, label='Rule-based Daily Avg', color='blue')
    ax4.bar(random_days_x, random_daily_power, alpha=0.7, label='Random Daily Avg', color='green')
    ax4.bar(control_days_x, control_daily_power, alpha=0.7, label='Control Daily Avg', color='red')
    ax4.set_ylabel('Daily Avg Power (W)')
    ax4.set_xlabel('Time (days)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add text annotations
    ax1.text(RULE_BASED_DAYS/2, ax1.get_ylim()[1]*0.95, 'RULE-BASED\nTRAINING', ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax1.text(RULE_BASED_DAYS + RANDOM_DAYS/2, ax1.get_ylim()[1]*0.95, 'RANDOM\nTRAINING', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    ax1.text(TRAINING_DAYS + CONTROL_DAYS/2, ax1.get_ylim()[1]*0.95, 'CONTROL', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    
    # Print power consumption statistics
    rule_based_avg_power = np.mean(rule_based_power)
    random_avg_power = np.mean(random_power)
    training_avg_power = np.mean(rule_based_power + random_power)
    control_avg_power = np.mean(control_power)
    power_reduction = ((training_avg_power - control_avg_power) / training_avg_power) * 100
    
    print(f"Power Consumption Statistics:")
    print(f"  Rule-based Training Average: {rule_based_avg_power:.2f} W")
    print(f"  Random Training Average: {random_avg_power:.2f} W")
    print(f"  Total Training Period Average: {training_avg_power:.2f} W")
    print(f"  Control Period Average: {control_avg_power:.2f} W")
    print(f"  Power Change: {power_reduction:.2f}% ({'reduction' if power_reduction > 0 else 'increase'})")
    
    plt.tight_layout()
    
    # Save figure - use passed folders and timestamp
    filename = f"{folders['figures']}/control_results_{mode}_{TRAINING_DAYS}days_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results plot saved: {filename}")

def plot_training_data(training_data, obs_mean, obs_var, args, folders, mode, timestamp):
    """
    Create visualization of training data collection with two-phase approach
    
    Args:
        training_data: Training data (combined from rule-based and random phases)
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
    """
    print("="*60)
    print("Step 3.5: Creating training data visualization (Two-Phase)")
    print("="*60)
    
    # Denormalize temperatures for plotting
    temp_mean = obs_mean[6]
    temp_std = math.sqrt(obs_var[6])
    
    # Separate data by phase
    rule_based_data = [d for d in training_data if d.get('phase') == 'rule_based']
    random_data = [d for d in training_data if d.get('phase') == 'random']
    
    print(f"Plotting {len(rule_based_data)} rule-based examples and {len(random_data)} random examples")
    
    # Extract data for plotting - rule-based phase
    rule_based_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in rule_based_data]
    rule_based_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in rule_based_data]
    rule_based_actions = [d['action'] for d in rule_based_data]
    rule_based_power = [d['total_power_demand'] for d in rule_based_data]
    rule_based_heating_setpoints = [d['continuous_action'][0] for d in rule_based_data]
    rule_based_cooling_setpoints = [d['continuous_action'][1] for d in rule_based_data]
    
    # Extract data for plotting - random phase
    random_indoor_temps = [d['indoor_temp'] * temp_std + temp_mean for d in random_data]
    random_outdoor_temps = [d['obs'][1] * math.sqrt(obs_var[1]) + obs_mean[1] for d in random_data]
    random_actions = [d['action'] for d in random_data]
    random_power = [d['total_power_demand'] for d in random_data]
    random_heating_setpoints = [d['continuous_action'][0] for d in random_data]
    random_cooling_setpoints = [d['continuous_action'][1] for d in random_data]
    
    # Create time indices (in days) for each phase
    rule_based_time = np.arange(len(rule_based_data)) / (args.timestep * 24)  # Days
    random_time = np.arange(len(random_data)) / (args.timestep * 24) + RULE_BASED_DAYS  # Days, offset by rule-based period
    
    # Create figure
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(12, 16))
    
    # Add comfort zone shading for occupied hours (8 AM to 6 PM)
    comfort_lower = 23  # Lower bound of comfort zone
    comfort_upper = 26  # Upper bound of comfort zone
    occupied_start = 8  # 8 AM
    occupied_end = 18   # 6 PM
    
    # Calculate total training days
    total_training_days = TRAINING_DAYS
    
    # Add comfort zone shading for each day of training period
    comfort_zone_labeled = False
    for day in range(total_training_days + 1):
        # Calculate start and end times for occupied hours each day
        occupied_day_start = day + occupied_start / 24  # 8 AM
        occupied_day_end = day + occupied_end / 24      # 6 PM
        
        # Only shade if within our training data range
        if occupied_day_start <= total_training_days:
            shade_end = min(occupied_day_end, total_training_days)
            label = 'Comfort Zone (8AM-6PM)' if not comfort_zone_labeled else ""
            ax1.fill_between([occupied_day_start, shade_end], 
                           comfort_lower, comfort_upper, 
                           color='lightgreen', alpha=0.3, 
                           label=label)
            comfort_zone_labeled = True
    
    # Add vertical line to separate phases
    separation_line = RULE_BASED_DAYS
    for ax in [ax1, ax2, ax3, ax4, ax5]:
        ax.axvline(x=separation_line, color='red', linestyle='--', alpha=0.8, linewidth=2)
    
    # Plot indoor and outdoor temperatures
    ax1.plot(rule_based_time, rule_based_indoor_temps, 'b-', label='Indoor Temperature (Rule-based)', alpha=0.7)
    ax1.plot(random_time, random_indoor_temps, 'g-', label='Indoor Temperature (Random)', alpha=0.7)
    ax1.plot(rule_based_time, rule_based_outdoor_temps, 'orange', label='Outdoor Temperature', alpha=0.7)
    ax1.plot(random_time, random_outdoor_temps, 'orange', alpha=0.7)
    ax1.set_ylabel('Temperature (°C)')
    ax1.set_title('Training Data Collection - Two-Phase Approach (Rule-based + Random)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot discrete actions
    ax2.plot(rule_based_time, rule_based_actions, 'b-', label='Rule-based Actions', alpha=0.7)
    ax2.plot(random_time, random_actions, 'g-', label='Random Actions', alpha=0.7)
    ax2.set_ylabel('Discrete Action')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.5, 9.5)
    
    # Plot heating setpoints
    ax3.plot(rule_based_time, rule_based_heating_setpoints, 'b-', label='Rule-based Heating Setpoint', alpha=0.7)
    ax3.plot(random_time, random_heating_setpoints, 'g-', label='Random Heating Setpoint', alpha=0.7)
    ax3.set_ylabel('Heating Setpoint (°C)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot cooling setpoints
    ax4.plot(rule_based_time, rule_based_cooling_setpoints, 'b-', label='Rule-based Cooling Setpoint', alpha=0.7)
    ax4.plot(random_time, random_cooling_setpoints, 'g-', label='Random Cooling Setpoint', alpha=0.7)
    ax4.set_ylabel('Cooling Setpoint (°C)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot power demand
    ax5.plot(rule_based_time, rule_based_power, 'b-', label='Rule-based Power Demand', alpha=0.7)
    ax5.plot(random_time, random_power, 'g-', label='Random Power Demand', alpha=0.7)
    ax5.set_ylabel('Total Power Demand (W)')
    ax5.set_xlabel('Time (days)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Add text annotations for phases
    ax1.text(RULE_BASED_DAYS/2, ax1.get_ylim()[1]*0.95, 'RULE-BASED\nPHASE', ha='center', va='top', 
             bbox=dict(boxstyle='round', facecolor='blue', alpha=0.3))
    ax1.text(RULE_BASED_DAYS + RANDOM_DAYS/2, ax1.get_ylim()[1]*0.95, 'RANDOM\nPHASE', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='green', alpha=0.3))
    
    plt.tight_layout()
    
    # Save figure
    filename = f"{folders['figures']}/training_data_{mode}_{TRAINING_DAYS}days_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Training data plot saved: {filename}")

def save_results(training_data, control_data, truth_table, metrics, obs_mean, obs_var, args, folders, mode, timestamp):
    """
    Save all results to files with organized structure
    
    Args:
        training_data: Training data
        control_data: Control data  
        truth_table: Truth table
        metrics: Control metrics
        obs_mean: Mean values for denormalization [9]
        obs_var: Variance values for denormalization [9]
        args: Arguments
        folders: Dictionary with organized folder paths
        mode: 'winter' or 'summer'
        timestamp: Timestamp string
    """
    print("="*60)
    print("Step 8: Saving random exploration results with organized structure")
    print("="*60)
    
    # Calculate energy consumption for both phases
    print("Calculating energy consumption and comfort violations...")
    
    # Determine comfort bounds based on season
    comfort_bounds = (20, 24) if args.winter else (23, 26)
    
    # Calculate training phase metrics
    training_energy = calculate_energy_consumption(training_data, obs_mean, obs_var)
    training_comfort_violations = calculate_comfort_violations(training_data, obs_mean, obs_var, comfort_bounds)
    
    # Calculate control phase metrics  
    control_energy = calculate_energy_consumption(control_data, obs_mean, obs_var)
    control_comfort_violations = calculate_comfort_violations(control_data, obs_mean, obs_var, comfort_bounds)
    
    # Create metrics dictionaries
    training_metrics = {
        'energy_kwh': training_energy,
        'comfort_violations': training_comfort_violations
    }
    
    control_metrics = {
        'energy_kwh': control_energy,
        'comfort_violations': control_comfort_violations
    }
    
    # Print key metrics
    print(f"\n🔋 ENERGY CONSUMPTION:")
    print(f"  Training Phase: {training_energy:.2f} kWh")
    print(f"  Control Phase: {control_energy:.2f} kWh")
    print(f"  Change: {((control_energy - training_energy) / training_energy * 100):+.1f}%")
    
    print(f"\n🌡️  COMFORT VIOLATIONS:")
    print(f"  Training Phase: {training_comfort_violations['violation_rate']*100:.1f}% ({training_comfort_violations['violation_hours']:.1f} hours)")
    print(f"  Control Phase: {control_comfort_violations['violation_rate']*100:.1f}% ({control_comfort_violations['violation_hours']:.1f} hours)")
    print(f"  Rate Change: {(control_comfort_violations['violation_rate'] - training_comfort_violations['violation_rate'])*100:+.1f}%")
    
    # Create energy & comfort summary figure
    create_energy_comfort_summary_figure(training_metrics, control_metrics, folders, timestamp, mode)
    
    # Save training data (collected exploration data folder)
    training_df = pd.DataFrame(training_data)
    training_file = f"{folders['collected_exploration_data']}/random_exploration_training_data_{mode}_{TRAINING_DAYS}days_{timestamp}.csv"
    training_df.to_csv(training_file, index=False)
    
    # Save control data (data folder)
    control_df = pd.DataFrame(control_data)
    control_file = f"{folders['data']}/random_exploration_control_data_{mode}_{TRAINING_DAYS}days_{timestamp}.csv"
    control_df.to_csv(control_file, index=False)
    
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
    norm_file = f"{folders['data']}/random_exploration_normalization_params_{mode}_{timestamp}.csv"
    pd.DataFrame([norm_params]).to_csv(norm_file, index=False)
    print(f"Normalization parameters saved to {norm_file}")
    
    # Save comprehensive metrics (data folder)
    metrics_file = f"{folders['data']}/random_exploration_metrics_{mode}_{TRAINING_DAYS}days_{timestamp}.txt"
    with open(metrics_file, 'w') as f:
        f.write(f"MPPI HVAC Control Experiment Results (Random Exploration)\n")
        f.write(f"{'='*60}\n")
        f.write(f"Mode: {mode}\n")
        f.write(f"Training days: {TRAINING_DAYS} ({RULE_BASED_DAYS} rule-based + {RANDOM_DAYS} random)\n")
        f.write(f"Control days: {CONTROL_DAYS} (1 week evaluation)\n")
        f.write(f"Environment: {args.environment}\n")
        f.write(f"Purpose: Random exploration experiment with two-phase training data\n")
        f.write(f"\n🔋 ENERGY CONSUMPTION:\n")
        f.write(f"  Training Phase: {training_energy:.2f} kWh\n")
        f.write(f"  Control Phase: {control_energy:.2f} kWh\n")
        f.write(f"  Change: {((control_energy - training_energy) / training_energy * 100):+.1f}%\n")
        f.write(f"\n🌡️ COMFORT VIOLATIONS:\n")
        f.write(f"  Training Phase: {training_comfort_violations['violation_rate']*100:.1f}% ({training_comfort_violations['violation_hours']:.1f} hours)\n")
        f.write(f"  Control Phase: {control_comfort_violations['violation_rate']*100:.1f}% ({control_comfort_violations['violation_hours']:.1f} hours)\n")
        f.write(f"  Rate Change: {(control_comfort_violations['violation_rate'] - training_comfort_violations['violation_rate'])*100:+.1f}%\n")
        f.write(f"\nHyperparameters:\n")
        f.write(f"  MPPI Horizon: {MPPI_HORIZON}\n")
        f.write(f"  MPPI Samples: {MPPI_NUM_SAMPLES}\n")
        f.write(f"  MPPI Gamma: {MPPI_GAMMA}\n")
        f.write(f"  MPPI Lambda: {MPPI_LAMBDA_UNCERTAINTY}\n")
        f.write(f"  MPPI Eta: {MPPI_ETA}\n")
        f.write(f"  MPPI Uncertainty Threshold: {MPPI_UNCERTAINTY_THRESHOLD}\n")
        f.write(f"\nControl Metrics:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")
    
    print(f"Results saved:")
    print(f"  Training data: {training_file}")
    print(f"  Control data: {control_file}")
    print(f"  Truth table: {truth_file}")
    print(f"  Normalization parameters: {norm_file}")
    print(f"  Metrics: {metrics_file}")

# ============================================================================
# MAIN EXPERIMENT FUNCTION
# ============================================================================

def run_experiment():
    """Main experiment function"""
    args = parse_args()
    
    # Create organized folder structure 
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    mode = 'winter' if args.winter else 'summer'
    folders = create_organized_results_folder('random', timestamp)
    
    print("="*60)
    print("MPPI HVAC Control Experiment v0.1 (Random Exploration)")
    print("="*60)
    print(f"Mode: {'Winter' if args.winter else 'Summer'}")
    print(f"Training days: {TRAINING_DAYS} ({RULE_BASED_DAYS} rule-based + {RANDOM_DAYS} random)")
    print(f"Control days: {CONTROL_DAYS} (1 week evaluation)")
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
        
        # Step 3: Collect training data with two-phase approach (rule-based + random)
        training_data, final_obs, final_info = collect_training_data(training_env, truth_table, args, use_default_controller=DEFAULT_CONTROLLER)
        
        # Step 3.5: Create training data visualization
        plot_training_data(training_data, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Close training environment
        training_env.close()
        
        # Step 4: Train GP model
        gp_model, gp_wrapper = train_gp_model(training_data, obs_mean, obs_var)
        
        # Step 5: Create new environment with discrete action space for control
        print("="*60)
        print("Creating control environment with discrete action space")
        print("="*60)
        control_env, _, _ = create_environment(args, use_default_controller=False)
        
        # Step 6: Run MPPI controller
        control_data, metrics = run_mppi_control(
            control_env, gp_wrapper, truth_table, training_data, obs_mean, obs_var, args
        )
        
        # Step 7: Create control visualization
        plot_results(training_data, control_data, obs_mean, obs_var, args, folders, mode, timestamp)
        
        # Step 8: Save results
        save_results(training_data, control_data, truth_table, metrics, obs_mean, obs_var, args, folders, mode, timestamp)
        
        print("="*60)
        print("RANDOM EXPLORATION EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Training data collected: {len(training_data)} samples ({RULE_BASED_DAYS} rule-based + {RANDOM_DAYS} random)")
        print(f"Control data collected: {len(control_data)} samples (1 week MPPI)")
        print(f"Total GP training data: {len(training_data)} samples")
        print("="*60)
        
    except Exception as e:
        print(f"Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Clean up
        if 'training_env' in locals():
            training_env.close()
        if 'control_env' in locals():
            control_env.close()

if __name__ == "__main__":
    run_experiment()