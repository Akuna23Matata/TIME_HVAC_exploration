import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class MPPIController:
    """
    Model Predictive Path Integral (MPPI) Controller for HVAC Control with Discrete Actions
    Based on the CLUE paper implementation with confidence-based control
    
    Modified to generate discrete actions from 0 to 9 representing temperature setpoints.
    The controller operates entirely in normalized space, but denormalizes values internally
    for meaningful reward computation.
    
    The controller filters out trajectory samples with high first-step uncertainty
    and returns them for evaluation purposes. When no trajectories pass the 
    uncertainty threshold, it falls back to a conservative action (action 5).
    
    Discrete Action Mapping (for reward computation only):
        Action 0: [12°C heating, 30°C cooling]
        Action 1: [13°C heating, 29°C cooling]
        Action 2: [14°C heating, 28°C cooling]
        Action 3: [15°C heating, 27°C cooling]
        Action 4: [16°C heating, 26°C cooling]
        Action 5: [17°C heating, 25°C cooling] (neutral/fallback)
        Action 6: [18°C heating, 24°C cooling]
        Action 7: [19°C heating, 23.25°C cooling]
        Action 8: [20°C heating, 23.25°C cooling]
        Action 9: [21°C heating, 23.25°C cooling]
        
    GP Input Normalization:
        Discrete action 'a' is normalized for GP input as: (a/4.5) - 1
        This maps action range [0,9] to normalized range [-1,1]
    
    Normalization Handling:
        - All inputs/outputs are in normalized space
        - Normalization parameters (mean, std) are used ONLY for reward computation
        - This allows physically meaningful rewards while keeping interface clean
    
    Returns:
        - dropped_state_action_pairs: List of (state, action) pairs dropped due to high uncertainty
          These represent the first step of trajectories that were filtered out
        - action: Optimal discrete action (integer 0-9)
        - is_fallback: Boolean indicating whether fallback action was used
    
    Usage:
        controller = MPPIController(
            gp_model=gp_model,
            temp_norm_params=(22.0, 8.0),      # (mean, std) for reward computation
            hour_norm_params=(12.0, 12.0),     # (mean, std) for reward computation
        )
        
        dropped_pairs, action, is_fallback = controller.plan(normalized_state, normalized_future_data)
        
    Evaluation:
        Use get_evaluation_metrics() to compute fallback rates and uncertainty statistics
    """
    
    def __init__(
        self,
        gp_model: Callable,  # GP model that returns (mean, variance)
        action_dim: int = 1,  # Changed to 1 for discrete scalar action
        horizon: int = 20,
        num_samples: int = 1000,
        gamma: float = 0.9,
        lambda_uncertainty: float = 1e-2,
        eta: float = 1.0,  # MPPI temperature parameter
        num_discrete_actions: int = 10,  # Actions 0-9
        uncertainty_threshold: Optional[float] = None,
        # Normalization parameters for reward computation: (mean, std) for each variable type
        temp_norm_params: Optional[Tuple[float, float]] = None,  # (mean, std) for temperature
        hour_norm_params: Optional[Tuple[float, float]] = None,  # (mean, std) for hours
        comfort_bounds: Optional[Tuple[float, float]] = (23, 26),  # (lower, upper) for comfort bounds
    ):
        """
        Initialize MPPI Controller for Discrete Actions
        
        Args:
            gp_model: Gaussian Process model function(state_action) -> (mean, variance)
            action_dim: Dimension of action space (1 for discrete action)
            horizon: Planning horizon
            num_samples: Number of trajectory samples for MPPI
            gamma: Discount factor
            lambda_uncertainty: Weight for uncertainty in objective function
            eta: Temperature parameter for MPPI exponential weighting
            num_discrete_actions: Number of discrete actions (10 for actions 0-9)
            uncertainty_threshold: Threshold for filtering high-uncertainty trajectories
            temp_norm_params: (mean, std) for temperature denormalization in reward computation
            hour_norm_params: (mean, std) for hour denormalization in reward computation
        """
        self.gp_model = gp_model
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.lambda_uncertainty = lambda_uncertainty
        self.eta = eta
        self.num_discrete_actions = num_discrete_actions
        self.uncertainty_threshold = uncertainty_threshold
        
        # Normalization parameters for reward computation only
        # Default assumes: temp ~N(22, 8²), hour ~N(12, 12²)
        self.temp_mean, self.temp_std = temp_norm_params if temp_norm_params else (22.0, 8.0)
        self.hour_mean, self.hour_std = hour_norm_params if hour_norm_params else (12.0, 12.0)
        
        # Previous action for smoothing (discrete action 0-9)
        self.prev_action = 5  # Start with neutral action
        
        # State indices for easier access
        self.state_indices = {
            'hour': 0,
            'outdoor_temp': 1, 
            'outdoor_humidity': 2,
            'wind_speed': 3,
            'wind_direction': 4,
            'direct_solar_radiation': 5,
            'air_temperature': 6,
            'air_humidity': 7,
            'people_occupant': 8
        }
        self.temp_lower, self.temp_upper = comfort_bounds
        
    def normalize_discrete_action_for_gp(self, discrete_action: int) -> float:
        """
        Normalize discrete action for GP input: normalized_action = (a/4.5) - 1
        Maps [0,9] to [-1,1]
        
        Args:
            discrete_action: Integer action from 0 to 9
            
        Returns:
            normalized_action: Float in range [-1,1] for GP input
        """
        return (discrete_action / 4.5) - 1.0
        
    def new_action_mapping(self, action: int) -> np.ndarray:
        """
        Map discrete action to [heating_setpoint, cooling_setpoint] for reward computation
        
        Args:
            action: Integer action from 0 to 9
            
        Returns:
            setpoints: [heating_setpoint, cooling_setpoint] in Celsius
        """
        # Handle case where action might be ndarray (from SB3 algorithms)
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
        
    def denormalize_temperature(self, norm_temp: float) -> float:
        """Convert normalized temperature to actual temperature for reward computation"""
        return norm_temp * self.temp_std + self.temp_mean
        
    def denormalize_hour(self, norm_hour: float) -> float:
        """Convert normalized hour to actual hour for reward computation"""
        return norm_hour * self.hour_std + self.hour_mean
    
    def sample_action_sequences(self, current_action: int) -> np.ndarray:
        """
        Sample random discrete action sequences
        
        Args:
            current_action: Previous discrete action (0-9)
            
        Returns:
            action_sequences: [num_samples, horizon] with integer actions 0-9
        """
        # Sample random discrete actions for each trajectory and timestep
        action_sequences = np.random.randint(
            0, self.num_discrete_actions, 
            size=(self.num_samples, self.horizon)
        )
        
        return action_sequences
    
    def rollout_trajectory(
        self, 
        initial_state: np.ndarray,
        action_sequence: np.ndarray,
        future_env_data: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Roll out a single trajectory using the GP model
        
        Args:
            initial_state: Current state [9] 
            action_sequence: Discrete actions for horizon [horizon] (integers 0-9)
            future_env_data: Future environmental data [horizon, 9]
            
        Returns:
            states: Predicted states [horizon+1, 9]
            rewards: Rewards for each step [horizon]
            uncertainties: GP uncertainties [horizon]
        """
        states = np.zeros((self.horizon + 1, len(initial_state)))
        states[0] = initial_state.copy()
        rewards = np.zeros(self.horizon)
        uncertainties = np.zeros(self.horizon)
        
        for t in range(self.horizon):
            current_state = states[t].copy()
            discrete_action = action_sequence[t]
            
            # Prepare next state with future environmental data
            next_state = future_env_data[t].copy()
            
            # Keep air_humidity constant from initial state
            next_state[self.state_indices['air_humidity']] = initial_state[self.state_indices['air_humidity']]
            
            # Normalize discrete action for GP input
            normalized_action = self.normalize_discrete_action_for_gp(discrete_action)
            
            # Prepare GP input: concatenate current state and normalized action
            gp_input = np.concatenate([current_state, [normalized_action]])
            
            # Get GP prediction for temperature change
            temp_change_mean, temp_change_var = self.gp_model(gp_input)
            
            # Update air temperature: current_temp + predicted_change
            current_temp = current_state[self.state_indices['air_temperature']]
            next_state[self.state_indices['air_temperature']] = current_temp + temp_change_mean
            
            # Clip temperature to reasonable bounds in normalized space
            next_state[self.state_indices['air_temperature']] = np.clip(
                next_state[self.state_indices['air_temperature']], -1.0, 1.0
            )
            
            states[t + 1] = next_state
            uncertainties[t] = temp_change_var
            
            # Compute reward for this step
            rewards[t] = self.compute_reward(next_state, discrete_action)
            
        return states, rewards, uncertainties
    
    def compute_reward(self, state: np.ndarray, discrete_action: int) -> float:
        """
        Compute reward for discrete action using heating/cooling setpoints
        r(s_t) = -w_e * E_t - (1 - w_e) * (|Z_t - z_upper|+ + |Z_t - z_lower|+)
        
        Args:
            state: Current state [9] (normalized)
            discrete_action: Discrete action (0-9)
            
        Returns:
            reward: Scalar reward value
        """
        # Denormalize temperature and get action setpoints
        air_temp = self.denormalize_temperature(state[self.state_indices['air_temperature']])
        setpoints = self.new_action_mapping(discrete_action)  # [heating_setpoint, cooling_setpoint]
        heating_setpoint, cooling_setpoint = setpoints[0], setpoints[1]
        hour = self.denormalize_hour(state[self.state_indices['hour']])
        
        # Determine whether it's an occupied period (simplified: 8AM-6PM weekdays)
        is_occupied = 8 <= hour <= 18
        
        # Energy consumption (heating and cooling energy based on setpoints)
        energy_heating = max(0, heating_setpoint - air_temp) if heating_setpoint > air_temp else 0
        energy_cooling = max(0, air_temp - cooling_setpoint) if cooling_setpoint < air_temp else 0
        energy = energy_heating + energy_cooling
        
        # Comfort bounds (simplified seasonal detection based on outdoor temp)
        outdoor_temp = self.denormalize_temperature(state[self.state_indices['outdoor_temp']])
        # if outdoor_temp < 15:  # Winter
        #     temp_lower, temp_upper = 20.0, 23.5
        # else:  # Summer
        temp_lower, temp_upper = self.temp_lower, self.temp_upper
        
        # Comfort violation (only positive deviations count)
        comfort_violation = max(0, temp_lower - air_temp) + max(0, air_temp - temp_upper)
        
        # Weight factor based on occupancy
        w_e = 1.0 if not is_occupied else 0.1
        
        # Compute reward (negative because MPPI maximizes)
        reward = -w_e * energy - (1 - w_e) * comfort_violation
        
        return reward
    
    def compute_trajectory_scores(
        self, 
        rewards: np.ndarray, 
        uncertainties: np.ndarray
    ) -> np.ndarray:
        """
        Compute trajectory scores using discounted rewards and uncertainties
        Based on Equation 9: sum_t gamma^t * (r(x_t) - lambda * sigma(x_t))
        
        Args:
            rewards: [num_samples, horizon]
            uncertainties: [num_samples, horizon]
            
        Returns:
            scores: [num_samples]
        """
        # Create discount factors
        discount_factors = np.array([self.gamma ** t for t in range(self.horizon)])
        
        # Combine rewards and uncertainty penalties
        combined_scores = rewards - self.lambda_uncertainty * uncertainties
        
        # Apply discounting and sum over horizon
        scores = np.sum(combined_scores * discount_factors[None, :], axis=1)
        
        return scores
    
    def filter_trajectories_by_uncertainty(
        self, 
        action_sequences: np.ndarray,
        rewards: np.ndarray,
        uncertainties: np.ndarray,
        current_state: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, int]]]:
        """
        Filter trajectories based on first-step uncertainty threshold (confidence-based control)
        
        Args:
            action_sequences: [num_samples, horizon] (discrete actions)
            rewards: [num_samples, horizon]
            uncertainties: [num_samples, horizon]
            current_state: Current state for tracking dropped state-action pairs
            
        Returns:
            filtered_actions: [filtered_samples, horizon] (discrete actions)
            filtered_rewards: [filtered_samples, horizon]
            filtered_uncertainties: [filtered_samples, horizon]
            dropped_state_action_pairs: List of (state, action) tuples that were dropped
        """
        dropped_state_action_pairs = []
        
        if self.uncertainty_threshold is None:
            return action_sequences, rewards, uncertainties, dropped_state_action_pairs
            
        # Filter based on first step uncertainty (as in paper)
        first_step_uncertainties = uncertainties[:, 0]
        valid_mask = first_step_uncertainties <= self.uncertainty_threshold
        dropped_mask = ~valid_mask
        
        # Collect dropped state-action pairs
        for i in np.where(dropped_mask)[0]:
            state_action_pair = (current_state.copy(), action_sequences[i, 0])
            dropped_state_action_pairs.append(state_action_pair)
        
        # If no trajectories pass threshold, return empty arrays for filtered data
        if not np.any(valid_mask):
            return (np.array([]).reshape(0, self.horizon), 
                    np.array([]).reshape(0, self.horizon), 
                    np.array([]).reshape(0, self.horizon),
                    dropped_state_action_pairs)
            
        return (action_sequences[valid_mask], 
                rewards[valid_mask], 
                uncertainties[valid_mask],
                dropped_state_action_pairs)
    
    def compute_mppi_action(
        self,
        action_sequences: np.ndarray,
        scores: np.ndarray
    ) -> int:
        """
        Compute MPPI discrete action using exponential weighting
        
        Args:
            action_sequences: [num_samples, horizon] (discrete actions)
            scores: [num_samples]
            
        Returns:
            optimal_action: Discrete action (0-9)
        """
        # Compute exponential weights
        weights = np.exp(scores / self.eta)
        
        # Normalize weights
        weights = weights / (np.sum(weights) + 1e-8)  # Add small epsilon for numerical stability
        
        # For discrete actions, use weighted voting
        # Count weighted votes for each discrete action (first timestep only)
        first_actions = action_sequences[:, 0]
        action_votes = np.zeros(self.num_discrete_actions)
        
        for action in range(self.num_discrete_actions):
            mask = first_actions == action
            action_votes[action] = np.sum(weights[mask])
        
        # Select action with highest weighted vote
        optimal_action = np.argmax(action_votes)
        
        return optimal_action
    
    def create_fallback_action(self) -> int:
        """
        Create fallback discrete action (neutral action 5, corresponding to ~22.5°C)
        
        Returns:
            fallback_action: Discrete action (integer)
        """
        # Use action 5 as neutral fallback (middle of range 0-9)
        return 9
    
    def get_evaluation_metrics(
        self, 
        all_dropped_pairs: List[List[Tuple[np.ndarray, int]]], 
        all_fallback_flags: List[bool],
        total_timesteps: int
    ) -> Dict:
        """
        Compute evaluation metrics for uncertainty-based control
        
        Args:
            all_dropped_pairs: List of dropped pairs for each timestep
            all_fallback_flags: List of fallback flags for each timestep  
            total_timesteps: Total number of control timesteps
            
        Returns:
            Dict with evaluation metrics
        """
        # Fallback statistics
        fallback_count = sum(all_fallback_flags)
        fallback_rate = fallback_count / total_timesteps if total_timesteps > 0 else 0.0
        
        # Trajectory filtering statistics
        total_dropped_pairs = sum(len(pairs) for pairs in all_dropped_pairs)
        total_evaluated_pairs = total_timesteps * self.num_samples
        drop_rate = total_dropped_pairs / total_evaluated_pairs if total_evaluated_pairs > 0 else 0.0
        
        # Uncertainty threshold effectiveness
        avg_dropped_per_timestep = total_dropped_pairs / total_timesteps if total_timesteps > 0 else 0.0
        
        return {
            'fallback_rate': fallback_rate,
            'fallback_count': fallback_count,
            'total_dropped_pairs': total_dropped_pairs,
            'drop_rate': drop_rate,
            'avg_dropped_per_timestep': avg_dropped_per_timestep,
            'uncertainty_threshold': self.uncertainty_threshold,
            'total_timesteps': total_timesteps
        }
    
    def plan(
        self,
        current_state: np.ndarray,
        future_env_data: np.ndarray
    ) -> Tuple[List[Tuple[np.ndarray, int]], int, bool]:
        """
        Main MPPI planning function
        
        Args:
            current_state: Current observation [9] (normalized)
            future_env_data: Future environmental data [horizon, 9] (normalized)
            
        Returns:
            Tuple containing:
                - dropped_state_action_pairs: List of (state, discrete_action) pairs dropped due to high uncertainty
                - action: Optimal discrete action (integer 0-9)
                - is_fallback: Boolean indicating whether fallback action was used
        """
        # Sample action sequences around previous action
        action_sequences = self.sample_action_sequences(self.prev_action)
        
        # Roll out all trajectories
        all_rewards = np.zeros((self.num_samples, self.horizon))
        all_uncertainties = np.zeros((self.num_samples, self.horizon))
        
        for i in range(self.num_samples):
            _, rewards, uncertainties = self.rollout_trajectory(
                current_state, action_sequences[i], future_env_data
            )
            all_rewards[i] = rewards
            all_uncertainties[i] = uncertainties
        
        # Filter trajectories by uncertainty (confidence-based control)
        filtered_actions, filtered_rewards, filtered_uncertainties, dropped_pairs = self.filter_trajectories_by_uncertainty(
            action_sequences, all_rewards, all_uncertainties, current_state
        )
        
        # If no valid trajectories after filtering, use fallback
        if len(filtered_actions) == 0:
            fallback_action = self.create_fallback_action()
            # Update previous action for next iteration
            self.prev_action = fallback_action
            
            return dropped_pairs, fallback_action, True
        
        # Compute trajectory scores
        scores = self.compute_trajectory_scores(filtered_rewards, filtered_uncertainties)
        
        # Compute optimal action using MPPI
        optimal_action = self.compute_mppi_action(filtered_actions, scores)
        
        # Update previous action for next iteration
        self.prev_action = optimal_action
        
        return dropped_pairs, optimal_action, False

# Example usage and helper functions
def create_dummy_gp_model():
    """Create a dummy GP model for testing"""
    def gp_model(state_action):
        # Dummy implementation - in practice, use your trained GP
        # state_action is [9 state features + 1 normalized action]
        # Returns (predicted_temperature_change, uncertainty)
        temp_change = np.random.normal(0, 0.1)  # Small random temperature change
        uncertainty = np.random.uniform(0.01, 0.1)  # Random uncertainty
        return temp_change, uncertainty
    return gp_model

def create_future_env_data(current_state, horizon=20):
    """Create dummy future environmental data"""
    # In practice, this would come from weather forecasts or schedules
    future_data = np.tile(current_state, (horizon, 1))
    
    # Add some variation to environmental variables
    for t in range(horizon):
        # Vary hour
        future_data[t, 0] = current_state[0] + t * 0.1  # Hour progression
        # Add small variations to weather
        future_data[t, 1:6] += np.random.normal(0, 0.05, 5)  # Weather variations
        # Keep air_temperature and air_humidity as placeholders (will be overwritten)
        # Vary occupancy based on hour
        hour = (current_state[0] * 12 + 12 + t) % 24
        future_data[t, 8] = 1.0 if 8 <= hour <= 18 else -1.0  # Occupied/unoccupied
    
    return future_data

# Example initialization and usage
if __name__ == "__main__":
    # Create controller with discrete actions
    gp_model = create_dummy_gp_model()
    controller = MPPIController(
        gp_model=gp_model,
        horizon=20,
        num_samples=1000,
        uncertainty_threshold=0.05,
        # Specify your environment's normalization parameters for reward computation
        temp_norm_params=(22.0, 8.0),      # temp = (temp - 22) / 8  
        hour_norm_params=(12.0, 12.0),     # hour = (hour - 12) / 12
    )
    
    # Example state (normalized)
    current_state = np.array([0.0, 0.2, 0.1, -0.3, 0.5, 0.8, 0.1, 0.0, 1.0])
    future_env_data = create_future_env_data(current_state)
    
    dropped_pairs, action, is_fallback = controller.plan(current_state, future_env_data)
    
    print(f"Optimal discrete action: {action}")
    setpoints = controller.new_action_mapping(action)
    print(f"Corresponding setpoints - Heating: {setpoints[0]:.1f}°C, Cooling: {setpoints[1]:.1f}°C")
    print(f"Is fallback action: {is_fallback}")
    print(f"Dropped pairs: {len(dropped_pairs)} {dropped_pairs[0]}")
    
    if is_fallback:
        fallback_setpoints = controller.new_action_mapping(action)
        print(f"Fallback setpoints - Heating: {fallback_setpoints[0]:.1f}°C, Cooling: {fallback_setpoints[1]:.1f}°C")