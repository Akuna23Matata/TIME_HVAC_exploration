import numpy as np
from typing import Dict, List, Tuple, Optional, Callable

class ExplorationMPPIController:
    """
    Exploration-based Model Predictive Path Integral (MPPI) Controller for HVAC Control
    
    This controller is designed specifically for exploration purposes in HVAC control systems,
    using information gain as the primary objective rather than comfort/energy optimization.
    
    The controller works with a GP model and Z dataset to select actions that maximize
    information gain about the system dynamics, enabling better model learning and
    reduced uncertainty in future predictions.
    
    Key differences from standard MPPI:
    - Reward function focuses on exploration/information gain rather than comfort/energy
    - Integrates with ZDataset for candidate exploration points
    - Uses GP model uncertainty and information gain estimates
    - Still maintains discrete action space (0-9) for compatibility
    
    Modified to generate discrete actions from 0 to 9 representing temperature setpoints.
    The controller operates entirely in normalized space for consistency.
    
    Discrete Action Mapping (for reference):
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
    
    Usage:
        controller = ExplorationMPPIController(
            gp_model=gp_model,
            z_dataset=z_dataset,  # ZDataset for exploration targets
            information_gain_fn=info_gain_fn,  # Function to compute information gain
        )
        
        dropped_pairs, action, is_fallback = controller.plan(normalized_state, normalized_future_data)
        
    Returns:
        - dropped_state_action_pairs: List of (state, action) pairs dropped due to high uncertainty
        - action: Optimal discrete action (integer 0-9) for exploration
        - is_fallback: Boolean indicating whether fallback action was used
    """
    
    def __init__(
        self,
        gp_model: Callable,  # GP model that returns (mean, variance)
        z_dataset: Optional[object] = None,  # ZDataset for exploration targets
        information_gain_fn: Optional[Callable] = None,  # Function to compute information gain
        action_dim: int = 1,  # Changed to 1 for discrete scalar action
        horizon: int = 20,
        num_samples: int = 1000,
        gamma: float = 0.9,
        lambda_uncertainty: float = 1e-2,
        eta: float = 1.0,  # MPPI temperature parameter
        num_discrete_actions: int = 10,  # Actions 0-9
        uncertainty_threshold: Optional[float] = None,
        # Parameters for fallback/safety behavior
        temp_norm_params: Optional[Tuple[float, float]] = None,  # (mean, std) for temperature
        hour_norm_params: Optional[Tuple[float, float]] = None,  # (mean, std) for hours
        comfort_bounds: Optional[Tuple[float, float]] = (23, 26),  # (lower, upper) for comfort bounds
        # Exploration-specific parameters
        exploration_weight: float = 1.0,  # Weight for exploration vs exploitation
        safety_weight: float = 0.1,  # Weight for safety constraints
    ):
        """
        Initialize Exploration MPPI Controller
        
        Args:
            gp_model: Gaussian Process model function(state_action) -> (mean, variance)
            z_dataset: ZDataset object containing exploration targets
            information_gain_fn: Function to compute information gain given state-action pairs
            action_dim: Dimension of action space (1 for discrete action)
            horizon: Planning horizon
            num_samples: Number of trajectory samples for MPPI
            gamma: Discount factor
            lambda_uncertainty: Weight for uncertainty in objective function
            eta: Temperature parameter for MPPI exponential weighting
            num_discrete_actions: Number of discrete actions (10 for actions 0-9)
            uncertainty_threshold: Threshold for filtering high-uncertainty trajectories
            temp_norm_params: (mean, std) for temperature denormalization
            hour_norm_params: (mean, std) for hour denormalization
            exploration_weight: Weight for exploration vs exploitation
            safety_weight: Weight for safety constraints
        """
        self.gp_model = gp_model
        self.z_dataset = z_dataset
        self.information_gain_fn = information_gain_fn
        self.action_dim = action_dim
        self.horizon = horizon
        self.num_samples = num_samples
        self.gamma = gamma
        self.lambda_uncertainty = lambda_uncertainty
        self.eta = eta
        self.num_discrete_actions = num_discrete_actions
        self.uncertainty_threshold = uncertainty_threshold
        
        # Normalization parameters (kept for compatibility/safety)
        self.temp_mean, self.temp_std = temp_norm_params if temp_norm_params else (22.0, 8.0)
        self.hour_mean, self.hour_std = hour_norm_params if hour_norm_params else (12.0, 12.0)
        
        # Exploration-specific parameters
        self.exploration_weight = exploration_weight
        self.safety_weight = safety_weight
        
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
        Map discrete action to [heating_setpoint, cooling_setpoint] for reference
        
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
        """Convert normalized temperature to actual temperature"""
        return norm_temp * self.temp_std + self.temp_mean
        
    def denormalize_hour(self, norm_hour: float) -> float:
        """Convert normalized hour to actual hour"""
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
        future_env_data: np.ndarray,
        exploration_flag: float,
        z_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Roll out a single trajectory using the GP model
        
        Args:
            initial_state: Current state [9] 
            action_sequence: Discrete actions for horizon [horizon] (integers 0-9)
            future_env_data: Future environmental data [horizon, 9]
            exploration_flag: f_e exploration flag (0 for normal operation, 1 for exploration)
            z_list: List of z points for information gain computation
            
        Returns:
            Tuple containing:
                states: Predicted states [horizon, 9]
                exploration_rewards: Exploration rewards for each step [horizon]
                uncertainties: Prediction uncertainties [horizon]
        """
        states = np.zeros((self.horizon, initial_state.shape[0]))
        exploration_rewards = np.zeros(self.horizon)
        uncertainties = np.zeros(self.horizon)
        
        current_state = initial_state.copy()
        
        for t in range(self.horizon):
            # Get current discrete action
            discrete_action = action_sequence[t]
            
            # Normalize action for GP input
            normalized_action = self.normalize_discrete_action_for_gp(discrete_action)
            
            # Create state-action pair for GP prediction
            state_action = np.concatenate([current_state, [normalized_action]])
            
            # Predict next state using GP
            predicted_temp_change, uncertainty = self.gp_model(state_action)
            
            # Update state (simplified: only temperature changes)
            next_state = current_state.copy()
            next_state[self.state_indices['air_temperature']] += predicted_temp_change
            
            # Update environmental conditions from future data
            if t < len(future_env_data):
                # Update environmental variables (keep predicted temperature)
                temp_idx = self.state_indices['air_temperature']
                predicted_temp = next_state[temp_idx]
                next_state = future_env_data[t].copy()
                next_state[temp_idx] = predicted_temp
                
            states[t] = next_state
            uncertainties[t] = uncertainty
            
            # Compute exploration reward for this step
            exploration_rewards[t] = self.compute_exploration_reward(
                next_state, discrete_action, uncertainty, exploration_flag, z_list
            )
            
            # Update current state for next iteration
            current_state = next_state
            
        return states, exploration_rewards, uncertainties
    
    def compute_information_gain(self, state_action_pair: np.ndarray, z_list: List[np.ndarray]) -> float:
        """
        Compute information gain for a given state-action pair using GP model and z_list
        
        Args:
            state_action_pair: [10] state-action pair (9 state dims + 1 action dim)
            z_list: List of z points for information gain computation
            
        Returns:
            information_gain: Computed information gain value
        """
        if not z_list or len(z_list) == 0:
            return 0.0
            
        # Compute information gain using GP model
        # This is a simplified implementation - you may need to adjust based on your GP model
        total_gain = 0.0
        
        for z_point in z_list:
            # Ensure z_point has correct dimensionality (should be 10D like state_action_pair)
            if len(z_point) != len(state_action_pair):
                continue
                
            # Compute distance/similarity between current state-action and z_point
            distance = np.linalg.norm(state_action_pair - z_point)
            
            # Get GP prediction uncertainty at z_point
            _, z_uncertainty = self.gp_model(z_point)
            
            # Information gain is inversely related to distance and proportional to uncertainty
            # This is a simplified formulation - adjust based on your information theory requirements
            if distance > 0:
                gain = z_uncertainty / (1 + distance)
                total_gain += gain
        
        # Normalize by number of z points
        information_gain = total_gain / len(z_list) if len(z_list) > 0 else 0.0
        
        return information_gain

    def compute_energy_cost(self, state: np.ndarray, discrete_action: int) -> float:
        """
        Compute energy cost E_t for the given state and action
        
        Args:
            state: Current state [9] (normalized)
            discrete_action: Discrete action (0-9)
            
        Returns:
            energy_cost: L1 norm of |temperature_setpoint - current_temperature|
        """
        # Get current temperature (denormalized)
        current_temp = self.denormalize_temperature(state[self.state_indices['air_temperature']])
        
        # Get temperature setpoints for this action
        setpoints = self.new_action_mapping(discrete_action)
        heating_setpoint, cooling_setpoint = setpoints[0], setpoints[1]
        
        # Compute L1 norm of temperature differences
        heating_diff = abs(heating_setpoint - current_temp)
        cooling_diff = abs(cooling_setpoint - current_temp)
        
        # Energy cost is the sum of deviations from setpoints
        energy_cost = heating_diff + cooling_diff
        
        return energy_cost

    def compute_comfort_violation(self, state: np.ndarray) -> float:
        """
        Compute comfort violation for the given state
        
        Args:
            state: Current state [9] (normalized)
            
        Returns:
            comfort_violation: Comfort violation penalty
        """
        # Get current temperature (denormalized)
        current_temp = self.denormalize_temperature(state[self.state_indices['air_temperature']])
        
        # Compute comfort violation
        comfort_violation = max(0, self.temp_lower - current_temp) + max(0, current_temp - self.temp_upper)
        
        return comfort_violation

    def is_occupied_period(self, state: np.ndarray) -> bool:
        """
        Determine if current time is occupied period (8AM-6PM)
        
        Args:
            state: Current state [9] (normalized)
            
        Returns:
            is_occupied: True if occupied period, False otherwise
        """
        # Get current hour (denormalized)
        current_hour = self.denormalize_hour(state[self.state_indices['hour']])
        
        # Occupied period is 8AM-6PM
        return 8 <= current_hour <= 18

    def compute_exploration_reward(
        self, 
        state: np.ndarray, 
        discrete_action: int, 
        uncertainty: float, 
        exploration_flag: float,
        z_list: List[np.ndarray]
    ) -> float:
        """
        Compute exploration reward using the revised formulation:
        r = f_e * (information_gain - 0.01*E_t) + (1 - f_e)(w_e*(-E_t) - (1-w_e)*comfort_violation)
        
        Args:
            state: Current state [9] (normalized)
            discrete_action: Discrete action (0-9)
            uncertainty: Prediction uncertainty from GP
            exploration_flag: f_e exploration flag (0 for normal operation, 1 for exploration)
            z_list: List of z points for information gain computation
            
        Returns:
            exploration_reward: Scalar reward value
        """
        # Compute normalized action for GP input
        normalized_action = self.normalize_discrete_action_for_gp(discrete_action)
        
        # Create state-action pair
        state_action_pair = np.concatenate([state, [normalized_action]])
        
        # Compute information gain
        information_gain = self.compute_information_gain(state_action_pair, z_list)
        
        # Compute energy cost E_t
        energy_cost = self.compute_energy_cost(state, discrete_action)
        
        # Compute comfort violation
        comfort_violation = self.compute_comfort_violation(state)
        
        # Determine w_e based on occupancy
        is_occupied = self.is_occupied_period(state)
        w_e = 0.1 if is_occupied else 1.0
        
        # Compute reward using the revised formulation
        exploration_term = information_gain - 0.01 * energy_cost
        normal_operation_term = w_e * (-energy_cost) - (1 - w_e) * comfort_violation
        
        reward = exploration_flag * exploration_term + (1 - exploration_flag) * normal_operation_term
        
        return reward
    
    def compute_trajectory_scores(
        self, 
        exploration_rewards: np.ndarray, 
        uncertainties: np.ndarray,
        exploration_flag: float
    ) -> np.ndarray:
        """
        Compute trajectory scores using the revised MPPI formulation:
        score = r - (1 - f_e) * lambda * sigma
        
        Args:
            exploration_rewards: [num_samples, horizon]
            uncertainties: [num_samples, horizon]
            exploration_flag: f_e exploration flag (0 for normal operation, 1 for exploration)
            
        Returns:
            scores: [num_samples]
        """
        # Create discount factors
        discount_factors = np.array([self.gamma ** t for t in range(self.horizon)])
        
        # Apply discounting to rewards
        discounted_rewards = exploration_rewards * discount_factors[None, :]
        
        # Apply uncertainty penalty with exploration flag modification
        uncertainty_penalty = (1 - exploration_flag) * self.lambda_uncertainty * uncertainties
        discounted_uncertainties = uncertainty_penalty * discount_factors[None, :]
        
        # Compute final scores: score = r - (1 - f_e) * lambda * sigma
        scores = np.sum(discounted_rewards - discounted_uncertainties, axis=1)
        
        return scores
    
    def filter_trajectories_by_uncertainty(
        self, 
        action_sequences: np.ndarray,
        exploration_rewards: np.ndarray,
        uncertainties: np.ndarray,
        current_state: np.ndarray,
        exploration_flag: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Tuple[np.ndarray, int]]]:
        """
        Filter trajectories based on uncertainty threshold (confidence-based control)
        
        Args:
            action_sequences: [num_samples, horizon]
            exploration_rewards: [num_samples, horizon]
            uncertainties: [num_samples, horizon]
            current_state: [9] current state
            exploration_flag: f_e exploration flag (0 for normal operation, 1 for exploration)
            
        Returns:
            filtered_actions: [filtered_samples, horizon]
            filtered_rewards: [filtered_samples, horizon]
            filtered_uncertainties: [filtered_samples, horizon]
            dropped_state_action_pairs: List of dropped (state, action) pairs
        """
        # During exploration mode, don't filter any trajectories - we want to explore uncertain regions
        if exploration_flag == 1.0 or self.uncertainty_threshold is None:
            # No filtering, return all trajectories
            return action_sequences, exploration_rewards, uncertainties, []
        
        # Filter based on first-step uncertainty (only during normal operation)
        first_step_uncertainties = uncertainties[:, 0]
        valid_mask = first_step_uncertainties <= self.uncertainty_threshold
        
        # Collect dropped state-action pairs
        dropped_state_action_pairs = []
        for i in range(len(action_sequences)):
            if not valid_mask[i]:
                first_action = action_sequences[i, 0]
                dropped_state_action_pairs.append((current_state.copy(), first_action))
        
        # Filter arrays
        filtered_actions = action_sequences[valid_mask]
        filtered_rewards = exploration_rewards[valid_mask]
        filtered_uncertainties = uncertainties[valid_mask]
        
        return filtered_actions, filtered_rewards, filtered_uncertainties, dropped_state_action_pairs
    
    def compute_mppi_action(self, action_sequences: np.ndarray, scores: np.ndarray) -> int:
        """
        Compute optimal action using MPPI exponential weighting
        
        Args:
            action_sequences: [num_samples, horizon]
            scores: [num_samples]
            
        Returns:
            optimal_action: Discrete action (0-9)
        """
        # Compute exponential weights
        weights = np.exp(self.eta * scores)
        weights = weights / np.sum(weights)
        
        # Compute weighted average of first actions
        first_actions = action_sequences[:, 0]
        
        # For discrete actions, we need to handle the averaging differently
        # Compute weighted probability for each discrete action
        action_probs = np.zeros(self.num_discrete_actions)
        for i in range(self.num_discrete_actions):
            action_probs[i] = np.sum(weights[first_actions == i])
        
        # Select action with highest probability
        optimal_action = np.argmax(action_probs)
        
        return optimal_action
    
    def create_fallback_action(self) -> int:
        """
        Create fallback action when no valid trajectories are found
        
        Returns:
            fallback_action: Conservative discrete action (5 - neutral)
        """
        return 5  # Neutral action
    
    def plan(
        self,
        current_state: np.ndarray,
        future_env_data: np.ndarray,
        exploration_flag: float,
        z_list: List[np.ndarray]
    ) -> Tuple[List[Tuple[np.ndarray, int]], int, bool]:
        """
        Main exploration MPPI planning function
        
        Args:
            current_state: Current observation [9] (normalized)
            future_env_data: Future environmental data [horizon, 9] (normalized)
            exploration_flag: f_e exploration flag (0 for normal operation, 1 for exploration)
            z_list: List of z points for information gain computation
            
        Returns:
            Tuple containing:
                - dropped_state_action_pairs: List of (state, discrete_action) pairs dropped due to high uncertainty
                - action: Optimal discrete action (integer 0-9) for exploration
                - is_fallback: Boolean indicating whether fallback action was used
        """
        # Sample action sequences
        action_sequences = self.sample_action_sequences(self.prev_action)
        
        # Roll out all trajectories
        all_exploration_rewards = np.zeros((self.num_samples, self.horizon))
        all_uncertainties = np.zeros((self.num_samples, self.horizon))
        
        for i in range(self.num_samples):
            _, exploration_rewards, uncertainties = self.rollout_trajectory(
                current_state, action_sequences[i], future_env_data, exploration_flag, z_list
            )
            all_exploration_rewards[i] = exploration_rewards
            all_uncertainties[i] = uncertainties
        
        # Filter trajectories by uncertainty (confidence-based control)
        filtered_actions, filtered_rewards, filtered_uncertainties, dropped_pairs = self.filter_trajectories_by_uncertainty(
            action_sequences, all_exploration_rewards, all_uncertainties, current_state, exploration_flag
        )
        
        # If no valid trajectories after filtering, use fallback
        if len(filtered_actions) == 0:
            fallback_action = self.create_fallback_action()
            # Update previous action for next iteration
            self.prev_action = fallback_action
            
            return dropped_pairs, fallback_action, True
        
        # Compute trajectory scores
        scores = self.compute_trajectory_scores(filtered_rewards, filtered_uncertainties, exploration_flag)
        
        # Compute optimal action using MPPI
        optimal_action = self.compute_mppi_action(filtered_actions, scores)
        
        # Update previous action for next iteration
        self.prev_action = optimal_action
        
        return dropped_pairs, optimal_action, False
    
    def get_evaluation_metrics(self) -> Dict:
        """
        Get evaluation metrics for the exploration controller
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Placeholder for evaluation metrics
        return {
            'controller_type': 'exploration_mppi',
            'num_discrete_actions': self.num_discrete_actions,
            'horizon': self.horizon,
            'num_samples': self.num_samples,
            'exploration_weight': self.exploration_weight,
            'safety_weight': self.safety_weight,
        }


# Example usage and helper functions
def create_dummy_exploration_setup():
    """Create a dummy setup for testing exploration MPPI"""
    def dummy_gp_model(state_action):
        # Dummy GP model - in practice, use your trained GP
        temp_change = np.random.normal(0, 0.1)
        uncertainty = np.random.uniform(0.01, 0.1)
        return temp_change, uncertainty
    
    def dummy_information_gain_fn(state_action_pairs):
        # Dummy information gain function
        # In practice, this would compute actual information gain
        return np.random.uniform(0, 1, len(state_action_pairs))
    
    # Create dummy Z dataset (mock object)
    class DummyZDataset:
        def get_z_targets(self):
            return [np.random.randn(10) for _ in range(50)], None
    
    return dummy_gp_model, dummy_information_gain_fn, DummyZDataset()

def create_future_env_data_exploration(current_state, horizon=20):
    """Create dummy future environmental data for exploration"""
    future_data = np.zeros((horizon, len(current_state)))
    
    for t in range(horizon):
        # Simulate environmental changes
        future_data[t] = current_state.copy()
        
        # Add some variation to environmental variables
        future_data[t, 0] += t * 0.1  # Hour progression
        future_data[t, 1] += np.random.normal(0, 0.05)  # Outdoor temp variation
        future_data[t, 2] += np.random.normal(0, 0.03)  # Humidity variation
        # Keep other variables relatively stable
        
    return future_data

if __name__ == "__main__":
    # Example usage
    print("Exploration MPPI Controller")
    print("=" * 50)
    
    # Create dummy setup
    gp_model, info_gain_fn, z_dataset = create_dummy_exploration_setup()
    
    # Create controller
    controller = ExplorationMPPIController(
        gp_model=gp_model,
        z_dataset=z_dataset,
        information_gain_fn=info_gain_fn,
        horizon=10,
        num_samples=100,
        exploration_weight=1.0,
        safety_weight=0.1
    )
    
    # Create dummy state and future data
    current_state = np.random.randn(9)
    future_env_data = create_future_env_data_exploration(current_state, horizon=10)
    
    # Create dummy z_list (exploration targets)
    z_list = [np.random.randn(10) for _ in range(20)]  # 20 z points, each 10D (state+action)
    
    # Test exploration mode
    exploration_flag = 1.0  # Full exploration
    dropped_pairs, action, is_fallback = controller.plan(
        current_state, future_env_data, exploration_flag, z_list
    )
    
    print(f"Exploration mode - Planned action: {action}")
    print(f"Is fallback: {is_fallback}")
    print(f"Dropped pairs: {len(dropped_pairs)}")
    
    # Test normal operation mode
    exploration_flag = 0.0  # Normal operation
    dropped_pairs, action, is_fallback = controller.plan(
        current_state, future_env_data, exploration_flag, z_list
    )
    
    print(f"Normal operation mode - Planned action: {action}")
    print(f"Is fallback: {is_fallback}")
    print(f"Dropped pairs: {len(dropped_pairs)}")
    print(f"Evaluation metrics: {controller.get_evaluation_metrics()}") 