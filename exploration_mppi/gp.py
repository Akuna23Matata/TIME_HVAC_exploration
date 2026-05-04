import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, WhiteKernel, ConstantKernel
from scipy.special import erf

class HVACGaussianProcess:
    def __init__(self, input_dim=10, predict_delta=True, safety_threshold=0.0):
        """
        Sklearn-based GP for HVAC temperature prediction with information gain
        
        Args:
            input_dim: Now 10 (9 observations + 1 action) instead of 11
            safety_threshold: Temperature change threshold for "safety" 
                            (e.g., 0.0 means no large temperature swings)
        """
        self.input_dim = input_dim
        self.predict_delta = predict_delta
        self.safety_threshold = safety_threshold  # For safety indicator Ψ(z)
        
        # Design kernel for HVAC dynamics
        self.kernel = self._create_hvac_kernel()
        
        # Initialize GP
        self.gp = GaussianProcessRegressor(
            kernel=self.kernel,
            alpha=1e-6,  # Small noise for numerical stability
            normalize_y=False,  # Don't normalize targets - data is already normalized
            n_restarts_optimizer=3
        )
        
        self.is_fitted = False
        
    def _create_hvac_kernel(self):
        """
        Simplified kernel for normalized inputs [-1, 1]
        """
        return RBF(length_scale=1.0) + WhiteKernel(noise_level=1e-5)
    
    def prepare_input(self, observation, action):
        """
        Combine observation and action into GP input
        Action is now a single scalar value
        """
        observation = np.asarray(observation)
        action = np.asarray(action)
        
        # Ensure action is a scalar (convert to scalar if it's a 1D array)
        if action.ndim > 0:
            action = action.item() if action.size == 1 else action[0]
        
        gp_input = np.concatenate([observation, [action]])
        return gp_input.reshape(1, -1)
    
    def fit(self, observations, actions, next_indoor_temps):
        """Train the GP model"""
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        next_indoor_temps = np.asarray(next_indoor_temps)
        
        # Handle single action case
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        elif actions.ndim == 2 and actions.shape[1] > 1:
            # If actions still has 2 columns, take only the first one
            actions = actions[:, 0:1]
            
        current_indoor_temps = observations[:, 6]
        X = np.concatenate([observations, actions], axis=1)
        
        if self.predict_delta:
            y = next_indoor_temps - current_indoor_temps
        else:
            y = next_indoor_temps
            
        self.gp.fit(X, y)
        self.is_fitted = True
        
    def predict(self, observation, action, return_std=True):
        """Predict next indoor temperature"""
        if not self.is_fitted:
            raise ValueError("GP must be fitted before making predictions")
            
        X = self.prepare_input(observation, action)
        
        if return_std:
            pred_mean, pred_std = self.gp.predict(X, return_std=True)
            pred_mean, pred_std = pred_mean[0], pred_std[0]
        else:
            pred_mean = self.gp.predict(X)[0]
            pred_std = None
            
        if self.predict_delta:
            current_indoor_temp = observation[6]
            next_indoor_temp = current_indoor_temp + pred_mean
        else:
            next_indoor_temp = pred_mean
            
        if return_std:
            return next_indoor_temp, pred_std
        else:
            return next_indoor_temp
    
    def predict_batch(self, observations, actions, return_std=True):
        """Batch prediction for efficiency"""
        if not self.is_fitted:
            raise ValueError("GP must be fitted before making predictions")
        
        observations = np.asarray(observations)
        actions = np.asarray(actions)
        
        # Handle single action case
        if actions.ndim == 1:
            actions = actions.reshape(-1, 1)
        elif actions.ndim == 2 and actions.shape[1] > 1:
            # If actions still has 2 columns, take only the first one
            actions = actions[:, 0:1]
            
        X = np.concatenate([observations, actions], axis=1)
        
        if return_std:
            pred_means, pred_stds = self.gp.predict(X, return_std=True)
        else:
            pred_means = self.gp.predict(X)
            pred_stds = None
            
        if self.predict_delta:
            current_indoor_temps = observations[:, 6]
            next_indoor_temps = current_indoor_temps + pred_means
        else:
            next_indoor_temps = pred_means
            
        if return_std:
            return next_indoor_temps, pred_stds
        else:
            return next_indoor_temps
    
    def get_gp_prediction_raw(self, x):
        """
        Get raw GP prediction (delta or absolute) without post-processing
        Used for information gain calculations
        """
        if not self.is_fitted:
            raise ValueError("GP must be fitted before making predictions")
        
        x = np.asarray(x).reshape(1, -1)
        pred_mean, pred_std = self.gp.predict(x, return_std=True)
        return pred_mean[0], pred_std[0]
    
    def correlation_coefficient(self, x1, x2):
        """
        Compute POSTERIOR correlation coefficient between GP predictions at x1 and x2
        """
        x1 = np.asarray(x1).reshape(1, -1)
        x2 = np.asarray(x2).reshape(1, -1)
        
        # Stack points and get posterior covariance
        points = np.vstack([x1, x2])
        
        # Get posterior mean and covariance (this depends on your GP library)
        # For sklearn-style: 
        posterior_cov = self.gp.predict(points, return_cov=True)[1]
        
        # Extract posterior variances and covariance
        var_x1 = posterior_cov[0, 0]
        var_x2 = posterior_cov[1, 1] 
        cov_x1_x2 = posterior_cov[0, 1]
    
        # Posterior correlation coefficient
        rho = cov_x1_x2 / np.sqrt(var_x1 * var_x2)
        return rho
    
    def entropy_psi(self, mu, sigma):
        """
        Compute entropy of safety indicator Ψ(z) using ISE approximation (Eq. 4)
        H[Ψ(z)] ≈ ln(2) * exp(-1/(π*ln(2)) * (μ/σ)²)
        """
        if sigma <= 1e-8:  # Avoid division by zero
            return 0.0
        
        c1 = 1.0 / (np.pi * np.log(2))  # Constant from ISE paper
        ratio_squared = (mu / sigma) ** 2
        entropy = np.log(2) * np.exp(-c1 * ratio_squared)
        return entropy
    
    def expected_entropy_after_observation(self, x, z):
        """
        Compute expected entropy of Ψ(z) after observing f(x)
        Using ISE analytical approximation (Eq. 5)
        """
        # Get GP predictions
        mu_x, sigma_x = self.get_gp_prediction_raw(x)
        mu_z, sigma_z = self.get_gp_prediction_raw(z)
        
        # Correlation coefficient
        rho = self.correlation_coefficient(x, z)
        
        # Constants from ISE paper
        c1 = 1.0 / (np.pi * np.log(2))
        c2 = 2 * c1 - 1
        
        # Noise variance (assuming observation noise)
        sigma_v_squared = 1e-6  # Small observation noise
        
        # ISE formula (Eq. 5) adapted
        rho_v_squared = sigma_x**2 / (sigma_v_squared + sigma_x**2)
        
        numerator_exp = -c1 * (mu_z**2 / sigma_z**2) * (
            (sigma_v_squared + sigma_x**2) / 
            (sigma_v_squared + sigma_x**2 * (1 + c2 * rho**2 * rho_v_squared))
        )
        
        denominator_sqrt = np.sqrt(
            (sigma_v_squared + sigma_x**2 * (1 - rho**2 * rho_v_squared)) /
            (sigma_v_squared + sigma_x**2 * (1 + c2 * rho**2 * rho_v_squared))
        )
        
        expected_entropy = np.log(2) * denominator_sqrt * np.exp(numerator_exp)
        
        return expected_entropy
    
    def information_gain(self, x, z):
        """
        Compute mutual information I({x,y}; Ψ(z)) using ISE approximation
        
        Args:
            x: Exploration point (observation + action)
            z: Target point we care about predicting
            
        Returns:
            Information gain value
        """
        if not self.is_fitted:
            raise ValueError("GP must be fitted before computing information gain")
        
        # Get GP prediction at target point z
        mu_z, sigma_z = self.get_gp_prediction_raw(z)
        
        # Prior entropy of safety indicator at z
        entropy_prior = self.entropy_psi(mu_z, sigma_z)
        
        # Expected entropy after observing at x
        entropy_posterior = self.expected_entropy_after_observation(x, z)
        
        # Information gain = reduction in entropy
        info_gain = entropy_prior - entropy_posterior
        
        return max(0.0, info_gain)  # Ensure non-negative
    
    def multi_target_information_gain(self, x, z_targets, weights=None):
        """
        Compute total information gain for multiple target points
        
        Args:
            x: Exploration point (observation + action)
            z_targets: List of target points we care about
            weights: Optional weights for each target (default: uniform)
            
        Returns:
            Total weighted information gain
        """
        if weights is None:
            weights = np.ones(len(z_targets)) / len(z_targets)
        
        total_gain = 0.0
        for i, z in enumerate(z_targets):
            gain = self.information_gain(x, z)
            total_gain += weights[i] * gain
        
        return total_gain
    
    def select_best_exploration_point(self, candidate_x_points, z_targets, weights=None):
        """
        Select the exploration point that maximizes information gain
        
        Args:
            candidate_x_points: List of candidate exploration points
            z_targets: List of target points we care about
            weights: Optional weights for each target
            
        Returns:
            best_x: Best exploration point
            best_gain: Information gain value
            all_gains: Information gains for all candidates
        """
        all_gains = []
        
        for x in candidate_x_points:
            gain = self.multi_target_information_gain(x, z_targets, weights)
            all_gains.append(gain)
        
        best_idx = np.argmax(all_gains)
        best_x = candidate_x_points[best_idx]
        best_gain = all_gains[best_idx]
        
        return best_x, best_gain, all_gains

# Updated test to include information gain with single action
if __name__ == "__main__":
    print("Testing HVAC GP with Single Action and Information Gain...")
    
    # Generate synthetic data
    n_samples = 100
    np.random.seed(423)
    
    observations = np.random.uniform(-1, 1, (n_samples, 9))
    print(observations[:10])
    actions = np.random.uniform(-1, 1, (n_samples, 1))  # Single action now
    print(actions[:10])
    current_temps = observations[:, 6]
    
    # Simplified temperature dynamics with single action
    # Positive action = heating, negative action = cooling
    temp_changes = 0.2 * actions[:, 0] + 0.01 * np.random.randn(n_samples)
    next_temps = current_temps + temp_changes
    
    # Train GP
    hvac_gp = HVACGaussianProcess(input_dim=10, predict_delta=True)  # 9 + 1 = 10
    hvac_gp.fit(observations, actions, next_temps)
    
    print("Testing single action prediction...")
    test_obs = np.random.uniform(-1, 1, 9)
    test_action = 0.5  # Single scalar action
    
    pred_temp, pred_std = hvac_gp.predict(test_obs, test_action)
    print(f"Predicted temperature: {pred_temp:.3f} ± {pred_std:.3f}")
    
    print("Testing information gain calculation...")
    
    # Define some target points (z) we care about - now with single action
    z_targets = [
        np.concatenate([np.random.uniform(-1, 1, 9), [0.8]]),   # Strong heating
        np.concatenate([np.random.uniform(-1, 1, 9), [-0.8]]),  # Strong cooling
        np.concatenate([np.random.uniform(-1, 1, 9), [0.0]]),   # No action
    ]
    
    # Define candidate exploration points (x) - now with single action
    candidate_x_points = [
        np.concatenate([np.random.uniform(-1, 1, 9), [0.3]]),   # Mild heating
        np.concatenate([np.random.uniform(-1, 1, 9), [-0.3]]),  # Mild cooling
        np.concatenate([np.random.uniform(-1, 1, 9), [0.9]]),   # Strong heating
    ]
    
    # Find best exploration point
    best_x, best_gain, all_gains = hvac_gp.select_best_exploration_point(
        candidate_x_points, z_targets
    )
    
    print(f"Best exploration point index: {np.argmax(all_gains)}")
    print(f"Best information gain: {best_gain:.6f}")
    print(f"All gains: {all_gains}")
    
    # Test batch prediction
    print("\nTesting batch prediction...")
    batch_obs = np.random.uniform(-1, 1, (5, 9))
    batch_actions = np.random.uniform(-1, 1, 5)  # Single actions
    
    batch_preds, batch_stds = hvac_gp.predict_batch(batch_obs, batch_actions)
    print(f"Batch predictions shape: {batch_preds.shape}")
    print(f"Batch predictions: {batch_preds}")