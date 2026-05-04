import numpy as np
from typing import List, Optional, Tuple, Union
import logging
import pandas as pd
from scipy.stats import truncnorm

class ZDataset:
    """
    Dataset class to manage z points (target points) for information gain calculations
    in GP-based HVAC control. The dataset maintains two types of points:
    - Type 1 (Cold Start): Points generated during initialization
    - Type 2 (Operational): Points generated during operation
    
    Each z point is a numpy array of shape (10,) containing:
    [observation (9 elements) + action (1 element)]
    
    The dataset is designed to be directly compatible with the GP model's
    multi_target_information_gain function's z_targets parameter.
    """
    
    def __init__(self, 
                 max_size_type1: int = 100, 
                 max_size_type2: int = 900,
                 observation_dim: int = 9, 
                 action_dim: int = 1):
        """
        Initialize the Z dataset with separate storage for two types of points
        
        Args:
            max_size_type1: Maximum number of Type 1 (cold start) points to store
            max_size_type2: Maximum number of Type 2 (operational) points to store
            observation_dim: Dimension of observation space (default: 9)
            action_dim: Dimension of action space (default: 1)
        """
        self.max_size_type1 = max_size_type1
        self.max_size_type2 = max_size_type2
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.total_dim = observation_dim + action_dim
        
        # Type 1 storage (Cold Start)
        self.z_points_type1: List[np.ndarray] = []
        self.metadata_type1: List[dict] = []
        
        # Type 2 storage (Operational)
        self.z_points_type2: List[np.ndarray] = []
        self.metadata_type2: List[dict] = []
        
        # Statistics
        self.num_points_type1 = 0
        self.num_points_type2 = 0
        self.num_added_type1 = 0
        self.num_added_type2 = 0
        self.num_purged_type1 = 0
        self.num_purged_type2 = 0
        
        self.logger = logging.getLogger(__name__)
        
    @property
    def num_points(self) -> int:
        """Total number of points (Type 1 + Type 2)"""
        return self.num_points_type1 + self.num_points_type2
    
    @property
    def max_size(self) -> int:
        """Total maximum size (Type 1 + Type 2)"""
        return self.max_size_type1 + self.max_size_type2
        
    def initialize_dataset(self, initial_points_type1: Optional[List[np.ndarray]] = None):
        """
        Initialize the dataset with optional initial Type 1 (cold start) points
        
        Args:
            initial_points_type1: Optional list of initial Type 1 z points to populate the dataset
        """
        # Clear Type 1 storage
        self.z_points_type1.clear()
        self.metadata_type1.clear()
        self.num_points_type1 = 0
        self.num_added_type1 = 0
        self.num_purged_type1 = 0
        
        # Clear Type 2 storage
        self.z_points_type2.clear()
        self.metadata_type2.clear()
        self.num_points_type2 = 0
        self.num_added_type2 = 0
        self.num_purged_type2 = 0
        
        if initial_points_type1 is not None:
            for point in initial_points_type1:
                self.add_type1_point(point)
                
        self.logger.info(f"Dataset initialized with {self.num_points_type1} Type 1 points")
        
    def add_type1_point(self, z_point: np.ndarray, metadata: Optional[dict] = None) -> bool:
        """
        Add a Type 1 (cold start) z point to the dataset
        
        Args:
            z_point: Numpy array of shape (10,) containing [observation + action]
            metadata: Optional metadata dictionary for the point
            
        Returns:
            bool: True if point was added successfully, False otherwise
        """
        # Validate input
        if not self._validate_point(z_point):
            return False
            
        # Check if Type 1 storage is full
        if self.num_points_type1 >= self.max_size_type1:
            self.logger.warning(f"Type 1 storage is full ({self.max_size_type1} points). Cannot add new point.")
            return False
            
        # Add point
        metadata = metadata if metadata is not None else {}
        metadata['type'] = 1  # Mark as Type 1
        
        self.z_points_type1.append(z_point.copy())
        self.metadata_type1.append(metadata)
        self.num_points_type1 += 1
        self.num_added_type1 += 1
        
        self.logger.debug(f"Added Type 1 point {self.num_added_type1}. Type 1 size: {self.num_points_type1}")
        return True
        
    def add_type2_point(self, z_point: np.ndarray, metadata: Optional[dict] = None) -> bool:
        """
        Add a Type 2 (operational) z point to the dataset
        
        Args:
            z_point: Numpy array of shape (10,) containing [observation + action]
            metadata: Optional metadata dictionary for the point
            
        Returns:
            bool: True if point was added successfully, False otherwise
        """
        # Validate input
        if not self._validate_point(z_point):
            return False
            
        # Check if Type 2 storage is full
        if self.num_points_type2 >= self.max_size_type2:
            self.logger.warning(f"Type 2 storage is full ({self.max_size_type2} points). Cannot add new point.")
            return False
            
        # Add point
        metadata = metadata if metadata is not None else {}
        metadata['type'] = 2  # Mark as Type 2
        
        self.z_points_type2.append(z_point.copy())
        self.metadata_type2.append(metadata)
        self.num_points_type2 += 1
        self.num_added_type2 += 1
        
        self.logger.debug(f"Added Type 2 point {self.num_added_type2}. Type 2 size: {self.num_points_type2}")
        return True
        
    def add_point(self, z_point: np.ndarray, metadata: Optional[dict] = None) -> bool:
        """
        Add a z point to the dataset (defaults to Type 1 for backward compatibility)
        
        Args:
            z_point: Numpy array of shape (10,) containing [observation + action]
            metadata: Optional metadata dictionary for the point
            
        Returns:
            bool: True if point was added successfully, False otherwise
        """
        return self.add_type1_point(z_point, metadata)
        
    def purge_dataset(self, purge_strategy: str = "oldest", purge_ratio: float = 0.5):
        """
        Cleanup the dataset by removing points based on the specified strategy
        
        Args:
            purge_strategy: Strategy for purging points
                - "oldest": Remove oldest points first
                - "random": Remove random points
                - "custom": Custom purging logic (to be implemented)
            purge_ratio: Ratio of points to remove (0.0 to 1.0)
        """
        if purge_ratio <= 0.0 or purge_ratio >= 1.0:
            self.logger.warning(f"Invalid purge ratio: {purge_ratio}. Must be between 0.0 and 1.0")
            return
            
        num_to_remove = int(self.num_points * purge_ratio)
        if num_to_remove <= 0:
            return
            
        if purge_strategy == "oldest":
            self._purge_oldest(num_to_remove)
        elif purge_strategy == "random":
            self._purge_random(num_to_remove)
        elif purge_strategy == "custom":
            self._purge_custom(num_to_remove)
        else:
            self.logger.error(f"Unknown purge strategy: {purge_strategy}")
            return
            
        self.logger.info(f"Purged {num_to_remove} points. Dataset size: {self.num_points}")
        
    def remove_type2_point(self, index: int) -> bool:
        """
        Remove a specific point from Type 2 (operational) data by index
        
        Args:
            index: Index of the point to remove (0-based within Type 2 data)
            
        Returns:
            bool: True if point was removed successfully, False otherwise
        """
        if index < 0 or index >= self.num_points_type2:
            self.logger.warning(f"Index {index} out of range for Type 2 data (size: {self.num_points_type2})")
            return False
            
        # Remove the point at specified index
        self.z_points_type2.pop(index)
        self.metadata_type2.pop(index)
        self.num_points_type2 -= 1
        self.num_purged_type2 += 1
        
        self.logger.debug(f"Removed Type 2 point at index {index}. Type 2 size: {self.num_points_type2}")
        return True
        
    def update_type1_dataset(self, truth_table: np.ndarray, obs_mean: np.ndarray, obs_var: np.ndarray, 
                           temp_mean: float = 24.5, temp_var: float = 3.0, occupied_ratio: float = 0.8) -> bool:
        """
        Update Type 1 dataset by clearing existing points and repopulating with cold start logic
        
        This is useful when external conditions change over time and we need to refresh 
        the Type 1 (cold start) points to reflect current conditions.
        
        Args:
            truth_table: Truth table data from environment
            obs_mean: Observation mean for normalization
            obs_var: Observation variance for normalization
            temp_mean: Mean temperature for sampling (default: 24.5°C)
            temp_var: Temperature variance for sampling (default: 3.0)
            occupied_ratio: Ratio of occupied to unoccupied hours (default: 0.8)
            
        Returns:
            bool: True if update was successful, False otherwise
        """
        try:
            # Clear existing Type 1 data
            old_count = self.num_points_type1
            self.clear_type1_dataset()
            
            # Repopulate with cold start logic
            cold_start_populate_dataset(self, truth_table, obs_mean, obs_var, 
                                      temp_mean, temp_var, occupied_ratio)
            
            self.logger.info(f"Updated Type 1 dataset: {old_count} → {self.num_points_type1} points")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to update Type 1 dataset: {e}")
            return False
        
    def get_z_targets(self, weights: Optional[List[float]] = None, balanced_sampling: bool = True) -> Tuple[List[np.ndarray], Optional[List[float]]]:
        """
        Get the combined z points list (Type 1 + Type 2) in the format expected by multi_target_information_gain
        
        Args:
            weights: Optional weights for each z point
            balanced_sampling: If True, sample Type 2 points to maintain 1:1 ratio with Type 1
                             If False, return all points (original behavior)
            
        Returns:
            Tuple of (z_targets, weights) where:
                - z_targets: List of numpy arrays compatible with multi_target_information_gain
                - weights: List of weights (None if not provided)
        """
        if balanced_sampling:
            # Get all Type 1 points
            type1_points = self.z_points_type1.copy()
            type1_count = len(type1_points)
            
            # Sample Type 2 points to match Type 1 count (1:1 ratio)
            if self.num_points_type2 > 0:
                if type1_count > 0:
                    # Sample min(type1_count, num_points_type2) points from Type 2
                    sample_size = min(type1_count, self.num_points_type2)
                    type2_indices = np.random.choice(self.num_points_type2, sample_size, replace=False)
                    type2_points = [self.z_points_type2[i] for i in type2_indices]
                else:
                    # No Type 1 points, return empty list
                    type2_points = []
            else:
                # No Type 2 points
                type2_points = []
            
            # Combine Type 1 and sampled Type 2 points
            combined_z_points = type1_points + type2_points
            
            # Update weights if provided
            if weights is not None:
                if len(weights) != len(combined_z_points):
                    self.logger.warning(f"Weights length ({len(weights)}) doesn't match balanced dataset size ({len(combined_z_points)})")
                    weights = None
            
            self.logger.debug(f"Balanced sampling: {len(type1_points)} Type 1 + {len(type2_points)} Type 2 = {len(combined_z_points)} total")
            
        else:
            # Original behavior: return all points
            if weights is not None and len(weights) != self.num_points:
                self.logger.warning(f"Weights length ({len(weights)}) doesn't match dataset size ({self.num_points})")
                weights = None
                
            # Combine all Type 1 and Type 2 points
            combined_z_points = self.z_points_type1.copy() + self.z_points_type2.copy()
        
        return combined_z_points, weights
        
    def get_balanced_z_targets(self, target_size: Optional[int] = None) -> Tuple[List[np.ndarray], Optional[List[float]]]:
        """
        Get z targets with balanced 1:1 ratio between Type 1 and Type 2 points
        
        Args:
            target_size: Optional target size. If None, uses 2 * num_points_type1
                        If specified, samples target_size//2 from each type
            
        Returns:
            Tuple of (z_targets, weights) where:
                - z_targets: List of numpy arrays with balanced Type 1 and Type 2 points
                - weights: None (weights not supported in balanced mode)
        """
        if target_size is None:
            # Default: 2 * num_points_type1 (1:1 ratio)
            target_size = 2 * self.num_points_type1
        
        # Calculate how many points to sample from each type
        points_per_type = target_size // 2
        
        # Get Type 1 points (all available, up to points_per_type)
        type1_points = self.z_points_type1.copy()
        if len(type1_points) > points_per_type:
            # Randomly sample from Type 1 if we have too many
            type1_indices = np.random.choice(len(type1_points), points_per_type, replace=False)
            type1_points = [type1_points[i] for i in type1_indices]
        
        # Sample Type 2 points
        if self.num_points_type2 > 0:
            # Sample min(points_per_type, num_points_type2) points from Type 2
            type2_sample_size = min(points_per_type, self.num_points_type2)
            type2_indices = np.random.choice(self.num_points_type2, type2_sample_size, replace=False)
            type2_points = [self.z_points_type2[i] for i in type2_indices]
        else:
            type2_points = []
        
        # Combine points
        combined_z_points = type1_points + type2_points
        
        self.logger.debug(f"Balanced z targets: {len(type1_points)} Type 1 + {len(type2_points)} Type 2 = {len(combined_z_points)} total (target: {target_size})")
        
        return combined_z_points, None
        
    def get_z_targets_by_type(self, point_type: int, weights: Optional[List[float]] = None) -> Tuple[List[np.ndarray], Optional[List[float]]]:
        """
        Get z points of a specific type
        
        Args:
            point_type: 1 for Type 1 (cold start), 2 for Type 2 (operational)
            weights: Optional weights for each z point
            
        Returns:
            Tuple of (z_targets, weights) where:
                - z_targets: List of numpy arrays of the specified type
                - weights: List of weights (None if not provided)
        """
        if point_type == 1:
            z_points = self.z_points_type1.copy()
            expected_size = self.num_points_type1
        elif point_type == 2:
            z_points = self.z_points_type2.copy()
            expected_size = self.num_points_type2
        else:
            raise ValueError(f"Invalid point_type: {point_type}. Must be 1 or 2.")
            
        if weights is not None and len(weights) != expected_size:
            self.logger.warning(f"Weights length ({len(weights)}) doesn't match Type {point_type} size ({expected_size})")
            weights = None
            
        return z_points, weights
        
    def get_dataset_info(self) -> dict:
        """
        Get information about the current dataset state
        
        Returns:
            dict: Dictionary containing dataset statistics
        """
        return {
            "num_points": self.num_points,
            "max_size": self.max_size,
            "num_points_type1": self.num_points_type1,
            "num_points_type2": self.num_points_type2,
            "max_size_type1": self.max_size_type1,
            "max_size_type2": self.max_size_type2,
            "num_added_type1": self.num_added_type1,
            "num_added_type2": self.num_added_type2,
            "num_purged_type1": self.num_purged_type1,
            "num_purged_type2": self.num_purged_type2,
            "observation_dim": self.observation_dim,
            "action_dim": self.action_dim,
            "total_dim": self.total_dim,
            "is_full": self.num_points >= self.max_size,
            "is_type1_full": self.num_points_type1 >= self.max_size_type1,
            "is_type2_full": self.num_points_type2 >= self.max_size_type2
        }
        
    def _validate_point(self, z_point: np.ndarray) -> bool:
        """
        Validate that a z point has the correct format
        
        Args:
            z_point: Point to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        if not isinstance(z_point, np.ndarray):
            self.logger.error("Z point must be a numpy array")
            return False
            
        if z_point.shape != (self.total_dim,):
            self.logger.error(f"Z point shape {z_point.shape} doesn't match expected {(self.total_dim,)}")
            return False
            
        if not np.all(np.isfinite(z_point)):
            self.logger.error("Z point contains non-finite values")
            return False
            
        return True
        
    def _purge_oldest(self, num_to_remove: int):
        """Remove oldest points from the dataset (prioritizes Type 1 first, then Type 2)"""
        removed_count = 0
        
        # First remove from Type 1 if available
        type1_to_remove = min(num_to_remove, self.num_points_type1)
        if type1_to_remove > 0:
            self.z_points_type1 = self.z_points_type1[type1_to_remove:]
            self.metadata_type1 = self.metadata_type1[type1_to_remove:]
            self.num_points_type1 -= type1_to_remove
            self.num_purged_type1 += type1_to_remove
            removed_count += type1_to_remove
            
        # Then remove from Type 2 if needed
        remaining_to_remove = num_to_remove - removed_count
        if remaining_to_remove > 0 and self.num_points_type2 > 0:
            type2_to_remove = min(remaining_to_remove, self.num_points_type2)
            self.z_points_type2 = self.z_points_type2[type2_to_remove:]
            self.metadata_type2 = self.metadata_type2[type2_to_remove:]
            self.num_points_type2 -= type2_to_remove
            self.num_purged_type2 += type2_to_remove
            removed_count += type2_to_remove
            
        self.logger.debug(f"Purged {removed_count} oldest points (Type1: {type1_to_remove}, Type2: {remaining_to_remove})")
        
    def _purge_random(self, num_to_remove: int):
        """Remove random points from the dataset (from both Type 1 and Type 2)"""
        if self.num_points == 0:
            return
            
        # Get indices for combined dataset
        all_indices = list(range(self.num_points))
        indices_to_remove = np.random.choice(all_indices, min(num_to_remove, len(all_indices)), replace=False)
        indices_to_remove = sorted(indices_to_remove, reverse=True)
        
        type1_removed = 0
        type2_removed = 0
        
        for idx in indices_to_remove:
            if idx < self.num_points_type1:
                # Remove from Type 1
                actual_idx = idx - type1_removed  # Adjust for already removed items
                self.z_points_type1.pop(actual_idx)
                self.metadata_type1.pop(actual_idx)
                self.num_points_type1 -= 1
                type1_removed += 1
            else:
                # Remove from Type 2
                type2_idx = idx - self.num_points_type1 - type2_removed
                self.z_points_type2.pop(type2_idx)
                self.metadata_type2.pop(type2_idx)
                self.num_points_type2 -= 1
                type2_removed += 1
                
        self.num_purged_type1 += type1_removed
        self.num_purged_type2 += type2_removed
        
        self.logger.debug(f"Purged {len(indices_to_remove)} random points (Type1: {type1_removed}, Type2: {type2_removed})")
        
    def _purge_custom(self, num_to_remove: int):
        """
        Custom purging logic - placeholder for future implementation
        
        This function can be extended with domain-specific logic such as:
        - Removing points with high prediction uncertainty
        - Removing points that are too similar to existing points
        - Removing points based on temporal relevance
        - etc.
        """
        # For now, fallback to oldest strategy
        self.logger.info("Using custom purging strategy (placeholder - falling back to oldest)")
        self._purge_oldest(num_to_remove)
        
    def purge_type2_by_uncertainty(self, gp_model, uncertainty_threshold: float) -> int:
        """
        Purge Type 2 data based on GP uncertainty evaluation
        
        This function evaluates the uncertainty of each Type 2 point using the provided GP model.
        Points with uncertainty below the threshold are considered "well-learned" and are removed
        to make space for new, potentially more uncertain points.
        
        Args:
            gp_model: Trained GP model with predict method that returns (mean, std)
            uncertainty_threshold: Points with uncertainty < threshold will be removed
            
        Returns:
            int: Number of points removed
        """
        if self.num_points_type2 == 0:
            self.logger.info("No Type 2 points to purge")
            return 0
            
        self.logger.info(f"Purging Type 2 data by uncertainty (threshold: {uncertainty_threshold})")
        
        # Evaluate uncertainty for each Type 2 point
        uncertainties = []
        for i, z_point in enumerate(self.z_points_type2):
            try:
                # Extract observation and action from z point
                obs = z_point[:9]  # First 9 elements are observation
                action = z_point[9]  # Last element is action
                
                # Get prediction uncertainty from GP model
                # Note: This assumes the GP model has a predict method that returns (mean, std)
                # Adjust the interface based on your actual GP model implementation
                if hasattr(gp_model, 'predict'):
                    # Try different prediction interfaces
                    try:
                        # Interface 1: predict(obs, action, return_std=True)
                        _, uncertainty = gp_model.predict(obs, action, return_std=True)
                    except TypeError:
                        try:
                            # Interface 2: predict(obs, action)
                            _, uncertainty = gp_model.predict(obs, action)
                        except:
                            # Interface 3: predict(z_point)
                            _, uncertainty = gp_model.predict(z_point.reshape(1, -1))
                else:
                    # Fallback: assume uniform uncertainty if GP model doesn't have predict method
                    self.logger.warning("GP model doesn't have predict method, using uniform uncertainty")
                    uncertainty = 1.0
                    
                uncertainties.append(uncertainty)
                
            except Exception as e:
                self.logger.warning(f"Failed to evaluate uncertainty for Type 2 point {i}: {e}")
                # Assign high uncertainty to keep the point if evaluation fails
                uncertainties.append(uncertainty_threshold + 1.0)
        
        # Find points to remove (uncertainty < threshold)
        points_to_remove = []
        for i, uncertainty in enumerate(uncertainties):
            if uncertainty < uncertainty_threshold:
                points_to_remove.append(i)
        
        # Remove points in reverse order to maintain correct indices
        points_to_remove.reverse()
        removed_count = 0
        
        for idx in points_to_remove:
            if self.remove_type2_point(idx):
                removed_count += 1
        
        # Log statistics
        if uncertainties:
            uncertainty_stats = {
                'mean': np.mean(uncertainties),
                'std': np.std(uncertainties),
                'min': np.min(uncertainties),
                'max': np.max(uncertainties),
                'below_threshold': len(points_to_remove),
                'above_threshold': len(uncertainties) - len(points_to_remove)
            }
            
            self.logger.info(f"Uncertainty stats: mean={uncertainty_stats['mean']:.4f}, "
                           f"std={uncertainty_stats['std']:.4f}, "
                           f"range=[{uncertainty_stats['min']:.4f}, {uncertainty_stats['max']:.4f}]")
            self.logger.info(f"Removed {removed_count} points with uncertainty < {uncertainty_threshold}")
            self.logger.info(f"Kept {uncertainty_stats['above_threshold']} points with uncertainty >= {uncertainty_threshold}")
        
        return removed_count
        
    def clear_dataset(self):
        """Clear all points from the dataset"""
        self.z_points_type1.clear()
        self.metadata_type1.clear()
        self.z_points_type2.clear()
        self.metadata_type2.clear()
        self.num_points_type1 = 0
        self.num_points_type2 = 0
        self.logger.info("Dataset cleared")
        
    def clear_type1_dataset(self):
        """Clear only Type 1 points from the dataset"""
        self.z_points_type1.clear()
        self.metadata_type1.clear()
        self.num_points_type1 = 0
        self.logger.info("Type 1 dataset cleared")
        
    def clear_type2_dataset(self):
        """Clear only Type 2 points from the dataset"""
        self.z_points_type2.clear()
        self.metadata_type2.clear()
        self.num_points_type2 = 0
        self.logger.info("Type 2 dataset cleared")
        
    def __len__(self) -> int:
        """Return the total number of points in the dataset"""
        return self.num_points
        
    def __getitem__(self, index: int) -> np.ndarray:
        """Get a specific z point by index (combined Type 1 + Type 2)"""
        if index >= self.num_points:
            raise IndexError(f"Index {index} out of range for dataset of size {self.num_points}")
        
        # Check if index is in Type 1 range
        if index < self.num_points_type1:
            return self.z_points_type1[index]
        else:
            # Index is in Type 2 range
            type2_index = index - self.num_points_type1
            return self.z_points_type2[type2_index]
        
    def __repr__(self) -> str:
        """String representation of the dataset"""
        return f"ZDataset(size={self.num_points}/{self.max_size}, Type1={self.num_points_type1}/{self.max_size_type1}, Type2={self.num_points_type2}/{self.max_size_type2}, dim={self.total_dim})"


# Helper functions for creating z points
def create_z_point(observation: np.ndarray, action: Union[int, float]) -> np.ndarray:
    """
    Create a z point from observation and action
    
    Args:
        observation: Observation array of shape (9,)
        action: Action value (will be converted to float)
        
    Returns:
        np.ndarray: Z point of shape (10,)
    """
    observation = np.asarray(observation)
    action = np.asarray([action])
    
    if observation.shape != (9,):
        raise ValueError(f"Observation shape {observation.shape} doesn't match expected (9,)")
    
    return np.concatenate([observation, action])


def create_future_z_points(current_observation: np.ndarray, 
                          future_env_data: np.ndarray, 
                          actions: List[Union[int, float]]) -> List[np.ndarray]:
    """
    Create z points for future states with different actions
    
    Args:
        current_observation: Current observation of shape (9,)
        future_env_data: Future environmental data of shape (horizon, 9)
        actions: List of actions to create z points for
        
    Returns:
        List[np.ndarray]: List of z points for future states
    """
    z_points = []
    
    # Create z points for current state with different actions
    for action in actions:
        z_point = create_z_point(current_observation, action)
        z_points.append(z_point)
    
    # Create z points for future states (can be extended)
    # For now, just use the first few future states
    for i in range(min(3, len(future_env_data))):  # Use first 3 future states
        future_obs = future_env_data[i]
        for action in actions:
            z_point = create_z_point(future_obs, action)
            z_points.append(z_point)
    
    return z_points


def cold_start_populate_dataset(
    dataset: ZDataset,
    truth_table: pd.DataFrame,
    obs_mean: List[float],
    obs_var: List[float],
    comfort_bounds: Tuple[float, float] = (23, 26),
    occupied_hours: Tuple[int, int] = (8, 17),
    temperature_std: float = 1.5,
    temperature_extension: float = 2.0,
    target_fill_ratio: float = 1.0,
    occupied_ratio: float = 0.8
) -> None:
    """
    Populate the dataset with z points for cold start
    
    Logic:
    1. Sample indoor temperatures from Gaussian distribution (centered on comfort zone)
    2. Sample observations from truth table (80% occupied hours, 20% unoccupied hours)
    3. Combine by replacing obs[6] with sampled temperature
    4. Add random normalized action (0-9 -> [-1,1])
    
    Args:
        dataset: ZDataset to populate
        truth_table: DataFrame with environmental data from collect_truth_table
        obs_mean: Mean values for normalization [9]
        obs_var: Variance values for normalization [9]
        comfort_bounds: Comfort zone bounds in actual temperature (lower, upper)
        occupied_hours: Occupied hours range (start, end) in 24-hour format
        temperature_std: Standard deviation for temperature sampling around comfort center
        temperature_extension: How far beyond comfort zone to extend sampling (°C)
        target_fill_ratio: Fraction of max_size to fill (0.0 to 1.0)
        occupied_ratio: Fraction of samples from occupied hours (default: 0.8)
    """
    logger = logging.getLogger(__name__)
    
    # Calculate target number of points to generate
    target_points = int(dataset.max_size_type1 * target_fill_ratio)
    
    logger.info(f"Cold start: Generating {target_points} z points")
    logger.info(f"Sampling ratio: {occupied_ratio*100:.0f}% occupied hours, {(1-occupied_ratio)*100:.0f}% unoccupied hours")
    
    # Separate occupied and unoccupied data
    occupied_data = truth_table[
        (truth_table['hour'] >= occupied_hours[0]) & 
        (truth_table['hour'] <= occupied_hours[1])
    ].copy()
    
    unoccupied_data = truth_table[
        (truth_table['hour'] < occupied_hours[0]) | 
        (truth_table['hour'] > occupied_hours[1])
    ].copy()
    
    if len(occupied_data) == 0:
        logger.warning("No occupied hours found in truth table")
        return
    
    if len(unoccupied_data) == 0:
        logger.warning("No unoccupied hours found in truth table")
        return
    
    logger.info(f"Truth table: {len(occupied_data)} occupied entries, {len(unoccupied_data)} unoccupied entries")
    
    # Setup temperature sampling
    comfort_center = (comfort_bounds[0] + comfort_bounds[1]) / 2.0  # 24.5°C
    temp_min = comfort_bounds[0] - temperature_extension  # 23 - 2 = 21°C
    temp_max = comfort_bounds[1] + temperature_extension  # 26 + 2 = 28°C
    
    # Create truncated normal distribution for temperature sampling
    # Convert to standard normal bounds
    a = (temp_min - comfort_center) / temperature_std
    b = (temp_max - comfort_center) / temperature_std
    temp_dist = truncnorm(a, b, loc=comfort_center, scale=temperature_std)
    
    logger.info(f"Temperature sampling: center={comfort_center:.1f}°C, "
                f"range=[{temp_min:.1f}, {temp_max:.1f}]°C, std={temperature_std:.1f}°C")
    
    # Pre-generate lists for sampling
    # 1. Indoor temperature list (Gaussian distribution)
    indoor_temp_samples = temp_dist.rvs(target_points)
    
    # 2. Observation list (80% occupied, 20% unoccupied)
    num_occupied_samples = int(target_points * occupied_ratio)
    num_unoccupied_samples = target_points - num_occupied_samples
    
    # Sample from occupied hours
    occupied_indices = np.random.choice(len(occupied_data), num_occupied_samples, replace=True)
    occupied_samples = [occupied_data.iloc[i] for i in occupied_indices]
    
    # Sample from unoccupied hours
    unoccupied_indices = np.random.choice(len(unoccupied_data), num_unoccupied_samples, replace=True)
    unoccupied_samples = [unoccupied_data.iloc[i] for i in unoccupied_indices]
    
    # Combine and shuffle observation samples
    observation_samples = occupied_samples + unoccupied_samples
    np.random.shuffle(observation_samples)
    
    # 3. Action list (uniform 0-9)
    action_samples = np.random.randint(0, 10, target_points)
    
    logger.info(f"Generated {len(indoor_temp_samples)} temperature samples")
    logger.info(f"Generated {len(observation_samples)} observation samples ({num_occupied_samples} occupied, {num_unoccupied_samples} unoccupied)")
    logger.info(f"Generated {len(action_samples)} action samples")
    
    # Generate z points by combining the lists
    points_generated = 0
    
    for i in range(target_points):
        # Get components
        sampled_temp = indoor_temp_samples[i]
        time_point = observation_samples[i]
        discrete_action = action_samples[i]
        
        # Get observation from truth table
        obs = time_point['obs'].copy()
        
        # Normalize the sampled temperature
        temp_mean = obs_mean[6]  # Index 6 is air_temperature
        temp_std_norm = np.sqrt(obs_var[6])
        normalized_temp = (sampled_temp - temp_mean) / temp_std_norm
        
        # Replace air_temperature (index 6) with sampled value
        obs[6] = normalized_temp
        
        # Normalize action for GP input: (action / 4.5) - 1
        normalized_action = (discrete_action / 4.5) - 1.0
        
        # Create z point
        z_point = np.concatenate([obs, [normalized_action]])
        
        # Add metadata for tracking
        metadata = {
            'source': 'cold_start',
            'original_step': time_point['step'],
            'hour': time_point['hour'],
            'day': time_point['day'],
            'month': time_point['month'],
            'discrete_action': discrete_action,
            'sampled_temp_celsius': sampled_temp,
            'normalized_temp': normalized_temp,
            'normalized_action': normalized_action,
            'is_occupied': occupied_hours[0] <= time_point['hour'] <= occupied_hours[1]
        }
        
        # Add point to dataset as Type 1 (cold start)
        success = dataset.add_type1_point(z_point, metadata=metadata)
        if success:
            points_generated += 1
            
            if points_generated % 100 == 0:
                logger.debug(f"Generated {points_generated}/{target_points} z points")
    
    logger.info(f"Cold start completed: Generated {points_generated} z points")
    
    # Log statistics
    if points_generated > 0:
        temp_samples = [dataset.metadata_type1[i]['sampled_temp_celsius'] for i in range(dataset.num_points_type1)]
        action_samples = [dataset.metadata_type1[i]['discrete_action'] for i in range(dataset.num_points_type1)]
        occupied_samples = [dataset.metadata_type1[i]['is_occupied'] for i in range(dataset.num_points_type1)]
        
        logger.info(f"Temperature stats: mean={np.mean(temp_samples):.2f}°C, "
                   f"std={np.std(temp_samples):.2f}°C, "
                   f"range=[{np.min(temp_samples):.2f}, {np.max(temp_samples):.2f}]°C")
        
        action_dist = np.bincount(action_samples, minlength=10)
        logger.info(f"Action distribution: {action_dist}")
        
        occupied_count = np.sum(occupied_samples)
        logger.info(f"Occupied samples: {occupied_count}/{points_generated} ({occupied_count/points_generated*100:.1f}%)")


def accept_operational_z_points(dataset: ZDataset, z_points: List[np.ndarray], metadata_list: Optional[List[dict]] = None) -> int:
    """
    Accept Type 2 (operational) z points from external functions
    
    Args:
        dataset: ZDataset to add points to
        z_points: List of z points to add
        metadata_list: Optional list of metadata dictionaries for each point
        
    Returns:
        int: Number of points successfully added
    """
    logger = logging.getLogger(__name__)
    
    if metadata_list is None:
        metadata_list = [{}] * len(z_points)
    
    if len(z_points) != len(metadata_list):
        logger.warning(f"z_points length ({len(z_points)}) doesn't match metadata_list length ({len(metadata_list)})")
        metadata_list = [{}] * len(z_points)
    
    points_added = 0
    
    for i, z_point in enumerate(z_points):
        metadata = metadata_list[i].copy() if metadata_list[i] else {}
        metadata['source'] = 'operational'
        metadata['external_index'] = i
        
        success = dataset.add_type2_point(z_point, metadata)
        if success:
            points_added += 1
        else:
            logger.warning(f"Failed to add operational z point {i}")
    
    logger.info(f"Accepted {points_added}/{len(z_points)} operational z points")
    return points_added


def purge_type2_by_uncertainty(dataset: ZDataset, gp_model, uncertainty_threshold: float) -> int:
    """
    Convenience function to purge Type 2 data based on GP uncertainty evaluation
    
    This is a wrapper around the dataset's purge_type2_by_uncertainty method.
    
    Args:
        dataset: ZDataset to purge
        gp_model: Trained GP model with predict method that returns (mean, std)
        uncertainty_threshold: Points with uncertainty < threshold will be removed
        
    Returns:
        int: Number of points removed
    """
    return dataset.purge_type2_by_uncertainty(gp_model, uncertainty_threshold)


def analyze_cold_start_dataset(dataset: ZDataset) -> dict:
    """
    Analyze the cold start dataset to understand the distribution of generated points
    
    Args:
        dataset: ZDataset to analyze
        
    Returns:
        dict: Analysis results
    """
    if dataset.num_points_type1 == 0:
        return {"error": "No Type 1 (cold start) points found"}
    
    # Extract metadata from Type 1 points
    cold_start_points = [i for i in range(dataset.num_points_type1) 
                        if dataset.metadata_type1[i].get('source') == 'cold_start']
    
    if len(cold_start_points) == 0:
        return {"error": "No cold start points found"}
    
    # Temperature analysis
    temp_samples = [dataset.metadata_type1[i]['sampled_temp_celsius'] for i in cold_start_points]
    temp_stats = {
        'mean': np.mean(temp_samples),
        'std': np.std(temp_samples),
        'min': np.min(temp_samples),
        'max': np.max(temp_samples),
        'median': np.median(temp_samples)
    }
    
    # Action analysis
    action_samples = [dataset.metadata_type1[i]['discrete_action'] for i in cold_start_points]
    action_distribution = np.bincount(action_samples, minlength=10)
    
    # Time analysis
    hour_samples = [dataset.metadata_type1[i]['hour'] for i in cold_start_points]
    hour_distribution = np.bincount(hour_samples, minlength=24)
    
    return {
        'num_cold_start_points': len(cold_start_points),
        'temperature_stats': temp_stats,
        'action_distribution': action_distribution.tolist(),
        'hour_distribution': hour_distribution.tolist(),
        'comfort_zone_coverage': {
            'in_comfort_zone': np.sum((np.array(temp_samples) >= 23) & (np.array(temp_samples) <= 26)),
            'below_comfort_zone': np.sum(np.array(temp_samples) < 23),
            'above_comfort_zone': np.sum(np.array(temp_samples) > 26)
        }
    }


def visualize_cold_start_distribution(dataset: ZDataset, comfort_bounds: Tuple[float, float] = (23, 26)):
    """
    Create visualizations of the cold start dataset distribution
    
    Args:
        dataset: ZDataset to visualize
        comfort_bounds: Comfort zone bounds for reference
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logging.getLogger(__name__).warning("matplotlib not available for visualization")
        return
    
    analysis = analyze_cold_start_dataset(dataset)
    
    if "error" in analysis:
        print(f"Cannot visualize: {analysis['error']}")
        return
    
    # Extract data for plotting
    cold_start_points = [i for i in range(dataset.num_points_type1) 
                        if dataset.metadata_type1[i].get('source') == 'cold_start']
    
    temp_samples = [dataset.metadata_type1[i]['sampled_temp_celsius'] for i in cold_start_points]
    action_samples = [dataset.metadata_type1[i]['discrete_action'] for i in cold_start_points]
    hour_samples = [dataset.metadata_type1[i]['hour'] for i in cold_start_points]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Temperature distribution
    axes[0, 0].hist(temp_samples, bins=30, alpha=0.7, color='blue', edgecolor='black')
    axes[0, 0].axvspan(comfort_bounds[0], comfort_bounds[1], alpha=0.3, color='green', label='Comfort Zone')
    axes[0, 0].set_xlabel('Temperature (°C)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Temperature Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Action distribution
    axes[0, 1].bar(range(10), analysis['action_distribution'], color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Discrete Action')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Action Distribution')
    axes[0, 1].set_xticks(range(10))
    axes[0, 1].grid(True, alpha=0.3)
    
    # Hour distribution
    axes[1, 0].bar(range(24), analysis['hour_distribution'], color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Hour of Day')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Hour Distribution')
    axes[1, 0].axvspan(8, 17, alpha=0.3, color='yellow', label='Occupied Hours')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Temperature vs Action scatter
    axes[1, 1].scatter(temp_samples, action_samples, alpha=0.6, color='purple')
    axes[1, 1].axvspan(comfort_bounds[0], comfort_bounds[1], alpha=0.3, color='green', label='Comfort Zone')
    axes[1, 1].set_xlabel('Temperature (°C)')
    axes[1, 1].set_ylabel('Discrete Action')
    axes[1, 1].set_title('Temperature vs Action')
    axes[1, 1].set_yticks(range(10))
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("\nCold Start Dataset Analysis:")
    print(f"Total points: {analysis['num_cold_start_points']}")
    print(f"Temperature stats: {analysis['temperature_stats']}")
    print(f"Comfort zone coverage: {analysis['comfort_zone_coverage']}")


# Example usage and testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("ZDataset Testing Suite")
    print("=" * 60)
    
    # Test 1: Dataset Initialization
    print("\n1. Testing Dataset Initialization")
    print("-" * 40)
    
    dataset = ZDataset(max_size_type1=50, max_size_type2=100)
    print(f"Created dataset: {dataset}")
    
    # Test 2: Adding Type 1 Points (Cold Start)
    print("\n2. Testing Type 1 Point Addition")
    print("-" * 40)
    
    np.random.seed(42)  # For reproducible testing
    
    # Create and add Type 1 points
    type1_points = []
    for i in range(10):
        obs = np.random.uniform(-1, 1, 9)
        action = np.random.uniform(-1, 1)
        z_point = create_z_point(obs, action)
        metadata = {'source': 'test_type1', 'index': i}
        success = dataset.add_type1_point(z_point, metadata)
        if success:
            type1_points.append(z_point)
    
    print(f"Added {len(type1_points)} Type 1 points")
    print(f"Dataset after Type 1 addition: {dataset}")
    
    # Test 3: Adding Type 2 Points (Operational)
    print("\n3. Testing Type 2 Point Addition")
    print("-" * 40)
    
    # Create Type 2 points
    type2_points = []
    for i in range(15):
        obs = np.random.uniform(-1, 1, 9)
        action = np.random.uniform(-1, 1)
        z_point = create_z_point(obs, action)
        type2_points.append(z_point)
    
    # Add Type 2 points using the convenience function
    points_added = accept_operational_z_points(dataset, type2_points)
    print(f"Added {points_added} Type 2 points")
    print(f"Dataset after Type 2 addition: {dataset}")
    
    # Test 4: Dataset Information and Statistics
    print("\n4. Testing Dataset Information")
    print("-" * 40)
    
    info = dataset.get_dataset_info()
    print("Dataset Statistics:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test 5: Getting Z Targets for GP
    print("\n5. Testing Z Targets Retrieval")
    print("-" * 40)
    
    # Get all z targets
    z_targets, weights = dataset.get_z_targets()
    print(f"Total z targets: {len(z_targets)}")
    print(f"Z target shape: {z_targets[0].shape}")
    
    # Get Type 1 targets only
    type1_targets, _ = dataset.get_z_targets_by_type(1)
    print(f"Type 1 targets: {len(type1_targets)}")
    
    # Get Type 2 targets only
    type2_targets, _ = dataset.get_z_targets_by_type(2)
    print(f"Type 2 targets: {len(type2_targets)}")
    
    # Test 5.5: Testing Balanced Sampling
    print("\n5.5. Testing Balanced Sampling")
    print("-" * 40)
    
    # Test balanced sampling (default behavior)
    balanced_targets, _ = dataset.get_z_targets(balanced_sampling=True)
    print(f"Balanced z targets: {len(balanced_targets)} (should be 2 * min(type1, type2))")
    
    # Test original behavior
    all_targets, _ = dataset.get_z_targets(balanced_sampling=False)
    print(f"All z targets: {len(all_targets)} (should be type1 + type2)")
    
    # Test get_balanced_z_targets with specific target size
    target_200 = dataset.get_balanced_z_targets(target_size=200)
    print(f"Target 200 z targets: {len(target_200[0])}")
    
    # Test get_balanced_z_targets with default size
    default_balanced = dataset.get_balanced_z_targets()
    print(f"Default balanced z targets: {len(default_balanced[0])}")
    
    # Test 6: GP-Based Uncertainty Purging
    print("\n6. Testing GP-Based Uncertainty Purging")
    print("-" * 40)
    
    # Create a realistic mock GP model
    class MockGPModel:
        def __init__(self):
            self.counter = 0
            
        def predict(self, obs, action, return_std=True):
            # Simulate different uncertainty patterns
            self.counter += 1
            
            # Create some points with low uncertainty (well-learned)
            # and some with high uncertainty (needs more data)
            if self.counter % 3 == 0:
                uncertainty = np.random.uniform(0.1, 0.5)  # Low uncertainty
            else:
                uncertainty = np.random.uniform(0.8, 2.0)  # High uncertainty
                
            return 0.0, uncertainty
    
    # Test uncertainty-based purging
    mock_gp = MockGPModel()
    print(f"Before purging: {dataset.num_points_type2} Type 2 points")
    
    # Purge points with uncertainty < 0.6
    removed_count = purge_type2_by_uncertainty(dataset, mock_gp, uncertainty_threshold=0.6)
    print(f"Removed {removed_count} Type 2 points with low uncertainty")
    print(f"After purging: {dataset.num_points_type2} Type 2 points remaining")
    print(f"Dataset after uncertainty purging: {dataset}")
    
    # Test 7: Traditional Purging Methods
    print("\n7. Testing Traditional Purging Methods")
    print("-" * 40)
    
    print(f"Before purging: {dataset}")
    
    # Test oldest purging
    dataset.purge_dataset(purge_strategy="oldest", purge_ratio=0.3)
    print(f"After oldest purging (30%): {dataset}")
    
    # Test random purging
    dataset.purge_dataset(purge_strategy="random", purge_ratio=0.2)
    print(f"After random purging (20%): {dataset}")
    
    # Test 8: Individual Point Removal
    print("\n8. Testing Individual Point Removal")
    print("-" * 40)
    
    if dataset.num_points_type2 > 0:
        print(f"Removing first Type 2 point (index 0)")
        success = dataset.remove_type2_point(0)
        print(f"Removal successful: {success}")
        print(f"Dataset after removal: {dataset}")
    
    # Test 9: Final Dataset State
    print("\n9. Final Dataset State")
    print("-" * 40)
    
    final_info = dataset.get_dataset_info()
    print("Final Dataset Statistics:")
    for key, value in final_info.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("ZDataset Testing Completed Successfully!")
    print("=" * 60)
