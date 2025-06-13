"""Unified utility functions for the shooter game."""

import numpy as np
import math

PI = 3.1415
SPEED = 1.0  # Simulation speed multiplier (higher = faster rollouts)

def set_simulation_speed(speed_multiplier):
    """Set the global simulation speed multiplier.
    
    Args:
        speed_multiplier (float): Speed multiplier. 1.0 = normal speed, 
                                 2.0 = 2x faster, 0.5 = 2x slower
    """
    global SPEED
    SPEED = max(0.1, float(speed_multiplier))  # Minimum speed to avoid issues
    
def get_simulation_speed():
    """Get the current simulation speed multiplier."""
    return SPEED

def normalize_vector(vector):
    """Return a normalized copy of a 2D vector."""
    if isinstance(vector, list):
        vector = np.array(vector)
    
    norm = np.linalg.norm(vector)
    if norm < 1e-5:
        return np.zeros_like(vector)
    return vector / norm

def normalize_vectors(vectors):
    """Normalize a list of vectors."""
    return [normalize_vector(v) for v in vectors]

def distance(pos1, pos2):
    """Calculate Euclidean distance between two points."""
    return np.linalg.norm(np.array(pos1) - np.array(pos2))

def angle_between(pos1, pos2):
    """Calculate angle from pos1 to pos2."""
    diff = np.array(pos2) - np.array(pos1)
    x, y = diff
    
    if abs(x) < 1e-5 and abs(y) < 1e-5:
        return 0
    
    angle = np.arccos(np.clip(x / distance(pos1, pos2), -1, 1))
    if y < 0:
        angle = 2 * PI - angle
    
    return angle

def rotate_vector(vector, theta):
    """Rotate a vector by theta radians."""
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)
    return np.array([
        vector[0] * cos_theta - vector[1] * sin_theta,
        vector[0] * sin_theta + vector[1] * cos_theta
    ])

def flatten_features(feature_list):
    """Flatten a list of features into a single numpy array."""
    flattened = []
    for feature in feature_list:
        flattened.append(np.array(feature).flatten())
    return np.concatenate(flattened)

# Action space utilities
def continuous_to_discrete_move(move_vec):
    """Convert continuous movement to 4-directional discrete."""
    x, y = move_vec
    if abs(x) > abs(y):
        return (1, 0) if x > 0 else (-1, 0)
    else:
        return (0, 1) if y > 0 else (0, -1)

def get_8_directions():
    """Get 8 directional unit vectors."""
    directions = []
    for i in range(8):
        angle = i * PI / 4
        directions.append(np.array([math.cos(angle), math.sin(angle)]))
    return directions

def get_4_directions():
    """Get 4 directional unit vectors."""
    return [
        np.array([1, 0]),   # Right
        np.array([0, 1]),   # Down
        np.array([-1, 0]),  # Left
        np.array([0, -1])   # Up
    ]
