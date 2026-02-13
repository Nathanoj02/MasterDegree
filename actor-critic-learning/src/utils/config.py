import os
import numpy as np

# Hyper-parameter
N = 100
dt = 0.01

# Multiprocessing parameters
num_workers = max(1, min(4, os.cpu_count() - 2))    # Limit to max 4 workers
maxtasksperchild = 50
chunk_size = 1  # Small chunk size to reduce memory usage

# State sampling ranges for each system
STATE_SAMPLING_CONFIG = {
    "single_integrator": {
        "x_min": -3.5,
        "x_max": 3.5,
        "num_samples": 1000  # For grid sampling
    },
    "double_integrator": {
        "p_min": -3.5,
        "p_max": 3.5,
        "v_min": -3.5,
        "v_max": 3.5,
        "num_samples_per_dim": 30  # Grid: 30x30 = 900 samples
    },
    "single_pendulum": {
        "theta_min": -np.pi,
        "theta_max": np.pi,
        "omega_min": -20.0,
        "omega_max": 20.0,
        "num_samples_per_dim": 30  # Grid: 30x30 = 900 samples
    },
    "double_pendulum": {
        "theta_min": -np.pi,
        "theta_max": np.pi,
        "omega_min": -6.0,
        "omega_max": 6.0,
        "num_samples_per_dim_theta": 6,
        "num_samples_per_dim_omega": 5  # Grid: 6x6x5x5 = 900 samples
    }
}