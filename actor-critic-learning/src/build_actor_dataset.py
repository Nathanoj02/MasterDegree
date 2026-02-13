import casadi as cs
import numpy as np
import torch
import joblib
import argparse
import tqdm
import l4casadi as l4c
import multiprocessing
import os
from utils.config import *
from utils.systems import *
from actor_critic_neural_networks.neural_network import NeuralNetwork
from actor_critic_neural_networks import conf_critic, conf_actor
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='torch.jit')

# Global variable for workers
worker_context = {}

def init_worker(model_path, f_scaler_path, t_scaler_path, system_name, dt_val):
    """
    Initialize the environment once for each child process.
    """
    pid = os.getpid()
    # Create a unique system and solver for this worker
    system = get_system(system_name, dt_val)
    
    # Load the critic (compile the .so only once for this worker)
    V_func = load_critic_local(model_path, f_scaler_path, t_scaler_path, name_suffix=str(pid))
    
    # Pre-define the symbolic solver here for reuse
    u = cs.MX.sym("u", system.nu)
    x_mx = cs.MX.sym("x", system.nx) # Use a symbol for the state
    
    x_next = system.dynamics_sym(x_mx, u)
    running_cost = system.running_cost_sym(x_mx, u)
    V_next = V_func(x_next)
    Q = running_cost + V_next

    nlp = {"x": u, "f": Q, "p": x_mx} # Parameterize the problem with the state x
    opts = {
        "ipopt.tol": 1e-6,
        "ipopt.print_level": 0,
        "print_time": 0,
        "ipopt.sb": "yes",
        "ipopt.hessian_approximation": "limited-memory"
    }
    
    solver = cs.nlpsol(f"solver_{pid}", "ipopt", nlp, opts)
    
    # Save everything in the global context of the worker
    global worker_context
    worker_context['solver'] = solver
    worker_context['nu'] = system.nu

def worker_u_star_fast(args_tuple):
    """
    Perform only the numerical resolution (very fast).
    """
    idx, x_val = args_tuple
    solver = worker_context['solver']
    nu = worker_context['nu']
    
    try:
        # Pass x_val as parameter 'p' to the already compiled solver
        sol = solver(x0=np.zeros(nu), p=x_val)
        return idx, np.array(sol["x"]).flatten()
    except Exception:
        return idx, np.full(nu, np.nan)

# Helper for loading the critic model locally in each worker
def load_critic_local(model_path, feature_scaler_path, target_scaler_path, name_suffix=""):
    feature_scaler = joblib.load(feature_scaler_path)
    target_scaler = joblib.load(target_scaler_path)

    x_mean = np.array(feature_scaler.mean_).reshape(1, -1)
    x_std = np.array(feature_scaler.scale_).reshape(1, -1)
    V_mean = float(target_scaler.mean_[0])
    V_std = float(target_scaler.scale_[0])

    checkpoint = torch.load(model_path, map_location="cpu")
    model = NeuralNetwork(
        input_size=checkpoint["config"]["input_size"],
        hidden_size=checkpoint["config"]["hidden_size"],
        output_size=checkpoint["config"]["output_size"]
    )
    model.load_state_dict(checkpoint['model'])
    model.eval()

    # Each worker has its own unique .so folder/file
    l4c_model = l4c.L4CasADi(model, device='cpu', name=f"critic_worker_{name_suffix}")

    x_sym = cs.MX.sym("x", checkpoint["config"]["input_size"])
    x_scaled = (x_sym.T - x_mean) / x_std
    V_scaled = l4c_model(x_scaled)
    V_expr = V_scaled * V_std + V_mean

    return cs.Function("V_func", [x_sym], [V_expr])


def build_dataset(critic_dataset, model_path, f_scaler, t_scaler, output_path, system_name, workers=None):
    if workers is None:
        workers = num_workers
    
    data = np.load(critic_dataset)
    X_samples = data['X']
    n_samples = X_samples.shape[0]
    
    # Pass only idx and x_val to the tasks, the rest is already in the worker
    worker_tasks = [(i, X_samples[i, :]) for i in range(n_samples)]

    U_results = [None] * n_samples

    print(f"Initializing {workers} workers with pre-compiled solvers...")
    
    # The initializer is executed once for each process at startup
    with multiprocessing.Pool(processes=workers,
                  maxtasksperchild=maxtasksperchild,
                  initializer=init_worker, 
                  initargs=(model_path, f_scaler, t_scaler, system_name, dt)) as pool:
        
        for idx, u_star in tqdm.tqdm(pool.imap_unordered(worker_u_star_fast, worker_tasks, chunksize=chunk_size), 
                                     total=n_samples, 
                                     desc=f"Solving OCPs for {system_name}"):
            U_results[idx] = u_star

    U_actor = np.array(U_results)
    valid_mask = ~np.isnan(U_actor).any(axis=1)
    np.savez(
        output_path,
        **{conf_actor.input_key: X_samples[valid_mask]},
        **{conf_actor.target_key: U_actor[valid_mask]}
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, required=True,
                        choices=["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"])
    parser.add_argument("--actor_dataset", type=str, default=conf_actor.dataset_path, help="Output actor dataset path (default: from config)")
    parser.add_argument("--critic_dataset", type=str, default=conf_critic.dataset_path, help="Input critic dataset path (default: from config)")
    parser.add_argument("--critic_model", type=str, default=conf_critic.model_path, help="Critic model path (default: from config)")
    parser.add_argument("--critic_feature_scaler", type=str, default=conf_critic.feature_scaler_path, help="Critic feature scaler path (default: from config)")
    parser.add_argument("--critic_target_scaler", type=str, default=conf_critic.target_scaler_path, help="Critic target scaler path (default: from config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers to use (default: auto-detect)")

    args = parser.parse_args()
    
    build_dataset(
        args.critic_dataset,
        args.critic_model,
        args.critic_feature_scaler,
        args.critic_target_scaler,
        args.actor_dataset,
        args.system,
        args.workers
    )

if __name__ == "__main__":
    main()