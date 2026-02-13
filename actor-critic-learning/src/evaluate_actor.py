import numpy as np
import matplotlib.pyplot as plt
import argparse
import casadi as cs
import torch
import joblib
import multiprocessing
from tqdm import tqdm
from utils.config import *
from utils.systems import *
from utils.plot_utils import *
from actor_critic_neural_networks.predictor import Predictor
from actor_critic_neural_networks import conf_critic, conf_actor
from build_critic_dataset import solve_instance

# -------------------------
# Analysis 1: Cost Accuracy (Dataset vs Actor)
# -------------------------
def evaluate_cost_accuracy(system_name, actor, dataset_path):
    system = get_system(system_name, dt)

    print(f"\n--- COST ACCURACY ANALYSIS ({type(system).__name__}) ---")
    
    # Load all dataset samples
    data = np.load(dataset_path)
    x_samples = data[conf_critic.input_key]  
    v_optimal = data[conf_critic.target_key].flatten()

    actor_costs = []
    
    # Configure Output Folders
    base_dir = f"../img/{system_name}"
    state_dir = os.path.join(base_dir, "policy_state_traj")
    control_dir = os.path.join(base_dir, "policy_control_traj")

    for d in [state_dir, control_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            print(f"Folder created: {d}")

    # CasADi Functions
    x_sym = cs.SX.sym('x', system.nx)
    u_sym = cs.SX.sym('u', system.nu)
    f_dyn = cs.Function('f', [x_sym, u_sym], [system.dynamics_sym(x_sym, u_sym)])
    f_cost = cs.Function('l', [x_sym, u_sym], [system.running_cost_sym(x_sym, u_sym)])

    time_array = np.linspace(0, N * dt, N + 1)

    # Simulation Loop
    for i in tqdm(range(len(x_samples)), desc="Simulating and Saving Trajectories"):
        x_init = x_samples[i, :system.nx]
        current_x = x_init.copy()
        cumulative_j = 0.0
        
        traj_x = [current_x]
        traj_u = [] # Accumulator for controls
        
        for _ in range(N):
            # Actor prediction with scalar handling
            u_val = actor.predict(current_x.reshape(1, -1))
            u_pred = np.atleast_1d(u_val).flatten()
            
            # Save control
            traj_u.append(u_pred)
            
            # Calculate cost and dynamics
            cumulative_j += float(f_cost(current_x, u_pred))
            current_x = np.array(f_dyn(current_x, u_pred)).flatten()
            
            traj_x.append(current_x)
            
        actor_costs.append(cumulative_j)

        # Save Trajectories (State and Control)
        x_values_np = np.array(traj_x).T
        u_values_np = np.array(traj_u).T # Shape: (nu, N)
        
        # File paths
        state_filename = os.path.join(state_dir, f"state_sample_{i:04d}.png")
        control_filename = os.path.join(control_dir, f"control_sample_{i:04d}.png")
        
        # Plot States
        save_state_trajectory(time_array, x_values_np, system_name, state_filename, bypass=True)
        
        # Plot Controls (using your save_control_trajectory function)
        save_control_trajectory(time_array, u_values_np, system_name, control_filename, x_init=x_init, bypass=True)

    # Statistics
    actor_costs = np.array(actor_costs)
    errors = actor_costs - v_optimal

    print(f"Mean Optimal Cost (OCP): {np.mean(v_optimal):.4f}")
    print(f"Mean Actor Policy Cost : {np.mean(actor_costs):.4f}")
    print(f"Mean Absolute Error    : {np.mean(np.abs(errors)):.4f}")
    print(f"Mean Relative Error    : {np.mean(np.abs(errors)) / np.mean(v_optimal):.4f}")

# -------------------------
# Analysis 2: Efficiency (Warm Start)
# -------------------------
def evaluate_warm_start(system_name, actor, dataset_path, workers=None):
    system = get_system(system_name, dt)

    if workers is None:
        workers = num_workers

    # Load all dataset samples
    data = np.load(dataset_path)
    X_all, V_all, Iters_all = data[conf_critic.input_key], data[conf_critic.target_key], data['iters']
    
    # Total number of samples in the dataset
    n_total = len(X_all)
    
    # Prepare inputs for processes
    worker_inputs = []
    
    # Helper CasADi to simulate dynamics quickly
    x_s = cs.SX.sym('x', system.nx); u_s = cs.SX.sym('u', system.nu)
    f_dyn = cs.Function('f', [x_s, u_s], [system.dynamics_sym(x_s, u_s)])

    for idx in tqdm(range(n_total), desc="Generating Actor guesses for all samples..."):
        x0 = X_all[idx]
        u_guess = np.zeros((system.nu, N))
        x_guess = np.zeros((system.nx, N + 1))
        
        # Rollout of the Actor to create the initial trajectory guess (Warm Start)
        curr_x = x0
        x_guess[:, 0] = curr_x
        for t in range(N):
            u_p = actor.predict(curr_x.reshape(1, -1))
            u_guess[:, t] = u_p
            curr_x = np.array(f_dyn(curr_x, u_p)).flatten()
            x_guess[:, t+1] = curr_x
        
        # System name formatted for get_system inside solve_instance
        sys_name = type(system).__name__.lower().replace('integrator', '_integrator').replace('pendulum', '_pendulum')
        log_path = f"../img/{sys_name}"
        
        worker_inputs.append((idx, x0, sys_name, dt, N, u_guess, x_guess, log_path, "warm_start"))

    # Parallel Execution (Solving OCP for the entire dataset)
    results = {}
    print(f"Starting parallel solve on {workers} cores...")
    with multiprocessing.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        for idx, cost_w, it_w in tqdm(pool.imap_unordered(solve_instance, worker_inputs, chunksize=chunk_size), total=len(worker_inputs), desc="Warm Starting OCPs"):
            results[idx] = {'cost': cost_w, 'iters': it_w}

    # Global Statistical Analysis
    print("\n" + "="*80)
    print(f"GLOBAL ANALYSIS RESULTS ({n_total} samples)")
    print("-" * 80)
    
    gains, diffs = [], []
    for idx in range(n_total):
        # Retrieve Cold Start from dataset
        it_c = Iters_all[idx]
        v_c = V_all[idx]
        
        # Retrieve Warm Start calculated now
        it_w = results[idx]['iters']
        v_w = results[idx]['cost']
        
        # Calculate gain and differences only if both solvers succeeded
        if not np.isnan(it_c) and not np.isnan(it_w) and it_c > 0:
            gain = (it_c - it_w) / it_c * 100
            diff = abs(v_c - v_w)
            
            gains.append(gain)
            diffs.append(diff)

    # Print only the first 10 for brevity in the table, but the statistics are global
    print(f"{'Idx':>5} | {'Cold Iters':>10} | {'Warm Iters':>10} | {'Gain %':>10} | {'Cost Diff':>12}")
    for idx in list(sorted(results.keys()))[:10]:
        print(f"{idx:5d} | {int(Iters_all[idx]):10d} | {int(results[idx]['iters']):10d} | {(Iters_all[idx]-results[idx]['iters'])/Iters_all[idx]*100:9.1f}% | {abs(V_all[idx]-results[idx]['cost']):12.2E}")
    print("...")

    print("-" * 80)
    print(f"AVERAGE ITERATION GAIN: {np.mean(gains):.2f}%")
    print(f"AVERAGE COST DEVIATION: {np.mean(diffs):.2E}")
    print(f"SUCCESS RATE: {(len(gains)/n_total)*100:.1f}%")
    print("="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate actor performance")
    parser.add_argument("--system", type=str, required=True, choices=["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"], help="System name")
    parser.add_argument("--dataset", type=str, default=conf_critic.dataset_path, help="Path to critic dataset (default: from config)")
    parser.add_argument("--model", type=str, default=conf_actor.model_path, help="Path to actor model (default: from config)")
    parser.add_argument("--feature_scaler", type=str, default=None, help="Path to feature scaler - optional for RL actors (default: from config)")
    parser.add_argument("--target_scaler", type=str, default=None, help="Path to target scaler - optional for RL actors (default: from config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers to use (default: auto-detect)")

    args = parser.parse_args()

    # Load the Actor Predictor
    actor_model = Predictor(
        model_path=args.model,
        feature_scaler_path=args.feature_scaler,
        target_scaler_path=args.target_scaler
    )

    plot_policy(args.system, actor_model, args.dataset)

    # Accuracy Evaluation
    evaluate_cost_accuracy(args.system, actor_model, args.dataset)

    # Warm Start Performance Evaluation
    evaluate_warm_start(args.system, actor_model, args.dataset, args.workers)