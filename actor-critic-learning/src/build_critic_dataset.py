import casadi as cs
import numpy as np
import argparse
import os
import shutil
from tqdm import tqdm
import multiprocessing
from utils.config import *
from utils.systems import *
from utils.plot_utils import *
from actor_critic_neural_networks import conf_critic

def solve_instance(args):
    # Unpack
    if len(args) == 9:
        idx, x_init_numeric, system_name, dt, N, u_guess, x_guess, log_path, folder_prefix = args
    else:
        idx, x_init_numeric, system_name, dt, N, u_guess, x_guess, log_path = args
        folder_prefix = "" # Default

    system = get_system(system_name, dt)
    
    opti = cs.Opti()
    X = opti.variable(system.nx, N + 1)
    U = opti.variable(system.nu, N)
    
    opti.subject_to(X[:, 0] == x_init_numeric)
    
    cost = 0
    for k in range(N):
        cost += system.running_cost_sym(X[:, k], U[:, k])
        opti.subject_to(X[:, k+1] == system.dynamics_sym(X[:, k], U[:, k]))
    
    opti.minimize(cost)
    
    if u_guess is not None: opti.set_initial(U, u_guess)
    if x_guess is not None: opti.set_initial(X, x_guess)

    opti.solver("ipopt", {"ipopt.print_level": 0, "print_time": 0, "ipopt.sb": "yes"})
    
    try:
        sol = opti.solve()
        v_opt = float(sol.value(cost))
        iter_count = int(opti.stats()["iter_count"])

        x_val = sol.value(X)
        u_val = sol.value(U)

        # If folder_prefix is "actor", the names will be "actor_state_traj", etc.
        s_folder = f"{folder_prefix}_state_traj" if folder_prefix else "state_traj"
        c_folder = f"{folder_prefix}_control_traj" if folder_prefix else "control_traj"

        state_dir = os.path.join(log_path, s_folder)
        control_dir = os.path.join(log_path, c_folder)

        os.makedirs(state_dir, exist_ok=True)
        os.makedirs(control_dir, exist_ok=True)

        time_axis = np.linspace(0, N*dt, N+1)
        state_filename = os.path.join(state_dir, f"sample_{idx}.png")
        control_filename = os.path.join(control_dir, f"sample_{idx}.png")

        save_state_trajectory(time_axis, x_val, system_name, state_filename, bypass=True)
        save_control_trajectory(time_axis, u_val, system_name, control_filename, x_init_numeric, bypass=True)

        return idx, v_opt, iter_count
    except Exception as e:
        print(f"Errore nel sample {idx}: {e}")
        iter_count = int(opti.stats()["iter_count"]) if "iter_count" in opti.stats() else 0

        return idx, np.nan, iter_count

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", type=str, choices=["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"], required=True)
    parser.add_argument("--dataset", type=str, default=conf_critic.dataset_path, help="Output dataset path (default: from config)")
    parser.add_argument("--workers", type=int, default=None, help="Number of parallel workers to use (default: auto-detect)")
    args = parser.parse_args()

    # Get config for this system
    cfg = STATE_SAMPLING_CONFIG[args.system]

    if args.system == "single_integrator":
        x_range = np.linspace(cfg["x_min"], cfg["x_max"], cfg["num_samples"])
        x_inits = x_range.reshape(-1, 1)

    elif args.system == "double_integrator":
        n_per_dim = cfg["num_samples_per_dim"]
        p_range = np.linspace(cfg["p_min"], cfg["p_max"], n_per_dim)
        v_range = np.linspace(cfg["v_min"], cfg["v_max"], n_per_dim)
        P, V = np.meshgrid(p_range, v_range)
        x_inits = np.column_stack([P.ravel(), V.ravel()])

    elif args.system == "single_pendulum":
        n_per_dim = cfg["num_samples_per_dim"]
        theta_range = np.linspace(cfg["theta_min"], cfg["theta_max"], n_per_dim)
        omega_range = np.linspace(cfg["omega_min"], cfg["omega_max"], n_per_dim)
        T, W = np.meshgrid(theta_range, omega_range)
        x_inits = np.column_stack([T.ravel(), W.ravel()])

    else:  # double_pendulum
        n_per_dim_theta = cfg["num_samples_per_dim_theta"]
        n_per_dim_omega = cfg["num_samples_per_dim_omega"]
        t1 = np.linspace(cfg["theta_min"], cfg["theta_max"], n_per_dim_theta)
        t2 = np.linspace(cfg["theta_min"], cfg["theta_max"], n_per_dim_theta)
        w1 = np.linspace(cfg["omega_min"], cfg["omega_max"], n_per_dim_omega)
        w2 = np.linspace(cfg["omega_min"], cfg["omega_max"], n_per_dim_omega)

        T1, T2, W1, W2 = np.meshgrid(t1, t2, w1, w2)
        x_inits = np.column_stack([
            T1.ravel(), T2.ravel(), W1.ravel(), W2.ravel()
        ])
    
    log_path = f"../img/{args.system}"
    if os.path.exists(log_path): shutil.rmtree(log_path)
    os.makedirs(log_path, exist_ok=True)

    n_samples = len(x_inits)
    worker_inputs = [
        (i, x_inits[i], args.system, dt, N, None, None, log_path) 
        for i in range(n_samples)
    ]
    
    v_results = np.full(n_samples, np.nan)
    iter_results = np.full(n_samples, np.nan) # Array to store iteration counts

    if args.workers is None:
        workers = num_workers
    else:
        workers = args.workers

    print(f"Starting parallel solve with {workers} workers...")

    with multiprocessing.Pool(processes=workers, maxtasksperchild=maxtasksperchild) as pool:
        # Unpack 3 values (idx, val, iters)
        for idx, val, iters in tqdm(pool.imap_unordered(solve_instance, worker_inputs, chunksize=chunk_size),
                                     total=n_samples, desc="Solving OCPs"):
            v_results[idx] = val
            iter_results[idx] = iters

    valid_mask = ~np.isnan(v_results)

    np.savez(
        args.dataset,
        **{conf_critic.input_key: x_inits[valid_mask]},
        **{conf_critic.target_key: v_results[valid_mask]},
        iters=iter_results[valid_mask]
    )
    
    print(f"Finished. Saved {np.sum(valid_mask)} Samples with iteration counts.")
    
    if args.system == "single_integrator":
        plot_path = os.path.join(log_path, "value_function.png")
        plot_value_function_1D(x_inits[valid_mask], v_results[valid_mask], plot_path)
        
    elif args.system in ["double_integrator", "single_pendulum"]:
        plot_path = os.path.join(log_path, "value_function_heatmap.png")
        plot_value_function_2D(x_inits[valid_mask], v_results[valid_mask], plot_path)
        
    elif args.system == "double_pendulum":
        print("The value function for the double pendulum has 4 dimensions")

if __name__ == "__main__":
    main()