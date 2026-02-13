import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from actor_critic_neural_networks import conf_critic
import os

def plot_policy(system_name, actor_model, dataset_path):
    # Load dataset
    data = np.load(dataset_path)
    x_samples = data[conf_critic.input_key] # Shape: (N, nx)
    
    save_dir = "../img/" + system_name
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # --- Single integrator ---
    if system_name == "single_integrator":
        u_ocp = data[conf_critic.target_key].flatten()
        x_flat = x_samples.flatten()
        sort_idx = np.argsort(x_flat)
        
        x_sorted = x_flat[sort_idx]
        # Iterative inference to avoid scaler error
        u_actor = np.array([actor_model.predict(x) for x in x_sorted]).flatten()

        plt.figure(figsize=(10, 6))
        plt.scatter(x_flat, u_ocp, color='red', alpha=0.3, s=15, label='OCP Ground Truth')
        plt.plot(x_sorted, u_actor, color='blue', linewidth=2, label='Actor Policy')
        plt.title('Actor Policy: Single Integrator')
        plt.xlabel('State x')
        plt.ylabel('Action u')
        plt.legend()
        plt.savefig(os.path.join(save_dir, "actor_policy_single_integrator.png"))
        plt.close()

    # --- Double integrator / Single pendulum (2D Heatmap from samples) ---
    elif system_name in ["double_integrator", "single_pendulum"]:
        # Perform inference on all states present in the dataset
        # Use a list to iterate over each row (sample) of x_samples
        u_actor = np.array([actor_model.predict(x_samples[i, :]) for i in range(len(x_samples))]).flatten()

        plt.figure(figsize=(10, 8))
        
        # Use scatter with colormap to visualize u with respect to x1 and x2
        sc = plt.scatter(x_samples[:, 0], x_samples[:, 1], c=u_actor, cmap='viridis', s=20)
        plt.colorbar(sc, label='Control Action u (Predicted)')
        
        if system_name == "double_integrator":
            plt.xlabel('Position (p)')
            plt.ylabel('Velocity (v)')
        else:
            plt.xlabel('Theta (rad)')
            plt.ylabel('Omega (rad/s)')
            
        plt.title(f'Actor Policy Samples: {system_name.replace("_", " ").title()}')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f"actor_heatmap_{system_name}.png"))
        plt.close()

    # --- Double pendulum ---
    elif system_name == "double_pendulum":
        return 

    else:
        print(f"System {system_name} not recognized.")

def save_state_trajectory(time, x_values, system_name, filename, bypass=False):
    if bypass:
        return
    
    if x_values.ndim == 1: x_values = x_values.reshape(1, -1)
    
    fig = Figure(figsize=(12, 6)) # Slightly increased height for text
    FigureCanvas(fig)
    
    # Prepare initial state string
    x0 = x_values[:, 0]
    if system_name == "single_integrator":
        x0_str = f"Initial State x0: [{x0[0]:.3f}] (Position)"
    elif system_name == "double_integrator":
        x0_str = f"Initial State x0: [p: {x0[0]:.3f}, v: {x0[1]:.3f}]"
    elif system_name == "single_pendulum":
        x0_str = f"Initial State x0: [theta: {x0[0]:.3f}, omega: {x0[1]:.3f}]"
    elif system_name == "double_pendulum":
        x0_str = f"Initial State x0:\nAngles: [{x0[0]:.3f}, {x0[1]:.3f}] | Velocities: [{x0[2]:.3f}, {x0[3]:.3f}]"

    # --- Subplots ---
    if system_name == "single_integrator":
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(time, x_values[0, :], 'b-', label='Position (x)')
        ax.set_title("State Evolution")
    elif system_name in ["double_integrator", "single_pendulum"]:
        ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(time, x_values[0, :], 'b-'); ax1.set_title("Position / Angle")
        ax2.plot(time, x_values[1, :], 'r-'); ax2.set_title("Velocity")
    elif system_name == "double_pendulum":
        ax1 = fig.add_subplot(1, 2, 1); ax2 = fig.add_subplot(1, 2, 2)
        ax1.plot(time, x_values[0, :], 'b-', label='q1'); ax1.plot(time, x_values[1, :], 'g-', label='q2')
        ax1.legend(); ax1.set_title("Angles")
        ax2.plot(time, x_values[2, :], 'r-', label='dq1'); ax2.plot(time, x_values[3, :], 'm-', label='dq2')
        ax2.legend(); ax2.set_title("Velocities")

    for ax in fig.get_axes(): ax.grid(True); ax.set_xlabel("Time [s]")
    
    # --- Add initial state text ---
    fig.text(0.05, 0.02, x0_str, fontsize=10, bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.05, 1, 0.95]) # Leave space for the text at the bottom
    fig.savefig(filename)
    fig.clf()
    
def save_control_trajectory(time, u_values, system_name, filename, x_init, bypass=False):
    if bypass:
        return
    
    # Force u_values to 2D
    if u_values.ndim == 1: 
        u_values = u_values.reshape(1, -1)
    
    # Control time has N points, while time (from X) has N+1
    time_u = time[:u_values.shape[1]]

    fig = Figure(figsize=(10, 6)) # Increased height for text
    FigureCanvas(fig)
    ax = fig.add_subplot(1, 1, 1)

    # Format initial state string (x0)
    if system_name == "single_integrator":
        x0_str = f"Initial state x0: [{x_init[0]:.3f}]"
    elif system_name == "double_integrator":
        x0_str = f"Initial state x0: [p: {x_init[0]:.3f}, v: {x_init[1]:.3f}]"
    elif system_name == "single_pendulum":
        x0_str = f"Initial state x0: [theta: {x_init[0]:.3f}, omega: {x_init[1]:.3f}]"
    elif system_name == "double_pendulum":
        x0_str = f"Initial state x0: Angles [{x_init[0]:.3f}, {x_init[1]:.3f}] | Velocities [{x_init[2]:.3f}, {x_init[3]:.3f}]"
    
    # --- Plot logic ---
    if system_name == "double_pendulum":
        ax.step(time_u, u_values[0, :], 'g-', where='post', label='Joint Torque 1')
        ax.step(time_u, u_values[1, :], 'm-', where='post', label='Joint Torque 2')
        ax.set_ylabel("Torque [Nm]")
    else:
        label = "Force [N]" if "integrator" in system_name else "Torque [Nm]"
        ax.step(time_u, u_values[0, :], 'g-', where='post', label='Control (u)')
        ax.set_ylabel(label)

    ax.set_title(f"Optimal Control Input: {system_name}")
    ax.set_xlabel("Time [s]")
    ax.legend()
    ax.grid(True)

    fig.text(0.05, 0.02, x0_str, fontsize=10, fontweight='bold', bbox=dict(facecolor='white', alpha=0.5))
    
    fig.tight_layout(rect=[0, 0.08, 1, 1]) 
    fig.savefig(filename)
    fig.clf()

def plot_value_function_1D(x_init_list, v_data, save_path):
    x_init_list = np.asarray(x_init_list)
    v_data = np.asarray(v_data)

    sort_idx = np.argsort(x_init_list.flatten())
    x_sorted = x_init_list[sort_idx]
    v_sorted = v_data[sort_idx]

    plt.figure(figsize=(10, 6))
    plt.plot(x_sorted, v_sorted, 'b-', linewidth=2)
    plt.xlabel("Initial state x_init")
    plt.ylabel("Value function V(x_init)")
    plt.title(f"Value Function - {save_path.split('/')[-2]}")
    plt.grid(True)
    
    plt.savefig(save_path)
    print(f"Plot saved in: {save_path}")
    plt.close()

def plot_value_function_2D(x_init_array, value_data, save_path):
    p_list = np.unique(x_init_array[:,0])
    v_list = np.unique(x_init_array[:,1])

    P, V = np.meshgrid(p_list, v_list)
    V_grid = np.zeros_like(P)
    
    for i, v_val in enumerate(v_list):
        for j, p_val in enumerate(p_list):
            idx = np.where((np.isclose(x_init_array[:,0], p_val)) & 
                           (np.isclose(x_init_array[:,1], v_val)))[0]
            if len(idx) > 0:
                V_grid[i, j] = value_data[idx[0]]
            else:
                V_grid[i, j] = np.nan

    plt.figure(figsize=(8, 6))
    plt.pcolormesh(P, V, V_grid, shading='auto', cmap='viridis')
    plt.colorbar(label="Value function V")
    plt.xlabel("State Dim 1 (Position/Theta)")
    plt.ylabel("State Dim 2 (Velocity/Omega)")
    plt.title(f"Value Function Heatmap - {save_path.split('/')[-2]}")
    
    plt.savefig(save_path)
    print(f"Plot saved in: {save_path}")
    plt.close()