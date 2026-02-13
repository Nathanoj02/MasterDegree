import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import joblib
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import os

from actor_critic_neural_networks import NeuralNetwork
from actor_critic_neural_networks import conf_actor_rl as conf
from actor_critic_neural_networks import conf_critic as conf_critic
from utils.systems import get_system
from utils.torch_wrappers import create_torch_dynamics_and_cost, sample_states_for_system


class DifferentiableScaler(nn.Module):
    """
    Wraps sklearn StandardScalers into a PyTorch module 
    so we can backpropagate through the scaling
    """
    def __init__(self, scaler_path, device):
        super().__init__()
        scaler = joblib.load(scaler_path)
        
        # Load mean and scale (std) as tensors
        self.mean = torch.tensor(scaler.mean_, dtype=torch.float32).to(device)
        self.scale = torch.tensor(scaler.scale_, dtype=torch.float32).to(device)
        
    def transform(self, x):
        return (x - self.mean) / self.scale

    def inverse_transform(self, x):
        return x * self.scale + self.mean


def train_actor_policy(
    system_name: str, critic_model_path: str = None, model_path: str = None, 
    feature_scaler_path: str = None, target_scaler_path: str = None
):
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    critic_model_path = critic_model_path if critic_model_path is not None else conf_critic.model_path
    model_path = model_path if model_path is not None else conf.model_path
    feature_scaler_path = feature_scaler_path if feature_scaler_path is not None else conf_critic.feature_scaler_path
    target_scaler_path = target_scaler_path if target_scaler_path is not None else conf_critic.target_scaler_path

    # Hyperparameters
    hidden_size = conf.hidden_size
    dt = conf.dt

    # Get system and dimensions
    system = get_system(system_name, dt)
    state_dim = system.nx
    action_dim = system.nu
    learning_rate = conf.learning_rate
    batch_size = conf.batch_size
    num_iterations = conf.num_iterations

    # Create PyTorch-compatible dynamics and cost functions from the system
    dynamics_torch, cost_torch = create_torch_dynamics_and_cost(system)

    # Load Critic (Pre-trained)
    checkpoint = torch.load(critic_model_path, map_location=device)
    critic_config = checkpoint['config']
    
    critic = NeuralNetwork(
        critic_config['input_size'], 
        critic_config['hidden_size'], 
        critic_config['output_size']
    ).to(device)
    critic.load_state_dict(checkpoint['model'])
    
    # Freeze Critic weights
    for param in critic.parameters():
        param.requires_grad = False
    critic.eval()
    
    # Load Scalers
    feature_scaler = DifferentiableScaler(feature_scaler_path, device)
    target_scaler = DifferentiableScaler(target_scaler_path, device)
    
    # Initialize Actor
    actor = NeuralNetwork(state_dim, hidden_size, action_dim).to(device)
    optimizer = optim.Adam(actor.parameters(), lr=learning_rate)
    
    loss_history = []
    
    # Optimization Loop
    pbar = tqdm(range(num_iterations))
    for i in pbar:
        # Sample Batch (system-appropriate states)
        x_batch = sample_states_for_system(system, batch_size, device)

        # Normalize Input State
        x_batch_norm = feature_scaler.transform(x_batch)

        # Forward Pass - Actor predicts control
        u_pred = actor(x_batch_norm)

        # Physics Step - Apply dynamics
        x_next = dynamics_torch(x_batch, u_pred)

        # Critic Evaluation -> Get V(x_{k+1})
        x_next_scaled = feature_scaler.transform(x_next)

        # Get scaled value from critic
        v_next_scaled = critic(x_next_scaled)

        # Unscale back to physical units for loss calculation
        v_next_physical = target_scaler.inverse_transform(v_next_scaled)

        # Calculate Loss -> L(x,u) + V(x')
        running_cost = cost_torch(x_batch, u_pred)
        
        # Total cost to minimize -> J = l(x, u) + V(f(x, u)) (mean over batch)
        loss = torch.mean(running_cost + v_next_physical)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()
        
        # Clip gradients to prevent explosion due to polynomial costs
        torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        loss_history.append(loss.item())
        pbar.set_description(f"Loss: {loss.item():.4f}")

    # Create configuration dictionary
    config = {
        'input_size': state_dim,
        'hidden_size': hidden_size,
        'output_size': action_dim,
        'batch_size': batch_size,
        'num_iterations': num_iterations,
        'learning_rate': learning_rate,
        'dt': dt
    }

    # Save Actor with configuration
    torch.save({
        'model': actor.state_dict(),
        'config': config,
    }, model_path)
    print(f"Actor saved to {model_path}")
    
    # Plot training curve with smoothing
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, alpha=0.3, label='Raw Loss', color='blue')

    # Compute smoothed loss using moving average
    window = 100
    if len(loss_history) >= window:
        smoothed = np.convolve(loss_history, np.ones(window)/window, mode='valid')
        plt.plot(range(window-1, len(loss_history)), smoothed,
                 label='Smoothed Loss', linewidth=2, color='orange')

    plt.xlabel("Iteration")
    plt.ylabel("Total Cost (J)")
    plt.title("Actor Training Convergence")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Create system-specific img directory
    img_dir = f"../img/{system_name}"
    os.makedirs(img_dir, exist_ok=True)

    plot_path = os.path.join(img_dir, f"actor_rl_{system_name}_training_loss.png")
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train actor via reinforcement learning")

    parser.add_argument(
        "--system",
        type=str,
        required=True,
        choices=["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"],
        help="System name"
    )
    parser.add_argument(
        "--model-critic",
        type=str,
        default=conf_critic.model_path,
        help="Critic model path (default: from config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=conf.model_path,
        help="Output actor model path (default: from config)"
    )
    parser.add_argument(
        "--critic-feature-scaler",
        type=str,
        default=conf_critic.feature_scaler_path,
        help="Critic feature scaler path (default: from config)"
    )
    parser.add_argument(
        "--critic-target-scaler",
        type=str,
        default=conf_critic.target_scaler_path,
        help="Critic target scaler path (default: from config)"
    )

    args = parser.parse_args()

    train_actor_policy(
        args.system,
        args.model_critic,
        args.model,
        args.critic_feature_scaler,
        args.critic_target_scaler
    )