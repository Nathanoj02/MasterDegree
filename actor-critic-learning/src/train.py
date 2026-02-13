import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib
import argparse
import os

from actor_critic_neural_networks import NeuralNetwork
from actor_critic_neural_networks import conf_actor as conf_actor
from actor_critic_neural_networks import conf_critic as conf_critic

def train_loop(model: nn.Module, device: torch.device, train_loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
    """
    One epoch training loop.
    Parameters:
    - model: neural network model
    - device: computation device (CPU or GPU)
    - train_loader: DataLoader for training data
    - criterion: loss function
    - optimizer: optimization algorithm
    Returns:
    - average loss over the epoch
    """
    model.train()
    total_loss = 0
    
    for batch in train_loader:
        # Move data to device
        inputs, targets = batch
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_loop(model: nn.Module, device: torch.device, test_loader: DataLoader, criterion: nn.Module) -> float:
    """
    One epoch evaluation loop.
    Parameters:
    - model: neural network model
    - device: computation device (CPU or GPU)
    - test_loader: DataLoader for test data
    - criterion: loss function
    Returns:
    - average loss over the epoch
    """
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in test_loader:
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
    
    return total_loss / len(test_loader)


def train_nn(conf, dataset_path: str = None, model_path: str = None, feature_scaler_path: str = None, target_scaler_path: str = None):
    """ 
    Train neural network to predict value function or control policy.

    Parameters:
    - conf: configuration module containing hyperparameters and paths
    """
    path = "../models/"
    os.makedirs(path, exist_ok=True)
    
    # Load hyperparameters from configuration
    hidden_size = conf.hidden_size
    batch_size = conf.batch_size
    num_epochs = conf.num_epochs
    learning_rate = conf.learning_rate
    patience = conf.patience
    min_delta = conf.min_delta
    random_seed = conf.random_seed

    dataset_path = conf.dataset_path if dataset_path is None else dataset_path
    model_path = conf.model_path if model_path is None else model_path
    feature_scaler_path = conf.feature_scaler_path if feature_scaler_path is None else feature_scaler_path
    target_scaler_path = conf.target_scaler_path if target_scaler_path is None else target_scaler_path

    # Load dataset from NPZ file
    data = np.load(dataset_path)
    X = data[conf.input_key]
    y = data[conf.target_key]
        
    # Normalize features and targets
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    input_size = X.shape[1]
    X = feature_scaler.fit_transform(X)

    if (y.ndim == 1):
        output_size = 1
        y = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()
    else:
        output_size = y.shape[1]
        y = target_scaler.fit_transform(y)

    # Save scalers
    joblib.dump(feature_scaler, feature_scaler_path)
    joblib.dump(target_scaler, target_scaler_path)

    input_size = X.shape[1]

    # Create configuration dictionary
    config = {
        'input_size': input_size,
        'hidden_size': hidden_size,
        'output_size': output_size,
        'batch_size': batch_size,
        'num_epochs': num_epochs,
        'learning_rate': learning_rate,
        'patience': patience,
        'min_delta': min_delta,
        'random_seed': random_seed
    }

    # Split: 70% train, 15% validation, 15% test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=random_seed)

    # Convert to PyTorch tensors
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

    if output_size == 1:
        y_train_tensor = y_train_tensor.reshape(-1, 1)
        y_val_tensor = y_val_tensor.reshape(-1, 1)
        y_test_tensor = y_test_tensor.reshape(-1, 1)

    train_dataset = TensorDataset(
                        torch.tensor(X_train, dtype=torch.float32), 
                        y_train_tensor)
    val_dataset = TensorDataset(
                        torch.tensor(X_val, dtype=torch.float32),
                        y_val_tensor)
    test_dataset = TensorDataset(
                        torch.tensor(X_test, dtype=torch.float32),
                        y_test_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeuralNetwork(input_size, hidden_size, output_size).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping variables
    best_loss = float('inf')
    epochs_no_improvement = 0

    # Store loss history
    train_losses = []
    val_losses = []

    # Train the model
    pbar = tqdm(range(num_epochs))
    for epoch in pbar:
        train_loss = train_loop(model, device, train_loader, criterion, optimizer)
        val_loss = eval_loop(model, device, val_loader, criterion)

        # Store losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        pbar.set_description(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

        # Early stopping check
        if val_loss < best_loss - min_delta:
            best_loss = val_loss
            epochs_no_improvement = 0

            # Save the best model
            torch.save({
                'model': model.state_dict(),
                'config': config,
            }, model_path)
        else:
            epochs_no_improvement += 1

        if epochs_no_improvement >= patience:
            pbar.close()
            print("Early stopping triggered.")
            break

    # Load the best model for final evaluation
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    test_loss = eval_loop(model, device, test_loader, criterion)
    model_name = os.path.splitext(os.path.basename(model_path))[0]

    print("Training complete.")
    print(f'Best Validation Loss: {best_loss:.4f}')
    print(f'Test Loss: {test_loss:.4f}')

    # Extract system name from model path
    system_name = None
    for sys in ["single_integrator", "double_integrator", "single_pendulum", "double_pendulum"]:
        if sys in model_name:
            system_name = sys
            break

    if system_name:
        img_dir = f"../img/{system_name}"
        os.makedirs(img_dir, exist_ok=True)
        plot_path = os.path.join(img_dir, f"{model_name}_loss_plot.png")
    else:
        plot_path = f"../img/{model_name}_loss_plot.png"

    # Plot training and validation loss
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Time')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_path)
    print(f"Loss plot saved to {plot_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train neural network")

    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['critic', 'actor'],
        help='Training mode: critic (value function) or actor (policy)'
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Dataset path (default: from config)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model path (default: from config)"
    )
    parser.add_argument(
        "--feature_scaler",
        type=str,
        default=None,
        help="Feature scaler path (default: from config)"
    )
    parser.add_argument(
        "--target_scaler",
        type=str,
        default=None,
        help="Target scaler path (default: from config)"
    )

    args = parser.parse_args()

    if args.mode == 'critic':
        conf = conf_critic
    else:
        conf = conf_actor

    train_nn(
        conf,
        args.dataset,
        args.model,
        args.feature_scaler,
        args.target_scaler
    )