import os

hidden_size = 60
batch_size = 64
num_epochs = 500
learning_rate = 1e-3
patience = 20       # Stop training if no improvement after this many epochs
min_delta = 1e-4    # Minimum change to qualify as an improvement
random_seed = 42

datasets_path = '../datasets'
dataset_path = os.path.join(datasets_path, 'critic_dataset_double_pendulum.npz')
input_key = 'X'
target_key = 'V'

models_path = '../models'
feature_scaler_path = os.path.join(models_path, 'critic_feature_scaler.pkl')
target_scaler_path = os.path.join(models_path, 'critic_target_scaler.pkl')
model_path = os.path.join(models_path, 'critic_best_model.pt')  