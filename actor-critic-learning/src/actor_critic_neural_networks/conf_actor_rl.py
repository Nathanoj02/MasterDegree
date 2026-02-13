import os

hidden_size = 60

dt = 0.05

batch_size = 128
num_iterations = 10000
learning_rate = 1e-4
random_seed = 42

models_path = '../models'
model_path = os.path.join(models_path, 'actor_rl_best_model.pt')