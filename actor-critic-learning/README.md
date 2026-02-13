# Actor-Critic Learning

This project implements actor-critic reinforcement learning for optimal control on classical mechanical systems. It trains neural network approximations of value functions (critic) and control policies (actor) for four systems: single integrator, double integrator, single pendulum, and double pendulum.

## Prerequisites

Install Docker to run the development environment. All dependencies are managed through the provided Dockerfile based on `docker.io/andreadelprete/orc24:v1`.

Build the Docker image from the project root:
```bash
docker build -t orc-project-custom .
```

## How to Run

Navigate to the source directory:
```bash
cd src
```

Run the complete pipeline for a system:
```bash
python3 run.py --system {system_name}
```

Supported systems: `single_integrator`, `double_integrator`, `single_pendulum`, `double_pendulum`

### Individual Scripts

Run each stage separately if needed (although we suggest to always run `run.py` to avoid name conflicts):

```bash
# Generate critic training data
python3 build_critic_dataset.py --system single_integrator [--workers 4]

# Train the critic (value function)
python3 train.py --mode critic

# Generate actor training data
python3 build_actor_dataset.py --system single_integrator

# Train actor via supervised learning
python3 train.py --mode actor

# Train actor via reinforcement learning
python3 train_actor_rl.py --system single_integrator

# Evaluate and compare actors
python3 evaluate_actor.py --system single_integrator
```

If you want you can run just the evaluation of the pipeline process by executing:
```bash
python3 run.py --system {system_name} --evaluate
```

## Configuration

Edit hyperparameters and training settings in `src/actor_critic_neural_networks/`:

- `conf_critic.py` - Critic network architecture and training parameters
- `conf_actor.py` - Supervised actor training configuration
- `conf_actor_rl.py` - Reinforcement learning actor configuration

Global settings like OCP horizon, timestep, and multiprocessing workers are in `src/utils/config.py`.
