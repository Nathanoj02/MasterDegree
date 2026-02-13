import torch
import joblib
import numpy as np
import argparse
import os

from .neural_network import NeuralNetwork
from . import conf_critic
from . import conf_actor
from . import conf_actor_rl

class Predictor:
    def __init__(self, model_path: str, feature_scaler_path: str = None, target_scaler_path: str = None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load Scalers (if provided)
        if feature_scaler_path and os.path.exists(feature_scaler_path):
            self.feature_scaler = joblib.load(feature_scaler_path)
        else:
            self.feature_scaler = None
            
        if target_scaler_path and os.path.exists(target_scaler_path):
            self.target_scaler = joblib.load(target_scaler_path)
        else:
            self.target_scaler = None

        # Load Model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Get model configuration
        config = checkpoint['config']
        input_size = config['input_size']
        hidden_size = config['hidden_size']
        output_size = config['output_size']

        # Load Model
        self.model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, output_size=output_size)
        self.model.load_state_dict(checkpoint['model'])
        self.model.to(self.device)
        self.model.eval()

    def predict(self, state: np.ndarray) -> np.ndarray:
        """
        Predict value function / policy action for given state.
        Parameters:
        - state: numpy array of shape (n,) representing the state
        Returns:
        - predicted value function / action: float or array
        """
        # Scale input features if scaler exists
        if self.feature_scaler is not None:
            state_scaled = self.feature_scaler.transform(state.reshape(1, -1))
        else:
            state_scaled = state.reshape(1, -1)

        # Convert to tensor
        state_tensor = torch.tensor(state_scaled, dtype=torch.float32).to(self.device)

        # Make inference
        with torch.no_grad():
            prediction_tensor = self.model(state_tensor)
            prediction = prediction_tensor.cpu().numpy()

        # Inverse scale the prediction if scaler exists
        if self.target_scaler is not None:
            prediction = self.target_scaler.inverse_transform(prediction)

        return prediction.item() if prediction.size == 1 else prediction.flatten()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict using saved model')
    parser.add_argument('--mode', type=str, default='actor',
                        choices=['critic', 'actor', 'actor_rl'],
                        help='Which model to load: critic, actor (supervised), or actor_rl (reinforcement learning)')
    args = parser.parse_args()

    if args.mode == 'critic':
        config = conf_critic
        print("Using Critic model")
        feature_scaler_path = config.feature_scaler_path
        target_scaler_path = config.target_scaler_path
    elif args.mode == 'actor':
        config = conf_actor
        print("Using Actor (supervised) model")
        feature_scaler_path = config.feature_scaler_path
        target_scaler_path = config.target_scaler_path
    elif args.mode == 'actor_rl':
        config = conf_actor_rl
        print("Using Actor (rl) model")
        feature_scaler_path = None
        target_scaler_path = None
    else:
        raise ValueError(f"Invalid mode: {args.mode}")

    predictor = Predictor(
        model_path=config.model_path,
        feature_scaler_path=feature_scaler_path,
        target_scaler_path=target_scaler_path
    )

    # Sample state: [theta1, theta2, dtheta1, dtheta2]
    sample_state = np.array([-0.788287681898749, -1.9783681552358543, -2.3829431626409403, 1.72702994208898])
    predicted_value = predictor.predict(sample_state)
    print(f"Predicted: {predicted_value}")