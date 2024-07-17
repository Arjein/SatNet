import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RescaleAction
from beta_distribution import BetaDistribution
from beta_policy import BetaPolicy, NormalPolicy
from graph_feature_extractor import GraphFeatureExtractor


from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
from itertools import product
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np

def evaluate_model(model, env, n_eval_episodes=10):
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, return_episode_rewards=False)
    return mean_reward, std_reward

def train_and_evaluate(sb_alg, hyperparams):
    
    env = gym.make("Ant-v4", render_mode=None)
    print("SB_ALG:", sb_alg)

    policy_kwargs = dict(
        features_extractor_class=GraphFeatureExtractor,
        last_layer_dim_pi=hyperparams.get('last_layer_dim_pi', 64),
        last_layer_dim_vf=hyperparams.get('last_layer_dim_vf', 64)
    )

    match sb_alg:
        case 'BETA':
            env = RescaleAction(env, min_action=0.0, max_action=1.0)
            model = PPO(
                BetaPolicy,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                seed=42,
                verbose=1,   
            )
        case 'GAUSSIAN':
            model = PPO(
                NormalPolicy,
                env,
                policy_kwargs=policy_kwargs,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                seed=42,
                verbose=1,   
            )
        case 'MLP':
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=hyperparams.get('learning_rate', 3e-4),
                n_steps=hyperparams.get('n_steps', 2048),
                batch_size=hyperparams.get('batch_size', 64),
                gamma=hyperparams.get('gamma', 0.99),
                gae_lambda=hyperparams.get('gae_lambda', 0.95),
                clip_range=hyperparams.get('clip_range', 0.2),
                seed=42,
                verbose=1,   
            )
        case _:
            print('Algorithm not found')
            return None, None

    TIMESTEPS = 20480
    model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps=False)
    # Evaluate the model
    mean_reward, std_reward = evaluate_model(model, env)
    return mean_reward, std_reward

def grid_search(hyperparams_grid, sb_alg):
    keys, values = zip(*hyperparams_grid.items())
    best_hyperparams = None
    best_mean_reward = -np.inf

    for v in product(*values):
        hyperparams = dict(zip(keys, v))
        print(f"Training with hyperparams: {hyperparams}")

        # Train and evaluate the model with the current hyperparameters
        mean_reward, std_reward = train_and_evaluate(sb_alg, hyperparams)
        
        print(f"Evaluation results - Mean reward: {mean_reward} ± {std_reward}")

        # Update the best hyperparameters if the current mean reward is better
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_hyperparams = hyperparams

    print(f"Best hyperparameters: {best_hyperparams}")
    print(f"Best mean reward: {best_mean_reward}")

# Define the hyperparameters grid

hyperparams_grid = {
    'learning_rate': [3e-4, 5e-4, 1e-3],
    'batch_size': [64, 128],
    'last_layer_dim_pi': [64,128, 256],
    'last_layer_dim_vf': [64,128, 256]
}

"""
hyperparams_grid = {
    'learning_rate': [1e-4, 3e-4, 1e-3],
    'batch_size': [32, 64, 128],
    'n_steps': [2048, 4096],
    'gamma': [0.98, 0.99, 0.995],
    'gae_lambda': [0.9, 0.95, 0.99],
    'clip_range': [0.1, 0.2, 0.3],
    'last_layer_dim_pi': [64, 128, 256],
    'last_layer_dim_vf': [64, 128, 256]
}
"""
# Run grid search
grid_search(hyperparams_grid, 'BETA')  # You can replace 'BETA' with 'GAUSSIAN' or

# Best hyperparameters: {'learning_rate': 0.0003, 'batch_size': 128, 'last_layer_dim_pi': 128, 'last_layer_dim_vf': 64}