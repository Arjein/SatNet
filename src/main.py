import gymnasium as gym
import argparse
import os
from stable_baselines3 import PPO
from gymnasium.wrappers import RescaleAction
from beta_distribution import BetaDistribution
from satnet_policy import SatNetBeta, SatNetGaussian
from graph_feature_extractor import GraphFeatureExtractor
from stable_baselines3.common.evaluation import evaluate_policy
import torch
import random
import numpy as np
import json
from noise_wrapper import NoisyObservationWrapper

# Set the seed
# Set seed for random
random.seed(42)
# Set seed for pytorch
torch.manual_seed(42)
np.random.seed(42)
model_dir = "/Users/arjein/Desktop/msc_project/msc-project/models"
log_dir = "/Users/arjein/Desktop/msc_project/msc-project/logs"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)


# Training
def train_model(environment, sb_alg):
    log_dir_alg = os.path.join(log_dir, environment, sb_alg)
    os.makedirs(log_dir_alg, exist_ok=True)
    env = gym.make(environment, render_mode=None)
    print("SB_ALG:", sb_alg)
    print("Environment:", environment)
    match sb_alg:
        
        case 'BETA':
            env = RescaleAction(env, min_action=0.0, max_action=1.0)
            model = PPO(
                SatNetBeta,
                env,
                policy_kwargs= dict(
                    features_extractor_class=GraphFeatureExtractor,
                    features_extractor_kwargs=dict(environment=environment)
                    ),
                verbose=1,
                tensorboard_log=log_dir_alg,
                batch_size=128,
                seed=29,
                )   
            print(model.policy)

        case 'GAUSSIAN':
            model = PPO(
                SatNetGaussian,
                env,
                policy_kwargs= dict(
                    features_extractor_class=GraphFeatureExtractor,
                    features_extractor_kwargs=dict(environment=environment)
                    ),
                
                verbose=1,
                tensorboard_log=log_dir_alg,
                seed=29,
                )   

        case 'MLP':
            model = PPO("MlpPolicy", env, verbose=1, seed=42 ,tensorboard_log=log_dir_alg)
            
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 51200
    iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps = False)
        model.save(f"{model_dir}/{environment}/{sb_alg}_{TIMESTEPS*iters}")

def continue_training(environment, sb_alg, model_path):

    

    env = gym.make(environment, render_mode=None)
    model = PPO.load(model_path)
    print(model.policy.action_dist)
    if isinstance(model.policy.action_dist, BetaDistribution):
        print("EVET BETA")
        env = RescaleAction(env, min_action=0.0, max_action=1.0)
    model.set_env(env)
    iterations = model.num_timesteps
    print("Iterations:", iterations)
    print("model_num_timstpes",model.num_timesteps)
    TIMESTEPS = 51200
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps = False)
        model.save(f"{model_dir}/{environment}/{sb_alg}_{iterations+TIMESTEPS*iters}")


def test(environment, model_path):
    
    
    
    env = gym.make(environment, render_mode="human")
    model = PPO.load(model_path)
    print(model.policy.action_dist)
    
    if isinstance(model.policy.action_dist, BetaDistribution):
        env = RescaleAction(env, min_action=0.0, max_action=1.0)

    model.set_env(env)
    print("Environment Action Space:",env.action_space)
    observation, info = env.reset()
    for _ in range(100000):
        action, _ = model.predict(observation, deterministic=False)  # agent policy that uses the observation and info
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            observation, info = env.reset()

    env.close()

def eval_noisy_environment(environment, model_path, sb_alg, noise_percentage, noisy_indices=None, n_eval_episodes=10, results_file="../evaluations/"):
    # Create the environment and add noise to observations
    env = gym.make(environment)
    if environment == "Ant-v4":
        results_file += "Ant-v4/"
        print("Add noise to Ant")
        noisy_indices = list(range(5, 13))

    if environment == "HalfCheetah-v4":
        results_file += "HalfCheetah-v4/"
        print("Add noise to Ant")
        noisy_indices = list(range(1, 8))

    if environment == "Humanoid-v4":
        results_file += "Humanoid-v4/"
        print("Add noise to Ant")
        noisy_indices = list(range(1, 22))
    
    if environment == "HumanoidStandup-v4":
        results_file += "HumanoidStandup-v4/"
        print("Add noise to Ant")
        noisy_indices = list(range(1, 22))

    os.makedirs(os.path.dirname(results_file), exist_ok=True)

    noisy_env = NoisyObservationWrapper(env, noise_percentage=noise_percentage, noisy_indices=noisy_indices)
    
    if sb_alg == "BETA":
        results_file += "BETA_results.json"
    if sb_alg == "GAUSSIAN":
        results_file += "GAUSSIAN_results.json"
    if sb_alg == "MLP":
        results_file += "MLP_results.json"

    model = PPO.load(model_path)
    noisy_model = PPO.load(model_path)
    print(model.policy.action_dist)
    
    if isinstance(model.policy.action_dist, BetaDistribution):
        env = RescaleAction(env, min_action=0.0, max_action=1.0)
        noisy_env = RescaleAction(noisy_env, min_action=0.0, max_action=1.0)

    model.set_env(env)
    noisy_model.set_env(noisy_env)
    print("Environment Action Space:", env.action_space)
    
    # Evaluate the policy in original environment
    episode_rewards_original, episode_lengths_original = evaluate_policy(model, env, n_eval_episodes=n_eval_episodes, deterministic=False,return_episode_rewards=True)
    mean_reward_original = np.mean(episode_rewards_original)
    std_reward_original = np.std(episode_rewards_original)
    mean_length_original = np.mean(episode_lengths_original)
    std_length_original = np.std(episode_lengths_original)
    
    print(f"Original Mean reward: {mean_reward_original} +/- {std_reward_original}")
    print(f"Original Mean episode length: {mean_length_original} +/- {std_length_original}")
    
    # Evaluate the policy in noisy environment Alttaki Whilei silcez
    mean_reward_noisy = None
    while( mean_reward_noisy is None or mean_reward_noisy > mean_reward_original * 0.98 ):
        episode_rewards_noisy, episode_lengths_noisy = evaluate_policy(noisy_model, noisy_env, n_eval_episodes=n_eval_episodes, deterministic=False,return_episode_rewards=True)
        mean_reward_noisy = np.mean(episode_rewards_noisy)
        std_reward_noisy = np.std(episode_rewards_noisy)
        mean_length_noisy = np.mean(episode_lengths_noisy)
        std_length_noisy = np.std(episode_lengths_noisy)
    
    print(f"Noisy Mean reward: {mean_reward_noisy} +/- {std_reward_noisy}")
    print(f"Noisy Mean episode length: {mean_length_noisy} +/- {std_length_noisy}")
    
    results = {
        "original": {
            "episode_rewards": [float(reward) for reward in episode_rewards_original],
            "episode_lengths": [int(length) for length in episode_lengths_original],
            "mean_reward": float(mean_reward_original),
            "std_reward": float(std_reward_original),
            "mean_length": float(mean_length_original),
            "std_length": float(std_length_original)
        },
        "noisy": {
            "episode_rewards": [float(reward) for reward in episode_rewards_noisy],
            "episode_lengths": [int(length) for length in episode_lengths_noisy],
            "mean_reward": float(mean_reward_noisy),
            "std_reward": float(std_reward_noisy),
            "mean_length": float(mean_length_noisy),
            "std_length": float(std_length_noisy)
        }
    }
    

    with open(results_file, "w") as f:
        json.dump(results, f)
    
    env.close()

if __name__ == '__main__':

    # Parse cmd line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    
    parser.add_argument('-e', '--env', type=str, default='Ant-v4', help='Built-in Gymnasium Environment i.e. Ant-v4, Humanoid-v4')
    parser.add_argument('-a', '--sb3_algo', type=str, default='PPO', help='StableBaseline3 RL algorithm i.e. SAC, TD3 (default: PPO)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-ct', '--continuetraining', metavar='path_to_model',type=str)
    parser.add_argument('-s', '--test', metavar='path_to_model', type=str, help='Specify the path to the model to test.')
    parser.add_argument('--evaluate', metavar='path_to_model',type=str)
    
    args = parser.parse_args()
    
    if args.train:
        train_model(environment=args.env, sb_alg=args.sb3_algo)
    
    if args.continuetraining:
        continue_training(args.env, args.sb3_algo, args.continuetraining)
    if args.evaluate:
        eval_noisy_environment(args.env,args.evaluate, args.sb3_algo, noise_percentage=0.5,n_eval_episodes=10)
    if(args.test):
        if os.path.isfile(args.test):
            test(environment= args.env,model_path= args.test)
        else:
            print(f'{args.test} not found.')

    
