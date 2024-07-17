import gymnasium as gym
import argparse
import os
from stable_baselines3 import PPO
from gymnasium.wrappers import RescaleAction
from beta_distribution import BetaDistribution
from beta_policy import BetaPolicy, NormalPolicy
from graph_feature_extractor import GraphFeatureExtractor

model_dir = "../models"
log_dir = "../logs"

os.makedirs(model_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

# Training
def train_model(sb_alg):
    log_dir_alg = os.path.join(log_dir, sb_alg)
    os.makedirs(log_dir_alg, exist_ok=True)
    env = gym.make("Ant-v4", render_mode=None)
    print("SB_ALG:", sb_alg)
    match sb_alg:
        
        case 'BETA':
            env = RescaleAction(env, min_action=0.0, max_action=1.0)
            model = PPO(
                BetaPolicy,
                env,
                policy_kwargs={'features_extractor_class': GraphFeatureExtractor},
                verbose=1,
                tensorboard_log=log_dir_alg,
                batch_size=128,
                seed=29,
                )   

        case 'GAUSSIAN':
            model = PPO(
                NormalPolicy,
                env,
                policy_kwargs={'features_extractor_class': GraphFeatureExtractor},
                verbose=1,
                tensorboard_log=log_dir_alg,
                seed=29,
                )   

        case 'MLP':
            model = PPO("MlpPolicy", env, verbose=1, seed=42 ,tensorboard_log=log_dir_alg)
            
        case _:
            print('Algorithm not found')
            return

    TIMESTEPS = 10240
    iters = 0

    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps = False)
        model.save(f"{model_dir}/{sb_alg}_{TIMESTEPS*iters}")

def continue_training(model_path):

    alg_name = model_path.split('/')[2].split('_')[0]

    print("Alg Name:",alg_name)
    
    env = gym.make("Ant-v4", render_mode=None)
    model = PPO.load(model_path)
    print(model.policy.action_dist)
    if isinstance(model.policy.action_dist, BetaDistribution):
        print("EVET BETA")
        env = RescaleAction(env, min_action=0.0, max_action=1.0)
    model.set_env(env)
    iterations = model.num_timesteps
    print("Iterations:", iterations)
    print("model_num_timstpes",model.num_timesteps)
    TIMESTEPS = 10240
    iters = 0
    while True:
        iters += 1
        model.learn(total_timesteps=TIMESTEPS, reset_num_timesteps = False)
        model.save(f"{model_dir}/{alg_name}_{iterations+TIMESTEPS*iters}")


def test(model_path):
    
    
    
    env = gym.make("Ant-v4", render_mode="human")
    
    model = PPO.load(model_path)
    print(model.policy.action_dist)
    if isinstance(model.policy.action_dist, BetaDistribution):
        print("EVET BETA")
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


if __name__ == '__main__':

    # Parse cmd line inputs
    parser = argparse.ArgumentParser(description='Train or test model.')
    
    parser.add_argument('-a', '--sb3_algo', type=str, default='PPO', help='StableBaseline3 RL algorithm i.e. SAC, TD3 (default: PPO)')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-ct', '--continuetraining', metavar='path_to_model',type=str)
    parser.add_argument('-s', '--test', metavar='path_to_model', type=str, help='Specify the path to the model to test.')
    
    args = parser.parse_args()
    
    if args.train:
        train_model(args.sb3_algo)
    if args.continuetraining:
        continue_training(args.continuetraining)

    if(args.test):
        if os.path.isfile(args.test):
            test(model_path= args.test)
        else:
            print(f'{args.test} not found.')

    
