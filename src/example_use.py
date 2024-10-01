from stable_baselines3.common.distributions import Distribution, sum_independent_dims
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union, Callable
from torch.distributions import Beta
from torch_geometric.nn import GATv2Conv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
from torch_geometric.data import Data
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
import numpy as np
from stable_baselines3.common.distributions import Distribution
from functools import partial
from stable_baselines3.common.preprocessing import get_action_dim
from graph_feature_extractor import GraphFeatureExtractor
from beta_policy import SatNetBeta, SatNetGaussian

from gymnasium.wrappers import RescaleAction
from stable_baselines3 import PPO




env = gym.make("Ant-v4")
# SatNet-Beta
env = RescaleAction(env, min_action=0.0, max_action=1.0)
SatNet_Beta = PPO(
    SatNetBeta,
    env,
    policy_kwargs= dict(
        features_extractor_class=GraphFeatureExtractor,
        features_extractor_kwargs=dict(environment="Ant-v4")
        ),
    verbose=1,
    )   


# SatNet-Gaussian
SatNet_Gaussian = PPO(
    SatNetGaussian,
    env,
    policy_kwargs= dict(
        features_extractor_class=GraphFeatureExtractor,
        features_extractor_kwargs=dict(environment="Ant-v4")
        ),
    verbose=1,
    
    
    )   

# MLP
MLP = PPO("MlpPolicy", env, verbose=1)



SatNet_Beta.learn(1000)