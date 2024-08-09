
from stable_baselines3.common.distributions import Distribution, sum_independent_dims
import torch as th
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, TypeVar, Union
from torch.distributions import Beta

class BetaDistribution(Distribution):
    """
    A custom distribution class that implements a Beta distribution for 
    SatNet implementation as sb_3 does not have this distribution built-in.
    
    This class handles the creation of a Beta distribution, sampling from it, calculating log probabilities,
    and other necessary operations for use in RL.
    """

    def __init__(self, action_dim: int):
        super().__init__()
        self.action_dim = action_dim
        self.alpha = None
        self.beta = None

    def proba_distribution_net(self, latent_dim: int) -> Tuple[nn.Module, nn.Module]:
        """
        Create the linear networks that produce the alpha and beta parameters of the Beta distribution.

        Parameters:
        latent_dim (int): The dimensionality of the latent features.

        Returns:
        Tuple[nn.Module, nn.Module]: The neural networks that produce alpha and beta parameters.
        """

        alpha_net = nn.Linear(latent_dim, self.action_dim)
        beta_net = nn.Linear(latent_dim, self.action_dim)
        return alpha_net, beta_net
    
    def proba_distribution(self, alpha: th.Tensor, beta: th.Tensor) -> 'BetaDistribution':
        """
        Create the Beta distribution given alpha and beta parameters.

        Parameters:
        alpha (Tensor): The alpha parameter tensor.
        beta (Tensor): The beta parameter tensor.

        Returns:
        BetaDistribution: The Beta distribution object.
        """
                
        self.distribution = Beta(alpha, beta)
        return self
    
    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        
        log_prob = self.distribution.log_prob(actions)
        return sum_independent_dims(log_prob)
    
    def entropy(self) -> th.Tensor:
        entropy = self.distribution.entropy()
        return sum_independent_dims(entropy)
    
    def sample(self) -> th.Tensor:
        actions = self.distribution.rsample()
        
        #print("Taken Actions:", actions)
        return actions

    def mode(self) -> th.Tensor:
        actions = self.distribution.mean
        
        return actions
    
    def actions_from_params(self, alpha: th.Tensor, beta: th.Tensor, deterministic: bool = False) -> th.Tensor:
        self.proba_distribution(alpha, beta)
        return self.get_actions(deterministic=deterministic)

    def log_prob_from_params(self, alpha: th.Tensor, beta: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        actions = self.actions_from_params(alpha, beta)
        log_prob = self.log_prob(actions)
        return actions, log_prob