from typing import Callable
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule
from beta_distribution import BetaDistribution
from nerve_attention_net import NerveAttentionNetwork
import numpy as np
from torch import nn
import torch as th
from stable_baselines3.common.distributions import Distribution
from functools import partial
from stable_baselines3.common.preprocessing import get_action_dim

class BetaPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NerveAttentionNetwork(self.features_dim)

    
        
    def _build(self, lr_schedule: Schedule) -> None:
        
        self.action_dist = BetaDistribution(get_action_dim(self.action_space))
    
        print("Beta Policy in Work")
        self._build_mlp_extractor()

        latent_dim_pi = self.mlp_extractor.latent_dim_pi

        if isinstance(self.action_dist, BetaDistribution):
            #print("Networks Initialized!")
            self.alpha_net, self.beta_net = self.action_dist.proba_distribution_net(latent_dim=latent_dim_pi)
            #print("Alpha_Net:", self.alpha_net)
            #print("Beta_Net:", self.beta_net)

        else:
            raise NotImplementedError(f"Unsupported distribution '{self.action_dist}'.")

        self.value_net = nn.Linear(self.mlp_extractor.latent_dim_vf, 1)
        # Init weights: use orthogonal initialization
        # with small initial weight for the output
        if self.ortho_init:
            
            module_gains = {
                self.features_extractor: np.sqrt(2),
                self.mlp_extractor: np.sqrt(2),
                self.alpha_net: 0.01,
                self.beta_net: 0.01,
                self.value_net: 1,
            }
            if not self.share_features_extractor:
                # Note(antonin): this is to keep SB3 results
                # consistent, see GH#1148
                del module_gains[self.features_extractor]
                module_gains[self.pi_features_extractor] = np.sqrt(2)
                module_gains[self.vf_features_extractor] = np.sqrt(2)

            for module, gain in module_gains.items():
                module.apply(partial(self.init_weights, gain=gain))

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        if isinstance(self.action_dist, BetaDistribution):
            softplus = th.nn.Softplus()
            alpha = softplus(self.alpha_net(latent_pi)) + 1.0
            beta = softplus(self.beta_net(latent_pi)) + 1.0
            return self.action_dist.proba_distribution(alpha, beta)
        else:
            return super()._get_action_dist_from_latent(latent_pi)


class NormalPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *args,
        **kwargs,
    ):
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )


    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = NerveAttentionNetwork(self.features_dim)

