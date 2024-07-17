from typing import Tuple
from gatv2_conv_wrapper import GATv2ConvWrapper
from torch import nn
import torch as th

class NerveAttentionNetwork(nn.Module):
    
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        

        # Policy network
        self.policy_net = nn.Sequential(
            GATv2ConvWrapper(feature_dim, last_layer_dim_pi), nn.Tanh(),
            GATv2ConvWrapper(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh(),
        )
        # Value network
        self.value_net = nn.Sequential(
            GATv2ConvWrapper(feature_dim, last_layer_dim_vf), nn.Tanh(),
            GATv2ConvWrapper(last_layer_dim_vf, last_layer_dim_vf), nn.Tanh(),   
        )


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        
        actions = self.policy_net(features)
        actions = actions.mean(dim=1)
        return actions

    def forward_critic(self, features: th.Tensor) -> th.Tensor:

        value = self.value_net(features)
        value = value.mean(dim=1)
        return value