from typing import Tuple
from gatv2_conv_wrapper import GATv2ConvWrapper
from torch import nn
import torch as th

class NerveAttentionNetwork(nn.Module):
    """
    SatNet is a neural network model that utilizes graph attention layers (GATv2Conv)
    for policy and value function approximation in reinforcement learning.
    
    This model is particularly designed for environments where observations are structured
    as graphs, and the relationships between different parts of the state space can be
    effectively captured by graph attention networks.
    """
    
    def __init__(
        self,
        feature_dim: int,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        """
        Initialize the SatNet model.

        Parameters:
        feature_dim (int): The dimensionality of the input features
        last_layer_dim_pi (int): The dimensionality of the last hidden layer in the policy network
        last_layer_dim_vf (int): The dimensionality of the last hidden layer in the value network 
        """
        super().__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        
        # Policy network
        self.policy_net = nn.Sequential(
            GATv2ConvWrapper(feature_dim, last_layer_dim_pi), nn.ReLU(),
            GATv2ConvWrapper(last_layer_dim_pi, last_layer_dim_pi), nn.ReLU(),
        )
        # Value network
        self.value_net = nn.Sequential(
            GATv2ConvWrapper(feature_dim, last_layer_dim_vf), nn.ReLU(),
            GATv2ConvWrapper(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU(),   
        )


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Forward pass to compute both the policy (actor) and value (critic) outputs.

        Parameters:
        features (Tensor): The input features tensor.

        Returns:
        Tuple[Tensor, Tensor]: The policy output (actions) and the value output (state value).
        """
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass through the policy network to compute the actions.

        Parameters:
        features (Tensor): The input features tensor.

        Returns:
        Tensor: The actions computed by the policy network.
        """
        actions = self.policy_net(features) 
        actions = actions.mean(dim=1) # Aggregate the actions across the node dimension
        return actions

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        """
        Forward pass through the value network to compute the state value.

        Parameters:
        features (Tensor): The input features tensor.

        Returns:
        Tensor: The state value computed by the value network.
        """
        value = self.value_net(features)
        value = value.mean(dim=1) # Aggregate the actions across the node dimension
        return value