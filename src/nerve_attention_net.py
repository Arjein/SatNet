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

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        

        # Policy network
        self.policy_net = nn.Sequential(
            GATv2ConvWrapper(feature_dim, last_layer_dim_pi), nn.Tanh(),
            GATv2ConvWrapper(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh(),
            #nn.Linear(feature_dim,last_layer_dim_pi), nn.Tanh(),
            #nn.Linear(last_layer_dim_pi,last_layer_dim_pi), nn.Tanh()
        )
        # Value network
        self.value_net = nn.Sequential(
            #GATv2ConvWrapper(feature_dim, last_layer_dim_vf), nn.ReLU(),
            #GATv2ConvWrapper(last_layer_dim_vf, last_layer_dim_vf), nn.ReLU(),
            GATv2ConvWrapper(feature_dim, last_layer_dim_vf), nn.Tanh(),
            GATv2ConvWrapper(last_layer_dim_vf, last_layer_dim_vf), nn.Tanh(),
            #nn.Linear(feature_dim,last_layer_dim_pi), nn.Tanh(),
            #nn.Linear(last_layer_dim_pi,last_layer_dim_pi), nn.Tanh()
            
        )


    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        #print("Batch size in forward pass:", features.x.size(0))
        
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        
        actions = self.policy_net(features)
        #print("Actions (Wout Aggregation):", actions)
        actions = actions.mean(dim=1)
        #print("Actions:", actions)
        #print("Actions:", actions)
#        print("Actions:", actions)
        #print("Policy Output:", actions)
        #print("Policy Output (Shape):", actions.shape)
        return actions

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        
        # Apply the value network first
        #print(features.x)
        #print(features.x)
        value = self.value_net(features)  # Ensure to use features.x to get the node features tensor
        value = value.mean(dim=1)
        #print("Value Shape", value.shape)
        #value = value.mean(dim=0, keepdim=True).view(1, -1)  # Flatten and maintain batch size
        #print("Value:", value)
        #print("Value Shape After Aggregation", value.shape)

        return value