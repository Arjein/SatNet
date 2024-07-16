import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
import gymnasium as gym


class GraphFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, feature_dim=11):
        super(GraphFeatureExtractor, self).__init__(observation_space, features_dim=feature_dim)  # features_dim doesn’t matter here
        # Define slices for position and velocity

        self.node_indices = {
            'torso': slice(0, 5),
            'hip_1': slice(5, 6), 'ankle_1': slice(6, 7),
            'hip_2': slice(7, 8), 'ankle_2': slice(8, 9),
            'hip_3': slice(9, 10), 'ankle_3': slice(10, 11),
            'hip_4': slice(11, 12), 'ankle_4': slice(12, 13),
        }
        self.velocity_indices = {
            'torso': slice(13, 19),
            'hip_1': slice(19, 20), 'ankle_1': slice(20, 21),
            'hip_2': slice(21, 22), 'ankle_2': slice(22, 23),
            'hip_3': slice(23, 24), 'ankle_3': slice(24, 25),
            'hip_4': slice(25, 26), 'ankle_4': slice(26, 27),
        }
        # Predefine the edge_index based on the known structure of the Ant
        self.edge_index = th.tensor([[0, 1], [0, 3], [0, 5], [0, 7], [1, 2], [3, 4], [5, 6], [7, 8]], dtype=th.long).t().contiguous()

    def forward(self, observations: th.Tensor):
        #print("Observation to extractor:", observations)
        #print("Shape of Observation to extractor:", observations.shape)
        #print("Tensor shape before operation:", observations.shape)
        #print("observation shape:", observations)

        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        #print("Tensor shape before operation:", observations.shape)
        node_features = []
        i = 0
        for key in self.node_indices.keys():
                        # Extract features for all instances in the batch
            pos_features = observations[:, self.node_indices[key]]
            vel_features = observations[:, self.velocity_indices[key]]
            # Concatenate and pad features for each node across the batch
            combined_features = th.cat((pos_features, vel_features), dim=-1)
            padded_features = F.pad(combined_features, (0, self.features_dim - combined_features.shape[-1]), "constant", 0)
            node_features.append(padded_features)
#        print("Node Features:",node_features)
        node_features_tensor = th.stack(node_features, dim=1)  # Adjust shape if necessary
        
        # Assuming the edge index is the same for all in the batch and does not change
        #graph = Data(x=node_features_tensor, edge_index=self.edge_index, num_nodes=9)
        
        return node_features_tensor