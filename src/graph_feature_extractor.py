import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn.functional as F
from torch_geometric.data import Data
import gymnasium as gym


class GraphFeatureExtractor(BaseFeaturesExtractor):
    """
    A custom feature extractor for environments with graph-structured observations.

    Attributes:
    features_dim (int): The dimensionality of the features for each node in the graph.
    node_indices (dict): A dictionary mapping node names to their corresponding slices in the observation tensor.
    velocity_indices (dict): A dictionary mapping node names to their corresponding velocity slices in the observation tensor.
    edge_index (Tensor): A tensor representing the edges in the graph.
    """
    def __init__(self, observation_space: gym.Space, features_dim = 11):
        """
        Initializes the GraphFeatureExtractor with observation space, environment, and features dimension.

        Parameters:
        observation_space (gym.Space): The observation space from Gymnasium.
        environment (str): The environment name, which determines the node and edge structure. Not fully implemented.
        features_dim (int): The feature dimension for each node in the graph which is equal to maximum action space dimension of environment.
        """
        super(GraphFeatureExtractor, self).__init__(observation_space, features_dim)  # features_dim doesn’t matter here
        
      
        environment = "Ant-v4"
        if environment == "Humanoid-v4" or environment == "HumanoidStandup-v4":
            print("Env: Humanoid GFE")    
            self.features_dim = 11
            self.node_indices = {
                'torso': slice(0, 5),  
                'abdomen': slice(5, 8),
                'right_hip': slice(8, 11),
                'right_knee': slice(11, 12),
                'left_hip': slice(12, 15),
                'left_knee': slice(15, 16),
                'right_shoulder': slice(16, 18),
                'right_elbow': slice(18, 19),
                'left_shoulder': slice(19, 21),
                'left_elbow': slice(21, 22),
            }
            self.velocity_indices = {
                'torso': slice(22, 28),          # 0
                'abdomen': slice(28, 31),        # 1
                'right_hip': slice(31, 34),      # 2
                'right_knee': slice(34, 35),     # 3
                'left_hip': slice(35, 38),       # 4
                'left_knee': slice(38, 39),      # 5
                'right_shoulder': slice(39, 41), # 6
                'right_elbow': slice(41, 42),    # 7
                'left_shoulder': slice(42, 44),  # 8
                'left_elbow': slice(44, 45),     # 9
            }
            self.edge_index = th.tensor([
            [0, 1], [1, 0],
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [1, 4], [4, 1],
            [4, 5], [5, 4],
            [0, 6], [6, 0],
            [6, 7], [7, 6],
            [0, 8], [8, 0],
            [8, 9], [9, 8],

            ], dtype=th.long).t().contiguous()

        if environment == "HalfCheetah-v4":
            print("Env: HalfCheetah-v4 GFE")    
            self.features_dim = 5 
            self.node_indices = {
                'f_tip': slice(0, 2),  # adjust these based on actual observation details
                'b_thigh': slice(2, 3),
                'b_shin': slice(3, 4),
                'b_foot': slice(4, 5),
                'f_thigh': slice(5, 6),
                'f_shin': slice(6, 7),
                'f_foot': slice(7, 8),
                
                
            }
            self.velocity_indices = {
                'f_tip': slice(8, 11),  # adjust these based on actual observation details
                'b_thigh': slice(11, 12),
                'b_shin': slice(12, 13),
                'b_foot': slice(13, 14),
                'f_thigh': slice(14, 15),
                'f_shin': slice(15, 16),
                'f_foot': slice(16, 17),
                
            }
            self.edge_index = th.tensor([
                [0, 4], [4, 0],
                [4, 1], [1, 4],
                [1, 2], [2, 1],
                [2, 3], [3, 2],
                [4, 5], [5, 4],
                [5, 6], [6, 5] 
                ], dtype=th.long).t().contiguous()
        
        if environment == "Ant-v4":
            print("Env: Ant-v4 GFE")    
            self.features_dim = 11
            self.num_of_nodes = 9
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
            self.edge_index = th.tensor([
            [0, 1], [1, 0],
            [0, 3], [3, 0],
            [0, 5], [5, 0],
            [0, 7], [7, 0],
            [1, 2], [2, 1],
            [3, 4], [4, 3],
            [5, 6], [6, 5],
            [7, 8], [8, 7]
            ], dtype=th.long).t().contiguous()
        
        
    @property
    def features_dim(self):
        """
        Retrieves the feature dimension.
        
        Returns:
        int: The current feature dimension.
        """

        return self._features_dim
    
    @features_dim.setter
    def features_dim(self, value):
        """
        Property to set the feature dimension.
        
        Parameters:
        value (int): The feature dimension to be set.
        """

        self._features_dim = value
        
    def forward(self, observations: th.Tensor):
        """
        Forward pass through the feature extractor.

        Parameters:
        observations (Tensor): The observation tensor.

        Returns:
        Tensor: A tensor of node features for the graph.
        """
        
        if observations.dim() == 1:
            observations = observations.unsqueeze(0)
        
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

        # Stack the node features to create a tensor with the shape required by the model
        node_features_tensor = th.stack(node_features, dim=1)  # Adjust shape if necessary
        
        return node_features_tensor