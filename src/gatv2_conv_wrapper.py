from torch import nn
from torch_geometric.nn import GATv2Conv
import torch as th

class GATv2ConvWrapper(nn.Module):
    """
    Wrapper class for GATv2Conv to enable its usage in nn.Sequential by internally managing a static edge_index.
    """
    def __init__(self, in_channels, out_channels):
        """
        Initialize the GATv2ConvWrapper.

        Parameters:
        in_channels (int): Number of input features per node.
        out_channels (int): Number of output features per node.
        """

        super(GATv2ConvWrapper, self).__init__()
        self.gatv2conv = GATv2Conv(in_channels, out_channels)
        
        # Define a static edge_index
        # The edge_index should be manually defined for the specific environment.
        # Uncomment the appropriate edge_index definition depending on the environment.
        
        # Ant
        
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
        
        
        # Half-Cheetah
        """
        self.edge_index = th.tensor([
            [0, 4], [4, 0],
            [4, 1], [1, 4],
            [1, 2], [2, 1],
            [2, 3], [3, 2],
            [4, 5], [5, 4],
            [5, 6], [6, 5] 
           
            ], dtype=th.long).t().contiguous()
        """
    
        
        # Humanoid
        """
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
        """
        
    def forward(self, x):
        """
        Forward pass through the GATv2Conv layer with a static edge_index.

        Parameters:
        x (Tensor): The input node features, which can be either a single graph or a batch of graphs.

        Returns:
        Tensor: The output node features after applying the GATv2Conv layer.
        """
        if x.dim() == 3:  # Batch processing
            outputs = []
            # Process each graph in the batch independently
            for single_graph_features in x:  
                output = self.gatv2conv(single_graph_features, self.edge_index)
                outputs.append(output)
            return th.stack(outputs, dim=0)
            # Process a single graph
        return self.gatv2conv(x, self.edge_index)

    