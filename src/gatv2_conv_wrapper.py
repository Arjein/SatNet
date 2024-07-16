from torch import nn
from torch_geometric.nn import GATv2Conv
import torch

class GATv2ConvWrapper(nn.Module):
    """
    Wrapper class for GATv2Conv to enable its usage in nn.Sequential by internally managing a static edge_index.
    """
    def __init__(self, in_channels, out_channels):
        super(GATv2ConvWrapper, self).__init__()
        self.gatv2conv = GATv2Conv(in_channels, out_channels)
        # Define a static edge_index
        self.edge_index = torch.tensor([[0, 1], [0, 3], [0, 5], [0, 7], [1, 2], [3, 4], [5, 6], [7, 8]], dtype=torch.long).t().contiguous()

    def forward(self, x):
        if x.dim() == 3:  # Batch processing
            outputs = []
            for single_graph_features in x:  # Process each graph in the batch
                output = self.gatv2conv(single_graph_features, self.edge_index)
                outputs.append(output)
            return torch.stack(outputs, dim=0)
        return self.gatv2conv(x, self.edge_index)
# Torch shape 3lü extracted features da sebebini bilmiyorum 2 olması gerekiyor sanırım
# Aggregation hatalı olabilir.
    