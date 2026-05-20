import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import global_mean_pool

class TrainableRBF(nn.Module):
    """
    A trainable Radial Basis Function layer.
    Learns the optimal distance shells/centers (mu) and sharpness (gamma) end-to-end.
    """
    def __init__(self, num_rbf=16, low=0.5, high=5.0):
        super().__init__()
        # Initialize centers (mu) evenly spaced between low and high bounds
        mu = torch.linspace(low, high, num_rbf)
        self.mu = nn.Parameter(mu) # Made a Parameter so backpropagation can tune it!
        
        # Initialize sharpness (gamma) based on the spacing gap
        gap = (high - low) / (num_rbf - 1)
        self.gamma = nn.Parameter(torch.ones(num_rbf) / (gap ** 2))

    def forward(self, dists):
        # dists shape: [Num_Edges, 1]
        # self.mu and self.gamma shape: [Num_RBF]
        # Output shape: [Num_Edges, Num_RBF]
        diff = dists - self.mu.view(1, -1)
        return torch.exp(-self.gamma.view(1, -1) * (diff ** 2))

class Simple3DGNNLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_rbf=16):
        super().__init__()
        self.rbf = TrainableRBF(num_rbf=num_rbf)
        self.distance_mlp = nn.Linear(num_rbf, out_dim)
        
        # ADD THIS: Projects the neighbor's raw input dim (4) to hidden dim (32)
        self.node_encoder = nn.Linear(in_dim, out_dim)
        
        self.node_mlp = nn.Linear(out_dim, out_dim) # Updated to take out_dim
        self.act = nn.ReLU()

    def forward(self, x, pos, edge_index):
        row, col = edge_index
        
        # 1. Compute invariant Euclidean distances
        coord_diff = pos[row] - pos[col]
        dists = torch.norm(coord_diff, p=2, dim=-1, keepdim=True)
        
        # 2. Pass distances through trainable RBF shells
        rbf_features = self.rbf(dists) 
        edge_weights = self.distance_mlp(rbf_features) # Shape: [Num_Edges, 32]
        
        # 3. FIX: Project neighbor node features to match the 32 hidden dimensions
        h_col = self.node_encoder(x[col])              # Shape: [Num_Edges, 32]

        # By convention, row indicates the info target and col the source
        # Here we get the weighted encoded info from the source
        # Later we will aggregate this to the target
        
        # 4. Message Generation: Now both sides are size 32!
        messages = h_col * edge_weights                # Shape: [Num_Edges, 32]
        
        # 5. Aggregation
        aggregated = torch.zeros(x.size(0), messages.size(-1), device=x.device)
        aggregated.index_add_(0, row, messages)
        
        # 6. Update
        return self.act(self.node_mlp(aggregated))

class CavityPredictorGNN(nn.Module):
    """
    The full Network architecture that encodes the 3D structure 
    and outputs a single global prediction value (e.g., Cavity Size).
    """
    def __init__(self, node_dim=4, hidden_dim=32, num_rbf=16):
        super().__init__()
        self.conv1 = Simple3DGNNLayer(node_dim, hidden_dim, num_rbf)
        self.conv2 = Simple3DGNNLayer(hidden_dim, hidden_dim, num_rbf)
        
        # Final output head to map pooled representations to a single scalar property
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, batch_data):
        x, pos, edge_index = batch_data.x, batch_data.pos, batch_data.edge_index
        
        # Core 3D message-passing iterations
        x = self.conv1(x, pos, edge_index)
        x = self.conv2(x, pos, edge_index)
        
        # Invariant Global Pooling: Compress atom-level vectors into a singular molecular fingerprint
        graph_vector = global_mean_pool(x, batch_data.batch) # [Batch_Size, hidden_dim]
        
        # Output spatial prediction
        return self.prediction_head(graph_vector)

def get_demo_molecule():
    # 4 Carbon atoms arranged in a flat 1.5Å x 1.5Å square (creating a central cavity)
    # One-hot node features: let's pretend [1, 0, 0, 4] means a Carbon atom with 4 valence electrons
    x = torch.tensor([
        [1.0, 0.0, 0.0, 4.0], # Atom 0
        [1.0, 0.0, 0.0, 4.0], # Atom 1
        [1.0, 0.0, 0.0, 4.0], # Atom 2
        [1.0, 0.0, 0.0, 4.0]  # Atom 3
    ], dtype=torch.float)

    # 3D Absolute Coordinates (X, Y, Z)
    pos = torch.tensor([
        [0.0, 0.0, 0.0],  # Atom 0 (Bottom-Left)
        [0.0, 1.5, 0.0],  # Atom 1 (Top-Left)
        [1.5, 1.5, 0.0],  # Atom 2 (Top-Right)
        [1.5, 0.0, 0.0]   # Atom 3 (Bottom-Right)
    ], dtype=torch.float)

    # Fully connected graph topology so all cross-cavity distances can be evaluated
    edge_index = torch.tensor([
        [0,0,0,1,1,1,2,2,2,3,3,3],
        [1,2,3,0,2,3,0,1,3,0,1,2]
    ], dtype=torch.long)

    # Ground truth property we want to predict (e.g., calculated internal cavity size)
    y = torch.tensor([[2.25]], dtype=torch.float)

    return Data(x=x, pos=pos, edge_index=edge_index, y=y)

if __name__ == "__main__":
    # Instantiate the network
    model = CavityPredictorGNN()

    # Generate the sample data structure and bundle it into a PyG batch structure [3]
    molecule = get_demo_molecule()
    batch = Batch.from_data_list([molecule]) 

    # Execute a forward step
    prediction = model(batch)

    print("--- 3D GNN Execution Complete ---")
    print(f"Input Node Features Shape: {batch.x.shape}")
    print(f"Input Coordinate Shape:    {batch.pos.shape}")
    print(f"Predicted Cavity Metric:    {prediction.item():.4f}")
    print("\nInitial unoptimized RBF Centers (mu):\n", model.conv1.rbf.mu.data)
