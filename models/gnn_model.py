import torch
import torch.nn as nn

class SimpleGNN(nn.Module):
    def __init__(self, node_features, num_classes):
        super().__init__()
        self.conv1 = nn.Linear(node_features, 64)
        self.conv2 = nn.Linear(64, num_classes)
        
    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x))
        return self.conv2(x)

class TGN_DiseasePredictor(SimpleGNN):
    def __init__(self, node_features, edge_features, num_classes):
        super().__init__(node_features, num_classes)
        self.edge_processor = nn.Linear(edge_features, 64)

class EvolveGCN_Predictor(SimpleGNN):
    def __init__(self, node_features, num_classes):
        super().__init__(node_features, num_classes)