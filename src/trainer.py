import argparse
import os
import sys
import torch
import torch.nn.functional as F

# Fixed import path with correct parenthesis
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.gnn_model import TGN_DiseasePredictor, EvolveGCN_Predictor

def train_model(data, model_type="tgn", epochs=10):
    # Fixed model initialization with balanced parentheses
    if model_type == "tgn":
        model = TGN_DiseasePredictor(
            node_features=data.x.shape[1],
            edge_features=data.edge_attr.shape[1],
            num_classes=len(torch.unique(data.y)))  # 3 closing parentheses
    else:
        model = EvolveGCN_Predictor(
            node_features=data.x.shape[1],
            num_classes=len(torch.unique(data.y)))  # 3 closing parentheses
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = F.cross_entropy(out, data.y)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", choices=["tgn", "evolvegcn"], default="tgn")
    args = parser.parse_args()
    
    # Dummy data
    class Data:
        def __init__(self):
            self.x = torch.randn(10, 5)  # 10 nodes, 5 features
            self.edge_index = torch.tensor([[0,1],[1,2],[2,3]]).t()  # 3 edges
            self.edge_attr = torch.randn(3, 2)  # Edge features
            self.y = torch.randint(0, 3, (10,))  # 3 classes
    
    data = Data()
    os.makedirs("models", exist_ok=True)
    model = train_model(data, args.model_type)
    torch.save(model.state_dict(), f"models/{args.model_type}_model.pt")
    print(f"SUCCESS! Model saved to models/{args.model_type}_model.pt")