import os
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx

def build_temporal_graph(ehr_df):
    G = nx.DiGraph()
    
    # Create mapping from node names to integer indices
    node_mapping = {}
    current_idx = 0
    
    for _, row in ehr_df.iterrows():
        patient = row["patient_id"]
        time = row["timestamp"]
        diag = row["diagnosis"]
        
        # Add diagnosis node with numeric index
        if diag not in node_mapping:
            node_mapping[diag] = current_idx
            current_idx += 1
        
        # Link symptoms → diagnosis
        for symptom in eval(row["symptoms"]):  # Use eval() to convert string to list
            if symptom not in node_mapping:
                node_mapping[symptom] = current_idx
                current_idx += 1
            G.add_edge(node_mapping[symptom], node_mapping[diag])
        
        # Link diagnosis → treatment
        treatment = row["treatment"]
        if treatment not in node_mapping:
            node_mapping[treatment] = current_idx
            current_idx += 1
        G.add_edge(node_mapping[diag], node_mapping[treatment])
    
    return G, node_mapping

def nx_to_pyg(G):
    # Convert NetworkX graph to PyG format
    pyg_data = from_networkx(G)
    
    # Add dummy node features if needed
    pyg_data.x = torch.ones((G.number_of_nodes(), 1))
    
    return pyg_data

if __name__ == "__main__":
    # Ensure directories exist
    os.makedirs("../data/processed", exist_ok=True)
    
    # Load and process data
    ehr_df = pd.read_csv("../data/raw/synthetic_ehr.csv")
    G, node_mapping = build_temporal_graph(ehr_df)
    
    # Save mapping for later use
    pd.DataFrame.from_dict(node_mapping, orient='index', columns=['index']).to_csv(
        "../data/processed/node_mapping.csv")
    
    # Convert and save graph
    pyg_graph = nx_to_pyg(G)
    torch.save(pyg_graph, "../data/processed/temporal_graph.pt")
    print("Graph built and saved successfully!")