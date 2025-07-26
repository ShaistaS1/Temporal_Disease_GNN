import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import ast

def build_temporal_graph(df):
    """Create a graph from patient data"""
    G = nx.DiGraph()
    
    for _, row in df.iterrows():
        try:
            # Get diagnosis
            diag = str(row['diagnosis']).lower().strip()
            G.add_node(diag, type='diagnosis', size=1000)
            
            # Process symptoms
            symptoms = ast.literal_eval(str(row['symptoms']))
            for symptom in symptoms:
                symptom_clean = str(symptom).lower().strip()
                G.add_node(symptom_clean, type='symptom', size=500)
                G.add_edge(symptom_clean, diag)
                
        except:
            continue
            
    return G

def main():
    st.title("Temporal Disease Onset Prediction Using Graph Neural Networks")
    
    uploaded_file = st.file_uploader("Upload patient CSV", type=["csv"])
    
    if uploaded_file:
        try:
            # Read and show data
            df = pd.read_csv(uploaded_file)
            st.write("## Data Preview", df.head())
            
            # Build and show graph
            G = build_temporal_graph(df)
            
            plt.figure(figsize=(12,8))
            pos = nx.spring_layout(G)
            
            # Color nodes by type
            node_colors = ['red' if G.nodes[n]['type'] == 'diagnosis' else 'skyblue' for n in G.nodes]
            node_sizes = [G.nodes[n]['size'] for n in G.nodes]
            
            nx.draw(G, pos, with_labels=True, 
                   node_color=node_colors,
                   node_size=node_sizes,
                   font_size=8,
                   arrowsize=20)
            
            st.pyplot(plt)
            
            # Show stats
            st.write(f"**Graph contains:**")
            st.write(f"- {len([n for n in G.nodes if G.nodes[n]['type']=='diagnosis'])} diagnoses")
            st.write(f"- {len([n for n in G.nodes if G.nodes[n]['type']=='symptom'])} symptoms")
            st.write(f"- {len(G.edges)} connections")
            
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == "__main__":
    main()