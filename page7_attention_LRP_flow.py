import torch
import numpy as np
import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
from transformers import BertModel, BertTokenizer

def compute_flows(attentions):
    """Compute the flow values for each token across layers, ensuring correct dimensions."""
    num_layers = len(attentions)
    seq_len = attentions[0].shape[-1]
    flow_values = np.zeros((num_layers, seq_len, seq_len))
    
    for layer_idx, layer_attn in enumerate(attentions):
        attn_matrix = layer_attn.mean(dim=1).squeeze(0).cpu().numpy()
        if attn_matrix.shape == (seq_len, seq_len):
            flow_values[layer_idx] = attn_matrix
    
    return flow_values

def draw_attention_graph(flow_values, n_layers, seq_len):
    """Draw attention flow graph using NetworkX."""
    G = nx.DiGraph()
    
    # Create nodes for each layer and token
    for layer in range(n_layers):
        for token in range(seq_len):
            G.add_node((layer, token), label=f"L{layer}_T{token}")
    
    # Add edges based on attention flow
    for layer in range(1, n_layers):
        for token in range(seq_len):
            for prev_token in range(seq_len):
                weight = flow_values[layer - 1, prev_token, token]
                if weight > 0.01:  # Threshold to filter out weak connections
                    G.add_edge((layer - 1, prev_token), (layer, token), weight=weight)
    
    # Define positions for visualization
    pos = {(layer, token): (layer, -token) for layer in range(n_layers) for token in range(seq_len)}
    edge_weights = [G[u][v]['weight'] for u, v in G.edges]
    
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=False, node_color='lightblue', edge_color=edge_weights, edge_cmap=plt.cm.Blues, 
            width=[w * 5 for w in edge_weights], node_size=300, alpha=0.8)
    
    labels = {node: G.nodes[node]['label'] for node in G.nodes}
    nx.draw_networkx_labels(G, pos, labels, font_size=8)
    
    plt.title("Attention Flow Across Layers")
    return plt

def render_page():
    st.title("BERT Attention Flow Visualization")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()
    
    text_input = st.text_area("Input Text", "She was a teacher for forty years.")
    if st.button("Compute Attention Flow"):
        inputs = tokenizer(text_input, return_tensors="pt", max_length=16, truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        attentions = outputs.attentions  # List of attention matrices
        
        flow_values = compute_flows(attentions)
        fig = draw_attention_graph(flow_values, n_layers=len(attentions), seq_len=inputs['input_ids'].shape[1])
        
        st.subheader("Attention Flow Graph")
        st.pyplot(fig)
    
if __name__ == "__main__":
    render_page()
