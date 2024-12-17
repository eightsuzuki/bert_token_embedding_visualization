import streamlit as st
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import plotly.express as px
import hashlib
import json
import os
import torch
import umap

import sys
sys.dont_write_bytecode = True

def render_page(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    # Create directory if it doesn't exist
    os.makedirs('cache/internal/embedding', exist_ok=True)
    os.makedirs('cache/internal/umap', exist_ok=True)
    tokens_text = []
    embeddings = [[] for _ in range(13)]  # 13 layers including the embedding layer

    cache_exists = True
    for layer_idx in range(13):
        file_path = f'cache/internal/embedding/hidden_state_layer_{layer_idx}.json'
        if not os.path.exists(file_path):
            cache_exists = False
            break

    if cache_exists:
        st.write("Loading embeddings from cache...")
        for layer_idx in range(13):
            file_path = f'cache/internal/embedding/hidden_state_layer_{layer_idx}.json'
            with open(file_path, 'r') as f:
                layer_embeddings = json.load(f)
                embeddings[layer_idx] = [np.array(layer_embeddings)]
    else:
        st.write("Generating embeddings...")
        total_chunks = (len(text) + 511) // 512  # Calculate the total number of chunks
        progress_bar = st.progress(0)  # Initialize the progress bar
        for i in range(0, len(text), 512):
            # Split text into chunks of 512 characters
            chunk = text[i:i+512]
            # Tokenize and pad/truncate as necessary
            tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            # Convert token IDs to tokens and add to tokens_text list
            tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
            # Get the model outputs
            outputs = model(**tokens, output_hidden_states=True)
            # Get the hidden states from the model outputs
            hidden_states = outputs.hidden_states

            # Append the hidden states for each layer to the embeddings list
            for layer_idx in range(13):
                embeddings[layer_idx].append(hidden_states[layer_idx].squeeze(0).detach().numpy())

            # Update the progress bar
            progress = (i // 512 + 1) / total_chunks
            progress_bar.progress(progress)

        # Save each layer's hidden states to individual JSON files
        for layer_idx in range(13):
            layer_embeddings = np.concatenate(embeddings[layer_idx], axis=0)
            file_path = f'cache/internal/embedding/hidden_state_layer_{layer_idx}.json'
            st.write(f"Layer {layer_idx} embeddings shape: {layer_embeddings.shape}")
            with open(file_path, 'w') as f:
                json.dump(layer_embeddings.tolist(), f)

    # Perform UMAP dimensionality reduction and cache the results
    umap_cache_exists = True
    umap_embeddings = []
    for layer_idx in range(13):
        file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
        if not os.path.exists(file_path):
            umap_cache_exists = False
            break

    if umap_cache_exists:
        st.write("Loading UMAP embeddings from cache...")
        for layer_idx in range(13):
            file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
            with open(file_path, 'r') as f:
                layer_umap_embeddings = json.load(f)
                umap_embeddings.append(np.array(layer_umap_embeddings))
    else:
        st.write("Performing UMAP dimensionality reduction...")
        for layer_idx in range(13):
            reducer = umap.UMAP(n_components=2)
            layer_embeddings = np.concatenate(embeddings[layer_idx], axis=0)
            umap_embedding = reducer.fit_transform(layer_embeddings)
            umap_embeddings.append(umap_embedding)
            file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
            with open(file_path, 'w') as f:
                json.dump(umap_embedding.tolist(), f)
            st.write(f"Layer {layer_idx} UMAP embeddings shape: {umap_embedding.shape}")

    # You can now use umap_embeddings for further processing or visualization
