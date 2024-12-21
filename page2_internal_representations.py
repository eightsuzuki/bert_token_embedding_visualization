import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import json
import os
import torch
import umap
import hashlib

import sys
sys.dont_write_bytecode = True

def render_page(text, model, tokenizer, tokens_text, embeddings=None):
    st.text_area("Input Text", text, height=300)

    if not embeddings:
        st.error("Embeddings list is empty.")
        return

    @st.cache_data(show_spinner=False)
    def get_cached_embeddings(text_hash, tokens_text, embeddings):
        def merge_tokens(tokens_text, embeddings):
            merged_tokens = []
            merged_embeddings = []
            temp_token = ""
            temp_embedding = []

            for idx, token in enumerate(tokens_text):
                if token.startswith("##"):
                    temp_token += token[2:]
                    temp_embedding.append(embeddings[idx])
                else:
                    if temp_token:
                        merged_tokens.append(temp_token)
                        merged_embeddings.append(np.mean(temp_embedding, axis=0))
                    temp_token = token
                    temp_embedding = [embeddings[idx]]
            if temp_token:
                merged_tokens.append(temp_token)
                merged_embeddings.append(np.mean(temp_embedding, axis=0))

            return merged_tokens, merged_embeddings

        merged_tokens, merged_embeddings = merge_tokens(tokens_text, embeddings)
        return merged_tokens, merged_embeddings

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    merged_tokens, merged_embeddings = get_cached_embeddings(text_hash, tokens_text, embeddings[0])

    st.sidebar.write(f"Total Tokens: {len(tokens_text)}")
    st.sidebar.write(f"Merged Tokens: {len(merged_tokens)}")

    dog_related_tokens = [
        ["dog", "dogs", "inu", "puppy", "canine", "doggy", "puppies", "pup"],
        ["chihuahua", "golden", "retriever", "shiba", "shepherd"],
        ["guide", "therapy", "police", "working", "service"],
        ["mammal", "mammals", "rodent", "primates", "marsupial", "carnivore", "herbivore", "bovine", "feline", "equine", "rodents", "marsupials", "bovines", "felines"],
        ["reptile", "amphibian", "bird", "fish", "insect", "arachnid", "crustacean", "mollusk", "algae", "bacteria", "reptiles", "amphibians", "fishes", "crustaceans", "mollusks", "birds", "plants"],
        [",", ".", "[SEP]"]
    ]

    category_counts = [0] * len(dog_related_tokens)
    dog_indices = []
    dog_labels = []
    for i, category in enumerate(dog_related_tokens):
        indices = [j for j, token in enumerate(merged_tokens) if token.lower() in category]
        dog_indices.extend(indices)
        dog_labels.extend([f"Category {i+1}"] * len(indices))
        category_counts[i] = len(indices)

    unrelated_tokens = [
        token.lower() for token in set(merged_tokens)
        if not any(word in token.lower() for category in dog_related_tokens for word in category)
    ]

    unrelated_indices = [
        i for i, token in enumerate(merged_tokens)
        if token.lower() in unrelated_tokens and i not in dog_indices
    ]

    selected_indices = dog_indices + unrelated_indices
    selected_embeddings = [merged_embeddings[idx] for idx in selected_indices]

    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    st.write(f"Tokenized input: {inputs}")

    os.makedirs('cache/internal/embedding', exist_ok=True)
    os.makedirs('cache/internal/umap', exist_ok=True)
    tokens_text = []
    embeddings = [[] for _ in range(13)]

    cache_exists = True
    for layer_idx in range(13):
        umap_file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
        if not os.path.exists(umap_file_path):
            cache_exists = False
            break

    if cache_exists:
        st.write("Loading UMAP embeddings from cache...")
        umap_embeddings = []
        for layer_idx in range(13):
            umap_file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
            with open(umap_file_path, 'r') as f:
                layer_umap_embeddings = json.load(f)
                umap_embeddings.append(np.array(layer_umap_embeddings))
    else:
        st.write("Generating embeddings...")
        total_chunks = (len(text) + 511) // 512
        progress_bar = st.progress(0)
        for i in range(0, len(text), 512):
            chunk = text[i:i+512]
            tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
            tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
            outputs = model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            for layer_idx in range(13):
                embeddings[layer_idx].append(hidden_states[layer_idx].squeeze(0).detach().numpy())

            progress = (i // 512 + 1) / total_chunks
            progress_bar.progress(progress)

        for layer_idx in range(13):
            layer_embeddings = np.concatenate(embeddings[layer_idx], axis=0)
            file_path = f'cache/internal/embedding/hidden_state_layer_{layer_idx}.json'
            st.write(f"Layer {layer_idx} embeddings shape: {layer_embeddings.shape}")
            with open(file_path, 'w') as f:
                json.dump(layer_embeddings.tolist(), f)

        umap_embeddings = []
        for layer_idx in range(13):
            st.write(f"Performing UMAP dimensionality reduction for layer {layer_idx}...")
            reducer = umap.UMAP(n_components=2)
            layer_embeddings = np.concatenate(embeddings[layer_idx], axis=0)
            umap_embedding = reducer.fit_transform(layer_embeddings)
            umap_embeddings.append(umap_embedding)
            umap_file_path = f'cache/internal/umap/umap_layer_{layer_idx}.json'
            with open(umap_file_path, 'w') as f:
                json.dump(umap_embedding.tolist(), f)
            st.write(f"Layer {layer_idx} UMAP embeddings shape: {umap_embedding.shape}")

    layer_idx = st.selectbox("Select BERT Layer", list(range(13)))

    colors = {
        "Category 1": "darkred",
        "Category 2": "red",
        "Category 3": "orange",
        "Category 4": "lightcoral",
        "Category 5": "coral",
        "Category 6": "purple",
        "Unrelated": "blue"
    }

    token_categories = []
    for token in tokens_text:
        category = "Unrelated"
        for idx, category_tokens in enumerate(dog_related_tokens):
            if token in category_tokens:
                category = f"Category {idx + 1}"
                break
        token_categories.append(category)

    min_length = min(len(umap_embeddings[layer_idx]), len(tokens_text), len(token_categories))
    df = pd.DataFrame({
        "x": umap_embeddings[layer_idx][:min_length, 0],
        "y": umap_embeddings[layer_idx][:min_length, 1],
        "Token": tokens_text[:min_length],
        "Category": token_categories[:min_length]
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Category",
        color_discrete_map=colors,
        hover_data={"Token": True},
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        title=f"UMAP Visualization of BERT Layer {layer_idx} Embeddings"
    )

    st.plotly_chart(fig)

    df_selected = pd.DataFrame({
        "x": [selected_embeddings[i][0] for i in range(len(selected_embeddings))],
        "y": [selected_embeddings[i][1] for i in range(len(selected_embeddings))],
        "Token": [tokens_text[idx] for idx in selected_indices],
        "Category": dog_labels + ["Unrelated"] * len(unrelated_indices)
    })

    fig_selected = px.scatter(
        df_selected,
        x="x",
        y="y",
        color="Category",
        color_discrete_map=colors,
        hover_data={"Token": True},
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        title=f"UMAP Visualization of Selected BERT Layer {layer_idx} Embeddings"
    )

    st.plotly_chart(fig_selected)

    st.sidebar.write("Category Counts:")
    for i, count in enumerate(category_counts):
        st.sidebar.write(f"Category {i+1}: {count} tokens")
    st.sidebar.write(f"Unrelated: {len(unrelated_indices)} tokens")
