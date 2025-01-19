import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import streamlit as st
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, FastICA
import umap
import pandas as pd
import plotly.express as px
import hashlib
import json
import torch
import os
import imageio
import plotly.io as pio

# Function to get embeddings from a specific layer
def get_layer_embeddings(model, inputs, layer_idx):
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    return outputs.hidden_states[layer_idx].squeeze().numpy()


# ページ 4: 文脈構造の表現
def render_page(tokenizer, model):
    # 選択肢を追加
    option = st.selectbox("Select option", ["Days of the Week", "Months"])
    
    if option == "Days of the Week":
        items = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    else:
        items = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
    
    if option == "Days of the Week":
        num_repeats = st.number_input("Number of repetitions", min_value=1, max_value=100, value=70)
    else:
        num_repeats = st.number_input("Number of repetitions", min_value=1, max_value=100, value=40)
    
    text = " ".join(items * num_repeats)
    st.text_area("Input Text", text, height=300)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = get_layer_embeddings(model, inputs, -1)
    st.write("Embeddings shape:", embeddings.shape)
    # 次元削減の方法を選択
    reduction_method = st.selectbox("Select dimensionality reduction method", ["UMAP", "ICA", "PCA", "t-SNE"])
    
    if reduction_method == "PCA":
        reducer = PCA(n_components=2)
        
    elif reduction_method == "ICA":
        reducer = FastICA(n_components=2)
        
    elif reduction_method == "t-SNE":
        reducer = TSNE(n_components=2)
        
    elif reduction_method == "UMAP":
        reducer = umap.UMAP(n_components=2)
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    # アイテムごとに色をつける
    labels = (items * num_repeats)[:embeddings.shape[0]]
    embeddings_reduced = embeddings_reduced[:len(labels)]
    df = pd.DataFrame(embeddings_reduced, columns=["Component 1", "Component 2"])
    df["Label"] = labels

    # Define a color map
    colors = {label: px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)] for i, label in enumerate(set(labels))}

    # Create the directory based on the selected option
    if option == "Days of the Week":
        output_dir = "./image/context_structure/days"
    else:
        output_dir = "./image/context_structure/months"
    
    os.makedirs(output_dir, exist_ok=True)

    title = f"{reduction_method} Reduced Embeddings"
    fig = px.scatter(df, x="Component 1", y="Component 2", color="Label", color_discrete_map=colors, title=title)
    st.plotly_chart(fig)

    # Save the figure as an image
    output_path = os.path.join(output_dir, "embedding_plot.png")
    pio.write_image(fig, output_path)

    if st.button("Plot Intermediate Layers"):
        intermediate_layers = [i for i in range(0, model.config.num_hidden_layers)]
        st.write(f"Number of intermediate layers: {len(intermediate_layers)}")
        images = []
        for layer_idx in intermediate_layers:
            # 中間層の埋め込みを取得
            intermediate_embeddings = get_layer_embeddings(model, inputs, layer_idx)

            # 次元削減
            intermediate_embeddings_reduced = reducer.fit_transform(intermediate_embeddings)
            intermediate_embeddings_reduced = intermediate_embeddings_reduced[:len(labels)]
            df_intermediate = pd.DataFrame(intermediate_embeddings_reduced, columns=["Component 1", "Component 2"])
            df_intermediate["Label"] = labels
            df_intermediate["Layer"] = layer_idx

            images.append(df_intermediate)

        # 最終層の埋め込みを追加
        final_layer_idx = model.config.num_hidden_layers
        final_embeddings = get_layer_embeddings(model, inputs, -1)
        final_embeddings_reduced = reducer.fit_transform(final_embeddings)
        final_embeddings_reduced = final_embeddings_reduced[:len(labels)]
        df_final = pd.DataFrame(final_embeddings_reduced, columns=["Component 1", "Component 2"])
        df_final["Label"] = labels
        df_final["Layer"] = final_layer_idx

        images.append(df_final)

        # Combine all layers into a single DataFrame
        df_combined = pd.concat(images)

        # Create an animated plot
        fig_animated = px.scatter(df_combined, x="Component 1", y="Component 2", color="Label", animation_frame="Layer", 
                                  color_discrete_map=colors, title=f"{reduction_method} Reduced Embeddings Across Layers")
        x_min = st.number_input("X-axis min", value=-20)
        x_max = st.number_input("X-axis max", value=22)
        y_min = st.number_input("Y-axis min", value=-20)
        y_max = st.number_input("Y-axis max", value=22)

        fig_animated.update_xaxes(range=[x_min, x_max])  # 座標範囲を固定
        fig_animated.update_yaxes(range=[y_min, y_max])  # 座標範囲を固定
        st.plotly_chart(fig_animated)
