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
import plotly.io as pio


# ページ 4: 文脈構造の表現
def render_page(tokenizer, model):
    # inputを{Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday}で、何回これを入力できるかの設定をできるようにして、それをtextに保存し、tokenizer, modelを使って埋め込みを取得して
    days_of_week = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    num_repeats = st.number_input("Number of repetitions", min_value=1, max_value=100, value=70)
    
    text = " ".join(days_of_week * num_repeats)
    st.text_area("Input Text", text, height=300)
    
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    
    embeddings = outputs.last_hidden_state.squeeze().numpy()
    st.write("Embeddings shape:", embeddings.shape)
    # 次元削減の方法を選択
    reduction_method = st.selectbox("Select dimensionality reduction method", ["PCA", "ICA", "t-SNE", "UMAP"])
    
    if reduction_method == "PCA":
        reducer = PCA(n_components=2)
        
    elif reduction_method == "ICA":
        reducer = FastICA(n_components=2)
        
    elif reduction_method == "t-SNE":
        reducer = TSNE(n_components=2)
    elif reduction_method == "UMAP":
        reducer = umap.UMAP(n_components=2)
    
    embeddings_reduced = reducer.fit_transform(embeddings)
    # 曜日ごとに色をつける
    labels = (days_of_week * num_repeats)[:embeddings.shape[0]]
    embeddings_reduced = embeddings_reduced[:len(labels)]
    df = pd.DataFrame(embeddings_reduced, columns=["Component 1", "Component 2"])
    df["Day"] = labels

    # Create the directory if it doesn't exist
    output_dir = "./image/day_of_week/context_structure"
    os.makedirs(output_dir, exist_ok=True)

    title = f"{reduction_method} Reduced Embeddings"
    fig = px.scatter(df, x="Component 1", y="Component 2", color="Day", title=title)
    st.plotly_chart(fig)

    # Save the figure as an image
    output_path = os.path.join(output_dir, "embedding_plot.png")
    pio.write_image(fig, output_path)
