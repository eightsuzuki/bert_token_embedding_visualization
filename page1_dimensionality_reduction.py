import streamlit as st
import numpy as np
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, FastICA
import umap
import pandas as pd
import plotly.express as px
import hashlib
import json
import os

# ページ 1: 次元削減技術の比較
def render_page(text, tokens_text, embeddings):
    st.text_area("Input Text", text, height=300)

    # トークンと埋め込みをキャッシュして取得
    @st.cache_data(show_spinner=False)
    def get_cached_embeddings(text_hash):
        def merge_tokens(tokens_text, embeddings):
            merged_tokens = []
            merged_embeddings = []
            temp_token = ""
            temp_embedding = []

            for idx, token in enumerate(tokens_text):
                if token.startswith("##"):
                    temp_token += token[2:]  # "##"を除いて結合
                    temp_embedding.append(embeddings[idx])
                else:
                    if temp_token:
                        # 結合したトークンを確定
                        merged_tokens.append(temp_token)
                        merged_embeddings.append(np.mean(temp_embedding, axis=0))
                    temp_token = token
                    temp_embedding = [embeddings[idx]]
            if temp_token:  # 最後のトークンを確定
                merged_tokens.append(temp_token)
                merged_embeddings.append(np.mean(temp_embedding, axis=0))

            return merged_tokens, merged_embeddings

        merged_tokens, merged_embeddings = merge_tokens(tokens_text, embeddings)
        return merged_tokens, merged_embeddings

    # テキストのハッシュを計算
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # キャッシュからトークンと埋め込みを取得
    merged_tokens, merged_embeddings = get_cached_embeddings(text_hash)

    # トークン総数をサイドバーに表示
    st.sidebar.write(f"Total Tokens: {len(tokens_text)}")
    st.sidebar.write(f"Merged Tokens: {len(merged_tokens)}")

    # 犬関連トークンを分類
    dog_related_tokens = [
        ["dog", "dogs", "inu", "puppy", "canine", "doggy", "puppies", "pup"],  # Directly representing dogs
        ["chihuahua", "golden", "retriever", "shiba", "shepherd"],  # Types of dogs
        ["guide", "therapy", "police", "working", "service"],  # Associated with dogs but not directly meaning 'dog'
        ["mammal", "mammals", "rodent", "primates", "marsupial", "carnivore", "herbivore", "bovine", "feline", "equine", "rodents", "marsupials", "bovines", "felines"],  # For other mammals
        ["reptile", "amphibian", "bird", "fish", "insect", "arachnid", "crustacean", "mollusk", "algae", "bacteria", "reptiles", "amphibians", "fishes", "crustaceans", "mollusks", "birds", "plants"]  # For non-mammal living organisms
    ]

    category_counts = [0] * len(dog_related_tokens)
    dog_indices = []
    dog_labels = []
    for i, category in enumerate(dog_related_tokens):
        indices = [j for j, token in enumerate(merged_tokens) if any(word in token.lower() for word in category)]
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

    method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "ICA", "ISOMAP", "HessianLLE", "t-SNE", "UMAP"]
    )

    @st.cache_data
    def reduce_dimensions_with_cache(method, embeddings, n_neighbors, text_hash):
        def reduce_dimensions(method, embeddings, n_neighbors=10):
            embeddings = np.array(embeddings)

            if method == "PCA":
                reducer = PCA(n_components=2)
            elif method == "ICA":
                reducer = FastICA(n_components=2)
            elif method == "ISOMAP":
                reducer = Isomap(n_components=2)
            elif method == "HessianLLE":
                reducer = LocallyLinearEmbedding(n_components=2, method='hessian', n_neighbors=n_neighbors, eigen_solver='dense')
            elif method == "t-SNE":
                reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
            elif method == "UMAP":
                reducer = umap.UMAP(n_components=2, random_state=42)
            else:
                raise ValueError(f"Unknown method: {method}")

            return reducer.fit_transform(embeddings)

        return reduce_dimensions(method, embeddings, n_neighbors)

    if method == "HessianLLE":
        n_neighbors = st.slider("Select n_neighbors for HessianLLE", min_value=6, max_value=30, value=10, step=1)
    else:
        n_neighbors = None

    # Save and load reduced embeddings to/from JSON
    cache_dir = f"cache/{method}"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{text_hash}_{n_neighbors}.json" if n_neighbors else f"{text_hash}.json")

    if os.path.exists(cache_file):
        with open(cache_file, "r") as f:
            reduced_embeddings = np.array(json.load(f))
        st.sidebar.write("Loaded reduced embeddings from cache.")
    else:
        reduced_embeddings = reduce_dimensions_with_cache(method, selected_embeddings, n_neighbors, text_hash)
        with open(cache_file, "w") as f:
            json.dump(reduced_embeddings.tolist(), f)
        st.sidebar.write("Calculated and cached reduced embeddings.")

    labels = dog_labels + ["Unrelated"] * len(unrelated_indices)
    tokens = [merged_tokens[idx] for idx in selected_indices]
    colors = {
        "Category 1": "darkred",
        "Category 2": "red",
        "Category 3": "orange",
        "Category 4": "lightcoral",
        "Category 5": "coral",
        "Unrelated": "blue"
    }
    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "Label": labels,
        "Token": tokens
    })

    fig = px.scatter(
        df[df["Label"] == "Unrelated"],
        x="x",
        y="y",
        color="Label",
        color_discrete_map=colors,
        hover_data={"Token": True},
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        title=f"{method} Visualization of Dog-Related and Unrelated Tokens"
    )
    fig.update_traces(marker=dict(opacity=0.8)) 

    for category in ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5"]:
        fig.add_trace(px.scatter(
            df[df["Label"] == category],
            x="x",
            y="y",
            color="Label",
            color_discrete_map=colors,
            hover_data={"Token": True},
            labels={"x": "Dimension 1", "y": "Dimension 2"}
        ).data[0])

    st.plotly_chart(fig)

    # Display category counts in the sidebar
    for i, count in enumerate(category_counts):
        st.sidebar.write(f"Category {i+1} Tokens: {count}")

