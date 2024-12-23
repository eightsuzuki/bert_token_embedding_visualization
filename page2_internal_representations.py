import os
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'

import streamlit as st
import numpy as np
from sklearn.decomposition import FastICA
import umap
import pandas as pd
import plotly.express as px
import hashlib
import json

# ページ 1: 次元削減技術の比較
def render_page(text, model, tokenizer, tokens_text):
    st.text_area("Input Text", text, height=300)

    # テキストを読み込み、トークン化して埋め込みを保存
    def process_and_save_embeddings():
        text = ""

        cache_dir = "./cache/BERT_embeddings"
        os.makedirs(cache_dir, exist_ok=True)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file = os.path.join(cache_dir, f"{text_hash}_embeddings.json")
        
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                text = data["text"]
                tokens_text = data["tokens_text"]
                embeddings = np.array(data["embeddings"])
                st.write(f"以前に保存された\n{cache_file}\n を使用")
        else:
            with open("input_text.txt", "r", encoding="utf-8") as input_file:
                text = input_file.read()
                tokens_text = []
                embeddings = []

                for i in range(0, len(text), 512):
                    chunk = text[i:i+512]
                    tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                    tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
                    outputs = model(**tokens)
                    embeddings.append(outputs.last_hidden_state.squeeze(0).detach().numpy())

                embeddings = np.concatenate(embeddings, axis=0)
                
            data = {
                "text": text,
                "tokens_text": tokens_text,
                "embeddings": embeddings.tolist()
            }
            with open(cache_file, "w", encoding="utf-8") as file:
                json.dump(data, file)   

            st.write(f"埋め込みをBERTから生成し、\n{cache_file}\n に保存されました")

        st.write(f"BERT embeddings:", embeddings.shape)
        return text, tokens_text, embeddings

    text, tokens_text, embeddings = process_and_save_embeddings()
    
    # トークンと埋め込みをキャッシュして取得
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

    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
    merged_tokens, merged_embeddings = get_cached_embeddings(text_hash)

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

    def reduce_dimensions_with_cache(method, embeddings, text_hash):
        def reduce_dimensions(method, embeddings):
            embeddings = np.array(embeddings)

            if method == "ICA":
                reducer = FastICA(n_components=2)
            else:
                raise ValueError(f"Unknown method: {method}")

            return reducer.fit_transform(embeddings)

        return reduce_dimensions(method, embeddings)

    method = "ICA"
    reduced_embeddings = reduce_dimensions_with_cache(method, selected_embeddings, text_hash)

    labels = dog_labels + ["Unrelated"] * len(unrelated_indices)
    tokens = [merged_tokens[idx] for idx in selected_indices]
    
    if not (len(reduced_embeddings) == len(labels) == len(tokens)):
        st.error(f"Error: The lengths of reduced_embeddings ({len(reduced_embeddings)}), labels ({len(labels)}), and tokens ({len(tokens)}) do not match.")
        return

    num_points = st.sidebar.slider("Number of points to plot", min_value=100, max_value=len(reduced_embeddings), value=5000, step=100)
    random_indices = np.random.choice(len(reduced_embeddings), num_points, replace=False)
    reduced_embeddings = reduced_embeddings[random_indices]
    labels = [labels[i] for i in random_indices]
    tokens = [tokens[i] for i in random_indices]
    colors = {
        "Category 1": "darkred",
        "Category 2": "red",
        "Category 3": "orange",
        "Category 4": "lightcoral",
        "Category 5": "coral",
        "Category 6": "purple",
        "Unrelated": "blue"
    }
    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "Label": labels,
        "Token": tokens
    })

    df["OriginalIndex"] = df.index
    df = df.sort_values(by="OriginalIndex")

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
    fig.update_traces(marker=dict(opacity=0.3)) 

    for category in ["Category 1", "Category 2", "Category 3", "Category 4", "Category 5", "Category 6"]:
        category_df = df[df["Label"] == category]
        if not category_df.empty:
            fig.add_trace(px.scatter(
                category_df,
                x="x",
                y="y",
                color="Label",
                color_discrete_map=colors,
                hover_data={"Token": True},
                labels={"x": "Dimension 1", "y": "Dimension 2"}
            ).data[0])

    st.plotly_chart(fig)

    for i, count in enumerate(category_counts):
        st.sidebar.write(f"Category {i+1} Tokens: {count}")
        # 他の隠れ層の埋め込みを取り出す
        def get_layer_embeddings(_model, _tokenizer, text, layer_indices):
            tokens = _tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            outputs = _model(**tokens, output_hidden_states=True)
            hidden_states = outputs.hidden_states

            layer_embeddings = {}
            for idx in layer_indices:
                layer_embeddings[idx] = hidden_states[idx].squeeze(0).detach().numpy()

            return layer_embeddings

        layer_indices = st.sidebar.multiselect("Select Layers", list(range(1, 13)), default=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], key=f"layer_multiselect_{i}")
        layer_embeddings = get_layer_embeddings(model, tokenizer, text, layer_indices)

        # 各層の埋め込みを次元削減
        reduced_layer_embeddings = {}
        for idx in layer_indices:
            reduced_layer_embeddings[idx] = reduce_dimensions_with_cache(method, layer_embeddings[idx], text_hash + f"_layer_{idx}")

        # 各層の次元削減された埋め込みを表示
        for idx in layer_indices:
            st.write(f"Layer {idx} Embeddings:", reduced_layer_embeddings[idx].shape)
            df_layer = pd.DataFrame({
                "x": reduced_layer_embeddings[idx][:, 0],
                "y": reduced_layer_embeddings[idx][:, 1],
                "Token": tokens_text[:len(reduced_layer_embeddings[idx])]
            })

            fig_layer = px.scatter(
                df_layer,
                x="x",
                y="y",
                hover_data={"Token": True},
                labels={"x": "Dimension 1", "y": "Dimension 2"},
                title=f"{method} Visualization of Layer {idx} Embeddings"
            )
            st.plotly_chart(fig_layer)
