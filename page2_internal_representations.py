import os
import logging
import streamlit as st
import numpy as np
import pandas as pd
import hashlib
import json
from transformers import BertModel, BertTokenizer
from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap, LocallyLinearEmbedding, TSNE
import umap
import plotly.express as px

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# ページ 2: 内部表現
def render_page(text):
    """
    指定されたテキストを入力として受け取り、BERTモデルとトークナイザーを使用して埋め込みを生成し、キャッシュに保存または読み込みを行います。
    また、トークンと埋め込みをマージして表示します。
    Args:
        text (str): 入力テキスト。
        model (BertModel): BERTモデル。
        tokenizer (BertTokenizer): BERTトークナイザー。
        tokens_text (list): トークン化されたテキストのリスト。
    Returns:
        None
    """
    st.text_area("Input Text", text, height=300)

    # モデルとトークナイザーのロード
    def load_model_and_tokenizer():
        st.write("Loading model and tokenizer...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
        st.write("Model and tokenizer loaded.")
        return tokenizer, model

    tokenizer, model = load_model_and_tokenizer()
    cache_dir = "./cache/internal"
    
    # テキストを読み込み、トークン化して埋め込みを保存
    def process_and_save_embeddings():
        text = ""

        os.makedirs(cache_dir, exist_ok=True)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        cache_file = os.path.join(cache_dir, f"{text_hash}_layer_0_embeddings.json")
        # キャッシュが存在する場合は読み込む
        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as file:
                data = json.load(file)
                text = data["text"]
                tokens_text = data["tokens_text"]
                # embeddings = np.array(data["layer_embeddings"])
                st.write(f"キャッシュから埋め込みを読み込みました: {cache_file}")
                logging.info(f"Loaded embeddings from cache file: {cache_file}")

            return text, tokens_text, embeddings, text_hash

        with open("input_text.txt", "r", encoding="utf-8") as input_file:
            text = input_file.read()
            tokens_text = []
            layer_embeddings = [[] for _ in range(model.config.num_hidden_layers + 1)]
            st.write(f"Now Token")
            for i in range(0, len(text), 512):
                st.write("{i*512} ~ {i*512+512}:")
                chunk = text[i:i+512]
                tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
                outputs = model(**tokens)
                
                # 中間層の隠れ層の埋め込みを取得し、結合
                hidden_states = outputs.hidden_states
                for layer_idx, state in enumerate(hidden_states):
                    layer_embeddings[layer_idx].append(state.squeeze(0).detach().numpy())

            for layer_idx in range(len(layer_embeddings)):
                layer_embeddings[layer_idx] = np.concatenate(layer_embeddings[layer_idx], axis=0)
                layer_cache_file = os.path.join(cache_dir, f"{text_hash}_layer_{layer_idx}_embeddings.json")
                layer_data = {
                    "text": text,
                    "tokens_text": tokens_text,
                    "layer_embeddings": layer_embeddings[layer_idx].tolist()
                }
                with open(layer_cache_file, "w", encoding="utf-8") as layer_file:
                    json.dump(layer_data, layer_file)
                logging.info(f"Generated and saved layer {layer_idx} embeddings to cache file: {layer_cache_file}")
                st.write(f"Layer {layer_idx} embeddings saved.")
                st.write(f"Layer {layer_idx} embeddings shape: {layer_embeddings[layer_idx].shape}")

        data = {
            "text": text,
            "tokens_text": tokens_text,
            "embeddings": layer_embeddings[-1].tolist()  # Save the final layer embeddings
        }
        with open(cache_file, "w", encoding="utf-8") as file:
            json.dump(data, file)

        st.write(f"埋め込みをBERTから生成し、\n{cache_file}\n に保存されました")
        logging.info(f"Generated and saved embeddings to cache file: {cache_file}")

        st.write(f"BERT embeddings:", layer_embeddings[-1].shape)
        logging.info(f"Text length: {len(text)}, Tokens length: {len(tokens_text)}, Embeddings shape: {layer_embeddings[-1].shape}")
        return text, tokens_text, layer_embeddings[-1], text_hash

    text, tokens_text, embeddings, text_hash = process_and_save_embeddings()

    # 文字数とトークン数を表示
    st.write(f"テキストの文字数: {len(text)}")
    st.write(f"トークン数: {len(tokens_text)}")
    
    def reduce_dimensions_with_tsne(layer_cache_file, layer_idx):
        st.write("t-SNEで次元削減中...")
        
        with open(layer_cache_file, "r", encoding="utf-8") as file:
            data = json.load(file)
            embeddings = np.array(data["layer_embeddings"])
            text = data["text"]
            tokens_text = data["tokens_text"]
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        tsne_cache_dir = "./cache/internal/tsne"
        os.makedirs(tsne_cache_dir, exist_ok=True)
        tsne_cache_file = os.path.join(tsne_cache_dir, f"{text_hash}__layer_{layer_idx}_tsne_embeddings.json")

        tsne_data = {
            "text": text,
            "tokens_text": tokens_text,
            "embeddings_2d": embeddings_2d.tolist()
        }
        with open(tsne_cache_file, "w", encoding="utf-8") as file:
            json.dump(tsne_data, file)

        st.write(f"t-SNEで次元削減した埋め込みを\n{tsne_cache_file}\nに保存しました")
        st.write(f"t-SNE embeddings shape: {embeddings_2d.shape}")

        return embeddings_2d

    for layer_idx in range(model.config.num_hidden_layers + 1):
        tsne_cache_dir = "./cache/internal/tsne"
        tsne_cache_file = os.path.join(tsne_cache_dir, f"{text_hash}__layer_{layer_idx}_tsne_embeddings.json")

        if os.path.exists(tsne_cache_file):
            with open(tsne_cache_file, "r", encoding="utf-8") as file:
                tsne_data = json.load(file)
                embeddings_2d = np.array(tsne_data["embeddings_2d"])
        else:
            layer_cache_file = os.path.join(cache_dir, f"{text_hash}_layer_{layer_idx}_embeddings.json")
            embeddings_2d = reduce_dimensions_with_tsne(layer_cache_file, layer_idx)
