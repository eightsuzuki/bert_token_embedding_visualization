import streamlit as st
from transformers import BertTokenizer, BertModel
import numpy as np
# import pandas as pd
# import plotly.express as px
import os
import json
import hashlib

# タイトル
st.title("トークン埋め込みの分析と可視化")

# ページ管理
page = st.sidebar.selectbox(
    "ページを選択してください",
    ["次元削減", "内部表現"]
)

# モデルとトークナイザーのロード
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# テキストを読み込み、トークン化して埋め込みを保存
@st.cache_data
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

    return text, tokens_text, embeddings

text, tokens_text, embeddings = process_and_save_embeddings()

# 文字数とトークン数を表示
st.write(f"テキストの文字数: {len(text)}")
st.write(f"トークン数: {len(tokens_text)}")

# ページ遷移
if page == "次元削減":
    import page1_dimensionality_reduction
    page1_dimensionality_reduction.render_page(text, tokens_text, embeddings)

elif page == "内部表現":
    import page2_internal_representations
    page2_internal_representations.render_page(text, tokens_text, embeddings, model)
