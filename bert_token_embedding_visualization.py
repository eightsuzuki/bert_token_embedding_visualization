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
    ["attention LRP flow", "qkv attention mapping", "attention map", "文脈構造の表現", "次元削減", "内部表現", "文字選択"]
)

if page == "attention map":
    import page5_attention_map
    page5_attention_map.render_page()
    st.stop()

if page == "qkv attention mapping":
    import page6_qkv_attention_mapping
    page6_qkv_attention_mapping.render_page()
    st.stop()
    
if page == "attention LRP flow":
    import page7_attention_LRP_flow
    page7_attention_LRP_flow.render_page()
    st.stop()
    
# モデルとトークナイザーのロード
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

if page == "文脈構造の表現":
    import page4_context_structure_representation
    page4_context_structure_representation.render_page(tokenizer, model)
    st.stop()
 
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
                # textの長さに基づいて、512文字ごとにループを回す
                chunk = text[i:i+512]
                # textから512文字のチャンクを取得する
                tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
                # チャンクをトークン化し、PyTorchテンソルとして返す。必要に応じてトランケーションとパディングを行う
                tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
                # トークンIDを対応するトークンに変換し、tokens_textリストに追加する
                outputs = model(**tokens)
                # トークンをモデルに入力し、出力を取得する
                embeddings.append(outputs.last_hidden_state.squeeze(0).detach().numpy())
                # モデルの出力から最後の隠れ層の状態を取得し、NumPy配列に変換してembeddingsリストに追加する

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

# ページ遷移
if page == "次元削減":

    # 文字数とトークン数を表示
    st.write(f"テキストの文字数: {len(text)}")
    st.write(f"トークン数: {len(tokens_text)}")

    import page1_dimensionality_reduction
    page1_dimensionality_reduction.render_page(text, tokens_text, embeddings)

elif page == "内部表現":
    import page2_internal_representations
    page2_internal_representations.render_page(text)

elif page == "文字選択":
    import page3_dimensionality_reduction_select_word
    page3_dimensionality_reduction_select_word.render_page(text, tokens_text, embeddings)
