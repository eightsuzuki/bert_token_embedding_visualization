import os
import tempfile
import io
import base64
import torch
import pickle
import joblib
import hashlib
import numpy as np
import seaborn as sns
import sklearn.manifold
from matplotlib import cm
import matplotlib.pyplot as plt
import streamlit as st

from io import BytesIO
import igraph as ig
import matplotlib.colors as mcolors
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.decomposition import PCA
import umap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# --- spaCy の読み込み（gold 依存解析用） ---
# spaCy とモデル en_core_web_sm がインストールされていることを確認してください。
import spacy
nlp = spacy.load("en_core_web_sm")

###############################################################################
# 0. カスタム BERT クラス（Attention マップ抽出用）
###############################################################################
st.markdown("## 0. カスタム BERT クラス")
st.markdown("""
このクラスは、BERT の各レイヤー・ヘッドからクエリ、キー、バリュー (q, k, v) を forward hook を用いて抽出します。  
※本実験（UAS 評価）では attention マップの抽出に直接 `BertModel` を用いますが、内部解析時に利用可能です。  
（対応：論文 Section 2: Background）
""")
class BertModelWithQKV(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.q_layers = {}
        self.k_layers = {}
        self.v_layers = {}
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )
    def _get_qkv_hook(self, layer_idx: int):
        def hook(module, module_input, module_output):
            hidden_states = module_input[0]
            query_layer = module.query(hidden_states)
            key_layer   = module.key(hidden_states)
            value_layer = module.value(hidden_states)
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook
    def get_qkv_from_layer(self, layer_idx: int):
        return (self.q_layers.get(layer_idx, None),
                self.k_layers.get(layer_idx, None),
                self.v_layers.get(layer_idx, None))

###############################################################################
# 1. Attention マップ抽出
###############################################################################
st.markdown("## 1. Attention マップの抽出")
st.markdown("""
ここでは、入力テキストから BERT の attention マップを抽出します。  
attention マップは形状 (num_layers, num_heads, seq_len, seq_len) を持ち、  
論文 Section 3 (Surface-Level Patterns in Attention) の解析に利用されます。
""")
def get_attention_data(text, tokenizer, model, max_length=128, cache_dir="./cache/attn/"):
    os.makedirs(cache_dir, exist_ok=True)
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    cache_path = os.path.join(cache_dir, f"attn_{text_hash}.pkl")
    if os.path.isfile(cache_path):
        st.write(f"Loading attention data from cache: {cache_path}")
        return joblib.load(cache_path)
    else:
        inputs = tokenizer(text, return_tensors="pt", max_length=max_length,
                           truncation=True, padding="max_length")
        with torch.no_grad():
            outputs = model(**inputs)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        # outputs.attentions はタプル。各要素は (batch, num_heads, seq_len, seq_len)
        attns = [attn.squeeze(0).detach().cpu().numpy() for attn in outputs.attentions]
        attns = np.stack(attns, axis=0)
        data = {"tokens": tokens, "attns": attns}
        joblib.dump(data, cache_path)
        st.write(f"Saved attention data to cache: {cache_path}")
        return data

###############################################################################
# UAS 評価: Attention ヘッドによる依存先予測 vs. spaCy Gold
###############################################################################
st.markdown("## UAS 評価")
st.markdown("""
このセクションは、論文 Section 5 (Probing Classifiers) の評価に対応します。  
入力文に対して、BERT の指定した attention ヘッド（Layer L, Head H）の注意分布から各単語の依存先（head）を予測し、  
spaCy による gold 依存解析結果と比較して UAS (Unlabeled Attachment Score) を計算します。
""")
def evaluate_attention_head_on_sentence(text, selected_layer, selected_head):
    # 1. spaCy による gold 依存解析
    doc = nlp(text)
    gold_heads = [token.head.i for token in doc]  # 各トークンの gold head インデックス（0始まり）
    spacy_tokens = [token.text for token in doc]
    
    # 2. BERT による attention マップの取得
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    bert_model = BertModel.from_pretrained(model_name, config=config)
    bert_model.eval()
    data = get_attention_data(text, tokenizer, bert_model, max_length=128, cache_dir="./cache/attn/")
    
    # BERT のトークンは [CLS] と [SEP] が追加されているため除去
    bert_tokens = data["tokens"][1:-1]
    attn_matrix = data["attns"][selected_layer][selected_head]
    # 最初と最後の行/列 ([CLS] と [SEP]) を除去
    attn_matrix = attn_matrix[1:-1, 1:-1]
    
    # spaCy のトークンと BERT のトークン数が一致するか確認
    if len(bert_tokens) != len(spacy_tokens):
        st.error(f"Token 数が一致しません：BERT: {len(bert_tokens)} vs spaCy: {len(spacy_tokens)}")
        st.write("BERT tokens:", bert_tokens)
        st.write("spaCy tokens:", spacy_tokens)
        return None

    # 3. 各単語について、自己注目を除外して最大の attention 値を持つ単語を依存先として予測
    n = len(bert_tokens)
    for i in range(n):
        attn_matrix[i, i] = -np.inf  # 自己注目を除外
    predicted_heads = np.argmax(attn_matrix, axis=-1)
    
    # 4. UAS の計算
    correct = 0
    for i in range(n):
        # spaCy の gold head は 0 始まり。BERT のトークンは [CLS] 除去済みなので、spaCy のインデックスとの差分を補正
        if predicted_heads[i] == gold_heads[i] - 1:
            correct += 1
    uas = 100 * correct / n
    return uas

###############################################################################
# render_page: Streamlit 全体の実行例
###############################################################################
def render_page():
    sns.set_style("darkgrid")
    st.title("BERT Attention UAS 評価")
    
    st.markdown("""
    このアプリでは、入力文に対して BERT の attention マップから  
    指定した attention ヘッドの依存先予測を行い、spaCy による gold 依存解析結果と比較して  
    UAS (Unlabeled Attachment Score) を評価します。
    
    **対応論文セクション：**  
    - Section 3: Surface-Level Patterns in Attention  
    - Section 5: Probing Classifiers  
    """)
    
    st.markdown("---")
    st.markdown("### 入力文の依存解析 (spaCy Gold)")
    text_to_evaluate = st.text_area("依存解析する文を入力してください", 
                                    "My dog is cute. He likes play running.")
    
    # spaCy による gold 解析結果の表示
    doc = nlp(text_to_evaluate)
    spacy_tokens = [token.text for token in doc]
    spacy_heads = [token.head.i for token in doc]
    st.write("spaCy 解析トークン:", spacy_tokens)
    st.write("spaCy Gold Head インデックス:", spacy_heads)
    
    st.markdown("---")
    st.markdown("### Attention ヘッドによる依存先予測と UAS 評価")
    num_layers = 12
    num_heads = 12
    selected_layer = st.number_input("選択レイヤー", min_value=0, max_value=num_layers-1, value=2, step=1)
    selected_head = st.number_input("選択ヘッド", min_value=0, max_value=num_heads-1, value=0, step=1)
    
    if st.button("UAS を評価する"):
        with st.spinner("Attention マップを取得中..."):
            uas = evaluate_attention_head_on_sentence(text_to_evaluate, selected_layer, selected_head)
        if uas is not None:
            st.write(f"Layer {selected_layer+1}, Head {selected_head+1} による UAS: {uas:.1f}%")
    
if __name__ == "__main__":
    render_page()
