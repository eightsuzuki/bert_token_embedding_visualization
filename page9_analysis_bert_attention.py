import base64
import os

import tempfile

import io
import torch
import pickle
import joblib
import hashlib
import numpy as np
import seaborn as sns
import sklearn.manifold
from matplotlib import cm
import matplotlib.pyplot as plt  # 追加
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

###############################################################################
# 0. カスタム BERT クラス（内部から q, k, v を抽出するためのフック付き）
###############################################################################
st.markdown("## 0. カスタム BERT クラス")
st.latex(r'''
Q_i = W_Q \cdot h_i,\quad
K_i = W_K \cdot h_i,\quad
V_i = W_V \cdot h_i
''')
st.latex(r'''
\alpha_{ij} = \frac{\exp\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right)}
{\sum_{l=1}^{n}\exp\left(\frac{Q_i \cdot K_l}{\sqrt{d_k}}\right)}
''')
st.markdown("このクラスでは、forward hook を用いて各レイヤーの q, k, v を抽出します。")

class BertModelWithQKV(BertModel):
    """
    BertModel を継承し、各レイヤーの q, k, v を forward hook で抽出するクラス。
    """
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
        q = self.q_layers.get(layer_idx, None)
        k = self.k_layers.get(layer_idx, None)
        v = self.v_layers.get(layer_idx, None)
        return q, k, v

###############################################################################
# 1. Attention マップ抽出・キャッシュ用関数群
###############################################################################
st.markdown("## 1. Attention マップの抽出とキャッシュ")
st.latex(r'''
\alpha_{ij} = \frac{\exp\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right)}
{\sum_{l=1}^{n}\exp\left(\frac{Q_i \cdot K_l}{\sqrt{d_k}}\right)}
''')
st.markdown("""
Attention マップは、形状 
\((\text{num\_layers}, \text{num\_heads}, \text{seq\_len}, \text{seq\_len})\)
となり、後の解析に利用され、キャッシュされます。
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
        attns = np.stack(attns, axis=0)  # (num_layers, num_heads, seq_len, seq_len)
        data = {"tokens": tokens, "attns": attns}
        joblib.dump(data, cache_path)
        st.write(f"Saved attention data to cache: {cache_path}")
        return data

st.markdown("### Jensen–Shannon Divergence の計算")
st.latex(r'''
JS(p \parallel q) = \frac{1}{2}KL\left(p \parallel \frac{p+q}{2}\right)
+ \frac{1}{2}KL\left(q \parallel \frac{p+q}{2}\right)
''')
st.markdown("""
ここで、\(KL\) は Kullback–Leibler Divergence です。  
各ヘッド間でこの値を計算し、類似度を距離行列として表現します。
""")

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    kl_p = np.sum(np.where(p != 0, p * np.log(p / m), 0))
    kl_q = np.sum(np.where(q != 0, q * np.log(q / m), 0))
    return 0.5 * (kl_p + kl_q)

def compute_js_divergences(attns):
    num_layers, num_heads, seq_len, _ = attns.shape
    total_heads = num_layers * num_heads
    attn_flat = attns.reshape(total_heads, seq_len, seq_len)
    js_div = np.zeros((total_heads, total_heads))
    for i in range(total_heads):
        for j in range(total_heads):
            js_vals = []
            for t in range(seq_len):
                p = attn_flat[i, t, :]
                q = attn_flat[j, t, :]
                js_vals.append(jensen_shannon_divergence(p, q))
            js_div[i, j] = np.mean(js_vals)
    return js_div

def get_js_divergences(attns, text_hash, cache_dir="./cache/attn/"):
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"head_distances_{text_hash}.pkl")
    if os.path.isfile(cache_path):
        st.write(f"Loading head distances from cache: {cache_path}")
        return joblib.load(cache_path)
    else:
        st.write("Computing Jensen–Shannon divergences between heads...")
        js_div = compute_js_divergences(attns)
        joblib.dump(js_div, cache_path)
        st.write(f"Saved head distances to cache: {cache_path}")
        return js_div

###############################################################################
# 2. 次元削減および抽出用関数（BERT内部の v を利用）
###############################################################################
st.markdown("## 2. 次元削減と内部表現の抽出")
st.latex(r'''
V_i = W_V \cdot h_i
''')
st.latex(r'''
\text{Reduced Coordinates} = \text{Reducer}(V)
''')
st.markdown("""
これにより、各ヘッドの内部表現が視覚化可能となります。
""")

def preprocess_text(text: str, tokenizer, max_length=128):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return inputs

def split_heads(tensor: torch.Tensor, num_heads: int):
    batch_size, seq_len, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor

def fit_umap_or_pca_per_head(texts, tokenizer, model, num_layers=12, num_heads=12, use_umap=True, cache_dir="./cache/qkv/"):
    import hashlib
    joined_text = " ".join(texts)
    text_hash = hashlib.md5(joined_text.encode("utf-8")).hexdigest()[:8]
    cache_file = f"reducers_umap_{text_hash}.pkl" if use_umap else f"reducers_pca_{text_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.isfile(cache_path):
        st.write(f"=== Loading reducers from local cache: {cache_path} ===")
        return joblib.load(cache_path)
    last_layer_idx = num_layers - 1
    head_accumulated_data = [[] for _ in range(num_heads)]
    for text in texts:
        inputs = preprocess_text(text, tokenizer)
        with torch.no_grad():
            model(**inputs)
        q, k, v = model.get_qkv_from_layer(last_layer_idx)
        v_split = split_heads(v, num_heads=num_heads)[0]
        for head_idx in range(num_heads):
            head_v = v_split[head_idx]
            head_accumulated_data[head_idx].append(head_v.detach().cpu().numpy())
    reducers = []
    for head_idx in range(num_heads):
        data_head = np.concatenate(head_accumulated_data[head_idx], axis=0)
        if use_umap:
            reducer = umap.UMAP(n_components=2, random_state=42, n_jobs=1)
        else:
            reducer = PCA(n_components=2)
        reducer.fit(data_head)
        reducers.append(reducer)
    st.write("=== Done fitting UMAP/PCA per head (using last-layer data only) ===")
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(reducers, cache_path)
    st.write(f"=== Saved reducers to local cache: {cache_path} ===")
    return reducers

def extract_plot_data(text: str, tokenizer, model, reducers_per_head, num_layers=12, num_heads=12, max_length=20, cache_dir="./cache/qkv/"):
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    cache_file = f"extracted_data_{text_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)
    if os.path.isfile(cache_path):
        st.write(f"=== Loading extracted data from local cache: {cache_path} ===")
        return joblib.load(cache_path)
    inputs = preprocess_text(text, tokenizer, max_length=max_length)
    with torch.no_grad():
        model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    seq_len = len(tokens)
    data_dict = {}
    x_vals_per_head = [[] for _ in range(num_heads)]
    y_vals_per_head = [[] for _ in range(num_heads)]
    for layer_idx in range(num_layers):
        q, k, v = model.get_qkv_from_layer(layer_idx)
        q_split = split_heads(q, num_heads=num_heads)[0]
        k_split = split_heads(k, num_heads=num_heads)[0]
        v_split = split_heads(v, num_heads=num_heads)[0]
        for head_idx in range(num_heads):
            head_q = q_split[head_idx]
            head_k = k_split[head_idx]
            head_v = v_split[head_idx].detach().cpu().numpy()
            attn_scores = (head_q @ head_k.transpose(-2, -1)) * (1.0 / np.sqrt(head_q.size(-1)))
            attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1).detach().cpu().numpy()
            reduced = reducers_per_head[head_idx].transform(head_v)
            data_dict[(layer_idx, head_idx)] = {"reduced": reduced, "attn_scores": attn_scores}
            x_vals_per_head[head_idx].append(reduced[:, 0])
            y_vals_per_head[head_idx].append(reduced[:, 1])
    x_min_per_head = {}
    x_max_per_head = {}
    y_min_per_head = {}
    y_max_per_head = {}
    for head_idx in range(num_heads):
        all_x = np.concatenate(x_vals_per_head[head_idx])
        all_y = np.concatenate(y_vals_per_head[head_idx])
        x_min_per_head[head_idx] = all_x.min()
        x_max_per_head[head_idx] = all_x.max()
        y_min_per_head[head_idx] = all_y.min()
        y_max_per_head[head_idx] = all_y.max()
    data_to_save = (data_dict, x_min_per_head, x_max_per_head, y_min_per_head, y_max_per_head, tokens, seq_len)
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(data_to_save, cache_path)
    st.write(f"=== Saved extracted data to local cache: {cache_path} ===")
    return data_to_save

###############################################################################
# 3. Plotly 用解析・可視化用関数
###############################################################################
def get_data_points(head_data):
    xs, ys, avgs = [], [], []
    # 固定で 12 レイヤー × 12 ヘッドとする
    for layer in range(12):
        for head in range(12):
            ys.append(head_data[layer, head])
            xs.append(layer + 1)
        avgs.append(head_data[layer].mean())
    return xs, ys, avgs



def plot_avg_attention_plotly(avg_attns):
    """
    Figure 2: 各ヘッドの平均 Attention をグループごとに Plotly でプロットする。
    """
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        subplot_titles=["[CLS], [SEP], punct", "other -> [SEP] & [SEP] -> [SEP]", "next, prev, current token"])
    group1 = [("cls", "#e74c3c", "[CLS]"), ("sep", "#3498db", "[SEP]"), ("punct", "#9b59b6", ". or ,")]
    group2 = [("rest_sep", "#3498db", "other -> [SEP]"), ("sep_sep", "#59d98e", "[SEP] -> [SEP]")]
    group3 = [("left", "#e74c3c", "next token"), ("right", "#3498db", "prev token"), ("self", "#9b59b6", "current token")]
    for key, color, label in group1:
        xs, ys, avgs = get_data_points(avg_attns[key])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=color, size=6), name=label), row=1, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,13)), y=avgs, mode="lines", line=dict(color=color), name=label+" avg"), row=1, col=1)
    for key, color, label in group2:
        xs, ys, avgs = get_data_points(avg_attns[key])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=color, size=6), name=label), row=2, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,13)), y=avgs, mode="lines", line=dict(color=color), name=label+" avg"), row=2, col=1)
    for key, color, label in group3:
        xs, ys, avgs = get_data_points(avg_attns[key])
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color=color, size=6), name=label), row=3, col=1)
        fig.add_trace(go.Scatter(x=list(range(1,13)), y=avgs, mode="lines", line=dict(color=color), name=label+" avg"), row=3, col=1)
    fig.update_layout(height=800, width=600, title_text="Average Attention per Head by Layer", xaxis_title="Layer", yaxis_title="Avg. Attention")
    return fig

def plot_entropy_plotly(entropies, entropies_cls, uniform_attn_entropy):
    """
    Figure 4: Attention エントロピーのプロットを Plotly で表示する。
    """
    xs, es, avg_es = get_data_points(entropies)
    xs, es_cls, avg_es_cls = get_data_points(entropies_cls)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=["BERT Heads", "BERT Heads from [CLS]"])
    fig.add_trace(go.Scatter(x=xs, y=es, mode="markers", marker=dict(color="#3498db", size=5), name="BERT Heads"), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(1,13)), y=avg_es, mode="lines", line=dict(color="#3498db"), name="BERT Heads Avg"), row=1, col=1)
    fig.add_trace(go.Scatter(x=xs, y=es_cls, mode="markers", marker=dict(color="#e74c3c", size=5), name="BERT Heads [CLS]"), row=2, col=1)
    fig.add_trace(go.Scatter(x=list(range(1,13)), y=avg_es_cls, mode="lines", line=dict(color="#e74c3c"), name="BERT Heads [CLS] Avg"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[1, 12], y=[uniform_attn_entropy, uniform_attn_entropy],
                             mode="lines", line=dict(color="black", dash="dash"), name="Uniform Attention"), row=1, col=1)
    fig.add_annotation(x=7, y=uniform_attn_entropy - 0.45, text="uniform attention", showarrow=False, row=1, col=1)
    fig.update_layout(height=600, width=600, title_text="Attention Entropy per Layer", xaxis_title="Layer", yaxis_title="Avg. Attention Entropy (nats)")
    return fig


def display_attn_info(example, heads):
    """
    Streamlit 上に、選択された各ヘッドの情報を表示します。
    使用するトークンは、example["tokens"] の先頭から最初の [PAD] まで（[PAD] がなければ全トークン）。
    また、該当ヘッドの attention マトリックスの shape も表示します。
    """
    st.markdown("#### 選択されたヘッドの情報")
    tokens_full = example["tokens"]
    if "[PAD]" in tokens_full:
        pad_index = tokens_full.index("[PAD]")
    else:
        pad_index = len(tokens_full)
    tokens = tokens_full[:pad_index]
    for (layer, head_idx) in heads:
        st.write(f"Processing Layer {layer+1} - Head {head_idx+1} ...")
        attn_matrix = example["attns"][layer][head_idx][:pad_index, :pad_index]
        st.write("使用するトークン（[PAD] まで）:", tokens)
        st.write("Attention マトリックスの shape:", attn_matrix.shape)
        st.write("---")
        
def plot_attn_igraph_svg(example, heads, attn_sep=3, pad=0.2,
                         token_range=15, scale=10):
    """
    指定した複数の層・ヘッドに対する attention マップ（各ヘッドは
    example["attns"][layer][head] の先頭から [PAD] までの部分）を igraph を用いて
    SVG 形式で描画し、その SVG テキストを返します。

    各ヘッドでは、元の Matplotlib の実装と同様に、
      ・左側にトークン（x = xoffset）
      ・右側に同じトークン（x = xoffset + width）
    を配置し、両列間に attention 重みが weight_threshold を超える場合にエッジを追加します。
    エッジ幅はその重みに scale 倍して設定され、さらにエッジの色は attention 値に応じて青色の濃さを変化させます。
    """
    # 固定の幅（各ヘッド内で左右の列間隔の基準となる値）
    width = 400
    # ノード間隔とサイズの調整
    node_spacing = 10   # 各ノードの縦間隔
    node_size = 15      # ノードサイズの基準
    
    st.write("Attention グラフの描画を開始します...")
    all_vertices = []
    all_edges = []
    all_edge_weights = []
    vertex_idx = 0  # 全体の頂点番号カウンタ

    # 使用するトークンは、example["tokens"] の先頭から最初の [PAD] まで
    tokens_full = example["tokens"]
    if "[PAD]" in tokens_full:
        pad_index = tokens_full.index("[PAD]")
    else:
        pad_index = len(tokens_full)
    tokens = tokens_full[:pad_index]
    
    # 進捗表示：各対象ヘッドの処理開始
    for ei, (layer, head_idx) in enumerate(heads):
        st.write(f"Processing Layer {layer+1} - Head {head_idx+1} ...")
        # xoffset: ヘッド毎の横方向オフセット = ヘッドインデックス × (width × attn_sep)
        xoffset = ei * width * attn_sep
        yoffset = 1  # 各ヘッドの基準 y 座標（固定値）
        try:
            attn_matrix = example["attns"][layer][head_idx][:pad_index, :pad_index]
        except Exception as e:
            st.error(f"Error extracting attention matrix for Layer {layer+1} - Head {head_idx+1}: {e}")
            continue
        attn_matrix = np.array(attn_matrix)
        # ゼロ除算を避けるため、各行の合計が 0 の場合は 1 に置換
        row_sums = attn_matrix.sum(axis=-1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        attn_matrix /= row_sums
        
        n = len(tokens)  # 使用するトークン数
        left_indices = []
        right_indices = []
        
        # ノードの並び順を逆にするため、各ノードの y 座標は下記のように計算する
        # y = yoffset - (n - 1 - i) * node_spacing  とすることで、i=0 のとき上部に、i=n-1 のとき yoffset となる
        for i in range(0, n):
            x = xoffset
            y = yoffset - (n - 1 - i) * node_spacing
            all_vertices.append({"label": tokens[i], "x": x, "y": y, "size": node_size})
            left_indices.append(vertex_idx)
            vertex_idx += 1
        
        for i in range(0, n):
            x = xoffset + width
            y = yoffset - (n - 1 - i) * node_spacing
            all_vertices.append({"label": tokens[i], "x": x, "y": y, "size": node_size})
            right_indices.append(vertex_idx)
            vertex_idx += 1
        
        
        
        # エッジの追加（左側から右側へ）
        for li, i in enumerate(range(0, n)):
            for lj, j in enumerate(range(0, n)):
                w = attn_matrix[i, j]
                # 左右ノードの座標の差がほぼゼロならエッジ追加をスキップ
                src_coord = np.array([xoffset, yoffset - (n - i) * node_spacing])
                tgt_coord = np.array([xoffset + width, yoffset - (n - j) * node_spacing])
                if np.linalg.norm(src_coord - tgt_coord) < 1e-6:
                    continue
                all_edges.append((left_indices[li], right_indices[lj]))
                all_edge_weights.append(w)
    
    total_vertices = len(all_vertices)
    st.write(f"全頂点数: {total_vertices}, エッジ数: {len(all_edges)}")
    g = ig.Graph(directed=False)
    g.add_vertices(total_vertices)
    g.vs["label"] = [v["label"] for v in all_vertices]
    layout = [(v["x"], v["y"]) for v in all_vertices]
    g.add_edges(all_edges)
    g.es["weight"] = all_edge_weights
    g.es["width"] = [w * scale for w in all_edge_weights]
    # エッジ色は、attention の値に応じて青色の透明度を変更
    edge_colors = []
    for w in all_edge_weights:
        alpha = min(w * 2, 1.0)
        edge_colors.append(f"rgba(0, 0, 255, {alpha})")
    g.es["color"] = edge_colors
    
    # グラフの描画領域の計算（ここでは全体の bbox を縮小サイズに設定）
    graph_width = 200  # 例: 幅 200px
    graph_height = 200  # 例: 高さ 200px
    visual_style = {
        "vertex_size": node_size,       # ノードサイズ
        "vertex_color": "lightblue",      # ノードの背景色
        "vertex_label": g.vs["label"],     # ノードラベル
        "vertex_label_size": 5,             # ラベルの文字サイズ
        "layout": layout,                    # ノードの座標リスト
        "bbox": (graph_width, graph_height), # 描画領域
        "margin": 20,                        # 余白
        "edge_width": 1,         # エッジ幅
        "edge_color": g.es["color"],         # エッジ色
        "edge_curved": 0.0,                  # 直線描画
    }
    
    with tempfile.NamedTemporaryFile(suffix=".svg", delete=False, mode="w", encoding="utf-8") as tmp_file:
        tmp_filename = tmp_file.name
    ig.plot(g, target=tmp_filename, **visual_style)
    with open(tmp_filename, "r", encoding="utf-8") as f:
        svg_str = f.read()
    os.remove(tmp_filename)
    # Base64 エンコードしてデータ URL を作成
    encoded_svg = base64.b64encode(svg_str.encode("utf-8")).decode("utf-8")
    data_url = f"data:image/svg+xml;base64,{encoded_svg}"
    return data_url


###############################################################################
# 6. Clustering Attention Heads (Section 6, Figure 6)
###############################################################################
@st.cache_data(show_spinner=True)
def cached_mds(js_divergences):
    mds = sklearn.manifold.MDS(
        metric=True, n_init=5, n_jobs=4, eps=1e-10,
        max_iter=1000, dissimilarity="precomputed", random_state=42
    )
    pts_flat = mds.fit_transform(js_divergences)  # shape (144, 2)
    pts = pts_flat.reshape((12, 12, 2))
    return pts


def plot_cluster_heads(js_divergences, avg_attns, entropies):
    """
    Section 6 の実装：各注意ヘッド間の Jensen–Shannon Divergence に基づく距離行列
    を多次元尺度法 (MDS) で2次元に埋め込み、各ヘッドの行動を Plotly で可視化します。
    """
    # 閾値の定義
    ENTROPY_THRESHOLD = 3.8
    POSITION_THRESHOLD = 0.5
    SPECIAL_TOKEN_THRESHOLD = 0.6
    # 言語学的行動のヘッド（例）
    LINGUISTIC_HEADS = {
        (4, 3): "Coreference",
        (7, 10): "Determiner",
        (7, 9): "Direct object",
        (8, 5): "Object of prep.",
        (3, 9): "Passive auxiliary",
        (6, 5): "Possessive",
    }
    # Plotly 用の色とマーカーの定義
    colors = {
         "attend to next": "#e74c3c",
         "attend to prev": "#3498db",
         "attend broadly": "#f39c12",
         "attend to [CLS]": "#9b59b6",
         "attend to [SEP]": "#59d98e",
         "attend to . and ,": "#159d82",
         "others": "#95a5a6"
    }
    markers = {
         "attend to next": "triangle-right",
         "attend to prev": "triangle-left",
         "attend broadly": "triangle-up",
         "attend to [CLS]": "star",
         "attend to [SEP]": "diamond",
         "attend to . and ,": "square",
         "others": "circle"
    }

    # キャッシュ済みの MDS 計算
    pts = cached_mds(js_divergences)  # shape: (12, 12, 2)

    # Prepare behavioral clusters (grouping by attention behavior)
    behavioral_data = {}
    for layer in range(12):
        for head in range(12):
            x, y = pts[layer, head]
            group = ""
            marker_symbol = markers["others"]
            color_val = colors["others"]
            # 条件を順次チェック（後の条件が上書き）
            if avg_attns["right"][layer, head] > POSITION_THRESHOLD:
                group = "attend to next"
                marker_symbol = markers["attend to next"]
                color_val = colors["attend to next"]
            if avg_attns["left"][layer, head] > POSITION_THRESHOLD:
                group = "attend to prev"
                marker_symbol = markers["attend to prev"]
                color_val = colors["attend to prev"]
            if entropies[layer, head] > ENTROPY_THRESHOLD:
                group = "attend broadly"
                marker_symbol = markers["attend broadly"]
                color_val = colors["attend broadly"]
            if avg_attns["cls"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
                group = "attend to [CLS]"
                marker_symbol = markers["attend to [CLS]"]
                color_val = colors["attend to [CLS]"]
            if avg_attns["sep"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
                group = "attend to [SEP]"
                marker_symbol = markers["attend to [SEP]"]
                color_val = colors["attend to [SEP]"]
            if avg_attns["punct"][layer, head] > SPECIAL_TOKEN_THRESHOLD:
                group = "attend to . and ,"
                marker_symbol = markers["attend to . and ,"]
                color_val = colors["attend to . and ,"]
            # override if linguistic head
            if (layer, head) in LINGUISTIC_HEADS:
                group = LINGUISTIC_HEADS[(layer, head)]
                marker_symbol = "x"
                color_val = "#000000"
            if group == "":
                group = "others"
            if group not in behavioral_data:
                behavioral_data[group] = {"x": [], "y": [], "marker": marker_symbol, "color": color_val}
            behavioral_data[group]["x"].append(x)
            behavioral_data[group]["y"].append(y)

    # Prepare layer-colored clusters (group by layer)
    layer_data = {}
    colormap = cm.seismic(np.linspace(0, 1.0, 12))
    for layer in range(12):
        layer_label = f"Layer {layer+1}"
        layer_data[layer_label] = {"x": [], "y": []}
        for head in range(12):
            x, y = pts[layer, head]
            layer_data[layer_label]["x"].append(x)
            layer_data[layer_label]["y"].append(y)

    # Create Plotly subplots (2 rows: behavioral, layer)
    fig = make_subplots(rows=2, cols=1,
                        subplot_titles=["Behavioral Clusters", "Colored by Layer"])

    # Behavioral clusters (row 1)
    for group, data_group in behavioral_data.items():
        fig.add_trace(go.Scatter(
            x=data_group["x"],
            y=data_group["y"],
            mode="markers",
            marker=dict(symbol=data_group["marker"],
                        color=data_group["color"],
                        size=10),
            name=group
        ), row=1, col=1)

    # Layer-colored clusters (row 2)
    for layer_label, data_layer in layer_data.items():
        layer_num = int(layer_label.split()[1]) - 1
        hex_color = mcolors.to_hex(colormap[layer_num])
        fig.add_trace(go.Scatter(
            x=data_layer["x"],
            y=data_layer["y"],
            mode="markers",
            marker=dict(symbol="circle", color=hex_color, size=10),
            name=layer_label
        ), row=2, col=1)

    fig.update_layout(height=800, width=600, title_text="Embedded BERT Attention Heads (Plotly)")
    return fig

###############################################################################
# 4. Plotly 用プロットユーティリティ関数（続き）
###############################################################################
def plot_individual_layer_and_head_plotly(data_dict, x_min_per_head, x_max_per_head, y_min_per_head, y_max_per_head, tokens, seq_len, selected_layer, selected_head, selected_token_idx):
    """
    個別のレイヤー・ヘッドの次元削減結果を Plotly で表示する。
    """
    reduced_selected = data_dict[(selected_layer, selected_head)]["reduced"]
    attn_matrix = data_dict[(selected_layer, selected_head)]["attn_scores"]
    selected_attn = attn_matrix[selected_token_idx, :]
    sizes = selected_attn * 1500
    colors = ["blue"] * seq_len
    colors[selected_token_idx] = "red"
    fig = go.Figure()
    for i in range(seq_len):
        fig.add_trace(go.Scatter(
            x=[reduced_selected[i, 0]],
            y=[reduced_selected[i, 1]],
            mode="markers+text",
            marker=dict(size=sizes[i], color=colors[i], opacity=0.5),
            text=[tokens[i]],
            textposition="top center",
            showlegend=False
        ))
    fig.update_layout(title=f"Layer {selected_layer}, Head {selected_head}",
                      xaxis_title="Component 1",
                      yaxis_title="Component 2",
                      xaxis_range=[x_min_per_head[selected_head], x_max_per_head[selected_head]],
                      yaxis_range=[y_min_per_head[selected_head], y_max_per_head[selected_head]])
    return fig

def animate_selected_head_across_layers_plotly(data_dict, x_min_per_head, x_max_per_head, y_min_per_head, y_max_per_head, tokens, seq_len, selected_head, selected_token_idx=0, num_layers=12, min_size=0):
    import pandas as pd
    rows = []
    for layer_idx in range(num_layers):
        reduced = data_dict[(layer_idx, selected_head)]["reduced"]
        attn_matrix = data_dict[(layer_idx, selected_head)]["attn_scores"]
        selected_attn = attn_matrix[selected_token_idx, :]
        sizes = selected_attn * 100
        sizes = np.maximum(sizes, min_size)
        for i in range(seq_len):
            label = "Selected" if i == selected_token_idx else "Other"
            rows.append({
                "x": reduced[i, 0],
                "y": reduced[i, 1],
                "Layer": layer_idx,
                "Token": tokens[i],
                "Label": label,
                "Size": sizes[i] + 5
            })
    df_combined = pd.DataFrame(rows)
    color_map = {"Selected": "red", "Other": "blue"}
    global_x_min = min(x_min_per_head.values())
    global_x_max = max(x_max_per_head.values())
    global_y_min = min(y_min_per_head.values())
    global_y_max = max(y_max_per_head.values())
    fig_animated = px.scatter(
        df_combined,
        x="x", y="y",
        color="Label",
        animation_frame="Layer",
        hover_name="Token",
        size="Size",
        range_x=[global_x_min, global_x_max],
        range_y=[global_y_min, global_y_max],
        color_discrete_map=color_map,
        title=f"Plotly Animated (Head {selected_head}), Token={tokens[selected_token_idx]}",
        opacity=0.5
    )
    return fig_animated



###############################################################################
# 9. 可視化の実行（Plotly 版と新規のクラスタリング図）
###############################################################################
def render_page():
    sns.set_style("darkgrid")
    st.title("General BERT Attention Analysis")
    
    st.markdown("""
    このページでは、論文 *"What Does BERT Look At? An Analysis of BERT’s Attention"* の Section 3 および Section 6 の理論に基づいて、  
    BERT の注意（Attention）マップの一般的なパターンを解析します。
    """)
    st.latex(r'''
    \alpha_{ij} = \frac{\exp\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right)}
    {\sum_{l=1}^{n}\exp\left(\frac{Q_i \cdot K_l}{\sqrt{d_k}}\right)}
    ''')
    st.latex(r'''
    JS(p \parallel q) = \frac{1}{2}KL\left(p \parallel \frac{p+q}{2}\right)
    + \frac{1}{2}KL\left(q \parallel \frac{p+q}{2}\right)
    ''')

    ###############################################################################
    # 1. モデル・トークナイザの準備と Attention マップの抽出
    ###############################################################################
    st.markdown("### 1. モデルと Attention マップの抽出")
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    bert_model = BertModel.from_pretrained(model_name, config=config)
    bert_model.eval()
    
    text_to_plot = st.text_area("Attention マップを抽出するテキストを入力してください", 
                                 "My dog is cute. He likes play running.")
    
    data = get_attention_data(text_to_plot, tokenizer, bert_model, max_length=128, cache_dir="./cache/attn/")
    tokens = data["tokens"]
    attns = data["attns"]
    seq_len = attns.shape[-1]
    st.write(f"抽出されたトークン数: {seq_len}")
    
    num_layers, num_heads, _, _ = attns.shape
    
    text_hash = hashlib.md5(text_to_plot.encode("utf-8")).hexdigest()[:8]
    js_divergences = get_js_divergences(attns, text_hash, cache_dir="./cache/attn/")
    
    ###############################################################################
    # 2. 平均 Attention の計算
    ###############################################################################
    st.markdown("### 2. 平均 Attention の計算")
    st.latex(r'''
    \text{AvgAttn}^{[SEP]} = \frac{1}{N_{[SEP]}} \sum_{i,j} \alpha_{ij} \cdot M_{ij}^{[SEP]}
    ''')
    avg_attns = {k: np.zeros((num_layers, num_heads)) for k in [
        "self", "right", "left", "sep", "sep_sep", "rest_sep", "cls", "punct"
    ]}
    n_docs = 1
    for tokens_doc, attns_doc in [(tokens, attns)]:
        n_tokens = attns_doc.shape[-1]
        seps = np.zeros(n_tokens)
        clss = np.zeros(n_tokens)
        puncts = np.zeros(n_tokens)
        for pos, token in enumerate(tokens_doc):
            if token == "[SEP]":
                seps[pos] = 1
            if token == "[CLS]":
                clss[pos] = 1
            if token == "." or token == ",":
                puncts[pos] = 1
        sep_seps = np.outer(seps, seps)
        rest_seps = np.outer(1 - seps, seps)
        selectors = {
            "self": np.eye(n_tokens),
            "right": np.eye(n_tokens, k=1),
            "left": np.eye(n_tokens, k=-1),
            "sep": np.tile(seps, (n_tokens, 1)),
            "sep_sep": sep_seps,
            "rest_sep": rest_seps,
            "cls": np.tile(clss, (n_tokens, 1)),
            "punct": np.tile(puncts, (n_tokens, 1)),
        }
        for key, selector in selectors.items():
            if key == "sep_sep":
                denom = 2
            elif key == "rest_sep":
                denom = n_tokens - 2 if n_tokens > 2 else n_tokens
            else:
                denom = n_tokens
            avg_attns[key] += ((attns_doc * selector[np.newaxis, ...]).sum(-1).sum(-1)) / (n_docs * denom)
    
    ###############################################################################
    # 3. Attention エントロピーの計算
    ###############################################################################
    st.markdown("### 3. Attention エントロピーの計算")
    st.latex(r'''
    H(\alpha_i) = -\sum_{j=1}^{n} \alpha_{ij}\log\alpha_{ij}
    ''')
    uniform_attn_entropy = 0
    entropies = np.zeros((num_layers, num_heads))
    entropies_cls = np.zeros((num_layers, num_heads))
    for tokens_doc, attns_doc in [(tokens, attns)]:
        attns_doc = 0.9999 * attns_doc + (0.0001 / attns_doc.shape[-1])
        uniform_attn_entropy -= np.log(1.0 / attns_doc.shape[-1])
        entropies -= (attns_doc * np.log(attns_doc)).sum(-1).mean(-1)
        entropies_cls -= (attns_doc * np.log(attns_doc))[:, :, 0].sum(-1)
    uniform_attn_entropy /= n_docs
    entropies /= n_docs
    entropies_cls /= n_docs
    
    ###############################################################################
    # 4. Plotly 用プロットユーティリティ関数（以下は既存の関数）
    ###############################################################################
    st.markdown("### 4. Plotly 用プロットユーティリティ関数")
    BLACK = "k"
    GREEN = "#59d98e"
    SEA = "#159d82"
    BLUE = "#3498db"
    PURPLE = "#9b59b6"
    GREY = "#95a5a6"
    RED = "#e74c3c"
    ORANGE = "#f39c12"
    
    ###############################################################################
    # 9. 可視化の実行（Plotly 版とクラスタリング図）
    ###############################################################################
    st.markdown("### 9. 可視化の実行")

    # ----- Figure 1 の説明 -----
    st.markdown("#### Figure 1: Attention マップの例")
    st.markdown("この図は、各 attention ヘッドが入力トークン間でどのような注意重み \(\\alpha_{ij}\\) を生成しているかを示します。")
    st.latex(r'''
    \alpha_{ij} = \frac{\exp\left(\frac{Q_i \cdot K_j}{\sqrt{d_k}}\right)}
    {\sum_{l=1}^{n}\exp\left(\frac{Q_i \cdot K_l}{\sqrt{d_k}}\right)}
    ''')
    st.markdown("線の太さが各重みの大きさに比例しており、実際の attention の分布を視覚化します。")
    selected_layer = st.number_input("選択レイヤー", min_value=0, max_value=num_layers-1, value=2, step=1)
    selected_head = st.number_input("選択ヘッド", min_value=0, max_value=num_heads-1, value=0, step=1)
    selected_heads = [(selected_layer, selected_head)]
    sns.set_style("darkgrid")
    
    # Streamlit上に各ヘッドの情報を表示
    if st.button("Show Attention Map (igraph SVG) for 選択ヘッド"):
        # import plot_attn_igraph
        # plot_attn_igraph.display_attn_info(data, selected_heads)
        with st.spinner("Attention マップを描画中..."):
            svg_data_url = plot_attn_igraph_svg(data, selected_heads,attn_sep=3, 
                                           pad=0.2, token_range=15, scale=10)
        st.image(svg_data_url, caption="Attention Map (igraph)", use_container_width=True)


    # ----- Figure 2 の説明 -----
    st.markdown("#### Figure 2: 平均 Attention プロット")
    st.markdown("この図は、各 attention ヘッドが特定のトークン（[CLS]、[SEP]、句読点等）に対してどれだけ注意を向けているかの平均値を示します。")
    st.latex(r'''
    \text{AvgAttn}^{[X]} = \frac{1}{N_{[X]}} \sum_{i,j} \alpha_{ij} \cdot M_{ij}^{[X]}
    ''')
    st.markdown("ここで、\(M_{ij}^{[X]}\) はトークンタイプ \([X]\) のマスクです。")
    if st.button("Show Clustering Attention Heads (Figure 2)"):
        st.markdown("#### (Figure 2) 平均 Attention プロット")
        fig_avg = plot_avg_attention_plotly(avg_attns)
        st.plotly_chart(fig_avg)

    # ----- Figure 4 の説明 -----
    st.markdown("#### Figure 4: Attention エントロピーのプロット")
    st.markdown("この図は、各 attention ヘッドの注意分布のエントロピー \(H(\\alpha_i)\) を示しています。")
    st.latex(r'''
    H(\alpha_i) = -\sum_{j=1}^{n} \alpha_{ij}\log\alpha_{ij}
    ''')
    st.markdown("エントロピーが高いほど、注意が広範囲に分散していることを意味します。")
    if st.button("Show Clustering Attention Heads (Figure 4)"):
        st.markdown("#### (Figure 4) Attention エントロピーのプロット")
        fig_entropy = plot_entropy_plotly(entropies, entropies_cls, uniform_attn_entropy)
        st.plotly_chart(fig_entropy)

    # ----- Figure 6 の説明 -----
    st.markdown("#### Figure 6: Attention Heads のクラスタリング")
    st.markdown("この図は、各 attention ヘッド間の類似性を Jensen–Shannon Divergence を用いて計算し、")
    st.latex(r'''
    D(H_i, H_j) = \sum_{\text{token}\in\text{data}} JS\Bigl(H_i(\text{token}) \parallel H_j(\text{token})\Bigr)
    ''')
    st.markdown("と定義した距離に基づいて、")
    st.latex(r'''
    \|\mathbf{p}_i - \mathbf{p}_j\|_2 \approx D(H_i, H_j)
    ''')
    st.markdown("とした2次元の埋め込みを、多次元尺度法 (MDS) により求めています。さらに、各ヘッドの平均注意値により、")
    st.latex(r'''
    \text{AvgAttn}_{\text{right}} > 0.5,\quad \text{AvgAttn}_{\text{left}} > 0.5,\quad H(\alpha) > 3.8,\quad \text{AvgAttn}_{\text{cls}} > 0.6,\quad \text{AvgAttn}_{\text{sep}} > 0.6,\quad \text{AvgAttn}_{\text{punct}} > 0.6
    ''')
    st.markdown("などの条件に基づいて、各ヘッドの行動を分類・色分けしています。")
    if st.button("Show Clustering Attention Heads (Figure 6)"):
        fig_cluster = plot_cluster_heads(js_divergences, avg_attns, entropies)
        st.plotly_chart(fig_cluster)
    
if __name__ == "__main__":
    render_page()
