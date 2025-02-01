import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import hashlib
from sklearn.decomposition import PCA

from typing import Dict, List
from transformers import BertModel, BertConfig, BertTokenizer


###############################################################################
# 1. BERT の内部から Q, K, V を取得するためのクラス (フックを使う)
###############################################################################
class BertModelWithQKV(BertModel):
    """
    BertModel を継承し、forward hook を使って各レイヤーの Q, K, V を保存します。
    具体的には、BertSelfAttention の部分にフックを仕込んで、
    モデル推論 (forward) 時に Q, K, V を取り出せるようにします。
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        # 各レイヤーの Q, K, V を保持する辞書: layer_idx -> Tensor
        self.q_layers: Dict[int, torch.Tensor] = {}
        self.k_layers: Dict[int, torch.Tensor] = {}
        self.v_layers: Dict[int, torch.Tensor] = {}

        # BertEncoder 内の各レイヤー (Transformer Block) にフックを登録
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx: int):
        """
        BertSelfAttention の forward 直前に呼び出されるフック。
        hidden_states から Q, K, V を計算して保存します。
        """
        def hook(module, module_input, module_output):
            # module_input[0] が (batch_size, seq_len, hidden_dim) の hidden_states
            hidden_states = module_input[0]
            query_layer = module.query(hidden_states)  # Q を計算
            key_layer   = module.key(hidden_states)    # K を計算
            value_layer = module.value(hidden_states)  # V を計算

            # CPU に移し、辞書に格納 (勾配は不要なので detach())
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()

        return hook

    def get_qkv_from_layer(self, layer_idx: int):
        """
        フォワード後に格納された layer_idx 層目の Q, K, V を返す。
        """
        q = self.q_layers.get(layer_idx, None)
        k = self.k_layers.get(layer_idx, None)
        v = self.v_layers.get(layer_idx, None)
        return q, k, v


###############################################################################
# 2. モデル読み込み & 前処理ユーティリティ
###############################################################################
def load_bert_with_qkv(model_name="bert-base-uncased") -> BertModelWithQKV:
    """
    HuggingFace Transformers から "bert-base-uncased" をロードし、
    BertModelWithQKV でラップして返します。
    """
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    model = BertModelWithQKV.from_pretrained(model_name, config=config)
    return model

def preprocess_text(text: str, tokenizer, max_length=128):
    """
    BERT 用の入力データ (input_ids, attention_mask, token_type_ids など) を作成し、
    PyTorch テンソルで返します。
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        max_length=max_length,
        truncation=True,
        padding="max_length",
    )
    return inputs

def split_heads(tensor: torch.Tensor, num_heads: int):
    """
    テンソル形状を (batch_size, seq_len, hidden_dim) から
    (batch_size, num_heads, seq_len, head_dim) に変換します。
    """
    batch_size, seq_len, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    # view & permute で次元を再配置
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor


###############################################################################
# 3. 最終レイヤーの V から UMAP または PCA を学習 (次元削減器を作成)
###############################################################################
def fit_umap_or_pca_on_last_layer(
    texts: List[str],
    tokenizer,
    model: BertModelWithQKV,
    last_layer_idx: int = 11,  # BERT-base は全12層 (0..11)
    num_heads: int = 12,
    use_umap: bool = True,
    cache_dir: str = "./cache/qkv/"
):
    """
    最終レイヤー (layer=11) の V をすべて集め、各ヘッドごとに UMAP あるいは PCA をフィットして
    2次元空間への変換器を学習します。
    -> こうすることで、各ヘッドのベクトルを 2D に投影するための軸を固定します。
    """
    os.makedirs(cache_dir, exist_ok=True)
    joined_text = " ".join(texts)
    # テキストリストからハッシュを作成 (キャッシュ用)
    text_hash = hashlib.md5(joined_text.encode("utf-8")).hexdigest()[:8]

    method_str = "umap" if use_umap else "pca"
    cache_file = f"reducers_{method_str}_lastlayer_{text_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)

    # すでに学習済みの次元削減器がある場合はロード
    if os.path.isfile(cache_path):
        print(f"Loading reducers from cache: {cache_path}")
        return joblib.load(cache_path)

    print("Fitting new reducers on the last layer's V ...")

    # ヘッドごとにデータを貯めるためのリスト
    head_accumulated_data = [[] for _ in range(num_heads)]

    # テキストごとにモデルを通して最終レイヤーの V を取得
    for text in texts:
        inputs = preprocess_text(text, tokenizer)
        with torch.no_grad():
            model(**inputs)

        # 最終レイヤーの V を取得
        _, _, v = model.get_qkv_from_layer(last_layer_idx)  # shape: (batch_size, seq_len, hidden_dim)
        # ヘッド次元に分割 (batch_size=1 を仮定)
        v_split = split_heads(v, num_heads=num_heads)[0]  # shape: (num_heads, seq_len, head_dim)

        # 各ヘッドごとにデータを蓄積
        for head_idx in range(num_heads):
            head_v = v_split[head_idx]  # shape: (seq_len, head_dim)
            head_accumulated_data[head_idx].append(head_v.numpy())

    # ヘッドごとに UMAP or PCA をフィット
    reducers = []
    for head_idx in range(num_heads):
        data_head = np.concatenate(head_accumulated_data[head_idx], axis=0)
        if use_umap:
            import umap
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            reducer = PCA(n_components=2)
        reducer.fit(data_head)
        reducers.append(reducer)

    joblib.dump(reducers, cache_path)
    print(f"Saved reducers to {cache_path}")
    return reducers


###############################################################################
# 4. Relevance / LRP を計算する関数
###############################################################################
def compute_relevance(
    attn_weights: np.ndarray,   # shape: (seq_len,) → 元々の ω_{ij} など
    v_prev: np.ndarray,         # shape: (seq_len, dim) → 前層の各トークンのベクトル
    v_i: np.ndarray,            # shape: (dim,) → 選択トークンの前層のベクトル
    v_i_prime: np.ndarray = None,  # shape: (dim,) → 更新後の選択トークンのベクトル (weighted_projection 用)
    base_rule: str = "raw",     # "raw", "weighted_distance", "weighted_projection"
    post_rule: str = "none"     # "none", "lrp_0", "lrp_epsilon", "lrp_gamma"
):
    """
    まず、以下の 3 つの基本ルールに従ってベースの relevance \(r_{ij}^{\rm base}\) を計算します：
    
      (1) raw: \(r_{ij}^{\rm base} = \omega_{ij}\)
      (2) weighted_distance: \(r_{ij}^{\rm base} \propto \omega_{ij}\,\|v_j - v_i\|\)
      (3) weighted_projection: \(r_{ij}^{\rm base} \propto \omega_{ij}\,\bigl|(v_j - v_i)\cdot (v'_i - v_i)\bigr|\)
    
    その後、もし post_rule が "none" でなければ、後処理として LRP のルールを適用します。
    具体的には、前層の各トークンの活性化 \(a_j\) を \( \|v_j\|_1 \) として、例えば LRP-0 では
    
    \[
      R_j = \frac{a_j\,r_{ij}^{\rm base}}{\sum_j a_j\,r_{ij}^{\rm base}}
    \]
    
    の形により再正規化します。
    """
    eps = 1e-12
    seq_len = len(attn_weights)
    
    # --- 基本ルールでベースの relevance を計算 ---
    if base_rule == "raw":
        r_base = attn_weights.copy()
    elif base_rule == "weighted_distance":
        dist = np.linalg.norm(v_prev - v_i, axis=1)  # (seq_len,)
        r_base = attn_weights * dist
    elif base_rule == "weighted_projection":
        if v_i_prime is None:
            r_base = attn_weights.copy()
        else:
            diff = v_prev - v_i            # (seq_len, dim)
            direction = v_i_prime - v_i      # (dim,)
            dotvals = np.einsum('md,d->m', diff, direction)  # (seq_len,)
            r_base = attn_weights * np.abs(dotvals)
    else:
        r_base = attn_weights.copy()
    
    # --- 後処理として LRP のルールを適用する場合 ---
    if post_rule == "none":
        r = r_base.copy()
    elif post_rule == "lrp_0":
        # LRP-0: \( r_j = \frac{a_j\,r^{base}_j}{\sum_j a_j\,r^{base}_j} \)
        a = np.linalg.norm(v_prev, ord=1, axis=1)  # a_j = ||v_j||_1
        numerator = a * r_base
        r = numerator / (np.sum(numerator) + eps)
    elif post_rule == "lrp_epsilon":
        epsilon = 1e-2  # 調整可能な定数
        a = np.linalg.norm(v_prev, ord=1, axis=1)
        numerator = a * r_base
        r = numerator / (epsilon + np.sum(numerator))
    elif post_rule == "lrp_gamma":
        gamma = 0.25  # 調整可能なパラメータ
        a = np.linalg.norm(v_prev, ord=1, axis=1)
        # softmax の出力は非負なので、実質 w⁺ = r_base, w⁻ = 0 として計算
        numerator = a * r_base
        r = numerator / (np.sum(numerator) + eps)
    else:
        r = r_base.copy()
    
    # 最終的に非負クリップ & 正規化
    r = np.clip(r, 0, None)
    sum_val = r.sum()
    if sum_val < eps:
        r = np.ones(seq_len) / seq_len
    else:
        r = r / sum_val
    
    return r


###############################################################################
# 5. 各レイヤーごとにアテンションで Value を合成し、2D 投影する関数
###############################################################################
def extract_plot_data_across_layers(
    text: str,
    tokenizer,
    model: BertModelWithQKV,
    reducers_per_head,
    num_layers: int = 12,
    num_heads: int = 12,
    max_length=20,
    cache_dir: str = "./cache/qkv/"
):
    """
    指定テキストに対して BERT を通し、
    各層で \( v^{(L)}_i = \sum_j \omega_{ij} \, v^{(L-1)}_j \) を計算します。
    [PAD] トークンは attention_mask を利用して除外します。
    その後、各ヘッドごとに 2D 投影 (UMAP または PCA) を行い、その結果を data_dict に格納します。
    
    data_dict[(layer, head)] = {
      "reduced": (valid_seq_len, 2),  → 2次元座標,
      "attn_scores": (valid_seq_len, valid_seq_len), → アテンション行列 ω_{ij},
      "v_prev": (valid_seq_len, hidden_dim) → 前層の v^{(L-1)}_j
    }
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    cache_file = f"extract_xlayer_{text_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)

    # キャッシュがあればロード
    if os.path.isfile(cache_path):
        print(f"Loading cross-layer data from cache: {cache_path}")
        return joblib.load(cache_path)

    # モデル推論
    inputs = preprocess_text(text, tokenizer, max_length=max_length)
    with torch.no_grad():
        model(**inputs)

    # attention_mask を用いて [PAD] を除外
    attention_mask = inputs["attention_mask"][0].bool()  # (seq_len,)
    valid_indices = attention_mask.nonzero().squeeze(1).tolist()

    # トークンリストも有効部分のみ抽出
    tokens_all = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    tokens = [tokens_all[i] for i in valid_indices]
    seq_len = len(tokens)

    # 各層の Q, K, V を辞書に格納（有効部分のみ抽出）
    q_layer_dict = {}
    k_layer_dict = {}
    v_layer_dict = {}
    for layer_idx in range(num_layers):
        q_full, k_full, v_full = model.get_qkv_from_layer(layer_idx)
        q_layer_dict[layer_idx] = q_full[0, valid_indices, :].unsqueeze(0)
        k_layer_dict[layer_idx] = k_full[0, valid_indices, :].unsqueeze(0)
        v_layer_dict[layer_idx] = v_full[0, valid_indices, :].unsqueeze(0)

    data_dict = {}
    x_vals_per_head = [[] for _ in range(num_heads)]
    y_vals_per_head = [[] for _ in range(num_heads)]

    for layer_idx in range(num_layers):
        if layer_idx == 0:
            v_prev = v_layer_dict[layer_idx]
            q = q_layer_dict[layer_idx]
            k = k_layer_dict[layer_idx]
        else:
            v_prev = v_layer_dict[layer_idx - 1]
            q = q_layer_dict[layer_idx - 1]
            k = k_layer_dict[layer_idx - 1]

        q_split = split_heads(q, num_heads=num_heads)[0]       # (num_heads, valid_seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]
        v_split_prev = split_heads(v_prev, num_heads=num_heads)[0]

        # アテンションスコアの計算
        attn_logits = torch.matmul(q_split, k_split.transpose(-2, -1))
        d_k = q_split.size(-1)
        attn_scores = torch.nn.functional.softmax(attn_logits / np.sqrt(d_k), dim=-1)

        for head_idx in range(num_heads):
            head_attn = attn_scores[head_idx]      # (valid_seq_len, valid_seq_len)
            head_v_prev = v_split_prev[head_idx]     # (valid_seq_len, head_dim)
            v_L = torch.matmul(head_attn, head_v_prev)  # (valid_seq_len, head_dim)

            reducer = reducers_per_head[head_idx]
            v_L_np = v_L.detach().cpu().numpy()
            reduced = reducer.transform(v_L_np - 1)  # (valid_seq_len, 2)

            data_dict[(layer_idx, head_idx)] = {
                "reduced": reduced,
                "attn_scores": head_attn.cpu().numpy(),
                "v_prev": head_v_prev.cpu().numpy()
            }
            x_vals_per_head[head_idx].append(reduced[:, 0])
            y_vals_per_head[head_idx].append(reduced[:, 1])

    # 各ヘッドごとのプロット範囲を計算
    x_min_per_head = {}
    x_max_per_head = {}
    y_min_per_head = {}
    y_max_per_head = {}
    for h in range(num_heads):
        if len(x_vals_per_head[h]) > 0:
            all_x = np.concatenate(x_vals_per_head[h])
            all_y = np.concatenate(y_vals_per_head[h])
        else:
            all_x = np.array([0])
            all_y = np.array([0])
        x_min_per_head[h] = all_x.min()
        x_max_per_head[h] = all_x.max()
        y_min_per_head[h] = all_y.min()
        y_max_per_head[h] = all_y.max()

    output_tuple = (
        data_dict,
        x_min_per_head,
        x_max_per_head,
        y_min_per_head,
        y_max_per_head,
        tokens,
        seq_len
    )
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(output_tuple, cache_path)
    print(f"Saved cross-layer data to {cache_path}")
    return output_tuple


###############################################################################
# 6. Plotly でアニメーション: Relevance / LRP の各ルールを円サイズに反映
###############################################################################
def animate_selected_head_across_layers_plotly_cross(
    data_dict,
    x_min_per_head,
    x_max_per_head,
    y_min_per_head,
    y_max_per_head,
    tokens,
    seq_len,
    selected_head,
    selected_token_idx,
    base_rule: str = "raw",      # "raw", "weighted_distance", "weighted_projection"
    post_rule: str = "none",      # "none", "lrp_0", "lrp_epsilon", "lrp_gamma"
    num_layers: int = 12,
    min_size: float = 5.0,
    margin_ratio: float = 0.1,
):
    """
    指定したヘッドにおいて、各層ごとに
    基本ルール (base_rule) で計算された relevance に対し、
    さらに後処理として LRP のルール (post_rule) を適用した結果を円サイズとして表示するアニメーションを作成します。
    """
    import pandas as pd
    import plotly.express as px

    rows = []
    for layer_idx in range(1, num_layers):
        key = (layer_idx, selected_head)
        if key not in data_dict:
            continue

        data_L = data_dict[key]
        reduced_L = data_L["reduced"]       # (valid_seq_len, 2)
        attn = data_L["attn_scores"]        # (valid_seq_len, valid_seq_len)
        v_prev = data_L["v_prev"]           # (valid_seq_len, hidden_dim)

        attn_sel = attn[selected_token_idx, :]  # (valid_seq_len,)
        v_i = v_prev[selected_token_idx]        # (hidden_dim,)
        v_i_prime = np.sum(v_prev * attn_sel[:, None], axis=0)

        # compute_relevance の呼び出しで base_rule と post_rule を渡す
        r = compute_relevance(
            attn_weights=attn_sel,
            v_prev=v_prev,
            v_i=v_i,
            v_i_prime=v_i_prime,
            base_rule=base_rule,
            post_rule=post_rule
        )

        data_prev_L = data_dict.get((layer_idx - 1, selected_head), None)
        if data_prev_L is None:
            continue
        reduced_prev_L = data_prev_L["reduced"]

        for j in range(seq_len):
            label = "SelectedToken" if j == selected_token_idx else "OtherToken"
            size_val = max(r[j] * 300, min_size)
            rows.append({
                "x": reduced_prev_L[j, 0],
                "y": reduced_prev_L[j, 1],
                "Layer": layer_idx,
                "Token": tokens[j],
                "Label": label,
                "Size": size_val,
                "Marker": "circle",
            })

        star_x = reduced_L[selected_token_idx, 0]
        star_y = reduced_L[selected_token_idx, 1]
        rows.append({
            "x": star_x,
            "y": star_y,
            "Layer": layer_idx,
            "Token": f"v(L={layer_idx}, i={selected_token_idx})",
            "Label": "NewVector",
            "Size": 20,
            "Marker": "star",
        })

    df = pd.DataFrame(rows)
    x_min = x_min_per_head[selected_head]
    x_max = x_max_per_head[selected_head]
    y_min = y_min_per_head[selected_head]
    y_max = y_max_per_head[selected_head]
    dx = x_max - x_min
    dy = y_max - y_min
    margin_x = margin_ratio * dx
    margin_y = margin_ratio * dy

    fig_animated = px.scatter(
        df,
        x="x",
        y="y",
        color="Label",
        animation_frame="Layer",
        hover_name="Token",
        size="Size",
        symbol="Marker",
        range_x=[x_min - margin_x, x_max + margin_x],
        range_y=[y_min - margin_y, y_max + margin_y],
        color_discrete_map={
            "SelectedToken": "red",
            "OtherToken": "blue",
            "NewVector": "gold",
        },
        title=f"Head={selected_head}, Base: {base_rule}, LRP: {post_rule}"
    )
    return fig_animated


###############################################################################
# 7. Streamlit アプリ (メイン)
###############################################################################
def render_page():
    st.title("Cross-Layer Weighted V Visualization for BERT (Relevance / LRP)")

    st.markdown("## BERT 内部のアテンション重みと Value ベクトルの更新可視化デモ")
    st.markdown("""
    以下では **BERT** の Self-Attention を題材に、
    **トークン間のアテンション重み** (\\(\omega_{ij}\\)) と
    **Value ベクトル** (\\(v_i\\)) の更新を可視化します。
    """)
    # 説明部分の例

    st.markdown("### 1. Attention Weight の定義")
    st.markdown("""
    BERT の各層において、入力トークン \(q_i\) と \(k_j\) の内積を計算し、スケーリングして softmax を適用することで、トークン間の注意重み \( \omega_{ij} \) を得ます。  
    このとき、各 \(i\) について \(\sum_j \omega_{ij} = 1\) かつ \(\omega_{ij} \ge 0\) となります。
    """)
    st.latex(r"\omega_{ij} = \mathrm{softmax}_j\!\Bigl(\frac{q_i \cdot k_j}{\sqrt{d_k}}\Bigr)")

    st.markdown("### 2. Value の更新")
    st.markdown("""
    各トークンの新しい Value \(v'_i\) は、前層の各トークンの Value \(v_j\) を注意重み \( \omega_{ij} \) で重み付けして合成することで得られます。  
    この操作は以下の式で表されます。
    """)
    st.latex(r"v'_i = \sum_j \omega_{ij}\,v_j")
    st.markdown("※ また、\(v'_i\) は \(v_i\) に \( \sum_j \omega_{ij}\,(v_j - v_i) \) を加えた形としても解釈できます。")

    st.markdown("### 3. Relevance / LRP の各ルール")
    st.markdown("""
    まず、以下の 3 つの基本ルールに従って、ベースとなる relevance \(r_{ij}^{\mathrm{base}}\) を計算します。
    """)
    st.latex(r"""
    \begin{aligned}
    (1)\;& r_{ij}^{\mathrm{base}} = \omega_{ij},\\[1mm]
    (2)\;& r_{ij}^{\mathrm{base}} \propto \omega_{ij}\,\|\,v_j - v_i\,\|,\\[1mm]
    (3)\;& r_{ij}^{\mathrm{base}} \propto \omega_{ij}\,\bigl|\,(v_j - v_i)\cdot(v'_i - v_i)\bigr|.
    \end{aligned}
    """)
    st.markdown("""
    その後、必要に応じて以下の LRP ルールを後処理として適用し、最終的な relevance \(R_j\) を計算します。  
    ここで、各トークンの活性化は \(a_j = \|v_j\|_1\) と仮定しています。
    """)
    st.latex(r"""
    \begin{aligned}
    \text{LRP-0:}\quad & R_j = \frac{a_j\,r_{ij}^{\mathrm{base}}}{\sum_j a_j\,r_{ij}^{\mathrm{base}}},\\[1mm]
    \text{LRP-}\epsilon:\quad & R_j = \frac{a_j\,r_{ij}^{\mathrm{base}}}{\epsilon + \sum_j a_j\,r_{ij}^{\mathrm{base}}},\\[1mm]
    \text{LRP-}\gamma:\quad & R_j = \frac{a_j\,r_{ij}^{\mathrm{base}}}{\sum_j a_j\,r_{ij}^{\mathrm{base}}}.
    \end{aligned}
    """)
    st.markdown("""
    なお、今回の実装では BERT の自己注意における注意重み \( \omega_{ij} \) は非負なため、  
    LRP-γ ルールは実質 LRP-0 と同じ形となります。  
    また、`post_rule` が "none" の場合は、基本ルールで計算された \(r_{ij}^{\mathrm{base}}\) をそのまま正規化して用います。
    """)


    # モデルのロード
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = load_bert_with_qkv(model_name)
    model.eval()

    num_layers = 12
    num_heads = 12
    use_umap = True

    texts_for_fitting = ["She was a teacher for forty years and her writing has appeared in journals and anthologies."]
    reducers_per_head = fit_umap_or_pca_on_last_layer(
        texts=texts_for_fitting,
        tokenizer=tokenizer,
        model=model,
        last_layer_idx=num_layers - 1,
        num_heads=num_heads,
        use_umap=use_umap,
        cache_dir="./cache/qkv/"
    )

    st.markdown("### デモ用テキストを入力")
    text_to_plot = st.text_area(
        "入力する文章 (英語)",
        "She was a teacher for forty years and her writing has appeared in journals and anthologies."
    )

    (data_dict, x_min_per_head, x_max_per_head,
     y_min_per_head, y_max_per_head,
     tokens, seq_len) = extract_plot_data_across_layers(
        text=text_to_plot,
        tokenizer=tokenizer,
        model=model,
        reducers_per_head=reducers_per_head,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=16,
        cache_dir="./cache/qkv/"
    )
    st.success("Cross-layer でのアテンション & Value 情報を取得しました！")

    st.markdown("### 可視化: Relevance / LRP の各ルールの比較")
    st.write("#### トークン一覧")
    st.write(tokens)

    token_options = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
    selected_token_str = st.selectbox("Relevance を計算する対象トークン (index: token)", token_options)
    selected_token_idx = int(selected_token_str.split(":")[0])
    selected_head = st.selectbox("可視化するヘッド (head_idx)", list(range(num_heads)))

    # まずベースとなるルールを選択
    base_rule = st.selectbox(
        "ベース Relevance ルールの選択",
        ["raw", "weighted_distance", "weighted_projection"],
        format_func=lambda x: {
            "raw": "① raw: r_{ij}^{base} = ω_{ij}",
            "weighted_distance": "② weighted_distance: r_{ij}^{base} ∝ ω_{ij} * ||v_j - v_i||",
            "weighted_projection": "③ weighted_projection: r_{ij}^{base} ∝ ω_{ij} * |(v_j - v_i)・(v'_i - v_i)|",
        }[x]
    )
    # その後、後処理として LRP ルールを選択（"none" で適用しない）
    post_rule = st.selectbox(
        "後処理として適用する LRP ルールの選択",
        ["none", "lrp_0", "lrp_epsilon", "lrp_gamma"],
        format_func=lambda x: {
            "none": "なし",
            "lrp_0": "④ LRP-0",
            "lrp_epsilon": "⑤ LRP-ε",
            "lrp_gamma": "⑥ LRP-γ",
        }[x]
    )

    if st.button("Plotly でアニメーションを表示"):
        fig_plotly = animate_selected_head_across_layers_plotly_cross(
            data_dict=data_dict,
            x_min_per_head=x_min_per_head,
            x_max_per_head=x_max_per_head,
            y_min_per_head=y_min_per_head,
            y_max_per_head=y_max_per_head,
            tokens=tokens,
            seq_len=seq_len,
            selected_head=selected_head,
            selected_token_idx=selected_token_idx,
            base_rule=base_rule,
            post_rule=post_rule,
            num_layers=num_layers
        )
        st.plotly_chart(fig_plotly)


if __name__ == "__main__":
    render_page()
