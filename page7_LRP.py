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
    BertModel を継承し、forward hook を使って各レイヤーの q, k, v を保存します。
    具体的には、BertSelfAttention の部分にフックを仕込んで、
    モデル推論 (forward) 時に Q, K, V を取り出せるようにします。
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)
        # 各レイヤーの q, k, v を保持する辞書: layer_idx -> Tensor
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
        フォワード後に格納された layer_idx 層目の q, k, v を返す。
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

        # 最終レイヤーの v を取得
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
# 4. Relevance を 3 種類の定義 (①,②,③) に従って計算する関数
###############################################################################
def compute_relevance(
    attn_weights: np.ndarray,  # shape: (seq_len,) => ω_{i j}
    v_prev: np.ndarray,        # shape: (seq_len, dim) => v_{(L-1), j} 一覧
    v_i: np.ndarray,           # shape: (dim,)        => v_{(L-1), i}
    v_i_prime: np.ndarray = None,  # shape: (dim,)    => v_{(L), i} = Σ_j [ ω_{i j} * v_j ] (+ skip)
    rule_type: str = "raw",    # "raw", "weighted_distance", "weighted_projection"
    use_skip_connection: bool = False
):
    """
    Relevance (r_{i j}) を計算する。
    ① raw               : r_{i j} = w_{i j}
    ② weighted_distance : r_{i j} ∝ w_{i j} * ||v_j - v_i||
    ③ weighted_projection : r_{i j} ∝ w_{i j} * |(v_j - v_i)・(v_i' - v_i)|

    Skip Connection がある場合は:
      v'_i = v_i + Σ_j [ ω_{i j} * v_j ]
    無い場合は:
      v'_i = Σ_j [ ω_{i j} * v_j ]

    最後に r_{i j} を正規化し、∑_j r_{i j} = 1, r_{i j} >= 0 となるようにする。
    """
    eps = 1e-12
    seq_len = len(attn_weights)

    # v_i_prime が None なら、ここで計算してもいいし、呼び出し元で計算済みなら渡してもよい
    if v_i_prime is None:
        if use_skip_connection:
            # v'_i = v_i + Σ_j [ ω_{i j} * v_j ]
            v_i_prime = v_i + np.sum(v_prev * attn_weights[:, None], axis=0)
        else:
            # v'_i = Σ_j [ ω_{i j} * v_j ]
            v_i_prime = np.sum(v_prev * attn_weights[:, None], axis=0)

    # ルールに応じて計算
    if rule_type == "raw":
        # ① 生のアテンション重みそのまま
        r = attn_weights.copy()

    elif rule_type == "weighted_distance":
        # ② アテンション重みにベクトル差のノルムを掛ける
        dist = np.linalg.norm(v_prev - v_i, axis=1)  # shape: (seq_len,)
        r = attn_weights * dist

    elif rule_type == "weighted_projection":
        # ③ アテンション重みに (v_j - v_i) と (v'_i - v_i) の内積の絶対値を掛ける
        diff = (v_prev - v_i)          # shape: (seq_len, dim)
        direction = (v_i_prime - v_i)  # shape: (dim,)
        dotvals = np.einsum('md,d->m', diff, direction)  # shape: (seq_len,)
        r = attn_weights * np.abs(dotvals)

    else:
        # それ以外は raw
        r = attn_weights.copy()

    # 非負クリップ & 正規化
    r = np.clip(r, 0, None)
    sum_val = r.sum()
    if sum_val < eps:
        r = np.ones(seq_len) / seq_len
    else:
        r = r / sum_val

    return r


###############################################################################
# 5. BERT の各レイヤーごとに ω_{i j} を計算して Value を合成 -> 2D 投影
#    Skip Connection ON/OFF を切り替え可能
###############################################################################
def extract_plot_data_across_layers(
    text: str,
    tokenizer,
    model: BertModelWithQKV,
    reducers_per_head,
    num_layers: int = 12,
    num_heads: int = 12,
    max_length=20,
    cache_dir: str = "./cache/qkv/",
    use_skip_connection: bool = False
):
    """
    指定テキストに対して BERT を通し、
    各レイヤー L(0..num_layers-1) の Q, K, V を取得して、
    v_{(L), i} = (skip ? v_{(L-1), i} + ) Σ_j [ ω_{i j} * v_{(L-1), j} ] を計算します。

    その上で、reducers_per_head[head_idx] によって 2D へ投影 (UMAP or PCA) した結果を
    data_dict に格納します。

    data_dict[(layer, head)] = {
      "reduced": (seq_len, 2),      # 2次元座標
      "attn_scores": (seq_len, seq_len), # アテンション行列 ω_{i j}
      "v_prev": (seq_len, hidden_dim)    # 前レイヤーの v_{(L-1), j}
    }

    Skip Connection は boolean 引数 use_skip_connection で切替。
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    cache_file = f"extract_xlayer_{text_hash}_{use_skip_connection}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)

    # キャッシュがあればロード
    if os.path.isfile(cache_path):
        print(f"Loading cross-layer data from cache: {cache_path}")
        return joblib.load(cache_path)

    # モデル推論
    inputs = preprocess_text(text, tokenizer, max_length=max_length)
    with torch.no_grad():
        model(**inputs)

    # トークナイズ結果を保存
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # [PAD] を除外するため、attention_mask またはトークン内容でフィルタリング
    valid_indices = [i for i, token in enumerate(tokens) if token != "[PAD]"]
    tokens = [tokens[i] for i in valid_indices]
    seq_len = len(valid_indices)

    # 全レイヤーの Q, K, V を辞書に格納
    q_layer_dict = {}
    k_layer_dict = {}
    v_layer_dict = {}
    for layer_idx in range(num_layers):
        q, k, v = model.get_qkv_from_layer(layer_idx)
        q_layer_dict[layer_idx] = q
        k_layer_dict[layer_idx] = k
        v_layer_dict[layer_idx] = v

    data_dict = {}
    x_vals_per_head = [[] for _ in range(num_heads)]
    y_vals_per_head = [[] for _ in range(num_heads)]

    for layer_idx in range(num_layers):
        if layer_idx == 0:
            # L=0 の場合
            v_prev = v_layer_dict[layer_idx]   # shape: (batch_size, seq_len, hidden_dim)
            q = q_layer_dict[layer_idx]
            k = k_layer_dict[layer_idx]
        else:
            # L>0 の場合
            v_prev = v_layer_dict[layer_idx - 1]
            q = q_layer_dict[layer_idx - 1]
            k = k_layer_dict[layer_idx - 1]

        # (batch=1 前提)
        q_split = split_heads(q, num_heads=num_heads)[0]          # (num_heads, full_seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]          # (num_heads, full_seq_len, head_dim)
        v_split_prev = split_heads(v_prev, num_heads=num_heads)[0]# (num_heads, full_seq_len, head_dim)

        attn_logits = torch.matmul(q_split, k_split.transpose(-2, -1))
        d_k = q_split.size(-1)
        attn_scores = attn_logits / np.sqrt(d_k)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

        for head_idx in range(num_heads):
            head_attn = attn_scores[head_idx]     # (full_seq_len, full_seq_len)
            head_v_prev = v_split_prev[head_idx]  # (full_seq_len, head_dim)

            # Attention の合成結果
            v_out_no_skip = torch.matmul(head_attn, head_v_prev)  # (full_seq_len, head_dim)

            # Skip Connection ON/OFF の切り替え
            if use_skip_connection:
                # BERT の Residual Connection (簡易版)
                v_out = head_v_prev + v_out_no_skip
            else:
                v_out = v_out_no_skip

            # 2次元に射影 (UMAP or PCA)
            v_out_np = v_out.detach().cpu().numpy()
            reduced = reducers_per_head[head_idx].transform(v_out_np)

            # [PAD] 部分のインデックス（valid_indices）だけを抽出
            reduced_valid = reduced[np.array(valid_indices)]
            head_attn_np = head_attn.cpu().numpy()
            attn_valid = head_attn_np[np.ix_(valid_indices, valid_indices)]
            head_v_prev_np = head_v_prev.cpu().numpy()
            v_prev_valid = head_v_prev_np[np.array(valid_indices)]

            data_dict[(layer_idx, head_idx)] = {
                "reduced": reduced_valid,
                "attn_scores": attn_valid,
                "v_prev": v_prev_valid
            }

            # x, y の min / max 記録用
            x_vals_per_head[head_idx].append(reduced_valid[:, 0])
            y_vals_per_head[head_idx].append(reduced_valid[:, 1])

    # ヘッドごとの x, y の min/max を取得 -> プロット範囲に利用
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
# 6. Plotly でアニメーション: 3 種類の Relevance を円サイズに反映
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
    rule_type: str = "raw",
    num_layers: int = 12,
    min_size: float = 5.0,
    margin_ratio: float = 0.1,
    use_skip_connection: bool = False,  # ← 追加
):
    """
    指定したヘッドを固定し、Layer=1..(num_layers-1) の変化をアニメーション表示。
    Relevance = r_{i j} を計算し、その値を円のサイズに利用して可視化する。

    Skip Connection = True の場合、v'_i = v_i + ∑ ω_{i j}v_j
    Skip Connection = False の場合、v'_i = ∑ ω_{i j}v_j
    """
    import pandas as pd
    import plotly.express as px

    rows = []

    # レイヤーごとに繰り返し
    for layer_idx in range(1, num_layers):
        key = (layer_idx, selected_head)
        if key not in data_dict:
            continue

        data_L = data_dict[key]
        reduced_L = data_L["reduced"]       # (seq_len, 2)
        attn = data_L["attn_scores"]        # (seq_len, seq_len)
        v_prev = data_L["v_prev"]           # (seq_len, hidden_dim)

        # 選択トークン i = selected_token_idx が他トークン j に払う注意を取り出す
        attn_sel = attn[selected_token_idx, :]  # shape: (seq_len,)
        # v_i
        v_i = v_prev[selected_token_idx]        # shape: (hidden_dim,)

        # v'_i (skip or not)
        if use_skip_connection:
            v_i_prime = v_i + np.sum(v_prev * attn_sel[:, None], axis=0)
        else:
            v_i_prime = np.sum(v_prev * attn_sel[:, None], axis=0)

        # Relevance を計算
        r = compute_relevance(
            attn_weights=attn_sel,
            v_prev=v_prev,
            v_i=v_i,
            v_i_prime=v_i_prime,
            rule_type=rule_type,
            use_skip_connection=use_skip_connection
        )

        # 前レイヤーの座標
        prev_key = (layer_idx - 1, selected_head)
        data_prev_L = data_dict.get(prev_key, None)
        if data_prev_L is None:
            continue
        reduced_prev_L = data_prev_L["reduced"]  # (seq_len, 2)

        for j in range(seq_len):
            label = "SelectedToken" if j == selected_token_idx else "OtherToken"
            # Relevance -> 円のサイズ
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

        # 合成後の v_{(L), i} を "★" (star) でプロット
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

    # DataFrame にまとめる
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
        title=f"Head={selected_head}, Relevance={rule_type}, Skip={use_skip_connection}"
    )
    return fig_animated


###############################################################################
# 7. Streamlit アプリ (メイン)
###############################################################################
def render_page():
    st.title("Cross-Layer Weighted V Visualization for BERT (Relevance) - Skip Connection ON/OFF")

    # --- はじめの説明 ---
    st.markdown("""
    ### BERT のアテンション + Value 更新可視化  
    **Skip Connection (Residual) を ON/OFF** で切り替えて、更新後の Value ベクトルがどう変わるかを可視化します。
    """)

    # モデルロード
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = load_bert_with_qkv(model_name)
    model.eval()

    num_layers = 12
    num_heads = 12

    # 次元削減器 (UMAP または PCA)
    st.markdown("### 次元削減 (UMAP または PCA) の準備")
    use_umap = True  # UMAP で行うか PCA で行うか; ここでは固定
    texts_for_fitting = ["My dog is cute. He likes play running."]
    reducers_per_head = fit_umap_or_pca_on_last_layer(
        texts=texts_for_fitting,
        tokenizer=tokenizer,
        model=model,
        last_layer_idx=num_layers - 1,
        num_heads=num_heads,
        use_umap=use_umap,
        cache_dir="./cache/qkv/"
    )

    # テキスト入力
    st.markdown("### テキスト入力")
    text_to_plot = st.text_area(
        "入力する文章 (英語)",
        "My dog is cute. He likes play running."
    )

    # Skip Connection チェックボックス
    use_skip_connection = st.checkbox("Use Skip Connection?", value=True)

    # データ取得
    (
        data_dict, 
        x_min_per_head, x_max_per_head,
        y_min_per_head, y_max_per_head,
        tokens, seq_len
    ) = extract_plot_data_across_layers(
        text=text_to_plot,
        tokenizer=tokenizer,
        model=model,
        reducers_per_head=reducers_per_head,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=16,
        cache_dir="./cache/qkv/",
        use_skip_connection=use_skip_connection
    )
    st.success("Cross-layer でのアテンション & Value 情報を取得しました。")

    # トークン一覧を表示
    st.markdown("#### トークン一覧")
    st.write(tokens)

    # 選択トークン/ヘッド/ルールを UI から選択
    token_options = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
    selected_token_str = st.selectbox("Relevance を計算する対象トークン (index: token)", token_options)
    selected_token_idx = int(selected_token_str.split(":")[0])

    selected_head = st.selectbox("可視化するヘッド (head_idx)", list(range(num_heads)))

    rule_type = st.selectbox(
        "Relevance の計算方法を選択",
        ["raw", "weighted_distance", "weighted_projection"],
        format_func=lambda x: {
            "raw": "① raw = w_{i j}",
            "weighted_distance": "② weighted_distance = w_{i j} * ||v_j - v_i||",
            "weighted_projection": "③ weighted_projection = w_{i j} * |(v_j - v_i)·(v'_i - v_i)|",
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
            rule_type=rule_type,
            num_layers=num_layers,
            use_skip_connection=use_skip_connection
        )
        st.plotly_chart(fig_plotly)


# Streamlit 実行エントリポイント
if __name__ == "__main__":
    render_page()
