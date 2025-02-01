import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import joblib
import hashlib  # 追加
import io
import matplotlib.animation as animation

from typing import Dict, List
from transformers import BertModel, BertConfig, BertTokenizer
from sklearn.decomposition import PCA
import umap


###############################################################################
# 1. BERT の内部から q, k, v を取得するためのカスタムクラス
###############################################################################
class BertModelWithQKV(BertModel):
    """
    BertModel を継承し、forward hook で各レイヤーの q, k, v を保存するクラス。
    model.encoder.layer[i].attention.self にフックを仕込む。
    """

    def __init__(self, config: BertConfig):
        super().__init__(config)

        # レイヤーごとの q, k, v を保持する辞書 (layer_idx -> Tensor)
        self.q_layers: Dict[int, torch.Tensor] = {}
        self.k_layers: Dict[int, torch.Tensor] = {}
        self.v_layers: Dict[int, torch.Tensor] = {}

        # BertEncoder の各レイヤーにフックを登録
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx: int):
        """
        指定レイヤーの BertSelfAttention から、query/key/value 計算結果をフックで取り出す。
        """
        def hook(module, module_input, module_output):
            # module_input: (hidden_states, ) のタプル
            hidden_states = module_input[0]  # shape: (batch_size, seq_len, hidden_dim)

            # BertSelfAttention 内部実装:
            # query_layer = self.query(hidden_states)
            # key_layer   = self.key(hidden_states)
            # value_layer = self.value(hidden_states)
            query_layer = module.query(hidden_states)
            key_layer   = module.key(hidden_states)
            value_layer = module.value(hidden_states)

            # CPU に移して保持しておく（GPU でもよければ detach のみでもOK）
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook

    def get_qkv_from_layer(self, layer_idx: int):
        """
        フォワード後にフックで格納された q, k, v を返す。
        """
        q = self.q_layers.get(layer_idx, None)
        k = self.k_layers.get(layer_idx, None)
        v = self.v_layers.get(layer_idx, None)
        return q, k, v


###############################################################################
# 2. モデル読み込み & テキスト前処理ユーティリティ
###############################################################################
def load_bert_with_qkv(model_name="bert-base-uncased") -> BertModelWithQKV:
    """
    HuggingFace の "bert-base-uncased" 等を元にカスタム BertModelWithQKV を作る。
    """
    config = BertConfig.from_pretrained(model_name, output_attentions=True)
    model = BertModelWithQKV.from_pretrained(model_name, config=config)
    return model

def preprocess_text(text: str, tokenizer, max_length=128):
    """
    BERT 用の入力 IDs, attention masks 等を作る。
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
    BERT の (batch_size, seq_len, hidden_dim) を
    (batch_size, num_heads, seq_len, head_dim) に reshape する。
    """
    batch_size, seq_len, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    # (batch_size, seq_len, num_heads, head_dim)
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
    # (batch_size, num_heads, seq_len, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor


###############################################################################
# 3. ヘッドごとに UMAP/PCA モデルをフィッティング
###############################################################################
def fit_umap_or_pca_per_head(
    texts: List[str],
    tokenizer,
    model: BertModelWithQKV,
    num_layers: int = 12,
    num_heads: int = 12,
    use_umap: bool = True,
    cache_dir: str = "./cache/qkv/"
):
    """
    全レイヤーの v を集めて、ヘッドごとに UMAP or PCA へフィットする。
    レイヤー情報は無視し、ヘッドごとに統一された次元削減モデルを作成。

    ★ここを変更して、最後のレイヤーのみからデータを集める★
    """
    import hashlib
    joined_text = " ".join(texts)
    text_hash = hashlib.md5(joined_text.encode("utf-8")).hexdigest()[:8]

    if use_umap:
        cache_file = f"reducers_umap_{text_hash}.pkl"
    else:
        cache_file = f"reducers_pca_{text_hash}.pkl"

    cache_path = os.path.join(cache_dir, cache_file)

    if os.path.isfile(cache_path):
        print(f"=== Loading reducers from local cache: {cache_path} ===")
        return joblib.load(cache_path)

    # 最後のレイヤー (num_layers - 1) の v だけ集める
    last_layer_idx = num_layers - 1

    head_accumulated_data = [[] for _ in range(num_heads)]

    for text in texts:
        inputs = preprocess_text(text, tokenizer)
        # フォワード -> フックが走る
        with torch.no_grad():
            model(**inputs)

        # "最後のレイヤー" だけから v を取得
        q, k, v = model.get_qkv_from_layer(last_layer_idx)
        v_split = split_heads(v, num_heads=num_heads)[0]  # batch_size=1

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

        # ここで "最後のレイヤー" のみを使って fit
        reducer.fit(data_head)
        reducers.append(reducer)

    print("=== Done fitting UMAP/PCA per head (using last-layer data only) ===")

    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(reducers, cache_path)
    print(f"=== Saved reducers to local cache: {cache_path} ===")

    return reducers


###############################################################################
# 4. 可視化用データ抽出関数
###############################################################################
def extract_plot_data(
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
    テキストを入力としてモデルをフォワードし、各レイヤー・ヘッドの v を 2次元マッピングする。
    その際に、CLSトークンのアテンションスコアなども含む可視化用のデータを一括で抽出して返す。

    戻り値:
    --------
    data_dict: dict
        {
          (layer_idx, head_idx): {
             "reduced": 2次元座標 (seq_len, 2),
             "attn_scores": (seq_len, seq_len)
          }
        }
    x_min_per_head, x_max_per_head, y_min_per_head, y_max_per_head: dict
        各ヘッドごとにレイヤーをまたいだときの x,y の最小・最大値
    tokens: List[str]
        トークナイザで分割したトークン列
    seq_len: int
        トークン列の長さ
    """
    import hashlib
    text_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    cache_file = f"extracted_data_{text_hash}.pkl"
    cache_path = os.path.join(cache_dir, cache_file)

    if os.path.isfile(cache_path):
        print(f"=== Loading extracted data from local cache: {cache_path} ===")
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

            # ← ここでは "fit_umap_or_pca_per_head" が最後のレイヤーだけで学習した reducer を使う
            reduced = reducers_per_head[head_idx].transform(head_v)

            data_dict[(layer_idx, head_idx)] = {
                "reduced": reduced,
                "attn_scores": attn_scores,
            }

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

    data_to_save = (
        data_dict,
        x_min_per_head,
        x_max_per_head,
        y_min_per_head,
        y_max_per_head,
        tokens,
        seq_len
    )
    os.makedirs(cache_dir, exist_ok=True)
    joblib.dump(data_to_save, cache_path)
    print(f"=== Saved extracted data to local cache: {cache_path} ===")

    return data_to_save


###############################################################################
# 5. すべてのレイヤー & ヘッドをまとめて可視化する関数
###############################################################################
def plot_all_layers_with_shared_head_reducers(
    data_dict,
    x_min_per_head,
    x_max_per_head,
    y_min_per_head,
    y_max_per_head,
    tokens,
    seq_len,
    num_layers: int = 12,
    num_heads: int = 12,
    output_dir: str = None,
    output_filename: str = None,
    show_plot=True,
    selected_token_idx=0
):
    """
    全レイヤーの v を 2 次元マッピングし、ヘッドごとに統一されたリダクションモデルを適用して可視化。
    さらに、同じヘッドであればレイヤーをまたいでも x/y 軸の表示範囲を固定する。
    """

    fig, axes = plt.subplots(
        num_layers, num_heads, figsize=(4 * num_heads, 4 * num_layers), squeeze=False
    )

    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            reduced = data_dict[(layer_idx, head_idx)]["reduced"]
            attn_scores = data_dict[(layer_idx, head_idx)]["attn_scores"]
            selected_attn = attn_scores[selected_token_idx, :]

            sizes = selected_attn * 1500
            colors = ["blue"] * seq_len
            colors[selected_token_idx] = "red"

            ax = axes[layer_idx, head_idx]
            ax.scatter(
                reduced[:, 0],
                reduced[:, 1],
                s=sizes,
                alpha=0.5,
                c=colors
            )
            ax.set_title(f"Layer {layer_idx}, Head {head_idx}")

            ax.set_xlim([x_min_per_head[head_idx], x_max_per_head[head_idx]])
            ax.set_ylim([y_min_per_head[head_idx], y_max_per_head[head_idx]])

            ax.set_xlabel("Component 1")
            ax.set_ylabel("Component 2")

            for i in range(seq_len):
                ax.text(
                    reduced[i, 0],
                    reduced[i, 1],
                    tokens[i],
                    fontsize=6,
                    ha="center",
                    va="center",
                    color="black",
                )

    plt.tight_layout()

    if output_dir is not None and output_filename is not None:
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, output_filename)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved at: {save_path}")

    if show_plot:
        plt.show()

    return fig


###############################################################################
# 6. 選択したレイヤー・ヘッドのみ可視化する関数
###############################################################################
def plot_individual_layer_and_head(
    data_dict,
    x_min_per_head,
    x_max_per_head,
    y_min_per_head,
    y_max_per_head,
    tokens,
    seq_len,
    selected_layer,
    selected_head,
    selected_token_idx=0
):
    """
    選択されたレイヤーとヘッドの次元削減結果を表示し、
    任意のトークン (selected_token_idx) が他のトークンに払うアテンションに応じて点のサイズを変化させる。
    さらに、その選択トークンを赤色で強調表示する。
    """

    fig_selected, ax_selected = plt.subplots()
    reduced_selected = data_dict[(selected_layer, selected_head)]["reduced"]
    attn_matrix = data_dict[(selected_layer, selected_head)]["attn_scores"]

    selected_attn = attn_matrix[selected_token_idx, :]
    sizes = selected_attn * 1500

    colors = ["blue"] * seq_len
    colors[selected_token_idx] = "red"

    ax_selected.scatter(
        reduced_selected[:, 0],
        reduced_selected[:, 1],
        s=sizes,
        alpha=0.5,
        c=colors
    )
    ax_selected.set_title(f"Layer {selected_layer}, Head {selected_head}")

    ax_selected.set_xlim([x_min_per_head[selected_head], x_max_per_head[selected_head]])
    ax_selected.set_ylim([y_min_per_head[selected_head], y_max_per_head[selected_head]])

    ax_selected.set_xlabel("Component 1")
    ax_selected.set_ylabel("Component 2")

    for i in range(seq_len):
        ax_selected.text(
            reduced_selected[i, 0],
            reduced_selected[i, 1],
            tokens[i],
            fontsize=6,
            ha="center",
            va="center",
            color="black",
        )

    return fig_selected


###############################################################################
# 9. 新規追加: 選択したヘッドを固定して、Plotly でレイヤーをアニメーション表示
###############################################################################
def animate_selected_head_across_layers_plotly(
    data_dict,
    x_min_per_head,
    x_max_per_head,
    y_min_per_head,
    y_max_per_head,
    tokens,
    seq_len,
    selected_head,
    selected_token_idx=0,
    num_layers: int = 12,
    min_size: int = 0  # 追加: サイズの最低値
):
    """
    選択したヘッドを固定し、レイヤー 0~(num_layers-1) の変化を
    Plotly Express のアニメーション付き散布図で可視化する関数。
    """

    import pandas as pd
    import plotly.express as px

    rows = []
    for layer_idx in range(num_layers):
        reduced = data_dict[(layer_idx, selected_head)]["reduced"]
        attn_matrix = data_dict[(layer_idx, selected_head)]["attn_scores"]
        selected_attn = attn_matrix[selected_token_idx, :]

        sizes = selected_attn * 100
        sizes = np.maximum(sizes, min_size)  # サイズの最低値を適用
        for i in range(seq_len):
            color_label = "Selected" if i == selected_token_idx else "Other"
            rows.append({
                "x": reduced[i, 0],
                "y": reduced[i, 1],
                "Layer": layer_idx,
                "Token": tokens[i],
                "Label": color_label,
                "Size": sizes[i] + 5  # サイズを調整
            })

    df_combined = pd.DataFrame(rows)

    color_map = {
        "Selected": "red",
        "Other": "blue"
    }

    # Calculate global min and max for x and y across all heads
    global_x_min = min(x_min_per_head.values())
    global_x_max = max(x_max_per_head.values())
    global_y_min = min(y_min_per_head.values())
    global_y_max = max(y_max_per_head.values())

    fig_animated = px.scatter(
        df_combined,
        x="x",
        y="y",
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
# 7. Streamlit用ページレンダリング関数（メイン）
###############################################################################
def render_page():
    # 説明文を表示
    st.markdown("""
    # BERT Embedding Mapping

    このコードは、BERTモデルの内部からクエリ (q)、キー (k)、バリュー (v) を取得し、それらを次元削減 (UMAPまたはPCA) を用いて可視化します。

    ## 数式
    BERTのセルフアテンション機構において、クエリ、キー、バリューは以下のように計算されます：
    """)

    st.latex(r'''
    \begin{align*}
    Q_n &= W_Q \cdot V_n \\
    K_n &= W_K \cdot V_n \\
    V_n &= W_V \cdot V_n
    \end{align*}
    ''')

    st.markdown("""
    ここで、\( W_Q \)、\( W_K \)、\( W_V \) はそれぞれクエリ、キー、バリューの重み行列であり、\( V_{n} \) は現在のレイヤーの出力です。

    アテンションスコアは以下のように計算されます：
    """)

    st.latex(r'''
    \text{Attention Scores} = \frac{Q_n \cdot K_n^T}{\sqrt{d_k}}
    ''')

    st.markdown("""
    ここで、\( d_k \) はキーの次元数です。

    ソフトマックス関数を適用して、スコアを確率に変換します：
    """)

    st.latex(r'''
    \text{Attention Weights} = \text{softmax}\left(\frac{Q_n \cdot K_n^T}{\sqrt{d_k}}\right)
    ''')

    st.markdown("""
    アテンションウェイトをバリューに適用して、最終的な出力を得ます：
    """)

    st.latex(r'''
    \text{Output} = \text{Attention Weights} \cdot V_n
    ''')

    st.markdown("""
    ## 出力
    このコードは、各レイヤーおよび各ヘッドにおけるバリュー (v) を2次元空間にマッピングし、可視化します。可視化にはUMAPまたはPCAを使用し、各トークンのアテンションスコアに基づいてプロットのサイズを調整します。

    ## 可視化の詳細
    - 各トークンの位置は次元削減されたバリュー (v) に基づいて決定されます。
    - 各トークンのサイズはアテンションスコアに基づいて調整されます。
    - 任意のトークンを選択して、そのトークンが他のトークンに対してどれだけ注意を払っているかを可視化します。

    ## 詳細な説明
    ### アテンションスコアの計算
    1. クエリ（Q_n）とキー（K_n）の内積を計算し、スケーリングします。
    2. ソフトマックス関数を適用して、スコアを確率に変換します。

    ### 任意のトークンのアテンションスコアの取得
    - 選択したトークンが他のすべてのトークンに対してどれだけ注意を払っているかを示すスコアを取得します。

    ### 具体的な流れ
    1. **クエリ、キー、バリューの計算**:
       - 各レイヤーの各ヘッドに対して、クエリ（Q_n）、キー（K_n）、バリュー（V_n）を計算します。

    2. **アテンションスコアの計算**:
       - クエリ（Q_n）とキー（K_n）の内積を計算し、スケーリングしてソフトマックス関数を適用します。
       - これにより、各トークンが他のトークンに対してどれだけ注意を払っているかを示すアテンションスコアが得られます。

    3. **任意のトークンのアテンションスコアの取得**:
       - 選択したトークンが他のすべてのトークンに対してどれだけ注意を払っているかを示すスコアを取得します。

    4. **次元削減の適用**:
       - バリュー（V_n）テンソルに対して次元削減（UMAPまたはPCA）を適用し、2次元座標を取得します。

    5. **データの保存**:
       - 次元削減された2次元座標と、各トークン間のアテンション行列を辞書に保存します。

    ### まとめ
    このコードは、各レイヤーおよび各ヘッドにおける任意のトークンのアテンションスコアを計算し、それを次元削減されたバリュー（V_n）テンソルの2次元座標とともに保存しています。これにより、特定のトークンが他のトークンに対してどれだけ注意を払っているかを視覚的に解析することができます。
    """)

    ###############################################################################
    # モデル・トークナイザ等の準備
    ###############################################################################
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = load_bert_with_qkv(model_name)
    model.eval()

    texts_for_fitting = [
        "She was a teacher for forty years and her writing has appeared in journals and anthologies since the early 1980s."
    ]

    num_layers = 12
    num_heads = 12
    use_umap = True

    # ★最後のレイヤーだけで学習し、それを全レイヤーに適用する★
    reducers_per_head = fit_umap_or_pca_per_head(
        texts=texts_for_fitting,
        tokenizer=tokenizer,
        model=model,
        num_layers=num_layers,
        num_heads=num_heads,
        use_umap=use_umap,
        cache_dir="./cache/qkv/"
    )

    text_to_plot = (
        "She was a teacher for forty years and her writing has appeared in journals and "
        "anthologies since the early 1980s."
    )

    (
        data_dict,
        x_min_per_head,
        x_max_per_head,
        y_min_per_head,
        y_max_per_head,
        tokens,
        seq_len
    ) = extract_plot_data(
        text=text_to_plot,
        tokenizer=tokenizer,
        model=model,
        reducers_per_head=reducers_per_head,
        num_layers=num_layers,
        num_heads=num_heads,
        max_length=16,
        cache_dir="./cache/qkv/"
    )
    
    
    if st.button("Plot All Layers and Heads"):
        fig_all = plot_all_layers_with_shared_head_reducers(
            data_dict=data_dict,
            x_min_per_head=x_min_per_head,
            x_max_per_head=x_max_per_head,
            y_min_per_head=y_min_per_head,
            y_max_per_head=y_max_per_head,
            tokens=tokens,
            seq_len=seq_len,
            num_layers=num_layers,
            num_heads=num_heads,
            output_dir="image/qkv_outputs",
            output_filename="all_layers_heads_values.png",
            show_plot=False,
            selected_token_idx=selected_token_idx
        )
        st.pyplot(fig_all)

    token_options = [f"{i}: {tok}" for i, tok in enumerate(tokens)]
    selected_token_str = st.selectbox("Select a Token to visualize its attention", token_options)
    selected_token_idx = int(selected_token_str.split(":")[0])
    selected_layer = st.selectbox("Select Layer", list(range(num_layers)))
    selected_head = st.selectbox("Select Head", list(range(num_heads)))

    if st.button("Plot Selected Layer and Head"):
        fig_selected = plot_individual_layer_and_head(
            data_dict=data_dict,
            x_min_per_head=x_min_per_head,
            x_max_per_head=x_max_per_head,
            y_min_per_head=y_min_per_head,
            y_max_per_head=y_max_per_head,
            tokens=tokens,
            seq_len=seq_len,
            selected_layer=selected_layer,
            selected_head=selected_head,
            selected_token_idx=selected_token_idx
        )
        st.pyplot(fig_selected)

    # 新規追加: Plotly でアニメーション
    if st.button("Animate with Plotly"):
        import plotly.express as px
        fig_plotly = animate_selected_head_across_layers_plotly(
            data_dict=data_dict,
            x_min_per_head=x_min_per_head,
            x_max_per_head=x_max_per_head,
            y_min_per_head=y_min_per_head,
            y_max_per_head=y_max_per_head,
            tokens=tokens,
            seq_len=seq_len,
            selected_head=selected_head,
            selected_token_idx=selected_token_idx,
            num_layers=num_layers
        )
        st.plotly_chart(fig_plotly)


if __name__ == "__main__":
    render_page()
