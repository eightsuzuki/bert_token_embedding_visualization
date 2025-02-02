import os
import torch
import numpy as np
import streamlit as st
from transformers import BertModel, BertConfig, BertTokenizer

###############################################################################
# 1. BERT 内部の Q, K, V を取得するクラス（フック利用）
###############################################################################
class BertModelWithQKV(BertModel):
    def __init__(self, config: BertConfig):
        super().__init__(config)
        self.q_layers = {}
        self.k_layers = {}
        self.v_layers = {}
        # 各層の self-attention 部分にフックを仕込み、実行時に q, k, v を取得する
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx):
        """
        layer_idx番目のEncoderLayerのSelf-Attentionにフックを仕込む。
        順伝搬の際に、クエリ・キー・バリュー行列を取得してキャッシュする。
        """
        def hook(module, module_input, module_output):
            # module_input[0] が "hidden_states" に相当
            hidden_states = module_input[0]
            query_layer = module.query(hidden_states)
            key_layer = module.key(hidden_states)
            value_layer = module.value(hidden_states)
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook

    def get_qkv_from_layer(self, layer_idx):
        """
        登録しておいた layer_idx 番目の Q, K, V を返す。
        """
        return (
            self.q_layers.get(layer_idx, None),
            self.k_layers.get(layer_idx, None),
            self.v_layers.get(layer_idx, None),
        )

###############################################################################
# 2. 前処理とヘッド分割のユーティリティ
###############################################################################
def preprocess_text(text: str, tokenizer, max_length=32):
    """
    入力テキストをトークナイズし、最大長を max_length に揃えて返す。
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
    hidden_dim をヘッド数(num_heads)で分割し、(batch_size, num_heads, seq_len, head_dim) 形状に変換する。
    """
    batch_size, seq_len, hidden_dim = tensor.shape
    head_dim = hidden_dim // num_heads
    tensor = tensor.view(batch_size, seq_len, num_heads, head_dim)
    tensor = tensor.permute(0, 2, 1, 3).contiguous()
    return tensor

###############################################################################
# 3. 各層の注意行列を抽出する関数
###############################################################################
def extract_attention_data(text, tokenizer, model,
                           num_layers=12, num_heads=12, max_length=32):
    """
    入力テキストを BERT (BertModelWithQKV) で処理し、
    各層・各ヘッドの注意行列 A^{(l)} を計算して返す。

    A^{(l)}_{ij}: トークン i がトークン j にどれだけ注意を向けているか（softmaxで行が1.0になるよう正規化）。

    なお [PAD] トークンは可視化時に除外する。
    """
    # 1. 入力をトークナイズ
    inputs = preprocess_text(text, tokenizer, max_length)
    # 2. 実際にモデルを推論（順伝搬）してフックで Q,K,V を取得
    with torch.no_grad():
        _ = model(**inputs)

    # 3. トークン列と [PAD] 以外のindexを特定
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    valid_indices = [i for i, token in enumerate(tokens) if token != "[PAD]"]
    valid_tokens = [tokens[i] for i in valid_indices]

    data = {}
    # BERT-base は通常 12層 x 12ヘッド
    for layer_idx in range(num_layers):
        q, k, _ = model.get_qkv_from_layer(layer_idx)
        # q,k,v は (batch_size, seq_len, hidden_dim) 形状 (batch_size=1 前提)
        q_split = split_heads(q, num_heads=num_heads)[0]  # -> (num_heads, seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]  # -> (num_heads, seq_len, head_dim)

        # 注意スコア（未softmax）: (num_heads, seq_len, seq_len)
        attn_logits = torch.matmul(q_split, k_split.transpose(-2, -1))

        # softmax の温度係数 1/sqrt(d_k)
        d_k = q_split.size(-1)
        attn_scores = attn_logits / np.sqrt(d_k)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)

        # (layer_idx, head_idx) をキーにして注意行列を保存
        for head_idx in range(num_heads):
            A = attn_scores[head_idx].cpu().numpy()  # (seq_len, seq_len)
            # [PAD] は除外して格納
            A_valid = A[np.ix_(valid_indices, valid_indices)]
            data[(layer_idx, head_idx)] = {"attn_scores": A_valid}

    return data, valid_tokens

###############################################################################
# 4. Attention Rollout / Flow の計算
###############################################################################
def compute_attention_rollout_from_layer(data_dict, selected_head,
                                         start_layer, end_layer,
                                         add_residual=True):
    """
    Attention Rollout を計算する関数。
    論文にある通り、各層の注意行列 A^{(l)} に I（単位行列）を足して正規化し、
    それを層をまたいで順次乗算していく。

    数式:
    R = A_hat^{(end_layer)} * A_hat^{(end_layer-1)} * ... * A_hat^{(start_layer+1)}

    ただし A_hat^{(l)} = (A^{(l)} + I) / row_sum(A^{(l)} + I)
    """
    attn_matrices = []
    for layer_idx in range(start_layer + 1, end_layer + 1):
        key = (layer_idx, selected_head)
        if key in data_dict:
            A = data_dict[key]["attn_scores"]  # (seq_len, seq_len)
            attn_matrices.append(A)

    # 指定した層の間でアテンションが取れなかった場合は単位行列を返す
    if not attn_matrices:
        seq_len = data_dict[(start_layer, selected_head)]["attn_scores"].shape[0]
        return np.eye(seq_len)

    # Rolloutの初期値 R を単位行列に設定
    R = np.eye(attn_matrices[0].shape[0])
    for A in attn_matrices:
        if add_residual:
            A_hat = A + np.eye(A.shape[0])
            A_hat = A_hat / A_hat.sum(axis=-1, keepdims=True)
        else:
            # 残差を足さない場合
            A_hat = A / (A.sum(axis=-1, keepdims=True) + 1e-8)
        R = A_hat @ R  # 右から掛けるか左から掛けるかは定義次第（ここでは左からA_hatをかけている）

    return R

def compute_attention_flow_from_layer(data_dict, selected_head,
                                      start_layer, end_layer):
    """
    Attention Flow (※厳密な最大フローではなく、簡易的な累積モデル) を計算する関数。

    ここでは、
      F^{(l)} = A^{(l)} + A^{(l)} * F^{(l-1)}
    の形で累積し、各層ごとに row-wise 正規化を行う近似実装。
    本来は論文で示されるように「最大フロー問題」として解く必要があるが、簡易化のための実装。

    数式イメージ:
      F^{(L_0)} = 0
      F^{(l)} = softmax_row( A^{(l)} + A^{(l)} * F^{(l-1)} )
    """
    attn_matrices = []
    for layer_idx in range(start_layer + 1, end_layer + 1):
        key = (layer_idx, selected_head)
        if key in data_dict:
            A = data_dict[key]["attn_scores"]
            attn_matrices.append(A)

    if not attn_matrices:
        seq_len = data_dict[(start_layer, selected_head)]["attn_scores"].shape[0]
        return np.eye(seq_len)

    F = np.zeros_like(attn_matrices[0])
    for A in attn_matrices:
        # Fを A で更新
        F = A + A @ F
        # 行ごとに正規化 (合計が0になる可能性を考慮して tiny value を足す)
        F = F / (F.sum(axis=-1, keepdims=True) + 1e-8)

    return F

def get_influence_vector(data_dict, selected_head, start_layer,
                         end_layer, selected_token_idx, method="rollout"):
    """
    指定した method に応じて累積行列 R (または F) を計算し、
    その最終的な行ベクトル (selected_token_idx の行) を取り出す。
    """
    if method == "rollout":
        # 論文に近づけるため、デフォルトで add_residual=True
        R = compute_attention_rollout_from_layer(
            data_dict, selected_head,
            start_layer, end_layer,
            add_residual=True
        )
        influence = R[selected_token_idx]

    elif method == "flow":
        F = compute_attention_flow_from_layer(
            data_dict, selected_head,
            start_layer, end_layer
        )
        influence = F[selected_token_idx]

    else:
        raise ValueError(f"Unknown method: {method}")

    return influence

###############################################################################
# 5. テキスト上に寄与度を背景色で示すための可視化関数
###############################################################################
def render_influence_on_text(tokens, influence, max_color_strength=5.0):
    """
    tokens: トークンのリスト
    influence: 各トークンへの寄与度（足すと1付近を想定）
    max_color_strength: スコアを何倍して背景色の濃さを決めるか
    """
    html = "<div style='font-size:16px; white-space:nowrap;'>"
    for i, (token, score) in enumerate(zip(tokens, influence)):
        alpha = float(np.clip(score * max_color_strength, 0, 1))
        # トークン番号もツールチップで確認できるよう title 属性を追加
        html += (
            f"<span title='Token index: {i}, Score: {score:.4f}' "
            f"style='background-color: rgba(255, 0, 0, {alpha:.2f}); "
            f"margin:2px; padding:2px;'>"
            f"{token}</span> "
        )
    html += "</div>"
    return html

###############################################################################
# 6. Streamlit アプリ本体
###############################################################################
def render_page():
    st.title("Attention Flow による情報伝播解析")
    st.markdown("""
    ## 概要
    このデモでは、BERT の各層の注意行列を解析し、最終層でのトークンごとの影響を計算します。
    注意行列は以下の数式で定義されます：
    """)
    st.latex(r"A^{(l)} = \text{softmax}\left(\frac{Q^{(l)} K^{(l)T}}{\sqrt{d_k}}\right)")
    
    st.markdown("""
    - ここで、\( Q^{(l)} \) はクエリ行列、\( K^{(l)} \) はキー行列、\( d_k \) はキーの次元数です。
    - 各行の合計が1となるようにSoftmax正規化が適用されます。
    
    ### Attention Rollout
    各層の注意行列 \( A^{(l)} \) に単位行列 \( I \) を加えたものを正規化し、行列積を計算することで、
    最終層での寄与 \( R \) を求めます。
    """)
    st.latex(r"R = \prod_{l=L_0+1}^{L} \frac{A^{(l)} + I}{\sum_{j}(A^{(l)} + I)_{ij}}")
    st.image("img/attention_rollout.gif", caption="Attention Rollout のイメージ図")
    st.markdown("""
    ### Attention Flow
    Attention Flow は以下のように再帰的に計算されます：
    """)
    st.latex(r"F^{(l)} = \frac{A^{(l)} + A^{(l)}F^{(l-1)}}{\sum_{j}(A^{(l)} + A^{(l)}F^{(l-1)})_{ij}},\quad F^{(L_0)}=0,\quad F=F^{(L)}")
    
    st.markdown("""
    各層で \( F^{(l)} \) を計算し、最終層でのトークンの影響度を可視化します。
    """)
    st.image("img/attention_flow.gif", caption="Attention Flow のイメージ図")
    
    st.markdown("---")
    st.markdown("### 入力テキストの設定")
    text_input = st.text_area(
        "入力テキスト (英語) を入力してください",
        "My dog is cute. He likes play running."
    )

    selected_method = st.selectbox("計算方法 (method)", ["rollout", "flow"],
        format_func=lambda x: "Attention Rollout" if x=="rollout" else "Attention Flow"
    )

    # BERT-base の標準構成
    num_layers = 12
    num_heads = 12

    selected_head = st.selectbox("対象ヘッド (0 ～ 11)", list(range(num_heads)))
    
    if text_input.strip():
        # 1. モデルの準備
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name, output_attentions=True)
        model = BertModelWithQKV.from_pretrained(model_name, config=config)
        model.eval()

        # 2. 注意行列の取得
        data_dict, tokens = extract_attention_data(
            text=text_input,
            tokenizer=tokenizer,
            model=model,
            num_layers=num_layers,
            num_heads=num_heads,
            max_length=32
        )

        # 3. 「どのトークンに対する最終層の寄与を追うか」選択
        #    -> トークンは "index: token文字列" の形で表示
        token_options = [f"{i}: {token}" for i, token in enumerate(tokens)]
        selected_token_option = st.selectbox(
            "最終層で注目するトークン（どのトークンに注目するか）",
            token_options
        )
        selected_token_idx = int(selected_token_option.split(":")[0])

        st.markdown("---")
        if st.button("計算して各層からの影響を可視化する"):
            st.markdown(f"#### 計算方法: **{selected_method}**, Head = {selected_head}")
            st.markdown(f"最終層で Token `{tokens[selected_token_idx]}` に注目したときの、各層における寄与可視化")

            # BERT の最終層は layer_idx = 11
            end_layer = num_layers - 1

            # テーブル表示: 各層 (0~10) を「ソース層」として累積した場合の可視化
            # ただし実運用上は「ソース層を 0 に固定」または「全部を一気にかけ合わせる」ほうが多い。
            # 学習・解釈用に、あえて層ごとの可視化をしている。
            table_html = "<div style='overflow-x:auto;'>"
            table_html += "<table border=1 style='border-collapse: collapse; white-space: nowrap;'>"
            table_html += (
                "<tr>"
                "<th style='padding:8px;'>Source Layer</th>"
                f"<th style='padding:8px;'>Head {selected_head} の影響可視化</th>"
                "</tr>"
            )

            for src_layer in range(num_layers - 1):
                influence = get_influence_vector(
                    data_dict=data_dict,
                    selected_head=selected_head,
                    start_layer=src_layer,
                    end_layer=end_layer,
                    selected_token_idx=selected_token_idx,
                    method=selected_method
                )
                cell_html = render_influence_on_text(tokens, influence)
                table_html += (
                    f"<tr>"
                    f"<td style='padding:8px;'>Layer {src_layer} → Layer {end_layer}</td>"
                    f"<td style='padding:8px;'>{cell_html}</td>"
                    f"</tr>"
                )
            table_html += "</table></div>"

            st.markdown("### 各層から最終層への累積的影響（Background Color 表示）", unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
