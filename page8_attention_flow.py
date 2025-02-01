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
        for layer_idx in range(config.num_hidden_layers):
            self.encoder.layer[layer_idx].attention.self.register_forward_hook(
                self._get_qkv_hook(layer_idx)
            )

    def _get_qkv_hook(self, layer_idx):
        def hook(module, module_input, module_output):
            hidden_states = module_input[0]
            query_layer = module.query(hidden_states)
            key_layer = module.key(hidden_states)
            value_layer = module.value(hidden_states)
            self.q_layers[layer_idx] = query_layer.detach().cpu()
            self.k_layers[layer_idx] = key_layer.detach().cpu()
            self.v_layers[layer_idx] = value_layer.detach().cpu()
        return hook

    def get_qkv_from_layer(self, layer_idx):
        return self.q_layers.get(layer_idx), self.k_layers.get(layer_idx), self.v_layers.get(layer_idx)

###############################################################################
# 2. 前処理とヘッド分割のユーティリティ
###############################################################################
def preprocess_text(text: str, tokenizer, max_length=32):
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
    tensor = tensor.permute(0, 2, 1, 3).contiguous()  # (batch_size, num_heads, seq_len, head_dim)
    return tensor

###############################################################################
# 3. 各層の注意行列の抽出（[PAD] トークンは無視）
###############################################################################
def extract_attention_data(text, tokenizer, model, num_layers=12, num_heads=12, max_length=32):
    """
    入力テキストを BERT で処理し、各層の注意行列（attn_scores）を取得します。
    ※トークン化後、"[PAD]" トークンは無視して計算対象から除外します。
    また、各層の注意行列は、
    \[
    A^{(l)} = \mathrm{softmax}\Bigl(\frac{q^{(l)}\cdot\bigl(k^{(l)}\bigr)^\top}{\sqrt{d_k}}\Bigr)
    \]
    により設定されます。
    """
    inputs = preprocess_text(text, tokenizer, max_length)
    with torch.no_grad():
        _ = model(**inputs)
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    # [PAD] トークンでないインデックスのみを抽出
    valid_indices = [i for i, token in enumerate(tokens) if token != "[PAD]"]
    data = {}
    for layer_idx in range(num_layers):
        # 層 0 はその層自身、層 >0 は直前の層の Q, K を使用
        if layer_idx == 0:
            q, k, _ = model.get_qkv_from_layer(layer_idx)
        else:
            q, k, _ = model.get_qkv_from_layer(layer_idx - 1)
        q_split = split_heads(q, num_heads=num_heads)[0]  # (num_heads, seq_len, head_dim)
        k_split = split_heads(k, num_heads=num_heads)[0]  # (num_heads, seq_len, head_dim)
        # 注意行列 A^{(l)} の計算： softmax((q·k^T)/sqrt(d_k))
        attn_logits = torch.matmul(q_split, k_split.transpose(-2, -1))  # (num_heads, seq_len, seq_len)
        d_k = q_split.size(-1)
        attn_scores = attn_logits / np.sqrt(d_k)
        attn_scores = torch.nn.functional.softmax(attn_scores, dim=-1)
        for head in range(num_heads):
            A = attn_scores[head].cpu().numpy()  # (seq_len, seq_len)
            # [PAD] 行・列を除外
            A_valid = A[np.ix_(valid_indices, valid_indices)]
            data[(layer_idx, head)] = {"attn_scores": A_valid}
    # [PAD] を除いたトークンリストを返す
    valid_tokens = [tokens[i] for i in valid_indices]
    return data, valid_tokens

###############################################################################
# 4. Attention Rollout / Flow の計算
###############################################################################
def compute_attention_rollout_from_layer(data_dict, selected_head, start_layer, end_layer, add_residual=True):
    """
    ソース層 (start_layer) から end_layer までの累積注意行列 R を計算します。
    各層で、残差接続（単位行列 I を加算）を行い、正規化した後、行列積を計算します。
    $$R = \prod_{l=L_0+1}^{L} \frac{A^{(l)} + I}{\sum_{j}(A^{(l)} + I)_{ij}}$$
    """
    attn_matrices = []
    for layer_idx in range(start_layer + 1, end_layer + 1):
        key = (layer_idx, selected_head)
        if key in data_dict:
            A = data_dict[key]["attn_scores"]  # (seq_len, seq_len)
            attn_matrices.append(A)
    if not attn_matrices:
        seq_len = data_dict[(start_layer, selected_head)]["attn_scores"].shape[0]
        return np.eye(seq_len)
    R = np.eye(attn_matrices[0].shape[0])
    for A in attn_matrices:
        if add_residual:
            A_hat = A + np.eye(A.shape[0])
            A_hat = A_hat / A_hat.sum(axis=-1, keepdims=True)
        else:
            A_hat = A
        R = np.matmul(A_hat, R)
    return R

def compute_attention_flow_from_layer(data_dict, selected_head, start_layer, end_layer):
    """
    Attention Flow を再帰的に計算します。
    $$F^{(l)} = \frac{A^{(l)} + A^{(l)}F^{(l-1)}}{\sum_{j}(A^{(l)} + A^{(l)}F^{(l-1)})_{ij}},\quad F^{(L_0)}=0,\quad F=F^{(L)}$$
    として計算し、正規化します。
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
    F = np.zeros(attn_matrices[0].shape)
    for A in attn_matrices:
        F = A + np.matmul(A, F)
        F = F / (F.sum(axis=-1, keepdims=True) + 1e-8)
    return F

def get_influence_vector(data_dict, selected_head, start_layer, end_layer, selected_token_idx, method="rollout"):
    """
    ソース層から end_layer までの累積注意行列を計算し、
    最終層で注目するトークン（selected_token_idx）の寄与を返します。
    method は "rollout" または "flow" を選択してください。
    """
    if method == "rollout":
        R = compute_attention_rollout_from_layer(data_dict, selected_head, start_layer, end_layer, add_residual=True)
        influence = R[selected_token_idx]
    elif method == "flow":
        F = compute_attention_flow_from_layer(data_dict, selected_head, start_layer, end_layer)
        influence = F[selected_token_idx]
    else:
        raise ValueError(f"Unknown method: {method}")
    return influence

###############################################################################
# 5. テキストの各トークンの背景色（赤の濃さ）で寄与度を可視化する関数
###############################################################################
def render_influence_on_text(tokens, influence):
    """
    tokens（リスト）と各トークンへの寄与度 influence（和が1）を受け取り、
    寄与度に応じた赤の背景色で HTML を生成します。
    """
    html = "<div style='font-size:16px; white-space:nowrap;'>"
    for token, score in zip(tokens, influence):
        alpha = np.clip(score * 5, 0, 1)  # スケール調整
        html += f"<span style='background-color: rgba(255, 0, 0, {alpha:.2f}); margin:2px; padding:2px;'>{token}</span> "
    html += "</div>"
    return html

###############################################################################
# 6. Streamlit アプリ本体
###############################################################################
def render_page():
    st.title("Attention Rollout / Flow による背景色可視化デモ")
    
    st.markdown("""
    ## 概要
    このデモでは、BERT の各層の注意行列  
    \[
    A^{(l)} = \mathrm{softmax}\Bigl(\frac{q^{(l)}\cdot\bigl(k^{(l)}\bigr)^\top}{\sqrt{d_k}}\Bigr)
    \]
    （[PAD] トークンは除外）から、指定した計算方法（Attention Rollout または Attention Flow）により、
    最終層で注目するトークンへの累積寄与を各【ソース層】（Layer 0～10）ごとに計算し、  
    その影響を背景色（赤の濃さ）付きのテーブル形式（各行に層番号）で表示します。
    """)
    
    st.markdown("### Attention Rollout")
    st.latex(r"R = \prod_{l=L_0+1}^{L} \frac{A^{(l)} + I}{\sum_{j}(A^{(l)} + I)_{ij}}")
    st.markdown("各層の注意行列 \(A^{(l)}\) に単位行列 \(I\) を加えたものを正規化し、行列積を計算することで、最終層での寄与 \(R\) を求めます。")
    
    st.markdown("### Attention Flow")
    st.latex(r"F^{(l)} = \frac{A^{(l)} + A^{(l)}F^{(l-1)}}{\sum_{j}(A^{(l)} + A^{(l)}F^{(l-1)})_{ij}},\quad F^{(L_0)}=0,\quad F=F^{(L)}")
    st.markdown("各層で \(F^{(l)} = A^{(l)} + A^{(l)}F^{(l-1)}\) を再帰的に計算し、正規化することで累積的な寄与 \(F\) を得ます。")
    
    st.markdown("---")
    
    st.markdown("### ユーザ入力")
    text_input = st.text_area("入力テキスト（英語）", 
                              "She was a teacher for forty years and her writing has appeared in journals and anthologies.")
    num_layers = 12
    num_heads = 12
    selected_method = st.selectbox("計算方法を選択", ["rollout", "flow"],
                                   format_func=lambda x: "Attention Rollout" if x=="rollout" else "Attention Flow")
    selected_head = st.selectbox("対象ヘッド (0～{0})".format(num_heads - 1), list(range(num_heads)))
    
    # ここで、入力テキストがある場合、事前に注意重みを抽出して Token リストを取得
    if text_input:
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        config = BertConfig.from_pretrained(model_name, output_attentions=True)
        model = BertModelWithQKV.from_pretrained(model_name, config=config)
        model.eval()
        data_dict, tokens = extract_attention_data(text_input, tokenizer, model,
                                                     num_layers=num_layers, num_heads=num_heads, max_length=32)
        # Token 選択肢は「番号: Token文字列」で表示
        token_options = [f"{i}: {token}" for i, token in enumerate(tokens)]
        selected_token_option = st.selectbox("最終層で注目するトークン", token_options)
        selected_token_idx = int(selected_token_option.split(":")[0])
        
        if st.button("計算して全層の影響を表示する"):
            end_layer = num_layers - 1
            # 各ソース層（Layer 0～10）ごとに寄与度を計算し、テーブル形式（縦方向）で表示
            table_html = "<div style='overflow-x:auto;'><table border=1 style='border-collapse: collapse; white-space: nowrap;'><tr><th style='padding:8px;'>Layer</th><th style='padding:8px;'>Influence (背景色付き)</th></tr>"
            for src_layer in range(num_layers - 1):
                influence = get_influence_vector(data_dict, selected_head,
                                                 start_layer=src_layer,
                                                 end_layer=end_layer,
                                                 selected_token_idx=selected_token_idx,
                                                 method=selected_method)
                cell_html = render_influence_on_text(tokens, influence)
                table_html += f"<tr><td style='padding:8px;'>Layer {src_layer}</td><td style='padding:8px;'>{cell_html}</td></tr>"
            table_html += "</table></div>"
            
            st.markdown("### 各層からの影響（背景色付き）", unsafe_allow_html=True)
            st.markdown(table_html, unsafe_allow_html=True)

if __name__ == "__main__":
    render_page()
