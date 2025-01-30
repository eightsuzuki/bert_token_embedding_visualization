import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import streamlit as st
import torch
from transformers import BertModel, BertTokenizer

# -------------------------------------------------------------------
# 隣接行列（adjacency matrix）を作成する関数
# -------------------------------------------------------------------
def get_adjmat(att_mat, input_tokens):
    """
    att_mat: shape = (seq_len, seq_len) の2次元numpy配列 (あるlayer,あるheadのアテンション)
    input_tokens: トークンのリスト（長さ seq_len）
    
    返り値:
      adj_mat: (seq_len x seq_len) の隣接行列 (numpy)
      labels_to_index: {token文字列: index} の辞書
    """
    seq_len = att_mat.shape[-1]
    adj_mat = np.zeros((seq_len, seq_len))
    labels_to_index = {token: idx for idx, token in enumerate(input_tokens)}
    
    for i in range(seq_len):
        for j in range(seq_len):
            adj_mat[i, j] = att_mat[i, j]
    
    return adj_mat, labels_to_index

# -------------------------------------------------------------------
# アテンション行列をネットワークグラフとして描画
# -------------------------------------------------------------------
def draw_attention_graph(adj_mat, labels_to_index):
    """
    adj_mat: (seq_len x seq_len)
    labels_to_index: {token文字列: index}
    """
    G = nx.DiGraph()
    
    # ノードを追加
    for token, idx in labels_to_index.items():
        G.add_node(idx, label=token)
    
    # エッジを追加（重みはアテンション値）
    for i in range(len(adj_mat)):
        for j in range(len(adj_mat)):
            if adj_mat[i, j] > 0:
                G.add_edge(i, j, weight=adj_mat[i, j])
    
    # レイアウト計算
    pos = nx.spring_layout(G, seed=42)
    
    # valid_labels: {ノード番号: トークン文字列}
    valid_labels = {idx: token for token, idx in labels_to_index.items() if idx in pos}
    
    # ノードの描画
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
    # ラベルの描画
    nx.draw_networkx_labels(G, pos, labels=valid_labels, font_size=10)
    # エッジの太さをアテンション値に応じて変更
    edges = G.edges(data=True)
    nx.draw_networkx_edges(
        G, pos,
        edgelist=[(u, v) for (u, v, d) in edges],
        width=[d['weight'] * 3 for (u, v, d) in edges],
        arrowstyle='->',
        arrowsize=10
    )
    
    plt.axis("off")
    return G

# -------------------------------------------------------------------
# 単純なAttention可視化用の関数
# （Streamlitのボタン押下時に呼び出す想定）
# -------------------------------------------------------------------
def show_attention_flow_button(all_attentions, tokens, selected_layer, selected_head):
    """
    all_attentions: shape = (num_layers, batch_size, num_heads, seq_len, seq_len)
    tokens: list of token strings, length = seq_len
    selected_layer: 選択されたレイヤー（int）
    selected_head: 選択されたヘッド（int, または -1 でAllHeads平均）
    """
    # all_attentionsは (num_layers, batch_size, num_heads, seq_len, seq_len)
    num_layers, _, num_heads, seq_len, _ = all_attentions.shape
    
    # batch次元はここでは1想定
    if selected_head == -1:
        # All Heads の平均アテンション (指定レイヤー)
        att_mat = all_attentions[selected_layer, 0, :, :, :].mean(axis=0)  # shape=(seq_len, seq_len)
    else:
        # 指定head
        att_mat = all_attentions[selected_layer, 0, selected_head, :, :]
    
    # 隣接行列作成
    adj_mat, labels_to_index = get_adjmat(att_mat.detach().cpu().numpy(), tokens)
    
    # 数値確認用: DataFrame化など
    st.write("### 選択したレイヤー・ヘッドのアテンション行列（上位左部分）")
    st.dataframe(adj_mat[:8, :8])  # 大きい場合は一部だけ表示
    
    # グラフ描画
    draw_attention_graph(adj_mat, labels_to_index)
    st.pyplot(plt.gcf())
    plt.clf()

# -------------------------------------------------------------------
# Streamlit アプリ本体
# -------------------------------------------------------------------
def render_page():
    st.title("BERT Attention Flow Visualization")
    
    # 1) 数式や概要の表示
    st.markdown("## Self-Attentionの計算式")
    st.latex(r"""
    \alpha_{ij} = \mathrm{softmax}\!\Bigl(\frac{Q_i K_j^{\mathsf{T}}}{\sqrt{d_k}}\Bigr)
    """)
    st.markdown("""
    上式により計算されたアテンション行列 \\( \\alpha_{ij} \\) を隣接行列とみなし、
    networkx を使って有向グラフとして可視化します。
    """)
    
    # 2) BERTモデルとトークナイザをロード
    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    # 3) テキスト入力 (固定 or ユーザ入力)
    st.write("### テキスト入力")
    default_text = "She was a teacher for forty years and her writing has appeared in journals and anthologies."
    text = st.text_area("テキストを入力してください", value=default_text, height=100)
    
    # トークナイズ
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=16)
    
    # 4) BERTで推論し、全アテンションを取得
    with torch.no_grad():
        outputs = model(**inputs)
    # all_attentions.shape = (num_layers=12, batch_size=1, num_heads=12, seq_len=16, seq_len=16)
    all_attentions = torch.stack(outputs.attentions, dim=0)
    
    # トークンリストを表示
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    st.write("### トークン列")
    st.write(tokens)
    
    # 5) レイヤー/ヘッドを選択
    st.write("### レイヤー・ヘッド選択")
    selected_layer = st.selectbox("Select Layer (0~11)", list(range(12)), index=0)
    head_options = ["All Heads"] + list(range(12))
    selected_head_label = st.selectbox("Select Head (0~11 or All Heads)", head_options, index=0)
    selected_head = -1 if selected_head_label == "All Heads" else int(selected_head_label)
    
    # 6) ボタン押下で可視化
    if st.button("Show Attention Flow"):
        show_attention_flow_button(
            all_attentions=all_attentions,
            tokens=tokens,
            selected_layer=int(selected_layer),
            selected_head=selected_head
        )

# メイン
if __name__ == "__main__":
    render_page()
