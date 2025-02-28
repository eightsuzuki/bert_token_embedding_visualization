import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import BertTokenizer, BertModel

def render_page():
    st.title("BERT Head Analysis")
    st.markdown("このページでは、BERT の各層・各ヘッドの Attention 特性を解析し、例えばトークン間の距離に対する注目の傾向などを可視化します。")

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    text = st.text_area("解析するテキストを入力してください", "The quick brown fox jumps over the lazy dog.")

    if st.button("Head の特性を解析する"):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        attentions = outputs.attentions  # (層数, batch, ヘッド数, seq_len, seq_len)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        seq_len = len(tokens)
        num_layers = len(attentions)
        num_heads = attentions[0].size(1)

        # 各ヘッドごとに、Attention がどれだけ遠くのトークンに集中しているか（距離の重み付け平均）を計算
        avg_distances = np.zeros((num_layers, num_heads))
        for l in range(num_layers):
            attn = attentions[l][0].detach().numpy()  # shape: (heads, seq_len, seq_len)
            for h in range(num_heads):
                attn_head = attn[h]  # (seq_len, seq_len)
                # 各行（各トークン）の位置インデックスの差の絶対値
                indices = np.arange(seq_len)
                distances = np.abs(indices.reshape(-1, 1) - indices.reshape(1, -1))
                weighted_distance = np.sum(attn_head * distances) / seq_len
                avg_distances[l, h] = weighted_distance

        plt.figure(figsize=(12, 6))
        plt.imshow(avg_distances, aspect='auto', cmap='coolwarm')
        plt.colorbar(label="平均 Attention 距離")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title("層・ヘッドごとの平均 Attention 距離")
        st.pyplot(plt.gcf())
