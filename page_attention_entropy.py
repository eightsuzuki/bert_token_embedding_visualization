import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel

def render_page():
    st.title("BERT Attention Entropy Analysis")
    st.markdown("このページでは、各層・各ヘッドの Attention 分布のエントロピーを計算し、Attention の分布の鋭さや広がりを定量的に示します。")

    model_name = "bert-base-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name, output_attentions=True)
    model.eval()

    text = st.text_area("解析するテキストを入力してください", "The quick brown fox jumps over the lazy dog.")

    if st.button("Attention エントロピーを計算する"):
        inputs = tokenizer(text, return_tensors="pt")
        outputs = model(**inputs)
        attentions = outputs.attentions  # (層数, batch, ヘッド数, seq_len, seq_len)
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        seq_len = len(tokens)
        num_layers = len(attentions)
        num_heads = attentions[0].size(1)

        # 確率分布のエントロピーを計算する関数
        def entropy(p):
            p = np.clip(p, 1e-12, 1.0)
            return -np.sum(p * np.log(p), axis=-1)

        entropy_values = np.zeros((num_layers, num_heads))
        for l in range(num_layers):
            attn = attentions[l][0].detach().numpy()  # (heads, seq_len, seq_len)
            for h in range(num_heads):
                attn_head = attn[h]  # (seq_len, seq_len)
                # 各トークンごとのエントロピーを計算し、平均値を求める
                entropies = entropy(attn_head)
                avg_entropy = np.mean(entropies)
                entropy_values[l, h] = avg_entropy

        plt.figure(figsize=(12, 6))
        sns.heatmap(entropy_values, annot=True, fmt=".2f", cmap="viridis")
        plt.xlabel("Head")
        plt.ylabel("Layer")
        plt.title("層・ヘッドごとの平均 Attention エントロピー")
        st.pyplot(plt.gcf())
