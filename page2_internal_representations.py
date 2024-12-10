import streamlit as st
import numpy as np

# ページ 2: 内部表現の表示
def render_page(text, tokens_text, embeddings, model):
    st.text_area("Input Text", text, height=300)

    layer_index = st.slider("Select Layer", 0, 11, 0)
    hidden_states = model(**{"input_ids": torch.tensor([tokens_text]), "attention_mask": torch.ones_like(tokens_text)}).hidden_states
    layer_embeddings = hidden_states[layer_index].squeeze(0).detach().numpy()

    st.write("Token Embeddings at Selected Layer:")
    for idx, token in enumerate(tokens_text):
        st.write(f"Token: {token}, Embedding: {layer_embeddings[idx]}")
