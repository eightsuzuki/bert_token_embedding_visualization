import streamlit as st
import numpy as np
from sklearn.decomposition import FastICA
import pandas as pd
import plotly.express as px
import hashlib
import json
import os
import torch

import sys
sys.dont_write_bytecode = True


def render_page(text, model, tokenizer):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)

    # Get the hidden states from the model
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states

    # Create directory if it doesn't exist
    os.makedirs('cache/internal/embedding', exist_ok=True)
    tokens_text = []
    embeddings = []

    # for i in range(0, len(text), 512):
    #     # Get chunks of text based on length of 512 characters
    #     chunk = text[i:i+512]
    #     # Tokenize the chunk and return as PyTorch tensors with truncation and padding if needed
    #     tokens = tokenizer(chunk, return_tensors="pt", truncation=True, padding=True)
    #     # Convert token IDs to corresponding tokens and add to tokens_text list
    #     tokens_text.extend(tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0)))
    #     # Get the output from the model
    #     outputs = model(**tokens, output_hidden_states=True)
    #     # Get the hidden states from the model's output
    #     hidden_states = outputs.hidden_states
    #     # Append the last hidden state to embeddings list
    #     embeddings.append(hidden_states[-1].squeeze(0).detach().numpy())

    # # Save each hidden state to a separate JSON file
    # for i, hidden_state in enumerate(hidden_states):
    #     hidden_state_np = hidden_state.squeeze().detach().numpy()
    #     hidden_state_list = hidden_state_np.tolist()
    #     file_path = f'cache/internal/embedding/hidden_state_{i}.json'
    #     st.write(f"Hidden State {i} Shape:", hidden_state_np.shape, len(hidden_state_np))
    #     with open(file_path, 'w') as f:
    #         json.dump(hidden_state_list, f)

    # # Display a part of the hidden states
    # st.write("Hidden State 0 Sample:", hidden_states[0].squeeze().numpy()[:5])
    # st.text_area("Input Text", text, height=300)


