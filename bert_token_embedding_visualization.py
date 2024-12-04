import streamlit as st
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from sklearn.decomposition import PCA, FastICA
import umap
import pandas as pd
import plotly.express as px

# タイトル
st.title("Token Embedding Analysis and Visualization")

# ページ管理
page = st.sidebar.selectbox(
    "Choose a Page",
    ["Dimensionality Reduction", "Internal Representations"]
)

# モデルとトークナイザーのロード
@st.cache_resource
def load_model_and_tokenizer():
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states=True)
    return tokenizer, model

tokenizer, model = load_model_and_tokenizer()

# ページ 1: 次元削減技術の比較
if page == "Dimensionality Reduction":
    # テキスト入力
    # テキスト入力 (ファイルから読み込み)
    with open("input_text.txt", "r", encoding="utf-8") as file:
        text = file.read()
    
    st.text_area("Input Text from File", text, height=300)

    # トークン化と埋め込み取得
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokens_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0))  # トークン化されたテキストを取得
    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state.squeeze(0).detach().numpy()

    # トークン化結果のデバッグ
    print("Tokenized Text:")
    for idx, token in enumerate(tokens_text):
        print(f"Index: {idx}, Token: {token}")


    # 英語に対応するトークンを選別（カテゴリごとに分類）
    dog_related_tokens = [
        ["dog", "puppy", "canine", "hound", "pets", "doggy", "puppies", "pup"],  # 一般的な犬関連単語
        ["chihuahua", "golden", "retriever", "shiba", "inu", "labrador", "shepherd"],  # 人気の犬種
        ["dachshund", "collie", "bulldog", "terrier", "beagle", "poodle", "akita"],  # 他の有名な犬種
        ["guide", "therapy", "police", "working", "service"]  # 犬の役割
    ]
    # トークン結合ロジックを追加
    def merge_tokens(tokens_text, embeddings):
        merged_tokens = []
        merged_embeddings = []
        temp_token = ""
        temp_embedding = []

        for idx, token in enumerate(tokens_text):
            if token.startswith("##"):
                temp_token += token[2:]  # "##"を除いて結合
                temp_embedding.append(embeddings[idx])
            else:
                if temp_token:
                    # 結合したトークンを確定
                    merged_tokens.append(temp_token)
                    merged_embeddings.append(torch.mean(torch.tensor(temp_embedding), dim=0).numpy())
                temp_token = token
                temp_embedding = [embeddings[idx]]
        if temp_token:  # 最後のトークンを確定
            merged_tokens.append(temp_token)
            merged_embeddings.append(torch.mean(torch.tensor(temp_embedding), dim=0).numpy())

        return merged_tokens, merged_embeddings

    # トークンと埋め込みの結合
    merged_tokens, merged_embeddings = merge_tokens(tokens_text, embeddings)

    # 犬関連トークンを含まないトークンをunrelated_tokensとして自動生成
    unrelated_tokens = [
        token.lower() for token in set(merged_tokens) 
        if not any(word in token.lower() for category in dog_related_tokens for word in category)
    ]

    # トークン化後の一致処理を更新
    dog_indices = []
    dog_labels = []
    for i, category in enumerate(dog_related_tokens):
        indices = [j for j, token in enumerate(merged_tokens) if any(word in token.lower() for word in category)]
        dog_indices.extend(indices)
        dog_labels.extend([f"Category {i+1}"] * len(indices))

    unrelated_indices = [
        i for i, token in enumerate(merged_tokens)
        if token.lower() in unrelated_tokens and i not in dog_indices
    ]

    # トークン化結果のデバッグ（結合後のトークン）
    print("Merged Tokenized Text:")
    for idx, token in enumerate(merged_tokens):
        print(f"Index: {idx}, Token: {token}")


    method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "ICA", "ISOMAP", "HessianLLE", "t-SNE", "UMAP"]  # "LLE" を "HessianLLE" に変更
    )

    # 次元削減を実行する関数
    def reduce_dimensions(method, embeddings, n_neighbors=10):
        # リストを numpy.ndarray に変換
        embeddings = np.array(embeddings)

        if method == "PCA":
            reducer = PCA(n_components=2)
        elif method == "ICA":
            reducer = FastICA(n_components=2)
        elif method == "ISOMAP":
            reducer = Isomap(n_components=2)
        elif method == "HessianLLE":
            if n_neighbors <= (2 * (2 + 3) // 2):  # n_neighborsの下限チェック
                raise ValueError(f"n_neighbors must be > {2 * (2 + 3) // 2} for HessianLLE with n_components=2.")
            reducer = LocallyLinearEmbedding(n_components=2, method='hessian', n_neighbors=n_neighbors, eigen_solver='dense')
        elif method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        elif method == "UMAP":
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)

    # HessianLLE用のn_neighborsを指定
    if method == "HessianLLE":
        n_neighbors = st.slider("Select n_neighbors for HessianLLE", min_value=6, max_value=30, value=10, step=1)
    else:
        n_neighbors = None

    # 次元削減の実行
    selected_indices = dog_indices + unrelated_indices
    selected_embeddings = [merged_embeddings[idx] for idx in selected_indices]
    reduced_embeddings = reduce_dimensions(method, selected_embeddings)

    # データフレームの作成
    labels = dog_labels + ["Unrelated"] * len(unrelated_indices)
    tokens = [tokens_text[idx] for idx in selected_indices]
    colors = {
        "Category 1": "darkred",
        "Category 2": "red",
        "Category 3": "orange",
        "Category 4": "lightcoral",
        "Unrelated": "blue"
    }
    df = pd.DataFrame({
        "x": reduced_embeddings[:, 0],
        "y": reduced_embeddings[:, 1],
        "Label": labels,
        "Token": tokens
    })

    # Plotlyでプロット
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="Label",
        color_discrete_map=colors,  # カスタムカラーを指定
        hover_data={"Token": True},
        labels={"x": "Dimension 1", "y": "Dimension 2"},
        title=f"{method} Visualization of Dog-Related and Unrelated Tokens"
    )
    fig.update_traces(marker=dict(opacity=0.8)) 

    # プロットの表示
    st.plotly_chart(fig)


# ページ 2: 内部表現の表示
elif page == "Internal Representations":
    # テキスト入力 (ファイルから読み込み)
    with open("input_text.txt", "r", encoding="utf-8") as file:
        text = file.read()

    st.text_area("Input Text from File (Page 2)", text, height=300)
    
    tokens = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    tokens_text = tokenizer.convert_ids_to_tokens(tokens["input_ids"].squeeze(0))
    outputs = model(**tokens)

    # 隠れ層の埋め込みを取得
    hidden_states = outputs.hidden_states  # 各層の埋め込み
    layer_index = st.slider("Select Layer", 0, len(hidden_states) - 1, 0)

    # 選択した層の埋め込み
    layer_embeddings = hidden_states[layer_index].squeeze(0).detach().numpy()

    # トークンと対応する埋め込みを表示
    st.write("Token Embeddings at Selected Layer:")
    for idx, token in enumerate(tokens_text):
        st.write(f"Token: {token}, Embedding: {layer_embeddings[idx]}")
