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
    text = st.text_area("Enter Text", """
    Dogs are said to be man's best friend. Dogs are loyal and playful, and loved by many people.
    Additionally, dogs are active as police dogs, guide dogs, and therapy dogs. They are known for their ability
    to detect scents, track missing persons, and provide comfort to those in need. Dogs have been companions to humans
    for thousands of years, playing significant roles in various cultures and histories.

    Dog breeds include Chihuahua, Golden Retriever, Shiba Inu, Labrador Retriever, German Shepherd, and Dachshund.
    Each breed has its unique characteristics. For example, the Golden Retriever is renowned for its friendly and
    intelligent demeanor, while the Shiba Inu is celebrated for its independence and spirited personality.
    Labrador Retrievers are popular for their reliability and use in guide dog programs, and Dachshunds are admired for
    their playful and bold nature despite their small size.

    In rural areas, dogs are often employed to protect livestock, guard properties, and assist in hunting. Breeds such as
    the Border Collie, Akita Inu, and Great Pyrenees excel in these roles due to their intelligence and resilience.
    In urban settings, dogs are cherished as household pets. Parks, cafes, and even office spaces are increasingly
    becoming dog-friendly, highlighting their integration into modern human life.

    Beyond dogs, the animal kingdom offers an incredible variety of species. Cats, for instance, are independent and
    curious animals often kept as pets. Birds, such as parrots and canaries, captivate humans with their vibrant colors
    and melodious songs. Fish, including goldfish and bettas, are admired for their serene presence and low-maintenance care.
    Reptiles, such as turtles, lizards, and snakes, attract those fascinated by exotic pets.

    Wild animals, like lions, tigers, elephants, and bears, play essential roles in their ecosystems. Elephants, often
    referred to as ecosystem engineers, help shape their environments by creating water holes and spreading seeds over
    vast distances. Tigers, as apex predators, ensure the balance of prey populations in forests, maintaining biodiversity.

    Marine life, including whales, dolphins, sharks, and jellyfish, showcases the immense diversity of the ocean.
    Whales, with their complex communication and migratory patterns, have fascinated scientists for generations.
    Dolphins are known for their playful behavior and intelligence, often interacting with humans in unique ways.

    The animal kingdom also includes insects like bees and butterflies, which play critical roles in pollination,
    supporting agriculture and plant diversity. Amphibians, such as frogs and salamanders, thrive in moist environments
    and serve as vital indicators of ecological health.

    Efforts to conserve endangered species are more important than ever. Animals such as pandas, rhinos, and cheetahs
    face threats due to habitat destruction, climate change, and poaching. Conservation programs aim to protect these
    species and raise awareness about their significance to global ecosystems.

    Throughout history, animals have inspired art, literature, and cultural traditions. From ancient cave paintings
    depicting wild animals to modern-day stories of heroic pets, the bond between humans and animals has always been profound.
    As we continue to learn about the natural world, fostering a deeper understanding of animals and their roles in
    ecosystems will be essential for achieving a sustainable future.

    From the loyal dog by your side to the majestic eagle soaring in the sky, animals enrich our lives and teach us
    valuable lessons about resilience, adaptation, and coexistence. The beauty and complexity of the animal kingdom
    are treasures worth protecting for generations to come.
    """)



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


    # 次元削減手法を選択
    method = st.selectbox(
        "Select Dimensionality Reduction Method",
        ["PCA", "ICA", "ISOMAP", "LLE", "t-SNE", "UMAP"]
    )


    # 次元削減を実行する関数
    def reduce_dimensions(method, embeddings):
        # リストを numpy.ndarray に変換
        embeddings = np.array(embeddings)

        if method == "PCA":
            reducer = PCA(n_components=2)
        elif method == "ICA":
            reducer = FastICA(n_components=2)
        elif method == "ISOMAP":
            reducer = Isomap(n_components=2)
        elif method == "LLE":
            reducer = LocallyLinearEmbedding(n_components=2)
        elif method == "t-SNE":
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
        elif method == "UMAP":
            reducer = umap.UMAP(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unknown method: {method}")

        return reducer.fit_transform(embeddings)

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
    text = st.text_area("Enter Text", "Dogs are loyal and intelligent animals.")
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
