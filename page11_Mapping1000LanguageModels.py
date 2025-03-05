import os
import pickle
import torch
import pandas as pd
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForMaskedLM, AutoTokenizer
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
# from sklearn.preprocessing import Standard

try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

def compute_log_likelihoods(model, tokenizer, texts):
    """
    各テキストに対する対数尤度（log-likelihood）を計算する。
    ここでは、クロスエントロピー損失をマスクトークンの尤度として使用。
    """
    log_likelihoods = []
    model.eval()
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
        # 出力lossは平均損失なので、トークン数をかけることで全トークンの対数尤度に戻す
        log_likelihood = -outputs.loss.item() * inputs["input_ids"].shape[1]
        log_likelihoods.append(log_likelihood)
    return torch.tensor(log_likelihoods)

def compute_log_likelihood_matrix(model_names, texts):
    """
    各モデルの対数尤度ベクトル ℓᵢ をそのまま計算し、
    L = (ℓ₁, …, ℓ_K)ᵀ ∈ ℝ^(K×N) を返す。
    """
    L_list = []
    for model_name in model_names:
        st.write(f"Loading {model_name} for L matrix ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        # 各テキストに対する対数尤度（非センタリング）を計算
        ll = compute_log_likelihoods(model, tokenizer, texts)
        L_list.append(ll.numpy())  # shape: (N,)
    L = np.stack(L_list)  # shape: (K, N)
    return L

def compute_tsne_embedding(L, random_state=42, perplexity=30):
    """
    行列 L (K x N) に対して、t-SNE を用いて2次元埋め込みを計算する。
    サンプル数が少ない場合は、perplexity を自動的に調整する。
    """
    n_samples = L.shape[0]
    if perplexity >= n_samples:
        perplexity = max(1, n_samples - 1)
    tsne = TSNE(n_components=2, random_state=random_state, metric="euclidean", perplexity=perplexity)
    embedding = tsne.fit_transform(L)
    return embedding

def compute_kl_divergence():
    """
    2KL(p_i, p_j) の近似値を計算する（bert-base-uncased と bert-base-cased の比較例）。
    ※加えて、テキストの平均バイト数で正規化し、バイトあたりのKLダイバージェンスも算出する。
    """
    model_1_name = "bert-base-uncased"
    model_2_name = "bert-base-cased"

    tokenizer_1 = AutoTokenizer.from_pretrained(model_1_name, use_fast=False)
    tokenizer_2 = AutoTokenizer.from_pretrained(model_2_name, use_fast=False)

    model_1 = AutoModelForMaskedLM.from_pretrained(model_1_name)
    model_2 = AutoModelForMaskedLM.from_pretrained(model_2_name)

    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the field of artificial intelligence.",
        "BERT is a popular model for natural language understanding tasks."
    ]
    N = len(test_texts)

    log_likelihoods_1 = compute_log_likelihoods(model_1, tokenizer_1, test_texts)
    log_likelihoods_2 = compute_log_likelihoods(model_2, tokenizer_2, test_texts)

    xi_1 = log_likelihoods_1 - log_likelihoods_1.mean()
    xi_2 = log_likelihoods_2 - log_likelihoods_2.mean()

    squared_euclidean_distance = torch.sum((xi_1 - xi_2) ** 2).item()
    kl_approx = squared_euclidean_distance / N

    # 平均テキストバイト数を計算
    avg_byte_length = sum(len(text.encode("utf-8")) for text in test_texts) / N
    kl_per_byte = kl_approx / avg_byte_length

    return kl_approx, kl_per_byte

def compute_kl_matrix(model_names, texts):
    """
    指定した複数モデルについて、各モデルの対数尤度ベクトルを
    中心化し、ペアワイズの2KL近似値（= (1/N)*||xi - xj||^2）を計算する。
    また、計算結果をテキストの平均バイト数で正規化して、バイトあたりの値も返す。
    さらに、各モデルの中心化済み対数尤度ベクトルを特徴量として返す。
    """
    N = len(texts)
    avg_byte_length = sum(len(text.encode("utf-8")) for text in texts) / N

    model2xi = {}
    for model_name in model_names:
        st.write(f"Loading {model_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        log_likelihoods = compute_log_likelihoods(model, tokenizer, texts)
        xi = log_likelihoods - log_likelihoods.mean()
        model2xi[model_name] = xi

    M = len(model_names)
    kl_matrix = torch.zeros((M, M))
    for i in range(M):
        for j in range(M):
            diff = model2xi[model_names[i]] - model2xi[model_names[j]]
            kl_matrix[i, j] = torch.sum(diff ** 2).item() / N
            
    kl_matrix_per_byte = kl_matrix / avg_byte_length
    feature_matrix = torch.stack([model2xi[m] for m in model_names]).numpy()

    return kl_matrix, kl_matrix_per_byte, feature_matrix, model_names


def load_or_compute_kl_matrix(model_names, texts, cache_dir="./cache/mapping1000", filename="kl_matrix.pkl"):
    """
    キャッシュディレクトリ "./cache/mapping1000" に、計算済みのKLダイバージェンス行列、バイト正規化行列、
    及び特徴量行列があれば読み込み、なければ計算して保存する。
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, filename)
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple) and len(data) == 4:
            kl_matrix, kl_matrix_per_byte, feature_matrix, cached_model_names = data
        elif isinstance(data, tuple) and len(data) == 3:
            kl_matrix, kl_matrix_per_byte, cached_model_names = data
            N = len(texts)
            avg_byte_length = sum(len(text.encode("utf-8")) for text in texts) / N
            kl_matrix_per_byte = kl_matrix / avg_byte_length
            feature_matrix = None
        else:
            raise ValueError("Cache file format is not recognized.")
        st.write("Cached KL行列を読み込みました。")
        return kl_matrix, kl_matrix_per_byte, feature_matrix, cached_model_names
    else:
        st.write("KL行列を計算中...")
        kl_matrix, kl_matrix_per_byte, feature_matrix, computed_model_names = compute_kl_matrix(model_names, texts)
        with open(cache_path, "wb") as f:
            pickle.dump((kl_matrix, kl_matrix_per_byte, feature_matrix, computed_model_names), f)
        st.write("KL行列の計算が完了し、キャッシュに保存しました。")
        return kl_matrix, kl_matrix_per_byte, feature_matrix, computed_model_names


def render():
    """
    Streamlit を用いた可視化・説明用の関数。
    """
    st.title("Mapping 1,000+ Language Models via the Log-Likelihood Vector")
    
    st.markdown("## 1. 計算の理論")
    st.markdown("このセクションでは、対数尤度ベクトルを用いてモデル間のKLダイバージェンスを近似する理論を、数理的な説明とともに解説します。")
    
    st.markdown("### 1. 対数尤度 (Log-Likelihood) の定義")
    st.markdown("各テキスト \\( x \\) に対して、モデル \\( p_i \\) の対数尤度は、以下の式で定義されます。")
    st.latex(r"""
    \ell_i(x) = \sum_{t=1}^{n} \log p_i(y_t \mid y_{<t})
    """)
    st.markdown("ここで、\\( y_{<t} \\) は時刻 \\( t \\) 以前のトークン列、\\( n \\) はテキストのトークン数を表します。")
    
    st.markdown("### 2. ログ尤度ベクトルの構築とセンタリング")
    st.markdown("データセット \\( D = \\{x_1, x_2, \\dots, x_N\\} \\) に対して、各モデル \\( p_i \\) のログ尤度ベクトルは次のように表されます。")
    st.latex(r"""
    \boldsymbol{\ell}_i = 
    \begin{pmatrix}
    \ell_i(x_1) \\
    \ell_i(x_2) \\
    \vdots \\
    \ell_i(x_N)
    \end{pmatrix} \in \mathbb{R}^N
    """)
    st.markdown("各モデルごとに平均ログ尤度 \\( \\bar{\\ell}_i \\) を計算し、その値を各要素から引くことで中心化したベクトル \\( \\xi_i \\) を定義します。")
    st.latex(r"""
    \xi_i = \boldsymbol{\ell}_i - \bar{\ell}_i,\quad 
    \bar{\ell}_i = \frac{1}{N}\sum_{s=1}^{N} \ell_i(x_s)
    """)
    st.markdown("さらに、全モデルの平均（列方向のセンタリング）を引くことで、ダブルセンタリングを行いますが、")
    st.latex(r"""
    q_i = \xi_i - \bar{\xi} \quad \text{(ただし } \bar{\xi} = \frac{1}{K}\sum_{i=1}^{K}\xi_i\text{)}
    """)
    st.markdown("と定義しても、モデル間の差は \\( q_i - q_j = \\xi_i - \\xi_j \\) となるため、ここでは \\( \\xi_i \\) をそのまま用いることが可能です。")
    
    st.markdown("### 3. KLダイバージェンスの定義とその近似")
    st.markdown("2つのモデル \\( p_i \\) と \\( p_j \\) のKLダイバージェンスは、以下のように定義されます。")
    st.latex(r"""
    KL(p_i \,\|\, p_j) = \mathbb{E}_{x \sim p_i}\left[ \log \frac{p_i(x)}{p_j(x)} \right]
    """)
    st.markdown("対数尤度の差を用いて表現すると、")
    st.latex(r"""
    KL(p_i \,\|\, p_j) = \mathbb{E}_{x \sim p_i}\left[ \ell_i(x) - \ell_j(x) \right]
    """)
    st.markdown("さらに、データセット \\( D \\) が十分大きい場合、以下の近似が成り立つことが示されています。")
    st.latex(r"""
    2KL(p_i, p_j) \approx \operatorname{Var}_{x \sim D}\left( \ell_i(x) - \ell_j(x) \right)
    """) 
    st.markdown("中心化された対数尤度の差 \\( \\xi_i - \\xi_j \\) の2乗和を \\( N \\) で割ると、サンプル分散の推定値となり、")
    st.latex(r"""
    2KL(p_i, p_j) \approx \frac{1}{N}\|\xi_i - \xi_j\|^2
    """)
    
    st.markdown("### 3.5 実験間比較のためのバイト正規化")
    st.markdown(
        """
        テキスト生成におけるKLダイバージェンスは、テキスト中のトークン数やバイト数に依存します。
        異なるトークナイザーを使用する場合、平均テキスト長（バイト数）で正規化することで、
        バイトあたりのKLダイバージェンスを得ることができます。
        例えば、もし KL(p_i, p_j) = 1000 であれば、平均テキスト長が 972.3 バイトの場合、
        バイトあたりのKLダイバージェンスは 1000 / 972.3 ≈ 1.028 nats（約 1.484 bits）となります。
        """
    )
    
    st.markdown("### 4. コード実装との対応")
    st.markdown("コードでは、")
    st.markdown("- 各テキストについての対数尤度を計算し、シーケンス長で乗じて全トークン分に戻しています。")
    st.markdown("- モデルごとに平均を引くことで中心化した \\( \\xi_i \\) を得ています。")
    st.markdown("- 2つのモデル間の距離 \\( \\|\\xi_i - \\xi_j\\|^2 \\) を \\( N \\) で割ることで、テキストあたりのKLダイバージェンス近似値を計算しています。")
    st.markdown("- さらに、テキストの平均バイト数で正規化することで、バイトあたりのKLダイバージェンスも算出しています。")
    st.latex(r"""
    2KL(p_i, p_j) \approx \frac{1}{N}\|\xi_i - \xi_j\|^2 \quad \Rightarrow \quad \frac{1}{N}\|\xi_i - \xi_j\|^2 \Big/ \text{(平均バイト数)}
    """)
    
    st.markdown("### まとめ")
    st.markdown("以上の理論に基づき、コードは対数尤度ベクトルのユークリッド距離を用いて、")
    st.markdown("モデル間のKLダイバージェンス近似値（テキストあたりおよびバイトあたり）を実現しています。")
    st.markdown("※ サンプル数 \\( N \\) やモデルの性質（例えば BERT のMasked LM である点）によっては、")
    st.markdown("理論上の近似精度に影響が出る可能性があるため、実験設定に合わせた注意が必要です。")
    
    st.markdown("## 2. 計算結果（bert-base-uncased と bert-base-cased の比較）")
    kl_value, kl_per_byte = compute_kl_divergence()
    st.markdown(f"### 計算された 2KL(p_i, p_j) の近似値（テキストあたり）: `{kl_value:.6f}`")
    st.markdown(f"### 計算された 2KL(p_i, p_j) の近似値（バイトあたり）: `{kl_per_byte:.6f}`")
    
    st.markdown("## 3. 複数モデル間の比較")
    st.markdown("以下では、10種類程度のMasked LMモデルについて、各モデルの対数尤度ベクトルから求めたKLダイバージェンス近似値のペアワイズ距離を計算し、比較します。")
    
    # 新しいモデルリスト（ChatGPT, Llama, DeepSeek, Claude, Gemini などの名称を含む候補）
    model_names = [
        # "gpt2",                                  # ChatGPT の代替候補
        # "decapoda-research/llama-7b-hf",           # Llama の代替候補（アクセス権がない場合はスキップ）
        # "deepseek-ai/deepseek-llm-7b-base",         # DeepSeek
        # "bigscience/bloom",                        # Claude の代替候補
        # "bigscience/bloomz-560m",                  # Gemini の代替候補
        "distilbert-base-uncased",
        "bert-base-uncased",
        "roberta-base",
        "albert-base-v2",
        "google/electra-base-discriminator"
    ]

    # サンプルテキスト（前述と同じ）
    test_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the field of artificial intelligence.",
        "BERT is a popular model for natural language understanding tasks."
    ]
    
    # キャッシュを利用してKL行列と特徴量行列を取得
    kl_matrix, kl_matrix_per_byte, feature_matrix, cached_models = load_or_compute_kl_matrix(model_names, test_texts)
    
    # DataFrameに変換して表示（テキストあたり）
    df = pd.DataFrame(kl_matrix.numpy(), index=cached_models, columns=cached_models)
    st.markdown("### モデル間のKLダイバージェンス近似値（テキストあたり）")
    df_styled = df.style.background_gradient(cmap="Blues")
    st.dataframe(df_styled)
    
    # DataFrameに変換して表示（バイトあたり）
    df_norm = pd.DataFrame(kl_matrix_per_byte.numpy(), index=cached_models, columns=cached_models)
    st.markdown("### モデル間のKLダイバージェンス近似値（バイトあたり）")
    df_norm_styled = df_norm.style.background_gradient(cmap="Blues")
    st.dataframe(df_norm_styled)
    

    st.markdown("## 6. L行列の t‑SNE プロット")
    st.markdown(
        """
        ここでは、各モデルの対数尤度ベクトル ℓᵢ をそのままスタックして得られる
        L = (ℓ₁, …, ℓ_K)ᵀ ∈ ℝ^(K×N) に対して t‑SNE を適用し、モデル間の類似性を2次元にプロットします。
        """
    )
    
    # L行列の計算
    L = compute_log_likelihood_matrix(model_names, test_texts)
    st.write("L行列の形状:", L.shape)
    
    # t-SNE による埋め込み（デフォルト t-SNE）
    embedding = compute_tsne_embedding(L)
    
    # プロット
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c="blue", alpha=0.7)
    for i, model_name in enumerate(model_names):
        ax.annotate(model_name, (embedding[i, 0], embedding[i, 1]), textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_title("t‑SNE によるモデル埋め込み (L 行列)")
    ax.set_xlabel("次元1")
    ax.set_ylabel("次元2")
    st.pyplot(fig)
    
    st.markdown("## 5. 考察")
    st.markdown(
        """
        - 各モデル間のKLダイバージェンス近似値は、対数尤度ベクトルの中心化後のユークリッド距離として計算される。
        - テキストあたりの値に加え、平均テキスト長（バイト数）で正規化することで、異なる実験間でも比較可能なバイトあたりのKLダイバージェンスが得られる。
        - 次元削減により、各モデルの相関構造を2次元空間上に視覚化することができ、クラスタリングの傾向が確認できる。
        - キャッシュ機構により、一度計算した結果は `./cache/mapping1000` に保存され、再実行時の計算負荷が軽減される。
        """
    )

if __name__ == "__main__":
    render()
