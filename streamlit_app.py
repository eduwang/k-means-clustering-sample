

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 한글 폰트 설정 (NanumGothic)
font_path = "fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(font_path)
rc('font', family='NanumGothic')

st.set_page_config(page_title="K-means Clustering Demo", layout="wide")
st.title("🔎 K-means Clustering 비지도 학습 데모")

# 사이드바: 데이터셋 선택
st.sidebar.header("데이터셋 선택")
dataset_name = st.sidebar.selectbox(
    "데이터셋 예시",
    ["make_blobs (샘플)", "Iris (Kaggle)", "Mall Customers (Kaggle)"]
)

# 데이터셋 로딩 함수
def load_dataset(name):
    if name == "make_blobs (샘플)":
        X, y = make_blobs(n_samples=300, centers=4, random_state=42)
        return pd.DataFrame(X, columns=["x1", "x2"])
    elif name == "Iris (Kaggle)":
        url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv"
        df = pd.read_csv(url)
        return df[["sepal_length", "sepal_width"]]
    elif name == "Mall Customers (Kaggle)":
        # 프로젝트 내 data/Mall_Customers.csv 파일에서 로드
        try:
            df = pd.read_csv("data/Mall_Customers.csv")
            return df[["Annual Income (k$)", "Spending Score (1-100)"]]
        except Exception as e:
            st.error("Mall Customers 데이터셋을 찾을 수 없습니다. data/Mall_Customers.csv 파일을 프로젝트에 추가해 주세요.")
            return pd.DataFrame()
    else:
        return pd.DataFrame()

data = load_dataset(dataset_name)

# K 값 입력
st.sidebar.header("K 값 설정")
K = st.sidebar.slider("클러스터 수 (K)", min_value=2, max_value=10, value=3)

# 단계별 centroid 이동 시각화
st.subheader("Centroid 이동 과정 시각화")
step = st.slider("K-means 단계 (Iteration)", min_value=1, max_value=10, value=1)

def plot_kmeans_steps(data, K, step):
    kmeans = KMeans(n_clusters=K, init="random", n_init=1, max_iter=step, random_state=42)
    kmeans.fit(data)
    centers = kmeans.cluster_centers_
    labels = kmeans.labels_
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.scatter(data.iloc[:,0], data.iloc[:,1], c=labels, cmap="viridis", alpha=0.6)
    ax.scatter(centers[:,0], centers[:,1], c="red", marker="X", s=200, label="Centroids")

    ax.set_xlabel(data.columns[0])
    ax.set_ylabel(data.columns[1])
    ax.legend()
    ax.set_title(f"Step {step}: Centroid 위치")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)

plot_kmeans_steps(data, K, step)

# Elbow 그래프 버튼
if st.button("Elbow 그래프 보기 (Inertia vs K)"):
    inertias = []
    K_range = range(1, 11)
    for k in K_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=42)
        km.fit(data)
        inertias.append(km.inertia_)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(K_range, inertias, marker="o")
    ax.set_xlabel("K (클러스터 수)")
    ax.set_ylabel("Inertia")
    ax.set_title("Elbow Method: 최적의 K 찾기")
    plt.tight_layout()
    st.pyplot(fig, use_container_width=False)
