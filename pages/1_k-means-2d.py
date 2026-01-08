import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans
import io

# 한글 폰트 설정 (NanumGothic)
font_path = "fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(font_path)
rc('font', family='NanumGothic')

st.set_page_config(page_title="K-means 2D 커스텀 데이터", layout="wide")
st.title("📊 K-means Clustering - 2D 커스텀 데이터 분석")

# 세션 상태 초기화
if 'custom_data' not in st.session_state:
    st.session_state.custom_data = pd.DataFrame(columns=['이름', 'X1', 'X2'])
if 'kmeans_model' not in st.session_state:
    st.session_state.kmeans_model = None
if 'clusters' not in st.session_state:
    st.session_state.clusters = None

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📥 데이터 입력", "📈 최적 K값 분석", "🎯 군집 시각화"])

# ============================================
# 탭 1: 데이터 입력
# ============================================
with tab1:
    st.header("2차원 데이터 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 스프레드시트에서 붙여넣기")
        st.info("💡 엑셀이나 구글시트에서 3개 열의 데이터를 복사한 후 아래에 붙여넣기\n첫 줄: 라벨 이름, 두 번째 줄부터 데이터\n예) 상품\t수량\t판매액")
        
        pasted_data = st.text_area(
            "데이터 붙여넣기 (탭으로 구분된 형식)",
            height=250,
            placeholder="상품\t수량\t판매액\n상품A\t1.5\t2.3\n상품B\t2.1\t3.2\n...",
            label_visibility="collapsed"
        )
        
        if st.button("✅ 붙여넣은 데이터 로드", key="paste_load"):
            try:
                if pasted_data.strip():
                    # 탭이나 공백으로 구분된 데이터 파싱
                    from io import StringIO
                    df = pd.read_csv(StringIO(pasted_data), sep='\t|\s+', engine='python')
                    
                    # 첫 3개 열만 사용
                    if len(df.columns) >= 3:
                        df = df.iloc[:, :3]
                        # 원래 열 이름 보존 (또는 사용자 정의 이름 사용 가능)
                        col_names = df.columns.tolist()
                        # 2, 3번째 열을 숫자로 변환
                        df[col_names[1]] = df[col_names[1]].astype(float)
                        df[col_names[2]] = df[col_names[2]].astype(float)
                        st.session_state.custom_data = df
                        st.success(f"✅ {len(df)}개의 데이터 포인트가 로드되었습니다!")
                        st.info(f"📌 열 이름: {col_names[0]} (라벨), {col_names[1]} (X축), {col_names[2]} (Y축)")
                    else:
                        st.error("❌ 최소 3개의 열이 필요합니다.")
            except Exception as e:
                st.error(f"❌ 데이터 파싱 오류: {str(e)}")
    
    with col2:
        st.subheader("2. 직접 입력")
        
        num_points = st.number_input("데이터 포인트 개수", min_value=1, max_value=100, value=5)
        
        # 동적 입력 필드
        data_input = []
        cols = st.columns(3)
        
        for i in range(num_points):
            col_idx = i % 3
            with cols[col_idx]:
                name = st.text_input(f"Point {i+1} - 이름", value=f"Data_{i+1}", key=f"name_{i}")
                x1 = st.number_input(f"Point {i+1} - X1", value=0.0, key=f"x1_{i}")
                x2 = st.number_input(f"Point {i+1} - X2", value=0.0, key=f"x2_{i}")
                data_input.append([name, x1, x2])
        
        if st.button("✅ 직접 입력 데이터 저장", key="manual_load"):
            df = pd.DataFrame(data_input, columns=['이름', 'X1', 'X2'])
            st.session_state.custom_data = df
            st.success(f"✅ {len(df)}개의 데이터 포인트가 저장되었습니다!")
    
    # 데이터 미리보기
    if not st.session_state.custom_data.empty:
        st.subheader("📋 데이터 미리보기")
        st.dataframe(st.session_state.custom_data, use_container_width=True)
        
        # 기본 통계
        x_col = st.session_state.custom_data.columns[1]
        y_col = st.session_state.custom_data.columns[2]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 데이터 포인트", len(st.session_state.custom_data))
        with col2:
            st.metric(f"{x_col} 평균", f"{st.session_state.custom_data[x_col].mean():.2f}")
        with col3:
            st.metric(f"{x_col} 범위", f"{st.session_state.custom_data[x_col].max() - st.session_state.custom_data[x_col].min():.2f}")
        with col4:
            st.metric(f"{y_col} 평균", f"{st.session_state.custom_data[y_col].mean():.2f}")
        
        # 데이터 다운로드
        csv = st.session_state.custom_data.to_csv(index=False)
        st.download_button(
            label="📥 CSV로 다운로드",
            data=csv,
            file_name="kmeans_data.csv",
            mime="text/csv"
        )

# ============================================
# 탭 2: 최적 K값 분석 (Elbow Method)
# ============================================
with tab2:
    st.header("최적 군집 수 찾기 (Elbow Method)")
    
    if st.session_state.custom_data.empty:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 입력해주세요.")
    else:
        # K값 범위 설정
        col1, col2 = st.columns(2)
        with col1:
            max_k = st.slider("최대 K값", min_value=3, max_value=15, value=10)
        with col2:
            st.info(f"K값을 1부터 {max_k}까지 분석합니다.")
        
        if st.button("🔍 Inertia 계산", key="calculate_inertia"):
            with st.spinner("계산 중..."):
                # 열 이름 가져오기
                x_col = st.session_state.custom_data.columns[1]
                y_col = st.session_state.custom_data.columns[2]
                
                inertias = []
                k_range = range(1, max_k + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(st.session_state.custom_data[[x_col, y_col]])
                    inertias.append(kmeans.inertia_)
                
                # 그래프 그리기
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
                ax.set_xlabel('클러스터 수 (K)', fontsize=12)
                ax.set_ylabel('Inertia (클러스터 내 거리 합)', fontsize=12)
                ax.set_title('Elbow Method를 통한 최적 K값 찾기', fontsize=14, fontweight='bold')
                ax.grid(True, alpha=0.3)
                ax.set_xticks(k_range)
                
                st.pyplot(fig)
                
                # Inertia 값 표시
                st.subheader("📊 Inertia 값 상세")
                inertia_df = pd.DataFrame({
                    'K': list(k_range),
                    'Inertia': inertias,
                    '감소율 (%)': ['-'] + [f"{(inertias[i-1] - inertias[i]) / inertias[i-1] * 100:.2f}%" 
                                          for i in range(1, len(inertias))]
                })
                st.dataframe(inertia_df, use_container_width=True)
                
                st.success("✅ Inertia 계산 완료! '군집 시각화' 탭에서 최적 K값을 선택하세요.")

# ============================================
# 탭 3: 군집 시각화
# ============================================
with tab3:
    st.header("K-means 군집 시각화")
    
    if st.session_state.custom_data.empty:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 입력해주세요.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ 설정")
            optimal_k = st.slider(
                "최적 K값 선택",
                min_value=1,
                max_value=min(10, len(st.session_state.custom_data) - 1),
                value=3
            )
            
            run_clustering = st.button("🚀 K-means 실행", key="run_clustering")
        
        if run_clustering or st.session_state.kmeans_model is not None:
            with col2:
                with st.spinner("클러스터링 중..."):
                    # 열 이름 가져오기
                    label_col = st.session_state.custom_data.columns[0]
                    x_col = st.session_state.custom_data.columns[1]
                    y_col = st.session_state.custom_data.columns[2]
                    
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(st.session_state.custom_data[[x_col, y_col]])
                    
                    st.session_state.kmeans_model = kmeans
                    st.session_state.clusters = clusters
                    
                    # 시각화
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    # 데이터 포인트와 클러스터 색칠
                    scatter = ax.scatter(
                        st.session_state.custom_data[x_col],
                        st.session_state.custom_data[y_col],
                        c=clusters,
                        cmap='viridis',
                        s=100,
                        alpha=0.6,
                        edgecolors='black',
                        linewidth=0.5
                    )
                    
                    # 각 데이터 포인트에 라벨 표시
                    for idx, row in st.session_state.custom_data.iterrows():
                        ax.annotate(
                            row[label_col],
                            (row[x_col], row[y_col]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7
                        )
                    
                    # 센트로이드 표시
                    centroids = kmeans.cluster_centers_
                    ax.scatter(
                        centroids[:, 0],
                        centroids[:, 1],
                        c='red',
                        marker='*',
                        s=500,
                        edgecolors='black',
                        linewidth=2,
                        label='Centroids'
                    )
                    
                    ax.set_xlabel(x_col, fontsize=12)
                    ax.set_ylabel(y_col, fontsize=12)
                    ax.set_title(f'K-means Clustering (K={optimal_k})', fontsize=14, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    plt.colorbar(scatter, ax=ax, label='Cluster')
                    st.pyplot(fig)
                    
                    # 클러스터 통계
                    st.subheader("📊 클러스터 통계")
                    
                    cluster_stats = []
                    for i in range(optimal_k):
                        mask = clusters == i
                        count = mask.sum()
                        centroid = centroids[i]
                        members = st.session_state.custom_data[mask][label_col].tolist()
                        cluster_stats.append({
                            '클러스터': i,
                            '데이터 포인트 수': count,
                            f'Centroid {x_col}': f"{centroid[0]:.2f}",
                            f'Centroid {y_col}': f"{centroid[1]:.2f}",
                            '비율': f"{count/len(clusters)*100:.1f}%",
                            '포함된 항목': ', '.join(members)
                        })
                    
                    stats_df = pd.DataFrame(cluster_stats)
                    st.dataframe(stats_df, use_container_width=True)
                    
                    # 모델 성능 지표
                    st.subheader("📈 모델 성능 지표")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Inertia", f"{kmeans.inertia_:.2f}")
                    with col2:
                        st.metric("클러스터 수", optimal_k)
                    with col3:
                        st.metric("총 데이터 포인트", len(st.session_state.custom_data))
                    
                    # 각 클러스터 상세 정보
                    st.subheader("🔍 클러스터별 상세 정보")
                    
                    for i in range(optimal_k):
                        with st.expander(f"클러스터 {i} ({(clusters == i).sum()}개 포인트)"):
                            cluster_data = st.session_state.custom_data[clusters == i].copy()
                            cluster_data['군집'] = i
                            st.dataframe(cluster_data.reset_index(drop=True), use_container_width=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(f"{x_col} 평균", f"{cluster_data[x_col].mean():.2f}")
                            with col2:
                                st.metric(f"{y_col} 평균", f"{cluster_data[y_col].mean():.2f}")
