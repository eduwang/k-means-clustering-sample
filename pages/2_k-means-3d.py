import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

# 한글 폰트 설정 (NanumGothic)
font_path = "fonts/NanumGothic-Regular.ttf"
font_manager.fontManager.addfont(font_path)
rc('font', family='NanumGothic')

st.set_page_config(page_title="K-means 3D 커스텀 데이터", layout="wide")
st.title("📊 K-means Clustering - 3D 커스텀 데이터 분석")

# 세션 상태 초기화
if 'custom_data_3d' not in st.session_state:
    st.session_state.custom_data_3d = pd.DataFrame(columns=['이름', 'X1', 'X2', 'X3'])
if 'kmeans_model_3d' not in st.session_state:
    st.session_state.kmeans_model_3d = None
if 'clusters_3d' not in st.session_state:
    st.session_state.clusters_3d = None

# 탭 구성
tab1, tab2, tab3 = st.tabs(["📥 데이터 입력", "📈 최적 K값 분석", "🎯 3D 군집 시각화"])

# ============================================
# 탭 1: 데이터 입력
# ============================================
with tab1:
    st.header("3차원 데이터 입력")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. 스프레드시트에서 붙여넣기")
        st.info("💡 엑셀이나 구글시트에서 4개 열의 데이터를 복사한 후 아래에 붙여넣기\n첫 줄: 라벨 이름, 그 다음 3개 열의 수치 데이터\n예) 상품\t수량\t판매액\t만족도")
        
        pasted_data = st.text_area(
            "데이터 붙여넣기 (탭으로 구분된 형식)",
            height=250,
            placeholder="상품\t수량\t판매액\t만족도\n상품A\t1.5\t2.3\t4.2\n상품B\t2.1\t3.2\t3.8\n...",
            label_visibility="collapsed"
        )
        
        if st.button("✅ 붙여넣은 데이터 로드", key="paste_load_3d"):
            try:
                if pasted_data.strip():
                    # 탭이나 공백으로 구분된 데이터 파싱
                    from io import StringIO
                    df = pd.read_csv(StringIO(pasted_data), sep='\t|\s+', engine='python')
                    
                    # 첫 4개 열만 사용
                    if len(df.columns) >= 4:
                        df = df.iloc[:, :4]
                        # 원래 열 이름 보존
                        col_names = df.columns.tolist()
                        # 2, 3, 4번째 열을 숫자로 변환
                        df[col_names[1]] = df[col_names[1]].astype(float)
                        df[col_names[2]] = df[col_names[2]].astype(float)
                        df[col_names[3]] = df[col_names[3]].astype(float)
                        st.session_state.custom_data_3d = df
                        st.success(f"✅ {len(df)}개의 데이터 포인트가 로드되었습니다!")
                        st.info(f"📌 열 이름: {col_names[0]} (라벨), {col_names[1]} (X축), {col_names[2]} (Y축), {col_names[3]} (Z축)")
                    else:
                        st.error("❌ 최소 4개의 열이 필요합니다.")
            except Exception as e:
                st.error(f"❌ 데이터 파싱 오류: {str(e)}")
    
    with col2:
        st.subheader("2. 직접 입력")
        
        num_points = st.number_input("데이터 포인트 개수", min_value=1, max_value=100, value=5, key="num_points_3d")
        
        # 동적 입력 필드
        data_input = []
        cols = st.columns(2)
        
        for i in range(num_points):
            col_idx = i % 2
            with cols[col_idx]:
                name = st.text_input(f"Point {i+1} - 이름", value=f"Data_{i+1}", key=f"name_3d_{i}")
                x1 = st.number_input(f"Point {i+1} - X1", value=0.0, key=f"x1_3d_{i}")
                x2 = st.number_input(f"Point {i+1} - X2", value=0.0, key=f"x2_3d_{i}")
                x3 = st.number_input(f"Point {i+1} - X3", value=0.0, key=f"x3_3d_{i}")
                data_input.append([name, x1, x2, x3])
        
        if st.button("✅ 직접 입력 데이터 저장", key="manual_load_3d"):
            df = pd.DataFrame(data_input, columns=['이름', 'X1', 'X2', 'X3'])
            st.session_state.custom_data_3d = df
            st.success(f"✅ {len(df)}개의 데이터 포인트가 저장되었습니다!")
    
    # 데이터 미리보기
    if not st.session_state.custom_data_3d.empty:
        st.subheader("📋 데이터 미리보기")
        st.dataframe(st.session_state.custom_data_3d, use_container_width=True)
        
        # 기본 통계
        x_col = st.session_state.custom_data_3d.columns[1]
        y_col = st.session_state.custom_data_3d.columns[2]
        z_col = st.session_state.custom_data_3d.columns[3]
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("총 데이터 포인트", len(st.session_state.custom_data_3d))
        with col2:
            st.metric(f"{x_col} 평균", f"{st.session_state.custom_data_3d[x_col].mean():.2f}")
        with col3:
            st.metric(f"{y_col} 평균", f"{st.session_state.custom_data_3d[y_col].mean():.2f}")
        with col4:
            st.metric(f"{z_col} 평균", f"{st.session_state.custom_data_3d[z_col].mean():.2f}")
        
        # 데이터 다운로드
        csv = st.session_state.custom_data_3d.to_csv(index=False)
        st.download_button(
            label="📥 CSV로 다운로드",
            data=csv,
            file_name="kmeans_3d_data.csv",
            mime="text/csv"
        )

# ============================================
# 탭 2: 최적 K값 분석 (Elbow Method)
# ============================================
with tab2:
    st.header("최적 군집 수 찾기 (Elbow Method)")
    
    if st.session_state.custom_data_3d.empty:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 입력해주세요.")
    else:
        # K값 범위 설정
        col1, col2 = st.columns(2)
        with col1:
            max_k = st.slider("최대 K값", min_value=3, max_value=15, value=10)
        with col2:
            st.info(f"K값을 1부터 {max_k}까지 분석합니다.")
        
        if st.button("🔍 Inertia 계산", key="calculate_inertia_3d"):
            with st.spinner("계산 중..."):
                # 열 이름 가져오기
                x_col = st.session_state.custom_data_3d.columns[1]
                y_col = st.session_state.custom_data_3d.columns[2]
                z_col = st.session_state.custom_data_3d.columns[3]
                
                inertias = []
                k_range = range(1, max_k + 1)
                
                for k in k_range:
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    kmeans.fit(st.session_state.custom_data_3d[[x_col, y_col, z_col]])
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
                
                st.success("✅ Inertia 계산 완료! '3D 군집 시각화' 탭에서 최적 K값을 선택하세요.")

# ============================================
# 탭 3: 3D 군집 시각화 (Plotly)
# ============================================
with tab3:
    st.header("K-means 3D 군집 시각화")
    
    if st.session_state.custom_data_3d.empty:
        st.warning("⚠️ 먼저 '데이터 입력' 탭에서 데이터를 입력해주세요.")
    else:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("⚙️ 설정")
            optimal_k = st.slider(
                "최적 K값 선택",
                min_value=1,
                max_value=min(10, len(st.session_state.custom_data_3d) - 1),
                value=3,
                key="optimal_k_3d"
            )
            
            run_clustering = st.button("🚀 K-means 실행", key="run_clustering_3d")
        
        if run_clustering or st.session_state.kmeans_model_3d is not None:
            with col2:
                with st.spinner("클러스터링 중..."):
                    # 열 이름 가져오기
                    label_col = st.session_state.custom_data_3d.columns[0]
                    x_col = st.session_state.custom_data_3d.columns[1]
                    y_col = st.session_state.custom_data_3d.columns[2]
                    z_col = st.session_state.custom_data_3d.columns[3]
                    
                    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
                    clusters = kmeans.fit_predict(st.session_state.custom_data_3d[[x_col, y_col, z_col]])
                    
                    st.session_state.kmeans_model_3d = kmeans
                    st.session_state.clusters_3d = clusters
                    
                    # 데이터에 클러스터 정보 추가
                    plot_data = st.session_state.custom_data_3d.copy()
                    plot_data['클러스터'] = clusters.astype(str)
                    
                    # 3D 시각화 (Plotly)
                    fig = go.Figure()
                    
                    # 데이터 포인트 추가
                    for i in range(optimal_k):
                        mask = clusters == i
                        cluster_points = plot_data[mask]
                        
                        fig.add_trace(go.Scatter3d(
                            x=cluster_points[x_col],
                            y=cluster_points[y_col],
                            z=cluster_points[z_col],
                            mode='markers+text',
                            name=f'클러스터 {i}',
                            marker=dict(
                                size=8,
                                opacity=0.8,
                                line=dict(width=0.5, color='white')
                            ),
                            text=cluster_points[label_col],
                            textposition='top center',
                            textfont=dict(size=8)
                        ))
                    
                    # 센트로이드 추가
                    centroids = kmeans.cluster_centers_
                    fig.add_trace(go.Scatter3d(
                        x=centroids[:, 0],
                        y=centroids[:, 1],
                        z=centroids[:, 2],
                        mode='markers',
                        name='Centroids',
                        marker=dict(
                            size=15,
                            color='red',
                            symbol='diamond',
                            line=dict(width=2, color='darkred')
                        )
                    ))
                    
                    # 레이아웃 설정
                    fig.update_layout(
                        title=f'K-means 3D Clustering (K={optimal_k})',
                        scene=dict(
                            xaxis_title=x_col,
                            yaxis_title=y_col,
                            zaxis_title=z_col,
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=1.3)
                            )
                        ),
                        hovermode='closest',
                        height=700,
                        font=dict(family='NanumGothic')
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 클러스터 통계
                    st.subheader("📊 클러스터 통계")
                    
                    cluster_stats = []
                    for i in range(optimal_k):
                        mask = clusters == i
                        count = mask.sum()
                        centroid = centroids[i]
                        members = st.session_state.custom_data_3d[mask][label_col].tolist()
                        cluster_stats.append({
                            '클러스터': i,
                            '데이터 포인트 수': count,
                            f'Centroid {x_col}': f"{centroid[0]:.2f}",
                            f'Centroid {y_col}': f"{centroid[1]:.2f}",
                            f'Centroid {z_col}': f"{centroid[2]:.2f}",
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
                        st.metric("총 데이터 포인트", len(st.session_state.custom_data_3d))
                    
                    # 각 클러스터 상세 정보
                    st.subheader("🔍 클러스터별 상세 정보")
                    
                    for i in range(optimal_k):
                        with st.expander(f"클러스터 {i} ({(clusters == i).sum()}개 포인트)"):
                            cluster_data = st.session_state.custom_data_3d[clusters == i].copy()
                            cluster_data['군집'] = i
                            st.dataframe(cluster_data.reset_index(drop=True), use_container_width=True)
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(f"{x_col} 평균", f"{cluster_data[x_col].mean():.2f}")
                            with col2:
                                st.metric(f"{y_col} 평균", f"{cluster_data[y_col].mean():.2f}")
                            with col3:
                                st.metric(f"{z_col} 평균", f"{cluster_data[z_col].mean():.2f}")
