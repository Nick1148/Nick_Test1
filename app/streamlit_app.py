"""
Module 3: Streamlit 웹 애플리케이션
===================================
다중 타겟 예측 모델의 시각화 및 사용자 인터페이스

주요 기능:
- 학습된 모델 파일 업로드
- 예측 실행 및 결과 비교
- 시각화 대시보드
- Feature Importance 분석
- 예측 결과 다운로드
"""

print("[STARTUP] Starting Streamlit app...")

import streamlit as st
print("[STARTUP] Streamlit imported")
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
from datetime import datetime
import json
import joblib
import tempfile
import os

# ============================================================
# Hugging Face Hub 모델 다운로드 설정
# ============================================================
# 여기에 Hugging Face 저장소 ID를 입력하세요
HF_REPO_ID = "nick1148/4gtp-models"  # 본인의 저장소로 변경하세요
HF_MODEL_FILENAME = "model_integrated_compressed.pkl"  # 압축된 통합 모델 (95MB)
HF_DATA_FILENAME = "4GTP_integrated_with_coal_Raw.xlsx"  # 데이터 파일

def download_model_from_huggingface():
    """Hugging Face Hub에서 모델 다운로드"""
    try:
        from huggingface_hub import hf_hub_download

        print(f"[HF] Downloading model from {HF_REPO_ID}...")

        # 캐시 디렉토리에 다운로드 (재시작 시 재사용)
        model_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_MODEL_FILENAME,
            cache_dir="/tmp/hf_cache"
        )

        print(f"[HF] Model downloaded to: {model_path}")
        return model_path
    except Exception as e:
        print(f"[HF] Download failed: {e}")
        return None

def download_data_from_huggingface():
    """Hugging Face Hub에서 데이터 파일 다운로드"""
    try:
        from huggingface_hub import hf_hub_download

        print(f"[HF] Downloading data from {HF_REPO_ID}...")

        data_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_DATA_FILENAME,
            cache_dir="/tmp/hf_cache"
        )

        print(f"[HF] Data downloaded to: {data_path}")
        return data_path
    except Exception as e:
        print(f"[HF] Data download failed: {e}")
        return None

def load_model_from_huggingface():
    """Hugging Face에서 모델을 다운로드하고 로드"""
    model_path = download_model_from_huggingface()
    if model_path:
        try:
            model_data = joblib.load(model_path)
            return model_data
        except Exception as e:
            print(f"[HF] Model load failed: {e}")
    return None

def load_data_from_huggingface():
    """Hugging Face에서 데이터를 다운로드하고 로드"""
    data_path = download_data_from_huggingface()
    if data_path:
        try:
            df = pd.read_excel(data_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
            return df
        except Exception as e:
            print(f"[HF] Data load failed: {e}")
    return None

# 경로 설정
APP_DIR = Path(__file__).parent
ROOT_DIR = APP_DIR.parent
sys.path.append(str(ROOT_DIR))

print("[STARTUP] Importing config.settings...")
from config.settings import (
    TARGET_COLUMNS, TARGET_NAMES_KR, TARGET_UNITS,
    COAL_CLASSES, FEATURE_GROUPS, FEATURE_NAMES_KR,
    UI_SETTINGS, CHART_COLORS, DATA_FILE, MODELS_DIR
)
print("[STARTUP] Config imported successfully")


# ============================================================
# 페이지 설정
# ============================================================
st.set_page_config(
    page_title=UI_SETTINGS['page_title'],
    page_icon=UI_SETTINGS['page_icon'],
    layout=UI_SETTINGS['layout'],
    initial_sidebar_state="expanded"
)

# 커스텀 CSS
st.markdown("""
<style>
    /* 메인 컨테이너 */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }

    /* 메트릭 카드 */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }

    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }

    /* 상태 뱃지 */
    .status-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .status-good {
        background-color: #d4edda;
        color: #155724;
    }

    .status-bad {
        background-color: #f8d7da;
        color: #721c24;
    }

    .status-neutral {
        background-color: #fff3cd;
        color: #856404;
    }

    /* 섹션 헤더 */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #333;
        border-left: 4px solid #1f77b4;
        padding-left: 1rem;
        margin: 2rem 0 1rem 0;
    }

    /* 정보 박스 */
    .info-box {
        background-color: #e7f3ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 0 10px 10px 0;
        margin: 1rem 0;
    }

    /* 사이드바 스타일 */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    /* 탭 스타일 */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }

    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }

    /* 테이블 스타일 */
    .dataframe {
        font-size: 0.9rem;
    }

    /* 버튼 스타일 */
    .stButton > button {
        width: 100%;
        border-radius: 10px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 세션 상태 초기화
# ============================================================
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
if 'preprocessor' not in st.session_state:
    st.session_state.preprocessor = None
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'raw_data' not in st.session_state:
    st.session_state.raw_data = None
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'uploaded_models' not in st.session_state:
    st.session_state.uploaded_models = {}
if 'model_upload_status' not in st.session_state:
    st.session_state.model_upload_status = {
        'integrated': False,
        'ClassA': False,
        'ClassB': False,
        'ClassC': False
    }


# ============================================================
# 유틸리티 함수
# ============================================================
@st.cache_data
def load_raw_data():
    """원시 데이터 로드"""
    try:
        # 로컬 파일이 있는지 확인
        if DATA_FILE.exists():
            df = pd.read_excel(DATA_FILE)
            df['Date'] = pd.to_datetime(df['Date'])
            return df
        else:
            # Streamlit Cloud에서는 파일이 없을 수 있음
            return None
    except Exception as e:
        # 오류를 조용히 처리 (Cloud 환경)
        return None


def load_uploaded_data(uploaded_file):
    """업로드된 데이터 파일 로드"""
    try:
        if uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return None

        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return None


def load_models():
    """모델 로드 (기존 models 폴더에서)"""
    try:
        # 모델 폴더가 존재하고 모델 파일이 있는지 먼저 확인
        if not MODELS_DIR.exists():
            return None

        model_files = list(MODELS_DIR.glob('*.pkl'))
        if not model_files:
            return None

        from src.prediction import MultiTargetPredictor
        predictor = MultiTargetPredictor()
        if predictor.load_models():
            return predictor
        return None
    except Exception as e:
        # Cloud 환경에서 오류 발생 시 조용히 처리
        return None


def load_uploaded_model(uploaded_file, model_name):
    """업로드된 모델 파일 로드"""
    try:
        # 임시 파일로 저장 후 로드
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        model_data = joblib.load(tmp_path)
        os.unlink(tmp_path)  # 임시 파일 삭제

        return model_data
    except Exception as e:
        st.error(f"모델 로드 실패 ({model_name}): {e}")
        return None


def load_combined_model(uploaded_file):
    """통합 모델 파일 로드 (models_combined.pkl 또는 models_combined_with_data.pkl)

    Returns:
        tuple: (model_data, raw_data) - raw_data는 없으면 None
    """
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        combined_data = joblib.load(tmp_path)
        os.unlink(tmp_path)

        # raw_data가 포함되어 있는지 확인
        raw_data = None
        if 'raw_data' in combined_data:
            raw_data = combined_data['raw_data']
            print(f"[INFO] raw_data 추출됨: {raw_data.shape}")
            # raw_data는 모델 데이터에서 제거 (메모리 효율)
            del combined_data['raw_data']

        return combined_data, raw_data
    except Exception as e:
        st.error(f"통합 모델 로드 실패: {e}")
        return None, None


def detect_model_format(model_data):
    """
    모델 파일 형식 감지

    Returns:
        'integrated': model_integrated.pkl 형식 (models 키 아래 직접 타겟별 모델)
        'combined': models_combined.pkl 형식 (models 키 아래 모델명/타겟별 모델)
        None: 알 수 없는 형식
    """
    if 'models' not in model_data:
        return None

    models = model_data['models']
    if not models:
        return None

    # models의 첫 번째 키 확인
    first_key = list(models.keys())[0]
    first_value = models[first_key]

    # combined 형식: models['integrated']['BTX_generation'] = sklearn model
    # integrated 형식: models['BTX_generation'] = sklearn model

    if isinstance(first_value, dict):
        # combined 형식 (중첩된 딕셔너리)
        return 'combined'
    elif hasattr(first_value, 'predict'):
        # integrated 형식 (직접 sklearn 모델)
        return 'integrated'

    return None


def create_predictor_from_combined(model_data):
    """
    모델 데이터로 Predictor 생성

    model_integrated.pkl과 models_combined.pkl 형식 모두 지원
    """
    try:
        from src.prediction import MultiTargetPredictor

        predictor = MultiTargetPredictor()

        # 모델 형식 감지
        model_format = detect_model_format(model_data)
        print(f"[INFO] Detected model format: {model_format}")

        if model_format == 'integrated':
            # model_integrated.pkl 형식
            # models에 직접 타겟별 모델이 있음
            # 이를 'integrated' 키 아래로 래핑
            predictor.models = {'integrated': model_data.get('models', {})}

            # performance와 feature_importance도 동일하게 래핑
            perf = model_data.get('performance', {})
            fi = model_data.get('feature_importance', {})

            # 이미 타겟별로 정리되어 있는지 확인
            if perf and list(perf.keys())[0] in TARGET_COLUMNS:
                predictor.model_performance = {'integrated': perf}
            else:
                predictor.model_performance = perf

            if fi and list(fi.keys())[0] in TARGET_COLUMNS:
                predictor.feature_importance = {'integrated': fi}
            else:
                predictor.feature_importance = fi

        elif model_format == 'combined':
            # models_combined.pkl 형식
            # 기존 방식 그대로
            predictor.models = model_data.get('models', {})
            predictor.model_performance = model_data.get('performance', {})
            predictor.feature_importance = model_data.get('feature_importance', {})
        else:
            print("[ERROR] Unknown model format")
            return None

        predictor.is_loaded = len(predictor.models) > 0

        return predictor if predictor.is_loaded else None
    except Exception as e:
        st.error(f"Predictor 생성 실패: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_predictor_from_uploaded():
    """업로드된 모델들로 Predictor 생성"""
    try:
        from src.prediction import MultiTargetPredictor

        predictor = MultiTargetPredictor()
        predictor.models = {}
        predictor.model_performance = {}
        predictor.feature_importance = {}

        for model_name, model_data in st.session_state.uploaded_models.items():
            if model_data is not None:
                predictor.models[model_name] = model_data.get('models', {})
                predictor.model_performance[model_name] = model_data.get('performance', {})
                predictor.feature_importance[model_name] = model_data.get('feature_importance', {})

        predictor.is_loaded = len(predictor.models) > 0
        return predictor if predictor.is_loaded else None
    except Exception as e:
        st.error(f"Predictor 생성 실패: {e}")
        return None


def create_gauge_chart(value, title, min_val=0, max_val=100, threshold_good=70, threshold_bad=30):
    """게이지 차트 생성"""
    if value >= threshold_good:
        color = CHART_COLORS['success']
    elif value >= threshold_bad:
        color = CHART_COLORS['warning']
    else:
        color = CHART_COLORS['danger']

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        title={'text': title, 'font': {'size': 16}},
        delta={'reference': threshold_good, 'relative': False},
        gauge={
            'axis': {'range': [min_val, max_val], 'tickwidth': 1},
            'bar': {'color': color},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [min_val, threshold_bad], 'color': '#ffebee'},
                {'range': [threshold_bad, threshold_good], 'color': '#fff3e0'},
                {'range': [threshold_good, max_val], 'color': '#e8f5e9'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_good
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


def create_comparison_chart(predictions, actuals, model_names):
    """예측값 vs 실제값 비교 차트"""
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=[TARGET_NAMES_KR[t] for t in TARGET_COLUMNS]
    )

    colors = [CHART_COLORS['primary'], CHART_COLORS['secondary'],
              CHART_COLORS['success'], CHART_COLORS['danger']]

    for i, target in enumerate(TARGET_COLUMNS):
        col = i + 1

        # 실제값
        if actuals and target in actuals:
            fig.add_trace(
                go.Bar(name='실제값', x=['실제값'], y=[actuals[target]],
                       marker_color='#2c3e50', showlegend=(i==0)),
                row=1, col=col
            )

        # 모델별 예측값
        for j, model_name in enumerate(model_names):
            if model_name in predictions and target in predictions[model_name]:
                pred_val = predictions[model_name][target]
                if isinstance(pred_val, (list, np.ndarray)):
                    pred_val = pred_val[0] if len(pred_val) > 0 else 0
                fig.add_trace(
                    go.Bar(name=model_name, x=[model_name], y=[pred_val],
                           marker_color=colors[j % len(colors)], showlegend=(i==0)),
                    row=1, col=col
                )

    fig.update_layout(
        height=400,
        barmode='group',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    return fig


def create_efficiency_chart(efficiency_data):
    """효율 분석 차트"""
    fig = go.Figure()

    targets = []
    values = []
    colors = []

    for target in TARGET_COLUMNS:
        if target in efficiency_data:
            data = efficiency_data[target]
            targets.append(data['target_kr'])
            values.append(data['difference_pct'])

            if target == 'BTX_generation':
                colors.append(CHART_COLORS['success'] if data['is_better'] else CHART_COLORS['danger'])
            else:
                colors.append(CHART_COLORS['success'] if data['is_better'] else CHART_COLORS['danger'])

    fig.add_trace(go.Bar(
        x=targets,
        y=values,
        marker_color=colors,
        text=[f"{v:+.1f}%" for v in values],
        textposition='outside'
    ))

    fig.update_layout(
        title="효율 개선 분석",
        xaxis_title="타겟 변수",
        yaxis_title="변화율 (%)",
        height=350,
        showlegend=False
    )

    # 0선 추가
    fig.add_hline(y=0, line_dash="dash", line_color="gray")

    return fig


def create_feature_importance_chart(importance_df, title="Feature Importance"):
    """Feature Importance 차트"""
    if importance_df.empty:
        return None

    df = importance_df.head(15).sort_values('importance', ascending=True)

    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature_kr'] if 'feature_kr' in df.columns else df['feature'],
        orientation='h',
        marker_color=CHART_COLORS['primary']
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=400,
        margin=dict(l=200)
    )

    return fig


def create_time_series_chart(df, target, title=None):
    """시계열 차트"""
    if title is None:
        title = TARGET_NAMES_KR.get(target, target)

    fig = px.line(df, x='Date', y=target, title=title)
    fig.update_layout(
        xaxis_title="날짜",
        yaxis_title=f"{title} ({TARGET_UNITS.get(target, '')})",
        height=350
    )

    return fig


def create_correlation_heatmap(df, columns=None):
    """상관관계 히트맵"""
    if columns is None:
        columns = TARGET_COLUMNS

    corr_matrix = df[columns].corr()

    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix.values,
        x=[TARGET_NAMES_KR.get(c, c) for c in corr_matrix.columns],
        y=[TARGET_NAMES_KR.get(c, c) for c in corr_matrix.index],
        colorscale='RdBu',
        zmin=-1, zmax=1,
        text=np.round(corr_matrix.values, 2),
        texttemplate='%{text}',
        textfont={"size": 12},
        hoverinfo='z'
    ))

    fig.update_layout(
        title="타겟 변수 상관관계",
        height=400
    )

    return fig


def create_distribution_chart(df, column, title=None):
    """분포 차트"""
    if title is None:
        title = FEATURE_NAMES_KR.get(column, column)

    fig = make_subplots(rows=1, cols=2, subplot_titles=["히스토그램", "박스플롯"])

    fig.add_trace(
        go.Histogram(x=df[column], nbinsx=50, marker_color=CHART_COLORS['primary']),
        row=1, col=1
    )

    fig.add_trace(
        go.Box(y=df[column], marker_color=CHART_COLORS['secondary']),
        row=1, col=2
    )

    fig.update_layout(title=title, height=350, showlegend=False)

    return fig


# ============================================================
# 사이드바
# ============================================================
print("[STARTUP] Setting up sidebar...")
with st.sidebar:
    # 외부 이미지 URL 제거 (Cloud 환경에서 블로킹 방지)
    st.markdown("# 🏭")
    st.title("4GTP 예측 시스템")
    st.markdown("---")

    # 메뉴 선택 (모델 업로드가 첫 번째)
    menu = st.radio(
        "메뉴 선택",
        ["모델 업로드", "대시보드", "예측 실행", "데이터 분석", "결과 분석", "설정"],
        index=0
    )

    st.markdown("---")

    # 모델 상태
    st.subheader("[MODEL] 모델 상태")

    # 업로드된 모델 상태 표시
    if st.session_state.models_loaded and st.session_state.predictor:
        available_models = st.session_state.predictor.get_available_models()
        for model in available_models:
            st.success(f"[OK] {model}")
        st.info(f"총 {len(available_models)}개 모델 로드됨")
    else:
        # 업로드 상태 표시
        upload_status = st.session_state.model_upload_status
        loaded_count = sum(1 for v in upload_status.values() if v)

        if loaded_count > 0:
            st.info(f"{loaded_count}/4 모델 업로드됨")
            for name, status in upload_status.items():
                if status:
                    st.success(f"[OK] {name}")
                else:
                    st.warning(f"[--] {name}")
        else:
            st.warning("[!] 모델 미업로드")
            st.caption("'모델 업로드' 메뉴에서 업로드하거나")
            st.caption("아래 버튼으로 자동 다운로드하세요")

            # Hugging Face에서 모델 자동 다운로드 버튼
            if st.button("🤗 HuggingFace에서 모델 다운로드", use_container_width=True):
                with st.spinner("모델 다운로드 중... (약 1분 소요)"):
                    combined_data = load_model_from_huggingface()
                    if combined_data:
                        predictor = create_predictor_from_combined(combined_data)
                        if predictor:
                            st.session_state.predictor = predictor
                            st.session_state.models_loaded = True
                            for model_name in predictor.get_available_models():
                                st.session_state.model_upload_status[model_name] = True
                            st.success("✅ 모델 다운로드 완료!")
                            st.rerun()
                        else:
                            st.error("모델 로드 실패")
                    else:
                        st.error("다운로드 실패 - 저장소를 확인하세요")

    st.markdown("---")

    # 데이터 상태
    st.subheader("📁 데이터 상태")
    if st.session_state.raw_data is not None:
        st.success(f"✅ {len(st.session_state.raw_data):,}행 로드됨")
    else:
        # Cloud 환경에서는 데이터 업로드 옵션 표시
        st.warning("📤 데이터 업로드 필요")

        # HuggingFace에서 데이터 다운로드 버튼
        if st.button("🤗 HuggingFace에서 데이터 다운로드", use_container_width=True):
            with st.spinner("데이터 다운로드 중..."):
                df = load_data_from_huggingface()
                if df is not None:
                    st.session_state.raw_data = df
                    st.session_state.data_loaded = True
                    st.success("✅ 데이터 다운로드 완료!")
                    st.rerun()
                else:
                    st.error("다운로드 실패 - 저장소를 확인하세요")

        # 데이터 파일 업로드
        uploaded_data = st.file_uploader(
            "또는 파일 직접 업로드",
            type=['xlsx', 'csv'],
            key="sidebar_data_upload",
            help="4GTP_integrated_with_coal_Raw.xlsx 파일을 업로드하세요"
        )

        if uploaded_data:
            with st.spinner("데이터 로드 중..."):
                df = load_uploaded_data(uploaded_data)
                if df is not None:
                    st.session_state.raw_data = df
                    st.session_state.data_loaded = True
                    st.success("✅ 데이터 로드 완료!")
                    st.rerun()

        # 로컬 파일 로드 버튼 (로컬 환경용)
        if DATA_FILE.exists():
            if st.button("🔄 로컬 데이터 로드", use_container_width=True):
                with st.spinner("데이터 로드 중..."):
                    df = load_raw_data()
                    if df is not None:
                        st.session_state.raw_data = df
                        st.session_state.data_loaded = True
                        st.success("✅ 데이터 로드 완료!")
                        st.rerun()

    st.markdown("---")
    st.caption("© 2025 4GTP 예측 시스템")

print("[STARTUP] Sidebar setup complete")

# ============================================================
# 메인 컨텐츠
# ============================================================

# 데이터 자동 로드 (로컬 환경에서만)
if st.session_state.raw_data is None and DATA_FILE.exists():
    df = load_raw_data()
    if df is not None:
        st.session_state.raw_data = df
        st.session_state.data_loaded = True


# ============================================================
# 1. 대시보드
# ============================================================
if menu == "대시보드":
    st.markdown('<div class="main-header">🏭 4GTP 다중 타겟 예측 시스템</div>', unsafe_allow_html=True)

    # 모델 미업로드시 경고
    if not st.session_state.models_loaded:
        st.warning("⚠️ 모델이 업로드되지 않았습니다. '모델 업로드' 메뉴에서 모델을 먼저 업로드해주세요.")
        st.info("모델을 업로드해야 대시보드의 모든 기능을 사용할 수 있습니다.")
        st.stop()

    # 주요 지표 카드
    col1, col2, col3, col4 = st.columns(4)

    df = st.session_state.raw_data

    if df is not None:
        with col1:
            st.metric(
                label="📊 전체 데이터",
                value=f"{len(df):,}",
                delta="시간별 기록"
            )

        with col2:
            date_range = (df['Date'].max() - df['Date'].min()).days
            st.metric(
                label="📅 데이터 기간",
                value=f"{date_range:,}일",
                delta=f"{df['Date'].min().strftime('%Y-%m')} ~ {df['Date'].max().strftime('%Y-%m')}"
            )

        with col3:
            if st.session_state.models_loaded:
                model_count = len(st.session_state.predictor.get_available_models())
                st.metric(
                    label="🤖 학습된 모델",
                    value=f"{model_count}개",
                    delta="사용 가능"
                )
            else:
                st.metric(
                    label="🤖 학습된 모델",
                    value="0개",
                    delta="로드 필요"
                )

        with col4:
            st.metric(
                label="🎯 타겟 변수",
                value=f"{len(TARGET_COLUMNS)}개",
                delta="다중 타겟"
            )

    st.markdown("---")

    # 시스템 개요
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown('<div class="section-header">📌 시스템 개요</div>', unsafe_allow_html=True)

        st.markdown("""
        이 시스템은 **4GTP 공정 데이터**를 기반으로 다음 3가지 타겟을 **동시 예측**합니다:

        | 타겟 | 설명 | 목적 |
        |------|------|------|
        | **BTX 생산량** | Benzene, Toluene, Xylene 총 생산량 | 생산성 최적화 |
        | **히터 증기 투입량** | 가열기에 투입되는 증기량 | 에너지 효율 |
        | **BTX 증류탑 증기 투입량** | 증류 공정에 투입되는 증기량 | 에너지 효율 |

        **핵심 기능:**
        - 🔄 통합 모델 + Coal Class별 개별 모델 비교
        - 📊 예측값 vs 실제값 효율 분석
        - 📈 Feature Importance 시각화
        - 💾 예측 결과 CSV 다운로드
        """)

    with col2:
        st.markdown('<div class="section-header">📊 Coal Class 분포</div>', unsafe_allow_html=True)

        if df is not None:
            class_dist = df['coal_class'].value_counts()
            fig = px.pie(
                values=class_dist.values,
                names=class_dist.index,
                color=class_dist.index,
                color_discrete_map={
                    'ClassA': CHART_COLORS['ClassA'],
                    'ClassB': CHART_COLORS['ClassB'],
                    'ClassC': CHART_COLORS['ClassC']
                },
                hole=0.4
            )
            fig.update_layout(height=300, margin=dict(l=20, r=20, t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

    # 타겟 변수 추이
    st.markdown('<div class="section-header">📈 타겟 변수 최근 추이</div>', unsafe_allow_html=True)

    if df is not None:
        # 최근 30일 데이터
        recent_df = df[df['Date'] >= df['Date'].max() - pd.Timedelta(days=30)]

        tab1, tab2, tab3 = st.tabs([TARGET_NAMES_KR[t] for t in TARGET_COLUMNS])

        for i, (tab, target) in enumerate(zip([tab1, tab2, tab3], TARGET_COLUMNS)):
            with tab:
                fig = create_time_series_chart(recent_df, target)
                st.plotly_chart(fig, use_container_width=True)


# ============================================================
# 2. 모델 업로드
# ============================================================
elif menu == "모델 업로드":
    st.markdown('<div class="main-header">[UPLOAD] 모델 업로드</div>', unsafe_allow_html=True)

    st.markdown("""
    ### 학습된 모델 파일 업로드

    이 시스템은 **사전 학습된 모델 파일(.pkl)**을 업로드하여 사용합니다.

    **지원 모델 파일:**
    - `model_integrated_compressed.pkl` (95MB, 권장) - 압축된 통합 모델
    - `model_integrated.pkl` (412MB) - 통합 모델 (비압축)
    - `models_combined.pkl` (924MB) - 전체 4개 모델 포함 (Cloud 미권장)

    > **Note:** Streamlit Cloud에서는 용량이 작은 `model_integrated_compressed.pkl`을 권장합니다.
    > HuggingFace 자동 다운로드 기능으로 편리하게 모델을 로드할 수 있습니다.

    > **⚠️ Streamlit Cloud 주의:**
    > - 메모리 제한: 약 1GB (큰 모델 사용 시 주의)
    > - 권장: 압축 모델 또는 HuggingFace 자동 다운로드 사용
    """)

    st.markdown("---")

    # 파일 업로드 섹션
    tab1, tab2 = st.tabs(["[FILE] 파일 업로드", "[STATUS] 업로드 현황"])

    with tab1:
        st.subheader("모델 파일 업로드")

        st.markdown("""
        **지원 파일 형식:**
        - `model_integrated_compressed.pkl` (95MB, 권장)
        - `model_integrated.pkl` (412MB)
        - `models_combined.pkl` (924MB)

        **통합 모델(Integrated)**은 전체 데이터로 학습된 범용 모델입니다.
        Coal Class 선택은 데이터 분석용으로 계속 제공됩니다.
        """)

        uploaded_combined = st.file_uploader(
            "모델 파일 (.pkl) 업로드",
            type=['pkl'],
            key="upload_combined",
            help="model_integrated.pkl, model_integrated_compressed.pkl, 또는 models_combined.pkl"
        )

        if uploaded_combined:
            with st.spinner("모델 파일 로드 중... (파일 크기가 클 수 있습니다)"):
                combined_data, embedded_raw_data = load_combined_model(uploaded_combined)
                if combined_data:
                    # Predictor 생성
                    predictor = create_predictor_from_combined(combined_data)
                    if predictor:
                        st.session_state.predictor = predictor
                        st.session_state.models_loaded = True

                        # 상태 업데이트
                        for model_name in predictor.get_available_models():
                            if model_name in st.session_state.model_upload_status:
                                st.session_state.model_upload_status[model_name] = True

                        st.success(f"[OK] 모델 로드 완료! ({len(predictor.get_available_models())}개 모델)")

                        # 내장된 raw_data가 있으면 자동 로드
                        if embedded_raw_data is not None:
                            if 'Date' in embedded_raw_data.columns:
                                embedded_raw_data['Date'] = pd.to_datetime(embedded_raw_data['Date'])
                            st.session_state.raw_data = embedded_raw_data
                            st.success(f"[OK] 내장 데이터 자동 로드! ({len(embedded_raw_data):,}행)")

                        st.balloons()

                        # 로드된 모델 목록 표시
                        st.markdown("**로드된 모델:**")
                        for model in predictor.get_available_models():
                            st.success(f"  - {model}")
                    else:
                        st.error("[ERROR] 모델 활성화 실패. 파일을 확인해주세요.")
                else:
                    st.error("[ERROR] 파일 로드 실패. 올바른 모델 파일인지 확인해주세요.")

        # 기존 모델 폴더에서 로드 옵션
        st.markdown("---")
        st.subheader("[FOLDER] 기존 모델 폴더에서 로드")
        st.caption("models/ 폴더에 이미 학습된 모델이 있다면 아래 버튼으로 로드할 수 있습니다.")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("[LOAD] 개별 모델 로드", use_container_width=True, help="model_integrated.pkl 등 개별 파일 로드"):
                with st.spinner("모델 로드 중..."):
                    predictor = load_models()
                    if predictor:
                        st.session_state.predictor = predictor
                        st.session_state.models_loaded = True
                        for model_name in predictor.get_available_models():
                            if model_name in st.session_state.model_upload_status:
                                st.session_state.model_upload_status[model_name] = True
                        st.success("[OK] 모델 로드 완료!")
                        st.rerun()
                    else:
                        st.error("[ERROR] models/ 폴더에 모델 파일이 없습니다.")

        with col2:
            if st.button("[LOAD] 통합 파일 로드", use_container_width=True, help="model_integrated.pkl 또는 models_combined.pkl 로드"):
                # 우선순위: 압축 통합 -> 통합 -> 전체 통합
                possible_paths = [
                    MODELS_DIR / 'model_integrated_compressed.pkl',
                    MODELS_DIR / 'model_integrated.pkl',
                    MODELS_DIR / 'models_combined.pkl'
                ]

                loaded = False
                for model_path in possible_paths:
                    if model_path.exists():
                        with st.spinner(f"{model_path.name} 로드 중..."):
                            try:
                                model_data = joblib.load(model_path)
                                predictor = create_predictor_from_combined(model_data)
                                if predictor:
                                    st.session_state.predictor = predictor
                                    st.session_state.models_loaded = True
                                    for model_name in predictor.get_available_models():
                                        if model_name in st.session_state.model_upload_status:
                                            st.session_state.model_upload_status[model_name] = True
                                    st.success(f"[OK] {model_path.name} 로드 완료!")
                                    loaded = True
                                    st.rerun()
                                    break
                            except Exception as e:
                                st.warning(f"{model_path.name} 로드 실패: {e}")
                                continue

                if not loaded:
                    st.error("[ERROR] models/ 폴더에 모델 파일이 없습니다.")

    with tab2:
        st.subheader("업로드 현황")

        # 상태 카드
        cols = st.columns(4)
        model_names = ['integrated', 'ClassA', 'ClassB', 'ClassC']
        model_labels = ['통합 모델', 'ClassA', 'ClassB', 'ClassC']

        for col, name, label in zip(cols, model_names, model_labels):
            with col:
                status = st.session_state.model_upload_status.get(name, False)
                if status:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #28a745, #20c997);
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h4 style="margin: 0;">{label}</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">[OK]</p>
                        <p style="font-size: 0.8rem; margin: 0;">업로드 완료</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background: linear-gradient(135deg, #6c757d, #adb5bd);
                                padding: 1rem; border-radius: 10px; text-align: center; color: white;">
                        <h4 style="margin: 0;">{label}</h4>
                        <p style="font-size: 1.5rem; margin: 0.5rem 0;">--</p>
                        <p style="font-size: 0.8rem; margin: 0;">미업로드</p>
                    </div>
                    """, unsafe_allow_html=True)

        st.markdown("---")

        # 모델 성능 정보 (업로드된 경우)
        if st.session_state.models_loaded and st.session_state.predictor:
            st.subheader("[PERF] 모델 성능 정보")

            performance = st.session_state.predictor.get_model_performance()

            if performance:
                for model_name, perf in performance.items():
                    with st.expander(f"[{model_name}] 성능 지표", expanded=True):
                        if perf:
                            perf_df = pd.DataFrame(perf).T
                            if not perf_df.empty:
                                perf_df.index = [TARGET_NAMES_KR.get(t, t) for t in perf_df.index]
                                display_cols = [c for c in ['R2', 'RMSE', 'MAE', 'MAPE'] if c in perf_df.columns]
                                if display_cols:
                                    st.dataframe(perf_df[display_cols].round(4), use_container_width=True)

                                    # R2 게이지 차트
                                    if 'R2' in perf_df.columns:
                                        gauge_cols = st.columns(len(TARGET_COLUMNS))
                                        for i, target in enumerate(TARGET_COLUMNS):
                                            if target in perf:
                                                with gauge_cols[i]:
                                                    r2 = perf[target].get('R2', 0)
                                                    fig = create_gauge_chart(
                                                        r2 * 100,
                                                        TARGET_NAMES_KR[target],
                                                        min_val=0, max_val=100,
                                                        threshold_good=80, threshold_bad=60
                                                    )
                                                    st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.info("성능 정보가 없습니다.")
            else:
                st.info("성능 정보를 로드할 수 없습니다.")

        # 초기화 버튼
        st.markdown("---")
        if st.button("[RESET] 모델 초기화", type="secondary"):
            st.session_state.models_loaded = False
            st.session_state.predictor = None
            st.session_state.uploaded_models = {}
            st.session_state.model_upload_status = {
                'integrated': False,
                'ClassA': False,
                'ClassB': False,
                'ClassC': False
            }
            st.success("[OK] 모델이 초기화되었습니다.")
            st.rerun()


# ============================================================
# 3. 데이터 분석
# ============================================================
elif menu == "데이터 분석":
    st.markdown('<div class="main-header">📊 데이터 분석</div>', unsafe_allow_html=True)

    # 데이터 분석은 모델 없이도 가능 - 데이터만 필요
    df = st.session_state.raw_data

    if df is None:
        st.warning("⚠️ 데이터를 먼저 로드해주세요.")
        st.info("💡 사이드바에서 'HuggingFace에서 데이터 다운로드' 또는 파일을 직접 업로드하세요.")
    else:
        # 탭 구성
        tab1, tab2, tab3, tab4 = st.tabs(["📋 데이터 개요", "📈 통계 분석", "🔗 상관관계", "📊 분포 분석"])

        with tab1:
            st.subheader("데이터 기본 정보")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("전체 행 수", f"{len(df):,}")
            with col2:
                st.metric("전체 열 수", f"{len(df.columns)}")
            with col3:
                st.metric("결측치", f"{df.isnull().sum().sum()}")

            st.markdown("---")

            st.subheader("컬럼 정보")
            col_info = pd.DataFrame({
                '컬럼명': df.columns,
                '데이터 타입': df.dtypes.astype(str).values,
                '결측치': df.isnull().sum().values,
                '고유값 수': df.nunique().values
            })
            st.dataframe(col_info, use_container_width=True, height=400)

            st.markdown("---")

            st.subheader("데이터 미리보기")
            st.dataframe(df.head(100), use_container_width=True)

        with tab2:
            st.subheader("기술 통계량")

            # Coal Class 선택
            selected_class = st.selectbox(
                "Coal Class 선택",
                ["전체"] + COAL_CLASSES
            )

            if selected_class == "전체":
                filtered_df = df
            else:
                filtered_df = df[df['coal_class'] == selected_class]

            st.write(f"**선택된 데이터: {len(filtered_df):,}개**")

            # 타겟 변수 통계
            st.subheader("🎯 타겟 변수 통계")
            target_stats = filtered_df[TARGET_COLUMNS].describe().T
            target_stats.index = [TARGET_NAMES_KR[t] for t in target_stats.index]
            st.dataframe(target_stats.round(3), use_container_width=True)

            # 시각화
            col1, col2 = st.columns(2)

            with col1:
                fig = px.box(
                    filtered_df.melt(value_vars=TARGET_COLUMNS),
                    x='variable', y='value',
                    color='variable',
                    labels={'variable': '타겟 변수', 'value': '값'}
                )
                fig.update_layout(title="타겟 변수 박스플롯", showlegend=False)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Coal Class별 타겟 평균
                class_means = df.groupby('coal_class')[TARGET_COLUMNS].mean()
                fig = px.bar(
                    class_means.reset_index().melt(id_vars='coal_class'),
                    x='coal_class', y='value', color='variable',
                    barmode='group',
                    labels={'coal_class': 'Coal Class', 'value': '평균값', 'variable': '타겟'}
                )
                fig.update_layout(title="Coal Class별 타겟 평균")
                st.plotly_chart(fig, use_container_width=True)

        with tab3:
            st.subheader("상관관계 분석")

            # 타겟 간 상관관계
            col1, col2 = st.columns([1, 1])

            with col1:
                fig = create_correlation_heatmap(df)
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                st.markdown("""
                **상관관계 해석:**

                - **BTX 생산량 ↔ 증류탑 증기**: 높은 양의 상관관계
                - **히터 증기 ↔ BTX 생산량**: 중간 양의 상관관계

                > 💡 타겟 변수들 간에 상관관계가 높아 **다중 타겟 모델**이 효과적입니다.
                """)

            st.markdown("---")

            # 피처와 타겟 간 상관관계
            st.subheader("피처-타겟 상관관계 (Top 10)")

            selected_target = st.selectbox(
                "타겟 변수 선택",
                TARGET_COLUMNS,
                format_func=lambda x: TARGET_NAMES_KR[x]
            )

            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c not in TARGET_COLUMNS]

            correlations = df[feature_cols].corrwith(df[selected_target]).abs().sort_values(ascending=False)

            fig = px.bar(
                x=[FEATURE_NAMES_KR.get(c, c) for c in correlations.head(10).index],
                y=correlations.head(10).values,
                labels={'x': '피처', 'y': '상관계수 (절댓값)'}
            )
            fig.update_layout(title=f"{TARGET_NAMES_KR[selected_target]}와의 상관관계 Top 10")
            st.plotly_chart(fig, use_container_width=True)

        with tab4:
            st.subheader("분포 분석")

            # 피처 그룹 선택
            selected_group = st.selectbox(
                "피처 그룹 선택",
                list(FEATURE_GROUPS.keys())
            )

            features = FEATURE_GROUPS[selected_group]

            # 분포 차트
            cols = st.columns(2)
            for i, feature in enumerate(features):
                if feature in df.columns:
                    with cols[i % 2]:
                        fig = create_distribution_chart(df, feature)
                        st.plotly_chart(fig, use_container_width=True)




# ============================================================
# 4. 예측 실행
# ============================================================
elif menu == "예측 실행":
    st.markdown('<div class="main-header">🔮 예측 실행 및 수익성 분석</div>', unsafe_allow_html=True)

    if not st.session_state.models_loaded:
        st.warning("⚠️ 모델을 먼저 로드해주세요. '모델 업로드' 메뉴에서 모델을 업로드하세요.")
        st.stop()

    predictor = st.session_state.predictor
    df = st.session_state.raw_data

    # 데이터 없을 때 경고
    if df is None:
        st.warning("⚠️ 데이터가 로드되지 않았습니다.")
        st.info("💡 최적값 계산을 위해 데이터가 필요합니다. 사이드바에서 'HuggingFace에서 데이터 다운로드'를 클릭하세요.")
        st.markdown("---")
        st.markdown("**데이터 없이도 현재값 입력은 가능하지만, 최적값 비교 기능은 제한됩니다.**")

    # 상위 5개 주요 인자 정의 (Feature Importance 기반)
    TOP_5_FEATURES = [
        'BS_aoil_flow',       # 흡수유 유량
        'BS_out_COG_F',       # BS 출구 COG 유량
        'BTXdistillator_RO_flow',  # BTX 증류탑 리치오일 유량
        'Heater_temp',        # 히터 온도
        'HE_VO_RO_T'          # 열교환기 VO 리치오일 온도
    ]

    # numeric 컬럼만 선택하여 기본값 계산
    if df is not None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        defaults = df[numeric_cols].iloc[-100:].mean()
    else:
        # 데이터 없을 때 기본값 설정
        defaults = {
            'BTX_generation': 1.5,
            'Heater_steam_input': 6.0,
            'BTXdistillator_distl_steam_input': 4.0,
            'BS_aoil_flow': 100.0,
            'BS_out_COG_F': 50.0,
            'BTXdistillator_RO_flow': 30.0,
            'Heater_temp': 180.0,
            'HE_VO_RO_T': 120.0
        }

    st.markdown("""
    ### 📋 예측 목적
    현재 운영값과 **데이터 기반 최적 운전값**을 비교하여 **수익성 개선 금액**을 계산합니다.

    - **현재값**: 사용자가 직접 입력하는 현재 운영 상태
    - **최적값**: 해당 Coal Class에서 효율이 가장 좋았던 상위 10% 데이터의 평균값

    **수익성 = 조경유(BTX) 추가 생산량 × 단가 - 스팀 추가 사용량 × 단가**
    """)

    st.markdown("---")

    # Coal Class 선택 (먼저)
    coal_class = st.selectbox(
        "Coal Class 선택",
        COAL_CLASSES,
        index=1,  # ClassB 기본
        key="coal_class_select"
    )

    # 해당 Coal Class의 최적값 계산 (상위 10% 효율)
    optimal_available = False  # 최적값 계산 가능 여부
    if df is not None and 'coal_class' in df.columns:
        class_df = df[df['coal_class'] == coal_class].copy()
        if len(class_df) > 10:
            # 효율 지표: BTX 생산량 / (총 스팀 사용량)
            class_df['efficiency'] = class_df['BTX_generation'] / (
                class_df['Heater_steam_input'] + class_df['BTXdistillator_distl_steam_input'] + 0.001
            )
            # 상위 10% 효율 데이터
            top_10_pct = class_df.nlargest(int(len(class_df) * 0.1), 'efficiency')
            optimal_from_data = {
                'BTX_generation': top_10_pct['BTX_generation'].mean(),
                'Heater_steam_input': top_10_pct['Heater_steam_input'].mean(),
                'BTXdistillator_distl_steam_input': top_10_pct['BTXdistillator_distl_steam_input'].mean()
            }
            optimal_available = True
            st.success(f"📊 {coal_class} 상위 10% 효율 데이터 ({len(top_10_pct)}개) 기반 최적값 계산 완료")
        else:
            optimal_from_data = defaults.to_dict() if hasattr(defaults, 'to_dict') else dict(defaults)
            st.warning(f"⚠️ {coal_class} 데이터가 부족하여 기본값을 사용합니다.")
    else:
        optimal_from_data = defaults.to_dict() if hasattr(defaults, 'to_dict') else dict(defaults)
        st.info("ℹ️ 데이터가 없어 최적값을 계산할 수 없습니다. 기본값을 표시합니다.")

    st.markdown("---")

    # 1. 현재 운영값 입력 (3개 타겟 모두)
    st.markdown("### 📊 현재 운영값 입력")
    st.caption("현재 공장에서 측정된 값을 직접 입력하세요. 키보드로 숫자를 직접 입력할 수 있습니다.")

    current_cols = st.columns(3)
    current_values = {}

    # BTX 생산량
    with current_cols[0]:
        default_btx = float(defaults.get('BTX_generation', 1.5))
        current_values['BTX_generation'] = st.number_input(
            f"🧪 {TARGET_NAMES_KR['BTX_generation']} ({TARGET_UNITS['BTX_generation']})",
            value=default_btx,
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            format="%.3f",
            key="current_btx",
            help="현재 BTX 생산량을 입력하세요"
        )

    # 히터 스팀
    with current_cols[1]:
        default_heater = float(defaults.get('Heater_steam_input', 6.0))
        current_values['Heater_steam_input'] = st.number_input(
            f"🔥 {TARGET_NAMES_KR['Heater_steam_input']} ({TARGET_UNITS['Heater_steam_input']})",
            value=default_heater,
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            format="%.3f",
            key="current_heater",
            help="현재 히터 증기 투입량을 입력하세요"
        )

    # BTX 증류탑 스팀
    with current_cols[2]:
        default_btx_steam = float(defaults.get('BTXdistillator_distl_steam_input', 4.0))
        current_values['BTXdistillator_distl_steam_input'] = st.number_input(
            f"💨 {TARGET_NAMES_KR['BTXdistillator_distl_steam_input']} ({TARGET_UNITS['BTXdistillator_distl_steam_input']})",
            value=default_btx_steam,
            min_value=0.0,
            max_value=100.0,
            step=0.01,
            format="%.3f",
            key="current_btx_steam",
            help="현재 BTX 증류탑 증기 투입량을 입력하세요"
        )

    st.markdown("---")

    # 2. 최적값 표시 (데이터 기반)
    st.markdown("### 🎯 최적 운전값 (데이터 기반)")
    if optimal_available:
        st.caption(f"{coal_class} 상위 10% 효율 데이터에서 추출한 최적값입니다.")
    else:
        st.caption("⚠️ 데이터가 없어 기본값을 표시합니다. 정확한 최적값 비교를 위해 데이터를 로드하세요.")

    optimal_cols = st.columns(3)
    with optimal_cols[0]:
        opt_btx = optimal_from_data.get('BTX_generation', current_values['BTX_generation'])
        st.metric(
            label=f"🧪 {TARGET_NAMES_KR['BTX_generation']}",
            value=f"{opt_btx:.3f} {TARGET_UNITS['BTX_generation']}",
            delta=f"{opt_btx - current_values['BTX_generation']:+.3f}"
        )
    with optimal_cols[1]:
        opt_heater = optimal_from_data.get('Heater_steam_input', current_values['Heater_steam_input'])
        st.metric(
            label=f"🔥 {TARGET_NAMES_KR['Heater_steam_input']}",
            value=f"{opt_heater:.3f} {TARGET_UNITS['Heater_steam_input']}",
            delta=f"{opt_heater - current_values['Heater_steam_input']:+.3f}",
            delta_color="inverse"  # 스팀은 낮을수록 좋음
        )
    with optimal_cols[2]:
        opt_btx_steam = optimal_from_data.get('BTXdistillator_distl_steam_input', current_values['BTXdistillator_distl_steam_input'])
        st.metric(
            label=f"💨 {TARGET_NAMES_KR['BTXdistillator_distl_steam_input']}",
            value=f"{opt_btx_steam:.3f} {TARGET_UNITS['BTXdistillator_distl_steam_input']}",
            delta=f"{opt_btx_steam - current_values['BTXdistillator_distl_steam_input']:+.3f}",
            delta_color="inverse"  # 스팀은 낮을수록 좋음
        )

    st.markdown("---")

    # 3. 주요 인자 입력 (선택사항)
    with st.expander("🔧 주요 인자 입력 (선택사항 - 모델 예측용)", expanded=False):
        st.caption("모델 예측을 사용하려면 주요 피처 값을 입력하세요.")
        input_data = {}

        feature_rows = [TOP_5_FEATURES[:3], TOP_5_FEATURES[3:]]
        for row_features in feature_rows:
            feature_cols = st.columns(3)
            for i, feature in enumerate(row_features):
                with feature_cols[i]:
                    default_val = float(defaults.get(feature, 0)) if feature in defaults else 0.0
                    input_data[feature] = st.number_input(
                        FEATURE_NAMES_KR.get(feature, feature),
                        value=default_val,
                        step=0.01,
                        format="%.3f",
                        key=f"input_{feature}"
                    )

        # 나머지 피처는 기본값으로 채움
        if df is not None:
            all_features = [c for c in df.columns if c not in TARGET_COLUMNS + ['Date', 'batch_no', 'coal_class']]
            for feature in all_features:
                if feature not in input_data:
                    input_data[feature] = float(defaults.get(feature, 0)) if feature in defaults else 0.0

    st.markdown("---")

    # 4. 단가 입력
    st.markdown("### 💰 단가 입력 (수익성 계산용)")
    price_cols = st.columns(2)
    with price_cols[0]:
        btx_price = st.number_input(
            "조경유(BTX) 단가 (원/ton)",
            value=500000.0,
            step=10000.0,
            format="%.0f",
            help="BTX 1톤당 판매 단가"
        )
    with price_cols[1]:
        steam_price = st.number_input(
            "스팀 단가 (원/ton)",
            value=50000.0,
            step=1000.0,
            format="%.0f",
            help="스팀 1톤당 비용"
        )

    st.markdown("---")

    # 5. 수익성 분석 버튼
    if st.button("💵 수익성 분석 실행", type="primary", use_container_width=True):

        # 현재값
        current_btx = current_values['BTX_generation']
        current_heater_steam = current_values['Heater_steam_input']
        current_btx_steam = current_values['BTXdistillator_distl_steam_input']

        # 최적값 (데이터 기반)
        optimal_btx = optimal_from_data.get('BTX_generation', current_btx)
        optimal_heater_steam = optimal_from_data.get('Heater_steam_input', current_heater_steam)
        optimal_btx_steam = optimal_from_data.get('BTXdistillator_distl_steam_input', current_btx_steam)

        # ========== 결과 표시 ==========
        st.markdown("## 📊 현재값 vs 최적값 비교")

        # 결과 테이블
        result_data = [
            {
                '항목': TARGET_NAMES_KR['BTX_generation'],
                '현재값': f"{current_btx:.3f}",
                '최적값': f"{optimal_btx:.3f}",
                '차이': f"{optimal_btx - current_btx:+.3f}",
                '단위': 'ton/hr',
                '평가': '📈 증가 필요' if optimal_btx > current_btx else ('✅ 양호' if abs(optimal_btx - current_btx) < 0.1 else '📉 과잉')
            },
            {
                '항목': TARGET_NAMES_KR['Heater_steam_input'],
                '현재값': f"{current_heater_steam:.3f}",
                '최적값': f"{optimal_heater_steam:.3f}",
                '차이': f"{optimal_heater_steam - current_heater_steam:+.3f}",
                '단위': 'ton/hr',
                '평가': '📉 절감 필요' if current_heater_steam > optimal_heater_steam else ('✅ 양호' if abs(optimal_heater_steam - current_heater_steam) < 0.1 else '📈 부족')
            },
            {
                '항목': TARGET_NAMES_KR['BTXdistillator_distl_steam_input'],
                '현재값': f"{current_btx_steam:.3f}",
                '최적값': f"{optimal_btx_steam:.3f}",
                '차이': f"{optimal_btx_steam - current_btx_steam:+.3f}",
                '단위': 'ton/hr',
                '평가': '📉 절감 필요' if current_btx_steam > optimal_btx_steam else ('✅ 양호' if abs(optimal_btx_steam - current_btx_steam) < 0.1 else '📈 부족')
            }
        ]

        result_df = pd.DataFrame(result_data)
        st.dataframe(result_df, use_container_width=True, hide_index=True)

        # ========== 수익성 분석 ==========
        st.markdown("## 💵 수익성 분석")
        st.markdown("**수익성 = 조경유(BTX) 추가 생산량 × 단가 - 스팀 추가 사용량 × 단가**")

        # 추가 생산/사용량 계산
        btx_diff = optimal_btx - current_btx  # BTX 추가 생산량
        heater_steam_diff = optimal_heater_steam - current_heater_steam  # 히터 스팀 변화량
        btx_steam_diff = optimal_btx_steam - current_btx_steam  # BTX증류 스팀 변화량
        total_steam_diff = heater_steam_diff + btx_steam_diff  # 총 스팀 변화량

        # 금액 계산 (시간당)
        btx_revenue_diff = btx_diff * btx_price  # 조경유 추가 수익
        steam_cost_diff = total_steam_diff * steam_price  # 스팀 비용 변화 (감소하면 음수)

        # 수익성 = 조경유 추가 생산 수익 - 스팀 추가 비용
        profit_diff_hr = btx_revenue_diff - steam_cost_diff

        # 일간/월간 계산
        profit_diff_day = profit_diff_hr * 24
        profit_diff_month = profit_diff_day * 30

        # 수익성 핵심 지표
        st.markdown("### 💰 수익성 핵심 지표 (최적 운전 시)")
        metric_cols = st.columns(3)
        with metric_cols[0]:
            st.metric(
                label="조경유 추가 생산 수익",
                value=f"{btx_revenue_diff:+,.0f}원/hr",
                delta=f"BTX {btx_diff:+.3f} ton/hr"
            )
        with metric_cols[1]:
            # 스팀 비용: 감소하면 절감(+), 증가하면 손실(-)
            steam_saving = -steam_cost_diff  # 음수가 되면 절감
            st.metric(
                label="스팀 비용 변화",
                value=f"{steam_cost_diff:+,.0f}원/hr",
                delta=f"스팀 {total_steam_diff:+.3f} ton/hr"
            )
        with metric_cols[2]:
            st.metric(
                label="💵 순 수익성 개선",
                value=f"{profit_diff_hr:+,.0f}원/hr",
                delta="개선 가능" if profit_diff_hr > 0 else ("현재 양호" if profit_diff_hr == 0 else "현재가 더 효율적")
            )

        # 기간별 수익성
        st.markdown("### 📅 기간별 수익성 개선 효과")
        period_cols = st.columns(3)
        with period_cols[0]:
            st.metric(
                label="일간 (24시간)",
                value=f"{profit_diff_day:+,.0f}원"
            )
        with period_cols[1]:
            st.metric(
                label="월간 (30일)",
                value=f"{profit_diff_month:+,.0f}원"
            )
        with period_cols[2]:
            st.metric(
                label="연간 (12개월)",
                value=f"{profit_diff_month * 12:+,.0f}원"
            )

        # 상세 수익성 테이블
        st.markdown("### 📋 상세 수익성 분석")
        profit_detail = pd.DataFrame([
            {
                '구분': 'BTX 생산량 변화',
                '현재': f"{current_btx:.3f} ton/hr",
                '최적': f"{optimal_btx:.3f} ton/hr",
                '차이': f"{btx_diff:+.3f} ton/hr",
                '금액 영향': f"{btx_revenue_diff:+,.0f}원/hr"
            },
            {
                '구분': '히터 스팀 변화',
                '현재': f"{current_heater_steam:.3f} ton/hr",
                '최적': f"{optimal_heater_steam:.3f} ton/hr",
                '차이': f"{heater_steam_diff:+.3f} ton/hr",
                '금액 영향': f"{heater_steam_diff * steam_price:+,.0f}원/hr"
            },
            {
                '구분': 'BTX증류 스팀 변화',
                '현재': f"{current_btx_steam:.3f} ton/hr",
                '최적': f"{optimal_btx_steam:.3f} ton/hr",
                '차이': f"{btx_steam_diff:+.3f} ton/hr",
                '금액 영향': f"{btx_steam_diff * steam_price:+,.0f}원/hr"
            },
            {
                '구분': '💵 순 수익성',
                '현재': '-',
                '최적': '-',
                '차이': '-',
                '금액 영향': f"{profit_diff_hr:+,.0f}원/hr"
            }
        ])
        st.dataframe(profit_detail, use_container_width=True, hide_index=True)

        # 수익성 종합 평가
        if profit_diff_hr > 0:
            st.success(f"""
            ### 🎉 최적화 적용 시 예상 수익 개선

            **시간당: +{profit_diff_hr:,.0f}원**

            **일간: +{profit_diff_day:,.0f}원**

            **월간: +{profit_diff_month:,.0f}원**

            **연간: +{profit_diff_month * 12:,.0f}원**

            > 💡 **권장사항**: {coal_class} 상위 10% 효율 조건으로 운전 조건을 조정하세요.
            """)
        elif profit_diff_hr < 0:
            st.info(f"""
            ### ✅ 현재 운영 상태 양호

            현재 운영 조건이 {coal_class} 상위 10% 평균보다 **{-profit_diff_hr:,.0f}원/hr** 더 효율적입니다!

            > 💡 현재 운전 조건을 유지하세요.
            """)
        else:
            st.info("현재 운영이 최적 상태에 근접합니다.")

        # ========== 비교 차트 ==========
        st.markdown("## 📈 시각화")

        # 현재 vs 최적 비교 차트
        fig = go.Figure()
        chart_labels = ['BTX 생산량', '히터 스팀', 'BTX증류 스팀']
        current_vals = [current_btx, current_heater_steam, current_btx_steam]
        optimal_vals = [optimal_btx, optimal_heater_steam, optimal_btx_steam]

        fig.add_trace(go.Bar(
            name='현재값',
            x=chart_labels,
            y=current_vals,
            marker_color=CHART_COLORS['primary'],
            text=[f"{v:.2f}" for v in current_vals],
            textposition='auto'
        ))
        fig.add_trace(go.Bar(
            name=f'최적값 ({coal_class} 상위 10%)',
            x=chart_labels,
            y=optimal_vals,
            marker_color=CHART_COLORS['success'],
            text=[f"{v:.2f}" for v in optimal_vals],
            textposition='auto'
        ))

        fig.update_layout(
            title="현재값 vs 최적값 비교 (ton/hr)",
            barmode='group',
            height=400,
            yaxis_title="ton/hr"
        )
        st.plotly_chart(fig, use_container_width=True)

        # 수익성 구성 차트
        fig_profit = go.Figure()
        fig_profit.add_trace(go.Bar(
            x=['BTX 추가 수익', '스팀 비용 변화', '순 수익성'],
            y=[btx_revenue_diff, steam_cost_diff, profit_diff_hr],
            marker_color=[
                CHART_COLORS['success'] if btx_revenue_diff >= 0 else CHART_COLORS['danger'],
                CHART_COLORS['danger'] if steam_cost_diff > 0 else CHART_COLORS['success'],
                CHART_COLORS['success'] if profit_diff_hr >= 0 else CHART_COLORS['danger']
            ],
            text=[f"{btx_revenue_diff:+,.0f}", f"{steam_cost_diff:+,.0f}", f"{profit_diff_hr:+,.0f}"],
            textposition='auto'
        ))
        fig_profit.update_layout(
            title="수익성 구성 (원/hr)",
            height=350,
            yaxis_title="원/hr"
        )
        st.plotly_chart(fig_profit, use_container_width=True)

        # 예측 기록 저장
        st.session_state.prediction_history.append({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'coal_class': coal_class,
            'current_values': current_values,
            'optimal_values': optimal_from_data,
            'profit_diff_hr': profit_diff_hr,
            'profit_diff_day': profit_diff_day,
            'btx_diff': btx_diff,
            'steam_diff': total_steam_diff
        })


# ============================================================
# 5. 결과 분석
# ============================================================
elif menu == "결과 분석":
    st.markdown('<div class="main-header">📈 결과 분석</div>', unsafe_allow_html=True)

    # 모델 미업로드시 경고
    if not st.session_state.models_loaded:
        st.warning("⚠️ 모델이 업로드되지 않았습니다. '모델 업로드' 메뉴에서 모델을 먼저 업로드해주세요.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🏆 Feature Importance", "📜 예측 기록", "💾 다운로드"])

    with tab1:
        st.subheader("Feature Importance 분석")

        if st.session_state.models_loaded and st.session_state.predictor:
            predictor = st.session_state.predictor

            # 모델 선택
            available_models = predictor.get_available_models()
            selected_model = st.selectbox("모델 선택", available_models)

            # 타겟 선택
            target_options = ["전체 평균"] + TARGET_COLUMNS
            selected_target = st.selectbox(
                "타겟 선택",
                target_options,
                format_func=lambda x: "전체 평균" if x == "전체 평균" else TARGET_NAMES_KR.get(x, x)
            )

            top_n = st.slider("표시할 피처 수", 5, 30, 15)

            # Feature Importance 조회
            target = None if selected_target == "전체 평균" else selected_target
            importance_df = predictor.get_feature_importance(selected_model, target, top_n)

            if not importance_df.empty:
                col1, col2 = st.columns([2, 1])

                with col1:
                    fig = create_feature_importance_chart(importance_df, f"{selected_model} - Feature Importance")
                    st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("**상위 피처 목록**")
                    display_df = importance_df[['feature_kr' if 'feature_kr' in importance_df.columns else 'feature', 'importance']].copy()
                    display_df.columns = ['피처', '중요도']
                    display_df['중요도'] = display_df['중요도'].round(4)
                    st.dataframe(display_df, use_container_width=True)
            else:
                st.info("Feature Importance 데이터가 없습니다.")
        else:
            st.warning("⚠️ 모델을 먼저 로드해주세요.")

    with tab2:
        st.subheader("예측 기록")

        if st.session_state.prediction_history:
            for i, record in enumerate(reversed(st.session_state.prediction_history[-10:])):
                with st.expander(f"📌 {record['timestamp']} - {record['coal_class']}"):

                    # 예측 결과
                    st.markdown("**예측값:**")
                    for model_name, preds in record['predictions'].items():
                        st.write(f"*{model_name}:*")
                        pred_df = pd.DataFrame([{
                            TARGET_NAMES_KR[t]: f"{v:.3f}" for t, v in preds.items()
                        }])
                        st.dataframe(pred_df, use_container_width=True)

                    # 실제값
                    st.markdown("**실제값:**")
                    actual_df = pd.DataFrame([{
                        TARGET_NAMES_KR[t]: f"{v:.3f}" for t, v in record['actuals'].items()
                    }])
                    st.dataframe(actual_df, use_container_width=True)

            if st.button("🗑️ 기록 초기화"):
                st.session_state.prediction_history = []
                st.rerun()
        else:
            st.info("예측 기록이 없습니다. '예측 실행' 메뉴에서 예측을 수행해주세요.")

    with tab3:
        st.subheader("데이터 다운로드")

        df = st.session_state.raw_data

        if df is not None:
            # 원본 데이터 다운로드
            st.markdown("**원본 데이터**")
            csv = df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 원본 데이터 (CSV)",
                data=csv,
                file_name="4GTP_raw_data.csv",
                mime="text/csv"
            )

        # 예측 기록 다운로드
        if st.session_state.prediction_history:
            st.markdown("**예측 기록**")

            history_data = []
            for record in st.session_state.prediction_history:
                row = {
                    'timestamp': record['timestamp'],
                    'coal_class': record['coal_class']
                }
                for target in TARGET_COLUMNS:
                    row[f'{target}_actual'] = record['actuals'].get(target)
                    for model_name, preds in record['predictions'].items():
                        row[f'{target}_{model_name}_pred'] = preds.get(target)
                history_data.append(row)

            history_df = pd.DataFrame(history_data)
            csv = history_df.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                label="📥 예측 기록 (CSV)",
                data=csv,
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        # 모델 성능 다운로드
        if st.session_state.models_loaded and st.session_state.predictor:
            st.markdown("**모델 성능 리포트**")

            performance = st.session_state.predictor.get_model_performance()
            perf_data = []
            for model_name, perf in performance.items():
                for target, metrics in perf.items():
                    row = {
                        'model': model_name,
                        'target': TARGET_NAMES_KR.get(target, target),
                        **{k: v for k, v in metrics.items() if not isinstance(v, dict)}
                    }
                    perf_data.append(row)

            if perf_data:
                perf_df = pd.DataFrame(perf_data)
                csv = perf_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button(
                    label="📥 모델 성능 리포트 (CSV)",
                    data=csv,
                    file_name="model_performance.csv",
                    mime="text/csv"
                )


# ============================================================
# 6. 설정
# ============================================================
elif menu == "설정":
    st.markdown('<div class="main-header">⚙️ 설정</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["📁 경로 설정", "🎨 테마", "ℹ️ 정보"])

    with tab1:
        st.subheader("경로 설정")

        st.markdown(f"""
        | 항목 | 경로 |
        |------|------|
        | 데이터 파일 | `{DATA_FILE}` |
        | 모델 디렉토리 | `{MODELS_DIR}` |
        | 앱 디렉토리 | `{APP_DIR}` |
        """)

    with tab2:
        st.subheader("테마 설정")

        st.info("Streamlit 설정에서 테마를 변경할 수 있습니다.")

        st.markdown("""
        **테마 변경 방법:**
        1. 우측 상단 메뉴 (≡) 클릭
        2. Settings 선택
        3. Theme에서 Light/Dark 선택
        """)

    with tab3:
        st.subheader("시스템 정보")

        st.markdown(f"""
        **4GTP 다중 타겟 예측 시스템**

        - 버전: 1.0.0
        - 개발일: 2025-11-28
        - 프레임워크: Streamlit

        **타겟 변수:**
        """)

        for target in TARGET_COLUMNS:
            st.markdown(f"- {TARGET_NAMES_KR[target]} ({target})")

        st.markdown("""
        ---

        **사용 기술:**
        - Python 3.9+
        - Streamlit
        - scikit-learn / PyCaret
        - Plotly
        - Pandas / NumPy
        """)


# ============================================================
# 푸터
# ============================================================
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #888; padding: 1rem;">
        🏭 4GTP 다중 타겟 예측 시스템 | © 2025 | Powered by Streamlit & Claude
    </div>
    """,
    unsafe_allow_html=True
)

print("[STARTUP] App initialization complete!")
