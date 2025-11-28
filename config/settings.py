"""
프로젝트 설정 파일
==================
다중 타겟 예측 모델의 전역 설정값을 관리합니다.
"""

import os
from pathlib import Path

# ============================================================
# 경로 설정
# ============================================================
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# 데이터 파일 경로
DATA_FILE = DATA_DIR / "4GTP_integrated_with_coal_Raw.xlsx"

# ============================================================
# 데이터 컬럼 설정
# ============================================================

# 타겟 변수 (예측 대상)
TARGET_COLUMNS = [
    'BTX_generation',
    'Heater_steam_input',
    'BTXdistillator_distl_steam_input'
]

# 타겟 변수 한글명
TARGET_NAMES_KR = {
    'BTX_generation': 'BTX 생산량',
    'Heater_steam_input': '히터 증기 투입량',
    'BTXdistillator_distl_steam_input': 'BTX 증류탑 증기 투입량'
}

# 타겟 변수 단위
TARGET_UNITS = {
    'BTX_generation': 'ton/hr',
    'Heater_steam_input': 'ton/hr',
    'BTXdistillator_distl_steam_input': 'ton/hr'
}

# 제외할 컬럼
EXCLUDE_COLUMNS = [
    'Date',
    'batch_no',  # 고유값이 너무 많음
]

# 범주형 변수
CATEGORICAL_COLUMNS = [
    'coal_class'
]

# Coal Class 목록
COAL_CLASSES = ['ClassA', 'ClassB', 'ClassC']

# ============================================================
# 피처 그룹 정의 (시각화용)
# ============================================================
FEATURE_GROUPS = {
    '가스/오일 관련': [
        'BS_out_COG_T', 'BS_out_COG_P', 'BS_out_COG_F', 'BS_aoil_flow'
    ],
    '온도 관련': [
        'Tank_RO_T', 'HE_VO_RO_T', 'HE_OO_RO_T', 'TOP_out_AO_T',
        'HE_OO_AO_T', 'HE_OW_AO_T', 'HE_spiral_AO_T', 'HE_VO_V_T', 'HE_VW_W_T'
    ],
    '압력 관련': [
        'HE_OO_RO_P', 'BTXdistillator_top_pres', 'BTXdistillator_bottom_pre', 'Heater_pres'
    ],
    'BTX 증류탑 관련': [
        'BTXdistillator_RO_flow', 'BTXdistillator_recycle_oil_flow', 'BTXdistillator_top_temp'
    ],
    '히터 관련': [
        'Heater_temp'
    ],
    '석탄 배합': [
        '호주탄', '러시아탄', '미국탄', '국내탄', '캐나다탄'
    ]
}

# 피처 한글명 매핑
FEATURE_NAMES_KR = {
    'BS_out_COG_T': 'BS 출구 COG 온도',
    'BS_out_COG_P': 'BS 출구 COG 압력',
    'BS_out_COG_F': 'BS 출구 COG 유량',
    'BS_aoil_flow': '흡수유 유량',
    'Tank_RO_T': '탱크 리치오일 온도',
    'HE_VO_RO_T': '열교환기 VO 리치오일 온도',
    'HE_OO_RO_T': '열교환기 OO 리치오일 온도',
    'HE_OO_RO_P': '열교환기 OO 리치오일 압력',
    'TOP_out_AO_T': 'TOP 출구 흡수유 온도',
    'HE_OO_AO_T': '열교환기 OO 흡수유 온도',
    'HE_OW_AO_T': '열교환기 OW 흡수유 온도',
    'HE_spiral_AO_T': '열교환기 spiral 흡수유 온도',
    'HE_VO_V_T': '열교환기 VO 증기 온도',
    'HE_VW_W_T': '열교환기 VW 물 온도',
    'BTXdistillator_RO_flow': 'BTX 증류탑 리치오일 유량',
    'BTXdistillator_recycle_oil_flow': 'BTX 증류탑 리사이클 오일 유량',
    'BTXdistillator_top_temp': 'BTX 증류탑 상부 온도',
    'BTXdistillator_top_pres': 'BTX 증류탑 상부 압력',
    'BTXdistillator_bottom_pre': 'BTX 증류탑 하부 압력',
    'Heater_pres': '히터 압력',
    'Heater_temp': '히터 온도',
    'coal_class': '석탄 등급',
    '호주탄': '호주탄 배합비',
    '러시아탄': '러시아탄 배합비',
    '미국탄': '미국탄 배합비',
    '국내탄': '국내탄 배합비',
    '캐나다탄': '캐나다탄 배합비'
}

# ============================================================
# 모델 설정
# ============================================================

# 모델 파일명
MODEL_FILES = {
    'integrated': 'model_integrated.pkl',
    'ClassA': 'model_ClassA.pkl',
    'ClassB': 'model_ClassB.pkl',
    'ClassC': 'model_ClassC.pkl'
}

# 데이터 분할 비율
TRAIN_RATIO = 0.7
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

# 시계열 분할 날짜 (옵션)
TRAIN_END_DATE = '2024-06-30'
VALIDATION_END_DATE = '2024-12-31'

# ============================================================
# AutoML 설정 (PyCaret)
# ============================================================
PYCARET_SETTINGS = {
    'session_id': 42,
    'normalize': True,
    'transformation': False,
    'ignore_low_variance': True,
    'remove_multicollinearity': True,
    'multicollinearity_threshold': 0.95,
    'n_jobs': -1,
    'use_gpu': False,
    'verbose': False
}

# 비교할 모델 목록
MODELS_TO_COMPARE = [
    'rf',      # Random Forest
    'et',      # Extra Trees
    'xgboost', # XGBoost
    'lightgbm', # LightGBM
    'catboost', # CatBoost
    'gbr',     # Gradient Boosting
]

# ============================================================
# 성능 목표
# ============================================================
PERFORMANCE_TARGETS = {
    'BTX_generation': {'R2': 0.85, 'RMSE': 0.3},
    'Heater_steam_input': {'R2': 0.75, 'RMSE': 0.5},
    'BTXdistillator_distl_steam_input': {'R2': 0.75, 'RMSE': 0.4}
}

# ============================================================
# UI 설정
# ============================================================
UI_SETTINGS = {
    'page_title': '4GTP 다중 타겟 예측 시스템',
    'page_icon': '🏭',
    'layout': 'wide',
    'theme_color': '#1f77b4',
    'success_color': '#2ecc71',
    'warning_color': '#f39c12',
    'danger_color': '#e74c3c'
}

# 차트 색상 팔레트
CHART_COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb78',
    'info': '#17becf',
    'ClassA': '#e74c3c',
    'ClassB': '#3498db',
    'ClassC': '#2ecc71',
    'integrated': '#9b59b6'
}
