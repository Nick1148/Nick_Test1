"""
Module 1: 데이터 전처리 모듈
============================
원시 데이터를 모델 학습에 적합한 형태로 변환합니다.

주요 기능:
- 데이터 로드 및 탐색
- 결측치/이상치 처리
- 범주형 변수 인코딩
- 수치형 변수 스케일링
- 시계열 특성 추가
- 데이터 분할 (시간 순서 유지)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    TARGET_COLUMNS, EXCLUDE_COLUMNS, CATEGORICAL_COLUMNS,
    COAL_CLASSES, TRAIN_RATIO, VALIDATION_RATIO, TEST_RATIO,
    DATA_FILE, MODELS_DIR, FEATURE_GROUPS
)


class DataPreprocessor:
    """
    데이터 전처리 클래스

    4GTP 공정 데이터를 전처리하여 모델 학습에 적합한 형태로 변환합니다.
    통합 데이터 및 Coal Class별 분리 데이터를 모두 지원합니다.
    """

    def __init__(self, data_path: Optional[str] = None):
        """
        초기화

        Args:
            data_path: 데이터 파일 경로 (기본값: config에서 설정된 경로)
        """
        self.data_path = Path(data_path) if data_path else DATA_FILE
        self.raw_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.scalers: Dict[str, StandardScaler] = {}
        self.encoders: Dict[str, LabelEncoder] = {}
        self.feature_columns: List[str] = []
        self.statistics: Dict[str, Any] = {}

    def load_data(self) -> pd.DataFrame:
        """
        데이터 로드

        Returns:
            로드된 원시 DataFrame
        """
        print(f"[INFO] 데이터 로드 중: {self.data_path}")

        self.raw_data = pd.read_excel(self.data_path)
        self.raw_data['Date'] = pd.to_datetime(self.raw_data['Date'])

        # 기본 통계 저장
        self.statistics['raw'] = {
            'n_rows': len(self.raw_data),
            'n_cols': len(self.raw_data.columns),
            'date_range': (
                self.raw_data['Date'].min().strftime('%Y-%m-%d'),
                self.raw_data['Date'].max().strftime('%Y-%m-%d')
            ),
            'missing_values': self.raw_data.isnull().sum().sum(),
            'coal_class_distribution': self.raw_data['coal_class'].value_counts().to_dict()
        }

        print(f"[OK] 데이터 로드 완료: {self.statistics['raw']['n_rows']:,}행 x {self.statistics['raw']['n_cols']}열")
        print(f"     기간: {self.statistics['raw']['date_range'][0]} ~ {self.statistics['raw']['date_range'][1]}")

        return self.raw_data

    def explore_data(self) -> Dict[str, Any]:
        """
        탐색적 데이터 분석 (EDA)

        Returns:
            EDA 결과 딕셔너리
        """
        if self.raw_data is None:
            self.load_data()

        df = self.raw_data

        eda_results = {
            'basic_info': {
                'shape': df.shape,
                'dtypes': df.dtypes.astype(str).to_dict(),
                'memory_usage': df.memory_usage(deep=True).sum() / 1024**2  # MB
            },
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'numeric_stats': df.describe().to_dict(),
            'target_stats': {},
            'target_correlations': df[TARGET_COLUMNS].corr().to_dict(),
            'coal_class_analysis': {}
        }

        # 타겟 변수별 통계
        for target in TARGET_COLUMNS:
            eda_results['target_stats'][target] = {
                'mean': df[target].mean(),
                'std': df[target].std(),
                'min': df[target].min(),
                'max': df[target].max(),
                'median': df[target].median(),
                'skewness': df[target].skew(),
                'kurtosis': df[target].kurtosis()
            }

        # Coal Class별 분석
        for coal_class in COAL_CLASSES:
            class_data = df[df['coal_class'] == coal_class]
            eda_results['coal_class_analysis'][coal_class] = {
                'count': len(class_data),
                'percentage': len(class_data) / len(df) * 100,
                'target_means': class_data[TARGET_COLUMNS].mean().to_dict(),
                'target_stds': class_data[TARGET_COLUMNS].std().to_dict()
            }

        self.statistics['eda'] = eda_results
        return eda_results

    def handle_missing_values(self, method: str = 'interpolate') -> pd.DataFrame:
        """
        결측치 처리

        Args:
            method: 처리 방법 ('drop', 'mean', 'median', 'interpolate')

        Returns:
            결측치 처리된 DataFrame
        """
        df = self.raw_data.copy()

        missing_before = df.isnull().sum().sum()

        if method == 'drop':
            df = df.dropna()
        elif method == 'mean':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
        elif method == 'median':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        elif method == 'interpolate':
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            df[numeric_cols] = df[numeric_cols].interpolate(method='linear')
            df = df.fillna(method='bfill').fillna(method='ffill')

        missing_after = df.isnull().sum().sum()

        self.statistics['missing_handling'] = {
            'method': method,
            'before': missing_before,
            'after': missing_after
        }

        print(f"[OK] 결측치 처리: {missing_before} -> {missing_after} ({method})")

        return df

    def handle_outliers(self, method: str = 'iqr', threshold: float = 1.5) -> pd.DataFrame:
        """
        이상치 처리

        Args:
            method: 처리 방법 ('iqr', 'zscore', 'clip')
            threshold: IQR 배수 또는 Z-score 임계값

        Returns:
            이상치 처리된 DataFrame
        """
        df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col not in TARGET_COLUMNS + ['Date']]

        outliers_count = 0

        if method == 'iqr':
            for col in numeric_cols:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - threshold * IQR
                upper = Q3 + threshold * IQR

                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outliers_count += outliers

                df[col] = df[col].clip(lower=lower, upper=upper)

        elif method == 'zscore':
            for col in numeric_cols:
                z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
                outliers = (z_scores > threshold).sum()
                outliers_count += outliers

                df.loc[z_scores > threshold, col] = df[col].median()

        elif method == 'clip':
            for col in numeric_cols:
                lower = df[col].quantile(0.01)
                upper = df[col].quantile(0.99)
                outliers = ((df[col] < lower) | (df[col] > upper)).sum()
                outliers_count += outliers
                df[col] = df[col].clip(lower=lower, upper=upper)

        self.statistics['outlier_handling'] = {
            'method': method,
            'threshold': threshold,
            'outliers_processed': outliers_count
        }

        print(f"[OK] 이상치 처리: {outliers_count}개 ({method}, threshold={threshold})")

        self.processed_data = df
        return df

    def add_time_features(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        시계열 특성 추가

        Args:
            df: 입력 DataFrame (None이면 processed_data 사용)

        Returns:
            시계열 특성이 추가된 DataFrame
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        df = df.copy()

        # 시간 관련 특성
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day'] = df['Date'].dt.day
        df['hour'] = df['Date'].dt.hour
        df['dayofweek'] = df['Date'].dt.dayofweek
        df['quarter'] = df['Date'].dt.quarter
        df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)

        # 주기적 특성 (사인/코사인 변환)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        print(f"[OK] 시계열 특성 추가 완료: +12개 컬럼")

        self.processed_data = df
        return df

    def add_lag_features(self, df: Optional[pd.DataFrame] = None,
                         lag_columns: Optional[List[str]] = None,
                         lags: List[int] = [1, 2, 3]) -> pd.DataFrame:
        """
        랙(Lag) 특성 추가

        Args:
            df: 입력 DataFrame
            lag_columns: 랙 특성을 추가할 컬럼 목록
            lags: 랙 간격 목록

        Returns:
            랙 특성이 추가된 DataFrame
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        df = df.copy()

        if lag_columns is None:
            lag_columns = TARGET_COLUMNS

        for col in lag_columns:
            for lag in lags:
                df[f'{col}_lag{lag}'] = df[col].shift(lag)

        # 결측값 제거 (랙으로 인한)
        df = df.dropna()

        print(f"[OK] 랙 특성 추가 완료: +{len(lag_columns) * len(lags)}개 컬럼")

        self.processed_data = df
        return df

    def add_rolling_features(self, df: Optional[pd.DataFrame] = None,
                            rolling_columns: Optional[List[str]] = None,
                            windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
        """
        롤링 통계 특성 추가

        Args:
            df: 입력 DataFrame
            rolling_columns: 롤링 특성을 추가할 컬럼 목록
            windows: 윈도우 크기 목록

        Returns:
            롤링 특성이 추가된 DataFrame
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        df = df.copy()

        if rolling_columns is None:
            rolling_columns = TARGET_COLUMNS

        for col in rolling_columns:
            for window in windows:
                df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window).mean()
                df[f'{col}_rolling_std_{window}'] = df[col].rolling(window=window).std()

        # 결측값 제거
        df = df.dropna()

        print(f"[OK] 롤링 특성 추가 완료: +{len(rolling_columns) * len(windows) * 2}개 컬럼")

        self.processed_data = df
        return df

    def encode_categorical(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        범주형 변수 인코딩

        Args:
            df: 입력 DataFrame

        Returns:
            인코딩된 DataFrame
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        df = df.copy()

        for col in CATEGORICAL_COLUMNS:
            if col in df.columns:
                le = LabelEncoder()
                df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
                self.encoders[col] = le

                # One-Hot 인코딩도 추가
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)

        print(f"[OK] 범주형 변수 인코딩 완료: {CATEGORICAL_COLUMNS}")

        self.processed_data = df
        return df

    def scale_features(self, df: Optional[pd.DataFrame] = None,
                       method: str = 'standard',
                       exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        수치형 변수 스케일링

        Args:
            df: 입력 DataFrame
            method: 스케일링 방법 ('standard', 'minmax')
            exclude_cols: 스케일링에서 제외할 컬럼

        Returns:
            스케일링된 DataFrame
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        df = df.copy()

        if exclude_cols is None:
            exclude_cols = []

        # 스케일링할 컬럼 선택 (수치형만, 타겟과 제외 컬럼 제외)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        scale_cols = [col for col in numeric_cols
                      if col not in TARGET_COLUMNS + exclude_cols + ['Date', 'year', 'month', 'day', 'hour', 'dayofweek', 'quarter', 'is_weekend']
                      and not col.endswith('_encoded') and not col.startswith('coal_class_')]

        if method == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()

        df[scale_cols] = scaler.fit_transform(df[scale_cols])
        self.scalers['features'] = scaler
        self.scalers['feature_columns'] = scale_cols

        print(f"[OK] 스케일링 완료: {len(scale_cols)}개 컬럼 ({method})")

        self.processed_data = df
        return df

    def prepare_features(self, df: Optional[pd.DataFrame] = None,
                        include_time_features: bool = True,
                        include_lag_features: bool = False,
                        include_rolling_features: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        최종 피처 준비

        Args:
            df: 입력 DataFrame
            include_time_features: 시계열 특성 포함 여부
            include_lag_features: 랙 특성 포함 여부
            include_rolling_features: 롤링 특성 포함 여부

        Returns:
            (준비된 DataFrame, 피처 컬럼 목록)
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        # 제외할 컬럼
        exclude = EXCLUDE_COLUMNS + TARGET_COLUMNS + CATEGORICAL_COLUMNS

        # 피처 컬럼 선택
        feature_cols = [col for col in df.columns
                        if col not in exclude
                        and col != 'Date'
                        and df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]

        self.feature_columns = feature_cols

        print(f"[OK] 최종 피처 수: {len(feature_cols)}개")

        return df, feature_cols

    def split_data(self, df: Optional[pd.DataFrame] = None,
                   feature_cols: Optional[List[str]] = None,
                   method: str = 'time') -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        데이터 분할 (시계열 순서 유지)

        Args:
            df: 입력 DataFrame
            feature_cols: 피처 컬럼 목록
            method: 분할 방법 ('time', 'random')

        Returns:
            {'train': (X, y), 'val': (X, y), 'test': (X, y)}
        """
        if df is None:
            df = self.processed_data
        if feature_cols is None:
            feature_cols = self.feature_columns

        df = df.sort_values('Date').reset_index(drop=True)

        n = len(df)
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VALIDATION_RATIO))

        if method == 'time':
            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]
        else:
            # Random split (시계열에는 권장하지 않음)
            train_df, temp_df = train_test_split(df, test_size=(1-TRAIN_RATIO), random_state=42)
            val_df, test_df = train_test_split(temp_df, test_size=TEST_RATIO/(TEST_RATIO+VALIDATION_RATIO), random_state=42)

        result = {
            'train': (train_df[feature_cols], train_df[TARGET_COLUMNS]),
            'val': (val_df[feature_cols], val_df[TARGET_COLUMNS]),
            'test': (test_df[feature_cols], test_df[TARGET_COLUMNS]),
            'train_df': train_df,
            'val_df': val_df,
            'test_df': test_df
        }

        self.statistics['data_split'] = {
            'train_size': len(train_df),
            'val_size': len(val_df),
            'test_size': len(test_df),
            'train_date_range': (train_df['Date'].min(), train_df['Date'].max()),
            'val_date_range': (val_df['Date'].min(), val_df['Date'].max()),
            'test_date_range': (test_df['Date'].min(), test_df['Date'].max())
        }

        print(f"[OK] 데이터 분할 완료:")
        print(f"     Train: {len(train_df):,}개 ({train_df['Date'].min().strftime('%Y-%m-%d')} ~ {train_df['Date'].max().strftime('%Y-%m-%d')})")
        print(f"     Validation: {len(val_df):,}개 ({val_df['Date'].min().strftime('%Y-%m-%d')} ~ {val_df['Date'].max().strftime('%Y-%m-%d')})")
        print(f"     Test: {len(test_df):,}개 ({test_df['Date'].min().strftime('%Y-%m-%d')} ~ {test_df['Date'].max().strftime('%Y-%m-%d')})")

        return result

    def split_by_coal_class(self, df: Optional[pd.DataFrame] = None) -> Dict[str, pd.DataFrame]:
        """
        Coal Class별 데이터 분리

        Args:
            df: 입력 DataFrame

        Returns:
            {'ClassA': df, 'ClassB': df, 'ClassC': df, 'integrated': df}
        """
        if df is None:
            df = self.processed_data if self.processed_data is not None else self.raw_data.copy()

        result = {'integrated': df}

        for coal_class in COAL_CLASSES:
            class_df = df[df['coal_class'] == coal_class].copy()
            result[coal_class] = class_df
            print(f"     {coal_class}: {len(class_df):,}개 ({len(class_df)/len(df)*100:.1f}%)")

        return result

    def run_full_pipeline(self,
                         missing_method: str = 'interpolate',
                         outlier_method: str = 'iqr',
                         scale_method: str = 'standard',
                         include_time_features: bool = True,
                         include_lag_features: bool = False,
                         include_rolling_features: bool = False) -> Dict[str, Any]:
        """
        전체 전처리 파이프라인 실행

        Args:
            missing_method: 결측치 처리 방법
            outlier_method: 이상치 처리 방법
            scale_method: 스케일링 방법
            include_time_features: 시계열 특성 포함 여부
            include_lag_features: 랙 특성 포함 여부
            include_rolling_features: 롤링 특성 포함 여부

        Returns:
            전처리 결과 딕셔너리
        """
        print("=" * 60)
        print("[START] 전처리 파이프라인 시작")
        print("=" * 60)

        # 1. 데이터 로드
        self.load_data()

        # 2. EDA
        self.explore_data()

        # 3. 결측치 처리
        self.processed_data = self.handle_missing_values(method=missing_method)

        # 4. 이상치 처리
        self.handle_outliers(method=outlier_method)

        # 5. 시계열 특성 추가
        if include_time_features:
            self.add_time_features()

        # 6. 랙 특성 추가
        if include_lag_features:
            self.add_lag_features()

        # 7. 롤링 특성 추가
        if include_rolling_features:
            self.add_rolling_features()

        # 8. 범주형 인코딩
        self.encode_categorical()

        # 9. 스케일링
        self.scale_features(method=scale_method)

        # 10. 피처 준비
        df, feature_cols = self.prepare_features(
            include_time_features=include_time_features,
            include_lag_features=include_lag_features,
            include_rolling_features=include_rolling_features
        )

        # 11. Coal Class별 분리
        print("\n[INFO] Coal Class별 데이터 분리:")
        class_data = self.split_by_coal_class(df)

        # 12. 각 데이터셋 분할
        print("\n[INFO] 데이터셋별 Train/Val/Test 분할:")
        split_results = {}
        for name, class_df in class_data.items():
            print(f"\n[{name}]")
            if len(class_df) > 100:  # 최소 데이터 수 확인
                split_results[name] = self.split_data(class_df, feature_cols)
            else:
                print(f"     [WARN] 데이터 부족으로 분할 스킵")

        print("\n" + "=" * 60)
        print("[DONE] 전처리 파이프라인 완료")
        print("=" * 60)

        return {
            'processed_data': df,
            'feature_columns': feature_cols,
            'class_data': class_data,
            'split_results': split_results,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'statistics': self.statistics
        }

    def save_preprocessor(self, path: Optional[str] = None):
        """
        전처리기 저장

        Args:
            path: 저장 경로
        """
        if path is None:
            path = MODELS_DIR / 'preprocessor.pkl'

        save_data = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_columns': self.feature_columns,
            'statistics': self.statistics
        }

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(save_data, path)
        print(f"[OK] 전처리기 저장 완료: {path}")

    @classmethod
    def load_preprocessor(cls, path: Optional[str] = None) -> 'DataPreprocessor':
        """
        전처리기 로드

        Args:
            path: 로드 경로

        Returns:
            로드된 DataPreprocessor 인스턴스
        """
        if path is None:
            path = MODELS_DIR / 'preprocessor.pkl'

        save_data = joblib.load(path)

        preprocessor = cls()
        preprocessor.scalers = save_data['scalers']
        preprocessor.encoders = save_data['encoders']
        preprocessor.feature_columns = save_data['feature_columns']
        preprocessor.statistics = save_data['statistics']

        print(f"[OK] 전처리기 로드 완료: {path}")

        return preprocessor

    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """
        새로운 데이터에 동일한 전처리 적용

        Args:
            new_data: 새로운 입력 데이터

        Returns:
            전처리된 DataFrame
        """
        df = new_data.copy()

        # 시계열 특성 추가
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df['year'] = df['Date'].dt.year
            df['month'] = df['Date'].dt.month
            df['day'] = df['Date'].dt.day
            df['hour'] = df['Date'].dt.hour
            df['dayofweek'] = df['Date'].dt.dayofweek
            df['quarter'] = df['Date'].dt.quarter
            df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
            df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
            df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
            df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)

        # 범주형 인코딩
        for col, encoder in self.encoders.items():
            if col in df.columns:
                df[f'{col}_encoded'] = encoder.transform(df[col].astype(str))
                # One-Hot 인코딩
                for cls in encoder.classes_:
                    df[f'{col}_{cls}'] = (df[col] == cls).astype(int)

        # 스케일링
        if 'features' in self.scalers and 'feature_columns' in self.scalers:
            scale_cols = [col for col in self.scalers['feature_columns'] if col in df.columns]
            if scale_cols:
                df[scale_cols] = self.scalers['features'].transform(df[scale_cols])

        return df


# 메인 실행
if __name__ == "__main__":
    # 전처리기 초기화 및 실행
    preprocessor = DataPreprocessor()
    results = preprocessor.run_full_pipeline(
        include_time_features=True,
        include_lag_features=False,
        include_rolling_features=False
    )

    # 전처리기 저장
    preprocessor.save_preprocessor()

    print("\n[INFO] 최종 결과 요약:")
    print(f"     전체 피처 수: {len(results['feature_columns'])}")
    print(f"     데이터셋 수: {len(results['split_results'])}")
