"""
예측 모듈
=========
학습된 모델을 사용하여 예측을 수행하고 결과를 분석합니다.

주요 기능:
- 다중 모델 예측 (통합 + Class별)
- 예측 결과 비교
- 효율 분석
- 신뢰구간 계산
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import joblib
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    TARGET_COLUMNS, TARGET_NAMES_KR, TARGET_UNITS,
    COAL_CLASSES, MODELS_DIR, MODEL_FILES, FEATURE_NAMES_KR
)


class MultiTargetPredictor:
    """
    다중 타겟 예측 클래스

    학습된 모델들을 로드하여 예측을 수행하고
    결과를 분석합니다.
    """

    def __init__(self, model_dir: Optional[Path] = None):
        """
        초기화

        Args:
            model_dir: 모델 디렉토리 경로
        """
        self.model_dir = Path(model_dir) if model_dir else MODELS_DIR
        self.models: Dict[str, Dict] = {}
        self.preprocessor = None
        self.model_performance: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, Dict] = {}
        self.is_loaded = False

    def load_models(self) -> bool:
        """
        모든 모델 로드

        Returns:
            로드 성공 여부
        """
        print("[LOAD] 모델 로드 중...")

        try:
            for model_name, filename in MODEL_FILES.items():
                filepath = self.model_dir / filename
                if filepath.exists():
                    model_data = joblib.load(filepath)
                    self.models[model_name] = model_data['models']
                    self.model_performance[model_name] = model_data.get('performance', {})
                    self.feature_importance[model_name] = model_data.get('feature_importance', {})
                    print(f"   [OK] {model_name} 로드 완료")

            # 전처리기 로드
            preprocessor_path = self.model_dir / 'preprocessor.pkl'
            if preprocessor_path.exists():
                self.preprocessor = joblib.load(preprocessor_path)
                print(f"   [OK] 전처리기 로드 완료")

            self.is_loaded = len(self.models) > 0
            print(f"\n[INFO] 로드된 모델: {list(self.models.keys())}")

            return self.is_loaded

        except Exception as e:
            print(f"[ERROR] 모델 로드 실패: {e}")
            return False

    def predict_single(self, input_data: Dict[str, float],
                       coal_class: str = 'ClassB') -> Dict[str, Any]:
        """
        단일 입력에 대한 예측

        Args:
            input_data: 입력 피처 딕셔너리
            coal_class: Coal Class

        Returns:
            예측 결과 딕셔너리
        """
        if not self.is_loaded:
            self.load_models()

        # 입력 데이터를 DataFrame으로 변환
        X = pd.DataFrame([input_data])

        # 통합 모델 예측
        integrated_pred = {}
        if 'integrated' in self.models:
            for target, model in self.models['integrated'].items():
                try:
                    pred = model.predict(X)[0]
                    integrated_pred[target] = pred
                except Exception as e:
                    print(f"통합 모델 예측 오류 ({target}): {e}")
                    integrated_pred[target] = None

        # 해당 Class 모델 예측
        class_pred = {}
        if coal_class in self.models:
            for target, model in self.models[coal_class].items():
                try:
                    pred = model.predict(X)[0]
                    class_pred[target] = pred
                except Exception as e:
                    print(f"{coal_class} 모델 예측 오류 ({target}): {e}")
                    class_pred[target] = None

        return {
            'integrated': integrated_pred,
            coal_class: class_pred,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def predict_batch(self, X: pd.DataFrame,
                     coal_class: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        배치 예측

        Args:
            X: 입력 피처 DataFrame
            coal_class: Coal Class (None이면 전체 모델 예측)

        Returns:
            각 모델별 예측 결과 DataFrame
        """
        if not self.is_loaded:
            self.load_models()

        results = {}

        model_names = [coal_class] if coal_class else list(self.models.keys())

        for model_name in model_names:
            if model_name not in self.models:
                continue

            predictions = {}
            for target, model in self.models[model_name].items():
                try:
                    predictions[target] = model.predict(X)
                except Exception as e:
                    print(f"{model_name} 모델 예측 오류 ({target}): {e}")
                    predictions[target] = np.full(len(X), np.nan)

            results[model_name] = pd.DataFrame(predictions)

        return results

    def predict_all_models(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        모든 모델로 예측

        Args:
            X: 입력 피처 DataFrame

        Returns:
            모든 모델의 예측 결과
        """
        return self.predict_batch(X, coal_class=None)

    def compare_predictions(self, X: pd.DataFrame,
                           actual_values: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        모델별 예측 비교

        Args:
            X: 입력 피처 DataFrame
            actual_values: 실제 값 (비교용)

        Returns:
            비교 결과 DataFrame
        """
        predictions = self.predict_all_models(X)

        comparison_data = []

        for target in TARGET_COLUMNS:
            row = {
                'target': target,
                'target_kr': TARGET_NAMES_KR.get(target, target),
                'unit': TARGET_UNITS.get(target, '')
            }

            # 각 모델의 예측값 평균
            for model_name, pred_df in predictions.items():
                if target in pred_df.columns:
                    row[f'{model_name}_pred'] = pred_df[target].mean()

            # 실제값이 있으면 추가
            if actual_values and target in actual_values:
                row['actual'] = actual_values[target]

                # 효율 차이 계산
                for model_name in predictions.keys():
                    if f'{model_name}_pred' in row:
                        diff = row[f'{model_name}_pred'] - row['actual']
                        pct_diff = (diff / row['actual']) * 100 if row['actual'] != 0 else 0
                        row[f'{model_name}_diff'] = diff
                        row[f'{model_name}_diff_pct'] = pct_diff

            comparison_data.append(row)

        return pd.DataFrame(comparison_data)

    def calculate_efficiency(self, predictions: Dict[str, float],
                            actual_values: Dict[str, float]) -> Dict[str, Any]:
        """
        효율 분석 계산

        Args:
            predictions: 예측값
            actual_values: 실제값

        Returns:
            효율 분석 결과
        """
        efficiency = {}

        for target in TARGET_COLUMNS:
            if target not in predictions or target not in actual_values:
                continue

            pred = predictions[target]
            actual = actual_values[target]

            diff = pred - actual
            pct_diff = (diff / actual) * 100 if actual != 0 else 0

            # BTX 생산량은 높을수록 좋음
            # 증기 투입량은 낮을수록 좋음
            if target == 'BTX_generation':
                improvement = "향상" if diff > 0 else "저하"
                is_better = diff > 0
            else:
                improvement = "절감" if diff < 0 else "증가"
                is_better = diff < 0

            efficiency[target] = {
                'prediction': pred,
                'actual': actual,
                'difference': diff,
                'difference_pct': pct_diff,
                'status': improvement,
                'is_better': is_better,
                'target_kr': TARGET_NAMES_KR.get(target, target),
                'unit': TARGET_UNITS.get(target, '')
            }

        # 종합 효율 점수 계산
        scores = []
        for target, data in efficiency.items():
            if target == 'BTX_generation':
                scores.append(data['difference_pct'])
            else:
                scores.append(-data['difference_pct'])

        efficiency['overall_score'] = np.mean(scores) if scores else 0
        efficiency['overall_status'] = "효율적" if efficiency['overall_score'] > 0 else "비효율적"

        return efficiency

    def get_prediction_with_confidence(self, X: pd.DataFrame,
                                       model_name: str = 'integrated') -> Dict[str, Any]:
        """
        신뢰구간과 함께 예측

        Args:
            X: 입력 피처 DataFrame
            model_name: 모델 이름

        Returns:
            예측값과 신뢰구간
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이(가) 없습니다.")

        results = {}

        for target, model in self.models[model_name].items():
            base_pred = model.predict(X)

            # RandomForest의 경우 트리별 예측으로 신뢰구간 계산
            if hasattr(model, 'estimators_'):
                tree_preds = np.array([tree.predict(X) for tree in model.estimators_])
                mean_pred = np.mean(tree_preds, axis=0)
                std_pred = np.std(tree_preds, axis=0)

                results[target] = {
                    'prediction': base_pred,
                    'mean': mean_pred,
                    'std': std_pred,
                    'ci_lower': mean_pred - 1.96 * std_pred,
                    'ci_upper': mean_pred + 1.96 * std_pred,
                    'confidence_level': 0.95
                }
            else:
                results[target] = {
                    'prediction': base_pred,
                    'mean': base_pred,
                    'std': np.zeros_like(base_pred),
                    'ci_lower': base_pred,
                    'ci_upper': base_pred,
                    'confidence_level': 1.0
                }

        return results

    def get_feature_importance(self, model_name: str = 'integrated',
                              target: Optional[str] = None,
                              top_n: int = 15) -> pd.DataFrame:
        """
        Feature Importance 조회

        Args:
            model_name: 모델 이름
            target: 타겟 변수
            top_n: 상위 N개

        Returns:
            Feature Importance DataFrame
        """
        if model_name not in self.models:
            return pd.DataFrame()

        if target:
            if target in self.models[model_name]:
                model = self.models[model_name][target]
                if hasattr(model, 'feature_importances_'):
                    features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(range(len(model.feature_importances_)))
                    df = pd.DataFrame({
                        'feature': features,
                        'importance': model.feature_importances_
                    })
                    df['feature_kr'] = df['feature'].map(lambda x: FEATURE_NAMES_KR.get(x, x))
                    return df.sort_values('importance', ascending=False).head(top_n)
            return pd.DataFrame()

        # 전체 타겟의 평균 importance
        all_importance = []
        for t, model in self.models[model_name].items():
            if hasattr(model, 'feature_importances_'):
                features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else list(range(len(model.feature_importances_)))
                df = pd.DataFrame({
                    'feature': features,
                    'importance': model.feature_importances_,
                    'target': t
                })
                all_importance.append(df)

        if all_importance:
            combined = pd.concat(all_importance)
            avg_df = combined.groupby('feature')['importance'].mean().reset_index()
            avg_df['feature_kr'] = avg_df['feature'].map(lambda x: FEATURE_NAMES_KR.get(x, x))
            return avg_df.sort_values('importance', ascending=False).head(top_n)

        return pd.DataFrame()

    def get_model_performance(self, model_name: Optional[str] = None) -> Dict[str, Any]:
        """
        모델 성능 조회

        Args:
            model_name: 모델 이름 (None이면 전체)

        Returns:
            성능 정보 딕셔너리
        """
        if model_name:
            return self.model_performance.get(model_name, {})
        return self.model_performance

    def get_available_models(self) -> List[str]:
        """
        사용 가능한 모델 목록 반환
        """
        return list(self.models.keys())

    def generate_report(self, X: pd.DataFrame,
                       actual_values: Optional[Dict[str, float]] = None,
                       coal_class: str = 'ClassB') -> Dict[str, Any]:
        """
        종합 예측 리포트 생성

        Args:
            X: 입력 피처 DataFrame
            actual_values: 실제 값 (선택)
            coal_class: Coal Class

        Returns:
            종합 리포트 딕셔너리
        """
        report = {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'coal_class': coal_class,
            'predictions': {},
            'comparison': None,
            'efficiency': None,
            'confidence_intervals': {},
            'feature_importance': {}
        }

        # 예측 수행
        predictions = self.predict_all_models(X)

        for model_name, pred_df in predictions.items():
            report['predictions'][model_name] = pred_df.to_dict('records')[0] if len(pred_df) == 1 else pred_df.to_dict('records')

        # 비교 분석
        if actual_values:
            report['comparison'] = self.compare_predictions(X, actual_values).to_dict('records')

            # 효율 분석
            integrated_pred = predictions.get('integrated', pd.DataFrame())
            if len(integrated_pred) > 0:
                pred_dict = integrated_pred.iloc[0].to_dict()
                report['efficiency'] = self.calculate_efficiency(pred_dict, actual_values)

        # 신뢰구간
        for model_name in self.models.keys():
            try:
                ci = self.get_prediction_with_confidence(X, model_name)
                report['confidence_intervals'][model_name] = {
                    target: {
                        'mean': float(data['mean'][0]) if len(data['mean']) > 0 else None,
                        'ci_lower': float(data['ci_lower'][0]) if len(data['ci_lower']) > 0 else None,
                        'ci_upper': float(data['ci_upper'][0]) if len(data['ci_upper']) > 0 else None
                    }
                    for target, data in ci.items()
                }
            except Exception as e:
                print(f"신뢰구간 계산 오류 ({model_name}): {e}")

        # Feature Importance
        for model_name in self.models.keys():
            fi = self.get_feature_importance(model_name, top_n=10)
            if not fi.empty:
                report['feature_importance'][model_name] = fi.to_dict('records')

        return report


# 테스트
if __name__ == "__main__":
    predictor = MultiTargetPredictor()

    if predictor.load_models():
        print("\n[OK] 모델 로드 성공")
        print(f"   사용 가능 모델: {predictor.get_available_models()}")

        # 성능 조회
        print("\n[INFO] 모델 성능:")
        for model_name, perf in predictor.get_model_performance().items():
            print(f"\n[{model_name}]")
            for target, metrics in perf.items():
                print(f"   {target}: R²={metrics.get('R2', 'N/A')}")
    else:
        print("[ERROR] 모델 로드 실패 - 먼저 학습을 실행하세요.")
