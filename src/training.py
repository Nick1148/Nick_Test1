"""
Module 2: 모델 학습 모듈
========================
AutoML(PyCaret)을 활용한 다중 타겟 예측 모델 학습

주요 기능:
- PyCaret AutoML을 통한 자동 모델 선택
- 다중 타겟 회귀 모델 학습
- 하이퍼파라미터 자동 튜닝
- 모델 성능 평가 및 비교
- PKL 파일로 모델 저장
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import joblib
import warnings
from datetime import datetime
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import json

warnings.filterwarnings('ignore')

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config.settings import (
    TARGET_COLUMNS, TARGET_NAMES_KR, COAL_CLASSES,
    MODELS_DIR, MODEL_FILES, PYCARET_SETTINGS,
    MODELS_TO_COMPARE, PERFORMANCE_TARGETS
)


class ModelTrainer:
    """
    모델 학습 클래스

    AutoML(PyCaret)을 활용하여 다중 타겟 예측 모델을 학습합니다.
    통합 모델 및 Coal Class별 개별 모델을 모두 지원합니다.
    """

    def __init__(self, use_pycaret: bool = True):
        """
        초기화

        Args:
            use_pycaret: PyCaret 사용 여부 (False면 sklearn 직접 사용)
        """
        self.use_pycaret = use_pycaret
        self.models: Dict[str, Any] = {}
        self.model_performance: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}
        self.training_history: Dict[str, Any] = {}
        self.best_models: Dict[str, str] = {}

        # PyCaret 사용 가능 여부 확인
        self.pycaret_available = False
        if use_pycaret:
            try:
                from pycaret.regression import setup, compare_models, tune_model, finalize_model
                self.pycaret_available = True
                print("[OK] PyCaret 사용 가능")
            except ImportError:
                print("[WARN] PyCaret 미설치 - sklearn으로 대체")
                self.pycaret_available = False

    def train_with_pycaret(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                           X_val: pd.DataFrame, y_val: pd.DataFrame,
                           model_name: str = 'integrated') -> Dict[str, Any]:
        """
        PyCaret을 사용한 모델 학습

        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_val: 검증 피처
            y_val: 검증 타겟
            model_name: 모델 이름 (integrated, ClassA, ClassB, ClassC)

        Returns:
            학습 결과 딕셔너리
        """
        from pycaret.regression import setup, compare_models, tune_model, finalize_model, predict_model, pull

        print(f"\n{'='*60}")
        print(f"[AutoML] PyCaret 학습 시작: {model_name}")
        print(f"{'='*60}")

        results = {}
        trained_models = {}
        performances = {}

        # 각 타겟별로 개별 모델 학습
        for target in TARGET_COLUMNS:
            print(f"\n[TARGET] {TARGET_NAMES_KR[target]} ({target})")

            # 학습 데이터 준비
            train_data = X_train.copy()
            train_data[target] = y_train[target].values

            # PyCaret 설정
            try:
                reg_setup = setup(
                    data=train_data,
                    target=target,
                    session_id=PYCARET_SETTINGS['session_id'],
                    normalize=PYCARET_SETTINGS['normalize'],
                    transformation=PYCARET_SETTINGS['transformation'],
                    ignore_low_variance=PYCARET_SETTINGS['ignore_low_variance'],
                    remove_multicollinearity=PYCARET_SETTINGS['remove_multicollinearity'],
                    multicollinearity_threshold=PYCARET_SETTINGS['multicollinearity_threshold'],
                    verbose=False,
                    html=False
                )

                # 모델 비교
                print(f"   [COMPARE] 모델 비교 중...")
                best_model = compare_models(
                    include=MODELS_TO_COMPARE,
                    n_select=1,
                    sort='R2',
                    verbose=False
                )

                # 결과 테이블 가져오기
                comparison_results = pull()

                # 모델 튜닝
                print(f"   [TUNE] 하이퍼파라미터 튜닝 중...")
                tuned_model = tune_model(best_model, n_iter=20, verbose=False)

                # 최종 모델 (전체 데이터로 재학습)
                final_model = finalize_model(tuned_model)

                # 검증 데이터로 예측
                val_data = X_val.copy()
                val_data[target] = y_val[target].values
                predictions = predict_model(final_model, data=val_data)

                # 성능 계산
                y_true = y_val[target].values
                y_pred = predictions['prediction_label'].values

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100

                performances[target] = {
                    'R2': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'MAPE': round(mape, 2),
                    'best_model': type(final_model).__name__,
                    'comparison_results': comparison_results.to_dict() if comparison_results is not None else None
                }

                trained_models[target] = final_model

                print(f"   [OK] 완료 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                print(f"      최적 모델: {type(final_model).__name__}")

            except Exception as e:
                print(f"   [ERROR] PyCaret 오류: {e}")
                print(f"   [WARN] sklearn으로 대체하여 학습...")

                # sklearn으로 대체
                model = self._train_sklearn_model(
                    X_train, y_train[[target]],
                    X_val, y_val[[target]]
                )
                trained_models[target] = model

                # 성능 계산
                y_pred = model.predict(X_val)
                y_true = y_val[target].values

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)

                performances[target] = {
                    'R2': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'MAPE': 0,
                    'best_model': 'RandomForestRegressor'
                }

        results = {
            'models': trained_models,
            'performances': performances,
            'model_name': model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        self.models[model_name] = trained_models
        self.model_performance[model_name] = performances

        return results

    def train_with_sklearn(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                          X_val: pd.DataFrame, y_val: pd.DataFrame,
                          model_name: str = 'integrated') -> Dict[str, Any]:
        """
        sklearn을 사용한 모델 학습 (PyCaret 대안)

        Args:
            X_train: 학습 피처
            y_train: 학습 타겟
            X_val: 검증 피처
            y_val: 검증 타겟
            model_name: 모델 이름

        Returns:
            학습 결과 딕셔너리
        """
        print(f"\n{'='*60}")
        print(f"[ML] sklearn 모델 학습 시작: {model_name}")
        print(f"{'='*60}")

        # 다양한 모델 후보
        model_candidates = {
            'RandomForest': RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'ExtraTrees': ExtraTreesRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, max_depth=10, random_state=42),
        }

        # XGBoost, LightGBM 추가 (설치된 경우)
        try:
            from xgboost import XGBRegressor
            model_candidates['XGBoost'] = XGBRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbosity=0
            )
        except ImportError:
            pass

        try:
            from lightgbm import LGBMRegressor
            model_candidates['LightGBM'] = LGBMRegressor(
                n_estimators=200, max_depth=10, learning_rate=0.1,
                random_state=42, n_jobs=-1, verbose=-1
            )
        except ImportError:
            pass

        results = {}
        trained_models = {}
        performances = {}

        for target in TARGET_COLUMNS:
            print(f"\n[TARGET] {TARGET_NAMES_KR[target]} ({target})")

            best_model = None
            best_r2 = -np.inf
            best_model_name = ""
            model_scores = {}

            # 각 모델 후보 평가
            for name, model in model_candidates.items():
                try:
                    model.fit(X_train, y_train[target])
                    y_pred = model.predict(X_val)
                    r2 = r2_score(y_val[target], y_pred)
                    model_scores[name] = r2

                    if r2 > best_r2:
                        best_r2 = r2
                        best_model = model
                        best_model_name = name

                except Exception as e:
                    print(f"   [WARN] {name} 오류: {e}")

            # 최적 모델로 최종 학습
            if best_model is not None:
                # 전체 학습 데이터로 재학습
                X_full = pd.concat([X_train, X_val])
                y_full = pd.concat([y_train[target], y_val[target]])
                best_model.fit(X_full, y_full)

                # 검증 성능
                y_pred = best_model.predict(X_val)
                y_true = y_val[target].values

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100

                performances[target] = {
                    'R2': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'MAPE': round(mape, 2),
                    'best_model': best_model_name,
                    'model_comparison': model_scores
                }

                trained_models[target] = best_model

                # Feature Importance 저장
                if hasattr(best_model, 'feature_importances_'):
                    importance_df = pd.DataFrame({
                        'feature': X_train.columns,
                        'importance': best_model.feature_importances_
                    }).sort_values('importance', ascending=False)

                    if model_name not in self.feature_importance:
                        self.feature_importance[model_name] = {}
                    self.feature_importance[model_name][target] = importance_df

                print(f"   [OK] 완료 - R²: {r2:.4f}, RMSE: {rmse:.4f}, MAE: {mae:.4f}")
                print(f"      최적 모델: {best_model_name}")
                print(f"      모델 비교: {model_scores}")

        results = {
            'models': trained_models,
            'performances': performances,
            'model_name': model_name,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        self.models[model_name] = trained_models
        self.model_performance[model_name] = performances

        return results

    def _train_sklearn_model(self, X_train: pd.DataFrame, y_train: pd.DataFrame,
                            X_val: pd.DataFrame, y_val: pd.DataFrame) -> Any:
        """
        단일 sklearn 모델 학습 (내부 헬퍼)
        """
        model = RandomForestRegressor(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train.values.ravel())
        return model

    def train_all_models(self, split_results: Dict[str, Dict],
                         feature_columns: List[str]) -> Dict[str, Any]:
        """
        모든 모델 학습 (통합 + Coal Class별)

        Args:
            split_results: 전처리에서 분할된 데이터
            feature_columns: 피처 컬럼 목록

        Returns:
            전체 학습 결과
        """
        print("\n" + "=" * 70)
        print("[START] 전체 모델 학습 시작 (통합 + Coal Class별)")
        print("=" * 70)

        all_results = {}

        for model_name, data in split_results.items():
            print(f"\n{'─'*50}")
            print(f"[MODEL] 모델: {model_name}")
            print(f"{'─'*50}")

            X_train, y_train = data['train']
            X_val, y_val = data['val']

            print(f"   학습 데이터: {len(X_train):,}개")
            print(f"   검증 데이터: {len(X_val):,}개")

            # 데이터 수가 충분한지 확인
            if len(X_train) < 100:
                print(f"   [WARN] 데이터 부족으로 학습 스킵")
                continue

            # 학습 실행
            if self.use_pycaret and self.pycaret_available:
                try:
                    results = self.train_with_pycaret(X_train, y_train, X_val, y_val, model_name)
                except Exception as e:
                    print(f"   [WARN] PyCaret 오류: {e}")
                    results = self.train_with_sklearn(X_train, y_train, X_val, y_val, model_name)
            else:
                results = self.train_with_sklearn(X_train, y_train, X_val, y_val, model_name)

            all_results[model_name] = results

        # 학습 히스토리 저장
        self.training_history = {
            'results': all_results,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_columns': feature_columns
        }

        # 성능 요약 출력
        self._print_performance_summary()

        return all_results

    def _print_performance_summary(self):
        """성능 요약 출력"""
        print("\n" + "=" * 70)
        print("[INFO] 모델 성능 요약")
        print("=" * 70)

        for model_name, performances in self.model_performance.items():
            print(f"\n[{model_name}]")
            print("-" * 50)
            print(f"{'타겟':<35} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
            print("-" * 50)

            for target, metrics in performances.items():
                target_kr = TARGET_NAMES_KR.get(target, target)
                print(f"{target_kr:<30} {metrics['R2']:>8.4f} {metrics['RMSE']:>8.4f} {metrics['MAE']:>8.4f}")

            # 평균 성능
            avg_r2 = np.mean([m['R2'] for m in performances.values()])
            avg_rmse = np.mean([m['RMSE'] for m in performances.values()])
            avg_mae = np.mean([m['MAE'] for m in performances.values()])
            print("-" * 50)
            print(f"{'평균':<30} {avg_r2:>8.4f} {avg_rmse:>8.4f} {avg_mae:>8.4f}")

    def evaluate_on_test(self, split_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """
        테스트 데이터로 최종 평가

        Args:
            split_results: 전처리에서 분할된 데이터

        Returns:
            테스트 성능 결과
        """
        print("\n" + "=" * 70)
        print("[TEST] 테스트 데이터 평가")
        print("=" * 70)

        test_results = {}

        for model_name, data in split_results.items():
            if model_name not in self.models:
                continue

            X_test, y_test = data['test']
            models = self.models[model_name]

            print(f"\n[{model_name}]")
            performances = {}

            for target in TARGET_COLUMNS:
                if target not in models:
                    continue

                model = models[target]
                y_pred = model.predict(X_test)
                y_true = y_test[target].values

                r2 = r2_score(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                mae = mean_absolute_error(y_true, y_pred)
                mape = mean_absolute_percentage_error(y_true, y_pred) * 100

                performances[target] = {
                    'R2': round(r2, 4),
                    'RMSE': round(rmse, 4),
                    'MAE': round(mae, 4),
                    'MAPE': round(mape, 2),
                    'predictions': y_pred,
                    'actuals': y_true
                }

                target_kr = TARGET_NAMES_KR.get(target, target)
                print(f"   {target_kr}: R²={r2:.4f}, RMSE={rmse:.4f}, MAE={mae:.4f}")

            test_results[model_name] = performances

        return test_results

    def predict(self, X: pd.DataFrame, model_name: str = 'integrated') -> pd.DataFrame:
        """
        예측 수행

        Args:
            X: 입력 피처
            model_name: 사용할 모델 이름

        Returns:
            예측 결과 DataFrame
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이(가) 학습되지 않았습니다.")

        models = self.models[model_name]
        predictions = {}

        for target in TARGET_COLUMNS:
            if target in models:
                predictions[target] = models[target].predict(X)

        return pd.DataFrame(predictions)

    def predict_with_confidence(self, X: pd.DataFrame, model_name: str = 'integrated',
                               n_iterations: int = 100) -> Dict[str, Any]:
        """
        신뢰구간과 함께 예측 (Bootstrap 방식)

        Args:
            X: 입력 피처
            model_name: 사용할 모델 이름
            n_iterations: Bootstrap 반복 횟수

        Returns:
            예측값, 신뢰구간 포함 딕셔너리
        """
        if model_name not in self.models:
            raise ValueError(f"모델 '{model_name}'이(가) 학습되지 않았습니다.")

        models = self.models[model_name]
        results = {}

        for target in TARGET_COLUMNS:
            if target not in models:
                continue

            model = models[target]

            # 기본 예측
            base_prediction = model.predict(X)

            # RandomForest의 경우 개별 트리 예측으로 신뢰구간 계산
            if hasattr(model, 'estimators_'):
                tree_predictions = np.array([tree.predict(X) for tree in model.estimators_])
                mean_pred = np.mean(tree_predictions, axis=0)
                std_pred = np.std(tree_predictions, axis=0)

                results[target] = {
                    'prediction': base_prediction,
                    'mean': mean_pred,
                    'std': std_pred,
                    'ci_lower': mean_pred - 1.96 * std_pred,
                    'ci_upper': mean_pred + 1.96 * std_pred
                }
            else:
                results[target] = {
                    'prediction': base_prediction,
                    'mean': base_prediction,
                    'std': np.zeros_like(base_prediction),
                    'ci_lower': base_prediction,
                    'ci_upper': base_prediction
                }

        return results

    def get_feature_importance(self, model_name: str = 'integrated',
                              target: Optional[str] = None,
                              top_n: int = 20) -> pd.DataFrame:
        """
        Feature Importance 조회

        Args:
            model_name: 모델 이름
            target: 타겟 변수 (None이면 전체)
            top_n: 상위 N개

        Returns:
            Feature Importance DataFrame
        """
        if model_name not in self.feature_importance:
            # 모델에서 직접 추출
            if model_name in self.models:
                self.feature_importance[model_name] = {}
                for t, model in self.models[model_name].items():
                    if hasattr(model, 'feature_importances_'):
                        importance_df = pd.DataFrame({
                            'feature': model.feature_names_in_ if hasattr(model, 'feature_names_in_') else range(len(model.feature_importances_)),
                            'importance': model.feature_importances_
                        }).sort_values('importance', ascending=False)
                        self.feature_importance[model_name][t] = importance_df

        if model_name not in self.feature_importance:
            return pd.DataFrame()

        if target:
            if target in self.feature_importance[model_name]:
                return self.feature_importance[model_name][target].head(top_n)
            return pd.DataFrame()
        else:
            # 전체 타겟의 평균 importance
            all_importance = []
            for t, df in self.feature_importance[model_name].items():
                df = df.copy()
                df['target'] = t
                all_importance.append(df)

            if all_importance:
                combined = pd.concat(all_importance)
                avg_importance = combined.groupby('feature')['importance'].mean().reset_index()
                return avg_importance.sort_values('importance', ascending=False).head(top_n)

            return pd.DataFrame()

    def save_models(self, output_dir: Optional[Path] = None):
        """
        모든 모델 저장

        Args:
            output_dir: 저장 디렉토리
        """
        if output_dir is None:
            output_dir = MODELS_DIR

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n[SAVE] 모델 저장 중: {output_dir}")

        for model_name, models in self.models.items():
            model_data = {
                'models': models,
                'performance': self.model_performance.get(model_name, {}),
                'feature_importance': self.feature_importance.get(model_name, {}),
                'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'target_columns': TARGET_COLUMNS
            }

            filename = MODEL_FILES.get(model_name, f'model_{model_name}.pkl')
            filepath = output_dir / filename
            joblib.dump(model_data, filepath)
            print(f"   [OK] {model_name}: {filepath}")

        # 메타데이터 저장
        metadata = {
            'model_names': list(self.models.keys()),
            'target_columns': TARGET_COLUMNS,
            'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'performances': self.model_performance
        }

        with open(output_dir / 'metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2, default=str)

        print(f"\n[OK] 모델 저장 완료: {len(self.models)}개")

    @classmethod
    def load_model(cls, model_name: str, model_dir: Optional[Path] = None) -> 'ModelTrainer':
        """
        저장된 모델 로드

        Args:
            model_name: 모델 이름
            model_dir: 모델 디렉토리

        Returns:
            로드된 ModelTrainer 인스턴스
        """
        if model_dir is None:
            model_dir = MODELS_DIR

        model_dir = Path(model_dir)
        filename = MODEL_FILES.get(model_name, f'model_{model_name}.pkl')
        filepath = model_dir / filename

        if not filepath.exists():
            raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {filepath}")

        model_data = joblib.load(filepath)

        trainer = cls(use_pycaret=False)
        trainer.models[model_name] = model_data['models']
        trainer.model_performance[model_name] = model_data.get('performance', {})
        trainer.feature_importance[model_name] = model_data.get('feature_importance', {})

        print(f"[LOAD] 모델 로드 완료: {model_name}")

        return trainer

    @classmethod
    def load_all_models(cls, model_dir: Optional[Path] = None) -> 'ModelTrainer':
        """
        모든 저장된 모델 로드

        Args:
            model_dir: 모델 디렉토리

        Returns:
            로드된 ModelTrainer 인스턴스
        """
        if model_dir is None:
            model_dir = MODELS_DIR

        model_dir = Path(model_dir)
        trainer = cls(use_pycaret=False)

        for model_name, filename in MODEL_FILES.items():
            filepath = model_dir / filename
            if filepath.exists():
                model_data = joblib.load(filepath)
                trainer.models[model_name] = model_data['models']
                trainer.model_performance[model_name] = model_data.get('performance', {})
                trainer.feature_importance[model_name] = model_data.get('feature_importance', {})
                print(f"   [OK] {model_name} 로드 완료")

        print(f"\n[LOAD] 전체 모델 로드 완료: {len(trainer.models)}개")

        return trainer


# 메인 실행
if __name__ == "__main__":
    from preprocessing import DataPreprocessor

    # 1. 전처리
    print("Step 1: 데이터 전처리")
    preprocessor = DataPreprocessor()
    prep_results = preprocessor.run_full_pipeline(
        include_time_features=True,
        include_lag_features=False,
        include_rolling_features=False
    )

    # 2. 모델 학습
    print("\nStep 2: 모델 학습")
    trainer = ModelTrainer(use_pycaret=True)
    training_results = trainer.train_all_models(
        prep_results['split_results'],
        prep_results['feature_columns']
    )

    # 3. 테스트 평가
    print("\nStep 3: 테스트 평가")
    test_results = trainer.evaluate_on_test(prep_results['split_results'])

    # 4. 모델 저장
    print("\nStep 4: 모델 저장")
    trainer.save_models()
    preprocessor.save_preprocessor()

    print("\n[DONE] 전체 학습 파이프라인 완료!")
