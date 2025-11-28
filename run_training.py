"""
모델 학습 실행 스크립트
======================
이 스크립트를 실행하여 모델을 학습시킵니다.

사용법:
    python run_training.py
"""

import sys
from pathlib import Path

# 경로 설정
ROOT_DIR = Path(__file__).parent
sys.path.insert(0, str(ROOT_DIR))

from src.preprocessing import DataPreprocessor
from src.training import ModelTrainer


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("🏭 4GTP 다중 타겟 예측 모델 학습")
    print("=" * 70)

    # 1. 전처리
    print("\n" + "=" * 70)
    print("📦 Step 1: 데이터 전처리")
    print("=" * 70)

    preprocessor = DataPreprocessor()
    prep_results = preprocessor.run_full_pipeline(
        missing_method='interpolate',
        outlier_method='iqr',
        scale_method='standard',
        include_time_features=True,
        include_lag_features=False,
        include_rolling_features=False
    )

    print(f"\n✅ 전처리 완료")
    print(f"   피처 수: {len(prep_results['feature_columns'])}")
    print(f"   데이터셋: {list(prep_results['split_results'].keys())}")

    # 2. 모델 학습
    print("\n" + "=" * 70)
    print("🤖 Step 2: 모델 학습")
    print("=" * 70)

    # PyCaret 사용 시도, 실패하면 sklearn 사용
    trainer = ModelTrainer(use_pycaret=True)

    training_results = trainer.train_all_models(
        prep_results['split_results'],
        prep_results['feature_columns']
    )

    # 3. 테스트 평가
    print("\n" + "=" * 70)
    print("🧪 Step 3: 테스트 데이터 평가")
    print("=" * 70)

    test_results = trainer.evaluate_on_test(prep_results['split_results'])

    # 4. 모델 저장
    print("\n" + "=" * 70)
    print("💾 Step 4: 모델 저장")
    print("=" * 70)

    trainer.save_models()
    preprocessor.save_preprocessor()

    # 5. 최종 요약
    print("\n" + "=" * 70)
    print("🎉 학습 완료!")
    print("=" * 70)

    print("\n📊 최종 성능 요약:")
    for model_name, perf in trainer.model_performance.items():
        print(f"\n[{model_name}]")
        for target, metrics in perf.items():
            print(f"   {target}: R²={metrics['R2']:.4f}, RMSE={metrics['RMSE']:.4f}")

    print("\n" + "=" * 70)
    print("🚀 Streamlit 앱 실행 방법:")
    print("   streamlit run app/streamlit_app.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
