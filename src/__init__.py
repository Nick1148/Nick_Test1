"""
4GTP 다중 타겟 예측 모델 소스 패키지
===================================
"""

from .preprocessing import DataPreprocessor
from .training import ModelTrainer
from .prediction import MultiTargetPredictor

__all__ = ['DataPreprocessor', 'ModelTrainer', 'MultiTargetPredictor']
__version__ = '1.0.0'
