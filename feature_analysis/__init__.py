# -*- coding: utf-8 -*-
"""
Feature Analysis Package

This package provides tools for analyzing feature importance across different models.

Modules:
    - feature_importance_analysis: Core PermutationImportance class
    - logistic_importance: Logistic Regression feature importance
    - mlp_importance: MLP feature importance
    - svm_importance: SVM feature importance
    - transformer_importance: Transformer feature importance

Usage:
    from feature_analysis import PermutationImportance, compute_importance_for_model
"""

from .feature_importance_analysis import PermutationImportance, compute_importance_for_model

__all__ = [
    'PermutationImportance',
    'compute_importance_for_model'
]
