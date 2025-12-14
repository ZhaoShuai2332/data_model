from sklearn.metrics import (confusion_matrix, precision_score, recall_score,
                             f1_score, roc_curve, auc, precision_recall_curve)
import numpy as np

class MetricsCalculator:
    """计算各种评估指标"""
    
    @staticmethod
    def calculate_metrics(y_true, y_pred, y_prob, threshold=0.5):
        """
        计算所有评估指标
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签（二值）
            y_prob: 预测概率
            threshold: 分类阈值
        """
        # 基础指标
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        
        # 计算各项指标
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # ROC AUC
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # PR AUC
        precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall_curve, precision_curve)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'specificity': specificity,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        return metrics
    
    @staticmethod
    def find_best_threshold(y_true, y_prob, metric='f1'):
        """
        寻找最佳阈值
        
        Args:
            y_true: 真实标签
            y_prob: 预测概率
            metric: 优化指标 ('f1', 'precision', 'recall')
        """
        thresholds = np.linspace(0, 1, 100)
        best_score = 0
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (y_prob >= threshold).astype(int)
            
            if metric == 'f1':
                score = f1_score(y_true, y_pred, zero_division=0)
            elif metric == 'precision':
                score = precision_score(y_true, y_pred, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_true, y_pred, zero_division=0)
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        return best_threshold, best_score