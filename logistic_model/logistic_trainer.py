import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
import joblib
import os

class LogisticTrainer:
    """逻辑回归模型训练器"""
    
    def __init__(self, params=None, random_state=42):
        """
        初始化训练器
        
        Args:
            params: 逻辑回归参数字典
            random_state: 随机种子
        """
        self.random_state = random_state
        if params is None:
            params = {
                'penalty': 'l2',
                'C': 1.0,
                'solver': 'lbfgs',
                'max_iter': 1000,
                'random_state': random_state,
                'class_weight': 'balanced'
            }
        self.params = params
        self.model = None
        self.history = {}
        
    def grid_search(self, X_train, y_train, param_grid, cv=5):
        """
        网格搜索最优参数
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            param_grid: 参数网格
            cv: 交叉验证折数
        """
        print("\n" + "="*50)
        print("开始网格搜索最优参数...")
        print("="*50)
        
        base_model = LogisticRegression(random_state=self.random_state, max_iter=1000)
        grid_search = GridSearchCV(base_model, param_grid, cv=cv, 
                                   scoring='roc_auc', n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"\n最优参数: {grid_search.best_params_}")
        print(f"最优AUC得分: {grid_search.best_score_:.4f}")
        
        # 更新参数
        self.params.update(grid_search.best_params_)
        
        return grid_search.best_params_
        
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_val: 验证特征
            y_val: 验证标签
        """
        print("\n" + "="*50)
        print("开始训练逻辑回归模型")
        print("="*50)
        
        # 创建并训练模型
        self.model = LogisticRegression(**self.params)
        self.model.fit(X_train, y_train)
        
        # 训练集评估
        train_pred = self.model.predict(X_train)
        train_pred_proba = self.model.predict_proba(X_train)[:, 1]
        
        train_metrics = self._calculate_metrics(y_train, train_pred, train_pred_proba)
        
        print("\n训练集性能:")
        self._print_metrics(train_metrics)
        print("\n训练集混淆矩阵:")
        print(confusion_matrix(y_train, train_pred))
        print("\n训练集分类报告:")
        print(classification_report(y_train, train_pred, digits=4))
        
        self.history['train'] = train_metrics
        self.history['train_confusion_matrix'] = confusion_matrix(y_train, train_pred)
        
        # 验证集评估
        if X_val is not None and y_val is not None:
            val_pred = self.model.predict(X_val)
            val_pred_proba = self.model.predict_proba(X_val)[:, 1]
            
            val_metrics = self._calculate_metrics(y_val, val_pred, val_pred_proba)
            
            print("\n验证集性能:")
            self._print_metrics(val_metrics)
            print("\n验证集混淆矩阵:")
            print(confusion_matrix(y_val, val_pred))
            print("\n验证集分类报告:")
            print(classification_report(y_val, val_pred, digits=4))
            
            self.history['val'] = val_metrics
            self.history['val_confusion_matrix'] = confusion_matrix(y_val, val_pred)
        
        print("\n模型训练完成!")
        print(f"模型参数: {self.params}")
        
        return self.history
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """计算评估指标"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }
        return metrics
    
    def _print_metrics(self, metrics):
        """打印评估指标"""
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None, title='Confusion Matrix'):
        """
        绘制混淆矩阵
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            save_path: 保存路径
            title: 图表标题
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Class 0', 'Class 1'],
                   yticklabels=['Class 0', 'Class 1'])
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"混淆矩阵已保存至: {save_path}")
        
        plt.close()
    
    def plot_roc_curve(self, y_true_train, y_score_train, y_true_val=None, 
                      y_score_val=None, save_path=None):
        """
        绘制ROC曲线
        
        Args:
            y_true_train: 训练集真实标签
            y_score_train: 训练集预测概率
            y_true_val: 验证集真实标签
            y_score_val: 验证集预测概率
            save_path: 保存路径
        """
        plt.figure(figsize=(10, 8))
        
        # 训练集ROC曲线
        fpr_train, tpr_train, _ = roc_curve(y_true_train, y_score_train)
        roc_auc_train = auc(fpr_train, tpr_train)
        plt.plot(fpr_train, tpr_train, lw=2, alpha=0.8,
                label=f'Train ROC (AUC = {roc_auc_train:.4f})')
        
        # 验证集ROC曲线
        if y_true_val is not None and y_score_val is not None:
            fpr_val, tpr_val, _ = roc_curve(y_true_val, y_score_val)
            roc_auc_val = auc(fpr_val, tpr_val)
            plt.plot(fpr_val, tpr_val, lw=2, alpha=0.8,
                    label=f'Validation ROC (AUC = {roc_auc_val:.4f})')
        
        # 对角线
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve', fontsize=14)
        plt.legend(loc='lower right', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC曲线已保存至: {save_path}")
        
        plt.close()
    
    def predict(self, X):
        """预测"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """预测概率"""
        if self.model is None:
            raise ValueError("模型未训练，请先调用train方法")
        return self.model.predict_proba(X)
    
    def save_model(self, filepath):
        """保存模型"""
        if self.model is None:
            raise ValueError("模型未训练，无法保存")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"\n模型已保存至: {filepath}")
    
    def load_model(self, filepath):
        """加载模型"""
        self.model = joblib.load(filepath)
        print(f"\n模型已从 {filepath} 加载")
    
    def get_feature_importance(self, feature_names=None, top_n=20):
        """获取特征重要性（系数绝对值）"""
        if self.model is None:
            raise ValueError("模型未训练")
        
        coefficients = np.abs(self.model.coef_[0])
        
        if feature_names is not None:
            importance = list(zip(feature_names, coefficients))
            importance.sort(key=lambda x: x[1], reverse=True)
            return importance[:top_n]
        else:
            return coefficients
