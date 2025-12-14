from data_process import DataPreprocessor
from res_net import ResidualBlock1D
from train import ResNetPredictor
from evaluate import MetricsCalculator
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, f1_score
)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")


def main():
    print("=" * 60)
    print("ResNet1D模型 - 高风险客户预测")
    print("使用残差网络 + F1-score评估")
    print("=" * 60)
    
    # 数据加载
    preprocessor = DataPreprocessor()
    train_data = preprocessor.load_data('train.csv')
    test_data = preprocessor.load_data('test.csv')
    
    # 预处理
    X, y = preprocessor.preprocess_train(train_data, target_col='是否高风险')
    
    # 划分数据集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {X_train.shape}")
    print(f"验证集: {X_val.shape}")
    
    # 创建ResNet1D模型
    input_dim = X_train.shape[1]
    
    predictor = ResNetPredictor(
        input_dim=input_dim,
        layers=[2, 2, 2, 2],  # ResNet18配置
        channels=[64, 128, 256, 512],
        dropout_rate=0.3,
        learning_rate=0.001,
        use_focal_loss=True
    )
    
    print(f"\nResNet1D模型结构:")
    print(predictor.model)
    print(f"参数量: {sum(p.numel() for p in predictor.model.parameters()):,}")
    
    # 训练
    best_auc, best_f1 = predictor.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=100,
        batch_size=128,
        patience=15
    )
    
    # 详细评估
    print("\n" + "=" * 60)
    print("验证集详细评估:")
    print("=" * 60)
    
    val_probs = predictor.predict(X_val)
    
    # 寻找最佳阈值
    print("\n寻找最佳分类阈值...")
    best_threshold_f1, best_f1_at_threshold = MetricsCalculator.find_best_threshold(
        y_val, val_probs, metric='f1'
    )
    print(f"最佳F1阈值: {best_threshold_f1:.4f}, F1-Score: {best_f1_at_threshold:.4f}")
    
    # 使用默认阈值0.5评估
    val_preds_05 = (val_probs >= 0.5).astype(int)
    metrics_05 = MetricsCalculator.calculate_metrics(y_val, val_preds_05, val_probs, 0.5)
    
    print(f"\n使用阈值 0.5:")
    print(f"  Accuracy:    {metrics_05['accuracy']:.4f}")
    print(f"  Precision:   {metrics_05['precision']:.4f}")
    print(f"  Recall:      {metrics_05['recall']:.4f}")
    print(f"  F1-Score:    {metrics_05['f1_score']:.4f}")
    print(f"  Specificity: {metrics_05['specificity']:.4f}")
    print(f"  ROC AUC:     {metrics_05['roc_auc']:.4f}")
    print(f"  PR AUC:      {metrics_05['pr_auc']:.4f}")
    
    # 使用最佳阈值评估
    val_preds_best = (val_probs >= best_threshold_f1).astype(int)
    metrics_best = MetricsCalculator.calculate_metrics(y_val, val_preds_best, val_probs, best_threshold_f1)
    
    print(f"\n使用最佳阈值 {best_threshold_f1:.4f}:")
    print(f"  Accuracy:    {metrics_best['accuracy']:.4f}")
    print(f"  Precision:   {metrics_best['precision']:.4f}")
    print(f"  Recall:      {metrics_best['recall']:.4f}")
    print(f"  F1-Score:    {metrics_best['f1_score']:.4f}")
    print(f"  Specificity: {metrics_best['specificity']:.4f}")
    
    # 混淆矩阵
    print("\n混淆矩阵 (阈值=0.5):")
    cm = confusion_matrix(y_val, val_preds_05)
    print(cm)
    print(f"\n  TN: {metrics_05['tn']}, FP: {metrics_05['fp']}")
    print(f"  FN: {metrics_05['fn']}, TP: {metrics_05['tp']}")
    
    # 分类报告
    print("\n分类报告 (阈值=0.5):")
    print(classification_report(y_val, val_preds_05, target_names=['低风险', '高风险']))
    
    # 绘制ROC曲线
    fpr, tpr, _ = roc_curve(y_val, val_probs)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.subplot(2, 2, 1)
    plt.plot(fpr, tpr, lw=2, label=f'ResNet1D (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # 绘制PR曲线
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, val_probs)
    pr_auc = auc(recall_curve, precision_curve)
    
    plt.subplot(2, 2, 2)
    plt.plot(recall_curve, precision_curve, lw=2, label=f'ResNet1D (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    
    # F1-score vs Threshold
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    for t in thresholds:
        preds = (val_probs >= t).astype(int)
        f1_scores.append(f1_score(y_val, preds, zero_division=0))
    
    plt.subplot(2, 2, 3)
    plt.plot(thresholds, f1_scores, lw=2)
    plt.axvline(x=best_threshold_f1, color='r', linestyle='--', 
                label=f'Best Threshold = {best_threshold_f1:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1-Score')
    plt.title('F1-Score vs Threshold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 混淆矩阵可视化
    plt.subplot(2, 2, 4)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['低风险', '高风险'])
    plt.yticks(tick_marks, ['低风险', '高风险'])
    
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    plt.tight_layout()
    plt.savefig('cnn_model\\outputs\\resnet1d_evaluation.png', dpi=150)
    print("\n评估图已保存: cnn_model\\outputs\\resnet1d_evaluation.png")
    
    # 训练历史
    predictor.plot_training_history()
    
    # 测试集预测
    print("\n" + "=" * 60)
    print("测试集预测:")
    print("=" * 60)
    
    X_test = preprocessor.preprocess_test(test_data)
    test_probs = predictor.predict(X_test)
    
    print(f"预测完成，样本数: {len(test_probs)}")
    print(f"预测高风险比例 (阈值=0.5): {(test_probs > 0.5).mean():.4f}")
    print(f"预测高风险比例 (最佳阈值={best_threshold_f1:.3f}): {(test_probs > best_threshold_f1).mean():.4f}")
    print(f"预测概率范围: [{test_probs.min():.4f}, {test_probs.max():.4f}]")
    
    # 保存结果（使用概率，让评估系统自己选择阈值）
    result_df = pd.DataFrame(test_probs)
    result_df.to_csv('Results_resnet1d.csv', 
                     encoding='utf-8', 
                     index=False, 
                     header=False)
    
    print("\n结果已保存: Results_resnet1d.csv")
    
    # 保存模型
    torch.save({
        'model_state_dict': predictor.model.state_dict(),
        'input_dim': input_dim,
        'best_threshold': best_threshold_f1,
        'best_f1': best_f1_at_threshold,
        'best_auc': best_auc,
    }, 'cnn_model\\outputs\\resnet1d_model_final.pth')
    
    print("模型已保存: cnn_model\\outputs\\resnet1d_model_final.pth")
    print("=" * 60)
    
    # 打印最终总结
    print("\n" + "=" * 60)
    print("最终总结:")
    print("=" * 60)
    print(f"最佳ROC AUC:     {best_auc:.4f}")
    print(f"最佳F1-Score:    {best_f1_at_threshold:.4f} (阈值={best_threshold_f1:.3f})")
    print(f"验证集Precision: {metrics_best['precision']:.4f}")
    print(f"验证集Recall:    {metrics_best['recall']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main()