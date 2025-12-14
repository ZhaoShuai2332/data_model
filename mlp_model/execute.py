from data_preprocess import DataPreprocessor
from train import RiskPredictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# 设置随机种子以保证可复现性
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)




def main():
    """主函数"""
    
    print("=" * 60)
    print("高风险客户预测 - PyTorch深度学习实现")
    print("=" * 60)
    
    # 1. 数据加载
    preprocessor = DataPreprocessor()
    
    # 注意: 请根据实际路径修改文件名
    train_data = preprocessor.load_data('train.csv')
    test_data = preprocessor.load_data('test.csv')
    
    # 2. 数据预处理
    X, y = preprocessor.preprocess_train(train_data, target_col='是否高风险')
    
    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    
    print(f"\n数据集划分:")
    print(f"训练集: {X_train.shape}, 正样本比例: {y_train.mean():.4f}")
    print(f"验证集: {X_val.shape}, 正样本比例: {y_val.mean():.4f}")
    
    # 4. 创建并训练模型
    input_dim = X_train.shape[1]
    
    predictor = RiskPredictor(
        input_dim=input_dim,
        hidden_dims=[256, 128, 64],  # 可调整网络结构
        dropout_rate=0.3,
        learning_rate=0.001,
        use_focal_loss=True  # 使用Focal Loss处理不平衡
    )
    
    # 训练
    best_auc = predictor.fit(
        X_train, y_train, 
        X_val, y_val,
        epochs=100,
        batch_size=256,
        patience=15
    )
    
    # 5. 在验证集上评估
    print("\n" + "=" * 60)
    print("验证集评估结果:")
    print("=" * 60)
    
    val_probs = predictor.predict(X_val)
    val_preds = (val_probs > 0.5).astype(int)
    
    print("\n混淆矩阵:")
    print(confusion_matrix(y_val, val_preds))
    
    print("\n分类报告:")
    print(classification_report(y_val, val_preds, 
                              target_names=['低风险', '高风险']))
    
    # 计算AUC
    fpr, tpr, thresholds = roc_curve(y_val, val_probs)
    final_auc = auc(fpr, tpr)
    print(f"\n最终AUC: {final_auc:.4f}")
    
    # 6. 绘制ROC曲线
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {final_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.savefig('mlp_model\\outputs\\roc_curve.png', dpi=150, bbox_inches='tight')
    print("\nROC曲线已保存: mlp_model\\outputs\\roc_curve.png")
    
    # 7. 绘制训练历史
    predictor.plot_training_history()
    
    # 8. 在测试集上预测
    print("\n" + "=" * 60)
    print("测试集预测:")
    print("=" * 60)
    
    X_test = preprocessor.preprocess_test(test_data)
    test_probs = predictor.predict(X_test)
    
    print(f"测试集预测完成，样本数: {len(test_probs)}")
    print(f"预测高风险比例: {(test_probs > 0.5).mean():.4f}")
    
    # 9. 保存预测结果
    result_df = pd.DataFrame(test_probs)
    result_df.to_csv('mlp_model\\outputs\\result.csv', 
                     encoding='utf-8', 
                     index=False, 
                     header=False)
    
    print("\n预测结果已保存: mlp_model\\outputs\\result.csv")
    print(f"文件格式: {len(test_probs)} 行 × 1 列")
    
    # 10. 保存模型
    torch.save({
        'model_state_dict': predictor.model.state_dict(),
        'input_dim': input_dim,
        'feature_cols': preprocessor.feature_cols,
        'scaler_mean': preprocessor.scaler.mean_,
        'scaler_scale': preprocessor.scaler.scale_,
    }, 'mlp_model\\outputs\\risk_model_final.pth')
    
    print("\n完整模型已保存: mlp_model\\outputs\\risk_model_final.pth")
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()


