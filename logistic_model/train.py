import sys
import os
from sklearn.model_selection import train_test_split
import joblib
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_process import DataPreprocessor
from logistic_trainer import LogisticTrainer
import config

def main():
    print("="*60)
    print("逻辑回归模型训练流程")
    print("="*60)
    
    # 1. 初始化数据预处理器
    preprocessor = DataPreprocessor()
    
    # 2. 加载和预处理训练数据
    train_data = preprocessor.load_data(config.TRAIN_DATA_PATH)
    X, y = preprocessor.preprocess_train(train_data)
    
    # 3. 划分训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE,
        stratify=y
    )
    
    print(f"\n训练集大小: {X_train.shape}")
    print(f"验证集大小: {X_val.shape}")
    print(f"训练集中正负样本比例: {(y_train==0).mean():.4f} / {(y_train==1).mean():.4f}")
    print(f"验证集中正负样本比例: {(y_val==0).mean():.4f} / {(y_val==1).mean():.4f}")
    
    # 4. 初始化训练器
    trainer = LogisticTrainer(
        params=config.LOGISTIC_PARAMS,
        random_state=config.RANDOM_STATE
    )
    
    # 5. 可选：网格搜索最优参数
    if config.ENABLE_GRID_SEARCH:
        best_params = trainer.grid_search(
            X_train, y_train, 
            config.GRID_SEARCH_PARAMS, 
            cv=config.GRID_SEARCH_CV
        )
        print(f"\n使用网格搜索得到的最优参数进行训练...")
    
    # 6. 训练模型
    history = trainer.train(X_train, y_train, X_val, y_val)
    
    # 7. 保存模型和预处理器
    os.makedirs(config.MODEL_SAVE_DIR, exist_ok=True)
    
    model_path = os.path.join(config.MODEL_SAVE_DIR, "logistic_model.pkl")
    trainer.save_model(model_path)
    
    joblib.dump(preprocessor, config.PREPROCESSOR_PATH)
    print(f"预处理器已保存至: {config.PREPROCESSOR_PATH}")
    
    # 8. 绘制并保存混淆矩阵
    print("\n" + "="*50)
    print("生成可视化图表...")
    print("="*50)
    
    os.makedirs(config.FIGURE_SAVE_DIR, exist_ok=True)
    
    # 验证集混淆矩阵
    val_pred = trainer.predict(X_val)
    confusion_matrix_path = os.path.join(config.FIGURE_SAVE_DIR, "confusion_matrix_val.png")
    trainer.plot_confusion_matrix(y_val, val_pred, 
                                 save_path=confusion_matrix_path,
                                 title='Validation Confusion Matrix')
    
    # 9. 绘制并保存ROC曲线
    train_pred_proba = trainer.predict_proba(X_train)[:, 1]
    val_pred_proba = trainer.predict_proba(X_val)[:, 1]
    
    trainer.plot_roc_curve(
        y_train, train_pred_proba,
        y_val, val_pred_proba,
        save_path=config.ROC_CURVE_PATH
    )
    
    # 10. 显示特征重要性（Top 20）
    print("\n" + "="*50)
    print("Top 20 重要特征（按系数绝对值排序）:")
    print("="*50)
    
    importance = trainer.get_feature_importance(preprocessor.feature_cols, top_n=20)
    for i, (feature, coef) in enumerate(importance, 1):
        print(f"{i:2d}. {feature:40s}: {coef:.6f}")
    
    # 11. 总结
    print("\n" + "="*60)
    print("训练流程总结")
    print("="*60)
    print(f"模型保存路径: {model_path}")
    print(f"预处理器保存路径: {config.PREPROCESSOR_PATH}")
    print(f"ROC曲线保存路径: {config.ROC_CURVE_PATH}")
    print(f"混淆矩阵保存路径: {confusion_matrix_path}")
    print(f"\n训练集 AUC: {history['train']['auc']:.4f}")
    print(f"验证集 AUC: {history['val']['auc']:.4f}")
    print(f"验证集 F1 Score: {history['val']['f1']:.4f}")
    print("\n训练完成！")

if __name__ == "__main__":
    main()
