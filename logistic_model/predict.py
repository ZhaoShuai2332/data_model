import sys
import os
import pandas as pd
import joblib

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logistic_trainer import LogisticTrainer
import config

def main():
    print("="*60)
    print("逻辑回归模型预测流程")
    print("="*60)
    
    # 1. 加载预处理器
    print(f"\n加载预处理器: {config.PREPROCESSOR_PATH}")
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    
    # 2. 加载测试数据
    test_data = preprocessor.load_data(config.TEST_DATA_PATH)
    
    # 3. 预处理测试数据
    X_test = preprocessor.preprocess_test(test_data)
    
    # 4. 加载模型
    model_path = os.path.join(config.MODEL_SAVE_DIR, "logistic_model.pkl")
    print(f"\n加载模型: {model_path}")
    
    trainer = LogisticTrainer()
    trainer.load_model(model_path)
    
    # 5. 进行预测
    print("\n开始预测...")
    predictions = trainer.predict(X_test)
    pred_proba = trainer.predict_proba(X_test)[:, 1]
    
    # 6. 保存结果
    results = pd.DataFrame({
        'id': preprocessor.test_ids if preprocessor.test_ids is not None else range(len(predictions)),
        '是否高风险': predictions,
        '高风险概率': pred_proba
    })
    
    # 确保结果目录存在
    os.makedirs(os.path.dirname(config.RESULT_SAVE_PATH), exist_ok=True)
    
    results.to_csv(config.RESULT_SAVE_PATH, index=False, encoding='utf-8-sig')
    
    print(f"\n预测完成！结果已保存至: {config.RESULT_SAVE_PATH}")
    print(f"总样本数: {len(results)}")
    print(f"预测为高风险的样本数: {predictions.sum()}")
    print(f"高风险比例: {predictions.mean():.4f}")
    
    # 显示前几行结果
    print("\n预测结果示例（前10行）:")
    print(results.head(10))

if __name__ == "__main__":
    main()
