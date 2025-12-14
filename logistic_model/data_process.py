import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

"""
主要对数据预处理的方法进行定义
"""

class DataPreprocessor:
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_cols = None
        self.test_ids = None  # 保存测试集ID
        
    def load_data(self, filepath):
        """加载数据"""
        print(f"正在加载数据: {filepath}")
        data = pd.read_csv(filepath)
        print(f"数据维度: {data.shape}")
        return data
    
    def preprocess_train(self, data, target_col='是否高风险'):
        """预处理训练数据"""
        print("\n开始预处理训练数据...")
        
        # 1. 删除缺失值过多的样本（超过50%变量缺失）
        missing_ratio = data.isnull().sum(axis=1) / data.shape[1]
        data = data[missing_ratio < 0.5].copy()
        print(f"删除缺失值过多样本后: {data.shape}")
        
        # 2. 删除全为0或方差为0的列
        # 排除目标列和ID列
        feature_cols = [col for col in data.columns if col not in [target_col, 'id']]
        
        # 选择数值型特征
        numeric_features = data[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
        
        # 删除全为0的列
        zero_cols = [col for col in numeric_features if (data[col] == 0).all()]
        if zero_cols:
            data = data.drop(columns=zero_cols)
            print(f"删除全为0的列: {len(zero_cols)}个")
        
        # 删除方差为0的列
        var_zero_cols = [col for col in numeric_features 
                        if col in data.columns and data[col].var() == 0]
        if var_zero_cols:
            data = data.drop(columns=var_zero_cols)
            print(f"删除方差为0的列: {len(var_zero_cols)}个")
        
        # 3. 更新特征列表
        self.feature_cols = [col for col in data.columns 
                            if col not in [target_col, 'id'] and 
                            col in data.select_dtypes(include=[np.number]).columns]
        
        print(f"最终特征数量: {len(self.feature_cols)}")
        
        # 4. 填补缺失值（使用均值）
        data[self.feature_cols] = data[self.feature_cols].fillna(
            data[self.feature_cols].mean()
        )

        
        # 5. 标准化
        X = data[self.feature_cols].values
        y = data[target_col].values
        # print(y.shape)
        
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"特征维度: {X_scaled.shape}, 标签维度: {y.shape}")
        print(f"正样本比例: {y.mean():.4f}")
        
        return X_scaled, y
    
    def preprocess_test(self, data):
        """预处理测试数据"""
        print("\n开始预处理测试数据...")
        
        # 保存ID列
        if 'id' in data.columns:
            self.test_ids = data['id'].values
        
        # 确保测试集有相同的特征列
        missing_cols = set(self.feature_cols) - set(data.columns)
        for col in missing_cols:
            data[col] = 0
        
        # 选择相同的特征列
        data = data[self.feature_cols].copy()
        
        # 填补缺失值
        data = data.fillna(data.mean())
        
        # 标准化（使用训练集的参数）
        X_scaled = self.scaler.transform(data.values)
        
        print(f"测试集特征维度: {X_scaled.shape}")
        
        return X_scaled
