"""
SVM模型配置文件
"""

# 数据路径
TRAIN_DATA_PATH = r"d:\projects\data_modeling\train.csv"
TEST_DATA_PATH = r"d:\projects\data_modeling\test.csv"

# 模型保存路径
MODEL_SAVE_DIR = r"d:\projects\data_modeling\svm_model\models"
PREPROCESSOR_PATH = r"d:\projects\data_modeling\svm_model\models\preprocessor.pkl"

# 结果保存路径
RESULT_SAVE_PATH = r"d:\projects\data_modeling\svm_model\results\predictions.csv"

# 图片保存路径
FIGURE_SAVE_DIR = r"d:\projects\data_modeling\svm_model\figures"
ROC_CURVE_PATH = r"d:\projects\data_modeling\svm_model\figures\roc_curve.png"
CONFUSION_MATRIX_PATH = r"d:\projects\data_modeling\svm_model\figures\confusion_matrix.png"

# 模型参数
RANDOM_STATE = 42
TEST_SIZE = 0.2

# SVM参数
SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'rbf',  # 'linear', 'poly', 'rbf', 'sigmoid'
    'gamma': 'scale',
    'probability': True,  # 必须为True才能使用predict_proba
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'  # 处理类别不平衡
}

# 参数调优配置（可选）
ENABLE_GRID_SEARCH = False  # 是否启用网格搜索
GRID_SEARCH_PARAMS = {
    'C': [0.1, 1.0, 10.0],
    'kernel': ['rbf', 'linear'],
    'gamma': ['scale', 'auto', 0.1, 1.0]
}
GRID_SEARCH_CV = 5  # 交叉验证折数
