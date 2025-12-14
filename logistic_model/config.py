"""
配置文件
"""

# 数据路径
TRAIN_DATA_PATH = r"d:\projects\data_modeling\train.csv"
TEST_DATA_PATH = r"d:\projects\data_modeling\test.csv"

# 模型保存路径
MODEL_SAVE_DIR = r"d:\projects\data_modeling\logistic_model\models"
PREPROCESSOR_PATH = r"d:\projects\data_modeling\logistic_model\models\preprocessor.pkl"

# 结果保存路径
RESULT_SAVE_PATH = r"d:\projects\data_modeling\logistic_model\results\predictions.csv"

# 图片保存路径
FIGURE_SAVE_DIR = r"d:\projects\data_modeling\logistic_model\figures"
ROC_CURVE_PATH = r"d:\projects\data_modeling\logistic_model\figures\roc_curve.png"
CONFUSION_MATRIX_PATH = r"d:\projects\data_modeling\logistic_model\figures\confusion_matrix.png"

# 模型参数
RANDOM_STATE = 42
TEST_SIZE = 0.2

# 逻辑回归参数
LOGISTIC_PARAMS = {
    'penalty': 'l2',
    'C': 1.0,
    'solver': 'lbfgs',
    'max_iter': 1000,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'  # 处理类别不平衡
}

# 参数调优配置（可选）
ENABLE_GRID_SEARCH = False  # 是否启用网格搜索
GRID_SEARCH_PARAMS = {
    'penalty': ['l1', 'l2'],
    'C': [0.1, 0.5, 1.0, 1.5, 2.0],
    'solver': ['liblinear', 'saga']  # l1需要这些solver
}
GRID_SEARCH_CV = 5  # 交叉验证折数
