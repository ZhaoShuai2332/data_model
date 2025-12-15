# P2P 逾期还款预测建模

本项目为解决Python 与大数据分析期末问题而生，旨在构建并评估多种机器学习与深度学习架构在高维表格数据上的风险预测能力。项目实现了从经典统计学习方法（逻辑回归、SVM）到现代深度学习架构（ResNet、Transformer）的完整技术栈，所有代码均遵循工业级开发规范与学术引用标准。

## 📚 项目概览

本项目针对高风险客户识别任务，通过对比实验探究了不同模型架构在表格数据上的性能表现。核心包含以下五大算法模块：

1.  **Transformer (Self-Attention)**: 采用多头自注意力机制捕捉特征间的全局依赖关系，参考 Vaswani et al. (2017) 与 Gorishniy et al. (2021) 架构。
2.  **ResNet-1D (CNN)**: 将残差网络 (Residual Networks) 适配于一维表格特征，通过跳跃连接解决深层网络退化问题。
3.  **Multilayer Perceptron (MLP)**: 经典的深度前馈神经网络，作为深度学习的基准模型。
4.  **Support Vector Machine (SVM)**: 基于核技巧 (Kernel Trick) 的分类器，适用于小样本高维空间。
5.  **Logistic Regression**: 广义线性模型，作为二分类任务的 Baseline。

## 🏗️ 项目架构

项目采用模块化设计，解耦数据流、模型定义与评估逻辑，便于扩展与复现。

```text
data_modeling/
├── transformer_model/  # Transformer 架构实现 (SOTA on Tabular)
├── cnn_model/          # ResNet-1D 卷积网络实现
├── mlp_model/          # 多层感知机实现
├── svm_model/          # 支持向量机实现
├── logistic_model/     # 逻辑回归基准实现
├── QUICKSTART.md       # 快速启动指南 (操作手册)
├── README.md           # 项目说明文档
└── requirements.txt    # 依赖环境配置文件
```

每个模型模块内部均遵循统一的工程结构：
- `execute.py`: **主执行入口**，负责串联数据加载、预处理、训练循环与评估输出。
- `[model_name].py`: **模型定义**，包含完整的网络结构与前向传播逻辑，附带学术文献引用。
- `data_process.py`: **数据工程**，实现缺失值处理、特征筛选、标准化等预处理流水线。
- `train.py`: **训练引擎**，封装了带有早停 (Early Stopping) 与日志记录的训练器。
- `evaluate.py`: **评估组件**，计算 AUC-ROC, F1-Score, Precision, Recall 等多维度指标。

## 🚀 快速开始

请查阅 [QUICKSTART.md](./QUICKSTART.md) 获取详细的环境配置、数据准备与运行指令。

## 📊 评估与复现

所有模型均内置了标准化的评估流程，支持：
- **自动日志记录**: 训练过程中的 Loss 与 Metric 实时输出。
- **多维度指标**: 自动计算并打印 Accuracy, F1-Score (Best Threshold), ROC-AUC, PR-AUC 等关键指标。
- **可复现性**: 所有实验均固定了随机种子 (Random Seed)，确保结果可复现。

## 📝 引用说明

代码中的模型定义部分已添加相关学术论文引用（如 NeurIPS, ICML 等顶级会议论文），以便查阅算法的原始理论依据。

## 📈 模型性能评估 (Model Performance)

基于各模型在验证集 (Validation Set) 和训练集 (Training Set) 上的表现，统计如下关键指标。**Transformer** 模型在验证集 AUC 与 F1-Score 上均取得了最优结果。

### 🏆 最佳模型 (Best Model): Transformer

综合各项指标，**Transformer** 被评选为本项目表现最佳的模型，理由如下：

1.  **综合指标最高**: 在最关键的 **ROC-AUC (0.8957)** 和 **F1-Score (0.7580)** 上均位列第一，表明其在处理不平衡分类任务上的综合优势。
2.  **泛化能力强**: 训练集 AUC (0.9002) 与验证集 AUC (0.8957) 差异极小 (<0.005)，相比于 SVM (差值 >0.03)，Transformer 展现了极佳的抗过拟合能力。
3.  **精确率与召回率平衡**: 相比于 MLP (高召回低精确) 和 SVM (高精确低召回)，Transformer 在 Precision (0.7212) 和 Recall (0.7987) 之间取得了最佳平衡，适合实际风控场景。

*   **Best ROC-AUC**: 0.8957
*   **Best F1-Score**: 0.7580

### 📊 详细对比表 (Performance Comparison)

| Model | Set | AUC | F1-Score (Best) | Accuracy | Precision | Recall | Specificity |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Transformer** | Validation | **0.8957** | **0.7580** | 0.8194 | 0.7212 | 0.7987 | 0.8200 |
| **Transformer** | Training | 0.9002 | 0.7418 | 0.8272 | 0.8248 | 0.6740 | 0.9166 |
| **MLP** | Validation | 0.8945 | 0.7558 | 0.8229 | 0.6828 | **0.8464** | 0.7708 |
| **MLP** | Training | 0.8992 | 0.7484 | 0.8275 | 0.8085 | 0.6967 | 0.9038 |
| **Logistic Reg** | Validation | 0.8932 | 0.7564 | 0.8142 | 0.7314 | 0.7831 | 0.8323 |
| **Logistic Reg** | Training | 0.8945 | 0.5671 | 0.4388 | 0.3960 | 0.9981 | 0.1128 |
| **CNN (ResNet1D)** | Validation | 0.8893 | 0.7514 | 0.8174 | 0.7149 | 0.7919 | 0.8159 |
| **CNN (ResNet1D)** | Training | 0.9048 | 0.7627 | 0.8306 | 0.7877 | 0.7392 | 0.8839 |
| **SVM** | Validation | 0.8829 | 0.7471 | 0.8117 | **0.7394** | 0.7549 | **0.8449** |
| **SVM** | Training | 0.9204 | 0.7997 | 0.8463 | 0.7690 | 0.8330 | 0.8541 |

> [!NOTE]
> *   **F1-Score (Best)**: 基于验证集上最优阈值 (Threshold) 计算得到的结果，而非默认 0.5 阈值。
> *   深度学习模型 (Transformer, MLP, CNN) 在训练集和验证集上的 AUC 非常接近，说明泛化能力良好。
> *   SVM 在训练集上表现出较高的 Metrics (AUC 0.9204)，但在验证集上下降至 0.8829，存在一定的过拟合现象。
> *   Logistic Regression 使用了极不平衡的类权重 (`{0: 0.02, 1: 0.98}`)，导致训练集上 Recall 极高 (0.9981) 但 Precision 和 Specificity 较低。


