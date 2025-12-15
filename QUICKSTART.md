# ⚡ 快速启动指南 (Quick Start Guide)

本手册旨在帮助开发者与研究人员快速搭建实验环境，并复现项目中各类模型的训练与评估结果。

## 🛠️ 1. 环境构建

为确保代码的稳定运行与结果复现，建议遵循以下步骤配置开发环境。

### 1.1 创建隔离环境 (Virtual Environment)

推荐使用 Python 虚拟环境以隔离项目依赖，避免版本冲突。

**Windows (PowerShell):**
```powershell
# 创建虚拟环境
python -m venv .venv

# 激活环境
.\.venv\Scripts\activate
```

**macOS / Linux:**
```bash
# 创建虚拟环境
python3 -m venv .venv

# 激活环境
source .venv/bin/activate
```

### 1.2 安装核心依赖

项目依赖包含基础科学计算库与深度学习框架。

```bash
# 升级 pip
pip install -U pip

# 安装项目依赖
pip install -r requirements.txt
```

> **注意**: 若缺少 `requirements.txt`，可手动安装核心库：
> ```bash
> pip install numpy pandas scikit-learn matplotlib
> ```

### 1.3 配置 PyTorch

Transformer 与 MLP 模型基于 PyTorch 框架构建。为获得最佳训练性能，强烈建议安装 CUDA (GPU) 版本。

**CUDA (GPU) 版本 (推荐，需 NVIDIA 显卡):**
```bash
# 示例：安装兼容 CUDA 11.8 的 PyTorch 版本
# 请访问 https://pytorch.org/ 获取匹配您硬件的最新指令
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

**CPU 版本 (仅用于调试):**
```bash
pip install torch torchvision torchaudio
```

---

## 📂 2. 数据集准备

由于数据隐私合规要求，原始数据集 (`train.csv`, `test.csv`) 未包含在代码仓库中。请按如下规范部署数据：

1.  **获取数据**: 确保您拥有课程提供的原始 CSV 文件。
2.  **部署路径**: 将数据文件放置于项目 **根目录** (`data_modeling/`) 下。
    *   `data_modeling/train.csv`
    *   `data_modeling/test.csv`

> **提示**: 各模型的预处理脚本已配置为自动搜索根目录数据。无需手动复制到每个子文件夹。

---

## 🚀 3. 脚本快速启动

项目提供了统一的模型运行脚本，可在项目根目录一键执行任意模型，**无需手动切换目录**。

### 3.1 Windows 用户

使用 `run_model.bat` 脚本：

```powershell
# 运行单个模型
.\run_model.bat mlp          # 训练 MLP 模型
.\run_model.bat logistic     # 训练 Logistic Regression 模型
.\run_model.bat svm          # 训练 SVM 模型
.\run_model.bat transformer  # 训练 Transformer 模型

# 运行所有模型（自动对比评估）
.\run_model.bat all

# 查看帮助信息
.\run_model.bat help
```

### 3.2 macOS / Linux 用户

使用 `run_model.sh` 脚本：

```bash
# 添加执行权限（首次运行需要）
chmod +x run_model.sh

# 运行单个模型
./run_model.sh mlp          # 训练 MLP 模型
./run_model.sh logistic     # 训练 Logistic Regression 模型
./run_model.sh svm          # 训练 SVM 模型
./run_model.sh transformer  # 训练 Transformer 模型

# 运行所有模型（自动对比评估）
./run_model.sh all

# 查看帮助信息
./run_model.sh help
```

### 3.3 可用模型参数

| 参数 | 模型 | 说明 |
|------|------|------|
| `mlp` | Multi-Layer Perceptron | 多层感知机 |
| `logistic` | Logistic Regression | 逻辑回归分类器 |
| `svm` | Support Vector Machine | 支持向量机 |
| `transformer` | Transformer | 基于自注意力机制的模型 |
| `all` | 全部模型 | 按顺序运行所有模型并生成对比报告 |

> **💡 提示**: 使用 `all` 参数运行所有模型后，脚本会自动生成执行摘要，显示每个模型的运行状态（成功/失败）。各模型的 `outputs/` 目录包含训练结果、评估图表及 `feature_importance.csv/png` 特征重要性分析文件。

---

## 📊 4. 模型训练与评估（手动方式）

如果您更倾向于手动运行单个模型，所有模型均封装了 `execute.py` 作为统一执行入口。执行后将自动完成以下流程：
1.  数据加载与清洗
2.  模型构建与训练
3.  验证集评估与指标输出

### 4.1 深度学习模型 (SOTA)

**Transformer (Self-Attention)**
```bash
cd transformer_model
python execute.py
```

**Multilayer Perceptron (MLP)**
```bash
cd mlp_model
python execute.py
```

### 4.2 经典机器学习模型 (Baselines)

**Support Vector Machine (SVM)**
```bash
cd svm_model
python execute.py
```
> *注: SVM 在大数据集上训练较慢，代码内置了采样逻辑以加速实验。*

**Logistic Regression**
```bash
cd logistic_model
python execute.py
```

---

## ❓ 5. 常见问题 (FAQ)

**Q1: 提示 `FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'`**
> **A**: 请检查 `train.csv` 是否确实位于项目根目录下，且文件名完全匹配（区分大小写）。

**Q2: 提示 `CUDA out of memory`**
> **A**: 显存不足。请在 `train.py` 或 `execute.py` 中尝试调小 `batch_size` (例如从 128 降至 64 或 32)。

**Q3: 训练过程 Loss 不下降**
> **A**: 可能是学习率 (Learning Rate) 设置不当或数据未标准化。本项目已内置 `StandardScaler` 标准化处理，请检查是否错误修改了预处理逻辑。

**Q4: 如何查看训练结果？**
> **A**: 模型训练完成后，评估结果 (图片、CSV、模型权重) 通常会保存在各模型目录下的 `outputs/` 文件夹中。其中 `feature_importance.csv` 和 `feature_importance.png` 包含特征重要性分析结果。
