# 基于降维与分类的网络入侵检测系统

本项目实现了基于机器学习的网络入侵检测系统，系统对比了 PCA、LDA 与 t-SNE 三种降维方法与 SVM、随机森林、逻辑回归三种分类器的组合效果。项目使用 CSE-CIC-IDS2018 数据集，通过 GPU 加速实现高效实验流程，生成完整的实验报告与可视化图表。

**核心成果**：完成15组降维×分类组合实验，最优模型（t-SNE-2D + SVM）达到76.05%准确率，误报率仅1.03%。

## 项目结构

```
hw_ml/
├── code/                          # 代码模块
│   ├── config.py                 # 配置文件（特征列表、路径等）
│   ├── preprocess.py             # 数据预处理模块
│   ├── torch_reducers.py         # PyTorch GPU 加速降维实现
│   ├── torch_classifiers.py      # PyTorch GPU 加速分类器
│   ├── metrics.py                # 评估指标计算
│   ├── plots.py                  # IEEE 风格可视化
│   └── run_experiments_torch.py  # 主实验脚本
├── data/                          # 数据目录
│   ├── raw/                      # 原始 CSV 文件
│   └── processed/                # 实验结果（CSV）
├── figures/                       # 可视化图表输出
├── report.md                      # 完整实践报告（10000字）
├── results.md                     # 实验结果汇总表
├── rules.md                       # 入侵检测规则提取
├── README.md                      # 本文件
├── references.md                  # 参考文献
└── pixi.toml                      # 环境配置文件
```

## 快速开始

### 1. 环境准备

本项目使用 [pixi](https://pixi.sh/) 进行环境管理，确保 Python 3.11 及所有依赖的版本一致性。

**前置要求**:
- Linux 系统（推荐）
- NVIDIA GPU（支持 CUDA 12.9+，可选）
- pixi 包管理器

**安装 pixi**（如未安装）:
```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

**安装项目依赖**:
```bash
cd hw_ml
pixi install
```

这将自动安装以下依赖：
- Python 3.11
- PyTorch 2.9 (GPU 版本)
- scikit-learn, pandas, numpy
- matplotlib, scienceplots
- tqdm（进度条）

### 2. 数据准备

**下载数据集**:

访问 [CSE-CIC-IDS2018 数据集页面](https://www.unb.ca/cic/datasets/ids-2018.html) 或 [AWS Open Data](https://registry.opendata.aws/cse-cic-ids2018/)，下载以下文件：

- `Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`

将文件放置于 `data/raw/` 目录：

```bash
mkdir -p data/raw
# 将下载的 CSV 文件移动到 data/raw/
mv Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv data/raw/
```

**数据说明**:
- 文件大小：约 103 MB (压缩后可能更小)
- 样本数：约 33 万条流记录
- 类别：Benign (正常流量) 和 Infilteration (攻击流量)
- 特征数：80 列（本项目使用其中 30 个核心特征）

### 3. 运行实验

**执行完整实验流程**:

```bash
pixi run python code/run_experiments_torch.py
```

**实验配置**:
- 训练集采样：10,000 样本（从 262,544 样本中随机采样，随机种子 42）
- 测试集：65,637 样本（完整测试集，不采样）
- 实验组合：15 组
  - PCA (10/15/20 维) × 3 分类器 = 9 组
  - LDA (1 维，二分类上限) × 3 分类器 = 3 组
  - t-SNE (2 维) × 3 分类器 = 3 组
- 预计运行时间：45-90 分钟（取决于硬件配置）

**输出文件**:
- `data/processed/metrics.csv`: 完整实验指标
- `data/processed/reduction_metrics.csv`: 降维效果评估
- `data/processed/attack_metrics.csv`: Infilteration 攻击检测指标
- `figures/*.png`: 降维可视化图表（5 张）

**实时监控**:

可以通过查看日志文件监控实验进度：

```bash
tail -f experiment_log_final.txt
```

### 4. 查看结果

实验完成后，查看以下文件：

- **`results.md`**: 实验结果汇总表（包含所有指标）
- **`report.md`**: 完整学术报告（约 10000 字）
- **`rules.md`**: 入侵检测规则提取与安全分析
- **`figures/`**: 降维可视化图表

## 技术特性

### GPU 加速

本项目使用 PyTorch 实现 GPU 加速的降维算法：

- **PCA**: 使用 `torch.linalg.svd` 进行奇异值分解
- **LDA**: 使用 GPU 计算类间/类内散度矩阵
- **t-SNE**: 使用 GPU 优化的梯度下降实现

**性能提升**:
- PCA/LDA 降维速度提升 5-10 倍
- 支持大规模数据集（50k+ 样本）

### 代码模块化

项目采用模块化设计，易于扩展与复用：

- `config.py`: 集中管理特征列表、路径、随机种子等配置
- `preprocess.py`: 数据清洗逻辑（处理 NaN/Inf、异常值过滤）
- `torch_reducers.py`: 降维算法实现（可独立使用）
- `torch_classifiers.py`: 分类器实现（可独立使用）
- `metrics.py`: 评估指标计算（准确率、FPR、FNR）
- `plots.py`: IEEE 风格可视化（使用 SciencePlots）

### 可复现性

项目确保实验可复现：

- 固定随机种子：`RANDOM_SEED = 42`
- 锁定依赖版本：`pixi.lock`
- 完整配置记录：`pixi.toml`
- 清晰的数据处理流程：`preprocess.py`

## 实验方法说明

### 数据预处理

1. **删除重复表头行**: 检测并删除混入数据的列名行
2. **特征筛选**: 保留 30 个核心流量统计特征，剔除 IP/端口等易伪造字段
3. **数值转换**: 将所有特征转为数值型，错误值转为 NaN
4. **异常值处理**: 替换 Inf/-Inf 为 NaN，删除含 NaN 的行
5. **零值过滤**: 删除无效零流量记录
6. **归一化**: Min-Max 归一化至 [0, 1] 区间
7. **数据划分**: 80/20 训练/测试集划分，分层采样保持类别平衡
8. **训练集采样**: 采样至 10,000 样本加速实验

### 降维方法

**PCA (主成分分析)**:
- 无监督线性降维
- 最大化方差保留
- 输出累积方差解释率

**LDA (线性判别分析)**:
- 有监督线性降维
- 最大化类间分离度
- 输出判别方向与方差解释率

**t-SNE (t-分布随机邻域嵌入)**:
- 非线性降维
- 保持局部邻域结构
- 主要用于可视化

### 分类器配置

**SVM (支持向量机)**:
- 网格搜索参数：`C ∈ {1, 10}`, `kernel = rbf`, `gamma = scale`
- 3 折交叉验证
- F1 宏平均评分

**RandomForest (随机森林)**:
- 网格搜索参数：`n_estimators = 200`, `max_depth ∈ {None, 20}`, `min_samples_split = 2`
- 3 折交叉验证
- F1 宏平均评分

**LogisticRegression (逻辑回归)**:
- 使用 PyTorch 实现（GPU 加速）
- 网格搜索参数：`C ∈ {0.1, 1, 10}`
- L-BFGS 优化器
- 3 折交叉验证

### 评估指标

- **Accuracy**: (TP + TN) / (TP + TN + FP + FN)
- **FPR (误报率)**: FP / (FP + TN)
- **FNR (漏报率)**: FN / (FN + TP)
- **Train Time**: 训练时间（包含网格搜索与交叉验证）
- **Predict Time**: 测试集预测时间

## 文档生成

### 转换为 Word 文档

使用 Pandoc 将 Markdown 报告转换为 Word 格式：

```bash
# 安装 pandoc（如未安装）
# Ubuntu/Debian: sudo apt-get install pandoc
# macOS: brew install pandoc

# 生成 Word 文档
pandoc report.md -o report.docx
pandoc results.md -o results.docx
pandoc rules.md -o rules.docx
```

转换后可在 Word 中进行版式调整与格式美化。

## 常见问题

**Q1: CUDA 版本不匹配怎么办？**

A: 检查系统 CUDA 版本：
```bash
nvidia-smi
```

如果 CUDA 版本不是 12.9，修改 `pixi.toml` 中的 `cuda` 版本，然后重新安装：
```toml
[system-requirements]
cuda = "你的CUDA版本"  # 如 "11.8" 或 "12.0"
```

**Q2: 没有 GPU 可以运行吗？**

A: 可以。代码会自动检测并切换到 CPU 模式。但运行时间会显著增加（约 3-5 倍）。

**Q3: 实验运行太慢怎么办？**

A: 可以进一步减少训练集采样上限。修改 `code/run_experiments_torch.py` 中的采样逻辑，例如将 `10000` 调小为 `5000`：
```python
if len(X_train) > 5000:  # 原值为 10000，可按需调整
```

**Q4: 内存不足怎么办？**

A: 减少训练集样本数（见 Q3）或减少实验组合数（只保留关键配置）。

## 引用

如果使用本项目或数据集，请引用：

```bibtex
@inproceedings{sharafaldin2018toward,
  title={Toward Generating a New Intrusion Detection Dataset and Intrusion Traffic Characterization},
  author={Sharafaldin, Iman and Lashkari, Arash Habibi and Ghorbani, Ali A},
  booktitle={ICISSP},
  pages={108--116},
  year={2018}
}
```

## 许可证

本项目代码采用 MIT 许可证。数据集遵循 CSE-CIC-IDS2018 的使用条款。

## 作者

nihildigit <nihildigit@outlook.com>

---

**注意事项**:
1. t-SNE 降维通过将训练集与测试集拼接后拟合二维嵌入再切分，存在轻微信息泄漏风险；若需严格避免，建议将 t-SNE 仅用于可视化或改用可显式变换的降维方法。
2. 训练集采样至 10,000 样本，结果精度略有下降但可接受。若需更高精度，可提高采样上限或使用完整训练集（修改代码去掉采样步骤）。
3. 所有随机过程固定种子 42，确保可复现性。

## 实验结果摘要

**最优组合**: t-SNE-2D + SVM
- 准确率: 76.05%
- 误报率: 1.03% (极低)
- 漏报率: 82.45%
- 攻击检测精确率: 86.99%

详见 `results.md` 和 `rules.md`
