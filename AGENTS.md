# AGENTS.md — 项目唯一执行指南

本文件是本项目唯一执行指南；后续所有开发、写作与交付仅依赖本文件。
全程使用 Markdown 编写，最终用 pandoc 转换为 docx（文档版式后续人工调整）。

## 目标
构建“基于降维与分类的网络入侵检测系统”课程实践报告与配套材料，覆盖：
- 数据集获取与预处理
- 三类降维算法（PCA/LDA/t-SNE）对比
- 三类分类算法（SVM/随机森林/逻辑回归）对比
- 至少 9 组“降维×分类”组合评测
- 结果分析与安全性评估

## 交付物（均为 Markdown）
1. `report.md`：完整实践报告正文（含图表说明与附录链接）。
2. `rules.md`：入侵检测规则/要点提取文档（按题目要求）。
3. `results.md`：实验结果汇总表（可单独导出到正文/附录）。
4. `README.md`：运行说明、数据准备与复现步骤。
5. `code/`：代码与脚本（模块化、可复用）。
6. `data/`：预处理后数据集与必要的中间产物（说明来源与处理步骤）。

## 报告结构（report.md）
严格遵循以下大纲，后续可在 docx 中调整版式：
1. 设计背景与要求  
2. 数据集说明与预处理  
3. 设计任务实现过程  
4. 算法对比与结果分析  
5. 总结与改进方向  
附：参考文献、附录（含规则文档、补充图表与参数表）

## 实验约束与指标
- 数据集：CSE-CIC-IDS2018（允许使用单天数据简化计算）
- 特征：约 30 个核心特征；剔除易伪造特征（如源/目的 IP）
- 划分：训练 80% / 测试 20%
- 降维：PCA、LDA、t-SNE；维度 10/15/20
- 分类：SVM、随机森林、逻辑回归；需网格搜索调参
- 指标：准确率、误报率、漏报率、检测耗时
- 可视化：降维后二维散点图、混淆矩阵、对比表格
- 需单独统计高发攻击类型（如 DDoS）的检测精度

## 工作流程（从零到交付）
1. 数据集准备：记录来源、选取子集、清洗与归一化策略  
2. 特征工程：列出保留/剔除特征与理由  
3. 降维实验：对 10/15/20 维记录重构误差/类间距等指标  
4. 分类实验：三模型 × 三降维，共 9 组组合  
5. 结果汇总：统一表格格式 + 关键结论  
6. 安全性分析：重点讨论误报/漏报与实际场景影响  
7. 输出文档：完善 `report.md`、`results.md`、`rules.md`、`README.md`

## 文件与目录约定
- `report.md`：主报告  
- `results.md`：核心表格与图表说明  
- `rules.md`：规则提取与安全分析补充  
- `README.md`：复现步骤  
- `code/`：训练、评估、可视化脚本  
- `data/`：预处理数据（含数据说明文件）  
- `figures/`：输出图表（在 report 中引用）

## Pandoc 转换
示例（按需要调整）：
```
pandoc report.md -o report.docx
```

## 执行原则
### 文风与表达
报告文风要求：正式、严谨、术语准确，强调“方法—结果—结论”的逻辑链条。正文尽量使用连贯段落表达，避免出现“清单式/要点式”的 AI 常见列表表达；必要时用自然段过渡或小节标题组织内容。

示例段落（作为表达基准）：
“为提升检测效率与泛化能力，本文在标准化预处理后引入 PCA、LDA 与 t-SNE 三种降维方法，并结合 SVM、随机森林与逻辑回归完成入侵检测建模。实验结果显示，降维至 15 维在信息保留与检测耗时之间达到较优平衡，因此后续对比以该维度作为主要基线进行分析。”

### 图表与可视化
所有图片统一使用 SciencePlots（https://github.com/garrettj403/SciencePlots）的 IEEE 风格（`plt.style.use(["science","ieee"])`），图表文字使用英文，确保可读性与美观性。

### 环境与依赖
统一使用 `pixi.toml` 进行环境管理与依赖锁定。Python 版本固定为 3.11，其余依赖使用最新稳定版本。

### 交付要求
所有新增内容先落到 Markdown，再导出 docx；数据与代码可复现，表格与结论必须可追溯；指标、参数、数据来源与处理步骤必须记录清楚。

## 数据集可用性与清洗要求（已确认）
已确认所用 CSV 为“Processed Traffic Data for ML Algorithms”，包含 80 列且 `Label` 为标签列，可直接用于建模。预处理必须包含以下清洗规则：
1) 去除混入数据的重复表头行（检测 `Label` 列出现字符串 `Label` 的行并删除）。
2) 处理无穷值与缺失值（示例：`inf`/`-inf`/`NaN` 等），统一替换为 `NaN` 后再按规则处理（如行删除或统计填充，需在报告中说明）。
3) 记录清洗前后样本量变化，确保可追溯。

## 核心特征清单（固定 30 项）
下列特征均来自 CICFlowMeter 输出，优先保留与流量统计、时序、包长分布相关字段，避免源/目的 IP、Flow ID 等易伪造或与拓扑强相关字段：
Flow Duration; Tot Fwd Pkts; Tot Bwd Pkts; TotLen Fwd Pkts; TotLen Bwd Pkts; Fwd Pkt Len Max; Fwd Pkt Len Min; Fwd Pkt Len Mean; Fwd Pkt Len Std; Bwd Pkt Len Max; Bwd Pkt Len Min; Bwd Pkt Len Mean; Bwd Pkt Len Std; Flow Byts/s; Flow Pkts/s; Flow IAT Mean; Flow IAT Std; Flow IAT Max; Flow IAT Min; Fwd IAT Mean; Fwd IAT Std; Bwd IAT Mean; Bwd IAT Std; SYN Flag Cnt; ACK Flag Cnt; PSH Flag Cnt; FIN Flag Cnt; RST Flag Cnt; Active Mean; Idle Mean.

## 清洗与预处理口径（固定）
1) 删除混入的重复表头行（`Label` 列值为 `Label` 的行）。  
2) 将 `inf/-inf` 统一替换为 `NaN`，再对含 `NaN` 的样本行做删除处理（不做数值填充）。  
3) 删除 `Flow Byts/s` 或 `Flow Pkts/s` 为 0 且 `Flow Duration` 为 0 的异常行。  
4) 对核心特征采用 Min-Max 归一化，输出归一化参数以便复现。

## 划分与采样
按 80%/20% 划分训练/测试集，使用固定随机种子 `42`；若类别极度不平衡，训练集中使用分层划分并记录各类占比。

## 降维配置
PCA 使用 `svd_solver='auto'` 与 `whiten=False`；LDA 使用 `solver='svd'`；t-SNE 用于可视化与降维对比，固定 `perplexity=30`、`learning_rate='auto'`、`init='pca'`、`n_iter=1000`。

## 分类器与超参数网格（优化版）
根据原始题目要求（"通过网格搜索优化超参数"），在保证网格搜索有效性的前提下，选择代表性参数组合以平衡准确性与效率：

SVM（RBF）：`C ∈ {1, 10}`，`kernel = rbf`，`gamma = scale`。
随机森林：`n_estimators = 200`，`max_depth ∈ {None, 20}`，`min_samples_split = 2`。
逻辑回归：`C ∈ {0.1, 1, 10}`，`penalty='l2'`，`solver='lbfgs'`，`max_iter=1000`。
网格搜索采用 3 折交叉验证，评分使用 `f1_macro`。

**说明**：原始题目要求"通过网格搜索优化超参数"（如SVM的核函数选择），并未固定具体参数范围。本配置选择代表性参数组合，既满足网格搜索要求，又显著提升计算效率（SVM网格从12组降至2组，速度提升6倍）。

## 指标计算口径
准确率 = (TP + TN) / (TP + TN + FP + FN)。  
误报率（FPR）= FP / (FP + TN)。  
漏报率（FNR）= FN / (FN + TP)。  
检测耗时 = 单次预测平均耗时（测试集总预测时间 / 测试样本数），同时记录训练耗时用于对比。

## 结果汇总格式
`results.md` 中固定使用三类表：  
1) 降维方法 × 维度 的信息保留或结构指标汇总；  
2) 9 组“降维×分类”组合的准确率/误报率/漏报率/耗时表；  
3) 重点攻击类型（如 DDoS）在最优组合下的检测指标表。
