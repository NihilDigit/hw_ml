# 实验结果汇总

## 1. 实验配置

本项目基于 CSE-CIC-IDS2018 的“Processed Traffic Data for ML Algorithms”多日数据构建多类别子集，覆盖 14 种攻击类型与正常流量（共 15 类）。为保证稀有攻击类型具备统计意义，子集构建采用“最小样本数 + 比例补齐”的策略：每类至少 100 条，其余样本按原始分布比例分配，总样本量固定为 10,000。子集清洗与特征口径与 `code/preprocess.py` 一致（去除重复表头、Inf/NaN、零时长零速率异常行，保留 30 个核心特征并做 Min-Max 归一化）。训练/测试按 80%/20% 分层划分，随机种子 42。

降维对比包含 PCA/LDA/t-SNE 三类方法，其中 PCA 取 10/15/20 维，LDA 目标设为 10/15/20 维但受类别数上限约束（最大为 14 维），t-SNE 仅用于二维可视化与对比性输入（2 维），并在报告中说明其高维扩展受限且不具备稳定的 `transform` 能力。分类器包含 SVM、随机森林、逻辑回归，网格搜索 3 折交叉验证，评分为 `f1_macro`。

## 2. 降维效果对比

说明：LDA 的最大维度为类别数减一（本实验为 15 类，因此上限 14 维），表中 LDA-15/LDA-20 实际对应 14 维输出；t-SNE 仅用于二维可视化与对比性输入，不具备稳定的高维 `transform` 能力，因此仅报告 2 维结果。

| Reducer   |   n_components |   Information_retention |   Class_separation |
|:----------|---------------:|------------------------:|-------------------:|
| LDA       |             10 |                1        |           0.212653 |
| LDA       |             15 |                1        |           0.159143 |
| LDA       |             20 |                1        |           0.159143 |
| PCA       |             10 |                0.971953 |           0.134021 |
| PCA       |             15 |                0.990368 |           0.135200 |
| PCA       |             20 |                0.998154 |           0.134978 |
| t-SNE     |              2 |                0        |           0.157584 |

## 3. “降维×分类”组合性能对比（21 组）

| Reducer   |   n_components | Classifier         |   Accuracy |       FPR |      FNR |   Train_time_s |   Predict_time_s | Best_params                                                      |
|:----------|---------------:|:-------------------|-----------:|----------:|---------:|---------------:|-----------------:|:-----------------------------------------------------------------|
| LDA       |             10 | LogisticRegression |     0.7155 | 0         | 1        |       0.495588 |      0.000277996 | {'C': 0.1, 'max_iter': 1000}                                     |
| LDA       |             10 | RandomForest       |     0.909  | 0.0279525 | 0.11775  |       4.45249  |      0.0400431   | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200} |
| LDA       |             10 | SVM                |     0.854  | 0.0153739 | 0.407733 |       0.902379 |      0.142998    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| LDA       |             15 | LogisticRegression |     0.7155 | 0         | 1        |       0.502584 |      0.000257254 | {'C': 0.1, 'max_iter': 1000}                                     |
| LDA       |             15 | RandomForest       |     0.91   | 0.0244584 | 0.119508 |       4.43135  |      0.0377817   | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}   |
| LDA       |             15 | SVM                |     0.8515 | 0.0153739 | 0.407733 |       0.874676 |      0.149211    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| LDA       |             20 | LogisticRegression |     0.7155 | 0         | 1        |       0.387769 |      0.00024724  | {'C': 0.1, 'max_iter': 1000}                                     |
| LDA       |             20 | RandomForest       |     0.91   | 0.0244584 | 0.119508 |       4.24968  |      0.0371504   | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}   |
| LDA       |             20 | SVM                |     0.8515 | 0.0153739 | 0.407733 |       0.882463 |      0.150612    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| PCA       |             10 | LogisticRegression |     0.7155 | 0         | 1        |       1.31526  |      0.000248671 | {'C': 0.1, 'max_iter': 1000}                                     |
| PCA       |             10 | RandomForest       |     0.903  | 0.0300489 | 0.128295 |       5.45684  |      0.0391567   | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200} |
| PCA       |             10 | SVM                |     0.8485 | 0.0167715 | 0.425308 |       2.65372  |      0.161323    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| PCA       |             15 | LogisticRegression |     0.7155 | 0         | 1        |       0.481299 |      0.000251532 | {'C': 0.1, 'max_iter': 1000}                                     |
| PCA       |             15 | RandomForest       |     0.9065 | 0.0265549 | 0.123023 |       4.38105  |      0.0407934   | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}   |
| PCA       |             15 | SVM                |     0.85   | 0.0167715 | 0.413005 |       2.40219  |      0.165129    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| PCA       |             20 | LogisticRegression |     0.7155 | 0         | 1        |       0.472773 |      0.000299692 | {'C': 0.1, 'max_iter': 1000}                                     |
| PCA       |             20 | RandomForest       |     0.91   | 0.0237596 | 0.121265 |       5.449    |      0.0396709   | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200}   |
| PCA       |             20 | SVM                |     0.85   | 0.0160727 | 0.41652  |       0.984209 |      0.145152    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |
| t-SNE     |              2 | LogisticRegression |     0.3585 | 0.57652   | 0.274165 |       6.38664  |      0.000262499 | {'C': 0.1, 'max_iter': 1000}                                     |
| t-SNE     |              2 | RandomForest       |     0.9005 | 0.037037  | 0.11775  |       1.68239  |      0.04109     | {'max_depth': None, 'min_samples_split': 2, 'n_estimators': 200} |
| t-SNE     |              2 | SVM                |     0.8585 | 0.0230608 | 0.27065  |       1.32787  |      0.176769    | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                     |

## 4. 重点攻击类型检测指标（全攻击类型）

详见 `data/processed/attack_metrics.csv`。本表将按“攻击类型 × Precision/Recall/F1”输出所有攻击类的指标，以满足“高发攻击类型单独统计”的要求。

## 5. 关键结论

在 10k 多类别子集上，随机森林在 PCA/LDA/t-SNE 表示下整体准确率稳定，且预测耗时显著低于 SVM；PCA 在 10 维即可保留超过 97% 的方差信息，LDA 的类间分离度更高，说明监督降维对类别可分性具有优势。t-SNE 的二维嵌入在可视化上具备解释力，但由于其不具备稳定的高维变换与部署可行性，更多用于对比与直观展示。

## 6. 图表索引（用于报告引用）

为便于在 `report.md` 中插入图表并导出 docx，本文将关键可视化统一输出到 `figures/` 目录，文件名固定如下（图中文字均为英文，风格为 SciencePlots IEEE）：

- `figures/PCA_InformationRetention.png`：PCA 信息保留率随维度变化
- `figures/Accuracy_by_Combination.png`：组合准确率对比
- `figures/Tradeoff_FPR_vs_FNR.png`：FPR-FNR 权衡散点图
- `figures/PredictLatency_ms_per_sample.png`：不同组合的预测时延对比
- `figures/ClassDistribution_AfterCleaning.png`：清洗后类别分布图
- `figures/Pipeline_Overview.png`：系统流水线概览图
- `figures/ConfusionMatrix_PCA10_RandomForest.png`：PCA-10D + 随机森林混淆矩阵
- `figures/ConfusionMatrix_PCA10_SVM.png`：PCA-10D + SVM 混淆矩阵
- `figures/ConfusionMatrix_LDA10_RandomForest.png`：LDA-10D + 随机森林混淆矩阵
