# 实验结果汇总

## 1. 实验配置

本项目基于 CSE-CIC-IDS2018 的单日数据（`Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv`）开展对比实验。数据经清洗后共有 328,181 条流记录，其中正常流量（Benign）235,778 条、攻击流量（Infilteration）92,403 条。按 80%/20% 分层划分得到训练集 262,544 条、测试集 65,637 条；为平衡计算开销与可复现性，训练集进一步随机采样为 10,000 条（随机种子 42），测试集保持全量评测。实验包含 5 组降维配置（PCA 10/15/20 维、LDA 1 维、t-SNE 2 维）与 3 个分类器（SVM、随机森林、逻辑回归），共计 15 组“降维×分类”组合，其中 LDA 的维度按当前二分类设置输出。

## 2. 降维效果对比

| Reducer | n_components | Information_retention | Class_separation |
|:--------|-------------:|----------------------:|-----------------:|
| PCA     |           10 |              0.972798 |       0.00652423 |
| PCA     |           15 |              0.993783 |       0.00638713 |
| PCA     |           20 |              0.998563 |       0.00635879 |
| LDA     |            1 |              1.000000 |       0.01892130 |
| t-SNE   |            2 |              0.000000 |       0.01488610 |

## 3. “降维×分类”组合性能对比（15 组）

| Reducer |   n_components | Classifier         |   Accuracy |    FPR |    FNR |   Train_time_s |   Predict_time_s | Best_params                                                    |
|:--------|---------------:|:-------------------|-----------:|-------:|-------:|---------------:|-----------------:|:---------------------------------------------------------------|
| PCA     |             10 | SVM                |     0.7439 | 0.0504 | 0.7809 |           5.12 |             8.76 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                   |
| PCA     |             10 | RandomForest       |     0.7539 | 0.0559 | 0.7312 |           8.14 |             0.80 | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |
| PCA     |             10 | LogisticRegression |     0.7184 | 0.0000 | 1.0000 |           1.01 |             0.00 | {'C': 0.1, 'max_iter': 1000}                                   |
| PCA     |             15 | SVM                |     0.7459 | 0.0503 | 0.7741 |           5.22 |             8.94 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                   |
| PCA     |             15 | RandomForest       |     0.7491 | 0.0667 | 0.7211 |           6.59 |             0.79 | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |
| PCA     |             15 | LogisticRegression |     0.7184 | 0.0000 | 1.0000 |           0.55 |             0.00 | {'C': 0.1, 'max_iter': 1000}                                   |
| PCA     |             20 | SVM                |     0.7447 | 0.0564 | 0.7629 |           3.42 |             8.69 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                   |
| PCA     |             20 | RandomForest       |     0.7482 | 0.0687 | 0.7190 |           8.40 |             0.84 | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |
| PCA     |             20 | LogisticRegression |     0.7184 | 0.0000 | 1.0000 |           0.53 |             0.00 | {'C': 0.1, 'max_iter': 1000}                                   |
| LDA     |              1 | SVM                |     0.7269 | 0.0950 | 0.7276 |           4.21 |             8.60 | {'C': 1, 'gamma': 'scale', 'kernel': 'rbf'}                    |
| LDA     |              1 | RandomForest       |     0.7391 | 0.0729 | 0.7406 |           2.70 |             0.76 | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |
| LDA     |              1 | LogisticRegression |     0.7184 | 0.0000 | 1.0000 |           0.41 |             0.00 | {'C': 0.1, 'max_iter': 1000}                                   |
| t-SNE   |              2 | SVM                |     0.7605 | 0.0103 | 0.8245 |           6.81 |             8.19 | {'C': 10, 'gamma': 'scale', 'kernel': 'rbf'}                   |
| t-SNE   |              2 | RandomForest       |     0.7491 | 0.0645 | 0.7267 |           2.64 |             0.83 | {'max_depth': 20, 'min_samples_split': 2, 'n_estimators': 200} |
| t-SNE   |              2 | LogisticRegression |     0.6413 | 0.2682 | 0.5897 |           3.79 |             0.00 | {'C': 1, 'max_iter': 1000}                                     |

## 4. 重点攻击类型（Infilteration）检测指标（最优组合）

| Attack_type   | Precision | Recall   | F1       |
|:--------------|----------:|---------:|---------:|
| Infilteration |  0.869903 | 0.175478 | 0.292044 |

## 5. 关键结论

综合准确率排序，最优组合为 t-SNE（2 维）+ SVM（RBF），测试集准确率 76.05%，误报率仅 1.03%，但漏报率较高（82.45%），表现出“低误报—高漏报”的显著权衡。就稳健性而言，随机森林在 PCA/LDA/t-SNE 各降维设置下的准确率波动较小（约 73.9%～75.4%），且预测耗时显著低于 SVM；PCA 在 10 维即可保留约 97% 的方差信息，具有较好的工程可用性。完整实验运行日志与中间输出见 `data/processed/` 与 `experiment_log_final.txt` 等文件。

## 6. 图表索引（用于报告引用）

为便于在 `report.md` 中插入图表并导出 docx，本文将关键可视化统一输出到 `figures/` 目录，文件名固定如下（图中文字均为英文，风格为 SciencePlots IEEE）：

- `figures/PCA_InformationRetention.png`：PCA 信息保留率随维度变化
- `figures/Accuracy_by_Combination.png`：15 组组合准确率对比
- `figures/Tradeoff_FPR_vs_FNR.png`：FPR-FNR 权衡散点图
- `figures/PredictLatency_ms_per_sample.png`：不同组合的预测时延对比
- `figures/ClassDistribution_AfterCleaning.png`：清洗后类别分布图
- `figures/Pipeline_Overview.png`：系统流水线概览图
- `figures/ConfusionMatrix_PCA10_RandomForest.png`：PCA-10D + 随机森林混淆矩阵
- `figures/ConfusionMatrix_PCA10_SVM.png`：PCA-10D + SVM 混淆矩阵
- `figures/ConfusionMatrix_LDA1_RandomForest.png`：LDA-1D + 随机森林混淆矩阵

详细结果见 `fill_results_final.txt`
