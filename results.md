# 实验结果汇总

本文件用于集中存放实验结果与图表说明，数据由 `code/run_experiments.py` 生成后填充。表格与数值需与代码输出一致，确保可复现与可追溯。

## 降维结果概览
表 1 记录 PCA/LDA/t‑SNE 在不同维度下的结构指标与信息保留情况，后续用于分析维度设置对检测效果与耗时的影响。

| Reducer | n_components | Information_retention | Class_separation | Notes |
| --- | --- | --- | --- | --- |
| PCA | 10 |  |  |  |
| PCA | 15 |  |  |  |
| PCA | 20 |  |  |  |
| LDA | 10 |  |  |  |
| LDA | 15 |  |  |  |
| LDA | 20 |  |  |  |
| t-SNE | 10 |  |  |  |
| t-SNE | 15 |  |  |  |
| t-SNE | 20 |  |  |  |

## 组合对比结果
表 2 为九组“降维×分类”组合的核心指标汇总，统一口径输出准确率、误报率、漏报率与预测耗时。

| Reducer | n_components | Classifier | Accuracy | FPR | FNR | Train_time_s | Predict_time_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| PCA | 10 | SVM |  |  |  |  |  |
| PCA | 10 | RandomForest |  |  |  |  |  |
| PCA | 10 | LogisticRegression |  |  |  |  |  |
| PCA | 15 | SVM |  |  |  |  |  |
| PCA | 15 | RandomForest |  |  |  |  |  |
| PCA | 15 | LogisticRegression |  |  |  |  |  |
| PCA | 20 | SVM |  |  |  |  |  |
| PCA | 20 | RandomForest |  |  |  |  |  |
| PCA | 20 | LogisticRegression |  |  |  |  |  |
| LDA | 10 | SVM |  |  |  |  |  |
| LDA | 10 | RandomForest |  |  |  |  |  |
| LDA | 10 | LogisticRegression |  |  |  |  |  |
| LDA | 15 | SVM |  |  |  |  |  |
| LDA | 15 | RandomForest |  |  |  |  |  |
| LDA | 15 | LogisticRegression |  |  |  |  |  |
| LDA | 20 | SVM |  |  |  |  |  |
| LDA | 20 | RandomForest |  |  |  |  |  |
| LDA | 20 | LogisticRegression |  |  |  |  |  |
| t-SNE | 10 | SVM |  |  |  |  |  |
| t-SNE | 10 | RandomForest |  |  |  |  |  |
| t-SNE | 10 | LogisticRegression |  |  |  |  |  |
| t-SNE | 15 | SVM |  |  |  |  |  |
| t-SNE | 15 | RandomForest |  |  |  |  |  |
| t-SNE | 15 | LogisticRegression |  |  |  |  |  |
| t-SNE | 20 | SVM |  |  |  |  |  |
| t-SNE | 20 | RandomForest |  |  |  |  |  |
| t-SNE | 20 | LogisticRegression |  |  |  |  |  |

## 重点攻击类型结果
表 3 用于记录高发攻击类型（例如 DDoS）在最优组合下的检测指标。

| Attack_type | Precision | Recall | F1 | Notes |
| --- | --- | --- | --- | --- |
| DDoS |  |  |  |  |
