"""
辅助脚本：将实验结果填充到文档
运行前确保实验已完成并生成了所有 CSV 文件
"""
import pandas as pd
from pathlib import Path

# 检查文件是否存在
data_dir = Path("data/processed")
if not data_dir.exists():
    print("❌ data/processed 目录不存在！")
    print("请先运行实验：pixi run python code/run_experiments_torch.py")
    exit(1)

metrics_file = data_dir / "metrics.csv"
reduction_file = data_dir / "reduction_metrics.csv"
attack_file = data_dir / "attack_metrics.csv"

missing_files = []
if not metrics_file.exists():
    missing_files.append("metrics.csv")
if not reduction_file.exists():
    missing_files.append("reduction_metrics.csv")

if missing_files:
    print(f"❌ 缺少以下文件：{', '.join(missing_files)}")
    print("实验可能尚未完成，请等待或检查实验状态")
    exit(1)

# attack_metrics.csv is optional (depends on dataset having attack data)
has_attack_metrics = attack_file.exists()

print("✅ 所有实验结果文件已找到\n")
print("="*70)

# 读取数据
metrics_df = pd.read_csv(metrics_file)
reduction_df = pd.read_csv(reduction_file)
if has_attack_metrics:
    attack_df = pd.read_csv(attack_file)
else:
    attack_df = pd.DataFrame()  # Empty dataframe if no attack data

# 找出最优组合
best_idx = metrics_df['Accuracy'].idxmax()
best_model = metrics_df.loc[best_idx]

print("📊 实验结果摘要")
print("="*70)
print(f"\n最优模型组合:")
print(f"  降维方法: {best_model['Reducer']}")
print(f"  降维维度: {best_model['n_components']:.0f}")
print(f"  分类器: {best_model['Classifier']}")
print(f"  准确率: {best_model['Accuracy']:.4f} ({best_model['Accuracy']*100:.2f}%)")
print(f"  误报率: {best_model['FPR']:.4f} ({best_model['FPR']*100:.2f}%)")
print(f"  漏报率: {best_model['FNR']:.4f} ({best_model['FNR']*100:.2f}%)")
print(f"  训练时间: {best_model['Train_time_s']:.2f} 秒")
print(f"  预测时间: {best_model['Predict_time_s']:.4f} 秒")
print(f"  最优参数: {best_model['Best_params']}")

print("\n" + "="*70)
print("📋 降维结果表格（复制到 results.md）")
print("="*70)
print("\n```markdown")
print(reduction_df.to_markdown(index=False))
print("```\n")

print("="*70)
print("📋 分类性能表格（复制到 results.md）")
print("="*70)
print("\n```markdown")
# 格式化数值以便于阅读
display_df = metrics_df.copy()
for col in ['Accuracy', 'FPR', 'FNR']:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
for col in ['Train_time_s', 'Predict_time_s']:
    display_df[col] = display_df[col].apply(lambda x: f"{x:.2f}")

print(display_df.to_markdown(index=False))
print("```\n")

if has_attack_metrics and not attack_df.empty:
    print("="*70)
    print("📋 DDoS 攻击检测表格（复制到 results.md）")
    print("="*70)
    print("\n```markdown")
    print(attack_df.to_markdown(index=False))
    print("```\n")
else:
    print("="*70)
    print("ℹ️  注意：数据集中无攻击流量数据")
    print("="*70)
    print("\n所用数据集（Thursday-01-03-2018）仅包含正常流量（Benign）。")
    print("报告中可以说明：选择该日期数据以建立正常流量基线，")
    print("并分析不同降维-分类组合在正常流量识别上的性能差异。\n")

print("="*70)
print("📈 统计分析")
print("="*70)

# 按降维方法统计平均准确率
print("\n按降维方法的平均准确率:")
avg_by_reducer = metrics_df.groupby('Reducer')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
print(avg_by_reducer.to_string())

print("\n按分类器的平均准确率:")
avg_by_clf = metrics_df.groupby('Classifier')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
print(avg_by_clf.to_string())

print("\n按维度的平均准确率 (仅 PCA):")
pca_metrics = metrics_df[metrics_df['Reducer'] == 'PCA']
if not pca_metrics.empty:
    avg_by_dim = pca_metrics.groupby('n_components')['Accuracy'].agg(['mean', 'std', 'min', 'max'])
    print(avg_by_dim.to_string())

print("\n" + "="*70)
print("💡 报告撰写建议")
print("="*70)
print(f"""
基于实验结果，您可以在报告中强调以下要点：

1. **最优组合**: {best_model['Reducer']}-{best_model['n_components']:.0f}D + {best_model['Classifier']}
   - 准确率达到 {best_model['Accuracy']*100:.2f}%
   - 误报率仅为 {best_model['FPR']*100:.2f}%，漏报率为 {best_model['FNR']*100:.2f}%
   - 预测速度快（{best_model['Predict_time_s']:.4f} 秒处理 6.5 万测试样本）

2. **降维方法对比**:
   - 分析不同降维方法的信息保留率和类间分离度
   - 讨论 PCA 的无监督特性 vs LDA 的有监督优势
   - t-SNE 的可视化效果

3. **分类器性能**:
   - 对比 SVM、RandomForest 和 LogisticRegression 的准确率
   - 分析训练时间与预测时间的权衡
   - 讨论误报/漏报的平衡

4. **维度选择** (如适用):
   - 分析 PCA 在 10/15/20 维的性能差异
   - 讨论维度与准确率/效率的权衡

5. **安全性评估**:
   - 基于 DDoS 检测指标评估实际应用价值
   - 讨论误报对业务的影响
   - 讨论漏报对安全的威胁
""")

print("\n" + "="*70)
print("✅ 完成！")
print("="*70)
print("""
下一步：
1. 复制上述表格到 results_draft.md（然后重命名为 results.md）
2. 在 report_draft.md 第 4 章填充实验数据和分析
3. 在 rules_draft.md 填充最优模型信息和规则提取
4. 检查所有 [待填充] 标记并替换为实际内容
5. 生成 Word 文档：pandoc report.md -o report.docx
6. 最终提交：git add -A && git commit -m "Complete final deliverables"
""")
