# 项目说明

本项目使用 CSE‑CIC‑IDS2018 处理后特征 CSV 完成降维与分类对比实验，核心结果与图表输出在 `results.md` 与 `figures/` 中，主报告在 `report.md`。

运行流程建议为：先准备数据到 `data/raw/`，再执行 `code/run_experiments.py` 生成指标与图表，最后将结果填入 `results.md` 并同步到 `report.md`。t‑SNE 在当前实现中需要对整体数据进行映射后再划分训练/测试，因此会产生轻微的信息泄漏风险；若需严格避免，可将 t‑SNE 仅用于可视化展示。

环境管理使用 `pixi.toml`，Python 固定为 3.11，其余依赖使用最新稳定版本。若需生成 docx，可使用 `pandoc report.md -o report.docx`。
