import os
import sys
import json
import joblib
import time
import torch
import numpy as np
import pandas as pd
import gradio as gr
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime

# Add 'code' directory to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), 'code'))

# Import project utilities
from config import FEATURES, LABEL_COL
from torch_reducers import TorchLDA

# Constants
MODEL_DIR = Path("data/models")
DATA_PATH = Path("data/processed/ids2018_subset_3k.csv")

# Load Models
def load_assets():
    scaler = joblib.load(MODEL_DIR / "minmax_scaler.joblib")
    rf = joblib.load(MODEL_DIR / "random_forest.joblib")
    lda_data = torch.load(MODEL_DIR / "lda_10d.pt", map_location="cpu")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reducer = TorchLDA(n_components=lda_data["n_components"], device=device)
    reducer.components_ = lda_data["components"].to(device)
    reducer.mean_ = lda_data["mean"].to(device)
    
    with open(MODEL_DIR / "labels.json", "r") as f:
        labels = json.load(f)
        
    df = pd.read_csv(DATA_PATH)
    df = df[df[LABEL_COL] != LABEL_COL]
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=FEATURES, inplace=True)
    
    return scaler, reducer, rf, labels, df

scaler, reducer, rf, labels, full_df = load_assets()

# Global state
stats = {"total": 0, "attacks": 0, "latencies": [], "class_counts": {}, "correct": 0}
history = []

def predict_single(row_values):
    start_time = time.perf_counter()
    x = row_values.reshape(1, -1)
    x_scaled = scaler.transform(x)
    x_red = reducer.transform(x_scaled)
    pred = rf.predict(x_red)
    if hasattr(pred, "get"): pred = pred.get()
    latency = (time.perf_counter() - start_time) * 1000 # ms
    return str(pred[0]), latency

def get_distribution_plot(counts_dict):
    if not counts_dict: return go.Figure()
    # Sort by count
    sorted_counts = sorted(counts_dict.items(), key=lambda x: x[1], reverse=True)
    labels = [x[0] for x in sorted_counts]
    values = [x[1] for x in sorted_counts]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color='#00cfd5'),
        text=values,
        textposition='auto',
    ))
    fig.update_layout(
        title="Detected Traffic Distribution (By Class)",
        template="plotly_dark",
        margin=dict(l=10, r=10, t=40, b=10),
        height=580,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(autorange="reversed") # Highest on top
    )
    return fig

def format_metric(value, unit="", color="#00cfd5"):
    return f"""
    <div style="text-align: center; padding: 10px; border-radius: 8px; background: rgba(255,255,255,0.05);">
        <p style="margin: 0; font-size: 14px; color: #888;">{value[0]}</p>
        <p style="margin: 0; font-size: 32px; font-weight: bold; color: {color};">
            {value[1]}<span style="font-size: 16px; margin-left: 4px;">{unit}</span>
        </p>
    </div>
    """

def stream_inference(count=1000, delay=0.15):
    global stats, history
    stats = {"total": 0, "attacks": 0, "latencies": [], "class_counts": {}, "correct": 0}
    history = []
    
    for _ in range(count):
        sample = full_df.sample(1)
        true_label = sample[LABEL_COL].values[0]
        row_values = sample[FEATURES].values
        
        pred_label, latency = predict_single(row_values)
        
        timestamp = datetime.now().strftime("%H:%M:%S")
        is_attack = "Benign" not in pred_label
        
        # Update metrics
        stats["total"] += 1
        if is_attack: stats["attacks"] += 1
        stats["latencies"].append(latency)
        stats["class_counts"][pred_label] = stats["class_counts"].get(pred_label, 0) + 1
        
        is_match = (pred_label == true_label)
        if is_match: stats["correct"] += 1
        
        # Update log history
        history.insert(0, [
            timestamp, 
            true_label, 
            pred_label, 
            "🎯 准确" if is_match else "🔍 偏差",
            f"{latency:.2f}", 
            "🛑 ATTACK" if is_attack else "🟢 NORMAL"
        ])
        if len(history) > 15: history.pop()
        
        avg_lat = np.mean(stats["latencies"])
        accuracy = (stats["correct"] / stats["total"]) * 100
        
        # Dashboard HTML
        dashboard_html = f"""
        <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
            {format_metric(("CURRENT LATENCY", f"{latency:.2f}"), "ms", "#ffea00")}
            {format_metric(("AVG LATENCY", f"{avg_lat:.2f}"), "ms")}
            {format_metric(("ACCURACY", f"{accuracy:.1f}"), "%", "#00ff88")}
            {format_metric(("THROUGHPUT", f"{stats['total']}"), "Flows")}
            {format_metric(("THREATS DETECTED", f"{stats['attacks']}"), "Blocked", "#ff4b4b")}
        </div>
        """
        
        yield (
            dashboard_html,
            history,
            get_distribution_plot(stats["class_counts"])
        )
        time.sleep(delay)

with gr.Blocks(theme=gr.themes.Soft(primary_hue="cyan", neutral_hue="slate"), title="SME IDS Monitor") as demo:
    gr.Markdown("""
    # 🛡️ 智能入侵检测监控系统 (IDS-ML Control Center)
    **实时流量 analysis 与 自动防护引擎** | 基于 LDA-RF 实时推理流水线
    """)
    
    # Header Dashboard
    dashboard = gr.HTML(f"""
    <div style="display: grid; grid-template-columns: repeat(5, 1fr); gap: 15px;">
        {format_metric(("CURRENT LATENCY", "0.00"), "ms")}
        {format_metric(("AVG LATENCY", "0.00"), "ms")}
        {format_metric(("ACCURACY", "0.0"), "%")}
        {format_metric(("THROUGHPUT", "0"), "Flows")}
        {format_metric(("THREATS DETECTED", "0"), "Blocked")}
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### 📟 系统运行控制")
            with gr.Row():
                start_btn = gr.Button("🚀 启动实时监控", variant="primary")
                stop_btn = gr.Button("🛑 停止系统", variant="stop")
            
            gr.Markdown("### 🛰️ 实时流量监控日志 (Live Log)")
            log_table = gr.DataFrame(
                headers=["时间", "真实标签", "识别结果", "预测状态", "耗时(ms)", "检测状态"],
                datatype=["str", "str", "str", "str", "str", "str"],
                value=[],
                interactive=False
            )
            
        with gr.Column(scale=2):
            gr.Markdown("### 📊 威胁分布统计")
            dist_chart = gr.Plot()

    # Events
    stream_event = start_btn.click(
        stream_inference,
        outputs=[dashboard, log_table, dist_chart]
    )
    stop_btn.click(None, None, None, cancels=[stream_event])

if __name__ == "__main__":
    demo.launch()