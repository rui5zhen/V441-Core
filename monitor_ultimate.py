import pandas as pd
import os
import time
from datetime import datetime

# --- 核心配置：锁定最新实验路径 ---
CSV_PATH = "/home/zhenr2024/yolov10/V3_Lite_Final_SOTA/Mamba_P2Ghost_Ultimate11/results.csv"

# --- 历史纪录红线 ---
V1_DISTILL = 0.4096
V2_SOLO = 0.4177
BASELINE = 0.4270
PURE_MAMBA = 0.3800

def get_latest_metrics():
    if not os.path.exists(CSV_PATH):
        return None
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty: return None
        # 清洗列名（去掉空格）
        df.columns = [c.strip() for c in df.columns]
        last = df.iloc[-1]
        best_map = df['metrics/mAP50(B)'].max()
        
        return {
            "epoch": int(last['epoch']),
            "mAP50": last['metrics/mAP50(B)'],
            "mAP95": last['metrics/mAP50-95(B)'],
            "p": last['metrics/precision(B)'],
            "r": last['metrics/recall(B)'],
            "loss_box": last['val/box_loss'],
            "loss_cls": last['val/cls_loss'],
            "best_map": best_map
        }
    except:
        return None

def print_dashboard(m):
    os.system('clear')
    now = datetime.now().strftime("%H:%M:%S")
    
    print(f"🔥 V3-Lite-Ghost [Ultimate Edition] 决胜时刻 | 🕒 {now}")
    print("=" * 72)
    
    print(f"📊 核心架构 (Architecture Status):")
    print(f"   🔹 引擎:      Octo-Scan Mamba + Ghost-P2 Dual-Path")
    print(f"   🔹 策略:      CA-TAL (类别感知损失) + LLRD 阶梯解冻")
    print(f"   🔹 检测头:    P2-Ghost, P3, P4, P5 (四头全火力覆盖)")

    # 计算差距
    gap_v1 = m['mAP50'] - V1_DISTILL
    gap_v2 = m['mAP50'] - V2_SOLO
    gap_base = m['mAP50'] - BASELINE
    
    print(f"\n📈 实时指标 (Real-time Metrics) [Epoch {m['epoch']} / 200]:")
    print(f"   📍 mAP50:      {m['mAP50']:.4f}  (历史最高: {m['best_map']:.4f})")
    print(f"   📍 mAP50-95:  {m['mAP95']:.4f}")
    print(f"   📍 Precision: {m['p']:.4f}  (Bias Reset 压制效果)")
    print(f"   📍 Recall:    {m['r']:.4f}  (👈 P2-Ghost 增益核心观察项)")

    print(f"\n⚔️ 战场对峙 (Battlefield Status):")
    
    def format_gap(gap, name):
        status = "✅ 已反超" if gap > 0 else "📉 距离"
        icon = "🔥" if gap > 0 else "🛡️"
        return f"   {icon} {status} {name}: " + (f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}")

    print(format_gap(gap_v1, "V1 蒸馏纪录 (0.4096)"))
    print(format_gap(gap_v2, "V2 Solo 纪录 (0.4177)"))
    print(format_gap(gap_base, "Baseline 红线 (0.4270)"))
    
    print(f"\n📉 损耗分析 (Loss Analysis):")
    print(f"   📦 Box Loss:  {m['loss_box']:.4f} (坐标回归精度)")
    print(f"   🏷️ Cls Loss:  {m['loss_cls']:.4f} (CA-TAL 惩罚强度)")

    print("-" * 72)
    
    # 动态诊断逻辑
    if m['epoch'] < 30:
        diag = "❄️ 骨干网冻结中。P2-Ghost 头正在接受 CA-TAL 梯度的“魔鬼训练”，重点攻克自行车类。"
    elif 30 <= m['epoch'] < 50:
        diag = "🔓 触发阶梯解冻。Octo-Scan 模块正在与 P2 特征进行对齐，预期会出现短暂指标波动。"
    else:
        diag = "🎯 进入高精度收割期。学习率已开始余弦退火，正在微米级修正小目标置信度。"
        
    print(f"💡 实时诊断：{diag}")
    print("=" * 72)

if __name__ == "__main__":
    print("🦊 狐狸专家正在刺探战况...")
    while True:
        metrics = get_latest_metrics()
        if metrics:
            print_dashboard(metrics)
        else:
            print(f"⏳ 等待 results.csv 生成... (当前路径: {CSV_PATH})")
        time.sleep(30)