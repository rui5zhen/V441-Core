import pandas as pd
import os
import time
from datetime import datetime

# --- 核心配置：锁定最新【终极版】实验路径 ---
# ⚠️ 注意：这里换成了我们新设定的 Ultimate 目录名
CSV_PATH = "/home/zhenr2024/yolov10/V3_Lite_Final_SOTA/Mamba_P2Ghost_Ultimate5/results.csv"

# --- 历史纪录与心魔红线 ---
BASELINE = 0.4270      # 原始 Baseline
V3_GHOST_OLD = 0.4415  # 我们上一版纯 CIoU 的极限
SOFT_NMS_MAX = 0.4693  # 刚才物理释放的极限，我们的心魔！

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
        best_map95 = df['metrics/mAP50-95(B)'].max()
        
        return {
            "epoch": int(last['epoch']),
            "mAP50": last['metrics/mAP50(B)'],
            "mAP95": last['metrics/mAP50-95(B)'],
            "p": last['metrics/precision(B)'],
            "r": last['metrics/recall(B)'],
            "loss_box": last['train/box_loss'], # 观察 NWD 的下降平滑度
            "loss_cls": last['train/cls_loss'],
            "val_box": last['val/box_loss'],
            "best_map": best_map,
            "best_map95": best_map95
        }
    except Exception as e:
        return None

def print_dashboard(m):
    os.system('clear')
    now = datetime.now().strftime("%H:%M:%S")
    
    print(f"🔥 V3-Lite-Ghost [NWD+TAL 终极收官战] 实时雷达 | 🕒 {now}")
    print("=" * 75)
    
    print(f"📊 战车武装状态 (Architecture & Weapons):")
    print(f"   🔹 引擎:      Octo-Scan Mamba (全局感知) + Ghost-P2 (高频显微镜)")
    print(f"   🔹 损失:      CA-TAL (特权痛感) + NWD (微小目标高斯包容)")
    print(f"   🔹 分配:      Scale-Aware 动态 Top-K (消灭语义混淆)")

    # 计算差距
    gap_base = m['mAP50'] - BASELINE
    gap_old  = m['mAP50'] - V3_GHOST_OLD
    gap_max  = m['mAP50'] - SOFT_NMS_MAX
    
    print(f"\n📈 核心指标矩阵 (Real-time Metrics) [Epoch {m['epoch']} / 200]:")
    print(f"   📍 mAP50:     {m['mAP50']:.4f}  (历史最高: {m['best_map']:.4f})")
    print(f"   📍 mAP50-95:  {m['mAP95']:.4f}  (历史最高: {m['best_map95']:.4f} 👈 NWD 专属观察点)")
    print(f"   📍 Precision: {m['p']:.4f}  (动态 TAL 净化效果)")
    print(f"   📍 Recall:    {m['r']:.4f}  (P2-Ghost 增益)")

    print(f"\n⚔️ 诸神黄昏对峙 (Battlefield Status):")
    
    def format_gap(gap, name, icon_win="👑", icon_lose="🚧"):
        status = "已碾压" if gap > 0 else "距超越"
        icon = icon_win if gap > 0 else icon_lose
        return f"   {icon} {status} {name:20s}: " + (f"+{gap:.4f}" if gap > 0 else f"{gap:.4f}")

    print(format_gap(gap_base, "Baseline 官方红线 (0.4270)"))
    print(format_gap(gap_old,  "上一代 V3 纪录     (0.4415)"))
    print(format_gap(gap_max,  "后处理极限心魔     (0.4693)", icon_win="🚀", icon_lose="🔥"))
    
    print(f"\n📉 损耗与梯度 (Loss Dynamics):")
    print(f"   📦 Box Loss (Train): {m['loss_box']:.4f}  | (Val): {m['val_box']:.4f}  (👈 观察 NWD 的平滑度)")
    print(f"   🏷️ Cls Loss (Train): {m['loss_cls']:.4f}  (CA-TAL 惩罚强度)")

    print("-" * 75)
    
    # 动态诊断逻辑
    if m['epoch'] < 5:
        diag = "🌪️ 阵痛期：NWD 正在重塑边界框的高斯分布认知，Box Loss 偏高属正常现象，请稳住！"
    elif 5 <= m['epoch'] < 30:
        diag = "❄️ 骨干网冻结中。动态 TAL 正在清洗脏标签，Precision 应该会比以前爬升得更快。"
    elif 30 <= m['epoch'] < 100:
        diag = "🔓 阶梯解冻触发。Mamba 正在与 P2 头连通，请盯紧 mAP50-95，它即将起飞！"
    elif 100 <= m['epoch'] < 180:
        diag = "🎯 深度微调期。梯度进入平滑下降，此时每一次 +0.001 的涨幅都是实打实的物理边界收敛。"
    else:
        diag = "🚀 终极封神！Mosaic 增强已关闭，网络正在直面真实世界，准备见证大爆炸！"
        
    print(f"💡 狐狸雷达诊断：{diag}")
    print("=" * 75)

if __name__ == "__main__":
    print("🦊 狐狸军师正在接入战地通讯网络...")
    while True:
        metrics = get_latest_metrics()
        if metrics:
            print_dashboard(metrics)
        else:
            print(f"⏳ 正在监听炼丹炉... 等待 {CSV_PATH} 生成第一个 Epoch 的战报...")
        # 每 10 秒刷新一次，给 CPU 一点喘息空间
        time.sleep(10)