import os
import shutil
from datetime import datetime
from pathlib import Path
import torch
from ultralytics import YOLO
from ultralytics.nn.modules.head import v10Detect
from ultralytics.utils import ops

# ===================== [1] 通信协议补丁（解决 Dict/Shape 报错的核心）=====================
_orig_nms = ops.non_max_suppression

def v3_lite_unboxed_nms(prediction, conf_thres=0.25, iou_thres=0.45, *args, **kwargs):
    """
    顶级专家补丁：从 YOLOv10 的输出字典中精准剥离 Tensor。
    解决 'dict' object has no attribute 'shape' 报错。
    """
    # 如果输出是字典（YOLOv10 默认），取 one2many 分支用于验证指标
    if isinstance(prediction, dict):
        prediction = prediction.get('one2many', prediction.get('one2one'))
    # 如果输出被包裹在列表或元组中，进行递归拆箱
    while isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    return _orig_nms(prediction, conf_thres, iou_thres, *args, **kwargs)

# 强行覆盖全局 NMS 算子
ops.non_max_suppression = v3_lite_unboxed_nms

# ===================== [2] 动力学控制与备份回调 =====================

def apply_scientific_reset(trainer):
    """[去噪涅槃] 重置检测头偏置，抑制 4 头架构带来的初始爆炸噪声"""
    print("\n" + "💉 " * 15)
    print("🔥 [Expert Ops] 偏置重置手术：注入 Prior = 0.01 抑制因子")
    model = trainer.model
    for m in model.modules():
        if isinstance(m, v10Detect):
            for cv in m.cv3:
                if hasattr(cv[-1], 'bias') and cv[-1].bias is not None:
                    cv[-1].bias.data.fill_(-4.59)
            # 确保在验证阶段不进入硬编码的 end2end 逻辑，配合我们的 NMS 补丁
            m.end2end = False
    print("💉 " * 15 + "\n")

def on_train_epoch_start(trainer):
    """[阶梯解冻] 保护 P2-Ghost 头在前期的学习純度"""
    epoch = trainer.epoch
    # 前 30 轮冻结 Backbone
    if epoch == 0:
        print("❄️ [Expert Ops] 初始状态：锁定 Backbone (Layer 0-9)")
        for k, v in trainer.model.named_parameters():
            if any(f'model.{i}.' in k for i in range(10)):
                v.requires_grad = False
    
    # 第 31 轮全网通车，此时 LR 已下降，适合微调主干
    elif epoch == 30:
        print("\n🔥 [Expert Ops] Epoch 30：全网解冻，Octo-Scan Mamba 参与最终收割")
        for param in trainer.model.parameters():
            param.requires_grad = True

def on_train_epoch_end(trainer):
    """[权重管理] 每 5 轮自动备份，防止停电或崩溃"""
    current_epoch = trainer.epoch + 1
    if current_epoch % 5 == 0:
        weights_dir = Path(trainer.save_dir) / 'weights'
        last_pt = weights_dir / 'last.pt'
        backup_dir = Path(trainer.save_dir) / 'backups'
        if last_pt.exists():
            backup_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(last_pt, backup_dir / f'epoch_{current_epoch}.pt')
            print(f"🛡️ 自动备份：epoch_{current_epoch}.pt 已存档")

# ===================== [3] 训练启动 =====================

if __name__ == "__main__":
    print("🚀 V3-Lite-Ghost Ultimate 启动 | 目标：0.427+ Baseline")
    
    # 初始化 4 头 P2 架构模型
    yaml_path = "/home/zhenr2024/yolov10/v3_lite_mamba.yaml"
    model = YOLO(yaml_path)

    # 挂载专家回调矩阵
    model.add_callback('on_pretrain_routine_start', apply_scientific_reset)
    model.add_callback('on_train_epoch_start', on_train_epoch_start)
    model.add_callback('on_train_epoch_end', on_train_epoch_end)

    # 启动全量总攻
    model.train(
        data="/home/zhenr2024/yolov10/VisDrone_DET.yaml",
        project="V3_Lite_Final_SOTA",
        name="Mamba_P2Ghost_Ultimate",
        epochs=200,
        batch=8,             # ⚠️ 降低 Batch 应对 4 头架构带来的 24.8G 显存压力
        imgsz=640,
        optimizer='AdamW',
        lr0=0.001,
        lrf=0.001,           # 极细打磨最后阶段
        close_mosaic=20,     # 最后 20 轮回归真实分布
        amp=False,           # 关闭混合精度，配合 loss.py 修改，保证小目标梯度稳定
        box=7.5,
        cls=1.5,             # 提升分类权重，配合 CA-TAL 手术
        dfl=1.5,
        patience=100         # 延长耐心，等待后期 mAP 爆发
    )