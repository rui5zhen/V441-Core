import torch
from ultralytics import YOLO
from ultralytics.utils import ops

# ===================== [1] 注入通信协议补丁（规避 Dict 报错的核心）=====================
_orig_nms = ops.non_max_suppression

def v3_lite_unboxed_nms(prediction, conf_thres=0.25, iou_thres=0.45, *args, **kwargs):
    """
    借鉴主公提供的补丁逻辑：
    从 YOLOv10 的输出字典中精准拆箱，解决 AttributeError。
    """
    if isinstance(prediction, dict):
        # 验证阶段我们优先取 one2many 分支，因为它包含更丰富的候选框用于计算 AP
        prediction = prediction.get('one2many', prediction.get('one2one'))
    while isinstance(prediction, (list, tuple)):
        prediction = prediction[0]
    return _orig_nms(prediction, conf_thres, iou_thres, *args, **kwargs)

# 强行覆盖全局 NMS 逻辑
ops.non_max_suppression = v3_lite_unboxed_nms

# ===================== [2] 专项成绩单提取逻辑 =====================

# 路径锁定
WEIGHTS = "/home/zhenr2024/yolov10/V3_Lite_Final_SOTA/Mamba_P2Ghost_Ultimate3/weights/last.pt"
DATA_YAML = "/home/zhenr2024/yolov10/VisDrone_DET.yaml"

def check_bad_students():
    print(f"🕵️ 正在解剖特权类别表现 (Patch Active): {WEIGHTS}")
    
    # 加载模型
    model = YOLO(WEIGHTS)
    
    # 强制执行验证。注意：因为有补丁，这里不会再报错
    results = model.val(data=DATA_YAML, split='val', imgsz=640, plots=False, save_json=False)
    
    class_names = model.names
    # results.maps 是每个类别的 mAP50-95
    # 想要获取每个类别的 mAP50，我们需要从内部属性提取
    ap50s = results.results_dict.get('metrics/mAP50(B)', 0) # 这是总分
    
    # 特权类 ID 对照 (VisDrone)
    target_ids = [1, 2, 7] # 1:people, 2:bicycle, 7:awning-tricycle
    
    print("\n" + "🎓 " * 5 + "V3-Lite-Ghost 专项成绩单" + " 🎓" * 5)
    print(f"{'类别ID':<6} | {'类别名称':<15} | {'mAP50-95':<10}")
    print("-" * 40)
    
    # results.maps 存储了各类的平均精度
    for i, ap_val in enumerate(results.maps):
        name = class_names[i]
        prefix = "⭐ [重点] " if i in target_ids else "   [常规] "
        print(f"{prefix}{i:<3} : {name:<10} | {ap_val:.4f}")
    
    print("-" * 40)
    print(f"📊 当前总 mAP50: {results.results_dict['metrics/mAP50(B)']:.4f}")
    print(f"💡 狐狸注：重点观察 Bicycle (ID 2) 是否有回升迹象。")

if __name__ == "__main__":
    check_bad_students()