from ultralytics import YOLO


model = YOLO('ultralytics/detection_results/yolov8n_CCPD2019_plate_detection/weights/best.pt')

# metrics = model.val(
#     data='ultralytics/data.yaml',  # 数据集配置文件
#     split='val',                   # 使用验证集（val/test）
#     device=1,
#     plots=True                     # 生成评估图表
# )

# # 查看关键指标[9](@ref)
# print(f"精确率 (Precision): {metrics.box.mp}")
# print(f"召回率 (Recall): {metrics.box.mr}")
# print(f"mAP@0.5: {metrics.box.map50}")
# print(f"mAP@0.5:0.95: {metrics.box.map}")

# # 如果需要访问每个类别的详细指标
# if hasattr(metrics, 'boxes'):
#     print(f"每个类别的mAP: {metrics.box.maps}")

# print("----------")

metrics = model.val(
    data='ultralytics/data.yaml',  # 数据集配置文件
    split='test',                   # 使用验证集（val/test）
    device=1,
    plots=True                     # 生成评估图表
)

# 查看关键指标[9](@ref)
print(f"精确率 (Precision): {metrics.box.mp}")
print(f"召回率 (Recall): {metrics.box.mr}")
print(f"mAP@0.5: {metrics.box.map50}")
print(f"mAP@0.5:0.95: {metrics.box.map}")

# 如果需要访问每个类别的详细指标
if hasattr(metrics, 'boxes'):
    print(f"每个类别的mAP: {metrics.box.maps}")