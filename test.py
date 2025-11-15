import os
import pandas as pd
import matplotlib.pyplot as plt

csv_path = "ultralytics/detection_results/yolov8n_CCPD2019_plate_detection/results.csv"
save_dir = "yolo_train"
os.makedirs(save_dir, exist_ok=True)

# 读取csv
df = pd.read_csv(csv_path)

# 你可以根据实际csv内容调整这些字段
metrics = [
    ("train/box_loss", "训练框损失"),
    ("train/cls_loss", "训练类别损失"),
    ("train/dfl_loss", "训练DFL损失"),
    ("metrics/precision(B)", "精度"),
    ("metrics/recall(B)", "召回率"),
    ("metrics/mAP_0.5(B)", "mAP@0.5"),
    ("metrics/mAP_0.5:0.95(B)", "mAP@0.5:0.95"),
    ("val/box_loss", "验证框损失"),
    ("val/cls_loss", "验证类别损失"),
    ("val/dfl_loss", "验证DFL损失"),
]

for col, name in metrics:
    if col in df.columns:
        plt.figure()
        plt.plot(df["epoch"], df[col], marker='o')
        plt.xlabel("Epoch")
        plt.ylabel(name)
        plt.title(f"{name} 随 Epoch 变化")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{col.replace('/', '_')}.png"))
        plt.close()