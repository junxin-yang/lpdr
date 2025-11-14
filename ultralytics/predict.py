from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os


def create_dir(image_path, output_dir):
    base_dir, img_save_path = image_path.split('/')[2:]
    new_dir = os.path.join(output_dir, base_dir)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir, img_save_path

# 检测车牌并返回四个角点
def detect_plate_corners(model, image_path, output_dir):
    results = model.predict(image_path, imgsz=640, conf=0.5)
    result = results[0].cpu()
    img = result.plot()
    base_dir, img_save_path = create_dir(image_path, output_dir)
    cv2.imwrite(os.path.join(base_dir, img_save_path), img)
    # 只取置信度最高的一个框
    if len(result.boxes.xyxy) == 0:
        return None
    box = result.boxes.xyxy[0].tolist()  # [x1, y1, x2, y2]
    x1, y1, x2, y2 = map(int, box)
    # 四个角点：左上、右上、右下、左下
    points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
    return points

# 根据四个角点裁剪车牌
def crop_plate_by_points(image_path, points):
    image = cv2.imread(image_path)
    src_pts = np.array(points, dtype='float32')
    width_top = np.linalg.norm(src_pts[0] - src_pts[1])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    max_width = int(max(width_top, width_bottom))
    height_left = np.linalg.norm(src_pts[0] - src_pts[3])
    height_right = np.linalg.norm(src_pts[1] - src_pts[2])
    max_height = int(max(height_left, height_right))
    dst_pts = np.array([[0, 0], [max_width - 1, 0], [max_width - 1, max_height - 1], [0, max_height - 1]], dtype='float32')
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    cropped_plate_cv = cv2.warpPerspective(image, M, (max_width, max_height))
    cropped_plate = cv2.cvtColor(cropped_plate_cv, cv2.COLOR_BGR2RGB)
    cropped_plate_pil = Image.fromarray(cropped_plate)
    return cropped_plate_pil

def main():
    detect_dir = "detect_dir"
    os.makedirs(detect_dir, exist_ok=True)
    crop_dir = "crop_results"
    os.makedirs(crop_dir, exist_ok=True)
    model_path = "ultralytics/detection_results/yolov8n_CCPD2019_plate_detection/weights/best.pt"
    model = YOLO(model_path)
    image_path = "data/CCPD2019/ccpd_db/02-1_4-352&371_544&458-539&454_352&458_357&375_544&371-0_0_20_24_21_30_27-19-2.jpg"
    points = detect_plate_corners(model, image_path, detect_dir)
    if points:
        cropped_plate = crop_plate_by_points(image_path, points)
        base_dir, img_save_path = create_dir(image_path, crop_dir)
        cropped_plate.save(os.path.join(base_dir, img_save_path))

# 示例用法
if __name__ == '__main__':
    main()