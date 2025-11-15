import sys
sys.path.append('/mnt/Disk1/yjx/code/LPDR')
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import os
from ocr.crnn.src.config import common_config as config
from ocr.crnn.src.model import CRNN
from ocr.crnn.src.ctc_decoder import ctc_decode
from ocr.crnn.src.dataset import CCPDDataset
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json


plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP',
    'AR PL UMing CN',
    'AR PL UKai CN',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False


def create_dir(image_path, output_dir):
    base_dir, img_save_path = image_path.split('/')[2:]
    new_dir = os.path.join(output_dir, base_dir)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir, img_save_path

# 检测车牌并返回四个角点
def detect_plate_corners(model, image_path, output_dir):
    results = model.predict(image_path, imgsz=640, conf=0.5, verbose=False, device=device)
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


def predict(crnn, data, label2char, decode_method, beam_size):
    crnn.eval()
    with torch.no_grad():
        images = data.to(device)
        logits = crnn(images)
        log_probs = torch.nn.functional.log_softmax(logits, dim=2)
        preds = ctc_decode(log_probs, method=decode_method, beam_size=beam_size,
                            label2char=label2char)
    return preds[0]


def show_result(path, points, pred, output_dir):
    
    base_dir, img_save_path = create_dir(path, output_dir)
    
    # 读取图片
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    pts = np.array(points, np.int32)
    polygon = patches.Polygon(pts, linewidth=2, edgecolor='lime', 
                             facecolor='none', alpha=0.8)
    ax.add_patch(polygon)
    
    text = ''.join(pred)
    x, y = points[0]
    
    text_bg = patches.Rectangle((x, y-30), len(text)*15, 25, 
                               facecolor='black', alpha=0.6, 
                               edgecolor='white', linewidth=1)
    ax.add_patch(text_bg)
    
    ax.text(x+5, y-10, text, fontsize=12, color='white', weight='bold', verticalalignment='top')
    
    
    plt.tight_layout()
    save_path = os.path.join(base_dir, img_save_path)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    return text


def isEqual(labelGT, labelP):
    if len(labelGT) != len(labelP):
        return 0
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare)


def load_points_gt(image_name):
    iname = image_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
    x1, y1 = int(leftUp[0]), int(leftUp[1])
    x2, y2 = int(rightDown[0]), int(rightDown[1])
    points_gt = [x1, y1, x2, y2]
    return points_gt


def corners_to_bbox(corners):

    x_coords = [point[0] for point in corners]
    y_coords = [point[1] for point in corners]
    
    # 计算边界框的坐标
    x_min = min(x_coords)
    y_min = min(y_coords)
    x_max = max(x_coords)
    y_max = max(y_coords)
    
    bbox = [x_min, y_min, x_max, y_max]
    return bbox


def calculate_iou(box1, box2):

    box1 = np.array(box1, dtype=np.float32)
    box2 = np.array(box2, dtype=np.float32)
    

    # 计算交集区域的坐标
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # 计算交集面积
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    intersection_area = inter_width * inter_height
    
    # 计算各自面积
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 计算并集面积
    union_area = area1 + area2 - intersection_area
    
    # 避免除零错误
    if union_area == 0:
        return 0.0
    
    # 计算IoU
    iou = intersection_area / union_area
    
    return iou



def main():

    base_dir = "data/CCPD2019"
    split_files = "ocr/crnn/data/splits/test.txt"
    detect_dir = "detect_dir"
    os.makedirs(detect_dir, exist_ok=True)
    crop_dir = "crop_results"
    os.makedirs(crop_dir, exist_ok=True)
    lpdr_dir = "lpdr_results"
    os.makedirs(lpdr_dir, exist_ok=True)

    model_path = "ultralytics/detection_results/yolov8n_CCPD2019_plate_detection/weights/best.pt"
    model = YOLO(model_path)

    num_class = len(CCPDDataset.LABEL2CHAR) + 1
    crnn = CRNN(3, config['img_height'], config['img_width'], num_class,
                map_to_seq_hidden=config['map_to_seq_hidden'],
                rnn_hidden=config['rnn_hidden'],
                leaky_relu=config['leaky_relu'])
    reload_checkpoint = "ocr/crnn/checkpoints/crnn_best_epoch23_iter069000_loss0.03195886522659069.pt"
    crnn.load_state_dict(torch.load(reload_checkpoint, map_location=device))
    crnn.to(device)

    decode_method = "greedy"
    beam_size = 10
    total_class = {}

    with open(split_files, "r") as f:
        data_dir = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            # 按第一个空格分割
            parts = line.split(' ', 1)
            if len(parts) == 2:
                img_path, label = parts
                basename = os.path.basename(img_path)
                parent_dir = os.path.basename(os.path.dirname(img_path))
                total_class[parent_dir] = total_class.get(parent_dir, {})
                total_class[parent_dir]['FN'] = total_class[parent_dir].get('FN', 0)
                total_class[parent_dir]['TN'] = total_class[parent_dir].get('TN', 0)
                result = os.path.join(parent_dir, basename)
                data_dir.append((result, label))
            else:
                data_dir.append((parts[0], None))

    correct = 0
    total = 0
    for image_name, label in tqdm(data_dir):
        if label is None:
            continue
        class_name = os.path.dirname(image_name)
        total_class[class_name]['total'] = total_class[class_name].get('total', 0) + 1
        total += 1

        image_path = os.path.join(base_dir, image_name)
        points = detect_plate_corners(model, image_path, detect_dir)
        if not points:
            print(f'No detection: {image_path}')
            total_class[class_name]['FN'] = total_class[class_name].get('FN', 0) + 1
            continue
        if points:
            cropped_plate = crop_plate_by_points(image_path, points)
            img_dir, img_save_path = create_dir(image_path, crop_dir)
            cropped_plate.save(os.path.join(img_dir, img_save_path))

        
        box_gt = load_points_gt(image_name)
        box_pred = corners_to_bbox(points)
        iou = calculate_iou(box_gt, box_pred)
        if iou < 0.5:
            total_class[class_name]['FP'] = total_class[class_name].get('FP', 0) + 1
        else:
            total_class[class_name]['TP'] = total_class[class_name].get('TP', 0) + 1
        
        cropped_resized = cropped_plate.resize(
            (config['img_width'], config['img_height']), resample=Image.LANCZOS)
        
        img_np = np.array(cropped_resized)  # (H, W, 3)
        img_np = img_np.transpose(2, 0, 1)  # (3, H, W)
        img_np = img_np.astype(np.float32) / 127.5 - 1.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0).to(device)  # (1, 3, H, W)

        lp_preds = predict(crnn, img_tensor, CCPDDataset.LABEL2CHAR,
                        decode_method=decode_method,
                        beam_size=beam_size)
        
        text = show_result(image_path, points, lp_preds, lpdr_dir)
        pred_text = [CCPDDataset.CHAR2LABEL[c] for c in text]
        label_text = [CCPDDataset.CHAR2LABEL[c] for c in label]
        if isEqual(pred_text, label_text) == 7:
            correct += 1
            total_class[class_name]['correct'] = total_class[class_name].get('correct', 0) + 1

    print(f'Accuracy: {correct}/{total} = {correct/total:.4f}')
    
    with open('accuracy_by_class.json', 'w') as f:
        json.dump(total_class, f, indent=4, ensure_ascii=False)
    with open('overall_accuracy.txt', 'w') as f:
        f.write(f'Overall Accuracy: {correct}/{total} = {correct/total:.4f}\n')


if __name__ == '__main__':
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(f'device: {device}')
    main()