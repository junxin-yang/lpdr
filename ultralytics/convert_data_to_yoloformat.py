import os
import shutil
from tqdm import tqdm
import cv2

base_dir = 'data/CCPD2019'
save_dir = 'ultralytics/CCPD2019_YOLOformat'
images_dir = os.path.join(save_dir, 'images')
os.makedirs(images_dir, exist_ok=True)
labels_dir = os.path.join(save_dir, 'labels')
os.makedirs(labels_dir, exist_ok=True)

for split in ['train', 'val', 'test']:
    split_img_dir = os.path.join(images_dir, split)
    os.makedirs(split_img_dir, exist_ok=True)
    split_label_dir = os.path.join(labels_dir, split)
    os.makedirs(split_label_dir, exist_ok=True)
    with open(os.path.join(base_dir, 'splits', f'{split}.txt'), 'r') as f:
        img_paths = [os.path.join(base_dir, line.strip()) for line in f if line.strip()]
    for img_path in tqdm(img_paths, desc=f'Processing {split} set'):
        shutil.copy(img_path, split_img_dir)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        img_name = img_path.rsplit('/', 1)[-1].rsplit('.', 1)[0]
        iname = img_name.split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        new_labels = [(leftUp[0] + rightDown[0])/(2*ori_w), 
                      (leftUp[1] + rightDown[1])/(2*ori_h), 
                      (rightDown[0]-leftUp[0])/ori_w, 
                      (rightDown[1]-leftUp[1])/ori_h]
        label_path = os.path.join(split_label_dir, f'{img_name}.txt')
        with open(label_path, 'w') as lf:
            lf.write(f'0 {new_labels[0]} {new_labels[1]} {new_labels[2]} {new_labels[3]}\n')
print("Conversion to YOLO format completed.")

