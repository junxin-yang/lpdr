import os
from matplotlib import pyplot as plt
import cv2
import numpy as np

files = os.listdir('data/CCPD2019/ccpd_base')
output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)

for file in files[:5]:
    img_name = os.path.join('data/CCPD2019/ccpd_base', file)
    img = cv2.imread(img_name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resizedImage = cv2.resize(img, (480, 480))  # HWC

    iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
    [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

    ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
    img_w, img_h = 480, 480

    # 直接对原图角点做缩放
    scale_x = img_w / ori_w
    scale_y = img_h / ori_h
    leftUp_scaled = [int(leftUp[0] * scale_x), int(leftUp[1] * scale_y)]
    rightDown_scaled = [int(rightDown[0] * scale_x), int(rightDown[1] * scale_y)]

    # 可视化
    plt.imshow(resizedImage)
    plt.plot([leftUp_scaled[0], rightDown_scaled[0], rightDown_scaled[0], leftUp_scaled[0], leftUp_scaled[0]],
             [leftUp_scaled[1], leftUp_scaled[1], rightDown_scaled[1], rightDown_scaled[1], leftUp_scaled[1]], color='blue')
    plt.axis('off')
    plt.savefig(f"{output_dir}/{file}.png")
    plt.close()