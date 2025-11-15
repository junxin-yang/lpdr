#encoding:utf-8
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import argparse
import numpy as np
from os import path, mkdir
from load_data import *
from time import time
from shutil import copyfile
from component import fh02
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import tqdm
from collections import OrderedDict

plt.rcParams['font.sans-serif'] = [
    'Noto Sans CJK JP',
    'AR PL UMing CN',
    'AR PL UKai CN',
    'DejaVu Sans'
]
plt.rcParams['axes.unicode_minus'] = False

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True,
                    help="path to the input folder")
    ap.add_argument("-m", "--model", required=True,
                    help="path to the model file")
    args = vars(ap.parse_args())
    return args
args = get_args()


def create_dir(image_path, output_dir):
    base_dir, img_save_path = image_path.split('/')[2:]
    new_dir = os.path.join(output_dir, base_dir)
    os.makedirs(new_dir, exist_ok=True)
    return new_dir, img_save_path


def show_result(ims, fps_pred, text_pred, output_dir=None):
    base_dir, img_save_path = create_dir(ims, output_dir)
    img = cv2.imread(ims)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    box = fps_pred
    if len(box) == 4:
        h, w = img.shape[:2]
        x_center, y_center, w_box, h_box = box
        x1 = int((x_center - w_box/2) * w)
        y1 = int((y_center - h_box/2) * h)
        x2 = int((x_center + w_box/2) * w)
        y2 = int((y_center + h_box/2) * h)
        points = [
            (x1, y1),  # 左上
            (x2, y1),  # 右上
            (x2, y2),  # 右下
            (x1, y2)   # 左下
            ]
    else:
        raise ValueError("fps_pred should be a list or array of length 4.")

    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.axis('off')
    
    pts = np.array(points, np.int32)
    polygon = patches.Polygon(pts, linewidth=2, edgecolor='lime', 
                             facecolor='none', alpha=0.8)
    ax.add_patch(polygon)
    
    text = ''.join(text_pred)
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


def decode_plate(labelPred):
    # labelPred: [省份, 字母, 字符1, 字符2, 字符3, 字符4, 字符5]
    if len(labelPred) != 7:
        return ""
    prov = provinces[labelPred[0]]
    alpha = alphabets[labelPred[1]]
    ad_chars = [ads[idx] for idx in labelPred[2:]]
    return prov + alpha + ''.join(ad_chars)


def isEqual(labelGT, labelP):
    compare = [1 if int(labelGT[i]) == int(labelP[i]) else 0 for i in range(7)]
    return sum(compare)


numClasses = 4
numPoints = 4
imgSize = (480, 480)
resume_file = str(args["model"])

provNum, alphaNum, adNum = 38, 25, 35
provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


model_conv = fh02(numPoints, numClasses, provNum=provNum, alphaNum=alphaNum, adNum=adNum, device_id=device)
checkpoint = torch.load(resume_file)
new_state_dict = OrderedDict()

for k, v in checkpoint.items():
    name = k.replace('module.', '')  # 移除module.前缀
    new_state_dict[name] = v

# 加载处理后的state_dict
model_conv.load_state_dict(new_state_dict)
model_conv = model_conv.to(device)
model_conv.eval()

count = 0
correct = 0
error = 0
sixCorrect = 0

testdataset = labelTestDataLoader(args["input"], imgSize, split='test')
testloader = DataLoader(testdataset, batch_size=1, shuffle=True, num_workers=1)

lpdr_dir = "lpdr_results_rpnet"
os.makedirs(lpdr_dir, exist_ok=True)
total_class = {}

start = time()
for i, (XI, labels, ims) in tqdm.tqdm(enumerate(testloader), total=len(testloader)):
    img_path = ims[0]
    basename = os.path.basename(img_path)
    parent_dir = os.path.basename(os.path.dirname(img_path))
    total_class[parent_dir] = total_class.get(parent_dir, {})
    total_class[parent_dir]['total'] = total_class[parent_dir].get('total', 0) + 1
    count += 1
    YI = [[int(ee) for ee in el.split('_')[:7]] for el in labels]
    x = Variable(XI.to(device))
    # Forward pass: Compute predicted y by passing x to the model

    fps_pred, y_pred = model_conv(x)

    outputY = [el.data.cpu().numpy().tolist() for el in y_pred]
    labelPred = [t[0].index(max(t[0])) for t in outputY]

    text = decode_plate(labelPred)
    show_result(img_path, fps_pred[0].data.cpu().numpy().tolist(), text, output_dir=lpdr_dir)

    if isEqual(labelPred, YI[0]) == 7:
        correct += 1
        total_class[parent_dir]['correct'] = total_class[parent_dir].get('correct', 0) + 1
        sixCorrect += 1
    else:
        sixCorrect += 1 if isEqual(labelPred, YI[0]) == 6 else 0

    if count % 50 == 0:
        print ('total: %s correct: %s error: %s precision: %s six: %s avg_time: %s' % (count, correct, error, float(correct)/count, float(sixCorrect)/count, (time() - start)/count))

with open('accuracy_by_class_rpnet.json', 'w') as f:
    json.dump(total_class, f, indent=4, ensure_ascii=False)
with open('overall_accuracy_rpnet.txt', 'w') as f:
    f.write(f'Overall Accuracy: {correct}/{count} = {correct/count:.4f}\n')
