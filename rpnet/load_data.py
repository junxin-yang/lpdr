from torch.utils.data import *
from imutils import paths
import os
import cv2
import numpy as np

class labelFpsDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None, split=None):
        self.img_dir = img_dir
        self.img_paths = []
        if split is not None:
            self.split_file = os.path.join(self.base_dir, 'splits', f'{split}.txt')
            with open(self.split_file, 'r') as f:
                self.img_paths = [os.path.join(self.base_dir, line.strip()) for line in f if line.strip()]
        else:
            for i in range(len(img_dir)):
                self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]
        ori_w, ori_h = [float(int(el)) for el in [img.shape[1], img.shape[0]]]
        assert img.shape[0] == 1160
        img_h, img_w = self.img_size
        scale_x = img_w / ori_w
        scale_y = img_h / ori_h
        leftUp_scaled = [int(leftUp[0] * scale_x), int(leftUp[1] * scale_y)]
        rightDown_scaled = [int(rightDown[0] * scale_x), int(rightDown[1] * scale_y)]
        new_labels = [(leftUp_scaled[0] + rightDown_scaled[0])/(2*img_w), 
                      (leftUp_scaled[1] + rightDown_scaled[1])/(2*img_h), 
                      (rightDown_scaled[0]-leftUp_scaled[0])/img_w, 
                      (rightDown_scaled[1]-leftUp_scaled[1])/img_h]
        return resizedImage, new_labels, lbl, img_name


class labelTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None, split=None):
        self.img_dir = img_dir
        self.img_paths = []
        if split is not None:
            self.split_file = os.path.join(self.base_dir, 'splits', f'{split}.txt')
            with open(self.split_file, 'r') as f:
                self.img_paths = [os.path.join(self.base_dir, line.strip()) for line in f if line.strip()]
        else:
            for i in range(len(img_dir)):
                self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        lbl = img_name.split('/')[-1].split('.')[0].split('-')[-3]
        return resizedImage, lbl, img_name


class ChaLocDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None, split=None):
        self.img_dir = img_dir
        self.img_paths = []
        self.base_dir = img_dir
        self.split = split
        if split is not None:
            self.split_file = os.path.join(self.base_dir, 'splits', f'{split}.txt')
            with open(self.split_file, 'r') as f:
                self.img_paths = [os.path.join(self.base_dir, line.strip()) for line in f if line.strip()]
        else:
            for i in range(len(img_dir)):
                self.img_paths += [el for el in paths.list_images(img_dir[i])]

        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2, 0, 1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0

        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]

        ori_w, ori_h = float(img.shape[1]), float(img.shape[0])
        assert img.shape[0] == 1160
        img_h, img_w = self.img_size
        scale_x = img_w / ori_w
        scale_y = img_h / ori_h
        leftUp_scaled = [leftUp[0] * scale_x, leftUp[1] * scale_y]
        rightDown_scaled = [rightDown[0] * scale_x, rightDown[1] * scale_y]
        new_labels = [(leftUp_scaled[0] + rightDown_scaled[0])/(2*img_w), 
                      (leftUp_scaled[1] + rightDown_scaled[1])/(2*img_h), 
                      (rightDown_scaled[0]-leftUp_scaled[0])/img_w, 
                      (rightDown_scaled[1]-leftUp_scaled[1])/img_h]
        return resizedImage, new_labels


class demoTestDataLoader(Dataset):
    def __init__(self, img_dir, imgSize, is_transform=None):
        self.img_dir = img_dir
        self.img_paths = []
        for i in range(len(img_dir)):
            self.img_paths += [el for el in paths.list_images(img_dir[i])]
        # self.img_paths = os.listdir(img_dir)
        # print self.img_paths
        self.img_size = imgSize
        self.is_transform = is_transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_name = self.img_paths[index]
        img = cv2.imread(img_name)
        # img = img.astype('float32')
        resizedImage = cv2.resize(img, self.img_size)
        resizedImage = np.transpose(resizedImage, (2,0,1))
        resizedImage = resizedImage.astype('float32')
        resizedImage /= 255.0
        return resizedImage, img_name

if __name__ == '__main__':
    data_loader = ChaLocDataLoader(
        img_dir='data/CCPD2019',
        imgSize=(480, 480),
        split='val'
    )
    print(len(data_loader))
    valloader = DataLoader(data_loader, batch_size=1, shuffle=False, num_workers=4)
    for im, lbl in valloader:
        print(im.shape, lbl)