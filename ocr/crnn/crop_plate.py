import os
import cv2
import numpy as np
from PIL import Image
from imutils import paths
from tqdm import tqdm

provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂",
             "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新", "警", "学", "O"]
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',
             'X', 'Y', 'Z', 'O']
ads = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X',
       'Y', 'Z', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'O']


def crop_plate(image_file, points):
    """
    Crop license plate from image using the given points.

    Args:
        image (PIL.Image): The input image.
        points (list): A list of four (x, y) tuples representing the corners of the plate.
    Returns:
        cropped_plate (PIL.Image): The cropped license plate image.
    """
    # Convert PIL image to OpenCV format
    image = cv2.imread(image_file)

    # Define source points from the given points
    src_pts = np.array(points, dtype='float32')

    # Calculate width and height of the cropped plate
    width_top = np.linalg.norm(src_pts[0] - src_pts[1])
    width_bottom = np.linalg.norm(src_pts[2] - src_pts[3])
    max_width = int(max(width_top, width_bottom))

    height_left = np.linalg.norm(src_pts[0] - src_pts[3])
    height_right = np.linalg.norm(src_pts[1] - src_pts[2])
    max_height = int(max(height_left, height_right))

    # Define destination points for the perspective transform
    dst_pts = np.array([[0, 0], # top-left
                        [max_width - 1, 0], # top-right
                        [max_width - 1, max_height - 1], # bottom-right
                        [0, max_height - 1]], dtype='float32') # bottom-left

    # Compute the perspective transform matrix
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)

    # Apply the perspective transformation to get the cropped plate
    cropped_plate_cv = cv2.warpPerspective(image, M, (max_width, max_height))

    # Convert back to PIL format
    cropped_plate = cv2.cvtColor(cropped_plate_cv, cv2.COLOR_BGR2RGB)
    cropped_plate_pil = Image.fromarray(cropped_plate)

    return cropped_plate_pil

def get_annotation(base_dir, out_dir=None, split=None):
    """
    Extract license plate annotation from the filename.

    Args:
        filename (str): The filename containing the annotation.
    Returns:
        annotation (str): The extracted license plate annotation.
    """
    if out_dir is not None:
        os.makedirs(out_dir, exist_ok=True)
        splits_dir = os.path.join(out_dir, 'splits')
        os.makedirs(splits_dir, exist_ok=True)
    img_paths = []
    if split is not None:
        split_file = os.path.join(base_dir, 'splits', f'{split}.txt')
        with open(split_file, 'r') as f:
            img_paths = [os.path.join(base_dir, line.strip()) for line in f if line.strip()]
    else:
        for i in range(len(base_dir)):
            img_paths += [el for el in paths.list_images(base_dir[i])]

    points = []

    for img_name in tqdm(img_paths):
        img = Image.open(img_name).convert('RGB')
        lbl = img_name.split('/')[-1].rsplit('.', 1)[0].split('-')[-3]
        YI = [int(ee) for ee in lbl.split('_')[:7]]
        iname = img_name.rsplit('/', 1)[-1].rsplit('.', 1)[0].split('-')
        [leftUp, rightDown] = [[int(eel) for eel in el.split('&')] for el in iname[2].split('_')]   # [[], []]
        [rightUp, leftDown] = [[rightDown[0], leftUp[1]], [leftUp[0], rightDown[1]]]
        points = [tuple(leftUp), tuple(rightUp), tuple(rightDown), tuple(leftDown)]
        cropped_plate = crop_plate(img_name, points)
        if out_dir is not None:
            temp_dir = os.path.join(out_dir, img_name.rsplit('/', 2)[-2])
            os.makedirs(temp_dir, exist_ok=True)
            out_path = os.path.join(temp_dir, img_name.rsplit('/', 1)[-1])
            cropped_plate.save(out_path)
            with open(os.path.join(splits_dir, f'{split}.txt'), 'a') as f:
                f.write(out_path + ' ' + ''.join([provinces[YI[0]], alphabets[YI[1]]] +
                                                 [ads[YI[i]] for i in range(2, 7)]) + '\n')
    print('Done!')

if __name__ == '__main__':
    base_dir = 'data/CCPD2019'
    out_dir = 'ocr/crnn/data'
    get_annotation(base_dir, out_dir=out_dir, split='train')
    get_annotation(base_dir, out_dir=out_dir, split='val')
    get_annotation(base_dir, out_dir=out_dir, split='test')