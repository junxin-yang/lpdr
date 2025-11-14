import os
from PIL import Image

img_dir = os.listdir('ocr/crnn/data/ccpd_base')
for img_name in img_dir:
    img_path = os.path.join('ocr/crnn/data/ccpd_base', img_name)
    img = Image.open(img_path)
    print(img.size)