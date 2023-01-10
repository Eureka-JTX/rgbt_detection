from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random
from tqdm import tqdm

cocoRoot = "data/align"
# annFile = os.path.join(cocoRoot, f'FLIR_val.json')
annFile = os.path.join(cocoRoot, f'FLIR_train.json')
print(f'Annotation file: {annFile}')

coco=COCO(annFile)
count = 0
# mean_sum = 0
imgs = []

            
for i in range(1):
    print(i)
    imgIds = coco.getImgIds()
    # for j in tqdm(len(imgIds)):
        # imgId = imgIds[j]
    for imgId in tqdm(imgIds):
        # if count >10:
        #     break
        imgInfo = coco.loadImgs(imgId)[0]

        imPath = os.path.join(cocoRoot, imgInfo['file_name_thermal'])
        # imPath = os.path.join(cocoRoot, imgInfo['file_name'])
            
        img = cv2.imread(imPath)
        img = img.reshape(-1, 3)

        imgs.append(img)

        count += 1

imgs = np.concatenate(imgs, axis=0)
print(imgs.shape)
means = []
stds = []
for i in range(3):
    means.append(np.mean(imgs[:, i]))
    stds.append(np.std(imgs[:, i]))
print(count)
print(means)
print(stds)
