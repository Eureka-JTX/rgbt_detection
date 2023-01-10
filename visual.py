from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import random

cocoRoot = "data/align"
annFile = os.path.join(cocoRoot, f'FLIR_val.json')
print(f'Annotation file: {annFile}')

coco=COCO(annFile)

ids = coco.getCatIds('person')[0]    
id = coco.getCatIds(['car'])[0]      
imgIds = coco.catToImgs[id]
cats = coco.loadCats(1)               
imgIds = coco.getImgIds(catIds=[1])

imgId = imgIds[11]
imgInfo = coco.loadImgs(imgId)[0]

# imPath = os.path.join(cocoRoot, imgInfo['file_name_thermal'])
imPath = os.path.join(cocoRoot, imgInfo['file_name'])
print(imPath)            
im = cv2.imread(imPath)
print(im.shape)
# plt.axis('off')
# plt.imshow(im)
# plt.show()
plt.imshow(im)
# plt.axis('off')
# plt.savefig('img_visual/img.jpg')
annIds = coco.getAnnIds(imgIds=imgInfo['id'])
print(annIds)

anns = coco.loadAnns(annIds)
# print(anns)
for i in anns:
    print(i)
coco.showAnns(anns)

# mask = coco.annToMask(anns[3])
# plt.imshow(mask)
plt.savefig('img_visual/mask.jpg')
plt.axis('off')