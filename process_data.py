import json
import cv2
from tqdm import tqdm
from mmdet.datasets.api_wrappers import COCO
import mmcv
import os

prefix = '../FLIR_ADAS_v2/'
fmap = open(prefix + 'rgb_to_thermal_vid_map.json', 'rb')
fmap = json.load(fmap)

count = 0

coco = COCO(f'{prefix}video_thermal_test/coco.json')

img_ids = coco.get_img_ids()

data_infos = []
day = ["dvZBYnphN2BwdMKBc", "msNEBxJE5PPDqenBM", "RMxN6a4CcCeLGu4tA"]
for i in img_ids:
    info = coco.load_imgs([i])[0]

    info['filename'] = "video_thermal_test/" + info['file_name']
    # info['filename_thermal'] = "video_thermal_test/data/" + fmap[info['file_name'].replace("data/", "")]
    data_infos.append(info)

print(len(data_infos))

for idx in tqdm(range(len(data_infos))):
    img_id = data_infos[idx]['id']
    ann_ids = coco.get_ann_ids(img_ids=[img_id])
    ann_info = coco.load_anns(ann_ids)

    img_info = data_infos[idx]
    img_file = img_info['filename']
    # img_file_thermal = img_info['filename_thermal']

    if img_file.split("/")[2][6:23] in day:
        hour = "day"
    else:
        hour = "night"

    img = cv2.imread(prefix + img_file)
    # img_thermal = cv2.imread(prefix + img_file_thermal)
    # cv2.resize(img_thermal, img)
    # print(img.shape, img_thermal.shape)

    for bbox in ann_info:
        x, y, h, w = bbox['bbox']
        y1, x1, y2, x2 = x, y, x + h, y + w
        inter_w = max(0, min(x + h, img_info['width']) - max(x, 0))
        inter_h = max(0, min(y + w, img_info['height']) - max(y, 0))
        y1, x1 = max(y1, 0), max(x1, 0)
        y2, x2 = min(y2, img_info['width']), min(x2, img_info['height'])
        if inter_w * inter_h == 0:
            continue
        if w <= 1 or h <= 1:
            continue
        category = int(bbox['category_id'])

        croped_img = img[x1:x2, y1:y2]
        # croped_img_thermal = img_thermal[x1:x2, y1:y2]

        if not os.path.exists(f'{prefix}crop_image/{hour}/rgb/{category}/'):
            os.makedirs(f'{prefix}crop_image/{hour}/rgb/{category}/')
        # if not os.path.exists(f'{prefix}crop_image/{hour}/thermal/{category}/'):
        #     os.makedirs(f'{prefix}crop_image/{hour}/thermal/{category}/')

        file_name = f'{prefix}crop_image/{hour}/rgb/{category}/{count}.jpg'
        # file_name_thermal = f'{prefix}crop_image/{hour}/thermal/{category}/{count}.jpeg'

        cv2.imwrite(file_name, croped_img)
        # cv2.imwrite(file_name_thermal, croped_img_thermal)

        count += 1
print(count)
