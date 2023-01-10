import json
import copy

train_path = '../FLIR_ADAS_v2/images_rgb_train/coco.json'
val_path = '../FLIR_ADAS_v2/images_rgb_val/coco.json'
f = open(train_path, 'rb')
info = json.load(f)

img_info = info['images']
anns = info['annotations']

img_day = []
img_night = []
img_night_dusk = []

cat = []

for ann in anns:
    c = ann['category_id']
    if c not in cat:
        cat.append(c)
print(cat)
print(len(cat))

for i in cat:
    print('\'{}\','.format(info['categories'][i]['name']))

print(len(img_info))
for img in img_info:
    w = img['extra_info']
    if 'hours' in w:
        w = w['hours']
        # assert w=='day'
        if w == 'day':
            img_day.append(copy.deepcopy(img))
        else:
            img_night_dusk.append(copy.deepcopy(img))
        if w == 'night':
            img_night.append(copy.deepcopy(img))

info_day = copy.deepcopy(info)
info_day['images'] = img_day

info_night = copy.deepcopy(info)
info_night['images'] = img_night

info_night_dusk = copy.deepcopy(info)
info_night_dusk['images'] = img_night_dusk

with open('../FLIR_ADAS_v2/coco_day.json', 'w', encoding='utf-8') as f:
    json.dump(info_day, f, ensure_ascii=False)
with open('../FLIR_ADAS_v2/coco_night.json', 'w', encoding='utf-8') as f:
    json.dump(info_night, f, ensure_ascii=False)
with open('../FLIR_ADAS_v2/coco_night_dusk.json', 'w', encoding='utf-8') as f:
    json.dump(info_night_dusk, f, ensure_ascii=False)
""" 
with open('/data/linzhiwei/data/images_rgb_val/coco_day.json','w',encoding='utf-8') as f:
    json.dump(info_day, f,ensure_ascii=False)

with open('/data/linzhiwei/data/images_rgb_val/coco_night.json','w',encoding='utf-8') as f:
    json.dump(info_night, f,ensure_ascii=False)

with open('/data/linzhiwei/data/images_rgb_val/coco_night_dusk.json','w',encoding='utf-8') as f:
    json.dump(info_night_dusk, f,ensure_ascii=False)


import json
import copy

# for path in ['/data/linzhiwei/data/images_rgb_val/coco_night_dusk.json',]:

#     f = open(path, 'rb')
#     info = json.load(f)

    

#     info['categories'] = 


#     with open(path,'w',encoding='utf-8') as f:
#         json.dump(info, f,ensure_ascii=False)
f = open('/data/linzhiwei/data/images_rgb_val/coco_night_dusk.json', 'rb')
info = json.load(f)

f2 = open('/data/linzhiwei/data/images_rgb_val/coco.json', 'rb')
info2 = json.load(f2)

info['categories'] = info2['categories']

with open('/data/linzhiwei/data/images_rgb_val/coco_night_dusk_old_cat.json','w',encoding='utf-8') as f:
    json.dump(info, f,ensure_ascii=False)
"""
