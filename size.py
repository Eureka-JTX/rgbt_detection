# import json
# import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

# fr = open('data/align/FLIR_val.json')
# f = json.load(fr)

# ann = f['annotations']
# area = [a['area'] for a in ann]

# plt.hist(area, bins=10, range=(0,10000))

# plt.savefig('tmp.jpg')
a = [3, 0, 0, 2, 1, 0, 3, 1, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 2, 0, 0, 1, 2, 0, 0, 1, 3, 1, 3, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 3, 0, 1]
l = 4+4+8+4

arch = dict(
    backbone_rgb=[
        a[:l]
    ],
    backbone_thermal=[
        a[l:2*l]
    ],
    head_rgb=[
        a[2*l:2*l+4]
    ],
    head_thermal=[
        a[2*l+4:]
    ]
)
print(arch)