import torch

f = torch.load('/data0/linzhiwei/pretrain_ckpt/COCO_FPN_300M_supernet.pkl')

arch = [3, 0, 0, 2, 1, 0, 3, 1, 2, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 0, 3, 2, 0, 0, 1, 2, 0, 0, 1, 3, 1, 3, 0, 2, 1, 2, 0, 2, 0, 2, 0, 1, 1, 3, 0, 1]
length = 4+4+8+4
arch_rgb = arch[:length]
arch_thermal = arch[length: 2*length]

weight_rgb = dict()
weight_thermal = dict()

# for i in range(length):
# keys = f.keys())
for key, value in f.items():
    if 'first_conv' in key:
        weight_rgb[key] = value
        weight_thermal[key] = value
    if 'features' in key:
        stage_id = int(key.split('.')[1])
        op_id = arch_rgb[stage_id]
        if f'features.{stage_id}.{op_id}' in key:
            new_key = f'features.{stage_id}' + key[len(f'features.{stage_id}.{op_id}'):]
            weight_rgb[new_key] = value
        
        op_id = arch_thermal[stage_id]
        if f'features.{stage_id}.{op_id}' in key:
            new_key = f'features.{stage_id}' + key[len(f'features.{stage_id}.{op_id}'):]
            weight_thermal[new_key] = value

torch.save(weight_rgb, 'work_dirs/top1_subnet_rgb.pkl')
torch.save(weight_thermal, 'work_dirs/top1_subnet_thermal.pkl')
        


