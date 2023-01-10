# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import multi_gpu_test, single_gpu_test
from mmdet.datasets import (build_dataloader, build_dataset,
                            replace_ImageToTensor)
from mmdet.models import build_detector
from mmdet.utils import (build_ddp, build_dp, compat_cfg, get_device,
                         replace_cfg_vals, setup_multi_processes,
                         update_data_root)
from random import choice
try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--gpu-ids',
        type=int,
        nargs='+',
        help='(Deprecated, please use --gpu-id) ids of gpus to use '
        '(only applicable to non-distributed training)')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    args = parser.parse_args()
    return args

def get_random_cand(arch):
        architecture = dict()
        for key in list(arch.keys()):
            architecture[key] = []
            for a in arch[key]:
                architecture[key].append(choice(a))
        return architecture

def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    # replace the ${key} with the value of cfg.key
    cfg = replace_cfg_vals(cfg)

    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids[0:1]
        warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                      'Because we only support single GPU mode in '
                      'non-distributed testing. Use the first GPU '
                      'in `gpu_ids` now.')
    else:
        cfg.gpu_ids = [args.gpu_id]
    cfg.device = get_device()

    # build the model and load checkpoint
    average_flops = 0
    average_params = 0
    n = 50
    for i in range(n):
        # print(cfg.arch)
        arch = get_random_cand(cfg.arch)
        cfg.model.train_cfg = None
        cfg.model.backbone_rgb.arch = arch['backbone_rgb']
        cfg.model.backbone_thermal.arch = arch['backbone_thermal']
        cfg.model.head.arch = list(arch['head_rgb'])
        cfg.model.head_thermal.arch = list(arch['head_thermal'])
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)

        model = model.cuda()
        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                    format(model.__class__.__name__))
        input_shape = (3, 224, 224)
        flops, params = get_model_complexity_info(model, input_shape, as_strings=False, print_per_layer_stat=False)
        flops = round(flops / 10.**9, 2)
        params = round(params / 10 ** 6, 2)
        print(flops, params)
        average_flops += flops
        average_params += params
        # return params, flops
    print(average_flops/n, average_params/n)

    

if __name__ == '__main__':
    main()