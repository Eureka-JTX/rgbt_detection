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

import torch.distributed as dist
from random import choice
import functools
import random
import numpy as np

try:
    from mmcv.cnn import get_model_complexity_info
except ImportError:
    raise ImportError('Please upgrade mmcv to >0.6.2')

from torch.autograd import Variable
import collections
import sys
import time
import copy
import logging
sys.setrecursionlimit(10000)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True


def seed(seed=0):
    import os
    import sys
    import torch
    import numpy as np
    import random
    sys.setrecursionlimit(100000)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

def get_broadcast_cand(arch, distributed, rank):
    if distributed:
        torch.cuda.synchronize(device=0)
        time.sleep(2)
        if isinstance(arch, dict):
            for key in arch:
                cand = arch[key]
                if isinstance(cand, tuple) or isinstance(cand, list):
                    cand = torch.tensor(cand, device='cuda')
                    dist.broadcast(cand, 0)
                    cand = tuple(cand.tolist())
                else:
                    cand = torch.tensor(cand, device='cuda')
                    dist.broadcast(cand, 0)
                    cand = cand.item()
                arch[key] = cand
        else:
            arch = torch.tensor(arch, device='cuda')
            dist.broadcast(arch, 0)
            arch = tuple(arch.tolist())

    return arch

def dict_to_tuple(cand):
    cand_tuple = []
    for key in cand:
        ar = cand[key]
        if isinstance(ar, tuple) or isinstance(ar, list):
            cand_tuple += list(cand[key])
        else:
            cand_tuple.append(cand[key])
    cand_tuple = tuple(cand_tuple)
    return cand_tuple


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument('--out', help='output result file in pickle format')
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
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        default='bbox',
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "bbox",'
        ' "segm", "proposal" for COCO, and "mAP", "recall" for PASCAL VOC')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--show-score-thr',
        type=float,
        default=0.3,
        help='score threshold (default: 0.3)')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)

    parser.add_argument('--seed', type=int, default=1, help='random seed')

    parser.add_argument('--log-dir', type=str, default='log')
    parser.add_argument('--max-epochs', type=int, default=20)
    parser.add_argument('--select-num', type=int, default=10)
    parser.add_argument('--population-num', type=int, default=50)
    parser.add_argument('--m_prob', type=float, default=0.1)
    parser.add_argument('--crossover-num', type=int, default=15)
    parser.add_argument('--mutation-num', type=int, default=15)
    parser.add_argument('--flops-limit', type=float, default=205) # 17.651 M 122.988 GFLOPS
    parser.add_argument('--shape',
        type=int,
        nargs='+',
        default=[1280, 800])

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args

def no_grad_wrapper(func):
    def new_func(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return new_func


@no_grad_wrapper
def get_cand_err(model, args, distributed, cfg, data_loader, dataset, architecture=None):

    if not distributed:
        model = build_dp(model, cfg.device, device_ids=cfg.gpu_ids)
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                args.show_score_thr, arch=architecture)
    else:
        model = build_ddp(
            model,
            cfg.device,
            device_ids=[int(os.environ['LOCAL_RANK'])],
            broadcast_buffers=False)
        outputs = multi_gpu_test(
            model, data_loader, args.tmpdir, args.gpu_collect
            or cfg.evaluation.get('gpu_collect', False), arch=architecture)
    model.eval()
    rank, _ = get_dist_info()
    if rank == 0:
        print('starting test....')
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)
        if args.eval:
            eval_kwargs = cfg.get('evaluation', {}).copy()
            # hard-code way to remove EvalHook args
            for key in [
                    'interval', 'tmpdir', 'start', 'gpu_collect', 'save_best',
                    'rule', 'dynamic_intervals'
            ]:
                eval_kwargs.pop(key, None)
            eval_kwargs.update(dict(metric='accuracy', **kwargs))
            metric = dataset.evaluate(outputs, **eval_kwargs)
            # print(metric)
            # print(metric)
            if metric:
                # print(metric)
                return metric['accuracy_top-1']
            else:
                return -1.0
    
    return -1.0

# @no_grad_wrapper
# def get_cand_err(model, args, distributed, cfg, data_loader, dataset, architecture=None):
#     rank, _ = get_dist_info()
#     if rank == 0:
#         return 1
    
#     return -1

class EvolutionSearcher(object):

    def __init__(self, args):
        self.args = args
        self.max_epochs = args.max_epochs
        self.select_num = args.select_num
        self.population_num = args.population_num
        self.m_prob = args.m_prob
        self.crossover_num = args.crossover_num
        self.mutation_num = args.mutation_num
        self.flops_limit = args.flops_limit
        cfg = Config.fromfile(args.config)
        self.arch = cfg.arch

        # replace the ${key} with the value of cfg.key
        cfg = replace_cfg_vals(cfg)

        # update data root according to MMDET_DATASETS
        update_data_root(cfg)

        if args.cfg_options is not None:
            cfg.merge_from_dict(args.cfg_options)

        cfg = compat_cfg(cfg)

        # set multi-process settings
        setup_multi_processes(cfg)

        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if 'pretrained' in cfg.model:
            cfg.model.pretrained = None
        # elif 'init_cfg' in cfg.model.backbone:
        #     cfg.model.backbone.init_cfg = None
        cfg.model.backbone_rgb.init_cfg = None
        cfg.model.backbone_thermal.init_cfg = None

        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None

        if args.gpu_ids is not None:
            cfg.gpu_ids = args.gpu_ids[0:1]
            warnings.warn('`--gpu-ids` is deprecated, please use `--gpu-id`. '
                        'Because we only support single GPU mode in '
                        'non-distributed testing. Use the first GPU '
                        'in `gpu_ids` now.')
        else:
            cfg.gpu_ids = [args.gpu_id]
        cfg.device = get_device()
        # init distributed env first, since logger depends on the dist info.
        if args.launcher == 'none':
            distributed = False
        else:
            distributed = True
            init_dist(args.launcher, **cfg.dist_params)
        self.distributed = distributed

        test_dataloader_default_args = dict(
            samples_per_gpu=1, workers_per_gpu=2, dist=distributed, shuffle=False)

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                # Replace 'ImageToTensor' to 'DefaultFormatBundle'
                cfg.data.test.pipeline = replace_ImageToTensor(
                    cfg.data.test.pipeline)
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True
            if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
                for ds_cfg in cfg.data.test:
                    ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

        test_loader_cfg = {
            **test_dataloader_default_args,
            **cfg.data.get('test_dataloader', {})
        }

        rank, _ = get_dist_info()
        # allows not to create
        if args.work_dir is not None and rank == 0:
            mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
            timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())

        # build the dataloader
        dataset = build_dataset(cfg.data.test)
        data_loader = build_dataloader(dataset, **test_loader_cfg)

        # self.data_loaders = data_loader
        self.data_loaders = data_loader
        self.datasets = dataset

        # build the model and load checkpoint
        cfg.model.train_cfg = None
        model = build_detector(cfg.model, test_cfg=cfg.get('test_cfg'))
        fp16_cfg = cfg.get('fp16', None)
        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')
        if args.fuse_conv_bn:
            model = fuse_conv_bn(model)
        # old versions did not save class info in checkpoints, this walkaround is
        # for backward compatibility
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES
        
        self.model = model
        self.cfg = cfg

        self.memory = []
        self.vis_dict = {}
        self.keep_top_k = {self.select_num: [], 50: []}
        self.epoch = 0
        self.candidates = []

        times = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        seed(0)
        self.log_path = args.checkpoint[:-4]
        self.logfile = args.checkpoint[:-4] + '_ea_fp{}_{}.log'.format(self.flops_limit, times)
        # self.input_shape = (3,) + tuple(args.shape)
        self.flop_upper_bound = 8.24 
        self.param_upper_bound = 46.87
        self.input_shape = (3, 224, 224)

    def get_param(self, cand):
        arch = cand

        cfg = Config.fromfile('configs/FLIR_cls/detnas_fusion_search_subnet.py')
        cfg.model.backbone_rgb.arch = arch['backbone_rgb']
        cfg.model.backbone_thermal.arch = arch['backbone_thermal']
        cfg.model.head.arch = list(arch['head_rgb'])
        cfg.model.head_thermal.arch = list(arch['head_thermal'])
        # cfg.model.head_fusion_thermal.arch = list(arch['head_2'])
        # print(cfg.model.roi_head)
        model = build_detector(
            cfg.model, test_cfg=cfg.get('test_cfg'))
        if torch.cuda.is_available():
            model.cuda()
        model.eval()

        if hasattr(model, 'forward_dummy'):
            model.forward = model.forward_dummy
        else:
            raise NotImplementedError(
                'FLOPs counter is currently not currently supported with {}'.
                    format(model.__class__.__name__))

        flops, params = get_model_complexity_info(model, self.input_shape, as_strings=False, print_per_layer_stat=False)
        flops = round(flops / 10.**9, 2)
        params = round(params / 10 ** 6, 2)
        rank, world_size = get_dist_info()
        if rank == 0:
            print(flops, params, arch)
        return params, flops

    def is_legal(self, cand):
        time.sleep(2)
        rank, world_size = get_dist_info()
        cand = get_broadcast_cand(cand, self.distributed, rank)

        cand_tuple = dict_to_tuple(cand)
        if cand_tuple not in self.vis_dict:
            self.vis_dict[cand_tuple] = {}
        info = self.vis_dict[cand_tuple]
        if 'visited' in info:
            return False
        param, fp = self.get_param(cand)
        if fp > self.flops_limit or param > self.param_upper_bound:
            return False
        # size, fp = 0, 0
        info['fp'] = fp
        info['size'] = param

        with open(self.log_path + '_gpu_{}.txt'.format(rank),'a') as f:
            f.write(str(cand)+'\n')
        # map = [1,1,1]
        map = get_cand_err(self.model, self.args, self.distributed, self.cfg, self.data_loaders, self.datasets, cand)
        if not isinstance(map, tuple):
            map = tuple([map])
        torch.cuda.synchronize(device=0)
        # print(map, rank)
        map = get_broadcast_cand(map, self.distributed, rank)

        if map:
            # print(map, rank)
            info['mAP'] = map[0]
            info['visited'] = True
            if rank == 0:
                print('is_legal', info, cand, cand_tuple)
            return map

        return False

    def update_top_k(self, candidates, *, k, key, reverse=True):
        assert k in self.keep_top_k
        print('select ......')
        t = self.keep_top_k[k]
        t += candidates

        t.sort(key=key, reverse=reverse)
        self.keep_top_k[k] = t[:k]

    def stack_random_cand(self, random_func, *, batchsize=10):

        while True:
            time.sleep(2)
            cands = [random_func() for _ in range(batchsize)]
            for cand in cands:
                if isinstance(cand, dict):
                    cand = dict_to_tuple(cand)
                # print(cand.__class__)
                # print(cand)
                if cand not in self.vis_dict:
                    self.vis_dict[cand] = {}
            for cand in cands:
                yield cand
    
    def get_random_cand(self):
        architecture = dict()
        for key in list(self.arch.keys()):
            architecture[key] = []
            for arch in self.arch[key]:
                architecture[key].append(choice(arch))
        return architecture

    def get_random(self, num):
        print('random select ........')
        # cand_iter = self.stack_random_cand(
        #             lambda: tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer)] + [np.random.randint(self.cb_type), np.random.randint(1, self.cb_step)] + [
        #                 np.random.randint(self.c_range[0], self.c_range[1]) // 16 * 16] + [np.random.randint(self.d_range[0], self.d_range[1])]))
        # cand_iter = self.stack_random_cand(
        #     lambda: {'fpn_arch': tuple([np.random.randint(self.nr_state) for i in range(self.nr_layer)]),
        #              'c_weight': np.random.randint(self.c_range[0], self.c_range[1]) // 16 * 16,
        #              'fpn_step': np.random.randint(self.d_range[0], self.d_range[1]),
        #              'cb_type': np.random.randint(self.cb_type),
        #              'cb_step': np.random.randint(1, self.cb_step)
        #     })

        cand_iter = self.stack_random_cand(self.get_random_cand)

        while len(self.candidates) < num:
            cand = next(cand_iter)
            rank, world_size = get_dist_info()
            cand = get_broadcast_cand(cand, self.distributed, rank)

            map = self.is_legal(cand)
            if not map:
                continue
            cand_tuple = dict_to_tuple(cand)
            self.candidates.append(cand_tuple)
            if rank == 0:
                print('random {}/{}, {}, {}'.format(len(self.candidates), num, rank, cand))
            # opt = [self.primitives[i] for i in cand['fpn_arch']]
            cand_tuple = dict_to_tuple(cand)
            # if self.args.eval[0]=="bbox":
            #     logging.info('random {}/{}, {}, AP {}, AP 50 {}, AP 75 {}, {}, {} M, {} GFLOPS'.format(len(self.candidates), num, opt, map[0], map[1], map[2], cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))
            # else:
            logging.info('random {}/{}, mAP {}, {}, {} M, {} GFLOPS'.format(len(self.candidates), num, map, cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))

        print('random_num = {}'.format(len(self.candidates)))


    def get_mutation(self, k, mutation_num, m_prob):
        assert k in self.keep_top_k
        print('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num * 10

        def random_func():
            cand = list(choice(self.keep_top_k[k]))
            # for i in range(len(cand)):
            i = 0
            for key in list(self.arch.keys()):
                for arch in self.arch[key]:
                    if np.random.random_sample() < m_prob:
                        cand[i] = choice(arch)
                    i += 1
            return tuple(cand)

        cand_iter = self.stack_random_cand(random_func)

        while len(res) < mutation_num and max_iters > 0:
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)

            # cand = {'fpn_arch': cand_tuple[:self.nr_layer],
            #  'c_weight': cand_tuple[self.nr_layer],
            #  'fpn_step': cand_tuple[self.nr_layer+1],
            #  'cb_type': cand_tuple[self.nr_layer+2],
            #  'cb_step': cand_tuple[self.nr_layer+3]
            #  }
            cand = dict()
            i = 0
            for key in list(self.arch.keys()):
                # for arch in self.arch[key]:
                cand[key] = cand_tuple[i:i+len(self.arch[key])]
                i += len(self.arch[key])

            cand = get_broadcast_cand(cand, self.distributed, rank)

            map = self.is_legal(cand)
            if not map:
                continue
            res.append(cand_tuple)
            if rank == 0:
                print('mutation {}/{}: {}, {}'.format(len(res), mutation_num, rank, cand))

            # opt = [self.primitives[i] for i in cand['fpn_arch']]

            cand_tuple = dict_to_tuple(cand)
            # if self.args.eval[0]=="bbox":
            #     logging.info('mutation {}/{}, {}, AP {}, AP 50 {}, AP 75 {}, {}, {} M, {} GFLOPS'.format(len(res), mutation_num, opt, map[0], map[1], map[2], cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))
            # else:
            logging.info('mutation {}/{}, {} mAP, {}, {} M, {} GFLOPS'.format(len(res), mutation_num, map, cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))

        print('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self, k, crossover_num):
        assert k in self.keep_top_k
        print('crossover ......')
        res = []
        max_iters = 10 * crossover_num

        def random_func():
            p1 = choice(self.keep_top_k[k])
            p2 = choice(self.keep_top_k[k])
            cand = []
            for i in range(len(p1)):
                rand = np.random.randint(2)
                if rand:
                    cand.append(p2[i])
                else:
                    cand.append(p1[i])

            return tuple(cand)


        cand_iter = self.stack_random_cand(random_func)
        while len(res) < crossover_num and max_iters > 0:
            max_iters -= 1
            cand_tuple = next(cand_iter)
            rank, world_size = get_dist_info()
            cand_tuple = get_broadcast_cand(cand_tuple, self.distributed, rank)
            cand = dict()
            i = 0
            for key in list(self.arch.keys()):
                # for arch in self.arch[key]:
                cand[key] = cand_tuple[i:i+len(self.arch[key])]
                i += len(self.arch[key])
            cand = get_broadcast_cand(cand, self.distributed, rank)

            map = self.is_legal(cand)
            if not map:
                continue
            res.append(cand_tuple)
            if rank == 0:
                print('crossover {}/{}: {}, {}'.format(len(res), crossover_num, rank, cand))
                # opt = [self.primitives[i] for i in cand['fpn_arch']]
                print(cand_tuple, cand, self.vis_dict[cand_tuple], rank)
            # if self.args.eval[0]=="bbox":
            #     logging.info('crossover {}/{}, {}, AP {}, AP 50 {}, AP 75 {}, {}, {} M, {} GFLOPS'.format(len(res), crossover_num, opt, map[0], map[1], map[2], cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))
            # else:
            logging.info('crossover {}/{}, mAP {}, {}, {}, {} GFLOPS'.format(len(res), crossover_num, map, cand, self.vis_dict[cand_tuple]['size'], self.vis_dict[cand_tuple]['fp']))
        print('crossover_num = {}'.format(len(res)))
        return res

    def search(self):
        rank, _ = get_dist_info()
        if rank == 0:
            print(
            'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
            logging.basicConfig(filename=self.logfile, level=logging.INFO)
            print(self.logfile)
            logging.info(self.cfg)
            logging.info(
                'population_num = {} select_num = {} mutation_num = {} crossover_num = {} random_num = {} max_epochs = {}'.format(
                    self.population_num, self.select_num, self.mutation_num, self.crossover_num,
                    self.population_num - self.mutation_num - self.crossover_num, self.max_epochs))
        self.get_random(self.population_num)
        torch.cuda.synchronize(device=0)
        while self.epoch < self.max_epochs:
            if rank == 0:
                print('epoch = {}'.format(self.epoch))

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
            if rank == 0:
                print(self.candidates, self.vis_dict)
            self.update_top_k(
                self.candidates, k=self.select_num, key=lambda x: self.vis_dict[x]['mAP'])
            self.update_top_k(
                self.candidates, k=50, key=lambda x: self.vis_dict[x]['mAP'])
            torch.cuda.synchronize(device=0)
            if rank == 0:
                print('epoch = {} : top {} result'.format(
                    self.epoch, len(self.keep_top_k[50])))
                logging.info('epoch = {} : top {} result'.format(
                    self.epoch, len(self.keep_top_k[50])))
            for i, cand in enumerate(self.keep_top_k[50]):

                # opt = [self.primitives[i] for i in cand[:self.nr_layer]]

                # if self.args.eval[0]=="bbox":
                #     logging.info('No.{} {}, channel:{}, depth:{}, size:{}M, {} GFLOPS,   mAP = {} mAP_50= {} mAP_75= {}, {}'.format(
                #         i + 1, opt, cand[-2], cand[-1] , self.vis_dict[cand]['size'], self.vis_dict[cand]['fp'], self.vis_dict[cand]['mAP'], self.vis_dict[cand]['mAP_50'], self.vis_dict[cand]['mAP_75'], cand))
                # else:
                logging.info('No.{} mAP = {}, {}M, {} GFLOPS ,arch={} '.format(
                    i + 1, self.vis_dict[cand]['mAP'], self.vis_dict[cand]['size'], self.vis_dict[cand]['fp'], cand))
            seed(0)
            mutation = self.get_mutation(
                self.select_num, self.mutation_num, self.m_prob)
            time.sleep(2)

            crossover = self.get_crossover(self.select_num, self.crossover_num)

            self.candidates = mutation + crossover

            self.get_random(self.population_num)

            self.epoch += 1

        if rank == 0:
            print('result------')
        info = self.vis_dict
        cands = sorted([cand for cand in info if 'mAP' in info[cand]],
                       key=lambda cand: info[cand]['mAP'])

        # opt = cands[-1][:(self.nr_layer-1)]
        # opt = cands[-1][:cands[-1][-1]]
        opt = cands[-1]

        # opt = [self.primitives[i] for i in opt]
        if rank == 0:
            print(opt)
            logging.info('result----- {}'.format(opt))
            logging.info('result----- {}'.format(cands[0]))


def main():
    args = parse_args()

    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')
    
    searcher = EvolutionSearcher(args)
    searcher.search()

if __name__ == '__main__':
    main()
