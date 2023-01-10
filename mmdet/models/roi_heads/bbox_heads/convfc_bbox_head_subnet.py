# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule

from mmdet.models.builder import HEADS
from mmdet.models.utils import build_linear_layer
from .bbox_head import BBoxHead
from mmcv.cnn import constant_init
from mmcv.runner import BaseModule, auto_fp16, force_fp32
import copy
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


PRIMITIVES = [
    # 'Identity',         # 0
    'concat_fusion',  # 4
    'conv_fusion',         # 1
    'res_fusion',        # 2
    'add_fusion',  # 3
]

OPS = {
    # 'Identity': lambda in_channels, flatten: Identity(),
    'concat_fusion': lambda in_channels, flatten: concat_fusion(in_channels, flatten),
    'conv_fusion': lambda in_channels, flatten: conv_fusion(in_channels, flatten),
    'res_fusion': lambda in_channels, flatten: res_fusion(),
    'add_fusion': lambda in_channels, flatten:   add_fusion(),
}

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
    def forward(self, x):
        return x

class conv_fusion(nn.Module):
    def __init__(self, in_channels, flatten=False):
        super(conv_fusion, self).__init__()
        self.in_channels = in_channels
        if flatten:
            self.res_connect = nn.Linear(self.in_channels, self.in_channels, bias=True)
        else:
            self.res_connect = nn.Conv2d(self.in_channels, self.in_channels, 1, bias=True)

        constant_init(self.res_connect, 0)

    def forward(self, x1, x2):
        x = x1 + self.res_connect(x2)

        return x

class res_fusion(nn.Module):
    def __init__(self):
        super(res_fusion, self).__init__()
        self.alpha = nn.Parameter(torch.zeros(1))

    def forward(self, x1, x2):
        x = (1 - self.alpha.sigmoid()) * x1 + self.alpha.sigmoid() * x2

        return x

class add_fusion(nn.Module):
    def __init__(self):
        super(add_fusion, self).__init__()
    def forward(self, x1, x2):
        x = x1 + x2

        return x

class concat_fusion(nn.Module):
    def __init__(self, in_channels, flatten=False):
        super(concat_fusion, self).__init__()
        self.in_channels = in_channels
        if flatten:
            self.conv = nn.Linear(self.in_channels * 2, self.in_channels, bias=True)
        else:
            self.conv = nn.Conv2d(self.in_channels * 2, self.in_channels, 1, bias=True)

        # constant_init(self.res_connect, 0)

    def forward(self, x1, x2):
        x = self.conv(torch.cat([x1, x2], dim=1))

        return x

@HEADS.register_module()
class ConvFCBBoxHeadSubnet(BBoxHead):
    r"""More general bbox head, with shared conv and fc layers and two optional
    separated branches.

    .. code-block:: none

                                    /-> cls convs -> cls fcs -> cls
        shared convs -> shared fcs
                                    \-> reg convs -> reg fcs -> reg
    """  # noqa: W605

    def __init__(self,
                 arch,
                 num_shared_convs=0,
                 num_shared_fcs=2,
                 num_cls_convs=0,
                 num_cls_fcs=0,
                 num_reg_convs=0,
                 num_reg_fcs=0,
                 conv_out_channels=256,
                 fc_out_channels=1024,
                 conv_cfg=None,
                 norm_cfg=None,
                 init_cfg=None,
                 *args,
                 **kwargs):
        super(ConvFCBBoxHeadSubnet, self).__init__(
            *args, init_cfg=init_cfg, **kwargs)

        self.num_shared_convs = num_shared_convs
        self.num_shared_fcs = num_shared_fcs
        self.num_cls_convs = num_cls_convs
        self.num_cls_fcs = num_cls_fcs
        self.num_reg_convs = num_reg_convs
        self.num_reg_fcs = num_reg_fcs
        self.conv_out_channels = conv_out_channels
        self.fc_out_channels = fc_out_channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg

        self.fusion_stage = arch[0]
        self.fusion_type = arch[1]

        stage_1 = []
        stage_2 = []
        if self.fusion_stage == 0:
            block_1 = []
            for i in range(arch[2]):
                block_1.append(ConvModule(
                                self.in_channels,
                                self.conv_out_channels,
                                3,
                                padding=1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg))
            stage_1.append(nn.ModuleList(block_1))

            block_2 = []
            fc_in_channels = self.conv_out_channels * self.roi_feat_area
            for i in range(arch[3]):
                if i == 0:
                    in_channels = fc_in_channels
                else:
                    in_channels = self.fc_out_channels
                block_2.append(nn.Linear(in_channels, self.fc_out_channels))
            stage_2.append(nn.ModuleList(block_2))
        
        elif self.fusion_stage == 1:
            block_1 = []
            for i in range(arch[2]):
                block_1.append(ConvModule(
                                self.in_channels,
                                self.conv_out_channels,
                                3,
                                padding=1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg))
            stage_1.append(nn.ModuleList(block_1))
            stage_1.append(copy.deepcopy(nn.ModuleList(block_1)))

            block_2 = []
            fc_in_channels = self.conv_out_channels * self.roi_feat_area
            for i in range(arch[3]):
                if i == 0:
                    in_channels = fc_in_channels
                else:
                    in_channels = self.fc_out_channels
                block_2.append(nn.Linear(in_channels, self.fc_out_channels))
            stage_2.append(nn.ModuleList(block_2))
        
        else:
            assert self.fusion_stage == 2
            block_1 = []
            for i in range(arch[2]):
                block_1.append(ConvModule(
                                self.in_channels,
                                self.conv_out_channels,
                                3,
                                padding=1,
                                conv_cfg=self.conv_cfg,
                                norm_cfg=self.norm_cfg))
            stage_1.append(nn.ModuleList(block_1))
            stage_1.append(copy.deepcopy(nn.ModuleList(block_1)))

            block_2 = []
            fc_in_channels = self.conv_out_channels * self.roi_feat_area
            for i in range(arch[3]):
                if i == 0:
                    in_channels = fc_in_channels
                else:
                    in_channels = self.fc_out_channels
                block_2.append(nn.Linear(in_channels, self.fc_out_channels))
            stage_2.append(nn.ModuleList(block_2))
            stage_2.append(copy.deepcopy(nn.ModuleList(block_2)))
        
        self.stage_1 = nn.ModuleList(stage_1)
        self.stage_2 = nn.ModuleList(stage_2)  
        
        stage_fusion = []
        in_channels = [self.in_channels, self.conv_out_channels, self.fc_out_channels]

        flatten = True if self.fusion_stage==2 else False
        # for op in PRIMITIVES:
        #     block.append(OPS[op](in_channels[i], flatten))
        self.stage_fusion = OPS[PRIMITIVES[self.fusion_type]](in_channels[self.fusion_stage], flatten)

        self.relu = nn.ReLU(inplace=False)
        # reconstruct fc_cls and fc_reg since input channels are changed
        self.cls_last_dim = self.fc_out_channels
        self.reg_last_dim = self.fc_out_channels
        if self.with_cls:
            if self.custom_cls_channels:
                cls_channels = self.loss_cls.get_cls_channels(self.num_classes)
            else:
                cls_channels = self.num_classes + 1
            self.fc_cls = build_linear_layer(
                self.cls_predictor_cfg,
                in_features=self.cls_last_dim,
                out_features=cls_channels)
        if self.with_reg:
            out_dim_reg = (4 if self.reg_class_agnostic else 4 *
                           self.num_classes)
            self.fc_reg = build_linear_layer(
                self.reg_predictor_cfg,
                in_features=self.reg_last_dim,
                out_features=out_dim_reg)

        # if init_cfg is None:
        #     self.init_cfg += [
        #         dict(
        #             type='Xavier',
        #             distribution='uniform',
        #             override=[
        #                 dict(name='stage_x1'),
        #                 dict(name='stage_x2'),
        #                 dict(name='stage_after_fusion')
        #             ])
        #     ]

    def forward(self, x1, x2):
        # print(arch)
        # arch = [stage_for_fusion, fusion_type, depth_1, depth_2]
        if self.fusion_stage == 0:
            x = self.stage_fusion(x1, x2)
            for conv in self.stage_1[0]:
                x = self.relu(conv(x))
            x = x.flatten(1)
            for conv in self.stage_2[0]:
                x = self.relu(conv(x))
        
        if self.fusion_stage == 1:
            for conv in self.stage_1[0]:
                x1 = self.relu(conv(x1))
            for conv in self.stage_1[1]:
                x2 = self.relu(conv(x2))
            x = self.stage_fusion(x1, x2)
            x = x.flatten(1)
            for conv in self.stage_2[0]:
                x = self.relu(conv(x))
        
        if self.fusion_stage == 2:
            for conv in self.stage_1[0]:
                x1 = self.relu(conv(x1))
            for conv in self.stage_1[1]:
                x2 = self.relu(conv(x2))
            x1 = x1.flatten(1)
            x2 = x2.flatten(1)
            for conv in self.stage_2[0]:
                x1 = self.relu(conv(x1))
            for conv in self.stage_2[1]:
                x2 = self.relu(conv(x2))
            x = self.stage_fusion(x1, x2)

        # separate branches
        x_cls = x
        x_reg = x

        if x_cls.dim() > 2:
            if self.with_avg_pool:
                x_cls = self.avg_pool(x_cls)
            x_cls = x_cls.flatten(1)

        if x_reg.dim() > 2:
            if self.with_avg_pool:
                x_reg = self.avg_pool(x_reg)
            x_reg = x_reg.flatten(1)

        cls_score = self.fc_cls(x_cls) if self.with_cls else None
        bbox_pred = self.fc_reg(x_reg) if self.with_reg else None
        return cls_score, bbox_pred
    
    def forward_dummy(self, x, x_thermal):
        out, _ = self(x, x_thermal)
        # out_thermal, _ = self(x_thermal, x)
        return out
    
    def forward_train(self, x, x_thermal, labels):
        labels = torch.cat(labels)
        out, _ = self(x, x_thermal)
        # out_thermal, _ = self(x_thermal)

        loss = dict()
        # print(out.size(), labels)
        loss_1 = self.loss(out, labels)
        # loss_2 = self.loss(out_thermal, labels)
        # for 
        # loss_thermal = {}
        # for name, value in loss_2.items():
        #     loss_thermal[f'{name}_thermal'] = value
        
        loss.update(loss_1)
        # loss.update(loss_thermal)
        return loss

    
    def simple_test(self, x, x_thermal):
        out, _ = self(x, x_thermal)
        # out_thermal, _ = self(x_thermal)

        out = F.softmax(out, dim=1)
        # out_thermal = F.softmax(out_thermal, dim=1)

        # return (out + out_thermal) / 2
        return out
        # return out_thermal
        # return torch.max(out, out_thermal)
        

    @force_fp32(apply_to=('cls_score'))
    def loss(self,
             cls_score,
             labels,
            #  label_weights,
            #  thermal=False,
             reduction_override=None):
        losses = dict()
        # print(labels)
        if cls_score is not None:
            # avg_factor = max(torch.sum(label_weights > 0).float().item(), 1.)
            if cls_score.numel() > 0:
                loss_cls_ = self.loss_cls(
                    cls_score,
                    labels,
                    # label_weights,
                    # avg_factor=avg_factor,
                    reduction_override=reduction_override)
                if isinstance(loss_cls_, dict):
                    losses.update(loss_cls_)
                else:
                    losses['loss_cls'] = loss_cls_
                if self.custom_activation:
                    acc_ = self.loss_cls.get_accuracy(cls_score, labels)
                    losses.update(acc_)
                else:
                    losses['acc'] = accuracy(cls_score, labels)

        return losses


# @HEADS.register_module()
# class Shared2FCBBoxHead(ConvFCBBoxHead):

#     def __init__(self, fc_out_channels=1024, *args, **kwargs):
#         super(Shared2FCBBoxHead, self).__init__(
#             num_shared_convs=0,
#             num_shared_fcs=2,
#             num_cls_convs=0,
#             num_cls_fcs=0,
#             num_reg_convs=0,
#             num_reg_fcs=0,
#             fc_out_channels=fc_out_channels,
#             *args,
#             **kwargs)


