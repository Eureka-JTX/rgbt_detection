# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn.modules.utils import _pair

from mmdet.core import build_bbox_coder, multi_apply, multiclass_nms
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy
from mmdet.models.utils import build_linear_layer
from mmcv.cnn import constant_init


@HEADS.register_module()
class LinearHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 in_channels=256,
                 num_classes=3,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 cls_predictor_cfg=dict(type='Linear'),
                 with_avg_pool=False,
                 thermal=False,
                 init_cfg=None):
        super(LinearHead, self).__init__(init_cfg)
        self.fp16_enabled = False

        self.thermal = thermal

        self.loss_cls = build_loss(loss_cls)
        self.with_avg_pool = with_avg_pool

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(7)
        else:
            in_channels *= 49

        self.fc_cls = build_linear_layer(
            # self.cls_predictor_cfg,
            cls_predictor_cfg,
            in_features=in_channels,
            out_features=num_classes)

        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            self.init_cfg += [
                dict(
                    type='Normal', std=0.01, override=dict(name='fc_cls'))
            ]
            

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x):
        # x = x[-1]
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        x = x.flatten(1)
        cls_score = self.fc_cls(x)
        return cls_score

    def forward_dummy(self, x, x_thermal):
        out = self(x)
        out_thermal = self(x_thermal)
        return out, out_thermal

    def forward_train(self, x, x_thermal, labels):
        # labels = labels[0]
        labels = torch.cat(labels)
        out = self(x)
        out_thermal = self(x_thermal)

        loss = dict()
        # print(out.size(), labels)
        loss_1 = self.loss(out, labels)
        loss_2 = self.loss(out_thermal, labels)
        # for 
        loss_thermal = {}
        for name, value in loss_2.items():
            loss_thermal[f'{name}_thermal'] = value
        
        loss.update(loss_1)
        loss.update(loss_thermal)
        return loss
    
    def simple_test(self, x, x_thermal):
        
        if self.thermal:
            out_thermal = self(x_thermal)
            out_thermal = F.softmax(out_thermal, dim=1)
            return out_thermal
        else:
            out = self(x)
            out = F.softmax(out, dim=1)
            return out

        # return (out + out_thermal) / 2
        # return out_thermal
        # return torch.max(out, out_thermal)
        # if self.thermal:
        #     return out_thermal
        # return out
        

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
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


@HEADS.register_module()
class LinearResHead(BaseModule):
    """Simplest RoI head, with only two fc layers for classification and
    regression respectively."""

    def __init__(self,
                 in_channels=256,
                 num_classes=3,
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=False,
                     loss_weight=1.0),
                 cls_predictor_cfg=dict(type='Linear'),
                 with_avg_pool=False,
                 init_cfg=None):
        super(LinearResHead, self).__init__(init_cfg)
        self.fp16_enabled = False

        self.loss_cls = build_loss(loss_cls)
        self.with_avg_pool = with_avg_pool

        self.res_connect = nn.Conv2d(in_channels, in_channels, 1, bias=True)
        constant_init(self.res_connect, 0)

        if self.with_avg_pool:
            self.avg_pool = nn.AvgPool2d(7)
        else:
            in_channels *= 49

        self.fc_cls = build_linear_layer(
            # self.cls_predictor_cfg,
            cls_predictor_cfg,
            in_features=in_channels,
            out_features=num_classes)

        self.debug_imgs = None
        if init_cfg is None:
            self.init_cfg = []
            self.init_cfg += [
                dict(
                    type='Normal', std=0.01, override=dict(name='fc_cls'))
            ]
            

    @property
    def custom_cls_channels(self):
        return getattr(self.loss_cls, 'custom_cls_channels', False)

    @property
    def custom_activation(self):
        return getattr(self.loss_cls, 'custom_activation', False)

    @property
    def custom_accuracy(self):
        return getattr(self.loss_cls, 'custom_accuracy', False)

    @auto_fp16()
    def forward(self, x, x_thermal):
        # x = x[-1]
        x = x + self.res_connect(x_thermal)
        if self.with_avg_pool:
            if x.numel() > 0:
                x = self.avg_pool(x)
                x = x.view(x.size(0), -1)
            else:
                # avg_pool does not support empty tensor,
                # so use torch.mean instead it
                x = torch.mean(x, dim=(-1, -2))
        x = x.flatten(1)
        cls_score = self.fc_cls(x)
        return cls_score

    def forward_dummy(self, x, x_thermal):
        # out = self(x)
        # out_thermal = self(x_thermal)
        # return out, out_thermal
        return self(x, x_thermal)

    def forward_train(self, x, x_thermal, labels):
        # labels = labels[0]
        labels = torch.cat(labels)
        out = self(x, x_thermal)
        # out_thermal = self(x_thermal)

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
        out = self(x, x_thermal)
        # out_thermal = self(x_thermal)

        out = F.softmax(out, dim=1)
        # out_thermal = F.softmax(out_thermal, dim=1)

        # return (out + out_thermal) / 2
        # return out_thermal
        # return torch.max(out, out_thermal)
        return out
        

    @force_fp32(apply_to=('cls_score', 'bbox_pred'))
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