# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from copy import deepcopy


@DETECTORS.register_module()
class TwoStreamClsSearch(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone_rgb,
                 head,
                 backbone_thermal=None,
                #  head=None,
                 pretrained=None,
                 init_cfg=None,
                 train_cfg=None,
                 test_cfg=None):
        super(TwoStreamClsSearch, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone_rgb = build_backbone(backbone_rgb)

        if backbone_thermal is None:
            backbone_thermal = deepcopy(backbone_rgb)
        self.backbone_thermal = build_backbone(backbone_thermal)

        self.head = build_head(head)
        self.head_thermal = build_head(head)
        self.last_arch = None

    def extract_feat(self, img, img_thermal, arch):
        """Directly extract features from the backbone+neck."""
        x = self.backbone_rgb(img, arch['backbone_rgb'])
        x_thermal = self.backbone_thermal(img_thermal, arch['backbone_thermal'])
        # if self.with_neck:
        #     x = self.neck(x)
        return x[-1], x_thermal[-1]

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x, x_thermal = self.extract_feat(img, img)
        
        # outs = outs + (roi_outs, )
        # outs = self.head.forward_dummy(x, x_thermal)
        outs = self.head.forward_dummy(x)
        outs_thermal = self.head_thermal.forward_dummy(x_thermal)
        return outs, outs_thermal

    def forward_train(self,
                      img,
                      img_metas,
                      img_thermal,
                      gt_labels,
                      arch=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if arch is None:
            arch = self.last_arch
        self.last_arch = arch

        x, x_thermal = self.extract_feat(img, img_thermal, arch)

        losses = dict()

        # RPN forward and loss
        # head_losses = self.head.forward_train(x, x_thermal, gt_labels, arch)
        head_losses = self.head.forward_train(x, x_thermal, gt_labels, arch['head_rgb'])
        head_losses_thermal = self.head_thermal.forward_train(x_thermal, x, gt_labels, arch['head_thermal'])
        losses.update(head_losses)
        loss_thermal = {}
        for name, value in head_losses_thermal.items():
            loss_thermal[f'{name}_thermal'] = value
        losses.update(loss_thermal)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, img_thermal, proposals=None, rescale=False, arch=None):
        """Test without augmentation."""

        # assert self.with_bbox, 'Bbox head must be implemented.'
        # print(img_thermal.__class__)
        if arch is None:
            arch = self.last_arch
        self.last_arch = arch
        img_thermal = img_thermal[0]
        x, x_thermal = self.extract_feat(img, img_thermal, arch)

        outs = self.head.simple_test(x, x_thermal,arch['head_rgb'])
        outs_thermal = self.head_thermal.simple_test(x_thermal, x, arch['head_thermal'])

        return (outs + outs_thermal)/2

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )
    
    def train_step(self, data, optimizer, arch):
        data['arch'] = arch
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer=None, arch=None):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        data['arch'] = arch
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs
    
