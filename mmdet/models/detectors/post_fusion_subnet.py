# Copyright (c) OpenMMLab. All rights reserved.
from multiprocessing.sharedctypes import Value
import warnings

import torch

from ..builder import DETECTORS
from .faster_rcnn import FasterRCNN
from ..builder import DETECTORS, build_backbone, build_head, build_neck

@DETECTORS.register_module()
class PostFusionSubnet(FasterRCNN):

    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(PostFusionSubnet, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)

        # self.backbone = build_backbone(backbone)
        self.last_arch = None
        self.backbone_thermal = build_backbone(backbone)

        if neck is not None:
            # self.neck = build_neck(neck)
            self.neck_thermal = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            # self.rpn_head = build_head(rpn_head_)
            self.rpn_head_thermal = build_head(rpn_head_)


    def extract_feat(self, img, img_thermal, arch=None):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img, arch)
        x_thermal = self.backbone_thermal(img_thermal, arch)
        if self.with_neck:
            x = self.neck(x)
            x_thermal = self.neck_thermal(x_thermal)
        return x, x_thermal

    def forward_train(self,
                      img,
                      img_metas,
                      img_thermal,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
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
        # img_thermal = img_thermal[0]
        if arch is None:
            arch = self.last_arch
        self.last_arch = arch
        x, x_thermal = self.extract_feat(img, img_thermal, arch=arch)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)

            rpn_losses_thermal, proposal_list_thermal = self.rpn_head_thermal.forward_train(
                x_thermal,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            # losses.update(rpn_losses_thermal)
            rpn_losses_thermal_tmp = dict()
            for name, value in rpn_losses_thermal.items():
                rpn_losses_thermal_tmp[f'{name}_thermal'] = value
            losses.update(rpn_losses_thermal_tmp)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(x, x_thermal, img_metas, 
                                                 proposal_list,
                                                 proposal_list_thermal,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 arch=arch,
                                                 **kwargs)
        losses.update(roi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                proposals_thermal=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x, x_thermal = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
            proposal_list_thermal = await self.rpn_head_thermal.async_simple_test_rpn(
                x_thermal, img_meta)
        else:
            proposal_list = proposals
            proposal_list_thermal = proposals_thermal
        
        return await self.roi_head.async_simple_test(x, x_thermal, proposal_list, proposal_list_thermal, img_meta, rescale=rescale)
        # out_thermal = await self.roi_head.async_simple_test(x_thermal, proposal_list_thermal, img_meta, rescale=rescale)

        # return out,

    def simple_test(self, img, img_metas, img_thermal, proposals=None, proposals_thermal=None, rescale=False, arch=None):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        assert proposals is None
        if arch is None:
            arch = self.last_arch
        self.last_arch = arch
        # print(img_thermal)
        # print(img_thermal.__class__)
        img_thermal = img_thermal[0]
        x, x_thermal = self.extract_feat(img, img_thermal, arch)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            proposal_list_thermal = self.rpn_head_thermal.simple_test_rpn(x_thermal, img_metas)
        else:
            proposal_list = proposals
            proposal_list_thermal = proposals_thermal

        return self.roi_head.simple_test(x, x_thermal, proposal_list, proposal_list_thermal, img_metas, rescale=rescale, arch=arch)
    
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
