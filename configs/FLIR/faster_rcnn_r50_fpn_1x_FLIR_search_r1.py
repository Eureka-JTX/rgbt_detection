_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/FLIR_double.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='PostFusionSearch',
    # init_cfg=dict(type='Pretrained', checkpoint='/data2/linzhiwei/download/faster_rcnn_r50_fpn_1x_coco.pth'),
    # init_cfg=dict(type='Pretrained', checkpoint='/data2/linzhiwei/download/faster_rcnn_r50_fpn_mstrain_3x_coco_double.pth'),
    backbone=dict(
        frozen_stages=-1,
    ),
    init_cfg=dict(type='Pretrained', checkpoint='work_dirs/faster_rcnn_r50_fpn_1x_FLIR_double_coco_pretrain_shared.pth'),
    roi_head=dict(
        type='PostFusionClassUnsharedRoIHeadSearch',
        bbox_head=dict(num_classes=3,
                loss_cls=dict(
                loss_weight=0.5),),
        bbox_roi_extractor_fusion=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256*2,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head_fusion=dict(
            type='ConvFCBBoxHeadSearch',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0., 0., 0., 0.],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            loss_bbox=dict(type='L1Loss', loss_weight=0.0)),
        # bbox_head=dict(
        #     loss_cls=dict(
        #         loss_weight=0.5),
        # )
        ),
    train_cfg=dict(
        rcnn=dict(
            score_thr_fusion=0.05,
            nms_fusion=dict(type='nms', iou_threshold=0.5),
            max_per_img_fusion=512
        )
        ),
    # test_cfg=dict(
    #     rcnn=dict(
    #         score_thr=0.05,
    #         nms=dict(type='nms', iou_threshold=0.5),
    #         max_per_img=100)
    
    )
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunnerSearch', max_epochs=4)  # actual epoch = 4 * 3 = 12

# load_from = 'work_dirs/faster_rcnn_r50_fpn_1x_FLIR_double.pth'
find_unused_parameters=True

arch = [[2, 3, 1, 1], [1, 0, 0, 1]]
fix_arch = True