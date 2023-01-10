_base_ = [
    '../_base_/models/faster_rcnn_r50_fpn.py', '../_base_/datasets/FLIR_vis.py',
    '../_base_/default_runtime.py'
]
model = dict(
    # init_cfg=dict(type='Pretrained', checkpoint='/data2/linzhiwei/download/faster_rcnn_r50_fpn_1x_coco.pth'),
    init_cfg=dict(type='Pretrained', checkpoint='/data2/linzhiwei/download/faster_rcnn_r50_fpn_mstrain_3x_coco.pth'),
    # frozen_stages=1,
    backbone=dict(
        frozen_stages=-1,
    ),
    roi_head=dict(bbox_head=dict(num_classes=3)))
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[3])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=4)  # actual epoch = 4 * 3 = 12
