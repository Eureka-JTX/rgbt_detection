_base_ = [
    '../_base_/models/r50_double.py', '../_base_/datasets/FLIR_double_cls.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='TwoStreamCls',
    backbone_rgb=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        init_cfg=dict(type='Pretrained', checkpoint='/data0/linzhiwei/pretrain_ckpt/resnet50-0676ba61.pth')
        ),
    head=dict(
        type='LinearResHead',
        in_channels=2048,
        num_classes=3,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
    head_thermal=dict(
        type='LinearResHead',
        in_channels=2048,
        num_classes=3,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
    train_cfg=None,
    test_cfg=None
    )

# optimizer
optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[6, 8])
# runtime settings
runner = dict(
    type='EpochBasedRunner', max_epochs=10)  # actual epoch = 4 * 3 = 12

# load_from = 'work_dirs/faster_rcnn_r50_fpn_1x_FLIR_double.pth'
# find_unused_parameters=True
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

auto_scale_lr = dict(enable=False, base_batch_size=16)
