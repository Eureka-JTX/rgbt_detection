# model settings
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
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')
        # init_cfg=dict(type='Pretrained', checkpoint='/data0/linzhiwei/pretrain_ckpt/resnet50-0676ba61.pth')
        ),
    head=dict(
        type='LinearHead',
        in_channels=2048,
        num_classes=3,
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
        # loss_bbox=dict(type='L1Loss', loss_weight=1.0)
        ),
    train_cfg=None,
    test_cfg=None
    )

    # model training and testing settings
