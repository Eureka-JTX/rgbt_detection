_base_ = [
    '../_base_/models/detnas_double.py', '../_base_/datasets/FLIR_double_cls.py',
    '../_base_/default_runtime.py'
]
model = dict(
    type='TwoStreamClsSearch',
    head=dict(
        type='ConvFCBBoxHeadSearch',
            in_channels=640,
            conv_out_channels=640,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=3,
            reg_class_agnostic=False,
            with_reg=False,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1),
            ),
)

optimizer = dict(type='SGD', lr=0.0005, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
# actual epoch = 3 * 3 = 9
lr_config = dict(policy='step', step=[12, 16])
# runtime settings
runner = dict(
    type='EpochBasedRunnerSearch', max_epochs=20)  # actual epoch = 4 * 3 = 12

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
find_unused_parameters=True

fix_arch = False

arch = dict(
    backbone_rgb=[
        [0,1,2,3] for i in range(sum([4, 4, 8, 4]))
    ],
    backbone_thermal=[
        [0,1,2,3] for i in range(sum([4, 4, 8, 4]))
    ],
    head_rgb=[
        [0,1,2],
        [0,1,2,3],
        # [0,1],
        [0],
        # [1,2]
        [1]
    ],
    head_thermal=[
        [0,1,2],
        [0,1,2,3],
        # [0,1],
        [0],
        # [1,2]
        [1],
    ]
)
