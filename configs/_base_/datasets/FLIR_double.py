# dataset settings
dataset_type = 'FLIRDataset'
data_root = 'data/align/'
img_norm_cfg = dict(
    mean=[159.881, 162.221, 160.283], std=[56.969, 59.579, 63.119], to_rgb=True,
    thermal_mean=[136.63746562356317, 136.63746562356317, 136.63746562356317], 
    thermal_std=[64.97730349740888, 64.97730349740888, 64.97730349740888],
    )

# [159.881, 162.221, 160.283]
# [56.969, 59.579, 63.119]

# [136.63746562356317, 136.63746562356317, 136.63746562356317]                                                                                                                                                     │···················································
# [64.97730349740888, 64.97730349740888, 64.97730349740888]

# CLASSES = ('bicycle',  'car', 'person')

train_pipeline = [
    dict(type='LoadImageFromFile', load_thermal=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'img_thermal','gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', load_thermal=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_thermal']),
            dict(type='Collect', keys=['img', 'img_thermal']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root + 'FLIR_train.json',
            img_prefix=data_root,
            pipeline=train_pipeline,
            # classes=CLASSES
            )
            ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'FLIR_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        # classes=CLASSES
        ),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'FLIR_val.json',
        img_prefix=data_root,
        pipeline=test_pipeline,
        # classes=CLASSES
        ))
evaluation = dict(interval=1, metric='bbox')
