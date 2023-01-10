# dataset settings
dataset_type = 'ClsDataset'
data_root = 'data/align/'
img_norm_cfg = dict(
    # mean=[159.881, 162.221, 160.283], std=[56.969, 59.579, 63.119], to_rgb=True,
    # thermal_mean=[136.63746562356317, 136.63746562356317, 136.63746562356317],
    # thermal_std=[64.97730349740888, 64.97730349740888, 64.97730349740888],
    mean=[151.61745, 152.50763, 152.57478], std=[39.68672, 40.339878, 40.068474], to_rgb=True,
    thermal_mean=[144.44142, 144.44142, 144.44142],
    thermal_std=[41.214504, 41.214504, 41.214504],
)

# [159.881, 162.221, 160.283]
# [56.969, 59.579, 63.119]

# [136.63746562356317, 136.63746562356317, 136.63746562356317]                                                                                                                                                     │···················································
# [64.97730349740888, 64.97730349740888, 64.97730349740888]

# CLASSES = ('bicycle',  'car', 'person')

train_pipeline = [
    dict(type='LoadImageFromFile', load_thermal=True),
    # dict(type='LoadAnnotations'),
    # dict(type='Resize', img_scale=(224, 224), keep_ratio=True),
    dict(type='Resize', img_scale=(256, 256), keep_ratio=False),
    dict(type='RandomResizedCrop', size=224),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_labels', 'img_thermal']),
]
test_pipeline = [
    dict(type='LoadImageFromFile', load_thermal=True),
    # dict(
    #     type='MultiScaleFlipAug',
    #     img_scale=(224, 224),
    #     flip=False,
    #     transforms=[
    #         dict(type='Resize', keep_ratio=True),
    #         dict(type='RandomFlip'),
    #         dict(type='Normalize', **img_norm_cfg),
    #         dict(type='Pad', size_divisor=32),
    #         dict(type='ImageToTensor', keys=['img', 'img_thermal']),
    #         dict(type='Collect', keys=['img', 'img_thermal']),
    #     ])
    dict(
        type='MultiScaleFlipAug',
        img_scale=(224, 224),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img', 'img_thermal']),
            dict(type='Collect', keys=['img', 'img_thermal']),
        ])
    # dict(type='Resize', img_scale=(224, 224), keep_ratio=False),
    # dict(type='Normalize', **img_norm_cfg),
    # dict(type='Pad', size_divisor=32),
    # dict(type='ImageToTensor', keys=['img', 'img_thermal']),
    # dict(type='Collect', keys=['img', 'img_thermal']),

]
data = dict(
    samples_per_gpu=32,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,
        dataset=dict(
            type=dataset_type,
            # ann_file=data_root + 'FLIR_train.json',
            data_prefix='FLIR_align_crop/night/train/rgb',
            # img_prefix=data_root,
            pipeline=train_pipeline,
            # classes=CLASSES
        )
    ),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'FLIR_val.json',
        # img_prefix=data_root,
        data_prefix='FLIR_align_crop/night/val/rgb',
        pipeline=test_pipeline,
        # classes=CLASSES
    ),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'FLIR_val.json',
        # img_prefix=data_root,
        data_prefix='FLIR_align_crop/night/val/rgb',
        pipeline=test_pipeline,
        # classes=CLASSES
    ))

evaluation = dict(interval=1, metric='accuracy')
