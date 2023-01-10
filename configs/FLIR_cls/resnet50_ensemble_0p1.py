_base_ = [
    '../_base_/models/r50_double.py', '../_base_/datasets/FLIR_double_cls.py',
    '../_base_/default_runtime.py'
]

# optimizer
optimizer = dict(type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0001)
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
