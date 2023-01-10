# model settings
model = dict(
    type='TwoStreamCls',
    backbone_rgb=dict(
        type='ShuffleNetV2DetNAS',
        out_indices=(0,1,2,3),
        stage_repeats = [4, 4, 8, 4],
        stage_out_channels = [-1, 16, 64, 160, 320, 640],
        init_cfg=dict(type='Pretrained', checkpoint='./pretrain_ckpt/COCO_FPN_300M_supernet.pkl')
        ),
    backbone_thermal=dict(
        type='ShuffleNetV2DetNAS',
        out_indices=(0,1,2,3),
        stage_repeats = [4, 4, 8, 4],
        stage_out_channels = [-1, 16, 64, 160, 320, 640],
        init_cfg=dict(type='Pretrained', checkpoint='./pretrain_ckpt/COCO_FPN_300M_supernet.pkl')
        ),
    train_cfg=None,
    test_cfg=None
)
    
