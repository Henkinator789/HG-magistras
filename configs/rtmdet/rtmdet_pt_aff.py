auto_scale_lr = dict(base_batch_size=16, enable=False)
backend_args = None
base_lr = 0.004
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
    dict(
        switch_epoch=280,
        switch_pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='RandomAffine'),
            dict(type='PackDetInputs'),
        ],
        type='PipelineSwitchHook'),
]
data_root = 'D:/Repositories/test/data/larch_casebearer/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=0,
        max_keep_ckpts=3,
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(interval=50, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=False,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
img_scales = [
    (
        640,
        640,
    ),
    (
        320,
        320,
    ),
    (
        960,
        960,
    ),
]
interval = 10
launcher = 'none'
load_from = 'D:/Repositories/test/25-05-09/rtmdet/weigths/rtmdet_m_8xb32-300e_coco_20220719_112220-229f527c.pth'
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
max_epochs = 300
metainfo = dict(
    classes=(
        'H',
        'LD',
        'HD',
        'other',
    ),
    palette=[
        (
            0,
            255,
            0,
        ),
        (
            255,
            128,
            0,
        ),
        (
            255,
            0,
            0,
        ),
        (
            0,
            0,
            255,
        ),
    ])
model = dict(
    backbone=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        arch='P5',
        channel_attention=True,
        deepen_factor=0.67,
        expand_ratio=0.5,
        norm_cfg=dict(type='SyncBN'),
        type='CSPNeXt',
        widen_factor=0.75),
    bbox_head=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        anchor_generator=dict(
            offset=0, strides=[
                8,
                16,
                32,
            ], type='MlvlPointGenerator'),
        bbox_coder=dict(type='DistancePointBBoxCoder'),
        exp_on_reg=True,
        feat_channels=192,
        in_channels=192,
        loss_bbox=dict(loss_weight=2.0, type='GIoULoss'),
        loss_cls=dict(
            beta=2.0,
            loss_weight=1.0,
            type='QualityFocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(type='SyncBN'),
        num_classes=4,
        pred_kernel_size=1,
        share_conv=True,
        stacked_convs=2,
        type='RTMDetSepBNHead',
        with_objectness=False),
    data_preprocessor=dict(
        batch_augments=None,
        bgr_to_rgb=False,
        mean=[
            103.53,
            116.28,
            123.675,
        ],
        std=[
            57.375,
            57.12,
            58.395,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        act_cfg=dict(inplace=True, type='SiLU'),
        expand_ratio=0.5,
        in_channels=[
            192,
            384,
            768,
        ],
        norm_cfg=dict(type='SyncBN'),
        num_csp_blocks=2,
        out_channels=192,
        type='CSPNeXtPAFPN'),
    test_cfg=dict(
        max_per_img=300,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.65, type='nms'),
        nms_pre=30000,
        score_thr=0.001),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(topk=13, type='DynamicSoftLabelAssigner'),
        debug=False,
        pos_weight=-1),
    type='RTMDet')
optim_wrapper = dict(
    optimizer=dict(lr=0.004, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=1000, start_factor=1e-05,
        type='LinearLR'),
    dict(
        T_max=150,
        begin=150,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0002,
        type='CosineAnnealingLR'),
]
resume = False
stage2_num_epochs = 20
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file='test/annotations.json',
        backend_args=None,
        data_prefix=dict(img='test/'),
        data_root='D:/Repositories/test/data/larch_casebearer/',
        metainfo=dict(
            classes=(
                'H',
                'LD',
                'HD',
                'other',
            ),
            palette=[
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    128,
                    0,
                ),
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    ann_file='D:/Repositories/test/data/larch_casebearer/test/annotations.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        meta_keys=(
            'img_id',
            'img_path',
            'ori_shape',
            'img_shape',
            'scale_factor',
        ),
        type='PackDetInputs'),
]
train_cfg = dict(
    dynamic_intervals=[
        (
            280,
            1,
        ),
    ],
    max_epochs=30,
    type='EpochBasedTrainLoop',
    val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=4,
    dataset=dict(
        ann_file='train/annotations.json',
        backend_args=None,
        data_prefix=dict(img='train/'),
        data_root='D:/Repositories/test/data/larch_casebearer/',
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        metainfo=dict(
            classes=(
                'H',
                'LD',
                'HD',
                'other',
            ),
            palette=[
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    128,
                    0,
                ),
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='RandomAffine'),
            dict(type='PackDetInputs'),
        ],
        type='CocoDataset'),
    num_workers=1,
    persistent_workers=True,
    pin_memory=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='RandomAffine'),
    dict(type='PackDetInputs'),
]
train_pipeline_stage2 = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        640,
        640,
    ), type='Resize'),
    dict(type='RandomAffine'),
    dict(type='PackDetInputs'),
]
tta_model = dict(
    tta_cfg=dict(max_per_img=100, nms=dict(iou_threshold=0.6, type='nms')),
    type='DetTTAModel')
tta_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(
        transforms=[
            [
                dict(keep_ratio=True, scale=(
                    640,
                    640,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    320,
                    320,
                ), type='Resize'),
                dict(keep_ratio=True, scale=(
                    960,
                    960,
                ), type='Resize'),
            ],
            [
                dict(prob=1.0, type='RandomFlip'),
                dict(prob=0.0, type='RandomFlip'),
            ],
            [
                dict(
                    pad_val=dict(img=(
                        114,
                        114,
                        114,
                    )),
                    size=(
                        960,
                        960,
                    ),
                    type='Pad'),
            ],
            [
                dict(type='LoadAnnotations', with_bbox=True),
            ],
            [
                dict(
                    meta_keys=(
                        'img_id',
                        'img_path',
                        'ori_shape',
                        'img_shape',
                        'scale_factor',
                        'flip',
                        'flip_direction',
                    ),
                    type='PackDetInputs'),
            ],
        ],
        type='TestTimeAug'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=5,
    dataset=dict(
        ann_file='val/annotations.json',
        backend_args=None,
        data_prefix=dict(img='val/'),
        data_root='D:/Repositories/test/data/larch_casebearer/',
        metainfo=dict(
            classes=(
                'H',
                'LD',
                'HD',
                'other',
            ),
            palette=[
                (
                    0,
                    255,
                    0,
                ),
                (
                    255,
                    128,
                    0,
                ),
                (
                    255,
                    0,
                    0,
                ),
                (
                    0,
                    0,
                    255,
                ),
            ]),
        pipeline=[
            dict(backend_args=None, type='LoadImageFromFile'),
            dict(keep_ratio=True, scale=(
                640,
                640,
            ), type='Resize'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                meta_keys=(
                    'img_id',
                    'img_path',
                    'ori_shape',
                    'img_shape',
                    'scale_factor',
                ),
                type='PackDetInputs'),
        ],
        test_mode=True,
        type='CocoDataset'),
    drop_last=False,
    num_workers=10,
    persistent_workers=True,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    ann_file='D:/Repositories/test/data/larch_casebearer/val/annotations.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
]
visualizer = dict(
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
    ])
work_dir = '25-05-09/rtmdet'
