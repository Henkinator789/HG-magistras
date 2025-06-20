auto_scale_lr = dict(base_batch_size=128, enable=False)
backend_args = None
batch_augments = [
    dict(size=(
        896,
        896,
    ), type='BatchFixedSizePad'),
]
custom_hooks = [
    dict(
        ema_type='ExpMomentumEMA',
        momentum=0.0002,
        priority=49,
        type='EMAHook',
        update_buffers=True),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'projects.EfficientDet.efficientdet',
    ])
data_root = 'D:/Repositories/test/data/larch_casebearer/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        _scope_='mmdet',
        interval=0,
        save_best='coco/bbox_mAP',
        type='CheckpointHook'),
    logger=dict(_scope_='mmdet', interval=50, type='LoggerHook'),
    param_scheduler=dict(_scope_='mmdet', type='ParamSchedulerHook'),
    sampler_seed=dict(_scope_='mmdet', type='DistSamplerSeedHook'),
    timer=dict(_scope_='mmdet', type='IterTimerHook'),
    visualization=dict(_scope_='mmdet', type='DetVisualizationHook'))
default_scope = 'mmdet'
env_cfg = dict(
    cudnn_benchmark=True,
    dist_cfg=dict(backend='nccl'),
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0))
evalute_type = 'CocoMetric'
image_size = 896
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(
    _scope_='mmdet', by_epoch=True, type='LogProcessor', window_size=50)
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
        arch='b3',
        conv_cfg=dict(type='Conv2dSamePadding'),
        drop_path_rate=0.3,
        frozen_stages=0,
        init_cfg=dict(
            checkpoint=
            'D:/Repositories/test/25-05-09/efficientdet/weights/efficientnet-b3_3rdparty_8xb32-aa-advprop_in1k_20220119-53b41118.pth',
            prefix='backbone',
            type='Pretrained'),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        norm_eval=False,
        out_indices=(
            3,
            4,
            5,
        ),
        type='EfficientNet'),
    bbox_head=dict(
        anchor_generator=dict(
            center_offset=0.5,
            octave_base_scale=4,
            ratios=[
                1.0,
                0.5,
                2.0,
            ],
            scales_per_octave=3,
            strides=[
                8,
                16,
                32,
                64,
                128,
            ],
            type='AnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            type='DeltaXYWHBBoxCoder'),
        feat_channels=160,
        in_channels=160,
        loss_bbox=dict(beta=0.1, loss_weight=50, type='HuberLoss'),
        loss_cls=dict(
            alpha=0.25,
            gamma=1.5,
            loss_weight=1.0,
            type='FocalLoss',
            use_sigmoid=True),
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_classes=4,
        num_ins=5,
        stacked_convs=4,
        type='EfficientDetSepBNHead'),
    data_preprocessor=dict(
        batch_augments=[
            dict(size=(
                896,
                896,
            ), type='BatchFixedSizePad'),
        ],
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=896,
        std=[
            58.395,
            57.12,
            57.375,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=[
            48,
            136,
            384,
        ],
        norm_cfg=dict(
            eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN'),
        num_stages=6,
        out_channels=160,
        start_level=0,
        type='BiFPN'),
    test_cfg=dict(
        max_per_img=100,
        min_bbox_size=0,
        nms=dict(
            iou_threshold=0.3,
            method='gaussian',
            min_score=0.001,
            sigma=0.5,
            type='soft_nms'),
        nms_pre=1000,
        score_thr=0.05),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            ignore_iof_thr=-1,
            min_pos_iou=0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler')),
    type='EfficientDet')
norm_cfg = dict(eps=0.001, momentum=0.01, requires_grad=True, type='SyncBN')
optim_wrapper = dict(
    _scope_='mmdet',
    clip_grad=dict(max_norm=10, norm_type=2),
    optimizer=dict(lr=0.16, momentum=0.9, type='SGD', weight_decay=4e-05),
    paramwise_cfg=dict(
        bias_decay_mult=0, bypass_duplicate=True, norm_decay_mult=0),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=False, end=917, start_factor=0.1, type='LinearLR'),
    dict(
        T_max=299,
        begin=1,
        by_epoch=True,
        convert_to_iter_based=True,
        end=300,
        eta_min=0.0,
        type='CosineAnnealingLR'),
]
resume = False
test_cfg = dict(_scope_='mmdet', type='TestLoop')
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
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
                896,
                896,
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
test_evaluator = dict(
    _scope_='mmdet',
    ann_file='D:/Repositories/test/data/larch_casebearer/test/annotations.json',
    backend_args=None,
    classwise=True,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
test_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(keep_ratio=True, scale=(
        896,
        896,
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
    _scope_='mmdet', max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=dict(_scope_='mmdet', type='AspectRatioBatchSampler'),
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
        dataset=dict(
            ann_file='train/annotations.json',
            backend_args=None,
            data_prefix=dict(img='train/'),
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
                dict(type='LoadAnnotations', with_bbox=True),
                dict(keep_ratio=True, scale=(
                    896,
                    896,
                ), type='Resize'),
                dict(type='RandomAffine'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        oversample_thr=0.25,
        type='ClassBalancedDataset'),
    num_workers=1,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=True, scale=(
        896,
        896,
    ), type='Resize'),
    dict(type='RandomAffine'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(_scope_='mmdet', type='ValLoop')
val_dataloader = dict(
    batch_size=1,
    dataset=dict(
        _scope_='mmdet',
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
                896,
                896,
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
    num_workers=2,
    persistent_workers=True,
    sampler=dict(_scope_='mmdet', shuffle=False, type='DefaultSampler'))
val_evaluator = dict(
    _scope_='mmdet',
    ann_file='D:/Repositories/test/data/larch_casebearer/val/annotations.json',
    backend_args=None,
    format_only=False,
    metric='bbox',
    type='CocoMetric')
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='TensorboardVisBackend'),
]
visualizer = dict(
    _scope_='mmdet',
    name='visualizer',
    type='DetLocalVisualizer',
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='TensorboardVisBackend'),
    ])
work_dir = '25-05-09/efficientdet'
