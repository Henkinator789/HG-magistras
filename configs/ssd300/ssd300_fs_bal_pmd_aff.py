auto_scale_lr = dict(base_batch_size=64, enable=False)
backend_args = None
cudnn_benchmark = True
custom_hooks = [
    dict(type='NumClassCheckHook'),
    dict(interval=50, priority='VERY_LOW', type='CheckInvalidLossHook'),
]
data_root = 'D:/Repositories/test/data/larch_casebearer/'
dataset_type = 'CocoDataset'
default_hooks = dict(
    checkpoint=dict(
        interval=0, save_best='coco/bbox_mAP', type='CheckpointHook'),
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
input_size = 300
launcher = 'none'
load_from = None
log_level = 'INFO'
log_processor = dict(by_epoch=True, type='LogProcessor', window_size=50)
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
        ceil_mode=True,
        depth=16,
        init_cfg=dict(
            checkpoint='open-mmlab://vgg16_caffe', type='Pretrained'),
        out_feature_indices=(
            22,
            34,
        ),
        out_indices=(
            3,
            4,
        ),
        type='SSDVGG',
        with_last_pool=False),
    bbox_head=dict(
        anchor_generator=dict(
            basesize_ratio_range=(
                0.15,
                0.9,
            ),
            input_size=300,
            ratios=[
                [
                    2,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                    3,
                ],
                [
                    2,
                ],
                [
                    2,
                ],
            ],
            scale_major=False,
            strides=[
                8,
                16,
                32,
                64,
                100,
                300,
            ],
            type='SSDAnchorGenerator'),
        bbox_coder=dict(
            target_means=[
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            target_stds=[
                0.1,
                0.1,
                0.2,
                0.2,
            ],
            type='DeltaXYWHBBoxCoder'),
        in_channels=(
            512,
            1024,
            512,
            256,
            256,
            256,
        ),
        num_classes=4,
        type='SSDHead'),
    data_preprocessor=dict(
        bgr_to_rgb=True,
        mean=[
            123.675,
            116.28,
            103.53,
        ],
        pad_size_divisor=1,
        std=[
            1,
            1,
            1,
        ],
        type='DetDataPreprocessor'),
    neck=dict(
        in_channels=(
            512,
            1024,
        ),
        l2_norm_scale=20,
        level_paddings=(
            1,
            1,
            0,
            0,
        ),
        level_strides=(
            2,
            2,
            1,
            1,
        ),
        out_channels=(
            512,
            1024,
            512,
            256,
            256,
            256,
        ),
        type='SSDNeck'),
    test_cfg=dict(
        max_per_img=200,
        min_bbox_size=0,
        nms=dict(iou_threshold=0.45, type='nms'),
        nms_pre=1000,
        score_thr=0.02),
    train_cfg=dict(
        allowed_border=-1,
        assigner=dict(
            gt_max_assign_all=False,
            ignore_iof_thr=-1,
            min_pos_iou=0.0,
            neg_iou_thr=0.5,
            pos_iou_thr=0.5,
            type='MaxIoUAssigner'),
        debug=False,
        neg_pos_ratio=3,
        pos_weight=-1,
        sampler=dict(type='PseudoSampler'),
        smoothl1_beta=1.0),
    type='SingleStageDetector')
optim_wrapper = dict(
    optimizer=dict(lr=0.002, momentum=0.9, type='SGD', weight_decay=0.0005),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0, by_epoch=False, end=500, start_factor=0.001, type='LinearLR'),
    dict(
        begin=0,
        by_epoch=True,
        end=24,
        gamma=0.1,
        milestones=[
            16,
            22,
        ],
        type='MultiStepLR'),
]
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=8,
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
            dict(keep_ratio=False, scale=(
                300,
                300,
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
    dict(keep_ratio=False, scale=(
        300,
        300,
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
train_cfg = dict(max_epochs=30, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_sampler=None,
    batch_size=8,
    dataset=dict(
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
                dict(keep_ratio=False, scale=(
                    300,
                    300,
                ), type='Resize'),
                dict(type='PhotoMetricDistortion'),
                dict(type='RandomAffine'),
                dict(type='PackDetInputs'),
            ],
            type='CocoDataset'),
        oversample_thr=0.25,
        type='ClassBalancedDataset'),
    num_workers=2,
    persistent_workers=True,
    sampler=dict(shuffle=True, type='DefaultSampler'))
train_pipeline = [
    dict(backend_args=None, type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(keep_ratio=False, scale=(
        300,
        300,
    ), type='Resize'),
    dict(type='PhotoMetricDistortion'),
    dict(type='RandomAffine'),
    dict(type='PackDetInputs'),
]
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=8,
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
            dict(keep_ratio=False, scale=(
                300,
                300,
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
work_dir = '25-05-09/ssd300'
