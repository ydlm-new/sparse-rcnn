_base_ = [
    '../_base_/models/sparse_rcnn_r50_fpn.py',
    '../_base_/default_runtime.py'
]

# 模型结构部分（自定义 VOC 类别数）
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                _delete_=True,
                type='DIIHead',
                num_classes=20,
                num_ffn_fcs=2,
                num_heads=8,
                feedforward_channels=2048,
                hidden_channels=256,
                dropout=0.0,
                ffn_act_cfg=dict(type='ReLU', inplace=True),
                loss_cls=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=2.0),
                loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                loss_iou=dict(type='GIoULoss', loss_weight=2.0)
            )
        ] * 6
    )
)

# VOC 类别
classes = (
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
    'dog', 'horse', 'motorbike', 'person', 'pottedplant',
    'sheep', 'sofa', 'train', 'tvmonitor'
)

# 图像预处理
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

# 训练 pipeline
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='PackDetInputs')
]

# 测试 pipeline
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='Resize', scale=(1000, 600), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='PackDetInputs')
        ]
    )
]

# dataloader 设置（替代原来的 data=...）
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOCdevkit/',
        ann_file='VOC2012/ImageSets/Main/train.txt',
        data_prefix=dict(sub_data_root='VOC2012/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        classes=classes
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='VOCDataset',
        data_root='data/VOCdevkit/',
        ann_file='VOC2012/ImageSets/Main/val.txt',
        data_prefix=dict(sub_data_root='VOC2012/'),
        test_mode=True,
        pipeline=test_pipeline,
        classes=classes
    )
)

test_dataloader = val_dataloader

# evaluator（替代 evaluation）
val_evaluator = dict(type='VOCMetric', metric='mAP')
test_evaluator = val_evaluator

# 优化器（替代 optimizer 和 optimizer_config）
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='SGD', lr=0.0025, momentum=0.9, weight_decay=0.0001)
)

# 学习率调度（替代 lr_config）
param_scheduler = [
    dict(type='MultiStepLR', begin=0, end=12, by_epoch=True, milestones=[8, 11], gamma=0.1)
]

# 训练控制器（替代 runner）
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)
val_cfg = dict()
test_cfg = dict()
