# voc2012.py
dataset_type = 'VOCDataset'
data_root = 'data/VOCdevkit/'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1000, 600), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2012/ImageSets/Main/trainval.txt',
        data_prefix=dict(sub_data_root='VOC2012/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32, bbox_min_size=32),
        pipeline=train_pipeline)
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='VOC2012/ImageSets/Main/val.txt',   # 你也可以用 test.txt
        data_prefix=dict(sub_data_root='VOC2012/'),
        test_mode=True,
        pipeline=test_pipeline)
)

test_dataloader = val_dataloader

val_evaluator = dict(type='VOCMetric', metric='mAP', eval_mode='area')  # VOC2012 推荐用area模式
test_evaluator = val_evaluator

