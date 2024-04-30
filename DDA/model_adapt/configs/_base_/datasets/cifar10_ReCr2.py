# dataset settings
dataset_type = 'Cifar10_2' 
corruption = ['jpeg', 'noise', 'blur', 'weak_seq', 'strong_seq', ]
corruption = corruption[1]
severity = 0

data_prefix1 = 'dataset/cifar10' 
data_prefix2 = 'dataset/generated/cifar10'

# Normalization values from https://github.com/open-mmlab/mmpretrain/blob/master/configs/_base_/datasets/cifar10_bs16.py
img_norm_cfg = dict(
    mean=[125.307, 122.961, 113.8575], std=[51.5865, 50.847, 51.255], to_rgb=True)
    
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='RandomCrop', size=32, padding=4),
    dict(type='RandomFlip', flip_prob=0.5, direction='horizontal'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='ToTensor', keys=['gt_label']),
    dict(type='Collect', keys=['img', 'gt_label'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='ImageToTensor', keys=['img']),
    dict(type='Collect', keys=['img'])
]

data = dict(
    samples_per_gpu=32,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=train_pipeline,
        corruption=corruption,
        severity=severity),
    val=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity),
    test=dict(
        type=dataset_type,
        data_prefix=data_prefix1,
        data_prefix2=data_prefix2,
        pipeline=test_pipeline,
        corruption=corruption,
        severity=severity))
evaluation = dict(interval=1, metric='accuracy')