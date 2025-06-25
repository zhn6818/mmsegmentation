# model settings
norm_cfg = dict(type='BN', requires_grad=True)

# dataset settings
dataset_type = 'ChaseDB1Dataset'
data_root = 'data/CHASE_DB1'
img_scale = (512, 512)

# 使用size_divisor确保图像尺寸是16的倍数（UNet下采样率）
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    # 使用size_divisor=16，确保图像尺寸能被UNet下采样率整除
    size_divisor=16,
    seg_pad_val=255)

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained=None,
    backbone=dict(
        type='UNet',
        in_channels=3,
        base_channels=64,
        num_stages=5,
        strides=(1, 1, 1, 1, 1),
        enc_num_convs=(2, 2, 2, 2, 2),
        dec_num_convs=(2, 2, 2, 2),
        downsamples=(True, True, True, True),  # 总下采样率=16，与size_divisor=16兼容
        enc_dilations=(1, 1, 1, 1, 1),
        dec_dilations=(1, 1, 1, 1),
        with_cp=False,
        conv_cfg=None,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU'),
        upsample_cfg=dict(type='InterpConv'),
        norm_eval=False),
    decode_head=dict(
        type='ASPPHead',
        in_channels=64,
        in_index=4,
        channels=16,
        dilations=(1, 12, 24, 36),
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=128,
        in_index=3,
        channels=64,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=2,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # 移除test_cfg中的crop_size和stride，使用完整图像推理
    test_cfg=dict(mode='whole'))

# 任务类型配置
task_type = 'segmentation'  # 可选值: 'classification', 'segmentation', 'detection' 等

# 训练管道 - 确保图像严格resize到256×256
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=img_scale,
        ratio_range=(1.0, 1.0),  # 固定比例，确保严格resize到256×256
        keep_ratio=False),
    # 移除RandomCrop，直接使用完整图像
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]

# 测试管道 - 确保图像严格resize到256×256
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=False),  # 修改：不保持宽高比
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
tta_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[
            [
                dict(type='Resize', scale_factor=r, keep_ratio=True)
                for r in img_ratios
            ],
            [
                dict(type='RandomFlip', prob=0., direction='horizontal'),
                dict(type='RandomFlip', prob=1., direction='horizontal')
            ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
        ])
]

# 由于使用完整图像，需要减小batch_size以避免内存不足
train_dataloader = dict(
    batch_size=2,  # 减小batch_size，因为图像尺寸更大
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=40000,
        dataset=dict(
            type=dataset_type,
            data_root=data_root,
            data_prefix=dict(
                img_path='images/training',
                seg_map_path='annotations/training'),
            pipeline=train_pipeline)))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(type='IoUMetric', iou_metrics=['mDice'])
test_evaluator = val_evaluator

default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='SegLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(by_epoch=False)
log_level = 'INFO'
load_from = None
resume = False

tta_model = dict(type='SegTTAModel')

# optimizer - 由于batch_size减小，可能需要调整学习率
optimizer = dict(type='SGD', lr=0.05, momentum=0.9, weight_decay=0.0005)  # 减小学习率
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)

# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-4,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=100)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=4, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000, save_best='auto'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook')) 

# 注册自定义hook，保存完整模型结构和权重
# 注意：这里使用的是devdeploy中的SaveFullModelHook，与mmpretrain独立
custom_hooks = [
    dict(
        type='SaveFullModelHook',
        # save_iter_best=True,  # 启用iter best保存，自动配置为iter-based训练
    ),
]

# 主要修改内容mu
# 1. 移除了所有裁剪相关的参数
# 删除了 crop_size = (128, 128)
# 删除了 stride = (85, 85)
# 从 data_preprocessor 中移除了 size 参数，改为使用 size_divisor=32
# 2. 修改了模型推理配置
# test_cfg 从 dict(mode='slide', crop_size=crop_size, stride=stride)
# 改为 dict(mode='whole')，使用完整图像推理
# 3. 简化了训练管道
# 移除了 RandomCrop 步骤
# 保留了其他数据增强（随机翻转、光度失真等）
# 4. 调整了训练参数
# batch_size 从 4 减小到 1（因为图像尺寸更大）
# 学习率从 0.01 减小到 0.005（因为batch_size减小）
# 5. 修复了BatchNorm问题
# 将 norm_cfg 从 SyncBN 改为 BN，因为batch_size=1时SyncBN无法正常工作