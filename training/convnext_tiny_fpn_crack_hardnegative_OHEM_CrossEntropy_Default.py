_base_ = [
    '../_base_/models/fpn_convnext-v2-atto_CrossEntropyLoss.py',
    '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_20k.py'
]

crop_size = (1024, 1024)
norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    data_preprocessor=dict(size=crop_size),
    decode_head=dict(
        num_classes=2,
        # OHEM Sampler: 어려운 픽셀에 집중
        sampler=dict(
            type='OHEMPixelSampler',
            thresh=0.7,        # confidence threshold (모호한 영역 선택)
            min_kept=50000     # 균열 픽셀의 ~8배, hard sample에 집중
        ),
        # CrossEntropyLoss with class weight (100units.py와 동일)
        loss_decode=dict(
            type='CrossEntropyLoss',
            use_sigmoid=False,           # Softmax 기반 (2-class)
            class_weight=[10., 20.],     # [배경, 균열] - 100units.py와 동일
            loss_weight=1.0,
            loss_name='loss_ce'
        ),
    ))

optim_wrapper = dict(
    _delete_=True,
    type='AmpOptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.05),
    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'stage_wise',
        'num_layers': 6
    },
    constructor='LearningRateDecayOptimizerConstructor',
    loss_scale='dynamic')

max_iters = 80000
param_scheduler = [
    dict(
        type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
    dict(
        type='PolyLR',
        power=1.0,
        begin=1500,
        end=max_iters,
        eta_min=0.0,
        by_epoch=False,
    )
]

dataset_type = 'KCQRCrackDataset'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(
        type='RandomResize',
        scale=(1024, 1024),
        ratio_range=(0.5, 2.0),
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=crop_size),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='LoadAnnotations'),
    dict(type='PackSegInputs')
]

img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]

label_map = {
    2:0,
    3:0,
    4:0,
    5:0,
    6:0,
    7:0,
    8:0,
}


# ============================================================================
# 학습 데이터셋 (10개) - train only
#   - 사용하려면 아래 각 *_root 경로를 본인 데이터 경로로 바꾸세요.
#   - 각 data_root 폴더 구조:
#       leftImg8bit/train/*_leftImg8bit.png        (입력 이미지)
#       gtFine/train/*_gtFine_labelIds.png  (정답 라벨)
# ============================================================================
kcqr_2023_v1_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/2022_현장촬영이미지'
kcqr_2023_v1 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v1_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v2_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/2023_현장촬영이미지_학습용데이터셋'
kcqr_2023_v2 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v2_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v3_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/AI_Hub_콘크리트균열_원천_34/crop'
kcqr_2023_v3 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v3_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v4_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/genenral_crack_v0.1.2'
kcqr_2023_v4 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v4_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v5_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/general_crack_v0.1.1'
kcqr_2023_v5 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v5_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v6_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/project0811_학습데이터'
kcqr_2023_v6 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v6_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v7_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/한국도로공사_split'
kcqr_2023_v7 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v7_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v8_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/v0.1.5_학습데이터'
kcqr_2023_v8 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v8_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v9_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/한국도로공사_전북본부_청운구조_와탄천교_P2전면부(목포)'
kcqr_2023_v9 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v9_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)

kcqr_2023_v10_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/학습데이터/실제촬영이미지/BKim_Thesis'
kcqr_2023_v10 = dict(
    train=dict(
        type=dataset_type,
        data_root=kcqr_2023_v10_root,
        data_prefix=dict(
            img_path='leftImg8bit/train',
            seg_map_path='gtFine/train'
        ),
        label_map=label_map,
        pipeline=train_pipeline,
    )
)


# ============================================================================
# 검증 데이터셋 - Dilation 9x9 gtFine 사용
#   - 각 data_root 폴더 구조:
#       leftImg8bit/test/*_leftImg8bit.png
#       Dilation9_gtFine/test/*_gtFine_labelIds.png
# ============================================================================
br_dilation9_val_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/테스트데이터/br전체/테스트데이터'
br_dilation9_val = dict(
    type=dataset_type,
    data_root=br_dilation9_val_root,
    data_prefix=dict(
        img_path='leftImg8bit/test',
        seg_map_path='Dilation9_gtFine/test'
    ),
    label_map=label_map,
    pipeline=test_pipeline,
)

joint_dilation9_val_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/테스트데이터/joint'
joint_dilation9_val = dict(
    type=dataset_type,
    data_root=joint_dilation9_val_root,
    data_prefix=dict(
        img_path='leftImg8bit/test',
        seg_map_path='Dilation9_gtFine/test'
    ),
    label_map=label_map,
    pipeline=test_pipeline,
)

leakage_dilation9_val_root='/home/user/WindowsShare/05. Data/04. Raw Images & Archive/206. hardnegative/테스트데이터/leakage'
leakage_dilation9_val = dict(
    type=dataset_type,
    data_root=leakage_dilation9_val_root,
    data_prefix=dict(
        img_path='leftImg8bit/test',
        seg_map_path='Dilation9_gtFine/test'
    ),
    label_map=label_map,
    pipeline=test_pipeline,
)


# ============================================================================
# Dataloaders
#   - 학습 데이터를 더 추가하려면 위에 kcqr_2023_vN 블록을 같은 형식으로 만들고
#     아래 train_dataloader 의 datasets 리스트에 kcqr_2023_vN['train'] 한 줄을 추가하세요.
# ============================================================================
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            # 균열 이미지 (10개)
            kcqr_2023_v1['train'],
            kcqr_2023_v2['train'],
            kcqr_2023_v3['train'],
            kcqr_2023_v4['train'],
            kcqr_2023_v5['train'],
            kcqr_2023_v6['train'],
            kcqr_2023_v7['train'],
            kcqr_2023_v8['train'],
            kcqr_2023_v9['train'],
            kcqr_2023_v10['train'],
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            br_dilation9_val,
            joint_dilation9_val,
            leakage_dilation9_val,
        ]))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='ConcatDataset',
        datasets=[
            br_dilation9_val,
            joint_dilation9_val,
            leakage_dilation9_val,
        ]))


val_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mFscore'])
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU'])
val_cfg = dict(type='ValLoopWithLoss')

work_dir = 'work_dirs/20260527_PSOnly_ReDilation9_gtFine_HardNegativeVal_OHEM_CrossEntropy_max_iters80000'
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=4000)
    )
train_cfg = dict(type='IterBasedTrainLoop', max_iters=max_iters, val_interval=4000)
