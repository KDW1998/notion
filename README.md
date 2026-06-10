# Crack Segmentation 모델 (ConvNeXt + FPN)

콘크리트 균열(crack) 세그멘테이션 모델의 **학습**과 **추론**을 그대로 재현하기 위한 패키지입니다.

## 폴더 구성

```
notion/
├── README.md            ← 지금 이 문서
├── ENVIRONMENT.md       ← 환경 설치 (가장 먼저 보세요)
├── training/            ← 모델 학습
│   ├── TRAINING.md
│   ├── convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py
│   ├── configs_base/    ← 학습 config가 상속하는 기본 설정
│   └── custom_mmseg/    ← 학습에 필요한 커스텀 코드
└── inference/           ← 모델 추론 + 성능 평가
    ├── INFERENCE.md
    ├── comprehensive_test.py
    ├── utils.py
    ├── quantify_seg_results.py
    └── requirements.txt
```

> **설계 원칙**: 환경 설치 후에는 **학습 데이터 경로**와 **터미널 명령어**만 바꾸면 됩니다.
> config 파일 내용이나 추론 코드는 수정할 필요가 없습니다.

## 시작 순서

1. **[ENVIRONMENT.md](ENVIRONMENT.md)** — conda 환경과 라이브러리를 설치합니다.
2. **[training/TRAINING.md](training/TRAINING.md)** — 데이터 경로를 지정하고 학습을 실행합니다.
3. **[inference/INFERENCE.md](inference/INFERENCE.md)** — 학습된 모델로 추론하고 Precision / Recall / F1 / IoU를 측정합니다.

## 모델 한눈에 보기

| 항목 | 값 |
|------|----|
| Backbone | ConvNeXt-V2-atto + FPN |
| 클래스 | 2 (background, crack) |
| 입력 크기 | 1024 × 1024 |
| 학습 iter | 80,000 |
| 평가 지표 | Precision, Recall, F1-score, IoU |
