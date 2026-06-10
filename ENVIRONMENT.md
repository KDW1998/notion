# 환경 설치

학습과 추론 모두 동일한 환경을 사용합니다. 아래 순서대로 실행하세요.

## 1. conda 환경 생성

```bash
conda create -n crack_gauge python=3.8 -y
conda activate crack_gauge
```

## 2. PyTorch 설치 (GPU)

CUDA 11.6 기준 예시입니다. 본인 GPU/CUDA에 맞게 설치하세요.

```bash
pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

## 3. OpenMMLab 라이브러리 설치

```bash
pip install openmim
mim install mmengine
mim install mmcv
mim install mmpretrain
mim install mmsegmentation
```

## 4. 추론용 추가 라이브러리

```bash
pip install -r inference/requirements.txt
```

## 5. 학습용 커스텀 코드 적용 (학습할 때만 필요)

이 모델의 학습 config는 mmsegmentation 기본 패키지에는 없는 커스텀 코드(`KCQRCrackDataset`,
`ValLoopWithLoss`)를 사용합니다. `training/custom_mmseg/` 폴더의 파일을 본인이 설치한
mmsegmentation 안의 **같은 경로**에 그대로 복사하세요.

```
training/custom_mmseg/mmseg/datasets/kcqr_crack.py      → mmseg/datasets/kcqr_crack.py
training/custom_mmseg/mmseg/datasets/kcqr_deterio.py    → mmseg/datasets/kcqr_deterio.py
training/custom_mmseg/mmseg/engine/loops/val_loss_loop.py → mmseg/engine/loops/val_loss_loop.py
training/custom_mmseg/mmseg/engine/loops/__init__.py    → mmseg/engine/loops/__init__.py
```

복사 후, mmsegmentation의 다음 파일에 등록이 되어 있는지 확인하세요.

- `mmseg/datasets/__init__.py` 에 `KCQRCrackDataset` 추가
- `mmseg/engine/__init__.py` 에 `ValLoopWithLoss` 추가

> mmsegmentation을 소스(github)에서 받아 `pip install -e .` 로 설치했다면, 위 파일들을 해당
> 소스 트리에 복사하는 것이 가장 간단합니다.

## 참고: 사전학습 가중치

Backbone 사전학습 가중치는 학습 시작 시 자동으로 다운로드됩니다 (인터넷 연결 필요).

```
https://download.openmmlab.com/mmclassification/v0/convnext-v2/convnext-v2-atto_3rdparty-fcmae_in1k_20230104-07514db4.pth
```
