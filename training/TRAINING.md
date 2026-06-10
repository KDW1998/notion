# 학습 (Training)

환경 설치는 [../ENVIRONMENT.md](../ENVIRONMENT.md)를 먼저 끝내세요.

> **수정할 것은 두 가지뿐입니다.** ① config 안의 학습 데이터 경로(`*_root`), ② 터미널 명령어.
> 그 외 config 내용이나 코드는 건드리지 않습니다.

## 0. config 파일 위치

학습 config 파일:

```
convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py
```

이 config는 `../_base_/...` 의 기본 설정을 상속합니다. mmsegmentation 소스 트리의
`configs/kcqr/` 폴더에 이 파일을 두고 실행하면 됩니다. (상속하는 기본 설정의 내용은
`configs_base/` 폴더에 참고용으로 함께 들어 있습니다.)

---

## 1. 학습 데이터 경로 수정

config 파일을 열어 `*_root` 로 끝나는 줄의 경로를 **본인 데이터 경로로** 바꾸세요.

학습 데이터 (10개):

```python
kcqr_2023_v1_root = '/.../2022_현장촬영이미지'
kcqr_2023_v2_root = '/.../2023_현장촬영이미지_학습용데이터셋'
...
kcqr_2023_v10_root = '/.../BKim_Thesis'
```

검증 데이터 (3개):

```python
br_dilation9_val_root      = '/.../br전체/테스트데이터'
joint_dilation9_val_root   = '/.../joint'
leakage_dilation9_val_root = '/.../leakage'
```

### 데이터 폴더 구조

각 `*_root` 폴더는 아래 구조여야 합니다.

학습 데이터:

```
<root>/
├── leftImg8bit/train/        *_leftImg8bit.png        (입력 이미지)
└── ReDilation_gtFine/train/  *_gtFine_labelIds.png    (정답 라벨)
```

검증 데이터:

```
<root>/
├── leftImg8bit/test/        *_leftImg8bit.png
└── Dilation9_gtFine/test/   *_gtFine_labelIds.png
```

> 라벨(`_gtFine_labelIds.png`)에서 균열 픽셀 값은 `1`, 배경은 `0` 입니다.

### 데이터셋을 더 추가하려면

1. config 안에 `kcqr_2023_v11` 블록을 기존 블록과 같은 형식으로 추가합니다.
2. `train_dataloader` 의 `datasets=[...]` 리스트에 `kcqr_2023_v11['train'],` 한 줄을 추가합니다.

---

## 2. 학습 실행

```bash
python tools/train.py configs/kcqr/convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py
```

학습 결과(체크포인트, 로그)는 config 안 `work_dir` 경로에 저장됩니다.

---

## 3. 이어서 학습 (학습된 모델에 데이터를 추가해 계속 학습)

### 방법 A — 가중치만 불러와서 다시 학습 (데이터 추가 시 권장)

이미 학습된 체크포인트의 **가중치만** 가져와, 새 데이터가 포함된 상태로 처음부터 학습합니다.

1. [1. 학습 데이터 경로 수정](#1-학습-데이터-경로-수정) 방법으로 새 데이터를 추가합니다.
2. 학습 명령에 `--cfg-options load_from=...` 를 붙여 실행합니다. (config 파일은 수정하지 않습니다.)

   ```bash
   python tools/train.py configs/kcqr/convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py \
     --cfg-options load_from=/경로/work_dirs/.../iter_best.pth
   ```

### 방법 B — 중단된 학습을 그대로 재개

학습이 중간에 멈췄을 때, optimizer 상태와 iter 번호까지 복원해 이어서 진행합니다.

```bash
python tools/train.py configs/kcqr/convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py --resume /경로/work_dirs/.../iter_xxxxx.pth
```

`--resume` 만 쓰면 `work_dir` 안의 가장 최근 체크포인트를 자동으로 찾아 재개합니다.

| | 방법 A (`load_from`) | 방법 B (`--resume`) |
|---|---|---|
| 가져오는 것 | 가중치만 | 가중치 + optimizer + iter 번호 |
| iter 시작 | 0부터 | 멈춘 지점부터 |
| 사용 상황 | **데이터를 추가해 다시 학습** | 중단된 동일 학습 이어가기 |
