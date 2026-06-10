# 추론 및 성능 평가 (Inference)

환경 설치는 [../ENVIRONMENT.md](../ENVIRONMENT.md)를 먼저 끝내세요.
이 폴더(`inference/`) 안에서 명령을 실행합니다.

## 실행 명령어

```bash
python comprehensive_test.py \
  --crack_config     <config.py> \
  --crack_checkpoint <checkpoint.pth> \
  --root_dir         <테스트 데이터 폴더> \
  --result_dir       <결과 저장 폴더>
```

예시:

```bash
python comprehensive_test.py \
  --crack_config     ./convnext_tiny_fpn_crack_hardnegative_OHEM_CrossEntropy_Default.py \
  --crack_checkpoint ./iter_best.pth \
  --root_dir         /data/테스트데이터 \
  --result_dir       ./results
```

## 인자 설명

| 인자 | 설명 |
|------|------|
| `--crack_config` | 모델 config 파일 (`.py`) |
| `--crack_checkpoint` | 학습된 체크포인트 (`.pth`) |
| `--root_dir` | 테스트 데이터가 있는 폴더 |
| `--result_dir` | 결과(이미지·CSV)를 저장할 폴더 |

자주 쓰는 선택 옵션:

| 옵션 | 기본값 | 설명 |
|------|--------|------|
| `--gt_subdir` | `gtFine` | 정답 라벨 폴더명. `leftImg8bit` 대신 이 이름의 폴더에서 GT를 찾음 |
| `--gt_base_dir` / `--gt_ext_subdir` | (없음) / `Dilation9_gtFine` | GT가 다른 위치에 있을 때 외부 GT 루트와 서브폴더명 지정 |
| `--alpha` | `0.8` | 균열 마스크 시각화 투명도 |
| `--rgb_to_bgr` | (꺼짐) | 팔레트 색상이 RGB로 저장된 경우 BGR로 변환 |
| `--overwrite_crack_palette` | (꺼짐) | 균열 색상을 빨강으로 고정 |

## 입력 폴더 구조

`--root_dir` 아래에서 `leftImg8bit/test` 폴더를 자동으로 찾습니다.

```
<root_dir>/
├── leftImg8bit/test/   *_leftImg8bit.png        (입력 이미지)
└── gtFine/test/        *_gtFine_labelIds.png    (정답 라벨, --gt_subdir 로 변경 가능)
```

## 결과물

`--result_dir` 에 다음이 저장됩니다.

- **시각화 이미지** — 균열 검출 결과 + 정답 비교(초록=TP, 자홍=FP, 빨강=FN)
- **예측 마스크** — `pred_gtFine/` 폴더에 라벨 이미지로 저장
- **`per_image_results.csv`** — 이미지별 `Precision, Recall, F1-score, IoU`
- **`global_results.csv`** — 전체 합산 `Precision, Recall, F1-score, IoU`

> 추론은 1024×1024 슬라이딩 윈도우(overlap 0.1)로 동작하며, 학습 시 입력 크기와 동일합니다.
