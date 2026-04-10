# Data

이 디렉토리는 학습/검증/테스트 데이터를 위한 공간입니다.  
데이터 용량 문제로 실제 이미지 파일은 포함되어 있지 않습니다. 아래 안내에 따라 다운로드하세요.

---

## 데이터 출처

**Kaggle — Chest X-Ray Images (Pneumonia)**  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

원출처 논문: Kermany et al., *Cell* 2018 — "Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning"

---

## 데이터 구성

| 분할 | Normal | Pneumonia | 합계 |
|------|--------|-----------|------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

**클래스 불균형**: 폐렴이 정상보다 약 3배 많음  
→ 데이터 증강(RandomHorizontalFlip, RandomRotation) 및 평가 지표 F1/Recall 중심 설계로 대응

---

## 다운로드 방법

### 방법 1 — Kaggle CLI

```bash
# Kaggle API 설정 필요 (~/.kaggle/kaggle.json)
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/
```

### 방법 2 — kagglehub (노트북 내 사용)

```python
import kagglehub
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
```

---

## 다운로드 후 예상 폴더 구조

```
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/       # 1,341장
    │   └── PNEUMONIA/    # 3,875장
    ├── val/
    │   ├── NORMAL/       # 8장
    │   └── PNEUMONIA/    # 8장
    └── test/
        ├── NORMAL/       # 234장
        └── PNEUMONIA/    # 390장
```

---

## 이미지 특성

- 포맷: JPEG
- 해상도: 다양함 (전처리 시 224×224로 통일)
- 채널: 그레이스케일 → RGB로 변환하여 ResNet 입력에 맞춤
- 대상: 소아 환자 흉부 정면 X-ray (AP/PA view)
- 폐렴 유형: 세균성(Bacterial) + 바이러스성(Viral) 혼합 — 세분류 레이블 없음
