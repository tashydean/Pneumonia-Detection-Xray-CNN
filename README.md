# 🫁 Pneumonia Diagnosis with ResNet18 + GradCAM
**흉부 X-ray 기반 폐렴 진단 딥러닝 모델**

---

## 📌 개요

ResNet18 전이학습 기반 폐렴 이진 분류 모델. 단순 fine-tuning을 넘어:
- **의료 이미지 특화 pre_conv 레이어** 설계
- **Ablation Study**로 파라미터 축소 실험 및 최적 구조 선정
- **GradCAM**으로 모델 판단 근거 시각화 및 임상적 유효성 검증
- **임계값 최적화**로 FN(폐렴 미진단) 최소화 설계

> 핵심 문제의식: X-ray 판독에서 FN(폐렴인데 정상 오진)이 치명적.  
> 단순 Accuracy보다 **Recall과 AUROC**를 우선 지표로 설계.

---

## 📊 최종 성능

| Threshold | Accuracy | Precision | Recall | F1 Score | AUROC |
|-----------|----------|-----------|--------|----------|-------|
| 0.3       | -        | ↓         | ↑↑     | -        | ~0.97 |
| **0.5**   | **89.7%**| -         | -      | **Best** | ~0.97 |
| 0.7       | -        | ↑↑        | ↓↓     | ↓        | ~0.97 |

---

## 🏗️ 모델 아키텍처

```
Input (224×224 RGB)
    ↓
[pre_conv block]        ← X-ray 특화 초기 특징 추출 (3→16→3)
    ↓
[ResNet18 Backbone]     ← ImageNet 사전학습 가중치
    ↓
[FC Head]
  Linear(512→128) + BN + ReLU + Dropout(0.5)
  Linear(128→1)
    ↓
BCEWithLogitsLoss
```

---

## 📁 프로젝트 구조

```
pneumonia_diagnosis/
├── notebooks/
│   └── pneumonia_diagnosis_final.ipynb   # 전체 실험 노트북
├── src/
│   ├── dataset.py      # ChestXRayDataset, DataLoader
│   ├── model.py        # PneumoniaResNet 아키텍처
│   ├── train.py        # 학습 루프 (EarlyStopping, Scheduler)
│   ├── experiment.py   # Ablation Study
│   ├── evaluate.py     # 임계값별 메트릭, 혼동행렬, ROC
│   └── gradcam.py      # GradCAM + 의료 유효성 검증
├── outputs/
│   ├── figures/        # 학습 곡선, GradCAM, 혼동행렬 등
│   └── models/         # 저장된 모델 체크포인트
├── docs/
│   └── analysis_report.md
├── data/
│   └── README.md       # 데이터 출처 및 다운로드 안내
├── requirements.txt
└── .gitignore
```

---

## ⚙️ 실험 설계 (Ablation Study)

파라미터 축소 실험으로 최적 모델 구조 선정.

| 설정 | FC 크기 | Dropout | Freeze | F1 |
|------|---------|---------|--------|----|
| baseline | 512→128 | 0.5 | 없음 | - |
| fc64 | 512→64 | 0.5 | 없음 | - |
| freeze_layer1-3 | 512→128 | 0.5 | layer1~3 | - |

→ F1 기준 best 설정으로 이후 평가 진행

---

## 🔍 GradCAM 해석

| 케이스 | 기대 패턴 | 판정 기준 |
|--------|-----------|-----------|
| 폐렴 | 폐 하엽·중엽 실질에 집중 | ✅ 폐 실질 영역 집중 (유효) |
| 정상 | 전반적 분산 | ℹ️ 분산된 activation |
| 불유효 | 뼈·기기 아티팩트에 집중 | ⚠️ 경계 영역 집중 |

---

## 🚀 실행 방법

```bash
pip install -r requirements.txt

# Kaggle 데이터 다운로드
kaggle datasets download paultimothymooney/chest-xray-pneumonia
unzip chest-xray-pneumonia.zip -d data/

# 실행 순서
# model.py → dataset.py → experiment.py → evaluate.py → gradcam.py
```

또는 `notebooks/pneumonia_diagnosis_final.ipynb`를 순서대로 실행.

---

## 🛠️ 기술 스택

`Python` `PyTorch` `torchvision` `ResNet18` `GradCAM` `scikit-learn` `matplotlib` `pandas`

---

## 📚 데이터 출처

[Kaggle: Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)  
Train 5,216장 / Val 16장 / Test 624장 | NORMAL : PNEUMONIA ≈ 1:3
