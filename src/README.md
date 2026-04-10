# src/

노트북(`notebooks/`)의 핵심 로직을 모듈화한 Python 파일들입니다.  
각 파일은 독립적으로 import하거나, 노트북에서 순서대로 실행할 수 있습니다.

---

## 모듈 구성

| 파일 | 역할 |
|------|------|
| `dataset.py` | `ChestXRayDataset` 클래스 + 학습/평가용 transform + DataLoader 팩토리 |
| `model.py` | `PneumoniaResNet` 아키텍처 (pre_conv block + ResNet18 backbone + FC Head) |
| `train.py` | `EarlyStopping` 클래스 + `ReduceLROnPlateau` 포함 학습 루프 |
| `experiment.py` | Ablation Study 실험 실행 — FC 크기 / Dropout / Backbone freeze 비교 |
| `evaluate.py` | 임계값별 Accuracy / Precision / Recall / F1 / AUROC 계산 + 혼동행렬 + ROC curve |
| `gradcam.py` | `GradCAM` 클래스 + 임상 유효성 검증 시각화 + pre_conv 필터 시각화 |

---

## 설계 포인트

**`model.py` — pre_conv block**  
ResNet18 앞단에 `Conv2d(3→16→3)` 블록을 추가. 자연 이미지에 편향된 ResNet 초기 레이어를 보완하여 흉부 X-ray의 저대비·음영 패턴을 선처리한다. ResNet 입력 형식(3채널)을 유지하므로 사전학습 가중치와 호환된다.

**`train.py` — EarlyStopping**  
val_loss 기준 patience=5. 개선이 없는 구간에서 학습을 중단하고 best 시점 가중치를 자동 복원한다. `ReduceLROnPlateau(patience=2, factor=0.5)`와 함께 사용하여 학습 정체 구간에서 LR을 단계적으로 줄인다.

**`experiment.py` — Ablation Study**  
동일 코드베이스에서 설정만 바꿔 반복 실험. F1 기준으로 best 모델을 자동 선정한다.

**`evaluate.py` — 임계값 선택 기준**  
임계값 선택은 Recall 최대화 기준. FN(폐렴 미검출)이 FP(과검출)보다 임상적으로 훨씬 치명적이기 때문이다. Ablation Study의 모델 선택(F1 기준)과 구분되는 설계 의도.

**`gradcam.py` — 임상 유효성 검증**  
단순 heatmap 시각화에 그치지 않고, 폐 중앙 영역 vs 경계 영역의 activation 비율을 계산하여 모델이 "올바른 근거"로 판단하는지를 자동 판정한다.

---

## 실행 순서 (독립 실행 시)

```
model.py → dataset.py → train.py → experiment.py → evaluate.py → gradcam.py
```
