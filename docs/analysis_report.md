# Pneumonia Detection with ResNet18 + GradCAM
## 분석 리포트

---

## 1. 프로젝트 개요

흉부 X-ray 이미지를 기반으로 정상/폐렴을 이진 분류하는 딥러닝 모델을 개발했다. 단순 fine-tuning에 그치지 않고, **의료 도메인 특화 설계**와 **모델 판단 근거의 임상적 검증**을 핵심 목표로 삼았다.

**핵심 문제의식**  
X-ray 판독에서 폐렴 환자를 정상으로 오진(FN, False Negative)하는 것은 치료 지연으로 이어져 임상적으로 치명적이다. 따라서 단순 Accuracy보다 **Recall과 AUROC**를 우선 설계 지표로 설정했다.

**평가 지표 선택 논리**

| 단계 | 기준 지표 | 이유 |
|------|-----------|------|
| Ablation Study (모델 선택) | F1 | Recall만 보면 무조건 폐렴으로 예측하는 편향 모델이 유리해짐. F1은 Precision·Recall 균형을 동시에 요구하므로 "진짜 잘 배운 모델"을 선별할 수 있음 |
| 임계값 선택 | Recall | 모델이 고정된 이후에는 FN 최소화가 임상적 우선순위. Precision↓(과검출)은 추가 검사로 대응 가능하지만, Recall↓(미검출)은 치료 기회 자체를 잃음 |
| 모델 판별력 확인 | AUROC | 임계값 무관하게 모델이 폐렴/정상을 얼마나 잘 분리하는지 확인. F1이 비슷한 모델 간 최종 판단 기준으로도 활용 |

---

## 2. 데이터셋

- **출처**: [Kaggle — Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **원출처 논문**: Kermany et al., *Cell* 2018

| 분할 | Normal | Pneumonia | 합계 |
|------|--------|-----------|------|
| Train | 1,341 | 3,875 | 5,216 |
| Val | 8 | 8 | 16 |
| Test | 234 | 390 | 624 |

**클래스 불균형**: 폐렴이 정상보다 약 3배 많음

<img src="../outputs/figures/Data_Distribution.png" width="600"/>

클래스 불균형에 대응하기 위해 데이터 증강과 평가 지표를 F1/Recall 중심으로 설계했다. Accuracy만 보면 "모든 이미지를 폐렴으로 예측"해도 높은 수치가 나오는 착시가 발생할 수 있다.

---

## 3. 데이터 전처리 및 증강

**학습 데이터 증강**

```python
train_transforms = v2.Compose([
    v2.ToImage(),
    v2.Resize((224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),      # 좌우폐 대칭성 활용
    v2.RandomRotation(degrees=10),        # 10도 제한: 의학적 구도 보존
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=ImageNet_mean, std=ImageNet_std),
])
```

수평 반전은 흉부 X-ray의 좌우 폐 대칭성을 고려할 때 유효한 증강이다. 회전은 의료 이미지의 촬영 구도를 과도하게 왜곡하지 않도록 10도로 제한했다.

<img src="../outputs/figures/Original_Image.png" width="550"/>
<img src="../outputs/figures/Augmented_Image.png" width="550"/>

---

## 4. 모델 아키텍처

### 설계 원칙

사전학습된 ResNet18 백본을 활용하되, **자연 이미지에 최적화된 ResNet의 초기 특징 추출 단계를 의료 이미지에 맞게 조정**하는 것이 핵심 설계 포인트였다.

```
Input (224×224 RGB)
    ↓
[pre_conv block]       ← X-ray 특화 초기 특징 추출 (채널: 3 → 16 → 3)
    ↓
[ResNet18 Backbone]    ← ImageNet 사전학습 가중치
    ↓
[FC Head]
  Linear(512→128) + BatchNorm1d + ReLU + Dropout(0.5)
  Linear(128→1)
    ↓
BCEWithLogitsLoss
```

### pre_conv 블록의 역할

흉부 X-ray는 자연 이미지와 달리 **저대비, 음영 기반의 패턴**이 주를 이룬다. ResNet의 첫 conv 레이어는 ImageNet으로 학습된 엣지·색상 필터에 편향되어 있어, X-ray 고유의 미세한 밀도 차이를 충분히 포착하지 못할 수 있다.

`pre_conv(3→16→3)`는 이 문제를 보완하기 위해 ResNet 앞단에 추가된 소형 블록으로, X-ray 도메인에 특화된 초기 표현을 학습하면서도 ResNet 입력 형식(3채널)을 그대로 유지한다. 이를 통해 ImageNet 사전학습 가중치를 온전히 활용하면서도 도메인 적응이 가능하다.

---

## 5. 학습 설정

| 항목 | 설정값 | 설계 이유 |
|------|--------|-----------|
| Optimizer | Adam (lr=0.0001) | 전이학습에서 작은 LR로 사전학습 가중치 보존 |
| Loss | BCEWithLogitsLoss | sigmoid + BCE를 수치 안정적으로 결합한 이진 분류 표준 |
| EarlyStopping | patience=5, val_loss 기준 | 과적합 방지 + best 가중치 자동 복원 |
| LR Scheduler | ReduceLROnPlateau (patience=2, factor=0.5) | val_loss 정체 구간에서 LR을 단계적으로 줄여 학습 재개 |
| Epoch | 최대 20 | 전이학습 + EarlyStopping 조합으로 충분 |
| Batch size | 32 | GPU 메모리와 일반화 성능의 균형 |
| 데이터 증강 | Flip, Rotation | 폐렴 이미지 편향 및 과적합 완화 |

**EarlyStopping 작동 방식**  
patience=5 카운터가 채워지기 전에 학습이 종료되면, 그것은 "얼리스탑이 발동되지 않은" 것이 아니라 모든 에포크 동안 val_loss가 꾸준히 개선되었다는 의미다 — 좋은 학습 신호.

---

## 6. Ablation Study — 파라미터 축소 실험

### 실험 설계

동일 데이터·하이퍼파라미터에서 **설정(구조)만 바꿔** 성능 변화를 측정했다.

| 실험명 | FC 크기 | Dropout | Backbone Freeze |
|--------|---------|---------|-----------------|
| baseline | 512→128 | 0.5 | 없음 |
| fc64 | 512→64 | 0.5 | 없음 |
| freeze_layer1-3 | 512→128 | 0.5 | layer1~3 고정 |

### 실험 결과

| 설정 | Val Loss | Accuracy | Recall | F1 | AUROC |
|------|----------|----------|--------|-----|-------|
| **baseline** | **0.0276** | 86.54% | **99.49%** | **90.23%** | 92.49% |
| fc64 | 0.0874 | 83.33% | 99.74% | 88.21% | 90.02% |
| freeze_layer1-3 | 0.3828 | 86.38% | 98.97% | 90.08% | **95.64%** |

<img src="../outputs/figures/ablation_metrics_valloss.png" width="760"/>
<img src="../outputs/figures/ablation_learning_convergence.png" width="580"/>

### 해석

**baseline 선정 (F1 기준 최우수)**

- **fc64**: 파라미터를 절반으로 줄였으나 val_loss가 3배 이상 높아졌다. 모델 크기를 줄이는 것이 항상 일반화에 도움이 되는 것은 아님 — 표현력 손실이 정규화 이득을 상회한 경우다.
- **freeze_layer1-3**: AUROC는 세 실험 중 가장 높지만 val_loss 0.38로 학습이 불안정했다. backbone의 초기 레이어까지 고정하면 X-ray 도메인 특화 표현을 충분히 학습하지 못하는 것으로 해석된다. layer4(고수준 특징)만 fine-tuning하는 강한 제약이 이 데이터셋에는 맞지 않았음을 시사한다.

---

## 7. 임계값 최적화 및 최종 평가

### 평가 지표 의미

| 지표 | 의미 | 폐렴 진단에서의 함의 |
|------|------|---------------------|
| **Accuracy** | 전체 중 맞힌 비율 | 클래스 불균형 시 착시 발생 가능 |
| **Precision** | 폐렴 예측 중 실제 폐렴 비율 | 낮으면 과진단 → 불필요한 추가 검사 |
| **Recall (민감도)** | 실제 폐렴 중 탐지한 비율 | 낮으면 미진단 → 치료 기회 상실 (치명적) |
| **F1 Score** | Precision × Recall 조화평균 | 둘의 균형 지표 |
| **AUROC** | 임계값 무관 모델 판별력 | 0.9249 → 임상적으로 유의미한 수준(≥0.90) |

### 임계값별 비교

<img src="../outputs/figures/Evaluation_Metrics_Summary_by_Threshold.png" width="700"/>

| Threshold | Accuracy | Precision | Recall | F1 Score | AUROC |
|-----------|----------|-----------|--------|----------|-------|
| 0.3 (채택) | 87.50% | 83.90% | **98.97%** | 90.82% | 92.49% |
| 0.5 | 89.74% | 89.25% | 96.41% | 92.70% | 92.49% |
| 0.7 | 90.22% | 93.03% | 92.56% | 92.80% | 92.49% |

임계값을 낮출수록 Recall↑ Precision↓, 높일수록 Precision↑ Recall↓ — Precision-Recall 트레이드오프의 전형적인 패턴이다. **Recall 최대화 기준으로 0.3을 채택**했다.

### 혼동행렬 (임계값 0.3)

<img src="../outputs/figures/Confusion_Matrix.png" width="380"/>

- **FN (폐렴 미진단)**: 약 4건 — 실제 폐렴 환자 390명 중 386명 탐지
- **FP (과진단)**: 약 73건 — 추가 검사로 대응 가능한 오류

**임상적 해석**: Recall 98.97%는 폐렴 환자를 거의 놓치지 않음을 의미한다. FP 증가는 의사의 최종 확인 과정에서 필터링될 수 있으나, FN은 치료 기회 자체를 박탈한다. 의료 AI 보조 도구로서 이 트레이드오프는 임상적으로 수용 가능한 설계다.

---

## 8. GradCAM — 모델 판단 근거의 임상적 검증

성능 수치만으로는 "모델이 올바른 이유로 맞혔는지" 알 수 없다. GradCAM은 예측에 가장 크게 기여한 이미지 영역을 시각화하여 모델의 판단 근거를 역추적한다.

<img src="../outputs/figures/GradCAM_Model_Decision_Basis_Verification.png" width="820"/>

### 검증 기준

임상의가 "유효하다"고 판단하는 기준을 코드로 구현했다.

```python
def _get_validity_note(heatmap):
    lung_region   = heatmap[H//4 : H*3//4, W//6 : W*5//6]   # 폐 실질 영역
    border_region = ...                                        # 뼈·기기 영역

    if lung_mean > 0.45 and lung_mean > border_mean * 1.5:
        return "Focused on Lung Parenchyma (Valid)"
    elif border_mean > lung_mean * 1.3:
        return "Focused on Border Area (Check Required)"
    else:
        return "Diffused Activation"
```

### 오분류(MISS) 케이스 해석

틀린 케이스의 GradCAM이 폐 영역을 보고 있더라도, 모델이 근거 없이 틀린 것이 아니다. 시각적으로 구별이 어려운 경계 케이스(borderline case)를 의미할 수 있다. 오분류 자체보다 **activation이 폐 실질에 집중했는지**가 임상적 신뢰성 판단의 기준이다.

### pre_conv 필터 시각화

<img src="../outputs/figures/Pre_trained_Filters.png" width="720"/>

학습된 16개 필터가 각기 다른 방향성·대비 패턴을 가지고 있어, 분업화된 특징 탐지 구조가 형성됐음을 확인했다. 이는 pre_conv 블록이 단순한 파라미터 추가가 아니라 실제로 의미 있는 X-ray 특화 표현을 학습했음을 방증한다.

### GradCAM 해석의 한계

- 입력마다 기울기가 달라 활성화 위치가 가변적임
- 시각적 설명이 모델 내부 의사결정과 완전히 일치하지 않을 수 있음
- GradCAM은 모델 신뢰성 검증의 **보조 도구**이며, 단독 임상 판단 근거로 사용 불가

---

## 9. 추론 샘플 시각화

<img src="../outputs/figures/inference_prediction_results.png" width="820"/>

테스트셋에서 정상/폐렴 각 5장을 무작위 추출하여 예측 결과를 시각화했다. 소수 샘플 기반이므로 이 정확도는 전체 성능 지표가 아니며, 실제 모델 성능은 임계값별 평가 테이블을 기준으로 한다.

---

## 10. 결론

### 성과 요약

- AUROC 92.49% — 임계값 무관하게 모델 자체의 판별력이 임상적으로 유의미한 수준(≥0.90)
- Recall 98.97% (threshold=0.3) — 폐렴 미진단(FN) 최소화 목표 달성
- Ablation Study로 구조를 최적화하여 val_loss 0.0276의 안정적인 일반화 모델 확보
- GradCAM으로 모델이 폐 실질 영역을 근거로 판단함을 임상적으로 검증

### 한계 및 향후 과제

- 학습 데이터가 단일 출처(Kaggle)로, 다기관 임상 데이터로의 일반화 검증 필요
- 세균성/바이러스성 폐렴 세분류 확장 가능성 탐색
- Val set이 16장(8+8)으로 극히 작아 val_loss 기반 EarlyStopping의 신뢰도에 한계 존재
- 실제 임상 배포를 위해서는 전문의 레이블링 데이터 기반 추가 검증 필수
