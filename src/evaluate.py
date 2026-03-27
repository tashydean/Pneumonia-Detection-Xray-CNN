"""
evaluate.py
임계값별 메트릭 + 혼동행렬 + ROC 시각화.

의료 AI 설계 철학:
  FN(폐렴→정상 오진)이 치명적 → Recall 우선, F1으로 임계값 선택.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import pandas as pd


def collect_predictions(model, loader, device):
    """모델 예측값 수집. (all_labels, all_probs) 반환."""
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in loader:
            probs = torch.sigmoid(model(images.to(device))).view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy().reshape(-1))
    return np.array(all_labels), np.array(all_probs)


def evaluate_thresholds(all_labels, all_probs, thresholds=[0.3, 0.5, 0.7]):
    """임계값별 Accuracy / Precision / Recall / F1 / AUROC 계산."""
    auroc = roc_auc_score(all_labels, all_probs)
    records = []
    for thr in thresholds:
        preds = (all_probs >= thr).astype(int)
        records.append({
            'Threshold': thr,
            'Accuracy':  round(accuracy_score(all_labels, preds), 4),
            'Precision': round(precision_score(all_labels, preds, zero_division=0), 4),
            'Recall':    round(recall_score(all_labels, preds, zero_division=0), 4),
            'F1 Score':  round(f1_score(all_labels, preds, zero_division=0), 4),
            'AUROC':     round(auroc, 4),
        })
    df = pd.DataFrame(records).set_index('Threshold')
    print("\n📊 Threshold Evaluation:")
    print(df.to_string())
    best = df['F1 Score'].idxmax()
    print(f"\n✅ Best threshold (F1 기준): {best}")
    return df


def plot_metrics_by_threshold(df):
    ax = df.plot(kind='bar', figsize=(12, 6), colormap='viridis', edgecolor='white')
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}",
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='bottom', fontsize=9,
                    xytext=(0, 3), textcoords='offset points')
    ax.set_title('Accuracy, Precision, Recall, F1, AUROC', fontsize=14, fontweight='bold')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Metric Value')
    ax.set_ylim(0, 1.2)
    plt.xticks(rotation=0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/figures/metrics_by_threshold.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(all_labels, all_probs, threshold, class_names=['Normal', 'Pneumonia']):
    preds = (all_probs >= threshold).astype(int)
    cm    = confusion_matrix(all_labels, preds)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(2), yticks=np.arange(2),
           xticklabels=class_names, yticklabels=class_names,
           xlabel='Predicted', ylabel='True',
           title=f'Confusion Matrix (threshold={threshold})')
    thresh = cm.max() / 2.
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black')

    fn = cm[1, 0]
    fp = cm[0, 1]
    fig.suptitle(f'FN(미진단): {fn}건  |  FP(과진단): {fp}건\n의료 맥락: FN이 더 치명적',
                 fontsize=10, color='darkred', y=1.02)
    plt.tight_layout()
    plt.savefig('outputs/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curve(all_labels, all_probs):
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    auroc = roc_auc_score(all_labels, all_probs)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, 'b-', lw=2, label=f'AUROC = {auroc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve — Pneumonia Detection', fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/figures/roc_curve.png', dpi=150, bbox_inches='tight')
    plt.show()
