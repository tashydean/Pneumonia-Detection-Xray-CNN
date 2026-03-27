"""
experiment.py
파라미터 축소 실험 (Ablation Study).

실험 축:
  1. FC Head 크기 (512→128 vs 512→64)
  2. Dropout 비율 (0.3 / 0.5 / 0.7)
  3. Backbone freeze 전략

best model을 model 변수에 저장 → 이후 evaluate.py, gradcam.py에서 사용.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, recall_score, accuracy_score, roc_auc_score
import numpy as np

from model import PneumoniaResNet
from train import EarlyStopping


# 실험 설정: (실험명, fc_hidden, dropout_rate, freeze_layers)
CONFIGS = [
    ("baseline",       128, 0.5, []),
    ("fc64",            64, 0.5, []),
    ("freeze_layer1-3",128, 0.5, ["layer1", "layer2", "layer3"]),
]


def build_experiment_model(fc_hidden, dropout_rate, freeze_layers, device):
    model = PneumoniaResNet(dropout_rate=dropout_rate).to(device)
    num_ftrs = model.model.fc[0].in_features  # 512

    if fc_hidden is not None:
        model.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, fc_hidden),
            nn.BatchNorm1d(fc_hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(fc_hidden, 1)
        ).to(device)
    else:
        model.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_ftrs, 1)
        ).to(device)

    for name, param in model.model.named_parameters():
        if any(layer in name for layer in freeze_layers):
            param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  학습 파라미터: {trainable:,}")
    return model


def run_single_experiment(model, train_loader, val_loader, test_loader, device, num_epochs=20):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)
    es        = EarlyStopping(patience=5)

    for epoch in range(num_epochs):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                val_loss += criterion(model(images), labels).item() * images.size(0)
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)

        if es.step(val_loss, model):
            print(f"    Early stop at epoch {epoch+1}")
            break

    model = es.restore_best(model)

    # Test 평가
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for images, labels in test_loader:
            probs = torch.sigmoid(model(images.to(device))).view(-1).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_labels = np.array(all_labels)
    all_probs  = np.array(all_probs)
    preds      = (all_probs >= 0.5).astype(int)

    return {
        'best_val_loss': round(es.best_loss, 4),
        'accuracy':      round(accuracy_score(all_labels, preds), 4),
        'recall':        round(recall_score(all_labels, preds, zero_division=0), 4),
        'f1':            round(f1_score(all_labels, preds, zero_division=0), 4),
        'auroc':         round(roc_auc_score(all_labels, all_probs), 4),
    }


def run_ablation(train_loader, val_loader, test_loader, device):
    """
    전체 ablation 실행.
    Returns: (best_model, df_ablation)
    """
    ablation_results = []

    for name, fc_hidden, dropout_rate, freeze_layers in CONFIGS:
        print(f"\n{'='*55}")
        print(f"실험: {name}")
        exp_model = build_experiment_model(fc_hidden, dropout_rate, freeze_layers, device)
        result    = run_single_experiment(exp_model, train_loader, val_loader, test_loader, device)
        result['name']  = name
        result['model'] = exp_model
        ablation_results.append(result)
        print(f"  → F1={result['f1']} | Recall={result['recall']} | AUROC={result['auroc']}")

    # best model 선택
    df_ablation = pd.DataFrame(
        [{k: v for k, v in r.items() if k != 'model'} for r in ablation_results]
    ).set_index('name')

    best_name  = df_ablation['f1'].idxmax()
    best_model = next(r['model'] for r in ablation_results if r['name'] == best_name)
    best_model.eval()

    print(f"\n🏆 Best 설정 (F1 기준): {best_name}")
    print(df_ablation.loc[best_name])

    # 시각화
    _plot_ablation(df_ablation)

    return best_model, df_ablation


def _plot_ablation(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Ablation Study: 파라미터 축소 실험', fontsize=13, fontweight='bold')

    df[['f1', 'recall', 'auroc']].plot(
        kind='bar', ax=axes[0], colormap='viridis', edgecolor='white'
    )
    axes[0].set_title('F1 / Recall / AUROC 비교')
    axes[0].set_ylim(0, 1.1)
    axes[0].set_xticklabels(df.index, rotation=35, ha='right')
    axes[0].grid(axis='y', alpha=0.3)

    df['best_val_loss'].plot(kind='bar', ax=axes[1], color='tomato', edgecolor='white')
    axes[1].set_title('Best Val Loss (낮을수록 좋음)')
    axes[1].set_xticklabels(df.index, rotation=35, ha='right')
    axes[1].grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/ablation_study.png', dpi=150, bbox_inches='tight')
    plt.show()
