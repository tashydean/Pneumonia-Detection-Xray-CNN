"""
train.py
EarlyStopping + ReduceLROnPlateau 포함 학습 루프.
"""

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class EarlyStopping:
    """
    val_loss가 patience 동안 개선 없으면 학습 중단.
    best 가중치를 내부에 보존해두고 나중에 복원 가능.
    """
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience   = patience
        self.min_delta  = min_delta
        self.counter    = 0
        self.best_loss  = float('inf')
        self.best_state = None
        self.triggered  = False

    def step(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss  = val_loss
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter    = 0
        else:
            self.counter += 1
            print(f"  EarlyStopping: {self.counter}/{self.patience} (best={self.best_loss:.4f})")
            if self.counter >= self.patience:
                self.triggered = True
                return True
        return False

    def restore_best(self, model):
        if self.best_state:
            model.load_state_dict(self.best_state)
            print(f"✅ Best model 복원 (val_loss={self.best_loss:.4f})")
        return model


def train(model, train_loader, val_loader, device,
          num_epochs=30, lr=1e-4, patience=5):
    """
    학습 루프. (model, history) 반환.
    history = {train_losses, val_losses, train_accs, val_accs, lr_history}
    """
    criterion  = nn.BCEWithLogitsLoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=2, factor=0.5, verbose=True
    )
    early_stop = EarlyStopping(patience=patience)

    history = {'train_losses': [], 'val_losses': [],
               'train_accs': [],   'val_accs': [], 'lr_history': []}

    for epoch in range(num_epochs):
        # ── Train ──────────────────────────────────────────────
        model.train()
        train_loss, train_correct = 0.0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.float().to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = nn.BCEWithLogitsLoss()(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss    += loss.item() * images.size(0)
            preds          = (torch.sigmoid(outputs) >= 0.5).long()
            train_correct += (preds == labels.long()).sum().item()

        train_epoch_loss = train_loss / len(train_loader.dataset)
        train_epoch_acc  = train_correct / len(train_loader.dataset)

        # ── Val ────────────────────────────────────────────────
        model.eval()
        val_loss, val_correct = 0.0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.float().to(device)
                outputs  = model(images)
                val_loss += nn.BCEWithLogitsLoss()(outputs, labels).item() * images.size(0)
                preds     = (torch.sigmoid(outputs) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()

        val_epoch_loss = val_loss / len(val_loader.dataset)
        val_epoch_acc  = val_correct / len(val_loader.dataset)

        scheduler.step(val_epoch_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_losses'].append(train_epoch_loss)
        history['val_losses'].append(val_epoch_loss)
        history['train_accs'].append(train_epoch_acc)
        history['val_accs'].append(val_epoch_acc)
        history['lr_history'].append(current_lr)

        print(f"[{epoch+1:02d}/{num_epochs}] "
              f"Train {train_epoch_loss:.4f}/{train_epoch_acc:.4f} | "
              f"Val {val_epoch_loss:.4f}/{val_epoch_acc:.4f} | LR {current_lr:.2e}")

        if early_stop.step(val_epoch_loss, model):
            print(f"\n🛑 EarlyStopping at epoch {epoch+1}")
            break

    model = early_stop.restore_best(model)
    return model, history, early_stop


def plot_history(history, early_stop):
    epochs = range(1, len(history['train_losses']) + 1)
    stopped = len(history['train_losses'])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Training History', fontsize=14, fontweight='bold')

    for ax, (train_key, val_key, title) in zip(axes[:2], [
        ('train_losses', 'val_losses', 'Loss'),
        ('train_accs',   'val_accs',   'Accuracy'),
    ]):
        ax.plot(epochs, history[train_key], 'b-o', label='Train', markersize=4)
        ax.plot(epochs, history[val_key],   'r-o', label='Val',   markersize=4)
        if early_stop.triggered:
            ax.axvline(stopped, color='gray', linestyle='--', label=f'Early Stop (ep{stopped})')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.legend()
        ax.grid(alpha=0.3)

    axes[2].plot(epochs, history['lr_history'], 'g-o', markersize=4)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch')
    axes[2].set_yscale('log')
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig('outputs/figures/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
