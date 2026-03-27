"""
gradcam.py
GradCAM 구현 + 의료 이미지 유효성 검증 시각화.

목적: 모델이 "올바른 근거"로 판단하는지 검증.
  - 폐렴: activation이 폐 실질(lung parenchyma) 영역에 집중해야 함
  - 정상: activation이 전반적으로 분산되어야 함
  - pre_conv 필터: X-ray 특화 패턴(엣지, 음영 경계)을 포착하는지 확인
"""

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]).reshape(3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225]).reshape(3, 1, 1)


class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # ResNet18 마지막 conv block
        self.target_layer = target_layer or model.model.layer4[-1]
        self._features = None
        self._gradients = None
        self._hooks = [
            self.target_layer.register_forward_hook(self._save_features),
            self.target_layer.register_full_backward_hook(self._save_gradients),
        ]

    def _save_features(self, module, input, output):
        self._features = output.detach()

    def _save_gradients(self, module, grad_input, grad_output):
        self._gradients = grad_output[0].detach()

    def __call__(self, x):
        self.model.zero_grad()
        output = self.model(x)
        output.backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # GAP
        cam = (weights * self._features).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=(x.shape[2], x.shape[3]),
                            mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        return cam

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


def denormalize(tensor):
    img = tensor.cpu().numpy()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1).transpose(1, 2, 0)


def _get_validity_note(heatmap):
    """폐 영역 vs 경계 영역 activation 비율로 간이 유효성 판정."""
    H, W = heatmap.shape
    lung_region   = heatmap[H//4: H*3//4, W//6: W*5//6]
    border_region = np.concatenate([
        heatmap[:H//4].flatten(), heatmap[H*3//4:].flatten(),
        heatmap[:, :W//6].flatten(), heatmap[:, W*5//6:].flatten(),
    ])
    lung_mean   = lung_region.mean()
    border_mean = border_region.mean() if len(border_region) > 0 else 0

    if lung_mean > 0.45 and lung_mean > border_mean * 1.5:
        return "✅ 폐 실질 영역 집중 (유효)"
    elif border_mean > lung_mean * 1.3:
        return "⚠️  경계 영역 집중 (재확인 필요)"
    else:
        return "ℹ️  분산된 activation"


def visualize_gradcam_panel(model, loader, device, num_normal=4, num_pneumonia=4, threshold=0.5):
    gradcam = GradCAM(model)
    model.eval()

    normal_samples, pneumonia_samples = [], []
    for images, labels in loader:
        for i in range(len(labels)):
            if labels[i].item() == 0 and len(normal_samples) < num_normal:
                normal_samples.append(images[i])
            elif labels[i].item() == 1 and len(pneumonia_samples) < num_pneumonia:
                pneumonia_samples.append(images[i])
        if len(normal_samples) >= num_normal and len(pneumonia_samples) >= num_pneumonia:
            break

    cases = [("NORMAL", s, 0) for s in normal_samples] + \
            [("PNEUMONIA", s, 1) for s in pneumonia_samples]

    colormap = cm.get_cmap("jet")
    fig, axes = plt.subplots(len(cases), 4, figsize=(16, 4 * len(cases)))
    fig.suptitle("GradCAM — Model Decision Basis Verification\n"
                 "[Original] [Heatmap] [Overlay] [Confidence]",
                 fontsize=13, fontweight="bold", y=1.01)

    for col, title in enumerate(["Original X-ray", "GradCAM Heatmap", "Overlay", "Confidence"]):
        axes[0, col].set_title(title, fontsize=11, fontweight="bold")

    for row, (true_cls, img_tensor, true_lbl) in enumerate(cases):
        x = img_tensor.unsqueeze(0).to(device)
        x.requires_grad_(True)
        with torch.enable_grad():
            heatmap = gradcam(x)

        prob     = torch.sigmoid(model(img_tensor.unsqueeze(0).to(device))).item()
        pred_cls = "PNEUMONIA" if prob >= threshold else "NORMAL"
        correct  = "✅" if (prob >= threshold) == bool(true_lbl) else "❌"
        orig_img = denormalize(img_tensor)
        overlay  = orig_img * 0.55 + colormap(heatmap)[..., :3] * 0.45

        axes[row, 0].imshow(orig_img, cmap="gray")
        axes[row, 0].set_ylabel(f"True: {true_cls}", fontsize=9)
        axes[row, 0].axis("off")

        im = axes[row, 1].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].axis("off")
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

        axes[row, 2].imshow(overlay)
        axes[row, 2].set_xlabel(_get_validity_note(heatmap), fontsize=8, color="navy")
        axes[row, 2].axis("off")

        bars = axes[row, 3].barh(["Normal", "Pneumonia"], [1 - prob, prob],
                                  color=["steelblue", "tomato"])
        axes[row, 3].set_xlim(0, 1)
        axes[row, 3].axvline(threshold, color="gray", linestyle="--", lw=1,
                              label=f"threshold={threshold}")
        axes[row, 3].set_title(f"Pred: {pred_cls} {correct}\n(p={prob:.3f})",
                                fontsize=9,
                                color="darkgreen" if correct == "✅" else "crimson")
        for bar, val in zip(bars, [1 - prob, prob]):
            axes[row, 3].text(min(val + 0.02, 0.93),
                              bar.get_y() + bar.get_height() / 2,
                              f"{val:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig('outputs/figures/gradcam_panel.png', dpi=150, bbox_inches='tight')
    plt.show()
    gradcam.remove_hooks()


def visualize_preconv_filters(model):
    """pre_conv 첫 번째 Conv2d 필터 시각화."""
    weight  = model.pre_conv[0].weight.data.cpu()  # (16, 3, 3, 3)
    filters = weight.mean(dim=1)                    # (16, 3, 3)

    fig, axes = plt.subplots(2, 8, figsize=(16, 4))
    fig.suptitle("Pre-trained Filters (pre_conv) — X-ray Specific Pattern Verification",
                 fontsize=12, fontweight="bold")
    for i, ax in enumerate(axes.flat):
        if i < len(filters):
            f = filters[i].numpy()
            f = (f - f.min()) / (f.max() - f.min() + 1e-8)
            ax.imshow(f, cmap="seismic", vmin=0, vmax=1)
            ax.set_title(f"Filter {i+1}", fontsize=8)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig('outputs/figures/preconv_filters.png', dpi=150, bbox_inches='tight')
    plt.show()
