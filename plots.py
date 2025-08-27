# viz.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

def colorize_depth(depth, vmax=10.0, cmap="plasma"):
    # depth: [H,W] en m
    d = np.clip(depth, 0, vmax) / vmax
    cm = plt.get_cmap(cmap)
    colored = cm(d)[:, :, :3]  # RGB
    return (colored*255).astype(np.uint8)

def save_triplet(rgb, pred, gt, out_path, vmax=10.0, title=None):
    """
    rgb: [3,H,W] tensor (normalizado imagenet) -> mostramos en [0,1]
    pred, gt: [1,H,W] en m
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rgb_show = rgb.clone()
    # des-normalizar ImageNet
    mean = torch.tensor([0.485,0.456,0.406], device=rgb.device).view(3,1,1)
    std  = torch.tensor([0.229,0.224,0.225], device=rgb.device).view(3,1,1)
    rgb_show = (rgb_show*std + mean).clamp(0,1)
    rgb_np = (rgb_show.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)

    pred_np = pred.squeeze(0).detach().cpu().numpy()
    gt_np   = gt.squeeze(0).detach().cpu().numpy()

    pred_col = colorize_depth(pred_np, vmax=vmax)
    gt_col   = colorize_depth(gt_np,   vmax=vmax)

    fig, axs = plt.subplots(1,3, figsize=(12,4))
    axs[0].imshow(rgb_np);    axs[0].set_title("RGB");  axs[0].axis("off")
    axs[1].imshow(pred_col);  axs[1].set_title("Pred (m)"); axs[1].axis("off")
    axs[2].imshow(gt_col);    axs[2].set_title("GT (m)");   axs[2].axis("off")
    if title: fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
