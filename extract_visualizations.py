"""
RSCL Visualization Extraction Script
=====================================
Extracts all feature maps for the three-panel method figure:
  Panel A: Input images, predictions, decoder features
  Panel B: r[i] reliability map, three-zone stratification, zone evolution
  Panel C: Memory bank prototypes, feature-to-prototype similarity, t-SNE

Usage:
  python extract_visualizations.py \
      --checkpoint_dir /public/home/zhsy/data/zhsy_data/RSCL/ACDC_20pct/model \
      --data_dir /public/home/zhsy/data/zhsy_data/training \
      --test_data_dir /public/home/zhsy/data/zhsy_data/testing \
      --train_data_csv ./data_dir/train_ACDC.csv \
      --valid_data_csv ./data_dir/valid_ACDC.csv \
      --test_data_list ./data_dir/test_subj_ACDC.csv \
      --output_dir ./vis_outputs \
      --device 1

All variable names match train.py and rscl.py exactly.
"""

import os
import sys
import argparse
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ---- Project imports (same as train.py) ----
from SegModel import ProjectUNet1
from rscl import RSCL
from utilities.MyDataSet import SemiSegDataset_2, SemiSegDataset_1
from utilities.val_2D import get_image_list, crop_image
import nibabel as nib


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint_dir", type=str,
                   default="/public/home/zhsy/data/zhsy_data/RSCL/ACDC_10pct/model")
    p.add_argument("--data_dir", type=str,
                   default="/public/home/zhsy/data/zhsy_data/training")
    p.add_argument("--test_data_dir", type=str,
                   default="/public/home/zhsy/data/zhsy_data/testing")
    p.add_argument("--train_data_csv", type=str, default="./data_dir/train_ACDC.csv")
    p.add_argument("--valid_data_csv", type=str, default="./data_dir/valid_ACDC.csv")
    p.add_argument("--test_data_list", type=str, default="./data_dir/test_subj_ACDC.csv")
    p.add_argument("--output_dir", type=str, default="./vis_outputs")
    p.add_argument("--num_classes", type=int, default=4)
    p.add_argument("--image_size", nargs='+', type=int, default=[224, 224])
    p.add_argument("--label_ratio", type=float, default=0.1)
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_iterations", type=int, default=30000)
    # Checkpoints at different iterations for zone evolution
    p.add_argument("--evo_checkpoints", nargs='+', type=str, default=None,
                   help="Paths to model checkpoints at different iterations for zone evolution. "
                        "If not provided, uses best_model only.")
    return p.parse_args()


# ============================= COLOR MAPS =============================

# Class colors: BG=black, RVC=red, Myo=green, LVC=blue (ACDC convention)
CLASS_COLORS = np.array([
    [0, 0, 0],        # 0: background
    [220, 60, 60],     # 1: RVC
    [60, 180, 75],     # 2: Myo
    [65, 105, 225],    # 3: LVC
], dtype=np.uint8)

# Zone colors: Anchor=teal, Learning=amber, Exclusion=red
ZONE_COLORS = np.array([
    [180, 180, 180],   # unassigned / background (should not appear)
    [15, 110, 86],     # Ω_a: anchor (teal)
    [186, 117, 23],    # Ω_l: learning (amber)
    [163, 45, 45],     # Ω_e: exclusion (red)
], dtype=np.uint8)


def class_to_rgb(seg, palette=CLASS_COLORS):
    """Convert integer segmentation map [H, W] to RGB [H, W, 3]."""
    h, w = seg.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(len(palette)):
        rgb[seg == c] = palette[c]
    return rgb


def zone_to_rgb(zone_map):
    """Convert zone map (1=anchor, 2=learning, 3=exclusion) to RGB."""
    h, w = zone_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for z in range(len(ZONE_COLORS)):
        rgb[zone_map == z] = ZONE_COLORS[z]
    return rgb


# ====================== RELIABILITY & ZONES (exact rscl.py logic) ======================

def compute_reliability(q1, q2):
    """Exact copy of RSCL.compute_reliability.
    q1, q2: [B, C, H, W] softmax probabilities
    Returns: r [B, H, W], agreement [B,H,W], confidence [B,H,W], jsd [B,H,W]
    """
    y1 = q1.argmax(dim=1)
    y2 = q2.argmax(dim=1)

    agree = (y1 == y2).float()
    conf = (q1.max(dim=1)[0] + q2.max(dim=1)[0]) / 2.0

    m = (q1 + q2) / 2.0
    kl1 = (q1 * (q1.clamp(min=1e-7).log() - m.clamp(min=1e-7).log())).sum(dim=1)
    kl2 = (q2 * (q2.clamp(min=1e-7).log() - m.clamp(min=1e-7).log())).sum(dim=1)
    jsd = (kl1 + kl2) / 2.0

    r = agree * conf * torch.exp(-jsd)
    return r, agree, conf, jsd


def get_thresholds(cur_iter, max_iter,
                   tau_high_init=0.50, tau_high_final=0.85,
                   tau_low_init=0.10, tau_low_final=0.30):
    """Exact copy of RSCL.get_thresholds."""
    ratio = cur_iter / max(max_iter, 1)
    tau_high = tau_high_final - (tau_high_final - tau_high_init) * math.cos(math.pi * ratio / 2)
    tau_low = tau_low_init + (tau_low_final - tau_low_init) * (1 - math.cos(math.pi * ratio / 2))
    return tau_high, tau_low


def compute_zone_map(r, tau_high, tau_low):
    """
    r: [B, H, W] reliability scores
    Returns: zone_map [B, H, W] with 1=anchor, 2=learning, 3=exclusion
    """
    zone = torch.full_like(r, 3, dtype=torch.long)  # default exclusion
    zone[r > tau_low] = 2     # learning
    zone[r > tau_high] = 1    # anchor
    return zone


# ====================== LOAD MODELS ======================

def load_models(checkpoint_dir, num_classes, device):
    """Load both Θ1 and Θ2, same as train.py model construction."""
    # Exact same construction as train.py line 133-134
    model1 = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).to(device)
    model2 = ProjectUNet1('resnet50', None, classes=num_classes, deep_stem=32).to(device)

    ckpt1 = os.path.join(checkpoint_dir, 'best_model1.pth')
    ckpt2 = os.path.join(checkpoint_dir, 'best_model2.pth')

    model1.load_state_dict(torch.load(ckpt1, map_location=device))
    model2.load_state_dict(torch.load(ckpt2, map_location=device))

    model1.eval()
    model2.eval()
    print(f"Loaded Θ1 from {ckpt1}")
    print(f"Loaded Θ2 from {ckpt2}")
    return model1, model2


def load_rscl(num_classes, device):
    """Construct RSCL module with default hyperparameters (same as train.py line 137-141)."""
    rscl = RSCL(num_classes=num_classes, feat_dim=128,
                contrastive_mode='dual',
                tau_high_final=0.85,
                tau_low_final=0.30,
                tau_soft=0.5).to(device)
    return rscl


# ====================== FORWARD PASS ======================

@torch.no_grad()
def forward_one_image(model1, model2, image, device):
    """Run both networks on a single image tensor [1, 1, H, W].
    Returns exact same variables as train.py lines 185-188.
    """
    image = image.to(device)
    # model forward: masks, out_list[1], features[2]
    # In train.py: y=logits, dx=decoder_feat(128ch), _=encoder_feat
    y1, dx1, _ = model1(image)   # y:[1,C,224,224], dx:[1,128,56,56]
    y2, dx2, _ = model2(image)

    return y1, y2, dx1, dx2


# ====================== PANEL B: RELIABILITY & ZONES ======================

def extract_panel_b(model1, model2, image_labeled, image_unlabeled,
                    label_labeled, device, cur_iter, max_iter, output_dir):
    """
    Extract all Panel B visualizations:
      - r[i] heatmap on unlabeled image
      - Three sub-signals: agreement, confidence, JSD
      - Zone stratification map
      - Overlay on original image
    """
    os.makedirs(os.path.join(output_dir, 'panel_b'), exist_ok=True)

    # Forward pass on unlabeled image (exact train.py variable names)
    y_u1, y_u2, dx_u1, dx_u2 = forward_one_image(model1, model2, image_unlabeled, device)

    # Softmax predictions (exact rscl.py lines 191-192)
    q_u1 = F.softmax(y_u1, dim=1)
    q_u2 = F.softmax(y_u2, dim=1)

    # Reliability decomposition
    r_u, agree, conf, jsd = compute_reliability(q_u1, q_u2)

    # Thresholds (exact rscl.py line 196)
    tau_high, tau_low = get_thresholds(cur_iter, max_iter)

    # Zone map at full prediction resolution [1, 224, 224]
    zone_map = compute_zone_map(r_u, tau_high, tau_low)

    # Also compute at feature resolution (exact rscl.py lines 202-215)
    Hf, Wf = dx_u1.shape[2], dx_u1.shape[3]  # 56, 56
    r_u_down = F.interpolate(r_u.unsqueeze(1), size=(Hf, Wf),
                             mode='bilinear', align_corners=False).squeeze(1)
    anchor_mask = (r_u_down > tau_high)
    learn_mask = (r_u_down > tau_low) & (~anchor_mask)

    # Convert to numpy for plotting
    img_np = image_unlabeled[0, 0].cpu().numpy()       # [224, 224]
    r_np = r_u[0].cpu().numpy()                         # [224, 224]
    agree_np = agree[0].cpu().numpy()
    conf_np = conf[0].cpu().numpy()
    jsd_np = jsd[0].cpu().numpy()
    zone_np = zone_map[0].cpu().numpy()                 # [224, 224]

    # ---- r[i] heatmap ----
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))

    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input $\\mathbf{x}_u$', fontsize=11)
    axes[0].axis('off')

    im1 = axes[1].imshow(agree_np, cmap='RdYlGn', vmin=0, vmax=1)
    axes[1].set_title('Agreement $a[i]$', fontsize=11)
    axes[1].axis('off')
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    im2 = axes[2].imshow(conf_np, cmap='hot', vmin=0, vmax=1)
    axes[2].set_title('Confidence $p[i]$', fontsize=11)
    axes[2].axis('off')
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    im3 = axes[3].imshow(jsd_np, cmap='viridis')
    axes[3].set_title('JSD $d[i]$', fontsize=11)
    axes[3].axis('off')
    plt.colorbar(im3, ax=axes[3], fraction=0.046)

    im4 = axes[4].imshow(r_np, cmap='hot', vmin=0, vmax=1)
    axes[4].set_title('Reliability $r[i]$', fontsize=11)
    axes[4].axis('off')
    plt.colorbar(im4, ax=axes[4], fraction=0.046)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_b', 'reliability_decomposition.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_b', 'reliability_decomposition.pdf'),
                bbox_inches='tight')
    plt.close()

    # ---- Zone map overlay ----
    zone_rgb = zone_to_rgb(zone_np)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Input $\\mathbf{x}_u$', fontsize=11)
    axes[0].axis('off')

    axes[1].imshow(r_np, cmap='hot', vmin=0, vmax=1)
    # Draw threshold lines on colorbar
    axes[1].set_title(f'$r[i]$ (τ_h={tau_high:.2f}, τ_l={tau_low:.2f})', fontsize=11)
    axes[1].axis('off')

    # Zone overlay: image with semi-transparent zone coloring
    axes[2].imshow(img_np, cmap='gray')
    axes[2].imshow(zone_rgb, alpha=0.45)
    axes[2].set_title('Zone map', fontsize=11)
    axes[2].axis('off')

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=np.array([15, 110, 86]) / 255., label='$\\Omega_a$ (anchor)'),
        Patch(facecolor=np.array([186, 117, 23]) / 255., label='$\\Omega_l$ (learning)'),
        Patch(facecolor=np.array([163, 45, 45]) / 255., label='$\\Omega_e$ (exclusion)'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right', fontsize=8,
                   framealpha=0.8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_b', 'zone_stratification.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_b', 'zone_stratification.pdf'),
                bbox_inches='tight')
    plt.close()

    # ---- Zone statistics ----
    n_total = zone_np.size
    n_anchor = (zone_np == 1).sum()
    n_learn = (zone_np == 2).sum()
    n_excl = (zone_np == 3).sum()
    print(f"  Zone stats (iter={cur_iter}): "
          f"Anchor={n_anchor} ({100*n_anchor/n_total:.1f}%), "
          f"Learning={n_learn} ({100*n_learn/n_total:.1f}%), "
          f"Exclusion={n_excl} ({100*n_excl/n_total:.1f}%)")

    # Save raw arrays for figure compositing
    np.savez(os.path.join(output_dir, 'panel_b', 'raw_arrays.npz'),
             image=img_np, reliability=r_np, agreement=agree_np,
             confidence=conf_np, jsd=jsd_np, zone_map=zone_np,
             tau_high=tau_high, tau_low=tau_low, cur_iter=cur_iter)

    return r_np, zone_np


def extract_zone_evolution(model1, model2, image_unlabeled, device,
                           max_iter, output_dir, evo_iters=None):
    """
    Extract zone maps at different simulated training iterations.
    Uses the same trained model but varies τ_high(t) and τ_low(t).
    This shows how thresholds shift even with the same predictions.
    For true evolution, provide checkpoints at different iterations.
    """
    if evo_iters is None:
        evo_iters = [500, 2000, 5000, 10000, 20000, 30000]

    os.makedirs(os.path.join(output_dir, 'panel_b'), exist_ok=True)

    y_u1, y_u2, _, _ = forward_one_image(model1, model2, image_unlabeled, device)
    q_u1 = F.softmax(y_u1, dim=1)
    q_u2 = F.softmax(y_u2, dim=1)
    r_u, _, _, _ = compute_reliability(q_u1, q_u2)
    r_np = r_u[0].cpu().numpy()
    img_np = image_unlabeled[0, 0].cpu().numpy()

    n_iters = len(evo_iters)
    fig, axes = plt.subplots(2, n_iters, figsize=(4 * n_iters, 8))

    for idx, t in enumerate(evo_iters):
        tau_high, tau_low = get_thresholds(t, max_iter)
        zone = compute_zone_map(r_u, tau_high, tau_low)
        zone_np = zone[0].cpu().numpy()
        zone_rgb = zone_to_rgb(zone_np)

        # Top row: r[i] with threshold annotation
        axes[0, idx].imshow(r_np, cmap='hot', vmin=0, vmax=1)
        axes[0, idx].axhline(y=0, color='w', alpha=0)  # placeholder
        axes[0, idx].set_title(f'iter={t}\nτ_h={tau_high:.2f}, τ_l={tau_low:.2f}',
                               fontsize=9)
        axes[0, idx].axis('off')

        # Bottom row: zone overlay
        axes[1, idx].imshow(img_np, cmap='gray')
        axes[1, idx].imshow(zone_rgb, alpha=0.45)
        n_a = (zone_np == 1).sum()
        n_l = (zone_np == 2).sum()
        n_e = (zone_np == 3).sum()
        axes[1, idx].set_title(f'A:{n_a/zone_np.size*100:.0f}% '
                               f'L:{n_l/zone_np.size*100:.0f}% '
                               f'E:{n_e/zone_np.size*100:.0f}%',
                               fontsize=9)
        axes[1, idx].axis('off')

    plt.suptitle('Zone evolution across training (threshold annealing)', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_b', 'zone_evolution.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_b', 'zone_evolution.pdf'),
                bbox_inches='tight')
    plt.close()


# ====================== PANEL C: DGPC VISUALIZATIONS ======================

def extract_panel_c(model1, model2, rscl_module,
                    image_labeled, label_labeled,
                    image_unlabeled, device, cur_iter, max_iter, output_dir):
    """
    Extract Panel C visualizations:
      - Memory bank prototype cosine similarity matrix
      - Feature-to-prototype similarity maps per class
      - Hard vs soft target distribution comparison
    """
    os.makedirs(os.path.join(output_dir, 'panel_c'), exist_ok=True)

    # Forward (exact train.py lines 185-188)
    y_l1, y_l2, dx_l1, dx_l2 = forward_one_image(model1, model2, image_labeled, device)
    y_u1, y_u2, dx_u1, dx_u2 = forward_one_image(model1, model2, image_unlabeled, device)

    q_u1 = F.softmax(y_u1, dim=1)
    q_u2 = F.softmax(y_u2, dim=1)

    # Run RSCL forward to populate memory bank (exact rscl.py forward)
    rscl_module.eval()
    _ = rscl_module(dx_l1, dx_l2, dx_u1, dx_u2,
                    y_l1, y_l2, y_u1, y_u2,
                    label_labeled.to(device), cur_iter, max_iter)

    # ---- Memory bank inter-class similarity matrix ----
    memory = rscl_module.memory.cpu().numpy()  # [C, D]
    memory_init = rscl_module.memory_init.cpu().numpy()  # [C]

    # Cosine similarity between all class prototypes
    C = memory.shape[0]
    sim_matrix = np.zeros((C, C))
    for i in range(C):
        for j in range(C):
            if memory_init[i] and memory_init[j]:
                sim_matrix[i, j] = np.dot(memory[i], memory[j])

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    im = ax.imshow(sim_matrix, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(C))
    ax.set_yticks(range(C))
    class_names = ['BG', 'RVC', 'Myo', 'LVC'][:C]
    ax.set_xticklabels(class_names, fontsize=10)
    ax.set_yticklabels(class_names, fontsize=10)
    ax.set_title('Memory bank $\\mathbf{B}$: inter-class cosine similarity', fontsize=11)
    for i in range(C):
        for j in range(C):
            ax.text(j, i, f'{sim_matrix[i,j]:.2f}', ha='center', va='center',
                    fontsize=9, color='white' if abs(sim_matrix[i,j]) > 0.5 else 'black')
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_c', 'memory_bank_similarity.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_c', 'memory_bank_similarity.pdf'),
                bbox_inches='tight')
    plt.close()

    # ---- Feature-to-prototype similarity maps for unlabeled image ----
    # Uses exact same logic as rscl.py _hard_contrastive_masked
    Hf, Wf = dx_u1.shape[2], dx_u1.shape[3]  # 56, 56
    feat_u = (dx_u1 + dx_u2) / 2.0  # [1, 128, 56, 56], same as rscl.py line 220
    feat_flat = F.normalize(feat_u.permute(0, 2, 3, 1).reshape(-1, 128), dim=1)  # [N, 128]
    proto_mat = rscl_module.memory  # [C, 128]

    # Hard similarity: logits / tau_hard (rscl.py line 148)
    hard_logits = (feat_flat @ proto_mat.t() / rscl_module.tau_hard)  # [N, C]
    hard_probs = F.softmax(hard_logits, dim=1).reshape(Hf, Wf, C).cpu().numpy()

    # Soft similarity: logits / tau_soft (rscl.py line 168)
    soft_logits = (feat_flat @ proto_mat.t() / rscl_module.tau_soft)  # [N, C]
    soft_probs = F.softmax(soft_logits, dim=1).reshape(Hf, Wf, C).cpu().numpy()

    fig, axes = plt.subplots(2, C + 1, figsize=(4 * (C + 1), 8))
    img_np = image_unlabeled[0, 0].cpu().numpy()

    # Row 0: Hard similarity per class
    axes[0, 0].imshow(img_np, cmap='gray')
    axes[0, 0].set_title('Input', fontsize=10)
    axes[0, 0].axis('off')
    for c in range(C):
        sim_up = F.interpolate(
            torch.tensor(hard_probs[:, :, c]).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        axes[0, c + 1].imshow(sim_up, cmap='hot', vmin=0, vmax=1)
        axes[0, c + 1].set_title(f'Hard sim: {class_names[c]}', fontsize=10)
        axes[0, c + 1].axis('off')

    # Row 1: Soft similarity per class
    axes[1, 0].imshow(img_np, cmap='gray')
    axes[1, 0].set_title('Input', fontsize=10)
    axes[1, 0].axis('off')
    for c in range(C):
        sim_up = F.interpolate(
            torch.tensor(soft_probs[:, :, c]).unsqueeze(0).unsqueeze(0),
            size=(224, 224), mode='bilinear', align_corners=False
        ).squeeze().numpy()
        axes[1, c + 1].imshow(sim_up, cmap='hot', vmin=0, vmax=1)
        axes[1, c + 1].set_title(f'Soft sim: {class_names[c]}', fontsize=10)
        axes[1, c + 1].axis('off')

    plt.suptitle('Feature-to-prototype similarity (hard τ=0.2 vs soft τ_s=0.5)',
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_c', 'feat_proto_similarity.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_c', 'feat_proto_similarity.pdf'),
                bbox_inches='tight')
    plt.close()

    # ---- Hard vs soft target distribution at a boundary pixel ----
    # Find a boundary pixel: where anchor and learning zone meet
    r_u, _, _, _ = compute_reliability(q_u1, q_u2)
    tau_high, tau_low = get_thresholds(cur_iter, max_iter)
    r_u_down = F.interpolate(r_u.unsqueeze(1), size=(Hf, Wf),
                             mode='bilinear', align_corners=False).squeeze(1)
    learn_mask = (r_u_down > tau_low) & (r_u_down <= tau_high)

    if learn_mask[0].any():
        # Pick a learning zone pixel
        learn_coords = learn_mask[0].nonzero(as_tuple=False)
        mid_idx = len(learn_coords) // 2
        py, px = learn_coords[mid_idx].tolist()

        # Hard target: one-hot from pseudo-label
        pseudo_u1 = y_u1.detach().argmax(dim=1)
        pseudo_u_down = F.interpolate(pseudo_u1.unsqueeze(1).float(),
                                      size=(Hf, Wf), mode='nearest').squeeze(1).long()
        hard_class = pseudo_u_down[0, py, px].item()
        hard_target = np.zeros(C)
        hard_target[hard_class] = 1.0

        # Soft target: averaged prediction (exact rscl.py line 276)
        q_avg_u = ((q_u1 + q_u2) / 2.0).detach()
        q_avg_u_down = F.interpolate(q_avg_u, size=(Hf, Wf),
                                     mode='bilinear', align_corners=False)
        soft_target = q_avg_u_down[0, :, py, px].cpu().numpy()

        # Feature-to-prototype similarity at this pixel
        feat_pixel = feat_flat[py * Wf + px]  # [128]
        sim_hard = F.softmax(feat_pixel @ proto_mat.t() / rscl_module.tau_hard, dim=0).cpu().numpy()
        sim_soft = F.softmax(feat_pixel @ proto_mat.t() / rscl_module.tau_soft, dim=0).cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(13, 4))
        x_pos = np.arange(C)
        width = 0.35

        # Hard target vs hard similarity
        axes[0].bar(x_pos - width/2, hard_target, width, label='Hard target (one-hot)',
                    color='#0F6E56', alpha=0.8)
        axes[0].bar(x_pos + width/2, sim_hard, width, label='Hard sim ($\\tau$=0.2)',
                    color='#0F6E56', alpha=0.4)
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(class_names)
        axes[0].set_title(f'Hard contrastive at ({py},{px})', fontsize=10)
        axes[0].legend(fontsize=8)
        axes[0].set_ylim(0, 1.1)

        # Soft target vs soft similarity
        axes[1].bar(x_pos - width/2, soft_target, width, label='Soft target $\\bar{q}$',
                    color='#BA7517', alpha=0.8)
        axes[1].bar(x_pos + width/2, sim_soft, width, label='Soft sim ($\\tau_s$=0.5)',
                    color='#BA7517', alpha=0.4)
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(class_names)
        axes[1].set_title(f'Soft contrastive at ({py},{px})', fontsize=10)
        axes[1].legend(fontsize=8)
        axes[1].set_ylim(0, 1.1)

        # Direct comparison: hard target vs soft target
        axes[2].bar(x_pos - width/2, hard_target, width, label='Hard target',
                    color='#0F6E56', alpha=0.8)
        axes[2].bar(x_pos + width/2, soft_target, width, label='Soft target',
                    color='#BA7517', alpha=0.8)
        axes[2].set_xticks(x_pos)
        axes[2].set_xticklabels(class_names)
        axes[2].set_title('Target comparison at boundary pixel', fontsize=10)
        axes[2].legend(fontsize=8)
        axes[2].set_ylim(0, 1.1)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'panel_c', 'hard_vs_soft_target.png'),
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(output_dir, 'panel_c', 'hard_vs_soft_target.pdf'),
                    bbox_inches='tight')
        plt.close()
        print(f"  Boundary pixel ({py},{px}): hard→class {hard_class}, "
              f"soft→{soft_target.round(3)}")

    # Save raw arrays
    np.savez(os.path.join(output_dir, 'panel_c', 'raw_arrays.npz'),
             memory_bank=memory, memory_init=memory_init,
             sim_matrix=sim_matrix,
             hard_similarity=hard_probs, soft_similarity=soft_probs)


# ====================== PANEL A: ARCHITECTURE OVERVIEW ======================

def extract_panel_a(model1, model2, image_labeled, label_labeled,
                    image_unlabeled, device, output_dir):
    """
    Extract Panel A visualizations:
      - Input images with GT overlay
      - Prediction maps from both networks
      - Decoder feature magnitude maps
    """
    os.makedirs(os.path.join(output_dir, 'panel_a'), exist_ok=True)

    y_l1, y_l2, dx_l1, dx_l2 = forward_one_image(model1, model2, image_labeled, device)
    y_u1, y_u2, dx_u1, dx_u2 = forward_one_image(model1, model2, image_unlabeled, device)

    # Predictions
    pred_l1 = y_l1.argmax(dim=1)[0].cpu().numpy()
    pred_l2 = y_l2.argmax(dim=1)[0].cpu().numpy()
    pred_u1 = y_u1.argmax(dim=1)[0].cpu().numpy()
    pred_u2 = y_u2.argmax(dim=1)[0].cpu().numpy()

    img_l = image_labeled[0, 0].cpu().numpy()
    img_u = image_unlabeled[0, 0].cpu().numpy()
    gt = label_labeled[0].cpu().numpy()

    # Decoder feature magnitude (L2 norm over channel dim)
    feat_mag_l1 = dx_l1[0].norm(dim=0).cpu().numpy()  # [56, 56]
    feat_mag_u1 = dx_u1[0].norm(dim=0).cpu().numpy()

    fig, axes = plt.subplots(2, 5, figsize=(22, 9))

    # Row 0: Labeled
    axes[0, 0].imshow(img_l, cmap='gray')
    axes[0, 0].set_title('$\\mathbf{x}_\\ell$ (labeled)', fontsize=11)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(class_to_rgb(gt.astype(int)))
    axes[0, 1].set_title('$\\mathbf{y}_\\ell$ (GT)', fontsize=11)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(class_to_rgb(pred_l1))
    axes[0, 2].set_title('$\\hat{\\mathbf{y}}^1_\\ell$ (Θ₁ pred)', fontsize=11)
    axes[0, 2].axis('off')

    axes[0, 3].imshow(class_to_rgb(pred_l2))
    axes[0, 3].set_title('$\\hat{\\mathbf{y}}^2_\\ell$ (Θ₂ pred)', fontsize=11)
    axes[0, 3].axis('off')

    axes[0, 4].imshow(feat_mag_l1, cmap='inferno')
    axes[0, 4].set_title('$\\|\\mathbf{g}^1_\\ell\\|_2$ (decoder feat)', fontsize=11)
    axes[0, 4].axis('off')

    # Row 1: Unlabeled
    axes[1, 0].imshow(img_u, cmap='gray')
    axes[1, 0].set_title('$\\mathbf{x}_u$ (unlabeled)', fontsize=11)
    axes[1, 0].axis('off')

    axes[1, 1].set_visible(False)  # no GT for unlabeled

    axes[1, 2].imshow(class_to_rgb(pred_u1))
    axes[1, 2].set_title('$\\hat{\\mathbf{y}}^1_u$ (Θ₁ pred)', fontsize=11)
    axes[1, 2].axis('off')

    axes[1, 3].imshow(class_to_rgb(pred_u2))
    axes[1, 3].set_title('$\\hat{\\mathbf{y}}^2_u$ (Θ₂ pred)', fontsize=11)
    axes[1, 3].axis('off')

    axes[1, 4].imshow(feat_mag_u1, cmap='inferno')
    axes[1, 4].set_title('$\\|\\mathbf{g}^1_u\\|_2$ (decoder feat)', fontsize=11)
    axes[1, 4].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'panel_a', 'architecture_outputs.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'panel_a', 'architecture_outputs.pdf'),
                bbox_inches='tight')
    plt.close()

    # Save individual images for figure compositing
    for name, arr in [('img_labeled', img_l), ('img_unlabeled', img_u),
                      ('gt', gt), ('pred_l1', pred_l1), ('pred_l2', pred_l2),
                      ('pred_u1', pred_u1), ('pred_u2', pred_u2)]:
        np.save(os.path.join(output_dir, 'panel_a', f'{name}.npy'), arr)

    for name, arr in [('feat_mag_l1', feat_mag_l1), ('feat_mag_u1', feat_mag_u1)]:
        np.save(os.path.join(output_dir, 'panel_a', f'{name}.npy'), arr)


# ====================== t-SNE VISUALIZATION ======================

def extract_tsne(model1, model2, device, test_data_dir, test_data_list,
                 image_size, num_classes, output_dir, max_samples=5000):
    """
    Extract decoder features from test data and generate t-SNE plot.
    Uses exact same test loop structure as train.py testing().
    """
    os.makedirs(os.path.join(output_dir, 'tsne'), exist_ok=True)
    from sklearn.manifold import TSNE

    data_list = get_image_list(test_data_list)
    all_feats = []
    all_labels = []

    for index in range(len(data_list['image_filenames'])):
        gt_name = data_list['label_filenames'][index]
        nib_gt = nib.load(os.path.join(test_data_dir, gt_name))
        gt = nib_gt.get_fdata()

        img_name = data_list['image_filenames'][index]
        nib_img = nib.load(os.path.join(test_data_dir, img_name))
        img = nib_img.get_fdata().astype('float32')

        clip_min = np.percentile(img, 1)
        clip_max = np.percentile(img, 99)
        img = np.clip(img, clip_min, clip_max)
        img = (img - img.min()) / float(img.max() - img.min())
        x, y, z = img.shape
        x_centre, y_centre = int(x / 2), int(y / 2)
        img = crop_image(img, x_centre, y_centre, image_size, constant_values=0)
        gt_crop = crop_image(gt, x_centre, y_centre, image_size, constant_values=0)

        for i in range(img.shape[2]):
            tmp_image = torch.from_numpy(img[:, :, i]).unsqueeze(0).unsqueeze(0).to(device)
            tmp_gt = gt_crop[:, :, i].astype(int)

            with torch.no_grad():
                _, dx1, _ = model1(tmp_image)
                _, dx2, _ = model2(tmp_image)

            # Average features, same as rscl.py line 219
            feat = ((dx1 + dx2) / 2.0)  # [1, 128, 56, 56]
            feat_np = F.normalize(feat, dim=1)[0].permute(1, 2, 0).cpu().numpy()  # [56, 56, 128]

            # Downsample GT to feature resolution
            gt_down = F.interpolate(
                torch.from_numpy(tmp_gt).float().unsqueeze(0).unsqueeze(0),
                size=(feat_np.shape[0], feat_np.shape[1]), mode='nearest'
            ).squeeze().numpy().astype(int)

            # Subsample pixels (skip background for cleaner plot)
            for c in range(1, num_classes):
                coords = np.argwhere(gt_down == c)
                if len(coords) > 0:
                    n_sample = min(50, len(coords))
                    idx = np.random.choice(len(coords), n_sample, replace=False)
                    for y_c, x_c in coords[idx]:
                        all_feats.append(feat_np[y_c, x_c])
                        all_labels.append(c)

        if len(all_feats) >= max_samples:
            break

    all_feats = np.array(all_feats[:max_samples])
    all_labels = np.array(all_labels[:max_samples])

    print(f"  t-SNE: {len(all_feats)} feature vectors, running...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
    coords_2d = tsne.fit_transform(all_feats)

    # Plot
    class_names = ['BG', 'RVC', 'Myo', 'LVC'][:num_classes]
    colors = ['#DC3C3C', '#3CB44B', '#4169E1']  # RVC, Myo, LVC

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    for c in range(1, num_classes):
        mask = all_labels == c
        ax.scatter(coords_2d[mask, 0], coords_2d[mask, 1],
                   c=colors[c - 1], label=class_names[c],
                   s=6, alpha=0.5, edgecolors='none')
    ax.legend(fontsize=11, markerscale=3)
    ax.set_title('t-SNE of decoder features (RSCL)', fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'tsne', 'tsne_decoder_features.png'),
                dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(output_dir, 'tsne', 'tsne_decoder_features.pdf'),
                bbox_inches='tight')
    plt.close()

    np.savez(os.path.join(output_dir, 'tsne', 'tsne_data.npz'),
             coords=coords_2d, labels=all_labels, features=all_feats)
    print("  t-SNE saved.")


# ====================== MAIN ======================

def main():
    args = parse_args()
    torch.cuda.set_device(args.device)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("=" * 60)
    print("RSCL Visualization Extraction")
    print("=" * 60)

    # Load models
    model1, model2 = load_models(args.checkpoint_dir, args.num_classes, device)
    rscl_module = load_rscl(args.num_classes, device)

    # Load a batch of data for visualization
    # Use validation set (has both image and GT)
    valset = SemiSegDataset_1(
        args.data_dir, args.train_data_csv, args.valid_data_csv,
        args.image_size, label_ratio=args.label_ratio, mode='valid',
        random_seed=args.seed)

    # Pick a representative sample (mid-ventricular slice typically has all structures)
    sample_idx = np.random.choice(len(valset),len(valset),replace=False)[0]
    image_labeled, label_labeled = valset[sample_idx]
    image_labeled = image_labeled.unsqueeze(0)   # [1, 1, 224, 224]
    label_labeled = label_labeled.unsqueeze(0)   # [1, 224, 224]

    # For unlabeled: use a different sample from validation (pretend it's unlabeled)
    unlabeled_idx = np.random.choice(len(valset),len(valset),replace=False)[-1]
    image_unlabeled, _ = valset[unlabeled_idx]
    image_unlabeled = image_unlabeled.unsqueeze(0)  # [1, 1, 224, 224]

    # Simulate a late-training iteration for visualization
    cur_iter = 25000

    print("\n--- Panel A: Architecture outputs ---")
    extract_panel_a(model1, model2, image_labeled, label_labeled,
                    image_unlabeled, device, args.output_dir)

    print("\n--- Panel B: Reliability & zones ---")
    extract_panel_b(model1, model2, image_labeled, image_unlabeled,
                    label_labeled, device, cur_iter, args.max_iterations,
                    args.output_dir)

    print("\n--- Panel B: Zone evolution ---")
    extract_zone_evolution(model1, model2, image_unlabeled, device,
                           args.max_iterations, args.output_dir)

    print("\n--- Panel C: DGPC ---")
    extract_panel_c(model1, model2, rscl_module,
                    image_labeled, label_labeled,
                    image_unlabeled, device, cur_iter, args.max_iterations,
                    args.output_dir)

    print("\n--- t-SNE ---")
    extract_tsne(model1, model2, device, args.test_data_dir,
                 args.test_data_list, args.image_size,
                 args.num_classes, args.output_dir)

    print("\n" + "=" * 60)
    print(f"All outputs saved to {args.output_dir}/")
    print("  panel_a/  - Input images, predictions, decoder features")
    print("  panel_b/  - Reliability maps, zone stratification, evolution")
    print("  panel_c/  - Memory bank, feat-proto similarity, hard vs soft")
    print("  tsne/     - t-SNE of decoder features")
    print("=" * 60)


if __name__ == '__main__':
    main()
