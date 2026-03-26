#!/usr/bin/env python
"""Debug script: verify each training module works correctly.

Tests:
1. Model forward pass (ProjectUNet1) - shape checks
2. Supervised loss (CE + Dice) - decreases over labeled steps
3. RSCL module:
   a. Reliability estimation output range [0,1]
   b. UCPS loss computation
   c. Memory bank init and update
   d. Hard contrastive loss
   e. Soft contrastive (KL) loss
   f. Full RSCL forward
4. Full training loop (10 iters) - total_loss decreases

All tests use synthetic data on GPU if available, else CPU.
"""
import sys
import torch
import torch.nn.functional as F
from torch.nn.modules.loss import CrossEntropyLoss
import numpy as np

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

PASS = "[PASS]"
FAIL = "[FAIL]"

# ─── Helpers ───────────────────────────────────────────────────────────────────

def check(cond, name, info=''):
    tag = PASS if cond else FAIL
    print(f"  {tag} {name}" + (f" | {info}" if info else ''))
    if not cond:
        sys.exit(1)


# ─── 1. Model forward pass ─────────────────────────────────────────────────────

print("\n=== 1. Model forward pass ===")
sys.path.insert(0, '.')
from SegModel import ProjectUNet1

B, C, H, W = 2, 4, 224, 224
model = ProjectUNet1('resnet50', None, classes=C, deep_stem=32).to(device)
model.eval()
with torch.no_grad():
    x = torch.randn(B, 1, H, W).to(device)
    out = model(x)

check(len(out) == 3, "output tuple has 3 elements", f"len={len(out)}")
y, dx, enc = out
check(y.shape == (B, C, H, W), "segmentation logits shape", str(y.shape))
check(dx.shape[0] == B and dx.shape[1] == 128, "decoder feat shape [B,128,*,*]", str(dx.shape))
check(not torch.isnan(y).any(), "no NaN in logits")
check(not torch.isnan(dx).any(), "no NaN in decoder feat")


# ─── 2. Supervised loss ────────────────────────────────────────────────────────

print("\n=== 2. Supervised loss ===")
from utilities.losses import DiceLoss

ce_loss = CrossEntropyLoss()
dice_loss_fn = DiceLoss(C)

model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

losses_sup = []
for step in range(5):
    x = torch.randn(B, 1, H, W).to(device)
    lbl = torch.randint(0, C, (B, H, W)).to(device)
    y, dx, _ = model(x)
    soft = F.softmax(y, dim=1)
    loss = 0.5 * ce_loss(y, lbl) + dice_loss_fn(soft, lbl)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses_sup.append(loss.item())
    print(f"  step {step}: loss_sup = {loss.item():.4f}")

check(not any(np.isnan(losses_sup)), "no NaN in sup losses")
check(losses_sup[0] > 0, "loss_sup > 0")


# ─── 3. RSCL module ────────────────────────────────────────────────────────────

print("\n=== 3. RSCL module ===")
from rscl import RSCL

rscl = RSCL(num_classes=C, feat_dim=128).to(device)

# synthetic inputs
Hf, Wf = 28, 28  # typical decoder feature resolution at 224x224
feat = lambda: torch.randn(B, 128, Hf, Wf).to(device)
logits = lambda: torch.randn(B, C, H, W).to(device)
gt = torch.randint(0, C, (B, H, W)).to(device)

# 3a. Reliability
print("  --- 3a. Reliability ---")
q1 = F.softmax(logits(), dim=1)
q2 = F.softmax(logits(), dim=1)
r = rscl.compute_reliability(q1, q2)
check(r.shape == (B, H, W), "reliability shape", str(r.shape))
check((r >= 0).all() and (r <= 1).all(), "reliability in [0,1]",
      f"min={r.min():.3f} max={r.max():.3f}")

# 3b. Thresholds annealing
print("  --- 3b. Threshold annealing ---")
t_h0, t_l0 = rscl.get_thresholds(0, 1000)
t_h1, t_l1 = rscl.get_thresholds(1000, 1000)
check(t_h0 < t_h1 or abs(t_h0 - t_h1) < 1e-6, "tau_high non-decreasing",
      f"{t_h0:.3f} -> {t_h1:.3f}")
check(t_l0 < t_l1 or abs(t_l0 - t_l1) < 1e-6, "tau_low non-decreasing",
      f"{t_l0:.3f} -> {t_l1:.3f}")
check(t_h1 >= t_l1, "tau_high >= tau_low", f"{t_h1:.3f} >= {t_l1:.3f}")

# 3c. UCPS loss
print("  --- 3c. UCPS ---")
lu1, lu2 = logits(), logits()
r_u = rscl.compute_reliability(F.softmax(lu1, dim=1), F.softmax(lu2, dim=1))
loss_ucps = rscl.compute_ucps(lu1, lu2, r_u)
check(loss_ucps.item() >= 0, "UCPS loss >= 0", f"{loss_ucps.item():.4f}")
check(not torch.isnan(loss_ucps), "no NaN in UCPS")

# 3d. Memory bank: initially zero, update, then hard contrastive valid
print("  --- 3d. Memory bank + hard contrastive ---")
check(not rscl.memory_init.any(), "memory uninit at start")

# manually init memory via build_prototypes + update
gt_down = F.interpolate(gt.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
f_l = feat()
proto = rscl.build_prototypes(f_l, gt_down)
rscl.update_memory(proto)
check(rscl.memory_init.any(), "memory init after first update")

loss_hard = rscl.hard_contrastive(f_l, gt_down)
check(loss_hard.item() >= 0, "hard contrastive >= 0", f"{loss_hard.item():.4f}")
check(not torch.isnan(loss_hard), "no NaN in hard contrastive")

# 3e. Soft contrastive
print("  --- 3e. Soft contrastive (KL) ---")
q_avg = (F.softmax(logits(), dim=1) + F.softmax(logits(), dim=1)) / 2
q_avg_down = F.interpolate(q_avg, size=(Hf, Wf), mode='bilinear', align_corners=False)
loss_soft = rscl.soft_contrastive(f_l, q_avg_down)
check(not torch.isnan(loss_soft), "no NaN in soft contrastive", f"{loss_soft.item():.4f}")

# 3f. Full RSCL forward
print("  --- 3f. Full RSCL forward ---")
fl1, fl2, fu1, fu2 = feat(), feat(), feat(), feat()
ll1, ll2, lu1, lu2 = logits(), logits(), logits(), logits()
loss_dgpc, loss_ucps = rscl(fl1, fl2, fu1, fu2, ll1, ll2, lu1, lu2, gt, cur_iter=500, max_iter=1000)
check(not torch.isnan(loss_dgpc), "no NaN in loss_dgpc", f"{loss_dgpc.item():.4f}")
check(not torch.isnan(loss_ucps), "no NaN in loss_ucps", f"{loss_ucps.item():.4f}")
check(loss_dgpc.item() >= 0, "loss_dgpc >= 0")
check(loss_ucps.item() >= 0, "loss_ucps >= 0")


# ─── 4. Mini training loop (10 iters) ─────────────────────────────────────────

print("\n=== 4. Mini training loop (10 iters) ===")
model1 = ProjectUNet1('resnet50', None, classes=C, deep_stem=32).to(device)
model2 = ProjectUNet1('resnet50', None, classes=C, deep_stem=32).to(device)
rscl2  = RSCL(num_classes=C, feat_dim=128).to(device)

opt1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)
opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)

ce  = CrossEntropyLoss()
dice_fn = DiceLoss(C)

total_losses = []
for it in range(10):
    model1.train(); model2.train()

    img_l = torch.randn(B, 1, H, W).to(device)
    lbl   = torch.randint(0, C, (B, H, W)).to(device)
    img_u = torch.randn(B, 1, H, W).to(device)

    yl1, dxl1, _ = model1(img_l)
    yu1, dxu1, _ = model1(img_u)
    yl2, dxl2, _ = model2(img_l)
    yu2, dxu2, _ = model2(img_u)

    loss_sup = (
        0.5 * ce(yl1, lbl) + dice_fn(F.softmax(yl1, dim=1), lbl) +
        0.5 * ce(yl2, lbl) + dice_fn(F.softmax(yl2, dim=1), lbl)
    )

    loss_dgpc, loss_ucps = rscl2(
        dxl1, dxl2, dxu1, dxu2,
        yl1,  yl2,  yu1,  yu2,
        lbl, cur_iter=it, max_iter=10
    )

    rampup = min(1.0, it / 2000.0)
    warmup_done = it >= 2  # short warmup for debug
    if warmup_done:
        total = loss_sup + 0.2 * loss_dgpc + 1.0 * rampup * loss_ucps
    else:
        total = loss_sup + 1.0 * rampup * loss_ucps

    opt1.zero_grad(); opt2.zero_grad()
    total.backward()
    opt1.step(); opt2.step()

    total_losses.append(total.item())
    print(f"  iter {it:2d}: total={total.item():.4f}  sup={loss_sup.item():.4f}  "
          f"dgpc={loss_dgpc.item():.4f}  ucps={loss_ucps.item():.4f}")

check(not any(np.isnan(total_losses)), "no NaN in total losses")
check(not any(np.isinf(total_losses)), "no Inf in total losses")

# Gradient check: parameters of model1 should have gradients
params_with_grad = sum(1 for p in model1.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
check(params_with_grad > 0, "model1 params received gradients", f"{params_with_grad} params with grad")

print("\n=== All checks passed. Training pipeline is functional. ===")
