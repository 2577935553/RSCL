import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class RSCL(nn.Module):
    """Reliability-Stratified Contrastive Learning module.
    
    Module A: Reliability estimation + three-zone stratification + UCPS
    Module B: Dual-granularity prototype contrastive (hard InfoNCE + soft KL)
    """
    def __init__(self, num_classes, feat_dim=128, tau_hard=0.2, tau_soft=0.5,
                 ema_momentum=0.999, beta=0.7, alpha=0.5,
                 tau_high_init=0.50, tau_high_final=0.85,
                 tau_low_init=0.10, tau_low_final=0.30,
                 contrastive_mode='dual'):
        super().__init__()
        self.C = num_classes
        self.feat_dim = feat_dim
        self.tau_hard = tau_hard
        self.tau_soft = tau_soft
        self.ema_momentum = ema_momentum
        self.beta = beta
        self.alpha = alpha
        self.tau_high_init = tau_high_init
        self.tau_high_final = tau_high_final
        self.tau_low_init = tau_low_init
        self.tau_low_final = tau_low_final
        self.contrastive_mode = contrastive_mode  # dual/all_hard/all_hard_filtered/hard_only/soft_only

        # Shared EMA memory bank [C, feat_dim], L2-normalized
        self.register_buffer('memory', torch.zeros(num_classes, feat_dim))
        self.register_buffer('memory_init', torch.zeros(num_classes, dtype=torch.bool))

    # ===================== Module A: Reliability Estimation =====================

    def compute_reliability(self, q1, q2):
        """Compute per-pixel reliability score from two networks' softmax outputs.
        q1, q2: [B, C, H, W] softmax probabilities
        Returns: r [B, H, W] in [0, 1]
        """
        y1 = q1.argmax(dim=1)  # [B, H, W]
        y2 = q2.argmax(dim=1)

        # Agreement indicator
        agree = (y1 == y2).float()

        # Average max confidence
        conf = (q1.max(dim=1)[0] + q2.max(dim=1)[0]) / 2.0

        # JSD
        m = (q1 + q2) / 2.0
        kl1 = (q1 * (q1.clamp(min=1e-7).log() - m.clamp(min=1e-7).log())).sum(dim=1)
        kl2 = (q2 * (q2.clamp(min=1e-7).log() - m.clamp(min=1e-7).log())).sum(dim=1)
        jsd = (kl1 + kl2) / 2.0  # [B, H, W]

        r = agree * conf * torch.exp(-jsd)
        return r

    def get_thresholds(self, cur_iter, max_iter):
        """Cosine annealing schedule for adaptive thresholds."""
        ratio = cur_iter / max(max_iter, 1)
        tau_high = self.tau_high_final - (self.tau_high_final - self.tau_high_init) * math.cos(math.pi * ratio / 2)
        tau_low = self.tau_low_init + (self.tau_low_final - self.tau_low_init) * (1 - math.cos(math.pi * ratio / 2))
        return tau_high, tau_low

    def compute_ucps(self, logits1, logits2, r):
        """Reliability-weighted cross pseudo supervision.
        logits1, logits2: [B, C, H, W] raw logits
        r: [B, H, W] reliability scores
        Returns: scalar loss
        """
        pseudo2 = logits2.detach().argmax(dim=1)  # [B, H, W]
        pseudo1 = logits1.detach().argmax(dim=1)

        loss_1 = F.cross_entropy(logits1, pseudo2, reduction='none')  # [B, H, W]
        loss_2 = F.cross_entropy(logits2, pseudo1, reduction='none')

        r_detach = r.detach()
        weighted = r_detach * (loss_1 + loss_2)
        return weighted.mean()

    # ===================== Module B: Dual-Granularity Prototype Contrastive =====================

    def build_prototypes(self, feat, mask, weights=None):
        """Build per-class prototypes by weighted averaging.
        feat: [B, D, Hf, Wf]
        mask: [B, Hf, Wf] class labels (long)
        weights: [B, Hf, Wf] optional per-pixel weights
        Returns: dict {class_id: prototype_vector [D]}
        """
        B, D, Hf, Wf = feat.shape
        feat_flat = feat.permute(0, 2, 3, 1).reshape(-1, D)       # [N, D]
        mask_flat = mask.reshape(-1)                                # [N]
        w_flat = weights.reshape(-1) if weights is not None else torch.ones_like(mask_flat, dtype=feat.dtype)

        protos = {}
        for c in range(self.C):
            idx = (mask_flat == c)
            if idx.sum() < 1:
                continue
            w_c = w_flat[idx].unsqueeze(1)  # [n, 1]
            f_c = feat_flat[idx]            # [n, D]
            proto = (w_c * f_c).sum(dim=0) / (w_c.sum() + 1e-7)
            protos[c] = proto
        return protos

    @torch.no_grad()
    def update_memory(self, protos):
        """EMA update the shared memory bank."""
        for c, p in protos.items():
            p_norm = F.normalize(p, dim=0)
            if not self.memory_init[c]:
                self.memory[c] = p_norm
                self.memory_init[c] = True
            else:
                self.memory[c] = self.ema_momentum * self.memory[c] + (1 - self.ema_momentum) * p_norm
                self.memory[c] = F.normalize(self.memory[c], dim=0)

    def hard_contrastive(self, feat, labels):
        """InfoNCE loss: pull features toward their class prototype, push away from others.
        feat: [B, D, Hf, Wf]
        labels: [B, Hf, Wf] (long)
        Only on pixels where memory is initialized for their class.
        """
        if not self.memory_init.any():
            return torch.tensor(0.0, device=feat.device)

        B, D, Hf, Wf = feat.shape
        feat_flat = F.normalize(feat.permute(0, 2, 3, 1).reshape(-1, D), dim=1)  # [N, D]
        labels_flat = labels.reshape(-1)  # [N]

        # Only use pixels whose class prototype is initialized
        valid = torch.zeros_like(labels_flat, dtype=torch.bool)
        for c in range(self.C):
            if self.memory_init[c]:
                valid |= (labels_flat == c)
        if valid.sum() < 1:
            return torch.tensor(0.0, device=feat.device)

        feat_sel = feat_flat[valid]       # [M, D]
        labels_sel = labels_flat[valid]   # [M]

        # Similarity to all prototypes [M, C]
        proto_mat = self.memory  # [C, D]
        logits = feat_sel @ proto_mat.t() / self.tau_hard  # [M, C]

        loss = F.cross_entropy(logits, labels_sel)
        return loss

    def soft_contrastive(self, feat, q_avg):
        """KL divergence: align feature-to-prototype distribution with averaged prediction.
        feat: [B, D, Hf, Wf]
        q_avg: [B, C, Hf, Wf] averaged softmax from both networks (soft target)
        Only on learning zone pixels (caller handles masking).
        """
        if not self.memory_init.any():
            return torch.tensor(0.0, device=feat.device)

        B, D, Hf, Wf = feat.shape
        feat_flat = F.normalize(feat.permute(0, 2, 3, 1).reshape(-1, D), dim=1)  # [N, D]
        q_flat = q_avg.permute(0, 2, 3, 1).reshape(-1, self.C)  # [N, C]

        # Feature-to-prototype similarity distribution
        proto_mat = self.memory  # [C, D]
        sim_logits = feat_flat @ proto_mat.t() / self.tau_soft  # [N, C]
        sim_dist = F.log_softmax(sim_logits, dim=1)

        # KL(q_avg || sim_dist)
        loss = F.kl_div(sim_dist, q_flat.detach(), reduction='batchmean')
        return loss

    # ===================== Main Forward =====================

    def forward(self, feat_l1, feat_l2, feat_u1, feat_u2,
                logits_l1, logits_l2, logits_u1, logits_u2,
                gt_labels, cur_iter, max_iter):
        """
        feat_*: decoder features [B, 128, Hf, Wf]
        logits_*: segmentation logits [B, C, H, W]
        gt_labels: ground truth for labeled data [B, H, W] (long)
        Returns: loss_dgpc, loss_ucps (scalars)
        """
        Hf, Wf = feat_l1.shape[2], feat_l1.shape[3]

        # --- Softmax predictions ---
        q_l1 = F.softmax(logits_l1, dim=1)
        q_l2 = F.softmax(logits_l2, dim=1)
        q_u1 = F.softmax(logits_u1, dim=1)
        q_u2 = F.softmax(logits_u2, dim=1)

        # --- Reliability for unlabeled data ---
        r_u = self.compute_reliability(q_u1, q_u2)  # [B, H, W]
        tau_high, tau_low = self.get_thresholds(cur_iter, max_iter)

        # --- UCPS loss (Module A) ---
        loss_ucps = self.compute_ucps(logits_u1, logits_u2, r_u)

        # --- Downsample labels and reliability to feature resolution ---
        gt_down = F.interpolate(gt_labels.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()
        r_u_down = F.interpolate(r_u.unsqueeze(1), size=(Hf, Wf), mode='bilinear', align_corners=False).squeeze(1)

        # Pseudo-labels for unlabeled (consensus)
        pseudo_u1 = logits_u1.detach().argmax(dim=1)
        pseudo_u2 = logits_u2.detach().argmax(dim=1)
        pseudo_consensus = pseudo_u1.clone()
        disagree = (pseudo_u1 != pseudo_u2)
        pseudo_consensus[disagree] = 0  # background for disagreed pixels (will be excluded anyway)
        pseudo_u_down = F.interpolate(pseudo_consensus.unsqueeze(1).float(), size=(Hf, Wf), mode='nearest').squeeze(1).long()

        # --- Zone masks at feature resolution ---
        anchor_mask = (r_u_down > tau_high)   # [B, Hf, Wf]
        learn_mask = (r_u_down > tau_low) & (~anchor_mask)

        # --- Prototype construction ---
        # Average features from both networks for stability
        feat_l = (feat_l1 + feat_l2) / 2.0
        feat_u = (feat_u1 + feat_u2) / 2.0

        # Labeled prototypes (from GT, all pixels)
        proto_labeled = self.build_prototypes(feat_l, gt_down)

        # Unlabeled prototypes (anchor zone only, weighted by reliability)
        anchor_r = r_u_down.clone()
        anchor_r[~anchor_mask] = 0.0
        proto_unlabeled = self.build_prototypes(feat_u, pseudo_u_down, weights=anchor_r)

        # Merge and update memory
        merged = {}
        for c in range(self.C):
            vecs = []
            if c in proto_labeled:
                vecs.append(proto_labeled[c])
            if c in proto_unlabeled:
                vecs.append(proto_unlabeled[c])
            if vecs:
                merged[c] = self.beta * vecs[0] + (1 - self.beta) * vecs[-1] if len(vecs) == 2 else vecs[0]
        self.update_memory(merged)

        # --- Hard contrastive: labeled (all) + unlabeled (mode-dependent) ---
        loss_hard_l = (self.hard_contrastive(feat_l1, gt_down) + self.hard_contrastive(feat_l2, gt_down)) / 2.0

        mode = self.contrastive_mode
        zero = torch.tensor(0.0, device=feat_l1.device)

        if mode == 'all_hard':
            # All unlabeled pixels get hard contrastive (no filtering)
            loss_hard_u = (self.hard_contrastive(feat_u1, pseudo_u_down) +
                           self.hard_contrastive(feat_u2, pseudo_u_down)) / 2.0
            loss_soft = zero

        elif mode == 'all_hard_filtered':
            # Anchor + learning zones get hard, exclusion filtered out
            valid_mask = anchor_mask | learn_mask
            filtered_labels = pseudo_u_down.clone()
            filtered_labels[~valid_mask] = 255
            loss_hard_u = (self._hard_contrastive_masked(feat_u1, filtered_labels) +
                           self._hard_contrastive_masked(feat_u2, filtered_labels)) / 2.0
            loss_soft = zero

        elif mode == 'hard_only':
            # Only anchor zone gets hard, learning zone excluded
            anchor_labels = pseudo_u_down.clone()
            anchor_labels[~anchor_mask] = 255
            loss_hard_u = (self._hard_contrastive_masked(feat_u1, anchor_labels) +
                           self._hard_contrastive_masked(feat_u2, anchor_labels)) / 2.0 if anchor_mask.any() else zero
            loss_soft = zero

        elif mode == 'soft_only':
            # Anchor + learning zones get soft contrastive
            loss_hard_u = zero
            soft_mask = anchor_mask | learn_mask
            if soft_mask.any():
                q_avg_u = ((q_u1 + q_u2) / 2.0).detach()
                q_avg_u_down = F.interpolate(q_avg_u, size=(Hf, Wf), mode='bilinear', align_corners=False)
                loss_soft = (self._soft_contrastive_masked(feat_u1, q_avg_u_down, soft_mask) +
                             self._soft_contrastive_masked(feat_u2, q_avg_u_down, soft_mask)) / 2.0
            else:
                loss_soft = zero

        else:  # 'dual' (default)
            # Anchor zone → hard, Learning zone → soft, Exclusion → none
            anchor_labels = pseudo_u_down.clone()
            anchor_labels[~anchor_mask] = 255
            loss_hard_u = (self._hard_contrastive_masked(feat_u1, anchor_labels) +
                           self._hard_contrastive_masked(feat_u2, anchor_labels)) / 2.0 if anchor_mask.any() else zero

            if learn_mask.any():
                q_avg_u = ((q_u1 + q_u2) / 2.0).detach()
                q_avg_u_down = F.interpolate(q_avg_u, size=(Hf, Wf), mode='bilinear', align_corners=False)
                loss_soft = (self._soft_contrastive_masked(feat_u1, q_avg_u_down, learn_mask) +
                             self._soft_contrastive_masked(feat_u2, q_avg_u_down, learn_mask)) / 2.0
            else:
                loss_soft = zero

        loss_hard = loss_hard_l + loss_hard_u

        loss_dgpc = loss_hard + self.alpha * loss_soft
        return loss_dgpc, loss_ucps

    def _hard_contrastive_masked(self, feat, labels):
        """Hard contrastive on valid pixels only (labels != 255)."""
        if not self.memory_init.any():
            return torch.tensor(0.0, device=feat.device)

        B, D, Hf, Wf = feat.shape
        feat_flat = F.normalize(feat.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
        labels_flat = labels.reshape(-1)

        valid = (labels_flat != 255)
        for c in range(self.C):
            if not self.memory_init[c]:
                valid &= (labels_flat != c)

        if valid.sum() < 1:
            return torch.tensor(0.0, device=feat.device)

        feat_sel = feat_flat[valid]
        labels_sel = labels_flat[valid]
        logits = feat_sel @ self.memory.t() / self.tau_hard
        return F.cross_entropy(logits, labels_sel)

    def _soft_contrastive_masked(self, feat, q_avg, mask):
        """Soft contrastive on masked pixels only."""
        if not self.memory_init.any():
            return torch.tensor(0.0, device=feat.device)

        B, D, Hf, Wf = feat.shape
        feat_flat = F.normalize(feat.permute(0, 2, 3, 1).reshape(-1, D), dim=1)
        q_flat = q_avg.permute(0, 2, 3, 1).reshape(-1, self.C)
        mask_flat = mask.reshape(-1)

        if mask_flat.sum() < 1:
            return torch.tensor(0.0, device=feat.device)

        feat_sel = feat_flat[mask_flat]
        q_sel = q_flat[mask_flat]

        sim_logits = feat_sel @ self.memory.t() / self.tau_soft
        sim_dist = F.log_softmax(sim_logits, dim=1)
        return F.kl_div(sim_dist, q_sel.detach(), reduction='batchmean')
