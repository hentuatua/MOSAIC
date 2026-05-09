from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionStage1Loss(nn.Module):
    def __init__(
        self,
        l1_text_feats: torch.Tensor,
        l2_text_feats: torch.Tensor,
        l1_to_l2_idxs: Dict[int, List[int]],
        l2_idx_to_l1_idx: Dict[int, int],
        tau_rel: float = 0.07,
        tau_div: float = 0.07,
        tau_set: float = 0.10,
        w_div_cls: float = 0.2,
        w_div_cons: float = 0.2,
        w_rel_sib: float = 0.7,
        w_fuse: float = 1.0,
        w_fuse_sib: float = 0.5,
        w_orth: float = 0.1,
        sibling_positive: bool = False,
    ):
        super().__init__()
        self.tau_rel = tau_rel
        self.tau_div = tau_div
        self.tau_set = tau_set
        self.w_div_cls = w_div_cls
        self.w_div_cons = w_div_cons
        self.w_rel_sib = w_rel_sib
        self.w_fuse = w_fuse
        self.w_fuse_sib = w_fuse_sib
        self.w_orth = w_orth
        self.sibling_positive = bool(sibling_positive)
        self.register_buffer('l1_text_feats', F.normalize(l1_text_feats.clone(), dim=-1))
        self.register_buffer('l2_text_feats', F.normalize(l2_text_feats.clone(), dim=-1))
        self.l1_to_l2_idxs = {int(k): list(map(int, v)) for k, v in l1_to_l2_idxs.items()}
        self.l2_idx_to_l1_idx = {int(k): int(v) for k, v in l2_idx_to_l1_idx.items()}
        self.l2_local_idx = {}
        for l1_idx, l2_list in self.l1_to_l2_idxs.items():
            for local_idx, global_l2 in enumerate(l2_list):
                self.l2_local_idx[int(global_l2)] = int(local_idx)
        residual_bank = []
        for global_l2 in range(self.l2_text_feats.size(0)):
            parent_l1 = self.l2_idx_to_l1_idx[global_l2]
            residual_bank.append(
                self._orth_project(
                    self.l2_text_feats[global_l2:global_l2 + 1],
                    self.l1_text_feats[parent_l1:parent_l1 + 1],
                )[0]
            )
        self.register_buffer('l2_residual_text_feats', torch.stack(residual_bank, dim=0))

    def _orth_project(self, feats: torch.Tensor, query_feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if feats.dim() == 2:
            residual = feats - (feats * query_feat).sum(dim=-1, keepdim=True) * query_feat
        else:
            residual = feats - torch.einsum('bnd,bd->bn', feats, query_feat).unsqueeze(-1) * query_feat.unsqueeze(1)
        return F.normalize(residual, dim=-1, eps=eps)

    def _branch_relevance_loss(
        self,
        query_feat: torch.Tensor,
        pos_a: torch.Tensor,
        pos_b: torch.Tensor,
        sibling_neg: torch.Tensor,
        global_neg: torch.Tensor,
        sib_weight: float,
    ):
        pos_parts = [pos_a, pos_b]
        if self.sibling_positive:
            pos_parts.append(sibling_neg)
        pos = torch.cat(pos_parts, dim=1)
        pos_logits = torch.einsum('bnd,bd->bn', pos, query_feat) / self.tau_rel
        global_logits = torch.einsum('bnd,bd->bn', global_neg, query_feat) / self.tau_rel

        pos_log = torch.logsumexp(pos_logits, dim=1)
        global_all_log = torch.logsumexp(torch.cat([pos_logits, global_logits], dim=1), dim=1)
        loss_global = -(pos_log - global_all_log).mean()
        if self.sibling_positive:
            loss_sibling = loss_global.new_zeros(())
            total = loss_global
        else:
            sibling_logits = torch.einsum('bnd,bd->bn', sibling_neg, query_feat) / self.tau_rel
            loss_sibling = F.softplus(torch.logsumexp(sibling_logits, dim=1) - pos_log).mean()
            total = loss_global + sib_weight * loss_sibling
        return total, loss_global, loss_sibling

    def diversity_cls_loss(
        self,
        set_div_a: torch.Tensor,
        set_div_b: torch.Tensor,
        l1_idx: torch.Tensor,
        l2_idx: torch.Tensor,
        set_div_sib: torch.Tensor = None,
        sibling_l2_idx: torch.Tensor = None,
    ) -> torch.Tensor:
        losses = []
        for set_div in [set_div_a, set_div_b]:
            per_sample = []
            for i in range(set_div.size(0)):
                cur_l1 = int(l1_idx[i].item())
                cur_l2 = int(l2_idx[i].item())
                candidate_ids = self.l1_to_l2_idxs[cur_l1]
                candidate_feats = self.l2_residual_text_feats[candidate_ids]
                logits = set_div[i:i + 1] @ candidate_feats.t() / self.tau_div
                target = torch.tensor([self.l2_local_idx[cur_l2]], device=set_div.device, dtype=torch.long)
                per_sample.append(F.cross_entropy(logits, target))
            losses.append(torch.stack(per_sample).mean())
        if set_div_sib is not None and sibling_l2_idx is not None:
            per_sample = []
            for i in range(set_div_sib.size(0)):
                cur_sib = int(sibling_l2_idx[i].item())
                if cur_sib < 0:
                    continue
                cur_l1 = int(l1_idx[i].item())
                candidate_ids = self.l1_to_l2_idxs[cur_l1]
                candidate_feats = self.l2_residual_text_feats[candidate_ids]
                logits = set_div_sib[i:i + 1] @ candidate_feats.t() / self.tau_div
                target = torch.tensor([self.l2_local_idx[cur_sib]], device=set_div_sib.device, dtype=torch.long)
                per_sample.append(F.cross_entropy(logits, target))
            if per_sample:
                losses.append(torch.stack(per_sample).mean())
        return torch.stack(losses).mean()

    def diversity_consistency_loss(
        self,
        set_div_a: torch.Tensor,
        set_div_b: torch.Tensor,
        set_div_sib: torch.Tensor,
    ) -> torch.Tensor:
        pos = (set_div_a * set_div_b).sum(dim=-1) / self.tau_set
        neg_a = (set_div_a * set_div_sib).sum(dim=-1) / self.tau_set
        neg_b = (set_div_b * set_div_sib).sum(dim=-1) / self.tau_set
        logits_a = torch.stack([pos, neg_a], dim=-1)
        logits_b = torch.stack([pos, neg_b], dim=-1)
        labels = torch.zeros(set_div_a.size(0), dtype=torch.long, device=set_div_a.device)
        return 0.5 * (F.cross_entropy(logits_a, labels) + F.cross_entropy(logits_b, labels))

    def orthogonality_loss(
        self,
        query_feat: torch.Tensor,
        set_div_a: torch.Tensor,
        set_div_b: torch.Tensor,
    ) -> torch.Tensor:
        qa = (set_div_a * query_feat).sum(dim=-1).pow(2).mean()
        qb = (set_div_b * query_feat).sum(dim=-1).pow(2).mean()
        return 0.5 * (qa + qb)

    def forward(
        self,
        query_feat: torch.Tensor,
        pos_rel_a: torch.Tensor,
        pos_rel_b: torch.Tensor,
        sibling_rel: torch.Tensor,
        global_rel: torch.Tensor,
        pos_fused_a: torch.Tensor,
        pos_fused_b: torch.Tensor,
        sibling_fused: torch.Tensor,
        global_fused: torch.Tensor,
        set_div_a: torch.Tensor,
        set_div_b: torch.Tensor,
        set_div_sib: torch.Tensor,
        l1_idx: torch.Tensor,
        l2_idx: torch.Tensor,
        sibling_l2_idx: torch.Tensor = None,
    ):
        loss_rel, loss_rel_global, loss_rel_sib = self._branch_relevance_loss(
            query_feat,
            pos_rel_a,
            pos_rel_b,
            sibling_rel,
            global_rel,
            self.w_rel_sib,
        )
        loss_fuse, loss_fuse_global, loss_fuse_sib = self._branch_relevance_loss(
            query_feat,
            pos_fused_a,
            pos_fused_b,
            sibling_fused,
            global_fused,
            self.w_fuse_sib,
        )
        loss_div_cls = self.diversity_cls_loss(set_div_a, set_div_b, l1_idx, l2_idx, set_div_sib, sibling_l2_idx)
        loss_div_cons = self.diversity_consistency_loss(set_div_a, set_div_b, set_div_sib)
        loss_orth = self.orthogonality_loss(query_feat, set_div_a, set_div_b)

        total = (
            loss_rel
            + self.w_fuse * loss_fuse
            + self.w_div_cls * loss_div_cls
            + self.w_div_cons * loss_div_cons
            + self.w_orth * loss_orth
        )
        return total, {
            'rel': float(loss_rel.detach().item()),
            'rel_global': float(loss_rel_global.detach().item()),
            'rel_sib': float(loss_rel_sib.detach().item()),
            'fuse': float(loss_fuse.detach().item()),
            'fuse_global': float(loss_fuse_global.detach().item()),
            'fuse_sib': float(loss_fuse_sib.detach().item()),
            'div_cls': float(loss_div_cls.detach().item()),
            'div_cons': float(loss_div_cons.detach().item()),
            'orth': float(loss_orth.detach().item()),
            'total': float(total.detach().item()),
        }


class ModeContrastiveDiversityLoss(nn.Module):
    def __init__(
        self,
        tau_mode: float = 0.07,
        w_scl: float = 0.3,
        w_proto: float = 0.3,
        w_spread: float = 0.1,
        max_positives: int = 0,
    ):
        super().__init__()
        self.tau_mode = tau_mode
        self.w_scl = w_scl
        self.w_proto = w_proto
        self.w_spread = w_spread
        self.max_positives = max_positives

    @staticmethod
    def _orth_project(feats: torch.Tensor, query_feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        residual = feats - torch.einsum('nd,d->n', feats, query_feat).unsqueeze(-1) * query_feat.unsqueeze(0)
        return F.normalize(residual, dim=-1, eps=eps)

    def _supcon_loss(self, div_feats: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        N = div_feats.size(0)
        if N < 2:
            return div_feats.new_zeros(())

        sim = torch.matmul(div_feats, div_feats.t()) / self.tau_mode
        self_mask = torch.eye(N, dtype=torch.bool, device=div_feats.device)
        sim = sim.masked_fill(self_mask, -1e4)

        pos_mask = (group_ids.unsqueeze(0) == group_ids.unsqueeze(1)) & ~self_mask
        has_pos = pos_mask.any(dim=1)
        if not has_pos.any():
            return div_feats.new_zeros(())

        log_denom = torch.logsumexp(sim, dim=1)

        neg_inf = torch.full_like(sim, -1e4)
        pos_sim = torch.where(pos_mask, sim, neg_inf)
        log_pos_sum = torch.logsumexp(pos_sim, dim=1)
        num_pos = pos_mask.float().sum(dim=1).clamp_min(1.0)
        loss_per_row = -log_pos_sum + torch.log(num_pos) + log_denom

        return loss_per_row[has_pos].mean()

    def _prototype_discrimination_loss(
        self, div_feats: torch.Tensor, group_ids: torch.Tensor, unique_groups: torch.Tensor,
    ) -> torch.Tensor:
        K = unique_groups.numel()
        if K <= 1:
            return div_feats.new_zeros(())

        proto_list = []
        targets = torch.zeros(div_feats.size(0), dtype=torch.long, device=div_feats.device)
        for local_idx, gid in enumerate(unique_groups.tolist()):
            mask = group_ids == gid
            proto = F.normalize(div_feats[mask].mean(dim=0, keepdim=True), dim=-1, eps=1e-12)
            proto_list.append(proto[0])
            targets[mask] = local_idx
        proto_bank = torch.stack(proto_list, dim=0)

        logits = torch.matmul(div_feats, proto_bank.t()) / self.tau_mode
        return F.cross_entropy(logits, targets)

    def _mode_spread_loss(
        self, div_feats: torch.Tensor, group_ids: torch.Tensor, unique_groups: torch.Tensor,
    ) -> torch.Tensor:
        K = unique_groups.numel()
        if K <= 1:
            return div_feats.new_zeros(())

        proto_list = []
        for gid in unique_groups.tolist():
            mask = group_ids == gid
            proto = F.normalize(div_feats[mask].mean(dim=0, keepdim=True), dim=-1, eps=1e-12)
            proto_list.append(proto[0])
        proto_bank = torch.stack(proto_list, dim=0)

        sim_matrix = torch.matmul(proto_bank, proto_bank.t())
        mask = ~torch.eye(K, dtype=torch.bool, device=div_feats.device)
        pairwise_sim = sim_matrix[mask]
        return pairwise_sim.mean()

    def forward(
        self,
        query_feat: torch.Tensor,
        div_bank: torch.Tensor,
        positive_indices: torch.Tensor,
        positive_group_ids: torch.Tensor,
    ):
        total_loss = query_feat.new_zeros(())
        scl_sum, proto_sum, spread_sum = 0.0, 0.0, 0.0
        valid_count = 0

        for row in range(query_feat.size(0)):
            valid = positive_indices[row] >= 0
            if not valid.any():
                continue
            pos_ids = positive_indices[row, valid]
            group_ids = positive_group_ids[row, valid]
            unique_groups = torch.unique(group_ids, sorted=True)
            if unique_groups.numel() <= 1:
                continue

            if self.max_positives > 0 and pos_ids.size(0) > self.max_positives:
                perm = torch.randperm(pos_ids.size(0), device=pos_ids.device)[:self.max_positives]
                pos_ids = pos_ids[perm]
                group_ids = group_ids[perm]
                unique_groups = torch.unique(group_ids, sorted=True)
                if unique_groups.numel() <= 1:
                    continue

            q = query_feat[row]
            pos_div = div_bank.index_select(0, pos_ids)
            pos_div_orth = self._orth_project(pos_div, q)

            scl = self._supcon_loss(pos_div_orth, group_ids)
            proto = self._prototype_discrimination_loss(pos_div_orth, group_ids, unique_groups)
            spread = self._mode_spread_loss(pos_div_orth, group_ids, unique_groups)

            row_loss = self.w_scl * scl + self.w_proto * proto + self.w_spread * spread
            total_loss = total_loss + row_loss
            scl_sum += float(scl.detach().item())
            proto_sum += float(proto.detach().item())
            spread_sum += float(spread.detach().item())
            valid_count += 1

        if valid_count > 0:
            total_loss = total_loss / valid_count

        return total_loss, {
            'mode_scl': scl_sum / max(valid_count, 1),
            'mode_proto': proto_sum / max(valid_count, 1),
            'mode_spread': spread_sum / max(valid_count, 1),
            'mode_total': float(total_loss.detach().item()),
        }
