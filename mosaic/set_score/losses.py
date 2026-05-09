from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class BudgetSetLoss(nn.Module):
    def __init__(
        self,
        weight_ret: float = 0.5,
        weight_cov: float = 1.0,
        weight_dup: float = 0.25,
        weight_proto: float = 0.1,
        weight_support: float = 0.0,
        weight_div: float = 0.0,
        cluster_alpha: float = 0.5,
        rank_inner_weight: float = 0.5,
        cluster_mass_weight: float = 0.5,
        slot_align_weight: float = 0.5,
        cluster_tau: float = 0.2,
        cluster_transport_iters: int = 12,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.weight_ret = weight_ret
        self.weight_cov = weight_cov
        self.weight_dup = weight_dup
        self.weight_proto = weight_proto
        self.weight_support = weight_support
        self.weight_div = weight_div
        self.cluster_alpha = cluster_alpha
        self.rank_inner_weight = rank_inner_weight
        self.cluster_mass_weight = cluster_mass_weight
        self.slot_align_weight = slot_align_weight
        self.cluster_tau = cluster_tau
        self.cluster_transport_iters = cluster_transport_iters
        self.eps = eps

    def _cluster_statistics(
        self,
        intent_states: torch.Tensor,
        img_feats: torch.Tensor,
        pos_mask: torch.Tensor,
        cluster: torch.Tensor,
    ):
        valid_cluster_mask = (cluster >= 0) & pos_mask
        if valid_cluster_mask.sum() == 0:
            return None
        group_ids = torch.unique(cluster[valid_cluster_mask])
        prototypes = []
        masks = []
        sizes = []
        for gid in group_ids.tolist():
            mask = (cluster == gid) & pos_mask
            prototypes.append(F.normalize(img_feats[mask].mean(dim=0, keepdim=True), dim=-1))
            masks.append(mask)
            sizes.append(mask.sum().to(img_feats.dtype))
        if not prototypes:
            return None
        prototypes = torch.cat(prototypes, dim=0)
        sizes = torch.stack(sizes).to(img_feats.device)
        sim = torch.matmul(intent_states, prototypes.t())
        return {
            'prototypes': prototypes,
            'masks': masks,
            'sizes': sizes,
            'sim': sim,
        }

    def _balanced_slot_cluster_plan(
        self,
        sim: torch.Tensor,
        slot_budgets: torch.Tensor,
        cluster_target: torch.Tensor,
    ) -> torch.Tensor:
        supply = slot_budgets.clamp_min(self.eps)
        demand = cluster_target.clamp_min(self.eps)
        kernel = torch.exp((sim - sim.amax()).clamp(min=-50.0) / self.cluster_tau).clamp_min(self.eps)
        u = torch.ones_like(supply)
        v = torch.ones_like(demand)
        for _ in range(self.cluster_transport_iters):
            u = supply / torch.matmul(kernel, v).clamp_min(self.eps)
            v = demand / torch.matmul(kernel.t(), u).clamp_min(self.eps)
        return u.unsqueeze(1) * kernel * v.unsqueeze(0)

    def _prototype_loss(
        self,
        intent_states: torch.Tensor,
        img_feats: torch.Tensor,
        pos_mask: torch.Tensor,
        cluster: torch.Tensor,
    ):
        stats = self._cluster_statistics(intent_states, img_feats, pos_mask, cluster)
        if stats is None:
            return intent_states.sum() * 0.0
        cluster_cover = stats['sim'].max(dim=0).values
        return (1.0 - cluster_cover).mean()

    def _coverage_loss(
        self,
        candidate_mass: torch.Tensor,
        transport: torch.Tensor,
        slot_budgets: torch.Tensor,
        intent_states: torch.Tensor,
        img_feats: torch.Tensor,
        pos_mask: torch.Tensor,
        cluster: torch.Tensor,
    ) -> torch.Tensor:
        stats = self._cluster_statistics(intent_states, img_feats, pos_mask, cluster)
        if stats is None:
            return candidate_mass.sum() * 0.0

        cluster_sizes = stats['sizes'].pow(self.cluster_alpha)
        cluster_target = slot_budgets.sum() * cluster_sizes / cluster_sizes.sum().clamp_min(self.eps)

        cluster_mass = []
        slot_cluster_mass = []
        for mask in stats['masks']:
            cluster_mass.append(candidate_mass[mask].sum())
            slot_cluster_mass.append(transport[:, mask].sum(dim=-1))
        cluster_mass = torch.stack(cluster_mass)
        slot_cluster_mass = torch.stack(slot_cluster_mass, dim=-1)

        cover_loss = F.relu(1.0 - cluster_mass).pow(2).mean()
        cluster_mass_loss = F.smooth_l1_loss(cluster_mass, cluster_target)
        cluster_plan = self._balanced_slot_cluster_plan(stats['sim'], slot_budgets, cluster_target)
        slot_align_loss = F.smooth_l1_loss(slot_cluster_mass, cluster_plan)

        return cover_loss + self.cluster_mass_weight * cluster_mass_loss + self.slot_align_weight * slot_align_loss

    def _binary_terms(self, logits: torch.Tensor, pos_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        pos_count = int(pos_mask.sum().item())
        neg_count = int((~pos_mask).sum().item())
        pos_weight = float(max(neg_count, 1)) / float(max(pos_count, 1))
        rel_loss = F.binary_cross_entropy_with_logits(
            logits,
            pos_mask.float(),
            pos_weight=torch.tensor(pos_weight, device=logits.device),
        )
        pos_logits = logits[pos_mask]
        neg_logits = logits[~pos_mask]
        if pos_logits.numel() > 0 and neg_logits.numel() > 0:
            rank_loss = F.softplus(neg_logits.unsqueeze(0) - pos_logits.unsqueeze(1)).mean()
        else:
            rank_loss = rel_loss.sum() * 0.0
        return {'rel': rel_loss, 'rank': rank_loss}

    def _support_loss(
        self,
        candidate_mass: torch.Tensor,
        support_probs: torch.Tensor,
        pos_mask: torch.Tensor,
    ) -> torch.Tensor:
        if pos_mask.sum() == 0:
            return candidate_mass.sum() * 0.0

        total_mass = candidate_mass.sum().clamp_min(self.eps)
        mass_dist = candidate_mass / total_mass
        support_dist = support_probs.clamp_min(self.eps)
        support_dist = (support_dist / support_dist.sum().clamp_min(self.eps)).detach()

        align_loss = torch.sum(
            mass_dist * (torch.log(mass_dist.clamp_min(self.eps)) - torch.log(support_dist))
        )
        neg_mass_ratio = candidate_mass[~pos_mask].sum() / total_mass if (~pos_mask).any() else total_mass * 0.0
        return align_loss + neg_mass_ratio

    def _sample_losses(self, outputs: Dict[str, torch.Tensor], quality: torch.Tensor, cluster: torch.Tensor) -> Dict[str, torch.Tensor]:
        candidate_mass = outputs['candidate_mass']
        transport = outputs['transport']
        slot_budgets = outputs['slot_budgets']
        rel_logits = outputs['rel_logits']
        support_probs = outputs.get('support_probs')
        intent_states = outputs['intent_states']
        img_feats = outputs['img_feats']

        pos_mask = quality > 0
        valid_cluster_mask = (cluster >= 0) & pos_mask
        zero = candidate_mass.sum() * 0.0
        binary_terms = self._binary_terms(rel_logits, pos_mask)
        dup_loss = F.relu(candidate_mass - 1.0).pow(2).mean()
        support_loss = self._support_loss(candidate_mass, support_probs, pos_mask) if support_probs is not None else zero

        if pos_mask.sum() == 0 or valid_cluster_mask.sum() == 0:
            return {
                'ret': binary_terms['rel'] + self.rank_inner_weight * binary_terms['rank'],
                'cov': zero,
                'dup': dup_loss,
                'proto': zero,
                'sup': support_loss,
                'div': zero,
            }

        cov_loss = self._coverage_loss(
            candidate_mass=candidate_mass,
            transport=transport,
            slot_budgets=slot_budgets,
            intent_states=intent_states,
            img_feats=img_feats,
            pos_mask=pos_mask,
            cluster=cluster,
        )
        proto_loss = self._prototype_loss(intent_states, img_feats, pos_mask, cluster)

        if self.weight_div > 0:
            mass_norm = candidate_mass / candidate_mass.sum().clamp_min(self.eps)
            feat_norm = F.normalize(img_feats.float(), dim=-1)
            feat_sim = torch.matmul(feat_norm, feat_norm.t())
            mass_outer = mass_norm.unsqueeze(0) * mass_norm.unsqueeze(1)
            div_loss = (mass_outer * feat_sim).sum() - mass_norm.pow(2).sum()
        else:
            div_loss = zero

        return {
            'ret': binary_terms['rel'] + self.rank_inner_weight * binary_terms['rank'],
            'cov': cov_loss,
            'dup': dup_loss,
            'proto': proto_loss,
            'sup': support_loss,
            'div': div_loss,
        }

    def forward(self, outputs: Dict[str, torch.Tensor], quality: torch.Tensor, cluster: torch.Tensor, pad_mask: torch.Tensor):
        metric_lists = {k: [] for k in ['ret', 'cov', 'dup', 'proto', 'sup', 'div']}
        for b in range(outputs['candidate_mass'].size(0)):
            valid = pad_mask[b].bool()
            sample_outputs = {
                'candidate_mass': outputs['candidate_mass'][b][valid],
                'transport': outputs['transport'][b][:, valid],
                'slot_budgets': outputs['slot_budgets'][b],
                'rel_logits': outputs['rel_logits'][b][valid],
                'support_probs': outputs['support_probs'][b][valid] if 'support_probs' in outputs else None,
                'intent_states': outputs['intent_states'][b],
                'img_feats': outputs['img_feats'][b][valid],
            }
            sample_terms = self._sample_losses(sample_outputs, quality[b, valid], cluster[b, valid])
            for k in metric_lists:
                metric_lists[k].append(sample_terms[k])

        metrics = {k: torch.stack(v).mean() for k, v in metric_lists.items()}
        total = (
            self.weight_ret * metrics['ret']
            + self.weight_cov * metrics['cov']
            + self.weight_dup * metrics['dup']
            + self.weight_proto * metrics['proto']
            + self.weight_support * metrics['sup']
            + self.weight_div * metrics['div']
        )
        detached = {'loss': total.detach()}
        detached.update({k: v.detach() for k, v in metrics.items()})
        return total, detached
