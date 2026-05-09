from typing import List, Optional

import torch


def _greedy_mmr(sim_matrix: torch.Tensor, rel_score: torch.Tensor, limit: int, lam: float) -> torch.Tensor:
    mask = torch.ones(rel_score.numel(), dtype=torch.bool, device=rel_score.device)
    picked: List[torch.Tensor] = []
    max_sim: Optional[torch.Tensor] = None
    for _ in range(limit):
        if not mask.any():
            break
        if max_sim is None:
            mmr_scores = rel_score
        else:
            mmr_scores = lam * rel_score - (1.0 - lam) * max_sim
        mmr_scores = mmr_scores.masked_fill(~mask, -1e4)
        best = mmr_scores.argmax()
        picked.append(best)
        mask[best] = False
        current_sim = sim_matrix[:, best]
        max_sim = current_sim if max_sim is None else torch.maximum(max_sim, current_sim)
    if not picked:
        return torch.empty(0, dtype=torch.long, device=rel_score.device)
    return torch.stack(picked)


def select_set_indices(
    candidate_mass: torch.Tensor,
    candidate_score: torch.Tensor,
    rel_logits: torch.Tensor,
    transport: Optional[torch.Tensor] = None,
    topk: int = 20,
    decode_mode: str = 'topk',
    score_mode: str = 'model',
    score_tau: float = 1.0,
    candidate_feats: Optional[torch.Tensor] = None,
    mmr_lambda: float = 0.7,
    pre_filter_n: int = 0,
) -> List[int]:
    target_k = min(topk, candidate_score.numel())
    if score_mode == 'model':
        score = candidate_score + 1e-4 * rel_logits
        pair_transport = transport
    elif score_mode == 'mass':
        score = candidate_mass + 1e-4 * rel_logits
        pair_transport = transport
    elif score_mode == 'gated':
        support = torch.sigmoid(rel_logits / max(score_tau, 1e-6))
        score = candidate_mass * support + 1e-4 * rel_logits
        pair_transport = None if transport is None else transport * support.unsqueeze(0)
    else:
        raise ValueError(f'Unknown score_mode: {score_mode}')

    if decode_mode == 'topk' or (transport is None and decode_mode != 'mmr'):
        return torch.argsort(score, descending=True)[:target_k].tolist()

    if decode_mode == 'mmr' and candidate_feats is not None:
        feats = torch.nn.functional.normalize(candidate_feats.float(), dim=-1)
        sim_matrix = torch.matmul(feats, feats.t())
        rel_score = score / (score.max() + 1e-8)
        return _greedy_mmr(sim_matrix, rel_score, target_k, mmr_lambda).tolist()

    if decode_mode == 'filter_mmr' and candidate_feats is not None:
        pfn = pre_filter_n if pre_filter_n > 0 else 50
        top_indices = torch.argsort(score, descending=True)[:pfn]
        feats = torch.nn.functional.normalize(candidate_feats.float(), dim=-1)
        filtered_feats = feats.index_select(0, top_indices)
        sim_matrix = torch.matmul(filtered_feats, filtered_feats.t())
        filtered_score = score.index_select(0, top_indices)
        rel_score = filtered_score / (filtered_score.max() + 1e-8)
        picked = _greedy_mmr(sim_matrix, rel_score, target_k, mmr_lambda)
        return top_indices.index_select(0, picked).tolist()

    if decode_mode == 'adaptive_mmr' and candidate_feats is not None:
        feats = torch.nn.functional.normalize(candidate_feats.float(), dim=-1)
        sim_matrix = torch.matmul(feats, feats.t())
        rel_score = score / (score.max() + 1e-8)
        picked: List[torch.Tensor] = []
        mask = torch.ones(score.numel(), dtype=torch.bool, device=score.device)
        max_sim: Optional[torch.Tensor] = None
        for pos in range(target_k):
            if not mask.any():
                break
            adaptive_lam = max(mmr_lambda, 1.0 - pos * (1.0 - mmr_lambda) / max(target_k * 0.3, 1))
            adaptive_lam = min(adaptive_lam, 1.0)
            if max_sim is None:
                mmr_scores = rel_score
            else:
                mmr_scores = adaptive_lam * rel_score - (1.0 - adaptive_lam) * max_sim
            mmr_scores = mmr_scores.masked_fill(~mask, -1e4)
            best = mmr_scores.argmax()
            picked.append(best)
            mask[best] = False
            current_sim = sim_matrix[:, best]
            max_sim = current_sim if max_sim is None else torch.maximum(max_sim, current_sim)
        return [int(idx.item()) for idx in picked]

    if decode_mode == 'pair_greedy':
        active_transport = transport if pair_transport is None else pair_transport
        flat = (active_transport + 1e-4 * rel_logits.unsqueeze(0)).reshape(-1)
        order = torch.argsort(flat, descending=True)
        picked = []
        used = set()
        num_candidates = candidate_score.numel()
        for idx in order.tolist():
            candidate_idx = idx % num_candidates
            if candidate_idx in used:
                continue
            used.add(candidate_idx)
            picked.append(candidate_idx)
            if len(picked) == target_k:
                return picked
        return picked

    raise ValueError(f'Unknown decode_mode: {decode_mode}')
