from typing import Dict, List, Optional

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset

from mosaic.data.evaluator import Evaluator
from mosaic.data.datasets import FusedPrecomputedEvalQueryDataset, collate_eval_batch


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


def _metric_keys(evaluator: Evaluator) -> List[int]:
    keys = [k for k in range(5, evaluator.R + 1, 5)]
    if evaluator.R not in keys:
        keys.append(evaluator.R)
    return keys


@torch.inference_mode()
def evaluate_retriever(
    model,
    dataset_dir: Optional[str] = None,
    topn: int = 150,
    feature_extractor: str = 'clip',
    device: str = 'cpu',
    max_queries: Optional[int] = None,
    eval_batch_size: int = 32,
    eval_dataset: Optional[FusedPrecomputedEvalQueryDataset] = None,
    distributed: bool = False,
    decode_mode: str = 'topk',
    stage1_ckpt: Optional[str] = None,
    chunk_size: int = 2048,
    candidate_order: str = 'raw',
    recall_source: str = 'raw',
    stage2_memory_source: str = 'fused',
    stage2_query_source: str = 'raw',
    score_fusion_mode: str = 'cosine',
    structured_bonus_scale: float = 1.0,
    score_mode: str = 'model',
    score_tau: float = 1.0,
) -> Dict[str, Dict[int, float]]:
    model.eval()
    if eval_dataset is None:
        if dataset_dir is None or stage1_ckpt is None:
            raise ValueError('dataset_dir and stage1_ckpt must be provided when eval_dataset is None.')
        eval_dataset = FusedPrecomputedEvalQueryDataset(
            dataset_dir,
            stage1_ckpt=stage1_ckpt,
            projector_device=device,
            topn=topn,
            feature_extractor=feature_extractor,
            max_queries=max_queries,
            chunk_size=chunk_size,
            candidate_order=candidate_order,
            recall_source=recall_source,
            stage2_memory_source=stage2_memory_source,
            stage2_query_source=stage2_query_source,
            score_fusion_mode=score_fusion_mode,
            structured_bonus_scale=structured_bonus_scale,
        )

    evaluator = Evaluator()
    metric_keys = _metric_keys(evaluator)
    reduce_device = torch.device(device) if str(device).startswith('cuda') or str(device) == 'cpu' else torch.device('cpu')
    metric_sums = torch.zeros(len(metric_keys) * 3 + 1, dtype=torch.float64, device=reduce_device)

    use_distributed = distributed and dist.is_available() and dist.is_initialized()
    if use_distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        query_indices = list(range(rank, len(eval_dataset), world_size))
        eval_subset = Subset(eval_dataset, query_indices)
    else:
        eval_subset = eval_dataset

    eval_loader = DataLoader(
        eval_subset,
        batch_size=max(1, eval_batch_size),
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and str(device).startswith('cuda'),
        collate_fn=collate_eval_batch,
    )

    for batch in eval_loader:
        query_feat = batch['query_feat'].to(device, non_blocking=True)
        rel_feats = batch['rel_feats'].to(device, non_blocking=True)
        div_feats = batch['div_feats'].to(device, non_blocking=True)
        img_feats = batch['img_feats'].to(device, non_blocking=True)
        recall_sims = batch['recall_sims'].to(device, non_blocking=True)
        pad_mask = batch['pad_mask'].to(device, non_blocking=True)
        outputs = model(
            query_feat,
            img_feats,
            recall_sims=recall_sims,
            pad_mask=pad_mask,
            rel_feats=rel_feats,
            div_feats=div_feats,
        )
        candidate_score = outputs['candidate_score'].cpu()
        candidate_mass = outputs['candidate_mass'].cpu()
        rel_logits = outputs['rel_logits'].cpu()
        transport = outputs['transport'].cpu()
        order_cpu = batch['order']
        pad_mask_cpu = batch['pad_mask']

        for b in range(query_feat.size(0)):
            recall_n = int(pad_mask_cpu[b].sum().item())
            local_pick = select_set_indices(
                candidate_mass[b, :recall_n],
                candidate_score[b, :recall_n],
                rel_logits[b, :recall_n],
                transport=transport[b, :, :recall_n],
                topk=min(20, recall_n),
                decode_mode=decode_mode,
                score_mode=score_mode,
                score_tau=score_tau,
                candidate_feats=img_feats[b, :recall_n].cpu() if decode_mode in ('mmr', 'filter_mmr', 'adaptive_mmr') else None,
                mmr_lambda=getattr(model, '_mmr_lambda', 0.7),
                pre_filter_n=getattr(model, '_pre_filter_n', 0),
            )
            ordered_local = order_cpu[b, :recall_n].tolist()
            selected_local = [int(ordered_local[i]) for i in local_pick]
            if len(selected_local) < 20:
                raise RuntimeError('A query has fewer than 20 retrievable candidates.')

            global_offset = int(batch['global_offset'][b].item())
            gt_positive_ids = batch['gt_positive_ids'][b]
            global_indices = [idx + global_offset for idx in selected_local[:20]]

            P = evaluator.P(global_indices, gt_positive_ids)
            nDCG = evaluator.nDCG(global_indices, gt_positive_ids)
            CR = evaluator.CR(
                [eval_dataset.id2cluster_mapper[i] for i in global_indices],
                int(batch['cluster_num'][b].item()),
                global_indices,
                gt_positive_ids,
            )
            for metric_idx, k in enumerate(metric_keys):
                metric_sums[metric_idx] += P[k]
                metric_sums[len(metric_keys) + metric_idx] += nDCG[k]
                metric_sums[2 * len(metric_keys) + metric_idx] += CR[k]
            metric_sums[-1] += 1.0

    if use_distributed:
        dist.all_reduce(metric_sums, op=dist.ReduceOp.SUM)

    summary = {'P': {}, 'nDCG': {}, 'CR': {}, 'F1': {}}
    total_count = max(metric_sums[-1].item(), 1.0)
    for metric_idx, k in enumerate(metric_keys):
        p = metric_sums[metric_idx].item() / total_count * 100.0
        ndcg = metric_sums[len(metric_keys) + metric_idx].item() / total_count * 100.0
        cr = metric_sums[2 * len(metric_keys) + metric_idx].item() / total_count * 100.0
        f1 = 2 * p * cr / (p + cr) if (p + cr) > 1e-8 else 0.0
        summary['P'][k] = p
        summary['nDCG'][k] = ndcg
        summary['CR'][k] = cr
        summary['F1'][k] = f1
    return summary


def format_summary(summary: Dict[str, Dict[int, float]]) -> str:
    lines = []
    for k in sorted(summary['P'].keys()):
        lines.append(
            '@%d [P:%.2f] [nDCG:%.2f] [CR:%.2f] [F1:%.2f]'
            % (k, summary['P'][k], summary['nDCG'][k], summary['CR'][k], summary['F1'][k])
        )
    return '\n'.join(lines)
