import argparse
import json
import math
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosaic.set_score.losses import BudgetSetLoss
from mosaic.data.query_training import ContrastiveTrainDataset, RetrievalEvalDataset, collate_train
from mosaic.data.evaluator import Evaluator
from mosaic.set_score.build import build_retriever
from mosaic.data.datasets import (
    FusedPrecomputedEvalQueryDataset,
    FusedRawClipQueryDataset,
    RawStage2TrainQueryDataset,
    _build_query_div_proto_map,
    _feature_dim,
    _build_stage1_modules_from_ckpt,
    _proto_affinity_score,
    collate_query_batch,
    collate_raw_stage2_query_batch,
    project_pool,
)
from mosaic.set_score.evaluate import evaluate_retriever, format_summary
from mosaic.query_module.losses import FusionStage1Loss
from mosaic.query_module.query_modulation import DiversitySetEncoder, QueryAdapter, RelDivFusionProjector

STAGE2_LOSS_KEYS = ['loss', 'ret', 'cov', 'dup', 'proto', 'sup', 'div']
DUAL_RECALL_SIM_MODES = {'dual', 'delta', 'structdelta', 'slotdelta', 'slotdiv', 'protodiv', 'factordiv', 'dualchannel', 'supportcov', 'supportbridge', 'supportroute', 'supportroutejoint', 'resroute', 'resrouteproto', 'resroutealloc', 'resroutetrans', 'modebudget'}


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args():
    parser = argparse.ArgumentParser(description='Train MOSAIC retrieval.')
    parser.add_argument('--train_dataset', default='data/devset/')
    parser.add_argument('--test_dataset', default='data/testset/')
    parser.add_argument('--feature_extractor', default='clip')
    parser.add_argument('--save_dir', default='runs/MOSAIC/default')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--chunk_size', type=int, default=2048)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--skip_stage1', action='store_true')
    parser.add_argument('--stage1_ckpt', default='')
    parser.add_argument('--only_stage1', action='store_true')
    parser.add_argument('--only_stage2', action='store_true')
    parser.add_argument('--dry_run', action='store_true')
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--max_train_queries', type=int, default=None)
    parser.add_argument('--max_eval_queries', type=int, default=None)

    parser.add_argument('--set_size', type=int, default=4)
    parser.add_argument('--p_mask', type=float, default=0.1)
    parser.add_argument('--stage1_epochs', type=int, default=5)
    parser.add_argument('--stage1_eval_every', type=int, default=5)
    parser.add_argument('--stage1_batch_size', type=int, default=256)
    parser.add_argument('--stage1_lr', type=float, default=5e-5)
    parser.add_argument('--stage1_dropout', type=float, default=0.0)
    parser.add_argument('--query_hidden_ratio', type=float, default=0.25)
    parser.add_argument('--query_residual_ratio', type=float, default=0.2)
    parser.add_argument('--projector_hidden_ratio', type=float, default=0.5)
    parser.add_argument('--projector_residual_ratio', type=float, default=0.2)
    parser.add_argument('--projector_fuse_ratio', type=float, default=0.25)
    parser.add_argument('--projector_fusion_mode', default='residual', choices=['residual', 'branch'])
    parser.add_argument('--stage1_select_metric', default='F1@20')
    parser.add_argument('--stage1_max_steps', type=int, default=0)
    parser.add_argument('--stage1_sibling_mode', default='exclusive', choices=['exclusive', 'aligned', 'legacy'])
    parser.add_argument('--enable_stage1_guided', action='store_true')
    parser.add_argument('--guided_stage2_ckpt', default='')
    parser.add_argument('--guided_stage1_epochs', type=int, default=20)
    parser.add_argument('--guided_stage1_eval_every', type=int, default=5)
    parser.add_argument('--guided_stage1_batch_size', type=int, default=4)
    parser.add_argument('--guided_stage1_lr', type=float, default=5e-5)
    parser.add_argument('--guided_stage1_max_steps', type=int, default=0)
    parser.add_argument('--guided_stage1_select_metric', default='F1@20')
    parser.add_argument('--guided_joint_delta', action='store_true')
    parser.add_argument('--tau_rel', type=float, default=0.07)
    parser.add_argument('--tau_div', type=float, default=0.07)
    parser.add_argument('--tau_set', type=float, default=0.10)
    parser.add_argument('--w_div_cls', type=float, default=0.2)
    parser.add_argument('--w_div_cons', type=float, default=0.2)
    parser.add_argument('--w_rel_sib', type=float, default=0.7)
    parser.add_argument('--w_fuse', type=float, default=1.0)
    parser.add_argument('--w_fuse_sib', type=float, default=0.5)
    parser.add_argument('--w_orth', type=float, default=0.1)
    parser.add_argument('--w_query_reg', type=float, default=0.1)

    parser.add_argument('--topn', type=int, default=150)
    parser.add_argument('--candidate_pool_topm', type=int, default=0)
    parser.add_argument('--candidate_order', default='raw', choices=['raw', 'fused'])
    parser.add_argument('--recall_source', default='raw', choices=['raw', 'fused', 'dual'])
    parser.add_argument('--stage2_memory_source', default='fused', choices=['fused', 'raw'])
    parser.add_argument('--stage2_query_source', default='raw', choices=['adapted', 'raw'])
    parser.add_argument('--score_fusion_mode', default='cosine', choices=['cosine', 'structured_proto'])
    parser.add_argument('--structured_bonus_scale', type=float, default=1.0)
    parser.add_argument('--sim_mode', default='single', choices=['single', 'dual', 'delta', 'structdelta', 'slotdelta', 'slotdiv', 'protodiv', 'factordiv', 'dualchannel', 'supportcov', 'supportbridge', 'supportroute', 'supportroutejoint', 'resroute', 'resrouteproto', 'resroutealloc', 'resroutetrans', 'modebudget'])
    parser.add_argument('--train_delta_only', action='store_true')
    parser.add_argument('--num_slots', type=int, default=20)
    parser.add_argument('--target_budget', type=float, default=20.0)
    parser.add_argument('--num_encoder_layers', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=4)
    parser.add_argument('--nhead', type=int, default=8)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--transport_tau', type=float, default=0.1)
    parser.add_argument('--proto_tau', type=float, default=0.7)
    parser.add_argument('--proto_rel_scale', type=float, default=0.2)
    parser.add_argument('--sinkhorn_iters', type=int, default=6)
    parser.add_argument('--enable_support_gate', action='store_true')
    parser.add_argument('--support_tau', type=float, default=1.0)
    parser.add_argument('--support_logit_scale', type=float, default=1.0)
    parser.add_argument('--support_score_mode', default='gated', choices=['gated', 'mass'])
    parser.add_argument('--support_column_cap', action='store_true')
    parser.add_argument('--enable_support_relprior', action='store_true')
    parser.add_argument('--min_mode_ratio', type=float, default=1.0/3.0)
    parser.add_argument('--minimal_memory', action='store_true',
                        help='Use a compact candidate memory without pairwise interaction features.')
    parser.add_argument('--disable_query_slots', action='store_true',
                        help='Use query-independent slot embeddings.')
    parser.add_argument('--disable_budget_learning', action='store_true',
                        help='Use uniform slot budgets.')
    parser.add_argument('--disable_proto_reasoning', action='store_true',
                        help='Skip prototype attention reasoning.')
    parser.add_argument('--disable_column_constraint', action='store_true',
                        help='Skip column normalization in the transport plan.')
    parser.add_argument('--disable_rank_embedding', action='store_true',
                        help='Disable candidate rank positional embeddings.')
    parser.add_argument('--disable_encoder', action='store_true',
                        help='Skip the candidate self-attention encoder.')
    parser.add_argument('--disable_decoder', action='store_true',
                        help='Skip demand-token cross-attention.')
    parser.add_argument('--enable_universal_stage2', action='store_true')
    parser.add_argument('--stage2_epochs', type=int, default=300)
    parser.add_argument('--stage2_eval_every', type=int, default=10)
    parser.add_argument('--stage2_batch_size', type=int, default=4)
    parser.add_argument('--generator_lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--clip_grad', type=float, default=1.0)
    parser.add_argument('--init_stage2_ckpt', default='')
    parser.add_argument('--early_stop_patience', type=int, default=80)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--decode_mode', default='pair_greedy', choices=['topk', 'pair_greedy', 'mmr', 'filter_mmr', 'adaptive_mmr'])
    parser.add_argument('--mmr_lambda', type=float, default=0.7)
    parser.add_argument('--pre_filter_n', type=int, default=0)
    parser.add_argument('--weight_ret', type=float, default=0.5)
    parser.add_argument('--weight_cov', type=float, default=1.0)
    parser.add_argument('--weight_dup', type=float, default=0.25)
    parser.add_argument('--weight_proto', type=float, default=0.1)
    parser.add_argument('--weight_support', type=float, default=0.0)
    parser.add_argument('--weight_div', type=float, default=0.0)
    parser.add_argument('--cluster_alpha', type=float, default=0.5)
    parser.add_argument('--stage2_max_steps', type=int, default=0)
    parser.add_argument('--proto_feature_source', default='fused', choices=['fused', 'div'])
    return parser.parse_args()


def setup_distributed(args):
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.use_ddp = args.ddp or args.world_size > 1
    if args.use_ddp:
        if not torch.cuda.is_available():
            raise RuntimeError('DDP training requires CUDA.')
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device(args.device)
    return device


def cleanup_distributed(args):
    if getattr(args, 'use_ddp', False) and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process(args):
    return (not getattr(args, 'use_ddp', False)) or args.rank == 0


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def set_decode_controls(model, args):
    target = unwrap_model(model)
    target._mmr_lambda = args.mmr_lambda
    target._pre_filter_n = args.pre_filter_n
    if hasattr(target, 'retriever'):
        target.retriever._mmr_lambda = args.mmr_lambda
        target.retriever._pre_filter_n = args.pre_filter_n


def save_json(path: str, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def _alpha_dcg(
    ranked_ids: List[int],
    positive_ids: set,
    id2cluster_mapper: Dict[int, int],
    alpha: float,
    max_rank: Optional[int] = None,
) -> float:
    if max_rank is None:
        max_rank = len(ranked_ids)
    seen = defaultdict(int)
    dcg = 0.0
    for rank, img_id in enumerate(ranked_ids[:max_rank], start=1):
        if img_id not in positive_ids:
            continue
        cluster_id = id2cluster_mapper[img_id]
        if cluster_id < 0:
            continue
        gain = (1.0 - alpha) ** seen[cluster_id]
        seen[cluster_id] += 1
        dcg += gain / math.log2(rank + 1.0)
    return dcg


def _ideal_alpha_dcg(cluster_ids: List[int], alpha: float, max_rank: Optional[int] = None) -> float:
    if not cluster_ids:
        return 1.0
    remaining = Counter(cluster_ids)
    seen = defaultdict(int)
    dcg = 0.0
    rank = 1
    limit = max_rank if max_rank is not None else len(cluster_ids)
    while remaining and rank <= limit:
        best_cluster = None
        best_gain = -1.0
        for cluster_id, cnt in remaining.items():
            if cnt <= 0:
                continue
            gain = (1.0 - alpha) ** seen[cluster_id]
            if gain > best_gain:
                best_gain = gain
                best_cluster = cluster_id
        if best_cluster is None:
            break
        dcg += best_gain / math.log2(rank + 1.0)
        seen[best_cluster] += 1
        remaining[best_cluster] -= 1
        if remaining[best_cluster] <= 0:
            del remaining[best_cluster]
        rank += 1
    return dcg if dcg > 0 else 1.0


def _qof_from_div(
    sample: dict,
    div_feat: torch.Tensor,
    id2cluster_mapper: Dict[int, int],
    eps: float = 1e-12,
) -> Optional[float]:
    positive_local_ids = []
    cluster_ids = []
    global_offset = sample['global_offset']
    for img_id in sample['gt_positive_ids']:
        cluster_id = id2cluster_mapper[img_id]
        if cluster_id < 0:
            continue
        local_id = img_id - global_offset
        if 0 <= local_id < div_feat.size(0):
            positive_local_ids.append(local_id)
            cluster_ids.append(cluster_id)
    if len(positive_local_ids) < 2:
        return None
    unique_clusters = sorted(set(cluster_ids))
    num_clusters = len(unique_clusters)
    num_points = len(positive_local_ids)
    if num_clusters < 2 or num_points <= num_clusters:
        return None
    pos_feat = F.normalize(div_feat[positive_local_ids].float(), dim=-1, eps=eps)
    overall_mean = pos_feat.mean(dim=0, keepdim=True)
    labels = torch.tensor(cluster_ids, dtype=torch.long)
    between = pos_feat.new_tensor(0.0)
    within = pos_feat.new_tensor(0.0)
    for cluster_id in unique_clusters:
        mask = labels == cluster_id
        cluster_feat = pos_feat[mask]
        cluster_mean = cluster_feat.mean(dim=0, keepdim=True)
        between = between + cluster_feat.size(0) * torch.sum((cluster_mean - overall_mean) ** 2)
        within = within + torch.sum((cluster_feat - cluster_mean) ** 2)
    qof = (between / max(num_clusters - 1, 1)) / (within / max(num_points - num_clusters, 1) + eps)
    return float(qof)


@torch.no_grad()
def evaluate_stage1(query_adapter, projector, eval_dataset, device: torch.device, chunk_size: int = 2048, alpha: float = 0.5):
    p20s = []
    cr20s = []
    alpha_scores = []
    qof_scores = []
    for sample in eval_dataset.samples:
        projected = project_pool(projector, query_adapter, sample['img_feats'], sample['query_feat'], device, chunk_size)
        scores = projected['fused_score']
        indices = torch.argsort(scores, descending=True).tolist()
        ranked_ids = [sample['global_offset'] + int(i) for i in indices]
        k = max(20, len(sample['gt_positive_ids']))
        ranked_at_k = ranked_ids[:k]
        evaluator = Evaluator(k)
        p20 = evaluator.P(ranked_at_k, sample['gt_positive_ids'])[20] * 100.0
        cr20 = evaluator.CR(
            [eval_dataset.id2cluster_mapper[i] for i in ranked_at_k],
            sample['cluster_num'],
            ranked_at_k,
            sample['gt_positive_ids'],
        )[20] * 100.0
        p20s.append(p20)
        cr20s.append(cr20)

        positive_ids = set(sample['gt_positive_ids'])
        positive_clusters = [
            eval_dataset.id2cluster_mapper[i]
            for i in sample['gt_positive_ids']
            if eval_dataset.id2cluster_mapper[i] >= 0
        ]
        dcg = _alpha_dcg(ranked_ids, positive_ids, eval_dataset.id2cluster_mapper, alpha, max_rank=None)
        idcg = _ideal_alpha_dcg(positive_clusters, alpha, max_rank=None)
        alpha_scores.append(dcg / idcg if idcg > 0 else 0.0)

        qof = _qof_from_div(sample, projected['div'], eval_dataset.id2cluster_mapper)
        if qof is not None:
            qof_scores.append(qof)

    p20 = sum(p20s) / len(p20s)
    cr20 = sum(cr20s) / len(cr20s)
    f120 = 2 * p20 * cr20 / (p20 + cr20 + 1e-12)
    return {
        'P@20': p20,
        'CR@20': cr20,
        'F1@20': f120,
        'alpha-nDCG@All': 100.0 * sum(alpha_scores) / len(alpha_scores),
        'QO-F': sum(qof_scores) / len(qof_scores) if qof_scores else 0.0,
    }


def train_stage1_epoch(query_adapter, projector, set_encoder, criterion, loader, optimizer, device, clip_grad, w_query_reg: float = 0.1, max_steps: int = 0):
    query_adapter.train()
    projector.train()
    set_encoder.train()
    meters = {}
    num_steps = 0
    for step_idx, batch in enumerate(loader):
        if max_steps > 0 and step_idx >= max_steps:
            break
        raw_query_feat = F.normalize(batch['query_feat'].to(device), dim=-1)
        query_feat = query_adapter(raw_query_feat)
        pos_a = batch['pos_a'].to(device)
        pos_b = batch['pos_b'].to(device)
        sibling_neg = batch['sibling_neg'].to(device)
        global_neg = batch['global_neg'].to(device)
        l1_idx = batch['l1_idx'].to(device)
        l2_idx = batch['l2_idx'].to(device)
        sibling_l2_idx = batch['sibling_l2_idx'].to(device)

        pos_a_out = projector.project(pos_a, query_feat)
        pos_b_out = projector.project(pos_b, query_feat)
        sibling_out = projector.project(sibling_neg, query_feat)
        global_out = projector.project(global_neg, query_feat)

        set_div_a = set_encoder(query_feat, pos_a_out['div'])
        set_div_b = set_encoder(query_feat, pos_b_out['div'])
        set_div_sib = set_encoder(query_feat, sibling_out['div'])

        loss, loss_dict = criterion(
            query_feat,
            pos_a_out['rel'],
            pos_b_out['rel'],
            sibling_out['rel'],
            global_out['rel'],
            pos_a_out['fused'],
            pos_b_out['fused'],
            sibling_out['fused'],
            global_out['fused'],
            set_div_a,
            set_div_b,
            set_div_sib,
            l1_idx,
            l2_idx,
            sibling_l2_idx,
        )
        query_reg = (1.0 - F.cosine_similarity(query_feat, raw_query_feat, dim=-1)).mean()
        loss = loss + w_query_reg * query_reg
        loss_dict['query_reg'] = float(query_reg.detach().item())
        loss_dict['total'] = float(loss.detach().item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(query_adapter.parameters()) + list(projector.parameters()) + list(set_encoder.parameters()), clip_grad)
        optimizer.step()
        for k, v in loss_dict.items():
            meters[k] = meters.get(k, 0.0) + float(v)
        num_steps += 1
    for k in list(meters.keys()):
        meters[k] /= max(num_steps, 1)
    return meters


def reduce_stage2_metrics(loss_meter, steps, device, args):
    tensor = torch.tensor([loss_meter[key] for key in STAGE2_LOSS_KEYS] + [float(steps)], device=device, dtype=torch.float64)
    if getattr(args, 'use_ddp', False):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    total_steps = max(float(tensor[-1].item()), 1.0)
    return {key: float(tensor[idx].item() / total_steps) for idx, key in enumerate(STAGE2_LOSS_KEYS)}


def _build_stage2_training_inputs(batch, device, args):
    query_feat = batch['query_feat'].to(device, non_blocking=True)
    img_feats = batch['img_feats'].to(device, non_blocking=True)
    recall_sims = batch['recall_sims'].to(device, non_blocking=True)
    if not args.enable_universal_stage2:
        return query_feat, img_feats, recall_sims

    raw_img_feats = batch['raw_img_feats'].to(device, non_blocking=True)
    fused_img_feats = batch['fused_img_feats'].to(device, non_blocking=True)
    raw_sims = batch['raw_sims'].to(device, non_blocking=True)
    fused_sims = batch['fused_sims'].to(device, non_blocking=True)

    mix = torch.rand(raw_img_feats.size(0), 1, 1, device=device, dtype=raw_img_feats.dtype)
    mixed_img_feats = F.normalize(raw_img_feats + mix * (fused_img_feats - raw_img_feats), dim=-1)
    mix_scalar = mix.squeeze(-1)
    if args.recall_source == 'dual':
        mixed_sims = raw_sims + mix_scalar * (fused_sims - raw_sims)
        mixed_recall_sims = torch.stack([raw_sims, mixed_sims], dim=-1)
    elif args.recall_source == 'fused':
        mixed_recall_sims = raw_sims + mix_scalar * (fused_sims - raw_sims)
    else:
        mixed_recall_sims = raw_sims
    return query_feat, mixed_img_feats, mixed_recall_sims


def train_stage2_epoch(model, loader, optimizer, criterion, device, args):
    model.train()
    loss_meter = {key: 0.0 for key in STAGE2_LOSS_KEYS}
    steps = 0
    for step_idx, batch in enumerate(loader):
        if args.stage2_max_steps > 0 and step_idx >= args.stage2_max_steps:
            break
        query_feat, img_feats, recall_sims = _build_stage2_training_inputs(batch, device, args)
        rel_feats = batch['rel_feats'].to(device, non_blocking=True)
        div_feats = batch['div_feats'].to(device, non_blocking=True)
        quality = batch['quality'].to(device, non_blocking=True)
        cluster = batch['cluster'].to(device, non_blocking=True)
        pad_mask = batch['pad_mask'].to(device, non_blocking=True)

        outputs = model(
            query_feat,
            img_feats,
            recall_sims=recall_sims,
            pad_mask=pad_mask,
            rel_feats=rel_feats,
            div_feats=div_feats,
        )
        outputs['img_feats'] = div_feats if args.proto_feature_source == 'div' else img_feats
        loss, metrics = criterion(outputs, quality, cluster, pad_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        optimizer.step()

        steps += 1
        for k in loss_meter.keys():
            loss_meter[k] += float(metrics[k])
    return reduce_stage2_metrics(loss_meter, steps, device, args)


def save_stage2_checkpoint(model, args, summary, epoch, stage1_ckpt: str):
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'budget_best.pt')
    payload = {
        'model_state': unwrap_model(model).state_dict(),
        'config': {
            'feature_extractor': args.feature_extractor,
            'topn': args.topn,
            'candidate_pool_topm': args.candidate_pool_topm,
            'num_slots': args.num_slots,
            'target_budget': args.target_budget,
            'num_encoder_layers': args.num_encoder_layers,
            'num_layers': args.num_layers,
            'nhead': args.nhead,
            'dim_feedforward': args.dim_feedforward,
            'dropout': args.dropout,
            'transport_tau': args.transport_tau,
            'proto_tau': args.proto_tau,
            'proto_rel_scale': args.proto_rel_scale,
            'sinkhorn_iters': args.sinkhorn_iters,
            'enable_support_gate': args.enable_support_gate,
            'support_tau': args.support_tau,
            'support_logit_scale': args.support_logit_scale,
            'support_score_mode': args.support_score_mode,
            'support_column_cap': args.support_column_cap,
            'enable_support_relprior': args.enable_support_relprior,
            'minimal_memory': getattr(args, 'minimal_memory', False),
            'disable_query_slots': getattr(args, 'disable_query_slots', False),
            'disable_budget_learning': getattr(args, 'disable_budget_learning', False),
            'disable_proto_reasoning': getattr(args, 'disable_proto_reasoning', False),
            'disable_column_constraint': getattr(args, 'disable_column_constraint', False),
            'disable_rank_embedding': getattr(args, 'disable_rank_embedding', False),
            'disable_encoder': getattr(args, 'disable_encoder', False),
            'disable_decoder': getattr(args, 'disable_decoder', False),
            'enable_universal_stage2': args.enable_universal_stage2,
            'candidate_order': args.candidate_order,
            'recall_source': args.recall_source,
            'stage2_memory_source': args.stage2_memory_source,
            'stage2_query_source': args.stage2_query_source,
            'score_fusion_mode': args.score_fusion_mode,
            'structured_bonus_scale': args.structured_bonus_scale,
            'sim_mode': args.sim_mode,
            'train_delta_only': args.train_delta_only,
            'proto_feature_source': args.proto_feature_source,
            'decode_mode': args.decode_mode,
            'mmr_lambda': args.mmr_lambda,
            'pre_filter_n': args.pre_filter_n,
            'stage1_ckpt': stage1_ckpt,
            'seed': args.seed,
            'generator_lr': args.generator_lr,
            'weight_decay': args.weight_decay,
            'clip_grad': args.clip_grad,
            'weight_ret': args.weight_ret,
            'weight_cov': args.weight_cov,
            'weight_dup': args.weight_dup,
            'weight_proto': args.weight_proto,
            'weight_support': args.weight_support,
            'weight_div': args.weight_div,
            'cluster_alpha': args.cluster_alpha,
            'stage2_epochs': args.stage2_epochs,
            'stage2_eval_every': args.stage2_eval_every,
            'early_stop_patience': args.early_stop_patience,
        },
        'epoch': epoch,
        'summary': summary,
    }
    torch.save(payload, ckpt_path)
    save_json(os.path.join(args.save_dir, 'budget_best_metrics.json'), summary)


def run_stage1(args, device):
    if args.use_ddp and not is_main_process(args):
        dist.barrier()
        return os.path.join(args.save_dir, 'stage1_best.pt')

    train_dataset = ContrastiveTrainDataset(args.train_dataset, args.feature_extractor, args.set_size, args.p_mask)
    eval_dataset = RetrievalEvalDataset(args.test_dataset, args.feature_extractor)
    if args.max_eval_queries is not None:
        eval_dataset.samples = eval_dataset.samples[:args.max_eval_queries]
    loader = DataLoader(
        train_dataset,
        batch_size=args.stage1_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_train,
        drop_last=False,
    )

    dim = train_dataset.l1_text_feats.size(-1)
    query_adapter = QueryAdapter(
        dim=dim,
        hidden_ratio=args.query_hidden_ratio,
        residual_ratio=args.query_residual_ratio,
        dropout=args.stage1_dropout,
    ).to(device)
    projector = RelDivFusionProjector(
        dim=dim,
        hidden_ratio=args.projector_hidden_ratio,
        residual_ratio=args.projector_residual_ratio,
        fuse_ratio=args.projector_fuse_ratio,
        dropout=args.stage1_dropout,
        fusion_mode=args.projector_fusion_mode,
    ).to(device)
    set_encoder = DiversitySetEncoder(dim=dim, dropout=args.stage1_dropout).to(device)
    criterion = FusionStage1Loss(
        train_dataset.l1_text_feats,
        train_dataset.l2_text_feats,
        train_dataset.l1_to_l2_idxs,
        train_dataset.l2_idx_to_l1_idx,
        tau_rel=args.tau_rel,
        tau_div=args.tau_div,
        tau_set=args.tau_set,
        w_div_cls=args.w_div_cls,
        w_div_cons=args.w_div_cons,
        w_rel_sib=args.w_rel_sib,
        w_fuse=args.w_fuse,
        w_fuse_sib=args.w_fuse_sib,
        w_orth=args.w_orth,
        sibling_positive=(args.stage1_sibling_mode == 'aligned'),
    ).to(device)

    optimizer = AdamW(
        list(query_adapter.parameters()) + list(projector.parameters()) + list(set_encoder.parameters()),
        lr=args.stage1_lr,
        weight_decay=args.weight_decay,
    )

    best_metric = -1e9
    best_epoch = 0
    best_metrics = None
    epochs = 1 if args.dry_run else args.stage1_epochs
    os.makedirs(args.save_dir, exist_ok=True)
    ckpt_path = os.path.join(args.save_dir, 'stage1_best.pt')

    if is_main_process(args):
        raw_metrics = evaluate_stage1(None, None, eval_dataset, device, args.chunk_size, args.alpha)
        print(
            'stage1 epoch 0 raw_eval '
            f"F1@20={raw_metrics['F1@20']:.4f} P@20={raw_metrics['P@20']:.4f} "
            f"CR@20={raw_metrics['CR@20']:.4f} alpha-nDCG@All={raw_metrics['alpha-nDCG@All']:.4f} QO-F={raw_metrics['QO-F']:.4f}"
        )

    for epoch in range(1, epochs + 1):
        meters = train_stage1_epoch(
            query_adapter,
            projector,
            set_encoder,
            criterion,
            loader,
            optimizer,
            device,
            args.clip_grad,
            w_query_reg=args.w_query_reg,
            max_steps=args.stage1_max_steps,
        )
        if is_main_process(args):
            meter_str = ' '.join([f'{k}={v:.4f}' for k, v in sorted(meters.items())])
            print(f'stage1 epoch {epoch} train {meter_str} lr={optimizer.param_groups[0]["lr"]:.6g}')

        should_eval = args.dry_run or (args.stage1_eval_every > 0 and (epoch % args.stage1_eval_every == 0 or epoch == epochs))
        if should_eval and is_main_process(args):
            metrics = evaluate_stage1(query_adapter, projector, eval_dataset, device, args.chunk_size, args.alpha)
            print(
                'stage1 epoch %d eval F1@20=%.4f P@20=%.4f CR@20=%.4f alpha-nDCG@All=%.4f QO-F=%.4f'
                % (
                    epoch,
                    metrics['F1@20'],
                    metrics['P@20'],
                    metrics['CR@20'],
                    metrics['alpha-nDCG@All'],
                    metrics['QO-F'],
                )
            )
            score = metrics[args.stage1_select_metric]
            if score > best_metric:
                best_metric = score
                best_epoch = epoch
                best_metrics = metrics
                torch.save(
                    {
                        'query_adapter': query_adapter.state_dict(),
                        'projector': projector.state_dict(),
                        'set_encoder': set_encoder.state_dict(),
                        'args': vars(args),
                        'metrics': metrics,
                        'epoch': epoch,
                    },
                    ckpt_path,
                )
    if is_main_process(args):
        summary = {
            'best_epoch': best_epoch,
            'best_metrics': best_metrics,
            'select_metric': args.stage1_select_metric,
            'checkpoint': ckpt_path,
        }
        save_json(os.path.join(args.save_dir, 'stage1_summary.json'), summary)
    if args.use_ddp:
        dist.barrier()
    return ckpt_path


def _build_fixed_stage2_from_ckpt(args, device, ckpt_path: str):
    payload = torch.load(ckpt_path, map_location='cpu')
    config = payload['config']
    model = build_retriever(
        feature_extractor=config['feature_extractor'],
        topn=config['topn'],
        num_slots=config['num_slots'],
        num_layers=config['num_layers'],
        nhead=config['nhead'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        num_encoder_layers=config.get('num_encoder_layers', 2),
        transport_tau=config.get('transport_tau', 0.1),
        target_budget=config.get('target_budget', 20.0),
        proto_tau=config.get('proto_tau', 0.7),
        proto_rel_scale=config.get('proto_rel_scale', 0.2),
        sinkhorn_iters=config.get('sinkhorn_iters', 6),
        sim_mode=config.get('sim_mode', 'single'),
        enable_support_gate=config.get('enable_support_gate', False),
        support_tau=config.get('support_tau', 1.0),
        support_logit_scale=config.get('support_logit_scale', 1.0),
        support_score_mode=config.get('support_score_mode', 'gated'),
        support_column_cap=config.get('support_column_cap', False),
        enable_support_relprior=config.get('enable_support_relprior', False),
        min_mode_ratio=config.get('min_mode_ratio', 1.0 / 3.0),
        minimal_memory=config.get('minimal_memory', False),
        disable_query_slots=config.get('disable_query_slots', False),
        disable_budget_learning=config.get('disable_budget_learning', False),
        disable_proto_reasoning=config.get('disable_proto_reasoning', False),
        disable_column_constraint=config.get('disable_column_constraint', False),
        disable_rank_embedding=config.get('disable_rank_embedding', False),
        disable_encoder=config.get('disable_encoder', False),
        disable_decoder=config.get('disable_decoder', False),
    ).to(device)
    current_state = model.state_dict()
    matched = {
        key: value
        for key, value in payload['model_state'].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    if args.guided_joint_delta:
        for name, param in model.named_parameters():
            param.requires_grad = 'delta_sim_proj' in name
    else:
        for param in model.parameters():
            param.requires_grad = False
    model.eval()
    return model, config


def _save_stage1_payload(path: str, query_adapter, projector, args, metrics=None, epoch: int = 0, stage1_args_override=None):
    payload = {
        'args': stage1_args_override if stage1_args_override is not None else vars(args),
        'metrics': metrics,
        'epoch': epoch,
    }
    if query_adapter is not None:
        payload['query_adapter'] = query_adapter.state_dict()
    if hasattr(projector, 'adapter'):
        payload['adapter'] = projector.adapter.state_dict()
    else:
        payload['projector'] = projector.state_dict()
    torch.save(payload, path)


def _build_guided_recall_sims(
    projected: dict,
    raw_sims: torch.Tensor,
    query_names: List[str],
    query_div_proto_map: Dict[str, torch.Tensor],
    score_fusion_mode: str,
    use_dual_recall: bool,
) -> torch.Tensor:
    fused_scores = projected['fused_score']
    if score_fusion_mode != 'structured_proto':
        return torch.stack([raw_sims, fused_scores], dim=-1) if use_dual_recall else fused_scores

    structured_bonus = torch.zeros_like(fused_scores)
    for i, query_name in enumerate(query_names):
        proto_bank = query_div_proto_map.get(query_name)
        if proto_bank is None or proto_bank.numel() == 0:
            continue
        affinity = _proto_affinity_score(projected['div'][i], proto_bank.to(projected['div'][i].device))
        structured_bonus[i] = projected['score'][i].clamp_min(0.0) * affinity
    guided_scores = fused_scores + structured_bonus
    return torch.stack([raw_sims, guided_scores], dim=-1) if use_dual_recall else guided_scores


def train_stage1_guided_epoch(
    query_adapter,
    projector,
    frozen_stage2,
    criterion,
    loader,
    optimizer,
    device,
    args,
    query_div_proto_map=None,
    use_dual_recall: bool = True,
):
    if query_adapter is not None:
        query_adapter.train()
    projector.train()
    frozen_stage2.eval()
    meters = {key: 0.0 for key in STAGE2_LOSS_KEYS}
    steps = 0
    for step_idx, batch in enumerate(loader):
        if args.guided_stage1_max_steps > 0 and step_idx >= args.guided_stage1_max_steps:
            break
        raw_query_feat = F.normalize(batch['query_feat'].to(device), dim=-1)
        raw_img_feats = F.normalize(batch['img_feats'].to(device), dim=-1)
        raw_sims = batch['raw_sims'].to(device)
        quality = batch['quality'].to(device)
        cluster = batch['cluster'].to(device)
        pad_mask = batch['pad_mask'].to(device)

        query_feat = query_adapter(raw_query_feat) if query_adapter is not None else raw_query_feat
        if hasattr(projector, 'project'):
            projected = projector.project(raw_img_feats, query_feat)
        else:
            projected = projector.project_query_images(raw_img_feats, query_feat)
        recall_sims = _build_guided_recall_sims(
            projected,
            raw_sims,
            batch['query_names'],
            query_div_proto_map or {},
            args.score_fusion_mode,
            use_dual_recall,
        )
        if use_dual_recall and args.score_fusion_mode == 'structured_proto':
            recall_sims[..., 1] = projected['fused_score'] + args.structured_bonus_scale * (recall_sims[..., 1] - projected['fused_score'])
        outputs = frozen_stage2(
            query_feat,
            projected['fused'],
            recall_sims=recall_sims,
            pad_mask=pad_mask,
            rel_feats=projected['rel'],
            div_feats=projected['div'],
        )
        outputs['img_feats'] = projected['fused']
        loss, metrics = criterion(outputs, quality, cluster, pad_mask)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        params = list(projector.parameters())
        if query_adapter is not None:
            params = list(query_adapter.parameters()) + params
        torch.nn.utils.clip_grad_norm_(params, args.clip_grad)
        optimizer.step()

        steps += 1
        for key in STAGE2_LOSS_KEYS:
            meters[key] += float(metrics[key])
    for key in meters:
        meters[key] /= max(steps, 1)
    return meters


def run_stage1_guided(args, device, stage1_ckpt: str):
    if not args.guided_stage2_ckpt:
        raise ValueError('--enable_stage1_guided requires --guided_stage2_ckpt.')
    guided_dir = os.path.join(args.save_dir, 'guided_stage1') if not args.only_stage1 else args.save_dir
    os.makedirs(guided_dir, exist_ok=True)

    query_adapter, projector = _build_stage1_modules_from_ckpt(stage1_ckpt, args.feature_extractor, device)
    source_stage1_payload = torch.load(stage1_ckpt, map_location='cpu')
    source_stage1_args = source_stage1_payload.get('args', {})
    if query_adapter is not None:
        query_adapter.train()
        for param in query_adapter.parameters():
            param.requires_grad = True
    projector.train()
    for param in projector.parameters():
        param.requires_grad = True
    frozen_stage2, stage2_config = _build_fixed_stage2_from_ckpt(args, device, args.guided_stage2_ckpt)

    train_dataset = RawStage2TrainQueryDataset(
        args.train_dataset,
        topn=stage2_config.get('topn', args.topn),
        feature_extractor=args.feature_extractor,
        max_queries=args.max_train_queries,
    )
    query_div_proto_map = _build_query_div_proto_map(args.train_dataset, args.feature_extractor)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.guided_stage1_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_raw_stage2_query_batch,
    )

    criterion = BudgetSetLoss(
        weight_ret=args.weight_ret,
        weight_cov=args.weight_cov,
        weight_dup=args.weight_dup,
        weight_proto=args.weight_proto,
        weight_support=args.weight_support,
        cluster_alpha=args.cluster_alpha,
    )
    guided_params = list(projector.parameters())
    if query_adapter is not None:
        guided_params = list(query_adapter.parameters()) + guided_params
    guided_params += [param for param in frozen_stage2.parameters() if param.requires_grad]
    optimizer = AdamW(guided_params, lr=args.guided_stage1_lr, weight_decay=args.weight_decay)

    best_metric = -1e9
    best_epoch = 0
    best_summary = None
    best_ckpt = os.path.join(guided_dir, 'stage1_guided_best.pt')
    tmp_ckpt = os.path.join(guided_dir, 'stage1_guided_tmp.pt')
    epochs = 1 if args.dry_run else args.guided_stage1_epochs
    guided_sim_mode = stage2_config.get('sim_mode', 'single')
    use_dual_recall = guided_sim_mode in DUAL_RECALL_SIM_MODES

    for epoch in range(1, epochs + 1):
        meters = train_stage1_guided_epoch(
            query_adapter,
            projector,
            frozen_stage2,
            criterion,
            train_loader,
            optimizer,
            device,
            args,
            query_div_proto_map=query_div_proto_map,
            use_dual_recall=use_dual_recall,
        )
        print(
            'stage1-guided epoch %d train loss %.4f ret %.4f cov %.4f dup %.4f proto %.4f sup %.4f lr=%g'
            % (epoch, meters['loss'], meters['ret'], meters['cov'], meters['dup'], meters['proto'], meters['sup'], args.guided_stage1_lr)
        )
        should_eval = args.dry_run or (args.guided_stage1_eval_every > 0 and (epoch % args.guided_stage1_eval_every == 0 or epoch == epochs))
        if not should_eval:
            continue
        _save_stage1_payload(tmp_ckpt, query_adapter, projector, args, epoch=epoch, stage1_args_override=source_stage1_args)
        eval_dataset = FusedPrecomputedEvalQueryDataset(
            args.test_dataset,
            stage1_ckpt=tmp_ckpt,
            projector_device=str(device),
            topn=stage2_config.get('topn', args.topn),
            candidate_pool_topm=stage2_config.get('candidate_pool_topm', args.candidate_pool_topm if args.candidate_pool_topm > 0 else stage2_config.get('topn', args.topn)),
            feature_extractor=args.feature_extractor,
            max_queries=args.max_eval_queries,
            chunk_size=args.chunk_size,
            candidate_order=stage2_config.get('candidate_order', 'raw'),
            recall_source=stage2_config.get('recall_source', 'raw'),
            stage2_memory_source=stage2_config.get('stage2_memory_source', args.stage2_memory_source),
            stage2_query_source=stage2_config.get('stage2_query_source', args.stage2_query_source),
            score_fusion_mode=stage2_config.get('score_fusion_mode', args.score_fusion_mode),
            structured_bonus_scale=stage2_config.get('structured_bonus_scale', args.structured_bonus_scale),
        )
        summary = evaluate_retriever(
            frozen_stage2,
            args.test_dataset,
            stage2_config.get('topn', args.topn),
            args.feature_extractor,
            str(device),
            args.max_eval_queries,
            eval_batch_size=args.eval_batch_size,
            eval_dataset=eval_dataset,
            decode_mode=args.decode_mode,
            stage1_ckpt=tmp_ckpt,
            chunk_size=args.chunk_size,
            candidate_order=stage2_config.get('candidate_order', 'raw'),
            recall_source=stage2_config.get('recall_source', 'raw'),
            stage2_memory_source=stage2_config.get('stage2_memory_source', args.stage2_memory_source),
            stage2_query_source=stage2_config.get('stage2_query_source', args.stage2_query_source),
            score_fusion_mode=stage2_config.get('score_fusion_mode', args.score_fusion_mode),
            structured_bonus_scale=stage2_config.get('structured_bonus_scale', args.structured_bonus_scale),
        )
        print(format_summary(summary))
        score = summary['F1'][20] if args.guided_stage1_select_metric == 'F1@20' else summary['F1'][20]
        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            best_summary = summary
            _save_stage1_payload(best_ckpt, query_adapter, projector, args, metrics=summary, epoch=epoch, stage1_args_override=source_stage1_args)
            save_json(os.path.join(guided_dir, 'stage1_guided_best_metrics.json'), summary)
            if args.guided_joint_delta:
                torch.save(
                    {
                        'model_state': frozen_stage2.state_dict(),
                        'config': stage2_config,
                        'summary': summary,
                        'epoch': epoch,
                        'stage1_ckpt': best_ckpt,
                    },
                    os.path.join(guided_dir, 'stage2_guided_best.pt'),
                )
            print('Saved best guided stage1 checkpoint with F1@20=%.2f' % score)

    save_json(
        os.path.join(guided_dir, 'stage1_guided_summary.json'),
        {
            'best_epoch': best_epoch,
            'best_metric': best_metric,
            'best_summary': best_summary,
            'checkpoint': best_ckpt,
            'guided_stage2_ckpt': args.guided_stage2_ckpt,
        },
    )
    return best_ckpt


def load_stage2_shell(model, ckpt_path: str):
    payload = torch.load(ckpt_path, map_location='cpu')
    current_state = model.state_dict()
    matched = {
        key: value
        for key, value in payload['model_state'].items()
        if key in current_state and current_state[key].shape == value.shape
    }

    def _copy_linear(dst_prefix: str, src_prefix: str):
        for suffix in ['weight', 'bias']:
            dst_key = f'{dst_prefix}.{suffix}'
            src_key = f'{src_prefix}.{suffix}'
            if dst_key in current_state and src_key in payload['model_state']:
                if current_state[dst_key].shape == payload['model_state'][src_key].shape:
                    matched[dst_key] = payload['model_state'][src_key]

    def _copy_norm(dst_prefix: str, src_prefix: str):
        for suffix in ['weight', 'bias']:
            dst_key = f'{dst_prefix}.{suffix}'
            src_key = f'{src_prefix}.{suffix}'
            if dst_key in current_state and src_key in payload['model_state']:
                if current_state[dst_key].shape == payload['model_state'][src_key].shape:
                    matched[dst_key] = payload['model_state'][src_key]

    def _copy_tensor(dst_key: str, tensor: torch.Tensor):
        if dst_key in current_state and current_state[dst_key].shape == tensor.shape:
            matched[dst_key] = tensor.to(dtype=current_state[dst_key].dtype)

    def _copy_adaptive_embed(dst_key: str, src_key: str):
        if dst_key not in current_state or src_key not in payload['model_state']:
            return
        dst_weight = current_state[dst_key]
        src_weight = payload['model_state'][src_key]
        if dst_weight.ndim != 2 or src_weight.ndim != 2 or dst_weight.shape[1] != src_weight.shape[1]:
            return
        pooled = F.adaptive_avg_pool1d(src_weight.t().unsqueeze(0), dst_weight.shape[0]).squeeze(0).t()
        matched[dst_key] = pooled.to(dtype=dst_weight.dtype)

    def _copy_decoder_block(dst_prefix: str, src_prefix: str):
        suffixes = [
            'self_attn.in_proj_weight',
            'self_attn.in_proj_bias',
            'self_attn.out_proj.weight',
            'self_attn.out_proj.bias',
            'cross_attn.in_proj_weight',
            'cross_attn.in_proj_bias',
            'cross_attn.out_proj.weight',
            'cross_attn.out_proj.bias',
            'linear1.weight',
            'linear1.bias',
            'linear2.weight',
            'linear2.bias',
            'norm1.weight',
            'norm1.bias',
            'norm2.weight',
            'norm2.bias',
            'norm3.weight',
            'norm3.bias',
        ]
        for suffix in suffixes:
            dst_key = f'{dst_prefix}.{suffix}'
            src_key = f'{src_prefix}.{suffix}'
            if dst_key in current_state and src_key in payload['model_state']:
                if current_state[dst_key].shape == payload['model_state'][src_key].shape:
                    matched[dst_key] = payload['model_state'][src_key]

    def _copy_concat_proj(dst_prefix: str, src_prefix: str):
        dst_weight_key = f'{dst_prefix}.weight'
        src_weight_key = f'{src_prefix}.weight'
        if dst_weight_key not in current_state or src_weight_key not in payload['model_state']:
            return
        dst_weight = current_state[dst_weight_key]
        src_weight = payload['model_state'][src_weight_key]
        if dst_weight.ndim != 2 or src_weight.ndim != 2:
            return
        if dst_weight.shape[0] != src_weight.shape[0] or dst_weight.shape[1] != src_weight.shape[1] * 2:
            return
        matched[dst_weight_key] = torch.cat([0.5 * src_weight, 0.5 * src_weight], dim=1)
        dst_bias_key = f'{dst_prefix}.bias'
        src_bias_key = f'{src_prefix}.bias'
        if dst_bias_key in current_state and src_bias_key in payload['model_state']:
            if current_state[dst_bias_key].shape == payload['model_state'][src_bias_key].shape:
                matched[dst_bias_key] = payload['model_state'][src_bias_key]

    dual_proj_suffixes = [
        '0.weight',
        '0.bias',
        '2.weight',
        '2.bias',
    ]
    for suffix in dual_proj_suffixes:
        old_key = f'retriever.sim_proj.{suffix}'
        if old_key not in payload['model_state']:
            continue
        for prefix in ['retriever.raw_sim_proj', 'retriever.fused_sim_proj']:
            new_key = f'{prefix}.{suffix}'
            if new_key in current_state and current_state[new_key].shape == payload['model_state'][old_key].shape:
                matched[new_key] = payload['model_state'][old_key]
        fused_key = f'retriever.fused_sim_proj.{suffix}'
        delta_key = f'retriever.delta_sim_proj.{suffix}'
        if fused_key in current_state and current_state[fused_key].shape == payload['model_state'][old_key].shape:
            matched[fused_key] = payload['model_state'][old_key]
        if delta_key in current_state:
            matched[delta_key] = torch.zeros_like(current_state[delta_key])
    has_slot_branch = 'retriever.slot_seed_query.weight' in current_state
    for prefix in ['retriever.rel_feat_proj', 'retriever.div_feat_proj']:
        if prefix == 'retriever.div_feat_proj' and has_slot_branch:
            continue
        weight_key = f'{prefix}.weight'
        bias_key = f'{prefix}.bias'
        if weight_key in current_state:
            matched[weight_key] = torch.zeros_like(current_state[weight_key])
        if bias_key in current_state:
            matched[bias_key] = torch.zeros_like(current_state[bias_key])
    if 'retriever.div_feat_proj.weight' in current_state:
        _copy_linear('retriever.div_feat_proj', 'retriever.img_proj')
    if 'retriever.struct_feat_proj.weight' in current_state:
        _copy_concat_proj('retriever.struct_feat_proj', 'retriever.img_proj')
    slot_seed_map = {
        'retriever.slot_seed_query': 'retriever.proto_query',
        'retriever.slot_seed_key': 'retriever.proto_key',
        'retriever.slot_seed_value': 'retriever.proto_value',
        'retriever.slot_seed_out': 'retriever.proto_ctx_proj',
    }
    for dst_prefix, src_prefix in slot_seed_map.items():
        _copy_linear(dst_prefix, src_prefix)
    _copy_norm('retriever.slot_seed_norm', 'retriever.proto_norm')
    if 'retriever.slot_gate' in current_state:
        matched['retriever.slot_gate'] = current_state['retriever.slot_gate']

    _copy_adaptive_embed('retriever.support_embed.weight', 'retriever.intent_embed.weight')
    _copy_linear('retriever.support_query_proj', 'retriever.intent_query_proj')
    _copy_linear('retriever.support_slot_query', 'retriever.proto_query')
    _copy_norm('retriever.support_init_norm', 'retriever.intent_norm')
    _copy_decoder_block('retriever.support_decoder.layers.0', 'retriever.decoder.layers.0')
    _copy_norm('retriever.support_decoder.norm', 'retriever.decoder.norm')
    _copy_linear('retriever.coverage_seed_proj', 'retriever.proto_ctx_proj')
    _copy_linear('retriever.support_mem_proj', 'retriever.proto_value')
    _copy_norm('retriever.support_mem_norm', 'retriever.encoder.norm')
    _copy_linear('retriever.div_state_proj', 'retriever.candidate_out')
    _copy_decoder_block('retriever.coverage_decoder.layers.0', 'retriever.decoder.layers.1')
    _copy_norm('retriever.coverage_decoder.norm', 'retriever.decoder.norm')
    for gate_name in ['retriever.div_state_gate', 'retriever.support_mem_gate']:
        if gate_name in current_state:
            matched[gate_name] = current_state[gate_name]
    missing, unexpected = model.load_state_dict(matched, strict=False)
    return missing, unexpected, len(matched)


def configure_trainable_stage2_params(model, args):
    if not args.train_delta_only:
        return
    for name, param in model.named_parameters():
        param.requires_grad = (
            ('delta_sim_proj' in name)
            or ('rel_feat_proj' in name)
            or ('div_feat_proj' in name)
            or ('struct_feat_proj' in name)
            or ('div_feat_proj' in name)
            or ('slot_seed_' in name)
            or ('slot_gate' in name)
            or ('support_' in name)
            or ('coverage_decoder' in name)
            or ('coverage_seed_proj' in name)
            or ('div_state_' in name)
            or ('route_score_scale' in name)
            or ('mode_budget_head' in name)
            or ('mode_rel_scale' in name)
        )


def run_stage2(args, device, stage1_ckpt: str):
    train_dataset = FusedRawClipQueryDataset(
        args.train_dataset,
        stage1_ckpt=stage1_ckpt,
        projector_device=str(device),
        topn=args.topn,
        candidate_pool_topm=(args.candidate_pool_topm if args.candidate_pool_topm > 0 else args.topn),
        feature_extractor=args.feature_extractor,
        max_queries=args.max_train_queries,
        chunk_size=args.chunk_size,
        candidate_order=args.candidate_order,
        recall_source=args.recall_source,
        stage2_memory_source=args.stage2_memory_source,
        stage2_query_source=args.stage2_query_source,
        score_fusion_mode=args.score_fusion_mode,
        structured_bonus_scale=args.structured_bonus_scale,
    )
    sampler = DistributedSampler(train_dataset, shuffle=True) if args.use_ddp else None
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.stage2_batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_query_batch,
    )
    eval_dataset = None

    model = build_retriever(
        feature_extractor=args.feature_extractor,
        topn=args.topn,
        num_slots=args.num_slots,
        num_layers=args.num_layers,
        nhead=args.nhead,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        num_encoder_layers=args.num_encoder_layers,
        transport_tau=args.transport_tau,
        target_budget=args.target_budget,
        proto_tau=args.proto_tau,
        proto_rel_scale=args.proto_rel_scale,
        sinkhorn_iters=args.sinkhorn_iters,
        sim_mode=args.sim_mode,
        enable_support_gate=args.enable_support_gate,
        support_tau=args.support_tau,
        support_logit_scale=args.support_logit_scale,
        support_score_mode=args.support_score_mode,
        support_column_cap=args.support_column_cap,
        enable_support_relprior=args.enable_support_relprior,
        min_mode_ratio=getattr(args, 'min_mode_ratio', 1.0 / 3.0),
        minimal_memory=getattr(args, 'minimal_memory', False),
        disable_query_slots=getattr(args, 'disable_query_slots', False),
        disable_budget_learning=getattr(args, 'disable_budget_learning', False),
        disable_proto_reasoning=getattr(args, 'disable_proto_reasoning', False),
        disable_column_constraint=getattr(args, 'disable_column_constraint', False),
        disable_rank_embedding=getattr(args, 'disable_rank_embedding', False),
        disable_encoder=getattr(args, 'disable_encoder', False),
        disable_decoder=getattr(args, 'disable_decoder', False),
    ).to(device)

    if args.enable_universal_stage2:
        if args.stage2_query_source != 'raw':
            raise ValueError('enable_universal_stage2 requires stage2_query_source=raw.')
        if getattr(unwrap_model(model).retriever, 'uses_rel_div', False):
            raise ValueError('enable_universal_stage2 currently supports generic stage2 models without rel/div side channels.')

    if args.init_stage2_ckpt:
        missing, unexpected, loaded_num = load_stage2_shell(model, args.init_stage2_ckpt)
        if is_main_process(args):
            print(f'stage2 shell init: loaded {loaded_num} tensors from {args.init_stage2_ckpt}')
            if missing:
                print('stage2 shell missing keys:', missing[:20], '...' if len(missing) > 20 else '')
            if unexpected:
                print('stage2 shell unexpected keys:', unexpected[:20], '...' if len(unexpected) > 20 else '')

    configure_trainable_stage2_params(model, args)
    if is_main_process(args) and args.train_delta_only:
        trainable = [name for name, param in model.named_parameters() if param.requires_grad]
        print('stage2 delta-only trainable params:', trainable)

    if args.use_ddp:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    set_decode_controls(model, args)

    criterion = BudgetSetLoss(
        weight_ret=args.weight_ret,
        weight_cov=args.weight_cov,
        weight_dup=args.weight_dup,
        weight_proto=args.weight_proto,
        weight_support=args.weight_support,
        weight_div=getattr(args, 'weight_div', 0.0),
        cluster_alpha=args.cluster_alpha,
    )
    optimizer = AdamW((param for param in model.parameters() if param.requires_grad), lr=args.generator_lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_epoch = -1
    epochs = 1 if args.dry_run else args.stage2_epochs

    if is_main_process(args):
        print(f'stage2 train queries: {len(train_dataset)}')
        print(f'stage2 topN: {args.topn}')
        print(f'stage2 embedding source: {stage1_ckpt}')
        print(f'stage2 candidate order: {args.candidate_order}')
        print(f'stage2 recall source: {args.recall_source}')
        print(f'stage2 memory source: {args.stage2_memory_source}')
        print(f'stage2 query source: {args.stage2_query_source}')
        print(f'stage2 score fusion: {args.score_fusion_mode}')
        print(f'stage2 structured bonus scale: {args.structured_bonus_scale}')
        print(f'stage2 support gate: {args.enable_support_gate}')
        print(f'stage2 support tau: {args.support_tau}')
        print(f'stage2 support logit scale: {args.support_logit_scale}')
        print(f'stage2 support score mode: {args.support_score_mode}')
        print(f'stage2 support column cap: {args.support_column_cap}')
        print(f'stage2 support rel prior: {args.enable_support_relprior}')
        print(f'stage2 universal mix: {args.enable_universal_stage2}')
        print(f'stage2 eval decode mode: {args.decode_mode}')
        print(
            'stage2 hparams: seed=%d lr=%g wd=%g clip=%g ret=%g cov=%g dup=%g proto=%g sup=%g div=%g alpha=%g epochs=%d eval_every=%d patience=%d'
            % (
                args.seed,
                args.generator_lr,
                args.weight_decay,
                args.clip_grad,
                args.weight_ret,
                args.weight_cov,
                args.weight_dup,
                args.weight_proto,
                args.weight_support,
                args.weight_div,
                args.cluster_alpha,
                args.stage2_epochs,
                args.stage2_eval_every,
                args.early_stop_patience,
            )
        )
        print(
            'stage2 component toggles: minimal_memory=%s disable_query_slots=%s disable_budget_learning=%s disable_proto_reasoning=%s disable_column_constraint=%s disable_rank_embedding=%s disable_encoder=%s disable_decoder=%s'
            % (
                getattr(args, 'minimal_memory', False),
                getattr(args, 'disable_query_slots', False),
                getattr(args, 'disable_budget_learning', False),
                getattr(args, 'disable_proto_reasoning', False),
                getattr(args, 'disable_column_constraint', False),
                getattr(args, 'disable_rank_embedding', False),
                getattr(args, 'disable_encoder', False),
                getattr(args, 'disable_decoder', False),
            )
        )
        if args.decode_mode in {'mmr', 'filter_mmr', 'adaptive_mmr'}:
            print(f'stage2 eval MMR lambda: {args.mmr_lambda}')
            print(f'stage2 eval pre-filter N: {args.pre_filter_n}')
        if args.use_ddp:
            print(f'DDP world_size: {args.world_size}, rank0 device: {device}')

    for epoch in range(epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        train_stats = train_stage2_epoch(model, train_loader, optimizer, criterion, device, args)
        if is_main_process(args):
            print(
                'stage2 epoch %d lr %.6g loss %.4f ret %.4f cov %.4f dup %.4f proto %.4f sup %.4f'
                % (
                    epoch,
                    args.generator_lr,
                    train_stats['loss'],
                    train_stats['ret'],
                    train_stats['cov'],
                    train_stats['dup'],
                    train_stats['proto'],
                    train_stats['sup'],
                )
            )

        should_eval = args.dry_run or (args.stage2_eval_every > 0 and ((epoch + 1) % args.stage2_eval_every == 0 or epoch + 1 == epochs))
        if should_eval:
            if eval_dataset is None:
                eval_dataset = FusedPrecomputedEvalQueryDataset(
                    args.test_dataset,
                    stage1_ckpt=stage1_ckpt,
                    projector_device=str(device),
                    topn=args.topn,
                    candidate_pool_topm=(args.candidate_pool_topm if args.candidate_pool_topm > 0 else args.topn),
                    feature_extractor=args.feature_extractor,
                    max_queries=args.max_eval_queries,
                    chunk_size=args.chunk_size,
                    candidate_order=args.candidate_order,
                    recall_source=args.recall_source,
                    stage2_memory_source=args.stage2_memory_source,
                    stage2_query_source=args.stage2_query_source,
                    score_fusion_mode=args.score_fusion_mode,
                    structured_bonus_scale=args.structured_bonus_scale,
                )
            summary = evaluate_retriever(
                unwrap_model(model),
                args.test_dataset,
                args.topn,
                args.feature_extractor,
                str(device),
                args.max_eval_queries,
                eval_batch_size=args.eval_batch_size,
                eval_dataset=eval_dataset,
                distributed=args.use_ddp,
                decode_mode=args.decode_mode,
                stage1_ckpt=stage1_ckpt,
                chunk_size=args.chunk_size,
                candidate_order=args.candidate_order,
                recall_source=args.recall_source,
                stage2_memory_source=args.stage2_memory_source,
                stage2_query_source=args.stage2_query_source,
                score_fusion_mode=args.score_fusion_mode,
                structured_bonus_scale=args.structured_bonus_scale,
            )
            stop_now = False
            if is_main_process(args):
                print(format_summary(summary))
                f1_20 = summary['F1'].get(20, 0.0)
                if f1_20 > best_f1:
                    best_f1 = f1_20
                    best_epoch = epoch
                    if not args.dry_run:
                        save_stage2_checkpoint(model, args, summary, epoch, stage1_ckpt)
                        print('Saved best stage2 checkpoint with F1@20=%.2f' % best_f1)
                elif args.early_stop_patience > 0 and best_epoch >= 0 and (epoch - best_epoch) >= args.early_stop_patience:
                    print('Early stopping at epoch %d: no F1@20 improvement for %d epochs' % (epoch, epoch - best_epoch))
                    stop_now = True
            if args.use_ddp:
                stop_tensor = torch.tensor([1 if stop_now else 0], device=device, dtype=torch.int64)
                dist.broadcast(stop_tensor, src=0)
                stop_now = bool(stop_tensor.item())
            if stop_now:
                break


def main():
    args = parse_args()
    if args.only_stage1 and args.only_stage2:
        raise ValueError('Cannot set both --only_stage1 and --only_stage2.')
    if args.recall_source == 'dual' and args.sim_mode not in DUAL_RECALL_SIM_MODES:
        raise ValueError('recall_source=dual requires a dual-recall sim_mode (delta, resrouteproto, resroutetrans, etc.).')
    if args.enable_universal_stage2 and args.stage2_query_source != 'raw':
        raise ValueError('enable_universal_stage2 requires stage2_query_source=raw.')
    device = setup_distributed(args)
    set_seed(args.seed + getattr(args, 'rank', 0))
    os.makedirs(args.save_dir, exist_ok=True)

    if hasattr(torch, 'set_float32_matmul_precision'):
        torch.set_float32_matmul_precision('high')

    try:
        stage1_ckpt = args.stage1_ckpt
        if not args.skip_stage1 and not args.only_stage2:
            if is_main_process(args):
                print('Running stage1 fusion pretraining')
            stage1_ckpt = run_stage1(args, device)
            if is_main_process(args):
                print(f'stage1 best checkpoint: {stage1_ckpt}')
            if args.only_stage1:
                return
        elif not stage1_ckpt:
            raise ValueError('When --skip_stage1 is set, --stage1_ckpt must be provided.')

        if args.enable_stage1_guided:
            if is_main_process(args):
                print('Running stage1 generator-guided finetuning')
            stage1_ckpt = run_stage1_guided(args, device, stage1_ckpt)
            if is_main_process(args):
                print(f'stage1 guided checkpoint: {stage1_ckpt}')

        if args.only_stage1:
            return
        if is_main_process(args):
            print('Running stage2 generator training')
        run_stage2(args, device, stage1_ckpt)
    finally:
        cleanup_distributed(args)


if __name__ == '__main__':
    main()
