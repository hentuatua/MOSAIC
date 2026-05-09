import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mosaic.set_score.build import build_retriever
from mosaic.data.datasets import FusedPrecomputedEvalQueryDataset
from mosaic.set_score.evaluate import evaluate_retriever, format_summary


def _resolve_resource_path(path: str) -> str:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return str(candidate)
    if candidate.exists():
        return str(candidate.resolve())
    for base in (Path.cwd(), ROOT, ROOT.parent):
        resolved = (base / candidate).resolve()
        if resolved.exists():
            return str(resolved)
    return str((ROOT.parent / candidate).resolve())


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate MOSAIC retrieval.')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--stage1_ckpt', default=None)
    parser.add_argument('--test_dataset', default='data/testset/')
    parser.add_argument('--feature_extractor', default=None)
    parser.add_argument('--topn', type=int, default=None)
    parser.add_argument('--candidate_pool_topm', type=int, default=None)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--max_eval_queries', type=int, default=None)
    parser.add_argument('--eval_batch_size', type=int, default=32)
    parser.add_argument('--decode_mode', default='pair_greedy', choices=['topk', 'pair_greedy', 'mmr', 'filter_mmr', 'adaptive_mmr'])
    parser.add_argument('--mmr_lambda', type=float, default=0.7)
    parser.add_argument('--pre_filter_n', type=int, default=0)
    parser.add_argument('--chunk_size', type=int, default=2048)
    parser.add_argument('--candidate_order', default=None, choices=['raw', 'fused'])
    parser.add_argument('--recall_source', default=None, choices=['raw', 'fused', 'dual'])
    parser.add_argument('--stage2_memory_source', default=None, choices=['fused', 'raw'])
    parser.add_argument('--stage2_query_source', default=None, choices=['adapted', 'raw'])
    parser.add_argument('--score_fusion_mode', default=None, choices=['cosine', 'structured_proto'])
    parser.add_argument('--structured_bonus_scale', type=float, default=None)
    parser.add_argument('--sim_mode', default=None, choices=['single', 'dual', 'delta', 'structdelta', 'slotdelta', 'slotdiv', 'protodiv', 'factordiv', 'dualchannel', 'supportcov', 'supportbridge', 'supportroute', 'supportroutejoint', 'resroute', 'resrouteproto', 'resroutealloc', 'resroutetrans', 'modebudget'])
    parser.add_argument('--score_mode', default='model', choices=['model', 'mass', 'gated'])
    parser.add_argument('--score_tau', type=float, default=1.0)
    parser.add_argument('--json_out', default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    checkpoint_path = _resolve_resource_path(args.checkpoint)
    stage1_path = _resolve_resource_path(args.stage1_ckpt) if args.stage1_ckpt else None
    payload = torch.load(checkpoint_path, map_location='cpu')
    config = payload['config']
    feature_extractor = args.feature_extractor or config['feature_extractor']
    topn = args.topn or config['topn']
    stage1_ckpt = stage1_path or config.get('stage1_ckpt')
    if not stage1_ckpt:
        raise ValueError('stage1_ckpt is required for fused-embedding evaluation.')
    candidate_order = args.candidate_order or config.get('candidate_order', 'raw')
    recall_source = args.recall_source or config.get('recall_source', 'raw')
    stage2_memory_source = args.stage2_memory_source or config.get('stage2_memory_source', 'fused')
    stage2_query_source = args.stage2_query_source or config.get('stage2_query_source', 'raw')
    score_fusion_mode = args.score_fusion_mode or config.get('score_fusion_mode', 'cosine')
    structured_bonus_scale = args.structured_bonus_scale if args.structured_bonus_scale is not None else config.get('structured_bonus_scale', 1.0)
    sim_mode = args.sim_mode or config.get('sim_mode', 'single')
    candidate_pool_topm = args.candidate_pool_topm or config.get('candidate_pool_topm', topn)
    if recall_source == 'dual' and sim_mode not in {'dual', 'delta', 'structdelta', 'slotdelta', 'slotdiv', 'protodiv', 'factordiv', 'dualchannel', 'supportcov', 'supportbridge', 'supportroute', 'supportroutejoint', 'resroute', 'resrouteproto', 'resroutealloc'}:
        raise ValueError('recall_source=dual requires sim_mode=dual, delta, structdelta, slotdelta, slotdiv, protodiv, factordiv, dualchannel, supportcov, supportbridge, supportroute, supportroutejoint, resroute, resrouteproto, or resroutealloc.')

    model = build_retriever(
        feature_extractor=feature_extractor,
        topn=topn,
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
        sim_mode=sim_mode,
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
    )
    current_state = model.state_dict()
    matched = {
        key: value
        for key, value in payload['model_state'].items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(matched, strict=False)
    if missing:
        print('Missing keys:', missing)
    if unexpected:
        print('Unexpected keys:', unexpected)
    model._mmr_lambda = args.mmr_lambda
    model._pre_filter_n = args.pre_filter_n
    model = model.to(args.device)
    eval_dataset = FusedPrecomputedEvalQueryDataset(
        args.test_dataset,
        stage1_ckpt=stage1_ckpt,
        projector_device=args.device,
        topn=topn,
        candidate_pool_topm=candidate_pool_topm,
        feature_extractor=feature_extractor,
        max_queries=args.max_eval_queries,
        chunk_size=args.chunk_size,
        candidate_order=candidate_order,
        recall_source=recall_source,
        stage2_memory_source=stage2_memory_source,
        stage2_query_source=stage2_query_source,
        score_fusion_mode=score_fusion_mode,
        structured_bonus_scale=structured_bonus_scale,
    )
    summary = evaluate_retriever(
        model,
        args.test_dataset,
        topn,
        feature_extractor,
        args.device,
        args.max_eval_queries,
        eval_batch_size=args.eval_batch_size,
        eval_dataset=eval_dataset,
        decode_mode=args.decode_mode,
        stage1_ckpt=stage1_ckpt,
        chunk_size=args.chunk_size,
        candidate_order=candidate_order,
        recall_source=recall_source,
        stage2_memory_source=stage2_memory_source,
        stage2_query_source=stage2_query_source,
        score_fusion_mode=score_fusion_mode,
        structured_bonus_scale=structured_bonus_scale,
        score_mode=args.score_mode,
        score_tau=args.score_tau,
    )
    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump(summary, f, indent=2)
    print(format_summary(summary))


if __name__ == '__main__':
    main()
