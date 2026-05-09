import torch.nn as nn

from mosaic.set_score.set_scorer import BudgetTransportModel
from mosaic.set_score.model_variants import (
    ResidualSupportAllocationBudgetTransportModel,
    ResidualProtoRoutingBudgetTransportModel,
    RouteTransportBudgetTransportModel,
    ModeBudgetTransportModel,
    ResidualSupportRoutingBudgetTransportModel,
    SupportBridgeBudgetTransportModel,
    SupportJointRoutingBudgetTransportModel,
    SupportRoutingBudgetTransportModel,
    DiversitySlotBudgetTransportModel,
    DiversityPrototypeBudgetTransportModel,
    DualChannelRelDivBudgetTransportModel,
    FactorizedRelDivBudgetTransportModel,
    DualSimBudgetTransportModel,
    ResidualSimBudgetTransportModel,
    SupportCoverageBudgetTransportModel,
    StructuredResidualBudgetTransportModel,
    StructuredSlotBudgetTransportModel,
)


class RetrievalModel(nn.Module):
    def __init__(self, retriever: nn.Module):
        super().__init__()
        self.retriever = retriever

    def forward(self, query_feat, img_feats, recall_sims=None, pad_mask=None, rel_feats=None, div_feats=None):
        if getattr(self.retriever, 'uses_rel_div', False):
            return self.retriever(
                query_feat,
                img_feats,
                recall_sims=recall_sims,
                pad_mask=pad_mask,
                rel_feats=rel_feats,
                div_feats=div_feats,
            )
        return self.retriever(query_feat, img_feats, recall_sims=recall_sims, pad_mask=pad_mask)


def build_retriever(
    feature_extractor: str,
    topn: int,
    num_slots: int,
    num_layers: int,
    nhead: int,
    dim_feedforward: int,
    dropout: float,
    num_encoder_layers: int = 2,
    transport_tau: float = 0.1,
    target_budget: float = 20.0,
    proto_tau: float = 0.7,
    proto_rel_scale: float = 0.2,
    sinkhorn_iters: int = 6,
    sim_mode: str = 'single',
    enable_support_gate: bool = False,
    support_tau: float = 1.0,
    support_logit_scale: float = 1.0,
    support_score_mode: str = 'gated',
    support_column_cap: bool = False,
    enable_support_relprior: bool = False,
    min_mode_ratio: float = 1.0 / 3.0,
    minimal_memory: bool = False,
    disable_query_slots: bool = False,
    disable_budget_learning: bool = False,
    disable_proto_reasoning: bool = False,
    disable_column_constraint: bool = False,
    disable_rank_embedding: bool = False,
    disable_encoder: bool = False,
    disable_decoder: bool = False,
):
    d_model = {'clip': 512, 'clip-r50': 1024, 'groupvit': 256}[feature_extractor]
    if sim_mode == 'dual':
        model_cls = DualSimBudgetTransportModel
    elif sim_mode == 'delta':
        model_cls = ResidualSimBudgetTransportModel
    elif sim_mode == 'structdelta':
        model_cls = StructuredResidualBudgetTransportModel
    elif sim_mode == 'slotdelta':
        model_cls = StructuredSlotBudgetTransportModel
    elif sim_mode == 'slotdiv':
        model_cls = DiversitySlotBudgetTransportModel
    elif sim_mode == 'protodiv':
        model_cls = DiversityPrototypeBudgetTransportModel
    elif sim_mode == 'factordiv':
        model_cls = FactorizedRelDivBudgetTransportModel
    elif sim_mode == 'dualchannel':
        model_cls = DualChannelRelDivBudgetTransportModel
    elif sim_mode == 'supportcov':
        model_cls = SupportCoverageBudgetTransportModel
    elif sim_mode == 'supportbridge':
        model_cls = SupportBridgeBudgetTransportModel
    elif sim_mode == 'supportroute':
        model_cls = SupportRoutingBudgetTransportModel
    elif sim_mode == 'supportroutejoint':
        model_cls = SupportJointRoutingBudgetTransportModel
    elif sim_mode == 'resroute':
        model_cls = ResidualSupportRoutingBudgetTransportModel
    elif sim_mode == 'resrouteproto':
        model_cls = ResidualProtoRoutingBudgetTransportModel
    elif sim_mode == 'resroutealloc':
        model_cls = ResidualSupportAllocationBudgetTransportModel
    elif sim_mode == 'resroutetrans':
        model_cls = RouteTransportBudgetTransportModel
    elif sim_mode == 'modebudget':
        model_cls = ModeBudgetTransportModel
    else:
        model_cls = BudgetTransportModel
    return RetrievalModel(
        model_cls(
            d_model=d_model,
            num_slots=num_slots,
            num_encoder_layers=num_encoder_layers,
            num_layers=num_layers,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            max_candidates=topn,
            transport_tau=transport_tau,
            target_budget=target_budget,
            proto_tau=proto_tau,
            proto_rel_scale=proto_rel_scale,
            sinkhorn_iters=sinkhorn_iters,
            enable_support_gate=enable_support_gate,
            support_tau=support_tau,
            support_logit_scale=support_logit_scale,
            support_score_mode=support_score_mode,
            support_column_cap=support_column_cap,
            enable_support_relprior=enable_support_relprior,
            minimal_memory=minimal_memory,
            disable_query_slots=disable_query_slots,
            disable_budget_learning=disable_budget_learning,
            disable_proto_reasoning=disable_proto_reasoning,
            disable_column_constraint=disable_column_constraint,
            disable_rank_embedding=disable_rank_embedding,
            disable_encoder=disable_encoder,
            disable_decoder=disable_decoder,
            **(dict(min_mode_ratio=min_mode_ratio) if sim_mode == 'modebudget' else {}),
        )
    )
