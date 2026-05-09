import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.self_attn(x, x, x, key_padding_mask=key_padding_mask, need_weights=False)[0]
        x = self.norm1(x + self.dropout1(y))
        y = self.linear2(self.dropout2(self.act(self.linear1(x))))
        x = self.norm2(x + y)
        return x


class MemoryEncoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, key_padding_mask=key_padding_mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.self_attn(tgt, tgt, tgt, need_weights=False)[0]
        tgt = self.norm1(tgt + self.dropout1(x))
        x = self.cross_attn(tgt, memory, memory, key_padding_mask=memory_key_padding_mask, need_weights=False)[0]
        tgt = self.norm2(tgt + self.dropout2(x))
        x = self.linear2(self.dropout3(self.act(self.linear1(tgt))))
        tgt = self.norm3(tgt + x)
        return tgt


class IntentDecoder(nn.Module):
    def __init__(self, num_layers: int, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList(
            [DecoderBlock(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(
        self,
        intents: torch.Tensor,
        memory: torch.Tensor,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            intents = layer(intents, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.norm(intents)


class BudgetTransportModel(nn.Module):
    def __init__(
        self,
        d_model: int = 512,
        num_slots: int = 20,
        num_encoder_layers: int = 2,
        num_layers: int = 4,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        max_candidates: int = 150,
        transport_tau: float = 0.1,
        rel_score_scale: float = 0.3,
        target_budget: float = 20.0,
        proto_tau: float = 0.7,
        proto_rel_scale: float = 0.2,
        sinkhorn_iters: int = 6,
        enable_support_gate: bool = False,
        support_tau: float = 1.0,
        support_logit_scale: float = 1.0,
        support_score_mode: str = 'gated',
        support_column_cap: bool = False,
        enable_support_relprior: bool = False,
        minimal_memory: bool = False,
        disable_query_slots: bool = False,
        disable_budget_learning: bool = False,
        disable_proto_reasoning: bool = False,
        disable_column_constraint: bool = False,
        disable_rank_embedding: bool = False,
        disable_encoder: bool = False,
        disable_decoder: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_slots = num_slots
        self.max_candidates = max_candidates
        self.transport_tau = transport_tau
        self.target_budget = float(target_budget)
        self.proto_tau = float(proto_tau)
        self.proto_rel_scale = float(proto_rel_scale)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.enable_support_gate = bool(enable_support_gate)
        self.support_tau = max(float(support_tau), 1e-6)
        self.support_logit_scale = float(support_logit_scale)
        self.support_column_cap = bool(support_column_cap)
        self.enable_support_relprior = bool(enable_support_relprior)
        self.minimal_memory = bool(minimal_memory)
        self.disable_query_slots = bool(disable_query_slots)
        self.disable_budget_learning = bool(disable_budget_learning)
        self.disable_proto_reasoning = bool(disable_proto_reasoning)
        self.disable_column_constraint = bool(disable_column_constraint)
        self.disable_rank_embedding = bool(disable_rank_embedding)
        self.disable_encoder = bool(disable_encoder)
        self.disable_decoder = bool(disable_decoder)
        self.uses_rel_div = bool(getattr(self, 'uses_rel_div', False) or self.enable_support_relprior)
        if support_score_mode not in {'gated', 'mass'}:
            raise ValueError(f'Unsupported support_score_mode: {support_score_mode}')
        self.support_score_mode = support_score_mode

        self.rank_embed = nn.Embedding(max_candidates, d_model)
        self.intent_embed = nn.Embedding(num_slots, d_model)

        self.img_proj = nn.Linear(d_model, d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.mul_proj = nn.Linear(d_model, d_model)
        self.diff_proj = nn.Linear(d_model, d_model)
        self.sim_proj = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fuse_norm = nn.LayerNorm(d_model)
        self.fuse_dropout = nn.Dropout(dropout)

        self.intent_query_proj = nn.Linear(d_model, d_model)
        self.intent_norm = nn.LayerNorm(d_model)

        self.encoder = MemoryEncoder(
            num_layers=num_encoder_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.decoder = IntentDecoder(
            num_layers=num_layers,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

        self.proto_query = nn.Linear(d_model, d_model)
        self.proto_key = nn.Linear(d_model, d_model)
        self.proto_value = nn.Linear(d_model, d_model)
        self.proto_ctx_proj = nn.Linear(d_model, d_model)
        self.proto_norm = nn.LayerNorm(d_model)

        self.candidate_out = nn.Linear(d_model, d_model)
        self.rel_head = nn.Linear(d_model, 1)
        self.slot_budget_head = nn.Linear(d_model, 1)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.proto_scale = nn.Parameter(torch.tensor(math.log(10.0)))
        self.rel_score_scale = nn.Parameter(torch.tensor(rel_score_scale))
        if self.enable_support_relprior:
            self.support_relprior_scale = nn.Parameter(torch.tensor(1.0))
            self.support_relprior_bias = nn.Parameter(torch.tensor(0.0))

    def build_candidate_memory(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = img_feats.shape
        rank_ids = torch.arange(num_candidates, device=img_feats.device).unsqueeze(0).expand(batch_size, -1)
        query_expand = query_feat.unsqueeze(1).expand(-1, num_candidates, -1)
        if recall_sims is None:
            recall_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
        rank_features = 0.0 if self.disable_rank_embedding else self.rank_embed(rank_ids)
        if self.minimal_memory:
            memory = (
                self.img_proj(img_feats)
                + self.query_proj(query_expand)
                + rank_features
            )
        else:
            memory = (
                self.img_proj(img_feats)
                + self.query_proj(query_expand)
                + self.mul_proj(img_feats * query_expand)
                + self.diff_proj(torch.abs(img_feats - query_expand))
                + self.sim_proj(recall_sims.unsqueeze(-1))
                + rank_features
            )
        return self.fuse_dropout(self.fuse_norm(memory))

    def build_intents(self, query_feat: torch.Tensor) -> torch.Tensor:
        intents = self.intent_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
        if not self.disable_query_slots:
            intents = intents + self.intent_query_proj(query_feat).unsqueeze(1)
        return self.intent_norm(intents)

    def rectangular_sinkhorn(
        self,
        logits: torch.Tensor,
        pad_mask: torch.Tensor,
        column_capacity: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        mask = pad_mask.unsqueeze(1).float()
        scores = logits / self.transport_tau
        scores = scores.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        scores = scores - scores.amax(dim=-1, keepdim=True)
        assign = torch.exp(scores) * mask
        cap = None
        if column_capacity is not None:
            cap = column_capacity.unsqueeze(1).clamp_min(1e-6)
        for _ in range(self.sinkhorn_iters):
            assign = assign / assign.sum(dim=-1, keepdim=True).clamp_min(1e-6)
            if not self.disable_column_constraint:
                col_sum = assign.sum(dim=1, keepdim=True)
                if cap is None:
                    assign = assign / col_sum.clamp_min(1.0)
                else:
                    assign = assign / torch.maximum(col_sum / cap, torch.ones_like(col_sum))
            assign = assign * mask
        assign = assign / assign.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        return assign

    def compute_slot_budgets(self, proto_hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        budget_logits = self.slot_budget_head(proto_hidden).squeeze(-1)
        if self.disable_budget_learning:
            uniform = self.target_budget / float(self.num_slots)
            slot_budgets = torch.full_like(budget_logits, uniform)
        else:
            budget_weights = F.softplus(budget_logits)
            slot_budgets = self.target_budget * budget_weights / budget_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_scores = slot_budgets / self.target_budget
        return {
            'slot_budget_logits': budget_logits,
            'slot_budgets': slot_budgets,
            'slot_scores': slot_scores,
        }

    def _compute_support(
        self,
        rel_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        query_feat: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
    ):
        support_prior_logits = None
        support_logits = rel_logits
        if self.enable_support_relprior and query_feat is not None and rel_feats is not None:
            support_prior_logits = torch.sum(F.normalize(rel_feats, dim=-1) * F.normalize(query_feat, dim=-1).unsqueeze(1), dim=-1)
            support_prior_logits = self.support_relprior_scale * support_prior_logits + self.support_relprior_bias
            support_prior_logits = support_prior_logits.masked_fill(~pad_mask, 0.0)
            support_logits = rel_logits + support_prior_logits
        support_probs = torch.sigmoid(support_logits / self.support_tau) * pad_mask.float()
        if self.enable_support_gate:
            support_log_bias = self.support_logit_scale * torch.log(support_probs.clamp_min(1e-6))
        else:
            support_log_bias = torch.zeros_like(rel_logits)
        support_log_bias = support_log_bias.masked_fill(~pad_mask, 0.0)
        return support_probs, support_log_bias, support_logits, support_prior_logits

    def _compute_column_capacity(
        self,
        support_probs: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if not (self.enable_support_gate and self.support_column_cap):
            return None
        capacity = support_probs.clamp_min(1e-6) * pad_mask.float()
        valid_count = pad_mask.float().sum(dim=-1, keepdim=True).clamp_min(1.0)
        slack = (float(self.num_slots) - capacity.sum(dim=-1, keepdim=True)).clamp_min(0.0) / valid_count
        capacity = (capacity + slack) * pad_mask.float()
        capacity = capacity.clamp(max=1.0)
        return capacity

    def infer_prototypes(
        self,
        intents: torch.Tensor,
        memory: torch.Tensor,
        candidate_states: torch.Tensor,
        rel_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        support_log_bias: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.disable_proto_reasoning:
            intent_states = F.normalize(intents, dim=-1)
            proto_hidden = self.proto_norm(intents)
            zeros_attn = torch.zeros(intents.size(0), intents.size(1), memory.size(1), device=intents.device, dtype=intents.dtype)
            return {
                'proto_logits': zeros_attn,
                'proto_attn': zeros_attn,
                'proto_hidden': proto_hidden,
                'intent_states': intent_states,
            }
        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)
        return {
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            'intent_states': intent_states,
        }

    def _compute_score_logits(
        self,
        intent_states: torch.Tensor,
        candidate_states: torch.Tensor,
        rel_logits: torch.Tensor,
        pad_mask: torch.Tensor,
        support_log_bias: Optional[torch.Tensor] = None,
        route_log_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=100.0)
        score_logits = scale * torch.matmul(intent_states, candidate_states.transpose(1, 2))
        score_logits = score_logits + self.rel_score_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            score_logits = score_logits + support_log_bias.unsqueeze(1)
        if route_log_bias is not None:
            score_logits = score_logits + route_log_bias
        score_logits = score_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        return score_logits

    def _compute_candidate_score(
        self,
        candidate_mass: torch.Tensor,
        support_probs: torch.Tensor,
    ) -> torch.Tensor:
        if self.enable_support_gate and self.support_score_mode == 'gated':
            return candidate_mass * support_probs + 1e-3 * support_probs
        return candidate_mass + 1e-3 * support_probs

    def _finalize_outputs(
        self,
        memory: torch.Tensor,
        intents: torch.Tensor,
        pad_mask: torch.Tensor,
        proto_memory: Optional[torch.Tensor] = None,
        query_feat: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )
        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)
        if proto_memory is None:
            proto_memory = memory

        proto_outputs = self.infer_prototypes(
            intents,
            proto_memory,
            candidate_states,
            rel_logits,
            pad_mask,
            support_log_bias=support_log_bias,
        )
        intent_states = proto_outputs['intent_states']
        proto_hidden = proto_outputs['proto_hidden']

        score_logits = self._compute_score_logits(
            intent_states,
            candidate_states,
            rel_logits,
            pad_mask,
            support_log_bias=support_log_bias,
        )

        budget_outputs = self.compute_slot_budgets(proto_hidden)
        column_capacity = self._compute_column_capacity(support_probs, pad_mask)
        row_probs = self.rectangular_sinkhorn(score_logits, pad_mask, column_capacity=column_capacity)
        transport = budget_outputs['slot_budgets'].unsqueeze(-1) * row_probs
        candidate_mass = transport.sum(dim=1)
        candidate_mass = candidate_mass * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            **proto_outputs,
            **budget_outputs,
        }

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        if not self.disable_encoder:
            memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat)
        if not self.disable_decoder:
            intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_outputs(memory, intents, pad_mask, query_feat=query_feat, rel_feats=rel_feats)
