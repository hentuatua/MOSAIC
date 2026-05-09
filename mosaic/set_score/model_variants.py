from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from mosaic.set_score.set_scorer import BudgetTransportModel, IntentDecoder


class DualSimBudgetTransportModel(BudgetTransportModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sim_proj
        self.raw_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.fused_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

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
            raw_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
            fused_sims = raw_sims
        elif recall_sims.dim() == 2:
            raw_sims = recall_sims
            fused_sims = recall_sims
        elif recall_sims.dim() == 3 and recall_sims.size(-1) == 2:
            raw_sims = recall_sims[..., 0]
            fused_sims = recall_sims[..., 1]
        else:
            raise ValueError(f'Unsupported recall_sims shape: {tuple(recall_sims.shape)}')

        sim_memory = 0.5 * (
            self.raw_sim_proj(raw_sims.unsqueeze(-1))
            + self.fused_sim_proj(fused_sims.unsqueeze(-1))
        )
        memory = (
            self.img_proj(img_feats)
            + self.query_proj(query_expand)
            + self.mul_proj(img_feats * query_expand)
            + self.diff_proj(torch.abs(img_feats - query_expand))
            + sim_memory
            + self.rank_embed(rank_ids)
        )
        return self.fuse_dropout(self.fuse_norm(memory))


class ResidualSimBudgetTransportModel(BudgetTransportModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sim_proj
        self.fused_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.delta_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )

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
            raw_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
            fused_sims = raw_sims
        elif recall_sims.dim() == 2:
            raw_sims = recall_sims
            fused_sims = recall_sims
        elif recall_sims.dim() == 3 and recall_sims.size(-1) == 2:
            raw_sims = recall_sims[..., 0]
            fused_sims = recall_sims[..., 1]
        else:
            raise ValueError(f'Unsupported recall_sims shape: {tuple(recall_sims.shape)}')

        delta_sims = fused_sims - raw_sims
        sim_memory = self.fused_sim_proj(fused_sims.unsqueeze(-1)) + self.delta_sim_proj(delta_sims.unsqueeze(-1))
        memory = (
            self.img_proj(img_feats)
            + self.query_proj(query_expand)
            + self.mul_proj(img_feats * query_expand)
            + self.diff_proj(torch.abs(img_feats - query_expand))
            + sim_memory
            + self.rank_embed(rank_ids)
        )
        return self.fuse_dropout(self.fuse_norm(memory))


class StructuredResidualBudgetTransportModel(BudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sim_proj
        self.fused_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.delta_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.rel_feat_proj = nn.Linear(self.d_model, self.d_model)
        self.div_feat_proj = nn.Linear(self.d_model, self.d_model)

    def build_candidate_memory(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, num_candidates, _ = img_feats.shape
        rank_ids = torch.arange(num_candidates, device=img_feats.device).unsqueeze(0).expand(batch_size, -1)
        query_expand = query_feat.unsqueeze(1).expand(-1, num_candidates, -1)

        if recall_sims is None:
            raw_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
            fused_sims = raw_sims
        elif recall_sims.dim() == 2:
            raw_sims = recall_sims
            fused_sims = recall_sims
        elif recall_sims.dim() == 3 and recall_sims.size(-1) == 2:
            raw_sims = recall_sims[..., 0]
            fused_sims = recall_sims[..., 1]
        else:
            raise ValueError(f'Unsupported recall_sims shape: {tuple(recall_sims.shape)}')

        if rel_feats is None:
            rel_feats = img_feats
        if div_feats is None:
            div_feats = torch.zeros_like(img_feats)

        delta_sims = fused_sims - raw_sims
        sim_memory = self.fused_sim_proj(fused_sims.unsqueeze(-1)) + self.delta_sim_proj(delta_sims.unsqueeze(-1))
        feature_memory = self.rel_feat_proj(rel_feats - img_feats) + self.div_feat_proj(div_feats)
        memory = (
            self.img_proj(img_feats)
            + self.query_proj(query_expand)
            + self.mul_proj(img_feats * query_expand)
            + self.diff_proj(torch.abs(img_feats - query_expand))
            + sim_memory
            + feature_memory
            + self.rank_embed(rank_ids)
        )
        return self.fuse_dropout(self.fuse_norm(memory))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(
            query_feat,
            img_feats,
            recall_sims=recall_sims,
            rel_feats=rel_feats,
            div_feats=div_feats,
        )
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_outputs(memory, intents, pad_mask, query_feat=query_feat, rel_feats=rel_feats)


class StructuredSlotBudgetTransportModel(BudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sim_proj
        self.fused_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.delta_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.struct_feat_proj = nn.Linear(self.d_model * 2, self.d_model)
        self.slot_seed_query = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_key = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_value = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_out = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_norm = nn.LayerNorm(self.d_model)

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
            raw_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
            fused_sims = raw_sims
        elif recall_sims.dim() == 2:
            raw_sims = recall_sims
            fused_sims = recall_sims
        elif recall_sims.dim() == 3 and recall_sims.size(-1) == 2:
            raw_sims = recall_sims[..., 0]
            fused_sims = recall_sims[..., 1]
        else:
            raise ValueError(f'Unsupported recall_sims shape: {tuple(recall_sims.shape)}')

        delta_sims = fused_sims - raw_sims
        sim_memory = self.fused_sim_proj(fused_sims.unsqueeze(-1)) + self.delta_sim_proj(delta_sims.unsqueeze(-1))
        memory = (
            self.img_proj(img_feats)
            + self.query_proj(query_expand)
            + self.mul_proj(img_feats * query_expand)
            + self.diff_proj(torch.abs(img_feats - query_expand))
            + sim_memory
            + self.rank_embed(rank_ids)
        )
        return self.fuse_dropout(self.fuse_norm(memory))

    def build_intents(
        self,
        query_feat: torch.Tensor,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        intents = super().build_intents(query_feat)
        if rel_feats is None or div_feats is None:
            return intents

        struct_feats = self.struct_feat_proj(torch.cat([rel_feats, div_feats], dim=-1))
        q = F.normalize(self.slot_seed_query(intents), dim=-1)
        k = F.normalize(self.slot_seed_key(struct_feats), dim=-1)
        v = self.slot_seed_value(struct_feats)
        attn = torch.matmul(q, k.transpose(1, 2))
        if pad_mask is not None:
            attn = attn.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(attn, dim=-1)
        if pad_mask is not None:
            attn = attn * pad_mask.unsqueeze(1).float()
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_ctx = torch.matmul(attn, v)
        return self.slot_seed_norm(intents + self.slot_seed_out(slot_ctx))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat, rel_feats=rel_feats, div_feats=div_feats, pad_mask=pad_mask)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_outputs(memory, intents, pad_mask, query_feat=query_feat, rel_feats=rel_feats)


class DiversitySlotBudgetTransportModel(BudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.sim_proj
        self.fused_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.delta_sim_proj = nn.Sequential(
            nn.Linear(1, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model),
        )
        self.div_feat_proj = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_query = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_key = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_value = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_out = nn.Linear(self.d_model, self.d_model)
        self.slot_seed_norm = nn.LayerNorm(self.d_model)
        self.slot_gate = nn.Parameter(torch.tensor(-2.0))

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
            raw_sims = torch.zeros(batch_size, num_candidates, device=img_feats.device, dtype=img_feats.dtype)
            fused_sims = raw_sims
        elif recall_sims.dim() == 2:
            raw_sims = recall_sims
            fused_sims = recall_sims
        elif recall_sims.dim() == 3 and recall_sims.size(-1) == 2:
            raw_sims = recall_sims[..., 0]
            fused_sims = recall_sims[..., 1]
        else:
            raise ValueError(f'Unsupported recall_sims shape: {tuple(recall_sims.shape)}')

        delta_sims = fused_sims - raw_sims
        sim_memory = self.fused_sim_proj(fused_sims.unsqueeze(-1)) + self.delta_sim_proj(delta_sims.unsqueeze(-1))
        memory = (
            self.img_proj(img_feats)
            + self.query_proj(query_expand)
            + self.mul_proj(img_feats * query_expand)
            + self.diff_proj(torch.abs(img_feats - query_expand))
            + sim_memory
            + self.rank_embed(rank_ids)
        )
        return self.fuse_dropout(self.fuse_norm(memory))

    def build_intents(
        self,
        query_feat: torch.Tensor,
        div_feats: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        intents = super().build_intents(query_feat)
        if div_feats is None:
            return intents

        div_tokens = self.div_feat_proj(div_feats)
        q = F.normalize(self.slot_seed_query(intents), dim=-1)
        k = F.normalize(self.slot_seed_key(div_tokens), dim=-1)
        v = self.slot_seed_value(div_tokens)
        attn = torch.matmul(q, k.transpose(1, 2))
        if pad_mask is not None:
            attn = attn.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        attn = torch.softmax(attn, dim=-1)
        if pad_mask is not None:
            attn = attn * pad_mask.unsqueeze(1).float()
            attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_ctx = torch.matmul(attn, v)
        gate = torch.sigmoid(self.slot_gate)
        return self.slot_seed_norm(intents + gate * self.slot_seed_out(slot_ctx))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat, div_feats=div_feats, pad_mask=pad_mask)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_outputs(memory, intents, pad_mask, query_feat=query_feat, rel_feats=rel_feats)


class DiversityPrototypeBudgetTransportModel(ResidualSimBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.proto_div_proj = nn.Linear(self.d_model, self.d_model)
        self.proto_div_norm = nn.LayerNorm(self.d_model)
        self.proto_div_gate = nn.Parameter(torch.tensor(-2.0))

    def build_proto_memory(
        self,
        memory: torch.Tensor,
        div_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if div_feats is None:
            return memory
        gate = torch.sigmoid(self.proto_div_gate)
        return self.proto_div_norm(memory + gate * self.proto_div_proj(div_feats))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)
        proto_memory = self.build_proto_memory(memory, div_feats=div_feats)

        intents = self.build_intents(query_feat)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_outputs(
            memory,
            intents,
            pad_mask,
            proto_memory=proto_memory,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )


class FactorizedRelDivBudgetTransportModel(ResidualSimBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_key_res_proj = nn.Linear(self.d_model, self.d_model)
        self.div_state_proj = nn.Linear(self.d_model, self.d_model)
        self.rel_key_gate = nn.Parameter(torch.tensor(0.0))
        self.div_state_gate = nn.Parameter(torch.tensor(0.0))

    def _build_rel_key_states(
        self,
        memory: torch.Tensor,
        rel_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_key = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is None:
            return base_key
        gate = torch.sigmoid(self.rel_key_gate)
        rel_res = self.rel_key_res_proj(rel_feats)
        return F.normalize(base_key + gate * rel_res, dim=-1)

    def _build_div_candidate_states(
        self,
        memory: torch.Tensor,
        div_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_states = F.normalize(self.candidate_out(memory), dim=-1)
        if div_feats is None:
            return base_states
        gate = torch.sigmoid(self.div_state_gate)
        div_res = self.div_state_proj(div_feats)
        return F.normalize(base_states + gate * div_res, dim=-1)

    def _finalize_factorized_outputs(
        self,
        memory: torch.Tensor,
        intents: torch.Tensor,
        pad_mask: torch.Tensor,
        query_feat: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        div_candidate_states = self._build_div_candidate_states(memory, div_feats=div_feats)
        rel_key_states = self._build_rel_key_states(memory, rel_feats=rel_feats)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, rel_key_states.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, div_candidate_states), dim=-1)

        score_logits = self._compute_score_logits(
            intent_states,
            div_candidate_states,
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
            'candidate_states': div_candidate_states,
            'rel_key_states': rel_key_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
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
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_factorized_outputs(
            memory,
            intents,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
            div_feats=div_feats,
        )


class DualChannelRelDivBudgetTransportModel(ResidualSimBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rel_state_proj = nn.Linear(self.d_model, self.d_model)
        self.div_state_proj = nn.Linear(self.d_model, self.d_model)
        self.rel_state_gate = nn.Parameter(torch.tensor(0.0))
        self.div_state_gate = nn.Parameter(torch.tensor(0.0))

    def _build_rel_candidate_states(
        self,
        memory: torch.Tensor,
        rel_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_states = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is None:
            return base_states
        gate = torch.sigmoid(self.rel_state_gate)
        rel_res = self.rel_state_proj(rel_feats)
        return F.normalize(base_states + gate * rel_res, dim=-1)

    def _build_div_candidate_states(
        self,
        memory: torch.Tensor,
        div_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_states = F.normalize(self.candidate_out(memory), dim=-1)
        if div_feats is None:
            return base_states
        gate = torch.sigmoid(self.div_state_gate)
        div_res = self.div_state_proj(div_feats)
        return F.normalize(base_states + gate * div_res, dim=-1)

    def _finalize_dual_outputs(
        self,
        memory: torch.Tensor,
        intents: torch.Tensor,
        pad_mask: torch.Tensor,
        query_feat: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        rel_candidate_states = self._build_rel_candidate_states(memory, rel_feats=rel_feats)
        div_candidate_states = self._build_div_candidate_states(memory, div_feats=div_feats)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, rel_candidate_states.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_rel_states = F.normalize(torch.matmul(proto_attn, rel_candidate_states), dim=-1)
        intent_div_states = F.normalize(torch.matmul(proto_attn, div_candidate_states), dim=-1)
        intent_states = F.normalize(intent_rel_states + intent_div_states, dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        rel_pair_logits = torch.matmul(intent_rel_states, rel_candidate_states.transpose(1, 2))
        div_pair_logits = torch.matmul(intent_div_states, div_candidate_states.transpose(1, 2))
        score_logits = scale * 0.5 * (rel_pair_logits + div_pair_logits)
        score_logits = score_logits + self.rel_score_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            score_logits = score_logits + support_log_bias.unsqueeze(1)
        score_logits = score_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

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
            'intent_rel_states': intent_rel_states,
            'intent_div_states': intent_div_states,
            'candidate_states': div_candidate_states,
            'rel_candidate_states': rel_candidate_states,
            'div_candidate_states': div_candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
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
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        intents = self.build_intents(query_feat)
        intents = self.decoder(intents, memory, memory_key_padding_mask=~pad_mask)
        return self._finalize_dual_outputs(
            memory,
            intents,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
            div_feats=div_feats,
        )


class SupportCoverageBudgetTransportModel(ResidualSimBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 2048)
        dropout = kwargs.get('dropout', 0.1)
        super().__init__(*args, **kwargs)
        self.num_support_slots = max(1, int(round(float(self.num_slots) ** 0.5)))
        self.support_embed = nn.Embedding(self.num_support_slots, self.d_model)
        self.support_query_proj = nn.Linear(self.d_model, self.d_model)
        self.support_slot_query = nn.Linear(self.d_model, self.d_model)
        self.support_init_norm = nn.LayerNorm(self.d_model)
        self.support_decoder = IntentDecoder(
            num_layers=1,
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.coverage_seed_proj = nn.Linear(self.d_model, self.d_model)
        self.support_mem_proj = nn.Linear(self.d_model, self.d_model)
        self.support_mem_norm = nn.LayerNorm(self.d_model)
        self.div_state_proj = nn.Linear(self.d_model, self.d_model)
        self.div_state_gate = nn.Parameter(torch.tensor(0.0))
        self.support_mem_gate = nn.Parameter(torch.tensor(0.0))

    def build_support_intents(self, query_feat: torch.Tensor) -> torch.Tensor:
        support = self.support_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
        support = support + self.support_query_proj(query_feat).unsqueeze(1)
        return self.support_init_norm(support)

    def build_support_memory(
        self,
        memory: torch.Tensor,
        support_probs: torch.Tensor,
    ) -> torch.Tensor:
        gate = torch.sigmoid(self.support_mem_gate)
        support_res = self.support_mem_proj(memory)
        return self.support_mem_norm(memory + gate * support_probs.unsqueeze(-1) * support_res)

    def build_div_candidate_states(
        self,
        support_memory: torch.Tensor,
        div_feats: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        base_states = F.normalize(self.candidate_out(support_memory), dim=-1)
        if div_feats is None:
            return base_states
        gate = torch.sigmoid(self.div_state_gate)
        div_res = self.div_state_proj(div_feats)
        return F.normalize(base_states + gate * div_res, dim=-1)

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_candidate_states = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is not None:
            rel_candidate_states = F.normalize(rel_candidate_states + rel_feats, dim=-1)

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            rel_candidate_states.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_logits = torch.logsumexp(support_pair_logits, dim=1) - torch.log(
            torch.tensor(float(self.num_support_slots), device=memory.device, dtype=memory.dtype)
        )

        rel_logits = self.rel_head(memory).squeeze(-1) + support_candidate_logits
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        support_memory = self.build_support_memory(memory, support_probs)
        coverage_seed = self.coverage_seed_proj(support_hidden.mean(dim=1)).unsqueeze(1)
        intents = self.intent_norm(
            self.intent_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
            + self.intent_query_proj(query_feat).unsqueeze(1)
            + coverage_seed
        )
        intents = self.decoder(intents, support_memory, memory_key_padding_mask=~pad_mask)

        candidate_states = self.build_div_candidate_states(support_memory, div_feats=div_feats)
        proto_outputs = self.infer_prototypes(
            intents,
            support_memory,
            candidate_states,
            rel_logits,
            pad_mask,
            support_log_bias=support_log_bias,
        )
        proto_hidden = proto_outputs['proto_hidden']
        intent_states = proto_outputs['intent_states']
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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'support_memory': support_memory,
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


class SupportBridgeBudgetTransportModel(SupportCoverageBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 2048)
        dropout = kwargs.get('dropout', 0.1)
        super().__init__(*args, **kwargs)
        self.coverage_decoder = IntentDecoder(
            num_layers=1,
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_candidate_states = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is not None:
            rel_candidate_states = F.normalize(rel_candidate_states + rel_feats, dim=-1)

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            rel_candidate_states.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_logits = torch.logsumexp(support_pair_logits, dim=1) - torch.log(
            torch.tensor(float(self.num_support_slots), device=memory.device, dtype=memory.dtype)
        )

        rel_logits = self.rel_head(memory).squeeze(-1) + support_candidate_logits
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        support_memory = self.build_support_memory(memory, support_probs)
        coverage_seed = self.coverage_seed_proj(support_hidden.mean(dim=1)).unsqueeze(1)
        base_intents = self.intent_norm(
            self.intent_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
            + self.intent_query_proj(query_feat).unsqueeze(1)
            + coverage_seed
        )
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, support_memory, memory_key_padding_mask=~pad_mask)

        candidate_states = self.build_div_candidate_states(support_memory, div_feats=div_feats)
        proto_outputs = self.infer_prototypes(
            intents,
            support_memory,
            candidate_states,
            rel_logits,
            pad_mask,
            support_log_bias=support_log_bias,
        )
        proto_hidden = proto_outputs['proto_hidden']
        intent_states = proto_outputs['intent_states']
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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'support_memory': support_memory,
            'bridge_intents': bridge_intents,
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


class SupportRoutingBudgetTransportModel(SupportBridgeBudgetTransportModel):
    uses_rel_div = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_route_scale = nn.Parameter(torch.tensor(0.0))
        self.support_route_prior_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_candidate_states = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is not None:
            rel_candidate_states = F.normalize(rel_candidate_states + rel_feats, dim=-1)

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            rel_candidate_states.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_logits = torch.logsumexp(support_pair_logits, dim=1) - torch.log(
            torch.tensor(float(self.num_support_slots), device=memory.device, dtype=memory.dtype)
        )

        rel_logits = self.rel_head(memory).squeeze(-1) + support_candidate_logits
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        support_memory = self.build_support_memory(memory, support_probs)
        coverage_seed = self.coverage_seed_proj(support_hidden.mean(dim=1)).unsqueeze(1)
        base_intents = self.intent_norm(
            self.intent_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
            + self.intent_query_proj(query_feat).unsqueeze(1)
            + coverage_seed
        )
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)

        route_scale = self.support_route_scale.exp().clamp(max=100.0)
        route_query = F.normalize(bridge_intents, dim=-1)
        route_key = F.normalize(support_hidden, dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        intents = self.decoder(bridge_intents, support_memory, memory_key_padding_mask=~pad_mask)
        candidate_states = self.build_div_candidate_states(support_memory, div_feats=div_feats)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_value = self.proto_value(support_memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, rel_candidate_states.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'support_memory': support_memory,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class SupportJointRoutingBudgetTransportModel(SupportRoutingBudgetTransportModel):
    uses_rel_div = True

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_candidate_states = F.normalize(self.proto_key(memory), dim=-1)
        if rel_feats is not None:
            rel_candidate_states = F.normalize(rel_candidate_states + rel_feats, dim=-1)

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            rel_candidate_states.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_logits = torch.logsumexp(support_pair_logits, dim=1) - torch.log(
            torch.tensor(float(self.num_support_slots), device=memory.device, dtype=memory.dtype)
        )

        rel_logits = self.rel_head(memory).squeeze(-1) + support_candidate_logits
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=rel_feats,
        )

        support_memory = self.build_support_memory(memory, support_probs)
        coverage_seed = self.coverage_seed_proj(support_hidden.mean(dim=1)).unsqueeze(1)
        base_intents = self.intent_norm(
            self.intent_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
            + self.intent_query_proj(query_feat).unsqueeze(1)
            + coverage_seed
        )
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)

        route_scale = self.support_route_scale.exp().clamp(max=100.0)
        route_query = F.normalize(bridge_intents, dim=-1)
        route_key = F.normalize(support_hidden, dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        intents = self.decoder(bridge_intents, support_memory, memory_key_padding_mask=~pad_mask)
        candidate_states = self.build_div_candidate_states(support_memory, div_feats=div_feats)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_value = self.proto_value(support_memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, rel_candidate_states.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

        scale = self.logit_scale.exp().clamp(max=100.0)
        score_logits = scale * torch.matmul(intent_states, candidate_states.transpose(1, 2))
        score_logits = score_logits + self.rel_score_scale * rel_logits.unsqueeze(1)
        score_logits = score_logits + route_log_bias
        if support_log_bias is not None:
            score_logits = score_logits + support_log_bias.unsqueeze(1)
        score_logits = score_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        budget_outputs = self.compute_slot_budgets(proto_hidden)
        column_capacity = self._compute_column_capacity(support_probs, pad_mask)
        row_probs = self.rectangular_sinkhorn(score_logits, pad_mask, column_capacity=column_capacity)
        transport = budget_outputs['slot_budgets'].unsqueeze(-1) * row_probs
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'support_memory': support_memory,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class ResidualSupportRoutingBudgetTransportModel(ResidualSimBudgetTransportModel):
    def __init__(self, *args, **kwargs):
        nhead = kwargs.get('nhead', 8)
        dim_feedforward = kwargs.get('dim_feedforward', 2048)
        dropout = kwargs.get('dropout', 0.1)
        super().__init__(*args, **kwargs)
        self.num_support_slots = max(1, int(round(float(self.num_slots) ** 0.5)))
        self.support_embed = nn.Embedding(self.num_support_slots, self.d_model)
        self.support_query_proj = nn.Linear(self.d_model, self.d_model)
        self.support_slot_query = nn.Linear(self.d_model, self.d_model)
        self.support_init_norm = nn.LayerNorm(self.d_model)
        self.support_decoder = IntentDecoder(
            num_layers=1,
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.coverage_decoder = IntentDecoder(
            num_layers=1,
            d_model=self.d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.support_route_scale = nn.Parameter(torch.tensor(0.0))
        self.support_route_prior_scale = nn.Parameter(torch.tensor(1.0))

    def build_support_intents(self, query_feat: torch.Tensor) -> torch.Tensor:
        support = self.support_embed.weight.unsqueeze(0).expand(query_feat.size(0), -1, -1)
        support = support + self.support_query_proj(query_feat).unsqueeze(1)
        return self.support_init_norm(support)

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=None,
        )

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_key = F.normalize(self.proto_key(memory), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            support_key.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            support_pair_logits = support_pair_logits + support_log_bias.unsqueeze(1)
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        base_intents = self.build_intents(query_feat)
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, memory, memory_key_padding_mask=~pad_mask)

        route_scale = self.support_route_scale.exp().clamp(max=100.0)
        route_query = F.normalize(intents, dim=-1)
        route_key = F.normalize(support_hidden, dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class ResidualProtoRoutingBudgetTransportModel(ResidualSupportRoutingBudgetTransportModel):
    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=None,
        )

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_key = F.normalize(self.proto_key(memory), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            support_key.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            support_pair_logits = support_pair_logits + support_log_bias.unsqueeze(1)
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        base_intents = self.build_intents(query_feat)
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, memory, memory_key_padding_mask=~pad_mask)

        route_scale = (self.proto_scale.exp() * self.support_route_scale.exp()).clamp(max=100.0)
        route_query = F.normalize(self.proto_query(intents), dim=-1)
        route_key = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class ModeBudgetTransportModel(ResidualProtoRoutingBudgetTransportModel):
    def __init__(self, *args, min_mode_ratio: float = 1.0 / 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.mode_budget_head = nn.Linear(self.d_model, 1)
        self.mode_rel_scale = nn.Parameter(torch.tensor(1.0))
        self.min_mode_ratio = float(min_mode_ratio)

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits, pad_mask, query_feat=query_feat, rel_feats=None,
        )

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_key = F.normalize(self.proto_key(memory), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query, support_key.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            support_pair_logits = support_pair_logits + support_log_bias.unsqueeze(1)
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        base_intents = self.build_intents(query_feat)
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, memory, memory_key_padding_mask=~pad_mask)

        route_scale = (self.proto_scale.exp() * self.support_route_scale.exp()).clamp(max=100.0)
        route_query = F.normalize(self.proto_query(intents), dim=-1)
        route_key = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)
        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

        mode_rel = (support_candidate_attn * support_probs.unsqueeze(1)).sum(dim=-1)

        mode_logits = self.mode_budget_head(support_hidden).squeeze(-1)
        mode_logits = mode_logits + self.mode_rel_scale * mode_rel

        K = self.num_support_slots
        mode_weights = F.softplus(mode_logits)
        mode_budgets = self.target_budget * mode_weights / mode_weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        min_budget = self.target_budget * self.min_mode_ratio / K
        mode_budgets = mode_budgets.clamp(min=min_budget)
        mode_budgets = self.target_budget * mode_budgets / mode_budgets.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        slot_budgets_raw = torch.matmul(route_attn, mode_budgets.unsqueeze(-1)).squeeze(-1)
        slot_budgets = self.target_budget * slot_budgets_raw / slot_budgets_raw.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        slot_scores = slot_budgets / self.target_budget

        budget_outputs = {
            'slot_budget_logits': mode_logits,
            'slot_budgets': slot_budgets,
            'slot_scores': slot_scores,
            'mode_budgets': mode_budgets,
        }

        score_logits = self._compute_score_logits(
            intent_states, candidate_states, rel_logits, pad_mask,
            support_log_bias=support_log_bias,
        )

        column_capacity = self._compute_column_capacity(support_probs, pad_mask)
        row_probs = self.rectangular_sinkhorn(score_logits, pad_mask, column_capacity=column_capacity)
        transport = slot_budgets.unsqueeze(-1) * row_probs
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class RouteTransportBudgetTransportModel(ResidualProtoRoutingBudgetTransportModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.route_score_scale = nn.Parameter(torch.tensor(1.0))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        support_probs, support_log_bias, support_logits, support_prior_logits = self._compute_support(
            rel_logits, pad_mask, query_feat=query_feat, rel_feats=None,
        )

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_key = F.normalize(self.proto_key(memory), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query, support_key.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        if support_log_bias is not None:
            support_pair_logits = support_pair_logits + support_log_bias.unsqueeze(1)
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        base_intents = self.build_intents(query_feat)
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, memory, memory_key_padding_mask=~pad_mask)

        route_scale = (self.proto_scale.exp() * self.support_route_scale.exp()).clamp(max=100.0)
        route_query = F.normalize(self.proto_query(intents), dim=-1)
        route_key = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        route_candidate_prior = torch.matmul(route_attn, support_candidate_attn)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        if support_log_bias is not None:
            proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

        scaled_route_bias = self.route_score_scale * route_log_bias
        score_logits = self._compute_score_logits(
            intent_states, candidate_states, rel_logits, pad_mask,
            support_log_bias=support_log_bias,
            route_log_bias=scaled_route_bias,
        )

        budget_outputs = self.compute_slot_budgets(proto_hidden)
        column_capacity = self._compute_column_capacity(support_probs, pad_mask)
        row_probs = self.rectangular_sinkhorn(score_logits, pad_mask, column_capacity=column_capacity)
        transport = budget_outputs['slot_budgets'].unsqueeze(-1) * row_probs
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }


class ResidualSupportAllocationBudgetTransportModel(ResidualProtoRoutingBudgetTransportModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.support_slot_budget_head = nn.Linear(self.d_model, 1)
        self.support_slot_rel_scale = nn.Parameter(torch.tensor(1.0))
        self.support_struct_mix = nn.Parameter(torch.tensor(-2.0))

    def forward(
        self,
        query_feat: torch.Tensor,
        img_feats: torch.Tensor,
        recall_sims: Optional[torch.Tensor] = None,
        pad_mask: Optional[torch.Tensor] = None,
        rel_feats: Optional[torch.Tensor] = None,
        div_feats: Optional[torch.Tensor] = None,
    ):
        if query_feat.dim() == 3:
            query_feat = query_feat.squeeze(1)
        if pad_mask is None:
            pad_mask = torch.ones(img_feats.size()[:2], dtype=torch.bool, device=img_feats.device)
        else:
            pad_mask = pad_mask.bool()

        memory = self.build_candidate_memory(query_feat, img_feats, recall_sims=recall_sims)
        memory = self.encoder(memory, key_padding_mask=~pad_mask)

        rel_logits = self.rel_head(memory).squeeze(-1)
        rel_logits = rel_logits.masked_fill(~pad_mask, -1e4)
        base_support_probs, _, support_logits, support_prior_logits = self._compute_support(
            rel_logits,
            pad_mask,
            query_feat=query_feat,
            rel_feats=None,
        )

        support_intents = self.build_support_intents(query_feat)
        support_hidden = self.support_decoder(support_intents, memory, memory_key_padding_mask=~pad_mask)
        support_query = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        support_key = F.normalize(self.proto_key(memory), dim=-1)
        support_pair_logits = self.proto_scale.exp().clamp(max=100.0) * torch.matmul(
            support_query,
            support_key.transpose(1, 2),
        )
        support_pair_logits = support_pair_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        support_pair_logits = support_pair_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)
        support_candidate_attn = torch.softmax(support_pair_logits / self.proto_tau, dim=-1)
        support_candidate_attn = support_candidate_attn * pad_mask.unsqueeze(1).float()
        support_candidate_attn = support_candidate_attn / support_candidate_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        support_slot_evidence = torch.sum(
            support_candidate_attn * base_support_probs.unsqueeze(1),
            dim=-1,
        )
        support_slot_logits = self.support_slot_budget_head(support_hidden).squeeze(-1)
        support_slot_logits = support_slot_logits + self.support_slot_rel_scale * support_slot_evidence
        support_slot_gates = torch.sigmoid(support_slot_logits / self.support_tau)

        structured_support_probs = torch.sum(
            support_slot_gates.unsqueeze(-1) * support_candidate_attn,
            dim=1,
        )
        structured_support_probs = structured_support_probs.clamp(max=1.0) * pad_mask.float()

        support_mix = torch.sigmoid(self.support_struct_mix)
        support_probs = (1.0 - support_mix) * base_support_probs + support_mix * structured_support_probs
        support_probs = support_probs.clamp_min(1e-6) * pad_mask.float()
        if self.enable_support_gate:
            support_log_bias = self.support_logit_scale * torch.log(support_probs)
        else:
            support_log_bias = torch.zeros_like(rel_logits)
        support_log_bias = support_log_bias.masked_fill(~pad_mask, 0.0)

        base_intents = self.build_intents(query_feat)
        bridge_intents = self.coverage_decoder(base_intents, support_hidden)
        intents = self.decoder(bridge_intents, memory, memory_key_padding_mask=~pad_mask)

        route_scale = (self.proto_scale.exp() * self.support_route_scale.exp()).clamp(max=100.0)
        route_query = F.normalize(self.proto_query(intents), dim=-1)
        route_key = F.normalize(self.support_slot_query(support_hidden), dim=-1)
        route_logits = route_scale * torch.matmul(route_query, route_key.transpose(1, 2))
        route_attn = torch.softmax(route_logits, dim=-1)
        weighted_route_attn = route_attn * support_slot_gates.unsqueeze(1)
        route_candidate_prior = torch.matmul(weighted_route_attn, support_candidate_attn)
        route_candidate_prior = route_candidate_prior + 1e-6 * pad_mask.unsqueeze(1).float()
        route_candidate_prior = route_candidate_prior * pad_mask.unsqueeze(1).float()
        route_candidate_prior = route_candidate_prior / route_candidate_prior.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        route_log_bias = self.support_route_prior_scale * torch.log(route_candidate_prior.clamp_min(1e-6))
        route_log_bias = route_log_bias.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        candidate_states = F.normalize(self.candidate_out(memory), dim=-1)

        proto_query = F.normalize(self.proto_query(intents), dim=-1)
        proto_key = F.normalize(self.proto_key(memory), dim=-1)
        proto_value = self.proto_value(memory)
        proto_scale = self.proto_scale.exp().clamp(max=100.0)
        proto_logits = proto_scale * torch.matmul(proto_query, proto_key.transpose(1, 2))
        proto_logits = proto_logits + self.proto_rel_scale * rel_logits.unsqueeze(1)
        proto_logits = proto_logits + route_log_bias
        proto_logits = proto_logits + support_log_bias.unsqueeze(1)
        proto_logits = proto_logits.masked_fill(~pad_mask.unsqueeze(1), -1e4)

        proto_attn = torch.softmax(proto_logits / self.proto_tau, dim=-1)
        proto_attn = proto_attn * pad_mask.unsqueeze(1).float()
        proto_attn = proto_attn / proto_attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        proto_context = torch.matmul(proto_attn, proto_value)
        proto_hidden = self.proto_norm(intents + self.proto_ctx_proj(proto_context))
        intent_states = F.normalize(torch.matmul(proto_attn, candidate_states), dim=-1)

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
        candidate_mass = transport.sum(dim=1) * pad_mask.float()
        candidate_score = self._compute_candidate_score(candidate_mass, support_probs)

        return {
            'score_logits': score_logits,
            'rel_logits': rel_logits,
            'support_logits': support_logits,
            'support_probs': support_probs,
            'support_prior_logits': support_prior_logits,
            'base_support_probs': base_support_probs,
            'structured_support_probs': structured_support_probs,
            'support_mix': support_mix,
            'support_pair_logits': support_pair_logits,
            'support_hidden': support_hidden,
            'support_slot_logits': support_slot_logits,
            'support_slot_gates': support_slot_gates,
            'bridge_intents': bridge_intents,
            'route_logits': route_logits,
            'route_attn': route_attn,
            'route_candidate_prior': route_candidate_prior,
            'intent_states': intent_states,
            'candidate_states': candidate_states,
            'row_probs': row_probs,
            'transport': transport,
            'candidate_mass': candidate_mass,
            'candidate_score': candidate_score,
            'column_capacity': column_capacity,
            'proto_logits': proto_logits,
            'proto_attn': proto_attn,
            'proto_hidden': proto_hidden,
            **budget_outputs,
        }
