import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class QueryAdapter(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, residual_ratio: float = 0.2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(128, int(dim * hidden_ratio))
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
        )
        self.gate = nn.Linear(dim, 1)
        self.residual_ratio = float(residual_ratio)

    def forward(self, query_feat: torch.Tensor) -> torch.Tensor:
        query_feat = F.normalize(query_feat, dim=-1)
        delta = self.net(query_feat)
        gate = torch.sigmoid(self.gate(query_feat)) * self.residual_ratio
        adapted = F.normalize((1.0 - gate) * query_feat + gate * delta, dim=-1)
        return adapted


class QueryConditionedImageAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 0.5,
        residual_ratio: float = 0.2,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(128, int(dim * hidden_ratio))
        self.input_norm = nn.LayerNorm(dim * 4)
        self.trunk = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(hidden_dim, dim)
        self.gate = nn.Linear(hidden_dim, 1)
        self.residual_ratio = float(residual_ratio)

    def _expand_query(self, x: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if query_feat.dim() == 1:
                return query_feat.unsqueeze(0).expand(x.size(0), -1)
            if query_feat.dim() == 2 and query_feat.size(0) == 1:
                return query_feat.expand(x.size(0), -1)
            return query_feat
        if query_feat.dim() == 2:
            return query_feat.unsqueeze(1).expand(-1, x.size(1), -1)
        return query_feat

    def project(self, x: torch.Tensor, query_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.normalize(x, dim=-1)
        query_feat = F.normalize(query_feat, dim=-1)
        query_feat = self._expand_query(x, query_feat)
        valid = (x.abs().sum(dim=-1, keepdim=True) > 0).float()

        pair = torch.cat([x, query_feat, x * query_feat, x - query_feat], dim=-1)
        h = self.trunk(self.input_norm(pair))
        delta = self.head(h)
        gate = torch.sigmoid(self.gate(h)) * self.residual_ratio
        enhanced = F.normalize((1.0 - gate) * x + gate * delta, dim=-1)
        enhanced = enhanced * valid

        fused_score = (enhanced * query_feat).sum(dim=-1)
        return {
            'fused': enhanced,
            'div': enhanced,
            'rel': enhanced,
            'fused_score': fused_score,
            'score': fused_score,
            'alpha': gate.squeeze(-1),
            'beta': gate.squeeze(-1),
        }

    def project_query_images(self, imgs: torch.Tensor, query_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.project(imgs, query_feat)

    def forward(self, x: torch.Tensor, query_feat: torch.Tensor = None, return_branches: bool = False):
        if query_feat is None:
            raise ValueError('QueryConditionedImageAdapter requires query_feat.')
        outputs = self.project(x, query_feat)
        if return_branches:
            return outputs
        return outputs['fused']

    @staticmethod
    def from_reldiv_projector(projector: 'RelDivFusionProjector') -> 'QueryConditionedImageAdapter':
        state = projector.state_dict()
        dim = state['rel_head.weight'].shape[0]
        hidden_dim = state['trunk.0.weight'].shape[0]
        adapter = QueryConditionedImageAdapter(
            dim=dim,
            hidden_ratio=hidden_dim / dim,
            residual_ratio=projector.residual_ratio,
        )
        adapter.input_norm.load_state_dict({
            'weight': state['input_norm.weight'],
            'bias': state['input_norm.bias'],
        })
        adapter.trunk.load_state_dict({
            k.replace('trunk.', ''): v for k, v in state.items() if k.startswith('trunk.')
        })
        adapter.head.load_state_dict({
            'weight': state['rel_head.weight'],
            'bias': state['rel_head.bias'],
        })
        adapter.gate.load_state_dict({
            'weight': state['rel_gate.weight'],
            'bias': state['rel_gate.bias'],
        })
        return adapter


class RelDivFusionProjector(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 0.5,
        residual_ratio: float = 0.2,
        fuse_ratio: float = 0.25,
        dropout: float = 0.0,
        fusion_mode: str = 'residual',
    ):
        super().__init__()
        hidden_dim = max(128, int(dim * hidden_ratio))
        self.input_norm = nn.LayerNorm(dim * 4)
        self.trunk = nn.Sequential(
            nn.Linear(dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.rel_head = nn.Linear(hidden_dim, dim)
        self.div_head = nn.Linear(hidden_dim, dim)
        self.rel_gate = nn.Linear(hidden_dim, 1)
        self.div_gate = nn.Linear(hidden_dim, 1)
        self.fuse_alpha = nn.Linear(hidden_dim, 1)
        self.fuse_beta = nn.Linear(hidden_dim, 1)
        self.residual_ratio = float(residual_ratio)
        self.fuse_ratio = float(fuse_ratio)
        self.fusion_mode = str(fusion_mode)

    def _expand_query(self, x: torch.Tensor, query_feat: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if query_feat.dim() == 1:
                return query_feat.unsqueeze(0).expand(x.size(0), -1)
            if query_feat.dim() == 2 and query_feat.size(0) == 1:
                return query_feat.expand(x.size(0), -1)
            return query_feat
        if query_feat.dim() == 2:
            return query_feat.unsqueeze(1).expand(-1, x.size(1), -1)
        return query_feat

    def _orth_project(self, feats: torch.Tensor, query_feat: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
        if feats.dim() == 2:
            residual = feats - (feats * query_feat).sum(dim=-1, keepdim=True) * query_feat
        else:
            residual = feats - torch.einsum('bnd,bnd->bn', feats, query_feat).unsqueeze(-1) * query_feat
        return F.normalize(residual, dim=-1, eps=eps)

    def project(self, x: torch.Tensor, query_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = F.normalize(x, dim=-1)
        query_feat = F.normalize(query_feat, dim=-1)
        query_feat = self._expand_query(x, query_feat)
        valid = (x.abs().sum(dim=-1, keepdim=True) > 0).float()

        pair = torch.cat([x, query_feat, x * query_feat, x - query_feat], dim=-1)
        h = self.trunk(self.input_norm(pair))

        rel_delta = self.rel_head(h)
        div_delta = self.div_head(h)
        rel_gate = torch.sigmoid(self.rel_gate(h)) * self.residual_ratio
        div_gate = torch.sigmoid(self.div_gate(h)) * self.residual_ratio
        alpha = torch.sigmoid(self.fuse_alpha(h))
        beta = torch.sigmoid(self.fuse_beta(h))

        rel_feat = F.normalize((1.0 - rel_gate) * x + rel_gate * rel_delta, dim=-1)
        div_base = F.normalize((1.0 - div_gate) * x + div_gate * div_delta, dim=-1)
        div_feat = self._orth_project(div_base, query_feat)
        if self.fusion_mode == 'residual':
            fused_feat = F.normalize(x + (alpha * self.fuse_ratio) * (rel_feat - x) + (beta * self.fuse_ratio) * div_feat, dim=-1)
        elif self.fusion_mode == 'branch':
            div_scale = beta * self.fuse_ratio * 4.0
            fused_branch = F.normalize(rel_feat + div_scale * div_feat, dim=-1)
            keep_raw = alpha * self.fuse_ratio
            fused_feat = F.normalize((1.0 - keep_raw) * fused_branch + keep_raw * x, dim=-1)
        else:
            raise ValueError(f'Unknown fusion_mode: {self.fusion_mode}')

        rel_feat = rel_feat * valid
        div_feat = div_feat * valid
        fused_feat = fused_feat * valid
        rel_score = (rel_feat * query_feat).sum(dim=-1)
        fused_score = (fused_feat * query_feat).sum(dim=-1)
        return {
            'rel': rel_feat,
            'div': div_feat,
            'fused': fused_feat,
            'score': rel_score,
            'fused_score': fused_score,
            'alpha': alpha.squeeze(-1),
            'beta': beta.squeeze(-1),
        }

    def project_query_images(self, imgs: torch.Tensor, query_feat: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.project(imgs, query_feat)

    def forward(self, x: torch.Tensor, query_feat: torch.Tensor = None, return_branches: bool = False):
        if query_feat is None:
            raise ValueError('RelDivFusionProjector requires query_feat.')
        outputs = self.project(x, query_feat)
        if return_branches:
            return outputs
        return outputs['fused']


class DiversitySetEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0):
        super().__init__()
        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.out_proj = nn.Linear(dim, dim)
        self.out_norm = nn.LayerNorm(dim)
        self.scale = 1.0 / math.sqrt(dim)

    def forward(self, query_feat: torch.Tensor, set_feats: torch.Tensor) -> torch.Tensor:
        query_feat = F.normalize(query_feat, dim=-1)
        set_feats = F.normalize(set_feats, dim=-1)

        valid = (set_feats.abs().sum(dim=-1) > 0).float()
        q = self.query_proj(query_feat).unsqueeze(1)
        k = self.key_proj(set_feats)
        v = self.value_proj(set_feats)
        dot = (q * k).sum(dim=-1) * self.scale
        gate = self.gate(torch.cat([set_feats, query_feat.unsqueeze(1).expand_as(set_feats)], dim=-1)).squeeze(-1)
        score = dot + gate
        score = score.masked_fill(valid == 0, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        mean_pool = torch.sum(set_feats * valid.unsqueeze(-1), dim=1) / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        out = self.out_proj(pooled + mean_pool)
        out = self.out_norm(out)
        return F.normalize(out, dim=-1)
