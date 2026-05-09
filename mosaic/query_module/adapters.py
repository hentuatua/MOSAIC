import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageAdapter(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.25, residual_ratio: float = 0.2, dropout: float = 0.0):
        super().__init__()
        hidden_dim = max(64, int(dim * hidden_ratio))
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.dropout = nn.Dropout(dropout)
        self.residual_ratio = residual_ratio

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc2(self.dropout(self.act(self.fc1(x))))
        x = self.residual_ratio * x + (1.0 - self.residual_ratio) * residual
        return F.normalize(x, dim=-1)


class RelDivSingleEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_ratio: float = 0.5,
        residual_ratio: float = 0.2,
        fuse_ratio: float = 0.15,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = max(128, int(dim * hidden_ratio))
        self.input_norm = nn.LayerNorm(dim)
        self.trunk = nn.Sequential(
            nn.Linear(dim, hidden_dim),
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
        self.fuse_gate = nn.Linear(hidden_dim, 1)
        self.residual_ratio = float(residual_ratio)
        self.fuse_ratio = float(fuse_ratio)

    def project(self, x: torch.Tensor):
        x = F.normalize(x, dim=-1)
        h = self.trunk(self.input_norm(x))
        rel_delta = self.rel_head(h)
        div_delta = self.div_head(h)
        rel_gate = torch.sigmoid(self.rel_gate(h)) * self.residual_ratio
        div_gate = torch.sigmoid(self.div_gate(h)) * self.residual_ratio
        fuse_gate = torch.sigmoid(self.fuse_gate(h)) * self.fuse_ratio

        rel_feat = F.normalize((1.0 - rel_gate) * x + rel_gate * rel_delta, dim=-1)
        div_feat = F.normalize((1.0 - div_gate) * x + div_gate * div_delta, dim=-1)
        fused_feat = F.normalize(rel_feat + fuse_gate * div_feat, dim=-1)
        return {
            'rel': rel_feat,
            'div': div_feat,
            'fused': fused_feat,
            'fuse_gate': fuse_gate.squeeze(-1),
        }

    def forward(self, x: torch.Tensor, return_branches: bool = False) -> torch.Tensor:
        outputs = self.project(x)
        if return_branches:
            return outputs
        return outputs['fused']


class QDPProjector(nn.Module):
    def __init__(self, dim: int, hidden_ratio: float = 0.5, residual_ratio: float = 0.2, dropout: float = 0.0):
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
        self.residual_ratio = residual_ratio

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

    def project(self, x: torch.Tensor, query_feat: torch.Tensor):
        query_feat = F.normalize(query_feat, dim=-1)
        query_feat = self._expand_query(x, query_feat)
        pair = torch.cat([x, query_feat, x * query_feat, x - query_feat], dim=-1)
        h = self.trunk(self.input_norm(pair))

        rel_delta = self.rel_head(h)
        div_delta = self.div_head(h)
        rel_gate = torch.sigmoid(self.rel_gate(h)) * self.residual_ratio
        div_gate = torch.sigmoid(self.div_gate(h)) * self.residual_ratio

        rel_feat = F.normalize((1.0 - rel_gate) * x + rel_gate * rel_delta, dim=-1)
        div_base = F.normalize((1.0 - div_gate) * x + div_gate * div_delta, dim=-1)
        div_feat = self._orth_project(div_base, query_feat)
        rel_score = (rel_feat * query_feat).sum(dim=-1)
        return {
            'rel': rel_feat,
            'div': div_feat,
            'score': rel_score,
        }

    def project_query_images(self, imgs: torch.Tensor, query_feat: torch.Tensor):
        return self.project(imgs, query_feat)

    def forward(self, x: torch.Tensor, query_feat: torch.Tensor = None, return_branches: bool = False) -> torch.Tensor:
        if query_feat is None:
            raise ValueError('QDPProjector requires query_feat.')
        outputs = self.project(x, query_feat)
        if return_branches:
            return outputs
        return outputs['rel']


class QueryConditionedSetEncoder(nn.Module):
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

        rel = (set_feats * query_feat.unsqueeze(1)).sum(dim=-1)
        dot = (q * k).sum(dim=-1) * self.scale
        gate = self.gate(torch.cat([set_feats, query_feat.unsqueeze(1).expand_as(set_feats)], dim=-1)).squeeze(-1)
        score = dot + rel + gate
        score = score.masked_fill(valid == 0, -1e4)
        attn = torch.softmax(score, dim=-1)
        attn = attn * valid
        attn = attn / attn.sum(dim=-1, keepdim=True).clamp_min(1e-6)

        pooled = torch.sum(attn.unsqueeze(-1) * v, dim=1)
        mean_pool = torch.sum(set_feats * valid.unsqueeze(-1), dim=1) / valid.sum(dim=-1, keepdim=True).clamp_min(1.0)
        out = self.out_proj(pooled + mean_pool)
        out = self.out_norm(out)
        return F.normalize(out, dim=-1)
