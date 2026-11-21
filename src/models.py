import torch
import torch.nn as nn

class TemporalEncoder(nn.Module):
    def __init__(self, d, layers, heads, dropout):
        super().__init__()
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=4 * d, dropout=dropout,
                                           batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)

    def forward(self, h):
        return self.norm(self.enc(h))

def _pos(d):
    p = nn.Parameter(torch.zeros(1, 12, d))
    nn.init.normal_(p, std=0.02)
    return p

class S2OnlyTransformer(nn.Module):
    """Baseline: Sentinel-2 bands only, ignores the availability mask."""
    def __init__(self, n_classes=10, d=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.proj, self.pos = nn.Linear(11, d), _pos(d)
        self.enc, self.head = TemporalEncoder(d, layers, heads, dropout), nn.Linear(d, n_classes)

    def forward(self, x, m):
        return self.head(self.enc(self.proj(x[:, :, 2:13]) + self.pos).mean(1))

class ConcatTransformer(nn.Module):
    """Baseline: all 13 bands concatenated per month, ignores the availability mask."""
    def __init__(self, n_classes=10, d=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        self.proj, self.pos = nn.Linear(13, d), _pos(d)
        self.enc, self.head = TemporalEncoder(d, layers, heads, dropout), nn.Linear(d, n_classes)

    def forward(self, x, m):
        return self.head(self.enc(self.proj(x) + self.pos).mean(1))

class MissingAwareFusion(nn.Module):
    """Proposed: per-sensor encoders, learned missing-token per sensor, bidirectional
    cross-attention between sensors (with a learned null key so an all-missing sensor
    never produces NaNs), concat-projection fusion, then a temporal transformer."""
    def __init__(self, n_classes=10, d=64, layers=2, heads=4, dropout=0.1):
        super().__init__()
        def enc(i): return nn.Sequential(nn.Linear(i, d), nn.LayerNorm(d), nn.GELU())
        self.enc1, self.enc2 = enc(2), enc(11)
        self.miss1, self.miss2 = nn.Parameter(torch.zeros(d)), nn.Parameter(torch.zeros(d))
        nn.init.normal_(self.miss1, std=0.02); nn.init.normal_(self.miss2, std=0.02)
        self.pos = _pos(d)
        self.null1, self.null2 = nn.Parameter(torch.zeros(1, 1, d)), nn.Parameter(torch.zeros(1, 1, d))
        self.attn_2from1 = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.attn_1from2 = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.fuse = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.LayerNorm(d))
        self.tenc = TemporalEncoder(d, layers, heads, dropout)
        self.head = nn.Linear(d, n_classes)

    @staticmethod
    def _cross(attn, norm, q, kv, kv_obs, null):
        B = q.shape[0]
        keys = torch.cat([null.expand(B, -1, -1), kv], dim=1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=q.device), ~kv_obs], dim=1)
        out, _ = attn(q, keys, keys, key_padding_mask=pad, need_weights=False)
        return norm(q + out)

    def forward(self, x, m):
        m1, m2 = m[:, :, 0], m[:, :, 1]
        h1 = torch.where(m1.unsqueeze(-1), self.enc1(x[:, :, 0:2]), self.miss1) + self.pos
        h2 = torch.where(m2.unsqueeze(-1), self.enc2(x[:, :, 2:13]), self.miss2) + self.pos
        c2 = self._cross(self.attn_2from1, self.norm2, h2, h1, m1, self.null1)
        c1 = self._cross(self.attn_1from2, self.norm1, h1, h2, m2, self.null2)
        h = self.fuse(torch.cat([c1, c2], dim=-1))
        return self.head(self.tenc(h).mean(1))

def build_model(name, **kw):
    return {"s2only": S2OnlyTransformer, "concat": ConcatTransformer,
            "ma": MissingAwareFusion}[name](**kw)