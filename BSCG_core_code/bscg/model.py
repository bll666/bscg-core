"""BSCG as specified in manuscript Sections 2.2 and Supplement S2.

This is a manuscript-aligned implementation; see docs/IMPLEMENTATION.md for
differences from the recovered development code. No outputs are embedded here.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F


def positions(length, width, device, dtype):
    phase = torch.arange(length, device=device, dtype=dtype)[:, None]
    scale = torch.exp(torch.arange(0, width, 2, device=device, dtype=dtype)
                      * (-math.log(10000.0) / width))
    p = torch.zeros(length, width, device=device, dtype=dtype)
    p[:, 0::2] = torch.sin(phase * scale)
    p[:, 1::2] = torch.cos(phase * scale[:p[:, 1::2].shape[1]])
    return p


def reorder_blocks(x, mask, segment_points, order):
    b, t, c = x.shape
    if t % segment_points:
        raise ValueError("Input length must be divisible by segment_points")
    k = t // segment_points
    if order.shape != (b, k):
        raise ValueError("Block order has the wrong shape")
    rows = torch.arange(b, device=x.device)[:, None]
    return (x.reshape(b, k, segment_points, c)[rows, order].reshape_as(x),
            mask.reshape(b, k, segment_points, c)[rows, order].reshape_as(mask))


def shuffled_view(x, mask, segment_points, generator=None):
    k = x.shape[1] // segment_points
    order = torch.rand(len(x), k, device=x.device, generator=generator).argsort(1)
    return reorder_blocks(x, mask, segment_points, order)


def noisy_view(x, mask, continuous_channels, noise_sd=0.02, generator=None):
    noise = torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator)
    use = torch.zeros(x.shape[-1], device=x.device, dtype=x.dtype)
    use[list(continuous_channels)] = 1
    return x + noise * noise_sd * mask * use


class BSCG(nn.Module):
    def __init__(self, channels, static_width, segment_points, width=32,
                 heads=4, feedforward=64, dropout=0.2,
                 local_layers=1, global_layers=2):
        super().__init__()
        if width % heads or local_layers < 0 or global_layers < 0:
            raise ValueError("Invalid attention configuration")
        self.segment_points = int(segment_points)
        self.width = width
        self.project = nn.Linear(2 * channels, width)

        def encoder(depth):
            if depth == 0:
                return nn.Identity()
            layer = nn.TransformerEncoderLayer(
                width, heads, feedforward, dropout, activation="gelu",
                layer_norm_eps=1e-5, batch_first=True, norm_first=True)
            return nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)

        self.local = encoder(local_layers)
        self.global_encoder = encoder(global_layers)
        self.fusion = nn.Sequential(nn.Linear(width + static_width, width),
                                    nn.GELU(), nn.Dropout(dropout), nn.Linear(width, 1))
        # Retained when lambda=0 so parameter accounting matches Supplement S5.
        self.projection = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                        nn.Linear(width, width))

    def encode(self, x, mask):
        b, t, c = x.shape
        l = self.segment_points
        if mask.shape != x.shape or t % l:
            raise ValueError("Incompatible sequence/mask/segment shapes")
        k = t // l
        a = torch.cat([x, mask], -1).reshape(b * k, l, 2 * c)
        h = self.project(a)
        h = self.local(h + positions(l, self.width, h.device, h.dtype))
        h = h.mean(1).reshape(b, k, self.width)
        h = self.global_encoder(h + positions(k, self.width, h.device, h.dtype))
        return h.mean(1)

    def forward(self, x, mask, static):
        h = self.encode(x, mask)
        return self.fusion(torch.cat([h, static], -1)).squeeze(-1), h

    def auxiliary(self, anchor, x, mask, continuous_channels,
                  noise_sd=0.02, temperature=0.2, generator=None):
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        positive = noisy_view(x, mask, continuous_channels, noise_sd, generator)
        negative, negative_mask = shuffled_view(x, mask, self.segment_points, generator)
        views = (anchor, self.encode(positive, mask), self.encode(negative, negative_mask))
        z, zp, zn = [F.normalize(self.projection(h), dim=-1) for h in views]
        logits = torch.stack([(z * zp).sum(-1), (z * zn).sum(-1)], -1) / temperature
        return F.cross_entropy(logits, torch.zeros(len(x), dtype=torch.long,
                               device=x.device), reduction="none")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
