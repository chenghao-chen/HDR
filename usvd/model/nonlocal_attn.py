"""Sparse non-local attention: each patch attends to its top-K most similar patches."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

__all__ = ["SparseNonLocalAttention"]


class SparseNonLocalAttention(nn.Module):
    """
    Patch-level non-local aggregation restricted to the K nearest patches.

    The full similarity matrix is O(N^2) in the patch count, which is the
    memory bottleneck of this design: at a 512x512 input with patch_size 8,
    N = 4096 and the matrix alone is ~67 MB per image in fp32 -- multiplied
    by the batch and by every unrolled iteration that has to keep it for the
    backward pass. `chunk_size` therefore scores the queries in blocks and
    keeps only the top-K per block, so peak memory scales with
    chunk_size x N rather than N x N.

    Inputs whose spatial dims are not multiples of patch_size are reflect
    padded and cropped back, so callers do not have to align their crops.
    """

    def __init__(self, channels: int, patch_size: int = 8, top_k: int = 16,
                 d_model: int = 256, chunk_size: int = 1024):
        super().__init__()
        self.patch_size = patch_size
        self.top_k = top_k
        self.d_model = d_model
        self.chunk_size = chunk_size
        self.embed = nn.Linear(patch_size * patch_size * channels, d_model)

    def forward(self, x):
        B, C, H, W = x.shape
        p = self.patch_size

        pad_h = (p - H % p) % p
        pad_w = (p - W % p) % p
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
        Hp, Wp = x.shape[-2:]

        tokens = rearrange(x, "b c (h p1) (w p2) -> b (h w) (p1 p2 c)",
                           p1=p, p2=p)                       # (B, N, p*p*C)
        N = tokens.shape[1]
        k_eff = min(self.top_k, N - 1) if N > 1 else 1

        keys = F.normalize(self.embed(tokens), dim=-1)       # (B, N, d_model)

        out = torch.empty_like(tokens)
        scale = self.d_model ** 0.5
        for start in range(0, N, self.chunk_size):
            stop = min(start + self.chunk_size, N)
            sim = keys[:, start:stop] @ keys.transpose(1, 2)  # (B, n, N)
            # A patch is trivially its own nearest neighbour; excluding it
            # keeps the K slots for genuinely other evidence. The identity
            # path is restored by ProxDenoiser's residual connection.
            idx = torch.arange(start, stop, device=x.device)
            sim[:, torch.arange(stop - start, device=x.device), idx] = -float("inf")

            scores, top_idx = sim.topk(k_eff, dim=-1)         # (B, n, K)
            weights = torch.softmax(scores / scale, dim=-1)

            gather_idx = top_idx.unsqueeze(-1).expand(-1, -1, -1, tokens.shape[-1])
            values = torch.gather(
                tokens.unsqueeze(1).expand(-1, stop - start, -1, -1), 2, gather_idx)
            out[:, start:stop] = torch.einsum("bnk,bnkd->bnd", weights, values)

        y = rearrange(out, "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                      h=Hp // p, w=Wp // p, p1=p, p2=p, c=C)
        if pad_h or pad_w:
            y = y[..., :H, :W]
        return y
