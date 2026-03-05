"""Single LLM decode step: RMSNorm → multi-head attention (static KV cache) → SwiGLU MLP → residual.

Fixed tensor shapes make the module safe for XPU graph capture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, kv_len: int, batch: int, dtype: torch.dtype, device: torch.device):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        # Pre-allocated KV cache: fixed shape, written in-place each step.
        cache_shape = (batch, n_heads, kv_len, self.head_dim)
        self.register_buffer("k_cache", torch.zeros(cache_shape, dtype=dtype, device=device))
        self.register_buffer("v_cache", torch.zeros(cache_shape, dtype=dtype, device=device))

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        # x: [batch_size, sequence_length, d_model]
        B, S, _ = x.shape

        def project_and_split(proj):
            # [B, 1, d_model] -> [B, n_heads, 1, head_dim]
            return proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)

        q = project_and_split(self.q_proj)
        k = project_and_split(self.k_proj)
        v = project_and_split(self.v_proj)

        # Write new K, V into the cache at current position — in-place.
        self.k_cache[:, :, position : position + 1, :] = k
        self.v_cache[:, :, position : position + 1, :] = v

        # Attend over the full cache (static shape — graph-safe).
        out = F.scaled_dot_product_attention(q, self.k_cache, self.v_cache, scale=1.0 / self.scale)
        # out: [B, n_heads, 1, head_dim] -> [B, 1, d_model]
        out = out.transpose(1, 2).reshape(B, S, -1)
        return self.o_proj(out)


class MLP(nn.Module):
    """swiglu MLP"""

    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up   = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class DecodeLayer(nn.Module):
    """One transformer decode layer.

    Args:
        d_model:  model dimension
        n_heads:  number of attention heads
        kv_len:   maximum sequence length (KV cache size)
        batch:    batch size (fixed for graph capture)
        dtype:    compute dtype (float16 / bfloat16)
        device:   target device
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        kv_len: int,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn  = Attention(d_model, n_heads, kv_len, batch, dtype, device)
        self.norm2 = RMSNorm(d_model)
        self.mlp   = MLP(d_model, hidden=4 * d_model)

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), position)
        x = x + self.mlp(self.norm2(x))
        return x


class TransformerStack(nn.Module):
    """A stack of N identical decode layers sharing the same config."""

    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        kv_len: int,
        batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DecodeLayer(d_model, n_heads, kv_len, batch, dtype, device)
                for _ in range(n_layers)
            ]
        )

    def forward(self, x: torch.Tensor, position: int) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, position)
        return x
