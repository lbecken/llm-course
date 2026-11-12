"""
MiniGPT — a small causal Transformer written to be read (Modules 06 and 07).

One class, two personalities. Every architectural difference between the 2019 GPT-2
design and the nanochat design is a switch on the config, so each can be studied (and
ablated) in isolation:

    switch            GPT-2 style            nanochat style
    ----------------  ---------------------  -----------------------------------------
    pos               "learned" (added)      "rope" (rotates q and k in every layer)
    norm              "layernorm" (learned)  "rmsnorm" (no learnable parameters)
    norm_after_embed  False                  True
    act               "gelu"                 "relu2"   (relu(x)^2)
    bias              True                   False
    tie_weights       True                   False     (separate embedding / unembedding)
    qk_norm           False                  True      (normalize q and k, then scale)
    logit_softcap     None                   15.0      (15 * tanh(logits / 15))

`MiniGPTConfig.gpt2_style()` and `MiniGPTConfig.nanochat_style()` set them all at once.

What MiniGPT leaves out on purpose (nanochat has them; Module 06 walks through each):
value embeddings, per-layer residual scalars (resid/x0 lambdas), "smear", "backout",
explicit bf16/fp8 dtype management, Flash Attention 3 kernels. With those extras
neutralised, labs/foundations/06_nanochat_parity.py copies nanochat's weights into a
MiniGPT and reproduces nanochat's logits.

Tensor-shape legend used in comments:
    B batch size   T sequence length   C = n_embd   H = n_head   D = head_dim = C / H   V = vocab_size
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mlle.foundations.attention import causal_mask
from mlle.foundations.positional import apply_rope, rope_cos_sin, sinusoidal_positions


@dataclass
class MiniGPTConfig:
    vocab_size: int
    block_size: int = 64  # maximum sequence length (nanochat: sequence_len)
    n_layer: int = 2
    n_head: int = 2
    n_embd: int = 64
    n_kv_head: int | None = None  # None = n_head. Fewer = grouped-query attention (Module 12)
    window: int | None = None  # None = full context. N = attend to N previous tokens + self
    mlp_ratio: int = 4
    pos: str = "learned"  # learned | sinusoidal | rope | none
    rope_base: float = 10000.0
    norm: str = "layernorm"  # layernorm | rmsnorm
    norm_after_embed: bool = False
    act: str = "gelu"  # gelu | relu2
    bias: bool = True
    tie_weights: bool = True
    qk_norm: bool = False
    qk_scale: float = 1.0  # multiplies q and k after qk_norm (nanochat: 1.2)
    logit_softcap: float | None = None
    init: str = "gpt2"  # gpt2 | nanochat

    def __post_init__(self):
        assert self.n_embd % self.n_head == 0
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_head % self.n_kv_head == 0
        assert self.pos in ("learned", "sinusoidal", "rope", "none")
        assert self.norm in ("layernorm", "rmsnorm")
        assert self.act in ("gelu", "relu2")

    @classmethod
    def gpt2_style(cls, vocab_size: int, **overrides) -> "MiniGPTConfig":
        return cls(vocab_size=vocab_size, **overrides)

    @classmethod
    def nanochat_style(cls, vocab_size: int, **overrides) -> "MiniGPTConfig":
        base = dict(
            pos="rope", rope_base=100000.0, norm="rmsnorm", norm_after_embed=True, act="relu2",
            bias=False, tie_weights=False, qk_norm=True, qk_scale=1.2, logit_softcap=15.0, init="nanochat",
        )
        base.update(overrides)
        return cls(vocab_size=vocab_size, **base)


def rms_norm(x: torch.Tensor) -> torch.Tensor:
    """Parameter-free RMSNorm: rescale each vector to root-mean-square 1.
    Identical to nanochat's `norm()` = F.rms_norm(x, (x.size(-1),))."""
    return F.rms_norm(x, (x.size(-1),))


class Norm(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.ln = nn.LayerNorm(config.n_embd, bias=config.bias) if config.norm == "layernorm" else None

    def forward(self, x):
        return self.ln(x) if self.ln is not None else rms_norm(x)


class CausalSelfAttention(nn.Module):
    def __init__(self, config: MiniGPTConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.n_head, self.n_kv_head = config.n_head, config.n_kv_head
        self.head_dim = config.n_embd // config.n_head
        self.c_q = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.last_weights: torch.Tensor | None = None  # filled when the model records internals
        # Intervention hook for experiments (Module 05): a list of (query_pos, key_pos) pairs whose
        # attention is forbidden in this layer, in addition to the causal mask. None = normal operation.
        self.blocked_edges: list[tuple[int, int]] | None = None

    def forward(self, x, cache=None, record: bool = False):
        B, T, C = x.shape
        H, Hkv, D = self.n_head, self.n_kv_head, self.head_dim
        pos0 = 0 if cache is None else cache.pos  # absolute position of x[:, 0]

        # 1) Project the residual stream into queries, keys and values, one set per head.
        q = self.c_q(x).view(B, T, H, D).transpose(1, 2)  # (B, H,   T, D)
        k = self.c_k(x).view(B, T, Hkv, D).transpose(1, 2)  # (B, Hkv, T, D)
        v = self.c_v(x).view(B, T, Hkv, D).transpose(1, 2)  # (B, Hkv, T, D)

        # 2) Positions: RoPE rotates q and k by an angle proportional to absolute position.
        if self.config.pos == "rope":
            cos, sin = rope_cos_sin(T, D, self.config.rope_base, offset=pos0)
            cos, sin = cos.to(x.device, x.dtype), sin.to(x.device, x.dtype)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        # 3) QK-norm: compare directions only, with a fixed sharpness.
        if self.config.qk_norm:
            q, k = rms_norm(q) * self.config.qk_scale, rms_norm(k) * self.config.qk_scale

        # 4) KV cache: store the new keys/values, read back everything seen so far.
        if cache is not None:
            k, v = cache.append(self.layer_idx, k, v)  # (B, Hkv, pos0 + T, D)
        Tk = k.shape[2]

        # 5) Grouped-query attention: each key/value head serves n_head / n_kv_head query heads.
        if Hkv != H:
            k = k.repeat_interleave(H // Hkv, dim=1)  # (B, H, Tk, D)
            v = v.repeat_interleave(H // Hkv, dim=1)

        # 6) Scores, mask, softmax, weighted average of values.
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(D)  # (B, H, T, Tk)
        mask = causal_mask(T, Tk, self.config.window, device=x.device)  # (T, Tk)
        if self.blocked_edges:
            mask = mask.clone()
            for q_pos, k_pos in self.blocked_edges:
                mask[q_pos, k_pos] = False
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (B, H, T, Tk) rows sum to 1
        if record:
            self.last_weights = weights.detach()
        y = weights @ v  # (B, H, T, D)

        # 7) Put the heads side by side again and mix them with one more linear map.
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.act = config.act
        self.c_fc = nn.Linear(config.n_embd, config.mlp_ratio * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.mlp_ratio * config.n_embd, config.n_embd, bias=config.bias)

    def forward(self, x):
        x = self.c_fc(x)  # (B, T, 4C)  expand
        x = F.gelu(x) if self.act == "gelu" else F.relu(x).square()
        return self.c_proj(x)  # (B, T, C)  contract


class Block(nn.Module):
    """Pre-norm residual block:  x = x + attn(norm(x));  x = x + mlp(norm(x))."""

    def __init__(self, config: MiniGPTConfig, layer_idx: int):
        super().__init__()
        self.norm1 = Norm(config)
        self.attn = CausalSelfAttention(config, layer_idx)
        self.norm2 = Norm(config)
        self.mlp = MLP(config)

    def forward(self, x, cache=None, record: bool = False):
        x = x + self.attn(self.norm1(x), cache, record)
        x = x + self.mlp(self.norm2(x))
        return x


class MiniGPT(nn.Module):
    def __init__(self, config: MiniGPTConfig):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)  # token id -> vector
        self.wpe = nn.Embedding(config.block_size, config.n_embd) if config.pos == "learned" else None
        if config.pos == "sinusoidal":
            self.register_buffer("pe", sinusoidal_positions(config.block_size, config.n_embd), persistent=False)
        self.h = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.norm_f = Norm(config)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # vector -> one score per token
        if config.tie_weights:
            self.lm_head.weight = self.wte.weight  # one matrix used for both reading and writing tokens
        self.internals: dict = {}
        self.init_weights()

    # ------------------------------------------------------------------ initialisation
    @torch.no_grad()
    def init_weights(self):
        c = self.config
        if c.init == "gpt2":
            # GPT-2: everything N(0, 0.02); projections that write into the residual stream
            # are scaled down by 1/sqrt(2 * n_layer) so the stream's variance does not grow with depth.
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, std=0.02)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
                elif isinstance(module, nn.Embedding):
                    nn.init.normal_(module.weight, std=0.02)
            for block in self.h:
                for proj in (block.attn.c_proj, block.mlp.c_proj):
                    nn.init.normal_(proj.weight, std=0.02 / math.sqrt(2 * c.n_layer))
        else:
            # nanochat (GPT.init_weights): residual-writing projections start at exactly zero,
            # so at initialisation every block is the identity and the model is a bigram-like
            # embedding -> unembedding map that the blocks then learn to correct.
            s = 3**0.5 * c.n_embd**-0.5  # uniform(-s, s) has standard deviation 1/sqrt(n_embd)
            nn.init.normal_(self.wte.weight, std=0.8)
            if not c.tie_weights:
                nn.init.normal_(self.lm_head.weight, std=0.001)
            if self.wpe is not None:
                nn.init.normal_(self.wpe.weight, std=0.02)
            for block in self.h:
                for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v):
                    nn.init.uniform_(lin.weight, -s, s)
                nn.init.zeros_(block.attn.c_proj.weight)
                nn.init.uniform_(block.mlp.c_fc.weight, -0.4 * s, 0.4 * s)
                nn.init.zeros_(block.mlp.c_proj.weight)
                for lin in (block.attn.c_q, block.attn.c_k, block.attn.c_v, block.attn.c_proj, block.mlp.c_fc, block.mlp.c_proj):
                    if lin.bias is not None:
                        nn.init.zeros_(lin.bias)

    def num_params(self, exclude_embeddings: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if exclude_embeddings:
            n -= self.wte.weight.numel()
            if self.wpe is not None:
                n -= self.wpe.weight.numel()
        return n

    # ------------------------------------------------------------------ forward
    def forward(self, idx, targets=None, cache=None, record: bool = False, loss_reduction: str = "mean"):
        """idx: (B, T) token ids. Returns logits (B, T, V), or the loss if `targets` is given.

        cache:  a mlle.inference.kvcache.KVCache; then idx holds only the NEW tokens.
        record: store attention weights and residual-stream snapshots in self.internals."""
        B, T = idx.shape
        pos0 = 0 if cache is None else cache.pos
        assert pos0 + T <= self.config.block_size, "sequence longer than block_size"

        x = self.wte(idx)  # (B, T, C)   one row of the embedding matrix per token
        if self.wpe is not None:
            x = x + self.wpe(torch.arange(pos0, pos0 + T, device=idx.device))  # (T, C) broadcast over B
        elif self.config.pos == "sinusoidal":
            x = x + self.pe[pos0 : pos0 + T]
        if self.config.norm_after_embed:
            x = rms_norm(x)

        stream = [x.detach()] if record else None
        for block in self.h:
            x = block(x, cache, record)  # (B, T, C)  each block ADDS its contribution
            if record:
                stream.append(x.detach())
        if cache is not None:
            cache.advance(T)
        x = self.norm_f(x)

        logits = self.lm_head(x)  # (B, T, V)   one score per vocabulary entry, per position
        if self.config.logit_softcap is not None:
            cap = self.config.logit_softcap
            logits = cap * torch.tanh(logits / cap)  # smoothly limit logits to (-cap, cap)

        if record:
            self.internals = {"residual_stream": stream, "attention": [b.attn.last_weights for b in self.h]}

        if targets is None:
            return logits
        # Cross-entropy at every position at once; targets equal to -1 are ignored (masking).
        return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)

    # ------------------------------------------------------------------ naive generation
    @torch.no_grad()
    def generate(self, idx, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, generator=None):
        """Autoregressive sampling WITHOUT a cache: every step re-runs the whole sequence.
        Simple and obviously correct; Module 11 replaces it with cached decoding and uses
        this function as the reference in the equality test."""
        for _ in range(max_new_tokens):
            context = idx[:, -self.config.block_size :]
            logits = self(context)[:, -1, :]  # (B, V) scores for the next token only
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            if temperature == 0.0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_id = torch.multinomial(probs, 1, generator=generator)
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    def describe(self) -> dict:
        return {"config": asdict(self.config), "num_params": self.num_params()}
