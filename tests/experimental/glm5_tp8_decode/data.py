# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic test data in the real GLM geometry.

Weights are synthetic checkpoint layers: full-shape tensors under the Hugging
Face GLM-5 names without the ``model.layers.{i}.`` prefix, as the packers
expect. Full-size weights are built only by the eight-GPU tests. The replicated
MoE router/normalization tensors are broadcast from one rank.
"""

from __future__ import annotations

import math
import torch

HIDDEN, QLORA, KVLORA = 6144, 2048, 512
NOPE, ROPE, VDIM, HEADS, NUM_HEADS = 192, 64, 256, 10, 64
IDX_HEADS, IDX_DIM, WORLD = 32, 128, 8
MOE_BANKS = ("mlp.shared_experts",) + tuple(f"mlp.experts.{e}" for e in range(256))
MOE_REPLICATED = (
    "post_attention_layernorm.weight",
    "mlp.gate.weight",
    "mlp.gate.e_score_correction_bias",
)
KI_CONC_SIGMA = {"L1": 0.003, "T": 0.05}


def make_input(q_len: int, seed: int, device, *, cancellation=False):
    generator = torch.Generator().manual_seed(seed)
    value = torch.randn(q_len, HIDDEN, generator=generator).bfloat16()
    if cancellation:
        value.zero_()
        value[:, 0] = 0 if seed == 23 else 2
    return value.to(device)


def make_rope_table(capacity: int, device, theta=8000000.0):
    # Built on the CPU in FP32, as a checkpoint loader would. The kernels and
    # the references read the same table; no reference recomputes sin/cos.
    inv = 1.0 / (theta ** (torch.arange(0, ROPE, 2, dtype=torch.float32) / ROPE))
    angles = torch.outer(torch.arange(capacity + 8), inv)
    table = torch.view_as_real(torch.polar(torch.ones_like(angles), angles))
    return table.reshape(capacity + 8, ROPE).contiguous().to(device)


def make_attention_input(q_len, seed, device):
    # Draw four rows even for q1 so that q1 and q4 share the same first row.
    generator = torch.Generator(device=device).manual_seed(seed * 1000 + 900)
    return (
        torch.randn(4, HIDDEN, generator=generator, device=device)
        .bfloat16()[:q_len]
        .contiguous()
    )


def make_selection(q_len, past, seed):
    """Explicit accepted rows, with changed membership across replay seeds."""
    ids = torch.full((q_len, 2048), -1, dtype=torch.int32)
    generator = torch.Generator().manual_seed(1000003 * seed + past)
    for query in range(q_len):
        visible = past + query + 1
        if visible <= 2048:
            ids[query, :visible] = torch.arange(visible, dtype=torch.int32)
        else:
            scores = torch.randn(visible, generator=generator, dtype=torch.float64)
            ids[query] = torch.topk(scores, 2048).indices.sort().values.int()
    return ids


def make_moe_layer(rank: int, seed: int, device):
    """One sparse layer's MoE checkpoint tensors in full shapes, as seen by ``rank``.

    Rank ``r`` draws only intermediate channels ``[256 * r, 256 * r + 256)`` of
    the shared and the 256 routed experts, from ``seed + r``; the other channels
    stay zero because the packer of rank ``r`` never reads them, so the eight
    ranks together hold one consistent layer. Rank zero also draws the
    replicated norm, router and correction bias, which the caller broadcasts.
    """
    generator = torch.Generator(device=device).manual_seed(seed + rank)

    def codes(shape):
        values = torch.randint(
            0, 32, shape, device=device, dtype=torch.uint8, generator=generator
        ).add_(0x28)
        signs = torch.randint(
            0, 2, shape, device=device, dtype=torch.uint8, generator=generator
        )
        values.bitwise_or_(signs.mul_(128))
        return values

    # Rank-local operands of every bank (shared expert first), drawn in one
    # stream: gate, up, their scales, then down and its scales.
    gate = codes((257, 256, 6144))
    up = codes((257, 256, 6144))
    gate_scale = 0.001 + 0.002 * torch.rand(
        257, 2, 48, device=device, generator=generator
    )
    up_scale = 0.001 + 0.002 * torch.rand(
        257, 2, 48, device=device, generator=generator
    )
    down = codes((257, 6144, 256))
    down_scale = 0.001 + 0.002 * torch.rand(
        257, 48, 2, device=device, generator=generator
    )
    rows = slice(256 * rank, 256 * rank + 256)
    blocks = slice(2 * rank, 2 * rank + 2)
    layer = {}

    def place(name, shape, region, local):
        full = torch.zeros(shape, dtype=local.dtype, device=device)
        full[region] = local
        layer[name] = (
            full.view(torch.float8_e4m3fn) if local.dtype == torch.uint8 else full
        )

    for bank, prefix in enumerate(MOE_BANKS):
        for name, local, scale in (
            ("gate_proj", gate[bank], gate_scale[bank]),
            ("up_proj", up[bank], up_scale[bank]),
        ):
            place(f"{prefix}.{name}.weight", (2048, 6144), rows, local)
            place(f"{prefix}.{name}.weight_scale_inv", (16, 48), blocks, scale)
        place(
            f"{prefix}.down_proj.weight", (6144, 2048), (slice(None), rows), down[bank]
        )
        place(
            f"{prefix}.down_proj.weight_scale_inv",
            (48, 16),
            (slice(None), blocks),
            down_scale[bank],
        )
    if rank == 0:
        layer["post_attention_layernorm.weight"] = (
            (0.8 + 0.4 * torch.rand(6144, device=device, generator=generator))
            .bfloat16()
            .float()
        )
        layer["mlp.gate.weight"] = (
            0.02 * torch.randn(256, 6144, device=device, generator=generator)
        ).bfloat16()
        layer["mlp.gate.e_score_correction_bias"] = 0.01 * torch.randn(
            256, device=device, generator=generator
        )
    return layer


def make_moe_cancellation(layer, rank, device):
    """Force a TP sum where an extra BF16 rounding loses the 1/256 result."""
    from flashinfer.experimental.glm5_tp8_decode.weights import shard_moe_weights

    from . import moe_reference as ref

    # Rank-local views: writing them edits the layer that the packer reads.
    local = shard_moe_weights(layer, rank=rank)
    local["post_attention_layernorm.weight"].fill_(1)
    for name, tensor in local.items():
        if "experts" in name and name.endswith(".weight"):
            tensor.view(torch.uint8).zero_()
        elif name.endswith(".weight_scale_inv"):
            tensor.fill_(1)
    shared = "mlp.shared_experts"
    for name in ("gate_proj", "up_proj"):
        local[f"{shared}.{name}.weight"].view(torch.uint8)[:, 0].fill_(0x38)
    x = make_input(1, 1, device, cancellation=True)
    norm = ref.rmsnorm(x, local["post_attention_layernorm.weight"]).to(device)
    quantized, scales = ref.q128(ref.shared_mid(norm, local))
    denominator = quantized[0, :128].float().sum() * scales[0, 0]
    assert denominator > 0
    if rank < 2:
        down = local[f"{shared}.down_proj.weight"].view(torch.uint8)
        down[0, :128].fill_(0xB8 if rank == 0 else 0x38)
        local[f"{shared}.down_proj.weight_scale_inv"][0, 0] = (
            2 if rank == 0 else 1 / 256
        ) / denominator
    return layer


def _gen(seed: int, device) -> torch.Generator:
    return torch.Generator(device=torch.device(device)).manual_seed(int(seed))


def _fp8(g: torch.Generator, *shape, device) -> torch.Tensor:
    return (
        torch.randn(*shape, generator=g, device=device, dtype=torch.float32) * 0.5
    ).to(torch.float8_e4m3fn)


def _scale(g: torch.Generator, *shape, K: int, device) -> torch.Tensor:
    return (
        torch.rand(*shape, generator=g, device=device, dtype=torch.float32) + 0.5
    ) / math.sqrt(K)


def _gamma(g: torch.Generator, n: int, device) -> torch.Tensor:
    return 1.0 + 0.1 * torch.randn(n, generator=g, device=device, dtype=torch.float32)


def make_attention_layer(seed: int) -> dict:
    """One layer's attention and indexer checkpoint tensors (all 64 heads), on the CPU.

    Every rank draws the same tensors from ``seed`` with a CPU generator, so
    replicated tensors agree across ranks; each rank packs and checks only its
    own share. Norm weights are BF16, as in the checkpoints.
    """
    g = _gen(seed * 1000, "cpu")
    dev = torch.device("cpu")

    def gamma(n):
        return _gamma(g, n, dev).bfloat16()

    W = {}
    W["input_layernorm.weight"] = gamma(HIDDEN)
    W["self_attn.q_a_proj.weight"] = _fp8(g, QLORA, HIDDEN, device=dev)
    W["self_attn.q_a_proj.weight_scale_inv"] = _scale(
        g, QLORA // 128, HIDDEN // 128, K=HIDDEN, device=dev
    )
    W["self_attn.q_a_layernorm.weight"] = gamma(QLORA)
    W["self_attn.kv_a_proj_with_mqa.weight"] = _fp8(
        g, KVLORA + ROPE, HIDDEN, device=dev
    )
    W["self_attn.kv_a_proj_with_mqa.weight_scale_inv"] = _scale(
        g, (KVLORA + ROPE + 127) // 128, HIDDEN // 128, K=HIDDEN, device=dev
    )
    W["self_attn.kv_a_layernorm.weight"] = gamma(KVLORA)
    W["self_attn.q_b_proj.weight"] = _fp8(
        g, NUM_HEADS * (NOPE + ROPE), QLORA, device=dev
    )
    W["self_attn.q_b_proj.weight_scale_inv"] = _scale(
        g, NUM_HEADS * (NOPE + ROPE) // 128, QLORA // 128, K=QLORA, device=dev
    )
    W["self_attn.kv_b_proj.weight"] = _fp8(
        g, NUM_HEADS * (NOPE + VDIM), KVLORA, device=dev
    )
    W["self_attn.kv_b_proj.weight_scale_inv"] = _scale(
        g, NUM_HEADS * (NOPE + VDIM) // 128, KVLORA // 128, K=KVLORA, device=dev
    )
    W["self_attn.o_proj.weight"] = _fp8(g, HIDDEN, NUM_HEADS * VDIM, device=dev)
    W["self_attn.o_proj.weight_scale_inv"] = _scale(
        g, HIDDEN // 128, NUM_HEADS * VDIM // 128, K=HEADS * VDIM, device=dev
    )
    W["self_attn.indexer.wk.weight"] = _fp8(g, IDX_DIM, HIDDEN, device=dev)
    W["self_attn.indexer.wk.weight_scale_inv"] = _scale(
        g, 1, HIDDEN // 128, K=HIDDEN, device=dev
    )
    W["self_attn.indexer.wq_b.weight"] = _fp8(g, IDX_HEADS * IDX_DIM, QLORA, device=dev)
    W["self_attn.indexer.wq_b.weight_scale_inv"] = _scale(
        g, IDX_HEADS * IDX_DIM // 128, QLORA // 128, K=QLORA, device=dev
    )
    W["self_attn.indexer.weights_proj.weight"] = (
        torch.randn(IDX_HEADS, HIDDEN, generator=g, dtype=torch.float32)
        * 0.5
        / math.sqrt(HIDDEN)
    ).to(torch.bfloat16)
    W["self_attn.indexer.k_norm.weight"] = gamma(IDX_DIM)
    W["self_attn.indexer.k_norm.bias"] = (
        0.1 * torch.randn(IDX_DIM, generator=g, dtype=torch.float32)
    ).bfloat16()
    return W


def to_device(weights: dict, device) -> dict:
    """Copy a weight mapping to ``device``; FP8 codes move as bytes."""
    result = {}
    for name, tensor in weights.items():
        if tensor.dtype == torch.float8_e4m3fn:
            result[name] = tensor.view(torch.uint8).to(device).view(tensor.dtype)
        else:
            result[name] = tensor.to(device)
    return result


def ki_sigma(seed: int):
    if 9000 <= seed < 9100:
        return KI_CONC_SIGMA["L1"]
    if 9100 <= seed < 10000:
        return KI_CONC_SIGMA["T"]
    return None


def make_state(rank: int, cache_rows: int, seed: int, device) -> dict:
    dev = torch.device(device)
    if rank == 0:
        g = _gen(seed * 1000 + 500, dev)
        sigma = ki_sigma(seed)
        if sigma is None:
            return {
                "indexer_k_cache": torch.randn(
                    cache_rows, IDX_DIM, generator=g, device=dev, dtype=torch.float32
                ).to(torch.bfloat16)
            }
        u = torch.randn(1, IDX_DIM, generator=g, device=dev, dtype=torch.float32)
        eps = torch.randn(
            cache_rows, IDX_DIM, generator=g, device=dev, dtype=torch.float32
        )
        return {"indexer_k_cache": (u + sigma * eps).to(torch.bfloat16)}
    g = _gen(seed * 1000 + 501, dev)
    kv = torch.randn(
        cache_rows, KVLORA, generator=g, device=dev, dtype=torch.float32
    ).to(torch.bfloat16)
    pe = torch.randn(cache_rows, ROPE, generator=g, device=dev, dtype=torch.float32).to(
        torch.bfloat16
    )
    return {"kv_cache": kv, "pe_cache": pe}
