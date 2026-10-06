from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
from safetensors import safe_open

from b12x.attention._shared.mla.reference import (
    dense_mla_reference,
    pack_mla_kv_cache_reference,
    sparse_mla_reference,
    unpack_mla_kv_cache_reference,
)
from b12x.attention._shared.mla.api import (
    MLASparseDecodeMetadata,
    MLASparseExtendMetadata,
    clear_mla_caches,
    sparse_mla_decode_forward,
    sparse_mla_extend_forward,
)
from b12x.attention.sparse_mla._scratch import (
    B12XSparseMLAScratchCaps,
    plan_sparse_mla_scratch,
)

from b12x.testing.reference.helpers import require_b12x


MODEL_PATH = Path("/data/models/GLM-5.1-NVFP4")
LAYER0_SHARD = MODEL_PATH / "model-00001-of-00084.safetensors"
_GLM_CACHE_BOUNDARY_CASES = [63, 64, 65, 129]


@dataclass(frozen=True)
class GLMMLAConfig:
    hidden_size: int
    num_heads: int
    q_lora_rank: int
    kv_lora_rank: int
    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    rms_norm_eps: float
    rope_theta: float

    @property
    def sm_scale(self) -> float:
        return (self.qk_nope_head_dim + self.qk_rope_head_dim) ** -0.5


@dataclass(frozen=True)
class GLMMLAWeights:
    q_a_proj: torch.Tensor
    kv_a_proj_with_mqa: torch.Tensor
    q_b_proj: torch.Tensor
    q_a_layernorm: torch.Tensor
    kv_a_layernorm: torch.Tensor
    w_kc: torch.Tensor


def _require_glm_weights() -> None:
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found at {MODEL_PATH}")
    if not LAYER0_SHARD.exists():
        pytest.skip(f"Layer-0 shard not found at {LAYER0_SHARD}")


@functools.lru_cache(maxsize=1)
def _load_glm_config() -> GLMMLAConfig:
    config = json.loads((MODEL_PATH / "config.json").read_text())
    return GLMMLAConfig(
        hidden_size=int(config["hidden_size"]),
        num_heads=int(config["num_attention_heads"]),
        q_lora_rank=int(config["q_lora_rank"]),
        kv_lora_rank=int(config["kv_lora_rank"]),
        qk_nope_head_dim=int(config["qk_nope_head_dim"]),
        qk_rope_head_dim=int(config["qk_rope_head_dim"]),
        v_head_dim=int(config["v_head_dim"]),
        rms_norm_eps=float(config["rms_norm_eps"]),
        rope_theta=float(config["rope_parameters"]["rope_theta"]),
    )


@functools.lru_cache(maxsize=1)
def _load_glm_layer0_cpu() -> GLMMLAWeights:
    cfg = _load_glm_config()
    keys = [
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
    ]
    with safe_open(str(LAYER0_SHARD), framework="pt", device="cpu") as handle:
        tensors = {key: handle.get_tensor(key).contiguous() for key in keys}

    kv_b_proj = tensors["model.layers.0.self_attn.kv_b_proj.weight"]
    w_kc, _ = kv_b_proj.unflatten(
        0,
        (-1, cfg.qk_nope_head_dim + cfg.v_head_dim),
    ).split([cfg.qk_nope_head_dim, cfg.v_head_dim], dim=1)
    w_kc = w_kc.transpose(1, 2).contiguous().transpose(1, 2)

    return GLMMLAWeights(
        q_a_proj=tensors["model.layers.0.self_attn.q_a_proj.weight"],
        kv_a_proj_with_mqa=tensors[
            "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"
        ],
        q_b_proj=tensors["model.layers.0.self_attn.q_b_proj.weight"],
        q_a_layernorm=tensors["model.layers.0.self_attn.q_a_layernorm.weight"],
        kv_a_layernorm=tensors["model.layers.0.self_attn.kv_a_layernorm.weight"],
        w_kc=w_kc,
    )


@functools.lru_cache(maxsize=4)
def _load_glm_layer0_cuda(device_index: int) -> tuple[GLMMLAConfig, GLMMLAWeights]:
    cfg = _load_glm_config()
    device = torch.device("cuda", device_index)
    cpu = _load_glm_layer0_cpu()
    return cfg, GLMMLAWeights(
        q_a_proj=cpu.q_a_proj.to(device=device),
        kv_a_proj_with_mqa=cpu.kv_a_proj_with_mqa.to(device=device),
        q_b_proj=cpu.q_b_proj.to(device=device),
        q_a_layernorm=cpu.q_a_layernorm.to(device=device),
        kv_a_layernorm=cpu.kv_a_layernorm.to(device=device),
        w_kc=cpu.w_kc.to(device=device),
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_f = x.to(torch.float32)
    inv_rms = torch.rsqrt(x_f.square().mean(dim=-1, keepdim=True) + eps)
    return (x_f * inv_rms).to(x.dtype) * weight


def _rope_interleaved(
    x: torch.Tensor, positions: torch.Tensor, theta: float
) -> torch.Tensor:
    half = x.shape[-1] // 2
    inv_freq = 1.0 / (
        theta ** (torch.arange(half, device=x.device, dtype=torch.float32) / half)
    )
    freqs = positions.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0)
    cos = freqs.cos().view(x.shape[0], 1, half)
    sin = freqs.sin().view(x.shape[0], 1, half)

    x_pairs = x.to(torch.float32).reshape(x.shape[0], x.shape[1], half, 2)
    even = x_pairs[..., 0]
    odd = x_pairs[..., 1]
    rotated = torch.empty_like(x_pairs)
    rotated[..., 0] = even * cos - odd * sin
    rotated[..., 1] = even * sin + odd * cos
    return rotated.reshape_as(x).to(x.dtype)


def _make_glm_case(
    *,
    cache_len: int,
    q_len: int,
    seed: int,
    device: torch.device,
) -> tuple[GLMMLAConfig, torch.Tensor, torch.Tensor, torch.Tensor]:
    cfg, weights = _load_glm_layer0_cuda(device.index or 0)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    hidden_states = torch.randn(
        (cache_len, cfg.hidden_size),
        generator=gen,
        dtype=torch.float32,
    ).to(device=device, dtype=torch.bfloat16)
    hidden_states /= 4
    positions = torch.arange(cache_len, device=device, dtype=torch.long)

    q_lora = F.linear(hidden_states, weights.q_a_proj)
    latent = F.linear(hidden_states, weights.kv_a_proj_with_mqa)
    q_norm = _rms_norm(q_lora, weights.q_a_layernorm, cfg.rms_norm_eps)
    k_nope = _rms_norm(
        latent[:, : cfg.kv_lora_rank],
        weights.kv_a_layernorm,
        cfg.rms_norm_eps,
    ).unsqueeze(1)
    k_rope = _rope_interleaved(
        latent[:, cfg.kv_lora_rank :].unsqueeze(1),
        positions,
        cfg.rope_theta,
    )

    q = F.linear(q_norm, weights.q_b_proj).view(
        cache_len,
        cfg.num_heads,
        cfg.qk_nope_head_dim + cfg.qk_rope_head_dim,
    )
    q_nope, q_rope = q.split([cfg.qk_nope_head_dim, cfg.qk_rope_head_dim], dim=-1)
    q_nope_out = torch.bmm(q_nope.transpose(0, 1), weights.w_kc).transpose(0, 1)
    q_rope = _rope_interleaved(q_rope, positions, cfg.rope_theta)
    q_all = torch.cat([q_nope_out[-q_len:], q_rope[-q_len:]], dim=-1).contiguous()
    return cfg, q_all, k_nope, k_rope


def _compare(a: torch.Tensor, b: torch.Tensor) -> tuple[float, float, float]:
    diff = (a - b).to(torch.float32)
    a_f = a.to(torch.float32).reshape(-1)
    b_f = b.to(torch.float32).reshape(-1)
    cos = torch.nn.functional.cosine_similarity(a_f, b_f, dim=0).item()
    return diff.abs().max().item(), torch.sqrt(diff.square().mean()).item(), cos


def _make_sparse_mla_binding(
    *,
    mode: str,
    device: torch.device,
    q_all: torch.Tensor,
    selected_indices: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    nsa_cache_seqlens_int32: torch.Tensor,
    max_total_q: int,
    max_batch: int,
    topk: int,
    cfg: GLMMLAConfig,
    kv_dtype: torch.dtype = torch.uint8,
    use_cuda_graph: bool = False,
):
    plan = plan_sparse_mla_scratch(
        B12XSparseMLAScratchCaps(
            softmax_scale=1.0,
            mode=mode,
            device=device,
            dtype=torch.bfloat16,
            kv_dtype=kv_dtype,
            num_q_heads=int(q_all.shape[1]),
            head_dim=cfg.kv_lora_rank + cfg.qk_rope_head_dim,
            v_head_dim=cfg.kv_lora_rank,
            max_width=topk,
            max_q_rows=max_total_q,
            max_batch=max_batch,
        )
    )
    (spec,) = plan.scratch_specs()
    scratch = torch.empty(spec.shape, dtype=spec.dtype, device=spec.device)
    binding = plan.bind(
        scratch=scratch,
        q=q_all,
        selected_indices=selected_indices,
        cache_seqlens_int32=cache_seqlens_int32,
        nsa_cache_seqlens_int32=nsa_cache_seqlens_int32,
    )
    binding.scratch.use_cuda_graph = bool(use_cuda_graph)
    return binding


def _make_decode_binding(
    *,
    device: torch.device,
    q_all: torch.Tensor,
    metadata: MLASparseDecodeMetadata,
    max_total_q: int,
    max_batch: int,
    topk: int,
    cfg: GLMMLAConfig,
    kv_dtype: torch.dtype = torch.uint8,
    use_cuda_graph: bool = False,
):
    return _make_sparse_mla_binding(
        mode="decode",
        device=device,
        q_all=q_all,
        selected_indices=metadata.page_table_1,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        max_total_q=max_total_q,
        max_batch=max_batch,
        topk=topk,
        cfg=cfg,
        kv_dtype=kv_dtype,
        use_cuda_graph=use_cuda_graph,
    )


def _make_extend_binding(
    *,
    device: torch.device,
    q_all: torch.Tensor,
    metadata: MLASparseExtendMetadata,
    max_total_q: int,
    max_batch: int,
    topk: int,
    cfg: GLMMLAConfig,
    kv_dtype: torch.dtype = torch.uint8,
):
    return _make_sparse_mla_binding(
        mode="extend",
        device=device,
        q_all=q_all,
        selected_indices=metadata.selected_token_offsets,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        nsa_cache_seqlens_int32=metadata.nsa_cache_seqlens_int32,
        max_total_q=max_total_q,
        max_batch=max_batch,
        topk=topk,
        cfg=cfg,
        kv_dtype=kv_dtype,
    )


def _make_sparse_pool_locs(
    *,
    active_tokens: int,
    pool_tokens: int,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    if pool_tokens < active_tokens:
        raise ValueError(
            f"pool_tokens {pool_tokens} must be at least active_tokens {active_tokens}"
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    locs = torch.randperm(pool_tokens, generator=gen)[:active_tokens]
    return locs.to(device=device, dtype=torch.int32)


def _embed_mla_cache_in_pool(
    *,
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    pool_locs: torch.Tensor,
    pool_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_nope_pool = torch.zeros(
        (pool_tokens, *k_nope.shape[1:]),
        dtype=k_nope.dtype,
        device=k_nope.device,
    )
    k_rope_pool = torch.zeros(
        (pool_tokens, *k_rope.shape[1:]),
        dtype=k_rope.dtype,
        device=k_rope.device,
    )
    pool_indices = pool_locs.to(torch.long)
    k_nope_pool[pool_indices] = k_nope
    k_rope_pool[pool_indices] = k_rope
    packed_pool = pack_mla_kv_cache_reference(k_nope_pool, k_rope_pool)
    return k_nope_pool, k_rope_pool, packed_pool


@pytest.mark.parametrize("cache_len", _GLM_CACHE_BOUNDARY_CASES)
def test_glm51_layer0_mla_pack_roundtrip_matches_unquantized_cache(
    cache_len: int,
) -> None:
    device = require_b12x()
    _require_glm_weights()

    cfg, _q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=10_000 + cache_len,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    unpacked = unpack_mla_kv_cache_reference(packed).squeeze(1)
    expected = torch.cat([k_nope.squeeze(1), k_rope.squeeze(1)], dim=-1)
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(unpacked, expected)
    assert max_abs <= 0.08, f"cache_len={cache_len}: max_abs={max_abs:.6f}"
    assert rmse <= 0.004, f"cache_len={cache_len}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cache_len={cache_len}: cos={cos:.6f}"
    assert unpacked.shape == (cache_len, cfg.kv_lora_rank + cfg.qk_rope_head_dim)


@pytest.mark.parametrize("cache_len", _GLM_CACHE_BOUNDARY_CASES)
def test_glm51_layer0_sparse_mla_reference_matches_dense_oracle_for_decode(
    cache_len: int,
) -> None:
    device = require_b12x()
    _require_glm_weights()

    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=20_000 + cache_len,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(cache_len, dtype=torch.int32, device=device).unsqueeze(
        0
    )

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"cache_len={cache_len}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"cache_len={cache_len}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cache_len={cache_len}: cos={cos:.6f}"


@pytest.mark.parametrize("cache_len", _GLM_CACHE_BOUNDARY_CASES)
def test_glm51_layer0_sparse_mla_reference_matches_dense_oracle_for_extend(
    cache_len: int,
) -> None:
    device = require_b12x()
    _require_glm_weights()

    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=5,
        seed=30_000 + cache_len,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(cache_len, dtype=torch.int32, device=device).repeat(
        5, 1
    )

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"cache_len={cache_len}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"cache_len={cache_len}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cache_len={cache_len}: cos={cos:.6f}"


def test_glm51_layer0_sparse_mla_reference_handles_sparse_indices_and_padding() -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = 129
    q_len = 4
    width = 64
    valid_per_row = 37
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=40_129,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(40129)
    rows = []
    for _ in range(q_len):
        valid = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[
            :valid_per_row
        ]
        valid = valid.to(torch.int32)
        padded = torch.full((width,), -1, dtype=torch.int32)
        padded[:valid_per_row] = valid
        rows.append(padded)
    page_table_1 = torch.stack(rows, dim=0).to(device=device)

    actual = sparse_mla_reference(
        q_all=q_all,
        kv_cache=packed,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_glm51_layer0_decode_api_handles_sparse_indices_and_padding() -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = 129
    width = 64
    valid_per_row = 37
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=45_129,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(45129)
    valid = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[
        :valid_per_row
    ].to(torch.int32)
    page_table_1 = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table_1[0, :valid_per_row] = valid.to(device=device)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_glm51_layer0_extend_api_handles_sparse_indices_and_padding() -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = 129
    q_len = 4
    width = 64
    valid_per_row = 37
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=46_129,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(46129)
    rows = []
    for _ in range(q_len):
        valid = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[
            :valid_per_row
        ]
        valid = valid.to(torch.int32)
        padded = torch.full((width,), -1, dtype=torch.int32)
        padded[:valid_per_row] = valid
        rows.append(padded)
    page_table_1 = torch.stack(rows, dim=0).to(device=device)
    cache_seqlens = torch.full((q_len,), cache_len, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, q_len + 1, dtype=torch.int32, device=device)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        nsa_cu_seqlens_q=cu_seqlens,
        nsa_cu_seqlens_k=cu_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        mode="extend",
    )
    binding = _make_extend_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=q_len,
        max_batch=q_len,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_extend_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


@pytest.mark.parametrize("width", [129, 511, 1024, 2048])
def test_glm51_layer0_decode_api_matches_dense_oracle_for_split_widths(
    width: int,
) -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = max(width, 2050)
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=47_000 + width,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"width={width}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"width={width}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"width={width}: cos={cos:.6f}"


@pytest.mark.parametrize("width", [129, 511, 1024, 2048])
def test_glm51_layer0_decode_api_split_handles_sparse_padding(width: int) -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = max(width + 17, 2050)
    valid_per_row = max(37, width - width // 3)
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=48_000 + width,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)

    gen = torch.Generator(device="cpu")
    gen.manual_seed(48_000 + width)
    valid = torch.randperm(cache_len, generator=gen, dtype=torch.int64)[
        :valid_per_row
    ].to(torch.int32)
    page_table_1 = torch.full((1, width), -1, dtype=torch.int32, device=device)
    page_table_1[0, :valid_per_row] = valid.to(device=device)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"width={width}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"width={width}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"width={width}: cos={cos:.6f}"


@pytest.mark.parametrize("width", [129, 2048])
def test_glm51_layer0_decode_api_matches_dense_oracle_for_boundary_widths(
    width: int,
) -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = max(width, 2050)
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=49_000 + width,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"width={width}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"width={width}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"width={width}: cos={cos:.6f}"


def test_glm51_layer0_decode_split_graph_replay_handles_runtime_padding_changes() -> (
    None
):
    device = require_b12x()
    _require_glm_weights()

    cache_len = 2050
    width = 2048
    replay_valid = 1537
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=49_537,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(width, dtype=torch.int32, device=device).unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=width,
        cfg=cfg,
        use_cuda_graph=True,
    )

    clear_mla_caches()
    captured_out = None

    def run() -> None:
        nonlocal captured_out
        captured_out = sparse_mla_decode_forward(
            kv_cache=packed,
            binding=binding,
            sm_scale=cfg.sm_scale,
            v_head_dim=cfg.kv_lora_rank,
        )

    run()
    torch.cuda.synchronize(device)
    run()
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    torch.cuda.synchronize(device)

    page_table_1[0, replay_valid:] = -1
    graph.replay()
    torch.cuda.synchronize(device)
    assert captured_out is not None

    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )

    max_abs, rmse, cos = _compare(captured_out, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_glm51_layer0_decode_api_matches_dense_oracle_for_local_tp_heads() -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=49_901,
        device=device,
    )
    pool_locs = _make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=cache_len * 2,
        seed=49_911,
        device=device,
    )
    k_nope_pool, k_rope_pool, packed = _embed_mla_cache_in_pool(
        k_nope=k_nope,
        k_rope=k_rope,
        pool_locs=pool_locs,
        pool_tokens=cache_len * 2,
    )
    q_local = q_all[:, :local_heads, :].contiguous()
    page_table_1 = pool_locs[:topk].unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    nsa_cache_seqlens = torch.tensor([topk], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_local,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=topk,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope_pool,
        k_rope=k_rope_pool,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


def test_glm51_layer0_decode_api_matches_dense_oracle_for_local_tp_heads_fp8_view_cache() -> (
    None
):
    device = require_b12x()
    _require_glm_weights()

    cache_len = 2050
    topk = 2048
    local_heads = 8
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=49_902,
        device=device,
    )
    pool_locs = _make_sparse_pool_locs(
        active_tokens=cache_len,
        pool_tokens=cache_len * 2,
        seed=49_912,
        device=device,
    )
    k_nope_pool, k_rope_pool, packed_pool = _embed_mla_cache_in_pool(
        k_nope=k_nope,
        k_rope=k_rope,
        pool_locs=pool_locs,
        pool_tokens=cache_len * 2,
    )
    packed = packed_pool.view(torch.float8_e4m3fn)
    q_local = q_all[:, :local_heads, :].contiguous()
    page_table_1 = pool_locs[:topk].unsqueeze(0)
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    nsa_cache_seqlens = torch.tensor([topk], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=nsa_cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_local,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=topk,
        cfg=cfg,
        kv_dtype=torch.float8_e4m3fn,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_local,
        k_nope=k_nope_pool,
        k_rope=k_rope_pool,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"


@pytest.mark.parametrize("cache_len", _GLM_CACHE_BOUNDARY_CASES)
def test_glm51_layer0_decode_api_matches_dense_oracle(cache_len: int) -> None:
    device = require_b12x()
    _require_glm_weights()

    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=1,
        seed=50_000 + cache_len,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(cache_len, dtype=torch.int32, device=device).unsqueeze(
        0
    )
    cache_seqlens = torch.tensor([cache_len], dtype=torch.int32, device=device)
    metadata = MLASparseDecodeMetadata(
        page_table_1=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        max_seq_len_k=cache_len,
    )
    binding = _make_decode_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=1,
        max_batch=1,
        topk=cache_len,
        cfg=cfg,
    )

    actual = sparse_mla_decode_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"cache_len={cache_len}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"cache_len={cache_len}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cache_len={cache_len}: cos={cos:.6f}"


@pytest.mark.parametrize("cache_len", _GLM_CACHE_BOUNDARY_CASES)
def test_glm51_layer0_extend_api_matches_dense_oracle(cache_len: int) -> None:
    device = require_b12x()
    _require_glm_weights()

    q_len = 5
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=60_000 + cache_len,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)
    page_table_1 = torch.arange(cache_len, dtype=torch.int32, device=device).repeat(
        q_len, 1
    )
    cache_seqlens = torch.full((q_len,), cache_len, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, q_len + 1, dtype=torch.int32, device=device)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_1,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=cache_seqlens,
        nsa_cu_seqlens_q=cu_seqlens,
        nsa_cu_seqlens_k=cu_seqlens,
        max_seq_len_q=1,
        max_seq_len_k=cache_len,
        mode="extend",
    )
    binding = _make_extend_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=q_len,
        max_batch=q_len,
        topk=cache_len,
        cfg=cfg,
    )

    actual = sparse_mla_extend_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_1,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"cache_len={cache_len}: max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"cache_len={cache_len}: rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cache_len={cache_len}: cos={cos:.6f}"


def test_glm51_layer0_extend_api_respects_active_token_counts() -> None:
    device = require_b12x()
    _require_glm_weights()

    cache_len = 2050
    width = 257
    q_len = 5
    cfg, q_all, k_nope, k_rope = _make_glm_case(
        cache_len=cache_len,
        q_len=q_len,
        seed=60_257,
        device=device,
    )
    packed = pack_mla_kv_cache_reference(k_nope, k_rope)

    active_counts = torch.tensor(
        [257, 193, 129, 65, 0], dtype=torch.int32, device=device
    )
    page_table_actual = torch.empty((q_len, width), dtype=torch.int32, device=device)
    page_table_expected = torch.full(
        (q_len, width), -1, dtype=torch.int32, device=device
    )
    base = torch.arange(width, dtype=torch.int32, device=device)
    for row in range(q_len):
        row_tokens = (base + row * 37) % cache_len
        page_table_actual[row] = row_tokens
        valid = int(active_counts[row].item())
        if valid > 0:
            page_table_expected[row, :valid] = row_tokens[:valid]

    cache_seqlens = torch.full((1,), cache_len, dtype=torch.int32, device=device)
    cu_seqlens = torch.arange(0, q_len + 1, dtype=torch.int32, device=device)
    metadata = MLASparseExtendMetadata(
        selected_token_offsets=page_table_actual,
        cache_seqlens_int32=cache_seqlens,
        nsa_cache_seqlens_int32=active_counts,
        nsa_cu_seqlens_q=cu_seqlens,
        nsa_cu_seqlens_k=cu_seqlens,
        max_seq_len_q=q_len,
        max_seq_len_k=cache_len,
        mode="extend",
    )
    binding = _make_extend_binding(
        device=device,
        q_all=q_all,
        metadata=metadata,
        max_total_q=q_len,
        max_batch=1,
        topk=width,
        cfg=cfg,
    )

    actual = sparse_mla_extend_forward(
        kv_cache=packed,
        binding=binding,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    expected = dense_mla_reference(
        q_all=q_all,
        k_nope=k_nope,
        k_rope=k_rope,
        page_table_1=page_table_expected,
        sm_scale=cfg.sm_scale,
        v_head_dim=cfg.kv_lora_rank,
    )
    torch.cuda.synchronize(device)

    max_abs, rmse, cos = _compare(actual, expected)
    assert max_abs <= 0.10, f"max_abs={max_abs:.6f}"
    assert rmse <= 0.005, f"rmse={rmse:.6f}"
    assert cos >= 0.9995, f"cos={cos:.6f}"
