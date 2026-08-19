"""Shared numerical fixtures for public MLA API tests."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pytest
import torch

from benchmarks.mla.reference import (
    MLAReferenceContract,
    mla_paged_attention_reference,
)


@dataclass(frozen=True)
class MLATestCase:
    """One reviewable public-API/backend configuration."""

    case_id: str
    architecture: tuple[int, int]
    backend: str
    q_len: int = 1
    page_size: int = 32
    q_dtype: torch.dtype = torch.bfloat16
    kv_dtype: torch.dtype = torch.bfloat16
    output_dtype: torch.dtype = torch.bfloat16
    kv_layout: str = "combined"
    lse_mode: str = "none"
    scale_mode: str = "default"
    output_scale: str = "none"
    skip_softmax: bool = False
    use_cuda_graph: bool = False
    metadata_form: str = "dense"
    enable_pdl: bool | None = None
    is_var_seq: bool | None = None
    uses_shared_paged_kv_idx: bool = True
    qk_nope_head_dim: int | None = None
    softmax_scale_qk_nope_head_dim: int | None = None

    @property
    def sm_scale(self) -> float:
        logical_nope_width = (
            self.softmax_scale_qk_nope_head_dim
            if self.softmax_scale_qk_nope_head_dim is not None
            else self.qk_nope_head_dim or 512
        )
        return 1.0 / math.sqrt(logical_nope_width + 64)


@dataclass(frozen=True)
class MLATestInputs:
    """Small deterministic inputs in both public metadata representations."""

    q_nope: torch.Tensor
    q_pe: torch.Tensor
    query: torch.Tensor
    kv_cache: torch.Tensor | None
    ckv_cache: torch.Tensor | None
    kpe_cache: torch.Tensor | None
    cum_seq_lens_q: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    qo_indptr: torch.Tensor
    kv_indptr: torch.Tensor
    kv_indices: torch.Tensor
    kv_len_arr: torch.Tensor


def make_mla_inputs(
    case: MLATestCase, device: torch.device | str = "cuda"
) -> MLATestInputs:
    """Create deterministic inputs without changing PyTorch's global RNG."""
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(20260727)
    batch_size = 2
    num_heads = 128
    total_queries = batch_size * case.q_len
    q_nope = torch.randn(
        (total_queries, num_heads, 512),
        device=device,
        dtype=case.q_dtype,
        generator=generator,
    )
    q_pe = torch.randn(
        (total_queries, num_heads, 64),
        device=device,
        dtype=case.q_dtype,
        generator=generator,
    )
    seed_dtype = (
        case.kv_dtype
        if case.kv_dtype in (torch.float16, torch.bfloat16, torch.float32)
        else torch.bfloat16
    )
    ckv = torch.randn(
        (8, case.page_size, 512),
        device=device,
        dtype=seed_dtype,
        generator=generator,
    ).to(case.kv_dtype)
    kpe = torch.randn(
        (8, case.page_size, 64),
        device=device,
        dtype=seed_dtype,
        generator=generator,
    ).to(case.kv_dtype)
    packed_kv = torch.cat((ckv, kpe), dim=-1)
    if case.kv_layout == "combined":
        kv_cache, ckv_cache, kpe_cache = packed_kv, None, None
    elif case.kv_layout == "adjacent-split":
        kv_cache = None
        ckv_cache, kpe_cache = packed_kv[..., :512], packed_kv[..., 512:]
    elif case.kv_layout == "independent-split":
        kv_cache, ckv_cache, kpe_cache = None, ckv, kpe
    else:
        raise ValueError(f"unsupported test KV layout {case.kv_layout!r}")

    cum_seq_lens_q = torch.arange(
        0,
        total_queries + 1,
        case.q_len,
        device=device,
        dtype=torch.int32,
    )
    block_tables = torch.tensor(
        [[0, 1, 0, 0], [2, 3, 0, 0]], device=device, dtype=torch.int32
    )
    seq_lens = torch.tensor(
        [case.page_size + case.page_size // 2, case.page_size + case.page_size // 4],
        device=device,
        dtype=torch.int32,
    )
    kv_indptr = torch.tensor([0, 2, 4], device=device, dtype=torch.int32)
    kv_indices = torch.arange(4, device=device, dtype=torch.int32)
    return MLATestInputs(
        q_nope=q_nope,
        q_pe=q_pe,
        query=torch.cat((q_nope, q_pe), dim=-1).reshape(
            batch_size, case.q_len, num_heads, 576
        ),
        kv_cache=kv_cache,
        ckv_cache=ckv_cache,
        kpe_cache=kpe_cache,
        cum_seq_lens_q=cum_seq_lens_q,
        block_tables=block_tables,
        seq_lens=seq_lens,
        qo_indptr=cum_seq_lens_q,
        kv_indptr=kv_indptr,
        kv_indices=kv_indices,
        kv_len_arr=seq_lens,
    )


def wrapper_plan_kwargs(case: MLATestCase, inputs: MLATestInputs) -> dict:
    return {
        "cum_seq_lens_q": inputs.cum_seq_lens_q,
        "block_tables": inputs.block_tables,
        "seq_lens": inputs.seq_lens,
        "max_q_len": case.q_len,
        "num_heads": 128,
        "head_dim_ckv": 512,
        "head_dim_kpe": 64,
        "page_size": case.page_size,
        "causal": False,
        "sm_scale": case.sm_scale,
        "q_data_type": case.q_dtype,
        "kv_data_type": case.kv_dtype,
        "qk_nope_head_dim": case.qk_nope_head_dim,
        "lse_mode": case.lse_mode,
        "query_layout": "packed",
        "kv_cache_layout": (
            "packed" if case.kv_layout != "independent-split" else "split"
        ),
        "output_dtype": case.output_dtype,
        "output_scale": case.output_scale,
        "scale_mode": case.scale_mode,
        "skip_softmax": case.skip_softmax,
        "enable_pdl": case.enable_pdl,
        "is_var_seq": case.is_var_seq,
    }


def wrapper_run_kwargs(case: MLATestCase, inputs: MLATestInputs) -> dict:
    kwargs = {
        "kv_cache": inputs.kv_cache,
        "ckv_cache": inputs.ckv_cache,
        "kpe_cache": inputs.kpe_cache,
        "return_lse": case.lse_mode != "none",
        "return_lse_base_on_e": case.lse_mode == "basee",
    }
    if case.output_scale == "per-tensor":
        kwargs["o_scale"] = 0.5
        kwargs["out"] = torch.empty_like(inputs.q_nope, dtype=case.output_dtype)
    if case.scale_mode == "kv-per-tensor":
        kwargs.update(ckv_scale=0.5, kpe_scale=1.0)
    elif case.scale_mode == "bmm-scalar":
        kwargs.update(bmm1_scale=case.sm_scale, bmm2_scale=1.0)
    elif case.scale_mode == "bmm-tensor":
        kwargs.update(
            bmm1_scale=torch.tensor(case.sm_scale, device=inputs.q_nope.device),
            bmm2_scale=torch.tensor(1.0, device=inputs.q_nope.device),
        )
    if case.skip_softmax:
        kwargs["skip_softmax_threshold_scale_factor"] = 1.0
    return kwargs


def functional_kwargs(case: MLATestCase, inputs: MLATestInputs) -> dict:
    packed_kv = (
        inputs.kv_cache
        if inputs.kv_cache is not None
        else torch.cat((inputs.ckv_cache, inputs.kpe_cache), dim=-1)
    )
    kwargs = {
        "query": inputs.query,
        "kv_cache": packed_kv,
        "workspace_buffer": torch.empty(
            128 * 1024 * 1024,
            device=inputs.query.device,
            dtype=torch.uint8,
        ),
        "qk_nope_head_dim": case.qk_nope_head_dim or 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "block_tables": inputs.block_tables,
        "seq_lens": inputs.seq_lens,
        "max_seq_len": int(inputs.seq_lens.max().item()),
        "bmm1_scale": case.sm_scale,
        "bmm2_scale": 1.0,
        "backend": case.backend,
        "return_lse": case.lse_mode != "none",
        "enable_pdl": case.enable_pdl,
        "is_var_seq": True if case.is_var_seq is None else case.is_var_seq,
        "uses_shared_paged_kv_idx": case.uses_shared_paged_kv_idx,
        "cum_seq_lens_q": None,
        "max_q_len": None,
    }
    if case.backend == "cutlass":
        kwargs["qk_nope_head_dim"] = 512
    if case.backend.startswith("cute-dsl-"):
        kwargs["cute_dsl_impl"] = case.backend.removeprefix("cute-dsl-")
    if case.scale_mode == "bmm-tensor":
        kwargs["bmm1_scale"] = torch.tensor(
            case.sm_scale, device=inputs.query.device, dtype=torch.float32
        )
        kwargs["bmm2_scale"] = torch.tensor(
            1.0, device=inputs.query.device, dtype=torch.float32
        )
    if case.skip_softmax:
        kwargs["skip_softmax_threshold_scale_factor"] = 1.0
    return kwargs


def reference_result(
    case: MLATestCase, inputs: MLATestInputs, *, causal: bool = False
) -> tuple[torch.Tensor, torch.Tensor | None]:
    contract = MLAReferenceContract(
        lse_mode=case.lse_mode,
        kv_layout=case.kv_layout,
        output_dtype=case.output_dtype,
        output_scale=case.output_scale,
        scale_mode=case.scale_mode,
        skip_softmax=case.skip_softmax,
    )
    run_kwargs = wrapper_run_kwargs(case, inputs)
    reference_kwargs = {
        key: run_kwargs[key]
        for key in ("o_scale", "ckv_scale", "kpe_scale", "bmm1_scale", "bmm2_scale")
        if key in run_kwargs
    }
    return mla_paged_attention_reference(
        q_nope=inputs.q_nope,
        q_pe=inputs.q_pe,
        qo_indptr=inputs.qo_indptr,
        block_tables=inputs.block_tables,
        seq_lens=inputs.seq_lens,
        page_size=case.page_size,
        contract=contract,
        kv_cache=inputs.kv_cache,
        ckv_cache=inputs.ckv_cache,
        kpe_cache=inputs.kpe_cache,
        sm_scale=case.sm_scale,
        causal=causal,
        **reference_kwargs,
    )


def unpack_mla_result(
    result: torch.Tensor | tuple[torch.Tensor, torch.Tensor], return_lse: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    return result if return_lse else (result, None)


def assert_mla_close(
    actual: torch.Tensor, expected: torch.Tensor, *, fp8: bool = False
) -> None:
    if fp8:
        actual, expected = actual.float().contiguous(), expected.float().contiguous()
    torch.testing.assert_close(
        actual,
        expected,
        rtol=0.15 if fp8 else 2e-2,
        atol=0.15 if fp8 else 2e-2,
    )


def require_architecture(architecture: tuple[int, int]) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires a matching CUDA architecture")
    current = torch.cuda.get_device_capability()
    if current != architecture:
        pytest.skip(
            "requires SM%d%d, current device is SM%d%d" % (*architecture, *current)
        )
