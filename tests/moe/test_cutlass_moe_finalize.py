"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Bit-exactness tests for the non-fused finalize of the CUTLASS fused MoE (top-k unpermute and
weighted reduction, used with ``use_fused_finalize=False`` and with GEMM2 tactics that have no
fused epilogue).

``finalizeMoeRoutingKernelLauncher`` runs ``finalizeMoeRoutingWarpKernel`` when it can and falls
back to the block-per-row kernels (``finalizeMoeRoutingKernel`` /
``finalizeMoeRoutingNoFillingKernel``) otherwise. The warp kernel performs the same per-element
floating-point operations in the same order, so its output must equal the block kernels' output
bit for bit (NaN payloads, infinities, signed zeros and flushed subnormals included) and it must
leave exactly the same output rows untouched. A test-only binding
(``tests/moe/csrc/cutlass_moe_finalize.cu``) compiled with the flags of the fused MoE module runs
both paths on the same inputs.
"""

import dataclasses
import functools
from pathlib import Path

import pytest
import torch

from flashinfer.utils import get_compute_capability

_SUPPORTED_ARCHES = {
    (8, 9),
    (9, 0),
    (10, 0),
    (10, 3),
    (10, 7),
    (11, 0),
    (12, 0),
    (12, 1),
}
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or get_compute_capability(torch.device("cuda")) not in _SUPPORTED_ARCHES,
    reason="CUTLASS fused MoE is not supported on this architecture",
)

_INT_VIEW = {
    torch.float16: torch.int16,
    torch.bfloat16: torch.int16,
    torch.float32: torch.int32,
}
_ELEM_PER_VEC = {torch.float16: 8, torch.bfloat16: 8, torch.float32: 4}


@functools.cache
def _finalize_module():
    """Build the test binding with the exact flags of the fused MoE module for this GPU."""
    from flashinfer.jit import fused_moe as jit_fused_moe

    major, minor = get_compute_capability(torch.device("cuda"))
    arch = major * 10 + minor
    if arch in (120, 121):
        prod_spec = jit_fused_moe.gen_cutlass_fused_moe_sm120_module()
    elif arch == 103:
        prod_spec = jit_fused_moe.gen_cutlass_fused_moe_sm103_module()
    elif arch in (100, 107, 110):
        prod_spec = jit_fused_moe.gen_cutlass_fused_moe_sm100_module()
    elif arch == 90:
        prod_spec = jit_fused_moe.gen_cutlass_fused_moe_sm90_module()
    else:
        prod_spec = jit_fused_moe.gen_cutlass_fused_moe_sm89_module()
    # The kernels header pulls in the TensorRT-LLM common utilities (logging, checks).
    common_sources = [
        src for src in prod_spec.sources if "nv_internal/cpp/common" in src.as_posix()
    ]
    test_source = Path(__file__).parent / "csrc" / "cutlass_moe_finalize.cu"
    spec = dataclasses.replace(
        prod_spec,
        name=f"cutlass_moe_finalize_test_{arch}",
        sources=[test_source, *common_sources],
    )
    return spec.build_and_load()


@dataclasses.dataclass
class _Case:
    num_rows: int
    hidden: int
    top_k: int
    dtype: torch.dtype = torch.bfloat16
    padded_hidden: int | None = None
    num_experts: int = 128  # total over all EP ranks
    ep_size: int = 1
    ep_rank: int = 0
    tp_rank: int = 0
    enable_alltoall: bool = False
    with_scales: bool = True
    with_bias: bool = False
    adversarial: bool = False
    invalid_expert_ids: bool = False
    duplicate_experts: bool = False
    out_of_range_permuted_rows: bool = False
    enable_pdl: bool = False
    seed: int = 0


def _special_values(dtype: torch.dtype) -> torch.Tensor:
    """Raw bit patterns of NaNs (several payloads, both signs), infinities, zeros, subnormals and
    extreme finite values."""
    if dtype == torch.float32:
        bits = [
            0x7FC00000,
            0x7F800001,
            0x7FFFFFFF,
            0xFFC00001,
            0xFF812345,  # NaN
            0x7F800000,
            0xFF800000,  # +-Inf
            0x00000000,
            0x80000000,  # +-0
            0x00000001,
            0x807FFFFF,
            0x00400000,  # subnormals
            0x00800000,
            0x7F7FFFFF,
            0xFF7FFFFF,
            0x3F800000,
            0xBF800000,
        ]
        return torch.tensor(bits, dtype=torch.int64).to(torch.int32).view(dtype)
    if dtype == torch.bfloat16:
        bits = [
            0x7FC0,
            0x7F81,
            0x7FFF,
            0xFFC1,
            0xFF85,  # NaN
            0x7F80,
            0xFF80,  # +-Inf
            0x0000,
            0x8000,  # +-0
            0x0001,
            0x807F,
            0x0040,  # subnormals
            0x0080,
            0x7F7F,
            0xFF7F,
            0x3F80,
            0xBF80,
        ]
    else:
        bits = [
            0x7E00,
            0x7C01,
            0x7FFF,
            0xFE01,
            0xFC85,  # NaN
            0x7C00,
            0xFC00,  # +-Inf
            0x0000,
            0x8000,  # +-0
            0x0001,
            0x83FF,
            0x0200,  # subnormals
            0x0400,
            0x7BFF,
            0xFBFF,
            0x3C00,
            0xBC00,
        ]
    return torch.tensor(bits, dtype=torch.int32).to(torch.int16).view(dtype)


def _random_values(shape, dtype, adversarial, gen) -> torch.Tensor:
    x = torch.randn(shape, generator=gen, dtype=torch.float32).to(dtype)
    if adversarial and x.numel() > 0:
        special = _special_values(dtype)
        # ~3% of the elements take a special value; whole rows are zero, subnormal or huge.
        mask = torch.rand(shape, generator=gen) < 0.03
        idx = torch.randint(0, special.numel(), shape, generator=gen)
        x = torch.where(mask, special[idx], x)
        if len(shape) == 2 and shape[0] > 3:
            x[0] = 0
            x[1] = special[9]  # smallest positive subnormal
            x[2] = torch.finfo(dtype).max / 2  # sums of a few of these overflow
    return x


def _make_inputs(c: _Case, device: torch.device):
    gen = torch.Generator().manual_seed(c.seed)
    padded = c.padded_hidden or c.hidden
    nepn = c.num_experts // c.ep_size
    start = nepn * c.ep_rank
    expanded_rows = c.num_rows * c.top_k

    if c.duplicate_experts:
        experts = torch.randint(0, c.num_experts, (c.num_rows, c.top_k), generator=gen)
    else:
        experts = torch.argsort(
            torch.rand(c.num_rows, c.num_experts, generator=gen), dim=1
        )[:, : c.top_k]
    experts = experts.to(torch.int32)
    if c.invalid_expert_ids and experts.numel() > 0:
        mask = torch.rand(experts.shape, generator=gen) < 0.1
        bad = torch.tensor(
            [-1, c.num_experts, c.num_experts + 7, -12345], dtype=torch.int32
        )
        experts = torch.where(
            mask, bad[torch.randint(0, 4, experts.shape, generator=gen)], experts
        )

    # Routing maps as built by the routing kernels: the slots with a local expert are sorted by
    # expert into permuted rows [0, num_valid); unpermuted index = k * num_rows + token.
    local = (experts >= start) & (experts < start + nepn)
    unpermuted = (
        torch.arange(c.top_k)[None, :] * c.num_rows + torch.arange(c.num_rows)[:, None]
    )
    keys = (experts - start).clamp(0, nepn).to(torch.int64)
    keys = torch.where(local, keys, torch.full_like(keys, nepn))
    order = torch.argsort(
        keys.flatten() * expanded_rows + torch.randperm(expanded_rows, generator=gen)
    )
    # Entries of non-local slots are never read; fill them with garbage to make sure of it.
    u2p = torch.randint(
        -(2**31), 2**31 - 1, (expanded_rows,), generator=gen, dtype=torch.int64
    )
    p2u = torch.randint(
        -(2**31), 2**31 - 1, (expanded_rows,), generator=gen, dtype=torch.int64
    )
    num_valid = int(local.sum())
    flat_unpermuted = unpermuted.flatten()[order][:num_valid]
    u2p[flat_unpermuted] = torch.arange(num_valid)
    p2u[:num_valid] = flat_unpermuted
    counts = (
        torch.bincount(keys[local], minlength=nepn)
        if num_valid
        else torch.zeros(nepn, dtype=torch.int64)
    )
    efto = torch.cat([torch.zeros(1, dtype=torch.int64), counts.cumsum(0)])
    if c.out_of_range_permuted_rows and num_valid:
        # Only meaningful for the zero-filling path, which skips such slots.
        lane = torch.nonzero(local.flatten()).flatten()
        pick = lane[torch.rand(lane.numel(), generator=gen) < 0.1]
        u2p[unpermuted.flatten()[pick]] = torch.where(
            torch.rand(pick.numel(), generator=gen) < 0.5,
            torch.full((pick.numel(),), -1, dtype=torch.int64),
            torch.full((pick.numel(),), expanded_rows, dtype=torch.int64),
        )

    gemm_output = _random_values((expanded_rows, padded), c.dtype, c.adversarial, gen)
    scales = None
    if c.with_scales:
        scales = torch.rand(c.num_rows, c.top_k, generator=gen) * 2 - 0.5
        if c.adversarial and scales.numel() > 0:
            special = _special_values(torch.float32)
            mask = torch.rand(scales.shape, generator=gen) < 0.05
            idx = torch.randint(0, special.numel(), scales.shape, generator=gen)
            scales = torch.where(mask, special[idx], scales)
    bias = None
    if c.with_bias:
        bias = _random_values((nepn, padded), c.dtype, c.adversarial, gen)

    to_dev = lambda t: None if t is None else t.to(device).contiguous()  # noqa: E731
    return dict(
        gemm_output=to_dev(gemm_output),
        bias=to_dev(bias),
        final_scales=to_dev(scales),
        unpermuted_row_to_permuted_row=to_dev(u2p.to(torch.int32)),
        permuted_row_to_unpermuted_row=to_dev(p2u.to(torch.int32)),
        token_selected_experts=to_dev(experts),
        expert_first_token_offset=to_dev(efto),
    )


def _run(c: _Case, block_kernels: bool, inputs, out_init):
    out = out_init.clone()
    _finalize_module().finalize_moe_routing(
        inputs["gemm_output"],
        out,
        inputs["bias"],
        inputs["final_scales"],
        inputs["unpermuted_row_to_permuted_row"],
        inputs["permuted_row_to_unpermuted_row"],
        inputs["token_selected_experts"],
        inputs["expert_first_token_offset"],
        c.num_rows,
        c.padded_hidden or c.hidden,
        c.hidden,
        c.top_k,
        c.num_experts // c.ep_size,
        2 if c.tp_rank else 1,
        c.tp_rank,
        c.ep_size,
        c.ep_rank,
        c.enable_alltoall,
        c.enable_pdl,
        block_kernels,
    )
    return out


def _launched_finalize_kernels(fn):
    """Run fn and return (its result, names of the finalize kernels it launched).

    The kernel names come from CUPTI via the PyTorch profiler, so this records the branch the
    production launcher actually took, independently of canUseFinalizeMoeRoutingWarpKernel.
    Without CUPTI the profiler cannot trace CUDA kernels; fn then runs untraced and the names
    are None.
    """
    cuda = torch.profiler.ProfilerActivity.CUDA
    if cuda not in torch.profiler.supported_activities():
        return fn(), None
    with torch.profiler.profile(activities=[cuda]) as prof:
        result = fn()
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if "finalizeMoeRouting" in e.name]
    return result, names


def _check(c: _Case, expect_warp_kernel: bool = True):
    device = torch.device("cuda")
    inputs = _make_inputs(c, device)
    # Untouched rows (no-filling path) keep this pattern; it must be identical in both outputs.
    gen = torch.Generator(device=device).manual_seed(c.seed + 1)
    ibits = _INT_VIEW[c.dtype]
    info = torch.iinfo(ibits)
    out_init = torch.randint(
        info.min,
        info.max,
        (c.num_rows, c.hidden),
        generator=gen,
        device=device,
        dtype=ibits,
    ).view(c.dtype)

    uses_warp = _finalize_module().uses_warp_kernel(
        inputs["gemm_output"],
        out_init,
        inputs["bias"] if c.tp_rank == 0 else None,
        c.padded_hidden or c.hidden,
        c.hidden,
        c.top_k,
    )
    assert uses_warp == expect_warp_kernel

    new, launched = _launched_finalize_kernels(lambda: _run(c, False, inputs, out_init))
    if c.num_rows > 0 and launched is not None:
        # The launcher must have taken the branch that uses_warp_kernel predicts.
        assert launched, "no finalize kernel was launched"
        is_warp = ["finalizeMoeRoutingWarpKernel" in name for name in launched]
        assert all(w == uses_warp for w in is_warp), (
            f"expected the {'warp' if uses_warp else 'block-per-row'} kernel, launched {launched}"
        )
    if c.num_rows == 0:
        torch.cuda.synchronize()
        return
    ref = _run(c, True, inputs, out_init)
    torch.cuda.synchronize()
    new_bits, ref_bits = new.view(ibits), ref.view(ibits)
    if not torch.equal(new_bits, ref_bits):
        diff = (new_bits != ref_bits).nonzero()
        raise AssertionError(
            f"{diff.shape[0]} elements differ from the block-per-row kernels, first at "
            f"{diff[0].tolist()}: {new[tuple(diff[0])].item()} vs {ref[tuple(diff[0])].item()}"
        )


DTYPES = [torch.bfloat16, torch.float16, torch.float32]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("hidden", [2048, 2880, 4096, 5120, 7168])
@pytest.mark.parametrize("top_k", [1, 2, 4, 6, 8, 10, 16])
@pytest.mark.parametrize(
    "ep_size, enable_alltoall",
    [(1, False), (8, False), (8, True)],
    ids=["tp", "ep", "ep_a2a"],
)
def test_finalize_matches_block_kernels(dtype, hidden, top_k, ep_size, enable_alltoall):
    _check(
        _Case(
            num_rows=257,
            hidden=hidden,
            top_k=top_k,
            dtype=dtype,
            ep_size=ep_size,
            ep_rank=ep_size // 2,
            enable_alltoall=enable_alltoall,
            seed=hidden * 31 + top_k,
        )
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("num_rows", [0, 1, 3, 32, 1000, 16384])
@pytest.mark.parametrize("enable_alltoall", [False, True])
def test_finalize_num_rows(dtype, num_rows, enable_alltoall):
    _check(
        _Case(
            num_rows=num_rows,
            hidden=2048,
            top_k=8,
            dtype=dtype,
            ep_size=8,
            ep_rank=3,
            enable_alltoall=enable_alltoall,
            seed=num_rows,
        )
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "ep_size, enable_alltoall", [(1, False), (4, False), (4, True)]
)
@pytest.mark.parametrize("with_scales", [True, False])
@pytest.mark.parametrize("with_bias, tp_rank", [(False, 0), (True, 0), (True, 1)])
@pytest.mark.parametrize("seed", [0, 1])
def test_finalize_adversarial_values(
    dtype, ep_size, enable_alltoall, with_scales, with_bias, tp_rank, seed
):
    _check(
        _Case(
            num_rows=300,
            hidden=2880,
            top_k=8,
            dtype=dtype,
            num_experts=64,
            ep_size=ep_size,
            ep_rank=ep_size - 1,
            tp_rank=tp_rank,
            enable_alltoall=enable_alltoall,
            with_scales=with_scales,
            with_bias=with_bias,
            adversarial=True,
            invalid_expert_ids=True,
            seed=seed,
        )
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("enable_alltoall", [False, True])
def test_finalize_routing_corner_cases(dtype, enable_alltoall):
    # Duplicate experts within a token, invalid expert ids, top_k up to the warp kernel's limit.
    for top_k in (3, 31, 32):
        _check(
            _Case(
                num_rows=111,
                hidden=1024,
                top_k=top_k,
                dtype=dtype,
                num_experts=48,
                ep_size=2,
                ep_rank=1,
                enable_alltoall=enable_alltoall,
                with_bias=True,
                duplicate_experts=True,
                invalid_expert_ids=True,
                enable_pdl=True,
                seed=top_k,
            )
        )
    if not enable_alltoall:
        # Local slots whose permuted row is out of range are skipped by the zero-filling kernel.
        _check(
            _Case(
                num_rows=200,
                hidden=2048,
                top_k=8,
                dtype=dtype,
                ep_size=2,
                ep_rank=0,
                out_of_range_permuted_rows=True,
                seed=7,
            )
        )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "hidden, padded_hidden",
    [(8, 8), (24, 24), (264, 264), (2880, 3072), (4096, 4224), (5120, 5120)],
)
@pytest.mark.parametrize("enable_alltoall", [False, True])
def test_finalize_padded_and_narrow_rows(dtype, hidden, padded_hidden, enable_alltoall):
    if hidden % _ELEM_PER_VEC[dtype] or padded_hidden % _ELEM_PER_VEC[dtype]:
        pytest.skip("row width is not a whole number of 128-bit vectors")
    _check(
        _Case(
            num_rows=129,
            hidden=hidden,
            padded_hidden=padded_hidden,
            top_k=4,
            dtype=dtype,
            num_experts=32,
            ep_size=2,
            ep_rank=1,
            enable_alltoall=enable_alltoall,
            with_bias=True,
            adversarial=True,
            seed=hidden,
        )
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_finalize_falls_back_to_block_kernels(dtype):
    # More than 32 experts per token and row widths that are not a whole number of 128-bit
    # vectors keep using the block-per-row kernels.
    _check(
        _Case(num_rows=64, hidden=2048, top_k=33, dtype=dtype, num_experts=64), False
    )
    _check(
        _Case(num_rows=64, hidden=2052, top_k=8, dtype=dtype), dtype == torch.float32
    )
    if dtype != torch.float32:
        _check(
            _Case(num_rows=64, hidden=2048, padded_hidden=2052, top_k=8, dtype=dtype),
            False,
        )
