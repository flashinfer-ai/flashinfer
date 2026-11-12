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
"""

# Host tests of the experimental MOSCA MegaMoE backend (no SM100 GPU needed):
# config contract, registration and the experimental warning, the generated
# kernel config header, the replicated row split, the weight transform port and
# the host planner (built by the JIT with nvcc; skipped when CUDA is
# unavailable). The catalog and the vendored sources have their own files.

from __future__ import annotations

import warnings

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.experimental.mosca.megamoe import (
    DSV4_PRO_DEP8_TILE_RULE,
    KERNEL_NAME,
    Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig,
    Sm100MoscaMegaKernelBackend,
    split_rows,
)
from flashinfer.experimental.mosca.megamoe import jit as mosca_jit
from flashinfer.experimental.mosca.megamoe.weights import (
    dequantize_fp4,
    per_token_cast_to_fp4,
    preprocess_mega_weights,
)
from flashinfer.moe_ep.core.kernel import registry

DSV4_RULE = DSV4_PRO_DEP8_TILE_RULE


def make_config(**kw) -> Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig:
    kw.setdefault("activation_clamp", 10.0)
    return Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(intermediate_size=3072, top_k=6, **kw)


# ----------------------------------------------------------------------------- config


def test_config_defaults() -> None:
    cfg = make_config()
    assert cfg.kernel_name == KERNEL_NAME == "sm100_fp8_fp4_bf16_mosca"
    assert cfg.small_t and not cfg.sparse_count
    assert cfg.token_layout == "partitioned" and not cfg.gathers_output
    assert cfg.tile_rule is None and cfg.num_sms is None and cfg.knobs is None
    assert cfg.precompile
    assert make_config(token_layout="replicated").gathers_output


@pytest.mark.parametrize(
    "kwargs",
    (
        dict(intermediate_size=3000),
        dict(top_k=0),
        dict(top_k=33),
        dict(activation_clamp=-1.0),
        dict(activation_clamp=float("inf")),
        dict(activation_clamp=None),  # no clamp: the host stub cannot spell +inf
        dict(small_t=False, sparse_count=True),
        dict(token_layout="tp"),
        dict(num_sms=147),
        dict(num_sms=0),
        dict(knobs="auto"),
        dict(precompile=1),
        dict(kernel_name="sm100_fp8_fp4_bf16_deepgemm"),
    ),
)
def test_config_rejects(kwargs) -> None:
    base = dict(intermediate_size=3072, top_k=6, activation_clamp=10.0)
    base.update(kwargs)
    with pytest.raises((ValueError, TypeError)):
        Sm100_Fp8_Fp4_Bf16_Mosca_MegaMoeConfig(**base)


def test_registry_resolves_kernel_name_and_warns_once() -> None:
    assert KERNEL_NAME in registry._available_mega_kernels()
    import flashinfer.api_logging as api_logging

    api_logging._WARNED_EXPERIMENTAL_BACKENDS.discard(("MoEEpMegaLayer", KERNEL_NAME))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        first = registry.create_mega_kernel(make_config())
        registry.create_mega_kernel(make_config())
    assert isinstance(first, Sm100MoscaMegaKernelBackend)
    experimental = [w for w in caught if issubclass(w.category, ExperimentalWarning)]
    assert len(experimental) == 1


def _run_python(code: str) -> None:
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", code], check=True)


def test_core_never_imports_mosca() -> None:
    # Containment (flashinfer/experimental/README.md): core reaches no
    # experimental module at import time; importing the config is the opt-in.
    _run_python(
        "import sys, flashinfer, flashinfer.moe_ep;"
        "assert not [m for m in sys.modules if m.startswith('flashinfer.experimental.mosca')]"
    )


def test_importing_the_package_registers_the_kernel_lightly() -> None:
    _run_python(
        "import sys, flashinfer.experimental.mosca.megamoe as m;"
        "from flashinfer.moe_ep.core.kernel import registry;"
        "assert m.KERNEL_NAME in registry._available_mega_kernels();"
        "assert 'flashinfer.experimental.mosca.megamoe.jit' not in sys.modules;"
        "assert 'flashinfer.experimental.mosca.megamoe.staging' not in sys.modules"
    )


# ----------------------------------------------------------------------------- generated config header


def test_kernel_config_header_carries_every_template_argument_and_feature() -> None:
    params = mosca_jit.KernelParams(
        num_max_tokens_per_rank=1920,
        hidden=7168,
        intermediate_hidden=3072,
        num_experts=384,
        num_topk=6,
        num_ranks=8,
        num_sms=148,
        block_m=16,
        block_n=128,
        block_k=128,
        store_block_m=8,
        sf_block_m=128,
        sf_block_n=128,
        gran_k=32,
        num_ring_tokens=19200,
        num_sf_ring_tokens=307200,
        swizzle_acts_mode=128,
        swizzle_weights_mode=128,
        num_stages=10,
        smem_size=219844,
        num_dispatch_threads=128,
        num_non_epilogue_threads=128,
        num_epilogue_threads=256,
        num_bytes_per_pull=7168,
        activation_clamp=10.0,
        fast_math=True,
        small_t=True,
        sparse_count=False,
        gather=True,
    )
    header = mosca_jit.render_config_header(params)
    assert "#define MOSCA_ACTIVATION_CLAMP 0x1.4000000000000p+3f" in header
    assert "#define DGX_SMALL_T" in header and "#define DGX_BCAST" in header
    assert "DGX_SPARSE_COUNT" not in header
    with pytest.raises(ValueError, match="finite activation clamp"):
        mosca_jit.render_config_header(
            params.__class__(**{**params.__dict__, "activation_clamp": float("inf")})
        )
    # A changed switch is a different module.
    other = params.__class__(**{**params.__dict__, "small_t": False})
    assert other.module_name() != params.module_name()


# ----------------------------------------------------------------------------- replicated row split


@pytest.mark.parametrize("world", (1, 2, 4, 8))
def test_split_rows_is_tensor_split(world) -> None:
    for total in range(0, 70):
        chunks = [len(c) for c in torch.arange(total).tensor_split(world)]
        starts = [sum(chunks[:r]) for r in range(world)]
        assert [split_rows(total, world, r) for r in range(world)] == list(
            zip(starts, chunks, strict=False)
        )


# ----------------------------------------------------------------------------- weights


def test_fp4_quantization_roundtrip() -> None:
    torch.manual_seed(0)
    w = torch.randn(64, 256)
    packed, sf = per_token_cast_to_fp4(w)
    assert packed.dtype == torch.int8 and packed.shape == (64, 128)
    assert sf.shape == (64, 8)
    # Scales are powers of two (UE8M0).
    assert torch.equal(sf, torch.exp2(torch.log2(sf).round()))
    rel = ((dequantize_fp4(packed, sf) - w).norm() / w.norm()).item()
    assert rel < 0.2


def test_weight_transform_layout() -> None:
    from flashinfer.moe_ep import MoEWeightPack

    torch.manual_seed(0)
    e, inter, hidden = 2, 256, 512
    pack = MoEWeightPack(
        w13=torch.randn(e, 2 * inter, hidden), w2=torch.randn(e, hidden, inter)
    )
    (l1, l1_sf), (l2, l2_sf) = preprocess_mega_weights(
        pack, intermediate_size=inter, hidden_size=hidden
    )
    assert l1.shape == (e, 2 * inter, hidden // 2) and l1.dtype == torch.int8
    assert l2.shape == (e, hidden, inter // 2) and l2.dtype == torch.int8
    # Scale factors: int32 words of four UE8M0 bytes, MN-major with a TMA-aligned stride.
    assert l1_sf.shape == (e, 2 * inter, hidden // 128) and l1_sf.stride()[1:] == (
        1,
        2 * inter,
    )
    assert l2_sf.shape == (e, hidden, inter // 128) and l2_sf.stride()[1:] == (
        1,
        hidden,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="deep_gemm transforms need CUDA"
)
def test_weight_transform_matches_deep_gemm_when_installed() -> None:
    pytest.importorskip("deep_gemm")
    from flashinfer.moe_ep import (
        MoEWeightPack,
        preprocess_mega_weights as deepgemm_preprocess,
    )

    g = torch.Generator(device="cuda").manual_seed(0)
    e, inter, hidden = 2, 512, 1024
    w13 = torch.randint(
        -128,
        128,
        (e, 2 * inter, hidden // 2),
        generator=g,
        device="cuda",
        dtype=torch.int16,
    )
    w2 = torch.randint(
        -128,
        128,
        (e, hidden, inter // 2),
        generator=g,
        device="cuda",
        dtype=torch.int16,
    )
    s13 = torch.exp2(
        torch.randint(
            -9, -5, (e, 2 * inter, hidden // 32), generator=g, device="cuda"
        ).float()
    )
    s2 = torch.exp2(
        torch.randint(
            -9, -5, (e, hidden, inter // 32), generator=g, device="cuda"
        ).float()
    )
    pack = MoEWeightPack(
        w13=w13.to(torch.int8), w2=w2.to(torch.int8), w13_scale=s13, w2_scale=s2
    )
    ours = preprocess_mega_weights(pack, intermediate_size=inter, hidden_size=hidden)
    ref = deepgemm_preprocess(pack, intermediate_size=inter, hidden_size=hidden)
    for pa, pb in zip(ours, ref, strict=False):
        for a, b in zip(pa, pb, strict=False):
            assert torch.equal(a, b) and a.stride() == b.stride()


# ----------------------------------------------------------------------------- host planner


@pytest.fixture(scope="module")
def host():
    if not torch.cuda.is_available():
        pytest.skip(
            "the host planner is built by FlashInfer's JIT (needs the CUDA toolkit)"
        )
    return mosca_jit.host_module()


def plan(host, tokens, *, cap=9600, rule="", ranks=8, experts=384, sms=148):
    v = [
        int(x) for x in host.plan(ranks, experts, cap, tokens, 6, 7168, 3072, sms, rule)
    ]
    return dict(
        block_m=v[0],
        store_block_m=v[5],
        num_ring_tokens=v[9],
        num_stages=v[13],
        smem=v[14],
        rule=v[19],
    )


def test_planner_reproduces_the_served_deepgemm_configs(host) -> None:
    # DeepSeek-V4-Pro at EP8 on 148 SMs: the configurations DeepGEMM's heuristic
    # serves (MOSCA provenance: small-T BLOCK_M 16 / 10 stages / 219844 B,
    # prefill BLOCK_M 128 / 7 stages / 227732 B, ring 472320 at capacity 9600).
    small = plan(host, 1)
    assert (small["block_m"], small["num_stages"], small["smem"]) == (16, 10, 219844)
    prefill = plan(host, 512)
    assert (prefill["block_m"], prefill["num_stages"], prefill["smem"]) == (
        128,
        7,
        227732,
    )
    assert small["num_ring_tokens"] == 472320 and not small["rule"]


def test_planner_applies_and_validates_the_tile_rule(host) -> None:
    assert plan(host, 48, rule=DSV4_RULE)["block_m"] == 32
    assert plan(host, 64, rule=DSV4_RULE)["block_m"] == 64
    p = plan(host, 300, rule=DSV4_RULE)
    assert (p["block_m"], p["store_block_m"], p["rule"]) == (96, 16, 1)
    assert plan(host, 100, rule=DSV4_RULE)["rule"] == 0  # no rule matches: heuristic
    for bad in (
        "t>=1:block_m=48",
        "t>=1:block_n=64",
        "t>=1:fp8_combine=1",
        "x>=1:block_m=32",
        "t>=1:block_m",
    ):
        with pytest.raises(Exception, match="tile_rule"):
            plan(host, 48, rule=bad)


def test_layout_appends_the_gathered_region(host) -> None:
    plain = [int(v) for v in host.layout(8, 384, 1920, 6, 7168, 3072, 148, False)]
    gather = [int(v) for v in host.layout(8, 384, 1920, 6, 7168, 3072, 148, True)]
    assert plain[0] == plain[1] and plain[2] == 0
    assert gather[1] == plain[1] and gather[2] % 256 == 0 and gather[2] >= plain[1]
    assert gather[0] == gather[2] + 8 * 1920 * 7168 * 2
    assert gather[3:] == plain[3:]


def test_planner_matches_deep_gemm_for_every_token_count(host) -> None:
    # The ported heuristic and buffer layout against the installed deep_gemm
    # (sgl-deep-gemm 0.2.0 is the upstream revision): block_m for every per-rank
    # token count, symmetric-buffer bytes and the view offsets.
    deep_gemm = pytest.importorskip("deep_gemm")
    ranks, experts, cap, top_k, hidden, inter, sms = 8, 384, 1920, 6, 7168, 3072, 148
    mismatches = []
    for t in range(cap + 1):
        ours = int(host.plan(ranks, experts, cap, t, top_k, hidden, inter, sms, "")[0])
        theirs = int(
            deep_gemm.get_block_m_for_mega_moe(ranks, experts, cap, t, top_k, "fp8xfp4")
        )
        if ours != theirs:
            mismatches.append((t, ours, theirs))
    assert not mismatches, mismatches[:5]
    layout = [
        int(v)
        for v in host.layout(ranks, experts, cap, top_k, hidden, inter, sms, False)
    ]
    nbytes, slicer = deep_gemm._C.get_symm_buffer_size_for_mega_moe(
        ranks, experts, cap, top_k, hidden, inter, "fp8xfp4", "swiglu", 0
    )
    assert int(nbytes) == layout[1]
    buf = torch.empty(int(nbytes), dtype=torch.int8, device="cuda")
    views = [torch.from_dlpack(v) for v in slicer(buf)]
    # x, x_sf, topk_idx, topk_weights, l1_acts, l1_acts_sf, l2_acts, l2_acts_sf
    offsets = [views[i].data_ptr() - buf.data_ptr() for i in (0, 1, 2, 3, 8, 9, 10, 11)]
    assert offsets == layout[5:13]


@pytest.mark.parametrize("rule", (None, DSV4_RULE))
def test_precompile_covers_every_launch_configuration(host, rule) -> None:
    from types import SimpleNamespace

    ws = SimpleNamespace(
        world_size=8,
        num_experts=384,
        num_max_tokens_per_rank=1920,
        capacity=512,
        hidden=7168,
        num_sms=148,
        plans={},
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        backend = Sm100MoscaMegaKernelBackend(make_config(tile_rule=rule))
    configs = backend.launch_configs(ws)
    # One cached plan per per-rank count a forward can stage: 0..capacity, not
    # the aligned 1920 (which would build modules no forward can use).
    assert len(ws.plans) == 513
    assert set(ws.plans.values()) == set(configs)
    assert 1 < len(configs) <= 32  # a handful of modules to build at warmup


def test_transformed_scales_must_pack_expert_groups_back_to_back() -> None:
    # The kernel's scale tensor map folds the experts into one dimension, so a
    # padded or sliced group would make every later expert read another
    # expert's scales; the validator refuses it (upstream: stride(-3) ==
    # stride(-1) * size(-1)).
    from flashinfer.experimental.mosca.megamoe.weights import (
        validate_transformed_mega_weights,
    )
    from flashinfer.moe_ep import MoEWeightPack
    from flashinfer.moe_ep.core.validation.common import MoEEpConfigError

    torch.manual_seed(0)
    e, inter, hidden = 2, 256, 512
    pack = MoEWeightPack(
        w13=torch.randn(e, 2 * inter, hidden), w2=torch.randn(e, hidden, inter)
    )
    (l1, l1_sf), (l2, l2_sf) = preprocess_mega_weights(
        pack, intermediate_size=inter, hidden_size=hidden
    )
    kwargs = dict(intermediate_size=inter, hidden_size=hidden, num_local_experts=e)
    validate_transformed_mega_weights(((l1, l1_sf), (l2, l2_sf)), **kwargs)
    padded = torch.empty_strided(
        l1_sf.shape,
        (l1_sf.stride(0) + 64, l1_sf.stride(1), l1_sf.stride(2)),
        dtype=l1_sf.dtype,
    )
    padded.copy_(l1_sf)
    with pytest.raises(MoEEpConfigError, match="back to back"):
        validate_transformed_mega_weights(((l1, padded), (l2, l2_sf)), **kwargs)


def test_planner_rejects_a_rule_whose_combine_chunk_the_kernel_cannot_split(
    host,
) -> None:
    # hidden 2304 with few stages leaves room for the combine only in two
    # chunks of 144 16-byte elements, which the kernel cannot handle (it needs
    # a multiple of 32). Upstream's override validator refuses this; without
    # the check the rule would fail later, in nvcc.
    with pytest.raises(Exception, match="combine chunk"):
        host.plan(8, 64, 1920, 16, 6, 2304, 1024, 148, "t>=1:num_stages=3")
    assert (
        int(host.plan(8, 64, 1920, 16, 6, 2304, 1024, 148, "t>=1:num_stages=5")[13])
        == 5
    )
