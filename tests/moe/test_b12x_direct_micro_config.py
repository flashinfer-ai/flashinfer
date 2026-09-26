"""CPU-only coverage checks for direct micro FC1 tasks and Q1 writers."""

import pytest
import torch

from flashinfer.cute_dsl import is_cute_dsl_available

pytestmark = pytest.mark.skipif(not is_cute_dsl_available(), reason="CuTe-DSL required")

if is_cute_dsl_available():
    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x.moe_direct_micro_kernel import (
        MoEDirectMicroKernel,
        build_direct_micro_kernel,
    )

    from flashinfer.fused_moe.cute_dsl.blackwell_sm12x import moe_dispatch


def _configured(n, *, k=2560, m=1, topk=10, max_active_ctas=110, **kwargs):
    kernel = MoEDirectMicroKernel(16, (32, 128), 1, **kwargs)
    # Supplying the resident CTA budget avoids querying a CUDA device.
    kernel.configure(m, k, n, topk, 512, max_active_ctas=max_active_ctas)
    return kernel


def _assert_fc1_q1_coverage(kernel, n):
    cfg = kernel._cfg
    assert cfg is not None
    assert cfg.fc1_chunks > 0
    assert cfg.fc1_chunks * cfg.i_chunk == n, cfg
    assert cfg.rows_per_warp_fc1 * 16 == cfg.i_chunk, cfg
    assert cfg.inter_blocks * 16 == cfg.i_chunk, cfg
    # FC1 loops over rows, but Q1 assigns at most one 16-value block per
    # thread. Divisibility alone cannot detect a chunk with missing writers.
    assert cfg.inter_blocks <= kernel.launch_block_dim, cfg


@pytest.mark.parametrize("n", [16, 32, 48, 80, 192, 320, 448, 512, 8336])
@pytest.mark.parametrize(
    "k,topk,activation,max_active_ctas",
    [
        (1024, 1, "silu", 110),
        (2560, 10, "silu", 110),
        (4096, 32, "silu", 110),
        (2560, 10, "gelu_tanh", 110),
        (2560, 10, "swigluoai_uninterleave", 110),
        (2560, 1, "silu", 512),
    ],
)
def test_single_token_fc1_q1_coverage(n, k, topk, activation, max_active_ctas):
    # 8336 = 521 * 16: an unconstrained chunk merger can collapse it to one
    # chunk, which needs 521 Q1 writers in a 512-thread CTA. The topk=1,
    # budget=512 case also exercises this hazard behind a wave-count guard;
    # it describes a possible CTA budget, not the current test host's GPU.
    kernel = _configured(
        n, k=k, topk=topk, activation=activation, max_active_ctas=max_active_ctas
    )
    _assert_fc1_q1_coverage(kernel, n)
    assert 1 <= kernel.grid_x <= max_active_ctas


@pytest.mark.parametrize("k", [1024, 2560, 4096, 6144])
@pytest.mark.parametrize("max_active_ctas", [1, 80, 100, 110, 140, 200, 512])
def test_single_token_fc1_q1_coverage_sweep(k, max_active_ctas):
    # Include odd numbers of FP4 blocks and values above the Q1 CTA limit.
    for n in range(16, 16384 + 1, 16):
        kernel = _configured(n, k=k, max_active_ctas=max_active_ctas)
        _assert_fc1_q1_coverage(kernel, n)
        assert 1 <= kernel.grid_x <= max_active_ctas


@pytest.mark.parametrize(
    "k,n,topk,phase",
    [
        pytest.param(4096, 320, 8, 0, id="long-k-regression"),
        pytest.param(2560, 448, 10, 0, id="multi-wave-regression"),
        pytest.param(2560, 320, 10, 1, id="standalone-fc1"),
        pytest.param(2560, 320, 10, 2, id="standalone-fc2"),
    ],
)
def test_preserve_geometry_outside_fused_merge_band(k, n, topk, phase):
    kernel = _configured(n, k=k, topk=topk, compile_time_phase=phase)
    _assert_fc1_q1_coverage(kernel, n)
    # Larger chunks regressed the first two shapes; independent launches
    # retain their separately tuned geometry as well.
    assert kernel._cfg.rows_per_warp_fc1 == 1


@pytest.mark.parametrize("n", [192, 320, 448, 512])
@pytest.mark.parametrize(
    "k,kwargs,rows_per_warp",
    [
        pytest.param(2560, {"dynamic_down_scale": True}, 1, id="dynamic-scale"),
        pytest.param(2560, {"activation": "relu2"}, 1, id="non-gated"),
        pytest.param(1024, {"w4a16_mode": True}, 2, id="w4a16-k1024"),
        pytest.param(4096, {"w4a16_mode": True}, 4, id="w4a16-k4096"),
        pytest.param(6144, {"w4a16_mode": True}, 1, id="w4a16-k6144"),
        pytest.param(
            4096,
            {"w4a16_mode": True, "scale_format": "e8m0_k32"},
            4,
            id="w4a16-mx",
        ),
        pytest.param(
            2560,
            {"a8_mx_mode": True, "scale_format": "e8m0_k32"},
            2,
            id="a8-mx",
        ),
        pytest.param(
            2560,
            {
                "a8_mx_mode": True,
                "scale_format": "e8m0_k32",
                "compile_time_phase": 1,
            },
            1,
            id="a8-mx-fc1-only",
        ),
    ],
)
def test_excluded_modes_retain_fc1_geometry(n, k, kwargs, rows_per_warp):
    kernel = _configured(n, k=k, **kwargs)
    _assert_fc1_q1_coverage(kernel, n)
    # Dynamic scaling takes an amax across the chunk; its grouping, and the
    # separately tuned W4A16/A8 geometry, must retain their previous values.
    assert kernel._cfg.rows_per_warp_fc1 == rows_per_warp


@pytest.mark.parametrize("m,n,rows_per_warp", [(2, 320, 4), (3, 320, 2), (8, 512, 8)])
def test_multiple_tokens_fc1_geometry(m, n, rows_per_warp):
    kernel = _configured(n, m=m)
    _assert_fc1_q1_coverage(kernel, n)
    assert kernel._cfg.rows_per_warp_fc1 == rows_per_warp


@pytest.mark.parametrize("m", [2, 3])
@pytest.mark.parametrize("k", [256, 2560, 4096])
@pytest.mark.parametrize("max_active_ctas", [80, 110, 512])
def test_few_token_fc1_q1_coverage_sweep(m, k, max_active_ctas):
    # Arbitrary 16-value blocks through the CTA writer limit, and larger
    # whole-N64 extents covered by the new merger. Large odd block counts
    # (e.g. N8336) retain the old helper's separate Q1-writer limitation.
    extents = list(range(16, 8192 + 1, 16)) + list(range(8256, 16384 + 1, 64))
    for n in extents:
        kernel = _configured(n, m=m, k=k, max_active_ctas=max_active_ctas)
        _assert_fc1_q1_coverage(kernel, n)
        assert 1 <= kernel.grid_x <= max_active_ctas


@pytest.mark.parametrize(
    "k,n,topk,budget,kwargs,rows",
    [
        (2560, 320, 10, 110, {}, 4),
        (1024, 320, 8, 80, {}, 4),
        (1024, 320, 8, 79, {}, 2),
        (4096, 320, 8, 110, {}, 2),
        (2560, 448, 10, 110, {}, 2),
        (2560, 320, 10, 110, {"compile_time_phase": 1}, 2),
        (2560, 320, 10, 110, {"compile_time_phase": 2}, 2),
        (2560, 320, 10, 110, {"dynamic_down_scale": True}, 2),
        (2560, 320, 10, 110, {"activation": "relu2"}, 2),
        (2560, 320, 10, 110, {"w4a16_mode": True}, 2),
        (
            2560,
            320,
            10,
            110,
            {"a8_mx_mode": True, "scale_format": "e8m0_k32"},
            2,
        ),
    ],
)
def test_two_token_merge_band(k, n, topk, budget, kwargs, rows):
    kernel = _configured(n, m=2, k=k, topk=topk, max_active_ctas=budget, **kwargs)
    _assert_fc1_q1_coverage(kernel, n)
    assert kernel._cfg.rows_per_warp_fc1 == rows


@pytest.mark.parametrize(
    "k,n,topk,budget,kwargs,rows",
    [
        (2560, 320, 10, 110, {}, 2),
        (2560, 320, 10, 80, {}, 4),
        (256, 320, 2, 110, {}, 4),
        (1024, 320, 8, 110, {}, 4),
        (4096, 320, 10, 110, {}, 4),
        (2560, 256, 10, 110, {}, 4),
        (2560, 384, 10, 110, {}, 3),
        (2560, 512, 10, 110, {}, 4),
        (2560, 320, 10, 110, {"compile_time_phase": 1}, 4),
        (2560, 320, 10, 110, {"compile_time_phase": 2}, 4),
        (2560, 320, 10, 110, {"dynamic_down_scale": True}, 4),
        (2560, 320, 10, 110, {"activation": "relu2"}, 4),
        (2560, 320, 10, 110, {"w4a16_mode": True}, 2),
        (
            2560,
            320,
            10,
            110,
            {"a8_mx_mode": True, "scale_format": "e8m0_k32"},
            4,
        ),
    ],
)
def test_three_token_narrow_chunk_band(k, n, topk, budget, kwargs, rows):
    kernel = _configured(n, m=3, k=k, topk=topk, max_active_ctas=budget, **kwargs)
    _assert_fc1_q1_coverage(kernel, n)
    assert kernel._cfg.rows_per_warp_fc1 == rows


@pytest.mark.parametrize("m,k", [(1, 1024), (1, 2560), (3, 2560), (4, 4096)])
@pytest.mark.parametrize("phase", [0, 1, 2])
def test_scale_mode_preserves_geometry_and_legacy_default(m, k, phase):
    def build(**kwargs):
        return build_direct_micro_kernel(
            512,
            m,
            k,
            320,
            10,
            compile_time_phase=phase,
            max_active_ctas=110,
            **kwargs,
        )

    legacy = build()
    reciprocal = build(input_scales_are_reciprocal=True)
    ordinary = build(input_scales_are_reciprocal=False)
    assert legacy.__cache_key__ == reciprocal.__cache_key__
    assert ordinary.__cache_key__ != reciprocal.__cache_key__
    assert legacy._cfg == reciprocal._cfg == ordinary._cfg
    assert legacy.grid_x == reciprocal.grid_x == ordinary.grid_x
    assert legacy.launch_block_dim == ordinary.launch_block_dim


@pytest.mark.parametrize("mode", ["w4a16_mode", "a8_mx_mode"])
def test_ordinary_scale_mode_rejects_other_quantization_contracts(mode):
    kwargs = {mode: True}
    legacy = MoEDirectMicroKernel(16, (64, 128), 1, **kwargs)
    reciprocal = MoEDirectMicroKernel(
        16, (64, 128), 1, input_scales_are_reciprocal=True, **kwargs
    )
    assert legacy.__cache_key__ == reciprocal.__cache_key__
    with pytest.raises(ValueError, match="only supported for NVFP4"):
        MoEDirectMicroKernel(
            16, (64, 128), 1, input_scales_are_reciprocal=False, **kwargs
        )


@pytest.mark.parametrize("first_mode", [True, False])
@pytest.mark.parametrize("ids_dtype", [torch.int32, torch.int64])
def test_direct_scale_modes_do_not_alias_launch_or_compile_caches(
    monkeypatch, first_mode, ids_dtype
):
    # Run the real getter and builder/configure. Only device queries, codegen,
    # and the device launchability probe are replaced so this stays CPU-only.
    monkeypatch.setattr(moe_dispatch, "_DIRECT_MICRO_LAUNCH_CACHE", {})
    monkeypatch.setattr(moe_dispatch, "_DIRECT_MICRO_KERNEL_CACHE", {})
    monkeypatch.setattr(moe_dispatch, "_refuse_during_capture", lambda _: None)
    original_build = moe_dispatch.build_direct_micro_kernel
    built = []
    compiled = []

    def build(*args, **kwargs):
        kernel = original_build(*args, max_active_ctas=110, **kwargs)
        built.append(kernel.input_scales_are_reciprocal)
        return kernel

    def compile_kernel(kernel, *, topk_ids_dtype):
        compiled.append((kernel.input_scales_are_reciprocal, topk_ids_dtype))
        return object()

    monkeypatch.setattr(moe_dispatch, "build_direct_micro_kernel", build)
    monkeypatch.setattr(moe_dispatch, "compile_direct_micro_kernel", compile_kernel)
    monkeypatch.setattr(
        moe_dispatch, "compiled_direct_micro_accepts_block_dim", lambda *_: True
    )

    def get(**kwargs):
        return moe_dispatch._get_direct_micro_kernel(
            512, 1, 2560, 320, 10, topk_ids_dtype=ids_dtype, **kwargs
        )

    first = get(input_scales_are_reciprocal=first_mode)
    second = get(input_scales_are_reciprocal=not first_mode)
    assert first[0] is not second[0]
    assert get(input_scales_are_reciprocal=first_mode) is first
    assert get(input_scales_are_reciprocal=not first_mode) is second
    legacy = get()
    assert legacy is (first if first_mode else second)
    assert built == [first_mode, not first_mode]
    assert compiled == [(first_mode, ids_dtype), (not first_mode, ids_dtype)]

    # A fresh launch cache must still retrieve the correct executable from
    # the compile cache, even if its two modes are requested in reverse order.
    moe_dispatch._DIRECT_MICRO_LAUNCH_CACHE.clear()
    assert get(input_scales_are_reciprocal=not first_mode)[0] is second[0]
    assert get(input_scales_are_reciprocal=first_mode)[0] is first[0]
    assert len(compiled) == 2
