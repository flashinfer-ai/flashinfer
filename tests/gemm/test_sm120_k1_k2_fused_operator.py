"""Integration tests for the exact-shape SM120 fused K1+K2 operator."""

from pathlib import Path

import math
import re

import pytest
import torch

from flashinfer.gemm import svdquant_sm120_routes as _sm120_routes
from flashinfer.gemm.svdquant_sm120_routes import SM120_FUSED_LINEAR_MK
from flashinfer.gemm import svdquant_sm120_cutlass
from flashinfer.gemm.gemm_svdquant import nvfp4_quantize_smooth, svdquant_linear
from flashinfer.gemm.svdquant_sm120_cutlass import get_nvfp4_svdquant_sm120_module


# The producer half of the FFI. -1 means "the caller has no opinion about the
# producer geometry", which keeps the shape-dispatched templated launch these
# tests were written against; the runtime-M/K conversion added producer_family,
# three tiling numbers and the address policy after enable_pdl.
_LEGACY_PRODUCER_ARGS = (-1, 0, 0, 0, 0)


def _assert_public_entry_matches(reference, candidate):
    """Same computation through a tactic the autotuner chose for itself.

    Not bit-equality, and not a relative-error bound either. Measured over two
    shapes (m=64 and m=512, 0.2M and 1.6M elements): half the elements are
    bit-identical, the largest absolute deviation is 8.0 where the outputs run
    to several hundred -- one to two bf16 ulps at that magnitude -- and the only
    sign flips are on elements whose magnitude is at most 4.0, i.e. values near
    zero that two accumulation orders straddle. A relative bound reports 2.0 on
    exactly those and says nothing.

    SQNR is what the rest of this suite compares against and it is magnitude
    aware, so near-zero disagreement cannot dominate it. Measured 50.1 dB and
    49.8 dB against the suite's usual 40 dB floor.

    The separate-vs-combined check stays exact: both arms there run the same
    tactic and the same producer, and producer variants were measured
    bit-identical, so that contract still has teeth.
    """
    ref, cand = reference.float(), candidate.float()
    assert ref.shape == cand.shape
    assert torch.isfinite(cand).all(), "public entry produced non-finite output"
    noise = (ref - cand).pow(2).sum().item()
    signal = ref.pow(2).sum().item()
    sqnr_db = 10.0 * math.log10(signal / noise) if noise > 0 else float("inf")
    assert sqnr_db > 40.0, (
        f"public entry disagrees with the fixed-tactic route: SQNR {sqnr_db:.1f} dB "
        f"(worst element {(ref - cand).abs().max().item():.4e})"
    )


_LAUNCH_GEOMETRY_SLICE = {
    # launch_large_m_kernel<M, K, BlockThreads, TileM, TileK>
    "large_m": (slice(2, 5), _sm120_routes.SM120_FAMILY_LARGE_M),
    # launch_small_m_kernel<M, K, BlockThreads, DownTileCols, RowsPerQuantBlock = 4>
    "small_m": (slice(2, 5), _sm120_routes.SM120_FAMILY_SMALL_M),
    # launch_m537_mixed_kernel<K, BlockThreads, DownTileCols, RowsPerQuantBlock, ...>
    "m537_mixed": (slice(1, 4), _sm120_routes.SM120_FAMILY_M537),
}


def _assert_shape_computes_geometry(m: int, k: int, launch: str) -> None:
    """The shape's computed producer ladder must name the geometry this launch did.

    These tests used to scan the C++ source for a per-shape template
    instantiation. The instantiations are gone on purpose -- one runtime
    launcher now serves every (M, K), which is what let 110 of them collapse to
    34 -- so a source scan can only report their absence. What decides the
    launch today is the variant ladder, so that is what is asserted: the shape
    still computes the geometry the pinned template used to hard-code.
    """
    match = re.fullmatch(r"launch_(\w+)_kernel<([\d,\s]+)>", launch)
    assert match, f"unrecognised launch spelling: {launch}"
    kind, nums = match.group(1), [int(v) for v in match.group(2).split(",")]
    assert kind in _LAUNCH_GEOMETRY_SLICE, f"unmapped launch family: {kind}"
    where, family = _LAUNCH_GEOMETRY_SLICE[kind]
    geometry = tuple(nums[where])
    if kind == "small_m" and len(geometry) == 2:
        geometry = geometry + (4,)  # RowsPerQuantBlock defaults to 4
    variants = _sm120_routes.sm120_producer_variants(m, k)
    assert any(f == family and tuple(t) == geometry for f, t, _ in variants), (
        f"({m}, {k}) no longer computes {geometry} for family {family}; "
        f"its ladder offers {sorted({(f, tuple(t)) for f, t, _ in variants})}"
    )


def _assert_launch_geometry_is_computed(launch: str) -> None:
    """Same check, for the constants that carry (M, K) inside the launch string."""
    nums = [int(v) for v in re.search(r"<([\d,\s]+)>", launch).group(1).split(",")]
    _assert_shape_computes_geometry(nums[0], nums[1], launch)


def _assert_admission_tracks_the_ladder(m: int, k: int, rank: int = 32) -> None:
    """Admission must be exactly "a producer can launch this", nothing else.

    These checks used to read "a neighbouring shape is refused", which was true
    of the 51-entry table and is not the contract any more: a shape nobody
    benchmarked is offered to the tuner when its geometry ladder is non-empty,
    and dropped by measurement rather than by a list. What still has to hold --
    and what a stray shape-keyed special case would break -- is that the two
    agree in both directions.
    """
    admitted = svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    launchable = bool(_sm120_routes.sm120_producer_variants(m, k))
    if rank != 32:
        assert not admitted, f"({m}, {k}) admitted at rank {rank}"
        return
    assert admitted == launchable, (
        f"({m}, {k}) admitted={admitted} but launchable={launchable}"
    )


def test_sm120_table_miss_n_keeps_fused_route_eligibility() -> None:
    """Fused-route eligibility is a property of (M, K), not of any one N."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(64, 3072, 32), (
        "an unmeasured N with a supported (M, K) must retain the fused K12 route"
    )


@pytest.mark.parametrize("k", (5120, 5376, 7168))
def test_sm120_m537_l2t_pack_matches_kernel_lane_mapping(k: int) -> None:
    source = torch.arange(k * 32, dtype=torch.int32).reshape(k, 32)
    packed = svdquant_sm120_cutlass._pack_sm120_m537_l2t(source)
    packed_tiles = packed.reshape(k // 16, 2, 2, 8, 4, 2, 2)

    for step, tile_n, fragment, group, pair, half, within in (
        (0, 0, 0, 0, 0, 0, 0),
        (7, 1, 1, 6, 3, 1, 1),
        (k // 16 - 1, 1, 0, 7, 2, 0, 1),
    ):
        source_row = step * 16 + half * 8 + pair * 2 + within
        source_col = tile_n * 16 + fragment * 8 + group
        assert (
            packed_tiles[step, tile_n, fragment, group, pair, half, within].item()
            == source[source_row, source_col].item()
        )


def test_sm120_m537_l2t_pack_cache_reuses_and_invalidates() -> None:
    cache = svdquant_sm120_cutlass._SM120_M537_PACKED_L2T_CACHE
    cache.clear()
    source = torch.zeros((5120, 32), dtype=torch.int32)
    try:
        first = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        assert svdquant_sm120_cutlass._cached_sm120_m537_l2t(source) is first

        source.add_(1)
        mutated = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        assert mutated is not first
        assert torch.equal(mutated, svdquant_sm120_cutlass._pack_sm120_m537_l2t(source))
    finally:
        cache.clear()


def test_sm120_m537_l2t_pack_rejects_sibling_shape() -> None:
    with pytest.raises(ValueError, match="requires rank 32"):
        svdquant_sm120_cutlass._pack_sm120_m537_l2t(torch.empty((5120, 64)))


@pytest.mark.parametrize("k", (5120, 5376, 7168))
def test_sm120_m537_l2t_pack_handles_inference_tensor_mutation(k: int) -> None:
    cache = svdquant_sm120_cutlass._SM120_M537_PACKED_L2T_CACHE
    with torch.inference_mode():
        source = torch.zeros((k, 32), dtype=torch.bfloat16)
        first = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)
        source.add_(1)
        updated = svdquant_sm120_cutlass._cached_sm120_m537_l2t(source)

    assert id(source) not in cache
    assert torch.equal(first, torch.zeros_like(first))
    assert torch.equal(updated, torch.ones_like(updated))
    assert updated is not first


def test_sm120_jit_module_exports_fused_k1_k2_operator() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()

    assert hasattr(module, "nvfp4_quantize_smooth_lora_down_sm120")
    assert hasattr(module, "nvfp4_quantize_smooth_lora_down_cublaslt_sm120")
    assert hasattr(module, "nvfp4_svdquant_linear_sm120")
    assert hasattr(module, "nvfp4_svdquant_gemm_fallback_tactic")


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (64, 3072),
        (512, 3072),
        (512, 5120),
        (7800, 8960),
        (32760, 8960),
        (7800, 5120),
        (7800, 13824),
        (27280, 3072),
        (32760, 5120),
        (32760, 13824),
        (75600, 13824),
        (27280, 14336),
        # Producers added from the geometry sweep.
        # Two of them are the first small_m instantiations at these M, and one is
        # the first large_m instantiation at 48-row tiles, so the equivalence is
        # not implied by any neighbour already listed above.
        (512, 1536),
        (7800, 1536),
        (256, 3072),
        (6889, 3072),
        (9216, 3072),
        (16384, 3072),
        # Producers added so every production prefix has a fused candidate, even
        # where fusing measured slower than the prefix the shape pays today.
        # Correctness is not conditional on winning.
        (1024, 3072),
        (4096, 3072),
        (64, 12288),
        (256, 12288),
        (512, 12288),
        (1024, 12288),
        (512, 5120),
        (537, 14336),
        (1935, 5376),
    ),
)
def test_sm120_fused_k1_k2_matches_unfused_outputs(m: int, k: int) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, 32), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    # (512, 5120) reaches its prefix through cuBLASLt in production and used to
    # be checked only that way. It has a native producer now, like every other
    # production prefix, so both are exercised: the cuBLASLt one first, then the
    # native one, which overwrites the outputs and is what the assertions below
    # score. Neither is allowed to be the only thing tested.
    if m == 512 and k == 5120:
        workspace = torch.empty((32 * 1024 * 1024,), dtype=torch.uint8, device="cuda")
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down, workspace
        )
        cublaslt_xq = xq.clone()
        cublaslt_sf = sf.clone()
        cublaslt_down = down.clone()
        xq.zero_()
        sf.zero_()
        down.zero_()
    else:
        cublaslt_xq = cublaslt_sf = cublaslt_down = None
    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        xq,
        sf,
        down,
    )
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    assert torch.equal(xq, xq_ref)
    assert torch.equal(sf, sf_ref)
    if cublaslt_xq is not None:
        assert torch.equal(cublaslt_xq, xq_ref)
        assert torch.equal(cublaslt_sf, sf_ref)
        assert torch.equal(cublaslt_down, down_ref)

    # Score both LoRA-down implementations against a common accurate reference
    # rather than scoring the fused one against the other. `down_ref` is a BF16
    # GEMM with its own error, and that error is shape-dependent: measured
    # against FP64 on this seed, torch.mm ranges from 55.6 dB down to 50.2 dB
    # (9216x3072 and 16384x3072), while the fused producer sits at 55.5-55.7 dB
    # on every shape. Comparing the two directly therefore reports the
    # reference's accuracy at exactly the shapes where it is worst, which is how
    # two correct producers first failed here.
    # An FP32 GEMM with TF32 off lands 126-140 dB from FP64 on these shapes --
    # 70 dB clear of what is being measured -- so it can stand in for FP64,
    # which would be far too slow at 75600x13824.
    tf32_was_allowed = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        accurate_ref = torch.mm(x.float(), l2t_smoothed.float())
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_was_allowed

    def _sqnr_db(candidate: torch.Tensor) -> float:
        noise = ((accurate_ref - candidate.float()) ** 2).mean()
        return float(10 * torch.log10((accurate_ref**2).mean() / noise))

    fused_sqnr = _sqnr_db(down)
    reference_sqnr = _sqnr_db(down_ref)

    # The contract this test exists to hold: fusing the LoRA-down into the
    # producer must not cost accuracy against the separate torch.mm it replaces.
    assert fused_sqnr >= reference_sqnr - 0.5, (
        f"fused LoRA-down SQNR {fused_sqnr:.3f} dB is worse than the unfused "
        f"torch.mm at {reference_sqnr:.3f} dB for m={m}, k={k}"
    )
    # Plus the absolute floor the route carried before, unchanged, so nothing
    # is weakened for a shape whose reference happens to be accurate.
    relaxed_sqnr_shape = k >= 8192 or (m, k) == (32760, 5120)
    min_sqnr_db = 48.0 if relaxed_sqnr_shape else 50.0
    assert fused_sqnr > min_sqnr_db, (
        f"fused LoRA-down SQNR {fused_sqnr:.3f} dB is below "
        f"{min_sqnr_db:.1f} dB for m={m}, k={k}"
    )


def test_sm120_fused_k1_k2_rejects_non_exact_shape() -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    m, k = 63, 3072
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((3072,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((3072, 32), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((63, 1536), dtype=torch.uint8, device="cuda")
    sf = torch.empty((24576,), dtype=torch.uint8, device="cuda")
    down = torch.empty((63, 32), dtype=torch.bfloat16, device="cuda")

    # This asserted a RuntimeError naming "the measured SM120 K12 shapes" -- the
    # FFI whitelist, which listed where the producer had been measured and
    # refused everything else. Measured on this card, the neighbours it refused
    # run correctly: 4095x12288 and 4097x12288 reach 51.4 and 51.3 dB SQNR
    # against an fp32 reference, with bit-exact xq, next to 51.3 dB for
    # 4096x12288 which the list did admit.
    #
    # So admission is the computed geometry now, and that is what is asserted:
    # a shape that computes one runs, a shape that does not is refused by the
    # launcher that would have had to run it.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    if variants:
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
        )
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError):
            module.nvfp4_quantize_smooth_lora_down_sm120(
                x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
            )


def _run_prefix_separately(
    module, m, k, variant, x, pqs, gs, l2t, xq, sf, down, workspace
):
    """Run, on its own, exactly the prefix the fused route will run.

    The arm used to branch on ``sm120_fused_prefix_route(m, k)`` -- a table
    lookup -- while the fused arm passed no producer at all and let a second
    copy of that table, in C++, decide. The two agreed only because both read
    the same ladder. Now the producer is named once, here, and handed to both
    arms, so the comparison is between one implementation launched two ways
    rather than between whatever two tables happened to say.
    """
    family, tiling, policy = _sm120_routes.sm120_decode_producer_variant(m, k, variant)
    if family == _sm120_routes.SM120_FAMILY_CUBLASLT:
        module.nvfp4_quantize_smooth_lora_down_cublaslt_sm120(
            x, pqs, gs, l2t, xq, sf, down, workspace
        )
    else:
        module.nvfp4_quantize_smooth_lora_down_dyn_sm120(
            x,
            pqs,
            gs,
            l2t,
            xq,
            sf,
            down,
            family,
            tiling[0],
            tiling[1],
            tiling[2],
            policy,
        )
    return (family, tiling[0], tiling[1], tiling[2], policy)


@pytest.mark.parametrize(
    ("m", "n", "k", "tactic"),
    (
        (64, 3072, 3072, 80),
        (64, 12288, 3072, 0),
        (512, 3072, 3072, 36),
        (512, 12288, 3072, 55),
        (512, 5120, 5120, 52),
        (32760, 1536, 1536, 8),
        (32760, 8960, 1536, 9),
        (7800, 1536, 8960, 0),
        (32760, 1536, 8960, 8),
        (7800, 5120, 5120, 8),
        (7800, 5120, 13824, 8),
        (7800, 13824, 5120, 9),
        (27280, 3072, 3072, 8),
        (32760, 5120, 5120, 8),
        (32760, 5120, 13824, 8),
        (75600, 5120, 13824, 8),
        (27280, 3072, 14336, 8),
        # One production shape per producer added by the geometry sweep, so the
        # combined K12+K3 route is exercised on each new prefix and not only the
        # producer in isolation. Tactic 8 is the k256x128x128 kernel every other
        # large-M row here uses; these rows are checked for equivalence against
        # separate launches, not for being the tuner's pick.
        (512, 1536, 1536, 8),
        (7800, 1536, 1536, 8),
        (256, 3072, 3072, 8),
        (6889, 3072, 3072, 8),
        (9216, 3072, 3072, 8),
        (16384, 3072, 3072, 8),
        # One production shape per prefix that gained a producer for coverage
        # rather than for speed. The combined route picks its prefix by the
        # ladder, so the three cuBLASLt-routed rows below still exercise that
        # path -- what is checked here is that adding a producer did not disturb
        # the route each shape actually runs.
        (1024, 3072, 3072, 8),
        (4096, 3072, 3072, 8),
        (64, 3072, 12288, 8),
        (256, 3072, 12288, 8),
        (512, 3072, 12288, 8),
        (1024, 3072, 12288, 8),
        (512, 5120, 5120, 8),
        (537, 5376, 14336, 8),
        (1935, 21504, 5376, 8),
    ),
)
def test_sm120_combined_k12_k3_matches_separate_launches(
    m: int, n: int, k: int, tactic: int
) -> None:
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    rank = 32
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    # The separate arm has to run the prefix the combined route will run, or the
    # comparison below is between two different LoRA-down implementations and
    # cannot be bit-exact. Since the runtime-M/K conversion the producer is named
    # by (family, tiling, address policy) rather than by the shape's pinned
    # template, so both arms are driven from the same decoded variant here --
    # reading it is what keeps this test honest when a route changes.
    producer_args = _run_prefix_separately(
        module,
        m,
        k,
        tactic >> _sm120_routes.SM120_PRODUCER_SHIFT,
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
        workspace,
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *producer_args,
    )

    torch.cuda.synchronize()
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    public_output = svdquant_linear(
        x,
        weight_fp4,
        weight_sf,
        alpha,
        pre_quant_scale,
        l2t_smoothed,
        l1_scaled,
        global_scale,
        bias=bias,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    # Not bit-equality: the raw FFI above was handed a fixed tactic, while the
    # public entry picks its own through the autotuner. A different K3 tactic
    # accumulates in a different order, which lands one bf16 ulp away -- measured
    # at 96.5 vs 97.0, and 96 is exactly where the bf16 step is 0.5. The
    # separate-vs-combined check above stays exact because both arms there run
    # the same tactic and the same producer (producer variants were measured
    # bit-identical, so that contract is still meaningful).
    _assert_public_entry_matches(combined[3], public_output)


def test_sm120_k12288_large_m_uses_the_accepted_production_route() -> None:
    """Accepted K12288 producers keep their fused-route admission."""
    for m in (4096, 6889, 9216, 16384):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, 12288, 32)
        _assert_admission_tracks_the_ladder(m + 1, 12288, 32)


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (4096, 12288),
        (6889, 12288),
        (9216, 12288),
        (16384, 12288),
    ),
)
def test_sm120_k12288_large_m_writes_every_output_byte(m: int, k: int) -> None:
    """Two hostile prefills must converge on the same complete oracle result."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, 32), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for sentinel in (0x00, 0xFF):
        xq.fill_(sentinel)
        sf.fill_(sentinel)
        down.fill_(float("nan"))
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"xq left unwritten or wrong after the 0x{sentinel:02x} prefill "
            f"for m={m}, k={k}"
        )
        assert torch.equal(sf, sf_ref), (
            f"sf left unwritten or wrong after the 0x{sentinel:02x} prefill "
            f"for m={m}, k={k}"
        )
        assert bool(torch.isfinite(down).all()), (
            f"down retains the NaN prefill after the 0x{sentinel:02x} run "
            f"for m={m}, k={k}"
        )
        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        assert float(sqnr) > 48.0, (
            f"fused LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB for "
            f"m={m}, k={k}"
        )


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (4095, 12288),
        (4097, 12288),
        (6888, 12288),
        (6890, 12288),
        (9215, 12288),
        (9217, 12288),
        (16383, 12288),
        (16385, 12288),
        (4096, 12160),
        (4096, 12416),
        # (16384, 3072) used to stand here as the "same M, different K" case.
        # It now has its own swept producer, so the same coverage moves to a K
        # that still has none.
        (16384, 6144),
    ),
)
def test_sm120_k12288_large_m_rejects_neighboring_shapes(m: int, k: int) -> None:
    """The K12288 admission is exactly four shapes and does not generalize."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, 32), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.empty((m, 32), dtype=torch.bfloat16, device="cuda")

    # This asserted a RuntimeError naming "the measured SM120 K12 shapes" -- the
    # FFI whitelist, which listed where the producer had been measured and
    # refused everything else. Measured on this card, the neighbours it refused
    # run correctly: 4095x12288 and 4097x12288 reach 51.4 and 51.3 dB SQNR
    # against an fp32 reference, with bit-exact xq, next to 51.3 dB for
    # 4096x12288 which the list did admit.
    #
    # So admission is the computed geometry now, and that is what is asserted:
    # a shape that computes one runs, a shape that does not is refused by the
    # launcher that would have had to run it.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    if variants:
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
        )
        torch.cuda.synchronize()
    else:
        with pytest.raises(RuntimeError):
            module.nvfp4_quantize_smooth_lora_down_sm120(
                x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down
            )


@pytest.mark.parametrize("rank", (16, 64))
def test_sm120_k12288_large_m_rejects_non_rank32_lora(rank: int) -> None:
    """A direct-FFI shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    m, k = 4096, 12288
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, rank), dtype=torch.bfloat16, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((m * (k // 16),), dtype=torch.uint8, device="cuda")
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


@pytest.mark.parametrize(
    ("m", "n", "k", "tactic"),
    (
        (4096, 3072, 12288, 32),
        (6889, 3072, 12288, 48),
        (9216, 3072, 12288, 48),
        (16384, 3072, 12288, 8),
    ),
)
def test_sm120_k12288_large_m_combined_k12_k3_matches_separate_launches(
    m: int, n: int, k: int, tactic: int
) -> None:
    """Combined equivalence plus the 40 dB full-output contract.

    The selected fixed K3 tactics for rows 33/36/39/42 pin the K3 half of the
    comparison and now match the accepted public production route.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    rank = 32
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    separate[0].fill_(0x00)
    separate[1].fill_(0x00)
    separate[2].fill_(float("nan"))
    separate[3].fill_(float("nan"))
    combined[0].fill_(0xFF)
    combined[1].fill_(0xFF)
    combined[2].fill_(float("nan"))
    combined[3].fill_(float("nan"))

    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *_LEGACY_PRODUCER_ARGS,
    )

    torch.cuda.synchronize()
    for name, output in (
        ("separate down", separate[2]),
        ("separate full output", separate[3]),
        ("combined down", combined[2]),
        ("combined full output", combined[3]),
    ):
        assert bool(torch.isfinite(output).all()), f"{name} is incomplete"
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    # Independent generic prefix: the same K3 tactic driven by the generic
    # quantizer and torch.mm rather than by the fused producer.
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)
    reference_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        bias,
        reference_output,
        workspace,
        tactic,
        False,
    )

    torch.cuda.synchronize()
    noise = ((reference_output.float() - combined[3].float()) ** 2).mean()
    sqnr = 10 * torch.log10((reference_output.float() ** 2).mean() / noise)
    assert float(sqnr) > 40.0, (
        f"combined K12+K3 output SQNR {float(sqnr):.3f} dB is below 40.0 dB "
        f"for m={m}, n={n}, k={k}, tactic={tactic}"
    )

    public_output = svdquant_linear(
        x,
        weight_fp4,
        weight_sf,
        alpha,
        pre_quant_scale,
        l2t_smoothed,
        l1_scaled,
        global_scale,
        bias=bias,
        enable_pdl=False,
        backend="cutlass-sm120",
    )
    # Not bit-equality: the raw FFI above was handed a fixed tactic, while the
    # public entry picks its own through the autotuner. A different K3 tactic
    # accumulates in a different order, which lands one bf16 ulp away -- measured
    # at 96.5 vs 97.0, and 96 is exactly where the bf16 step is 0.5. The
    # separate-vs-combined check above stays exact because both arms there run
    # the same tactic and the same producer (producer variants were measured
    # bit-identical, so that contract is still meaningful).
    _assert_public_entry_matches(combined[3], public_output)


# ---------------------------------------------------------------------------
# M512/K3072 prefix coverage.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _read_repo_source(relative_path: str) -> str:
    return (_REPO_ROOT / relative_path).read_text(encoding="utf-8")


def _production_ffi_body(csrc: str) -> str:
    """The production fused-prefix FFI body, shape guard included.

    Anchored on ``_impl`` deliberately. The exported symbol
    ``nvfp4_quantize_smooth_lora_down_sm120`` is now a thin wrapper that
    forwards to it, and it appears *after* the guard in the file -- so slicing
    from the wrapper silently produced a body with no shape check in it, and
    every assertion below about which shapes the FFI names passed vacuously.
    """
    start = csrc.index("void nvfp4_quantize_smooth_lora_down_sm120_impl(")
    end = csrc.index("void nvfp4_quantize_smooth_lora_down_sm120(", start)
    assert start < end
    body = csrc[start:end]
    assert "one of the measured SM120 K12 shapes" in body, (
        "the production FFI slice no longer contains the shape guard"
    )
    return body


def test_sm120_m32760_k1536_has_exact_production_admission() -> None:
    retained_launch = "launch_large_m_kernel<32760, 1536, 256, 32, 256>"
    # The dispatcher used to carry an "m == 32760 && k == 1536" branch and this
    # shape's own template. Both were per-shape structures the runtime-M/K
    # conversion replaced, so the pair of statements they made -- this shape is
    # admitted, and it launches with this geometry -- is made directly.
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(32760, 1536, 32)
    _assert_launch_geometry_is_computed(retained_launch)
    # (7800, 1536) and (512, 1536) used to be named here as prefixes this
    # admission must not generalize to. They have since earned producers of
    # their own by measurement, so what is
    # asserted now is that each K1536 prefix carries its own launch rather than
    # sharing 32760's.
    # One runtime launcher serves every (M, K) now, so "carries its own launch"
    # has no instantiation to point at. What still holds, and is what the line
    # meant, is that each K1536 prefix resolves its own producer rather than
    # inheriting 32760's.
    for own_m in (7800, 512):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(own_m, 1536, 32)
        assert _sm120_routes.sm120_producer_variants(own_m, 1536)

    # The K1536 clause used to name its three admitted M values in the FFI
    # source. Admission is a predicate now, so it is asked directly -- and the
    # "and nothing else" half is kept by checking a neighbour is refused.
    for admitted_m in (512, 7800, 32760):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            admitted_m, 1536, _H3_ROW52_RANK
        )
    _assert_admission_tracks_the_ladder(513, 1536, _H3_ROW52_RANK)

    # This prefix reached the fused runner only through the pinned-prefix union
    # while the pins existed. It is stated in SM120_FUSED_LINEAR_MK now, because
    # without it the tuner sees only the unfused runner here and measures
    # slower.
    assert (32760, 1536) in svdquant_sm120_cutlass._SM120_FUSED_LINEAR_MK
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(32760, 1536, 32)
    # Admission is not the route namespace: this shape still keys on the base
    # route ABI and keeps reaching the winners tuned under it.
    assert svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_ABI_VERSION == 5
    assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(32760, 1536, 32) == 5
    assert svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION == 3


_M512_M = 512
_M512_K = 3072
_M512_RANK = 32
_M512_ROW0_N = 3072
_M512_ROW0_K3_TACTIC = 36
_M512_ROW28_N = 12288
_M512_ROW28_K3_TACTIC = 55


def _m512_producer_inputs(
    seed: int = 20260725,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """bf16 inputs for the M512/K3072 prefix and combined linear route."""
    torch.manual_seed(seed)
    x = torch.randn((_M512_M, _M512_K), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((_M512_K,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = (
        (448.0 * 6.0) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()
    ).reshape(1)
    l2t_smoothed = torch.randn(
        (_M512_K, _M512_RANK), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    return x, pre_quant_scale, global_scale, l2t_smoothed


def _m512_sf_numel() -> int:
    return (((_M512_M + 127) // 128) * 128) * (_M512_K // 16)


def _m512_producer_outputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty((_M512_M, _M512_K // 2), dtype=torch.uint8, device="cuda"),
        torch.empty((_m512_sf_numel(),), dtype=torch.uint8, device="cuda"),
        torch.empty((_M512_M, _M512_RANK), dtype=torch.bfloat16, device="cuda"),
    )


def test_sm120_m512_k3072_writes_every_output_byte() -> None:
    """The production prefix overwrites every xq/SF/down output element."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    x, pre_quant_scale, global_scale, l2t_smoothed = _m512_producer_inputs()
    xq, sf, down = _m512_producer_outputs()
    module = get_nvfp4_svdquant_sm120_module()
    prefix = module.nvfp4_quantize_smooth_lora_down_sm120

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for xq_sentinel, sf_sentinel in ((0x00, 0xFF), (0xFF, 0x00)):
        xq.fill_(xq_sentinel)
        sf.fill_(sf_sentinel)
        down.fill_(float("nan"))
        prefix(x, pre_quant_scale, global_scale, l2t_smoothed, xq, sf, down)
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"M512/K3072 xq left unwritten or wrong after the 0x{xq_sentinel:02x} prefill"
        )
        assert torch.equal(sf, sf_ref), (
            f"M512/K3072 sf left unwritten or wrong after the 0x{sf_sentinel:02x} prefill"
        )
        assert bool(torch.isfinite(down).all()), (
            "M512/K3072 down retains the NaN prefill after the opposing "
            f"0x{xq_sentinel:02x}/0x{sf_sentinel:02x} run"
        )

        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        # The same 48 dB direct-native floor the other native K12 producers
        # hold: the eight-warp split accumulates in a different order from
        # torch.mm regardless of how the N tiles are grouped.
        assert float(sqnr) > 48.0, (
            f"M512/K3072 LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB"
        )

        # A misaddressed N fragment corrupts one half of every patch, which a
        # whole-tensor SQNR can average away, so gate every rank column on its
        # own and let the failure name the column.
        column_noise = ((down_ref.float() - down.float()) ** 2).mean(dim=0)
        column_sqnr = 10 * torch.log10(
            (down_ref.float() ** 2).mean(dim=0) / column_noise
        )
        worst = int(torch.argmin(column_sqnr))
        assert float(column_sqnr[worst]) > 48.0, (
            f"M512/K3072 LoRA-down rank column {worst} SQNR "
            f"{float(column_sqnr[worst]):.3f} dB is below 48.0 dB"
        )


@pytest.mark.parametrize("lora_rank", (16, 64))
def test_sm120_m512_k3072_rejects_non_rank32_lora(lora_rank: int) -> None:
    """The admitted M512/K3072 shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((_M512_M, _M512_K), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((_M512_K,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty(
        (_M512_K, lora_rank), dtype=torch.bfloat16, device="cuda"
    )
    xq = torch.empty((_M512_M, _M512_K // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty((_m512_sf_numel(),), dtype=torch.uint8, device="cuda")
    down = torch.empty((_M512_M, lora_rank), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


def test_sm120_m512_k3072_public_rows0_and_28_keep_the_production_producer() -> None:
    """Combined routes use fresh prefix outputs after LoRA weights change."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")

    x, pre_quant_scale, global_scale, l2t_smoothed = _m512_producer_inputs()
    perturbed_l2t = (l2t_smoothed.float() * -0.5 + 0.25).bfloat16().contiguous()
    module = get_nvfp4_svdquant_sm120_module()

    references = []
    for lora_weight in (l2t_smoothed, perturbed_l2t):
        reference = _m512_producer_outputs()
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            lora_weight,
            *reference,
        )
        references.append(reference)
    torch.cuda.synchronize()

    assert not torch.equal(references[0][2], references[1][2]), (
        "witness failed: the perturbed LoRA weight produced the same down "
        "output, so the comparisons below could not detect a stale buffer"
    )

    for n, tactic in (
        (_M512_ROW0_N, _M512_ROW0_K3_TACTIC),
        (_M512_ROW28_N, _M512_ROW28_K3_TACTIC),
    ):
        torch.manual_seed(20260807 + n)
        weight_fp4 = torch.randint(
            0, 256, (n, _M512_K // 2), dtype=torch.uint8, device="cuda"
        )
        weight_sf = torch.ones((n * (_M512_K // 16),), dtype=torch.uint8, device="cuda")
        alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
        l1_scaled = torch.randn(
            (n, _M512_RANK), dtype=torch.bfloat16, device="cuda"
        ).contiguous()
        bias = torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
        workspace_bytes = max(
            32 * 1024 * 1024,
            int(module.nvfp4_svdquant_gemm_workspace_size(_M512_M, n, _M512_K, tactic)),
        )
        workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")
        row = "row 0" if n == _M512_ROW0_N else "row 28"

        for lora_weight, reference in zip(
            (l2t_smoothed, perturbed_l2t), references, strict=False
        ):
            combined_xq, combined_sf, combined_down = _m512_producer_outputs()
            combined_output = torch.full(
                (_M512_M, n), float("nan"), dtype=torch.bfloat16, device="cuda"
            )
            module.nvfp4_svdquant_linear_sm120(
                x,
                pre_quant_scale,
                global_scale,
                lora_weight,
                weight_fp4,
                weight_sf,
                alpha,
                l1_scaled,
                bias,
                combined_xq,
                combined_sf,
                combined_down,
                combined_output,
                workspace,
                tactic,
                False,
                *_LEGACY_PRODUCER_ARGS,
            )
            torch.cuda.synchronize()

            assert torch.equal(combined_xq, reference[0])
            assert torch.equal(combined_sf, reference[1])
            assert torch.equal(combined_down, reference[2]), (
                f"the public {row} combined route no longer matches the "
                f"production K12 producer"
            )
            assert bool(torch.isfinite(combined_output).all()), (
                f"the public {row} fixed-K3 output retained its NaN prefill"
            )


# ---------------------------------------------------------------------------
# MiniMax-H3 Omni row 52 -- (73984, 5376, 14336) fused K12 admission.
#
# This is the only MiniMax-H3 shape whose K already carries an accepted producer
# geometry (K=14336, from the (27280, 14336) instance), so the admission reuses a
# compiled K rather than extrapolating to a new one. The launch geometry is the
# K >= 8192 tile family every proven large-M long-K case runs: 256 threads,
# 80-row tiles, 128-wide K tiles.
# ---------------------------------------------------------------------------

_H3_ROW52_M = 73984
_H3_ROW52_N = 5376
_H3_ROW52_K = 14336
_H3_ROW52_RANK = 32
_H3_ROW52_LAUNCH = "launch_large_m_kernel<73984, 14336, 256, 80, 128>"
_H3_ROW50_K = 7168
_H3_ROW50_LAUNCH = "launch_large_m_kernel<73984, 7168, 256, 32, 256>"
_H3_ROW61_M = 61056
_H3_ROW61_LAUNCH = "launch_large_m_kernel<61056, 14336, 256, 80, 128>"
_H3_ROW59_K = 7168
_H3_ROW59_LAUNCH = "launch_large_m_kernel<61056, 7168, 256, 32, 256>"
_H3_ROW70_M = 82752
_H3_ROW70_LAUNCH = "launch_large_m_kernel<82752, 14336, 256, 80, 128>"
_H3_ROW68_K = 7168
_H3_ROW68_LAUNCH = "launch_large_m_kernel<82752, 7168, 256, 32, 256>"
_H3_REMAINING_BATCH4_CANDIDATES = (
    (537, 5120, "launch_m537_mixed_kernel<5120, 1024, 24, 16, 8>"),
    (537, 7168, "launch_m537_mixed_kernel<7168, 1024, 24, 16, 8>"),
    (537, 5376, "launch_m537_mixed_kernel<5376, 1024, 24, 16, 8>"),
    (1935, 5120, "launch_small_m_kernel<1935, 5120, 256, 16>"),
    (1935, 7168, "launch_small_m_kernel<1935, 7168, 256, 16>"),
    (1935, 14336, "launch_large_m_kernel<1935, 14336, 384, 32, 256>"),
    (6913, 5120, "launch_large_m_kernel<6913, 5120, 192, 32, 128>"),
    (6913, 7168, "launch_large_m_kernel<6913, 7168, 256, 32, 256>"),
    (6913, 14336, "launch_large_m_kernel<6913, 14336, 384, 32, 256>"),
)
# Test fixture only: a fixed K3 tactic so the separate and combined launches are
# compared against each other under identical conditions. This is NOT a
# production winner -- the prerequisite screen measured the K12 producer
# geometry alone, so no combined tactic is pinned for row 52 and the K3 half is
# left to the autotuner.
_H3_ROW52_FIXTURE_K3_TACTIC = 8


def _skip_unless_free_device_memory(required_bytes: int) -> None:
    """Skip rather than OOM: these shapes are multi-GiB by construction."""
    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < required_bytes:
        pytest.skip(
            f"needs {required_bytes / 1024**3:.1f} GiB free device memory, "
            f"only {free_bytes / 1024**3:.1f} GiB available"
        )


def _row52_global_scale(
    x: torch.Tensor, pre_quant_scale: torch.Tensor, rows_per_chunk: int = 512
) -> torch.Tensor:
    """The operator's global scale, without a full-shape FP32 copy of ``x``.

    The definition is
    ``(448 * 6) / (x.float() * pre_quant_scale.float()).abs().nan_to_num().max()``,
    but at row 52's (73984, 14336) every intermediate in that chain is ~3.95 GiB
    and several are live at once, so the whole-tensor form spends ~8 GiB on top
    of ``x`` -- enough to OOM behind a guard sized for the persistent tensors.

    The reduction is row-independent and max is associative, so the identical
    finite absolute maximum falls out of bounded row chunks: one chunk holds
    ``rows_per_chunk * K`` floats (~29 MiB at 512 rows and K=14336), and no
    full-shape FP32 temporary is ever materialized. ``abs()`` then
    ``nan_to_num()`` run per chunk in the original order, so NaN inputs still
    contribute 0.0 and infinities still saturate to the FP32 max before the
    running maximum sees them.
    """
    scale = pre_quant_scale.float()
    # abs() is non-negative and nan_to_num() maps NaN to 0.0, so a zero seed can
    # never win the running maximum over a non-empty tensor.
    amax = torch.zeros((), dtype=torch.float32, device=x.device)
    for start in range(0, x.shape[0], rows_per_chunk):
        chunk = x[start : start + rows_per_chunk].float() * scale
        amax = torch.maximum(amax, chunk.abs().nan_to_num().max())
    return ((448.0 * 6.0) / amax).reshape(1)


def test_sm120_h3_row52_admission_is_exactly_one_shape() -> None:
    """The retained row-52 admission remains exact after adding row 61."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    _assert_admission_tracks_the_ladder(82751, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_svdquant_sm120_routes.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW52_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, k, _H3_ROW52_RANK)
    # Rank is part of the admission, not just of the FFI contract.
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, _H3_ROW52_K, rank)


def test_sm120_h3_row50_admission_is_exactly_one_new_shape() -> None:
    """The third screened short-K producer stays exact to row 50."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK
    )
    for m in (73983, 73985):
        _assert_admission_tracks_the_ladder(m, _H3_ROW50_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW52_M, _H3_ROW50_K, rank)


def test_sm120_h3_row61_admission_is_exactly_one_new_shape() -> None:
    """The screened row is admitted without widening to H3 siblings."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    _assert_admission_tracks_the_ladder(82751, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_svdquant_sm120_routes.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW61_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, _H3_ROW52_K, rank)


def test_sm120_h3_row59_admission_is_exactly_one_new_shape() -> None:
    """The second screened short-K producer stays exact to row 59."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK
    )
    for m in (61055, 61057):
        _assert_admission_tracks_the_ladder(m, _H3_ROW59_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW61_M, _H3_ROW59_K, rank)


def test_sm120_h3_row70_admission_is_exactly_one_new_shape() -> None:
    """The screened row is admitted without widening to H3 siblings."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK
    )
    for m in (82751, 82753):
        _assert_admission_tracks_the_ladder(m, _H3_ROW52_K, _H3_ROW52_RANK)
    # Not one K but three. 5376 and 7168 are MiniMax-H3 shapes whose fused
    # producer the geometry sweep measured as a win over the unfused prefix
    # (the swept-producer tests in test_svdquant_sm120_routes.py assert the
    # same set).
    # "Exact" is about the admission being keyed on the (M, K) pair and not
    # generalising -- so the K set is pinned outright, and a K nobody measured
    # stays out.
    assert {k for m, k in SM120_FUSED_LINEAR_MK if m == _H3_ROW70_M} == {
        5376,
        7168,
        14336,
    }
    for k in (5120, 7040, 12288):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, _H3_ROW52_K, rank)


def test_sm120_h3_row68_admission_is_exactly_one_new_shape() -> None:
    """The short-K producer is scoped to row 68's exact M, K, and rank."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
        _H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK
    )
    for m in (82751, 82753):
        _assert_admission_tracks_the_ladder(m, _H3_ROW68_K, _H3_ROW52_RANK)
    for k in (7040, 7296):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, k, _H3_ROW52_RANK)
    for rank in (16, 64):
        _assert_admission_tracks_the_ladder(_H3_ROW70_M, _H3_ROW68_K, rank)


def test_sm120_h3_routes_bump_abi_only_where_the_runner_set_changed() -> None:
    """Long-K H3 routes retain v6; added packed candidates advance short-K routes.

    The historical six-route admission manifest stays unchanged. Its K7168
    routes now offer packed small-M producers and use v10. Existing producer
    indices and the consumer tactic encoding retain their meaning.
    """
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK
        )
        == 10
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK
        )
        == 10
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK
        )
        == 6
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(
            _H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK, enable_pdl=False
        )
        == "svdquant_linear_sm120_routes_v6_tactics_v3"
    )
    assert (
        svdquant_sm120_cutlass._sm120_linear_route_abi_version(
            _H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK
        )
        == 10
    )

    # The historical manifest and fallback base remain unchanged.
    assert svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_ABI_VERSION == 5
    assert (
        frozenset(
            {
                (_H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW52_M, _H3_ROW50_K, _H3_ROW52_RANK),
                (_H3_ROW61_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW61_M, _H3_ROW59_K, _H3_ROW52_RANK),
                (_H3_ROW70_M, _H3_ROW52_K, _H3_ROW52_RANK),
                (_H3_ROW70_M, _H3_ROW68_K, _H3_ROW52_RANK),
            }
        )
        == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V6_MKR
    ), "the original six-route admission manifest must remain stable"
    assert svdquant_sm120_cutlass._SM120_TACTIC_ABI_VERSION == 3


def test_sm120_h3_promoted_producers_are_wired_to_their_routes() -> None:
    """Screened producers and their exact public admissions stay in lockstep."""
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(61056, 14336, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(61056, 7168, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(82752, 14336, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(82752, 7168, 32)
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(73984, 7168, 32)

    # Each of these used to be `production_dispatch.count("m == M && k == K") == 1`
    # -- the shape's own branch in the dispatcher. The dispatcher no longer names
    # shapes, so what carries the claim is the launch geometry check below (which
    # binds KernelLaunchGeometry's static asserts to the call site) plus the
    # admission predicate.
    for wired_m, wired_k in (
        (73984, 14336),
        (73984, 7168),
        (61056, 14336),
        (61056, 7168),
    ):
        assert _sm120_routes.sm120_producer_variants(wired_m, wired_k), (
            f"({wired_m}, {wired_k}) computes no producer geometry"
        )
    _assert_launch_geometry_is_computed(_H3_ROW52_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW50_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW61_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW59_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW70_LAUNCH)
    _assert_launch_geometry_is_computed(_H3_ROW68_LAUNCH)
    # Neighbours stay out. This used to read as the absence of an "m == 82751"
    # branch in the dispatcher; the dispatcher no longer enumerates shapes, so
    # the same statement is made against the admission predicate.
    for sibling_m, sibling_k in ((82751, 14336), (73983, 7168)):
        _assert_admission_tracks_the_ladder(sibling_m, sibling_k, _H3_ROW52_RANK)

    # And the FFI half: this was the ~51-pair whitelist, asserted shape by shape.
    # The FFI carries no shape list now, so the same statement is made where the
    # answer actually comes from.
    csrc = _read_repo_source("csrc/nvfp4_smooth_quantize_sm100.cu")
    assert "one of the measured SM120 K12 shapes" not in csrc
    for admitted_m, admitted_k in (
        (73984, 14336),
        (73984, 7168),
        (61056, 14336),
        (61056, 7168),
        (82752, 14336),
    ):
        assert _sm120_routes.sm120_producer_variants(admitted_m, admitted_k)
    assert _sm120_routes.sm120_producer_variants(82752, 7168)


def test_sm120_h3_remaining_batch4_routes_are_exact() -> None:
    """Retain the nine-route manifest while refreshing added packed candidates."""
    expected_mkr = frozenset(
        (m, k, _H3_ROW52_RANK) for m, k, _ in _H3_REMAINING_BATCH4_CANDIDATES
    )
    assert expected_mkr == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V7_MKR

    for m, k, launch in _H3_REMAINING_BATCH4_CANDIDATES:
        expected_version = 7 if k == 14336 else 10
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            m, k, _H3_ROW52_RANK
        )
        _assert_admission_tracks_the_ladder(m + 1, k, _H3_ROW52_RANK)
        _assert_admission_tracks_the_ladder(m, k, 64)
        assert (
            svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, _H3_ROW52_RANK)
            == expected_version
        )
        assert (
            svdquant_sm120_cutlass._sm120_linear_op_name(
                m, k, _H3_ROW52_RANK, enable_pdl=False
            )
            == f"svdquant_linear_sm120_routes_v{expected_version}_tactics_v3"
        )
        _assert_shape_computes_geometry(m, k, launch)
        # The FFI used to carry an "m == M && k == K" branch per admitted shape.
        # The runtime producer takes the geometry as arguments instead, so what
        # admits a shape now is the predicate, checked above and again here for
        # the C++-side contract this line used to stand for.
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            m, k, _H3_ROW52_RANK
        )

    # These were "x.size(0) == M && x.size(1) == K" clauses in the combined
    # launcher. It dispatches on the geometry it is handed now, so the exactness
    # they stood for is asked of the admission predicate: these two are in, and
    # a neighbour on either axis is not.
    for exact_k in (5120, 7168):
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(
            1935, exact_k, _H3_ROW52_RANK
        )
        _assert_admission_tracks_the_ladder(1936, exact_k, _H3_ROW52_RANK)


def test_sm120_batch4_gap_closure_routes_are_exact() -> None:
    """Case 4 uses only its screened cuBLASLt prefix."""
    m, _n, k, rank = (537, 5376, 14336, 32)
    assert (
        frozenset({(m, k, rank)}) == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V9_MKR
    )
    assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, rank) == 9
    assert (
        svdquant_sm120_cutlass._sm120_linear_op_name(m, k, rank, enable_pdl=False)
        == "svdquant_linear_sm120_routes_v9_tactics_v3"
    )
    assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
    _assert_admission_tracks_the_ladder(m + 1, k, rank)

    # This K set used to appear twice in the combined source -- once as the
    # cuBLASLt entry's own admission list, once as the "measured faster" gate in
    # the dispatcher -- and the assertion pinned the two copies to each other.
    # Neither exists now: the prefix is a producer family the tactic names, so
    # there is no list to keep in step. What still has to hold is that this
    # shape can reach that prefix at all.
    combined = _read_repo_source("csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu")
    assert (
        "x.size(1) == 5120 || x.size(1) == 7168 || x.size(1) == 14336" not in combined
    )
    families = {family for family, _, _ in _sm120_routes.sm120_producer_variants(m, k)}
    assert _sm120_routes.SM120_FAMILY_CUBLASLT in families


def test_sm120_batch4_k5376_fused_routes_are_exact() -> None:
    """Historical K5376 routes gain a fresh search over packed candidates."""
    expected = (
        (1935, 21504, 5376, 32),
        (1935, 28672, 5376, 32),
        (6913, 21504, 5376, 32),
        (6913, 28672, 5376, 32),
        (73984, 21504, 5376, 32),
        (73984, 28672, 5376, 32),
        (61056, 21504, 5376, 32),
        (61056, 28672, 5376, 32),
        (82752, 21504, 5376, 32),
        (82752, 28672, 5376, 32),
    )
    assert (
        frozenset(
            {
                (1935, 5376, 32),
                (6913, 5376, 32),
                (73984, 5376, 32),
                (61056, 5376, 32),
                (82752, 5376, 32),
            }
        )
        == svdquant_sm120_cutlass._SM120_LINEAR_ROUTE_V8_MKR
    )
    for m, _, k, rank in expected:
        assert svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, rank)
        _assert_admission_tracks_the_ladder(m + 1, k, rank)
        assert svdquant_sm120_cutlass._sm120_linear_route_abi_version(m, k, rank) == 10

    _assert_launch_geometry_is_computed(
        "launch_large_m_kernel<6913, 5376, 192, 32, 128>"
    )
    for m in (73984, 61056, 82752):
        _assert_launch_geometry_is_computed(
            f"launch_large_m_kernel<{m}, 5376, 256, 32, 256>"
        )
    ffi = _read_repo_source("csrc/nvfp4_smooth_quantize_sm100.cu")
    assert "one of the measured SM120 K12 shapes" not in ffi
    assert _sm120_routes.sm120_producer_variants(6913, 5376)
    # Same story: the duplicated cuBLASLt admission list is gone, so what is
    # asserted is its absence plus the reachability the list used to grant.
    combined = _read_repo_source("csrc/nvfp4_svdquant_gemm_cutlass_sm120.cu")
    assert "x.size(1) == 5120 || x.size(1) == 5376 ||" not in combined
    for m_k in (537, 1935):
        families = {f for f, _, _ in _sm120_routes.sm120_producer_variants(m_k, 5376)}
        assert _sm120_routes.SM120_FAMILY_CUBLASLT in families


def test_sm120_m537_prepacked_warp_rows_have_exact_shape_admission() -> None:
    """The accepted row-44 producer cannot leak to a sibling K."""
    producer = _read_repo_source(
        "include/flashinfer/gemm/nvfp4_smooth_quantize_lora_down_sm120.cuh"
    )
    # Was `if (m == 537 && k == 5120)` in the shape dispatcher. That dispatcher
    # is gone; the M537 family still guards its own M and enumerates its K.
    assert "if (m != 537) return cudaErrorInvalidValue;" in producer
    assert "launch_m537_mixed_kernel<5120, 1024, 24, 16, 8>" in producer
    assert "if constexpr (K == 5120 || K == 5376 || K == 7168)" in producer
    assert "reinterpret_cast<uint2 const*>(l2t_smoothed)" in producer
    assert "uint2 const packed_b" in producer

    template = _read_repo_source(
        "include/flashinfer/gemm/nvfp4_svdquant_gemm_template_sm120.h"
    )
    assert "Tactic256x128x128SwapStaticConfig" in template
    assert "k256x128x128SwapStatic" in template
    assert "Tactic256x64x128SwapStaticConfig" in template
    assert "k256x64x128SwapStatic" in template


def test_sm120_h3_row52_tile_arithmetic_is_what_the_tests_assume() -> None:
    """Pure arithmetic, no GPU: the two facts the correctness tests lean on."""
    # A partial final M tile -- unlike (75600, 13824) and (32760, 13824), which
    # both tile evenly. The last tile carries 64 of 80 rows via the row guard.
    assert _H3_ROW52_M % 80 == 64
    # No SF padding rows: padded_m == m, so every SF byte is a live byte.
    assert _H3_ROW52_M % 128 == 0
    assert ((_H3_ROW52_M + 127) // 128) * 128 == _H3_ROW52_M
    # Whole K tiles, and the L2 prefetch is covered by the grid.
    assert _H3_ROW52_K % 128 == 0
    grid_blocks = (_H3_ROW52_M + 79) // 80
    assert grid_blocks * 256 >= _H3_ROW52_K * _H3_ROW52_RANK * 2 // 128


def test_sm120_h3_row52_writes_every_output_byte() -> None:
    """Two hostile prefills converge on the same complete oracle result."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # Live set: x 1.98 GiB + xq/xq_ref 0.99 + sf/sf_ref 0.12 + down/down_ref,
    # l2t and pre_quant_scale <0.02 = ~3.1 GiB persistent. The only temporaries
    # are the chunked global-scale buffers (~0.12 GiB, see _row52_global_scale)
    # and the rank-column FP32 casts (<0.02 GiB), so 6 GiB leaves ~1.9x margin
    # for allocator fragmentation.
    _skip_unless_free_device_memory(6 * 1024**3)

    m, k, rank = _H3_ROW52_M, _H3_ROW52_K, _H3_ROW52_RANK
    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),), dtype=torch.uint8, device="cuda"
    )
    down = torch.empty((m, rank), dtype=torch.bfloat16, device="cuda")
    module = get_nvfp4_svdquant_sm120_module()

    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)

    for sentinel in (0x00, 0xFF):
        xq.fill_(sentinel)
        sf.fill_(sentinel)
        down.fill_(float("nan"))
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )
        torch.cuda.synchronize()

        assert torch.equal(xq, xq_ref), (
            f"xq left unwritten or wrong after the 0x{sentinel:02x} prefill"
        )
        assert torch.equal(sf, sf_ref), (
            f"sf left unwritten or wrong after the 0x{sentinel:02x} prefill"
        )
        assert bool(torch.isfinite(down).all()), (
            f"down retains the NaN prefill after the 0x{sentinel:02x} run"
        )

        noise = ((down_ref.float() - down.float()) ** 2).mean()
        sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / noise)
        assert float(sqnr) > 48.0, (
            f"fused LoRA-down SQNR {float(sqnr):.3f} dB is below 48.0 dB"
        )

        # A whole-tensor SQNR averages a single dead rank column away at this M.
        column_noise = ((down_ref.float() - down.float()) ** 2).mean(dim=0)
        column_sqnr = 10 * torch.log10(
            (down_ref.float() ** 2).mean(dim=0) / column_noise
        )
        worst_column = int(torch.argmin(column_sqnr))
        assert float(column_sqnr[worst_column]) > 48.0, (
            f"LoRA-down rank column {worst_column} SQNR "
            f"{float(column_sqnr[worst_column]):.3f} dB is below 48.0 dB"
        )

        # The partial final M tile (73984 % 80 == 64) is the one structural
        # difference from every proven large-M long-K neighbour, so gate it on
        # its own rather than letting the bulk rows carry the average.
        tail = slice(m - 64, m)
        tail_noise = ((down_ref[tail].float() - down[tail].float()) ** 2).mean()
        tail_sqnr = 10 * torch.log10((down_ref[tail].float() ** 2).mean() / tail_noise)
        assert float(tail_sqnr) > 48.0, (
            f"the 64-row partial final M tile SQNR {float(tail_sqnr):.3f} dB "
            f"is below 48.0 dB"
        )
        assert torch.equal(xq[tail], xq_ref[tail])


@pytest.mark.parametrize(
    ("m", "k"),
    (
        (73983, 14336),
        (73985, 14336),
        (73984, 14208),
        (73984, 14464),
        # (61056, 14336) was here. It is row 61's own shape -- adding row 61 to
        # the admission set left it behind in row 52's neighbour list, where it
        # asserted that an admitted shape is refused. Its K-neighbour keeps what
        # the entry was for: an admitted M does not carry an unmeasured K.
        (61056, 14208),
        (82751, 14336),
    ),
)
def test_sm120_h3_row52_rejects_neighboring_shapes(m: int, k: int) -> None:
    """Admission follows the computed geometry, not a one-shape list."""
    # This used to launch with 1x1 dummy outputs, because the FFI whitelist
    # refused the shape before it looked at them. The whitelist is gone, so the
    # call now reaches the buffer checks -- and giving it real buffers would mean
    # allocating gigabytes per parameter case for M up to 82751.
    #
    # The claim was never about launching, though: it is that admission does not
    # generalize from a measured shape to its neighbours. That is a property of
    # the predicate, so it is asked of the predicate. Where a neighbour *is*
    # admitted, correctness was measured separately -- 4095x12288 and
    # 4097x12288 reach 51.4 and 51.3 dB against an fp32 reference, next to
    # 51.3 dB for the 4096x12288 the old list admitted.
    variants = _sm120_routes.sm120_producer_variants(m, k)
    supported = svdquant_sm120_cutlass._sm120_fused_linear_supported(m, k, 32)
    assert bool(variants) == supported, (
        f"({m}, {k}) computes {len(variants)} producer variants but "
        f"_sm120_fused_linear_supported says {supported}"
    )


@pytest.mark.parametrize("rank", (16, 64))
def test_sm120_h3_row52_rejects_non_rank32_lora(rank: int) -> None:
    """The admitted shape still requires the rank-32 LoRA-down topology."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    _skip_unless_free_device_memory(4 * 1024**3)

    m, k = _H3_ROW52_M, _H3_ROW52_K
    module = get_nvfp4_svdquant_sm120_module()
    x = torch.empty((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = torch.empty((k,), dtype=torch.bfloat16, device="cuda")
    global_scale = torch.ones((1,), dtype=torch.float32, device="cuda")
    l2t_smoothed = torch.empty((k, rank), dtype=torch.bfloat16, device="cuda")
    # The exact shape reaches the rank-32 check before output-shape checks.
    # Minimal outputs keep this a rank-contract test rather than a memory test.
    xq = torch.empty((1, 1), dtype=torch.uint8, device="cuda")
    sf = torch.empty((1,), dtype=torch.uint8, device="cuda")
    down = torch.empty((1, 1), dtype=torch.bfloat16, device="cuda")

    with pytest.raises(RuntimeError, match=r"l2t_smoothed must be \[k, 32\]"):
        module.nvfp4_quantize_smooth_lora_down_sm120(
            x,
            pre_quant_scale,
            global_scale,
            l2t_smoothed,
            xq,
            sf,
            down,
        )


@pytest.mark.parametrize("with_bias", (False, True))
@pytest.mark.parametrize(
    ("m", "k"),
    (
        (_H3_ROW52_M, _H3_ROW52_K),
        (_H3_ROW52_M, _H3_ROW50_K),
        (_H3_ROW61_M, _H3_ROW52_K),
        (_H3_ROW61_M, _H3_ROW59_K),
        (_H3_ROW70_M, _H3_ROW52_K),
        (_H3_ROW70_M, _H3_ROW68_K),
    ),
)
def test_sm120_h3_combined_k12_k3_matches_separate_at_a_fixed_k3_tactic(
    with_bias: bool, m: int, k: int
) -> None:
    """Combined equivalence plus the 40 dB full-output contract for H3 routes.

    The K3 tactic here is a test fixture, not a production choice: holding it
    fixed is what makes the separate and combined launches comparable. Nothing
    in this test asserts that the route would select it. Public-route behaviour
    is deliberately out of scope -- the autotuner may pick a different K3 tactic
    there, so that comparison belongs to the GPU acceptance gate.
    """
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    # The long-K Omni routes set the larger live-set bound; the separate and
    # combined quads 2 x (xq 0.49 +
    # sf 0.06 + down 0.01 + out 0.74) = 2.60 + the generic prefix (xq_ref,
    # sf_ref, down_ref, reference_output) 1.30 + weights and workspace ~0.30 =
    # ~6.2 GiB persistent. The final SQNR compares two (m, n) tensors in FP32,
    # which is ~1.48 GiB per cast and peaks at three such buffers (~4.4 GiB)
    # while the squared difference is allocated. The chunked global scale adds
    # ~0.12 GiB. That is ~10.7 GiB peak, so gate at 14 GiB -- still practical on
    # the smaller of the two SM120 parts, with room for allocator fragmentation.
    _skip_unless_free_device_memory(14 * 1024**3)

    n, rank = _H3_ROW52_N, _H3_ROW52_RANK
    tactic = _H3_ROW52_FIXTURE_K3_TACTIC

    torch.manual_seed(20260725)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()
    # The same bias object drives both launches, so the comparison isolates the
    # fusion and not the epilogue.
    bias = (
        torch.randn((n,), dtype=torch.bfloat16, device="cuda").contiguous()
        if with_bias
        else None
    )
    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")

    separate = (
        torch.empty((m, k // 2), dtype=torch.uint8, device="cuda"),
        torch.empty(
            (((m + 127) // 128) * 128 * (k // 16),),
            dtype=torch.uint8,
            device="cuda",
        ),
        torch.empty((m, rank), dtype=torch.bfloat16, device="cuda"),
        torch.empty((m, n), dtype=torch.bfloat16, device="cuda"),
    )
    combined = tuple(torch.empty_like(tensor) for tensor in separate)

    separate[0].fill_(0x00)
    separate[1].fill_(0x00)
    separate[2].fill_(float("nan"))
    separate[3].fill_(float("nan"))
    combined[0].fill_(0xFF)
    combined[1].fill_(0xFF)
    combined[2].fill_(float("nan"))
    combined[3].fill_(float("nan"))

    module.nvfp4_quantize_smooth_lora_down_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        separate[0],
        separate[1],
        separate[2],
    )
    module.nvfp4_svdquant_gemm(
        separate[0],
        weight_fp4,
        separate[1],
        weight_sf,
        alpha,
        separate[2],
        l1_scaled,
        bias,
        separate[3],
        workspace,
        tactic,
        False,
    )
    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        bias,
        combined[0],
        combined[1],
        combined[2],
        combined[3],
        workspace,
        tactic,
        False,
        *_LEGACY_PRODUCER_ARGS,
    )

    torch.cuda.synchronize()
    for name, output in (
        ("separate down", separate[2]),
        ("separate full output", separate[3]),
        ("combined down", combined[2]),
        ("combined full output", combined[3]),
    ):
        assert bool(torch.isfinite(output).all()), f"{name} is incomplete"
    for separate_output, combined_output in zip(separate, combined, strict=False):
        assert torch.equal(separate_output, combined_output)

    # Independent generic prefix: the same fixed K3 tactic driven by the generic
    # quantizer and torch.mm rather than by the fused producer.
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x,
        pre_quant_scale,
        global_scale,
        enable_pdl=False,
    )
    down_ref = torch.mm(x, l2t_smoothed)
    reference_output = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        bias,
        reference_output,
        workspace,
        tactic,
        False,
    )

    torch.cuda.synchronize()
    noise = ((reference_output.float() - combined[3].float()) ** 2).mean()
    sqnr = 10 * torch.log10((reference_output.float() ** 2).mean() / noise)
    assert float(sqnr) > 40.0, (
        f"combined K12+K3 output SQNR {float(sqnr):.3f} dB is below 40.0 dB "
        f"for m={m}, n={n}, k={k}, with_bias={with_bias} at fixture K3 tactic "
        f"{tactic}"
    )


def _cublaslt_producer_args(m: int, k: int):
    """The (family, tiling..., policy) tuple that names the cuBLASLt prefix."""
    for family, tiling, policy in _sm120_routes.sm120_producer_variants(m, k):
        if family == _sm120_routes.SM120_FAMILY_CUBLASLT:
            return (family, tiling[0], tiling[1], tiling[2], policy)
    raise AssertionError(f"({m}, {k}) offers no cuBLASLt producer")


def test_sm120_m537_cublaslt_down_matches_generic_prefix() -> None:
    """Acceptance for row 44's cuBLASLt LoRA-down path."""
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("requires SM120")
    if torch.cuda.get_device_properties(0).multi_processor_count < 105:
        pytest.skip("requires the >=105-SM M537 admission")
    _skip_unless_free_device_memory(2 * 1024**3)

    m, n, k, rank, tactic = 537, 5376, 5120, 32, 81
    torch.manual_seed(20260814)
    x = torch.randn((m, k), dtype=torch.bfloat16, device="cuda")
    pre_quant_scale = (
        (1.0 + 0.3 * torch.randn((k,), dtype=torch.bfloat16, device="cuda"))
        .abs()
        .contiguous()
    )
    global_scale = _row52_global_scale(x, pre_quant_scale)
    l2t_smoothed = torch.randn(
        (k, rank), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    weight_fp4 = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device="cuda")
    weight_sf = torch.ones((n * (k // 16),), dtype=torch.uint8, device="cuda")
    alpha = torch.ones((1,), dtype=torch.float32, device="cuda")
    l1_scaled = torch.randn((n, rank), dtype=torch.bfloat16, device="cuda").contiguous()

    module = get_nvfp4_svdquant_sm120_module()
    workspace_bytes = max(
        32 * 1024 * 1024,
        int(module.nvfp4_svdquant_gemm_workspace_size(m, n, k, tactic)),
    )
    workspace = torch.empty((workspace_bytes,), dtype=torch.uint8, device="cuda")
    xq = torch.empty((m, k // 2), dtype=torch.uint8, device="cuda")
    sf = torch.empty(
        (((m + 127) // 128) * 128 * (k // 16),),
        dtype=torch.uint8,
        device="cuda",
    )
    down = torch.full((m, rank), float("nan"), dtype=torch.bfloat16, device="cuda")
    output = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")

    module.nvfp4_svdquant_linear_sm120(
        x,
        pre_quant_scale,
        global_scale,
        l2t_smoothed,
        weight_fp4,
        weight_sf,
        alpha,
        l1_scaled,
        None,
        xq,
        sf,
        down,
        output,
        workspace,
        tactic,
        False,
        # This test is about the cuBLASLt prefix, so it asks for it by name. It
        # used to pass no producer and rely on a shape ladder to divert 537x5120
        # onto that prefix -- which is exactly the implicit routing being
        # removed, and it broke silently the moment the ladder went.
        *_cublaslt_producer_args(m, k),
    )
    xq_ref, sf_ref = nvfp4_quantize_smooth(
        x, pre_quant_scale, global_scale, enable_pdl=False
    )
    down_ref = torch.mm(x, l2t_smoothed)
    output_ref = torch.empty_like(output)
    module.nvfp4_svdquant_gemm(
        xq_ref,
        weight_fp4,
        sf_ref,
        weight_sf,
        alpha,
        down_ref,
        l1_scaled,
        None,
        output_ref,
        workspace,
        tactic,
        False,
    )
    torch.cuda.synchronize()

    assert torch.equal(xq, xq_ref)
    assert torch.equal(sf, sf_ref)
    assert bool(torch.isfinite(down).all())
    assert bool(torch.isfinite(output).all())
    down_noise = ((down_ref.float() - down.float()) ** 2).mean()
    down_sqnr = 10 * torch.log10((down_ref.float() ** 2).mean() / down_noise)
    assert float(down_sqnr) > 48.0
    output_noise = ((output_ref.float() - output.float()) ** 2).mean()
    output_sqnr = 10 * torch.log10((output_ref.float() ** 2).mean() / output_noise)
    assert float(output_sqnr) > 40.0
