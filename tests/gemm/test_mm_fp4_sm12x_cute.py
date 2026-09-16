"""SM12x CuTe NVFP4 regressions for graph reuse and cached tactic validation."""

import pytest
import torch
from flashinfer import mm_fp4
from flashinfer.autotuner import AutoTuner
from flashinfer.gemm import gemm_base
from flashinfer.gemm.kernels.sm12x_cute import policy
from flashinfer.gemm.kernels.sm12x_cute.runner import get_runner

from flashinfer.utils import get_compute_capability, version_at_least
from flashinfer.quantization.fp4_quantization import (
    _e2m1_and_ufp8sf_scale_to_float_cpu,
)


@pytest.fixture(scope="module", autouse=True)
def _require_sm12x_cuda13():
    if not torch.cuda.is_available():
        pytest.skip("Requires an SM120 or SM121 GPU")
    cc = get_compute_capability(torch.device("cuda"))
    if cc not in ((12, 0), (12, 1)):
        pytest.skip("Requires an SM120 or SM121 GPU")
    if torch.version.cuda is None or not version_at_least(torch.version.cuda, "13.0"):
        pytest.skip("SM12x CuTe NVFP4 requires CUDA 13 or newer")
    pytest.importorskip("cutlass.cute")
    assert mm_fp4.is_backend_supported("cute-dsl", cc[0] * 10 + cc[1])


def _operands(m, n, k, seed):
    gen = torch.Generator(device="cuda").manual_seed(seed)
    a = torch.randint(256, (m, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    b = torch.randint(256, (n, k // 2), dtype=torch.uint8, device="cuda", generator=gen)
    # Binary-exact E4M3 scales keep the small reference dot exact in FP32.
    scales = torch.tensor([0x30, 0x38, 0x40], dtype=torch.uint8, device="cuda")
    sfa = scales[torch.randint(3, (m, k // 16), device="cuda", generator=gen)]
    sfb = scales[torch.randint(3, (n, k // 16), device="cuda", generator=gen)]
    return a, b.T, sfa, sfb.T


def _reference(a, b, sfa, sfb, alpha):
    one = torch.ones(1, dtype=torch.float32)

    def linear_sf(sf, m, k):
        # Inverse of the standard 128x4 layout, including its 32-row interleave.
        return (
            sf.cpu()
            .reshape(m // 128, k // 64, 32, 4, 4)
            .permute(0, 3, 2, 1, 4)
            .reshape(m, k // 16)
            .contiguous()
        )

    k = a.shape[1] * 2
    da = _e2m1_and_ufp8sf_scale_to_float_cpu(
        a.cpu(), linear_sf(sfa, a.shape[0], k), one, 16, 1, False
    ).double()
    db = _e2m1_and_ufp8sf_scale_to_float_cpu(
        b.T.cpu(), linear_sf(sfb.T, b.shape[1], k), one, 16, 1, False
    ).double()
    # Apply alpha in FP32, matching the public NVFP4 contract.
    return ((da @ db.T).float() * alpha.cpu()).bfloat16().cuda()


def _assert_bits(actual, expected):
    assert torch.isfinite(actual).all()
    assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))


def make_inputs(m, n, k, seed):
    padded_m = (m + 127) // 128 * 128
    padded_n = (n + 127) // 128 * 128
    full = _operands(padded_m, padded_n, k, seed)
    # Logical A/B are compact; scale storage keeps physical 128x4 row padding.
    return full, (full[0][:m].clone(), full[1].T[:n].clone().T, *full[2:])


@pytest.mark.parametrize(
    "m,n,k",
    [(m, 256, 256) for m in (1, 4, 127, 128, 129, 5000, 8192)]
    + [(1, 192, 256), (5, 256, 320), (129, 192, 320)],
)
@pytest.mark.parametrize("pdl", [False, True])
def test_public_and_each_tactic_full_bits_graph_mutation(m, n, k, pdl):
    full, operands = make_inputs(m, n, k, 42)
    alpha = torch.tensor(1.25, dtype=torch.float32, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    expected = _reference(*full, alpha)[:m, :n]
    saved = [x.clone() for x in (*operands, alpha)]
    actual = mm_fp4(*operands, alpha, out=out, backend="cute-dsl", enable_pdl=pdl)
    assert actual.data_ptr() == out.data_ptr()
    _assert_bits(actual, expected)
    allocated = mm_fp4(*operands, alpha, backend="cute-dsl", enable_pdl=pdl)
    _assert_bits(allocated, expected)
    public_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(public_graph):
        mm_fp4(*operands, alpha, out=out, backend="cute-dsl", enable_pdl=pdl)
    for live, before in zip((*operands, alpha), saved, strict=True):
        assert torch.equal(
            live.reshape(-1).view(torch.uint8), before.reshape(-1).view(torch.uint8)
        )

    runner = get_runner()
    inputs = [*operands, alpha, out.dtype, out, 16, True, None]
    for tactic in runner.get_valid_tactics(inputs, None):
        for dst, src in zip((*operands, alpha), saved, strict=True):
            dst.copy_(src)
        out.fill_(float("nan"))
        assert runner(inputs, tactic=tactic).data_ptr() == out.data_ptr()
        _assert_bits(out, expected)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner(inputs, tactic=tactic)
        changed_full, changed = make_inputs(m, n, k, 123)
        for live, update in zip(operands, changed, strict=True):
            live.copy_(update)
        alpha.fill_(-0.75)
        changed_expected = _reference(*changed_full, alpha)[:m, :n]
        changed_saved = [x.clone() for x in (*operands, alpha)]
        out.fill_(float("nan"))
        graph.replay()
        _assert_bits(out, changed_expected)
        for live, before in zip((*operands, alpha), changed_saved, strict=True):
            assert torch.equal(
                live.reshape(-1).view(torch.uint8), before.reshape(-1).view(torch.uint8)
            )

    out.fill_(float("nan"))
    public_graph.replay()
    _assert_bits(out, changed_expected)


def test_lmhead_shape_against_current_public_cutlass():
    m, n, k = 1, 248320, 5120
    _, operands = make_inputs(m, n, k, 42)
    alpha = torch.tensor(-0.75, dtype=torch.float32, device="cuda")
    anchor = mm_fp4(*operands, alpha, backend="cutlass")
    out = torch.full_like(anchor, float("nan"))
    saved = [x.clone() for x in (*operands, alpha)]
    runner = get_runner()
    inputs = [*operands, alpha, out.dtype, out, 16, True, None]
    for tactic in [-1, *runner.get_valid_tactics(inputs, None)]:
        out.fill_(float("nan"))
        runner(inputs, tactic=tactic)
        _assert_bits(out, anchor)
        for live, before in zip((*operands, alpha), saved, strict=True):
            assert torch.equal(
                live.reshape(-1).view(torch.uint8), before.reshape(-1).view(torch.uint8)
            )
    actual = mm_fp4(*operands, alpha, backend="cute-dsl")
    _assert_bits(actual, anchor)


def test_public_rejects_cached_raw_for_ragged_actual_m(monkeypatch):
    m, n, k = 5000, 256, 256
    full, operands = make_inputs(m, n, k, 42)
    _, aligned = make_inputs(8192, n, k, 42)
    alpha = torch.tensor(-0.75, dtype=torch.float32, device="cuda")
    out = torch.full((m, n), float("nan"), dtype=torch.bfloat16, device="cuda")
    workspace = gemm_base._get_cache_buf(
        "mm_fp4_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, operands[0].device
    )
    runner, tuner = get_runner(), AutoTuner.get()
    assert not tuner.is_tuning_mode
    config = gemm_base._MM_FP4_TUNING_CONFIG_128x4
    actual_inputs = [*operands, alpha, out.dtype, out, 16, True, workspace]
    aligned_out = torch.empty((8192, n), dtype=out.dtype, device="cuda")
    aligned_inputs = [*aligned, alpha, out.dtype, aligned_out, 16, True, workspace]

    def cache_key(inputs):
        return tuner._get_cache_key(
            "fp4_gemm",
            runner,
            tuner._get_input_sizes(inputs),
            config,
            runner.get_cache_key_extras(inputs),
        )

    raw = policy.RAW_TACTICS[0]
    assert runner.validate_tactic(aligned_inputs, raw)
    validation, compiled = [], []
    original_validate = type(runner).validate_tactic
    original_compiled = type(runner)._get_compiled

    def validate(self, inputs, tactic):
        result = original_validate(self, inputs, tactic)
        validation.append((inputs[0].shape[0], tactic, result))
        return result

    def get_compiled(self, inputs, tactic):
        compiled.append((inputs[0].shape[0], tactic))
        return original_compiled(self, inputs, tactic)

    monkeypatch.setattr(type(runner), "validate_tactic", validate)
    monkeypatch.setattr(type(runner), "_get_compiled", get_compiled)
    key = cache_key(actual_inputs)
    assert key == cache_key(aligned_inputs), "The actual public M bucket must alias"
    winners = tuner._winner_cache()
    sentinel = object()
    previous = winners.get(key, sentinel)
    winners[key] = (raw, None)
    saved = [x.clone() for x in (*operands, alpha)]
    try:
        actual = mm_fp4(*operands, alpha, out=out, backend="cute-dsl")
        _assert_bits(actual, _reference(*full, alpha)[:m])
        assert (m, raw, False) in validation
        assert compiled and all(tactic[0] != "raw" for _, tactic in compiled)
        for live, before in zip((*operands, alpha), saved, strict=True):
            assert torch.equal(
                live.reshape(-1).view(torch.uint8), before.reshape(-1).view(torch.uint8)
            )
    finally:
        if previous is sentinel:
            winners.pop(key, None)
        else:
            winners[key] = previous


@pytest.mark.parametrize("m", [129, 256])
@pytest.mark.parametrize("case", ["zero_a", "zero_b", "sparse"])
def test_zero_and_sparse_full_output(m, case):
    n, k = 256, 256
    full, _ = make_inputs(m, n, k, 42)
    if case == "zero_a":
        full[0].zero_()
    elif case == "zero_b":
        full[1].zero_()
    else:
        full[0].zero_()
        full[1].zero_()
        full[0][:, ::17] = 0x76
        full[1][::19, :] = 0xEF
    operands = (full[0][:m].clone(), *full[1:])
    alpha = torch.tensor(-0.75, dtype=torch.float32, device="cuda")
    expected = _reference(*full, alpha)[:m]
    saved = [x.clone() for x in (*operands, alpha)]
    out = torch.empty_like(expected)
    runner = get_runner()
    inputs = [*operands, alpha, out.dtype, out, 16, True, None]
    out.fill_(float("nan"))
    _assert_bits(mm_fp4(*operands, alpha, out=out, backend="cute-dsl"), expected)
    for tactic in runner.get_valid_tactics(inputs, None):
        out.fill_(float("nan"))
        runner(inputs, tactic=tactic)
        _assert_bits(out, expected)
    for live, before in zip((*operands, alpha), saved, strict=True):
        assert torch.equal(
            live.reshape(-1).view(torch.uint8), before.reshape(-1).view(torch.uint8)
        )


@pytest.mark.parametrize("n,k", [(32, 256), (256, 96)])
def test_public_rejects_dimensions_not_aligned_to_64(n, k):
    _, operands = make_inputs(1, n, k, 42)
    alpha = torch.tensor(1.25, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="N % 64 = 0, K % 64 = 0"):
        mm_fp4(*operands, alpha, backend="cute-dsl")


def test_public_rejects_tail_scales_without_physical_n_padding():
    m, n, k = 1, 192, 256
    _, operands = make_inputs(m, n, k, 42)
    a, b, sfa, sfb = operands
    alpha = torch.tensor(1.25, dtype=torch.float32, device="cuda")
    truncated_sfb = sfb.T[:n].contiguous().T
    with pytest.raises(ValueError, match="physical 128x4 scale layout"):
        mm_fp4(a, b, sfa, truncated_sfb, alpha, backend="cute-dsl")


def test_sm121_defaults_preserve_other_devices_and_unmeasured_shapes():
    targets = [
        (m, n, k)
        for m in [*range(1, 17), 32, 64, 128]
        for n, k in [(34816, 5120), (5120, 17408)]
    ]
    for m, n, k in targets:
        old = policy.default_tactic(m, n, k)
        choices = policy.valid_tactics(m, n, k)
        for cc in [None, (12, 0)]:
            assert policy.default_tactic(m, n, k, compute_capability=cc) == old
            assert policy.valid_tactics(m, n, k, compute_capability=cc) == choices
        expected = (
            ("narrow", 32, 128, 512)
            if m <= 32
            else ("b12x", 64, 128, 512)
            if m == 64
            else ("b12x_single_cta", 128, 128, 256)
        )
        assert policy.default_tactic(m, n, k, compute_capability=(12, 1)) == expected
        updated = policy.valid_tactics(m, n, k, compute_capability=(12, 1))
        assert len(updated) == len(choices)
        assert expected in updated
        assert all(
            policy.compatible(m, n, k, t, compute_capability=(12, 1)) for t in choices
        )
        assert not policy.compatible(m, n, k, expected, compute_capability=(12, 0))
    larger = [
        ((256, 896, 1024), ("cooperative", 128, 64, 256)),
        ((512, 8192, 2048), ("raw", 64, 32, 4, False, True, 256, True)),
        ((256, 9216, 7168), ("raw", 64, 32, 2, False, True, 256, True)),
        ((1024, 896, 1024), ("raw", 32, 64, 13, True, True)),
        ((512, 7168, 5120), ("raw", 64, 32, 8, False, True)),
        ((256, 7168, 4608), ("b12x", 64, 128, 256)),
        ((1024, 1024, 7168), ("cooperative", 256, 128, 128)),
        ((64, 896, 5120), ("b12x", 64, 64, 256)),
        ((64, 512, 7168), ("b12x", 64, 64, 256)),
        ((512, 5120, 640), ("cooperative", 128, 128, 128)),
        ((512, 5120, 2560), ("cooperative", 128, 64, 256)),
        ((256, 7168, 256), ("raw", 32, 64, 13, True, True, 256, False)),
        ((256, 7168, 512), ("raw", 32, 64, 13, True, True, 256, False)),
        ((128, 2688, 1856), ("b12x", 128, 128, 256)),
        ((128, 3712, 2688), ("b12x", 128, 128, 256)),
        ((128, 2688, 3712), ("b12x", 128, 128, 256)),
        ((512, 1792, 5120), ("cooperative", 128, 64, 256)),
        ((512, 5120, 1024), ("cooperative", 128, 64, 256)),
        ((512, 5120, 1280), ("cooperative", 128, 64, 256)),
        ((512, 1280, 8192), ("cooperative", 128, 128, 256)),
        ((512, 896, 5120), ("cooperative", 128, 128, 256)),
        ((512, 5120, 2048), ("cooperative", 128, 64, 256)),
        ((512, 5120, 8192), ("cooperative", 128, 64, 256)),
        ((512, 8192, 3584), ("cooperative", 128, 64, 256)),
        ((512, 3584, 5120), ("cooperative", 128, 64, 256)),
        ((512, 2560, 8192), ("cooperative", 128, 128, 256)),
        ((512, 5120, 5120), ("cooperative", 128, 64, 256)),
        ((1024, 512, 7168), ("b12x", 128, 128, 128)),
        ((8192, 34816, 5120), ("raw", 32, 64, 13, True, True, 256, False)),
        ((2048, 34816, 5120), ("cooperative", 256, 128, 128)),
        ((2048, 5120, 17408), ("raw", 64, 32, 8, False, True, 256, True)),
        ((2000, 1856, 2688), ("cooperative", 128, 128, 128)),
        ((2000, 2688, 1856), ("cooperative", 128, 128, 128)),
        ((2000, 2688, 3712), ("cooperative", 128, 128, 128)),
        ((2000, 3712, 2688), ("cooperative", 128, 128, 128)),
        ((512, 8192, 4096), ("cooperative", 128, 64, 256)),
        ((1024, 7168, 4608), ("raw", 64, 32, 8, False, True, 256, True)),
        ((512, 8192, 8192), ("cooperative", 128, 64, 256)),
        ((512, 8192, 7168), ("cooperative", 128, 64, 256)),
        ((1024, 9216, 7168), ("raw", 64, 32, 8, False, True, 256, True)),
        ((512, 10240, 8192), ("cooperative", 128, 128, 256)),
        ((512, 8192, 14336), ("cooperative", 128, 64, 256)),
        ((512, 5120, 16384), ("cooperative", 128, 128, 256)),
        ((1024, 4608, 7168), ("raw", 64, 32, 8, False, True, 256, True)),
        ((512, 8192, 28672), ("cooperative", 256, 128, 128)),
        ((512, 5120, 4096), ("cooperative", 128, 64, 256)),
        ((256, 34816, 5120), ("cooperative", 128, 128, 256)),
        ((256, 5120, 17408), ("cooperative", 128, 128, 256)),
        ((512, 34816, 5120), ("cooperative", 128, 128, 256)),
        ((512, 5120, 17408), ("cooperative", 128, 64, 256)),
        ((1024, 34816, 5120), ("cooperative", 256, 128, 128)),
        ((1024, 5120, 17408), ("raw", 64, 32, 8, False, True, 256, True)),
    ]
    for shape, preferred in larger:
        choices = policy.valid_tactics(*shape)
        assert policy.default_tactic(*shape, compute_capability=(12, 1)) == preferred
        assert len(policy.valid_tactics(*shape, compute_capability=(12, 1))) == len(
            choices
        )
        assert all(
            policy.compatible(*shape, t, compute_capability=(12, 1)) for t in choices
        )
        assert policy.compatible(*shape, preferred, compute_capability=(12, 0)) == (
            preferred in choices
        )
        if preferred in choices:
            assert policy.valid_tactics(*shape, compute_capability=(12, 1)) == choices
        if (
            preferred == ("cooperative", 128, 128, 128) and shape[0] == 2000
        ) or shape == (512, 8192, 28672):
            previous = ("cooperative", 128, 128, 256)
            assert policy.compatible(*shape, previous, compute_capability=(12, 1))
            assert previous not in policy.valid_tactics(
                *shape, compute_capability=(12, 1)
            )
            assert not policy.compatible(*shape, previous, compute_capability=(12, 0))
        if shape == (1024, 1024, 7168):
            previous = ("b12x", 64, 128, 128)
            assert policy.compatible(*shape, previous, compute_capability=(12, 1))
            assert previous not in policy.valid_tactics(
                *shape, compute_capability=(12, 1)
            )
            assert not policy.compatible(*shape, previous, compute_capability=(12, 0))
        assert policy.default_tactic(
            *shape, compute_capability=(12, 0)
        ) == policy.default_tactic(*shape)
        assert policy.valid_tactics(*shape, compute_capability=(12, 0)) == choices
    neighbors = [
        (m, n, k)
        for m in [17, 31, 33, 63, 65, 127, 129, 257, 513, 1025, 2047, 2049]
        for n, k in [(34816, 5120), (5120, 17408)]
    ]
    neighbors += [
        (255, 896, 1024),
        (257, 896, 1024),
        (256, 768, 1024),
        (256, 1024, 1024),
        (256, 896, 768),
        (256, 896, 1280),
        (511, 8192, 2048),
        (513, 8192, 2048),
        (512, 8064, 2048),
        (512, 8320, 2048),
        (512, 8192, 1792),
        (512, 8192, 2304),
        (128, 1856, 2688),
        (127, 2688, 1856),
        (129, 2688, 1856),
        (128, 2816, 1856),
        (128, 2688, 2112),
        (127, 3712, 2688),
        (129, 3712, 2688),
        (128, 3840, 2688),
        (128, 3712, 2944),
        (127, 2688, 3712),
        (129, 2688, 3712),
        (128, 2816, 3712),
        (128, 2688, 3968),
        (8191, 34816, 5120),
        (8193, 34816, 5120),
        (8192, 34944, 5120),
        (8192, 34816, 5376),
        (1999, 1856, 2688),
        (2001, 1856, 2688),
        (2000, 1984, 2688),
        (2000, 1856, 2944),
        (1999, 2688, 1856),
        (2001, 2688, 1856),
        (2000, 2816, 1856),
        (2000, 2688, 2112),
        (1999, 2688, 3712),
        (2001, 2688, 3712),
        (2000, 2816, 3712),
        (2000, 2688, 3968),
        (1999, 3712, 2688),
        (2001, 3712, 2688),
        (2000, 3840, 2688),
        (2000, 3712, 2944),
        (256, 8192, 4096),
        (1024, 8192, 4096),
        (512, 8320, 4096),
        (512, 8192, 4352),
        (512, 7168, 4608),
        (2048, 7168, 4608),
        (1024, 7296, 4608),
        (1024, 7168, 4864),
    ]
    neighbors += [
        (256, 8192, 8192),
        (1024, 8192, 8192),
        (512, 8320, 8192),
        (512, 8192, 8448),
        (256, 8192, 7168),
        (1024, 8192, 7168),
        (512, 8320, 7168),
        (512, 8192, 7424),
    ]
    neighbors += [
        (128, 9216, 7168),
        (255, 9216, 7168),
        (257, 9216, 7168),
        (256, 9088, 7168),
        (256, 9344, 7168),
        (256, 9216, 6912),
        (256, 9216, 7424),
        (512, 9216, 7168),
        (2048, 9216, 7168),
        (1024, 9344, 7168),
        (1024, 9216, 7424),
    ]
    neighbors += [(256, 10240, 8192), (1024, 10240, 8192), (512, 10368, 8192)]
    neighbors += [
        (256, 8192, 14336),
        (1024, 8192, 14336),
        (512, 8320, 14336),
        (512, 8192, 14592),
        (256, 5120, 16384),
        (1024, 5120, 16384),
        (512, 5248, 16384),
        (512, 5120, 16640),
    ]
    neighbors += [
        (512, 4608, 7168),
        (2048, 4608, 7168),
        (1024, 4736, 7168),
        (1024, 4608, 7424),
    ]
    neighbors += [
        (256, 8192, 28672),
        (1024, 8192, 28672),
        (512, 8320, 28672),
        (512, 8192, 28928),
    ]
    neighbors += [(256, 5120, 4096), (1024, 5120, 4096), (512, 5120, 4352)]
    neighbors += [
        (m, n, k)
        for m in [1, 16, 32, 64, 128]
        for n, k in [(34944, 5120), (34816, 5376), (5120, 17664)]
    ]
    neighbors += [
        (512, 512, 7168),
        (2048, 512, 7168),
        (1024, 640, 7168),
        (1024, 512, 7424),
    ]
    neighbors += [
        (511, 2560, 8192),
        (513, 2560, 8192),
        (512, 2688, 8192),
        (512, 2560, 8448),
        (511, 5120, 5120),
        (513, 5120, 5120),
        (512, 5248, 5120),
        (512, 5120, 5376),
    ]
    neighbors += [
        (511, 5120, 8192),
        (513, 5120, 8192),
        (511, 8192, 3584),
        (513, 8192, 3584),
        (511, 3584, 5120),
        (513, 3584, 5120),
        (512, 5248, 8192),
        (512, 8320, 3584),
        (512, 3712, 5120),
        (512, 5120, 8448),
        (512, 8192, 3840),
        (512, 3584, 5376),
    ]
    neighbors += [
        (513, 1792, 5120),
        (513, 5120, 1024),
        (513, 1280, 8192),
        (513, 896, 5120),
        (513, 5120, 2048),
        (512, 1920, 5120),
        (512, 5248, 1024),
        (512, 1408, 8192),
        (512, 1024, 5120),
        (512, 5248, 2048),
        (512, 1792, 5376),
        (512, 1280, 8448),
    ]
    neighbors += [
        (128, 7168, 256),
        (512, 7168, 256),
        (128, 7168, 512),
        (512, 7168, 512),
        (256, 7296, 256),
        (256, 7296, 512),
        (256, 7168, 768),
        (1023, 896, 1024),
        (1025, 896, 1024),
        (1024, 832, 1024),
        (1024, 960, 1024),
        (1024, 896, 960),
        (1024, 896, 1088),
    ]
    neighbors += [
        (511, 5120, 640),
        (513, 5120, 640),
        (512, 5248, 640),
        (512, 5120, 704),
        (511, 5120, 2560),
        (513, 5120, 2560),
        (512, 5248, 2560),
        (512, 5120, 2816),
    ]
    assert not policy.compatible(
        512, 5120, 640, ("cooperative", 128, 128, 256), compute_capability=(12, 1)
    )
    neighbors += [
        (511, 5120, 1280),
        (513, 5120, 1280),
        (512, 5056, 1280),
        (512, 5184, 1280),
        (512, 5120, 1216),
        (512, 5120, 1344),
    ]
    neighbors += [
        (511, 7168, 5120),
        (513, 7168, 5120),
        (512, 7104, 5120),
        (512, 7232, 5120),
        (512, 7168, 5056),
        (512, 7168, 5184),
    ]
    neighbors += [
        (511, 8192, 28672),
        (513, 8192, 28672),
        (512, 8128, 28672),
        (512, 8256, 28672),
        (512, 8192, 28608),
        (512, 8192, 28736),
    ]
    neighbors += [
        (255, 7168, 4608),
        (257, 7168, 4608),
        (256, 7104, 4608),
        (256, 7232, 4608),
        (256, 7168, 4544),
        (256, 7168, 4672),
    ]
    neighbors += [
        (1023, 1024, 7168),
        (1025, 1024, 7168),
        (1024, 960, 7168),
        (1024, 1088, 7168),
        (1024, 1024, 7104),
        (1024, 1024, 7232),
        (63, 896, 5120),
        (65, 896, 5120),
        (64, 832, 5120),
        (64, 960, 5120),
        (64, 896, 5056),
        (64, 896, 5184),
    ]
    neighbors += [
        (63, 512, 7168),
        (65, 512, 7168),
        (64, 448, 7168),
        (64, 576, 7168),
        (64, 512, 7104),
        (64, 512, 7232),
    ]
    for shape in neighbors:
        assert policy.default_tactic(
            *shape, compute_capability=(12, 1)
        ) == policy.default_tactic(*shape)
        assert policy.valid_tactics(
            *shape, compute_capability=(12, 1)
        ) == policy.valid_tactics(*shape)


@pytest.mark.parametrize(
    "m,n,k",
    [
        (m, n, k)
        for m in [16, 32, 64, 128, 256, 512, 1024, 2048]
        for n, k in [(34816, 5120), (5120, 17408)]
    ]
    + [
        (256, 896, 1024),
        (512, 8192, 2048),
        (256, 9216, 7168),
        (1024, 896, 1024),
        (512, 7168, 5120),
        (256, 7168, 4608),
        (1024, 1024, 7168),
        (64, 896, 5120),
        (64, 512, 7168),
        (512, 5120, 640),
        (512, 5120, 2560),
        (256, 7168, 256),
        (256, 7168, 512),
        (128, 2688, 1856),
        (128, 3712, 2688),
        (128, 2688, 3712),
        (512, 1792, 5120),
        (512, 5120, 1024),
        (512, 5120, 1280),
        (512, 1280, 8192),
        (512, 896, 5120),
        (512, 5120, 2048),
        (512, 5120, 8192),
        (512, 8192, 3584),
        (512, 3584, 5120),
        (512, 2560, 8192),
        (512, 5120, 5120),
        (1024, 512, 7168),
        (8192, 34816, 5120),
        (4096, 5120, 17408),
        (8192, 5120, 17408),
        (512, 5120, 4096),
        (512, 8192, 28672),
        (1024, 4608, 7168),
        (512, 8192, 14336),
        (512, 5120, 16384),
        (512, 10240, 8192),
        (1024, 9216, 7168),
        (512, 8192, 8192),
        (512, 8192, 7168),
        (512, 8192, 4096),
        (1024, 7168, 4608),
        (2000, 2688, 3712),
        (2000, 3712, 2688),
        (2000, 1856, 2688),
        (2000, 2688, 1856),
    ],
)
def test_sm121_measured_default_public_graph_and_cached_choice(m, n, k, monkeypatch):
    cc = get_compute_capability(torch.device("cuda"))
    if cc != (12, 1):
        pytest.skip("New defaults are qualified only on SM121")
    _, operands = make_inputs(m, n, k, 42)
    alpha = torch.tensor(1.25, dtype=torch.float32, device="cuda")
    torch.cuda.synchronize()
    expected = mm_fp4(*operands, alpha, backend="cutlass")
    out = torch.empty_like(expected)
    runner, tuner = get_runner(), AutoTuner.get()
    workspace = gemm_base._get_cache_buf(
        "mm_fp4_workspace", gemm_base.DEFAULT_WORKSPACE_SIZE, operands[0].device
    )
    inputs = [*operands, alpha, out.dtype, out, 16, True, workspace]
    key = tuner._get_cache_key(
        "fp4_gemm",
        runner,
        tuner._get_input_sizes(inputs),
        gemm_base._MM_FP4_TUNING_CONFIG_128x4,
        runner.get_cache_key_extras(inputs),
    )
    winners = tuner._winner_cache()
    sentinel = object()
    previous = winners.pop(key, sentinel)
    selected = []
    original_get = type(runner)._get_compiled

    def observe(self, inputs, tactic):
        selected.append(tactic)
        return original_get(self, inputs, tactic)

    # Instance attributes participate in the tuner's runner hash.
    monkeypatch.setattr(type(runner), "_get_compiled", observe)
    assert hash(runner) == key.runner_hash
    preferred = policy.default_tactic(m, n, k, compute_capability=cc)
    try:
        _assert_bits(mm_fp4(*operands, alpha, out=out, backend="cute-dsl"), expected)
        assert selected and selected[-1] == preferred
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            mm_fp4(*operands, alpha, out=out, backend="cute-dsl")
        _, changed = make_inputs(m, n, k, 123)
        reference_alpha = torch.tensor(-0.75, dtype=torch.float32, device="cuda")
        torch.cuda.synchronize()
        changed_expected = mm_fp4(*changed, reference_alpha, backend="cutlass")
        for live, update in zip(operands, changed, strict=True):
            live.copy_(update)
        out.fill_(float("nan"))
        alpha.fill_(-0.75)
        graph.replay()
        _assert_bits(out, changed_expected)
        # The new default does not silently migrate an existing cached choice.
        legacy_choices = [policy.default_tactic(m, n, k)]
        if (m, n, k) == (512, 8192, 2048):
            legacy_choices.append(("raw", 64, 32, 8, False, True))
        if (m, n, k) in (
            (512, 5120, 2560),
            (256, 9216, 7168),
            (256, 896, 1024),
        ):
            legacy_choices.append(("b12x", 64, 128, 256))
        if preferred == ("cooperative", 128, 128, 128) and m == 2000:
            legacy_choices.append(("cooperative", 128, 128, 256))
        if (m, n, k) == (512, 8192, 28672):
            legacy_choices.append(("cooperative", 128, 128, 256))
        if (m, n, k) == (1024, 512, 7168):
            legacy_choices.append(("cooperative", 128, 64, 256))
        if (m, n, k) == (1024, 1024, 7168):
            legacy_choices.append(("b12x", 64, 128, 128))
        for legacy in legacy_choices:
            winners[key] = (legacy, None)
            _assert_bits(
                mm_fp4(*operands, alpha, out=out, backend="cute-dsl"), changed_expected
            )
            assert selected[-1] == legacy
            if (m, n, k) == (512, 8192, 28672) and legacy == (
                "cooperative",
                128,
                128,
                256,
            ):
                cached_graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(cached_graph):
                    mm_fp4(*operands, alpha, out=out, backend="cute-dsl")
                out.fill_(float("nan"))
                cached_graph.replay()
                _assert_bits(out, changed_expected)
    finally:
        if previous is sentinel:
            winners.pop(key, None)
        else:
            winners[key] = previous


def test_compile_cache_binds_input_device_capability(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace

    from flashinfer.gemm.kernels.sm12x_cute import runner as native

    device = object()
    capability = [(12, 1)]
    calls = []
    a = SimpleNamespace(shape=(64, 2560), device=device)
    b = SimpleNamespace(shape=(2560, 34816))
    tactic = ("narrow", 32, 128, 256)

    def get_capability(actual_device):
        assert actual_device is device
        return capability[0]

    def compile_kernel(m, n, k, choice, *, compute_capability):
        calls.append((m, n, k, choice, compute_capability))
        return object()

    monkeypatch.setattr(native, "_COMPILED", {})
    monkeypatch.setattr(native, "_compile", compile_kernel)
    monkeypatch.setattr(native, "get_compute_capability", get_capability)
    monkeypatch.setattr(native, "get_device_index", lambda _: 3)
    monkeypatch.setattr(
        native,
        "torch",
        SimpleNamespace(
            cuda=SimpleNamespace(
                is_current_stream_capturing=lambda: False,
                device=lambda _: nullcontext(),
            )
        ),
    )
    runner = native.Sm12xCuTeFp4GemmRunner()
    first = runner._get_compiled([a, b], tactic)
    assert runner._get_compiled([a, b], tactic) is first
    capability[0] = (12, 0)
    assert runner._get_compiled([a, b], tactic) is not first
    assert calls == [
        (64, 34816, 5120, tactic, (12, 1)),
        (64, 34816, 5120, tactic, (12, 0)),
    ]
