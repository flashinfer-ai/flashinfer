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


@pytest.mark.parametrize(
    "m,n,k",
    [
        (1, 1856, 2688),
        (128, 1856, 2688),
        (2000, 1856, 2688),
        (1, 2688, 1856),
        (128, 2688, 1856),
        (2000, 2688, 1856),
        (1, 3712, 2688),
        (128, 3712, 2688),
        (2000, 3712, 2688),
        (1, 2688, 3712),
        (128, 2688, 3712),
        (2000, 2688, 3712),
        (1, 1024, 7168),
        (4, 1024, 7168),
        (16, 1024, 7168),
        (256, 1024, 7168),
        (1024, 1024, 7168),
        (1, 7168, 512),
        (4, 7168, 512),
        (16, 7168, 512),
        (64, 7168, 512),
        (256, 7168, 512),
        (1024, 7168, 512),
        (1, 7168, 4608),
        (4, 7168, 4608),
        (16, 7168, 4608),
        (64, 7168, 4608),
        (256, 7168, 4608),
        (1024, 7168, 4608),
        (1, 9216, 7168),
        (4, 9216, 7168),
        (16, 9216, 7168),
        (64, 9216, 7168),
        (256, 9216, 7168),
        (1024, 9216, 7168),
        (1, 512, 7168),
        (4, 512, 7168),
        (16, 512, 7168),
        (64, 512, 7168),
        (256, 512, 7168),
        (1024, 512, 7168),
        (1, 7168, 256),
        (4, 7168, 256),
        (16, 7168, 256),
        (64, 7168, 256),
        (256, 7168, 256),
        (1024, 7168, 256),
        (4, 7168, 2304),
        (16, 7168, 2304),
        (64, 7168, 2304),
        (256, 7168, 2304),
        (1, 4608, 7168),
        (4, 4608, 7168),
        (16, 4608, 7168),
        (64, 4608, 7168),
        (256, 4608, 7168),
        (1024, 4608, 7168),
        (1, 896, 1024),
        (4, 896, 1024),
        (16, 896, 1024),
        (64, 896, 1024),
        (256, 896, 1024),
        (1024, 896, 1024),
        (1, 10240, 8192),
        (8, 10240, 8192),
        (64, 10240, 8192),
        (512, 10240, 8192),
        (1, 8192, 8192),
        (8, 8192, 8192),
        (64, 8192, 8192),
        (512, 8192, 8192),
        (1, 8192, 28672),
        (8, 8192, 28672),
        (64, 8192, 28672),
        (512, 8192, 28672),
        (1, 7168, 5120),
        (8, 7168, 5120),
        (64, 7168, 5120),
        (512, 7168, 5120),
        (1, 5120, 5120),
        (8, 5120, 5120),
        (64, 5120, 5120),
        (512, 5120, 5120),
        (1, 5120, 16384),
        (8, 5120, 16384),
        (64, 5120, 16384),
        (512, 5120, 16384),
        (1, 5120, 8192),
        (8, 5120, 8192),
        (64, 5120, 8192),
        (512, 5120, 8192),
        (1, 8192, 4096),
        (8, 8192, 4096),
        (64, 8192, 4096),
        (512, 8192, 4096),
        (1, 8192, 14336),
        (8, 8192, 14336),
        (64, 8192, 14336),
        (512, 8192, 14336),
        (1, 3584, 5120),
        (8, 3584, 5120),
        (64, 3584, 5120),
        (512, 3584, 5120),
        (1, 5120, 2560),
        (8, 5120, 2560),
        (64, 5120, 2560),
        (512, 5120, 2560),
        (1, 5120, 4096),
        (8, 5120, 4096),
        (64, 5120, 4096),
        (512, 5120, 4096),
        (1, 2560, 8192),
        (8, 2560, 8192),
        (64, 2560, 8192),
        (512, 2560, 8192),
        (1, 8192, 2048),
        (8, 8192, 2048),
        (64, 8192, 2048),
        (512, 8192, 2048),
        (1, 8192, 7168),
        (8, 8192, 7168),
        (64, 8192, 7168),
        (512, 8192, 7168),
        (1, 1792, 5120),
        (8, 1792, 5120),
        (64, 1792, 5120),
        (512, 1792, 5120),
        (1, 5120, 1280),
        (8, 5120, 1280),
        (64, 5120, 1280),
        (512, 5120, 1280),
        (1, 5120, 2048),
        (8, 5120, 2048),
        (64, 5120, 2048),
        (512, 5120, 2048),
        (1, 1280, 8192),
        (8, 1280, 8192),
        (64, 1280, 8192),
        (512, 1280, 8192),
        (1, 8192, 1024),
        (8, 8192, 1024),
        (64, 8192, 1024),
        (512, 8192, 1024),
        (1, 8192, 3584),
        (8, 8192, 3584),
        (64, 8192, 3584),
        (512, 8192, 3584),
        (1, 896, 5120),
        (8, 896, 5120),
        (64, 896, 5120),
        (512, 896, 5120),
        (1, 5120, 640),
        (8, 5120, 640),
        (64, 5120, 640),
        (512, 5120, 640),
        (1, 5120, 1024),
        (8, 5120, 1024),
        (64, 5120, 1024),
        (512, 5120, 1024),
        (16, 34816, 5120),
        (16, 5120, 17408),
        (32, 34816, 5120),
        (32, 5120, 17408),
        (64, 34816, 5120),
        (64, 5120, 17408),
        (128, 34816, 5120),
        (128, 5120, 17408),
        (256, 34816, 5120),
        (256, 5120, 17408),
        (512, 34816, 5120),
        (512, 5120, 17408),
        (1024, 34816, 5120),
        (1024, 5120, 17408),
        (2048, 34816, 5120),
        (2048, 5120, 17408),
        (8192, 34816, 5120),
        (4096, 5120, 17408),
        (8192, 5120, 17408),
    ],
)
def test_public_default_graph_and_cached_choice(m, n, k, monkeypatch):
    cc = get_compute_capability(torch.device("cuda"))
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
        for choice in runner.get_valid_tactics(inputs, None):
            # Compile before graph capture and install a genuine current tactic.
            runner(inputs, tactic=choice)
            winners[key] = (choice, None)
            _assert_bits(
                mm_fp4(*operands, alpha, out=out, backend="cute-dsl"), changed_expected
            )
            assert selected[-1] == choice
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
    tactic = policy.SMALL

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
