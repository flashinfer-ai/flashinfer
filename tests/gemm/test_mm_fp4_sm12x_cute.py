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
    padded = (m + 127) // 128 * 128
    full = _operands(padded, n, k, seed)
    # Logical A has no allocated padding; scale storage keeps physical128x4 rows.
    return full, (full[0][:m].clone(), *full[1:])


@pytest.mark.parametrize("m", [1, 4, 127, 128, 129, 5000, 8192])
@pytest.mark.parametrize("pdl", [False, True])
def test_public_and_each_tactic_full_bits_graph_mutation(m, pdl):
    n, k = 256, 256
    full, operands = make_inputs(m, n, k, 42)
    alpha = torch.tensor(1.25, dtype=torch.float32, device="cuda")
    out = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
    expected = _reference(*full, alpha)[:m]
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
        changed_expected = _reference(*changed_full, alpha)[:m]
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
    original_validate = runner.validate_tactic
    original_compiled = runner._get_compiled

    def validate(inputs, tactic):
        result = original_validate(inputs, tactic)
        validation.append((inputs[0].shape[0], tactic, result))
        return result

    def get_compiled(inputs, tactic):
        compiled.append((inputs[0].shape[0], tactic))
        return original_compiled(inputs, tactic)

    monkeypatch.setattr(runner, "validate_tactic", validate)
    monkeypatch.setattr(runner, "_get_compiled", get_compiled)
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
