# SPDX-License-Identifier: Apache-2.0
# The reference math (_staged_reference, _checkpoint_reference, _check_apply)
# and its tolerances are derived from SGLang's test_hc_sum_state.py at
# sgl-project/sglang 9cdd6624fe115979b0a0e6c588d9de5ddfeaafe2 (Apache-2.0).
# Modified 2026-11-04: rewritten for the standalone FlashInfer plan API, with FP64
# reference arithmetic and the [4,H] checkpoint RMS layout. No SGLang import or
# framework test is used.
"""Tests for the experimental SM100/SM103 HC sum-state API in
``flashinfer.hyperconnection``.

The tests use only the documented plan contract: ``bootstrap``,
``allocate_workspace``, ``mix`` and ``apply``, plus the tensors that these
calls return. The references are independent of the packed buffers:

- ``_checkpoint_reference`` normalizes the stored activations and then applies
  the raw checkpoint weights.
- ``_staged_reference`` applies the documented fold to the supplied FP32 sums,
  with the storage casts of each path as validated by the SGLang reference.
- ``_check_apply`` checks the Apply output against an FP64 model of
  ``fma(alpha, block_output, residual)``. The square sums must equal the sums of
  that update before it is rounded for storage.

``T <= 24`` takes the small fused path and ``T > 24`` takes the batched path.
Determinism is tested only for repeatability at a fixed T, never across T.
The tests set no global Torch state; FP64 references make them independent of
TF32 settings.

Smoke subset: ``pytest tests/experimental/test_hc_sum_sm100.py -k smoke``.
"""

import os

import pytest
import torch
import torch.nn.functional as F

from flashinfer.hyperconnection import (
    HCSumRuntime,
    bind_hc_sum_weights,
    hc_apply_sum,
    prepare_hc_sum_weights,
    hc_sum_weight_shapes,
    hc_sum_support_reason,
    pack_hc_sum_weights_into,
)

EPS = 1e-6
# Documented dispatch boundary: T <= 24 is the fused small path, T > 24 batched.
SMALL_T_MAX = 24
BF16, FP16 = torch.bfloat16, torch.float16
_NAMES = {BF16: "bf16", FP16: "fp16"}


def _is_sm100_family():
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in (
        (10, 0),
        (10, 3),
    )


requires_sm100 = pytest.mark.skipif(
    not _is_sm100_family(), reason="HC sum-state backend requires SM100 or SM103"
)


def _id(dtype, h, lowrank, rows, smoke=False):
    return f"{_NAMES[dtype]}-h{h}-l{lowrank}-t{rows}" + ("-smoke" if smoke else "")


def _randn(shape, dtype, seed, scale=1.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    values = torch.randn(shape, device="cuda", generator=generator)
    return (values * scale).to(dtype)


def _checkpoint(h, lowrank, dtype, seed=311):
    """Checkpoint-layout weights; gamma is ``1 + norm``.

    Unlike SGLang's 0.02-scaled synthetic checkpoint, these scales keep raw/4
    and the Up logits O(1), so a wrong Up layout or gate changes Mix by more
    than the tolerance instead of hiding near sigmoid(0).
    """
    return {
        "down": _randn((lowrank, 4 * h), dtype, seed, h**-0.5),
        "up": _randn((4 * h, lowrank), dtype, seed + 1, 2 * lowrank**-0.5),
        "inject": _randn((4, 4 * h), dtype, seed + 2, h**-0.5),
        "norm": _randn((4, h), dtype, seed + 3, 0.15),
    }


def _prepare(weights, rms_eps=EPS):
    return prepare_hc_sum_weights(
        weights["down"],
        weights["up"],
        weights["inject"],
        weights["norm"],
        rms_eps=rms_eps,
    )


def _fold(weights):
    """Documented fold of checkpoint Down+Inject: ``[L+4, 4H]`` storage dtype."""
    gamma = 1.0 + weights["norm"].float().reshape(1, -1)
    stacked = torch.cat((weights["down"], weights["inject"])).float()
    return (stacked * gamma).to(weights["down"].dtype)


def _native(weights):
    """Native bind layout: Down+Inject ``[4, round_up(L+4, 8), H]`` with zero
    padding, and the original RMS weights interleaved as ``[H, 4]``."""
    folded = _fold(weights)
    rows, width = folded.shape
    h = width // 4
    packed = folded.new_zeros((4, (rows + 7) // 8 * 8, h))
    packed[:, :rows].copy_(folded.view(rows, 4, h).transpose(0, 1))
    return packed, weights["norm"].T.contiguous()


def _square_sums(values, h):
    return values.double().view(values.shape[0], 4, h).square().sum(-1)


def _staged_reference(residual, sum_sq, folded, up, norm, rms_eps):
    """Reference from the supplied FP32 sums and folded storage-dtype weights.

    The reference keeps the documented storage boundaries: the folded weights,
    the normalized residual and the Down output on both paths. The batched path
    also rounds the merged raw Down logits, the Up GEMM output, the gate and the
    gate*norm product.
    """
    rows, width = residual.shape
    h = width // 4
    dtype = residual.dtype
    lowrank = folded.shape[0] - 4
    grouped = residual.double().view(rows, 4, h)
    weight = folded.double().view(-1, 4, h)
    inv = torch.rsqrt(sum_sq.double() / h + rms_eps)
    raw = sum(
        (grouped[:, branch] @ weight[:, branch].T) * inv[:, branch : branch + 1]
        for branch in range(4)
    )
    if rows > SMALL_T_MAX:
        raw = raw.to(dtype).double()
    down = F.silu(raw[:, :lowrank] / 4).to(dtype).double()
    alpha = 2 * torch.sigmoid(raw[:, lowrank:] / 4)
    gamma = 1.0 + norm.double()
    normed = (grouped * inv[:, :, None] * gamma).to(dtype).double()
    logits = (down @ up.double().T).view(rows, 4, h)
    if rows <= SMALL_T_MAX:
        mixed = (torch.sigmoid(logits) * normed).mean(1)
    else:
        gate = torch.sigmoid(logits.to(dtype).double()).to(dtype).double()
        mixed = (gate * normed).to(dtype).double().mean(1)
    return mixed.to(dtype), alpha.float()


def _checkpoint_reference(residual, weights, rms_eps):
    """Unfused checkpoint math: normalize the stored activations, then project
    them with the raw Down, Up and Inject weights. Nothing is folded."""
    rows, width = residual.shape
    h = width // 4
    dtype = residual.dtype
    grouped = residual.double().view(rows, 4, h)
    rms = torch.rsqrt(grouped.square().mean(-1, keepdim=True) + rms_eps)
    normed = (grouped * rms * (1.0 + weights["norm"].double())).to(dtype).double()
    flat = normed.flatten(1)
    down = F.silu(flat @ weights["down"].double().T / 4).to(dtype).double()
    logits = (down @ weights["up"].double().T).view(rows, 4, h)
    mixed = (torch.sigmoid(logits) * normed).mean(1)
    alpha = 2 * torch.sigmoid(flat @ weights["inject"].double().T / 4)
    return mixed.to(dtype), alpha.float()


def _mix_tolerance(dtype):
    # SGLang's validated Mix tolerances against the staged and unfused references.
    if dtype == BF16:
        return dict(rtol=0.02, atol=0.004)
    return dict(rtol=0.003, atol=0.0005)


def _alpha_tolerance(dtype, rows):
    if rows <= SMALL_T_MAX:
        return dict(rtol=3e-5, atol=3e-6)
    # Independent FP32 accumulations can straddle a storage midpoint before the
    # batched path's raw-logit cast; allow one storage ULP after the sigmoid.
    return dict(rtol=3e-5, atol=0.004 if dtype == BF16 else 0.0005)


def _check_mix(mixed, alpha, expected, alpha_tolerance=None):
    ref_mixed, ref_alpha = expected
    if alpha_tolerance is None:
        alpha_tolerance = _alpha_tolerance(mixed.dtype, mixed.shape[0])
    torch.testing.assert_close(mixed, ref_mixed, **_mix_tolerance(mixed.dtype))
    torch.testing.assert_close(alpha, ref_alpha, **alpha_tolerance)


def _check_bootstrap(residual, sums):
    h = residual.shape[1] // 4
    expected = _square_sums(residual, h).float()
    torch.testing.assert_close(sums, expected, rtol=3e-6, atol=1e-5)


def _check_apply(residual, alpha, block_output, new_residual, next_sum):
    """Check Apply against an FP64 model of the FP32 FMA update.

    The FP64 multiply-add is followed by one FP32 rounding, which models
    ``fma(alpha, block_output, residual)``. The stored residual is that value
    cast to the storage dtype, so it must match bitwise. The FP32 sums observe
    the update before the storage cast.
    """
    rows, h = block_output.shape
    update = (
        residual.double().view(rows, 4, h)
        + alpha.double()[:, :, None] * block_output.double()[:, None, :]
    ).float()
    expected_residual = update.to(residual.dtype).flatten(1)
    torch.testing.assert_close(new_residual, expected_residual, rtol=0, atol=0)
    pre_store = update.double().square().sum(-1)
    torch.testing.assert_close(next_sum, pre_store.float(), rtol=3e-6, atol=1e-5)
    if rows * 4 >= 32:
        # Show that the sums are not those of the stored residual. With at
        # least 32 sums, the storage-rounding term (relative error about
        # 2u/sqrt(H)) is much larger than the FP32 reduction error.
        stored = _square_sums(new_residual, h)
        pre_error = (next_sum.double() - pre_store).abs().sum().item()
        stored_error = (next_sum.double() - stored).abs().sum().item()
        assert stored_error > 4 * pre_error, (pre_error, stored_error)


def _step(plan, workspace, residual, sum_sq, block_output):
    mixed, alpha = plan.mix(residual, sum_sq, workspace)
    new_residual, next_sum = plan.apply(block_output, residual, workspace)
    return mixed, alpha, new_residual, next_sum


def _bits(tensor):
    return tensor.view(torch.uint8)


def _poison(tensors):
    for tensor in tensors:
        tensor.fill_(float("nan"))


def _rejects(call, case, errors=ValueError):
    try:
        call()
    except errors:
        return
    pytest.fail(f"{case}: expected {errors}")


@pytest.mark.cpu_only
def test_entry_points_are_experimental():
    assert prepare_hc_sum_weights.is_experimental is True
    assert bind_hc_sum_weights.is_experimental is True
    assert hc_apply_sum.is_experimental is True


@requires_sm100
@pytest.mark.parametrize("deterministic", [False, True], ids=["atomic-smoke", "fixed"])
def test_tensor_apply_sliced_handoff(deterministic):
    # A row partition shifts four FP32 statistics by16 bytes, while residual
    # and core rows remain32-byte aligned. No whole mix workspace is required.
    rows, h = 3, 512
    residual = _randn((rows, 4 * h), BF16, 501)
    block = _randn((rows, h), BF16, 502)
    alpha = _randn((rows, 4), torch.float32, 503).sigmoid_()
    output = torch.empty_like(residual)
    output[0].fill_(7.25)
    sums = torch.zeros((rows, 4), device="cuda", dtype=torch.float32)
    sums[0].fill_(13)
    partials = torch.empty((rows, 4, 10), device="cuda") if deterministic else None
    if partials is not None:
        partials[0].fill_(17)
    assert alpha[1:].data_ptr() % 32 == 16
    hc_apply_sum(
        block[1:],
        residual[1:],
        alpha[1:],
        out=output[1:],
        sum_sq=sums[1:],
        partial_sums=partials[1:] if partials is not None else None,
    )
    _check_apply(residual[1:], alpha[1:], block[1:], output[1:], sums[1:])
    assert torch.equal(output[0], torch.full_like(output[0], 7.25))
    assert torch.equal(sums[0], torch.full_like(sums[0], 13))
    if partials is not None:
        assert torch.equal(partials[0], torch.full_like(partials[0], 17))


@requires_sm100
@pytest.mark.parametrize("rows", [24, 25])
def test_current_sum_alignment_boundary(rows):
    plan = _prepare(_checkpoint(512, 64, BF16))
    residual = _randn((rows, 2048), BF16, 601)
    sums = torch.empty(rows * 4 + 4, device="cuda")[4:].view(rows, 4)
    assert sums.data_ptr() % 32 == 16
    plan.bootstrap(residual, out_sum=sums)
    workspace = plan.allocate_workspace(rows)
    if rows == 24:
        _rejects(lambda: plan.mix(residual, sums, workspace), "small-T32-byte sums")
    else:
        actual = tuple(t.clone() for t in plan.mix(residual, sums, workspace))
        aligned = plan.bootstrap(residual)
        expected = plan.mix(residual, aligned, workspace)
        for a, b in zip(actual, expected, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("dtype", [BF16, FP16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("up_on_gpu", [False, True], ids=["cpu-up", "gpu-up"])
def test_cpu_checkpoint_preparation(dtype, up_on_gpu, monkeypatch):
    gpu_weights = _checkpoint(512, 64, dtype)
    cpu_weights = {}
    for name, tensor in gpu_weights.items():
        # Contiguous memory-mapped-style views need not have CUDA alignment.
        view = torch.empty(tensor.numel() + 4, dtype=dtype)[4:].view(tensor.shape)
        view.copy_(tensor.cpu())
        assert view.data_ptr() % 16 == 8
        cpu_weights[name] = view
    up = gpu_weights["up"] if up_on_gpu else cpu_weights["up"]
    uploads = []
    original_to = torch.Tensor.to

    def record_to(tensor, *args, **kwargs):
        result = original_to(tensor, *args, **kwargs)
        if tensor.device.type == "cpu" and result.device.type == "cuda":
            uploads.append(tuple(tensor.shape))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "to", record_to)
        allocations = torch.cuda.memory_stats()["allocation.all.allocated"]
        cpu_plan = prepare_hc_sum_weights(
            cpu_weights["down"],
            up,
            cpu_weights["inject"],
            cpu_weights["norm"],
            device="cuda",
        )
        allocated = torch.cuda.memory_stats()["allocation.all.allocated"] - allocations
    assert allocated == (3 if up_on_gpu else 4)
    # Only the final packed Down+Inject, ordinary Up (if needed), and interleaved
    # RMS weights cross devices. No checkpoint-format Down/Inject GPU staging.
    assert uploads == (
        [(4, 72, 512), (512, 4)] if up_on_gpu else [(4, 72, 512), (2048, 64), (512, 4)]
    )
    gpu_plan = _prepare(gpu_weights)
    for name in ("packed_down", "up", "norm", "up_padded"):
        assert torch.equal(getattr(cpu_plan, name), getattr(gpu_plan, name))
    if up_on_gpu:
        assert cpu_plan.up.data_ptr() == up.data_ptr()
    rows = 24 if up_on_gpu else 25
    residual = _randn((rows, 2048), dtype, 701)
    core = _randn((rows, 512), dtype, 702)
    sums = cpu_plan.bootstrap(residual)
    a = _step(
        cpu_plan,
        cpu_plan.allocate_workspace(rows, deterministic=True),
        residual,
        sums,
        core,
    )
    b = _step(
        gpu_plan,
        gpu_plan.allocate_workspace(rows, deterministic=True),
        residual,
        sums,
        core,
    )
    for actual, expected in zip(a, b, strict=True):
        assert torch.equal(_bits(actual), _bits(expected))


@requires_sm100
def test_cpu_preparation_checks_target_device_capture():
    if torch.cuda.device_count() < 2:
        pytest.skip("needs two visible SM100-family GPUs")
    if torch.cuda.get_device_capability(1) not in ((10, 0), (10, 3)):
        pytest.skip("second GPU must be SM100/SM103")
    weights = {k: v.cpu() for k, v in _checkpoint(512, 64, BF16).items()}

    def prepare():
        return prepare_hc_sum_weights(
            weights["down"],
            weights["up"],
            weights["inject"],
            weights["norm"],
            device="cuda:1",
        )

    with torch.cuda.device(0):
        plan = prepare()
        assert plan.packed_down.device == torch.device("cuda:1")
    with torch.cuda.device(1):
        counter = torch.zeros(1, device="cuda:1")
        counter.add_(1)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            counter.add_(1)
            with torch.cuda.device(0):
                _rejects(prepare, "capture on target device", errors=RuntimeError)
                _rejects(
                    lambda: plan.allocate_workspace(8),
                    "workspace capture on target device",
                    errors=RuntimeError,
                )
        graph.replay()
        torch.cuda.synchronize()


@requires_sm100
def test_apply_preserves_bf16_subnormal_residual():
    residual = torch.full((1, 2048), 1e-39, device="cuda", dtype=BF16)
    block = torch.zeros((1, 512), device="cuda", dtype=BF16)
    alpha = torch.zeros((1, 4), device="cuda")
    output = torch.empty_like(residual)
    sums = torch.zeros_like(alpha)
    hc_apply_sum(block, residual, alpha, out=output, sum_sq=sums)
    assert torch.equal(output.view(torch.int16), residual.view(torch.int16))
    assert torch.equal(sums, torch.zeros_like(sums))


@requires_sm100
@pytest.mark.skipif(
    os.environ.get("FLASHINFER_TEST_HC_LARGE_INDEX") != "1",
    reason="opt-in output-index boundary test needs about16GiB of GPU buffers",
)
def test_large_deterministic_bmm_output_indices():
    from flashinfer.experimental.hc_sum_sm100.deterministic import bmm

    free_bytes, _ = torch.cuda.mem_get_info()
    if free_bytes < 24 * 1024**3:
        pytest.skip("need24GiB free for the signed32-bit output-index boundary")
    rows, h, npad = 1_020_000, 512, 712
    residual = torch.ones((rows, 4 * h), device="cuda", dtype=BF16)
    packed = torch.ones((4, npad, h), device="cuda", dtype=BF16)
    a = residual.view(rows, 4, h).transpose(0, 1)
    b = packed.transpose(1, 2)
    out = torch.empty((4, rows, npad), device="cuda", dtype=torch.float32)
    assert a.numel() < 2**31 and b.numel() < 2**31 < out.numel()
    assert 3 * out.stride(0) > 2**31
    bmm(a, b, out)
    torch.cuda.synchronize()
    for branch in range(4):
        for row in (0, rows // 2, rows - 1):
            sample = out[branch, row, -8:]
            torch.testing.assert_close(
                sample, torch.full_like(sample, h), rtol=0, atol=0
            )


# The public H2560/L320 shape and the small H512/L64 shape cover every token
# count. Two more bf16 shapes cover the alpha-only Inject tile at L=192 and the
# largest supported H.
_NUMERIC_SMOKE = {
    (BF16, 512, 64, 0),
    (BF16, 512, 64, 24),
    (BF16, 512, 64, 25),
    (BF16, 2560, 320, 1),
    (BF16, 2560, 320, 64),
}
_NUMERIC_CASES = [
    (dtype, h, lowrank, rows)
    for dtype in (BF16, FP16)
    for h, lowrank in ((512, 64), (2560, 320))
    for rows in (0, 1, 8, 16, 24, 25, 64)
]
_NUMERIC_CASES += [(BF16, 2048, 192, rows) for rows in (3, 25)]
_NUMERIC_CASES += [(BF16, 16384, 64, rows) for rows in (1, 25)]


@requires_sm100
@pytest.mark.parametrize(
    "dtype,h,lowrank,rows",
    [
        pytest.param(*case, id=_id(*case, smoke=case in _NUMERIC_SMOKE))
        for case in _NUMERIC_CASES
    ],
)
def test_bootstrap_mix_apply_match_references(dtype, h, lowrank, rows):
    weights = _checkpoint(h, lowrank, dtype)
    plan = _prepare(weights)
    residual = _randn((rows, 4 * h), dtype, seed=rows + 1)
    block_output = _randn((rows, h), dtype, seed=rows + 2)
    saved_residual = residual.clone()

    sums = plan.bootstrap(residual)
    assert sums.shape == (rows, 4) and sums.dtype == torch.float32
    _check_bootstrap(residual, sums)
    saved_sums = sums.clone()
    staged = _staged_reference(
        residual, sums, _fold(weights), weights["up"], weights["norm"], EPS
    )

    workspace = plan.allocate_workspace(rows)
    previous = None
    for _ in range(2):
        if previous is not None:
            # Every output of the second pass must be rewritten.
            _poison(previous)
        mixed, alpha = plan.mix(residual, sums, workspace)
        if previous is not None:
            # Mix clears NEXT sums before the atomic Apply.
            torch.testing.assert_close(
                previous[3], torch.zeros_like(previous[3]), rtol=0, atol=0
            )
        new_residual, next_sum = plan.apply(block_output, residual, workspace)
        outputs = (mixed, alpha, new_residual, next_sum)
        assert [(tuple(t.shape), t.dtype) for t in outputs] == [
            ((rows, h), dtype),
            ((rows, 4), torch.float32),
            ((rows, 4 * h), dtype),
            ((rows, 4), torch.float32),
        ]
        if previous is not None and rows:
            # Outputs live in workspace storage, so their addresses stay fixed.
            assert [t.data_ptr() for t in outputs] == [t.data_ptr() for t in previous]
        _check_mix(mixed, alpha, staged)
        _check_apply(residual, alpha, block_output, new_residual, next_sum)
        previous = outputs

    # Rounding x to unit roundoff u changes sum(x^2) by at most (2u + u^2) of
    # it; add the two FP32 reductions' 3e-6 budgets.
    u = torch.finfo(dtype).eps / 2
    torch.testing.assert_close(
        plan.bootstrap(new_residual),
        next_sum,
        rtol=2 * u + u * u + 6e-6,
        atol=2e-5,
    )
    torch.testing.assert_close(residual, saved_residual, rtol=0, atol=0)
    torch.testing.assert_close(sums, saved_sums, rtol=0, atol=0)


@requires_sm100
@pytest.mark.parametrize("dtype", [BF16, FP16], ids=["bf16", "fp16"])
@pytest.mark.parametrize(
    "rows", [1, 24, 25, 64], ids=["t1", "t24", "t25", "t64-checkpoint-smoke"]
)
def test_unfused_checkpoint_baseline_scale(dtype, rows):
    # Folding gamma and rounding weights changes the floating-point program.
    # The stronger projections above therefore use the staged contract. Keep
    # the original SGLang0.02 projection /0.15 RMS distribution as a separate
    # unfused-checkpoint regression, with its original tolerances unchanged.
    h, lowrank = 2560, 320
    weights = _checkpoint(h, lowrank, dtype)
    for index, name in enumerate(("down", "up", "inject")):
        weights[name] = _randn(weights[name].shape, dtype, 311 + index, 0.02)
    residual = _randn((rows, 4 * h), dtype, rows + 1)
    plan = _prepare(weights)
    sums = plan.bootstrap(residual)
    workspace = plan.allocate_workspace(rows)
    mixed, alpha = plan.mix(residual, sums, workspace)
    _check_mix(
        mixed,
        alpha,
        _checkpoint_reference(residual, weights, EPS),
        alpha_tolerance=_mix_tolerance(dtype),
    )


@requires_sm100
@pytest.mark.parametrize(
    "rows", [pytest.param(24, id="t24"), pytest.param(25, id="t25-smoke")]
)
def test_state_handoff_between_blocks(rows):
    dtype, h, lowrank = BF16, 512, 64
    first_weights = _checkpoint(h, lowrank, dtype, seed=11)
    second_weights = _checkpoint(h, lowrank, dtype, seed=23)
    # Large enough that using the first block's epsilon is detectable.
    second_eps = 0.5
    first = _prepare(first_weights)
    second = _prepare(second_weights, rms_eps=second_eps)
    first_ws = first.allocate_workspace(rows)
    second_ws = second.allocate_workspace(rows)

    residual0 = _randn((rows, 4 * h), dtype, seed=1)
    sums0 = first.bootstrap(residual0)
    mixed, alpha = first.mix(residual0, sums0, first_ws)
    core = _randn((rows, h), dtype, seed=2)
    residual1, sums1 = first.apply(core, residual0, first_ws)
    _check_apply(residual0, alpha, core, residual1, sums1)
    current = (residual1.clone(), sums1.clone())

    # The next block uses the pre-store sums from Apply directly. It does not
    # bootstrap them again, and it applies its own epsilon to them.
    mixed, alpha = second.mix(residual1, sums1, second_ws)
    expected = _staged_reference(
        residual1,
        sums1,
        _fold(second_weights),
        second_weights["up"],
        second_weights["norm"],
        second_eps,
    )
    _check_mix(mixed, alpha, expected)
    wrong_eps = _staged_reference(
        residual1,
        sums1,
        _fold(second_weights),
        second_weights["up"],
        second_weights["norm"],
        EPS,
    )
    assert not torch.allclose(alpha, wrong_eps[1], **_alpha_tolerance(dtype, rows))
    # Mix resets only its own NEXT sums. The CURRENT state is left unchanged.
    torch.testing.assert_close(residual1, current[0], rtol=0, atol=0)
    torch.testing.assert_close(sums1, current[1], rtol=0, atol=0)
    core = _randn((rows, h), dtype, seed=3)
    residual2, sums2 = second.apply(core, residual1, second_ws)
    _check_apply(residual1, alpha, core, residual2, sums2)
    torch.testing.assert_close(sums1, current[1], rtol=0, atol=0)

    # A recurrent chain alternates workspaces, so the first workspace is reused.
    mixed, alpha = first.mix(residual2, sums2, first_ws)
    expected = _staged_reference(
        residual2,
        sums2,
        _fold(first_weights),
        first_weights["up"],
        first_weights["norm"],
        EPS,
    )
    _check_mix(mixed, alpha, expected)
    core = _randn((rows, h), dtype, seed=4)
    residual3, sums3 = first.apply(core, residual2, first_ws)
    _check_apply(residual2, alpha, core, residual3, sums3)


_GRAPH_SMOKE = {(BF16, 24, False), (BF16, 25, True)}


@requires_sm100
@pytest.mark.parametrize(
    "dtype,rows,deterministic",
    [
        pytest.param(
            dtype,
            rows,
            deterministic,
            id=f"{_NAMES[dtype]}-t{rows}-{'deterministic' if deterministic else 'atomic'}"
            + ("-smoke" if (dtype, rows, deterministic) in _GRAPH_SMOKE else ""),
        )
        for dtype in (BF16, FP16)
        for rows in (24, 25)
        for deterministic in (False, True)
    ],
)
def test_graph_replay_two_block_chain(dtype, rows, deterministic):
    h, lowrank = 512, 64
    first_weights = _checkpoint(h, lowrank, dtype, seed=11)
    second_weights = _checkpoint(h, lowrank, dtype, seed=23)
    first, second = _prepare(first_weights), _prepare(second_weights)
    first_fold, second_fold = _fold(first_weights), _fold(second_weights)
    first_ws = first.allocate_workspace(rows, deterministic=deterministic)
    second_ws = second.allocate_workspace(rows, deterministic=deterministic)
    residual = torch.empty((rows, 4 * h), dtype=dtype, device="cuda")
    sums = torch.empty((rows, 4), dtype=torch.float32, device="cuda")
    noise = torch.empty((2, rows, h), dtype=dtype, device="cuda")
    cores = torch.empty_like(noise)

    def load(seed):
        residual.copy_(_randn(residual.shape, dtype, seed))
        noise.copy_(_randn(noise.shape, dtype, seed + 1))

    def step():
        first.bootstrap(residual, out_sum=sums)
        mixed0, alpha0 = first.mix(residual, sums, first_ws)
        # Stand-in for the caller's block. It reads Mix, so replay order matters.
        torch.add(mixed0, noise[0], out=cores[0])
        residual1, sums1 = first.apply(cores[0], residual, first_ws)
        mixed1, alpha1 = second.mix(residual1, sums1, second_ws)
        torch.add(mixed1, noise[1], out=cores[1])
        residual2, sums2 = second.apply(cores[1], residual1, second_ws)
        return mixed0, alpha0, residual1, sums1, mixed1, alpha1, residual2, sums2

    def check(outputs):
        mixed0, alpha0, residual1, sums1, mixed1, alpha1, residual2, sums2 = outputs
        _check_bootstrap(residual, sums)
        _check_mix(
            mixed0,
            alpha0,
            _staged_reference(
                residual,
                sums,
                first_fold,
                first_weights["up"],
                first_weights["norm"],
                EPS,
            ),
        )
        _check_apply(residual, alpha0, cores[0], residual1, sums1)
        _check_mix(
            mixed1,
            alpha1,
            _staged_reference(
                residual1,
                sums1,
                second_fold,
                second_weights["up"],
                second_weights["norm"],
                EPS,
            ),
        )
        _check_apply(residual1, alpha1, cores[1], residual2, sums2)

    load(seed=1)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        step()  # Compile and load every kernel before capture.
        allocations = torch.cuda.memory_stats().get("allocation.all.allocated", 0)
        eager = step()
        # The explicit buffers make the repeated chain allocation-free.
        assert (
            torch.cuda.memory_stats().get("allocation.all.allocated", 0) == allocations
        )
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()
    check(eager)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step()
    for seed in (101, 202, 303):
        load(seed)
        _poison((sums, cores, *captured))
        graph.replay()
        torch.cuda.synchronize()
        check(captured)

    # Replays with unchanged inputs: every captured Mix resets the atomic NEXT
    # sums, so a replay never adds to the previous replay's sums.
    expected = [_bits(t).clone() for t in captured]
    for _ in range(3):
        graph.replay()
        torch.cuda.synchronize()
        check(captured)
        if deterministic:
            for actual, bits in zip(captured, expected, strict=True):
                assert torch.equal(_bits(actual), bits)


_DETERMINISTIC_SMOKE = {(BF16, 512, 64, 24), (BF16, 512, 64, 25)}
_DETERMINISTIC_CASES = [
    (dtype, 512, 64, rows) for dtype in (BF16, FP16) for rows in (0, 1, 24, 25, 64)
]
_DETERMINISTIC_CASES += [(BF16, 2560, 320, 25)]


@requires_sm100
@pytest.mark.parametrize(
    "dtype,h,lowrank,rows",
    [
        pytest.param(*case, id=_id(*case, smoke=case in _DETERMINISTIC_SMOKE))
        for case in _DETERMINISTIC_CASES
    ],
)
def test_deterministic_mode_repeats_bits_at_fixed_rows(dtype, h, lowrank, rows):
    # No cross-T claim: only repeated calls at this T must produce equal bits.
    weights = _checkpoint(h, lowrank, dtype)
    plan = _prepare(weights)
    residual = _randn((rows, 4 * h), dtype, seed=5)
    block_output = _randn((rows, h), dtype, seed=6)
    sums = plan.bootstrap(residual)
    workspace = plan.allocate_workspace(rows, deterministic=True)

    outputs = _step(plan, workspace, residual, sums, block_output)
    _check_mix(
        outputs[0],
        outputs[1],
        _staged_reference(
            residual, sums, _fold(weights), weights["up"], weights["norm"], EPS
        ),
    )
    _check_apply(residual, outputs[1], block_output, outputs[2], outputs[3])
    expected = [_bits(t).clone() for t in outputs]
    for _ in range(3):
        _poison(outputs)
        outputs = _step(plan, workspace, residual, sums, block_output)
        for actual, bits in zip(outputs, expected, strict=True):
            assert torch.equal(_bits(actual), bits)
    # A second workspace at the same T reproduces the same bits.
    other = plan.allocate_workspace(rows, deterministic=True)
    for actual, bits in zip(
        _step(plan, other, residual, sums, block_output), expected, strict=True
    ):
        assert torch.equal(_bits(actual), bits)


@requires_sm100
@pytest.mark.parametrize(
    "rows", [pytest.param(24, id="t24"), pytest.param(25, id="t25-smoke")]
)
def test_separate_workspaces_do_not_share_scratch(rows):
    dtype, h, lowrank = BF16, 512, 64
    weights = _checkpoint(h, lowrank, dtype)
    plan = _prepare(weights)
    folded = _fold(weights)
    inputs = []
    for seed in (1, 2):
        residual = _randn((rows, 4 * h), dtype, seed)
        block_output = _randn((rows, h), dtype, seed + 100)
        sums = plan.bootstrap(residual)
        expected = _staged_reference(
            residual, sums, folded, weights["up"], weights["norm"], EPS
        )
        inputs.append((residual, sums, block_output, expected))
    workspaces = [plan.allocate_workspace(rows) for _ in inputs]

    def check(index, mixed, alpha, new_residual, next_sum):
        residual, _, block_output, expected = inputs[index]
        _check_mix(mixed, alpha, expected)
        _check_apply(residual, alpha, block_output, new_residual, next_sum)

    # Interleaved on one stream: Mix 0, Mix 1, Apply 0, Apply 1.
    mixes = [
        plan.mix(residual, sums, workspace)
        for (residual, sums, _, _), workspace in zip(inputs, workspaces, strict=True)
    ]
    applies = [
        plan.apply(block_output, residual, workspace)
        for (residual, _, block_output, _), workspace in zip(
            inputs, workspaces, strict=True
        )
    ]
    for index in range(len(inputs)):
        check(index, *mixes[index], *applies[index])

    # Concurrent streams. Hidden plan-level scratch would race here.
    streams = [torch.cuda.Stream() for _ in inputs]
    for _ in range(3):
        _poison([t for outputs in (*mixes, *applies) for t in outputs])
        for stream in streams:
            stream.wait_stream(torch.cuda.current_stream())
        for _ in range(4):
            for (residual, sums, block_output, _), workspace, stream in zip(
                inputs, workspaces, streams, strict=True
            ):
                with torch.cuda.stream(stream):
                    _step(plan, workspace, residual, sums, block_output)
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for index in range(len(inputs)):
            check(index, *mixes[index], *applies[index])


@requires_sm100
@pytest.mark.parametrize(
    "dtype,rows",
    [
        pytest.param(BF16, 24, id="bf16-t24-smoke"),
        pytest.param(BF16, 25, id="bf16-t25"),
        pytest.param(FP16, 24, id="fp16-t24"),
        pytest.param(FP16, 25, id="fp16-t25"),
    ],
)
def test_bind_native_weights_without_refold(dtype, rows):
    h, lowrank = 512, 64
    weights = _checkpoint(h, lowrank, dtype)
    # With gamma = 0 and gamma = 2 in some columns, a missing or repeated fold
    # gives a visibly different result.
    weights["norm"][0, :64] = -1
    weights["norm"][1, :64] = 1
    folded = _fold(weights)
    packed, packed_norm = _native(weights)
    prepared = _prepare(weights)
    native = bind_hc_sum_weights(
        packed, weights["up"], packed_norm, lowrank=lowrank, rms_eps=EPS
    )
    residual = _randn((rows, 4 * h), dtype, seed=7)
    block_output = _randn((rows, h), dtype, seed=8)
    sums = native.bootstrap(residual)
    native_ws = native.allocate_workspace(rows, deterministic=True)
    prepared_ws = prepared.allocate_workspace(rows, deterministic=True)

    outputs = _step(native, native_ws, residual, sums, block_output)
    expected = _staged_reference(
        residual, sums, folded, weights["up"], weights["norm"], EPS
    )
    _check_mix(outputs[0], outputs[1], expected)
    _check_apply(residual, outputs[1], block_output, outputs[2], outputs[3])
    native_bits = [_bits(t).clone() for t in outputs]

    # The single-storage contract is observable through the prepared views.
    assert native.packed_down.data_ptr() == packed.data_ptr()
    assert native.norm.data_ptr() == packed_norm.data_ptr()
    assert native.up.data_ptr() == weights["up"].data_ptr()

    # A content update requires rebinding. Zero folded Down+Inject makes every
    # raw logit exactly zero, so alpha = 2 * sigmoid(0) = 1.
    saved = packed.clone()
    packed.zero_()
    zero_plan = bind_hc_sum_weights(packed, weights["up"], packed_norm, lowrank=lowrank)
    zero_ws = zero_plan.allocate_workspace(rows, deterministic=True)
    mixed, alpha = zero_plan.mix(residual, sums, zero_ws)
    torch.testing.assert_close(alpha, torch.ones_like(alpha), rtol=0, atol=3e-6)
    zero_expected = _staged_reference(
        residual,
        sums,
        torch.zeros_like(folded),
        weights["up"],
        weights["norm"],
        EPS,
    )
    _check_mix(mixed, alpha, zero_expected)
    packed.copy_(saved)

    # The bound RMS weights scale only the Mix input; nothing folds them again.
    # With gamma = 0, Mix is exactly zero and alpha keeps its earlier bits.
    packed_norm.fill_(-1)
    zero_norm_plan = bind_hc_sum_weights(
        packed, weights["up"], packed_norm, lowrank=lowrank
    )
    zero_norm_ws = zero_norm_plan.allocate_workspace(rows, deterministic=True)
    mixed, alpha = zero_norm_plan.mix(residual, sums, zero_norm_ws)
    torch.testing.assert_close(mixed, torch.zeros_like(mixed), rtol=0, atol=0)
    assert torch.equal(_bits(alpha), native_bits[1])
    packed_norm.copy_(weights["norm"].T)

    # prepare folds the checkpoint once into its own storage and keeps no raw
    # Down/Inject/RMS tensor. Poisoning those tensors afterwards changes
    # nothing, and the prepared plan gives the same bits as the bound plan.
    for name in ("down", "inject", "norm"):
        weights[name].fill_(float("nan"))
    prepared_outputs = _step(prepared, prepared_ws, residual, sums, block_output)
    for actual, bits in zip(prepared_outputs, native_bits, strict=True):
        assert torch.equal(_bits(actual), bits)


_CONTRACT_CASES = [
    pytest.param(512, 64, EPS, True, id="h512-l64-smoke"),
    pytest.param(512, 192, EPS, True, id="l192"),
    pytest.param(2560, 320, EPS, True, id="h2560-l320-smoke"),
    pytest.param(16384, 64, EPS, True, id="h16384"),
    pytest.param(256, 64, EPS, False, id="h256"),
    pytest.param(768, 64, EPS, False, id="h768-smoke"),
    pytest.param(16896, 64, EPS, False, id="h16896"),
    pytest.param(512, 32, EPS, False, id="l32"),
    pytest.param(512, 96, EPS, False, id="l96"),
    pytest.param(512, 128, EPS, False, id="l128-no-inject-room-smoke"),
    pytest.param(512, 256, EPS, False, id="l256-no-inject-room"),
    pytest.param(512, 64, 0.0, False, id="eps-zero-smoke"),
    pytest.param(512, 64, -1e-6, False, id="eps-negative"),
    pytest.param(512, 64, float("nan"), False, id="eps-nan"),
    pytest.param(512, 64, float("inf"), False, id="eps-inf"),
]


@requires_sm100
@pytest.mark.parametrize("h,lowrank,rms_eps,valid", _CONTRACT_CASES)
def test_contract_limits(h, lowrank, rms_eps, valid):
    weights = _checkpoint(h, lowrank, BF16)
    packed, packed_norm = _native(weights)

    def prepare():
        return _prepare(weights, rms_eps)

    def bind():
        return bind_hc_sum_weights(
            packed, weights["up"], packed_norm, lowrank=lowrank, rms_eps=rms_eps
        )

    if valid:
        prepare()
        bind()
    else:
        _rejects(prepare, "prepare")
        _rejects(bind, "bind")


@requires_sm100
@pytest.mark.parametrize(
    "dtype", [pytest.param(BF16, id="bf16-smoke"), pytest.param(FP16, id="fp16")]
)
def test_weight_metadata_rejected(dtype):
    h, lowrank = 512, 64
    other = FP16 if dtype == BF16 else BF16
    weights = _checkpoint(h, lowrank, dtype)
    packed, packed_norm = _native(weights)
    _prepare(weights)
    bind_hc_sum_weights(packed, weights["up"], packed_norm, lowrank=lowrank)

    prepare_cases = {
        "up_dtype": {"up": weights["up"].to(other)},
        "up_rank": {"up": _randn((4 * h, lowrank + 64), dtype, seed=9)},
        "inject_rows": {"inject": weights["inject"][:3].contiguous()},
        "norm_native_layout": {"norm": packed_norm},
        "down_strided": {"down": weights["down"].T.contiguous().T},
        "inject_cpu": {"inject": weights["inject"].cpu()},
        "fp32": {name: tensor.float() for name, tensor in weights.items()},
        "cpu": {name: tensor.cpu() for name, tensor in weights.items()},
    }
    for case, override in prepare_cases.items():
        _rejects(lambda: _prepare({**weights, **override}), f"prepare {case}")

    native = {
        "packed_down": packed,
        "up": weights["up"],
        "packed_norm": packed_norm,
        "lowrank": lowrank,
    }
    bind_cases = {
        "unpadded_down": {"packed_down": packed[:, : lowrank + 4].contiguous()},
        "transposed_down": {"packed_down": packed.transpose(1, 2).contiguous()},
        "strided_down": {
            "packed_down": packed.transpose(1, 2).contiguous().transpose(1, 2)
        },
        "checkpoint_norm_layout": {"packed_norm": weights["norm"]},
        "up_dtype": {"up": weights["up"].to(other)},
        "lowrank_mismatch": {"lowrank": 192},
        "cpu": {
            "packed_down": packed.cpu(),
            "up": weights["up"].cpu(),
            "packed_norm": packed_norm.cpu(),
        },
    }
    for case, override in bind_cases.items():
        args = {**native, **override}
        _rejects(
            lambda: bind_hc_sum_weights(
                args["packed_down"],
                args["up"],
                args["packed_norm"],
                lowrank=args["lowrank"],
            ),
            f"bind {case}",
        )


@requires_sm100
@pytest.mark.parametrize(
    "dtype,rows",
    [
        pytest.param(BF16, 24, id="bf16-t24-smoke"),
        pytest.param(BF16, 25, id="bf16-t25"),
        pytest.param(FP16, 25, id="fp16-t25"),
    ],
)
def test_runtime_metadata_rejected(dtype, rows):
    h, lowrank = 512, 64
    other = FP16 if dtype == BF16 else BF16
    plan = _prepare(_checkpoint(h, lowrank, dtype))
    other_plan = _prepare(_checkpoint(h, lowrank, dtype, seed=5))
    residual = _randn((rows, 4 * h), dtype, seed=10)
    block_output = _randn((rows, h), dtype, seed=11)
    sums = plan.bootstrap(residual)
    workspace = plan.allocate_workspace(rows)

    for bad_rows in (-1, 2.0):
        _rejects(lambda: plan.allocate_workspace(bad_rows), f"rows {bad_rows}")
    _rejects(lambda: plan.mix(residual, None, workspace), "mix requires explicit sums")

    wide = torch.empty((rows, 8 * h), dtype=dtype, device="cuda")
    bad_residuals = {
        "dtype": residual.to(other),
        "width": residual[:, : 3 * h].contiguous(),
        "grouped_3d": residual.view(rows, 4, h),
        "strided": wide[:, : 4 * h],
        "cpu": residual.cpu(),
    }
    for case, bad in bad_residuals.items():
        _rejects(lambda: plan.bootstrap(bad), f"bootstrap residual {case}")
        _rejects(lambda: plan.mix(bad, sums, workspace), f"mix residual {case}")
        _rejects(
            lambda: plan.apply(block_output, bad, workspace), f"apply residual {case}"
        )

    misaligned = torch.empty(rows * 4 + 1, device="cuda")[1:].view(rows, 4)
    bad_sums = {
        "dtype": sums.to(dtype),
        "fp64": sums.double(),
        "shape": sums[:, :3].contiguous(),
        "cpu": sums.cpu(),
        "misaligned": misaligned,
    }
    for case, bad in bad_sums.items():
        _rejects(lambda: plan.mix(residual, bad, workspace), f"mix sums {case}")
        _rejects(
            lambda: plan.bootstrap(residual, out_sum=bad), f"bootstrap out_sum {case}"
        )

    mixed, alpha = plan.mix(residual, sums, workspace)
    bad_outputs = {
        "dtype": block_output.to(other),
        "fp32": block_output.float(),
        "width": block_output[:, : h - 64].contiguous(),
        "cpu": block_output.cpu(),
    }
    for case, bad in bad_outputs.items():
        _rejects(lambda: plan.apply(bad, residual, workspace), f"block output {case}")

    foreign = {
        "rows": plan.allocate_workspace(rows + 1),
        "plan": other_plan.allocate_workspace(rows),
    }
    for case, bad in foreign.items():
        _rejects(lambda: plan.mix(residual, sums, bad), f"mix workspace {case}")
        _rejects(
            lambda: plan.apply(block_output, residual, bad), f"apply workspace {case}"
        )
    if torch.cuda.device_count() > 1:
        _rejects(
            lambda: plan.mix(residual.to("cuda:1"), sums.to("cuda:1"), workspace),
            "mix on another device",
        )

    # Validation fails before any launch, so the earlier Mix still feeds Apply.
    new_residual, next_sum = plan.apply(block_output, residual, workspace)
    _check_apply(residual, alpha, block_output, new_residual, next_sum)


@requires_sm100
@pytest.mark.parametrize(
    "rows", [pytest.param(24, id="t24-smoke"), pytest.param(25, id="t25")]
)
def test_aliasing_rejected(rows):
    dtype, h, lowrank = BF16, 512, 64
    weights = _checkpoint(h, lowrank, dtype)
    plan = _prepare(weights)
    residual = _randn((rows, 4 * h), dtype, seed=12)
    block_output = _randn((rows, h), dtype, seed=13)
    sums = plan.bootstrap(residual)
    workspace = plan.allocate_workspace(rows)
    outputs = _step(plan, workspace, residual, sums, block_output)
    mixed, alpha, new_residual, next_sum = outputs
    snapshot = [_bits(t).clone() for t in outputs]

    cases = {
        "current_sums_are_next": lambda: plan.mix(residual, next_sum, workspace),
        "current_sums_are_alpha": lambda: plan.mix(residual, alpha, workspace),
        # A recurrent chain must alternate workspaces.
        "mix_residual_is_output": lambda: plan.mix(new_residual, sums, workspace),
        "apply_residual_is_output": lambda: plan.apply(
            block_output, new_residual, workspace
        ),
        "block_output_in_destination": lambda: plan.apply(
            new_residual.view(-1)[: rows * h].view(rows, h), residual, workspace
        ),
        "bootstrap_into_residual": lambda: plan.bootstrap(
            residual,
            out_sum=residual.view(-1).view(torch.float32)[: rows * 4].view(rows, 4),
        ),
        "bootstrap_into_weights": lambda: plan.bootstrap(
            residual,
            out_sum=weights["up"]
            .view(-1)
            .view(torch.float32)[: rows * 4]
            .view(rows, 4),
        ),
    }
    for case, call in cases.items():
        _rejects(call, case)
    for actual, bits in zip(outputs, snapshot, strict=True):
        assert torch.equal(_bits(actual), bits)

    # Mix's own output can be used directly as the block output (identity block).
    mixed, alpha = plan.mix(residual, sums, workspace)
    new_residual, next_sum = plan.apply(mixed, residual, workspace)
    _check_apply(residual, alpha, mixed, new_residual, next_sum)


@requires_sm100
@pytest.mark.parametrize(
    "dtype", [pytest.param(BF16, id="bf16-smoke"), pytest.param(FP16, id="fp16")]
)
def test_preparation_rejected_during_capture(dtype):
    h, lowrank = 512, 64
    weights = _checkpoint(h, lowrank, dtype)
    packed, packed_norm = _native(weights)
    plan = _prepare(weights)
    counter = torch.zeros(1, device="cuda")
    cases = {
        "prepare": lambda: _prepare(weights),
        "bind": lambda: bind_hc_sum_weights(
            packed, weights["up"], packed_norm, lowrank=lowrank
        ),
        "allocate_workspace": lambda: plan.allocate_workspace(8),
    }
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        counter.add_(1)  # Keep the capture non-empty.
        for case, call in cases.items():
            _rejects(call, case, errors=(RuntimeError, ValueError))
    graph.replay()
    torch.cuda.synchronize()
    assert counter.item() == 1


@requires_sm100
@pytest.mark.parametrize("dtype", [BF16, FP16], ids=["bf16", "fp16"])
@pytest.mark.parametrize("rows", [0, 1, 3, 24, 25, 64])
@pytest.mark.parametrize("deterministic", [False, True])
def test_runtime_matches_explicit_workspace(dtype, rows, deterministic):
    plan = _prepare(_checkpoint(512, 64, dtype))
    runtime = HCSumRuntime(plan, deterministic=deterministic)
    residual = _randn((rows, 2048), dtype, 801)
    core = _randn((rows, 512), dtype, 802)
    workspace = plan.allocate_workspace(rows, deterministic=deterministic)
    mixed_ref, alpha_ref = plan.mix(residual, plan.bootstrap(residual), workspace)
    out_ref, sums_ref = plan.apply(core, residual, workspace)
    mixed, context = runtime.mix(residual)
    state = context.apply(core, residual)
    torch.testing.assert_close(mixed, mixed_ref, rtol=0, atol=0)
    torch.testing.assert_close(context.alpha, alpha_ref, rtol=0, atol=0)
    torch.testing.assert_close(state.residual, out_ref, rtol=0, atol=0)
    torch.testing.assert_close(
        state.sum_sq,
        sums_ref,
        rtol=0 if deterministic else 1e-6,
        atol=0 if deterministic else 1e-3,
    )
    assert state.matches(state.residual)
    assert not state.matches(state.residual.clone()) or rows == 0
    with pytest.raises(ValueError, match="residual differs"):
        context.apply(core, torch.empty((rows + 1, 2048), device="cuda", dtype=dtype))


@requires_sm100
@pytest.mark.parametrize("deterministic", [False, True])
def test_runtime_odd_row_partition_carries_exact_sums(deterministic):
    plan = _prepare(_checkpoint(512, 64, BF16))
    runtime = HCSumRuntime(plan, deterministic=deterministic)
    residual = _randn((5, 2048), BF16, 811)
    core = _randn((5, 512), BF16, 812)
    _, context = runtime.mix(residual)
    part = context.slice_rows(2, 1)  # Starts at row 3: sums are 16, not 32, aligned.
    state = part.apply(core.tensor_split(2)[1])
    _check_apply(part.residual, part.alpha, core[3:], state.residual, state.sum_sq)
    first = context.slice_rows(2, 0)
    first_state = first.apply(core[:3], residual[:3])
    _check_apply(
        first.residual, first.alpha, core[:3], first_state.residual, first_state.sum_sq
    )
    assert state.sum_sq.data_ptr() % 32 == 16
    mixed, next_context = runtime.mix(state.residual, state)
    # Use a distinct explicit workspace and the same PRE-storage-cast sums.
    workspace = plan.allocate_workspace(2, deterministic=deterministic)
    expected, alpha = plan.mix(state.residual, state.sum_sq.clone(), workspace)
    torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
    torch.testing.assert_close(next_context.alpha, alpha, rtol=0, atol=0)
    with pytest.raises(ValueError, match="residual differs"):
        part.with_residual(residual[:2])


@requires_sm100
@pytest.mark.parametrize("rows", [3, 25])
@pytest.mark.parametrize("deterministic", [False, True])
def test_runtime_graph_output_lifetime_and_reset(rows, deterministic):
    plan = _prepare(_checkpoint(512, 64, BF16))
    runtime = HCSumRuntime(plan, deterministic=deterministic)
    residual = _randn((rows, 2048), BF16, 821)
    core = _randn((rows, 512), BF16, 822)

    def run():
        _, first_context = runtime.mix(residual)
        first = first_context.apply(core)
        _, second_context = runtime.mix(first.residual, first)
        second = second_context.apply(core)
        return first, second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first, second = run()
    assert first.residual.data_ptr() != second.residual.data_ptr()
    assert first.sum_sq.data_ptr() != second.sum_sq.data_ptr()
    for seed in (823, 824, 825):
        residual.copy_(_randn((rows, 2048), BF16, seed))
        graph.replay()
        first_ref, second_ref = run()
        for actual, expected in ((first, first_ref), (second, second_ref)):
            torch.testing.assert_close(
                actual.residual, expected.residual, rtol=0, atol=0
            )
            torch.testing.assert_close(
                actual.sum_sq, expected.sum_sq, rtol=1e-6, atol=1e-3
            )


@requires_sm100
@pytest.mark.parametrize("dtype", [BF16, FP16], ids=["bf16", "fp16"])
def test_pack_into_native_preserves_storage_and_plan_math(dtype):
    weights = _checkpoint(512, 64, dtype)
    down_shape, norm_shape = hc_sum_weight_shapes(512, 64)
    packed = torch.full(down_shape, float("nan"), dtype=dtype, device="cuda")
    norm = torch.full(norm_shape, float("nan"), dtype=dtype, device="cuda")
    addresses = packed.data_ptr(), norm.data_ptr()
    cpu = {name: weights[name].cpu() for name in ("down", "inject", "norm")}
    allocations_before = torch.cuda.memory_stats()["allocation.all.allocated"]
    pack_hc_sum_weights_into(
        cpu["down"],
        cpu["inject"],
        cpu["norm"],
        out_down=packed,
        out_norm=norm,
    )
    assert torch.cuda.memory_stats()["allocation.all.allocated"] == allocations_before
    assert addresses == (packed.data_ptr(), norm.data_ptr())
    reference = _prepare(weights)
    torch.testing.assert_close(packed, reference.packed_down, rtol=0, atol=0)
    torch.testing.assert_close(norm.flatten(), reference.norm, rtol=0, atol=0)
    assert torch.count_nonzero(packed[:, 68:]) == 0
    native = bind_hc_sum_weights(packed, weights["up"], norm, lowrank=64)
    assert native.packed_down.data_ptr() == packed.data_ptr()
    assert native.norm.data_ptr() == norm.data_ptr()
    assert native.up.data_ptr() == weights["up"].data_ptr()
    bad_norm = weights["norm"].cpu().float()
    before = packed.clone()
    with pytest.raises(ValueError):
        pack_hc_sum_weights_into(
            weights["down"].cpu(),
            weights["inject"].cpu(),
            bad_norm,
            out_down=packed,
            out_norm=norm,
        )
    torch.testing.assert_close(packed, before, rtol=0, atol=0)


def test_runtime_public_markers_and_source_eligibility(monkeypatch):
    from flashinfer.experimental.hc_sum_sm100 import jit
    from flashinfer.hyperconnection import HCApplyContext, HCSumState

    assert HCSumRuntime.is_experimental
    assert HCSumState.is_experimental
    assert HCApplyContext.is_experimental
    assert pack_hc_sum_weights_into.is_experimental
    assert hc_sum_support_reason(hidden_size=512, lowrank=64, dtype=BF16, device="cpu")
    assert hc_sum_support_reason(hidden_size=511, lowrank=64, dtype=BF16, device="cuda")
    with pytest.raises(ValueError):
        hc_sum_weight_shapes(512, 128)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (10, 0))

    def missing():
        raise RuntimeError("HC CUDA sources are absent")

    monkeypatch.setattr(jit, "require_sources", missing)
    reason = hc_sum_support_reason(
        hidden_size=512, lowrank=64, dtype=BF16, device="cuda"
    )
    assert "sources are absent" in reason


def test_pack_into_cpu_outputs_and_rejected_aliases():
    h, lowrank = 512, 64
    gen = torch.Generator().manual_seed(881)
    down = torch.randn((lowrank, 4 * h), generator=gen).to(BF16)
    inject = torch.randn((4, 4 * h), generator=gen).to(BF16)
    norm = (torch.randn((4, h), generator=gen) * 0.1).to(BF16)
    down_shape, norm_shape = hc_sum_weight_shapes(h, lowrank)
    packed, interleaved = (
        torch.empty(down_shape, dtype=BF16),
        torch.empty(norm_shape, dtype=BF16),
    )
    pack_hc_sum_weights_into(down, inject, norm, out_down=packed, out_norm=interleaved)
    expected = (torch.cat((down, inject)).float() * (1 + norm.float()).flatten()).to(
        BF16
    )
    torch.testing.assert_close(
        packed[:, :68], expected.view(68, 4, h).transpose(0, 1), rtol=0, atol=0
    )
    torch.testing.assert_close(interleaved, norm.T, rtol=0, atol=0)
    assert torch.count_nonzero(packed[:, 68:]) == 0
    before = packed.clone()
    with pytest.raises(ValueError, match="overlap"):
        pack_hc_sum_weights_into(
            down,
            inject,
            norm,
            out_down=packed,
            out_norm=packed.flatten()[: 4 * h].view(h, 4),
        )
    with pytest.raises(ValueError, match="Tensor"):
        pack_hc_sum_weights_into(
            down, inject, norm, out_down=None, out_norm=interleaved
        )
    torch.testing.assert_close(packed, before, rtol=0, atol=0)


@requires_sm100
def test_runtime_rejects_bad_inputs_and_bootstraps_replaced_residual():
    from flashinfer.hyperconnection import HCSumState

    plan = _prepare(_checkpoint(512, 64, BF16))
    residual = _randn((3, 2048), BF16, 891)
    runtime = HCSumRuntime(plan)
    with pytest.raises(ValueError):
        HCSumRuntime(plan, deterministic="false")
    with pytest.raises(ValueError):
        runtime.mix(residual, "not a state")
    with pytest.raises(ValueError, match="sum_sq"):
        runtime.mix(residual, HCSumState(residual, None))
    replaced = residual.clone()
    stale = HCSumState(residual, torch.full((3, 4), float("nan"), device="cuda"))
    mixed, ctx = runtime.mix(replaced, stale)
    ws = plan.allocate_workspace(3)
    expected, _ = plan.mix(replaced, plan.bootstrap(replaced), ws)
    torch.testing.assert_close(mixed, expected, rtol=0, atol=0)
    with pytest.raises(ValueError):
        ctx.slice_rows(0, 0)
    with pytest.raises(ValueError):
        ctx.slice_rows(2, 2)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        with pytest.raises(RuntimeError, match="outside CUDA Graph"):
            HCSumRuntime(plan)
        residual.zero_()


@requires_sm100
@pytest.mark.parametrize("deterministic", [False, True])
def test_runtime_graph_odd_partition_and_poisoned_destinations(deterministic):
    plan = _prepare(_checkpoint(512, 64, BF16))
    runtime = HCSumRuntime(plan, deterministic=deterministic)
    residual = _randn((5, 2048), BF16, 901)
    core = _randn((5, 512), BF16, 902)

    def run():
        mixed1, context = runtime.mix(residual)
        part = context.slice_rows(2, 1)
        first = part.apply(core[3:], residual[3:])
        mixed2, next_context = runtime.mix(first.residual, first)
        second = next_context.apply(core[3:])
        return first, second, part.alpha, next_context.alpha, mixed1, mixed2

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            run()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        first, second, alpha1, alpha2, mixed1, mixed2 = run()
    for seed in (903, 904, 905):
        residual.copy_(_randn((5, 2048), BF16, seed))
        for state in (first, second):
            state.residual.fill_(float("nan"))
            state.sum_sq.fill_(float("nan"))
        graph.replay()
        _check_apply(residual[3:], alpha1, core[3:], first.residual, first.sum_sq)
        _check_apply(first.residual, alpha2, core[3:], second.residual, second.sum_sq)
        _, _, expected_alpha1, expected_alpha2, expected_mixed1, expected_mixed2 = run()
        for actual, expected in ((alpha1, expected_alpha1), (alpha2, expected_alpha2)):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-6)
        for actual, expected in ((mixed1, expected_mixed1), (mixed2, expected_mixed2)):
            torch.testing.assert_close(actual, expected, **_mix_tolerance(BF16))


@requires_sm100
def test_independent_runtimes_on_concurrent_streams():
    plan = _prepare(_checkpoint(512, 64, BF16))
    lanes = []
    for rows in (3, 24):
        runtime = HCSumRuntime(plan, deterministic=True)
        residual = _randn((rows, 2048), BF16, 911 + rows)
        core = _randn((rows, 512), BF16, 912 + rows)
        mixed, context = runtime.mix(residual)
        context.apply(core)
        expected_mixed, expected_alpha = mixed.clone(), context.alpha.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        lanes.append((runtime, residual, core, expected_mixed, expected_alpha, stream))
    results = []
    for runtime, residual, core, expected_mixed, expected_alpha, stream in lanes:
        with torch.cuda.stream(stream):
            for _ in range(3):
                mixed, context = runtime.mix(residual)
                state = context.apply(core)
                results.append(
                    (
                        residual,
                        context.alpha,
                        core,
                        state,
                        mixed,
                        expected_mixed,
                        expected_alpha,
                    )
                )
    for *_, stream in lanes:
        torch.cuda.current_stream().wait_stream(stream)
    for residual, alpha, core, state, mixed, expected_mixed, expected_alpha in results:
        torch.testing.assert_close(mixed, expected_mixed, rtol=0, atol=0)
        torch.testing.assert_close(alpha, expected_alpha, rtol=0, atol=0)
        _check_apply(residual, alpha, core, state.residual, state.sum_sq)
