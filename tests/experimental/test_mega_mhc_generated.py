"""Independent fused mHC values, scale bytes, any token count, current stream and graph epochs."""

import gc

import pytest
import torch
from flashinfer.experimental.deepgemm_mega_mhc import mega_mhc as _runtime
from flashinfer.mega_mhc import prepare_mega_mhc

EXPORTED_TOKENS = (1, 64, 65, 200, 1025, 4096)
HELD_OUT_TOKENS = (3, 100, 257, 384, 448, 1000, 2048, 5000)
# The BF16-output route (hidden 4096): both sides of every split-count switch of
# its rule on the SM counts it ships for, small and irregular counts, and the
# large counts of its performance table.
H4096_TOKENS = (
    1,
    2,
    3,
    7,
    33,
    64,
    65,
    127,
    192,
    193,
    256,
    384,
    385,
    576,
    577,
    832,
    833,
    1024,
    1025,
    1152,
    1153,
    1280,
    1281,
    1664,
    1665,
    2048,
    2049,
    4096,
    8192,
)
H4096_SM_COUNTS = (148, 152, 208, 212, 216)
# The architectures every hidden-4096 program ships for, in the records' order.
H4096_ARCHES = ["sm_107a", "sm_100a", "sm_103a"]
# The hidden-4096 output layouts: the BF16 output alone, or the BF16 output beside
# its quantization (vLLM's per-token-group FP8 of it; bitwise, the tests hold).
H4096_LAYOUTS = ("bf16", "fp8_group128", "nvfp4")
BF16_OUTPUTS = ("new_residual", "new_post_mix", "new_comb_res_mix", "y_bf16")
QUANT_OUTPUTS = {
    "bf16": (),
    "fp8_group128": ("y_fp8", "y_fp8_sf"),
    "nvfp4": ("y_fp4", "y_fp4_sf"),
}
H4096_CONSTANTS = dict(
    hc_norm_eps=1e-5,
    hc_pre_eps=1e-6,
    hc_post_scale=2.0,
    sinkhorn_eps=1e-6,
    num_sinkhorn_iters=20,
    rmsnorm_eps=1e-5,
    rmsnorm_scale=1.0,
)


def inputs(tokens, shifted):
    hidden = 5120
    token_scale = (
        1.0 + (torch.arange(tokens, device="cuda", dtype=torch.float32) % 2) * 0.5
    )
    pattern = torch.tensor([1.0, 0.5, -1.0, -0.5], device="cuda")
    return dict(
        x=token_scale[:, None].expand(tokens, hidden).to(torch.bfloat16).contiguous(),
        residual=torch.tensor(
            [0.25, 0.5, 0.75, 1.0], device="cuda", dtype=torch.bfloat16
        )
        .view(1, 4, 1)
        .expand(tokens, 4, hidden)
        .contiguous(),
        post_mix=torch.full((tokens, 4, 1), 0.5, device="cuda"),
        comb_res_mix=torch.eye(4, device="cuda").expand(tokens, 4, 4).contiguous(),
        shifted_prev_mix=torch.full((tokens, 4, 1), 0.25, device="cuda")
        if shifted
        else None,
        fn=((torch.arange(24, device="cuda", dtype=torch.float32) - 12) / 1024)
        .view(24, 1)
        .expand(24, 4 * hidden)
        .contiguous(),
        mix_scales=torch.tensor([1 / 16, 1 / 32, 1 / 64], device="cuda"),
        mix_bases=(torch.arange(24, device="cuda", dtype=torch.float32) - 12) / 128,
        rmsnorm_weight=pattern[
            (torch.arange(hidden, device="cuda") // 32) % 4
        ].bfloat16(),
        hc_mult=4,
        hc_norm_eps=2e-5,
        hc_pre_eps=3e-4,
        hc_post_scale=1.25,
        sinkhorn_eps=2e-6,
        num_sinkhorn_iters=20,
        rmsnorm_eps=7e-6,
        rmsnorm_scale=1.25,
    )


def reference(values, layout):
    tokens, hidden = values["x"].shape
    # comb_res_mix is the identity and every projection row is constant. This
    # closed evaluation avoids a second GPU GEMM implementation as the oracle.
    residual = (
        values["x"].float().unsqueeze(1) * values["post_mix"]
        + values["residual"].float()
    ).bfloat16()
    mixes = residual.float().sum((1, 2))[:, None] * values["fn"][:, 0]
    mixes *= torch.rsqrt(
        residual.float().square().mean((1, 2)) + values["hc_norm_eps"]
    ).unsqueeze(1)
    scales = torch.cat(
        (
            values["mix_scales"][0].expand(4),
            values["mix_scales"][1].expand(4),
            values["mix_scales"][2].expand(16),
        )
    )
    mixes = mixes * scales + values["mix_bases"]
    prev_mix = mixes[:, :4].sigmoid().unsqueeze(2) + values["hc_pre_eps"]
    comb = mixes[:, 8:].reshape(tokens, 4, 4).softmax(-1) + values["sinkhorn_eps"]
    comb = comb / (comb.sum(-2, keepdim=True) + values["sinkhorn_eps"])
    for _ in range(1, values["num_sinkhorn_iters"]):
        comb = comb / (comb.sum(-1, keepdim=True) + values["sinkhorn_eps"])
        comb = comb / (comb.sum(-2, keepdim=True) + values["sinkhorn_eps"])
    pre = (
        values["shifted_prev_mix"]
        if values["shifted_prev_mix"] is not None
        else prev_mix
    )
    norm_input = (residual.float() * pre).sum(1).bfloat16().float()
    norm_scale = (
        torch.rsqrt(norm_input.square().mean(1) + values["rmsnorm_eps"])
        * values["rmsnorm_scale"]
    )
    normalized = (
        norm_input * norm_scale[:, None] * values["rmsnorm_weight"].float()
    ).bfloat16()
    result = dict(
        new_residual=residual,
        new_post_mix=(mixes[:, 4:8].sigmoid() * values["hc_post_scale"]).unsqueeze(2),
        new_comb_res_mix=comb,
        y_bf16=normalized,
    )
    if values["shifted_prev_mix"] is not None:
        result["new_prev_mix"] = prev_mix
    groups = normalized.reshape(tokens, hidden // 32, 32)
    amax_bits = (
        groups.float().abs().amax(-1).bfloat16().view(torch.int16).to(torch.int32)
        & 0xFFFF
    )
    exponents = ((amax_bits + 0x1F) >> 7).clamp_min(113) - 8
    sf_inv = torch.ldexp(
        torch.ones_like(exponents, dtype=torch.float32), 127 - exponents
    )
    result["y_fp8"] = (
        (groups.float() * sf_inv[:, :, None])
        .to(torch.float8_e4m3fn)
        .reshape(tokens, hidden)
    )
    words = (
        (
            exponents.to(torch.int64).reshape(tokens, hidden // 128, 4)
            << (torch.arange(4, device="cuda", dtype=torch.int64) * 8)
        )
        .sum(-1)
        .to(torch.int32)
    )
    result["y_gemm_sf" if layout == "col" else "y_routed_sf"] = words
    if layout == "extra":
        result["y_shared_sf"] = words
    return result


def logical(outputs):
    values = {
        key: value
        for key, value in outputs.items()
        if isinstance(value, torch.Tensor)
        and key not in ("y_shared_sf", "y_shared_sf_storage")
    }
    if "y_shared_sf_storage" in outputs:
        tokens = outputs["y_bf16"].shape[0]
        block = outputs["shared_sf_block_m"]
        rows = torch.arange(tokens, device="cuda")
        index = rows % block
        physical = (
            rows // block * ((block + 127) // 128 * 128)
            + (index & ~127)
            + (index & 31) * 4
            + ((index >> 5) & 3)
        )
        values["y_shared_sf"] = outputs["y_shared_sf_storage"][physical]
    return values


def check(outputs, expected):
    for name, value in logical(outputs).items():
        tolerance = (
            0
            if value.dtype == torch.int32
            else (
                0.1
                if name == "y_fp8"
                else 1e-2
                if value.dtype == torch.bfloat16
                else 1e-5
            )
        )
        if value.dtype == torch.int32:
            torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
        else:
            torch.testing.assert_close(
                value.float(), expected[name].float(), atol=tolerance, rtol=tolerance
            )


def poison(outputs):
    for value in outputs.values():
        if not isinstance(value, torch.Tensor):
            continue
        if value.dtype == torch.int32:
            value.fill_(-559038737)
        elif value.dtype == torch.uint8:
            value.fill_(0xFF)  # two saturated negative E2M1 codes
        else:
            value.fill_(float("nan"))


def _skip_unless_exported(hidden=5120, sf_layout="col"):
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch, _sms = _runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))
    if not _runtime.supported(arch, hidden, sf_layout):
        pytest.skip(
            f"no exported mHC program for hidden={hidden} {sf_layout} on {arch}"
        )


def _replay_identical(plan, expected, stream, repeats):
    first = None
    for _ in range(repeats):
        with torch.cuda.stream(stream):
            poison(plan.outputs)
            plan.run()
        stream.synchronize()
        check(plan.outputs, expected)
        actual = logical(plan.outputs)
        if first is None:
            first = {name: value.clone() for name, value in actual.items()}
        else:
            for name, value in actual.items():
                if value.dtype == torch.int32:
                    torch.testing.assert_close(value, first[name], atol=0, rtol=0)
                else:
                    torch.testing.assert_close(
                        value.float(), first[name].float(), atol=0, rtol=0
                    )
        stream.wait_stream(torch.cuda.current_stream())


@pytest.mark.parametrize("tokens", EXPORTED_TOKENS)
@pytest.mark.parametrize("shifted", (False, True))
@pytest.mark.parametrize("layout", ("col", "extra"))
def test_fused_values_scales_and_changed_input_replay(tokens, shifted, layout):
    _skip_unless_exported()
    values = inputs(tokens, shifted)
    plan = prepare_mega_mhc(**values, sf_layout=layout)
    expected = reference(values, layout)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _replay_identical(plan, expected, stream, 3)
    if not (tokens in (65, 1025) or (shifted and tokens in (1, 64, 200))):
        return
    graphs = []
    for _ in range(2):
        with torch.cuda.stream(stream):
            plan.run()
            plan.run()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            plan.run()
        graphs.append(graph)
    stream.synchronize()
    previous = plan.launch_epochs.cpu().tolist()
    active = plan.config["num_launch_sms"]
    for repeat in range(12):
        with torch.cuda.stream(stream):
            values["x"].add_(0.015625)
            expected = reference(values, layout)
            poison(plan.outputs)
            if repeat in (5, 10):
                plan.run()
            else:
                graphs[0 if repeat < 7 else 1].replay()
        stream.synchronize()
        check(plan.outputs, expected)
        current = plan.launch_epochs.cpu().tolist()
        # Epoch storage always covers the device's full SM count.
        assert len(current) == plan.config["num_sms"]
        assert len(set(current[:active])) == 1
        assert all(
            after == before + 1
            for before, after in zip(previous[:active], current[:active], strict=True)
        )
        assert current[active:] == previous[active:]
        previous = current
        stream.wait_stream(torch.cuda.current_stream())


@pytest.mark.parametrize("tokens", HELD_OUT_TOKENS)
@pytest.mark.parametrize("shifted", (False, True))
@pytest.mark.parametrize("layout", ("col", "extra"))
def test_any_token_count(tokens, shifted, layout):
    """Token counts outside the exported denominator route by split class, not by exact count."""
    _skip_unless_exported()
    values = inputs(tokens, shifted)
    plan = prepare_mega_mhc(**values, sf_layout=layout)
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    assert plan.config["num_splits"] == _runtime.num_splits(tokens, 5120, sms)
    assert plan.config["num_sms"] == sms
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _replay_identical(plan, reference(values, layout), stream, 2)


@pytest.mark.parametrize("tokens", (1, 200, 384))
def test_deterministic_mode_uses_sixteen_splits(tokens):
    _skip_unless_exported()
    values = inputs(tokens, True)
    plan = prepare_mega_mhc(**values, sf_layout="col", deterministic=True)
    assert plan.config["num_splits"] == 16
    assert plan.config["num_launch_sms"] == plan.config["num_sms"]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    _replay_identical(plan, reference(values, "col"), stream, 2)


def test_bf16_only_layout_is_not_exported():
    _skip_unless_exported()
    with pytest.raises(NotImplementedError):
        prepare_mega_mhc(**inputs(64, False), sf_layout="bf16")


def test_fp8_plan_refuses_a_replacement_fn():
    """The FP8 programs read the ``fn`` bound at plan time: ``update_fn()`` is a
    no-op there, and a replacement tensor is refused rather than ignored."""
    _skip_unless_exported()
    values = inputs(64, False)
    plan = prepare_mega_mhc(**values, sf_layout="col")
    assert plan.fnp is None
    assert plan.update_fn() is None
    with pytest.raises(ValueError, match="bound at plan time"):
        plan.update_fn(values["fn"].clone())


# ---------------------------------------------------------------------------- the BF16-output route (hidden 4096)


def inputs_h4096(tokens, seed=2026):
    hidden = 4096
    generator = torch.Generator(device="cuda").manual_seed(seed)

    def randn(*shape):
        return torch.randn(shape, device="cuda", generator=generator)

    comb = torch.rand(tokens, 4, 4, device="cuda", generator=generator)
    for _ in range(3):
        comb /= comb.sum(-1, keepdim=True)
        comb /= comb.sum(-2, keepdim=True)
    return dict(
        x=randn(tokens, hidden).bfloat16(),
        residual=randn(tokens, 4, hidden).bfloat16(),
        post_mix=randn(tokens, 4, 1).sigmoid(),
        comb_res_mix=comb,
        shifted_prev_mix=None,
        fn=randn(24, 4 * hidden) * 0.01,
        mix_scales=randn(3) * 0.1,
        mix_bases=randn(24) * 0.1,
        rmsnorm_weight=(1 + randn(hidden) * 0.1).bfloat16(),
        hc_mult=4,
        **H4096_CONSTANTS,
    )


def reference_h4096(values, dtype=torch.float64, residual_cur=None):
    """The fused step in ``dtype`` (float64, the oracle; float32, the IEEE
    comparator) with BF16 rounding only at its two output points; the FP32
    outputs stay in ``dtype``. Given ``residual_cur`` (a BF16 ``new_residual``)
    the step starts from that value instead of its own post sum."""
    tokens, hidden = values["x"].shape
    if residual_cur is None:
        x, residual = values["x"].to(dtype), values["residual"].to(dtype)
        post, comb = values["post_mix"].to(dtype), values["comb_res_mix"].to(dtype)
        residual_cur = (
            post * x.unsqueeze(1) + torch.einsum("tij,tih->tjh", comb, residual)
        ).bfloat16()
    u = residual_cur.to(dtype).reshape(tokens, 4 * hidden)
    z = (u @ values["fn"].to(dtype).T) * torch.rsqrt(
        u.square().mean(-1, keepdim=True) + values["hc_norm_eps"]
    )
    scales = torch.cat(
        (
            values["mix_scales"][0].expand(4),
            values["mix_scales"][1].expand(4),
            values["mix_scales"][2].expand(16),
        )
    ).to(dtype)
    a = z * scales + values["mix_bases"].to(dtype)
    pre = a[:, :4].sigmoid() + values["hc_pre_eps"]
    eps = values["sinkhorn_eps"]
    c = a[:, 8:].reshape(tokens, 4, 4).softmax(-1) + eps
    c = c / (c.sum(-2, keepdim=True) + eps)
    for _ in range(1, values["num_sinkhorn_iters"]):
        c = c / (c.sum(-1, keepdim=True) + eps)
        c = c / (c.sum(-2, keepdim=True) + eps)
    w = (pre.unsqueeze(2) * residual_cur.to(dtype)).sum(1)
    rstd = torch.rsqrt(w.square().mean(-1, keepdim=True) + values["rmsnorm_eps"])
    rstd = rstd * values["rmsnorm_scale"]
    layer_input = (
        w.bfloat16().to(dtype) * rstd * values["rmsnorm_weight"].to(dtype)
    ).bfloat16()
    return dict(
        new_residual=residual_cur,
        new_post_mix=(values["hc_post_scale"] * a[:, 4:8].sigmoid()).unsqueeze(2),
        new_comb_res_mix=c,
        y_bf16=layer_input,
    )


def ieee_fp32_comparator(values):
    """``reference_h4096`` in float32 with TF32 off and the matmul precision at
    ``highest`` for its duration (restored afterwards)."""
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    precision = torch.get_float32_matmul_precision()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    try:
        return reference_h4096(values, torch.float32)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.set_float32_matmul_precision(precision)


def fp32_errors(value, oracle):
    """``(rel-L2, max abs)`` of an FP32 output against the oracle, in float64."""
    error = value.double() - oracle.double()
    return (error.norm() / oracle.double().norm()).item(), error.abs().max().item()


FP32_ULP = 2.0**-23


def check_h4096(outputs, values, label=""):
    """The BF16 outputs at ``atol = rtol = 1e-2`` against the FP64 oracle; the
    FP32 outputs given each side's own BF16 ``new_residual`` (the arithmetic
    past the post sum's rounding), since an FP32-quality implementation's error
    there is bounded by the IEEE FP32 comparator's on the same inputs: rel-L2
    <= 1e-5 and (<= 2x the comparator's or below one FP32 ulp, as differences
    below one FP32 ulp are the comparator's own rounding, not implementation
    error), max abs <= 2e-5 or <= 2x the comparator's. The readings against the
    oracle itself and the literal ``atol=2e-5/rtol=1e-5`` clause are printed,
    not gated."""
    oracle = reference_h4096(values)
    comparator = ieee_fp32_comparator(values)
    given = reference_h4096(values, residual_cur=outputs["new_residual"])
    given_c = reference_h4096(values, residual_cur=comparator["new_residual"])
    met = {True: "met", False: "unmet"}
    for name, value in outputs.items():
        if value.dtype == torch.bfloat16:
            torch.testing.assert_close(
                value.float(), oracle[name].float(), atol=1e-2, rtol=1e-2
            )
            continue
        rel_l2, max_abs = fp32_errors(value, given[name])
        rel_l2_c, max_abs_c = fp32_errors(comparator[name], given_c[name])
        direct = fp32_errors(value, oracle[name])
        direct_c = fp32_errors(comparator[name], oracle[name])
        literal, literal_c = (
            torch.allclose(side[name].double(), oracle[name], atol=2e-5, rtol=1e-5)
            for side in (outputs, comparator)
        )
        print(
            f"{label}{name}: given u rel-L2 {rel_l2:.3e} (comparator {rel_l2_c:.3e}),"
            f" max abs {max_abs:.3e} (comparator {max_abs_c:.3e});"
            f" vs oracle rel-L2 {direct[0]:.3e} (comparator {direct_c[0]:.3e}),"
            f" max abs {direct[1]:.3e} (comparator {direct_c[1]:.3e});"
            f" literal atol=2e-5/rtol=1e-5 {met[literal]}"
            f" (comparator {met[literal_c]})"
        )
        assert rel_l2 <= 1e-5 and (rel_l2 <= 2 * rel_l2_c or rel_l2 <= FP32_ULP), (
            name,
            rel_l2,
            rel_l2_c,
        )
        assert max_abs <= 2e-5 or max_abs <= 2 * max_abs_c, (name, max_abs, max_abs_c)


FP8_GROUP = 128
FP8_MAX = 448.0  # the largest finite float8_e4m3fn magnitude
FP8_EPS = 1e-10  # the absmax floor of the pinned op, read as an FP32 literal


def _fp32(value):
    return float(torch.tensor(value, dtype=torch.float32).item())


def _pow2_ceil(x):
    """The smallest power of two >= x (x > 0), exact in x's dtype."""
    m, e = torch.frexp(x)
    return torch.where(
        m == 0.5, torch.exp2((e - 1).to(x.dtype)), torch.exp2(e.to(x.dtype))
    )


def _round_to_e4m3(x):
    """x (float64, |x| <= 448) rounded to the float8_e4m3fn grid, nearest even."""
    a = x.abs()
    _, e = torch.frexp(a)
    step = torch.exp2((e.to(torch.float64) - 1).clamp_min(-6.0) - 3)
    return torch.copysign((torch.round(a / step) * step).clamp_max(FP8_MAX), x)


def quant_fp8_group128(y_bf16, ue8m0):
    """vLLM's ``per_token_group_quant_fp8(y_bf16, 128, use_ue8m0=ue8m0)`` (its CUDA
    op) in torch: per token and group of 128, ``absmax = max(max |y|, 1e-10)``,
    ``scale = fp32(absmax / 448)`` (with ``ue8m0`` the smallest power of two
    >= max(scale, 1e-10)), codes ``e4m3(fp32(y / scale))`` clamped to +-448 and
    rounded to nearest even; the sign of a zero code is kept."""
    tokens, hidden = y_bf16.shape
    y = y_bf16.double().reshape(tokens, hidden // FP8_GROUP, FP8_GROUP)
    absmax = y.abs().amax(dim=-1).clamp_min(_fp32(FP8_EPS))
    scale = (absmax / FP8_MAX).float()
    if ue8m0:
        scale = _pow2_ceil(scale.double().clamp_min(_fp32(FP8_EPS))).float()
    quotient = (y.float() / scale[..., None]).double()
    codes = _round_to_e4m3(quotient.clamp(-FP8_MAX, FP8_MAX))
    return codes.float().to(torch.float8_e4m3fn).reshape(tokens, hidden), scale


def bitwise_equal(a, b):
    """Bitwise equality of two tensors of one dtype (float8 through uint8 views)."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype == torch.float8_e4m3fn:
        return torch.equal(a.view(torch.uint8), b.view(torch.uint8))
    return torch.equal(
        a.view(torch.int32) if a.dtype == torch.float32 else a,
        b.view(torch.int32) if b.dtype == torch.float32 else b,
    )


def check_fp8(outputs, ue8m0=True, label=""):
    """``y_fp8`` and ``y_fp8_sf`` bitwise what the pinned op computes from the
    plan's own ``y_bf16``: the codes as bytes, the scales as FP32 words."""
    codes, scales = quant_fp8_group128(outputs["y_bf16"], ue8m0)
    code_mismatches = int(
        (outputs["y_fp8"].view(torch.uint8) != codes.view(torch.uint8)).sum()
    )
    scale_mismatches = int(
        (outputs["y_fp8_sf"].view(torch.int32) != scales.view(torch.int32)).sum()
    )
    assert code_mismatches == 0 and scale_mismatches == 0, (
        f"{label}y_fp8: {code_mismatches} of {codes.numel()} codes differ, "
        f"y_fp8_sf: {scale_mismatches} of {scales.numel()} scales differ (ue8m0={ue8m0})"
    )


E2M1_VALUES = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
)  # the E2M1 magnitude of code 0..7; bit 3 is the sign
E2M1_MAX = 6.0
FP4_GROUP = 16
FP4_CODE_STEP_MAX = 1  # the op's fast-math chain leaves the exact formula by one code step on a few codes
FP4_BYTE_MISMATCH_MAX = (
    2.0e-2  # ... on at most this share of the packed bytes (measured 0.25-1.32 %)
)
FP4_DEQUANT_FACTOR = 1.02


def _round_to_e2m1(x):
    """x (float64, |x| <= 6) rounded to the E2M1 grid, nearest even."""
    a = x.abs()
    step = torch.where(
        a < 2.0,
        torch.full_like(a, 0.5),
        torch.where(a < 4.0, torch.ones_like(a), torch.full_like(a, 2.0)),
    )
    return torch.copysign((torch.round(a / step) * step).clamp_max(E2M1_MAX), x)


def _e2m1_codes(values):
    table = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=values.device)
    code = (values.abs()[..., None] == table).to(torch.int16).argmax(-1).to(torch.int16)
    return torch.where(torch.signbit(values), code | 0x8, code)


def _e2m1_values(codes):
    table = torch.tensor(E2M1_VALUES, dtype=torch.float64, device=codes.device)
    magnitude = table[(codes & 0x7).to(torch.int64)]
    return torch.where((codes & 0x8) != 0, -magnitude, magnitude)


def unpack_e2m1(packed):
    """The nibbles (int16 [T, 2N]) of packed bytes [T, N]: element 2k is the low nibble of byte k."""
    lo = (packed & 0xF).to(torch.int16)
    hi = (packed >> 4).to(torch.int16)
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def pack_e2m1(codes):
    return (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)


def quant_nvfp4(y_bf16, global_scale):
    """vLLM's ``scaled_fp4_quant(y_bf16, gs, is_sf_swizzled_layout=False)`` as the
    exact formula in torch: per token and block of 16, ``SF = e4m3(min(fp32(gs *
    fp32(amax * fp32(1/6))), 448))`` (the op's FP32 product order, bitwise the
    op), codes ``e2m1(y * gs / SF)`` clamped to +-6 (zero where SF == 0), element
    2k in the low nibble of byte k; the op's own code arithmetic (an approximate
    reciprocal chain) differs from this by one code step on a few codes."""
    tokens, hidden = y_bf16.shape
    gs = global_scale.reshape(()).to(torch.float32)
    inv6 = torch.tensor(1.0 / E2M1_MAX, dtype=torch.float32, device=y_bf16.device)
    y = y_bf16.double().reshape(tokens, hidden // FP4_GROUP, FP4_GROUP)
    amax = y.abs().amax(dim=-1).to(torch.float32)
    sf = _round_to_e4m3((gs * (amax * inv6)).clamp_max(FP8_MAX).double())
    inv = torch.where(sf > 0, gs.double() / sf, torch.zeros_like(sf))
    codes = _e2m1_codes(_round_to_e2m1((y * inv[..., None]).clamp(-E2M1_MAX, E2M1_MAX)))
    return pack_e2m1(codes.reshape(tokens, hidden)), sf.float().to(torch.float8_e4m3fn)


def dequant_nvfp4(packed, sf, global_scale):
    """The consumer's ``x * (SF / gs)`` in float64."""
    gs = global_scale.reshape(()).double()
    block = (sf.double() / gs).repeat_interleave(FP4_GROUP, dim=-1)
    return _e2m1_values(unpack_e2m1(packed)) * block


def fp4_global_scale_for(y_bf16):
    """The consumer's global scale fitted to an output: ``448 * 6 / amax`` as a
    float32 [1] device tensor (1.0 for an all-zero output)."""
    amax = y_bf16.float().abs().amax()
    scale = torch.where(amax > 0, FP8_MAX * E2M1_MAX / amax, torch.ones_like(amax))
    return scale.reshape(1).to(torch.float32)


def layout_kwargs(layout, values):
    """The extra plan keywords of a layout: the NVFP4 global scale fitted to the
    oracle's BF16 output (the kernel's is within one BF16 ulp of it)."""
    if layout == "nvfp4":
        return {
            "fp4_global_scale": fp4_global_scale_for(reference_h4096(values)["y_bf16"])
        }
    return {}


def check_nvfp4(outputs, fp4_global_scale, oracle_y_bf16=None, label=""):
    """``y_fp4_sf`` bitwise the formula on the plan's own ``y_bf16``; ``y_fp4``
    within one code step of it on at most 2 % of the bytes; and, given the
    oracle's BF16 output, the dequantized output's rel-L2 against it within
    1.02x that of the formula applied to the oracle's output itself."""
    gs = fp4_global_scale
    codes, sf = quant_nvfp4(outputs["y_bf16"], gs)
    sf_mismatches = int(
        (outputs["y_fp4_sf"].view(torch.uint8) != sf.view(torch.uint8)).sum()
    )
    assert sf_mismatches == 0, (
        f"{label}y_fp4_sf: {sf_mismatches} of {sf.numel()} scale bytes differ"
    )
    got, ref = unpack_e2m1(outputs["y_fp4"]), unpack_e2m1(codes)

    def ordered(c):
        magnitude = (c & 0x7).to(torch.int32)
        return torch.where((c & 0x8) != 0, -magnitude, magnitude)

    step = int((ordered(got) - ordered(ref)).abs().max())
    byte_rate = float((outputs["y_fp4"] != codes).float().mean())
    print(f"{label}y_fp4: byte mismatch rate {byte_rate:.3e}, max code step {step}")
    assert step <= FP4_CODE_STEP_MAX, (
        f"{label}y_fp4: max code step {step} > {FP4_CODE_STEP_MAX}"
    )
    assert byte_rate <= FP4_BYTE_MISMATCH_MAX, (
        f"{label}y_fp4: byte mismatch rate {byte_rate:.3e}"
    )
    if oracle_y_bf16 is not None:
        target = oracle_y_bf16.double()
        norm = target.norm()
        err = (
            dequant_nvfp4(outputs["y_fp4"], outputs["y_fp4_sf"], gs) - target
        ).norm() / norm
        path = quant_nvfp4(oracle_y_bf16, gs)
        err_path = (dequant_nvfp4(path[0], path[1], gs) - target).norm() / norm
        print(
            f"{label}y_fp4: dequantized rel-L2 {err:.4e} vs the formula's {err_path:.4e}"
        )
        assert err <= FP4_DEQUANT_FACTOR * err_path, (float(err), float(err_path))


def check_h4096_layout(outputs, values, layout, label="", **kw):
    """``check_h4096`` on the four outputs every layout stores, then the layout's
    quantized outputs against the pinned op's definition on the plan's own BF16
    (``kw``: the plan's ``fp8_sf_ue8m0`` or ``fp4_global_scale``)."""
    check_h4096({name: outputs[name] for name in BF16_OUTPUTS}, values, label)
    if layout == "fp8_group128":
        check_fp8(outputs, kw.get("fp8_sf_ue8m0", True), label)
    elif layout == "nvfp4":
        oracle = reference_h4096(values)["y_bf16"]
        check_nvfp4(outputs, kw["fp4_global_scale"], oracle, label)


# The token counts at which the hidden-4096 split count changes, per SM count the
# program ships for (the two-wave rule of its source of record: a count of at
# least ten where the tokens fit two tail rounds of the grid, the deepgemm count
# of at least 16 elsewhere), and the count at each point on the 208-SM board.
H4096_SWITCH_POINTS = {
    148: (129, 257, 385, 577, 705, 833, 897, 1153, 1409),
    152: (129, 257, 385, 577, 705, 833, 961, 1217, 1473),
    208: (193, 385, 577, 833, 1025, 1153, 1281, 1665, 2049),
    212: (193, 385, 577, 833, 1025, 1217, 1345, 1665, 2049),
    216: (193, 385, 577, 833, 1025, 1217, 1345, 1729, 2113),
}
H4096_SPLITS_208 = {
    1: 64,
    193: 32,
    385: 22,
    577: 16,
    833: 13,
    1025: 11,
    1153: 10,
    1281: 16,
    1665: 13,
    2049: 16,
}
# The three BF16-output programs by the regime the host rule names (the loader's
# route key): "tailpipe_wsmem" at WEIGHT_SMEM_MAX_SPLITS (16) splits or fewer
# when the tasks run in WEIGHT_SMEM_MIN_WAVES (5) waves of the grid or more,
# "tailpipe" at the other counts up to TAIL_PIPE_MAX_SPLITS (22: 22, 16 and the
# two-wave counts below them), "bf16" at 32 and 64; the split rule yields no
# count between 22 and 32. All three write bitwise the same output.
BF16_PROGRAM = "cake_mega_mhc_f24c2f3f98bfe3356398"
TAILPIPE_PROGRAM = "cake_mega_mhc_c549b064530c3765f9ba"
WSMEM_PROGRAM = "cake_mega_mhc_41024b807c02040677b3"
TAIL_PROGRAMS = {
    "bf16": BF16_PROGRAM,
    "tailpipe": TAILPIPE_PROGRAM,
    "tailpipe_wsmem": WSMEM_PROGRAM,
}
# The quantized variants' programs (hidden 4096) by the regime the host rule
# mhc_infer_variant_program names: the plain variant ("bf16", every listed
# architecture) and its tail forms ("tailpipe" / "tailpipe_wsmem", the production
# programs' tail forms with the variant's quantization, VARIANT_TAIL_ARCHES alone).
# A tail form writes bitwise the plain variant's outputs at the same split count.
VARIANT_PROGRAMS = {
    "fp8_group128": {
        "bf16": "cake_mega_mhc_a940c94296afad2b2497",
        "tailpipe": "cake_mega_mhc_ed3f407e5a68eb65478e",
        "tailpipe_wsmem": "cake_mega_mhc_892d774130f06423fd7c",
    },
    "nvfp4": {
        "bf16": "cake_mega_mhc_1c61fed11caf93d1b762",
        "tailpipe": "cake_mega_mhc_5018692c75652529e6fd",
        "tailpipe_wsmem": "cake_mega_mhc_12e5bcc41e7940e7d511",
    },
}
VARIANT_TAIL_ARCHES = ["sm_107a"]
# The FP8 group-128 programs of the UE8M0 scale mode (fp8_sf_ue8m0=True) on
# sm_107a, one per regime: the plain variant's programs with the scale's
# reciprocal an exact power of two compiled in. The raw mode, a selection
# without the mode and the other architectures take VARIANT_PROGRAMS.
UE8M0_PROGRAMS = {
    "bf16": "cake_mega_mhc_91c9979888321fdbf94f",
    "tailpipe": "cake_mega_mhc_9823742e18a9ffa6515b",
    "tailpipe_wsmem": "cake_mega_mhc_5c5eaa3072b0d4f69ccd",
}
UE8M0_ARCHES = ["sm_107a"]


def _variant_program(layout, regime, arch, ue8m0=True):
    """The program a plan of the quantized ``layout`` launches in ``regime`` on
    ``arch`` (``ue8m0``: the FP8 scale mode, True by default)."""
    if layout == "fp8_group128" and ue8m0 and arch in UE8M0_ARCHES:
        return UE8M0_PROGRAMS[regime]
    return VARIANT_PROGRAMS[layout][regime]


# The first token count at which the rule leaves the BF16 program for the
# tailpipe program (the split count's 32 -> 22 point) and the first at which it
# leaves the tailpipe program for its weight-staging twin (16 splits over five
# task waves of the full grid: the first m-block count with
# ceil(16 m / num_sms) >= 5), per SM count (two switches each, never back), and
# the token counts of the producer's own rules test, all as that test spells them.
H4096_PROGRAM_SWITCH = {148: 257, 152: 257, 208: 385, 212: 385, 216: 385}
H4096_WSMEM_SWITCH = {148: 2369, 152: 2433, 208: 3329, 212: 3393, 216: 3457}
# The first token count at which the quantized variants take 8 splits: the
# production count's tasks in seven waves of the full grid (the first m-block
# count with ceil(16 m / num_sms) >= 7), past the weight-staging switch.
H4096_VARIANT_SWITCH = {148: 3521, 152: 3649, 208: 4993, 212: 5057, 216: 5185}
H4096_RULE_TOKENS = (
    *(1, 2, 3, 4, 5, 7, 63, 64, 65, 127, 128, 129, 192, 193, 200, 255, 256, 257),
    *(320, 321, 384, 385, 448, 449, 512, 513, 576, 577, 832, 833, 1000, 1024),
    *(1025, 1153, 1216, 1280, 1281, 1665, 2048, 2049, 2496, 2497, 3520, 3521),
    *(3648, 3649, 4096, 4992, 4993, 5056, 5057, 5184, 5185, 8192, 16263, 65536),
    1 << 20,
)


def _bf16_routes(regime, arch="sm_107a"):
    """The BF16-output routes of hidden 4096 keyed on ``regime`` on ``arch``."""
    return [
        route
        for route in _runtime.ROUTES
        if route["hidden"] == 4096
        and route.get("sf_layout") == "bf16"
        and _runtime.route_regime(route) == regime
        and arch in _runtime.record_arches(_runtime.PROGRAMS[route["program"]])
    ]


def _deepgemm_splits(tokens, num_sms):
    """The one-wave count the rule starts from: the fewest splits reaching the
    shortest longest task, never below 16."""
    m_blocks = (tokens + 63) // 64
    max_splits = min(64, max(16, num_sms // m_blocks))
    longest = (64 + max_splits - 1) // max_splits
    return (64 + longest - 1) // longest


@pytest.mark.parametrize("num_sms", H4096_SM_COUNTS)
def test_h4096_host_rules(num_sms):
    """The split and grid rules of the BF16-output program: the switch points of
    its rule on every SM count it ships for (each the first token count of its
    count, one token less keeps the previous one), the count's bounds (at least
    ten, never above the deepgemm count of at least 16; a smaller count only
    where the tokens fit two tail rounds of the grid, its tasks in two waves),
    a resident grid, the scratch size, and the deterministic mode's 16; the
    quantized variants' count (8 from seven task waves of the production count on)."""
    points = _runtime.mhc_infer_switch_points(num_sms)
    assert points == H4096_SWITCH_POINTS[num_sms]
    assert points == _runtime.mhc_infer_switch_points(num_sms, max_tokens=1 << 14)
    tokens = sorted(
        {
            1,
            2,
            63,
            64,
            65,
            *(p + d for p in points for d in (-1, 0)),
            *(H4096_VARIANT_SWITCH[num_sms] + d for d in (-1, 0)),
            4096,
            8192,
            1 << 20,
        }
    )
    for t in tokens:
        splits = _runtime.mhc_infer_num_splits(t, num_sms)
        legacy = _deepgemm_splits(t, num_sms)
        assert 10 <= splits <= legacy <= 64
        assert _runtime.mhc_infer_num_splits(t, num_sms, deterministic=True) == 16
        ctas = _runtime.mhc_infer_num_ctas(t, num_sms, splits)
        m_blocks = (t + 63) // 64
        assert min(num_sms, m_blocks) <= ctas <= num_sms
        assert ctas == min(num_sms, max(m_blocks * splits, (t + 3) // 4))
        if splits != legacy:
            assert m_blocks * splits <= 2 * ctas and m_blocks * 64 <= 12 * ctas
        assert _runtime.mhc_infer_scratch_floats(t, splits) == m_blocks * splits * 1664
    for t in points:
        assert t % 64 == 1
        assert _runtime.mhc_infer_num_splits(
            t, num_sms
        ) != _runtime.mhc_infer_num_splits(t - 1, num_sms)
    assert _runtime.mhc_infer_num_splits(points[-1], num_sms) == 16
    assert _runtime.mhc_infer_num_splits(1 << 20, num_sms) == 16
    assert _runtime.mhc_infer_num_splits(1, num_sms) == 64
    assert _runtime.mhc_infer_num_ctas(1, num_sms, 64) == 64
    assert _runtime.mhc_infer_num_ctas(4096, num_sms, 16) == num_sms
    if num_sms == 208:
        table = {t: _runtime.mhc_infer_num_splits(t, num_sms) for t in H4096_SPLITS_208}
        assert table == H4096_SPLITS_208
    # The quantized variants' count: the production count, except 8 where the
    # BF16 output runs the weight-staging program and the tasks run in at least
    # seven waves of the grid; 16 when deterministic. Its one switch of its own
    # is that wave count's first token count (H4096_VARIANT_SWITCH).
    for t in tokens:
        splits = _runtime.mhc_infer_num_splits(t, num_sms)
        variant = _runtime.mhc_infer_variant_num_splits(t, num_sms)
        grid = _runtime.mhc_infer_num_ctas(t, num_sms, splits)
        waves = ((t + 63) // 64 * splits + grid - 1) // grid
        wsmem = _runtime.mhc_infer_tail_program(t, num_sms) == "tailpipe_wsmem"
        plain = wsmem and waves >= _runtime.VARIANT_PLAIN_MIN_WAVES
        assert variant == (_runtime.VARIANT_WSMEM_REGIME_SPLITS if plain else splits)
        assert 8 <= variant <= splits and (variant != splits) == plain
        assert (
            _runtime.mhc_infer_variant_num_splits(t, num_sms, deterministic=True) == 16
        )
    assert _runtime.VARIANT_WSMEM_REGIME_SPLITS == 8
    assert _runtime.VARIANT_PLAIN_MIN_WAVES == 7
    switch = H4096_VARIANT_SWITCH[num_sms]
    assert switch > H4096_WSMEM_SWITCH[num_sms]
    assert _runtime.mhc_infer_variant_num_splits(switch - 1, num_sms) == 16
    assert _runtime.mhc_infer_variant_num_splits(switch, num_sms) == 8
    assert _runtime.mhc_infer_variant_num_splits(4096, num_sms) == (
        8 if num_sms in (148, 152) else 16
    )
    assert _runtime.mhc_infer_variant_num_splits(8192, num_sms) == 8


def _expected_h4096_program(t, num_sms, deterministic=False):
    """The host rule restated as the producer's rules test spells it: the
    weight-staging twin at <= 16 splits over >= 5 task waves of the grid, the
    tailpipe program at the other counts <= 22, the BF16 program at 32 and 64."""
    splits = _runtime.mhc_infer_num_splits(t, num_sms, deterministic)
    grid = _runtime.mhc_infer_num_ctas(t, num_sms, splits)
    waves = -(-((t + 63) // 64 * splits) // grid)
    if splits <= 16 and waves >= 5:
        return "tailpipe_wsmem"
    return "tailpipe" if splits <= 22 else "bf16"


@pytest.mark.parametrize("num_sms", H4096_SM_COUNTS)
def test_h4096_tail_program_follows_the_split_count(num_sms):
    """The BF16-output route launches exactly one program for every (T, SM count):
    the host rule names the regime ("tailpipe_wsmem" at the split counts up to
    WEIGHT_SMEM_MAX_SPLITS over WEIGHT_SMEM_MIN_WAVES task waves or more,
    "tailpipe" at the other counts up to TAIL_PIPE_MAX_SPLITS, "bf16" at 32 and
    64; the counts 23..31 are unreachable by the rule, asserted here), every
    regime the rule names has exactly one route and no other route matches, and
    a selection without the regime is refused; two program switches per SM
    count, at the token counts the producer's rules test spells (the split
    count's 32 -> 22 point, then the five-wave point at a constant 16 splits);
    the deterministic 16 is the tailpipe program's below the wave switch and
    the twin's from it."""
    assert _runtime.TAIL_PIPE_MAX_SPLITS == 22
    assert _runtime.WEIGHT_SMEM_MAX_SPLITS == 16
    assert _runtime.WEIGHT_SMEM_MIN_WAVES == 5
    by_regime = {
        regime: [r for r in _runtime.ROUTES if r["program"] == program]
        for regime, program in TAIL_PROGRAMS.items()
    }
    for regime, routes in by_regime.items():
        assert len(routes) == 1 and routes[0]["num_splits"] == "runtime", regime
        assert _runtime.route_regime(routes[0]) == routes[0]["regime"] == regime
    keyed = [r for r in _runtime.ROUTES if _runtime.route_regime(r) is not None]
    assert [r for r in keyed if r["sf_layout"] == "bf16"] == [
        routes[0] for routes in by_regime.values()
    ]
    ue8m0 = [r for r in keyed if "fp8_sf_ue8m0" in r]
    assert [r["sf_layout"] for r in ue8m0] == ["fp8_group128"] * len(UE8M0_PROGRAMS)
    assert all(r["fp8_sf_ue8m0"] is True for r in ue8m0)
    assert all(r["num_splits"] == "runtime" for r in ue8m0)
    assert {_runtime.route_regime(r): r["program"] for r in ue8m0} == UE8M0_PROGRAMS
    assert all("fp8_sf_ue8m0" not in r for r in _runtime.ROUTES if r not in ue8m0)
    for layout, programs in VARIANT_PROGRAMS.items():
        routes = [r for r in keyed if r["sf_layout"] == layout and r not in ue8m0]
        assert len(routes) == len(programs), layout
        assert {_runtime.route_regime(r): r["program"] for r in routes} == programs
        assert all(r["num_splits"] == "runtime" for r in routes), layout
    assert {r["sf_layout"] for r in keyed} == {"bf16", *VARIANT_PROGRAMS}, (
        "only the hidden-4096 BF16-output and quantized-variant routes are keyed "
        "on a regime"
    )
    assert all(
        _runtime.route_regime(r) is None for r in _runtime.ROUTES if r not in keyed
    )
    points = _runtime.mhc_infer_switch_points(num_sms)
    program_switch = H4096_PROGRAM_SWITCH[num_sms]
    wsmem_switch = H4096_WSMEM_SWITCH[num_sms]
    tokens = sorted(
        {
            *H4096_TOKENS,
            *H4096_RULE_TOKENS,
            *(p + d for p in points for d in (-1, 0)),
            *(program_switch + d for d in (-1, 0)),
            *(wsmem_switch + d for d in (-1, 0, 63, 64)),
            2 * wsmem_switch,
            1 << 14,
        }
    )
    image, regimes = set(), set()
    for t in tokens:
        splits = _runtime.mhc_infer_num_splits(t, num_sms)
        image.add(splits)
        regime = _runtime.mhc_infer_tail_program(t, num_sms)
        regimes.add(regime)
        assert regime == _expected_h4096_program(t, num_sms)
        assert regime == (
            "tailpipe_wsmem"
            if t >= wsmem_switch
            else "tailpipe"
            if t >= program_switch
            else "bf16"
        )
        if regime == "tailpipe_wsmem":
            assert splits == 16
            assert _runtime.mhc_infer_num_ctas(t, num_sms, splits) == num_sms
        fixed = _runtime.mhc_infer_tail_program(t, num_sms, deterministic=True)
        assert fixed == _expected_h4096_program(t, num_sms, deterministic=True)
        assert fixed == ("tailpipe_wsmem" if t >= wsmem_switch else "tailpipe")
        routes = _bf16_routes(regime)
        assert [r["program"] for r in routes] == [TAIL_PROGRAMS[regime]], (t, splits)
        selected, defines = _runtime.select_program(
            4096, splits, False, False, "bf16", "sm_107a", regime=regime
        )
        assert selected == TAIL_PROGRAMS[regime] and defines == {}
        with pytest.raises(NotImplementedError, match="regime=None"):
            _runtime.select_program(4096, splits, False, False, "bf16", "sm_107a")
    # The split count is constant between its switch points, so the counts seen
    # at the points and their predecessors are its whole image on this SM count;
    # the regimes the rule names are exactly the routes' keys, each reached.
    assert image <= {*range(10, 23), 32, 64} and {16, 32, 64} <= image
    assert not image & set(range(23, 32))
    assert regimes == set(TAIL_PROGRAMS) == {r["regime"] for r in keyed}
    with pytest.raises(NotImplementedError, match="regime='other'"):
        _runtime.select_program(
            4096, 16, False, False, "bf16", "sm_107a", regime="other"
        )
    # The program changes only at m-block boundaries: scanning them to 2^14
    # tokens finds exactly the two switches, each the first token count of its
    # program; the wave switch lies beyond the last split-count switch point, at
    # the first m-block count with ceil(16 m / num_sms) >= 5.
    switches = [
        t
        for t in range(65, (1 << 14) + 1, 64)
        if _runtime.mhc_infer_tail_program(t, num_sms)
        != _runtime.mhc_infer_tail_program(t - 1, num_sms)
    ]
    assert switches == [program_switch, wsmem_switch]
    assert points[-1] < wsmem_switch and wsmem_switch % 64 == 1
    assert (wsmem_switch + 63) // 64 == (4 * num_sms) // 16 + 1
    assert _runtime.mhc_infer_tail_program(1, num_sms) == "bf16"
    assert _runtime.mhc_infer_tail_program(1 << 20, num_sms) == "tailpipe_wsmem"
    assert _runtime.mhc_infer_tail_program(1, num_sms, deterministic=True) == "tailpipe"
    matrix = _runtime._support_matrix()
    assert all(
        f"({regime}: {program})" in matrix for regime, program in TAIL_PROGRAMS.items()
    )
    if num_sms == 208:
        spots = (256, 385, 512, 3328, 3329, 4096, 8192)
        assert [_runtime.mhc_infer_tail_program(t, 208) for t in spots] == [
            "bf16",
            "tailpipe",
            "tailpipe",
            "tailpipe",
            "tailpipe_wsmem",
            "tailpipe_wsmem",
            "tailpipe_wsmem",
        ]


def _expected_variant_program(t, num_sms, layout, deterministic=False):
    """The quantized variants' program on an architecture that ships their tail
    forms, restated as the producer's rules test spells it: the production
    program's regime, except the plain variant ("bf16") where that regime is
    "tailpipe" and the tasks fit one wave of the grid ("fp8_group128" below 16
    splits, "nvfp4" at 16 or more), and in deterministic mode below the
    weight-staging regime."""
    regime = _expected_h4096_program(t, num_sms, deterministic)
    if regime != "tailpipe":
        return regime
    if deterministic:
        return "bf16"
    splits = _runtime.mhc_infer_num_splits(t, num_sms)
    grid = _runtime.mhc_infer_num_ctas(t, num_sms, splits)
    one_wave = (t + 63) // 64 * splits <= grid
    if one_wave and (splits < 16 if layout == "fp8_group128" else splits >= 16):
        return "bf16"
    return "tailpipe"


@pytest.mark.parametrize("num_sms", H4096_SM_COUNTS)
def test_h4096_variant_program_follows_the_rule(num_sms):
    """The quantized variants' program: the plain variant where the architecture
    ships no tail forms (``tail_variants`` False: sm_100a / sm_103a), else the
    regime ``mhc_infer_variant_program`` names, which selects that layout's route
    on sm_107a at the variant's own split count; every regime is reached on every
    SM count, and on 208 SMs the plain variant stays where the tail form measured
    slower (one-wave FP8 below 16 splits, one-wave NVFP4 at 16 or more)."""
    tokens = sorted(
        {
            *H4096_TOKENS,
            *H4096_RULE_TOKENS,
            *(H4096_PROGRAM_SWITCH[num_sms] + d for d in (-1, 0)),
            *(H4096_WSMEM_SWITCH[num_sms] + d for d in (-1, 0)),
            *(H4096_VARIANT_SWITCH[num_sms] + d for d in (-1, 0)),
        }
    )
    for layout, programs in VARIANT_PROGRAMS.items():
        seen = set()
        for t in tokens:
            for deterministic in (False, True):
                rule = _runtime.mhc_infer_variant_program
                assert rule(t, num_sms, layout, deterministic) == "bf16"
                regime = rule(t, num_sms, layout, deterministic, True)
                expected = _expected_variant_program(t, num_sms, layout, deterministic)
                assert regime == expected, (layout, t, deterministic)
                seen.add(regime)
                splits = _runtime.mhc_infer_variant_num_splits(
                    t, num_sms, deterministic
                )
                assert _runtime.select_program(
                    4096, splits, False, False, layout, "sm_107a", regime=regime
                ) == (programs[regime], {})
                if layout == "fp8_group128":
                    for ue8m0 in (True, False):
                        assert _runtime.select_program(
                            4096,
                            splits,
                            False,
                            False,
                            layout,
                            "sm_107a",
                            regime=regime,
                            fp8_sf_ue8m0=ue8m0,
                        ) == (_variant_program(layout, regime, "sm_107a", ue8m0), {})
        assert seen == set(programs), (layout, seen)
    if num_sms == 208:
        spots = (1, 384, 385, 832, 833, 1280, 1281, 3328, 3329, 8192)
        table = {
            layout: [
                _runtime.mhc_infer_variant_program(t, 208, layout, False, True)
                for t in spots
            ]
            for layout in VARIANT_PROGRAMS
        }
        assert table == {
            "fp8_group128": [
                "bf16",
                "bf16",
                "tailpipe",
                "tailpipe",
                "bf16",
                "bf16",
                "tailpipe",
                "tailpipe",
                "tailpipe_wsmem",
                "tailpipe_wsmem",
            ],
            "nvfp4": [
                "bf16",
                "bf16",
                "bf16",
                "bf16",
                "tailpipe",
                "tailpipe",
                "tailpipe",
                "tailpipe",
                "tailpipe_wsmem",
                "tailpipe_wsmem",
            ],
        }


def test_split_fn_is_the_exact_three_way_split():
    fn = torch.randn(24, 4 * 4096, generator=torch.Generator().manual_seed(7)) * 0.01
    fnp = _runtime.split_fn(fn)
    assert fnp.shape == (3, 32, 4 * 4096) and fnp.dtype == torch.bfloat16
    assert not fnp[:, 24:].any()
    pieces = fnp[:, :24].double()
    assert torch.equal(fnp[0, :24], fn.bfloat16())
    assert torch.equal(fnp[1, :24], (fn - fn.bfloat16().float()).bfloat16())
    residual = (fn.double() - pieces.sum(0)).abs()
    assert bool((residual <= fn.double().abs() * 2.0**-27 + 1e-30).all())
    held = torch.full_like(fnp, float("nan"))
    assert _runtime.split_fn(fn, out=held) is held and torch.equal(held, fnp)


def test_h4096_unsupported_combinations_name_the_support_matrix():
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        _runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))
    values = inputs_h4096(64)
    with pytest.raises(NotImplementedError, match="hidden=4096"):
        prepare_mega_mhc(**values, sf_layout="col")
    shifted = dict(values, shifted_prev_mix=torch.full((64, 4, 1), 0.25, device="cuda"))
    with pytest.raises(NotImplementedError, match="normal mode"):
        prepare_mega_mhc(**shifted, sf_layout="bf16")


@pytest.mark.parametrize("tokens", H4096_TOKENS)
@pytest.mark.parametrize("layout", H4096_LAYOUTS)
def test_h4096_values_and_replay(tokens, layout):
    _skip_unless_exported(4096, layout)
    values = inputs_h4096(tokens)
    kw = layout_kwargs(layout, values)
    plan = prepare_mega_mhc(**values, sf_layout=layout, **kw)
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    rule = (
        _runtime.mhc_infer_variant_num_splits
        if layout in _runtime.QUANT_LAYOUTS
        else _runtime.mhc_infer_num_splits
    )
    assert plan.config["num_splits"] == rule(tokens, sms)
    assert plan.config["num_launch_sms"] == _runtime.mhc_infer_num_ctas(
        tokens, sms, plan.config["num_splits"]
    )
    if layout == "bf16":
        regime = _runtime.mhc_infer_tail_program(tokens, sms)
        assert plan.config["regime"] == regime
        assert plan.program == plan.config["program"] == TAIL_PROGRAMS[regime]
    else:
        assert plan.program not in TAIL_PROGRAMS.values()
        regime = _runtime.mhc_infer_variant_program(
            tokens, sms, layout, False, _arch in VARIANT_TAIL_ARCHES
        )
        assert plan.config["regime"] == regime
        assert plan.program == plan.config["program"]
        assert plan.program == _variant_program(layout, regime, _arch)
    assert set(plan.outputs) == {*BF16_OUTPUTS, *QUANT_OUTPUTS[layout]}
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    first = None
    for _ in range(3):
        with torch.cuda.stream(stream):
            poison(plan.outputs)
            before = torch.cuda.memory_allocated()
            plan.run()
            assert torch.cuda.memory_allocated() == before
        stream.synchronize()
        check_h4096_layout(plan.outputs, values, layout, **kw)
        current = {name: value.clone() for name, value in plan.outputs.items()}
        if first is None:
            first = current
        for name, value in current.items():
            assert bitwise_equal(value, first[name]), f"{name} differs between launches"
    stream.wait_stream(torch.cuda.current_stream())


@pytest.mark.parametrize("tokens", (1, 200, 1025))
@pytest.mark.parametrize("layout", H4096_LAYOUTS)
def test_h4096_graph_replay_changed_inputs_and_update_fn(tokens, layout):
    """Six replays over inputs rewritten in place, the fourth after ``update_fn``
    on a fresh projection; the replacement keeps fn's 0.01 scale, so the FP32
    coefficients stay in the regime of the other tests' inputs."""
    _skip_unless_exported(4096, layout)
    values = inputs_h4096(tokens)
    kw = layout_kwargs(layout, values)
    plan = prepare_mega_mhc(**values, sf_layout=layout, **kw)
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    if layout == "bf16":
        # 1 and 200 tokens replay the BF16 program, 1025 the tailpipe program.
        assert (
            plan.program == TAIL_PROGRAMS[_runtime.mhc_infer_tail_program(tokens, sms)]
        )
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
        plan.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    stream.synchronize()
    previous = plan.launch_epochs.cpu().tolist()
    active = plan.config["num_launch_sms"]
    for repeat in range(6):
        with torch.cuda.stream(stream):
            values["x"].add_(0.015625)
            values["post_mix"].mul_(0.5)
            if repeat == 3:
                fresh = torch.Generator(device="cuda").manual_seed(2027)
                values["fn"].copy_(
                    torch.randn(values["fn"].shape, device="cuda", generator=fresh)
                    * 0.01
                )
                plan.update_fn()
            poison(plan.outputs)
            if repeat == 4:
                plan.run()
            else:
                graph.replay()
        stream.synchronize()
        check_h4096_layout(
            plan.outputs, values, layout, label=f"repeat {repeat} ", **kw
        )
        current = plan.launch_epochs.cpu().tolist()
        assert len(current) == plan.config["num_sms"]
        assert all(
            after == before + 1
            for before, after in zip(previous[:active], current[:active], strict=True)
        )
        assert current[active:] == previous[active:]
        previous = current
    stream.wait_stream(torch.cuda.current_stream())


def test_h4096_caller_held_fnp_and_deterministic_splits():
    _skip_unless_exported(4096, "bf16")
    values = inputs_h4096(384)
    fnp = _runtime.split_fn(values["fn"])
    plan = prepare_mega_mhc(**values, sf_layout="bf16", fnp=fnp, deterministic=True)
    assert plan.fnp is fnp and plan.config["num_splits"] == 16
    plan.run()
    torch.cuda.synchronize()
    check_h4096(plan.outputs, values)
    replacement = values["fn"] * 0.5
    plan.update_fn(replacement)
    assert torch.equal(fnp, _runtime.split_fn(replacement))
    plan.run()
    torch.cuda.synchronize()
    check_h4096(plan.outputs, dict(values, fn=replacement))


def test_h4096_tailpipe_programs_are_bitwise_the_bf16_program(monkeypatch):
    """At every token count the rule dispatches to the tailpipe program or its
    weight-staging twin, a plan forced onto the BF16 program (the route read as
    the BF16 program's) writes bitwise the same four outputs over the same
    inputs; the forced plan launches the same grid and split count, so the two
    differ in the program alone. The candidates reach both tailpipe regimes on
    the device (3329 is the twin's first token count on 208 SMs)."""
    _skip_unless_exported(4096, "bf16")
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    candidates = (385, 577, 1025, 1281, 2048, 3329, 4096, 8192)
    regimes = {t: _runtime.mhc_infer_tail_program(t, sms) for t in candidates}
    tokens = [t for t in candidates if regimes[t] != "bf16"]
    assert {regimes[t] for t in tokens} == {"tailpipe", "tailpipe_wsmem"}, regimes
    select = _runtime.select_program

    def forced(
        hidden,
        splits,
        shifted,
        shared_sf,
        sf_layout,
        arch,
        regime=None,
        fp8_sf_ue8m0=None,
    ):
        assert (hidden, sf_layout) == (4096, "bf16") and regime != "bf16"
        return select(
            hidden,
            splits,
            shifted,
            shared_sf,
            sf_layout,
            arch,
            regime="bf16",
            fp8_sf_ue8m0=fp8_sf_ue8m0,
        )

    for t in tokens:
        values = inputs_h4096(t)
        dispatched = prepare_mega_mhc(**values, sf_layout="bf16")
        assert dispatched.program == TAIL_PROGRAMS[regimes[t]]
        assert dispatched.config["regime"] == regimes[t]
        with monkeypatch.context() as patch:
            patch.setattr(_runtime, "select_program", forced)
            base = prepare_mega_mhc(**values, sf_layout="bf16")
        assert base.program == BF16_PROGRAM
        assert base.config["num_splits"] == dispatched.config["num_splits"]
        assert base.config["num_launch_sms"] == dispatched.config["num_launch_sms"]
        for plan in (dispatched, base):
            poison(plan.outputs)
            plan.run()
        torch.cuda.synchronize()
        check_h4096(dispatched.outputs, values)
        for name in BF16_OUTPUTS:
            assert bitwise_equal(dispatched.outputs[name], base.outputs[name]), (
                t,
                name,
            )


@pytest.mark.parametrize("layout", ("fp8_group128", "nvfp4"))
def test_h4096_variant_tail_forms_are_bitwise_the_plain_variant(monkeypatch, layout):
    """At every token count the variants' rule dispatches a quantized layout to a
    tail form, a plan forced onto the plain variant program (the route read as
    the plain variant's) writes bitwise the same outputs over the same inputs; the
    forced plan launches the same grid and split count, so the two differ in the
    program alone. The candidates reach both tail forms on the device."""
    _skip_unless_exported(4096, layout)
    arch, sms = _runtime.device_facts(torch.cuda.current_device())
    if arch not in VARIANT_TAIL_ARCHES:
        pytest.skip(f"the variants' tail forms ship for {VARIANT_TAIL_ARCHES} alone")
    candidates = (385, 577, 833, 1025, 1281, 2048, 3329, 4096, 8192)
    rule = _runtime.mhc_infer_variant_program
    regimes = {t: rule(t, sms, layout, False, True) for t in candidates}
    tokens = [t for t in candidates if regimes[t] != "bf16"]
    assert {regimes[t] for t in tokens} == {"tailpipe", "tailpipe_wsmem"}, regimes
    select = _runtime.select_program

    def forced(
        hidden,
        splits,
        shifted,
        shared_sf,
        sf_layout,
        arch,
        regime=None,
        fp8_sf_ue8m0=None,
    ):
        assert (hidden, sf_layout) == (4096, layout) and regime != "bf16"
        return select(
            hidden,
            splits,
            shifted,
            shared_sf,
            sf_layout,
            arch,
            regime="bf16",
            fp8_sf_ue8m0=fp8_sf_ue8m0,
        )

    for t in tokens:
        values = inputs_h4096(t)
        kw = layout_kwargs(layout, values)
        dispatched = prepare_mega_mhc(**values, sf_layout=layout, **kw)
        assert dispatched.program == _variant_program(layout, regimes[t], arch)
        assert dispatched.config["regime"] == regimes[t]
        with monkeypatch.context() as patch:
            patch.setattr(_runtime, "select_program", forced)
            base = prepare_mega_mhc(**values, sf_layout=layout, **kw)
        assert base.program == _variant_program(layout, "bf16", arch)
        assert base.config["num_splits"] == dispatched.config["num_splits"]
        assert base.config["num_launch_sms"] == dispatched.config["num_launch_sms"]
        for plan in (dispatched, base):
            poison(plan.outputs)
            plan.run()
        torch.cuda.synchronize()
        check_h4096_layout(dispatched.outputs, values, layout, **kw)
        for name in (*BF16_OUTPUTS, *QUANT_OUTPUTS[layout]):
            assert bitwise_equal(dispatched.outputs[name], base.outputs[name]), (
                t,
                name,
            )


# ------------------------------------------------- the FP8 group-128 output variant (hidden 4096)


def test_h4096_quant_routes_select_their_programs():
    """The hidden-4096 layouts dispatch on (hidden, output layout, architecture,
    regime): each exported layout names its own programs (a quantized layout its
    plain variant on every listed architecture and, on sm_107a alone, its two
    tail forms, one per regime), an unexported pair is refused with the support
    matrix, and the support matrix names every exported pair."""
    bf16 = _runtime.select_program(
        4096, 32, False, False, "bf16", "sm_107a", regime="bf16"
    )
    tailpipe = _runtime.select_program(
        4096, 16, False, False, "bf16", "sm_107a", regime="tailpipe"
    )[0]
    wsmem = _runtime.select_program(
        4096, 16, False, False, "bf16", "sm_107a", regime="tailpipe_wsmem"
    )[0]
    assert (bf16[0], tailpipe, wsmem) == (BF16_PROGRAM, TAILPIPE_PROGRAM, WSMEM_PROGRAM)
    bf16 = bf16[0]
    assert _runtime.supported("sm_107a", 4096, "bf16")
    base = _runtime.PROGRAMS[bf16]
    for program in (tailpipe, wsmem):
        record = _runtime.PROGRAMS[program]
        assert record["hidden"] == base["hidden"] == 4096
        assert record["arg_plan"] == base["arg_plan"]
        assert record["launch"]["block"] == base["launch"]["block"]
        assert record["launch"]["cooperative"] == base["launch"]["cooperative"]
        assert record["launch"]["use_pdl"] == base["launch"]["use_pdl"]
    assert _runtime.PROGRAMS[tailpipe]["launch"] == base["launch"]
    # The twin stages the RMSNorm weight (4096 BF16 values) in shared memory.
    assert (
        _runtime.PROGRAMS[wsmem]["launch"]["dynamic_smem_bytes"]
        == base["launch"]["dynamic_smem_bytes"] + 4096 * 2
    )
    matrix = _runtime._support_matrix()
    assert f"hidden=4096 output=bf16 on {H4096_ARCHES}" in matrix
    for layout in ("fp8_group128", "nvfp4"):
        exported = _runtime.supported("sm_107a", 4096, layout)
        assert not _runtime.supported("sm_107a", 5120, layout)
        assert _runtime.supported("sm_100a", 4096, layout) == exported
        assert _runtime.supported("sm_103a", 4096, layout) == exported
        assert not _runtime.supported("sm_120a", 4096, layout)
        if not exported:
            assert f"output={layout}" not in matrix
            with pytest.raises(NotImplementedError, match="(?i)exported:"):
                _runtime.select_program(4096, 16, False, False, layout, "sm_107a")
            continue
        programs = VARIANT_PROGRAMS[layout]
        program, defines = _runtime.select_program(
            4096, 16, False, False, layout, "sm_107a", regime="bf16"
        )
        assert program == programs["bf16"]
        assert program not in (bf16, tailpipe, wsmem) and defines == {}
        assert len(set(programs.values())) == 3
        assert not set(programs.values()) & {bf16, tailpipe, wsmem}
        for splits, regime in (
            (64, "bf16"),
            (16, "tailpipe"),
            (16, "tailpipe_wsmem"),
        ):
            same = _runtime.select_program(
                4096, splits, False, False, layout, "sm_107a", regime=regime
            )
            assert same == (programs[regime], {}), "the regime selects the program"
        with pytest.raises(NotImplementedError, match="regime=None"):
            _runtime.select_program(4096, 32, False, False, layout, "sm_107a")
        for arch in ("sm_100a", "sm_103a"):
            assert _runtime.select_program(
                4096, 16, False, False, layout, arch, regime="bf16"
            ) == (program, {})
            for regime in ("tailpipe", "tailpipe_wsmem"):
                with pytest.raises(NotImplementedError, match="(?i)exported:"):
                    _runtime.select_program(
                        4096, 16, False, False, layout, arch, regime=regime
                    )
        record = _runtime.PROGRAMS[program]
        assert record["hidden"] == 4096 and record["arches"] == H4096_ARCHES
        routes = [r for r in _runtime.ROUTES if r["program"] == program]
        assert [r["sf_layout"] for r in routes] == [layout]
        assert f"hidden=4096 output={layout} on {H4096_ARCHES}" in matrix
        for regime in ("tailpipe", "tailpipe_wsmem"):
            tail = _runtime.PROGRAMS[programs[regime]]
            assert tail["hidden"] == 4096 and tail["arches"] == VARIANT_TAIL_ARCHES
            assert tail["arg_plan"] == record["arg_plan"]
            assert dict(tail["launch"], dynamic_smem_bytes=0) == dict(
                record["launch"], dynamic_smem_bytes=0
            )
            # A tail form takes the production program's dynamic SMEM of its form.
            production = _runtime.PROGRAMS[TAIL_PROGRAMS[regime]]["launch"]
            assert (
                tail["launch"]["dynamic_smem_bytes"] == production["dynamic_smem_bytes"]
            )
            routes = [r for r in _runtime.ROUTES if r["program"] == programs[regime]]
            assert [(r["sf_layout"], r["regime"]) for r in routes] == [(layout, regime)]
            assert (
                f"hidden=4096 output={layout} on {VARIANT_TAIL_ARCHES} "
                f"({regime}: {programs[regime]})" in matrix
            )
        names = [name for _kind, name in _runtime.record_arg_plan(record)]
        assert names[-3:] == ["grid_x", "grid_y", "grid_z"]
        assert set(QUANT_OUTPUTS[layout]) < set(names)


def test_h4096_every_program_ships_for_every_listed_architecture():
    """Each hidden-4096 record lists the same three architectures, sm_107a first,
    except the quantized variants' four tail forms, which list VARIANT_TAIL_ARCHES
    alone; the routes select the same BF16-output program and the same plain
    variant on each architecture, and a tail form on sm_107a alone; hidden 5120
    stays unexported on sm_107a."""
    h4096 = [
        name for name, record in _runtime.PROGRAMS.items() if record["hidden"] == 4096
    ]
    tails = {
        programs[regime]
        for programs in VARIANT_PROGRAMS.values()
        for regime in ("tailpipe", "tailpipe_wsmem")
    }
    pow2 = set(UE8M0_PROGRAMS.values())
    assert len(h4096) == 12 and len(tails) == 4 and tails < set(h4096)
    assert len(pow2) == 3 and pow2 < set(h4096) and not pow2 & tails
    for name in h4096:
        arches = (
            UE8M0_ARCHES
            if name in pow2
            else VARIANT_TAIL_ARCHES
            if name in tails
            else H4096_ARCHES
        )
        assert _runtime.record_arches(_runtime.PROGRAMS[name]) == arches, name
    for arch in H4096_ARCHES:
        for splits, regime, program in (
            (64, "bf16", BF16_PROGRAM),
            (16, "tailpipe", TAILPIPE_PROGRAM),
            (16, "tailpipe_wsmem", WSMEM_PROGRAM),
        ):
            assert _runtime.select_program(
                4096, splits, False, False, "bf16", arch, regime=regime
            ) == (program, {})
        for layout in ("fp8_group128", "nvfp4"):
            if _runtime.supported(arch, 4096, layout):
                program, defines = _runtime.select_program(
                    4096, 16, False, False, layout, arch, regime="bf16"
                )
                assert program not in (BF16_PROGRAM, TAILPIPE_PROGRAM, WSMEM_PROGRAM)
                assert program == VARIANT_PROGRAMS[layout]["bf16"]
                assert defines == {}
                for regime in ("tailpipe", "tailpipe_wsmem"):
                    key = (4096, 16, False, False, layout, arch)
                    if arch in VARIANT_TAIL_ARCHES:
                        assert _runtime.select_program(*key, regime=regime) == (
                            VARIANT_PROGRAMS[layout][regime],
                            {},
                        )
                    else:
                        with pytest.raises(NotImplementedError, match="(?i)exported:"):
                            _runtime.select_program(*key, regime=regime)
        assert _runtime.supported(arch, 5120, "col") == (arch != "sm_107a")
    assert _runtime.shipped_arches() == sorted(H4096_ARCHES)


def test_h4096_fp8_ue8m0_routes_select_the_pow2_programs():
    """The FP8 group-128 layout's UE8M0 scale mode on sm_107a takes programs of
    its own, one per regime (UE8M0_PROGRAMS); the raw mode, a selection without
    the mode and sm_100a / sm_103a take the plain variant's programs. Each UE8M0
    program's record is the plain variant's record of its regime (arg_plan,
    launch) with its own kernel on UE8M0_ARCHES alone, and its route is the
    plain variant's route of its regime keyed on fp8_sf_ue8m0=True."""
    programs = VARIANT_PROGRAMS["fp8_group128"]
    assert len(set(UE8M0_PROGRAMS.values())) == 3
    assert not set(UE8M0_PROGRAMS.values()) & set(programs.values())
    for regime, splits in (("bf16", 64), ("tailpipe", 16), ("tailpipe_wsmem", 16)):
        key = (4096, splits, False, False, "fp8_group128")
        for ue8m0 in (True, False, None):
            kw = {} if ue8m0 is None else {"fp8_sf_ue8m0": ue8m0}
            expected = UE8M0_PROGRAMS[regime] if ue8m0 else programs[regime]
            selected = _runtime.select_program(*key, "sm_107a", regime=regime, **kw)
            assert selected == (expected, {}), (regime, ue8m0)
        for arch in ("sm_100a", "sm_103a"):
            for ue8m0 in (True, False):
                if regime == "bf16":
                    assert _runtime.select_program(
                        *key, arch, regime=regime, fp8_sf_ue8m0=ue8m0
                    ) == (programs["bf16"], {})
                    continue
                with pytest.raises(NotImplementedError, match="(?i)exported:"):
                    _runtime.select_program(
                        *key, arch, regime=regime, fp8_sf_ue8m0=ue8m0
                    )
        pow2 = _runtime.PROGRAMS[UE8M0_PROGRAMS[regime]]
        plain = _runtime.PROGRAMS[programs[regime]]
        assert _runtime.record_arches(pow2) == UE8M0_ARCHES
        assert pow2["hidden"] == plain["hidden"] == 4096
        assert pow2["arg_plan"] == plain["arg_plan"]
        assert pow2["launch"] == plain["launch"]
        assert pow2["kernel_symbol"] != plain["kernel_symbol"]
        route = [r for r in _runtime.ROUTES if r["program"] == UE8M0_PROGRAMS[regime]]
        twin = [r for r in _runtime.ROUTES if r["program"] == programs[regime]]
        assert len(route) == len(twin) == 1, regime
        assert route[0]["fp8_sf_ue8m0"] is True
        rest = {k: v for k, v in route[0].items() if k != "fp8_sf_ue8m0"}
        assert dict(rest, program=None) == dict(twin[0], program=None)


def test_h4096_quant_keyword_rules():
    """``fp8_sf_ue8m0`` belongs to the FP8 layout and ``fp4_global_scale`` to the
    NVFP4 layout; a quantized layout of hidden 5120 is not exported."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    try:
        arch, _sms = _runtime.device_facts(torch.cuda.current_device())
    except RuntimeError as error:
        pytest.skip(str(error))
    values = inputs_h4096(64)
    with pytest.raises(ValueError, match="fp8_sf_ue8m0 selects"):
        prepare_mega_mhc(**values, sf_layout="bf16", fp8_sf_ue8m0=False)
    scale = torch.ones(1, device="cuda")
    with pytest.raises(ValueError, match="fp4_global_scale is the operand"):
        prepare_mega_mhc(**values, sf_layout="fp8_group128", fp4_global_scale=scale)
    with pytest.raises(ValueError, match="fp4_global_scale is the operand"):
        prepare_mega_mhc(**values, sf_layout="bf16", fp4_global_scale=scale)
    if _runtime.supported(arch, 4096, "nvfp4"):
        with pytest.raises(ValueError, match="requires fp4_global_scale"):
            prepare_mega_mhc(**values, sf_layout="nvfp4")
        with pytest.raises(ValueError, match="fp4_global_scale must be"):
            prepare_mega_mhc(
                **values, sf_layout="nvfp4", fp4_global_scale=scale.double()
            )
        with pytest.raises(ValueError, match="fp4_global_scale must be"):
            prepare_mega_mhc(**values, sf_layout="nvfp4", fp4_global_scale=scale.cpu())
        with pytest.raises(ValueError, match="fp8_sf_ue8m0 selects"):
            prepare_mega_mhc(
                **values, sf_layout="nvfp4", fp4_global_scale=scale, fp8_sf_ue8m0=False
            )
    else:
        with pytest.raises(NotImplementedError, match="(?i)exported:"):
            prepare_mega_mhc(**values, sf_layout="nvfp4", fp4_global_scale=scale)
    if _runtime.supported(arch, 5120, "col"):
        with pytest.raises(NotImplementedError, match="hidden=5120"):
            prepare_mega_mhc(**inputs(64, False), sf_layout="fp8_group128")


@pytest.mark.parametrize("tokens", H4096_TOKENS)
def test_h4096_fp8_raw_scales(tokens):
    """``fp8_sf_ue8m0=False`` keeps the exact ``absmax / 448`` scales: the four
    other outputs equal the power-of-two plan's bitwise, both quantizations are
    bitwise the pinned op in their mode, and the two scale tensors differ exactly
    where the exact scale is not a power of two."""
    _skip_unless_exported(4096, "fp8_group128")
    values = inputs_h4096(tokens)
    pow2 = prepare_mega_mhc(**values, sf_layout="fp8_group128")
    raw = prepare_mega_mhc(**values, sf_layout="fp8_group128", fp8_sf_ue8m0=False)
    pow2.run()
    raw.run()
    torch.cuda.synchronize()
    for name in BF16_OUTPUTS:
        assert bitwise_equal(pow2.outputs[name], raw.outputs[name]), name
    check_fp8(pow2.outputs, ue8m0=True)
    check_fp8(raw.outputs, ue8m0=False)
    exact = raw.outputs["y_fp8_sf"]
    is_pow2 = torch.frexp(exact.double())[0] == 0.5
    assert torch.equal(pow2.outputs["y_fp8_sf"] == exact, is_pow2)
    assert bool((pow2.outputs["y_fp8_sf"] >= exact).all())


@pytest.mark.parametrize("ue8m0", (True, False))
def test_h4096_fp8_constructed_groups(ue8m0):
    """Groups the RMSNorm weight drives to the op's edges: an all-zero group (the
    scale floor: 1e-10 / 448, or 2^-33 as a power of two, every code a zero), a
    group saturating E4M3 (the absmax element's code is +-448 under the exact
    scale, in [224, 448] under the power-of-two scale), a group of tiny
    values (scales far below one); every group bitwise the pinned op."""
    _skip_unless_exported(4096, "fp8_group128")
    values = inputs_h4096(64)
    weight = values["rmsnorm_weight"].float()
    weight[0:128] = 0.0
    weight[128:256] *= 256.0
    weight[256:384] *= 2.0**-20
    values["rmsnorm_weight"] = weight.bfloat16()
    plan = prepare_mega_mhc(**values, sf_layout="fp8_group128", fp8_sf_ue8m0=ue8m0)
    plan.run()
    torch.cuda.synchronize()
    out = plan.outputs
    check_h4096_layout(out, values, "fp8_group128", fp8_sf_ue8m0=ue8m0)
    assert not out["y_bf16"][:, :128].float().any()
    assert not (out["y_fp8"][:, :128].view(torch.uint8) & 0x7F).any()
    floor = 2.0**-33 if ue8m0 else _fp32(_fp32(FP8_EPS) / FP8_MAX)
    assert torch.equal(out["y_fp8_sf"][:, 0], torch.full((64,), floor, device="cuda"))
    top = out["y_fp8"][:, 128:256].float().abs().amax(dim=1)
    if ue8m0:  # the power-of-two scale is up to twice the exact one: the top code is in [224, 448]
        assert bool(((top >= 224.0) & (top <= FP8_MAX)).all()), top
    else:
        assert bool((top == FP8_MAX).all()), top
    assert bool((out["y_fp8_sf"][:, 2] < 2.0**-12).all())
    assert bool((out["y_fp8_sf"][:, 3:] > out["y_fp8_sf"][:, 2:3]).all())


@pytest.mark.parametrize("layout", ("fp8_group128", "nvfp4"))
def test_h4096_quant_graph_replay_equals_eager(layout):
    """A captured quantized-output plan replays to the eager result bitwise, on a
    side stream, after the inputs change in place."""
    _skip_unless_exported(4096, layout)
    values = inputs_h4096(385)
    kw = layout_kwargs(layout, values)
    plan = prepare_mega_mhc(**values, sf_layout=layout, **kw)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        plan.run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        plan.run()
    stream.synchronize()
    for repeat in range(3):
        with torch.cuda.stream(stream):
            values["x"].mul_(1.5)
            values["rmsnorm_weight"].mul_(0.75)
            poison(plan.outputs)
            graph.replay()
        stream.synchronize()
        replayed = {name: value.clone() for name, value in plan.outputs.items()}
        with torch.cuda.stream(stream):
            poison(plan.outputs)
            plan.run()
        stream.synchronize()
        for name, value in plan.outputs.items():
            assert bitwise_equal(value, replayed[name]), f"repeat {repeat}: {name}"
        check_h4096_layout(
            plan.outputs, values, layout, label=f"repeat {repeat} ", **kw
        )
    stream.wait_stream(torch.cuda.current_stream())


def test_h4096_shared_launch_epochs_must_launch_the_same_grid():
    """Plans over one caller-owned ``launch_epochs`` must launch the same grid
    (one epoch slot per active CTA): a second plan with another grid is refused
    naming both grids while the first is alive, the same grid shares, and a
    collected plan releases the storage."""
    _skip_unless_exported(4096, "bf16")
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    epochs = torch.zeros((sms,), dtype=torch.uint64, device="cuda")
    small = prepare_mega_mhc(**inputs_h4096(64), sf_layout="bf16", launch_epochs=epochs)
    grid = small.config["num_launch_sms"]
    assert grid < sms and small.program == BF16_PROGRAM
    with pytest.raises(
        ValueError, match=f"launching {grid} CTAs; this plan launches {sms}"
    ):
        prepare_mega_mhc(**inputs_h4096(4096), sf_layout="bf16", launch_epochs=epochs)
    twin = prepare_mega_mhc(**inputs_h4096(64), sf_layout="bf16", launch_epochs=epochs)
    assert twin.launch_epochs is epochs and small.launch_epochs is epochs
    for plan in (small, twin, small):
        plan.run()
    torch.cuda.synchronize()
    counts = epochs.cpu().tolist()
    assert counts[:grid] == [3] * grid and counts[grid:] == [0] * (sms - grid)
    full = TAIL_PROGRAMS[_runtime.mhc_infer_tail_program(4096, sms)]
    assert full != BF16_PROGRAM
    other = prepare_mega_mhc(**inputs_h4096(4096), sf_layout="bf16")
    assert other.launch_epochs is not epochs and other.config["num_launch_sms"] == sms
    assert other.program == full, "the guard spans the programs"
    del small, twin, plan
    gc.collect()
    big = prepare_mega_mhc(**inputs_h4096(4096), sf_layout="bf16", launch_epochs=epochs)
    assert big.launch_epochs is epochs and big.program == full
    # Two full-grid plans share fresh storage (on 208 SMs the twin at 4096 tokens
    # and the tailpipe program at 2048); every CTA advances in step. (The storage
    # above is not launched again: its slots are no longer uniform after the
    # smaller grid, and the kernel's epoch stamps must agree across the active
    # CTAs.)
    fresh = torch.zeros((sms,), dtype=torch.uint64, device="cuda")
    pair = {
        t: prepare_mega_mhc(**inputs_h4096(t), sf_layout="bf16", launch_epochs=fresh)
        for t in (4096, 2048)
    }
    for t, plan in pair.items():
        assert plan.program == TAIL_PROGRAMS[_runtime.mhc_infer_tail_program(t, sms)]
        assert plan.program != BF16_PROGRAM and plan.launch_epochs is fresh
        assert plan.config["num_launch_sms"] == sms
        plan.run()
    torch.cuda.synchronize()
    assert fresh.cpu().tolist() == [2] * sms


def test_h4096_a_plans_own_launch_epochs_is_guarded_like_a_caller_owned_one():
    """A plan's internally allocated ``launch_epochs`` handed to a second plan is
    refused with another grid and shared with the same grid, exactly as a
    caller-owned tensor: every plan registers its epoch storage."""
    _skip_unless_exported(4096, "bf16")
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    first = prepare_mega_mhc(**inputs_h4096(64), sf_layout="bf16")
    grid = first.config["num_launch_sms"]
    assert grid < sms
    with pytest.raises(
        ValueError, match=f"launching {grid} CTAs; this plan launches {sms}"
    ):
        prepare_mega_mhc(
            **inputs_h4096(4096), sf_layout="bf16", launch_epochs=first.launch_epochs
        )
    same = prepare_mega_mhc(
        **inputs_h4096(64), sf_layout="bf16", launch_epochs=first.launch_epochs
    )
    assert same.launch_epochs is first.launch_epochs
    for plan in (first, same):
        plan.run()
    torch.cuda.synchronize()
    counts = first.launch_epochs.cpu().tolist()
    assert counts[:grid] == [2] * grid and counts[grid:] == [0] * (sms - grid)


def test_h4096_launch_epochs_must_own_its_whole_storage():
    """A caller-owned ``launch_epochs`` that is a sliced or offset view of a larger
    storage is refused (two views of overlapping memory would carry different
    registry keys); a tensor owning its whole storage is accepted."""
    _skip_unless_exported(4096, "bf16")
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    storage = torch.zeros((sms + 8,), dtype=torch.uint64, device="cuda")
    for view in (storage[8:], storage[:sms]):
        assert view.shape == (sms,)
        with pytest.raises(ValueError, match="must own its whole storage"):
            prepare_mega_mhc(**inputs_h4096(64), sf_layout="bf16", launch_epochs=view)
    whole = torch.zeros((sms,), dtype=torch.uint64, device="cuda")
    plan = prepare_mega_mhc(**inputs_h4096(64), sf_layout="bf16", launch_epochs=whole)
    assert plan.launch_epochs is whole
    plan.run()
    torch.cuda.synchronize()
    grid = plan.config["num_launch_sms"]
    assert whole.cpu().tolist()[:grid] == [1] * grid


@pytest.mark.parametrize("fitted", (True, False))
def test_h4096_nvfp4_constructed_blocks(fitted):
    """Blocks the RMSNorm weight drives to the op's edges under the fitted global
    scale (448 * 6 / amax of the output) or gs = 1: an all-zero block (scale byte
    0, every code a zero with its sign kept, as the op writes it), the block
    holding the output's amax (its scale 448 and the amax element's code +-6
    under the fitted scale), a block of tiny values; every block within the
    formula's tolerance and the scale bytes bitwise."""
    _skip_unless_exported(4096, "nvfp4")
    values = inputs_h4096(64)
    weight = values["rmsnorm_weight"].float()
    weight[0:16] = 0.0
    weight[16:32] *= 256.0
    weight[32:48] *= 2.0**-20
    values["rmsnorm_weight"] = weight.bfloat16()
    oracle = reference_h4096(values)["y_bf16"]
    gs = fp4_global_scale_for(oracle) if fitted else torch.ones(1, device="cuda")
    plan = prepare_mega_mhc(**values, sf_layout="nvfp4", fp4_global_scale=gs)
    plan.run()
    torch.cuda.synchronize()
    out = plan.outputs
    check_h4096_layout(out, values, "nvfp4", fp4_global_scale=gs)
    assert not out["y_bf16"][:, :16].float().any()
    assert (
        not (out["y_fp4"][:, :8] & 0x77).any()
        and not out["y_fp4_sf"][:, 0].view(torch.uint8).any()
    )
    sf = out["y_fp4_sf"].float()
    if fitted:
        y = out["y_bf16"].float().abs()
        rows = (y[:, 16:32].amax(dim=1) == y.amax(dim=1)) & (y.amax(dim=1) == y.max())
        assert bool(rows.any()), (
            "the token holding the output's amax has it in the scaled block"
        )
        assert bool((sf[rows, 1] == FP8_MAX).all())
        codes = unpack_e2m1(out["y_fp4"][rows, 8:16]) & 0x7
        assert bool((codes.amax(dim=1) == 7).all()), "the amax element saturates E2M1"
    assert bool((sf[:, 2] < sf[:, 3:].amin(dim=1)).all())
