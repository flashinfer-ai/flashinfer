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
        if isinstance(value, torch.Tensor):
            value.fill_(-559038737 if value.dtype == torch.int32 else float("nan"))


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
    a resident grid, the scratch size, and the deterministic mode's 16."""
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
def test_h4096_values_and_replay(tokens):
    _skip_unless_exported(4096, "bf16")
    values = inputs_h4096(tokens)
    plan = prepare_mega_mhc(**values, sf_layout="bf16")
    _arch, sms = _runtime.device_facts(torch.cuda.current_device())
    assert plan.config["num_splits"] == _runtime.mhc_infer_num_splits(tokens, sms)
    assert plan.config["num_launch_sms"] == _runtime.mhc_infer_num_ctas(
        tokens, sms, plan.config["num_splits"]
    )
    assert set(plan.outputs) == {
        "new_residual",
        "new_post_mix",
        "new_comb_res_mix",
        "y_bf16",
    }
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
        check_h4096(plan.outputs, values)
        current = {name: value.clone() for name, value in plan.outputs.items()}
        if first is None:
            first = current
        for name, value in current.items():
            torch.testing.assert_close(value, first[name], atol=0, rtol=0)
    stream.wait_stream(torch.cuda.current_stream())


@pytest.mark.parametrize("tokens", (1, 200, 1025))
def test_h4096_graph_replay_changed_inputs_and_update_fn(tokens):
    """Six replays over inputs rewritten in place, the fourth after ``update_fn``
    on a fresh projection; the replacement keeps fn's 0.01 scale, so the FP32
    coefficients stay in the regime of the other tests' inputs."""
    _skip_unless_exported(4096, "bf16")
    values = inputs_h4096(tokens)
    plan = prepare_mega_mhc(**values, sf_layout="bf16")
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
        check_h4096(plan.outputs, values, label=f"repeat {repeat} ")
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
    assert grid < sms
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
    other = prepare_mega_mhc(**inputs_h4096(4096), sf_layout="bf16")
    assert other.launch_epochs is not epochs and other.config["num_launch_sms"] == sms
    del small, twin, plan
    gc.collect()
    big = prepare_mega_mhc(**inputs_h4096(4096), sf_layout="bf16", launch_epochs=epochs)
    assert big.launch_epochs is epochs


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
