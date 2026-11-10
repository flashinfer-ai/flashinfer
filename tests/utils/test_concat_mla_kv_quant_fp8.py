"""
Tests for concat_mla_kv_quant_fp8 — the fused bf16 -> fp8 e4m3 MLA context K/V
pack (one generated Cake program per head group). It is a saturating cast plus
pure memory movement, so the output must be **byte-exact** against the explicit
saturating reference (and against torch's own GPU cast on torch >= 2.13).
"""

import pytest
import torch

import flashinfer
from flashinfer import mla_kv_pack
from flashinfer.jit.cake_concat_mla_kv_quant_fp8 import HG1, HG2, HG3, HG4, PROGRAMS
from flashinfer.jit.cake_program import GRID_ARGS
from flashinfer.utils import get_compute_capability

NOPE, ROPE, V = 128, 64, 128


def _cake_dispatches_here() -> bool:
    """The Cake programs are built for the exact compute capabilities 10.0 and
    10.3 only."""
    return torch.cuda.is_available() and (
        mla_kv_pack.cake_target_for_capability(
            get_compute_capability(torch.device("cuda"))
        )
        is not None
    )


def _specialized_dispatches_here() -> bool:
    """The specialized kernel is compiled for every compute capability >= 10.0."""
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) >= (10, 0)


def _fused_kernel_dispatches_here() -> bool:
    return _cake_dispatches_here() or _specialized_dispatches_here()


requires_fused_dispatch = pytest.mark.skipif(
    not _fused_kernel_dispatches_here(),
    reason="the fused concat_mla_kv_quant_fp8 backends dispatch on compute "
    "capability 10.0+ only; this GPU takes the fallback path",
)
requires_cake = pytest.mark.skipif(
    not _cake_dispatches_here(),
    reason="the Cake backend is built for compute capability 10.0 / 10.3 only",
)
requires_specialized = pytest.mark.skipif(
    not _specialized_dispatches_here(),
    reason="the specialized backend dispatches on compute capability 10.0+ only",
)


def _backend_available(backend: str) -> bool:
    return {
        "auto": _fused_kernel_dispatches_here(),
        "specialized": _specialized_dispatches_here(),
        "cake": _cake_dispatches_here(),
    }[backend]


def _expected_auto_backend(H: int) -> str:
    """What ``backend="auto"`` resolves to on this GPU for ``H`` local heads."""
    preferred = (
        ("specialized", "cake")
        if H in mla_kv_pack._SPECIALIZED_AUTO_HEADS
        else ("cake", "specialized")
    )
    for candidate in preferred:
        if _backend_available(candidate):
            return candidate
    raise AssertionError("no fused backend on this GPU")


def _sat_e4m3_bytes(x: torch.Tensor) -> torch.Tensor:
    """Explicit saturating bf16 -> e4m3fn (RNE; overflow/inf -> 448; NaN -> 0x7F)."""
    f = x.float()
    sign = (torch.signbit(f)).to(torch.uint8) << 7
    a = f.abs()
    nan = torch.isnan(f)
    # e4m3fn grid: normals 2^-6..448, subnormals multiples of 2^-9.
    e = torch.floor(torch.log2(a.clamp_min(2.0**-6)))
    step = torch.pow(2.0, e - 3)
    step = torch.where(a < 2.0**-6, torch.full_like(step, 2.0**-9), step)
    q = torch.round(a / step)  # RNE on ties (torch.round is half-to-even)
    val = (q * step).clamp(max=448.0)
    val = torch.where(torch.isinf(a), torch.full_like(val, 448.0), val)
    out = val.to(torch.float8_e4m3fn).view(torch.uint8) | sign
    out = torch.where(nan, torch.full_like(out, 0x7F), out)
    return out


def _reference(kv_nope: torch.Tensor, k_pe: torch.Tensor):
    T, H, _ = kv_nope.shape
    kv8 = _sat_e4m3_bytes(kv_nope)
    pe8 = _sat_e4m3_bytes(k_pe.reshape(T, ROPE))
    key = torch.empty(T, H, NOPE + ROPE, dtype=torch.uint8, device=kv_nope.device)
    key[..., :NOPE] = kv8[..., :NOPE]
    key[..., NOPE:] = pe8.unsqueeze(1).expand(T, H, ROPE)
    value = kv8[..., NOPE:].contiguous()
    return key, value


def _inputs(T: int, H: int, device="cuda"):
    gen = torch.Generator(device=device).manual_seed(T * 131 + H)
    kv = (
        torch.randn(T, H, NOPE + V, device=device, dtype=torch.float32, generator=gen)
        * 2.0
    )
    pe = torch.randn(T, ROPE, device=device, dtype=torch.float32, generator=gen) * 2.0
    kv = kv.to(torch.bfloat16)
    pe = pe.to(torch.bfloat16)
    # Saturation / NaN / sign edge values on every row.
    edges = torch.tensor(
        [
            466.0,
            480.0,
            1e4,
            -1e4,
            float("inf"),
            -float("inf"),
            float("nan"),
            -0.0,
            2.0**-10,
            448.0,
        ],
        device=device,
        dtype=torch.bfloat16,
    )
    if T:
        n = min(T * H, 64)
        idx = torch.randint(0, T * H * (NOPE + V), (n,), device=device, generator=gen)
        kv.view(-1)[idx] = edges[torch.arange(n, device=device) % edges.numel()]
        pe.view(-1)[
            torch.randint(0, T * ROPE, (min(T, 16),), device=device, generator=gen)
        ] = edges[0]
    return kv, pe


def _torch_cast_saturates() -> bool:
    return not torch.isnan(
        torch.tensor([1e4], device="cuda", dtype=torch.bfloat16)
        .to(torch.float8_e4m3fn)
        .float()
    ).item()


def _assert_fused_byte_exact(T, H, backend="auto"):
    if not _backend_available(backend):
        pytest.skip(f"backend {backend!r} does not dispatch on this GPU")
    expected = _expected_auto_backend(H) if backend == "auto" else backend
    kv, pe = _inputs(T, H)
    ref_key, ref_value = _reference(kv, pe)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe, backend=backend)
    assert key.dtype == torch.float8_e4m3fn and value.dtype == torch.float8_e4m3fn
    assert key.shape == (T, H, NOPE + ROPE) and value.shape == (T, H, V)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["fused_dispatches"] == before["fused_dispatches"] + 1, stats
    assert (
        stats["backend_dispatches"][expected]
        == before["backend_dispatches"][expected] + 1
    ), (expected, stats)


# ---------------------------------------------------------------------------
# Host-only tests (no GPU)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("T", "H", "head_group", "grid", "warps_per_token"),
    [
        # Frozen from the Cake production launcher (_plan_head_group / _plan_grid).
        (1, 1, 1, 1, 1),
        (131072, 1, 1, 16384, 1),
        (1024, 1, 1, 128, 1),
        (127, 3, 2, 16, 1),
        (1, 6, 3, 1, 1),
        (1536, 6, 3, 192, 1),
        (65536, 6, 3, 8192, 1),
        (1, 12, 2, 1, 3),
        (1536, 12, 2, 576, 3),
        (2048, 12, 2, 768, 3),
        (2049, 12, 4, 513, 2),  # 6 pairs: 4 + 2 (the register budget's maximum first)
        (66048, 12, 4, 16512, 2),
        (4096, 18, 4, 1536, 3),  # 9 pairs: 4 + 4 + 1 over three warps
        (1536, 24, 2, 1152, 6),
        (16384, 24, 4, 6144, 3),
        (65536, 96, 4, 98304, 12),
        (1, 128, 2, 4, 32),
        (65536, 128, 4, 131072, 16),
        (131072, 128, 4, 262144, 16),
        (4096, 10, 4, 1024, 2),  # 5 pairs: 4 + 1
        (4096, 13, 4, 1024, 2),  # 7 pairs, odd head count (last pair half predicated)
    ],
)
def test_head_group_plan_matches_the_cake_launcher(
    T, H, head_group, grid, warps_per_token
):
    assert mla_kv_pack._plan_head_group(T, H) == head_group
    assert mla_kv_pack._plan_grid(T, H, head_group) == (grid, warps_per_token)


def test_every_head_count_has_a_delivered_program():
    """Every H in the allowlist resolves to a generated program at every token
    count, and every generated program serves one head group."""
    for T in (1, 2048, 2049, 131072):
        for H in range(1, 129):
            head_group = mla_kv_pack._plan_head_group(T, H)
            assert head_group in mla_kv_pack._CAKE_PROGRAMS, (T, H, head_group)
    assert tuple(sorted(mla_kv_pack._CAKE_PROGRAMS)) == (1, 2, 3, 4)
    assert mla_kv_pack._CAKE_PROGRAMS == {1: HG1, 2: HG2, 3: HG3, 4: HG4}
    assert set(mla_kv_pack._CAKE_PROGRAMS.values()) == set(PROGRAMS)
    assert len({program.name for program in PROGRAMS}) == len(PROGRAMS) == 4


def test_every_program_states_the_pack_launch():
    """Every generated program is the pack kernel at one head group: the same
    exact targets, the kernel symbol of its own name, the eight-warp block, no
    cluster, no dynamic shared memory, no PDL, not cooperative."""
    for program in PROGRAMS:
        assert program.targets == ("sm_100a", "sm_103a")
        assert program.kernel == f"kernel_{program.name}"
        assert program.block == (32 * mla_kv_pack._WARPS_PER_CTA, 1, 1)
        assert program.cluster == (1, 1, 1)
        assert program.launch_record() == {
            "block": (256, 1, 1),
            "cluster": (1, 1, 1),
            "cooperative": False,
            "dynamic_smem_bytes": 0,
            "use_pdl": False,
        }
        assert program.tensors == ("kv_nope", "k_pe", "key", "value")
        assert program.args[-3:] == GRID_ARGS
        assert program.args[: len(program.tensors)] == program.tensors


@pytest.mark.parametrize(
    ("capability", "expected"),
    [((10, 0), "sm_100a"), ((10, 3), "sm_103a"), ((9, 0), None), ((12, 0), None)],
)
def test_exact_architecture_router(capability, expected):
    for program in PROGRAMS:
        assert program.target_for(capability) == expected
    assert mla_kv_pack.cake_target_for_capability(capability) == expected


def test_architecture_router_rejects_cross_routing(monkeypatch):
    monkeypatch.setattr(mla_kv_pack, "get_compute_capability", lambda _device: (12, 0))
    with pytest.raises(RuntimeError, match="exact compute capability 10.0 or 10.3"):
        mla_kv_pack._cake_target(HG4, torch.device("cuda"))
    monkeypatch.setattr(mla_kv_pack, "get_compute_capability", lambda _device: (10, 3))
    assert mla_kv_pack._cake_target(HG1, torch.device("cuda")) == "sm_103a"


def test_host_arguments_follow_the_program_order():
    """The host assembles the binding's positional call from the program's
    argument order: the four tensors, the five launch scalars, the grid; a
    row-strided ``k_pe`` passes its row stride in elements."""
    T, H = 7, 12
    kv = torch.empty(T, H, NOPE + V, dtype=torch.bfloat16)
    workspace = torch.empty(T, 576, dtype=torch.bfloat16)
    pe = workspace[:, 512:]
    key = torch.empty(T, H, NOPE + ROPE, dtype=torch.float8_e4m3fn)
    value = torch.empty(T, H, V, dtype=torch.float8_e4m3fn)
    head_group = mla_kv_pack._plan_head_group(T, H)
    grid_x, warps_per_token = mla_kv_pack._plan_grid(T, H, head_group)
    values = mla_kv_pack._cake_launch_values(
        kv, pe, key, value, warps_per_token=warps_per_token, grid_x=grid_x
    )
    program = mla_kv_pack._CAKE_PROGRAMS[head_group]
    assert set(values) == set(program.args)
    arguments = [values[name] for name in program.args]
    assert all(a is b for a, b in zip(arguments[:4], (kv, pe, key, value), strict=True))
    assert arguments[4:] == [T, H, 6, 3, 576, grid_x, 1, 1]
    assert (head_group, grid_x, warps_per_token) == (2, 3, 3)
    for other in PROGRAMS:
        assert set(other.args) == set(values), other.name
    # One token: the row stride is never dereferenced and the guards admit any
    # value, so the dense stride is passed (the binding checks the stride atoms).
    odd = torch.empty(1, 100, dtype=torch.bfloat16)[:, 36:]
    assert odd.stride() == (100, 1) and mla_kv_pack._k_pe_rows_admissible(odd)
    one = mla_kv_pack._cake_launch_values(
        kv[:1], odd, key[:1], value[:1], warps_per_token=3, grid_x=1
    )
    assert one["k_pe_row_stride"] == ROPE and one["num_tokens"] == 1


def test_allowlist_guards_are_gpu_free():
    """The allowlist is checked before the device: exercisable on CPU tensors."""
    allowlist = mla_kv_pack._load_allowlist()
    assert allowlist is not None
    assert allowlist["heads"] == (1, 128) and allowlist["tokens"] == (1, 131072)
    assert allowlist["min_cc"] == (10, 0)
    fp8 = torch.float8_e4m3fn

    def reason(T, H):
        kv = torch.empty(T, H, NOPE + V, dtype=torch.bfloat16)
        pe = torch.empty(T, ROPE, dtype=torch.bfloat16)
        key = torch.empty(T, H, NOPE + ROPE, dtype=fp8)
        value = torch.empty(T, H, V, dtype=fp8)
        return mla_kv_pack._fused_supported(kv, pe, key, value, NOPE)

    assert reason(1024, 12) == "device"  # every GPU-free guard passed
    assert reason(1024, 129) == "num_heads_not_allowlisted"
    assert reason(131073, 12) == "num_tokens_not_allowlisted"
    kv = torch.empty(8, 12, NOPE + V, dtype=torch.bfloat16)
    pe = torch.empty(8, ROPE, dtype=torch.bfloat16)
    key = torch.empty(8, 12, NOPE + ROPE, dtype=fp8)
    bad_value = torch.empty(8, 12, V, dtype=torch.float8_e5m2)
    assert mla_kv_pack._fused_supported(kv, pe, key, bad_value, NOPE) == (
        "output_dtype"
    )
    assert mla_kv_pack._fused_supported(kv, pe, key, key, NOPE) == ("output_geometry")


# ---------------------------------------------------------------------------
# GPU tests
# ---------------------------------------------------------------------------


@requires_fused_dispatch
@pytest.mark.parametrize("backend", ["auto", "specialized", "cake"])
@pytest.mark.parametrize("T", [1, 7, 1536, 66038, 66048])
@pytest.mark.parametrize("H", [12, 8])
def test_byte_exact(T, H, backend):
    """Both fused backends, and the auto route, are byte-exact. 12 heads is the
    specialized kernel's unrolled variant (and auto's pick); 8 heads its
    runtime-head-count variant (auto picks Cake). Cake: 12 heads is two pairs
    per warp up to 2048 tokens, 4 + 2 above; 8 heads two, then four."""
    _assert_fused_byte_exact(T, H, backend)


@requires_fused_dispatch
@pytest.mark.parametrize("H", [1, 6, 12, 13, 24, 96, 128])
def test_auto_routes_by_head_count(H):
    """auto -> specialized for 12 local heads, Cake otherwise, falling over to
    the other backend where the preferred one is not built for this GPU."""
    kv, pe = _inputs(1536, H)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["backend_dispatches"]
    flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["backend_dispatches"]
    expected = _expected_auto_backend(H)
    assert after[expected] == before[expected] + 1, (H, expected, before, after)
    other = "cake" if expected == "specialized" else "specialized"
    assert after[other] == before[other]


def test_backend_order_is_the_documented_rule():
    assert mla_kv_pack._backend_order("auto", 12) == ("specialized", "cake")
    for H in (1, 6, 8, 13, 24, 96, 128):
        assert mla_kv_pack._backend_order("auto", H) == ("cake", "specialized")
    assert mla_kv_pack._backend_order("cake", 12) == ("cake",)
    assert mla_kv_pack._backend_order("specialized", 24) == ("specialized",)
    assert mla_kv_pack.BACKENDS == ("auto", "specialized", "cake")


def test_rejects_unknown_backend():
    kv, pe = _inputs(8, 12)
    with pytest.raises(ValueError, match="backend must be one of"):
        flashinfer.concat_mla_kv_quant_fp8(kv, pe, backend="cuda")


@requires_fused_dispatch
def test_explicit_backend_that_cannot_serve_takes_fallback():
    """An explicit backend is never silently swapped for the other one: the
    specialized kernel needs a contiguous k_pe, so a row-strided k_pe under
    backend="specialized" takes the composable path (byte-exactly)."""
    kv, pe = _inputs(257, 12)
    pe_view = _strided_k_pe(pe, 576, 512)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe_view, backend="specialized")
    after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert after["fused_dispatches"] == before["fused_dispatches"]
    assert after["fallback_reasons"]["non_contiguous"] == (
        before["fallback_reasons"].get("non_contiguous", 0) + 1
    )
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)


@requires_fused_dispatch
@pytest.mark.parametrize(
    "T,H", [(129, 1), (127, 3), (1536, 13), (4096, 10), (2048, 128)]
)
def test_byte_exact_odd_head_counts(T, H):
    """One pair per warp (H <= 2), odd head counts (half-predicated last pair),
    pair counts four does not divide, and the allowlist's largest head count."""
    _assert_fused_byte_exact(T, H)


@requires_fused_dispatch
@pytest.mark.parametrize("H", [1, 6, 12, 24])
def test_byte_exact_every_head_group(H):
    """Every delivered program (head group 1, 3, 2 and 4) at a token count above
    the small-T threshold and at one below it."""
    for T in (2048, 2049):
        _assert_fused_byte_exact(T, H)


def test_matches_torch_cast_when_saturating():
    """On torch >= 2.13 the GPU cast saturates: kernel == cast + slice copies."""
    if not _torch_cast_saturates():
        pytest.skip("this torch's fp8 cast is NaN-on-overflow (< 2.13)")
    kv, pe = _inputs(4096, 12)
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    kv8 = kv.to(torch.float8_e4m3fn)
    exp_key = torch.cat(
        [kv8[..., :NOPE], pe.to(torch.float8_e4m3fn).unsqueeze(1).expand(-1, 12, -1)],
        -1,
    )
    assert torch.equal(key.view(torch.uint8), exp_key.contiguous().view(torch.uint8))
    assert torch.equal(
        value.view(torch.uint8), kv8[..., NOPE:].contiguous().view(torch.uint8)
    )


@requires_fused_dispatch
def test_all_bf16_patterns_byte_exact():
    """Every bf16 bit pattern through the fused kernel (saturation, NaN sign
    dropping, subnormal RNE, -0.0) against the explicit reference.  Fused
    dispatch only: outside the dispatch surface the operator is the composable
    torch path, whose fp8 cast bits on NaN / overflow / -0.0 patterns depend on
    the torch build (that path is covered by test_matches_torch_cast_when_saturating)."""
    allb = (
        torch.arange(0, 65536, dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
        .cuda()
    )
    kv = allb.repeat(48).view(1024, 12, NOPE + V)  # 48 x 65536 = 1024 x 12 x 256
    pe = allb.view(1024, ROPE)  # every pattern once through the rope path too
    ref_key, ref_value = _reference(kv, pe)
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)


def test_kill_switch_takes_fallback(monkeypatch):
    kv, pe = _inputs(2048, 12)
    ref_key, ref_value = _reference(kv, pe)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"]
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["fused_dispatches"] == before
    assert stats["fallback_reasons"].get("kill_switch", 0) >= 1
    # Random data: compared where torch's own cast saturates (torch >= 2.13); the
    # fallback's NaN / overflow encoding is pinned on any torch build by
    # test_fallback_encodes_nan_and_overflow_like_the_kernel.
    if _torch_cast_saturates():
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


def test_fallback_encodes_nan_and_overflow_like_the_kernel(monkeypatch):
    """The composable path matches the fused kernel's saturating cast on every
    torch build: finite overflow and +-inf saturate to +-448 and every NaN,
    either sign, encodes as 0x7F.  torch < 2.13's software cast alone does
    neither (NaN on overflow, 0xFF for -NaN)."""
    # bf16 bit patterns: +NaN, -NaN, +inf, -inf, -0.0, 0.0 (as signed int16)
    bits = torch.tensor([0x7FC0, -64, 0x7F80, -128, -32768, 0], dtype=torch.int16)
    edge = torch.cat(
        [
            bits.view(torch.bfloat16),
            torch.tensor(
                [1e4, -1e4, 466.0, -466.0, 480.0, 448.0, -448.0, 2.0**-10],
                dtype=torch.bfloat16,
            ),
        ]
    ).cuda()
    T, H = 64, 12
    n_kv, n_pe = T * H * (NOPE + V), T * ROPE
    kv = edge.repeat(n_kv // edge.numel() + 1)[:n_kv].view(T, H, NOPE + V).clone()
    pe = edge.repeat(n_pe // edge.numel() + 1)[:n_pe].view(T, ROPE).clone()
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"]
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["fused_dispatches"] == before
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
    # Every NaN input (both signs) -> 0x7F, in the nope, rope and value columns.
    key_u8, value_u8 = key.view(torch.uint8), value.view(torch.uint8)
    assert (key_u8[..., :NOPE][torch.isnan(kv[..., :NOPE])] == 0x7F).all()
    assert (value_u8[torch.isnan(kv[..., NOPE:])] == 0x7F).all()
    assert (key_u8[:, 0, NOPE:][torch.isnan(pe)] == 0x7F).all()


def test_k_pe_3d_and_preallocated_outputs():
    kv, pe = _inputs(512, 12)
    key = torch.empty(512, 12, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    value = torch.empty(512, 12, V, dtype=torch.float8_e4m3fn, device="cuda")
    k2, v2 = flashinfer.concat_mla_kv_quant_fp8(kv, pe.unsqueeze(1), key, value)
    assert k2.data_ptr() == key.data_ptr() and v2.data_ptr() == value.data_ptr()
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)


def _strided_k_pe(pe: torch.Tensor, row_stride: int, col_offset: int) -> torch.Tensor:
    """``pe`` placed as columns [col_offset, col_offset + 64) of a [T, row_stride] bf16 workspace (other columns noise)."""
    T = pe.shape[0]
    workspace = (torch.randn(T, row_stride, device="cuda") * 3).to(torch.bfloat16)
    view = workspace[:, col_offset : col_offset + ROPE]
    view.copy_(pe)
    assert view.stride() == (row_stride, 1) and not view.is_contiguous()
    return view


@requires_cake
@pytest.mark.parametrize(("T", "H"), [(129, 13), (1536, 12), (3, 1)])
@pytest.mark.parametrize(
    ("row_stride", "col_offset"),
    [(576, 512), (576, 0), (128, 64), (80, 16)],
)
def test_strided_k_pe_byte_exact(T, H, row_stride, col_offset):
    """A row-strided k_pe (vLLM's non-DCP prefill passes the last 64 columns of
    the [T, 576] latent; 32-byte-aligned rows) is served by the fused kernel
    byte-exactly, without a contiguous copy."""
    kv, pe = _inputs(T, H)
    pe_view = _strided_k_pe(pe, row_stride, col_offset)
    assert (
        mla_kv_pack._fused_supported(
            kv,
            pe_view,
            torch.empty(T, H, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda"),
            torch.empty(T, H, V, dtype=torch.float8_e4m3fn, device="cuda"),
            NOPE,
        )
        is None
    )
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"]
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe_view)
    assert (
        mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"] == before + 1
    )
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
    # the [T, 1, 64] view of the same strided columns
    key3, value3 = flashinfer.concat_mla_kv_quant_fp8(kv, pe_view.unsqueeze(1))
    assert torch.equal(key3.view(torch.uint8), ref_key)
    assert torch.equal(value3.view(torch.uint8), ref_value)


@pytest.mark.parametrize(
    ("row_stride", "col_offset", "reason"),
    [(72, 8, "non_contiguous"), (576, 8, "alignment")],
)
def test_inadmissible_k_pe_layout_takes_fallback(row_stride, col_offset, reason):
    """Row strides that are not 32-byte multiples, or a base that is only
    16-byte aligned, take the composable torch path (256-bit lane loads)."""
    kv, pe = _inputs(257, 12)
    pe_view = _strided_k_pe(pe, row_stride, col_offset)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fallback_reasons"].get(
        reason, 0
    )
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe_view)
    assert (
        mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fallback_reasons"][reason]
        == before + 1
    )
    if _torch_cast_saturates():
        ref_key, ref_value = _reference(kv, pe)
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


def test_non_contiguous_takes_fallback():
    kv, pe = _inputs(256, 12)
    kv_strided = torch.empty(
        256, 12, 2 * (NOPE + V), device="cuda", dtype=torch.bfloat16
    )[..., : NOPE + V]
    kv_strided.copy_(kv)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fallback_reasons"].get(
        "non_contiguous", 0
    )
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv_strided, pe)
    assert (
        mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fallback_reasons"][
            "non_contiguous"
        ]
        == before + 1
    )
    if _torch_cast_saturates():
        ref_key, ref_value = _reference(kv, pe)
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


def test_rejects_non_fp8_output_buffers():
    """Caller-provided outputs must honour the float8_e4m3fn contract; a wrong
    dtype is an error, never a silent fallback into that buffer."""
    kv, pe = _inputs(8, 12)
    key = torch.empty(8, 12, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    value_e5m2 = torch.empty(8, 12, V, dtype=torch.float8_e5m2, device="cuda")
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        flashinfer.concat_mla_kv_quant_fp8(kv, pe, key=key, value=value_e5m2)
    key_bf16 = torch.empty(8, 12, NOPE + ROPE, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError, match="float8_e4m3fn"):
        flashinfer.concat_mla_kv_quant_fp8(kv, pe, key=key_bf16)


def test_pre_blackwell_device_takes_fallback(monkeypatch):
    """The fused kernel is dispatched on compute capability 10.0+ only."""
    monkeypatch.setattr(mla_kv_pack, "get_compute_capability", lambda device: (9, 0))
    kv, pe = _inputs(512, 12)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert after["fused_dispatches"] == before["fused_dispatches"]
    assert after["fallback_reasons"]["compute_capability"] == (
        before["fallback_reasons"].get("compute_capability", 0) + 1
    )
    if _torch_cast_saturates():
        ref_key, ref_value = _reference(kv, pe)
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


def test_unbuilt_exact_target_takes_fallback(monkeypatch):
    """A 10.x / 12.x part without a built Cake program takes the composable path
    under backend="cake", byte-exactly (auto would use the specialized kernel)."""
    monkeypatch.setattr(mla_kv_pack, "get_compute_capability", lambda device: (12, 0))
    kv, pe = _inputs(512, 12)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe, backend="cake")
    after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert after["fused_dispatches"] == before["fused_dispatches"]
    assert after["fallback_reasons"]["exact_target"] == (
        before["fallback_reasons"].get("exact_target", 0) + 1
    )
    if _torch_cast_saturates():
        ref_key, ref_value = _reference(kv, pe)
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


@requires_fused_dispatch
def test_stats_hook_reports_compile_footprint():
    flashinfer.concat_mla_kv_quant_fp8(*_inputs(64, 12))
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["module_loaded"] and stats["precompiled"]
    assert stats["compiled_variants"] >= 1
    assert stats["distinct_kernels_for_allowlist"] == len(PROGRAMS) == 4
    assert stats["allowlist_loaded"]
    assert not stats["module_errors"]


def test_zero_tokens():
    kv, pe = _inputs(0, 12)
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    assert key.shape == (0, 12, NOPE + ROPE) and value.shape == (0, 12, V)


@requires_fused_dispatch
def test_cuda_graph_capture_after_warmup():
    kv, pe = _inputs(1024, 12)
    flashinfer.concat_mla_kv_quant_fp8(kv, pe)  # builds/loads the module
    key = torch.empty(1024, 12, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    value = torch.empty(1024, 12, V, dtype=torch.float8_e4m3fn, device="cuda")
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.concat_mla_kv_quant_fp8(kv, pe, key, value)
    torch.cuda.current_stream().wait_stream(s)
    assert (
        mla_kv_pack._concat_mla_kv_quant_fp8_stats()["fused_dispatches"] == before + 1
    )
    key.zero_()
    value.zero_()
    g.replay()
    torch.cuda.synchronize()
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
    # Same graph, new source contents: the replay must follow the data.
    kv.mul_(-0.5)
    pe.mul_(-0.5)
    g.replay()
    torch.cuda.synchronize()
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)


@requires_cake
def test_capture_before_jit_takes_fallback():
    """Capturing a head group whose module is not loaded yet must not JIT inside
    the capture: the stock path serves that capture, byte-exactly."""
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    # Head group 1 (H <= 2) is rarely warmed by the other tests; force the cold state.
    head_group = mla_kv_pack._plan_head_group(256, 1)
    program = mla_kv_pack._CAKE_PROGRAMS[head_group]
    target = mla_kv_pack._cake_target(program, torch.device("cuda"))
    saved = mla_kv_pack._cake_modules.pop((target, head_group), None)
    try:
        kv, pe = _inputs(256, 1)
        key = torch.empty(256, 1, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
        value = torch.empty(256, 1, V, dtype=torch.float8_e4m3fn, device="cuda")
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                flashinfer.concat_mla_kv_quant_fp8(kv, pe, key, value, backend="cake")
        torch.cuda.current_stream().wait_stream(s)
        after = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
        assert after["fallback_reasons"]["capturing_before_jit"] == (
            stats["fallback_reasons"].get("capturing_before_jit", 0) + 1
        )
        g.replay()
        torch.cuda.synchronize()
        if _torch_cast_saturates():
            ref_key, ref_value = _reference(kv, pe)
            assert torch.equal(key.view(torch.uint8), ref_key)
            assert torch.equal(value.view(torch.uint8), ref_value)
    finally:
        if saved is not None:
            mla_kv_pack._cake_modules[(target, head_group)] = saved
