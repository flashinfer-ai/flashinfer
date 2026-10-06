"""
Tests for concat_mla_kv_quant_fp8 — the fused bf16 -> fp8 e4m3 MLA context K/V
pack. It is a saturating cast plus pure memory movement, so the output must be
**byte-exact** against the explicit saturating reference (and against torch's
own GPU cast on torch >= 2.13).
"""

import pytest
import torch

import flashinfer
from flashinfer import mla_kv_pack
from flashinfer.utils import get_compute_capability

NOPE, ROPE, V = 128, 64, 128


def _fused_kernel_dispatches_here() -> bool:
    """The shipped allowlist dispatches the fused kernel on CC 10.0+ only; older
    GPUs take the PyTorch fallback (covered by the fallback tests below)."""
    return torch.cuda.is_available() and get_compute_capability(
        torch.device("cuda")
    ) >= (10, 0)


requires_fused_dispatch = pytest.mark.skipif(
    not _fused_kernel_dispatches_here(),
    reason="fused concat_mla_kv_quant_fp8 kernel dispatches on compute capability "
    "10.0+ only; this GPU takes the fallback path",
)


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


def _assert_fused_byte_exact(T, H):
    kv, pe = _inputs(T, H)
    ref_key, ref_value = _reference(kv, pe)
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["specialized_dispatches"]
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    assert key.dtype == torch.float8_e4m3fn and value.dtype == torch.float8_e4m3fn
    assert key.shape == (T, H, NOPE + ROPE) and value.shape == (T, H, V)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["specialized_dispatches"] == before + 1, stats


@requires_fused_dispatch
@pytest.mark.parametrize("T", [1, 7, 1536, 66038, 66048])
@pytest.mark.parametrize("H", [12, 8])
def test_byte_exact(T, H):
    """12 heads takes the unrolled kernel, 8 heads the runtime-head-count one."""
    _assert_fused_byte_exact(T, H)


@requires_fused_dispatch
@pytest.mark.parametrize("T,H", [(129, 1), (127, 3), (1536, 13), (2048, 128)])
def test_byte_exact_odd_head_counts(T, H):
    """Runtime head count: heads below the 2-way / 4-way lane strides and the
    allowlist's largest head count."""
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


def test_all_bf16_patterns_byte_exact():
    """Every bf16 bit pattern through the fused kernel (saturation, NaN sign
    dropping, subnormal RNE, -0.0) against the explicit reference."""
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
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["specialized_dispatches"]
    monkeypatch.setenv("FLASHINFER_SPECIALIZED_KERNEL_DISABLE", "1")
    key, value = flashinfer.concat_mla_kv_quant_fp8(kv, pe)
    stats = mla_kv_pack._concat_mla_kv_quant_fp8_stats()
    assert stats["specialized_dispatches"] == before
    assert stats["fallback_reasons"].get("kill_switch", 0) >= 1
    # Fallback is torch's cast: byte-exact with the kernel only where the cast saturates.
    if _torch_cast_saturates():
        assert torch.equal(key.view(torch.uint8), ref_key)
        assert torch.equal(value.view(torch.uint8), ref_value)


def test_k_pe_3d_and_preallocated_outputs():
    kv, pe = _inputs(512, 12)
    key = torch.empty(512, 12, NOPE + ROPE, dtype=torch.float8_e4m3fn, device="cuda")
    value = torch.empty(512, 12, V, dtype=torch.float8_e4m3fn, device="cuda")
    k2, v2 = flashinfer.concat_mla_kv_quant_fp8(kv, pe.unsqueeze(1), key, value)
    assert k2.data_ptr() == key.data_ptr() and v2.data_ptr() == value.data_ptr()
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
        return mla_kv_pack._specialized_supported(kv, pe, key, value, NOPE)

    assert reason(1024, 12) == "device"  # every GPU-free guard passed
    assert reason(1024, 129) == "num_heads_not_allowlisted"
    assert reason(131073, 12) == "num_tokens_not_allowlisted"
    kv = torch.empty(8, 12, NOPE + V, dtype=torch.bfloat16)
    pe = torch.empty(8, ROPE, dtype=torch.bfloat16)
    key = torch.empty(8, 12, NOPE + ROPE, dtype=fp8)
    bad_value = torch.empty(8, 12, V, dtype=torch.float8_e5m2)
    assert mla_kv_pack._specialized_supported(kv, pe, key, bad_value, NOPE) == (
        "output_dtype"
    )
    assert mla_kv_pack._specialized_supported(kv, pe, key, key, NOPE) == (
        "output_geometry"
    )


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
    assert after["specialized_dispatches"] == before["specialized_dispatches"]
    assert after["fallback_reasons"]["compute_capability"] == (
        before["fallback_reasons"].get("compute_capability", 0) + 1
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
    assert stats["compiled_variants"] == 2
    assert stats["distinct_kernels_for_allowlist"] == 2
    assert stats["allowlist_loaded"]


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
    before = mla_kv_pack._concat_mla_kv_quant_fp8_stats()["specialized_dispatches"]
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            flashinfer.concat_mla_kv_quant_fp8(kv, pe, key, value)
    torch.cuda.current_stream().wait_stream(s)
    assert (
        mla_kv_pack._concat_mla_kv_quant_fp8_stats()["specialized_dispatches"]
        == before + 1
    )
    key.zero_()
    value.zero_()
    g.replay()
    torch.cuda.synchronize()
    ref_key, ref_value = _reference(kv, pe)
    assert torch.equal(key.view(torch.uint8), ref_key)
    assert torch.equal(value.view(torch.uint8), ref_value)
