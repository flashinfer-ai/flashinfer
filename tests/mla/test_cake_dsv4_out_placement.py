"""Route-conditional output placement of the Cake DSv4 host.

The kernel time of the bf16 H128 persistent and the fp8 H64 SWA single-CTA routes is a 4 KiB-periodic function of the
output buffer's base address (slow at ``base % 4096 == 0``, fast at ``0x800``); with ``out=None`` the host places the
output at the fast phase on exactly those routes and leaves every other route at the allocator default.  CPU only.
"""

import pytest
import torch

from flashinfer.mla import cake_dsv4 as cake
from flashinfer.mla.cake_dsv4 import allocate_cake_dsv4_output, cake_dsv4_out_phase


def test_out_phase_table_is_route_conditional():
    assert (
        cake_dsv4_out_phase("bf16_h128_prefill_v42", num_heads=128, sparse_topk=128)
        == 0x800
    )
    assert (
        cake_dsv4_out_phase("bf16_h128_prefill_v42", num_heads=128, sparse_topk=1152)
        == 0x800
    )
    assert (
        cake_dsv4_out_phase("fp8_lowhead_prefill", num_heads=64, sparse_topk=128)
        == 0x800
    )
    for heads in (8, 16, 32):
        assert (
            cake_dsv4_out_phase("fp8_lowhead_prefill", num_heads=heads, sparse_topk=128)
            is None
        )
    assert (
        cake_dsv4_out_phase("fp8_lowhead_prefill", num_heads=64, sparse_topk=640)
        is None
    )
    for route in (
        "bf16_h64_prefill",
        "fp8_h64_source_exact",
        "fp8_lowhead_h64",
        "bf16_h128_topk128x",
    ):
        assert cake_dsv4_out_phase(route, num_heads=64, sparse_topk=128) == 0x800
    for route in (
        "bf16_h128_swa128",
        "bf16_swa128_single_cta",
        "bf16_h64_compressed_q8_v38",
        "fp8_h128_persistent",
        "fp8_h128_prefill_source_persistent",
        "bf16_h64_guard_q_tma_batch_r25",
        "fp8_lowhead_h64_split",
    ):
        assert cake_dsv4_out_phase(route, num_heads=64, sparse_topk=128) is None


@pytest.mark.parametrize("shape", [(12, 64, 512), (3, 4, 128, 512), (7, 512)])
def test_allocate_output_pins_the_base_phase(shape):
    for _ in range(4):
        out = allocate_cake_dsv4_output(shape, torch.device("cpu"), phase=0x800)
        assert (
            tuple(out.shape) == shape
            and out.dtype == torch.bfloat16
            and out.is_contiguous()
        )
        assert out.data_ptr() % 4096 == 0x800
    plain = allocate_cake_dsv4_output(shape, torch.device("cpu"), phase=None)
    assert tuple(plain.shape) == shape and plain.dtype == torch.bfloat16
    with pytest.raises(ValueError):
        allocate_cake_dsv4_output(shape, torch.device("cpu"), phase=4096)
    with pytest.raises(ValueError):
        allocate_cake_dsv4_output(shape, torch.device("cpu"), phase=1)


def _aligned_u8(num_bytes, align=128):
    """A zeroed uint8 CPU view whose data pointer is ``align``-byte aligned (CPU allocations are only 64 B aligned)."""
    raw = torch.zeros((num_bytes + align,), dtype=torch.uint8)
    off = (-raw.data_ptr()) % align
    return raw[off : off + num_bytes]


def _run_fp8_swa(monkeypatch, *, num_heads, out, out_shape=None):
    seen = {}

    def fake_dispatch(route, launcher):
        seen["route"] = route
        seen["O"] = launcher.values["O"]

    monkeypatch.setattr(cake, "_dispatch_route", fake_dispatch)
    monkeypatch.setattr(cake, "_target_arch", lambda device: "sm_103a")
    monkeypatch.setattr(cake, "_stream_ptr", lambda device: 0)
    rows = 12
    query = torch.zeros((rows, num_heads, 512), dtype=torch.bfloat16).to(
        torch.float8_e4m3fn
    )
    table = torch.arange(128, dtype=torch.int32).repeat(rows, 1)
    lens = torch.full((rows,), 128, dtype=torch.int32)
    result = cake.run_cake_dsv4(
        query=query,
        swa_kv_cache=torch.zeros((4, 1, 256, 512), dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        ),
        compressed_kv_cache=torch.zeros((8, 1, 64, 512), dtype=torch.bfloat16).to(
            torch.float8_e4m3fn
        ),
        workspace_buffer=_aligned_u8(
            cake.get_cake_dsv4_workspace_bytes(rows, num_heads, 128, torch.float8_e4m3fn)
        ),
        sparse_indices=table,
        sparse_topk_lens=lens,
        out=out,
        bmm1_scale=0.5,
        bmm2_scale=1.0,
        sinks=None,
        max_q_len=1,
        cum_seq_lens_q=None,
        seq_lens=torch.full((rows,), 1000, dtype=torch.int32),
        backend="cake",
        out_shape=out_shape,
    )
    return result, seen


def test_run_cake_dsv4_places_out_none_by_route(monkeypatch):
    result, seen = _run_fp8_swa(monkeypatch, num_heads=64, out=None)
    assert seen["route"] == "fp8_lowhead_prefill"
    assert tuple(result.shape) == (12, 64, 512) and result.data_ptr() % 4096 == 0x800
    assert seen["O"].data_ptr() == result.data_ptr()
    result, seen = _run_fp8_swa(
        monkeypatch, num_heads=64, out=None, out_shape=(12, 1, 64, 512)
    )
    assert tuple(result.shape) == (12, 1, 64, 512) and result.data_ptr() % 4096 == 0x800
    result, seen = _run_fp8_swa(monkeypatch, num_heads=8, out=None)
    assert seen["route"] == "fp8_lowhead_prefill" and tuple(result.shape) == (
        12,
        8,
        512,
    )


def test_run_cake_dsv4_keeps_a_caller_out_as_is(monkeypatch):
    raw = torch.zeros((12 * 64 * 512 + 2048,), dtype=torch.bfloat16)
    start = (
        (0 - raw.data_ptr()) % 4096
    ) // 2  # a 4 KiB-aligned caller buffer = the slow phase
    out = raw[start : start + 12 * 64 * 512].view(12, 64, 512)
    result, seen = _run_fp8_swa(monkeypatch, num_heads=64, out=out)
    assert result is out and result.data_ptr() % 4096 == 0
    with pytest.raises(ValueError):
        _run_fp8_swa(monkeypatch, num_heads=64, out=None, out_shape=(2, 64, 512))
