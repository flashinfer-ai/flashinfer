"""Source tests (no GPU) of the sm_107a batched long-item pair route
(``_BATCHED_TAIL_WINDOWS``, :func:`batched_tail_selected`): batched calls
without z and without initial states, of a bf16 state, with 768..4096
(sequence, head) items of 16..32 chunks take the ``_tail`` pair program on
R200."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


MIN_ITEMS, MAX_ITEMS, MIN_CHUNKS, MAX_CHUNKS = 768, 4096, 16, 32


def _selected(
    module,
    *,
    items,
    chunks,
    has_z=False,
    has_init=False,
    state_dtype=torch.bfloat16,
    capability=(10, 7),
    nheads=8,
):
    return module.batched_tail_selected(
        capability=capability,
        nheads=nheads,
        num_sequences=items // nheads,
        nchunks=chunks,
        has_z=has_z,
        has_initial_states=has_init,
        state_dtype=state_dtype,
    )


@pytest.mark.cpu_only
def test_source_batched_tail_window_is_rubin_bf16_only():
    module = _module()
    windows = module._BATCHED_TAIL_WINDOWS
    assert windows == {(10, 7): (MIN_ITEMS, MAX_ITEMS, MIN_CHUNKS, MAX_CHUNKS)}
    states = module._BATCHED_TAIL_STATE_DTYPES
    assert states == {(10, 7): frozenset({torch.bfloat16})}
    for items in (MIN_ITEMS, 1024, 2048, MAX_ITEMS):
        for chunks in (MIN_CHUNKS, 24, MAX_CHUNKS):
            assert _selected(module, items=items, chunks=chunks)
            assert not _selected(module, items=items, chunks=chunks, has_z=True)
            assert not _selected(module, items=items, chunks=chunks, has_init=True)
            for state_dtype in (torch.float16, torch.float32):
                assert not _selected(
                    module, items=items, chunks=chunks, state_dtype=state_dtype
                )
            for capability in ((10, 0), (10, 3), None):
                assert not _selected(
                    module, items=items, chunks=chunks, capability=capability
                )
    # bounds inclusive; items are num_sequences x nheads
    assert not _selected(module, items=MIN_ITEMS - 8, chunks=MIN_CHUNKS)
    assert not _selected(module, items=MAX_ITEMS + 8, chunks=MIN_CHUNKS)
    assert not _selected(module, items=MIN_ITEMS, chunks=MIN_CHUNKS - 1)
    assert not _selected(module, items=MIN_ITEMS, chunks=MAX_CHUNKS + 1)
    assert _selected(module, items=1024, chunks=16, nheads=128)
    assert _selected(module, items=1024, chunks=16, nheads=32)
    assert not _selected(module, items=640, chunks=16, nheads=128)


def _rubin(module, monkeypatch, calls, arch="sm_107a"):
    """The device answers as ``arch`` (212 SMs on sm_107a); launches are
    captured, nothing runs."""

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: arch)
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(
        module, "_sm_count", lambda _: {"sm_100a": 148, "sm_103a": 152}.get(arch, 212)
    )
    monkeypatch.setattr(
        module, "_launch_program", lambda name, _arch, **kwargs: calls.append(name)
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=0x1234)
    )


def _batched_program(
    module,
    monkeypatch,
    *,
    batch,
    seqlen,
    nheads=8,
    state_dtype=torch.bfloat16,
    has_z=False,
    has_init=False,
    arch="sm_107a",
):
    """The program a batched call of ``batch`` sequences of ``seqlen`` tokens
    runs."""

    calls = []
    _rubin(module, monkeypatch, calls, arch)
    runner = module.CakeSSDCombined(
        128,
        nheads,
        64,
        128,
        8,
        io_dtype=torch.bfloat16,
        state_dtype=state_dtype,
        has_d=False,
        d_has_hdim=False,
        has_initial_states=has_init,
        has_varlen=False,
        has_z=has_z,
        seq_idx_dtype=torch.int32,
    )
    x = torch.empty((batch, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((batch, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((batch, seqlen, 8, 128), dtype=torch.bfloat16)
    kwargs = {}
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if has_init:
        kwargs["initial_states"] = torch.empty(
            (batch, nheads, 64, 128), dtype=state_dtype
        )
    runner.run(x, dt, A, B, torch.empty_like(B), **kwargs)
    (name,) = calls
    return name


@pytest.mark.cpu_only
def test_source_rubin_batched_long_calls_take_the_pair_program(monkeypatch):
    """In-window batched bf16 calls without z and initial states take the
    ``_tail`` program; calls with z or initial states, other state dtypes,
    calls outside the window and other architectures run what they run
    without the rule."""

    module = _module()
    windows = module._BATCHED_TAIL_WINDOWS

    def both(batch, chunks, **kwargs):
        on = _batched_program(
            module, monkeypatch, batch=batch, seqlen=chunks * 128, **kwargs
        )
        monkeypatch.setattr(module, "_BATCHED_TAIL_WINDOWS", {})
        try:
            off = _batched_program(
                module, monkeypatch, batch=batch, seqlen=chunks * 128, **kwargs
            )
        finally:
            monkeypatch.setattr(module, "_BATCHED_TAIL_WINDOWS", windows)
        return on, off

    for items in (MIN_ITEMS, MAX_ITEMS):
        for chunks in (MIN_CHUNKS, MAX_CHUNKS):
            on, off = both(items // 8, chunks)
            assert (on, off) == (
                "exactpre_bf16_batched_tail",
                "exactpre_bf16_batched",
            ), (items, chunks)
            for kwargs in (
                {"has_z": True},
                {"has_init": True},
                {"state_dtype": torch.float32},
                {"state_dtype": torch.float16},
                {"arch": "sm_103a"},
                {"arch": "sm_100a"},
            ):
                on, off = both(items // 8, chunks, **kwargs)
                assert on == off, (items, chunks, kwargs, on, off)
    for items, chunks in (
        (MIN_ITEMS - 8, MIN_CHUNKS),
        (MAX_ITEMS + 8, MAX_CHUNKS),
        (MIN_ITEMS, MIN_CHUNKS - 1),
        (MAX_ITEMS, MAX_CHUNKS + 1),
    ):
        on, off = both(items // 8, chunks)
        assert on == off, (items, chunks, on, off)
    # a pair call the full-grid admission already takes keeps it
    assert _batched_program(module, monkeypatch, batch=64, seqlen=16 * 128) == (
        "exactpre_bf16_batched_tail"
    )
