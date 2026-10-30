"""Source tests (no GPU) of the sm_107a fp32 claim-route bounds
(``_CLAIM_WIDE_BOUNDS``): fp32-state one-chunk packed-varlen ring calls take
the claim program from 1024 items and 4 items per CTA of the persistent grid,
bf16 calls keep the 2048-item / 8-per-CTA rule."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


def _claim(
    module,
    *,
    num_sequences,
    tokens=128,
    nheads=128,
    sm_count=212,
    capability=(10, 7),
    state_dtype=torch.float32,
    variant="",
    mode_varlen=True,
):
    return module.claim_selected(
        mode_varlen=mode_varlen,
        num_sequences=num_sequences,
        seqlen=num_sequences * tokens,
        nheads=nheads,
        sm_count=sm_count,
        capability=capability,
        variant=variant,
        state_dtype=state_dtype,
    )


@pytest.mark.cpu_only
def test_source_claim_wide_bounds_are_rubin_fp32_only():
    module = _module()
    wide_bounds = module._CLAIM_WIDE_BOUNDS
    assert wide_bounds == {(10, 7): {torch.float32: (1024, 4)}}
    # the shared bounds stay those of the bf16 program
    assert module._CLAIM_MIN_ITEMS == {(10, 7): 2048}
    assert module._CLAIM_MIN_ITEMS_PER_CTA == 8
    assert (
        set(module._CLAIM_WIDE_BOUNDS[(10, 7)]) <= module._CLAIM_STATE_DTYPES[(10, 7)]
    )


@pytest.mark.cpu_only
def test_source_claim_fp32_wide_rule_rows():
    """The measured rows of the class on R200 (128 heads, one chunk per
    item)."""

    module = _module()
    # 16 x 128 tokens = 2048 items: not above the shared 2048 bound, admitted
    # by the fp32 bound
    assert _claim(module, num_sequences=16)
    # 8 x 64 tokens = 1024 items, grid 212, 4.8 per CTA
    assert _claim(module, num_sequences=8, tokens=64)
    assert _claim(module, num_sequences=8)
    # 7 x 128 = 896 items: below 1024; 4 x 128 (512 items) too
    assert not _claim(module, num_sequences=7)
    assert not _claim(module, num_sequences=4)
    # calls the shared rule admits keep the program (64 / 160 x 128 items)
    assert _claim(module, num_sequences=64) and _claim(module, num_sequences=160)
    # bf16 keeps its rule: 1024 / 2048 items no, 2176 items yes
    for num_sequences in (8, 16):
        assert not _claim(
            module, num_sequences=num_sequences, state_dtype=torch.bfloat16
        )
    assert _claim(module, num_sequences=17, state_dtype=torch.bfloat16)
    # f16 has no claim program
    for num_sequences in (8, 16, 64):
        assert not _claim(
            module, num_sequences=num_sequences, state_dtype=torch.float16
        )
    # two host-known chunks per item, other schedules, batched calls, other
    # architectures
    assert not _claim(module, num_sequences=16, tokens=129)
    assert _claim(module, num_sequences=16, tokens=1)
    for variant in ("noring", "tail"):
        assert not _claim(module, num_sequences=16, variant=variant)
    assert not _claim(module, num_sequences=16, mode_varlen=False)
    for capability in ((10, 3), (10, 0), None):
        assert not _claim(module, num_sequences=16, capability=capability)


@pytest.mark.cpu_only
def test_source_claim_rule_matches_its_definition():
    """``claim_selected`` over a grid of shapes and SM counts equals the rule
    written out: one host-known chunk per item and either (fp32) at least 1024
    items and 4 per CTA, or (bf16 / fp32) more than 2048 items and 8 per
    CTA."""

    module = _module()
    for sm_count in (148, 152, 212, 257, 330):
        for nheads in (8, 32, 128):
            for num_sequences in range(1, 300, 7):
                for tokens in (1, 64, 128, 129):
                    for state_dtype in (torch.bfloat16, torch.float16, torch.float32):
                        items = num_sequences * nheads
                        grid = module._persistent_grid_size(
                            total_work=items, sm_count=sm_count
                        )
                        one_chunk = -(-tokens // 128) == 1
                        wide = (
                            state_dtype == torch.float32
                            and items >= 1024
                            and items >= 4 * grid
                        )
                        shared = (
                            state_dtype in (torch.bfloat16, torch.float32)
                            and items > 2048
                            and items >= 8 * grid
                        )
                        expected = one_chunk and (wide or shared)
                        got = _claim(
                            module,
                            num_sequences=num_sequences,
                            tokens=tokens,
                            nheads=nheads,
                            sm_count=sm_count,
                            state_dtype=state_dtype,
                        )
                        assert got is expected, (
                            sm_count,
                            nheads,
                            num_sequences,
                            tokens,
                            state_dtype,
                        )


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


def _triple(lengths):
    """The seq_idx triple of a packed batch (every sequence start and every
    128-token chunk boundary inside a sequence starts a segment)."""

    total = sum(lengths)
    seq = []
    for sequence, length in enumerate(lengths):
        seq += [sequence] * length
    chunk_indices, chunk_offsets = [], []
    for chunk in range(-(-total // 128)):
        for offset in range(min(128, total - chunk * 128)):
            token = chunk * 128 + offset
            if offset == 0 or seq[token] != seq[token - 1]:
                chunk_indices.append(chunk)
                chunk_offsets.append(offset)
    return (
        torch.tensor([seq], dtype=torch.int32),
        torch.tensor(chunk_indices, dtype=torch.int32),
        torch.tensor(chunk_offsets, dtype=torch.int32),
    )


def _varlen_program(
    module,
    monkeypatch,
    *,
    lengths,
    nheads=128,
    state_dtype=torch.bfloat16,
    has_z=True,
    has_init=True,
    form="triple",
    arch="sm_107a",
):
    """The program a packed-varlen call of ``lengths`` runs; ``form`` is
    ``"triple"`` (seq_idx triple, the table derived in-kernel) or
    ``"cu_seqlens"``."""

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
        has_varlen=True,
        has_z=has_z,
        seq_idx_dtype=torch.int32,
    )
    seqlen = sum(lengths)
    x = torch.empty((1, seqlen, nheads, 64), dtype=torch.bfloat16)
    dt = torch.empty((1, seqlen, nheads), dtype=torch.float32)
    A = torch.empty((nheads,), dtype=torch.float32)
    B = torch.empty((1, seqlen, 8, 128), dtype=torch.bfloat16)
    kwargs = {}
    if has_init:
        kwargs["initial_states"] = torch.empty(
            (len(lengths), nheads, 64, 128), dtype=state_dtype
        )
    if has_z:
        kwargs["z"] = torch.empty_like(x)
    if form == "cu_seqlens":
        bounds = [0]
        for length in lengths:
            bounds.append(bounds[-1] + length)
        kwargs["cu_seqlens"] = torch.tensor(bounds, dtype=torch.int32)
    else:
        seq_idx, chunk_indices, chunk_offsets = _triple(lengths)
        kwargs.update(
            seq_idx=seq_idx, chunk_indices=chunk_indices, chunk_offsets=chunk_offsets
        )
        if not has_init:
            kwargs["num_seqs"] = len(lengths)
    runner.run(x, dt, A, B, torch.empty_like(B), **kwargs)
    (name,) = calls
    return name


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("num_seqs", "tokens", "state_dtype", "has_z", "program"),
    (
        # 16 x 128 and 8 x 64 tokens, fp32: the claim program
        (16, 128, torch.float32, True, "exactpre_f32_varlen_claim"),
        (8, 64, torch.float32, True, "exactpre_f32_varlen_claim"),
        (16, 128, torch.float32, False, "exactpre_f32_varlen_claim"),
        # 4 x 128 tokens (512 items) keeps the static-order program (or its
        # one-CTA-cluster twin where that rule takes it)
        (4, 128, torch.float32, True, "exactpre_f32_varlen"),
        # bf16 calls of the same shapes keep their programs (no claim)
        (16, 128, torch.bfloat16, True, None),
        (8, 64, torch.bfloat16, True, None),
    ),
)
@pytest.mark.parametrize("form", ("triple", "cu_seqlens"))
def test_source_rubin_fp32_one_chunk_calls_route_to_the_claim_program(
    monkeypatch, num_seqs, tokens, state_dtype, has_z, program, form
):
    module = _module()
    name = _varlen_program(
        module,
        monkeypatch,
        lengths=[tokens] * num_seqs,
        state_dtype=state_dtype,
        has_z=has_z,
        form=form,
    )
    if program is None:
        assert not name.endswith("_claim"), name
    elif program == "exactpre_f32_varlen":
        twin = module.unit_cluster_variant(
            capability=(10, 7),
            family="exactpre",
            variant="",
            state_dtype=state_dtype,
            mode_varlen=True,
            num_sequences=num_seqs,
            nheads=128,
            total_tokens=num_seqs * tokens,
        )
        assert name == program + (f"_{twin}" if twin else "")
    else:
        assert name == program
