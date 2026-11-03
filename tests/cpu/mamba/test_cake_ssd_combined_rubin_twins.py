"""Source tests (no GPU) of two sm_107a program choices of the same code:

* the fused batched z-prefetch programs (``exactfused_<state>_batched_zpf``,
  :func:`fused_z_prefetch_selected`): fused batched calls with z of at least
  two chunks per item, bf16 / f32 state;
* the one-CTA-cluster twins (``exactpre_<state>_varlen_unit``,
  ``exactone_<state>_varlen_tablefirstunit``,
  ``exactone_f32_varlen_tablerolesunit``, :func:`unit_cluster_variant`,
  ``_UNIT_CLUSTER_RULES``)."""

import importlib
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


TWINS = {
    "exactpre_bf16_varlen_unit": "exactpre_bf16_varlen",
    "exactpre_f32_varlen_unit": "exactpre_f32_varlen",
    "exactone_bf16_varlen_tablefirstunit": "exactone_bf16_varlen_tablefirst",
    "exactone_f32_varlen_tablefirstunit": "exactone_f32_varlen_tablefirst",
    "exactone_f32_varlen_tablerolesunit": "exactone_f32_varlen_tableroles",
    "exactfused_bf16_batched_zpf": "exactfused_bf16_batched",
    "exactfused_f32_batched_zpf": "exactfused_f32_batched",
}


@pytest.mark.cpu_only
def test_source_twin_programs_bind_their_base_program_launch():
    """Each new program is sm_107a-only and launches exactly as the program it
    stands beside (family, host template, arguments, SMEM, cluster shape,
    preprocess, state code, dependent launch) from a module of its own."""

    module = _module()
    assert set(TWINS) <= module._SM107_ONLY_PROGRAMS
    for name, base in TWINS.items():
        twin = module._arch_program(name, "sm_107a")
        plain = module._arch_program(base, "sm_107a")
        assert twin.main.module == module._SM107_MODULES[name]
        assert plain.main.module == module._SM107_MODULES[base]
        assert twin.main.module != plain.main.module
        assert twin.family == plain.family
        assert twin.host_template == plain.host_template
        assert twin.main_args == plain.main_args
        assert twin.main_smem_bytes == plain.main_smem_bytes
        assert twin.cluster_dims == plain.cluster_dims == (1, 1, 1)
        assert twin.preprocess == plain.preprocess
        assert twin.preprocess_args == plain.preprocess_args
        assert (twin.state_dtype_code, twin.state_dtype_bits) == (
            plain.state_dtype_code,
            plain.state_dtype_bits,
        )
        assert twin.main.threads == plain.main.threads
        assert module._main_pdl(name, "sm_107a") == module._main_pdl(base, "sm_107a")


@pytest.mark.cpu_only
def test_source_fused_z_prefetch_rule():
    module = _module()
    capabilities = module._FUSED_Z_PREFETCH_CAPABILITIES
    assert capabilities == frozenset({(10, 7)})
    states = module._FUSED_Z_PREFETCH_STATE_DTYPES
    assert states == frozenset({torch.bfloat16, torch.float32})
    min_chunks = module._FUSED_Z_PREFETCH_MIN_CHUNKS
    assert min_chunks == 2

    def selected(**kwargs):
        args = dict(
            capability=(10, 7),
            state_dtype=torch.bfloat16,
            mode_varlen=False,
            has_z=True,
            nchunks=2,
        )
        args.update(kwargs)
        return module.fused_z_prefetch_selected(**args)

    for nchunks in (2, 3, 8, 16):
        assert selected(nchunks=nchunks)
        assert selected(nchunks=nchunks, state_dtype=torch.float32)
    assert not selected(nchunks=1)
    assert not selected(has_z=False)
    assert not selected(mode_varlen=True)
    assert not selected(state_dtype=torch.float16)
    for capability in ((10, 0), (10, 3), None):
        assert not selected(capability=capability)


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
@pytest.mark.parametrize("state_dtype", (torch.bfloat16, torch.float32, torch.float16))
def test_source_rubin_fused_batched_calls_with_z_take_the_prefetch_program(
    monkeypatch, state_dtype
):
    module = _module()
    state = {torch.bfloat16: "bf16", torch.float32: "f32", torch.float16: "f16"}[
        state_dtype
    ]
    prefetch = state_dtype != torch.float16
    # 4 x 128 and 32 x 8 heads (more items than SMs: never chunk-parallel)
    # inside the fused window (at most 2048 chunk-items)
    for batch, nheads in ((4, 128), (32, 8)):
        for seqlen, chunks in ((300, 3), (256, 2), (1024, 8), (128, 1), (100, 1)):
            if batch * nheads * chunks > 2048:
                continue
            for has_init in (False, True):
                name = _batched_program(
                    module,
                    monkeypatch,
                    batch=batch,
                    seqlen=seqlen,
                    nheads=nheads,
                    state_dtype=state_dtype,
                    has_z=True,
                    has_init=has_init,
                )
                want = f"exactfused_{state}_batched" + (
                    "_zpf" if prefetch and chunks >= 2 else ""
                )
                assert name == want, (nheads, seqlen, has_init, name)
        # without z: the ring (or start-schedule) program
        name = _batched_program(
            module,
            monkeypatch,
            batch=batch,
            seqlen=300,
            nheads=nheads,
            state_dtype=state_dtype,
        )
        assert name.startswith(f"exactfused_{state}_batched") and "zpf" not in name
        # other architectures
        for arch in ("sm_100a", "sm_103a"):
            name = _batched_program(
                module,
                monkeypatch,
                batch=batch,
                seqlen=300,
                nheads=nheads,
                state_dtype=state_dtype,
                has_z=True,
                arch=arch,
            )
            assert name == f"exactfused_{state}_batched", (arch, name)
    # a fused varlen call with z keeps its program
    name = _varlen_program(
        module,
        monkeypatch,
        lengths=[300] * 4,
        nheads=8,
        state_dtype=state_dtype,
        form="cu_seqlens",
    )
    assert "zpf" not in name, name


# The one-CTA-cluster rule (generated from the final rule)
UNIT_STATES = frozenset({torch.bfloat16, torch.float32})
UNIT_EXACTPRE_MAX_ITEMS = 384
UNIT_TABLEFIRST_MAX_SEQUENCES = 1
UNIT_TABLEFIRST_MAX_HEADS = 8
UNIT_TABLEROLES_STATES = frozenset({torch.float32})
UNIT_TABLEROLES_MIN_MEAN_TOKENS = 128
UNIT_TABLEROLES_MAX_SEQUENCES = None


def _within(value, bound, low=False):
    return bound is None or (value >= bound if low else value <= bound)


def _expected_twin(family, variant, state_dtype, num_sequences, nheads, tokens):
    """The rule written out: the variant the call runs on sm_107a."""

    sequences = max(num_sequences, 1)
    if family == "exactpre" and variant == "":
        ok = state_dtype in UNIT_STATES and _within(
            sequences * nheads, UNIT_EXACTPRE_MAX_ITEMS
        )
        return "unit" if ok else variant
    if family == "exactone" and variant == "tablefirst":
        ok = (
            state_dtype in UNIT_STATES
            and _within(sequences, UNIT_TABLEFIRST_MAX_SEQUENCES)
            and _within(nheads, UNIT_TABLEFIRST_MAX_HEADS)
        )
        return "tablefirstunit" if ok else variant
    if family == "exactone" and variant == "tableroles":
        ok = (
            state_dtype in UNIT_TABLEROLES_STATES
            and _within(
                tokens,
                None
                if UNIT_TABLEROLES_MIN_MEAN_TOKENS is None
                else UNIT_TABLEROLES_MIN_MEAN_TOKENS * sequences,
                low=True,
            )
            and _within(sequences, UNIT_TABLEROLES_MAX_SEQUENCES)
        )
        return "tablerolesunit" if ok else variant
    return variant


@pytest.mark.cpu_only
def test_source_unit_cluster_rule_is_rubin_only():
    module = _module()
    rules = module._UNIT_CLUSTER_RULES
    assert rules == {
        (10, 7): module._UnitClusterRule(
            state_dtypes=UNIT_STATES,
            exactpre_max_items=UNIT_EXACTPRE_MAX_ITEMS,
            tablefirst_max_sequences=UNIT_TABLEFIRST_MAX_SEQUENCES,
            tablefirst_max_heads=UNIT_TABLEFIRST_MAX_HEADS,
            tableroles_state_dtypes=UNIT_TABLEROLES_STATES,
            tableroles_min_mean_tokens=UNIT_TABLEROLES_MIN_MEAN_TOKENS,
            tableroles_max_sequences=UNIT_TABLEROLES_MAX_SEQUENCES,
        )
    }
    # the twins exist for exactly the admitted (family, state) pairs
    assert UNIT_STATES.issubset({torch.bfloat16, torch.float32})
    assert UNIT_TABLEROLES_STATES.issubset({torch.float32})


@pytest.mark.cpu_only
def test_source_unit_cluster_variant_matches_its_definition():
    module = _module()
    for family, variant in (
        ("exactpre", ""),
        ("exactpre", "claim"),
        ("exactpre", "ibulk"),
        ("exactpre", "tail"),
        ("exactpre", "tailone"),
        ("exactone", ""),
        ("exactone", "noring"),
        ("exactone", "tablefirst"),
        ("exactone", "tableroles"),
        ("exactfused", ""),
        ("exactfused", "ibulk"),
        ("exact", ""),
    ):
        for state_dtype in (torch.bfloat16, torch.float16, torch.float32):
            for num_sequences in (0, 1, 2, 3, 4, 5, 8, 64):
                for nheads in (1, 8, 16, 32, 64, 128):
                    for mean in (1, 32, 64, 96, 127, 128, 129, 384, 1024):
                        tokens = max(num_sequences, 1) * mean
                        kwargs = dict(
                            family=family,
                            variant=variant,
                            state_dtype=state_dtype,
                            num_sequences=num_sequences,
                            nheads=nheads,
                            total_tokens=tokens,
                        )
                        got = module.unit_cluster_variant(
                            capability=(10, 7), mode_varlen=True, **kwargs
                        )
                        assert got == _expected_twin(
                            family, variant, state_dtype, num_sequences, nheads, tokens
                        ), kwargs
                        # batched calls and other architectures keep the variant
                        assert (
                            module.unit_cluster_variant(
                                capability=(10, 7), mode_varlen=False, **kwargs
                            )
                            == variant
                        )
                        for capability in ((10, 0), (10, 3), None):
                            assert (
                                module.unit_cluster_variant(
                                    capability=capability, mode_varlen=True, **kwargs
                                )
                                == variant
                            )


@pytest.mark.cpu_only
@pytest.mark.parametrize(
    ("state", "num_seqs", "tokens", "nheads", "form", "plain"),
    (
        # two-kernel static-order calls (outside the fused window)
        ("bf16", 2, 384, 128, "triple", "exactpre_bf16_varlen"),
        ("f32", 2, 384, 128, "triple", "exactpre_f32_varlen"),
        ("bf16", 3, 384, 128, "cu_seqlens", "exactpre_bf16_varlen"),
        ("f32", 3, 128, 128, "triple", "exactpre_f32_varlen"),
        ("bf16", 4, 384, 128, "triple", "exactpre_bf16_varlen"),
        ("f32", 4, 384, 128, "cu_seqlens", "exactpre_f32_varlen"),
        ("bf16", 4, 128, 128, "triple", "exactpre_bf16_varlen"),
        ("bf16", 8, 384, 128, "triple", "exactpre_bf16_varlen"),
        ("f32", 8, 384, 128, "triple", "exactpre_f32_varlen"),
        # one-launch calls (seq_idx triple): tablefirst below 128 CTAs,
        # tableroles from it on (the single-sequence calls among them run the
        # fused batched program with the seq_idx check CTA,
        # :func:`single_sequence_batched_variant`; their twin is judged with
        # that rule off)
        ("bf16", 1, 128, 8, "triple", "exactone_bf16_varlen_tablefirst"),
        ("bf16", 1, 127, 8, "triple", "exactone_bf16_varlen_tablefirst"),
        ("bf16", 1, 128, 32, "triple", "exactone_bf16_varlen_tablefirst"),
        ("bf16", 1, 128, 64, "triple", "exactone_bf16_varlen_tablefirst"),
        ("f32", 1, 128, 8, "triple", "exactone_f32_varlen_tablefirst"),
        ("bf16", 2, 128, 32, "triple", "exactone_bf16_varlen_tablefirst"),
        ("f32", 1, 128, 128, "triple", "exactone_f32_varlen_tableroles"),
        ("f32", 1, 64, 128, "triple", "exactone_f32_varlen_tableroles"),
        ("f32", 2, 128, 128, "triple", "exactone_f32_varlen_tableroles"),
        ("f32", 2, 96, 128, "triple", "exactone_f32_varlen_tableroles"),
        ("f32", 2, 192, 128, "triple", "exactone_f32_varlen_tableroles"),
        ("bf16", 2, 128, 128, "triple", "exactone_bf16_varlen_tableroles"),
    ),
)
def test_source_rubin_varlen_calls_take_the_twin_by_the_rule(
    monkeypatch, state, num_seqs, tokens, nheads, form, plain
):
    module = _module()
    state_dtype = {"bf16": torch.bfloat16, "f32": torch.float32}[state]
    rules = module._UNIT_CLUSTER_RULES

    def run(**kwargs):
        return _varlen_program(
            module,
            monkeypatch,
            lengths=[tokens] * num_seqs,
            nheads=nheads,
            state_dtype=state_dtype,
            form=form,
            **kwargs,
        )

    if num_seqs == 1 and form == "triple":
        assert run() == f"exactfused_{state}_batched_chk"
        monkeypatch.setattr(module, "single_sequence_batched_variant", lambda **_: None)
    monkeypatch.setattr(module, "_UNIT_CLUSTER_RULES", {})
    try:
        off = run()
    finally:
        monkeypatch.setattr(module, "_UNIT_CLUSTER_RULES", rules)
    assert off == plain
    family, _state, _mode, *variant = plain.split("_")
    twin = _expected_twin(
        family,
        variant[0] if variant else "",
        state_dtype,
        num_seqs,
        nheads,
        num_seqs * tokens,
    )
    on = run()
    assert on == f"{family}_{state}_varlen" + (f"_{twin}" if twin else ""), on
    # the other architectures never take a twin
    for arch in ("sm_100a", "sm_103a"):
        assert not run(arch=arch).endswith("unit")
