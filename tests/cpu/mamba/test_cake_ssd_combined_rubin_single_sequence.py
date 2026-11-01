"""Source tests (no GPU) of the sm_107a fused batched programs with the
appended seq_idx check CTA (``exactfused_<state>_batched_chk`` and
``_zpfchk``) and their rule (:func:`single_sequence_batched_variant`):
single-sequence packed-varlen calls (batch 1) in the seq_idx triple form
whose table the device derives run the fused batched program on their
``[1, T]`` view in one launch.  The cu_seqlens form and caller tables keep
their program."""

import hashlib
import importlib
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

STATES = {"bf16": torch.bfloat16, "f32": torch.float32}
BASE = {
    ("bf16", "chk"): "exactfused_bf16_batched",
    ("f32", "chk"): "exactfused_f32_batched",
    ("bf16", "zpfchk"): "exactfused_bf16_batched_zpf",
    ("f32", "zpfchk"): "exactfused_f32_batched_zpf",
}


def _module():
    return importlib.import_module("flashinfer.mamba.cake_ssd_combined")


_RULE = _module().single_sequence_batched_variant


def _device_index(kw):
    return kw["seq_idx"].device.index


def _device_dir():
    return (
        Path(_module().__file__).resolve().parents[2]
        / "csrc"
        / "cake_mamba_ssd_combined"
        / "generated"
        / "device"
    )


@pytest.fixture(autouse=True)
def _no_override(monkeypatch):
    monkeypatch.delenv("FLASHINFER_CAKE_SSD_CHUNK_PARALLEL", raising=False)


@pytest.mark.cpu_only
def test_programs_and_units():
    module = _module()
    for (state, variant), base in BASE.items():
        name = f"exactfused_{state}_batched_{variant}"
        assert name in module._SM107_MODULES
        assert name in module._SM107_ONLY_PROGRAMS
        program = module._PROGRAMS[name]
        assert program.family == "exactfused" and program.variant == variant
        assert (
            program.main_smem_bytes
            == module._PROGRAMS[base].main_smem_bytes
            == module._EXACTFUSED_SMEM_BYTES
        )
        assert program.cluster_dims == (1, 1, 1)
        assert program.main_args == module._PROGRAMS[base].main_args
        assert not module._main_pdl(name, module._SM107_ARCH)
        mod = module._SM107_MODULES[name]
        assert module._arch_program(name, module._SM107_ARCH).main.module == mod
        assert (
            program.main.kernel
            == f"kernel_mamba_ssd_q_tmem_alias_{state}_batched_{variant}"
        )
        data = (_device_dir() / f"{mod}.cu").read_bytes()
        assert hashlib.sha256(data).hexdigest()[:10] == mod[-10:]
        text = data.decode()
        base_text = (_device_dir() / f"{module._SM107_MODULES[base]}.cu").read_text()
        # the check CTA flags the status word; the base unit writes none
        assert text.count("preprocess_status[0] = 1;") == 1
        assert "preprocess_status[0] = 1;" not in base_text
        assert module._program_name("exactfused", STATES[state], False, variant) == name
    assert "exactfused_f16_batched_chk" not in module._PROGRAMS


def _want(
    module, cap, batch, nseq, derives, ckpt, dtype, nheads, nchunks, has_z, family
):
    if cap != (10, 7) or batch != 1 or nseq != 1 or not derives or ckpt:
        return None
    if dtype not in STATES.values() or family in ("chunkpar", "chunkpar3"):
        return None
    if nheads >= 212:
        return None
    exact = module.chunk_parallel_family(
        nheads=nheads,
        num_sequences=1,
        num_segments=nchunks,
        nchunks=nchunks,
        mode_varlen=False,
        sm_count=212,
        capability=cap,
        mode="auto",
    )
    if exact != "exact":
        return None
    # the batched fused window
    if not (nheads <= 1024 and nchunks <= 16 and nheads * nchunks <= 2048):
        return None
    return "zpfchk" if has_z and nchunks >= 2 else "chk"


@pytest.mark.cpu_only
def test_rule_is_its_definition():
    module = _module()
    families = ("exactpre", "exactone", "exactfused", "chunkpar", "chunkpar3")
    for cap in ((10, 7), (10, 0), (10, 3), None):
        for batch in (1, 2):
            for nseq in (1, 2):
                for derives in (False, True):
                    for ckpt in (0, 1):
                        for dtype in (torch.bfloat16, torch.float32, torch.float16):
                            for nheads in (1, 8, 32, 128, 211, 212, 256):
                                for nchunks in (1, 2, 8, 16, 17, 32):
                                    for has_z in (False, True):
                                        for family in families:
                                            kw = dict(
                                                capability=cap,
                                                mode_varlen=True,
                                                batch=batch,
                                                num_sequences=nseq,
                                                derives_table=derives,
                                                checkpoint_state_count=ckpt,
                                                state_dtype=dtype,
                                                nheads=nheads,
                                                nchunks=nchunks,
                                                sm_count=212,
                                                has_z=has_z,
                                                varlen_family=family,
                                                chunk_parallel_mode="auto",
                                            )
                                            want = _want(
                                                module,
                                                cap,
                                                batch,
                                                nseq,
                                                derives,
                                                ckpt,
                                                dtype,
                                                nheads,
                                                nchunks,
                                                has_z,
                                                family,
                                            )
                                            got = (
                                                module.single_sequence_batched_variant(
                                                    **kw
                                                )
                                            )
                                            assert got == want, kw
    kw = dict(
        capability=(10, 7),
        mode_varlen=True,
        batch=1,
        num_sequences=1,
        derives_table=True,
        checkpoint_state_count=0,
        state_dtype=torch.bfloat16,
        nheads=128,
        nchunks=16,
        sm_count=212,
        has_z=True,
        varlen_family="exactpre",
        chunk_parallel_mode="auto",
    )
    assert module.single_sequence_batched_variant(**kw) == "zpfchk"
    assert module.single_sequence_batched_variant(**dict(kw, has_z=False)) == "chk"
    assert module.single_sequence_batched_variant(**dict(kw, mode_varlen=False)) is None
    assert module.single_sequence_batched_variant(**dict(kw, batch=2)) is None
    assert module.single_sequence_batched_variant(**dict(kw, sm_count=128)) is None
    assert (
        module.single_sequence_batched_variant(**dict(kw, derives_table=False)) is None
    )


def _rubin(module, monkeypatch, launches, arch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_target_arch", lambda *_: arch)
    monkeypatch.setattr(module, "_cuda_device_index", lambda _: 0)
    monkeypatch.setattr(
        module, "_sm_count", lambda _: {"sm_100a": 148, "sm_103a": 152}.get(arch, 212)
    )
    monkeypatch.setattr(
        module,
        "_launch_program",
        lambda name, _arch, **kw: launches.append((name, kw)),
    )
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda *_: SimpleNamespace(cuda_stream=0x1234)
    )


def _varlen(
    module,
    monkeypatch,
    *,
    lengths,
    nheads=128,
    state_dtype=torch.bfloat16,
    has_z=True,
    has_init=True,
    form="triple",
    int64=False,
    arch="sm_107a",
    caller_cumsum=False,
    checkpoint=False,
    batch=1,
    rule=True,
):
    """The program and launch of a packed-varlen call of ``lengths``;
    ``rule=False`` runs it with :func:`single_sequence_batched_variant`
    off."""

    launches = []
    _rubin(module, monkeypatch, launches, arch)
    monkeypatch.setattr(
        module,
        "single_sequence_batched_variant",
        _RULE if rule else (lambda **_: None),
    )
    T, S = sum(lengths), len(lengths)
    runner = module.CakeSSDCombined(
        128,
        nheads,
        64,
        128,
        8,
        io_dtype=torch.bfloat16,
        state_dtype=state_dtype,
        has_d=True,
        d_has_hdim=False,
        has_initial_states=has_init,
        has_varlen=True,
        has_z=has_z,
        seq_idx_dtype=torch.int64 if int64 else torch.int32,
    )
    x = torch.empty((batch, T, nheads, 64), dtype=torch.bfloat16)
    B = torch.empty((batch, T, 8, 128), dtype=torch.bfloat16)
    kw = dict(D=torch.empty((nheads,), dtype=torch.bfloat16), dt_softplus=True)
    if has_z:
        kw["z"] = torch.empty_like(x)
    if has_init:
        kw["initial_states"] = torch.empty((S, nheads, 64, 128), dtype=state_dtype)
    else:
        kw["num_seqs"] = S
    if form == "cu":
        cu = [0]
        for length in lengths:
            cu.append(cu[-1] + length)
        kw["cu_seqlens"] = torch.tensor(cu, dtype=torch.int32)
    else:
        ids = torch.cat(
            [
                torch.full((n,), s, dtype=torch.int64 if int64 else torch.int32)
                for s, n in enumerate(lengths)
            ]
        )
        kw["seq_idx"] = ids.view(1, T).expand(batch, T).contiguous()
        ci, co, starts = [], [], []
        for c in range(-(-T // 128)):
            vals = ids[c * 128 : c * 128 + 128]
            prev = torch.cat([vals[:1] - 1, vals[:-1]])
            for off in (vals != prev).nonzero(as_tuple=True)[0].tolist():
                ci.append(c)
                co.append(off)
                starts.append(c * 128 + off)
        kw["chunk_indices"] = torch.tensor(ci, dtype=torch.int32)
        kw["chunk_offsets"] = torch.tensor(co, dtype=torch.int32)
        if caller_cumsum:
            bounds = [0]
            for length in lengths:
                bounds.append(bounds[-1] + length)
            first = [
                next((i for i, s in enumerate(starts) if s >= bounds[q]), len(starts))
                for q in range(S)
            ]
            kw["seq_chunk_cumsum"] = torch.tensor(
                first + [len(starts)], dtype=torch.int32
            )
    if checkpoint:
        kw.update(
            checkpoint_token_indices=torch.full((S,), -1, dtype=torch.int32),
            checkpoint_state_slots=torch.full((S,), -1, dtype=torch.int32),
            checkpoint_states=torch.empty((1, nheads, 64, 128), dtype=state_dtype),
        )
    runner.run(
        x,
        torch.empty((batch, T, nheads)),
        torch.empty((nheads,)),
        B,
        torch.empty_like(B),
        **kw,
    )
    ((name, launch),) = launches
    return name, launch, dict(kw, runner=runner)


@pytest.mark.cpu_only
def test_route_through_the_runner(monkeypatch):
    module = _module()
    for state, dtype in STATES.items():
        for T in (1, 8, 127, 128, 256, 1000, 1024, 2048):
            for has_z in (False, True):
                for form, int64 in (("triple", False), ("triple", True), ("cu", False)):
                    call = dict(
                        lengths=[T],
                        state_dtype=dtype,
                        has_z=has_z,
                        form=form,
                        int64=int64,
                    )
                    name, launch, kw = _varlen(module, monkeypatch, **call)
                    off, off_launch, _ = _varlen(
                        module, monkeypatch, rule=False, **call
                    )
                    assert "_varlen" in off and "chk" not in off, off
                    if form == "cu":
                        # the cu_seqlens form keeps its program and launch
                        assert name == off, (state, T, has_z, name, off)
                        for field in ("main_grid", "preprocess_grid"):
                            assert launch[field] == off_launch[field]
                        for field in ("metadata_from_cu_seqlens", "mode_varlen"):
                            assert launch["main"][field] == off_launch["main"][field]
                        continue
                    chunks = -(-T // 128)
                    variant = "zpfchk" if has_z and chunks >= 2 else "chk"
                    assert name == f"exactfused_{state}_batched_{variant}", (
                        state,
                        T,
                        has_z,
                        int64,
                        name,
                    )
                    # one launch: the persistent grid of 128 items + the check CTA
                    assert launch["main_grid"] == (128 + 1, 1, 1)
                    assert launch["preprocess_grid"][0] == 0
                    main = launch["main"]
                    assert main["mode_varlen"] == 0
                    assert main["sequence_count"] == 1 and main["batch"] == 1
                    assert main["seqlen"] == T and main["nchunks"] == chunks
                    assert main["metadata_from_cu_seqlens"] == (3 if int64 else 2)
                    # the check CTA reads the caller's triple and flags the
                    # runner's status word
                    assert main["chunk_indices"] is kw["chunk_indices"]
                    assert main["chunk_offsets"] is kw["chunk_offsets"]
                    assert main["num_logical_chunks"] == len(kw["chunk_indices"])
                    seq = main["seq_idx_i64" if int64 else "seq_idx_i32"]
                    assert seq.data_ptr() == kw["seq_idx"].data_ptr()
                    assert (
                        main["preprocess_status"]
                        is kw["runner"]._preprocess_status[_device_index(kw)]
                    )
    # calls that keep their program
    for kwargs in (
        dict(lengths=[1024, 1024]),
        dict(lengths=[2048], caller_cumsum=True),
        dict(lengths=[2048], checkpoint=True),
        dict(lengths=[2048], state_dtype=torch.float16),
        dict(lengths=[4096]),
        dict(lengths=[2049]),
        dict(lengths=[2048], arch="sm_100a"),
        dict(lengths=[2048], arch="sm_103a"),
        dict(lengths=[2048], nheads=8, has_z=False),
        dict(lengths=[2048], form="cu"),
        dict(lengths=[1000], form="cu"),
        dict(lengths=[128], form="cu", has_z=False),
        dict(lengths=[1024, 1024], form="cu"),
    ):
        on = _varlen(module, monkeypatch, **kwargs)[0]
        off = _varlen(module, monkeypatch, rule=False, **kwargs)[0]
        assert on == off and "chk" not in on, (kwargs, on, off)
    # a triple-form call with batch 2 and one sequence is never routed (the
    # same program, or the same refusal, with the rule on and off)
    for T in (128, 1000, 2048):
        outcome = []
        for rule in (True, False):
            try:
                outcome.append(
                    _varlen(module, monkeypatch, lengths=[T], batch=2, rule=rule)[0]
                )
            except Exception as exc:  # noqa: BLE001
                outcome.append(f"{type(exc).__name__}: {exc}")
        assert outcome[0] == outcome[1] and "chk" not in outcome[0], (T, outcome)


@pytest.mark.cpu_only
@pytest.mark.parametrize("state", tuple(STATES))
def test_checkpoint_outputs_under_prepare_and_capture(monkeypatch, state):
    """The rule reads the call's checkpoint outputs, which the workspace key
    does not: one call shape runs the check program without them and its own
    program with them.  ``prepare`` and an eager run of the check program
    load (build) both programs; a captured call whose check program has not
    been launched in this process keeps the shape's launched program (bitwise
    the same results and status word); a program that is loaded but never
    launched never passes the capture gate; a captured call with checkpoint
    outputs whose program has never been launched is refused, saying that
    checkpoint outputs select a different program."""

    module = _module()
    dtype = STATES[state]
    launches, loads = [], []
    _rubin(module, monkeypatch, launches, "sm_107a")
    monkeypatch.setattr(module, "_LOADED_PROGRAMS", set())

    def launch(name, arch, **kw):
        launches.append((name, kw))
        module._LOADED_PROGRAMS.add((name, arch))

    def load(name, arch):
        loads.append((name, arch))
        return SimpleNamespace(run=None)

    monkeypatch.setattr(module, "_launch_program", launch)
    monkeypatch.setattr(module, "_load_generated_program", load)
    capturing = {"value": False}
    monkeypatch.setattr(module, "_stream_capturing", lambda _d: capturing["value"])
    T, nheads = 1024, 128
    x = torch.empty((1, T, nheads, 64), dtype=torch.bfloat16)
    B = torch.empty((1, T, 8, 128), dtype=torch.bfloat16)
    inputs = dict(
        x=x,
        dt=torch.empty((1, T, nheads)),
        A=torch.empty((nheads,)),
        B=B,
        C=torch.empty_like(B),
        D=torch.empty((nheads,), dtype=torch.bfloat16),
        z=torch.empty_like(x),
        dt_softplus=True,
        initial_states=torch.empty((1, nheads, 64, 128), dtype=dtype),
        seq_idx=torch.zeros((1, T), dtype=torch.int32),
        chunk_indices=torch.arange(T // 128, dtype=torch.int32),
        chunk_offsets=torch.zeros((T // 128,), dtype=torch.int32),
    )
    checkpoints = dict(
        checkpoint_token_indices=torch.full((1,), -1, dtype=torch.int32),
        checkpoint_state_slots=torch.full((1,), -1, dtype=torch.int32),
        checkpoint_states=torch.empty((1, nheads, 64, 128), dtype=dtype),
    )
    check = f"exactfused_{state}_batched_zpfchk"
    refused = module.CakeSSDCombinedCaptureError
    different = "checkpoint outputs select a different program"

    def runner():
        return module.CakeSSDCombined(
            128,
            nheads,
            64,
            128,
            8,
            io_dtype=torch.bfloat16,
            state_dtype=dtype,
            has_d=True,
            d_has_hdim=False,
            has_initial_states=True,
            has_varlen=True,
            has_z=True,
            seq_idx_dtype=torch.int32,
        )

    def run(cake, capture, call=inputs, **extra):
        capturing["value"] = capture
        try:
            cake.run(**call, **extra)
        finally:
            capturing["value"] = False
        return launches[-1]

    # Warm with checkpoint outputs, capture without: the shape's own
    # program (launched) serves the captured call.
    first = runner()
    own, own_launch = run(first, False, **checkpoints)
    assert "chk" not in own and own_launch["main"]["checkpoint_state_count"] == 1
    assert loads == []  # the own program loads in its launch
    name, launch_values = run(first, True)
    assert name == own and launch_values["main"]["checkpoint_state_count"] == 0
    assert launch_values["main"]["mode_varlen"] == 1
    assert launch_values["main"]["metadata_from_cu_seqlens"] == 0
    assert launch_values["main_grid"] == own_launch["main_grid"]
    assert loads == []  # nothing loads under capture
    # The eager check-program run also loads the shape's own program (load
    # only, no extra launch); afterwards captured calls take the check program.
    count = len(launches)
    assert run(first, False)[0] == check
    assert len(launches) == count + 1 and loads == [(own, "sm_107a")]
    name, launch_values = run(first, True)
    assert name == check and launch_values["main"]["metadata_from_cu_seqlens"] == 2
    assert launch_values["main_grid"] == (nheads + 1, 1, 1)
    assert run(first, True, **checkpoints)[0] == own

    # Warm without checkpoint outputs, capture with them: the shape's own
    # program is loaded but has never been launched -- refused, and the
    # message names the checkpoint outputs.
    monkeypatch.setattr(module, "_LOADED_PROGRAMS", set())
    loads.clear()
    second = runner()
    assert run(second, False)[0] == check
    assert loads == [(own, "sm_107a")]
    count = len(launches)
    with pytest.raises(refused, match=f"program {own} .*{different}"):
        run(second, True, **checkpoints)
    assert len(launches) == count
    # without checkpoint outputs the launched check program still serves
    assert run(second, True)[0] == check

    # prepare loads both programs of the shape (the call it plans has no
    # checkpoint outputs) and launches neither: loaded is not launched, so
    # both capture orders stay refused until an eager run.
    monkeypatch.setattr(module, "_LOADED_PROGRAMS", set())
    loads.clear()
    third = runner()
    prepared = third.prepare(**inputs)
    assert prepared.program_name == check
    assert loads == [(check, "sm_107a"), (own, "sm_107a")]
    count = len(launches)
    with pytest.raises(refused, match=f"program {check} .*eagerly once before"):
        run(third, True)
    with pytest.raises(refused, match=f"program {own} .*{different}"):
        run(third, True, **checkpoints)
    assert len(launches) == count
    assert run(third, False, **checkpoints)[0] == own
    assert run(third, True)[0] == own
    assert loads == [(check, "sm_107a"), (own, "sm_107a")]
    # a shape the rule leaves alone loads its one program and never gets the
    # checkpoint wording
    monkeypatch.setattr(module, "_LOADED_PROGRAMS", set())
    loads.clear()
    cu = dict(inputs, cu_seqlens=torch.tensor([0, T], dtype=torch.int32))
    for key in ("seq_idx", "chunk_indices", "chunk_offsets"):
        del cu[key]
    fourth = runner()
    prepared = fourth.prepare(**cu)
    assert "chk" not in prepared.program_name
    assert loads == [(prepared.program_name, "sm_107a")]
    with pytest.raises(refused, match="eagerly once before") as info:
        run(fourth, True, cu, **checkpoints)
    assert different not in str(info.value)
