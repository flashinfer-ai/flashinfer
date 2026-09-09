"""What `use_cp="auto"` picks, per architecture, with no kernel running.

These pin dispatch, not numerics. Every kernel entry is replaced with a
recorder, so the calls here are shape and capability checks only -- which is
the point: the choice has to be checkable without a device of each generation.

They exist because a prepared-execution experiment once added a shape rule to
`chunk_gated_delta_rule` and wired it into the SM90/100/120 auto condition.
The rule needs a host-side maximum that no existing caller passes, so it read
None, returned False, and turned off Hopper/Blackwell auto CP for every caller
-- silently, since falling back to the fused kernel is not an error. Nothing
here would have let that through.
"""

import pytest
import torch

import flashinfer.gdn_prefill as gp


HEAD_SIZE = 128
CP_ENTRIES = (
    "cp_delta_rule_dsl_sm80",
    "cp_delta_rule_dsl_sm90",
    "cp_delta_rule_dsl_sm100",
    "cp_delta_rule_dsl_sm120",
)
FUSED_ENTRIES = (
    "chunk_gated_delta_rule_sm80",
    "chunk_gated_delta_rule_sm90",
    "chunk_gated_delta_rule_sm100",
    "chunk_gated_delta_rule_sm120",
)


def _inputs(seq_lens, heads, dtype=torch.bfloat16):
    dev = torch.device("cuda")
    total = sum(seq_lens)
    q = torch.randn(total, heads, HEAD_SIZE, dtype=dtype, device=dev)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    g = torch.zeros(total, heads, dtype=torch.float32, device=dev)
    beta = torch.ones(total, heads, dtype=torch.float32, device=dev)
    cu = torch.tensor(
        [0, *torch.tensor(seq_lens).cumsum(0).tolist()], dtype=torch.int32, device=dev
    )
    return dict(q=q, k=k, v=v, g=g, beta=beta, cu_seqlens=cu)


def _dispatch(monkeypatch, *, arch, sm_count=80, minor=0, kwargs_out=None, **call):
    """Which entry ran, under a mocked capability. Returns its name.

    `minor` matters now: SM80 auto CP is restricted to compute capability 8.0,
    and 8.6 and 8.9 have to stay on the fused path. `kwargs_out`, when given, is
    filled with the keyword arguments the entry was called with -- which is how
    the pointer ABI is checked without running a kernel.
    """
    fired = []

    def recorder(name):
        def f(*a, **kw):
            fired.append(name)
            if kwargs_out is not None:
                kwargs_out.update(kw)

        return f

    monkeypatch.setattr(gp, "get_compute_capability", lambda dev: (arch, minor))
    monkeypatch.setattr(gp, "get_device_name", lambda dev: f"mock-sm{arch}0")
    monkeypatch.setattr(gp, "get_device_sm_count", lambda dev: sm_count)
    # The SM100 path refuses CUDA < 13 before looking at anything else, and
    # this runs wherever the test host's toolkit happens to be.
    monkeypatch.setattr(torch.version, "cuda", "13.0")
    for name in CP_ENTRIES + FUSED_ENTRIES:
        monkeypatch.setattr(gp, name, recorder(name))
    # A recorder writes nothing into `output`, which is fine: nothing here
    # reads the result.
    gp.chunk_gated_delta_rule(**call)
    assert len(fired) == 1, f"expected one entry, got {fired}"
    return fired[0]


@pytest.mark.parametrize("arch", [9, 10, 12])
def test_auto_cp_unchanged_on_sm90_sm100_sm120(monkeypatch, arch):
    """The regression this file exists for.

    One sequence at one head is the shape `should_use_cp_host` was written for:
    too little work for the fused kernel to fill the device. No caller passes a
    host-side maximum, so auto CP has to hold without one.
    """
    call = _inputs([4096], 1)
    got = _dispatch(monkeypatch, arch=arch, **call)
    assert (
        got
        == {
            9: "cp_delta_rule_dsl_sm90",
            10: "cp_delta_rule_dsl_sm100",
            12: "cp_delta_rule_dsl_sm120",
        }[arch]
    )


@pytest.mark.parametrize("arch", [9, 10, 12])
def test_auto_still_declines_cp_when_there_is_plenty_of_work(monkeypatch, arch):
    """The other half of the heuristic still applies: it is not always CP."""
    call = _inputs([512] * 32, 8)
    got = _dispatch(monkeypatch, arch=arch, sm_count=80, **call)
    assert (
        got
        == {
            9: "chunk_gated_delta_rule_sm90",
            10: "chunk_gated_delta_rule_sm100",
            12: "chunk_gated_delta_rule_sm120",
        }[arch]
    )


# ─── SM80 auto ───────────────────────────────────────────────────────────────
# Two conditions, both on numbers the host already has: the exact longest
# sequence at 8192 or more, and `num_seqs * max(q, v heads)` at 8 or less.


@pytest.mark.parametrize(
    "seq_lens,heads",
    [
        ([8192], 1),
        ([8192], 8),
        ([16384], 1),
        ([8192, 8192], 4),
    ],
)
def test_auto_on_sm80_picks_cp_inside_the_rule(monkeypatch, seq_lens, heads):
    call = _inputs(seq_lens, heads)
    got = _dispatch(monkeypatch, arch=8, _max_seq_len=max(seq_lens), **call)
    assert got == "cp_delta_rule_dsl_sm80"


@pytest.mark.parametrize(
    "seq_lens,heads,why",
    [
        ([8191], 1, "the longest sequence is one token short"),
        ([4096], 1, "the longest sequence is well short"),
        ([8192], 9, "nine units of work is one too many"),
        ([8192] * 2, 8, "two sequences at eight heads is sixteen"),
        ([512] * 32, 8, "plenty of work and nothing long"),
    ],
)
def test_auto_on_sm80_picks_fused_outside_the_rule(monkeypatch, seq_lens, heads, why):
    call = _inputs(seq_lens, heads)
    got = _dispatch(monkeypatch, arch=8, _max_seq_len=max(seq_lens), **call)
    assert got == "chunk_gated_delta_rule_sm80", why


def test_auto_on_sm80_needs_the_exact_maximum(monkeypatch):
    """No exact maximum, no CP -- and `total_seq_len` is not a substitute.

    `cu_seqlens` is on the device and reading it here would synchronize on
    every call, so the only source is the caller. Two sequences of 8192 have a
    `total_seq_len` of 16384 and a maximum of 8192: at eight heads the rule
    says fused either way, but at one head it says CP on the maximum and would
    also say CP on the total, so the shape that separates them is a batch whose
    total clears 8192 and whose maximum does not.
    """
    call = _inputs([4096, 4096], 1)
    assert _dispatch(monkeypatch, arch=8, **call) == "chunk_gated_delta_rule_sm80"
    assert (
        _dispatch(monkeypatch, arch=8, _max_seq_len=4096, **call)
        == "chunk_gated_delta_rule_sm80"
    )


@pytest.mark.parametrize("minor", [6, 9])
def test_auto_on_sm86_and_sm89_stays_fused(monkeypatch, minor):
    """Fitted on 8.0 only. Different SM count, L2 and register file."""
    call = _inputs([8192], 1)
    got = _dispatch(monkeypatch, arch=8, minor=minor, _max_seq_len=8192, **call)
    assert got == "chunk_gated_delta_rule_sm80"


def test_auto_on_sm80_falls_back_when_cp_cannot_take_the_call(monkeypatch):
    """Checkpointing is not offered on SM8x CP: auto falls back, True raises."""
    call = _inputs([8192], 1)
    total = call["q"].shape[0]
    ckpt = torch.zeros(
        4, 1, HEAD_SIZE, HEAD_SIZE, dtype=torch.float32, device=call["q"].device
    )
    starts = torch.tensor([0, 4], dtype=torch.int32, device=call["q"].device)
    extra = dict(
        state_checkpoints=ckpt,
        checkpoint_cu_starts=starts,
        checkpoint_every_n_tokens=2048,
    )
    with pytest.warns(RuntimeWarning, match="checkpointing"):
        got = _dispatch(monkeypatch, arch=8, _max_seq_len=8192, **call, **extra)
    assert got == "chunk_gated_delta_rule_sm80"
    with pytest.raises(ValueError, match="checkpointing"):
        _dispatch(monkeypatch, arch=8, use_cp=True, _max_seq_len=8192, **call, **extra)
    del total


def test_sm80_cp_is_entered_through_the_pointer_abi(monkeypatch):
    """Both ways in, and neither exposes the switch to a caller.

    `_ptr_abi` is a private argument of the composition. The wrapper sets it
    for SM80 and nothing else does; the public signature has no ABI parameter,
    which is what the second half of this asserts.
    """
    import inspect

    for kw in ({"use_cp": True}, {"use_cp": "auto", "_max_seq_len": 8192}):
        seen: dict = {}
        call = _inputs([8192], 1)
        got = _dispatch(monkeypatch, arch=8, kwargs_out=seen, **kw, **call)
        assert got == "cp_delta_rule_dsl_sm80"
        assert seen.get("_ptr_abi") is True, kw

    for arch in (9, 10, 12):
        seen = {}
        call = _inputs([4096], 1)
        _dispatch(monkeypatch, arch=arch, kwargs_out=seen, **call)
        assert "_ptr_abi" not in seen, arch

    public = inspect.signature(gp.chunk_gated_delta_rule).parameters
    assert not [n for n in public if "abi" in n.lower()]


def test_explicit_cp_on_sm80_still_runs_cp(monkeypatch):
    """`use_cp=True` still reaches CP at shapes the rule would decline."""
    call = _inputs([4096], 1)
    assert (
        _dispatch(monkeypatch, arch=8, use_cp=True, **call) == "cp_delta_rule_dsl_sm80"
    )


def test_explicit_cp_false_never_runs_cp(monkeypatch):
    for arch, fused in (
        (8, "chunk_gated_delta_rule_sm80"),
        (9, "chunk_gated_delta_rule_sm90"),
        (10, "chunk_gated_delta_rule_sm100"),
        (12, "chunk_gated_delta_rule_sm120"),
    ):
        call = _inputs([4096], 1)
        assert _dispatch(monkeypatch, arch=arch, use_cp=False, **call) == fused


def test_the_private_maximum_does_not_change_dispatch_above_sm80(monkeypatch):
    """On SM90/SM100/SM120 it sizes per-sequence indexing and nothing else.

    It *is* a routing input on SM80 now, which is why that architecture is not
    in this list: the rule there is stated on the exact maximum. Above SM80 the
    heuristic is `should_use_cp_host` on parallelism alone, and if the maximum
    starts changing those choices it has crept somewhere it does not belong --
    which is the regression this file was written for.
    """
    for arch in (9, 10, 12):
        base = _dispatch(monkeypatch, arch=arch, **_inputs([4096], 1))
        for mx in (None, 1, 4096):
            got = _dispatch(
                monkeypatch, arch=arch, _max_seq_len=mx, **_inputs([4096], 1)
            )
            assert got == base, f"arch {arch}, _max_seq_len={mx}"


@pytest.mark.parametrize("bad", [0, -1, 4097, 1.0, True, "4096"])
def test_the_private_maximum_is_validated(bad):
    """A value below the real maximum under-sizes indexing, so it is checked."""
    call = _inputs([4096], 1)
    with pytest.raises(ValueError):
        gp.chunk_gated_delta_rule(_max_seq_len=bad, **call)


def test_the_private_maximum_accepts_the_edges():
    """1 and `total_seq_len` are both legal; only the kernel entry is mocked."""
    for mx in (1, 4096):
        call = _inputs([4096], 1)
        out = gp.chunk_gated_delta_rule(_max_seq_len=mx, use_cp=False, **call)
        assert out.shape == (4096, 1, HEAD_SIZE)


# ─── the V32 fused specialization's auto rule ────────────────────────────────
# One validated shape, not a region: `1x8192` with 4 query heads and 16 value
# heads, bf16, an initial state, a final state, no checkpointing and no state
# indices, on compute capability 8.0. Everything else keeps the kernel it had.
# The offer is a private keyword on the SM80 fused entry, so these check the
# keyword rather than the numbers.

V32_HEADS = (4, 16)


def _v32_inputs(
    seq_lens=(8192,), heads=V32_HEADS, dtype=torch.bfloat16, state_dtype=torch.bfloat16
):
    """The V32 contract's shape, with separate query and value head counts."""
    dev = torch.device("cuda")
    total = sum(seq_lens)
    h_qk, h_v = heads
    h = max(h_qk, h_v)
    call = dict(
        q=torch.randn(total, h_qk, HEAD_SIZE, dtype=dtype, device=dev),
        k=torch.randn(total, h_qk, HEAD_SIZE, dtype=dtype, device=dev),
        v=torch.randn(total, h_v, HEAD_SIZE, dtype=dtype, device=dev),
        g=torch.zeros(total, h, dtype=torch.float32, device=dev),
        beta=torch.ones(total, h, dtype=torch.float32, device=dev),
        cu_seqlens=torch.tensor(
            [0, *torch.tensor(list(seq_lens)).cumsum(0).tolist()],
            dtype=torch.int32,
            device=dev,
        ),
        initial_state=torch.zeros(
            len(seq_lens), h, HEAD_SIZE, HEAD_SIZE, dtype=state_dtype, device=dev
        ),
        output_final_state=True,
        _max_seq_len=max(seq_lens),
    )
    return call


def _offered_v32(monkeypatch, **overrides):
    """Whether the SM80 fused entry was offered the specialization."""
    call = _v32_inputs(
        **{
            k: v
            for k, v in overrides.items()
            if k in ("seq_lens", "heads", "dtype", "state_dtype")
        }
    )
    for key in ("minor", "_max_seq_len", "use_cp"):
        if key in overrides:
            if key == "minor":
                continue
            call[key] = overrides[key]
    if overrides.get("_max_seq_len", "keep") is None:
        call.pop("_max_seq_len", None)
    kwargs = {}
    got = _dispatch(
        monkeypatch, arch=8, minor=overrides.get("minor", 0), kwargs_out=kwargs, **call
    )
    return got, kwargs.get("_v32")


def test_v32_offered_on_its_exact_contract(monkeypatch):
    got, offered = _offered_v32(monkeypatch)
    assert got == "chunk_gated_delta_rule_sm80"
    assert offered is True


@pytest.mark.parametrize("minor", [6, 9])
def test_v32_not_offered_off_cc80(monkeypatch, minor):
    """8.6 and 8.9 have a different shared budget and have never been run."""
    got, offered = _offered_v32(monkeypatch, minor=minor)
    assert got == "chunk_gated_delta_rule_sm80"
    assert offered is False


def test_v32_not_offered_without_an_exact_maximum(monkeypatch):
    got, offered = _offered_v32(monkeypatch, _max_seq_len=None)
    assert got == "chunk_gated_delta_rule_sm80"
    assert offered is False


@pytest.mark.parametrize("length", [8191, 8193, 4096])
def test_v32_not_offered_at_other_lengths(monkeypatch, length):
    got, offered = _offered_v32(monkeypatch, seq_lens=(length,))
    assert got == "chunk_gated_delta_rule_sm80"
    assert offered is False


def test_v32_not_offered_for_more_than_one_sequence(monkeypatch):
    got, offered = _offered_v32(monkeypatch, seq_lens=(8192, 8192))
    assert got == "chunk_gated_delta_rule_sm80"
    assert offered is False


@pytest.mark.parametrize("heads", [(2, 8), (8, 32), (4, 4), (16, 4)])
def test_v32_not_offered_at_other_head_counts(monkeypatch, heads):
    got, offered = _offered_v32(monkeypatch, heads=heads)
    assert got in ("chunk_gated_delta_rule_sm80", "cp_delta_rule_dsl_sm80")
    if got == "chunk_gated_delta_rule_sm80":
        assert offered is False


def test_v32_not_offered_when_the_caller_asks_for_cp(monkeypatch):
    """`use_cp=True` is the caller naming a path; the offer stays out of it."""
    got, offered = _offered_v32(monkeypatch, use_cp=True)
    assert got == "cp_delta_rule_dsl_sm80"
    assert offered is None


@pytest.mark.parametrize("arch", [9, 10, 12])
def test_v32_keyword_never_reaches_other_arches(monkeypatch, arch):
    call = _v32_inputs()
    kwargs = {}
    got = _dispatch(monkeypatch, arch=arch, kwargs_out=kwargs, **call)
    assert got.endswith(f"sm{arch}0")
    assert "_v32" not in kwargs


def test_v32_is_not_a_public_parameter():
    import inspect

    assert "_v32" not in inspect.signature(gp.chunk_gated_delta_rule).parameters
