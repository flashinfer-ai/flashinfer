"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host tests of the in-kernel Q/K L2 norm of the Cake packed KDA chunked training package
(``use_qk_l2norm_in_kernel``): the raw q / k operand rule (``_check_qk_raw``: refused, never copied), the mode
keyword, the saved set's ``qk_rstd`` / ``qk_l2norm`` fields and their checks, the layouts' transient ``qn`` /
``kn`` and the byte queries, the registry's l2n classes and the autograd node's arity.  None of them launches a
kernel; the device rows (bitwise forward / backward ``qn``, parity with the shared reference, accuracy) are in
``test_cake_kda_det_train.py``.
"""

import dataclasses
import types

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake.cake_kda_det_train", "cake_jit")

D = cake_backend.D
CPU = torch.device("cpu")


def _packed(S: int, H: int, c0: int, width: int) -> torch.Tensor:
    """The ``q`` columns of a packed projection output ``[S, width]``: ``out[:, c0:c0 + 128 * H].view(S, H, 128)``."""
    out = torch.randn(S, width, dtype=torch.float32).to(torch.bfloat16)
    return out[:, c0 : c0 + D * H].view(S, H, D)


# ---------------------------------------------------------------------------------------------------------------
# _check_qk_raw (contract section 1.1)


def test_qk_raw_accepts_contiguous_and_packed_views_without_a_copy() -> None:
    S, H = 5, 3
    x = torch.randn(S, H, D).to(torch.bfloat16)
    assert cake_backend._check_qk_raw(x, "q", S, H, CPU) is x
    for c0, width in ((0, 3 * D * H), (8, 3 * D * H + 16), (D * H, 2 * D * H)):
        view = _packed(S, H, c0, width)
        got = cake_backend._check_qk_raw(view, "q", S, H, CPU)
        assert got.data_ptr() == view.data_ptr()
        assert got.stride() == view.stride() == (width, D, 1)


def test_qk_raw_canonical_axes_of_extent_one_are_restrided_without_a_copy() -> None:
    base = torch.zeros(4096, dtype=torch.bfloat16)
    one_head = base.as_strided((3, 1, D), (512, 7, 1))  # H == 1: any head stride
    got = cake_backend._check_qk_raw(one_head, "k", 3, 1, CPU)
    assert got.data_ptr() == one_head.data_ptr() and got.stride() == (512, D, 1)
    one_token = base.as_strided((1, 2, D), (3, D, 1))  # S == 1: any token stride
    got = cake_backend._check_qk_raw(one_token, "k", 1, 2, CPU)
    assert got.data_ptr() == one_token.data_ptr() and got.stride() == (2 * D, D, 1)


def test_qk_raw_empty_is_checked_for_shape_dtype_device_only() -> None:
    empty = torch.empty(0, 4, D, dtype=torch.bfloat16)
    assert cake_backend._check_qk_raw(empty, "q", 0, 4, CPU) is empty
    with pytest.raises(ValueError, match="bfloat16"):
        cake_backend._check_qk_raw(empty.float(), "q", 0, 4, CPU)


@pytest.mark.parametrize(
    "case",
    [
        "inner_stride",
        "head_stride",
        "token_stride_mod8",
        "overlapping_rows",
        "misaligned_base",
        "shape",
        "dtype",
    ],
)
def test_qk_raw_refusals(case: str) -> None:
    S, H = 4, 2
    base = torch.zeros(1 << 16, dtype=torch.bfloat16)
    views = {
        "inner_stride": base.as_strided((S, H, D), (H * D * 2, D * 2, 2)),
        "head_stride": base.as_strided((S, H, D), (4 * D, 2 * D, 1)),
        "token_stride_mod8": base.as_strided((S, H, D), (H * D + 4, D, 1)),
        "overlapping_rows": base.as_strided((S, H, D), (D, D, 1)),
        "misaligned_base": base[4:].as_strided((S, H, D), (H * D, D, 1)),
        "shape": torch.zeros(S, H, 64, dtype=torch.bfloat16),
        "dtype": torch.zeros(S, H, D, dtype=torch.float32),
    }
    x = views[case]
    with pytest.raises(ValueError):
        cake_backend._check_qk_raw(x, "q", S, H, CPU)


def test_operands_take_raw_views_only_with_the_flag() -> None:
    S, H = 6, 2
    q = _packed(S, H, 8, 3 * D * H + 8)
    k = _packed(S, H, D * H + 8, 3 * D * H + 8)
    v = torch.zeros(S, H, D, dtype=torch.bfloat16)
    g = torch.zeros(S, H, D, dtype=torch.float32)
    beta = torch.zeros(S, H, dtype=torch.bfloat16)
    cu = torch.tensor([0, S], dtype=torch.int32)
    got = cake_backend._operands(q, k, v, g, beta, cu, l2n="fused")
    assert got[0].data_ptr() == q.data_ptr() and got[1].data_ptr() == k.data_ptr()
    with pytest.raises(ValueError, match="contiguous"):
        cake_backend._operands(q, k, v, g, beta, cu)


# ---------------------------------------------------------------------------------------------------------------
# mode keyword, saved set


@pytest.mark.parametrize("bad", [1, 0, None, "fused", 1.0])
def test_mode_keyword_must_be_a_bool(bad) -> None:
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel must be a bool"):
        cake_backend._l2n_mode(bad)
    assert cake_backend._l2n_mode(True) == "fused"
    assert cake_backend._l2n_mode(False) == "off"


def test_l2n_modes_are_the_registrys() -> None:
    assert cake_backend.L2N_MODES == cake_jit.L2N_MODES == ("off", "fused", "fallback")
    assert {"fwd_intra", "bwd_prep", "glm_intra"} <= set(cake_jit.L2N_STAGES)
    for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
        assert stage in cake_jit.STAGES and stage in cake_jit.OPTIONAL_STAGES
        assert stage in cake_jit.L2N_STAGES
    for name in ("q_raw", "k_raw", "qn", "kn", "qk_rstd", "qn_int", "kn_int"):
        assert name in cake_jit.LOGICAL_TENSORS
    for name in (
        "num_tokens",
        "q_raw_row_stride",
        "q_raw_head_stride",
        "k_raw_row_stride",
        "k_raw_head_stride",
        "dq_row_stride",
        "dk_row_stride",
        "read_rstd",
        "write_rstd",
        "write_qkn",
    ):
        assert name in cake_jit.LOGICAL_SCALARS


def _saved(S: int, H: int, **fields):
    a = torch.zeros(S, H, 64, dtype=torch.bfloat16)
    return cake_backend.CakeKDADetChunkSaved(
        cake_backend.SAVED_VERSION, a, a, None, S, H, 1, cake_backend.SCALE, **fields
    )


def test_saved_set_fields_and_their_checks() -> None:
    S, H = 3, 2
    fields = {
        f.name: f.default for f in dataclasses.fields(cake_backend.CakeKDADetChunkSaved)
    }
    assert fields["qk_rstd"] is None and fields["qk_l2norm"] is False
    for version in (cake_backend.SAVED_VERSION, cake_backend.SAVED_VERSION_TAPE):
        assert cake_backend._EXTRA_FIELDS[version][-1] == "qk_rstd"
    rstd = torch.zeros(2, S, H, dtype=torch.float32)
    on = _saved(S, H, qk_rstd=rstd, qk_l2norm=True)
    off = _saved(S, H)
    cake_backend._check_saved_l2n(on, "fused", S, H, CPU)
    cake_backend._check_saved_l2n(on, "fallback", S, H, CPU)
    cake_backend._check_saved_l2n(off, "off", S, H, CPU)
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel=True; this backward"):
        cake_backend._check_saved_l2n(on, "off", S, H, CPU)
    with pytest.raises(
        ValueError, match="use_qk_l2norm_in_kernel=False; this backward"
    ):
        cake_backend._check_saved_l2n(off, "fused", S, H, CPU)
    with pytest.raises(ValueError, match="carries none|has none"):
        cake_backend._check_saved_l2n(_saved(S, H, qk_l2norm=True), "fused", S, H, CPU)
    with pytest.raises(ValueError, match="without the in-kernel"):
        cake_backend._check_saved_l2n(_saved(S, H, qk_rstd=rstd), "off", S, H, CPU)
    for bad in (
        torch.zeros(2, S + 1, H),
        torch.zeros(2, S, H, dtype=torch.float64),
        torch.zeros(2, H, S).transpose(1, 2),
    ):
        with pytest.raises(ValueError):
            cake_backend._check_saved_l2n(
                _saved(S, H, qk_rstd=bad, qk_l2norm=True), "fused", S, H, CPU
            )
    # the autograd extras: qk_rstd rides last, absent with the flag off
    extra, names = cake_backend._saved_extra(on)
    assert names == ("qk_rstd",) and extra[0] is rstd
    assert cake_backend._saved_extra(off) == ((), ())


def test_saved_and_output_bytes() -> None:
    S, H = 1000, 8
    off = cake_backend.saved_bytes(S, H)
    assert cake_backend.saved_bytes(S, H, l2n="fused") == off + 8 * S * H
    assert cake_backend.saved_bytes(S, H, l2n="fallback") == off + 8 * S * H
    for backward in (False, True):
        assert cake_backend.output_bytes(
            S, H, backward=backward, l2n="fused"
        ) == cake_backend.output_bytes(S, H, backward=backward)
    with pytest.raises(ValueError, match="l2n must be one of"):
        cake_backend.saved_bytes(S, H, l2n="on")


# ---------------------------------------------------------------------------------------------------------------
# layouts


def _rows(mode: str, **kwargs) -> dict[str, dict]:
    return {r["buffer"]: r for r in cake_backend.lifetime_table(mode, **kwargs)}


@pytest.mark.parametrize(
    "mode", ["fwd", "fwd_tape", "fwd_kcp", "replay", "replay_tape"]
)
@pytest.mark.parametrize("l2n", ["fused", "fallback"])
def test_layouts_carry_transient_qn_kn_int_on_the_u_w_slots(
    mode: str, l2n: str
) -> None:
    """Contract v1.1: the fused intra program's CTA-private internal rows qn_int / kn_int ride at sub-offset 0 of the
    u / w slots of the forward and replay layouts, lifetime (0, 1), bytes unchanged; the saved-set backward layouts
    have none."""
    off = _rows(mode)
    on = _rows(mode, l2n=l2n)
    assert "qn_int" not in off and "kn_int" not in off
    for name, host in (("qn_int", "u"), ("kn_int", "w")):
        r = on[name]
        assert r["slot"] == on[host]["slot"] and r["kind"] == "B16"
        assert r["sub_offset_per_th"] == 0
        assert (r["first_write"], r["last_read"]) == (0, 1)
    # the same bytes: the layout's per-token-head size does not grow
    R, H, NC = 128, 2, 2
    assert cake_backend._mode_bytes(
        R, H, NC, 0, mode, 1, True, l2n
    ) == cake_backend._mode_bytes(R, H, NC, 0, mode, 1, True, "off")


@pytest.mark.parametrize("mode", ["bwd", "bwd_tape"])
def test_saved_set_backward_layouts_have_no_internal_qn(mode: str) -> None:
    assert "qn_int" not in _rows(mode, l2n="fused") and "kn_int" not in _rows(
        mode, l2n="fused"
    )


@pytest.mark.parametrize("mode", ["bwd", "replay", "bwd_tape", "replay_tape"])
def test_fallback_backward_keeps_dq_f_dk_f_until_the_pullback(mode: str) -> None:
    fused = _rows(mode, l2n="fused")
    fallback = _rows(mode, l2n="fallback")
    for name in ("dq_f", "dk_f"):
        assert (fused[name]["first_write"], fused[name]["last_read"]) == (6, 7)
        assert (fallback[name]["first_write"], fallback[name]["last_read"]) == (6, 8)
    # every l2n mode drops the gate epilogue's buffers (the l2n GLM intra programs fuse it)
    for rows in (fused, fallback):
        assert "dg_int" not in rows and "db_int" not in rows
    with pytest.raises(ValueError, match="gate_fused must be True"):
        cake_backend.lifetime_table(mode, l2n="fused", gate_fused=False)


def _shipped_l2n_modes(gate: str = "g") -> list[str]:
    """The l2n modes whose fwd_intra / bwd_prep / glm_intra records of gate mode ``gate`` this build ships."""
    return [
        mode
        for mode in ("fused", "fallback")
        if all(
            any(
                mode in cake_jit.MODULES[n].get("l2n", ["off"])
                for n in cake_jit.records_of_stage(stage, gate)
            )
            for stage in ("fwd_intra", "bwd_prep", "glm_intra")
        )
    ]


@pytest.mark.parametrize("arch", list(cake_jit.ARCHES))
def test_saved_set_backward_keeps_qn_kn_in_the_arena(arch: str) -> None:
    """The saved-set backward of an l2n mode keeps qn / kn on the arena's do_int / dv2 bytes (dv2 on the dq / dk
    bytes): no record of those paths binds do_int, only the dq / dk writers bind dq / dk and none of them dv2, so
    memory_report counts no per-call qn / kn there; the replay, a lean layout, two tail passes and 2 S < R keep them
    per call."""
    modes = _shipped_l2n_modes()
    if not modes:
        pytest.skip("this build ships no l2n mode")
    rows = 2 * D * 2  # per-call qn + kn bytes per token-head
    for mode in modes:
        for gate in (x for x in ("g", "raw") if mode in _shipped_l2n_modes(x)):
            for S, N, H in ((5000, 3, 16), (16384, 1, 64), (16384, 4, 64)):
                p = cake_backend.plan(S, N, H, 148, arch, gate=gate, l2n=mode)
                saved = ("light", "light_states", "tape")
                for path in (x for x in saved if x in p.records):
                    layout = "bwd_tape" if path == "tape" else "bwd"
                    names = {r.name for r in p.layout(layout)}
                    assert {"do_int", "dv2"} <= names
                    ok = cake_backend._l2n_rows_in_arena(p, path, names, 1)
                    assert ok, (path, mode, gate)
                    assert not cake_backend._l2n_rows_in_arena(p, path, names, 2)
                    lean = {r.name for r in p.layout(layout + "_lean")}
                    assert not cake_backend._l2n_rows_in_arena(p, path, lean, 1)
                replay = {r.name for r in p.layout("replay")}
                assert not cake_backend._l2n_rows_in_arena(p, "replay", replay, 1)
                for save_mode in ("light", "tape"):
                    report = cake_backend.memory_report(
                        S, N, H, save_mode, sm_count=148, arch=arch, gate=gate, l2n=mode
                    )
                    assert report["bwd_qn_kn"] == 0, (save_mode, mode, gate)
                    assert report["replay_qn_kn"] == rows * S * H
                lean_report = cake_backend.memory_report(
                    S,
                    N,
                    H,
                    sm_count=148,
                    arch=arch,
                    gate=gate,
                    l2n=mode,
                    arena=cake_backend.CakeKDADetArena(lean=True),
                )
                assert lean_report["bwd_qn_kn"] == rows * S * H
        # 64 tokens in 8 documents: R = 640 internal rows > 2 S, dv2 does not fit the dq / dk bytes
        p = cake_backend.plan(64, 8, 2, 148, arch, l2n=mode)
        assert 2 * p.S < p.R
        names = {r.name for r in p.layout("bwd")}
        assert not cake_backend._l2n_rows_in_arena(p, "light", names, 1)
        report = cake_backend.memory_report(64, 8, 2, sm_count=148, arch=arch, l2n=mode)
        assert report["bwd_qn_kn"] == rows * 64 * 2


def test_gate_fused_on_every_arch_in_l2n_modes() -> None:
    for arch in cake_jit.ARCHES:
        assert cake_backend._gate_fused(arch, "g", "fused") is True
        assert cake_backend._gate_fused(arch, "g", "fallback") is True


# ---------------------------------------------------------------------------------------------------------------
# registry


def test_select_record_refuses_a_mode_without_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-optional l2n stage whose records do not serve the call's mode is refused, never silently routed to
    another class; an optional stage selects nothing."""
    scalars = dict(
        num_chunks_pad=2, num_seqs=1, num_heads=1, sm_count=148, arch=cake_jit.ARCHES[0]
    )
    for stage in ("fwd_intra", "bwd_prep", "glm_intra"):
        path = "fwd" if stage == "fwd_intra" else "light"
        if any(
            "fused" in cake_jit.MODULES[n].get("l2n", ["off"])
            for n in cake_jit.records_of_stage(stage)
        ):
            continue  # the class ships: covered by the plan tests
        with pytest.raises(
            ValueError, match="no record serves the in-kernel Q/K L2 norm mode"
        ):
            cake_jit.select_record(stage, **scalars, path=path, l2n="fused")
    for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
        path = "fwd" if stage == "qk_l2norm" else "light"
        if cake_jit.records_of_stage(stage):
            continue
        assert (
            cake_jit.select_record(stage, **scalars, path=path, l2n="fallback") is None
        )
    with pytest.raises(ValueError, match="l2n 'on' is not one of"):
        cake_jit.select_record("dqkg", **scalars, path="light", l2n="on")


def test_l2n_classes_and_the_l2n_in_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """A plain record kept on some paths of the fused route narrows its rule with ``l2n_in``; the twin serves the
    others: exactly one record per (mode, path)."""
    fake = {
        "x_plain": {
            "stage": "qk_l2norm",
            "l2n": ["off", "fused"],
            "select_rule": {
                "kind": "any_of",
                "rules": [
                    {"kind": "l2n_in", "modes": ["off"]},
                    {"kind": "path_in", "paths": ["replay"]},
                ],
            },
            "grid_rule": {"kind": "tokens"},
        },
        "x_l2n": {
            "stage": "qk_l2norm",
            "l2n": ["fused"],
            "select_rule": {"kind": "path_in", "paths": ["fwd", "fwd_nosave"]},
            "grid_rule": {"kind": "tokens"},
        },
    }
    for name, rec in fake.items():
        monkeypatch.setitem(cake_jit.MODULES, name, rec)
    scalars = dict(
        num_chunks_pad=2, num_seqs=1, num_heads=1, sm_count=148, arch=cake_jit.ARCHES[0]
    )

    def sel(path: str, mode: str) -> str | None:
        return cake_jit.select_record("qk_l2norm", **scalars, path=path, l2n=mode)

    assert sel("fwd", "off") == "x_plain" and sel("replay", "off") == "x_plain"
    assert sel("fwd", "fused") == "x_l2n" and sel("fwd_nosave", "fused") == "x_l2n"
    assert sel("replay", "fused") == "x_plain"
    # the fallback class: the registry's own token-local record (none of the two above serves the mode)
    shipped = [
        n
        for n in cake_jit.records_of_stage("qk_l2norm")
        if "fallback" in cake_jit.MODULES[n].get("l2n", ["off"])
    ]
    assert sel("fwd", "fallback") == (shipped[0] if shipped else None)


def test_tokens_grid_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {
        "stage": "qk_l2norm",
        "grid_rule": {"kind": "tokens"},
        "select_rule": {"kind": "always"},
    }
    monkeypatch.setitem(cake_jit.MODULES, "qk_l2norm_test", record)
    common = dict(num_chunks_pad=4, num_seqs=1, num_heads=3, sm_count=148)
    assert cake_jit.launch_grid("qk_l2norm_test", **common, num_tokens=129) == (3, 3, 1)
    assert cake_jit.launch_grid("qk_l2norm_test", **common, num_tokens=64) == (1, 3, 1)
    assert cake_jit.launch_grid("qk_l2norm_test", **common, num_tokens=0) == (1, 3, 1)


# ---------------------------------------------------------------------------------------------------------------
# values, test hook, autograd arity


def test_l2n_values_bind_normalized_rows_under_q_and_k() -> None:
    S, H = 4, 2
    q = _packed(S, H, 8, 3 * D * H + 8)
    k = torch.zeros(S, H, D, dtype=torch.bfloat16)
    qn, kn = torch.empty_like(k), torch.empty_like(k)
    rstd = torch.empty(2, S, H)
    v = cake_backend._l2n_values(q, k, qn, kn, rstd, read_rstd=1, write_rstd=0)
    assert v["q"] is qn and v["k"] is kn and v["q_raw"] is q and v["k_raw"] is k
    assert v["qk_rstd"] is rstd and v["num_tokens"] == S
    assert v["q_raw_row_stride"] == 3 * D * H + 8 and v["k_raw_row_stride"] == H * D
    assert v["q_raw_head_stride"] == v["k_raw_head_stride"] == D
    assert v["dq_row_stride"] == v["dk_row_stride"] == H * D
    assert (v["read_rstd"], v["write_rstd"]) == (1, 0)
    # contract v1.1: the internal rows default to the token rows (no record reads them there), write_qkn to 1
    assert v["qn_int"] is qn and v["kn_int"] is kn and v["write_qkn"] == 1
    qi, ki = (
        torch.empty(64, H, D, dtype=torch.bfloat16),
        torch.empty(64, H, D, dtype=torch.bfloat16),
    )
    v = cake_backend._l2n_values(
        q, k, qn, kn, rstd, read_rstd=0, write_rstd=1, write_qkn=0, qn_int=qi, kn_int=ki
    )
    assert v["qn_int"] is qi and v["kn_int"] is ki and v["write_qkn"] == 0


def test_internal_rows_scatter_to_token_rows() -> None:
    """The hook's map of the internal rows (row 64 c + i of chunk c) to token rows chunk_bos[c] + i, real tokens
    only."""
    H = 1
    lengths = (3, 64, 70)
    bos, ln = [], []
    t = 0
    for n in lengths:
        for c0 in range(0, n, 64):
            bos.append(t + c0)
            ln.append(min(64, n - c0))
        t += n
    bos += [0]  # a pad chunk
    ln += [0]
    S, nc = t, len(bos)
    tok = torch.arange(S, dtype=torch.float32)
    rows = torch.full((nc * 64, H, D), -1.0)
    for c in range(nc):
        rows[64 * c : 64 * c + ln[c], :, :] = tok[bos[c] : bos[c] + ln[c], None, None]
    out = torch.full((S, H, D), -2.0)
    cake_backend._internal_to_tokens(
        rows,
        torch.tensor(bos, dtype=torch.int32),
        torch.tensor(ln, dtype=torch.int32),
        out,
    )
    assert torch.equal(out[:, 0, 0], tok)


def test_record_hook_clones_and_is_off_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert cake_backend._l2n_record is None
    x = torch.arange(4.0)
    cake_backend._record_l2n("fwd", qn=x)  # no hook: nothing recorded, no error
    store: dict = {}
    monkeypatch.setattr(cake_backend, "_l2n_record", store)
    cake_backend._record_l2n("bwd", qn=x, path="light")
    assert store["bwd.path"] == "light" and torch.equal(store["bwd.qn"], x)
    assert store["bwd.qn"].data_ptr() != x.data_ptr()


def test_autograd_node_returns_one_gradient_slot_per_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import inspect

    n_inputs = (
        len(inspect.signature(cake_backend.CakeKDADetFunction.forward).parameters) - 1
    )
    seen = {}

    def fake_bwd(*args, **kwargs):
        seen.update(kwargs)
        return tuple(torch.zeros(1) for _ in range(5))

    monkeypatch.setattr(cake_backend, "cake_kda_det_train_backward", fake_bwd)
    t = torch.zeros(1)
    ctx = types.SimpleNamespace(
        saved_tensors=(
            t,
            t,
            t,
            t,
            t,
            t,
            t,
            t,
            t,
        ),  # q k v g beta cu_seqlens Aqk Akk qk_rstd
        gate="g",
        lower_bound=None,
        qk_l2norm=True,
        scale=cake_backend.SCALE,
        workspace_ref=None,
        arena=None,
        doc_lengths=None,
        version=cake_backend.SAVED_VERSION,
        n_tables=0,
        extra_fields=("qk_rstd",),
        problem=(1, 1, 1),
    )
    grads = cake_backend.CakeKDADetFunction.backward(ctx, torch.zeros(1))
    assert len(grads) == n_inputs
    assert seen["use_qk_l2norm_in_kernel"] is True
    assert seen["saved"].qk_l2norm is True and seen["saved"].qk_rstd is t


# ---------------------------------------------------------------------------------------------------------------
# the KDA reference against the shared GDN reference (cake_l2n, blob dcfaa8a36492): bit for bit


def _cake_l2n():
    try:
        return cake_package("cake_gdn_det_train_kcp", "cake_l2n")
    except (ImportError, ModuleNotFoundError):
        pytest.skip(
            "the shared GDN reference cake_gdn_det_train_kcp/cake_l2n.py is not in this tree (FlashInfer #8252)"
        )


def _edge_rows(H: int = 3) -> torch.Tensor:
    """BF16 raw rows ``[T, H, 128]`` with the edge cases of the norm: random of several magnitudes, an all-zero row
    (``ss = 0``: ``r = 1000``), signed zeros, BF16 subnormals, a row whose ``ss`` overflows FP32 (``r = 0``), one
    large channel among tiny ones."""
    g = torch.Generator().manual_seed(7817)
    rows = [torch.randn(64, H, D, generator=g) * s for s in (1e-3, 1.0, 30.0, 3e3)]
    special = torch.zeros(6, H, D)
    special[1] = -0.0
    special[2] = torch.tensor(2.0**-130)  # BF16 subnormal
    special[3, :, ::2] = -(2.0**-133)
    special[4] = 3e19  # ss overflows FP32
    special[5] = 1e-20
    special[5, :, 17] = 5.0
    return torch.cat(rows + [special]).to(torch.bfloat16)


def _bits(x: torch.Tensor) -> torch.Tensor:
    return x.contiguous().view(
        torch.int16 if x.dtype == torch.bfloat16 else torch.int32
    )


def test_kda_reference_is_bitwise_the_shared_gdn_reference() -> None:
    from tests.experimental.test_helpers import (
        cake_kda_det_train_reference_l2n as kda_l2n,
    )

    gdn = _cake_l2n()
    assert not gdn.flush_denormal_active()
    x = _edge_rows()
    assert torch.equal(
        _bits(kda_l2n.sum_of_squares(x)), _bits(gdn.sum_of_squares(x.float()))
    )
    qn_k, r_k = kda_l2n.qk_l2norm(x)
    qn_g, r_g = gdn.qk_l2norm_ref(x)
    assert torch.equal(_bits(r_k), _bits(r_g)) and torch.equal(_bits(qn_k), _bits(qn_g))
    # a strided view gives the contiguous copy's bits (contract C11 at the reference level)
    wide = torch.zeros(x.shape[0], 2 * x.shape[1] * D + 8, dtype=torch.bfloat16)
    wide[:, 8 : 8 + x.shape[1] * D] = x.reshape(x.shape[0], -1)
    view = wide[:, 8 : 8 + x.shape[1] * D].view(x.shape)
    qn_v, r_v = kda_l2n.qk_l2norm(view)
    assert torch.equal(_bits(qn_v), _bits(qn_k)) and torch.equal(_bits(r_v), _bits(r_k))
    g = torch.Generator().manual_seed(5747)
    for scale in (1e-6, 1.0, 1e4):
        Dq = torch.randn(x.shape, generator=g) * scale
        Dq[-6:-4] = 0.0
        a = kda_l2n.l2n_pullback(x, r_k, Dq)
        b = gdn.l2n_pullback(x, r_g, Dq)
        assert torch.equal(_bits(a), _bits(b))
        assert torch.equal(_bits(a.to(torch.bfloat16)), _bits(b.to(torch.bfloat16)))
    assert torch.equal(_bits(kda_l2n.l2n_dot(Dq)), _bits(gdn.l2n_dot(Dq)))


def test_kda_reference_normalizes_and_pulls_back_like_fp64() -> None:
    """The device-order FP32 reference is within a few FP32 ulps of the FP64 normalization and its exact pullback
    (a guard against an order that is bitwise equal to a wrong formula)."""
    from tests.experimental.test_helpers import (
        cake_kda_det_train_reference_l2n as kda_l2n,
    )

    g = torch.Generator().manual_seed(1)
    x = (torch.randn(256, 2, D, generator=g) * 3).to(torch.bfloat16)
    qn, r = kda_l2n.qk_l2norm(x)
    x64 = x.double().requires_grad_(True)
    y64 = x64 / torch.sqrt((x64 * x64).sum(-1, keepdim=True) + kda_l2n.EPS)
    assert (qn.double() - y64.detach()).abs().max() <= 2.0**-8
    Dq = torch.randn(x.shape, generator=g)
    (dx64,) = torch.autograd.grad(y64, x64, Dq.double())
    dx = kda_l2n.l2n_pullback(x, r, Dq)
    err = (dx.double() - dx64).abs().max() / dx64.abs().max()
    assert err < 1e-5, err


# ---------------------------------------------------------------------------------------------------------------
# plans


@pytest.mark.parametrize("arch", list(cake_jit.ARCHES))
def test_flag_off_plans_are_the_default_plans(arch: str) -> None:
    for S, N, H in ((256, 1, 2), (5000, 3, 16), (64 * 40, 300, 64)):
        default = cake_backend.plan(S, N, H, 148, arch)
        assert cake_backend.plan(S, N, H, 148, arch, l2n="off") is default
        assert default.l2n == "off"
        assert not any(
            stage in ("qk_l2norm", "qk_apply", "qk_pullback") and name is not None
            for by_stage in default.records.values()
            for stage, name in by_stage.items()
        )


@pytest.mark.parametrize("arch", list(cake_jit.ARCHES))
def test_l2n_plans_need_l2n_records(arch: str) -> None:
    """A mode whose records the build does not ship is refused at planning (no silent route)."""
    for mode in ("fused", "fallback"):
        ships = all(
            any(
                mode in cake_jit.MODULES[n].get("l2n", ["off"])
                for n in cake_jit.records_of_stage(stage)
            )
            for stage in ("fwd_intra", "bwd_prep", "glm_intra")
        )
        if ships:
            p = cake_backend.plan(256, 1, 2, 148, arch, l2n=mode)
            assert p.l2n == mode and p.gate_fused
            assert p.dispatched("qk_pullback", "light") == (mode == "fallback")
        else:
            with pytest.raises(
                ValueError, match="no record serves the in-kernel Q/K L2 norm mode"
            ):
                cake_backend.plan(256, 1, 2, 148, arch, l2n=mode)
    with pytest.raises(ValueError, match="l2n must be one of"):
        cake_backend.plan(256, 1, 2, 148, arch, l2n="on")


@pytest.mark.parametrize("arch", list(cake_jit.ARCHES))
def test_fallback_replay_counts_no_qk_apply(arch: str) -> None:
    """The fallback plan selects a ``qk_apply`` record on the replay path too (the KCP's local backward plans on
    that path), but the replay normalizes q / k with ``qk_l2norm`` and never launches ``qk_apply``: the replay's
    stages and launch count leave it out, while the saved-set paths keep it and launch no ``qk_l2norm``."""
    try:
        cake_backend.plan(256, 1, 2, 148, arch, l2n="fallback")
    except ValueError:
        pytest.skip(f"this build ships no fallback records for {arch}")
    for S, N, H in ((256, 1, 2), (5000, 3, 16), (64 * 40, 300, 64)):
        p = cake_backend.plan(S, N, H, 148, arch, l2n="fallback")
        assert p.dispatched("qk_apply", "replay") and p.dispatched(
            "qk_l2norm", "replay"
        )
        selected = {s for s, r in p.records["replay"].items() if r is not None}
        assert set(p.stages("replay")) == selected - {"qk_apply"}
        assert "qk_l2norm" in p.stages("replay") and "qk_pullback" in p.stages("replay")
        for path in ("light", "tape"):
            assert "qk_apply" in p.stages(path) and "qk_l2norm" not in p.stages(path)
        try:
            fused = cake_backend.plan(S, N, H, 148, arch, l2n="fused")
        except ValueError:
            continue
        # the fused class runs the same stages but the token-local ones: the fallback replay adds exactly the
        # normalize and the pullback launches
        assert p.launch_count("replay") == fused.launch_count("replay") + p.launches(
            "qk_l2norm", "replay"
        ) + p.launches("qk_pullback", "replay")


# ---------------------------------------------------------------------------------------------------------------
# device rows (skip without an exported device or without the l2n records of its architecture)

from tests.experimental.test_helpers.cake_kda_det_train_reference import (  # noqa: E402
    LOWER_BOUND,
    elementwise_gate,
    error_metrics,
    make_inputs,
    make_raw_inputs,
)
from tests.experimental.test_helpers import cake_kda_det_train_reference_l2n as ref_l2n  # noqa: E402

ATOL = RTOL = (
    1e-2  # the package's declared elementwise gate (test_cake_kda_det_train.py)
)
SHAPES_L2N = (
    ((64,), 1),
    ((1000, 37, 300), 4),
    ((2048,), 16),
    ((4000, 96), 32),
    ((5000, 3000), 64),
)


def _l2n_device(mode: str = "fused") -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    arch = cake_jit.arch_for_capability(tuple(torch.cuda.get_device_capability(index)))
    if arch is None:
        pytest.skip("not an exported compute capability")
    for stage in ("fwd_intra", "bwd_prep", "glm_intra"):
        if not any(
            mode in cake_jit.MODULES[n].get("l2n", ["off"])
            and arch in cake_jit.MODULES[n]["arches"]
            for n in cake_jit.records_of_stage(stage)
        ):
            pytest.skip(
                f"this build ships no l2n={mode!r} record of {stage} for {arch}"
            )
    return torch.device("cuda", index)


def _raw_qk(
    lengths, H: int, device, seed: int = 0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Raw (not normalized) BF16 q / k of a few magnitudes per token (the caller's projections)."""
    S = int(sum(lengths))
    gen = torch.Generator().manual_seed(7817 + seed + 31 * S + H)
    mag = torch.exp(torch.randn(S, H, 1, generator=gen) * 1.5)
    q = (torch.randn(S, H, D, generator=gen) * mag).to(torch.bfloat16)
    k = (torch.randn(S, H, D, generator=gen) * mag.flip(0)).to(torch.bfloat16)
    return q.to(device), k.to(device)


def _l2n_inputs(lengths, H: int, device, *, raw_gate: bool = False) -> dict:
    inputs = (make_raw_inputs if raw_gate else make_inputs)(lengths, H, device)
    inputs["q"], inputs["k"] = _raw_qk(lengths, H, device)
    return inputs


def _gate_kw(inputs: dict, raw_gate: bool) -> dict:
    if not raw_gate:
        return {}
    return {
        "use_gate_in_kernel": True,
        "A_log": inputs["A_log"],
        "dt_bias": inputs["dt_bias"],
        "lower_bound": LOWER_BOUND,
    }


def _g(inputs: dict, raw_gate: bool) -> torch.Tensor:
    """``g``: the activated FP32 gate, or the raw BF16 gate of make_raw_inputs."""
    return inputs["g"]


def _fwd(inputs, raw_gate=False, **kwargs):
    i = inputs
    return cake_backend.cake_kda_det_train_forward(
        i["q"],
        i["k"],
        i["v"],
        _g(i, raw_gate),
        i["beta"],
        i["cu_seqlens"],
        use_qk_l2norm_in_kernel=True,
        **_gate_kw(i, raw_gate),
        **kwargs,
    )


def _bwd(inputs, saved, raw_gate=False, **kwargs):
    i = inputs
    return cake_backend.cake_kda_det_train_backward(
        i["q"],
        i["k"],
        i["v"],
        _g(i, raw_gate),
        i["beta"],
        i["cu_seqlens"],
        i["dout"],
        saved=saved,
        use_qk_l2norm_in_kernel=True,
        **_gate_kw(i, raw_gate),
        **kwargs,
    )


@pytest.fixture
def l2n_hook(monkeypatch: pytest.MonkeyPatch) -> dict:
    store: dict = {}
    monkeypatch.setattr(cake_backend, "_l2n_record", store)
    return store


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize(
    "shape", SHAPES_L2N, ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_matches_the_exact_reference(shape, raw_gate: bool) -> None:
    """Accuracy with use_qk_l2norm_in_kernel=True against the exact FP64 recurrence whose q / k are normalized in
    FP64 inside the graph (``dq`` / ``dk`` the raw rows' gradients): the declared elementwise gate on ``out`` and
    every gradient."""
    device = _l2n_device()
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    out, saved = _fwd(inputs, raw_gate)
    grads = _bwd(inputs, saved, raw_gate)
    torch.cuda.synchronize(device)
    assert saved.qk_l2norm is True and saved.qk_rstd.shape == (2, sum(lengths), H)
    i = inputs
    if raw_gate:
        ref_out, ref = ref_l2n.kda_exact_raw_l2n_with_grads(
            i["q"],
            i["k"],
            i["v"],
            i["g"],
            i["beta"],
            i["A_log"],
            i["dt_bias"],
            i["dout"],
            i["cu_seqlens"],
            cake_backend.SCALE,
            LOWER_BOUND,
        )
        names = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")
    else:
        ref_out, ref = ref_l2n.kda_exact_l2n_with_grads(
            i["q"],
            i["k"],
            i["v"],
            i["g"],
            i["beta"],
            i["dout"],
            i["cu_seqlens"],
            cake_backend.SCALE,
        )
        names = ("dq", "dk", "dv", "dbeta", "dg")
    candidates = {"out": out, **dict(zip(names, grads, strict=True))}
    references = {"out": ref_out, **ref}
    failures = []
    for name, cand in candidates.items():
        r = references[name].reshape(cand.shape)
        gate = elementwise_gate(cand, r, ATOL, RTOL)
        if not gate["pass"]:
            failures.append((name, gate, error_metrics(cand, r)))
    assert not failures, failures


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:4], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_forward_rows_equal_the_reference_and_every_backward_route(
    shape, raw_gate: bool, l2n_hook: dict
) -> None:
    """Lead requirements (2) / (3): the forward's rstd / qn / kn are bitwise the shared reference's; the backward's
    qn / kn (light / light_states, tape, replay table route or intra-first, checkpoint recompute) are bitwise the
    forward's; the replay's rstd is the saved one; repeats are bitwise.  The hook rows (``write_qkn``) validate the
    qn / kn / rstd formation; v35 production applies rstd after the MMA and never forms a BF16 qn, so its
    correctness is covered by out / grad (``test_l2n_matches_the_exact_reference``,
    ``test_l2n_hook_leaves_out_and_every_gradient_bitwise_unchanged``) and R-L3-2."""
    device = _l2n_device()
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    q, k = inputs["q"].cpu(), inputs["k"].cpu()
    qn_ref, rq_ref = ref_l2n.qk_l2norm(q)
    kn_ref, rk_ref = ref_l2n.qk_l2norm(k)
    out, saved = _fwd(inputs, raw_gate)
    fwd = {n: l2n_hook[f"fwd.{n}"].cpu() for n in ("qn", "kn", "qk_rstd")}
    assert _bits(fwd["qn"]).equal(_bits(qn_ref)) and _bits(fwd["kn"]).equal(
        _bits(kn_ref)
    )
    assert _bits(fwd["qk_rstd"]).equal(_bits(torch.stack((rq_ref, rk_ref))))
    assert _bits(saved.qk_rstd.cpu()).equal(_bits(fwd["qk_rstd"]))
    grads = _bwd(inputs, saved, raw_gate)
    paths = {l2n_hook["bwd.path"]: [g.clone() for g in grads]}
    for name in ("qn", "kn"):
        assert _bits(l2n_hook[f"bwd.{name}"].cpu()).equal(_bits(fwd[name])), (
            name,
            l2n_hook["bwd.path"],
        )
    replay = _bwd(inputs, None, raw_gate)
    assert l2n_hook["bwd.path"] == "replay"
    for name in ("qn", "kn", "qk_rstd"):
        assert _bits(l2n_hook[f"bwd.{name}"].cpu()).equal(_bits(fwd[name])), (
            "replay",
            name,
        )
    out_t, saved_t = _fwd(inputs, raw_gate, save_mode="tape")
    grads_t = _bwd(inputs, saved_t, raw_gate)
    assert l2n_hook["bwd.path"] == "tape"
    for name in ("qn", "kn"):
        assert _bits(l2n_hook[f"bwd.{name}"].cpu()).equal(_bits(fwd[name])), (
            "tape",
            name,
        )
    ref = list(paths.values())[0]
    for tag, other in (("replay", replay), ("tape", grads_t)):
        for a, b in zip(ref, other, strict=True):
            assert _bits(a).equal(_bits(b)), tag
    assert _bits(out_t).equal(_bits(out))
    out2, saved2 = _fwd(inputs, raw_gate)
    assert _bits(out2).equal(_bits(out)) and _bits(saved2.qk_rstd).equal(
        _bits(saved.qk_rstd)
    )
    for a, b in zip(ref, _bwd(inputs, saved2, raw_gate), strict=True):
        assert _bits(a).equal(_bits(b)), "repeat"


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:4], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_hook_leaves_out_and_every_gradient_bitwise_unchanged(
    shape, raw_gate: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The test hook (``_l2n_record``; the fused intra program then also writes the token-layout qn / kn,
    ``write_qkn = 1``) does not perturb the computation: on the same inputs, ``out`` and every gradient of each
    backward route the hook rows cover (light, tape, replay) are bitwise the hook-off ones."""
    device = _l2n_device()
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)

    def run(route: str) -> list[torch.Tensor]:
        out, saved = _fwd(
            inputs, raw_gate, **({"save_mode": "tape"} if route == "tape" else {})
        )
        grads = _bwd(inputs, None if route == "replay" else saved, raw_gate)
        torch.cuda.synchronize(device)
        return [out.clone(), *(g.clone() for g in grads)]

    for route in ("light", "tape", "replay"):
        monkeypatch.setattr(cake_backend, "_l2n_record", None)
        off = run(route)
        store: dict = {}
        monkeypatch.setattr(cake_backend, "_l2n_record", store)
        on = run(route)
        monkeypatch.setattr(cake_backend, "_l2n_record", None)
        assert "fwd.qn" in store and "bwd.qn" in store, route  # the hook was on
        if route == "light":
            assert store["bwd.path"] not in ("tape", "replay"), store["bwd.path"]
        else:
            assert store["bwd.path"] == route, (route, store["bwd.path"])
        for i, (a, b) in enumerate(zip(off, on, strict=True)):
            name = "out" if i == 0 else f"grad[{i - 1}]"
            assert a.dtype == b.dtype and a.shape == b.shape, (route, name)
            assert torch.equal(_bits(a), _bits(b)), (route, name)


@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:3], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_checkpoint_recompute_is_bitwise_the_first_forward(
    shape, l2n_hook: dict
) -> None:
    """Under torch.utils.checkpoint the recompute's rstd / qn / kn and every gradient equal the plain autograd
    run's bit for bit."""
    device = _l2n_device()
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device)
    names = ("q", "k", "v", "g", "beta")

    def run(ckpt: bool):
        leaves = [inputs[n].detach().clone().requires_grad_(True) for n in names]

        def f(*xs):
            return cake_backend.cake_kda_det(
                *xs, inputs["cu_seqlens"], use_qk_l2norm_in_kernel=True
            )

        out = (
            torch.utils.checkpoint.checkpoint(f, *leaves, use_reentrant=False)
            if ckpt
            else f(*leaves)
        )
        out.backward(inputs["dout"])
        return (
            out.detach(),
            [x.grad for x in leaves],
            {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in l2n_hook.items()
            },
        )

    out_a, grads_a, rec_a = run(False)
    out_b, grads_b, rec_b = run(True)
    assert _bits(out_a).equal(_bits(out_b))
    for a, b in zip(grads_a, grads_b, strict=True):
        assert _bits(a).equal(_bits(b))
    for key in ("fwd.qn", "fwd.kn", "fwd.qk_rstd", "bwd.qn", "bwd.kn"):
        assert _bits(rec_a[key]).equal(_bits(rec_b[key])), key


@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:3], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_packed_views_equal_their_contiguous_copies(shape) -> None:
    """Contract C11: q / k as column slices of one packed projection output give the bits of their contiguous
    copies (forward, saved rstd and every gradient); no copy is made of the views."""
    device = _l2n_device()
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device)
    S = sum(lengths)
    packed = torch.zeros(S, 3 * H * D + 8, dtype=torch.bfloat16, device=device)
    packed[:, 8 : 8 + H * D] = inputs["q"].reshape(S, -1)
    packed[:, 8 + H * D : 8 + 2 * H * D] = inputs["k"].reshape(S, -1)
    viewed = dict(
        inputs,
        q=packed[:, 8 : 8 + H * D].view(S, H, D),
        k=packed[:, 8 + H * D : 8 + 2 * H * D].view(S, H, D),
    )
    out_a, saved_a = _fwd(inputs)
    out_b, saved_b = _fwd(viewed)
    assert _bits(out_a).equal(_bits(out_b)) and _bits(saved_a.qk_rstd).equal(
        _bits(saved_b.qk_rstd)
    )
    for a, b in zip(_bwd(inputs, saved_a), _bwd(viewed, saved_b), strict=True):
        assert _bits(a).equal(_bits(b))


@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:3], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_fused_equals_the_fallback_route(shape) -> None:
    """The force-fallback knob (an explicit fallback plan / workspace): forward, saved rstd and every gradient bit
    for bit the fused route's."""
    device = _l2n_device()
    _l2n_device("fallback")
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device)
    S, N = sum(lengths), len(lengths)
    ws = cake_backend.CakeKDADetWorkspace.for_problem(S, N, H, device, l2n="fallback")
    assert ws.l2n == "fallback" and ws.plan.dispatched("qk_pullback", "light")
    out_a, saved_a = _fwd(inputs)
    out_b, saved_b = _fwd(inputs, workspace=ws)
    assert _bits(out_a).equal(_bits(out_b)) and _bits(saved_a.qk_rstd).equal(
        _bits(saved_b.qk_rstd)
    )
    for a, b in zip(
        _bwd(inputs, saved_a), _bwd(inputs, saved_b, workspace=ws), strict=True
    ):
        assert _bits(a).equal(_bits(b))
    for a, b in zip(_bwd(inputs, None), _bwd(inputs, None, workspace=ws), strict=True):
        assert _bits(a).equal(_bits(b)), "replay"


@pytest.mark.parametrize(
    "shape", SHAPES_L2N[1:3], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_fallback_launch_counts_are_the_launches(shape) -> None:
    """On the fallback route ``Plan.launch_count`` is the number of programs each backward launches: the saved-set
    backward applies the saved rstd (``qk_apply``), the replay recomputes it (``qk_l2norm``) and launches no
    ``qk_apply`` although its plan selects one."""
    device = _l2n_device("fallback")
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device)
    S, N = sum(lengths), len(lengths)
    launched: list[str] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    ws = cake_backend.CakeKDADetWorkspace.for_problem(
        S, N, H, device, launcher=recording, l2n="fallback"
    )
    p = ws.plan
    _, saved = _fwd(inputs, workspace=ws)
    launched.clear()
    _bwd(inputs, saved, workspace=ws)
    torch.cuda.synchronize(device)
    assert len(launched) == p.launch_count("bwd")
    assert p.record("qk_apply", p.light_path) in launched
    launched.clear()
    _bwd(inputs, None, workspace=ws)
    torch.cuda.synchronize(device)
    assert len(launched) == p.launch_count("replay")
    assert p.dispatched("qk_apply", "replay")
    assert p.record("qk_apply", "replay") not in launched
    assert p.record("qk_l2norm", "replay") in launched


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize("save_mode", ["light", "tape"])
@pytest.mark.parametrize("mode", ["fused", "fallback"])
def test_l2n_saved_set_backward_allocates_what_the_flag_off_backward_does(
    mode: str, save_mode: str, raw_gate: bool
) -> None:
    """The saved-set backward with use_qk_l2norm_in_kernel=True keeps qn / kn in its arena: on caller-owned
    workspaces of the same problem its per-call peak is at most the flag-off backward's (dq / dk / dv / dg / dbeta,
    no per-call qn / kn), and memory_report counts none."""
    device = _l2n_device(mode)
    gate = "raw" if raw_gate else "g"
    if mode not in _shipped_l2n_modes(gate):
        pytest.skip(f"this build ships no l2n={mode!r} records of gate mode {gate!r}")
    lengths, H = (4000, 96), 32
    S, N = sum(lengths), len(lengths)
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    ws_on = cake_backend.CakeKDADetWorkspace.for_problem(
        S, N, H, device, save_mode=save_mode, gate=gate, l2n=mode
    )
    ws_off = cake_backend.CakeKDADetWorkspace.for_problem(
        S, N, H, device, save_mode=save_mode, gate=gate
    )
    i = inputs
    qn = ref_l2n.qk_l2norm(i["q"].cpu())[0].to(device)
    kn = ref_l2n.qk_l2norm(i["k"].cpu())[0].to(device)
    _, saved_on = _fwd(inputs, raw_gate, workspace=ws_on)
    _, saved_off = cake_backend.cake_kda_det_train_forward(
        qn,
        kn,
        i["v"],
        _g(i, raw_gate),
        i["beta"],
        i["cu_seqlens"],
        workspace=ws_off,
        **_gate_kw(i, raw_gate),
    )

    def peak(fn) -> int:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        base = torch.cuda.memory_allocated(device)
        grads = fn()
        torch.cuda.synchronize(device)
        del grads
        return torch.cuda.max_memory_allocated(device) - base

    peak_on = peak(lambda: _bwd(inputs, saved_on, raw_gate, workspace=ws_on))
    peak_off = peak(
        lambda: cake_backend.cake_kda_det_train_backward(
            qn,
            kn,
            i["v"],
            _g(i, raw_gate),
            i["beta"],
            i["cu_seqlens"],
            i["dout"],
            saved=saved_off,
            workspace=ws_off,
            **_gate_kw(i, raw_gate),
        )
    )
    assert peak_on <= peak_off, (peak_on, peak_off)
    report = cake_backend.memory_report(
        S,
        N,
        H,
        save_mode,
        sm_count=ws_on.plan.sm_count,
        arch=ws_on.plan.arch,
        gate=gate,
        l2n=mode,
    )
    assert report["bwd_qn_kn"] == 0


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize(
    "shape", SHAPES_L2N[:4], ids=lambda s: f"{len(s[0])}docs-S{sum(s[0])}-H{s[1]}"
)
def test_l2n_fallback_route_matches_the_exact_reference(
    shape, raw_gate: bool, l2n_hook: dict
) -> None:
    """The force-fallback route alone (token-local normalize / apply, the GLM intra capture forms, the pullback),
    runnable before the fused records ship: the forward's rstd / qn / kn are bitwise the shared reference's, the
    saved-set backward's qn / kn are the forward's, out and every gradient pass the declared elementwise gate against
    the exact FP64 layer, and the replay and a repeat give the saved-set backward's bits."""
    device = _l2n_device("fallback")
    lengths, H = shape
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    S, N = sum(lengths), len(lengths)
    ws = cake_backend.CakeKDADetWorkspace.for_problem(
        S, N, H, device, gate="raw" if raw_gate else "g", l2n="fallback"
    )
    assert ws.l2n == "fallback" and ws.plan.dispatched("qk_pullback", "light")
    out, saved = _fwd(inputs, raw_gate, workspace=ws)
    fwd = {n: l2n_hook[f"fwd.{n}"].cpu() for n in ("qn", "kn", "qk_rstd")}
    grads = _bwd(inputs, saved, raw_gate, workspace=ws)
    torch.cuda.synchronize(device)
    i = inputs
    qn_ref, rq_ref = ref_l2n.qk_l2norm(i["q"].cpu())
    kn_ref, rk_ref = ref_l2n.qk_l2norm(i["k"].cpu())
    assert _bits(fwd["qn"]).equal(_bits(qn_ref)) and _bits(fwd["kn"]).equal(
        _bits(kn_ref)
    )
    assert _bits(fwd["qk_rstd"]).equal(_bits(torch.stack((rq_ref, rk_ref))))
    assert _bits(saved.qk_rstd.cpu()).equal(_bits(fwd["qk_rstd"]))
    for name in ("qn", "kn"):
        assert _bits(l2n_hook[f"bwd.{name}"].cpu()).equal(_bits(fwd[name])), name
    if raw_gate:
        ref_out, ref = ref_l2n.kda_exact_raw_l2n_with_grads(
            i["q"],
            i["k"],
            i["v"],
            i["g"],
            i["beta"],
            i["A_log"],
            i["dt_bias"],
            i["dout"],
            i["cu_seqlens"],
            cake_backend.SCALE,
            LOWER_BOUND,
        )
        names = ("dq", "dk", "dv", "dbeta", "dg", "dA_log", "ddt_bias")
    else:
        ref_out, ref = ref_l2n.kda_exact_l2n_with_grads(
            i["q"],
            i["k"],
            i["v"],
            i["g"],
            i["beta"],
            i["dout"],
            i["cu_seqlens"],
            cake_backend.SCALE,
        )
        names = ("dq", "dk", "dv", "dbeta", "dg")
    candidates = {"out": out, **dict(zip(names, grads, strict=True))}
    references = {"out": ref_out, **ref}
    failures = []
    for name, cand in candidates.items():
        r = references[name].reshape(cand.shape)
        gate = elementwise_gate(cand, r, ATOL, RTOL)
        if not gate["pass"]:
            failures.append((name, gate, error_metrics(cand, r)))
    assert not failures, failures
    ref_grads = [g.clone() for g in grads]
    replay = _bwd(inputs, None, raw_gate, workspace=ws)
    assert l2n_hook["bwd.path"] == "replay"
    out2, saved2 = _fwd(inputs, raw_gate, workspace=ws)
    assert _bits(out2).equal(_bits(out))
    repeat = _bwd(inputs, saved2, raw_gate, workspace=ws)
    for tag, other in (("replay", replay), ("repeat", repeat)):
        for a, b in zip(ref_grads, other, strict=True):
            assert _bits(a).equal(_bits(b)), tag


def test_l2n_refusals_on_device() -> None:
    """Saved sets and workspaces of the other mode are refused; flag-off workspaces refuse flag-on calls."""
    device = _l2n_device()
    lengths, H = (300, 200), 2
    inputs = _l2n_inputs(lengths, H, device)
    i = inputs
    out, saved = _fwd(inputs)
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        cake_backend.cake_kda_det_train_backward(
            i["q"].contiguous(),
            i["k"].contiguous(),
            i["v"],
            i["g"],
            i["beta"],
            i["cu_seqlens"],
            i["dout"],
            saved=saved,
        )
    ws_off = cake_backend.CakeKDADetWorkspace.for_problem(sum(lengths), 2, H, device)
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
        _fwd(inputs, workspace=ws_off)
    bad_q = torch.zeros(sum(lengths), H, 2 * D, dtype=torch.bfloat16, device=device)[
        :, :, ::2
    ]
    with pytest.raises(ValueError, match="never copied"):
        _fwd(dict(inputs, q=bad_q))


@pytest.mark.parametrize("raw_gate", [False, True], ids=["g", "raw"])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize(
    "lengths, batch_tokens",
    [((100, 126, 50), 226), ((100, 128, 50), 228)],
    ids=["overhang", "aligned"],
)
def test_l2n_batched_replay_reads_no_unwritten_row_of_the_next_group(
    lengths, batch_tokens: int, H: int, raw_gate: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The batched arena's replay with use_qk_l2norm_in_kernel=True on the serial route (H 32 on every
    architecture, H 16 on sm_100a): the plain intra program of group 0 box-reads up to BT - 1 rows past the group
    end of the per-call qn / kn (``"overhang"``: document 1 ends 62 rows into its last chunk), which group 1's
    prep writes only later.  With every per-call float tensor and the arena filled with NaN, the replay before and
    after a forward and a saved-set backward on the workspace gives finite gradients bitwise the per-call whole
    problem's (``"aligned"``: the group ends on a chunk boundary, the control)."""
    device = _l2n_device()
    gate = "raw" if raw_gate else "g"
    S, N = sum(lengths), len(lengths)
    assert cake_backend.doc_batches(lengths, batch_tokens) == ((0, 2), (2, 3))
    inputs = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    ref = [t.clone() for t in _bwd(inputs, None, raw_gate)]
    alloc = cake_backend._alloc

    def nan_alloc(shape, dtype, dev):
        t = alloc(shape, dtype, dev)
        if t.is_floating_point() and t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    monkeypatch.setattr(cake_backend, "_alloc", nan_alloc)
    arena = cake_backend.CakeKDADetArena(batch_tokens=batch_tokens)
    for case in ("replay first", "replay after"):
        ws = cake_backend.CakeKDADetWorkspace.for_problem(
            S, N, H, device, arena=arena, doc_lengths=lengths, gate=gate, l2n="fused"
        )
        ws.arena.fill_(0xFF)
        kw = {"workspace": ws, "doc_lengths": lengths}
        runs = []
        if case == "replay after":
            _, saved = _fwd(inputs, raw_gate, **kw)
            runs.append(("saved", _bwd(inputs, saved, raw_gate, **kw)))
        runs.append((case, _bwd(inputs, None, raw_gate, **kw)))
        for tag, got in runs:
            for i, (a, b) in enumerate(zip(got, ref, strict=True)):
                assert bool(torch.isfinite(a).all()), (H, gate, tag, i)
                assert _bits(a).equal(_bits(b)), (H, gate, tag, i)
        del ws
