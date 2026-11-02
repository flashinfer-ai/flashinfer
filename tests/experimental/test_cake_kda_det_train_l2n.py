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

from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")

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
    for name in ("q_raw", "k_raw", "qn", "kn", "qk_rstd"):
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
    ):
        assert name in cake_jit.LOGICAL_SCALARS


def _saved(S: int, H: int, **fields):
    a = torch.zeros(S, H, 64, dtype=torch.bfloat16)
    return cake_backend.KdaDetChunkSaved(
        cake_backend.SAVED_VERSION, a, a, None, S, H, 1, cake_backend.SCALE, **fields
    )


def test_saved_set_fields_and_their_checks() -> None:
    S, H = 3, 2
    fields = {f.name: f.default for f in dataclasses.fields(cake_backend.KdaDetChunkSaved)}
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
    with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel=False; this backward"):
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


@pytest.mark.parametrize("mode", ["fwd", "fwd_tape", "fwd_kcp"])
@pytest.mark.parametrize("l2n", ["fused", "fallback"])
def test_forward_layouts_carry_transient_qn_kn_on_the_u_w_slots(mode: str, l2n: str) -> None:
    off = _rows(mode)
    on = _rows(mode, l2n=l2n)
    assert "qn" not in off and "kn" not in off
    for name, host in (("qn", "u"), ("kn", "w")):
        r = on[name]
        assert r["slot"] == on[host]["slot"] and r["kind"] == "B16"
        assert r["sub_offset_per_th"] == 0
        assert (r["first_write"], r["last_read"]) == (0, 1)
    # the same bytes: the layout's per-token-head size does not grow
    R, H, NC = 128, 2, 2
    assert cake_backend._mode_bytes(R, H, NC, 0, mode, 1, True, l2n) == cake_backend._mode_bytes(
        R, H, NC, 0, mode, 1, True, "off"
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


def test_gate_fused_on_every_arch_in_l2n_modes() -> None:
    for arch in cake_jit.ARCHES:
        assert cake_backend._gate_fused(arch, "g", "fused") is True
        assert cake_backend._gate_fused(arch, "g", "fallback") is True


# ---------------------------------------------------------------------------------------------------------------
# registry


def test_select_record_refuses_a_mode_without_records(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-optional l2n stage whose records do not serve the call's mode is refused, never silently routed to
    another class; an optional stage selects nothing."""
    scalars = dict(num_chunks_pad=2, num_seqs=1, num_heads=1, sm_count=148, arch=cake_jit.ARCHES[0])
    for stage in ("fwd_intra", "bwd_prep", "glm_intra"):
        path = "fwd" if stage == "fwd_intra" else "light"
        if any("fused" in cake_jit.MODULES[n].get("l2n", ["off"]) for n in cake_jit.records_of_stage(stage)):
            continue  # the class ships: covered by the plan tests
        with pytest.raises(ValueError, match="no record serves the in-kernel Q/K L2 norm mode"):
            cake_jit.select_record(stage, **scalars, path=path, l2n="fused")
    for stage in ("qk_l2norm", "qk_apply", "qk_pullback"):
        path = "fwd" if stage == "qk_l2norm" else "light"
        if cake_jit.records_of_stage(stage):
            continue
        assert cake_jit.select_record(stage, **scalars, path=path, l2n="fallback") is None
    with pytest.raises(ValueError, match="l2n 'on' is not one of"):
        cake_jit.select_record("dqkg", **scalars, path="light", l2n="on")


def test_tokens_grid_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    record = {"stage": "qk_l2norm", "grid_rule": {"kind": "tokens"}, "select_rule": {"kind": "always"}}
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


def test_record_hook_clones_and_is_off_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    assert cake_backend._l2n_record is None
    x = torch.arange(4.0)
    cake_backend._record_l2n("fwd", qn=x)  # no hook: nothing recorded, no error
    store: dict = {}
    monkeypatch.setattr(cake_backend, "_l2n_record", store)
    cake_backend._record_l2n("bwd", qn=x, path="light")
    assert store["bwd.path"] == "light" and torch.equal(store["bwd.qn"], x)
    assert store["bwd.qn"].data_ptr() != x.data_ptr()


def test_autograd_node_returns_one_gradient_slot_per_input(monkeypatch: pytest.MonkeyPatch) -> None:
    import inspect

    n_inputs = len(inspect.signature(cake_backend.ChunkKDADetFunction.forward).parameters) - 1
    seen = {}

    def fake_bwd(*args, **kwargs):
        seen.update(kwargs)
        return tuple(torch.zeros(1) for _ in range(5))

    monkeypatch.setattr(cake_backend, "chunk_kda_det_bwd", fake_bwd)
    t = torch.zeros(1)
    ctx = types.SimpleNamespace(
        saved_tensors=(t, t, t, t, t, t, t, t, t),  # q k v g beta cu_seqlens Aqk Akk qk_rstd
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
    grads = cake_backend.ChunkKDADetFunction.backward(ctx, torch.zeros(1))
    assert len(grads) == n_inputs
    assert seen["use_qk_l2norm_in_kernel"] is True
    assert seen["saved"].qk_l2norm is True and seen["saved"].qk_rstd is t
