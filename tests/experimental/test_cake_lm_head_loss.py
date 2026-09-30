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
"""

import contextlib
import math
import os
import subprocess
import sys
import threading
import warnings
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from flashinfer.api_logging import ExperimentalWarning
from flashinfer.chunked_lm_head import chunked_lm_head_logprob, chunked_lm_head_loss
from flashinfer.experimental.cake_lm_head_loss import cake_backend, cake_jit
from flashinfer.experimental.cake_lm_head_loss.cake_backend import (
    ABI_CONTRACT,
    COMMON_SCALARS,
    COMMON_TENSORS,
    CONTRACT_ALIASES,
    MODE_CE,
    MODE_NONE,
    MODE_POLICY,
    STAGE_TENSORS,
    SUPPORTED_ABIS,
    SUPPORTED_COMPUTE_CAPABILITIES,
    WORKSPACE_ALIGN,
    BindingCache,
    Geometry,
    bind_stage,
    cluster_resident,
    forward_binding_key,
    generated_program_available,
    grid_dims,
    lm_head_loss_workspace_size,
    logprob_backward_binding_key,
    memory_report,
    plan_chunks,
    prepare_lm_head_loss,
    recommended_k_slices,
    record_abi,
    stage_values,
    stages_for_entry,
    validate_lm_head_inputs,
    wave_efficiency,
    workspace_layout,
)
from tests.test_helpers.cake_lm_head_loss_reference import (
    DEFAULT_H,
    DEFAULT_V,
    IGNORE_INDEX,
    KNEE_MARGIN,
    error_report,
    make_inputs,
    reference_fp64,
    reference_unchunked,
    rel_l2,
)

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 20260930
# Host-layer shapes (the reference backend runs at any multiple of 256).
H_HOST, V_HOST = 256, 512
# Accuracy gates against the FP64 oracle: at most 1.05x the error of the unchunked PyTorch path on
# the same inputs plus a floor per metric.
GATE_MARGIN = 1.05
GATE_TINY = dict(loss_rel=1e-6, logp_max_abs=1e-5, dX_rel_l2=1e-3, dW_rel_l2=1e-3)
# Errors of the unchunked PyTorch path against the FP64 oracle measured at the production geometry
# (H = 6144, V = 154880).  The device tests also hold every error below twice these figures as a
# documented sanity ceiling; the primary gate stays the per-input 1.05x.
PRODUCTION_B0 = dict(dX_rel_l2=1.85e-3, dW_rel_l2=1.88e-3, logp_max_abs=4.1e-3)
# Per-row gate between two BF16 implementations of the same rows (knee test).
ROW_REL_L2 = 1e-2
# Host values every stage may name (the kernels' own argument names, see ``stage_values``).
_GEMM_VALUES = {"A", "B", "C", "STATS_OUT", "M", "m_tiles", "k_iters"}
_STAGE_VALUES = {
    stage: set(tensors) | set(COMMON_TENSORS) | set(COMMON_SCALARS) | {"num_vecs"} | (_GEMM_VALUES if stage.startswith("gemm") else set())
    for stage, tensors in STAGE_TENSORS.items()
}


def _device_supported() -> bool:
    return torch.cuda.is_available() and (torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES)


def _require_program(*, entry: str = "loss"):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not generated_program_available(torch.device("cuda"), entry=entry):
        pytest.skip(f"generated chunked LM-head program ({entry} entry) not registered for this device")


@contextlib.contextmanager
def _quiet_experimental():
    # The experimental banner fires once per process; the API's opt-in is
    # exercised by test_public_api_is_marked_experimental.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ExperimentalWarning)
        yield


@contextlib.contextmanager
def _cache(enabled: bool):
    cache = cake_backend.BINDING_CACHE
    was = cache.enabled
    cache.enabled = enabled
    try:
        yield cache
    finally:
        cache.enabled = was


def _host_inputs(T, objective="ce", *, seed=SEED, **kw):
    return make_inputs(T, objective=objective, seed=seed + T, H=H_HOST, V=V_HOST, device=DEV, **kw)


def _run(inp, C, *, entry="loss", grad_weight_dtype=torch.bfloat16, train_x=True, train_w=True, backend="reference", scale=None,
         compact_rows=None):
    """One forward + backward through the autograd entry points; ``backend="cake"`` goes
    through the public module.  Returns ``loss`` / ``logp`` / ``dX`` / ``dW`` and the leaves."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    loss_fn = chunked_lm_head_loss if backend == "cake" else cake_backend.chunked_lm_head_loss
    logprob_fn = chunked_lm_head_logprob if backend == "cake" else cake_backend.chunked_lm_head_logprob
    with _quiet_experimental():
        if entry == "loss":
            loss, logp = loss_fn(
                X, W, inp.labels, objective=inp.objective, loss_div=inp.loss_div if inp.objective == "ce" else None,
                infer_logp=inp.infer_logp, loss_weights=inp.loss_weights, chunk_size=C, return_logp=True,
                grad_weight_dtype=grad_weight_dtype, backend=backend, compact_rows=compact_rows,
            )
            if train_x or train_w:
                (loss if scale is None else loss * scale).backward()
        else:
            logp = logprob_fn(X, W, inp.labels, chunk_size=C, backend=backend, compact_rows=compact_rows)
            loss = (logp.detach() * inp.dlogp)[inp.valid].sum()
            if train_x or train_w:
                logp.backward(inp.dlogp if scale is None else inp.dlogp * scale)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad, X=X, W=W)


def _check_against_references(result, inp, *, entry="loss", grad_weight_dtype=torch.bfloat16, ceiling=False):
    oracle = reference_fp64(inp, entry=entry)
    b0 = reference_unchunked(inp, entry=entry, grad_weight_dtype=grad_weight_dtype)
    errors = error_report(result, oracle, inp.labels)
    b0_errors = error_report(b0, oracle, inp.labels)
    assert not errors["nan"]
    for key, tiny in GATE_TINY.items():
        if key in errors:
            assert errors[key] <= GATE_MARGIN * b0_errors[key] + tiny, f"{key}: {errors[key]:.3e} vs unchunked {b0_errors[key]:.3e}"
    if ceiling:
        for key, measured in PRODUCTION_B0.items():
            if key in errors:
                assert errors[key] <= 2.0 * measured, f"{key}: {errors[key]:.3e} above twice the production-geometry unchunked error"
    assert errors["logp_ignored_zero"]
    if result.get("dX") is not None:
        assert errors["dX_ignored_zero"]
    return errors, b0_errors


def _check_dtypes(result, inp, *, grad_weight_dtype=torch.bfloat16):
    assert result["loss"].dtype == torch.float32 and result["loss"].shape == ()
    assert result["logp"].dtype == torch.float32 and tuple(result["logp"].shape) == (inp.T,)
    if result["dX"] is not None:
        assert result["dX"].dtype == torch.bfloat16 and tuple(result["dX"].shape) == (inp.T, inp.H)
    if result["dW"] is not None:
        assert result["dW"].dtype == grad_weight_dtype and tuple(result["dW"].shape) == (inp.V, inp.H)


def _same(a, b) -> bool:
    if a is None or b is None:
        return a is b
    return torch.equal(a, b)


# ---------------------------------------------------------------------------
# Host layer (any device; the reference backend)
# ---------------------------------------------------------------------------


def test_public_api_is_marked_experimental():
    assert chunked_lm_head_loss.is_experimental is True
    assert chunked_lm_head_logprob.is_experimental is True
    assert "SM100" in chunked_lm_head_loss.__doc__
    assert "SM100" in chunked_lm_head_logprob.__doc__
    X = torch.zeros(4, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(4, dtype=torch.int64)
    # the public module serves the cake backend only
    with pytest.raises(ValueError, match="backend"):
        chunked_lm_head_loss(X, W, labels, loss_div=1.0, backend="reference")
    with pytest.raises(ValueError, match="backend"):
        chunked_lm_head_logprob(X, W, labels, backend="reference")


def test_registry_records_are_well_formed():
    assert cake_jit.STAGES == (
        "gemm_logits", "gemm_logits_nostats", "row_finalize", "loss_reduce", "row_grad", "gemm_dx", "gemm_dx_s2",
        "gemm_dx_s3", "gemm_dx_s4", "gemm_dw_acc", "scale_cast_bf16", "scale_cast_f32",
    )
    assert set(STAGE_TENSORS) == set(cake_jit.STAGES)
    assert set(CONTRACT_ALIASES.values()) <= set().union(*_STAGE_VALUES.values())
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCH_NVCC_FLAGS
        assert record["arch"] in SUPPORTED_COMPUTE_CAPABILITIES.values()
        assert record_abi(record) in SUPPORTED_ABIS
        assert len(record["closure_sha256"]) == 64
        assert cake_jit.select_module(record["arch"]) == name
        geometry = Geometry.from_record(record)
        assert geometry.hidden is None or geometry.hidden % geometry.hidden_multiple == 0
        assert geometry.vocab is None or geometry.vocab % geometry.vocab_multiple == 0
        stages = cake_jit.registered_stages(name)
        assert stages and set(stages) <= set(cake_jit.STAGES)
        for stage in stages:
            physical = record[stage]
            assert physical["module"] and physical["ffi_entry"]
            assert len(physical["sources"]) == 2
            assert all(s.startswith("cake_lm_head_loss/") for s in physical["sources"])
            assert isinstance(physical["compile_flags"], list)
            assert len(physical["closure_sha256"]) == 64
            kinds = {kind for kind, _ in physical["arg_plan"]}
            assert kinds <= {"buffer", "tma_buffer", "workspace", "parameter", "grid"}
            names = [n for _, n in physical["arg_plan"]]
            assert len(names) == len(set(names))
            for kind, arg in physical["arg_plan"]:
                if kind == "grid":
                    assert arg in ("grid_x", "grid_y", "grid_z")
                else:  # every kernel argument is a host value of its stage (directly or through an alias)
                    assert CONTRACT_ALIASES.get(arg, arg) in _STAGE_VALUES[stage], (stage, arg)
            assert int(physical.get("tma_workspace_bytes", 0)) >= 0
            assert int(physical.get("workspace_bytes", 0)) >= 0
            assert len(physical.get("grid", ["rows_c", 1, 1])) == 3
            launch = physical.get("launch")
            if launch is not None:
                assert len(launch["block"]) == 3 and all(int(b) >= 1 for b in launch["block"])
                assert len(launch["cluster"]) == 3 and all(int(c) >= 1 for c in launch["cluster"])


@pytest.mark.parametrize("C", [4096, 2048, 8192])
@pytest.mark.parametrize("T", [0, 1, 4095, 4096, 4097, 16231, 16172, 32463])
def test_plan_chunks(T, C):
    chunks = plan_chunks(T, C)
    assert len(chunks) == -(-T // C)
    assert sum(rows for _, rows in chunks) == T
    next_row = 0
    for i, (row0, rows) in enumerate(chunks):
        assert row0 == next_row and 1 <= rows <= C
        if i < len(chunks) - 1:
            assert rows == C  # only the last chunk may be short
        next_row += rows


def test_plan_chunks_examples():
    assert plan_chunks(0, 4096) == ()
    assert plan_chunks(1, 4096) == ((0, 1),)
    assert plan_chunks(4096, 4096) == ((0, 4096),)
    assert plan_chunks(4097, 4096) == ((0, 4096), (4096, 1))
    assert plan_chunks(16231, 4096)[-1] == (12288, 3943)
    assert plan_chunks(16231, 4096)[:-1] == ((0, 4096), (4096, 4096), (8192, 4096))
    with pytest.raises(ValueError):
        plan_chunks(5, 0)


def test_stages_for_entry():
    full = stages_for_entry("loss")
    assert full == ("gemm_logits", "row_finalize", "loss_reduce", "row_grad", "gemm_dx", "gemm_dw_acc", "scale_cast_bf16")
    assert list(full) == [s for s in cake_jit.STAGES if s in full]  # launch order
    frozen_x = stages_for_entry("loss", need_dx=False)
    assert "gemm_dx" not in frozen_x and "gemm_dw_acc" in frozen_x and "scale_cast_bf16" in frozen_x
    frozen_w = stages_for_entry("loss", need_dw=False)
    assert "gemm_dw_acc" not in frozen_w and "gemm_dx" in frozen_w and "scale_cast_f32" not in frozen_w
    frozen_w32 = stages_for_entry("loss", need_dw=False, grad_weight_dtype=torch.float32)
    assert "scale_cast_f32" not in frozen_w32  # the FP32 cast belongs to dW only
    neither = stages_for_entry("loss", need_dx=False, need_dw=False)
    assert neither == ("gemm_logits", "row_finalize", "loss_reduce")
    fp32 = stages_for_entry("loss", grad_weight_dtype=torch.float32)
    assert "scale_cast_f32" in fp32 and "scale_cast_bf16" in fp32  # dW in FP32, dX in BF16
    logprob = stages_for_entry("logprob")
    assert "gemm_logits_nostats" in logprob and "loss_reduce" not in logprob
    assert logprob == ("gemm_logits", "gemm_logits_nostats", "row_finalize", "row_grad", "gemm_dx", "gemm_dw_acc", "scale_cast_bf16")
    assert "scale_cast_f32" not in stages_for_entry("logprob", grad_weight_dtype=torch.float32)  # dW is BF16 there
    assert stages_for_entry("logprob", need_dx=False, need_dw=False) == ("gemm_logits", "gemm_logits_nostats", "row_finalize")
    with pytest.raises(ValueError):
        stages_for_entry("other")


def _valid_call(T=4):
    X = torch.zeros(T, H_HOST, dtype=torch.bfloat16)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16)
    labels = torch.zeros(T, dtype=torch.int64)
    return dict(X=X, W=W, labels=labels, objective="ce", loss_div=1.0)


def test_validate_accepts_strided_x():
    kw = _valid_call()
    p = validate_lm_head_inputs(**kw)
    assert (p.num_rows, p.hidden, p.vocab, p.chunk) == (4, H_HOST, V_HOST, 4096)
    assert (p.ld_x, p.x_copy, p.objective, p.loss_div, p.entry, p.mode) == (H_HOST, False, "ce", 1.0, "loss", MODE_CE)
    wide = torch.zeros(4, H_HOST + 8, dtype=torch.bfloat16)[:, :H_HOST]
    p = validate_lm_head_inputs(**dict(kw, X=wide))
    assert not p.x_copy and p.ld_x == H_HOST + 8  # a 16 B pitch is launched as is
    odd = torch.zeros(4, H_HOST + 3, dtype=torch.bfloat16)[:, :H_HOST]
    p = validate_lm_head_inputs(**dict(kw, X=odd))
    assert p.x_copy and p.ld_x == H_HOST  # copied to a contiguous tensor
    p = validate_lm_head_inputs(**dict(kw, X=kw["X"][:0], labels=kw["labels"][:0]))
    assert p.num_rows == 0 and not p.x_copy
    policy = validate_lm_head_inputs(**dict(kw, objective="policy", loss_div=None, infer_logp=torch.zeros(4), loss_weights=torch.zeros(4)))
    assert policy.mode == MODE_POLICY and policy.loss_div is None
    logprob = validate_lm_head_inputs(**dict(kw, loss_div=None), entry="logprob")
    assert logprob.mode == MODE_NONE and logprob.entry == "logprob"
    # loss_div also as a CPU scalar tensor / an int
    assert validate_lm_head_inputs(**dict(kw, loss_div=torch.tensor(2.5))).loss_div == 2.5
    assert validate_lm_head_inputs(**dict(kw, loss_div=3)).loss_div == 3.0


def _cuda_scalar():
    if not torch.cuda.is_available():
        pytest.skip("a CUDA scalar needs a CUDA device")
    return torch.tensor(1.0, device="cuda")


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda kw: dict(kw, X=kw["X"].float()), ValueError, "BF16"),
        (lambda kw: dict(kw, W=torch.zeros(H_HOST, V_HOST, dtype=torch.bfloat16).t()), ValueError, "contiguous"),
        (lambda kw: dict(kw, X=torch.zeros(H_HOST, 4, dtype=torch.bfloat16).t()), ValueError, "contiguous"),
        (lambda kw: dict(kw, X=torch.zeros(4, 320, dtype=torch.bfloat16), W=torch.zeros(V_HOST, 320, dtype=torch.bfloat16)), ValueError, "multiple of 256"),
        (lambda kw: dict(kw, W=torch.zeros(500, H_HOST, dtype=torch.bfloat16)), ValueError, "multiple of 256"),
        (lambda kw: dict(kw, W=torch.zeros(V_HOST, 2 * H_HOST, dtype=torch.bfloat16)), ValueError, "differ"),
        (lambda kw: dict(kw, labels=kw["labels"].int()), ValueError, "int64"),
        (lambda kw: dict(kw, labels=kw["labels"][:3]), ValueError, r"\[T\]"),
        (lambda kw: dict(kw, chunk_size=0), ValueError, "positive integer"),
        (lambda kw: dict(kw, chunk_size=True), ValueError, "positive integer"),
        (lambda kw: dict(kw, chunk_size=70000), ValueError, "65535"),
        (lambda kw: dict(kw, grad_weight_dtype=torch.float16), ValueError, "grad_weight_dtype"),
        (lambda kw: dict(kw, loss_div=None), ValueError, "loss_div"),
        (lambda kw: dict(kw, infer_logp=torch.zeros(4)), ValueError, "policy"),
        (lambda kw: dict(kw, loss_div=0.0), ValueError, "positive"),
        (lambda kw: dict(kw, loss_div=-1.0), ValueError, "positive"),
        (lambda kw: dict(kw, loss_div=_cuda_scalar()), ValueError, "synchronize"),
        (lambda kw: dict(kw, objective="policy", loss_div=None, infer_logp=torch.zeros(4)), ValueError, "loss_weights"),
        (lambda kw: dict(kw, objective="policy", loss_div=None, infer_logp=torch.zeros(4, dtype=torch.float64), loss_weights=torch.zeros(4)), ValueError, "FP32"),
        (lambda kw: dict(kw, objective="policy", loss_div=1.0, infer_logp=torch.zeros(4), loss_weights=torch.zeros(4)), ValueError, "loss_div"),
        (lambda kw: dict(kw, objective="other"), ValueError, "objective"),
        (lambda kw: dict(kw, entry="logprob"), ValueError, "objective arguments"),
        (lambda kw: dict(kw, deterministic=False), NotImplementedError, "deterministic"),
        (lambda kw: dict(kw, geometry=Geometry.from_record({"geometry": {"hidden": DEFAULT_H, "vocab": DEFAULT_V}})), ValueError, "specialized"),
    ],
    ids=[
        "x_fp32", "w_noncontiguous", "x_column_major", "h_not_256", "v_not_256", "hidden_mismatch", "labels_int32",
        "labels_length", "chunk_zero", "chunk_bool", "chunk_too_large", "grad_dtype_fp16", "ce_without_loss_div",
        "ce_with_infer_logp", "loss_div_zero", "loss_div_negative", "loss_div_cuda", "policy_without_weights",
        "policy_infer_fp64", "policy_with_loss_div", "unknown_objective", "logprob_with_objective_args",
        "nondeterministic", "pinned_geometry",
    ],
)
def test_validate_rejects(mutate, exc, match):
    with pytest.raises(exc, match=match):
        validate_lm_head_inputs(**mutate(_valid_call()))


def test_workspace_layout_and_memory_rule():
    V, H, C = DEFAULT_V, DEFAULT_H, 4096
    for T in (1, 100, 4097, 16231):
        rows = min(T, C)
        layout = workspace_layout(T, V, C)
        regions = [k for k in layout if k != "total"]
        assert regions == ["logits", "stats", "d", "term", "loss_acc", "grad_scale"]
        offsets = [layout[k][0] for k in regions]
        assert offsets == sorted(offsets) and all(o % WORKSPACE_ALIGN == 0 for o in offsets)
        assert layout["logits"][1] == rows * V * 2
        assert layout["stats"][1] == rows * (V // 256) * 8
        assert layout["d"][1] == rows * 4 and layout["term"][1] == rows * 4
        assert layout["loss_acc"][1] == 8 and layout["grad_scale"][1] == 4  # FP64 loss accumulator, FP32 scale
        assert layout["total"] % WORKSPACE_ALIGN == 0 and layout["total"] >= sum(layout[k][1] for k in regions)
        assert layout == workspace_layout(rows, V, C)  # depends on T only through min(T, C)
    assert workspace_layout(4097, V, C) == workspace_layout(16231, V, C) == workspace_layout(4096, V, C)
    assert workspace_layout(100, V, C)["logits"][1] < workspace_layout(4096, V, C)["logits"][1]
    assert workspace_layout(0, V, C) == workspace_layout(1, V, C)
    with_scratch = workspace_layout(100, V, C, tma_workspace_bytes=1024, scratch_bytes=4096)
    assert with_scratch["workspace"][1] == 4096 and with_scratch["tma_descriptor_workspace"][1] == 1024
    assert with_scratch["total"] == workspace_layout(100, V, C)["total"] + 4096 + 1024
    assert workspace_layout(100, V, C, stats_tile=128)["stats"][1] == 2 * workspace_layout(100, V, C)["stats"][1]

    T = 16231
    m = memory_report(T, H, V, C)
    for key in ("temporary_bytes", "temporary", "outputs_bytes", "outputs", "accumulator_bytes", "accumulators", "weights_bytes", "weights"):
        assert key in m
    assert m["vocab_rows_max"] == 4096 and m["chunk"] == C and m["num_chunks"] == 4
    assert m["temporary"]["logits"] == 4096 * V * 2 and m["temporary"]["lse"] == T * 4 and m["temporary"]["logp"] == T * 4
    assert m["temporary_bytes"] == sum(m["temporary"].values())
    assert m["accumulators"] == {"dX_acc": T * H * 4, "dW_acc": V * H * 4}
    assert m["outputs"] == {"loss": 4, "logp": 0, "dX": T * H * 2, "dW": V * H * 2}
    assert m["weights"] == {"W": V * H * 2, "X": T * H * 2}
    assert memory_report(T, H, V, C, grad_weight_dtype=torch.float32)["outputs"]["dW"] == 2 * m["outputs"]["dW"]
    returned = memory_report(T, H, V, C, return_logp=True)
    assert returned["outputs"]["logp"] == T * 4 and "logp" not in returned["temporary"]
    assert memory_report(T, H, V, C, x_copy=True)["temporary"]["x_copy"] == T * H * 2
    frozen = memory_report(T, H, V, C, need_dx=False)
    assert "dX_acc" not in frozen["accumulators"] and "dX" not in frozen["outputs"]
    lp = memory_report(T, H, V, C, entry="logprob", need_dx=False)
    assert lp["accumulators"] == {"dW_acc": V * H * 4, "saved_lse": T * 4}
    assert lp["outputs"]["loss"] == 0 and lp["outputs"]["logp"] == T * 4 and lp["temporary"]["dlogp"] == T * 4
    assert lp["outputs"]["dW"] == V * H * 2  # BF16 in the log-probability entry
    assert memory_report(1, H, V, C)["vocab_rows_max"] == 1
    empty = memory_report(0, H, V, C)
    assert empty["num_chunks"] == 0 and empty["vocab_rows_max"] == 1
    # every vocabulary-sized temporary spans at most C rows regardless of T
    for T in (1, 4095, 4096, 4097, 16231, 32463):
        rep = memory_report(T, H, V, C)
        assert rep["vocab_rows_max"] == min(T, C)
        assert rep["temporary"]["logits"] == min(T, C) * V * 2
    assert lm_head_loss_workspace_size(16231, V, C, backend="reference") == workspace_layout(16231, V, C)["total"]
    assert lm_head_loss_workspace_size(16231, V, C, entry="logprob", backend="reference") == workspace_layout(16231, V, C, entry="logprob")["total"]


def test_grid_dims():
    scalars = {"m_tiles": 32, "rows_c": 4097, "V": DEFAULT_V, "num_vecs": 24}
    assert grid_dims(["max(1, min(m_tiles//2*605, sms//2))*2", "rows_c/8", 1], scalars, 148) == (148, 513, 1)
    assert grid_dims(["rows_c//8", 1, 1], scalars, 148) == (512, 1, 1)  # floor
    assert grid_dims(["rows_c/8", 1, 1], scalars, 148) == (513, 1, 1)  # ceil
    assert grid_dims(["V/256", "rows_c", 1], scalars, 148) == (605, 4097, 1)
    assert grid_dims(["sms", "sms*2", "sms-100"], scalars, 148) == (148, 296, 48)
    assert grid_dims([4, 2, 1], scalars, 148) == (4, 2, 1)
    assert grid_dims(["min(num_vecs, 8) + max(2, 3)", 1, 1], scalars, 148) == (11, 1, 1)
    assert grid_dims(["(rows_c - 1) // 4096 + 1", 1, 1], scalars, 148) == (2, 1, 1)
    assert grid_dims([0, "rows_c - rows_c", "1 - 6"], scalars, 148) == (1, 1, 1)  # clamped to one
    with pytest.raises(ValueError):
        grid_dims(["-rows_c", 1, 1], scalars, 148)  # unary operators are not part of the grammar
    with pytest.raises(KeyError, match="unknown"):
        grid_dims(["not_a_scalar", 1, 1], scalars, 148)
    with pytest.raises(KeyError):
        grid_dims(["rows_c", 1, 1], {"rows_c": None}, 148)
    with pytest.raises(KeyError):
        grid_dims(["rows_c", 1, 1], {"rows_c": torch.tensor(4)}, 148)
    with pytest.raises(ValueError):
        grid_dims(["rows_c +", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["rows_c ** 2", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["rows_c / 0", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["min(rows_c)", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["abs(rows_c)", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["2.5", 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims([True, 1, 1], scalars, 148)
    with pytest.raises(ValueError):
        grid_dims(["rows_c", 1], scalars, 148)


def test_grid_dims_resident():
    scalars = {"m_tiles": 32}
    # a persistent four-CTA cluster grid is capped by the co-resident clusters (the occupancy answer, not sms // 4)
    assert grid_dims(["max(1, min(m_tiles//4*24, resident))*4", 1, 1], scalars, 148, resident=33) == (132, 1, 1)
    assert grid_dims(["max(1, min(m_tiles//4*24, resident))*4", 1, 1], {"m_tiles": 4}, 148, resident=33) == (96, 1, 1)
    assert grid_dims(["max(1, min(m_tiles//2*24, resident))*2", 1, 1], scalars, 148, resident=74) == (148, 1, 1)
    # a dynamically scheduled kernel launches its whole work-item domain
    assert grid_dims(["max(1, m_tiles//2*605)*2", 1, 1], scalars, 148) == (19360, 1, 1)
    with pytest.raises(KeyError, match="unknown"):  # only a clustered stage provides ``resident``
        grid_dims(["min(m_tiles, resident)", 1, 1], scalars, 148)


def test_geometry_cluster_ctas():
    g = Geometry.from_record({"geometry": {"logits_cluster_ctas": 2, "dx_cluster_ctas": 4, "dw_cluster_ctas": 2}})
    assert (g.logits_cluster_ctas, g.dx_cluster_ctas, g.dw_cluster_ctas) == (2, 4, 2)
    assert g.row_tiles(4097) == 34 and g.row_tiles(4097, 4) == 36 and g.row_tiles(1, 4) == 4 and g.row_tiles(4096, 4) == 32
    assert g.cluster_ctas_of("gemm_logits") == 2 and g.cluster_ctas_of("gemm_logits_nostats") == 2
    assert g.cluster_ctas_of("gemm_dx") == 4 and g.cluster_ctas_of("gemm_dx_s3") == 4 and g.cluster_ctas_of("gemm_dw_acc") == 2
    assert g.cluster_ctas_of("row_grad") is None and g.cluster_ctas_of("scale_cast_bf16") is None
    default = Geometry.from_record(None)
    assert (default.logits_cluster_ctas, default.dx_cluster_ctas, default.dw_cluster_ctas) == (2, 2, 2)
    with pytest.raises(ValueError):
        Geometry.from_record({"geometry": {"dx_cluster_ctas": 0}})


def test_cluster_resident_rule():
    assert cluster_resident(None, 2, 148) == 74 and cluster_resident(None, 2, 152) == 76 and cluster_resident(None, 1, 5) == 5
    with pytest.raises(ValueError):  # wider clusters take the device's occupancy answer
        cluster_resident(None, 4, 148)
    with pytest.raises(ValueError):
        cluster_resident(torch.device("cpu"), 4, 148)


def test_wave_efficiency_uses_the_dx_cluster():
    g4 = Geometry.from_record({"geometry": {"dx_cluster_ctas": 4}})
    g2 = Geometry.from_record(None)
    # 4096 rows: 32 row tiles = 8 four-CTA clusters x 24 column tiles = 192 items over 33 co-resident clusters
    assert wave_efficiency(4096, 6144, 148, 1, g4, resident=33) == pytest.approx(192 / (6 * 33))
    assert wave_efficiency(4096, 6144, 148, 1, g2) == pytest.approx((16 * 24) / (6 * 74))
    assert 1 <= recommended_k_slices(4096, 6144, 148, 4, g4, resident=33) <= 4
    assert recommended_k_slices(4096, 6144, 148, 4, g2) == recommended_k_slices(4096, 6144, 148, 4, g2, resident=74)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="the cluster occupancy probe needs a CUDA device")
def test_device_cluster_resident_probe():
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip("thread-block clusters of four CTAs are probed on the supported parts only")
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    four = cluster_resident(device, 4, sms)
    assert 1 <= four <= sms // 4
    assert cluster_resident(device, 4, sms) == four  # cached per (device, width)
    assert cluster_resident(device, 2, sms) == sms // 2


def test_stage_values_names():
    T, C = 37, 16
    inp = _host_inputs(T)
    runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=C, backend="reference")
    plan, t = runner.plan, runner.tensors
    assert plan.chunks == ((0, 16), (16, 16), (32, 5)) and plan.chunks[-1][1] == T % C
    assert runner.stages == stages_for_entry("loss")
    assert runner.forward_order[:2] == (("gemm_logits", 0), ("row_finalize", 0))
    assert runner.backward_order == (("scale_cast_bf16", "dx"), ("scale_cast_bf16", "dw"))
    for index, (row0, rows_c) in enumerate(plan.chunks):
        v = stage_values("gemm_logits", t, plan, index)
        assert (v["rows_c"], v["row0"], v["T"], v["H"], v["V"], v["chunk"]) == (rows_c, row0, T, H_HOST, V_HOST, C)
        assert (v["first_chunk"], v["last_chunk"]) == (0, int(index == len(plan.chunks) - 1))
        assert v["mode"] == MODE_CE and v["loss_div"] == inp.loss_div and v["num_tiles"] == V_HOST // 256
        assert v["A"].data_ptr() == inp.X[row0].data_ptr() and tuple(v["A"].shape) == (rows_c, H_HOST)
        assert v["B"] is inp.W and v["STATS_OUT"] is t["stats"]
        assert v["C"].data_ptr() == t["logits"].data_ptr() and tuple(v["C"].shape) == (rows_c, V_HOST)  # the chunk's rows
        assert v["M"] == rows_c and v["m_tiles"] % 2 == 0 and v["m_tiles"] >= -(-rows_c // 128) and v["k_iters"] == 1
        dx = stage_values("gemm_dx", t, plan, index)
        assert dx["A"].data_ptr() == t["logits"].data_ptr() and tuple(dx["A"].shape) == (rows_c, V_HOST)
        assert dx["C"].data_ptr() == t["dx_acc"][row0].data_ptr() and dx["M"] == rows_c and dx["first_chunk"] == 1  # stores its rows
        dw = stage_values("gemm_dw_acc", t, plan, index)
        assert dw["M"] == V_HOST and dw["k_iters"] == -(-rows_c // 64) and dw["C"] is t["dw_acc"]
        assert dw["first_chunk"] == int(index == 0)
        assert dw["A"].data_ptr() == t["logits"].data_ptr() and dw["B"].data_ptr() == inp.X[row0].data_ptr()
        rg = stage_values("row_grad", t, plan, index)
        assert rg["d_off"] == 0 and rg["d"] is t["d"] and rg["z"] is t["logits"] and rg["lse"] is t["lse"]
        rf = stage_values("row_finalize", t, plan, index)
        assert rf["infer_logp"] is t["d"] and rf["loss_weights"] is t["d"] and rf["d_in"] is t["d"]  # CE reads none of them
        assert rf["logp"] is t["logp"] and rf["term"] is t["term"]
        lr = stage_values("loss_reduce", t, plan, index)
        assert lr["loss_acc"] is t["loss_acc"] and lr["loss_out"] is t["loss"]
    cast = stage_values("scale_cast_bf16", t, plan, "dw")
    assert cast["acc"].data_ptr() == t["dw_acc"].data_ptr() and cast["out"].data_ptr() == t["dw_out"].data_ptr()
    assert cast["num_vecs"] == V_HOST * H_HOST // 8 and cast["g"] is t["grad_scale"]
    with pytest.raises(ValueError):
        stage_values("not_a_stage", t, plan, 0)
    # the log-probability entry reads the caller's [T] dlogp at d[row0 + r]
    lp = prepare_lm_head_loss(inp.X, inp.W, inp.labels, chunk_size=C, entry="logprob", backend="reference")
    assert lp.stages == stages_for_entry("logprob") and lp.dlogp is lp.tensors["d_in"]
    assert lp.forward_order == tuple(k for i in range(3) for k in (("gemm_logits", i), ("row_finalize", i)))
    assert lp.backward_order[0] == ("gemm_logits_nostats", 0) and lp.backward_order[-1] == ("scale_cast_bf16", "dw")
    for index, (row0, _) in enumerate(lp.plan.chunks):
        rg = stage_values("row_grad", lp.tensors, lp.plan, index)
        assert rg["d_off"] == row0 and rg["d"] is lp.tensors["d_in"]
        assert stage_values("row_finalize", lp.tensors, lp.plan, index)["mode"] == MODE_NONE
    with pytest.raises(ValueError, match="T > 0"):
        prepare_lm_head_loss(inp.X[:0], inp.W, inp.labels[:0], objective="ce", loss_div=1.0, backend="reference")


_ROW_GRAD_PLAN = [
    ["buffer", "z"], ["buffer", "labels"], ["buffer", "lse"], ["buffer", "d"], ["parameter", "row0"],
    ["parameter", "d_off"], ["parameter", "V"], ["grid", "grid_x"], ["grid", "grid_y"], ["grid", "grid_z"],
]


def _fake_record(arg_plan):
    return {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["row_grad"],
        "row_grad": {
            "module": "fake", "sources": ["cake_lm_head_loss/fake/a.cu", "cake_lm_head_loss/fake/b.cu"], "compile_flags": [],
            "ffi_entry": "run", "arg_plan": arg_plan, "closure_sha256": "0" * 64, "tma_workspace_bytes": 0, "workspace_bytes": 0,
            "grid": ["rows_c", 1, 1], "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "0" * 64,
    }


def _row_grad_values():
    return dict(
        z=torch.zeros(4, V_HOST, dtype=torch.bfloat16), labels=torch.zeros(4, dtype=torch.int64), lse=torch.zeros(4),
        d=torch.zeros(4), row0=0, d_off=0, V=V_HOST, rows_c=4, T=4, workspace=None, tma_descriptor_workspace=None,
    )


def test_bind_stage_fails_closed_on_unknown_argument(monkeypatch):
    name = "cake_lm_head_loss_fake"
    monkeypatch.setitem(cake_jit.MODULES, name, _fake_record(_ROW_GRAD_PLAN + [["parameter", "not_a_host_value"]]))
    monkeypatch.setattr(cake_backend, "load_cake_lm_head_loss_module", lambda n, s: SimpleNamespace(run=lambda *a: None))
    with pytest.raises(KeyError, match="not_a_host_value"):
        bind_stage(name, "row_grad", _row_grad_values(), (4, 1, 1))
    # a host value the call leaves at None is missing too
    values = _row_grad_values()
    values["lse"] = None
    with pytest.raises(KeyError, match="lse"):
        bind_stage(name, "row_grad", values, (4, 1, 1))


def test_bind_stage_records_tensor_slots(monkeypatch):
    name = "cake_lm_head_loss_fake"
    monkeypatch.setitem(cake_jit.MODULES, name, _fake_record(_ROW_GRAD_PLAN))
    calls = []
    monkeypatch.setattr(cake_backend, "load_cake_lm_head_loss_module", lambda n, s: SimpleNamespace(run=lambda *a: calls.append(a)))
    values = _row_grad_values()
    launch = bind_stage(name, "row_grad", values, (4, 3, 1))
    assert launch.slots == ((0, "z"), (1, "labels"), (2, "lse"), (3, "d"))
    assert launch.arguments[4:] == (0, 0, V_HOST, 4, 3, 1) and launch.grid == (4, 3, 1)
    assert all(launch.arguments[i] is values[k] for i, k in launch.slots)
    launch()
    assert len(calls) == 1 and calls[0][0] is values["z"] and calls[0][7:] == (4, 3, 1)
    template = launch.templated()
    assert [template.arguments[i] for i, _ in launch.slots] == [None] * 4
    assert not any(isinstance(a, torch.Tensor) for a in template.arguments)
    fresh = dict(z=torch.ones(4, V_HOST, dtype=torch.bfloat16), labels=values["labels"], lse=torch.ones(4), d=values["d"])
    rebound = template.arguments_for(fresh)
    assert rebound[0] is fresh["z"] and rebound[2] is fresh["lse"] and rebound[4:] == [0, 0, V_HOST, 4, 3, 1]
    with pytest.raises(RuntimeError, match="'d'"):
        template.arguments_for({k: v for k, v in fresh.items() if k != "d"})
    # kernel-side spellings resolve through the contract aliases to the same host slots
    monkeypatch.setitem(cake_jit.MODULES, name, _fake_record([["buffer", "dlogits"], ["buffer", "targets"], ["parameter", "vocab"], ["grid", "grid_x"]]))
    aliased = bind_stage(name, "row_grad", values, (4, 1, 1))
    assert aliased.slots == ((0, "z"), (1, "labels")) and aliased.arguments[2:] == (V_HOST, 4)


def test_binding_keys_cover_pointer_shape_stride_dtype_and_options():
    kw = _valid_call(T=8)
    X, W, labels = kw["X"], kw["W"], kw["labels"]
    common = dict(objective="ce", loss_div=1.0, infer_logp=None, loss_weights=None, chunk_size=4096, need_dx=True, need_dw=True,
                  grad_weight_dtype=torch.bfloat16, entry="loss")
    base = forward_binding_key(X, W, labels, **common)
    assert base[0] == "fwd"
    assert base == forward_binding_key(X.view_as(X), W, labels, **common)  # another object, same binding
    assert base != forward_binding_key(X.clone(), W, labels, **common)  # pointer
    assert base != forward_binding_key(X[:4], W, labels, **common)  # shape (same pointer, stride, dtype)
    same_ptr_other_stride = X.as_strided((4, H_HOST), (2 * H_HOST, 1))
    assert same_ptr_other_stride.data_ptr() == X[:4].data_ptr() and same_ptr_other_stride.shape == X[:4].shape
    assert forward_binding_key(X[:4], W, labels, **common) != forward_binding_key(same_ptr_other_stride, W, labels, **common)  # stride
    assert base != forward_binding_key(X.view(torch.float16), W, labels, **common)  # dtype
    assert base != forward_binding_key(X, W.clone(), labels, **common)
    assert base != forward_binding_key(X, W, labels.clone(), **common)
    for option in (dict(loss_div=2.0), dict(chunk_size=2048), dict(need_dx=False), dict(need_dw=False),
                   dict(grad_weight_dtype=torch.float32), dict(entry="logprob")):
        assert base != forward_binding_key(X, W, labels, **dict(common, **option)), option
    infer, weights = torch.zeros(8), torch.zeros(8)
    policy = dict(common, objective="policy", loss_div=None, infer_logp=infer, loss_weights=weights)
    pkey = forward_binding_key(X, W, labels, **policy)
    assert pkey != base
    assert pkey != forward_binding_key(X, W, labels, **dict(policy, infer_logp=infer.clone()))
    assert pkey != forward_binding_key(X, W, labels, **dict(policy, loss_weights=weights.clone()))
    lse, dlogp = torch.zeros(8), torch.zeros(8)
    bwd = logprob_backward_binding_key(X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=True)
    assert bwd[:2] == ("bwd", "logprob") and bwd != base
    assert bwd != logprob_backward_binding_key(X, W, labels, lse, dlogp.clone(), chunk_size=4096, need_dx=True, need_dw=True)
    assert bwd != logprob_backward_binding_key(X, W, labels, lse.clone(), dlogp, chunk_size=4096, need_dx=True, need_dw=True)
    assert bwd != logprob_backward_binding_key(X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=False)
    assert bwd != logprob_backward_binding_key(X, W, labels, lse, dlogp, chunk_size=2048, need_dx=True, need_dw=True)
    # the compacted row count is a label-dependent fact of the plan
    assert base != forward_binding_key(X, W, labels, valid_rows=4, **common)
    assert forward_binding_key(X, W, labels, valid_rows=4, **common) != forward_binding_key(X, W, labels, valid_rows=5, **common)
    assert bwd != logprob_backward_binding_key(X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=True, valid_rows=4)


def test_binding_cache_is_lru_and_bounded():
    cache = BindingCache(capacity=3)
    assert cache.enabled and len(cache) == 0

    def keys():
        return list(cache._bindings)

    for tag in ("a", "b", "c"):
        cache.remember(("fwd", tag), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "a"), ("fwd", "b"), ("fwd", "c")] and cache.owned_bytes == 30
    # a hit refreshes its binding; a miss changes nothing
    assert cache.lookup(("fwd", "a")) is not None
    assert keys() == [("fwd", "b"), ("fwd", "c"), ("fwd", "a")]
    assert cache.lookup(("fwd", "z")) is None and (cache.hits, cache.misses) == (1, 1)
    # beyond the capacity the least recently used binding goes (b, not the refreshed a)
    cache.remember(("bwd", "d"), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "c"), ("fwd", "a"), ("bwd", "d")]
    # re-remembering a key makes it the most recently used without duplicating it
    cache.remember(("fwd", "c"), SimpleNamespace(owned_bytes=10))
    assert keys() == [("fwd", "a"), ("bwd", "d"), ("fwd", "c")] and len(cache) == 3
    # peek neither counts nor refreshes
    assert cache.peek(("fwd", "a")) is not None and keys()[0] == ("fwd", "a")
    assert (cache.hits, cache.misses) == (1, 1)
    cache.clear()
    assert len(cache) == 0 and cache.owned_bytes == 0
    # a model cycling through N layer bindings per step binds each once in an N-binding cache
    layer_keys = [(kind, layer) for layer in range(4) for kind in ("fwd", "bwd")]

    def run_steps(capacity, steps=3):
        c = BindingCache(capacity=capacity)
        for _ in range(steps):
            for key in layer_keys:
                if c.lookup(key) is None:
                    c.remember(key, SimpleNamespace(owned_bytes=1))
        return c.hits, c.misses, len(c)

    assert run_steps(8) == (16, 8, 8)
    assert run_steps(64) == (16, 8, 8)
    assert run_steps(7) == (0, 24, 7)  # one binding short thrashes under a cyclic pattern
    with pytest.raises(ValueError, match="capacity"):
        BindingCache(capacity=0)
    assert not BindingCache(capacity=1, enabled=False).enabled


def test_binding_cache_capacity_from_environment(monkeypatch):
    env = cake_backend.BINDING_CACHE_CAPACITY_ENV
    monkeypatch.delenv(env, raising=False)
    assert BindingCache().capacity == cake_backend.BINDING_CACHE_DEFAULT_CAPACITY == 64
    assert cake_backend.BINDING_CACHE.capacity >= 1
    monkeypatch.setenv(env, "5")
    assert cake_backend.binding_cache_capacity() == 5 and BindingCache().capacity == 5
    assert BindingCache(capacity=3).capacity == 3  # an explicit capacity wins
    monkeypatch.setenv(env, " ")
    assert BindingCache().capacity == 64
    for bad in ("0", "-1", "many", "2.5"):
        monkeypatch.setenv(env, bad)
        with pytest.raises(ValueError, match="positive integer"):
            BindingCache()


HOST_SHAPES = [(1, 16), (5, 4), (37, 16), (64, 64), (65, 64), (200, 64)]


@pytest.mark.parametrize("T, C", HOST_SHAPES, ids=[f"t{t}_c{c}" for t, c in HOST_SHAPES])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_reference_matches_unchunked_and_oracle(objective, T, C):
    inp = _host_inputs(T, objective)
    result = _run(inp, C)
    _check_dtypes(result, inp)
    _check_against_references(result, inp)
    assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(result["dX"][~inp.valid] == 0)
    assert torch.isfinite(result["dW"].float()).all()


def test_reference_policy_ratio_boundary():
    inp = _host_inputs(200, "policy")
    regime = inp.regime
    assert all(bool((regime == k).any()) for k in (0, 2, 3)) and not bool((regime == 1).any())
    assert torch.equal(regime == -1, ~inp.valid)
    # every valid row keeps its margin from the clip knee (in the oracle's FP32 logp)
    logp32 = reference_fp64(inp, need_dx=False, need_dw=False)["logp"].float()
    delta = logp32 - inp.infer_logp
    assert (delta[inp.valid] - math.log(2.0)).abs().min().item() >= KNEE_MARGIN - 1e-6
    assert torch.allclose(delta[regime == 0], torch.full_like(delta[regime == 0], -1.0), atol=1e-6)
    assert torch.allclose(delta[regime == 2], torch.full_like(delta[regime == 2], 1.0), atol=1e-6)
    assert torch.all(inp.infer_logp[~inp.valid] == 0) and torch.all(inp.loss_weights[~inp.valid] == 0)
    result = _run(inp, 64)
    oracle = reference_fp64(inp)
    b0 = reference_unchunked(inp)
    dX = result["dX"]
    # above the clip the logit gradient vanishes analytically: those rows are exactly zero
    assert torch.all(dX[regime == 2] == 0) and torch.all(oracle["dX"][regime == 2] == 0)
    # below the clip (and on the random rows) every row carries a gradient within the unchunked path's error
    # (random rows above the knee, ratio > 2, legitimately carry no gradient; keep the ones below it)
    for rows in (regime == 0, (regime == 3) & (delta < math.log(2.0))):
        assert torch.all(dX[rows].float().abs().sum(-1) > 0)
        assert rel_l2(dX[rows], oracle["dX"][rows]) <= GATE_MARGIN * rel_l2(b0["dX"][rows], oracle["dX"][rows]) + GATE_TINY["dX_rel_l2"]
    _check_against_references(result, inp)


def test_reference_policy_clamp_knee():
    """Rows at the clip knee: ``ratio <= 2`` keeps ``d_t = -w_t * ratio_t`` (gradient rows present),
    ``ratio > 2`` zeroes the row and the loss term saturates at ``2 w_t``.  The knee is placed
    around the implementation's own FP32 ``logp`` with ``1e-3`` margins: ``exp(logp - (logp -
    ln 2))`` lands within a few FP32 ulp of 2 on either side, so exact equality is not a testable
    point in FP32 arithmetic."""
    T, C = 64, 16
    inp = _host_inputs(T, "policy")
    valid = inp.valid
    logp_impl = _run(inp, C, train_x=False, train_w=False)["logp"]  # the backend's FP32 logp (independent of infer_logp)
    assert (logp_impl - reference_fp64(inp, need_dx=False, need_dw=False)["logp"].float())[valid].abs().max().item() <= 1e-2
    ln2 = torch.tensor(math.log(2.0), dtype=torch.float32, device=DEV)
    rows = torch.nonzero(valid).squeeze(1)
    under, over = rows[0::2], rows[1::2]  # alternate valid rows: ratio 2 e^-1e-3 (kept) / 2 e^+1e-3 (clipped)
    assert len(under) >= 8 and len(over) >= 8
    infer = inp.infer_logp.clone()
    infer[under] = logp_impl[under] - ln2 + 1e-3
    infer[over] = logp_impl[over] - ln2 - 1e-3
    knee = replace(inp, infer_logp=infer.contiguous(), regime=None)
    result = _run(knee, C)
    dX = result["dX"]
    assert torch.all(dX[over] == 0)
    assert torch.all(dX[under].float().abs().sum(-1) > 0)
    assert torch.equal(result["logp"], logp_impl)
    # the backend's own quantities: ratio below / above the knee, loss = -sum w_t min(ratio_t, 2)
    ratio = torch.exp(logp_impl - infer)
    assert torch.all(ratio[under] < 2) and torch.all(ratio[over] > 2)
    expected = -(inp.loss_weights * torch.clamp_max(ratio, 2.0))[valid].sum().item()
    assert abs(result["loss"].item() - expected) <= 1e-5 * abs(expected) + 1e-6
    # kept rows carry the unclipped analytic gradient -w * ratio * (onehot - p) @ W of the same BF16 inputs
    X64, W64 = inp.X[under].double(), inp.W.double()
    z = X64 @ W64.t()
    lse = torch.logsumexp(z, -1)
    p = torch.exp(z - lse[:, None])
    y = inp.labels[under]
    ratio64 = torch.exp(z.gather(1, y[:, None]).squeeze(1) - lse - infer[under].double())
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    unclipped = ((-inp.loss_weights[under].double() * ratio64)[:, None] * (onehot - p)) @ W64
    assert rel_l2(dX[under], unclipped) <= ROW_REL_L2
    assert torch.all(result["dX"][~valid] == 0)
    # the loss is continuous through the knee: the usual gates hold with knee rows present
    _check_against_references(result, knee)


def test_logprob_entry_matches_unchunked():
    inp = _host_inputs(200)
    result = _run(inp, 64, entry="logprob")
    _check_dtypes(result, inp)
    _check_against_references(result, inp, entry="logprob")
    assert torch.all(result["dX"][~inp.valid] == 0)
    # the eager forward exposes the saved row statistic (compacted by default: the valid rows of row_index)
    T_v = int(inp.valid.sum())
    fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=64, backend="reference")
    assert fr.loss is None and torch.equal(fr.logp, result["logp"]) and tuple(fr.lse.shape) == (T_v,) and fr.num_rows == inp.T
    oracle = reference_fp64(inp, entry="logprob")
    assert (fr.lse.double() - oracle["lse"][inp.valid]).abs().max().item() <= 1e-2
    dx, dw = cake_backend.backward_logprob(inp.X, inp.W, inp.labels, fr.lse, inp.dlogp, chunk_size=64, backend="reference")
    assert torch.equal(dx, result["dX"]) and torch.equal(dw, result["dW"])
    plain = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=64, backend="reference", compact_rows=False)
    assert tuple(plain.lse.shape) == (inp.T,) and plain.row_index is None and torch.equal(plain.lse[inp.valid], fr.lse)
    with pytest.raises(ValueError, match="lse"):  # the uncompacted statistic does not fit a compacted backward
        cake_backend.backward_logprob(inp.X, inp.W, inp.labels, plain.lse, inp.dlogp, chunk_size=64, backend="reference", compact_rows=True)
    assert cake_backend.backward_logprob(inp.X, inp.W, inp.labels, fr.lse, inp.dlogp, chunk_size=64, need_dx=False, need_dw=False, backend="reference") == (None, None)


def test_grad_weight_dtype_fp32():
    inp = _host_inputs(65)
    bf16 = _run(inp, 64)
    # the autograd entry cannot return an FP32 dW for a BF16 leaf (the engine casts to the leaf's dtype)
    with pytest.raises(ValueError, match="W.dtype"):
        _run(inp, 64, grad_weight_dtype=torch.float32)
    with pytest.raises(ValueError, match="W.dtype"), _quiet_experimental():
        chunked_lm_head_loss(inp.X, inp.W, inp.labels, loss_div=inp.loss_div, grad_weight_dtype=torch.float32)
    # the explicit pair: FP32 accumulators, one cast to the requested dtype
    fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=64, need_dx=True, need_dw=True,
                                   grad_weight_dtype=torch.float32, backend="reference")
    dX, dW32 = cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, None, grad_weight_dtype=torch.float32, backend="reference",
                                          row_index=fr.row_index, num_rows=fr.num_rows)  # the compacted forward's rows
    assert dW32.dtype == torch.float32 and dX.dtype == torch.bfloat16 and tuple(dX.shape) == (inp.T, H_HOST)
    assert torch.equal(fr.loss, bf16["loss"]) and torch.equal(fr.logp, bf16["logp"]) and torch.equal(dX, bf16["dX"])
    assert torch.equal(dW32.to(torch.bfloat16), bf16["dW"])  # the same accumulator, cast once
    assert torch.equal(dW32, fr.dw_acc) and dW32.data_ptr() != fr.dw_acc.data_ptr()
    result = dict(loss=fr.loss, logp=fr.logp, dX=dX, dW=dW32)
    _check_dtypes(result, inp, grad_weight_dtype=torch.float32)
    _check_against_references(result, inp, grad_weight_dtype=torch.float32)
    oracle = reference_fp64(inp)
    assert rel_l2(dW32, oracle["dW"]) <= rel_l2(bf16["dW"], oracle["dW"])
    assert fr.memory["outputs"]["dW"] == 2 * inp.V * inp.H * 2
    assert stages_for_entry("loss", grad_weight_dtype=torch.float32)[-1] == "scale_cast_f32"


def test_frozen_inputs():
    inp = _host_inputs(37)
    both = _run(inp, 16)
    x_only = _run(inp, 16, train_w=False)
    w_only = _run(inp, 16, train_x=False)
    neither = _run(inp, 16, train_x=False, train_w=False)
    assert x_only["dW"] is None and w_only["dX"] is None and neither["dX"] is None and neither["dW"] is None
    assert torch.equal(x_only["dX"], both["dX"]) and torch.equal(w_only["dW"], both["dW"])
    for r in (x_only, w_only, neither):
        assert torch.equal(r["loss"], both["loss"]) and torch.equal(r["logp"], both["logp"])
    assert not neither["loss"].requires_grad


def test_retain_graph_repeated_backward():
    inp = _host_inputs(37, "policy")
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    loss = cake_backend.chunked_lm_head_loss(X, W, inp.labels, objective="policy", infer_logp=inp.infer_logp, loss_weights=inp.loss_weights,
                                             chunk_size=16, backend="reference")
    loss.backward(retain_graph=True)
    dx1, dw1 = X.grad.clone(), W.grad.clone()
    X.grad, W.grad = None, None
    loss.backward(retain_graph=True)
    assert torch.equal(X.grad, dx1) and torch.equal(W.grad, dw1)  # the saved accumulators are read, never written
    X.grad, W.grad = None, None
    (loss * 2.0).backward()
    assert torch.equal(X.grad, (2.0 * dx1.float()).to(torch.bfloat16)) and torch.equal(W.grad, (2.0 * dw1.float()).to(torch.bfloat16))


def test_upstream_scale_is_applied_once():
    inp = _host_inputs(65)
    scaled = _run(inp, 64, scale=3.0)
    fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=64, backend="reference")
    T_v = int(inp.valid.sum())  # the compacted loop: 62 of the 65 rows, one chunk
    assert fr.backend == "reference" and fr.loss.shape == () and fr.memory["num_chunks"] == -(-T_v // 64) == 1
    assert torch.equal(scaled["loss"], fr.loss)
    scatter = lambda rows: cake_backend.scatter_rows(rows, fr.row_index, fr.num_rows)
    assert torch.equal(scaled["dX"], scatter((3.0 * fr.dx_acc).to(torch.bfloat16)))
    assert torch.equal(scaled["dW"], (3.0 * fr.dw_acc).to(torch.bfloat16))
    dx, dw = cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, torch.tensor(3.0), backend="reference", row_index=fr.row_index, num_rows=fr.num_rows)
    assert torch.equal(dx, scaled["dX"]) and torch.equal(dw, scaled["dW"])
    dx32, dw32 = cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, torch.tensor(3.0), grad_weight_dtype=torch.float32, backend="reference",
                                            row_index=fr.row_index, num_rows=fr.num_rows)
    assert dw32.dtype == torch.float32 and torch.equal(dw32, 3.0 * fr.dw_acc) and torch.equal(dx32, dx)
    with pytest.raises(ValueError, match="num_rows"):
        cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, None, backend="reference", row_index=fr.row_index)
    # the log-probability entry scales through dlogp
    plain = _run(inp, 64, entry="logprob")
    twice = _run(inp, 64, entry="logprob", scale=2.0)
    assert torch.equal(plain["logp"], twice["logp"])
    assert rel_l2(twice["dX"], 2.0 * plain["dX"].float()) <= 1e-2 and rel_l2(twice["dW"], 2.0 * plain["dW"].float()) <= 1e-2


def test_return_logp_detached():
    inp = _host_inputs(37)
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference")
    loss, logp = cake_backend.chunked_lm_head_loss(X, W, inp.labels, return_logp=True, **kw)
    assert loss.requires_grad and loss.grad_fn is not None
    assert not logp.requires_grad and logp.grad_fn is None and logp.dtype == torch.float32
    only = cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)
    assert isinstance(only, torch.Tensor) and torch.equal(only, loss.detach())
    lp = cake_backend.chunked_lm_head_logprob(X, W, inp.labels, chunk_size=16, backend="reference")
    assert lp.requires_grad and torch.equal(lp.detach(), logp)


def test_zero_rows_return_zeros_without_binding(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a call without rows must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    monkeypatch.setattr(cake_backend, "load_cake_lm_head_loss_module", refuse)
    W = torch.zeros(V_HOST, H_HOST, dtype=torch.bfloat16, device=DEV)
    X0 = torch.zeros(0, H_HOST, dtype=torch.bfloat16, device=DEV)
    labels0 = torch.zeros(0, dtype=torch.int64, device=DEV)
    cache = cake_backend.BINDING_CACHE
    hits, misses = cache.hits, cache.misses
    for backend in ("reference", "cake"):
        # the eager forwards return zeros without consulting the registry (either backend)
        fr = cake_backend.forward_loss(X0, W, labels0, objective="ce", loss_div=1.0, backend=backend)
        assert fr.loss.shape == () and fr.loss.item() == 0.0 and tuple(fr.logp.shape) == (0,)
        assert tuple(fr.dx_acc.shape) == (0, H_HOST) and torch.all(fr.dw_acc == 0) and fr.memory["num_chunks"] == 0
        lr = cake_backend.forward_logprob(X0, W, labels0, backend=backend)
        assert lr.loss is None and tuple(lr.logp.shape) == (0,) and tuple(lr.lse.shape) == (0,)
        dx, dw = cake_backend.backward_logprob(X0, W, labels0, lr.lse, torch.zeros(0, device=DEV), backend=backend)
        assert tuple(dx.shape) == (0, H_HOST) and dx.dtype == torch.bfloat16 and torch.all(dw == 0) and dw.dtype == torch.bfloat16
    # the autograd path (reference backend: the empty dW cast needs no generated program)
    for objective in ("ce", "policy"):
        X = X0.clone().requires_grad_()
        Wt = W.clone().requires_grad_()
        kw = dict(objective=objective, loss_div=1.0) if objective == "ce" else dict(
            objective=objective, infer_logp=torch.zeros(0, device=DEV), loss_weights=torch.zeros(0, device=DEV))
        loss, logp = cake_backend.chunked_lm_head_loss(X, Wt, labels0, return_logp=True, backend="reference", **kw)
        assert loss.dtype == torch.float32 and loss.shape == () and loss.item() == 0.0 and tuple(logp.shape) == (0,)
        loss.backward()
        assert tuple(X.grad.shape) == (0, H_HOST) and X.grad.dtype == torch.bfloat16
        assert tuple(Wt.grad.shape) == (V_HOST, H_HOST) and Wt.grad.dtype == torch.bfloat16 and torch.all(Wt.grad == 0)
    X = X0.clone().requires_grad_()
    Wt = W.clone().requires_grad_()
    lp = cake_backend.chunked_lm_head_logprob(X, Wt, labels0, backend="reference")
    assert tuple(lp.shape) == (0,) and lp.dtype == torch.float32
    lp.sum().backward()
    assert tuple(X.grad.shape) == (0, H_HOST) and torch.all(Wt.grad == 0)
    # the shape / dtype checks still apply to an empty call
    with pytest.raises(ValueError, match="int64"):
        cake_backend.forward_loss(X0, W, labels0.int(), objective="ce", loss_div=1.0, backend="reference")
    with pytest.raises(ValueError, match="loss_div"):
        cake_backend.forward_loss(X0, W, labels0, objective="ce", backend="reference")
    assert (cache.hits, cache.misses) == (hits, misses)  # never consulted


@pytest.mark.parametrize("objective", ["ce", "policy"])
@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_all_ignored_rows(objective, entry):
    if entry == "logprob" and objective == "policy":
        pytest.skip("the log-probability entry has no objective")
    inp = _host_inputs(37, objective, ignore_frac=1.0)
    assert not inp.valid.any() and torch.all(inp.labels == IGNORE_INDEX) and torch.all(inp.dlogp == 0)
    result = _run(inp, 16, entry=entry)
    _check_dtypes(result, inp)
    assert result["loss"].item() == 0.0
    assert torch.all(result["logp"] == 0) and torch.all(result["dX"] == 0) and torch.all(result["dW"] == 0)
    assert not error_report(result, reference_fp64(inp, entry=entry), inp.labels)["nan"]


@pytest.mark.parametrize("ld_pad, copied", [(8, False), (3, True)])
def test_noncontiguous_x_strides(ld_pad, copied):
    inp = _host_inputs(37, ld_pad=ld_pad)
    assert inp.X.stride() == (H_HOST + ld_pad, 1) and not inp.X.is_contiguous()
    problem = validate_lm_head_inputs(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div)
    assert problem.x_copy is copied and problem.ld_x == (H_HOST if copied else H_HOST + ld_pad)
    strided = _run(inp, 16)
    contiguous = _run(replace(inp, X=inp.X.contiguous()), 16)
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(strided[key], contiguous[key]), key
    assert tuple(strided["dX"].shape) == (37, H_HOST)
    _check_against_references(strided, inp)
    fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference", compact_rows=False)
    assert ("x_copy" in fr.memory["temporary"]) is copied
    # the compacted loop gathers the chunk's rows into a contiguous buffer: no copy of X whatever its stride
    compact = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference", compact_rows=True)
    assert compact.memory["compact_rows"] and "x_copy" not in compact.memory["temporary"] and torch.equal(compact.logp, strided["logp"])


def test_reference_deterministic_three_runs():
    inp = _host_inputs(200, "policy")
    runs = [_run(inp, 64) for _ in range(3)]
    for r in runs[1:]:
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(r[key], runs[0][key]), key


def test_chunk_size_changes_only_rounding():
    inp = _host_inputs(200)
    small = _run(inp, 16)
    large = _run(inp, 64)
    single = _run(inp, 4096)  # one chunk
    for r in (small, large, single):
        _check_against_references(r, inp)
    for r in (small, single):
        assert abs(r["loss"].item() - large["loss"].item()) <= 1e-5 * abs(large["loss"].item())
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference")
    T_v = int(inp.valid.sum())  # the default compacted loop chunks the valid rows (190 of 200): 12 chunks, not 13
    assert cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw).memory["num_chunks"] == -(-T_v // 16) == 12
    assert cake_backend.forward_loss(inp.X, inp.W, inp.labels, compact_rows=False, **kw).memory["num_chunks"] == 13
    # the row statistics do not depend on the chunking at all
    assert torch.equal(small["logp"], large["logp"]) or (small["logp"] - large["logp"]).abs().max().item() <= 1e-5


def test_t_changes_between_calls():
    W = _host_inputs(1).W
    inputs = [_host_inputs(T, W=W) for T in (37, 64, 37)]
    assert all(inp.W is W for inp in inputs)
    fresh = [_run(inp, 16) for inp in inputs]  # each shape computed on its own
    again = [_run(inp, 16) for inp in inputs]  # the sequence 37 -> 64 -> 37 in one process
    for a, b in zip(fresh, again, strict=True):
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(a[key], b[key]), key
    assert torch.equal(fresh[0]["loss"], fresh[2]["loss"]) and not torch.equal(fresh[0]["loss"], fresh[1]["loss"])


# ---------------------------------------------------------------------------
# Valid-row compaction (the chunk loop over the rows with labels >= 0 only)
# ---------------------------------------------------------------------------


def test_valid_row_index_and_scatter():
    labels = torch.tensor([3, IGNORE_INDEX, 5, 7, IGNORE_INDEX], device=DEV)
    idx = cake_backend.valid_row_index(labels)
    assert idx.dtype == torch.int64 and idx.tolist() == [0, 2, 3]
    assert cake_backend.valid_row_index(torch.tensor([1, 2], device=DEV)) is None  # every row valid: the uncompacted path
    assert cake_backend.valid_row_index(torch.zeros(0, dtype=torch.int64, device=DEV)) is None
    assert cake_backend.valid_row_index(torch.full((4,), IGNORE_INDEX, device=DEV)).numel() == 0
    rows = torch.arange(6, dtype=torch.float32, device=DEV).view(3, 2)
    out = cake_backend.scatter_rows(rows, idx, 5)
    assert tuple(out.shape) == (5, 2) and torch.equal(out[idx], rows) and torch.all(out[[1, 4]] == 0)
    assert cake_backend.scatter_rows(rows, None, 5) is rows
    empty = cake_backend.scatter_rows(rows[:0], idx[:0], 5)
    assert tuple(empty.shape) == (5, 2) and torch.all(empty == 0)


def test_compact_rows_default_env(monkeypatch):
    monkeypatch.delenv(cake_backend.COMPACT_ROWS_ENV, raising=False)
    assert cake_backend.compact_rows_default() is True
    monkeypatch.setenv(cake_backend.COMPACT_ROWS_ENV, "0")
    assert cake_backend.compact_rows_default() is False
    monkeypatch.setenv(cake_backend.COMPACT_ROWS_ENV, "1")
    assert cake_backend.compact_rows_default() is True


def test_memory_report_compaction():
    V, H, C, T, T_v = DEFAULT_V, DEFAULT_H, 4096, 4097, 3892
    align = lambda n: (n + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN
    plain = memory_report(T, H, V, C)
    assert not plain["compact_rows"] and plain["valid_rows"] == T and plain["gather_bytes"] == 0 and "x_c" not in plain["temporary"]
    m = memory_report(T, H, V, C, valid_rows=T_v)
    assert m["compact_rows"] and m["valid_rows"] == T_v and m["num_chunks"] == 1 and m["vocab_rows_max"] == T_v
    assert m["gather_bytes"] == m["temporary"]["x_c"] == T_v * H * 2
    assert m["temporary"]["logits"] == T_v * V * 2 < plain["temporary"]["logits"]
    assert m["temporary"]["row_index"] == T_v * 8 and m["temporary"]["dx_compact"] == T_v * H * 2
    assert m["temporary"]["lse"] == T_v * 4 and m["temporary"]["logp"] == T_v * 4  # the compact rows before the scatter
    assert m["accumulators"]["dX_acc"] == T_v * H * 4 and m["outputs"]["dX"] == T * H * 2  # accumulate compact, return [T, H]
    assert m["outputs"] == plain["outputs"] and m["weights"] == plain["weights"]
    assert "x_copy" not in memory_report(T, H, V, C, x_copy=True, valid_rows=T_v)["temporary"]  # the gather output is contiguous
    lp = memory_report(T, H, V, C, entry="logprob", valid_rows=T_v)
    assert lp["temporary"]["dlogp"] == T * 4 and lp["temporary"]["dlogp_compact"] == T_v * 4 and lp["accumulators"]["saved_lse"] == T_v * 4
    assert memory_report(T, H, V, C, valid_rows=0)["num_chunks"] == 0
    layout = workspace_layout(T_v, V, C, hidden=H, compact=True)
    assert layout["x_c"][1] == T_v * H * 2 and layout["total"] == workspace_layout(T_v, V, C)["total"] + align(T_v * H * 2)
    with pytest.raises(ValueError, match="hidden"):
        workspace_layout(T_v, V, C, compact=True)
    assert lm_head_loss_workspace_size(T, V, C, backend="reference", hidden=H, compact_rows=True) == workspace_layout(T, V, C, hidden=H, compact=True)["total"]
    with pytest.raises(ValueError, match="hidden"):
        lm_head_loss_workspace_size(T, V, C, backend="reference", compact_rows=True)


COMPACTION_CASES = [("ce", "loss"), ("policy", "loss"), ("ce", "logprob")]


@pytest.mark.parametrize("frac", ["none", "five_percent", "half", "all_but_one"])
@pytest.mark.parametrize("objective, entry", COMPACTION_CASES, ids=["ce", "policy", "logprob"])
def test_compaction_matches_uncompacted(objective, entry, frac):
    T, C = 37, 16
    ignore = {"none": 0.0, "five_percent": 0.05, "half": 0.5, "all_but_one": (T - 1) / T}[frac]
    inp = _host_inputs(T, objective, ignore_frac=ignore)
    num_ignored = int((~inp.valid).sum())
    assert num_ignored == (T - 1 if frac == "all_but_one" else round(ignore * T))
    T_v = T - num_ignored
    compact = _run(inp, C, entry=entry, compact_rows=True)
    plain = _run(inp, C, entry=entry, compact_rows=False)
    for result in (compact, plain):
        _check_dtypes(result, inp)
        _check_against_references(result, inp, entry=entry)
        assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(result["dX"][~inp.valid] == 0)
    if num_ignored == 0:  # every row valid: the very same (uncompacted) path
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(compact[key], plain[key]), key
    else:
        # the same per-row work over fewer rows; the reference GEMMs may round differently for another M, and
        # the loss / dW reductions run over different chunk boundaries -- the same precision level, not bitwise
        assert torch.allclose(compact["logp"], plain["logp"], rtol=1e-5, atol=2e-5)
        assert rel_l2(compact["dX"].float(), plain["dX"].float()) <= 1e-3
        assert rel_l2(compact["dW"].float(), plain["dW"].float()) <= 1e-3
        assert torch.allclose(compact["loss"], plain["loss"], rtol=1e-5, atol=1e-6)
    # the plan of the compacted forward covers the valid rows only; the outputs keep the caller's [T]
    kw = dict(objective=objective, loss_div=inp.loss_div if objective == "ce" else None, infer_logp=inp.infer_logp,
              loss_weights=inp.loss_weights)
    if entry == "loss":
        fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, chunk_size=C, backend="reference", compact_rows=True, **kw)
        assert tuple(fr.dx_acc.shape) == (T_v, H_HOST)
    else:
        fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=C, backend="reference", compact_rows=True)
        assert tuple(fr.lse.shape) == (T_v,)  # the saved statistic stays compact
    assert tuple(fr.logp.shape) == (T,) and fr.num_rows == T and torch.equal(fr.logp, compact["logp"])
    assert (fr.row_index is None) == (num_ignored == 0)
    assert fr.memory["compact_rows"] == (num_ignored > 0) and fr.memory["valid_rows"] == T_v
    assert fr.memory["num_chunks"] == -(-T_v // C) and fr.memory["vocab_rows_max"] == min(T_v, C)
    if num_ignored:
        assert torch.equal(fr.row_index, inp.valid.nonzero().squeeze(1))
        assert fr.memory["gather_bytes"] == min(T_v, C) * H_HOST * 2
    plain_fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, chunk_size=C, backend="reference", compact_rows=False, **kw) if entry == "loss" else None
    if plain_fr is not None:
        assert plain_fr.row_index is None and not plain_fr.memory["compact_rows"] and plain_fr.memory["num_chunks"] == -(-T // C)


def test_compacted_runner_binds_the_valid_rows():
    T, C = 37, 16
    inp = _host_inputs(T, "policy", ignore_frac=0.5)
    idx = cake_backend.valid_row_index(inp.labels)
    T_v = int(idx.numel())
    runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, objective="policy", infer_logp=inp.infer_logp, loss_weights=inp.loss_weights,
                                  chunk_size=C, backend="reference", compact_rows=True)
    plan, t = runner.plan, runner.tensors
    assert plan.compact and plan.rows == T_v and runner.valid_rows == T_v and torch.equal(runner.row_index, idx)
    assert plan.chunks == plan_chunks(T_v, C) and runner.problem.num_rows == T
    assert runner.stages == stages_for_entry("loss")  # the host gathers are not stages of the program
    # the row operands are gathered once per step, the chunk's X rows before each chunk
    assert runner.forward_order[:3] == (("gather_rows", "infer_logp"), ("gather_rows", "loss_weights"), ("gather_rows", 0))
    assert runner.forward_order[3:5] == (("gemm_logits", 0), ("row_finalize", 0))
    assert [k for k in runner.forward_order if k[0] == "gather_rows" and isinstance(k[1], int)] == [("gather_rows", i) for i in range(plan.num_chunks)]
    assert tuple(t["labels"].shape) == (T_v,) and torch.equal(t["labels"].long(), inp.labels[idx])
    assert tuple(t["x_c"].shape) == (min(T_v, C), H_HOST) and t["x_c"].dtype == torch.bfloat16
    assert tuple(t["lse"].shape) == tuple(t["logp"].shape) == (T_v,) and tuple(t["dx_acc"].shape) == tuple(t["dx_out"].shape) == (T_v, H_HOST)
    assert t["infer_logp_full"] is inp.infer_logp and tuple(t["infer_logp"].shape) == (T_v,)
    for index, (row0, rows_c) in enumerate(plan.chunks):
        g = stage_values("gather_rows", t, plan, index)
        assert g["src"] is t["X"] and tuple(g["out"].shape) == (rows_c, H_HOST) and g["out"].data_ptr() == t["x_c"].data_ptr()
        assert torch.equal(g["idx"], idx[row0:row0 + rows_c])
        v = stage_values("gemm_logits", t, plan, index)
        assert v["T"] == T_v and v["A"].data_ptr() == t["x_c"].data_ptr() and tuple(v["A"].shape) == (rows_c, H_HOST)
        assert stage_values("gemm_dw_acc", t, plan, index)["B"].data_ptr() == t["x_c"].data_ptr()
    op = stage_values("gather_rows", t, plan, "infer_logp")
    assert op["src"] is inp.infer_logp and op["out"] is t["infer_logp"] and torch.equal(op["idx"], idx)
    runner.step()
    assert torch.equal(t["infer_logp"], inp.infer_logp[idx]) and torch.equal(t["x_c"][: plan.chunks[-1][1]], inp.X[idx[plan.chunks[-1][0]:]])
    autograd = _run(inp, C, compact_rows=True)
    assert torch.equal(runner.scatter(runner.logp), autograd["logp"]) and torch.equal(runner.scatter(runner.dx_out), autograd["dX"])
    assert torch.equal(runner.loss.reshape(()), autograd["loss"]) and torch.equal(runner.dw_out, autograd["dW"])
    assert torch.all(runner.scatter(runner.logp)[~inp.valid] == 0)
    with pytest.raises(ValueError, match="gather_rows"):
        stage_values("gather_rows", t, cake_backend.make_plan(runner.problem, need_dx=True, need_dw=True), 0)
    # the log-probability runner gathers the caller's [T] dlogp at the start of its backward
    lp = prepare_lm_head_loss(inp.X, inp.W, inp.labels, chunk_size=C, entry="logprob", dlogp=inp.dlogp, backend="reference", compact_rows=idx)
    assert lp.backward_order[:2] == (("gather_rows", "d_in"), ("gather_rows", 0)) and lp.dlogp is inp.dlogp
    assert tuple(lp.tensors["d_in"].shape) == (T_v,) and tuple(lp.lse.shape) == (T_v,)
    lp.forward()
    dx, dw = lp.backward()
    assert torch.equal(lp.tensors["d_in"], inp.dlogp[idx]) and tuple(dx.shape) == (T_v, H_HOST)
    ref = _run(inp, C, entry="logprob", compact_rows=True)
    assert torch.equal(lp.scatter(dx), ref["dX"]) and torch.equal(dw, ref["dW"]) and torch.equal(lp.scatter(lp.logp), ref["logp"])
    # a runner cannot be prepared over no valid rows; a mismatching index is rejected
    with pytest.raises(ValueError, match="valid row"):
        prepare_lm_head_loss(inp.X, inp.W, torch.full_like(inp.labels, IGNORE_INDEX), objective="ce", loss_div=1.0, chunk_size=C,
                             backend="reference", compact_rows=True)
    with pytest.raises(ValueError, match="int64"):
        prepare_lm_head_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=1.0, chunk_size=C, backend="reference", compact_rows=idx.int())


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_all_ignored_rows_compacted_return_zeros_without_binding(entry, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("a call whose every row is ignored must neither bind nor launch")

    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    inp = _host_inputs(37, ignore_frac=1.0)
    cache = cake_backend.BINDING_CACHE
    hits, misses = cache.hits, cache.misses
    if entry == "loss":
        fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, backend="reference", compact_rows=True)
        assert fr.loss.item() == 0.0 and tuple(fr.logp.shape) == (37,) and torch.all(fr.logp == 0)
        assert tuple(fr.dx_acc.shape) == (0, H_HOST) and torch.all(fr.dw_acc == 0) and fr.row_index.numel() == 0
        assert fr.memory["compact_rows"] and fr.memory["valid_rows"] == 0 and fr.memory["num_chunks"] == 0
        dx, dw = cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, None, backend="reference", row_index=fr.row_index, num_rows=fr.num_rows)
        assert tuple(dx.shape) == (37, H_HOST) and torch.all(dx == 0) and torch.all(dw == 0)
    else:
        fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, backend="reference", compact_rows=True)
        assert tuple(fr.logp.shape) == tuple(fr.lse.shape) == (37,) and torch.all(fr.logp == 0) and torch.all(fr.lse == 0)
        dx, dw = cake_backend.backward_logprob(inp.X, inp.W, inp.labels, fr.lse, inp.dlogp, backend="reference", compact_rows=True)
        assert tuple(dx.shape) == (37, H_HOST) and torch.all(dx == 0) and torch.all(dw == 0)
    result = _run(inp, 16, entry=entry, compact_rows=True)
    _check_dtypes(result, inp)
    assert result["loss"].item() == 0.0 and torch.all(result["logp"] == 0) and torch.all(result["dX"] == 0) and torch.all(result["dW"] == 0)
    assert (cache.hits, cache.misses) == (hits, misses)


# ---------------------------------------------------------------------------
# Device tests (compute capability 10.0 / 10.3 with a registered program; the pinned
# GLM-class geometry H = 6144, V = 154880 with small T)
# ---------------------------------------------------------------------------

CUDA = torch.device("cuda")


@pytest.fixture(scope="module")
def glm_weight():
    _require_program(entry="loss")
    return make_inputs(1, seed=SEED, device=CUDA, ignore_frac=0.0).W


def _device_inputs(T, objective="ce", *, W, seed=SEED):
    return make_inputs(T, objective=objective, seed=seed + T, device=CUDA, W=W, ignore_frac=0.0 if T < 16 else 0.05)


@pytest.mark.parametrize("T", [1, 4095, 4097])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_forward_backward_matches_reference(glm_weight, objective, T):
    _require_program(entry="loss")
    inp = _device_inputs(T, objective, W=glm_weight)
    result = _run(inp, 4096, backend="cake")
    _check_dtypes(result, inp)
    _check_against_references(result, inp, ceiling=True)
    assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(result["dX"][~inp.valid] == 0)


def test_device_logprob_entry(glm_weight):
    _require_program(entry="logprob")
    inp = _device_inputs(4097, W=glm_weight)
    result = _run(inp, 4096, entry="logprob", backend="cake")
    _check_dtypes(result, inp)
    _check_against_references(result, inp, entry="logprob", ceiling=True)
    assert torch.all(result["dX"][~inp.valid] == 0)


def test_device_deterministic_three_runs(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, "policy", W=glm_weight)
    runs = [_run(inp, 4096, backend="cake") for _ in range(3)]
    for r in runs[1:]:
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(r[key], runs[0][key]), key


def test_device_frozen_inputs(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4095, W=glm_weight)
    both = _run(inp, 4096, backend="cake")
    x_only = _run(inp, 4096, backend="cake", train_w=False)
    w_only = _run(inp, 4096, backend="cake", train_x=False)
    neither = _run(inp, 4096, backend="cake", train_x=False, train_w=False)
    assert x_only["dW"] is None and w_only["dX"] is None and neither["dX"] is None and neither["dW"] is None
    assert torch.equal(x_only["dX"], both["dX"]) and torch.equal(w_only["dW"], both["dW"])
    for r in (x_only, w_only, neither):
        assert torch.equal(r["loss"], both["loss"]) and torch.equal(r["logp"], both["logp"])


def test_device_grad_weight_dtype_fp32(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    bf16 = _run(inp, 4096, backend="cake")
    with pytest.raises(ValueError, match="W.dtype"), _quiet_experimental():
        chunked_lm_head_loss(inp.X.detach().requires_grad_(), inp.W, inp.labels, loss_div=inp.loss_div, grad_weight_dtype=torch.float32, backend="cake")
    fr = cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=4096, need_dx=True, need_dw=True,
                                   grad_weight_dtype=torch.float32, backend="cake")
    dX, dW32 = cake_backend.backward_loss(fr.dx_acc, fr.dw_acc, None, grad_weight_dtype=torch.float32, backend="cake",
                                          row_index=fr.row_index, num_rows=fr.num_rows)  # the compacted forward's rows
    torch.cuda.synchronize()
    assert dW32.dtype == torch.float32 and dX.dtype == torch.bfloat16 and tuple(dX.shape) == (inp.T, DEFAULT_H)
    assert torch.equal(fr.loss, bf16["loss"]) and torch.equal(dX, bf16["dX"])
    assert torch.equal(dW32.to(torch.bfloat16), bf16["dW"]) and torch.equal(dW32, fr.dw_acc)
    result = dict(loss=fr.loss, logp=fr.logp, dX=dX, dW=dW32)
    _check_dtypes(result, inp, grad_weight_dtype=torch.float32)
    _check_against_references(result, inp, grad_weight_dtype=torch.float32, ceiling=True)


def test_device_binding_scratch_binds_constant_cells(glm_weight):
    """A remembered binding's per-call scratch binds the device's constant cells (``grad_scale`` / ``unit_scale`` /
    ``f32_dummy``), not fresh ``ones`` / ``zeros``: no fill launches ride along on every remembered call."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    with _cache(True) as cache:
        cache.clear()
        cake_backend.forward_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=4096, backend="cake")
        assert len(cache) == 1
        binding = next(iter(cache._bindings.values()))
    constants = cake_backend._device_constants(binding.device_index)
    assert constants is cake_backend._device_constants(binding.device_index)
    t = {}
    binding._scratch(t)
    assert all(t[name] is constants[name] for name in ("grad_scale", "unit_scale", "f32_dummy"))
    torch.cuda.synchronize()
    assert float(t["grad_scale"]) == 1.0 == float(t["unit_scale"]) and not bool(t["f32_dummy"].any())
    assert tuple(t["f32_dummy"].shape) == (16,) and t["f32_dummy"].dtype == torch.float32


def test_device_binding_cache_hits_are_bitwise_and_pin_nothing(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=4096)
    with _cache(True) as cache:
        cache.clear()
        hits0, misses0 = cache.hits, cache.misses
        first = cake_backend.forward_loss(inp.X, inp.W, inp.labels, backend="cake", **kw)  # validating path, remembers the binding
        second = cake_backend.forward_loss(inp.X, inp.W, inp.labels, backend="cake", **kw)  # remembered binding
        assert (cache.misses, cache.hits) == (misses0 + 1, hits0 + 1)
        with _cache(False):
            fresh = cake_backend.forward_loss(inp.X, inp.W, inp.labels, backend="cake", **kw)
        torch.cuda.synchronize()
        for name in ("loss", "logp", "dx_acc", "dw_acc"):
            a, b, c = getattr(first, name), getattr(second, name), getattr(fresh, name)
            assert torch.equal(a, b) and torch.equal(b, c), name
        assert second.loss.data_ptr() != first.loss.data_ptr()  # outputs are fresh allocations
        key = forward_binding_key(inp.X, inp.W, inp.labels, infer_logp=None, loss_weights=None, need_dx=True, need_dw=True,
                                  grad_weight_dtype=torch.bfloat16, entry="loss", valid_rows=int(inp.valid.sum()), **kw)
        binding = cache.peek(key)
        assert binding is not None and binding.holds_no_tensor() and binding.plan.compact
        assert set(binding.owned) <= {"workspace", "tma_descriptor_workspace"}
        assert cache.owned_bytes == sum(t.numel() * t.element_size() for t in binding.owned.values())
        # another chunk size is another binding (misses once), then hits
        third = cake_backend.forward_loss(inp.X, inp.W, inp.labels, backend="cake", **dict(kw, chunk_size=2048))
        fourth = cake_backend.forward_loss(inp.X, inp.W, inp.labels, backend="cake", **dict(kw, chunk_size=2048))
        torch.cuda.synchronize()
        assert (cache.misses, cache.hits) == (misses0 + 2, hits0 + 2) and len(cache) == 2
        assert torch.equal(third.logp, fourth.logp) and torch.equal(third.dw_acc, fourth.dw_acc)
        # the autograd path reuses the forward binding
        result = _run(inp, 4096, backend="cake")
        assert cache.hits == hits0 + 3 and torch.equal(result["loss"], first.loss)
        assert torch.equal(result["dX"], cake_backend.scatter_rows(first.dx_acc.to(torch.bfloat16), first.row_index, inp.T))


def test_device_runner_launches_without_allocation(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, "policy", W=glm_weight)
    runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, objective="policy", infer_logp=inp.infer_logp, loss_weights=inp.loss_weights,
                                  chunk_size=4096, backend="cake", compact_rows=True)  # the autograd entries' default plan
    runner.step()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    dx, dw = runner.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert dx is runner.dx_out and dw is runner.dw_out
    result = _run(inp, 4096, backend="cake")
    assert torch.equal(runner.loss.reshape(()), result["loss"]) and torch.equal(runner.scatter(runner.logp), result["logp"])
    assert torch.equal(runner.scatter(dx), result["dX"]) and torch.equal(dw, result["dW"])


def test_device_t_changes_between_calls(glm_weight):
    _require_program(entry="loss")
    inputs = [_device_inputs(T, W=glm_weight) for T in (4095, 4097, 4095)]
    fresh = [_run(inp, 4096, backend="cake") for inp in inputs]
    again = [_run(inp, 4096, backend="cake") for inp in inputs]
    for a, b in zip(fresh, again, strict=True):
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(a[key], b[key]), key
    assert torch.equal(fresh[0]["loss"], fresh[2]["loss"])
    _check_against_references(again[1], inputs[1], ceiling=True)


def test_device_compaction_parity(glm_weight):
    """Compacted and uncompacted chunk loops over the same kernels: per-row logp bitwise, dX rows bitwise when the
    chunk's K-slice count agrees, loss / dW at the same precision (different chunk boundaries); the compacted
    runner's step allocates nothing and equals the autograd path."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)  # five percent of the rows ignored
    T_v = int(inp.valid.sum())
    assert 0 < T_v < inp.T
    compact = _run(inp, 4096, backend="cake", compact_rows=True)
    plain = _run(inp, 4096, backend="cake", compact_rows=False)
    _check_dtypes(compact, inp)
    _check_against_references(compact, inp, ceiling=True)
    assert torch.equal(compact["logp"], plain["logp"])
    assert torch.all(compact["dX"][~inp.valid] == 0)
    assert rel_l2(compact["dW"].float(), plain["dW"].float()) <= GATE_TINY["dW_rel_l2"]
    assert torch.allclose(compact["loss"], plain["loss"], rtol=1e-5, atol=1e-6)
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=4096, backend="cake")
    rc = prepare_lm_head_loss(inp.X, inp.W, inp.labels, compact_rows=True, **kw)
    rp = prepare_lm_head_loss(inp.X, inp.W, inp.labels, compact_rows=False, **kw)
    assert rc.plan.compact and rc.plan.rows == T_v and rc.plan.chunks == plan_chunks(T_v, 4096) and rp.plan.chunks == ((0, 4096), (4096, 1))
    assert tuple(rc.logp.shape) == (T_v,) and tuple(rc.dx_out.shape) == (T_v, DEFAULT_H)
    rc.step()
    rp.step()
    torch.cuda.synchronize()
    logp_c = rc.scatter(rc.logp)
    assert torch.equal(logp_c, plain["logp"]) and torch.equal(logp_c, compact["logp"])
    assert torch.equal(rc.scatter(rc.dx_out), compact["dX"]) and torch.equal(rc.dw_out, compact["dW"]) and torch.equal(rc.loss.reshape(()), compact["loss"])
    if rc.plan.dx_slices_of(0) == rp.plan.dx_slices_of(0):  # same K-slice count: the rows of the first plain chunk are bitwise
        assert torch.equal(rc.scatter(rc.dx_out)[:4096], rp.dx_out[:4096])
    assert rc.memory["compact_rows"] and rc.memory["valid_rows"] == T_v and rc.memory["gather_bytes"] == T_v * DEFAULT_H * 2
    before = torch.cuda.memory_stats()
    rc.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    # the log-probability entry: compact saved statistic, scattered gradients
    lp_c = _run(inp, 4096, entry="logprob", backend="cake", compact_rows=True)
    lp_p = _run(inp, 4096, entry="logprob", backend="cake", compact_rows=False)
    _check_against_references(lp_c, inp, entry="logprob", ceiling=True)
    assert torch.equal(lp_c["logp"], lp_p["logp"]) and torch.all(lp_c["dX"][~inp.valid] == 0)
    assert rel_l2(lp_c["dW"].float(), lp_p["dW"].float()) <= GATE_TINY["dW_rel_l2"]


_FRESH_PROCESS_SCRIPT = """
import sys, torch
sys.path.insert(0, {root!r})
from flashinfer.chunked_lm_head import chunked_lm_head_logprob
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs
W = make_inputs(1, seed={seed_w}, device="cuda", ignore_frac=0.0).W  # the module fixture's weight
inp = make_inputs({T}, seed={seed}, device="cuda", W=W, ignore_frac=0.05)
X = inp.X.detach().requires_grad_(True)
W = inp.W.detach().requires_grad_(True)
import warnings
warnings.simplefilter("ignore")
logp = chunked_lm_head_logprob(X, W, inp.labels, chunk_size={C}, backend="cake")
dX, dW = torch.autograd.grad(logp, (X, W), inp.dlogp)
torch.cuda.synchronize()
torch.save(dict(logp=logp.detach().cpu(), dX=dX.cpu(), dW=dW.cpu()), {out!r})
"""


def test_device_logprob_grad_from_fresh_thread_and_process(glm_weight, tmp_path):
    """The log-probability backward recomputes the logits on PyTorch's autograd worker thread under
    ``torch.autograd.grad``: a thread whose first CUDA work is the generated launch has no current CUDA
    context, so the generated host shim must bind the device's primary context itself.  The result must be
    bitwise the main-thread path's, from a fresh thread and from a fresh process."""
    _require_program(entry="logprob")
    T, C = 4097, 4096
    inp = _device_inputs(T, W=glm_weight)

    def grad_path():
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        with _quiet_experimental():
            logp = chunked_lm_head_logprob(X, W, inp.labels, chunk_size=C, backend="cake")
            dX, dW = torch.autograd.grad(logp, (X, W), inp.dlogp)
        torch.cuda.synchronize()
        return dict(logp=logp.detach(), dX=dX, dW=dW)

    main = grad_path()
    _check_against_references(main, inp, entry="logprob", ceiling=True)
    outcome = {}

    def worker():
        try:
            outcome["result"] = grad_path()
        except BaseException as exc:  # surfaced by the assertion below
            outcome["error"] = exc

    thread = threading.Thread(target=worker, name="fresh-launch-thread")
    thread.start()
    thread.join()
    assert "error" not in outcome, f"launch from a fresh thread failed: {outcome.get('error')!r}"
    for key in ("logp", "dX", "dW"):
        assert torch.equal(outcome["result"][key], main[key]), key
    # a fresh process: the same seeded inputs, the same entry, bitwise the same result
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = tmp_path / "fresh_process.pt"
    script = _FRESH_PROCESS_SCRIPT.format(root=root, T=T, seed=SEED + T, seed_w=SEED, C=C, out=str(out))
    env = dict(os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", ""))
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env, cwd=root, timeout=1800)
    assert proc.returncode == 0, f"fresh process failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    fresh = torch.load(out)
    for key in ("logp", "dX", "dW"):
        assert torch.equal(fresh[key].to(main[key].device), main[key]), key


def test_device_memory_rule(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div, chunk_size=4096, backend="cake")
    runner.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    m = runner.memory
    assert m["vocab_rows_max"] == 4096 and m["num_chunks"] == 2
    assert m["temporary"]["logits"] == 4096 * DEFAULT_V * 2  # never the full [T, V]
    assert peak <= m["temporary_bytes"] + m["accumulator_bytes"] + m["outputs_bytes"] + (256 << 20)
    # the vocabulary-sized part of the peak is the reported chunk workspace (logits + statistics + K-slice slabs), never a
    # [T, V] buffer: the unchunked path at the contract's boundaries holds BF16 z plus its FP32 promotion (T * V * 6 bytes)
    assert peak - m["accumulator_bytes"] - m["outputs_bytes"] <= m["temporary_bytes"] + (256 << 20)
    assert m["temporary_bytes"] < inp.T * DEFAULT_V * 6 and m["temporary"]["logits"] < inp.T * DEFAULT_V * 2
    # a step through the prepared runner allocates nothing more
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    runner.step()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base == 0
