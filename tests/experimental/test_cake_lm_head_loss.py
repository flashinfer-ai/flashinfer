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
    stage: set(tensors)
    | set(COMMON_TENSORS)
    | set(COMMON_SCALARS)
    | {"num_vecs"}
    | (_GEMM_VALUES if stage.startswith("gemm") else set())
    | ({"n_slabs"} if stage == "slab_sum" else set())
    | ({"row_vecs", "num_rows"} if stage == "scale_cast_scatter_bf16" else set())
    for stage, tensors in STAGE_TENSORS.items()
}


def _device_supported() -> bool:
    return torch.cuda.is_available() and (
        torch.cuda.get_device_capability(0) in SUPPORTED_COMPUTE_CAPABILITIES
    )


def _require_program(*, entry: str = "loss"):
    if not _device_supported():
        pytest.skip("requires a compute capability 10.0 / 10.3 device")
    if not generated_program_available(torch.device("cuda"), entry=entry):
        pytest.skip(
            f"generated chunked LM-head program ({entry} entry) not registered for this device"
        )


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
    return make_inputs(
        T, objective=objective, seed=seed + T, H=H_HOST, V=V_HOST, device=DEV, **kw
    )


def _run(
    inp,
    C,
    *,
    entry="loss",
    grad_weight_dtype=torch.bfloat16,
    train_x=True,
    train_w=True,
    backend="reference",
    scale=None,
    compact_rows=None,
    fuse_dw_cast=None,
):
    """One forward + backward through the autograd entry points; ``backend="cake"`` goes
    through the public module.  Returns ``loss`` / ``logp`` / ``dX`` / ``dW`` and the leaves."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    loss_fn = (
        chunked_lm_head_loss if backend == "cake" else cake_backend.chunked_lm_head_loss
    )
    logprob_fn = (
        chunked_lm_head_logprob
        if backend == "cake"
        else cake_backend.chunked_lm_head_logprob
    )
    with _quiet_experimental():
        if entry == "loss":
            loss, logp = loss_fn(
                X,
                W,
                inp.labels,
                objective=inp.objective,
                loss_div=inp.loss_div if inp.objective == "ce" else None,
                infer_logp=inp.infer_logp,
                loss_weights=inp.loss_weights,
                chunk_size=C,
                return_logp=True,
                grad_weight_dtype=grad_weight_dtype,
                backend=backend,
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
            )
            if train_x or train_w:
                (loss if scale is None else loss * scale).backward()
        else:
            logp = logprob_fn(
                X,
                W,
                inp.labels,
                chunk_size=C,
                backend=backend,
                compact_rows=compact_rows,
                fuse_dw_cast=fuse_dw_cast,
            )
            loss = (logp.detach() * inp.dlogp)[inp.valid].sum()
            if train_x or train_w:
                logp.backward(inp.dlogp if scale is None else inp.dlogp * scale)
    if DEV.type == "cuda":
        torch.cuda.synchronize()
    return dict(loss=loss.detach(), logp=logp.detach(), dX=X.grad, dW=W.grad, X=X, W=W)


def _check_against_references(
    result, inp, *, entry="loss", grad_weight_dtype=torch.bfloat16, ceiling=False
):
    oracle = reference_fp64(inp, entry=entry)
    b0 = reference_unchunked(inp, entry=entry, grad_weight_dtype=grad_weight_dtype)
    errors = error_report(result, oracle, inp.labels)
    b0_errors = error_report(b0, oracle, inp.labels)
    assert not errors["nan"]
    for key, tiny in GATE_TINY.items():
        if key in errors:
            assert errors[key] <= GATE_MARGIN * b0_errors[key] + tiny, (
                f"{key}: {errors[key]:.3e} vs unchunked {b0_errors[key]:.3e}"
            )
    if ceiling:
        for key, measured in PRODUCTION_B0.items():
            if key in errors:
                assert errors[key] <= 2.0 * measured, (
                    f"{key}: {errors[key]:.3e} above twice the production-geometry unchunked error"
                )
    assert errors["logp_ignored_zero"]
    if result.get("dX") is not None:
        assert errors["dX_ignored_zero"]
    return errors, b0_errors


def _check_dtypes(result, inp, *, grad_weight_dtype=torch.bfloat16):
    assert result["loss"].dtype == torch.float32 and result["loss"].shape == ()
    assert result["logp"].dtype == torch.float32 and tuple(result["logp"].shape) == (
        inp.T,
    )
    if result["dX"] is not None:
        assert result["dX"].dtype == torch.bfloat16 and tuple(result["dX"].shape) == (
            inp.T,
            inp.H,
        )
    if result["dW"] is not None:
        assert result["dW"].dtype == grad_weight_dtype and tuple(
            result["dW"].shape
        ) == (inp.V, inp.H)


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
        "gemm_logits",
        "gemm_logits_g16",
        "gemm_logits_nostats",
        "gemm_logits_nostats_g16",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx",
        "gemm_dx_s2",
        "gemm_dx_s3",
        "gemm_dx_s4",
        "gemm_dx_tn256",
        "gemm_dx_s2_tn256",
        "gemm_dx_s3_tn256",
        "gemm_dx_s4_tn256",
        "gemm_dx_st3",
        "gemm_dx_s2_st3",
        "gemm_dx_s3_st3",
        "gemm_dx_s4_st3",
        "slab_sum",
        "gemm_dw_acc",
        "gemm_dw_acc_g16",
        "gemm_dw_acc_g32",
        "gemm_dw_acc_gn12",
        "gemm_dw_cast_bf16",
        "gemm_dw_cast_bf16_g16",
        "gemm_dw_cast_bf16_g32",
        "gemm_dw_cast_f32",
        "gemm_dw_cast_f32_g16",
        "gemm_dw_cast_f32_g32",
        "gemm_dw_cast_f32_gn12",
        "scale_cast_bf16",
        "scale_cast_f32",
        "scale_cast_scatter_bf16",
    )
    assert (
        tuple(s for s in cake_jit.STAGES if cake_backend.base_stage(s) == s)
        == cake_jit.BASE_STAGES
    )
    assert (
        tuple(s for s in cake_jit.STAGES if s.startswith("gemm_"))
        == cake_jit.GEMM_STAGES
    )
    assert set(STAGE_TENSORS) == set(cake_jit.STAGES)
    assert set(CONTRACT_ALIASES.values()) <= set().union(*_STAGE_VALUES.values())
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in cake_jit.ARCH_NVCC_FLAGS
        assert record["arch"] in SUPPORTED_COMPUTE_CAPABILITIES.values()
        assert record_abi(record) in SUPPORTED_ABIS
        assert len(record["closure_sha256"]) == 64
        assert (
            cake_jit.select_module(record["arch"], *cake_jit.record_geometry(record))
            == name
        )
        if name == cake_jit.PRIMARY_RECORD.format(arch=record["arch"]):
            assert cake_jit.select_module(record["arch"]) == name
        geometry = Geometry.from_record(record)
        assert (
            geometry.hidden is None or geometry.hidden % geometry.hidden_multiple == 0
        )
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
                    assert CONTRACT_ALIASES.get(arg, arg) in _STAGE_VALUES[stage], (
                        stage,
                        arg,
                    )
            assert int(physical.get("tma_workspace_bytes", 0)) >= 0
            assert int(physical.get("workspace_bytes", 0)) >= 0
            assert len(physical.get("grid", ["rows_c", 1, 1])) == 3
            launch = physical.get("launch")
            if launch is not None:
                assert len(launch["block"]) == 3 and all(
                    int(b) >= 1 for b in launch["block"]
                )
                assert len(launch["cluster"]) == 3 and all(
                    int(c) >= 1 for c in launch["cluster"]
                )


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
    unfused = dict(fuse_dw_cast=False)
    full = stages_for_entry("loss", **unfused)
    assert full == (
        "gemm_logits",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx",
        "gemm_dw_acc",
        "scale_cast_bf16",
    )
    assert list(full) == [s for s in cake_jit.STAGES if s in full]  # launch order
    frozen_x = stages_for_entry("loss", need_dx=False, **unfused)
    assert (
        "gemm_dx" not in frozen_x
        and "gemm_dw_acc" in frozen_x
        and "scale_cast_bf16" in frozen_x
    )
    frozen_w = stages_for_entry("loss", need_dw=False, **unfused)
    assert (
        "gemm_dw_acc" not in frozen_w
        and "gemm_dx" in frozen_w
        and "scale_cast_f32" not in frozen_w
    )
    frozen_w32 = stages_for_entry(
        "loss", need_dw=False, grad_weight_dtype=torch.float32, **unfused
    )
    assert "scale_cast_f32" not in frozen_w32  # the FP32 cast belongs to dW only
    neither = stages_for_entry("loss", need_dx=False, need_dw=False, **unfused)
    assert neither == ("gemm_logits", "row_finalize", "loss_reduce")
    fp32 = stages_for_entry("loss", grad_weight_dtype=torch.float32, **unfused)
    assert (
        "scale_cast_f32" in fp32 and "scale_cast_bf16" in fp32
    )  # dW in FP32, dX in BF16
    logprob = stages_for_entry("logprob", **unfused)
    assert "gemm_logits_nostats" in logprob and "loss_reduce" not in logprob
    assert logprob == (
        "gemm_logits",
        "gemm_logits_nostats",
        "row_finalize",
        "row_grad",
        "gemm_dx",
        "gemm_dw_acc",
        "scale_cast_bf16",
    )
    assert "scale_cast_f32" not in stages_for_entry(
        "logprob", grad_weight_dtype=torch.float32, **unfused
    )  # dW is BF16 there
    assert stages_for_entry("logprob", need_dx=False, need_dw=False, **unfused) == (
        "gemm_logits",
        "gemm_logits_nostats",
        "row_finalize",
    )
    # the fused weight-gradient cast: the last chunk's GEMM replaces the flat dW cast (dX keeps its cast),
    # the accumulator GEMM stays for the chunks before it and is absent from a one-chunk plan
    fused = stages_for_entry("loss", fuse_dw_cast=True)
    assert fused == (
        "gemm_logits",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx",
        "gemm_dw_acc",
        "gemm_dw_cast_bf16",
        "scale_cast_bf16",
    )
    assert stages_for_entry("loss", fuse_dw_cast=True, num_chunks=2) == fused
    one = stages_for_entry("loss", fuse_dw_cast=True, num_chunks=1)
    assert "gemm_dw_acc" not in one and "gemm_dw_cast_bf16" in one
    fused32 = stages_for_entry(
        "loss", grad_weight_dtype=torch.float32, fuse_dw_cast=True
    )
    assert (
        "gemm_dw_cast_f32" in fused32
        and "gemm_dw_cast_bf16" not in fused32
        and "scale_cast_f32" not in fused32
        and "scale_cast_bf16" in fused32
    )
    assert stages_for_entry("logprob", fuse_dw_cast=True)[-3:] == (
        "gemm_dw_acc",
        "gemm_dw_cast_bf16",
        "scale_cast_bf16",
    )
    frozen_w_fused = stages_for_entry("loss", need_dw=False, fuse_dw_cast=True)
    assert frozen_w_fused == frozen_w  # no weight gradient, no fused stage
    assert stages_for_entry("loss", need_dx=False, fuse_dw_cast=True, num_chunks=1) == (
        "gemm_logits",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dw_cast_bf16",
    )
    # the default form follows the environment knob
    assert stages_for_entry("loss") == stages_for_entry(
        "loss", fuse_dw_cast=cake_backend.fuse_dw_cast_default()
    )
    # dx_cast=False: the flat dX cast leaves (the caller finalizes dx_acc); the dW cast stays
    uncast = stages_for_entry("logprob", dx_cast=False, need_dw=False, **unfused)
    assert "scale_cast_bf16" not in uncast and "gemm_dx" in uncast
    assert "scale_cast_bf16" in stages_for_entry(
        "logprob", dx_cast=False, **unfused
    )  # the unfused bf16 dW cast stays
    assert "scale_cast_bf16" in stages_for_entry(
        "loss", dx_cast=False, **unfused
    )  # the unfused bf16 dW cast
    assert "scale_cast_bf16" not in stages_for_entry(
        "loss", dx_cast=False, fuse_dw_cast=True
    )
    assert "slab_sum" not in stages_for_entry(
        "loss"
    ) and "scale_cast_scatter_bf16" not in stages_for_entry("loss")
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
    assert (p.ld_x, p.x_copy, p.objective, p.loss_div, p.entry, p.mode) == (
        H_HOST,
        False,
        "ce",
        1.0,
        "loss",
        MODE_CE,
    )
    wide = torch.zeros(4, H_HOST + 8, dtype=torch.bfloat16)[:, :H_HOST]
    p = validate_lm_head_inputs(**dict(kw, X=wide))
    assert not p.x_copy and p.ld_x == H_HOST + 8  # a 16 B pitch is launched as is
    odd = torch.zeros(4, H_HOST + 3, dtype=torch.bfloat16)[:, :H_HOST]
    p = validate_lm_head_inputs(**dict(kw, X=odd))
    assert p.x_copy and p.ld_x == H_HOST  # copied to a contiguous tensor
    p = validate_lm_head_inputs(**dict(kw, X=kw["X"][:0], labels=kw["labels"][:0]))
    assert p.num_rows == 0 and not p.x_copy
    policy = validate_lm_head_inputs(
        **dict(
            kw,
            objective="policy",
            loss_div=None,
            infer_logp=torch.zeros(4),
            loss_weights=torch.zeros(4),
        )
    )
    assert policy.mode == MODE_POLICY and policy.loss_div is None
    logprob = validate_lm_head_inputs(**dict(kw, loss_div=None), entry="logprob")
    assert logprob.mode == MODE_NONE and logprob.entry == "logprob"
    # loss_div also as a CPU scalar tensor / an int
    assert (
        validate_lm_head_inputs(**dict(kw, loss_div=torch.tensor(2.5))).loss_div == 2.5
    )
    assert validate_lm_head_inputs(**dict(kw, loss_div=3)).loss_div == 3.0


def _cuda_scalar():
    if not torch.cuda.is_available():
        pytest.skip("a CUDA scalar needs a CUDA device")
    return torch.tensor(1.0, device="cuda")


@pytest.mark.parametrize(
    "mutate, exc, match",
    [
        (lambda kw: dict(kw, X=kw["X"].float()), ValueError, "BF16"),
        (
            lambda kw: dict(
                kw, W=torch.zeros(H_HOST, V_HOST, dtype=torch.bfloat16).t()
            ),
            ValueError,
            "contiguous",
        ),
        (
            lambda kw: dict(kw, X=torch.zeros(H_HOST, 4, dtype=torch.bfloat16).t()),
            ValueError,
            "contiguous",
        ),
        (
            lambda kw: dict(
                kw,
                X=torch.zeros(4, 320, dtype=torch.bfloat16),
                W=torch.zeros(V_HOST, 320, dtype=torch.bfloat16),
            ),
            ValueError,
            "multiple of 256",
        ),
        (
            lambda kw: dict(kw, W=torch.zeros(500, H_HOST, dtype=torch.bfloat16)),
            ValueError,
            "multiple of 256",
        ),
        (
            lambda kw: dict(
                kw, W=torch.zeros(V_HOST, 2 * H_HOST, dtype=torch.bfloat16)
            ),
            ValueError,
            "differ",
        ),
        (lambda kw: dict(kw, labels=kw["labels"].int()), ValueError, "int64"),
        (lambda kw: dict(kw, labels=kw["labels"][:3]), ValueError, r"\[T\]"),
        (lambda kw: dict(kw, chunk_size=0), ValueError, "positive integer"),
        (lambda kw: dict(kw, chunk_size=True), ValueError, "positive integer"),
        (lambda kw: dict(kw, chunk_size=70000), ValueError, "65535"),
        (
            lambda kw: dict(kw, grad_weight_dtype=torch.float16),
            ValueError,
            "grad_weight_dtype",
        ),
        (lambda kw: dict(kw, loss_div=None), ValueError, "loss_div"),
        (lambda kw: dict(kw, infer_logp=torch.zeros(4)), ValueError, "policy"),
        (lambda kw: dict(kw, loss_div=0.0), ValueError, "positive"),
        (lambda kw: dict(kw, loss_div=-1.0), ValueError, "positive"),
        (lambda kw: dict(kw, loss_div=_cuda_scalar()), ValueError, "synchronize"),
        (
            lambda kw: dict(
                kw, objective="policy", loss_div=None, infer_logp=torch.zeros(4)
            ),
            ValueError,
            "loss_weights",
        ),
        (
            lambda kw: dict(
                kw,
                objective="policy",
                loss_div=None,
                infer_logp=torch.zeros(4, dtype=torch.float64),
                loss_weights=torch.zeros(4),
            ),
            ValueError,
            "FP32",
        ),
        (
            lambda kw: dict(
                kw,
                objective="policy",
                loss_div=1.0,
                infer_logp=torch.zeros(4),
                loss_weights=torch.zeros(4),
            ),
            ValueError,
            "loss_div",
        ),
        (lambda kw: dict(kw, objective="other"), ValueError, "objective"),
        (lambda kw: dict(kw, entry="logprob"), ValueError, "objective arguments"),
        (
            lambda kw: dict(kw, deterministic=False),
            NotImplementedError,
            "deterministic",
        ),
        (
            lambda kw: dict(
                kw,
                geometry=Geometry.from_record(
                    {"geometry": {"hidden": DEFAULT_H, "vocab": DEFAULT_V}}
                ),
            ),
            ValueError,
            "specialized",
        ),
    ],
    ids=[
        "x_fp32",
        "w_noncontiguous",
        "x_column_major",
        "h_not_256",
        "v_not_256",
        "hidden_mismatch",
        "labels_int32",
        "labels_length",
        "chunk_zero",
        "chunk_bool",
        "chunk_too_large",
        "grad_dtype_fp16",
        "ce_without_loss_div",
        "ce_with_infer_logp",
        "loss_div_zero",
        "loss_div_negative",
        "loss_div_cuda",
        "policy_without_weights",
        "policy_infer_fp64",
        "policy_with_loss_div",
        "unknown_objective",
        "logprob_with_objective_args",
        "nondeterministic",
        "pinned_geometry",
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
        assert offsets == sorted(offsets) and all(
            o % WORKSPACE_ALIGN == 0 for o in offsets
        )
        assert layout["logits"][1] == rows * V * 2
        assert layout["stats"][1] == rows * (V // 256) * 8
        assert layout["d"][1] == rows * 4 and layout["term"][1] == rows * 4
        assert (
            layout["loss_acc"][1] == 8 and layout["grad_scale"][1] == 4
        )  # FP64 loss accumulator, FP32 scale
        assert layout["total"] % WORKSPACE_ALIGN == 0 and layout["total"] >= sum(
            layout[k][1] for k in regions
        )
        assert layout == workspace_layout(
            rows, V, C
        )  # depends on T only through min(T, C)
    assert (
        workspace_layout(4097, V, C)
        == workspace_layout(16231, V, C)
        == workspace_layout(4096, V, C)
    )
    assert (
        workspace_layout(100, V, C)["logits"][1]
        < workspace_layout(4096, V, C)["logits"][1]
    )
    assert workspace_layout(0, V, C) == workspace_layout(1, V, C)
    with_scratch = workspace_layout(
        100, V, C, tma_workspace_bytes=1024, scratch_bytes=4096
    )
    assert (
        with_scratch["workspace"][1] == 4096
        and with_scratch["tma_descriptor_workspace"][1] == 1024
    )
    assert with_scratch["total"] == workspace_layout(100, V, C)["total"] + 4096 + 1024
    assert (
        workspace_layout(100, V, C, stats_tile=128)["stats"][1]
        == 2 * workspace_layout(100, V, C)["stats"][1]
    )

    T = 16231
    m = memory_report(T, H, V, C)
    for key in (
        "temporary_bytes",
        "temporary",
        "outputs_bytes",
        "outputs",
        "accumulator_bytes",
        "accumulators",
        "weights_bytes",
        "weights",
    ):
        assert key in m
    assert m["vocab_rows_max"] == 4096 and m["chunk"] == C and m["num_chunks"] == 4
    assert (
        m["temporary"]["logits"] == 4096 * V * 2
        and m["temporary"]["lse"] == T * 4
        and m["temporary"]["logp"] == T * 4
    )
    assert m["temporary_bytes"] == sum(m["temporary"].values())
    assert m["accumulators"] == {"dX_acc": T * H * 4, "dW_acc": V * H * 4}
    assert m["outputs"] == {"loss": 4, "logp": 0, "dX": T * H * 2, "dW": V * H * 2}
    assert m["weights"] == {"W": V * H * 2, "X": T * H * 2}
    assert (
        memory_report(T, H, V, C, grad_weight_dtype=torch.float32)["outputs"]["dW"]
        == 2 * m["outputs"]["dW"]
    )
    returned = memory_report(T, H, V, C, return_logp=True)
    assert returned["outputs"]["logp"] == T * 4 and "logp" not in returned["temporary"]
    assert memory_report(T, H, V, C, x_copy=True)["temporary"]["x_copy"] == T * H * 2
    frozen = memory_report(T, H, V, C, need_dx=False)
    assert "dX_acc" not in frozen["accumulators"] and "dX" not in frozen["outputs"]
    lp = memory_report(T, H, V, C, entry="logprob", need_dx=False)
    assert lp["accumulators"] == {"dW_acc": V * H * 4, "saved_lse": T * 4}
    assert (
        lp["outputs"]["loss"] == 0
        and lp["outputs"]["logp"] == T * 4
        and lp["temporary"]["dlogp"] == T * 4
    )
    assert lp["outputs"]["dW"] == V * H * 2  # BF16 in the log-probability entry
    assert memory_report(1, H, V, C)["vocab_rows_max"] == 1
    empty = memory_report(0, H, V, C)
    assert empty["num_chunks"] == 0 and empty["vocab_rows_max"] == 1
    # every vocabulary-sized temporary spans at most C rows regardless of T
    for T in (1, 4095, 4096, 4097, 16231, 32463):
        rep = memory_report(T, H, V, C)
        assert rep["vocab_rows_max"] == min(T, C)
        assert rep["temporary"]["logits"] == min(T, C) * V * 2
    assert (
        lm_head_loss_workspace_size(16231, V, C, backend="reference")
        == workspace_layout(16231, V, C)["total"]
    )
    assert (
        lm_head_loss_workspace_size(16231, V, C, entry="logprob", backend="reference")
        == workspace_layout(16231, V, C, entry="logprob")["total"]
    )


def test_grid_dims_holds_no_reference_to_the_host_values():
    """The grid evaluator must not capture the stage's host values: scale_cast binds a fresh launch per call, and a
    recursive closure over ``scalars`` formed a reference cycle that kept every call's accumulator and output alive
    until the cyclic GC (about 5.4 GiB per training step at the GLM geometry -- OOM within one benchmark campaign)."""
    import gc
    import weakref

    gc.collect()
    gc.disable()
    try:
        acc = torch.zeros(
            8
        )  # rides along in the host values like scale_cast's accumulator
        ref = weakref.ref(acc)
        scalars = {
            "rows_c": 4097,
            "m_tiles": 33,
            "num_vecs": 9,
            "acc": acc,
            "out": None,
        }
        assert grid_dims(
            ["max(1, min(m_tiles//4*24, resident))*4", "rows_c/8", "num_vecs"],
            scalars,
            148,
            resident=30,
        ) == (120, 513, 9)
        del scalars, acc
        assert ref() is None, (
            "grid_dims left a reference cycle over the host values (freed only by the cyclic GC)"
        )
    finally:
        gc.enable()


def test_grid_dims():
    scalars = {"m_tiles": 32, "rows_c": 4097, "V": DEFAULT_V, "num_vecs": 24}
    assert grid_dims(
        ["max(1, min(m_tiles//2*605, sms//2))*2", "rows_c/8", 1], scalars, 148
    ) == (148, 513, 1)
    assert grid_dims(["rows_c//8", 1, 1], scalars, 148) == (512, 1, 1)  # floor
    assert grid_dims(["rows_c/8", 1, 1], scalars, 148) == (513, 1, 1)  # ceil
    assert grid_dims(["V/256", "rows_c", 1], scalars, 148) == (605, 4097, 1)
    assert grid_dims(["sms", "sms*2", "sms-100"], scalars, 148) == (148, 296, 48)
    assert grid_dims([4, 2, 1], scalars, 148) == (4, 2, 1)
    assert grid_dims(["min(num_vecs, 8) + max(2, 3)", 1, 1], scalars, 148) == (11, 1, 1)
    assert grid_dims(["(rows_c - 1) // 4096 + 1", 1, 1], scalars, 148) == (2, 1, 1)
    assert grid_dims([0, "rows_c - rows_c", "1 - 6"], scalars, 148) == (
        1,
        1,
        1,
    )  # clamped to one
    with pytest.raises(ValueError):
        grid_dims(
            ["-rows_c", 1, 1], scalars, 148
        )  # unary operators are not part of the grammar
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
    assert grid_dims(
        ["max(1, min(m_tiles//4*24, resident))*4", 1, 1], scalars, 148, resident=33
    ) == (132, 1, 1)
    assert grid_dims(
        ["max(1, min(m_tiles//4*24, resident))*4", 1, 1],
        {"m_tiles": 4},
        148,
        resident=33,
    ) == (96, 1, 1)
    assert grid_dims(
        ["max(1, min(m_tiles//2*24, resident))*2", 1, 1], scalars, 148, resident=74
    ) == (148, 1, 1)
    # a dynamically scheduled kernel launches its whole work-item domain
    assert grid_dims(["max(1, m_tiles//2*605)*2", 1, 1], scalars, 148) == (19360, 1, 1)
    with pytest.raises(
        KeyError, match="unknown"
    ):  # only a clustered stage provides ``resident``
        grid_dims(["min(m_tiles, resident)", 1, 1], scalars, 148)


def test_geometry_cluster_ctas():
    g = Geometry.from_record(
        {
            "geometry": {
                "logits_cluster_ctas": 2,
                "dx_cluster_ctas": 4,
                "dw_cluster_ctas": 2,
            }
        }
    )
    assert (g.logits_cluster_ctas, g.dx_cluster_ctas, g.dw_cluster_ctas) == (2, 4, 2)
    assert (
        g.row_tiles(4097) == 34
        and g.row_tiles(4097, 4) == 36
        and g.row_tiles(1, 4) == 4
        and g.row_tiles(4096, 4) == 32
    )
    assert (
        g.cluster_ctas_of("gemm_logits") == 2
        and g.cluster_ctas_of("gemm_logits_nostats") == 2
    )
    assert (
        g.cluster_ctas_of("gemm_dx") == 4
        and g.cluster_ctas_of("gemm_dx_s3") == 4
        and g.cluster_ctas_of("gemm_dx_s3_tn256") == 4
        and g.cluster_ctas_of("gemm_dw_acc") == 2
        and g.cluster_ctas_of("gemm_dw_acc_g32") == 2
        and g.cluster_ctas_of("gemm_dw_acc_g16") == 2
        and g.cluster_ctas_of("gemm_logits_g16") == 2
        and g.cluster_ctas_of("gemm_dw_cast_f32_g32") == 2
    )
    assert (
        g.cluster_ctas_of("row_grad") is None
        and g.cluster_ctas_of("scale_cast_bf16") is None
    )
    default = Geometry.from_record(None)
    assert (
        default.logits_cluster_ctas,
        default.dx_cluster_ctas,
        default.dw_cluster_ctas,
    ) == (2, 2, 2)
    with pytest.raises(ValueError):
        Geometry.from_record({"geometry": {"dx_cluster_ctas": 0}})


# --------------------------------------------------------------------------- per-chunk instance variants


def test_stage_variant_grammar():
    bases = cake_jit.BASE_STAGES
    assert len(bases) == 13 and set(bases) <= set(cake_jit.STAGES)
    for stage in cake_jit.STAGES:
        base, knobs = cake_backend.parse_stage(stage)
        assert base in bases and cake_backend.stage_variant(base, **knobs) == stage
        assert set(knobs) == {
            "k_slices",
            "tile_n",
            "group_m",
            "epi_store",
            "stages",
            "group_n",
        }
    assert (
        cake_backend.stage_variant("gemm_dx", k_slices=3, tile_n=256)
        == "gemm_dx_s3_tn256"
    )
    assert cake_backend.stage_variant("gemm_dx", k_slices=1, tile_n=512) == "gemm_dx"
    # the 3-deep ring of the 512-wide dX tile, the 2-D blocked weight-gradient raster
    assert (
        cake_backend.stage_variant("gemm_dx", k_slices=3, stages=3) == "gemm_dx_s3_st3"
    )
    assert cake_backend.stage_variant("gemm_dx", stages=3, tile_n=512) == "gemm_dx_st3"
    assert (
        cake_backend.stage_variant("gemm_dx", stages=4) == "gemm_dx"
    )  # the table depth
    assert cake_backend.stage_variant("gemm_dw_acc", group_n=12) == "gemm_dw_acc_gn12"
    assert cake_backend.stage_variant("gemm_dw_acc", group_n=0) == "gemm_dw_acc"
    assert (
        cake_backend.stage_variant("gemm_dw_cast_f32", group_n=12)
        == "gemm_dw_cast_f32_gn12"
    )
    assert cake_backend.parse_stage("gemm_dx_s2_st3") == (
        "gemm_dx",
        {
            "k_slices": 2,
            "tile_n": None,
            "group_m": None,
            "epi_store": None,
            "stages": 3,
            "group_n": None,
        },
    )
    assert cake_backend.parse_stage("gemm_dw_cast_f32_gn12")[1]["group_n"] == 12
    for bad in (
        dict(base="gemm_dx", tile_n=256, stages=3),  # the narrow tile keeps its depth
        dict(base="gemm_dx", stages=5),
        dict(base="gemm_dw_acc", stages=3),
        dict(
            base="gemm_dw_cast_bf16", group_n=12
        ),  # the bf16 cast keeps the 1-D raster
        dict(base="gemm_dw_acc", group_n=8),
        dict(base="gemm_logits", group_n=12),
    ):
        with pytest.raises(ValueError):
            cake_backend.stage_variant(**bad)
    for bad_name in ("gemm_dw_cast_bf16_gn12", "gemm_dx_tn256_st3", "gemm_dx_gn12"):
        with pytest.raises(ValueError):
            cake_backend.parse_stage(bad_name)
    assert (
        cake_backend.stage_variant("gemm_dw_acc", group_m=32, epi_store="tma")
        == "gemm_dw_acc_g32_tma"
    )
    assert (
        cake_backend.stage_variant("gemm_dw_acc", group_m=16, epi_store="tma")
        == "gemm_dw_acc_g16_tma"
    )  # grammar only: no rule selects an epilogue form and no record registers a _tma stage
    assert cake_backend.stage_variant("gemm_dw_acc", group_m=16) == "gemm_dw_acc_g16"
    assert (
        cake_backend.stage_variant("gemm_dw_cast_bf16", group_m=16)
        == "gemm_dw_cast_bf16_g16"
    )
    assert cake_backend.parse_stage("gemm_dw_cast_f32_g16") == (
        "gemm_dw_cast_f32",
        dict(
            k_slices=1,
            tile_n=None,
            group_m=16,
            epi_store=None,
            stages=None,
            group_n=None,
        ),
    )
    assert cake_backend.stage_variant("gemm_dw_acc", epi_store="redsm") == "gemm_dw_acc"
    assert (
        cake_backend.stage_variant("gemm_dw_cast_f32", group_m=32)
        == "gemm_dw_cast_f32_g32"
    )
    assert (
        cake_backend.stage_variant("gemm_logits_nostats", group_m=16)
        == "gemm_logits_nostats_g16"
    )
    assert (
        cake_backend.dx_stage(2) == "gemm_dx_s2"
        and cake_backend.dx_stage(1, 256) == "gemm_dx_tn256"
    )
    assert cake_backend.dx_stage(4, 512) == "gemm_dx_s4"
    for bad in (
        dict(base="row_grad", group_m=16),
        dict(base="gemm_logits", k_slices=2),
        dict(base="gemm_dx", tile_n=128),
        dict(base="gemm_dx", k_slices=5),
        dict(base="gemm_dw_cast_bf16", epi_store="tma"),
        dict(base="gemm_logits", epi_store="tma"),
        dict(base="gemm_dw_acc", epi_store="cast_bf16"),
        dict(base="other"),
    ):
        with pytest.raises(ValueError):
            cake_backend.stage_variant(bad.pop("base"), **bad)
    for bad in (
        "gemm_dx_g16",
        "gemm_logits_s2",
        "gemm_dw_acc_tma_g32",
        "gemm_dx_tn128",
        "row_grad_g16",
        "scale_cast",
        "gemm_dw_cast_bf16_tma",
    ):
        with pytest.raises(ValueError):
            cake_backend.parse_stage(bad)
    with pytest.raises(ValueError):
        cake_backend.base_stage("gemm_dx_s9")


def test_instance_rules_mirror_the_launchers():
    assert (cake_backend.RASTER_RULE_MIN_HIDDEN, cake_backend.RASTER_RULE_MIN_ROWS) == (
        7168,
        2049,
    )
    assert cake_backend.RASTER_WIDE_GROUPS == {"sm_100a": (16, 32), "sm_103a": (16, 32)}
    assert cake_backend.DW_LONG_CHUNK_GROUP_M == 16
    assert cake_backend.DW_LONG_CHUNK_GROUPS == {"sm_100a": 16, "sm_103a": 16}
    assert cake_backend.DW_LONG_CHUNK_MIN_ROWS == 4097
    assert cake_backend.LOGITS_LONG_RASTER_GROUPS == {"sm_100a": 16, "sm_103a": 16}
    assert cake_backend.LOGITS_LONG_RASTER_MIN_ROWS == 3841  # 30 row tiles of 128 + 1
    assert not hasattr(
        cake_backend, "dw_epilogue_variant"
    )  # no epilogue rule: the SM103 TMA reduce-add above 4096 rows was retired
    assert (cake_backend.DX_TILE_WIDE, cake_backend.DX_TILE_NARROW) == (512, 256)
    assert cake_backend.DX_LONG_CHUNK_STAGES == {"sm_100a": 3, "sm_103a": 3}
    assert cake_backend.DX_WIDE_TILE_STAGES == 4
    assert cake_backend.DW_BLOCK_GROUPS == {"sm_100a": 12, "sm_103a": 12}
    assert (cake_backend.DW_BLOCK_MIN_ROWS, cake_backend.DW_BLOCK_MAX_ROWS) == (
        2049,
        4096,
    )
    assert cake_backend.DW_BLOCK_BASES == ("gemm_dw_acc", "gemm_dw_cast_f32")
    # the dX tile rule's wave-efficiency floor: SM100 only, one-slice launches below H 7168 (T 65031 / 32463 tails at
    # chunk 4096: 15 row pairs x 12 = 180 wide items on 74 SM pairs = 2.43 waves, 81 % < 83 % -> the 256-wide form)
    assert cake_backend.DX_WIDE_MIN_EFF == {"sm_100a": 0.83}
    g_glm = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    g_wide = Geometry.from_record({"geometry": {"hidden": 7168, "vocab": 129280}})
    assert cake_backend.dx_tile_rule(3591, 6144, 148, 1, g_glm, "sm_100a") == 256
    assert cake_backend.dx_tile_rule(3663, 6144, 148, 1, g_glm, "sm_100a") == 256
    assert cake_backend.dx_tile_rule(2049, 6144, 148, 1, g_glm, "sm_100a") == 256
    # the floor's bands at chunk 4096 (one slice): 2049-2304 (73 %) and 3585-3840 (81 %) take the 256-wide form, 5121-5376
    # (85 %) and 6657-6912 (88 %) keep the wide tile; SM103 has no floor
    for rows, tile in ((2200, 256), (3700, 256), (3791, 256), (5200, 512), (6800, 512)):
        assert cake_backend.dx_tile_rule(rows, 6144, 148, 1, g_glm, "sm_100a") == tile
        assert cake_backend.dx_tile_rule(rows, 6144, 148, 1, g_glm, "sm_103a") == 512
    assert (
        cake_backend.dx_tile_rule(3591, 6144, 148, 2, g_glm, "sm_100a") == 512
    )  # two slices: 360 items / 5 waves = 97 %
    assert (
        cake_backend.dx_tile_rule(3591, 6144, 148, 1, g_glm, "sm_103a") == 512
    )  # no floor on SM103
    assert cake_backend.dx_tile_rule(3591, 6144, 148, 1, g_glm, None) == 512
    assert (
        cake_backend.dx_tile_rule(3591, 7168, 148, 1, g_wide, "sm_100a") == 512
    )  # the floor is GLM-class only
    assert (
        cake_backend.dx_tile_rule(1, 6144, 148, 1, g_glm, "sm_103a") == 256
    )  # the wave-fill rule is unchanged on both architectures
    for arch in ("sm_100a", "sm_103a"):
        # the dX ring rule: the 512-wide tile at the default geometry's long chunks, both architectures
        assert cake_backend.dx_stages_variant(4097, 6144, None, arch) == 3
        assert cake_backend.dx_stages_variant(8192, 6144, 512, arch) == 3
        assert cake_backend.dx_stages_variant(4096, 6144, None, arch) is None
        assert cake_backend.dx_stages_variant(8192, 6144, 256, arch) is None
        assert cake_backend.dx_stages_variant(8192, 7168, None, arch) is None
    assert cake_backend.dx_stages_variant(8192, 6144, None, None) is None
    # the dW block rule: both architectures, the C 4096 chunk and its tails of the default geometry
    for arch in ("sm_100a", "sm_103a"):
        assert cake_backend.dw_block_variant(4096, 6144, arch) == 12
        assert cake_backend.dw_block_variant(3943, 6144, arch) == 12
        assert cake_backend.dw_block_variant(2049, 6144, arch) == 12
        assert cake_backend.dw_block_variant(2048, 6144, arch) is None
        assert cake_backend.dw_block_variant(4097, 6144, arch) is None
        assert cake_backend.dw_block_variant(4096, 7168, arch) is None
        assert cake_backend._dw_block_group(arch, 6144) == 12
        assert (
            cake_backend._dw_block_group(arch, 6912) is None
        )  # 27 column tiles: the block would not tile them
    assert cake_backend.dw_block_variant(4096, 6144, None) is None
    for arch in ("sm_100a", "sm_103a"):
        for rows in (3841, 3884, 3943, 4021, 4096):
            assert (
                cake_backend.raster_variant(6144, rows, arch)
                == (
                    16,
                    None,
                )
            )  # the default geometry at 31+ row tiles up to 4096 rows: the logits raster alone
        for rows in (1, 1895, 2048, 3591, 3791, 3840):
            assert (
                cake_backend.raster_variant(6144, rows, arch) is None
            )  # up to 30 row tiles: the default rasters
        assert (
            cake_backend.raster_variant(7168, 2048, arch) is None
        )  # one row short of the rule
        assert cake_backend.raster_variant(7168, 2049, arch) == (16, 32)
        assert cake_backend.raster_variant(8192, 4096, arch) == (16, 32)
        assert cake_backend.raster_variant(8192, 8192, arch) == (
            16,
            32,
        )  # the wide rule owns H >= 7168 at every length
    # the default geometry's long chunks (more than 4096 rows): both rasters, both architectures
    for rows in (4097, 8039, 8117, 8192, 16231, 65536):
        for arch in ("sm_100a", "sm_103a"):
            assert cake_backend.raster_variant(6144, rows, arch) == (16, 16)
            assert cake_backend.raster_variant(6912, rows, arch) == (
                16,
                16,
            )  # every H below 7168
    for arch in ("sm_100a", "sm_103a"):
        assert (
            cake_backend.raster_variant(6144, 4096, arch)
            == (
                16,
                None,
            )
        )  # one row short of the long-chunk weight-gradient rule: the logits raster alone
        assert cake_backend.raster_variant(6912, 4096, arch) == (16, None)
        assert cake_backend.raster_variant(6912, 3840, arch) is None
    assert cake_backend.raster_variant(8192, 4096, None) is None
    assert cake_backend.raster_variant(8192, 4096, "sm_90a") is None
    assert cake_backend.raster_variant(6144, 8192, "sm_90a") is None
    assert cake_backend._dw_raster_groups("sm_103a", 6144) == (None, 16)
    assert cake_backend._dw_raster_groups("sm_100a", 6144) == (None, 16)
    assert cake_backend._dw_raster_groups("sm_103a", 7168) == (None, 32)
    assert cake_backend._dw_raster_groups("sm_103a", None) == (
        None,
    ) and cake_backend._dw_raster_groups(None, 6144) == (None,)
    g = Geometry.from_record(None)
    rule = cake_backend.dx_tile_rule
    assert (
        rule(4096, 6144, 148, 1, g) == 512
    )  # 16 row pairs x 12 = 192 wide items >= 74 SM pairs
    assert rule(4096, 6144, 148, 4, g) == 512
    assert rule(3943, 6144, 148, 1, g) == 512 and rule(1895, 6144, 148, 1, g) == 512
    assert rule(128, 6144, 148, 3, g) == 256  # 1 x 12 x 3 = 36 items < 74
    assert rule(1, 6144, 148, 4, g) == 256  # 48 < 74
    assert rule(1, 8192, 160, 4, g) == 256  # 1 x 16 x 4 = 64 < 80
    assert rule(1024, 8192, 160, 1, g) == 256  # 4 x 16 = 64 < 80
    assert rule(2048, 8192, 160, 1, g) == 512  # 8 x 16 = 128 >= 80
    assert rule(4096, 7168, 148, 1, g) == 512  # 7168 = 14 x 512
    assert rule(4096, 6400, 148, 1, g) == 256  # H % 512 != 0
    assert rule(4096, 6144, 1, 1, g) == 512  # one SM: the wide tile always fills


def _problem(T, H, V, C, objective="ce", entry="loss", dtype=torch.bfloat16):
    return cake_backend.Problem(
        num_rows=T,
        hidden=H,
        vocab=V,
        chunk=C,
        ld_x=H,
        x_copy=False,
        objective=objective,
        loss_div=1.0 if objective == "ce" else None,
        grad_weight_dtype=dtype,
        entry=entry,
    )


def test_make_plan_selects_the_rule_variants_per_chunk():
    g = Geometry.from_record({"geometry": {"hidden": 7168, "vocab": 129280}})
    plan = cake_backend.make_plan(
        _problem(16231, 7168, 129280, 4096),
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
    )
    assert plan.chunks == ((0, 4096), (4096, 4096), (8192, 4096), (12288, 3943))
    for i, (_, rows_c) in enumerate(plan.chunks):
        v = plan.variants_of(i)
        assert (v.logits_group_m, v.dw_group_m, v.dw_epi_store) == (16, 32, None)
        assert plan.logits_stage(i) == "gemm_logits_g16"
        assert plan.logits_stage(i, stats=False) == "gemm_logits_nostats_g16"
        k = recommended_k_slices(rows_c, 7168, 148, 4, g, 74)
        tile = cake_backend.dx_tile_rule(rows_c, 7168, 148, k, g)
        assert plan.dx_slices_of(i) == k
        assert plan.dx_stage_of(i) == cake_backend.stage_variant(
            "gemm_dx", k_slices=k, tile_n=tile
        )
    assert all(plan.dw_acc_stage(i) == "gemm_dw_acc_g32" for i in range(3))
    assert plan.dw_cast_stage == "gemm_dw_cast_bf16_g32"
    assert "gemm_logits" not in plan.stages and "gemm_dw_acc" not in plan.stages
    assert {
        "gemm_logits_g16",
        "gemm_dw_acc_g32",
        "gemm_dw_cast_bf16_g32",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "scale_cast_bf16",
    } <= set(plan.stages)
    assert list(plan.stages) == [s for s in cake_jit.STAGES if s in plan.stages]
    first = [
        k[0]
        for k in cake_backend.forward_keys(plan)
        if k[1] == 0 and k[0].startswith("gemm")
    ]
    assert first == ["gemm_logits_g16", plan.dx_stage_of(0), "gemm_dw_acc_g32"]
    assert cake_backend.cast_keys(plan)[-1] == ("gemm_dw_cast_bf16_g32", 3)
    logprob = cake_backend.make_plan(
        _problem(16231, 7168, 129280, 4096, entry="logprob"),
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
    )
    recompute = [
        k[0]
        for k in cake_backend.recompute_keys(logprob)
        if k[1] == 3 and k[0].startswith("gemm")
    ]
    assert recompute == [
        "gemm_logits_nostats_g16",
        logprob.dx_stage_of(3),
        "gemm_dw_cast_bf16_g32",
    ]
    # the default geometry: the blocked weight-gradient raster at the 4096-row chunk, the default raster elsewhere; a
    # one-row tail takes the K-sliced narrow dX tile
    g0 = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    tail = cake_backend.make_plan(
        _problem(4097, 6144, 154880, 4096),
        need_dx=True,
        need_dw=True,
        geometry=g0,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
    )
    assert tail.chunks == ((0, 4096), (4096, 1))
    assert (
        tail.logits_stage(0) == "gemm_logits_g16"  # 32 row tiles: the logits raster
        and tail.logits_stage(1) == "gemm_logits"
        and tail.dw_acc_stage(0) == "gemm_dw_acc_gn12"
    )
    assert tail.dw_cast_stage == "gemm_dw_cast_bf16"
    assert (
        tail.dx_slices_of(0) == 4 and tail.dx_stage_of(0) == "gemm_dx_s4"
    )  # 16 x 12 x 4 = 768 wide items
    assert (
        tail.dx_slices_of(1) == 3 and tail.dx_stage_of(1) == "gemm_dx_s3_tn256"
    )  # 1 x 12 x 3 = 36 < 74 pairs
    assert tail.stages == (
        "gemm_logits",
        "gemm_logits_g16",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx_s4",
        "gemm_dx_s3_tn256",
        "gemm_dw_acc_gn12",
        "gemm_dw_cast_bf16",
        "scale_cast_bf16",
    )
    # SM103 at a wide geometry: the wide weight-gradient raster of the accumulate (no epilogue rule)
    big = cake_backend.make_plan(
        _problem(16384, 7168, 129280, 8192),
        need_dx=False,
        need_dw=True,
        geometry=g,
        num_sms=160,
        arch="sm_103a",
    )
    assert big.chunks == ((0, 8192), (8192, 8192))
    assert (
        big.dw_acc_stage(0) == "gemm_dw_acc_g32"
        and big.dw_cast_stage == "scale_cast_bf16"
    )
    # the default geometry's long chunks (more than 4096 rows): the taller weight-gradient raster (group_m 16) in the
    # accumulate and in the fused cast; the one-row tail and chunks of up to 4096 rows keep the defaults
    long_chunks = cake_backend.make_plan(
        _problem(8193, 6144, 154880, 8192),
        need_dx=False,
        need_dw=True,
        geometry=g0,
        num_sms=160,
        arch="sm_103a",
    )
    assert long_chunks.chunks == ((0, 8192), (8192, 1))
    assert (
        long_chunks.dw_acc_stage(0) == "gemm_dw_acc_g16"
        and long_chunks.dw_acc_stage(1) == "gemm_dw_acc"
    )
    assert (
        long_chunks.logits_stage(0) == "gemm_logits_g16"
        and long_chunks.logits_stage(1) == "gemm_logits"
    )  # the 8192-row chunk takes the logits raster (64 row tiles), the 1-row tail the default
    assert (
        long_chunks.variants_of(0).dw_group_m,
        long_chunks.variants_of(0).dw_epi_store,
    ) == (16, None)
    assert (
        cake_backend.make_plan(
            _problem(8192, 6144, 154880, 4096),
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=160,
            arch="sm_103a",
        ).dw_acc_stage(1)
        == "gemm_dw_acc_gn12"
    )  # SM103 at the C 4096 chunk: the 2-D blocked raster
    assert (
        cake_backend.make_plan(
            _problem(8192, 6144, 154880, 4096),
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=148,
            arch="sm_100a",
        ).dw_acc_stage(1)
        == "gemm_dw_acc_gn12"
    )  # SM100 as well: the 2-D blocked raster at the C 4096 chunk
    assert (
        cake_backend.make_plan(
            _problem(8193, 6144, 154880, 8192),
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=148,
            arch="sm_100a",
        ).dw_acc_stage(0)
        == "gemm_dw_acc_g16"
    )  # SM100 too: the long-chunk rule is the same on both architectures
    one = cake_backend.make_plan(
        _problem(8192, 6144, 154880, 8192),
        need_dx=False,
        need_dw=True,
        geometry=g0,
        num_sms=160,
        fuse_dw_cast=True,
        arch="sm_103a",
    )
    assert one.stages == (
        "gemm_logits_g16",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dw_cast_bf16_g16",
    )
    assert one.dw_cast_stage == "gemm_dw_cast_bf16_g16"
    deferred_tail = cake_backend.make_plan(
        _problem(16231, 6144, 154880, 8192),
        need_dx=False,
        need_dw=True,
        geometry=g0,
        num_sms=160,
        fuse_dw_cast=True,
        arch="sm_103a",
    )
    assert deferred_tail.chunks == ((0, 8192), (8192, 8039))
    assert (
        deferred_tail.dw_acc_stage(0) == "gemm_dw_acc_g16"
        and deferred_tail.dw_cast_stage == "gemm_dw_cast_bf16_g16"
    )
    assert (
        cake_backend.make_plan(
            _problem(8192, 6144, 154880, 8192),
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=148,
            fuse_dw_cast=True,
            arch="sm_100a",
        ).dw_cast_stage
        == "gemm_dw_cast_bf16_g16"
    )  # one chunk of 8192 rows: the long-chunk raster in the fused cast on SM100 as well
    # the default geometry's long chunks with dX: the 3-deep operand ring of the 512-wide tile, both architectures
    for arch, sms, resident in (("sm_103a", 160, 80), ("sm_100a", 148, 74)):
        ring = cake_backend.make_plan(
            _problem(16231, 6144, 154880, 8192),
            need_dx=True,
            need_dw=True,
            geometry=g0,
            dx_max_slices=4,
            num_sms=sms,
            dx_resident=resident,
            fuse_dw_cast=True,
            arch=arch,
        )
        assert ring.chunks == ((0, 8192), (8192, 8039))
        for i, (_, rows_c) in enumerate(ring.chunks):
            k = recommended_k_slices(rows_c, 6144, sms, 4, g0, resident)
            tile = cake_backend.dx_tile_rule(rows_c, 6144, sms, k, g0, arch)
            assert ring.dx_stage_of(i) == cake_backend.stage_variant(
                "gemm_dx", k_slices=k, tile_n=tile, stages=3 if tile == 512 else None
            )
            if tile == 512:
                assert ring.dx_stage_of(i).endswith("_st3")
                assert ring.variants_of(i).dx_stages == 3
            else:  # the tile rule's wave-efficiency floor (SM100): the 256-wide form keeps its ring depth
                assert ring.dx_stage_of(i).endswith("_tn256")
                assert ring.variants_of(i).dx_stages is None
            assert (
                ring.variants_of(i).dw_group_n is None
            )  # above 4096 rows: the raster rule, not the block
        assert ring.dw_acc_stage(0) == "gemm_dw_acc_g16"
        pinned_ring = cake_backend.make_plan(
            _problem(16231, 6144, 154880, 8192),
            gemm_tuning={"dx": {"stages": 4}},
            need_dx=True,
            need_dw=True,
            geometry=g0,
            dx_max_slices=4,
            num_sms=sms,
            dx_resident=resident,
            fuse_dw_cast=True,
            arch=arch,
        )
        assert all(not s.endswith("_st3") for s in pinned_ring.stages)
    # the default geometry's C 4096 chunks on both architectures: the blocked weight-gradient raster in the accumulate
    # and in the fp32 fused cast; the bf16 cast keeps the 1-D raster; a pinned ``group_n`` (0 or null) keeps the default
    for dtype, cast in (
        (torch.float32, "gemm_dw_cast_f32_gn12"),
        (torch.bfloat16, "gemm_dw_cast_bf16"),
    ):
        for arch, sms, resident in (("sm_100a", 148, 74), ("sm_103a", 160, 80)):
            blocked = cake_backend.make_plan(
                _problem(16231, 6144, 154880, 4096, dtype=dtype),
                need_dx=True,
                need_dw=True,
                geometry=g0,
                dx_max_slices=4,
                num_sms=sms,
                dx_resident=resident,
                fuse_dw_cast=True,
                arch=arch,
            )
            assert blocked.chunks == (
                (0, 4096),
                (4096, 4096),
                (8192, 4096),
                (12288, 3943),
            )
            assert all(blocked.dw_acc_stage(i) == "gemm_dw_acc_gn12" for i in range(4))
            assert all(blocked.variants_of(i).dw_group_n == 12 for i in range(4))
            assert all(
                not s.endswith("_st3") for s in blocked.stages
            )  # up to 4096 rows: the table ring
            assert blocked.dw_cast_stage == cast
    # the T 65031 tail (3591 rows) at chunk 4096: one slice on both architectures; SM100 takes the 256-wide cluster form
    # (the wave-efficiency floor), SM103 keeps the wide tile; a pinned tile_n=512 wins
    for arch, sms, resident, tail_stage in (
        ("sm_100a", 148, 74, "gemm_dx_tn256"),
        ("sm_103a", 160, 80, "gemm_dx"),
    ):
        tail_plan = cake_backend.make_plan(
            _problem(65031, 6144, 154880, 4096),
            need_dx=True,
            need_dw=True,
            geometry=g0,
            dx_max_slices=4,
            num_sms=sms,
            dx_resident=resident,
            fuse_dw_cast=True,
            arch=arch,
        )
        assert tail_plan.chunks[-1] == (61440, 3591)
        last = tail_plan.num_chunks - 1
        if tail_plan.dx_slices_of(last) == 1:
            assert tail_plan.dx_stage_of(last) == tail_stage
        else:  # the slice rule took more than one slice here: no floor, the wide tile
            assert not tail_plan.dx_stage_of(last).endswith("_tn256")
        pinned_tail = cake_backend.make_plan(
            _problem(65031, 6144, 154880, 4096),
            gemm_tuning={"dx": {"tile_n": 512}},
            need_dx=True,
            need_dw=True,
            geometry=g0,
            dx_max_slices=4,
            num_sms=sms,
            dx_resident=resident,
            fuse_dw_cast=True,
            arch=arch,
        )
        assert not pinned_tail.dx_stage_of(last).endswith("_tn256")
    for pin in ({"dw": {"group_n": 0}}, {"dw": {"group_n": None}}):
        unblocked = cake_backend.make_plan(
            _problem(16231, 6144, 154880, 4096, dtype=torch.float32),
            gemm_tuning=pin,
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=160,
            fuse_dw_cast=True,
            arch="sm_103a",
        )
        assert unblocked.dw_acc_stage(0) == "gemm_dw_acc"
        assert unblocked.dw_cast_stage == "gemm_dw_cast_f32"
    # the reference path (no architecture) plans the base stages
    ref = cake_backend.make_plan(
        _problem(16231, 7168, 129280, 4096),
        need_dx=True,
        need_dw=True,
        geometry=g,
        num_sms=148,
    )
    assert ref.variants == () and set(ref.stages) <= set(cake_jit.BASE_STAGES)


def test_make_plan_explicit_gemm_tuning_wins_over_the_rules():
    g = Geometry.from_record({"geometry": {"hidden": 7168, "vocab": 129280}})
    kw = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
    )
    problem = _problem(16231, 7168, 129280, 4096)
    pinned = cake_backend.make_plan(
        problem,
        gemm_tuning={
            "logits": {"group_m": None},
            "dw": {"group_m": None},
            "dx": {"tile_n": 256, "k_slices": 2},
        },
        **kw,
    )
    assert (
        pinned.logits_stage(0) == "gemm_logits"
        and pinned.dw_acc_stage(0) == "gemm_dw_acc"
    )
    assert pinned.dw_cast_stage == "gemm_dw_cast_bf16"
    assert all(pinned.dx_stage_of(i) == "gemm_dx_s2_tn256" for i in range(4))
    partial = cake_backend.make_plan(problem, gemm_tuning={"dx": {"k_slices": 2}}, **kw)
    assert (
        partial.logits_stage(0) == "gemm_logits_g16"
    )  # the rules still decide the knobs not named
    assert partial.dw_acc_stage(0) == "gemm_dw_acc_g32"
    assert all(
        s.startswith("gemm_dx_s2") for s in partial.stages if s.startswith("gemm_dx")
    )
    # a present ``null`` pins the default form -- one dX slice, the base ``gemm_dx`` stage -- like every other
    # pinned knob; an absent knob keeps the slice rule
    one = cake_backend.make_plan(problem, gemm_tuning={"dx": {"k_slices": None}}, **kw)
    ruled = cake_backend.make_plan(problem, gemm_tuning={"dx": {}}, **kw)
    unpinned = cake_backend.make_plan(problem, **kw)
    assert len(one.chunks) == 4
    for i, (_, rows_c) in enumerate(one.chunks):
        assert one.dx_slices_of(i) == 1
        assert one.dx_stage_of(i) == cake_backend.stage_variant(
            "gemm_dx",
            k_slices=1,
            tile_n=cake_backend.dx_tile_rule(rows_c, 7168, 148, 1, g, "sm_100a"),
        )
        assert ruled.dx_slices_of(i) == unpinned.dx_slices_of(i)
    # the default geometry's long chunks: a pinned knob keeps the other rule's output
    g0 = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    long_kw = dict(
        need_dx=False,
        need_dw=True,
        geometry=g0,
        num_sms=160,
        fuse_dw_cast=True,
        arch="sm_103a",
    )
    long_problem = _problem(8193, 6144, 154880, 8192)
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"group_m": None}}, **long_kw
        ).dw_acc_stage(0)
        == "gemm_dw_acc"
    )
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"epi_store": None}}, **long_kw
        ).dw_acc_stage(0)
        == "gemm_dw_acc_g16"
    )
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"group_m": None}}, **long_kw
        ).dw_cast_stage
        == "gemm_dw_cast_bf16"
    )
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"group_m": 32}}, **long_kw
        ).dw_acc_stage(0)
        == "gemm_dw_acc_g32"
    )
    wide = cake_backend.make_plan(problem, gemm_tuning={"dx": {"tile_n": 512}}, **kw)
    assert all(not s.endswith("_tn256") for s in wide.stages)
    with pytest.raises(ValueError, match="k_slices"):
        cake_backend.make_plan(problem, gemm_tuning={"dx": {"k_slices": 5}}, **kw)
    for bad in (
        {"gemm_logits": {"group_m": 16}},
        {"dx": {"group_m": 16}},
        {"dw": "tma"},
        ["dx"],
    ):
        with pytest.raises(ValueError):
            cake_backend._resolve_gemm_tuning(bad)
    assert cake_backend._resolve_gemm_tuning({"logits": {"group_m": "16"}}) == {
        "logits": {"group_m": 16}
    }
    assert cake_backend._resolve_gemm_tuning(
        {"dw": {"epi_store": "tma", "group_m": None}}
    ) == {"dw": {"epi_store": "tma", "group_m": None}}


def test_gemm_tuning_default_env(monkeypatch):
    monkeypatch.delenv(cake_backend.GEMM_TUNING_ENV, raising=False)
    assert (
        cake_backend.gemm_tuning_default() == {}
        and cake_backend._resolve_gemm_tuning(None) == {}
    )
    monkeypatch.setenv(
        cake_backend.GEMM_TUNING_ENV,
        '{"logits": {"group_m": 16}, "dx": {"tile_n": 256}}',
    )
    assert cake_backend.gemm_tuning_default() == {
        "logits": {"group_m": 16},
        "dx": {"tile_n": 256},
    }
    monkeypatch.setenv(cake_backend.GEMM_TUNING_ENV, "not json")
    with pytest.raises(ValueError, match="JSON"):
        cake_backend.gemm_tuning_default()
    monkeypatch.setenv(cake_backend.GEMM_TUNING_ENV, '{"dx": {"tile_n": 256}}')
    kw = _valid_call(T=8)
    X, W, labels = kw["X"], kw["W"], kw["labels"]
    common = dict(
        objective="ce",
        loss_div=1.0,
        infer_logp=None,
        loss_weights=None,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.bfloat16,
        entry="loss",
    )
    env_key = forward_binding_key(
        X, W, labels, **common
    )  # the environment's knobs are part of the binding
    assert env_key == forward_binding_key(
        X, W, labels, gemm_tuning={"dx": {"tile_n": 256}}, **common
    )
    assert env_key != forward_binding_key(X, W, labels, gemm_tuning={}, **common)
    # the reference path with an explicit knob plans the pinned variant; the reference engine runs its base stage
    inp = _host_inputs(37)
    common = dict(
        objective="ce", loss_div=inp.loss_div, backend="reference", chunk_size=16
    )
    narrow = prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, gemm_tuning={"dx": {"tile_n": 256}}, **common
    )
    assert "gemm_dx_tn256" in narrow.stages and "gemm_dx" not in narrow.stages
    plain = prepare_lm_head_loss(inp.X, inp.W, inp.labels, gemm_tuning={}, **common)
    assert "gemm_dx" in plain.stages and plain.plan.variants == ()
    narrow.step(torch.tensor(2.0))
    plain.step(torch.tensor(2.0))
    assert torch.equal(narrow.dx_out, plain.dx_out) and torch.equal(
        narrow.loss, plain.loss
    )
    with pytest.raises(ValueError, match="plans"):
        stage_values("gemm_dx", narrow.tensors, narrow.plan, 0)


def test_select_module_by_geometry(monkeypatch):
    def rec(arch, H, V):
        return {
            "arch": arch,
            "abi": ABI_CONTRACT,
            "geometry": {"hidden": H, "vocab": V},
            "stages": [],
        }

    monkeypatch.setattr(
        cake_jit,
        "MODULES",
        {
            "cake_lm_head_loss_sm_100a": rec("sm_100a", 6144, 154880),
            "cake_lm_head_loss_sm_100a_h7168_v129280": rec("sm_100a", 7168, 129280),
            "cake_lm_head_loss_sm_103a_h8192_v128256": rec("sm_103a", 8192, 128256),
        },
    )
    assert (
        cake_jit.select_module("sm_100a") == "cake_lm_head_loss_sm_100a"
    )  # the default-geometry record
    assert (
        cake_jit.select_module("sm_100a", 6144, 154880) == "cake_lm_head_loss_sm_100a"
    )
    assert (
        cake_jit.select_module("sm_100a", 7168, 129280)
        == "cake_lm_head_loss_sm_100a_h7168_v129280"
    )
    assert (
        cake_jit.select_module("sm_103a") == "cake_lm_head_loss_sm_103a_h8192_v128256"
    )  # the sole record
    assert (
        cake_jit.select_module("sm_103a", hidden=8192)
        == "cake_lm_head_loss_sm_103a_h8192_v128256"
    )
    with pytest.raises(ValueError, match="specialized"):
        cake_jit.select_module("sm_100a", 8192, 128256)
    with pytest.raises(ValueError, match="specialized"):
        cake_jit.select_module("sm_103a", 6144, 154880)
    with pytest.raises(NotImplementedError):
        cake_jit.select_module("sm_90a")
    monkeypatch.setattr(
        cake_jit,
        "MODULES",
        {"a": rec("sm_100a", 6144, 154880), "b": rec("sm_100a", 7168, 129280)},
    )
    with pytest.raises(NotImplementedError, match="default-geometry"):
        cake_jit.select_module("sm_100a")
    assert cake_jit.select_module("sm_100a", 7168, 129280) == "b"
    monkeypatch.setattr(
        cake_jit,
        "MODULES",
        {"a": rec("sm_100a", None, None), "b": rec("sm_100a", 7168, 129280)},
    )
    with pytest.raises(NotImplementedError, match="more than one"):
        cake_jit.select_module(
            "sm_100a", 7168, 129280
        )  # an unpinned record accepts every geometry
    assert cake_jit.record_geometry(rec("sm_100a", None, 4)) == (None, 4)


def test_reachable_variants_complete_the_rule_closure():
    bases = (
        set(stages_for_entry("loss"))
        | set(stages_for_entry("loss", grad_weight_dtype=torch.float32))
        | set(stages_for_entry("logprob"))
    )
    glm = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    wide = Geometry.from_record({"geometry": {"hidden": 7168, "vocab": 129280}})
    rv = cake_backend.reachable_variants
    dx_all = {
        "gemm_dx",
        "gemm_dx_s2",
        "gemm_dx_s3",
        "gemm_dx_s4",
        "gemm_dx_tn256",
        "gemm_dx_s2_tn256",
        "gemm_dx_s3_tn256",
        "gemm_dx_s4_tn256",
    }
    glm_dw = {
        "gemm_dw_acc",
        "gemm_dw_acc_g16",
        "gemm_dw_cast_bf16_g16",
        "gemm_dw_cast_f32_g16",
    }
    glm_dx_st3 = {"gemm_dx_st3", "gemm_dx_s2_st3", "gemm_dx_s3_st3", "gemm_dx_s4_st3"}
    glm_dw_gn12 = {"gemm_dw_acc_gn12", "gemm_dw_cast_f32_gn12"}
    glm_logits = {
        "gemm_logits_g16",
        "gemm_logits_nostats_g16",
    }  # the 31-row-tile logits raster
    assert (
        rv(bases, "sm_100a", glm, 4)
        == dx_all | glm_dx_st3 | glm_dw | glm_dw_gn12 | glm_logits
    )
    # the default geometry: the long-chunk raster of the weight-gradient GEMMs, the dX ring rule and the 2-D blocked
    # weight-gradient raster on both architectures (no epilogue rule)
    assert rv(bases, "sm_103a", glm, 4) == rv(bases, "sm_100a", glm, 4)
    assert rv(bases, "sm_100a", wide, 4) - rv(bases, "sm_100a", glm, 4) == {
        "gemm_dw_acc_g32",
        "gemm_dw_cast_bf16_g32",
        "gemm_dw_cast_f32_g32",
    }
    assert glm_logits <= rv(
        bases, "sm_100a", wide, 4
    )  # the same logits raster forms on both geometry classes
    assert rv(bases, "sm_103a", wide, 4) == rv(bases, "sm_100a", wide, 4)
    assert (
        rv(bases, "sm_100a", glm, 2)
        == {
            "gemm_dx",
            "gemm_dx_s2",
            "gemm_dx_tn256",
            "gemm_dx_s2_tn256",
            "gemm_dx_st3",
            "gemm_dx_s2_st3",
        }
        | glm_dw
        | glm_dw_gn12
        | glm_logits
    )
    assert rv({"row_grad", "scale_cast_bf16"}, "sm_103a", wide, 4) == set()
    # every name of STAGES is a base stage or a variant some record can reach
    assert set(cake_jit.STAGES) == set(cake_jit.BASE_STAGES) | rv(
        bases, "sm_103a", wide, 4
    ) | rv(bases, "sm_100a", wide, 4) | rv(bases, "sm_103a", glm, 4)
    assert len(cake_jit.STAGES) == 34
    unpinned = Geometry.from_record(None)
    assert rv(bases, "sm_100a", unpinned, 4) == dx_all | {
        "gemm_dw_acc"
    }  # no pinned H: no raster variants required


def test_cluster_resident_rule():
    assert (
        cluster_resident(None, 2, 148) == 74
        and cluster_resident(None, 2, 152) == 76
        and cluster_resident(None, 1, 5) == 5
    )
    with pytest.raises(ValueError):  # wider clusters take the device's occupancy answer
        cluster_resident(None, 4, 148)
    with pytest.raises(ValueError):
        cluster_resident(torch.device("cpu"), 4, 148)


def test_wave_efficiency_uses_the_dx_cluster():
    g4 = Geometry.from_record({"geometry": {"dx_cluster_ctas": 4}})
    g2 = Geometry.from_record(None)
    # 4096 rows: 32 row tiles = 8 four-CTA clusters x 24 column tiles = 192 items over 33 co-resident clusters
    assert wave_efficiency(4096, 6144, 148, 1, g4, resident=33) == pytest.approx(
        192 / (6 * 33)
    )
    assert wave_efficiency(4096, 6144, 148, 1, g2) == pytest.approx(
        (16 * 24) / (6 * 74)
    )
    assert 1 <= recommended_k_slices(4096, 6144, 148, 4, g4, resident=33) <= 4
    assert recommended_k_slices(4096, 6144, 148, 4, g2) == recommended_k_slices(
        4096, 6144, 148, 4, g2, resident=74
    )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the cluster occupancy probe needs a CUDA device",
)
def test_device_cluster_resident_probe():
    device = torch.device("cuda", torch.cuda.current_device())
    if torch.cuda.get_device_capability(device) not in SUPPORTED_COMPUTE_CAPABILITIES:
        pytest.skip(
            "thread-block clusters of four CTAs are probed on the supported parts only"
        )
    sms = int(torch.cuda.get_device_properties(device).multi_processor_count)
    four = cluster_resident(device, 4, sms)
    assert 1 <= four <= sms // 4
    assert cluster_resident(device, 4, sms) == four  # cached per (device, width)
    assert cluster_resident(device, 2, sms) == sms // 2


def test_stage_values_names():
    T, C = 37, 16
    inp = _host_inputs(T)
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=C,
        backend="reference",
        fuse_dw_cast=False,
    )
    plan, t = runner.plan, runner.tensors
    assert plan.chunks == ((0, 16), (16, 16), (32, 5)) and plan.chunks[-1][1] == T % C
    assert runner.stages == stages_for_entry("loss", fuse_dw_cast=False)
    assert runner.forward_order[:2] == (("gemm_logits", 0), ("row_finalize", 0))
    assert runner.backward_order == (
        ("scale_cast_bf16", "dx"),
        ("scale_cast_bf16", "dw"),
    )
    for index, (row0, rows_c) in enumerate(plan.chunks):
        v = stage_values("gemm_logits", t, plan, index)
        assert (v["rows_c"], v["row0"], v["T"], v["H"], v["V"], v["chunk"]) == (
            rows_c,
            row0,
            T,
            H_HOST,
            V_HOST,
            C,
        )
        assert (v["first_chunk"], v["last_chunk"]) == (
            0,
            int(index == len(plan.chunks) - 1),
        )
        assert (
            v["mode"] == MODE_CE
            and v["loss_div"] == inp.loss_div
            and v["num_tiles"] == V_HOST // 256
        )
        assert v["A"].data_ptr() == inp.X[row0].data_ptr() and tuple(v["A"].shape) == (
            rows_c,
            H_HOST,
        )
        assert v["B"] is inp.W and v["STATS_OUT"] is t["stats"]
        assert v["C"].data_ptr() == t["logits"].data_ptr() and tuple(v["C"].shape) == (
            rows_c,
            V_HOST,
        )  # the chunk's rows
        assert (
            v["M"] == rows_c
            and v["m_tiles"] % 2 == 0
            and v["m_tiles"] >= -(-rows_c // 128)
            and v["k_iters"] == 1
        )
        dx = stage_values("gemm_dx", t, plan, index)
        assert dx["A"].data_ptr() == t["logits"].data_ptr() and tuple(
            dx["A"].shape
        ) == (rows_c, V_HOST)
        assert (
            dx["C"].data_ptr() == t["dx_acc"][row0].data_ptr()
            and dx["M"] == rows_c
            and dx["first_chunk"] == 1
        )  # stores its rows
        dw = stage_values("gemm_dw_acc", t, plan, index)
        assert (
            dw["M"] == V_HOST
            and dw["k_iters"] == -(-rows_c // 64)
            and dw["C"] is t["dw_acc"]
        )
        assert dw["first_chunk"] == int(index == 0)
        assert (
            dw["A"].data_ptr() == t["logits"].data_ptr()
            and dw["B"].data_ptr() == inp.X[row0].data_ptr()
        )
        rg = stage_values("row_grad", t, plan, index)
        assert (
            rg["d_off"] == 0
            and rg["d"] is t["d"]
            and rg["z"] is t["logits"]
            and rg["lse"] is t["lse"]
        )
        rf = stage_values("row_finalize", t, plan, index)
        assert (
            rf["infer_logp"] is t["d"]
            and rf["loss_weights"] is t["d"]
            and rf["d_in"] is t["d"]
        )  # CE reads none of them
        assert rf["logp"] is t["logp"] and rf["term"] is t["term"]
        lr = stage_values("loss_reduce", t, plan, index)
        assert lr["loss_acc"] is t["loss_acc"] and lr["loss_out"] is t["loss"]
    cast = stage_values("scale_cast_bf16", t, plan, "dw")
    assert (
        cast["acc"].data_ptr() == t["dw_acc"].data_ptr()
        and cast["out"].data_ptr() == t["dw_out"].data_ptr()
    )
    assert cast["num_vecs"] == V_HOST * H_HOST // 8 and cast["g"] is t["grad_scale"]
    with pytest.raises(ValueError):
        stage_values("not_a_stage", t, plan, 0)
    # the log-probability entry reads the caller's [T] dlogp at d[row0 + r]
    lp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        entry="logprob",
        backend="reference",
        fuse_dw_cast=False,
    )
    assert lp.stages == stages_for_entry("logprob", fuse_dw_cast=False) and (
        lp.dlogp is lp.tensors["d_in"]
    )
    assert lp.forward_order == tuple(
        k for i in range(3) for k in (("gemm_logits", i), ("row_finalize", i))
    )
    assert lp.backward_order[0] == ("gemm_logits_nostats", 0) and lp.backward_order[
        -1
    ] == ("scale_cast_bf16", "dw")
    for index, (row0, _) in enumerate(lp.plan.chunks):
        rg = stage_values("row_grad", lp.tensors, lp.plan, index)
        assert rg["d_off"] == row0 and rg["d"] is lp.tensors["d_in"]
        assert (
            stage_values("row_finalize", lp.tensors, lp.plan, index)["mode"]
            == MODE_NONE
        )
    with pytest.raises(ValueError, match="T > 0"):
        prepare_lm_head_loss(
            inp.X[:0],
            inp.W,
            inp.labels[:0],
            objective="ce",
            loss_div=1.0,
            backend="reference",
        )


_ROW_GRAD_PLAN = [
    ["buffer", "z"],
    ["buffer", "labels"],
    ["buffer", "lse"],
    ["buffer", "d"],
    ["parameter", "row0"],
    ["parameter", "d_off"],
    ["parameter", "V"],
    ["grid", "grid_x"],
    ["grid", "grid_y"],
    ["grid", "grid_z"],
]


def _fake_record(arg_plan):
    return {
        "arch": "sm_100a",
        "abi": ABI_CONTRACT,
        "stages": ["row_grad"],
        "row_grad": {
            "module": "fake",
            "sources": ["cake_lm_head_loss/fake/a.cu", "cake_lm_head_loss/fake/b.cu"],
            "compile_flags": [],
            "ffi_entry": "run",
            "arg_plan": arg_plan,
            "closure_sha256": "0" * 64,
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["rows_c", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
        "closure_sha256": "0" * 64,
    }


def _row_grad_values():
    return dict(
        z=torch.zeros(4, V_HOST, dtype=torch.bfloat16),
        labels=torch.zeros(4, dtype=torch.int64),
        lse=torch.zeros(4),
        d=torch.zeros(4),
        row0=0,
        d_off=0,
        V=V_HOST,
        rows_c=4,
        T=4,
        workspace=None,
        tma_descriptor_workspace=None,
    )


def test_bind_stage_fails_closed_on_unknown_argument(monkeypatch):
    name = "cake_lm_head_loss_fake"
    monkeypatch.setitem(
        cake_jit.MODULES,
        name,
        _fake_record(_ROW_GRAD_PLAN + [["parameter", "not_a_host_value"]]),
    )
    monkeypatch.setattr(
        cake_backend,
        "load_cake_lm_head_loss_module",
        lambda n, s: SimpleNamespace(run=lambda *a: None),
    )
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
    monkeypatch.setattr(
        cake_backend,
        "load_cake_lm_head_loss_module",
        lambda n, s: SimpleNamespace(run=lambda *a: calls.append(a)),
    )
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
    fresh = dict(
        z=torch.ones(4, V_HOST, dtype=torch.bfloat16),
        labels=values["labels"],
        lse=torch.ones(4),
        d=values["d"],
    )
    rebound = template.arguments_for(fresh)
    assert (
        rebound[0] is fresh["z"]
        and rebound[2] is fresh["lse"]
        and rebound[4:] == [0, 0, V_HOST, 4, 3, 1]
    )
    with pytest.raises(RuntimeError, match="'d'"):
        template.arguments_for({k: v for k, v in fresh.items() if k != "d"})
    # kernel-side spellings resolve through the contract aliases to the same host slots
    monkeypatch.setitem(
        cake_jit.MODULES,
        name,
        _fake_record(
            [
                ["buffer", "dlogits"],
                ["buffer", "targets"],
                ["parameter", "vocab"],
                ["grid", "grid_x"],
            ]
        ),
    )
    aliased = bind_stage(name, "row_grad", values, (4, 1, 1))
    assert aliased.slots == ((0, "z"), (1, "labels")) and aliased.arguments[2:] == (
        V_HOST,
        4,
    )


def test_binding_keys_cover_pointer_shape_stride_dtype_and_options():
    kw = _valid_call(T=8)
    X, W, labels = kw["X"], kw["W"], kw["labels"]
    common = dict(
        objective="ce",
        loss_div=1.0,
        infer_logp=None,
        loss_weights=None,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.bfloat16,
        entry="loss",
    )
    base = forward_binding_key(X, W, labels, **common)
    assert base[0] == "fwd"
    assert base == forward_binding_key(
        X.view_as(X), W, labels, **common
    )  # another object, same binding
    assert base != forward_binding_key(X.clone(), W, labels, **common)  # pointer
    assert base != forward_binding_key(
        X[:4], W, labels, **common
    )  # shape (same pointer, stride, dtype)
    same_ptr_other_stride = X.as_strided((4, H_HOST), (2 * H_HOST, 1))
    assert (
        same_ptr_other_stride.data_ptr() == X[:4].data_ptr()
        and same_ptr_other_stride.shape == X[:4].shape
    )
    assert forward_binding_key(X[:4], W, labels, **common) != forward_binding_key(
        same_ptr_other_stride, W, labels, **common
    )  # stride
    assert base != forward_binding_key(
        X.view(torch.float16), W, labels, **common
    )  # dtype
    assert base != forward_binding_key(X, W.clone(), labels, **common)
    assert base != forward_binding_key(X, W, labels.clone(), **common)
    for option in (
        dict(loss_div=2.0),
        dict(chunk_size=2048),
        dict(need_dx=False),
        dict(need_dw=False),
        dict(grad_weight_dtype=torch.float32),
        dict(entry="logprob"),
        dict(fuse_dw_cast=True),
        dict(gemm_tuning={"logits": {"group_m": 16}}),
    ):
        assert base != forward_binding_key(X, W, labels, **dict(common, **option)), (
            option
        )
    assert forward_binding_key(
        X, W, labels, gemm_tuning={"dx": {"tile_n": 256}}, **common
    ) != forward_binding_key(
        X, W, labels, gemm_tuning={"dx": {"tile_n": 512}}, **common
    )
    infer, weights = torch.zeros(8), torch.zeros(8)
    policy = dict(
        common,
        objective="policy",
        loss_div=None,
        infer_logp=infer,
        loss_weights=weights,
    )
    pkey = forward_binding_key(X, W, labels, **policy)
    assert pkey != base
    assert pkey != forward_binding_key(
        X, W, labels, **dict(policy, infer_logp=infer.clone())
    )
    assert pkey != forward_binding_key(
        X, W, labels, **dict(policy, loss_weights=weights.clone())
    )
    lse, dlogp = torch.zeros(8), torch.zeros(8)
    bwd = logprob_backward_binding_key(
        X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=True
    )
    assert bwd[:2] == ("bwd", "logprob") and bwd != base
    assert bwd != logprob_backward_binding_key(
        X, W, labels, lse, dlogp.clone(), chunk_size=4096, need_dx=True, need_dw=True
    )
    assert bwd != logprob_backward_binding_key(
        X, W, labels, lse.clone(), dlogp, chunk_size=4096, need_dx=True, need_dw=True
    )
    assert bwd != logprob_backward_binding_key(
        X, W, labels, lse, dlogp, chunk_size=4096, need_dx=True, need_dw=False
    )
    assert bwd != logprob_backward_binding_key(
        X, W, labels, lse, dlogp, chunk_size=2048, need_dx=True, need_dw=True
    )
    assert bwd != logprob_backward_binding_key(
        X,
        W,
        labels,
        lse,
        dlogp,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        fuse_dw_cast=True,
    )
    assert bwd != logprob_backward_binding_key(
        X,
        W,
        labels,
        lse,
        dlogp,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        gemm_tuning={"dw": {"group_m": 32}},
    )
    # the compacted row count is a label-dependent fact of the plan
    assert base != forward_binding_key(X, W, labels, valid_rows=4, **common)
    assert forward_binding_key(
        X, W, labels, valid_rows=4, **common
    ) != forward_binding_key(X, W, labels, valid_rows=5, **common)
    assert bwd != logprob_backward_binding_key(
        X,
        W,
        labels,
        lse,
        dlogp,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        valid_rows=4,
    )


def test_binding_cache_is_lru_and_bounded():
    cache = BindingCache(capacity=3)
    assert cache.enabled and len(cache) == 0

    def keys():
        return list(cache._bindings)

    for tag in ("a", "b", "c"):
        cache.remember(("fwd", tag), SimpleNamespace(owned_bytes=10))
    assert (
        keys() == [("fwd", "a"), ("fwd", "b"), ("fwd", "c")] and cache.owned_bytes == 30
    )
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
    assert run_steps(7) == (
        0,
        24,
        7,
    )  # one binding short thrashes under a cyclic pattern
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


@pytest.mark.parametrize(
    "T, C", HOST_SHAPES, ids=[f"t{t}_c{c}" for t, c in HOST_SHAPES]
)
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_reference_matches_unchunked_and_oracle(objective, T, C):
    inp = _host_inputs(T, objective)
    result = _run(inp, C)
    _check_dtypes(result, inp)
    _check_against_references(result, inp)
    assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(
        result["dX"][~inp.valid] == 0
    )
    assert torch.isfinite(result["dW"].float()).all()


def test_reference_policy_ratio_boundary():
    inp = _host_inputs(200, "policy")
    regime = inp.regime
    assert all(bool((regime == k).any()) for k in (0, 2, 3)) and not bool(
        (regime == 1).any()
    )
    assert torch.equal(regime == -1, ~inp.valid)
    # every valid row keeps its margin from the clip knee (in the oracle's FP32 logp)
    logp32 = reference_fp64(inp, need_dx=False, need_dw=False)["logp"].float()
    delta = logp32 - inp.infer_logp
    assert (delta[inp.valid] - math.log(2.0)).abs().min().item() >= KNEE_MARGIN - 1e-6
    assert torch.allclose(
        delta[regime == 0], torch.full_like(delta[regime == 0], -1.0), atol=1e-6
    )
    assert torch.allclose(
        delta[regime == 2], torch.full_like(delta[regime == 2], 1.0), atol=1e-6
    )
    assert torch.all(inp.infer_logp[~inp.valid] == 0) and torch.all(
        inp.loss_weights[~inp.valid] == 0
    )
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
        assert (
            rel_l2(dX[rows], oracle["dX"][rows])
            <= GATE_MARGIN * rel_l2(b0["dX"][rows], oracle["dX"][rows])
            + GATE_TINY["dX_rel_l2"]
        )
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
    logp_impl = _run(inp, C, train_x=False, train_w=False)[
        "logp"
    ]  # the backend's FP32 logp (independent of infer_logp)
    assert (
        logp_impl - reference_fp64(inp, need_dx=False, need_dw=False)["logp"].float()
    )[valid].abs().max().item() <= 1e-2
    ln2 = torch.tensor(math.log(2.0), dtype=torch.float32, device=DEV)
    rows = torch.nonzero(valid).squeeze(1)
    under, over = (
        rows[0::2],
        rows[1::2],
    )  # alternate valid rows: ratio 2 e^-1e-3 (kept) / 2 e^+1e-3 (clipped)
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
    ratio64 = torch.exp(
        z.gather(1, y[:, None]).squeeze(1) - lse - infer[under].double()
    )
    onehot = torch.zeros_like(p).scatter_(1, y[:, None], 1.0)
    unclipped = (
        (-inp.loss_weights[under].double() * ratio64)[:, None] * (onehot - p)
    ) @ W64
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
    fr = cake_backend.forward_logprob(
        inp.X, inp.W, inp.labels, chunk_size=64, backend="reference"
    )
    assert (
        fr.loss is None
        and torch.equal(fr.logp, result["logp"])
        and tuple(fr.lse.shape) == (T_v,)
        and fr.num_rows == inp.T
    )
    oracle = reference_fp64(inp, entry="logprob")
    assert (fr.lse.double() - oracle["lse"][inp.valid]).abs().max().item() <= 1e-2
    dx, dw = cake_backend.backward_logprob(
        inp.X, inp.W, inp.labels, fr.lse, inp.dlogp, chunk_size=64, backend="reference"
    )
    assert torch.equal(dx, result["dX"]) and torch.equal(dw, result["dW"])
    plain = cake_backend.forward_logprob(
        inp.X, inp.W, inp.labels, chunk_size=64, backend="reference", compact_rows=False
    )
    assert (
        tuple(plain.lse.shape) == (inp.T,)
        and plain.row_index is None
        and torch.equal(plain.lse[inp.valid], fr.lse)
    )
    with pytest.raises(
        ValueError, match="lse"
    ):  # the uncompacted statistic does not fit a compacted backward
        cake_backend.backward_logprob(
            inp.X,
            inp.W,
            inp.labels,
            plain.lse,
            inp.dlogp,
            chunk_size=64,
            backend="reference",
            compact_rows=True,
        )
    assert cake_backend.backward_logprob(
        inp.X,
        inp.W,
        inp.labels,
        fr.lse,
        inp.dlogp,
        chunk_size=64,
        need_dx=False,
        need_dw=False,
        backend="reference",
    ) == (None, None)


def test_grad_weight_dtype_fp32():
    inp = _host_inputs(65)
    bf16 = _run(inp, 64)
    # the autograd entry cannot return an FP32 dW for a BF16 leaf (the engine casts to the leaf's dtype)
    with pytest.raises(ValueError, match="W.dtype"):
        _run(inp, 64, grad_weight_dtype=torch.float32)
    with pytest.raises(ValueError, match="W.dtype"), _quiet_experimental():
        chunked_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            grad_weight_dtype=torch.float32,
        )
    # the explicit pair: FP32 accumulators, one cast to the requested dtype
    fr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=64,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.float32,
        backend="reference",
        fuse_dw_cast=False,
    )
    dX, dW32 = cake_backend.backward_loss(
        fr.dx_acc,
        fr.dw_acc,
        None,
        grad_weight_dtype=torch.float32,
        backend="reference",
        row_index=fr.row_index,
        num_rows=fr.num_rows,
    )  # the compacted forward's rows
    assert (
        dW32.dtype == torch.float32
        and dX.dtype == torch.bfloat16
        and tuple(dX.shape) == (inp.T, H_HOST)
    )
    assert (
        torch.equal(fr.loss, bf16["loss"])
        and torch.equal(fr.logp, bf16["logp"])
        and torch.equal(dX, bf16["dX"])
    )
    assert torch.equal(
        dW32.to(torch.bfloat16), bf16["dW"]
    )  # the same accumulator, cast once
    assert torch.equal(dW32, fr.dw_acc) and dW32.data_ptr() != fr.dw_acc.data_ptr()
    result = dict(loss=fr.loss, logp=fr.logp, dX=dX, dW=dW32)
    _check_dtypes(result, inp, grad_weight_dtype=torch.float32)
    _check_against_references(result, inp, grad_weight_dtype=torch.float32)
    oracle = reference_fp64(inp)
    assert rel_l2(dW32, oracle["dW"]) <= rel_l2(bf16["dW"], oracle["dW"])
    assert fr.memory["outputs"]["dW"] == 2 * inp.V * inp.H * 2
    assert (
        stages_for_entry("loss", grad_weight_dtype=torch.float32, fuse_dw_cast=False)[
            -1
        ]
        == "scale_cast_f32"
    )
    assert "gemm_dw_cast_f32" in stages_for_entry(
        "loss", grad_weight_dtype=torch.float32, fuse_dw_cast=True
    )
    # the fused form of the same pair: one compacted chunk -> no FP32 accumulator at all, the backward's
    # GEMM epilogue writes the FP32 dW directly; bitwise the unfused result
    fused = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=64,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.float32,
        backend="reference",
        fuse_dw_cast=True,
    )
    assert fused.dw_acc is None and fused.dz_last is not None and fused.x_src is inp.X
    assert fused.x_last is None and torch.equal(
        fused.x_idx, fr.row_index
    )  # compacted: one chunk = every valid row
    assert "dW_acc" not in fused.memory["accumulators"] and fused.memory["fuse_dw_cast"]
    dXf, dW32f = fused.backward(None, grad_weight_dtype=torch.float32)
    assert (
        torch.equal(dW32f, dW32)
        and torch.equal(dXf, dX)
        and dW32f.dtype == torch.float32
    )


def test_frozen_inputs():
    inp = _host_inputs(37)
    both = _run(inp, 16)
    x_only = _run(inp, 16, train_w=False)
    w_only = _run(inp, 16, train_x=False)
    neither = _run(inp, 16, train_x=False, train_w=False)
    assert (
        x_only["dW"] is None
        and w_only["dX"] is None
        and neither["dX"] is None
        and neither["dW"] is None
    )
    assert torch.equal(x_only["dX"], both["dX"]) and torch.equal(
        w_only["dW"], both["dW"]
    )
    for r in (x_only, w_only, neither):
        assert torch.equal(r["loss"], both["loss"]) and torch.equal(
            r["logp"], both["logp"]
        )
    assert not neither["loss"].requires_grad


def test_retain_graph_repeated_backward():
    inp = _host_inputs(37, "policy")
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    loss = cake_backend.chunked_lm_head_loss(
        X,
        W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=16,
        backend="reference",
    )
    loss.backward(retain_graph=True)
    dx1, dw1 = X.grad.clone(), W.grad.clone()
    X.grad, W.grad = None, None
    loss.backward(retain_graph=True)
    assert torch.equal(X.grad, dx1) and torch.equal(
        W.grad, dw1
    )  # the saved accumulators are read, never written
    X.grad, W.grad = None, None
    (loss * 2.0).backward()
    assert torch.equal(X.grad, (2.0 * dx1.float()).to(torch.bfloat16)) and torch.equal(
        W.grad, (2.0 * dw1.float()).to(torch.bfloat16)
    )


def test_upstream_scale_is_applied_once():
    inp = _host_inputs(65)
    scaled = _run(inp, 64, scale=3.0)
    fr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=64,
        backend="reference",
        fuse_dw_cast=False,
    )
    T_v = int(inp.valid.sum())  # the compacted loop: 62 of the 65 rows, one chunk
    assert (
        fr.backend == "reference"
        and fr.loss.shape == ()
        and fr.memory["num_chunks"] == -(-T_v // 64) == 1
    )
    assert torch.equal(scaled["loss"], fr.loss)
    scatter = lambda rows: cake_backend.scatter_rows(rows, fr.row_index, fr.num_rows)
    assert torch.equal(scaled["dX"], scatter((3.0 * fr.dx_acc).to(torch.bfloat16)))
    assert torch.equal(scaled["dW"], (3.0 * fr.dw_acc).to(torch.bfloat16))
    dx, dw = cake_backend.backward_loss(
        fr.dx_acc,
        fr.dw_acc,
        torch.tensor(3.0),
        backend="reference",
        row_index=fr.row_index,
        num_rows=fr.num_rows,
    )
    assert torch.equal(dx, scaled["dX"]) and torch.equal(dw, scaled["dW"])
    dx32, dw32 = cake_backend.backward_loss(
        fr.dx_acc,
        fr.dw_acc,
        torch.tensor(3.0),
        grad_weight_dtype=torch.float32,
        backend="reference",
        row_index=fr.row_index,
        num_rows=fr.num_rows,
    )
    assert (
        dw32.dtype == torch.float32
        and torch.equal(dw32, 3.0 * fr.dw_acc)
        and torch.equal(dx32, dx)
    )
    with pytest.raises(ValueError, match="num_rows"):
        cake_backend.backward_loss(
            fr.dx_acc, fr.dw_acc, None, backend="reference", row_index=fr.row_index
        )
    # the log-probability entry scales through dlogp
    plain = _run(inp, 64, entry="logprob")
    twice = _run(inp, 64, entry="logprob", scale=2.0)
    assert torch.equal(plain["logp"], twice["logp"])
    assert (
        rel_l2(twice["dX"], 2.0 * plain["dX"].float()) <= 1e-2
        and rel_l2(twice["dW"], 2.0 * plain["dW"].float()) <= 1e-2
    )


def test_return_logp_detached():
    inp = _host_inputs(37)
    X = inp.X.detach().requires_grad_()
    W = inp.W.detach().requires_grad_()
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference")
    loss, logp = cake_backend.chunked_lm_head_loss(
        X, W, inp.labels, return_logp=True, **kw
    )
    assert loss.requires_grad and loss.grad_fn is not None
    assert (
        not logp.requires_grad and logp.grad_fn is None and logp.dtype == torch.float32
    )
    only = cake_backend.chunked_lm_head_loss(X, W, inp.labels, **kw)
    assert isinstance(only, torch.Tensor) and torch.equal(only, loss.detach())
    lp = cake_backend.chunked_lm_head_logprob(
        X, W, inp.labels, chunk_size=16, backend="reference"
    )
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
        fr = cake_backend.forward_loss(
            X0, W, labels0, objective="ce", loss_div=1.0, backend=backend
        )
        assert (
            fr.loss.shape == ()
            and fr.loss.item() == 0.0
            and tuple(fr.logp.shape) == (0,)
        )
        assert (
            tuple(fr.dx_acc.shape) == (0, H_HOST)
            and torch.all(fr.dw_acc == 0)
            and fr.memory["num_chunks"] == 0
        )
        lr = cake_backend.forward_logprob(X0, W, labels0, backend=backend)
        assert (
            lr.loss is None
            and tuple(lr.logp.shape) == (0,)
            and tuple(lr.lse.shape) == (0,)
        )
        dx, dw = cake_backend.backward_logprob(
            X0, W, labels0, lr.lse, torch.zeros(0, device=DEV), backend=backend
        )
        assert (
            tuple(dx.shape) == (0, H_HOST)
            and dx.dtype == torch.bfloat16
            and torch.all(dw == 0)
            and dw.dtype == torch.bfloat16
        )
    # the autograd path (reference backend: the empty dW cast needs no generated program)
    for objective in ("ce", "policy"):
        X = X0.clone().requires_grad_()
        Wt = W.clone().requires_grad_()
        kw = (
            dict(objective=objective, loss_div=1.0)
            if objective == "ce"
            else dict(
                objective=objective,
                infer_logp=torch.zeros(0, device=DEV),
                loss_weights=torch.zeros(0, device=DEV),
            )
        )
        loss, logp = cake_backend.chunked_lm_head_loss(
            X, Wt, labels0, return_logp=True, backend="reference", **kw
        )
        assert (
            loss.dtype == torch.float32
            and loss.shape == ()
            and loss.item() == 0.0
            and tuple(logp.shape) == (0,)
        )
        loss.backward()
        assert tuple(X.grad.shape) == (0, H_HOST) and X.grad.dtype == torch.bfloat16
        assert (
            tuple(Wt.grad.shape) == (V_HOST, H_HOST)
            and Wt.grad.dtype == torch.bfloat16
            and torch.all(Wt.grad == 0)
        )
    X = X0.clone().requires_grad_()
    Wt = W.clone().requires_grad_()
    lp = cake_backend.chunked_lm_head_logprob(X, Wt, labels0, backend="reference")
    assert tuple(lp.shape) == (0,) and lp.dtype == torch.float32
    lp.sum().backward()
    assert tuple(X.grad.shape) == (0, H_HOST) and torch.all(Wt.grad == 0)
    # the shape / dtype checks still apply to an empty call
    with pytest.raises(ValueError, match="int64"):
        cake_backend.forward_loss(
            X0, W, labels0.int(), objective="ce", loss_div=1.0, backend="reference"
        )
    with pytest.raises(ValueError, match="loss_div"):
        cake_backend.forward_loss(X0, W, labels0, objective="ce", backend="reference")
    assert (cache.hits, cache.misses) == (hits, misses)  # never consulted


@pytest.mark.parametrize("objective", ["ce", "policy"])
@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_all_ignored_rows(objective, entry):
    if entry == "logprob" and objective == "policy":
        pytest.skip("the log-probability entry has no objective")
    inp = _host_inputs(37, objective, ignore_frac=1.0)
    assert (
        not inp.valid.any()
        and torch.all(inp.labels == IGNORE_INDEX)
        and torch.all(inp.dlogp == 0)
    )
    result = _run(inp, 16, entry=entry)
    _check_dtypes(result, inp)
    assert result["loss"].item() == 0.0
    assert (
        torch.all(result["logp"] == 0)
        and torch.all(result["dX"] == 0)
        and torch.all(result["dW"] == 0)
    )
    assert not error_report(result, reference_fp64(inp, entry=entry), inp.labels)["nan"]


@pytest.mark.parametrize("ld_pad, copied", [(8, False), (3, True)])
def test_noncontiguous_x_strides(ld_pad, copied):
    inp = _host_inputs(37, ld_pad=ld_pad)
    assert inp.X.stride() == (H_HOST + ld_pad, 1) and not inp.X.is_contiguous()
    problem = validate_lm_head_inputs(
        inp.X, inp.W, inp.labels, objective="ce", loss_div=inp.loss_div
    )
    assert problem.x_copy is copied and problem.ld_x == (
        H_HOST if copied else H_HOST + ld_pad
    )
    strided = _run(inp, 16)
    contiguous = _run(replace(inp, X=inp.X.contiguous()), 16)
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(strided[key], contiguous[key]), key
    assert tuple(strided["dX"].shape) == (37, H_HOST)
    _check_against_references(strided, inp)
    fr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=16,
        backend="reference",
        compact_rows=False,
    )
    assert ("x_copy" in fr.memory["temporary"]) is copied
    # the compacted loop gathers the chunk's rows into a contiguous buffer: no copy of X whatever its stride
    compact = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=16,
        backend="reference",
        compact_rows=True,
    )
    assert (
        compact.memory["compact_rows"]
        and "x_copy" not in compact.memory["temporary"]
        and torch.equal(compact.logp, strided["logp"])
    )


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
        assert abs(r["loss"].item() - large["loss"].item()) <= 1e-5 * abs(
            large["loss"].item()
        )
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference")
    T_v = int(
        inp.valid.sum()
    )  # the default compacted loop chunks the valid rows (190 of 200): 12 chunks, not 13
    assert (
        cake_backend.forward_loss(inp.X, inp.W, inp.labels, **kw).memory["num_chunks"]
        == -(-T_v // 16)
        == 12
    )
    assert (
        cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, compact_rows=False, **kw
        ).memory["num_chunks"]
        == 13
    )
    # the row statistics do not depend on the chunking at all
    assert (
        torch.equal(small["logp"], large["logp"])
        or (small["logp"] - large["logp"]).abs().max().item() <= 1e-5
    )


def test_t_changes_between_calls():
    W = _host_inputs(1).W
    inputs = [_host_inputs(T, W=W) for T in (37, 64, 37)]
    assert all(inp.W is W for inp in inputs)
    fresh = [_run(inp, 16) for inp in inputs]  # each shape computed on its own
    again = [
        _run(inp, 16) for inp in inputs
    ]  # the sequence 37 -> 64 -> 37 in one process
    for a, b in zip(fresh, again, strict=True):
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(a[key], b[key]), key
    assert torch.equal(fresh[0]["loss"], fresh[2]["loss"]) and not torch.equal(
        fresh[0]["loss"], fresh[1]["loss"]
    )


# ---------------------------------------------------------------------------
# Valid-row compaction (the chunk loop over the rows with labels >= 0 only)
# ---------------------------------------------------------------------------


def test_valid_row_index_and_scatter():
    labels = torch.tensor([3, IGNORE_INDEX, 5, 7, IGNORE_INDEX], device=DEV)
    idx = cake_backend.valid_row_index(labels)
    assert idx.dtype == torch.int64 and idx.tolist() == [0, 2, 3]
    assert (
        cake_backend.valid_row_index(torch.tensor([1, 2], device=DEV)) is None
    )  # every row valid: the uncompacted path
    assert (
        cake_backend.valid_row_index(torch.zeros(0, dtype=torch.int64, device=DEV))
        is None
    )
    assert (
        cake_backend.valid_row_index(torch.full((4,), IGNORE_INDEX, device=DEV)).numel()
        == 0
    )
    rows = torch.arange(6, dtype=torch.float32, device=DEV).view(3, 2)
    out = cake_backend.scatter_rows(rows, idx, 5)
    assert (
        tuple(out.shape) == (5, 2)
        and torch.equal(out[idx], rows)
        and torch.all(out[[1, 4]] == 0)
    )
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


def test_fuse_dw_cast_default_env(monkeypatch):
    monkeypatch.delenv(cake_backend.FUSE_DW_CAST_ENV, raising=False)
    assert cake_backend.fuse_dw_cast_default() is True
    monkeypatch.setenv(cake_backend.FUSE_DW_CAST_ENV, "0")
    assert cake_backend.fuse_dw_cast_default() is False
    assert cake_backend._resolve_fuse(None) is False and cake_backend._resolve_fuse(
        True
    )
    assert "gemm_dw_cast_bf16" not in stages_for_entry("loss")
    monkeypatch.setenv(cake_backend.FUSE_DW_CAST_ENV, "1")
    assert cake_backend.fuse_dw_cast_default() is True
    assert cake_backend._resolve_fuse(False) is False
    assert "gemm_dw_cast_bf16" in stages_for_entry("loss")


def test_dx_finalize_default_env(monkeypatch):
    monkeypatch.delenv(cake_backend.DX_FINALIZE_ENV, raising=False)
    assert cake_backend.dx_finalize_default() is True
    assert cake_backend.DX_FINALIZE_STAGES == ("slab_sum", "scale_cast_scatter_bf16")
    monkeypatch.setenv(cake_backend.DX_FINALIZE_ENV, "0")
    assert cake_backend.dx_finalize_default() is False
    assert cake_backend._resolve_dx_finalize(None) is False
    assert cake_backend._resolve_dx_finalize(True) is True
    monkeypatch.setenv(cake_backend.DX_FINALIZE_ENV, "1")
    assert cake_backend.dx_finalize_default() is True
    assert cake_backend._resolve_dx_finalize(False) is False


def test_valid_rows_mask():
    labels = torch.tensor([3, IGNORE_INDEX, 0, IGNORE_INDEX, 7])
    idx, mask = cake_backend.valid_rows(labels, mask=True)
    assert torch.equal(idx, torch.tensor([0, 2, 4])) and idx.dtype == torch.int64
    assert torch.equal(mask, labels >= 0) and mask.dtype == torch.bool
    plain_idx, plain_mask = cake_backend.valid_rows(labels)
    assert torch.equal(plain_idx, idx) and plain_mask is None
    assert torch.equal(cake_backend.valid_row_index(labels), idx)
    assert cake_backend.valid_rows(labels.clamp(min=0), mask=True) == (None, None)
    assert cake_backend.valid_rows(labels[:0], mask=True) == (None, None)
    all_ignored = torch.full((4,), IGNORE_INDEX)
    idx0, mask0 = cake_backend.valid_rows(all_ignored, mask=True)
    assert idx0.numel() == 0 and mask0.shape == (4,) and not mask0.any()


def test_scale_cast_scatter_reference_matches_scatter_rows():
    gen = torch.Generator().manual_seed(SEED)
    for T, frac in ((1, 0.0), (37, 0.3), (64, 1.0), (5, 0.5)):
        valid = torch.rand(T, generator=gen) >= frac
        if frac >= 1.0:
            valid[:] = False
        idx = valid.nonzero().squeeze(1)
        scan = torch.cumsum(valid, 0, dtype=torch.int32)
        acc = torch.randn(int(idx.numel()), H_HOST, generator=gen)
        g = torch.tensor(0.75)
        fused = cake_backend.scale_cast_scatter(
            acc, g, idx, scan, T, backend="reference"
        )
        shipped = cake_backend.scatter_rows(
            cake_backend.scale_cast(acc, g, torch.bfloat16, backend="reference"), idx, T
        )
        assert fused.dtype == torch.bfloat16 and tuple(fused.shape) == (T, H_HOST)
        assert torch.equal(fused, shipped) and torch.all(fused[~valid] == 0)
        # finalize_dx: every row valid -> the flat cast; the mask selects the one-pass form; no mask -> cast + scatter
        if idx.numel() == T:
            flat = cake_backend.finalize_dx(acc, g, None, None, T, backend="reference")
            assert torch.equal(
                flat,
                cake_backend.scale_cast(acc, g, torch.bfloat16, backend="reference"),
            )
        else:
            assert torch.equal(
                cake_backend.finalize_dx(acc, g, idx, valid, T, backend="reference"),
                shipped,
            )
            assert torch.equal(
                cake_backend.finalize_dx(acc, g, idx, None, T, backend="reference"),
                shipped,
            )
    assert tuple(
        cake_backend.scale_cast_scatter(
            torch.zeros(0, H_HOST),
            None,
            torch.zeros(0, dtype=torch.int64),
            torch.zeros(0, dtype=torch.int32),
            0,
            backend="reference",
        ).shape
    ) == (0, H_HOST)
    with pytest.raises(ValueError, match="scan"):
        cake_backend.scale_cast_scatter(
            torch.zeros(2, H_HOST),
            None,
            torch.tensor([0, 1]),
            torch.ones(3, dtype=torch.int32),
            2,
            backend="reference",
        )
    with pytest.raises(ValueError, match="row_valid"):
        cake_backend.finalize_dx(
            torch.zeros(2, H_HOST),
            None,
            torch.tensor([0, 1]),
            torch.ones(3, dtype=torch.bool),
            2,
            backend="reference",
        )
    with pytest.raises(ValueError, match="num_rows"):
        cake_backend.finalize_dx(
            torch.zeros(2, H_HOST),
            None,
            torch.tensor([0, 1]),
            None,
            None,
            backend="reference",
        )


_DX_FINALIZE_CASES = [  # (objective, entry, ignored fraction): multi-chunk + tail, the all-ignored call, no ignored rows
    ("ce", "loss", "five_percent"),
    ("policy", "loss", "half"),
    ("ce", "logprob", "five_percent"),
    ("ce", "loss", "none"),
    ("ce", "logprob", "none"),
    ("ce", "loss", "all"),
    ("ce", "logprob", "all"),
    ("policy", "loss", "all_but_one"),
]


@pytest.mark.parametrize(
    "objective, entry, frac",
    _DX_FINALIZE_CASES,
    ids=["_".join(c) for c in _DX_FINALIZE_CASES],
)
def test_dx_finalize_switch_is_bitwise_on_the_reference_backend(
    objective, entry, frac, monkeypatch
):
    """The fused dX finalize off (``0``: cast + zero fill + ``index_copy_``) and on (one-pass scatter) through the
    autograd entry points and the explicit pair, on the reference backend: ``loss`` / ``logp`` / ``dX`` / ``dW``
    bitwise; the valid-row mask accompanies the compact index exactly when the rows are compacted and the switch is on;
    the all-ignored call returns zeros without the mask being read."""
    T, C = 37, 16
    ignore = {
        "none": 0.0,
        "five_percent": 0.05,
        "half": 0.5,
        "all_but_one": (T - 1) / T,
        "all": 1.0,
    }[frac]
    inp = _host_inputs(T, objective, ignore_frac=ignore)
    n_valid = int(inp.valid.sum())
    compacted = n_valid < T
    out = {}
    for env in ("0", "1"):
        monkeypatch.setenv(cake_backend.DX_FINALIZE_ENV, env)
        result = _run(inp, C, entry=entry, compact_rows=True)
        _check_dtypes(result, inp)
        assert torch.all(result["dX"][~inp.valid] == 0)
        if entry == "loss":
            fr = cake_backend.forward_loss(
                inp.X,
                inp.W,
                inp.labels,
                objective=objective,
                loss_div=inp.loss_div if objective == "ce" else None,
                infer_logp=inp.infer_logp,
                loss_weights=inp.loss_weights,
                chunk_size=C,
                backend="reference",
                compact_rows=True,
            )
            assert (fr.row_index is None) == (not compacted)
            assert (fr.row_valid is not None) == (env == "1" and compacted)
            if fr.row_valid is not None:
                assert fr.row_valid.dtype == torch.bool and tuple(
                    fr.row_valid.shape
                ) == (T,)
                assert int(fr.row_valid.sum()) == n_valid
            assert fr.memory["dx_finalize"] is (env == "1")
            g = torch.tensor(2.5)
            dx, dw = fr.backward(
                g, grad_weight_dtype=torch.float32, backend="reference"
            )
            frozen_dx, _ = cake_backend.backward_loss(
                fr.dx_acc,
                fr.dw_acc,
                g,
                grad_weight_dtype=torch.float32,
                backend="reference",
                row_index=fr.row_index,
                num_rows=fr.num_rows,
                dz_last=fr.dz_last,
                x_last=fr.x_last,
                x_src=fr.x_src,
                x_idx=fr.x_idx,
            )  # without the mask: the previous host path, the same bits
            assert torch.equal(dx, frozen_dx) and dw.dtype == torch.float32
            result["pair"] = (fr.loss, fr.logp, dx, dw)
        out[env] = result
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(out["1"][key], out["0"][key]), key
    if entry == "loss":
        for a, b in zip(out["1"]["pair"], out["0"]["pair"], strict=True):
            assert torch.equal(a, b)
    if frac == "all":
        assert torch.all(out["1"]["dX"] == 0) and torch.all(out["1"]["dW"] == 0)


def test_dx_finalize_plan_keys():
    """A plan with ``dx_finalize`` adds a sliced dX GEMM's slabs with the ``slab_sum`` kernel (one launch key in place
    of the host ``dx_reduce`` step, registered in the plan's stages in launch order); ``dx_cast=False`` leaves the flat
    dX cast out of the cast keys; the reference path never slices K, so it never plans the kernel."""
    g = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    for finalize in (False, True):
        plan = cake_backend.make_plan(
            _problem(4097, 6144, 154880, 4096),
            need_dx=True,
            need_dw=True,
            geometry=g,
            dx_max_slices=4,
            num_sms=148,
            dx_resident=74,
            fuse_dw_cast=True,
            arch="sm_100a",
            dx_finalize=finalize,
        )
        sliced = [i for i in range(plan.num_chunks) if plan.dx_slices_of(i) > 1]
        assert sliced == [0, 1], (
            "both chunks of the 4097-row tail plan take K slices (4 and 3)"
        )
        assert plan.dx_finalize is finalize and plan.dx_cast
        keys = cake_backend.forward_keys(plan)
        for i in range(plan.num_chunks):
            k = plan.dx_slices_of(i)
            reduce = ("slab_sum" if finalize else "dx_reduce", i)
            dx = (plan.dx_stage_of(i), i)
            if k > 1:
                assert reduce in keys and keys.index(reduce) == keys.index(dx) + 1
                assert ("dx_reduce" if finalize else "slab_sum", i) not in keys
            else:
                assert ("slab_sum", i) not in keys and ("dx_reduce", i) not in keys
        assert ("slab_sum" in plan.stages) is finalize
        assert list(plan.stages) == [s for s in cake_jit.STAGES if s in plan.stages]
        assert ("scale_cast_bf16", "dx") in cake_backend.cast_keys(plan)
    uncast = cake_backend.make_plan(
        _problem(4097, 6144, 154880, 4096, entry="logprob"),
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        arch="sm_100a",
        dx_finalize=True,
        dx_cast=False,
    )
    assert ("scale_cast_bf16", "dx") not in cake_backend.cast_keys(uncast)
    assert "scale_cast_bf16" not in uncast.stages and "slab_sum" in uncast.stages
    # the reference runner: the switch is recorded, no slab (one K slice everywhere), dx_out follows dx_cast
    inp = _host_inputs(37)
    common = dict(
        objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference"
    )
    on = prepare_lm_head_loss(inp.X, inp.W, inp.labels, dx_finalize=True, **common)
    off = prepare_lm_head_loss(inp.X, inp.W, inp.labels, dx_finalize=False, **common)
    assert on.plan.dx_finalize and not off.plan.dx_finalize
    assert "slab_sum" not in on.stages and on.stages == off.stages
    assert ("scale_cast_bf16", "dx") in on.backward_order
    on.step(torch.tensor(3.0))
    off.step(torch.tensor(3.0))
    assert torch.equal(on.dx_out, off.dx_out) and torch.equal(on.dw_out, off.dw_out)
    lp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=16,
        entry="logprob",
        backend="reference",
        dx_cast=False,
    )
    assert (
        "dx_out" not in lp.tensors
        and lp.dx_out is None
        and ("scale_cast_bf16", "dx") not in lp.backward_order
    )
    lp.forward()
    lp.dlogp.copy_(inp.dlogp)
    dx_acc, dw = lp.backward()
    assert (
        dx_acc is lp.dx_acc
        and dx_acc.dtype == torch.float32
        and dw.dtype == torch.bfloat16
    )
    with pytest.raises(ValueError, match="slab_sum"):
        stage_values("slab_sum", on.tensors, on.plan, 0)
    # S >= 2 on the host: the reference ``slab_sum`` over a 3-slice workspace is the ``dx_reduce`` add_ chain, bitwise
    gen = torch.Generator().manual_seed(SEED)
    rows_c, rows_ws = 37, 64
    ws = torch.randn(2, rows_ws, H_HOST, generator=gen)
    base = torch.randn(rows_c, H_HOST, generator=gen)
    fused_acc, plain_acc = base.clone(), base.clone()
    cake_backend.ReferenceEngine.slab_sum(
        dict(
            dx=fused_acc,
            ws=ws,
            ws_slab=rows_ws * H_HOST,
            num_vecs=rows_c * H_HOST // 8,
            n_slabs=2,
        )
    )
    cake_backend.ReferenceEngine.dx_reduce(
        dict(acc=plain_acc, ws=ws, rows_c=rows_c, k_slices=3)
    )
    assert torch.equal(fused_acc, plain_acc) and not torch.equal(fused_acc, base)
    with pytest.raises(ValueError, match="finalize_dx"):
        stage_values("scale_cast_scatter_bf16", on.tensors, on.plan, "dx")


def test_dw_stream_default_env(monkeypatch):
    """The dW side-stream knob: unset / ``auto`` = calls of :data:`DW_STREAM_MIN_CHUNKS` (three) or more chunks,
    ``1`` = every multi-chunk call, ``0`` = never; a one-chunk call never (its only accumulate is deferred to the
    backward)."""
    assert cake_backend.DW_STREAM_MIN_CHUNKS == 3
    monkeypatch.delenv(cake_backend.DW_STREAM_ENV, raising=False)
    assert cake_backend.dw_stream_mode() == "auto"
    assert [cake_backend.dw_side_stream(n) for n in (0, 1, 2, 3, 4, 8)] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    for value in ("0", "false", "off", "no", " OFF "):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, value)
        assert cake_backend.dw_stream_mode() == "0"
        assert not any(cake_backend.dw_side_stream(n) for n in (1, 2, 3, 8))
    for value in ("1", "true", "on", "yes", " On "):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, value)
        assert cake_backend.dw_stream_mode() == "1"
        assert [cake_backend.dw_side_stream(n) for n in (1, 2, 3)] == [
            False,
            True,
            True,
        ]
    for value in ("auto", "AUTO", "", "anything-else"):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, value)
        assert cake_backend.dw_stream_mode() == "auto"
        assert [cake_backend.dw_side_stream(n) for n in (2, 3)] == [False, True]


def test_dw_stream_schedule_keys(monkeypatch):
    """The side-stream placement over the launch keys: joins at the first key of every chunk after the first and
    before the keys that follow the chunk loop, forks at each chunk's ``row_grad``, sides at the per-chunk
    accumulates only -- a deferred last chunk's cast GEMM and the flat casts stay on the caller's stream; ``None``
    for the loss entry's cast keys, for a plan without ``dW`` and whenever the rule says no.  The reference runner
    takes no part (host operators on one stream)."""
    g = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})

    def plan_for(T, *, entry="loss", fuse=True, need_dw=True):
        return cake_backend.make_plan(
            _problem(T, 6144, 154880, 4096, entry=entry),
            need_dx=True,
            need_dw=need_dw,
            geometry=g,
            dx_max_slices=4,
            num_sms=148,
            dx_resident=74,
            fuse_dw_cast=fuse,
            arch="sm_100a",
            dx_finalize=True,
        )

    def first_keys(keys):
        first = {}
        for pos, (_, index) in enumerate(keys):
            if isinstance(index, int):
                first.setdefault(index, pos)
        return first

    monkeypatch.delenv(cake_backend.DW_STREAM_ENV, raising=False)
    three = plan_for(
        10000
    )  # chunks (0, 4096), (4096, 4096), (8192, 1808); the last deferred
    assert three.num_chunks == 3 and three.dw_deferred
    fwd = cake_backend.forward_keys(three)
    joins, forks, sides = cake_backend.side_stream_schedule(three, fwd)
    first = first_keys(fwd)
    assert joins == frozenset({first[1], first[2]})
    assert forks == frozenset(p for p, (s, _) in enumerate(fwd) if s == "row_grad")
    assert len(forks) == 3
    assert sides == frozenset(
        p for p, (s, i) in enumerate(fwd) if i in (0, 1) and s == three.dw_acc_stage(i)
    )
    assert len(sides) == 2 and all(p > min(forks) for p in sides)
    assert (three.dw_cast_stage, 2) in cake_backend.cast_keys(three)
    assert (
        cake_backend.side_stream_schedule(three, cake_backend.cast_keys(three)) is None
    )
    unfused = plan_for(10000, fuse=False)
    fwd_u = cake_backend.forward_keys(unfused)
    joins_u, _, sides_u = cake_backend.side_stream_schedule(unfused, fwd_u)
    assert len(sides_u) == 3 and max(sides_u) == len(fwd_u) - 1
    assert joins_u == frozenset(first_keys(fwd_u)[i] for i in (1, 2))
    lp = plan_for(10000, entry="logprob")
    bwd = cake_backend.recompute_keys(lp) + cake_backend.cast_keys(lp)
    joins_l, forks_l, sides_l = cake_backend.side_stream_schedule(lp, bwd)
    last_chunk_key = max(p for p, (_, i) in enumerate(bwd) if isinstance(i, int))
    assert len(sides_l) == 2 and len(forks_l) == 3
    assert last_chunk_key + 1 < len(bwd) and last_chunk_key + 1 in joins_l
    assert bwd[last_chunk_key] == (lp.dw_cast_stage, 2)
    assert first_keys(bwd)[2] in joins_l and first_keys(bwd)[2] < last_chunk_key
    no_dw = plan_for(10000, need_dw=False)
    assert (
        cake_backend.side_stream_schedule(no_dw, cake_backend.forward_keys(no_dw))
        is None
    )
    two = plan_for(4097)
    assert two.num_chunks == 2
    assert (
        cake_backend.side_stream_schedule(two, cake_backend.forward_keys(two)) is None
    )
    monkeypatch.setenv(cake_backend.DW_STREAM_ENV, "1")
    _, forks_2, sides_2 = cake_backend.side_stream_schedule(
        two, cake_backend.forward_keys(two)
    )
    assert len(sides_2) == 1 and len(forks_2) == 2
    monkeypatch.setenv(cake_backend.DW_STREAM_ENV, "0")
    assert cake_backend.side_stream_schedule(three, fwd) is None
    inp = _host_inputs(37)
    common = dict(
        objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference"
    )
    results = []
    for value in ("0", "1"):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, value)
        runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, **common)
        runner.step(torch.tensor(3.0))
        results.append(
            (runner.loss.clone(), runner.dx_out.clone(), runner.dw_out.clone())
        )
    for a, b in zip(results[0], results[1], strict=True):
        assert torch.equal(a, b)


def test_memory_report_fused_dw_cast():
    V, H, C = DEFAULT_V, DEFAULT_H, 4096
    four = memory_report(16231, H, V, C, fuse_dw_cast=True)
    assert four["fuse_dw_cast"] and four["accumulators"]["dW_acc"] == V * H * 4
    assert (
        four["saved_dz_bytes"] == 4096 * V * 2
    )  # the last chunk's dz rows are a view of the whole [C, V] chunk buffer, which stays alive
    one = memory_report(4096, H, V, C, fuse_dw_cast=True)
    assert "dW_acc" not in one["accumulators"] and one["saved_dz_bytes"] == 4096 * V * 2
    assert "dW_acc" in memory_report(4096, H, V, C, fuse_dw_cast=False)["accumulators"]
    plain = memory_report(16231, H, V, C, fuse_dw_cast=False)
    assert not plain["fuse_dw_cast"] and plain["saved_dz_bytes"] == 0
    frozen = memory_report(16231, H, V, C, need_dw=False, fuse_dw_cast=True)
    assert not frozen["fuse_dw_cast"] and frozen["saved_dz_bytes"] == 0
    assert memory_report(0, H, V, C, fuse_dw_cast=True)["saved_dz_bytes"] == 0
    compact = memory_report(4097, H, V, C, valid_rows=3892, fuse_dw_cast=True)
    assert (
        compact["saved_dz_bytes"] == 3892 * V * 2
        and "dW_acc" not in compact["accumulators"]
    )
    assert (
        memory_report(16231, H, V, C)["fuse_dw_cast"]
        is cake_backend.fuse_dw_cast_default()
    )


def test_fused_dw_cast_plan_and_stage_values():
    inp = _host_inputs(37)
    C = 16
    common = dict(objective="ce", loss_div=inp.loss_div, backend="reference")
    fused = prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, chunk_size=C, fuse_dw_cast=True, **common
    )
    plain = prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, chunk_size=C, fuse_dw_cast=False, **common
    )
    plan, t = fused.plan, fused.tensors
    assert plan.fuse_dw_cast and plan.dw_deferred and plan.dw_acc_needed
    assert plan.last_chunk == (32, 5) and plan.dw_cast_stage == "gemm_dw_cast_bf16"
    assert not plain.plan.dw_deferred and plain.plan.dw_cast_stage == "scale_cast_bf16"
    assert fused.stages == stages_for_entry("loss", fuse_dw_cast=True, num_chunks=3)
    # the forward accumulates the chunks before the last one; the backward runs the last chunk's fused GEMM
    assert ("gemm_dw_acc", 1) in fused.forward_order and (
        "gemm_dw_acc",
        2,
    ) not in fused.forward_order
    assert ("gemm_dw_acc", 2) in plain.forward_order
    assert fused.backward_order == (("scale_cast_bf16", "dx"), ("gemm_dw_cast_bf16", 2))
    v = stage_values("gemm_dw_cast_bf16", t, plan, 2)
    assert v["A"].data_ptr() == t["logits"].data_ptr() and tuple(v["A"].shape) == (
        5,
        V_HOST,
    )
    assert v["B"].data_ptr() == inp.X[32].data_ptr() and tuple(v["B"].shape) == (
        5,
        H_HOST,
    )
    assert (
        v["C"] is t["dw_out"]
        and v["WS"] is t["dw_acc"]
        and v["STATS_OUT"] is t["grad_scale"]
    )
    assert (v["M"], v["k_iters"], v["first_chunk"], v["last_chunk"]) == (
        V_HOST,
        1,
        0,
        1,
    )
    with pytest.raises(ValueError, match="last chunk"):
        stage_values("gemm_dw_cast_bf16", t, plan, 0)
    with pytest.raises(ValueError, match="casts its weight gradient"):
        stage_values("gemm_dw_cast_f32", t, plan, 2)
    with pytest.raises(ValueError, match="fuse_dw_cast"):
        stage_values("gemm_dw_cast_bf16", plain.tensors, plain.plan, 2)
    # the same FP32 operations in the same order: bitwise outputs; the accumulator holds the first chunks
    fused.step(torch.tensor(3.0))
    plain.step(torch.tensor(3.0))
    assert torch.equal(fused.loss, plain.loss) and torch.equal(
        fused.dx_out, plain.dx_out
    )
    assert (
        torch.equal(fused.dw_out, plain.dw_out) and fused.dw_out.dtype == torch.bfloat16
    )
    assert torch.equal(
        t["dw_acc"] + cake_backend._mm_fp32(v["A"].t(), v["B"]), plain.tensors["dw_acc"]
    )
    # one chunk: no FP32 [V, H] accumulator exists, the GEMM stores cast(g * tile)
    one = prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, chunk_size=64, fuse_dw_cast=True, **common
    )
    assert one.plan.dw_deferred and not one.plan.dw_acc_needed and one.dw_acc is None
    assert "dw_acc" not in one.tensors and "gemm_dw_acc" not in one.stages
    assert one.stages == stages_for_entry("loss", fuse_dw_cast=True, num_chunks=1)
    assert (
        "dW_acc" not in one.memory["accumulators"]
        and one.memory["saved_dz_bytes"] == 37 * V_HOST * 2
    )
    v1 = stage_values("gemm_dw_cast_bf16", one.tensors, one.plan, 0)
    assert v1["WS"] is one.tensors["f32_dummy"] and v1["first_chunk"] == 1
    one_plain = prepare_lm_head_loss(
        inp.X, inp.W, inp.labels, chunk_size=64, fuse_dw_cast=False, **common
    )
    one.step()
    one_plain.step()
    assert torch.equal(one.dw_out, one_plain.dw_out) and torch.equal(
        one.loss, one_plain.loss
    )
    # compacted FP32 dW: the last chunk's X rows are gathered again before the fused GEMM
    cf = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        compact_rows=True,
        grad_weight_dtype=torch.float32,
        fuse_dw_cast=True,
        **common,
    )
    last = cf.plan.num_chunks - 1
    assert (
        cf.plan.dw_cast_stage == "gemm_dw_cast_f32"
        and cf.tensors["dw_out"].dtype == torch.float32
    )
    assert cf.backward_order == (
        ("scale_cast_bf16", "dx"),
        ("gather_rows", last),
        ("gemm_dw_cast_f32", last),
    )
    vc = stage_values("gemm_dw_cast_f32", cf.tensors, cf.plan, last)
    assert (
        vc["B"].data_ptr() == cf.tensors["x_c"].data_ptr()
        and vc["C"].dtype == torch.float32
    )
    cp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        compact_rows=True,
        grad_weight_dtype=torch.float32,
        fuse_dw_cast=False,
        **common,
    )
    cf.step(torch.tensor(0.5))
    cp.step(torch.tensor(0.5))
    assert torch.equal(cf.dw_out, cp.dw_out) and torch.equal(cf.dx_out, cp.dx_out)
    # the log-probability entry runs the fused GEMM inside the recompute (its scale is the constant 1)
    lp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        entry="logprob",
        backend="reference",
        fuse_dw_cast=True,
    )
    assert ("gemm_dw_cast_bf16", 2) in lp.backward_order
    assert ("gemm_dw_acc", 2) not in lp.backward_order and (
        "gemm_dw_acc",
        1,
    ) in lp.backward_order
    assert lp.backward_order[-1] == ("scale_cast_bf16", "dx")
    assert (
        stage_values("gemm_dw_cast_bf16", lp.tensors, lp.plan, 2)["STATS_OUT"]
        is lp.tensors["unit_scale"]
    )
    lpp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        entry="logprob",
        backend="reference",
        fuse_dw_cast=False,
    )
    lp.dlogp.copy_(inp.dlogp)
    lpp.dlogp.copy_(inp.dlogp)
    lp.step()
    lpp.step()
    assert torch.equal(lp.dw_out, lpp.dw_out) and torch.equal(lp.dx_out, lpp.dx_out)


def test_fused_dw_cast_autograd_is_bitwise():
    # the autograd entries hand out dW in W's dtype (bf16; an FP32 weight gradient is served by the explicit pair, below)
    for objective in ("ce", "policy"):
        inp = _host_inputs(37, objective)
        for scale in (None, 3.0):
            plain = _run(inp, 16, scale=scale, fuse_dw_cast=False)
            fused = _run(inp, 16, scale=scale, fuse_dw_cast=True)
            for key in ("loss", "logp", "dX", "dW"):
                assert torch.equal(plain[key], fused[key]), (objective, scale, key)
            assert fused["dW"].dtype == torch.bfloat16
        plain = _run(inp, 16, entry="logprob", fuse_dw_cast=False)
        fused = _run(inp, 16, entry="logprob", fuse_dw_cast=True)
        for key in ("logp", "dX", "dW"):
            assert torch.equal(plain[key], fused[key]), (objective, key)
    # the deferred chunk's operands are saved through the autograd context: an in-place write to X between
    # the forward and the backward raises PyTorch's saved-tensor version error instead of a stale dW (X is a
    # fresh leaf: a view of a no_grad-created base would raise autograd's view-base message instead)
    inp = _host_inputs(37)
    for compact in (False, True):
        X = inp.X.detach().clone().requires_grad_(True)
        W = inp.W.detach().clone().requires_grad_(True)
        loss = cake_backend.chunked_lm_head_loss(
            X,
            W,
            inp.labels,
            objective="ce",
            loss_div=inp.loss_div,
            chunk_size=16,
            backend="reference",
            compact_rows=compact,
            fuse_dw_cast=True,
        )
        with torch.no_grad():
            X.add_(1.0)
        with pytest.raises(
            RuntimeError, match="modified by an inplace operation|modified inplace"
        ):
            loss.backward()
    # the explicit pair carries the operands; ForwardResult.backward passes them along
    fr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=16,
        backend="reference",
        compact_rows=False,
        fuse_dw_cast=True,
    )
    assert (
        fr.x_src is None
        and fr.x_idx is None
        and fr.x_last.data_ptr() == inp.X[32].data_ptr()
    )
    assert tuple(fr.dz_last.shape) == (5, V_HOST) and fr.dz_last.dtype == torch.bfloat16
    pr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=16,
        backend="reference",
        compact_rows=False,
        fuse_dw_cast=False,
    )
    assert pr.dz_last is None and pr.x_last is None
    for grad, dtype in ((None, torch.bfloat16), (torch.tensor(2.5), torch.float32)):
        dxf, dwf = fr.backward(grad, grad_weight_dtype=dtype)
        dxp, dwp = pr.backward(grad, grad_weight_dtype=dtype)
        assert torch.equal(dxf, dxp) and torch.equal(dwf, dwp) and dwf.dtype == dtype
    with pytest.raises(ValueError, match="x_last"):
        cake_backend.backward_loss(
            fr.dx_acc, fr.dw_acc, None, backend="reference", dz_last=fr.dz_last
        )
    # dw_cast alone: the fused GEMM as an eager primitive (reference arithmetic)
    out = cake_backend.dw_cast(
        fr.dz_last, fr.x_last, fr.dw_acc, None, torch.float32, backend="reference"
    )
    assert torch.equal(
        out, fr.dw_acc + cake_backend._mm_fp32(fr.dz_last.t(), fr.x_last)
    )
    with pytest.raises(ValueError, match="dz_last"):
        cake_backend.dw_cast(
            fr.dz_last.float(),
            fr.x_last,
            fr.dw_acc,
            None,
            torch.bfloat16,
            backend="reference",
        )
    with pytest.raises(ValueError, match="x_rows"):
        cake_backend.dw_cast(
            fr.dz_last,
            fr.x_last[:2],
            fr.dw_acc,
            None,
            torch.bfloat16,
            backend="reference",
        )
    with pytest.raises(ValueError, match="dw_acc"):
        cake_backend.dw_cast(
            fr.dz_last,
            fr.x_last,
            fr.dw_acc[:1],
            None,
            torch.bfloat16,
            backend="reference",
        )


def test_memory_report_compaction():
    V, H, C, T, T_v = DEFAULT_V, DEFAULT_H, 4096, 4097, 3892
    align = lambda n: (n + WORKSPACE_ALIGN - 1) // WORKSPACE_ALIGN * WORKSPACE_ALIGN
    plain = memory_report(T, H, V, C)
    assert (
        not plain["compact_rows"]
        and plain["valid_rows"] == T
        and plain["gather_bytes"] == 0
        and "x_c" not in plain["temporary"]
    )
    m = memory_report(T, H, V, C, valid_rows=T_v, dx_finalize=False)
    assert (
        m["compact_rows"]
        and m["valid_rows"] == T_v
        and m["num_chunks"] == 1
        and m["vocab_rows_max"] == T_v
    )
    assert m["gather_bytes"] == m["temporary"]["x_c"] == T_v * H * 2
    assert m["temporary"]["logits"] == T_v * V * 2 < plain["temporary"]["logits"]
    assert (
        m["temporary"]["row_index"] == T_v * 8
        and m["temporary"]["dx_compact"] == T_v * H * 2
    )
    fused = memory_report(T, H, V, C, valid_rows=T_v, dx_finalize=True)
    assert fused["dx_finalize"] and not m["dx_finalize"]
    assert "dx_compact" not in fused["temporary"]
    assert (
        fused["temporary"]["row_valid"] == T and fused["temporary"]["row_scan"] == T * 4
    )
    assert (
        fused["outputs"] == m["outputs"] and fused["accumulators"] == m["accumulators"]
    )
    assert (
        memory_report(T, H, V, C, valid_rows=T_v)["dx_finalize"]
        is cake_backend.dx_finalize_default()
    )
    assert (
        "row_scan" not in memory_report(T, H, V, C, dx_finalize=True)["temporary"]
    )  # uncompacted: no scatter
    assert (
        "row_scan"
        not in memory_report(
            T, H, V, C, valid_rows=T_v, need_dx=False, dx_finalize=True
        )["temporary"]
    )
    assert (
        m["temporary"]["lse"] == T_v * 4 and m["temporary"]["logp"] == T_v * 4
    )  # the compact rows before the scatter
    assert (
        m["accumulators"]["dX_acc"] == T_v * H * 4 and m["outputs"]["dX"] == T * H * 2
    )  # accumulate compact, return [T, H]
    assert m["outputs"] == plain["outputs"] and m["weights"] == plain["weights"]
    assert (
        "x_copy"
        not in memory_report(T, H, V, C, x_copy=True, valid_rows=T_v)["temporary"]
    )  # the gather output is contiguous
    lp = memory_report(T, H, V, C, entry="logprob", valid_rows=T_v)
    assert (
        lp["temporary"]["dlogp"] == T * 4
        and lp["temporary"]["dlogp_compact"] == T_v * 4
        and lp["accumulators"]["saved_lse"] == T_v * 4
    )
    assert memory_report(T, H, V, C, valid_rows=0)["num_chunks"] == 0
    layout = workspace_layout(T_v, V, C, hidden=H, compact=True)
    assert layout["x_c"][1] == T_v * H * 2 and layout["total"] == workspace_layout(
        T_v, V, C
    )["total"] + align(T_v * H * 2)
    with pytest.raises(ValueError, match="hidden"):
        workspace_layout(T_v, V, C, compact=True)
    assert (
        lm_head_loss_workspace_size(
            T, V, C, backend="reference", hidden=H, compact_rows=True
        )
        == workspace_layout(T, V, C, hidden=H, compact=True)["total"]
    )
    with pytest.raises(ValueError, match="hidden"):
        lm_head_loss_workspace_size(T, V, C, backend="reference", compact_rows=True)


COMPACTION_CASES = [("ce", "loss"), ("policy", "loss"), ("ce", "logprob")]


@pytest.mark.parametrize("frac", ["none", "five_percent", "half", "all_but_one"])
@pytest.mark.parametrize(
    "objective, entry", COMPACTION_CASES, ids=["ce", "policy", "logprob"]
)
def test_compaction_matches_uncompacted(objective, entry, frac):
    T, C = 37, 16
    ignore = {
        "none": 0.0,
        "five_percent": 0.05,
        "half": 0.5,
        "all_but_one": (T - 1) / T,
    }[frac]
    inp = _host_inputs(T, objective, ignore_frac=ignore)
    num_ignored = int((~inp.valid).sum())
    assert num_ignored == (T - 1 if frac == "all_but_one" else round(ignore * T))
    T_v = T - num_ignored
    compact = _run(inp, C, entry=entry, compact_rows=True)
    plain = _run(inp, C, entry=entry, compact_rows=False)
    for result in (compact, plain):
        _check_dtypes(result, inp)
        _check_against_references(result, inp, entry=entry)
        assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(
            result["dX"][~inp.valid] == 0
        )
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
    kw = dict(
        objective=objective,
        loss_div=inp.loss_div if objective == "ce" else None,
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
    )
    if entry == "loss":
        fr = cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=C,
            backend="reference",
            compact_rows=True,
            **kw,
        )
        assert tuple(fr.dx_acc.shape) == (T_v, H_HOST)
    else:
        fr = cake_backend.forward_logprob(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=C,
            backend="reference",
            compact_rows=True,
        )
        assert tuple(fr.lse.shape) == (T_v,)  # the saved statistic stays compact
    assert (
        tuple(fr.logp.shape) == (T,)
        and fr.num_rows == T
        and torch.equal(fr.logp, compact["logp"])
    )
    assert (fr.row_index is None) == (num_ignored == 0)
    assert (
        fr.memory["compact_rows"] == (num_ignored > 0)
        and fr.memory["valid_rows"] == T_v
    )
    assert fr.memory["num_chunks"] == -(-T_v // C) and fr.memory[
        "vocab_rows_max"
    ] == min(T_v, C)
    if num_ignored:
        assert torch.equal(fr.row_index, inp.valid.nonzero().squeeze(1))
        assert fr.memory["gather_bytes"] == min(T_v, C) * H_HOST * 2
    plain_fr = (
        cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=C,
            backend="reference",
            compact_rows=False,
            **kw,
        )
        if entry == "loss"
        else None
    )
    if plain_fr is not None:
        assert (
            plain_fr.row_index is None
            and not plain_fr.memory["compact_rows"]
            and plain_fr.memory["num_chunks"] == -(-T // C)
        )


def test_compacted_runner_binds_the_valid_rows():
    T, C = 37, 16
    inp = _host_inputs(T, "policy", ignore_frac=0.5)
    idx = cake_backend.valid_row_index(inp.labels)
    T_v = int(idx.numel())
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=C,
        backend="reference",
        compact_rows=True,
        fuse_dw_cast=False,
    )
    plan, t = runner.plan, runner.tensors
    assert (
        plan.compact
        and plan.rows == T_v
        and runner.valid_rows == T_v
        and torch.equal(runner.row_index, idx)
    )
    assert plan.chunks == plan_chunks(T_v, C) and runner.problem.num_rows == T
    assert runner.stages == stages_for_entry(
        "loss", fuse_dw_cast=False
    )  # the host gathers are not stages of the program
    # the row operands are gathered once per step, the chunk's X rows before each chunk
    assert runner.forward_order[:3] == (
        ("gather_rows", "infer_logp"),
        ("gather_rows", "loss_weights"),
        ("gather_rows", 0),
    )
    assert runner.forward_order[3:5] == (("gemm_logits", 0), ("row_finalize", 0))
    assert [
        k
        for k in runner.forward_order
        if k[0] == "gather_rows" and isinstance(k[1], int)
    ] == [("gather_rows", i) for i in range(plan.num_chunks)]
    assert tuple(t["labels"].shape) == (T_v,) and torch.equal(
        t["labels"].long(), inp.labels[idx]
    )
    assert (
        tuple(t["x_c"].shape) == (min(T_v, C), H_HOST)
        and t["x_c"].dtype == torch.bfloat16
    )
    assert tuple(t["lse"].shape) == tuple(t["logp"].shape) == (T_v,) and tuple(
        t["dx_acc"].shape
    ) == tuple(t["dx_out"].shape) == (T_v, H_HOST)
    assert t["infer_logp_full"] is inp.infer_logp and tuple(t["infer_logp"].shape) == (
        T_v,
    )
    for index, (row0, rows_c) in enumerate(plan.chunks):
        g = stage_values("gather_rows", t, plan, index)
        assert (
            g["src"] is t["X"]
            and tuple(g["out"].shape) == (rows_c, H_HOST)
            and g["out"].data_ptr() == t["x_c"].data_ptr()
        )
        assert torch.equal(g["idx"], idx[row0 : row0 + rows_c])
        v = stage_values("gemm_logits", t, plan, index)
        assert (
            v["T"] == T_v
            and v["A"].data_ptr() == t["x_c"].data_ptr()
            and tuple(v["A"].shape) == (rows_c, H_HOST)
        )
        assert (
            stage_values("gemm_dw_acc", t, plan, index)["B"].data_ptr()
            == t["x_c"].data_ptr()
        )
    op = stage_values("gather_rows", t, plan, "infer_logp")
    assert (
        op["src"] is inp.infer_logp
        and op["out"] is t["infer_logp"]
        and torch.equal(op["idx"], idx)
    )
    runner.step()
    assert torch.equal(t["infer_logp"], inp.infer_logp[idx]) and torch.equal(
        t["x_c"][: plan.chunks[-1][1]], inp.X[idx[plan.chunks[-1][0] :]]
    )
    autograd = _run(inp, C, compact_rows=True)
    assert torch.equal(runner.scatter(runner.logp), autograd["logp"]) and torch.equal(
        runner.scatter(runner.dx_out), autograd["dX"]
    )
    assert torch.equal(runner.loss.reshape(()), autograd["loss"]) and torch.equal(
        runner.dw_out, autograd["dW"]
    )
    assert torch.all(runner.scatter(runner.logp)[~inp.valid] == 0)
    with pytest.raises(ValueError, match="gather_rows"):
        stage_values(
            "gather_rows",
            t,
            cake_backend.make_plan(runner.problem, need_dx=True, need_dw=True),
            0,
        )
    # the log-probability runner gathers the caller's [T] dlogp at the start of its backward
    lp = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=C,
        entry="logprob",
        dlogp=inp.dlogp,
        backend="reference",
        compact_rows=idx,
    )
    assert (
        lp.backward_order[:2] == (("gather_rows", "d_in"), ("gather_rows", 0))
        and lp.dlogp is inp.dlogp
    )
    assert tuple(lp.tensors["d_in"].shape) == (T_v,) and tuple(lp.lse.shape) == (T_v,)
    lp.forward()
    dx, dw = lp.backward()
    assert torch.equal(lp.tensors["d_in"], inp.dlogp[idx]) and tuple(dx.shape) == (
        T_v,
        H_HOST,
    )
    ref = _run(inp, C, entry="logprob", compact_rows=True)
    assert (
        torch.equal(lp.scatter(dx), ref["dX"])
        and torch.equal(dw, ref["dW"])
        and torch.equal(lp.scatter(lp.logp), ref["logp"])
    )
    # a runner cannot be prepared over no valid rows; a mismatching index is rejected
    with pytest.raises(ValueError, match="valid row"):
        prepare_lm_head_loss(
            inp.X,
            inp.W,
            torch.full_like(inp.labels, IGNORE_INDEX),
            objective="ce",
            loss_div=1.0,
            chunk_size=C,
            backend="reference",
            compact_rows=True,
        )
    with pytest.raises(ValueError, match="int64"):
        prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="ce",
            loss_div=1.0,
            chunk_size=C,
            backend="reference",
            compact_rows=idx.int(),
        )


@pytest.mark.parametrize("entry", ["loss", "logprob"])
def test_all_ignored_rows_compacted_return_zeros_without_binding(entry, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError(
            "a call whose every row is ignored must neither bind nor launch"
        )

    monkeypatch.setattr(cake_backend, "prepare_lm_head_loss", refuse)
    monkeypatch.setattr(cake_backend, "record_for", refuse)
    inp = _host_inputs(37, ignore_frac=1.0)
    cache = cake_backend.BINDING_CACHE
    hits, misses = cache.hits, cache.misses
    if entry == "loss":
        fr = cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="ce",
            loss_div=inp.loss_div,
            backend="reference",
            compact_rows=True,
        )
        assert (
            fr.loss.item() == 0.0
            and tuple(fr.logp.shape) == (37,)
            and torch.all(fr.logp == 0)
        )
        assert (
            tuple(fr.dx_acc.shape) == (0, H_HOST)
            and torch.all(fr.dw_acc == 0)
            and fr.row_index.numel() == 0
        )
        assert (
            fr.memory["compact_rows"]
            and fr.memory["valid_rows"] == 0
            and fr.memory["num_chunks"] == 0
        )
        dx, dw = cake_backend.backward_loss(
            fr.dx_acc,
            fr.dw_acc,
            None,
            backend="reference",
            row_index=fr.row_index,
            num_rows=fr.num_rows,
        )
        assert (
            tuple(dx.shape) == (37, H_HOST)
            and torch.all(dx == 0)
            and torch.all(dw == 0)
        )
    else:
        fr = cake_backend.forward_logprob(
            inp.X, inp.W, inp.labels, backend="reference", compact_rows=True
        )
        assert (
            tuple(fr.logp.shape) == tuple(fr.lse.shape) == (37,)
            and torch.all(fr.logp == 0)
            and torch.all(fr.lse == 0)
        )
        dx, dw = cake_backend.backward_logprob(
            inp.X,
            inp.W,
            inp.labels,
            fr.lse,
            inp.dlogp,
            backend="reference",
            compact_rows=True,
        )
        assert (
            tuple(dx.shape) == (37, H_HOST)
            and torch.all(dx == 0)
            and torch.all(dw == 0)
        )
    result = _run(inp, 16, entry=entry, compact_rows=True)
    _check_dtypes(result, inp)
    assert (
        result["loss"].item() == 0.0
        and torch.all(result["logp"] == 0)
        and torch.all(result["dX"] == 0)
        and torch.all(result["dW"] == 0)
    )
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
    return make_inputs(
        T,
        objective=objective,
        seed=seed + T,
        device=CUDA,
        W=W,
        ignore_frac=0.0 if T < 16 else 0.05,
    )


@pytest.mark.parametrize("T", [1, 4095, 4097])
@pytest.mark.parametrize("objective", ["ce", "policy"])
def test_device_forward_backward_matches_reference(glm_weight, objective, T):
    _require_program(entry="loss")
    inp = _device_inputs(T, objective, W=glm_weight)
    result = _run(inp, 4096, backend="cake")
    _check_dtypes(result, inp)
    _check_against_references(result, inp, ceiling=True)
    assert torch.all(result["logp"][~inp.valid] == 0) and torch.all(
        result["dX"][~inp.valid] == 0
    )


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
    assert (
        x_only["dW"] is None
        and w_only["dX"] is None
        and neither["dX"] is None
        and neither["dW"] is None
    )
    assert torch.equal(x_only["dX"], both["dX"]) and torch.equal(
        w_only["dW"], both["dW"]
    )
    for r in (x_only, w_only, neither):
        assert torch.equal(r["loss"], both["loss"]) and torch.equal(
            r["logp"], both["logp"]
        )


def test_device_grad_weight_dtype_fp32(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    bf16 = _run(inp, 4096, backend="cake")
    with pytest.raises(ValueError, match="W.dtype"), _quiet_experimental():
        chunked_lm_head_loss(
            inp.X.detach().requires_grad_(),
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            grad_weight_dtype=torch.float32,
            backend="cake",
        )
    fr = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.float32,
        backend="cake",
        fuse_dw_cast=False,
    )
    dX, dW32 = cake_backend.backward_loss(
        fr.dx_acc,
        fr.dw_acc,
        None,
        grad_weight_dtype=torch.float32,
        backend="cake",
        row_index=fr.row_index,
        num_rows=fr.num_rows,
    )  # the compacted forward's rows
    torch.cuda.synchronize()
    assert (
        dW32.dtype == torch.float32
        and dX.dtype == torch.bfloat16
        and tuple(dX.shape) == (inp.T, DEFAULT_H)
    )
    assert torch.equal(fr.loss, bf16["loss"]) and torch.equal(dX, bf16["dX"])
    assert torch.equal(dW32.to(torch.bfloat16), bf16["dW"]) and torch.equal(
        dW32, fr.dw_acc
    )
    # the fused form: one compacted chunk, no FP32 accumulator, the GEMM epilogue writes the FP32 dW (bitwise)
    fused = cake_backend.forward_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        grad_weight_dtype=torch.float32,
        backend="cake",
        fuse_dw_cast=True,
    )
    assert fused.dw_acc is None and fused.dz_last is not None
    _, dW32f = fused.backward(None, grad_weight_dtype=torch.float32)
    torch.cuda.synchronize()
    assert torch.equal(dW32f, dW32)
    result = dict(loss=fr.loss, logp=fr.logp, dX=dX, dW=dW32)
    _check_dtypes(result, inp, grad_weight_dtype=torch.float32)
    _check_against_references(
        result, inp, grad_weight_dtype=torch.float32, ceiling=True
    )


def test_device_binding_scratch_binds_constant_cells(glm_weight):
    """A remembered binding's per-call scratch binds the device's constant cells (``grad_scale`` / ``unit_scale`` /
    ``f32_dummy``), not fresh ``ones`` / ``zeros``: no fill launches ride along on every remembered call."""
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    with _cache(True) as cache:
        cache.clear()
        cake_backend.forward_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="ce",
            loss_div=inp.loss_div,
            chunk_size=4096,
            backend="cake",
        )
        assert len(cache) == 1
        binding = next(iter(cache._bindings.values()))
    constants = cake_backend._device_constants(binding.device_index)
    assert constants is cake_backend._device_constants(binding.device_index)
    t = {}
    binding._scratch(t)
    assert all(
        t[name] is constants[name] for name in ("grad_scale", "unit_scale", "f32_dummy")
    )
    torch.cuda.synchronize()
    assert float(t["grad_scale"]) == 1.0 == float(t["unit_scale"]) and not bool(
        t["f32_dummy"].any()
    )
    assert (
        tuple(t["f32_dummy"].shape) == (16,) and t["f32_dummy"].dtype == torch.float32
    )


def test_device_binding_cache_hits_are_bitwise_and_pin_nothing(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, W=glm_weight)
    kw = dict(objective="ce", loss_div=inp.loss_div, chunk_size=4096)
    with _cache(True) as cache:
        cache.clear()
        hits0, misses0 = cache.hits, cache.misses
        first = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, backend="cake", **kw
        )  # validating path, remembers the binding
        second = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, backend="cake", **kw
        )  # remembered binding
        assert (cache.misses, cache.hits) == (misses0 + 1, hits0 + 1)
        with _cache(False):
            fresh = cake_backend.forward_loss(
                inp.X, inp.W, inp.labels, backend="cake", **kw
            )
        torch.cuda.synchronize()
        for name in ("loss", "logp", "dx_acc", "dw_acc", "dz_last", "x_idx"):
            a, b, c = getattr(first, name), getattr(second, name), getattr(fresh, name)
            assert _same(a, b) and _same(b, c), (
                name
            )  # dw_acc is None for a one-chunk fused plan
        assert (
            second.loss.data_ptr() != first.loss.data_ptr()
        )  # outputs are fresh allocations
        key = forward_binding_key(
            inp.X,
            inp.W,
            inp.labels,
            infer_logp=None,
            loss_weights=None,
            need_dx=True,
            need_dw=True,
            grad_weight_dtype=torch.bfloat16,
            entry="loss",
            valid_rows=int(inp.valid.sum()),
            fuse_dw_cast=cake_backend.fuse_dw_cast_default(),
            dx_finalize=cake_backend.dx_finalize_default(),
            **kw,
        )
        binding = cache.peek(key)
        assert (
            binding is not None and binding.holds_no_tensor() and binding.plan.compact
        )
        assert set(binding.owned) <= {"workspace", "tma_descriptor_workspace"}
        assert cache.owned_bytes == sum(
            t.numel() * t.element_size() for t in binding.owned.values()
        )
        # another chunk size is another binding (misses once), then hits
        third = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, backend="cake", **dict(kw, chunk_size=2048)
        )
        fourth = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, backend="cake", **dict(kw, chunk_size=2048)
        )
        torch.cuda.synchronize()
        assert (cache.misses, cache.hits) == (misses0 + 2, hits0 + 2) and len(
            cache
        ) == 2
        assert torch.equal(third.logp, fourth.logp) and torch.equal(
            third.dw_acc, fourth.dw_acc
        )
        # the autograd path reuses the forward binding
        result = _run(inp, 4096, backend="cake")
        assert cache.hits == hits0 + 3 and torch.equal(result["loss"], first.loss)
        assert torch.equal(
            result["dX"],
            cake_backend.scatter_rows(
                first.dx_acc.to(torch.bfloat16), first.row_index, inp.T
            ),
        )


def test_device_runner_launches_without_allocation(glm_weight):
    _require_program(entry="loss")
    inp = _device_inputs(4097, "policy", W=glm_weight)
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="policy",
        infer_logp=inp.infer_logp,
        loss_weights=inp.loss_weights,
        chunk_size=4096,
        backend="cake",
        compact_rows=True,
    )  # the autograd entries' default plan
    runner.step()
    torch.cuda.synchronize()
    before = torch.cuda.memory_stats()
    dx, dw = runner.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    assert dx is runner.dx_out and dw is runner.dw_out
    result = _run(inp, 4096, backend="cake")
    assert torch.equal(runner.loss.reshape(()), result["loss"]) and torch.equal(
        runner.scatter(runner.logp), result["logp"]
    )
    assert torch.equal(runner.scatter(dx), result["dX"]) and torch.equal(
        dw, result["dW"]
    )


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
    assert (
        rc.plan.compact
        and rc.plan.rows == T_v
        and rc.plan.chunks == plan_chunks(T_v, 4096)
        and rp.plan.chunks == ((0, 4096), (4096, 1))
    )
    assert tuple(rc.logp.shape) == (T_v,) and tuple(rc.dx_out.shape) == (T_v, DEFAULT_H)
    rc.step()
    rp.step()
    torch.cuda.synchronize()
    logp_c = rc.scatter(rc.logp)
    assert torch.equal(logp_c, plain["logp"]) and torch.equal(logp_c, compact["logp"])
    assert (
        torch.equal(rc.scatter(rc.dx_out), compact["dX"])
        and torch.equal(rc.dw_out, compact["dW"])
        and torch.equal(rc.loss.reshape(()), compact["loss"])
    )
    if rc.plan.dx_slices_of(0) == rp.plan.dx_slices_of(
        0
    ):  # same K-slice count: the rows of the first plain chunk are bitwise
        assert torch.equal(rc.scatter(rc.dx_out)[:4096], rp.dx_out[:4096])
    assert (
        rc.memory["compact_rows"]
        and rc.memory["valid_rows"] == T_v
        and rc.memory["gather_bytes"] == T_v * DEFAULT_H * 2
    )
    before = torch.cuda.memory_stats()
    rc.step()
    torch.cuda.synchronize()
    after = torch.cuda.memory_stats()
    assert after["allocation.all.allocated"] == before["allocation.all.allocated"]
    # the log-probability entry: compact saved statistic, scattered gradients
    lp_c = _run(inp, 4096, entry="logprob", backend="cake", compact_rows=True)
    lp_p = _run(inp, 4096, entry="logprob", backend="cake", compact_rows=False)
    _check_against_references(lp_c, inp, entry="logprob", ceiling=True)
    assert torch.equal(lp_c["logp"], lp_p["logp"]) and torch.all(
        lp_c["dX"][~inp.valid] == 0
    )
    assert rel_l2(lp_c["dW"].float(), lp_p["dW"].float()) <= GATE_TINY["dW_rel_l2"]


def test_device_dw_stream_switch_is_bitwise(glm_weight, monkeypatch):
    """The dW side stream (``FLASHINFER_CAKE_LM_HEAD_LOSS_DW_STREAM``) off, forced on and at the ``auto`` rule,
    through the autograd entry points and a prepared runner: ``loss`` / ``logp`` / ``dX`` / ``dW`` bitwise for both
    entries (only the launch stream of the per-chunk accumulate differs); the launches of a call with the side stream
    enter exactly two streams, those of a single-stream call one; the remembered binding serves every mode."""
    _require_program(entry="loss")
    streams: list[int] = []
    real = cake_backend._ffi_stream_context

    def spy(index):
        streams.append(torch.cuda.current_stream(index).cuda_stream)
        return real(index)

    monkeypatch.setattr(cake_backend, "_ffi_stream_context", spy)
    two = _device_inputs(
        5000, "policy", W=glm_weight
    )  # 4750 valid rows = two compact chunks: auto = off
    three = _device_inputs(
        8700, W=glm_weight
    )  # 8265 valid rows = three chunks: auto = on
    assert int(two.valid.sum()) == 4750 and int(three.valid.sum()) == 8265
    out = {}
    for name, inp, env, expect in (
        ("two_off", two, "0", 1),
        ("two_on", two, "1", 2),
        ("two_auto", two, "auto", 1),
        ("three_off", three, "0", 1),
        ("three_auto", three, "auto", 2),
    ):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, env)
        streams.clear()
        res = _run(inp, 4096, backend="cake", compact_rows=True)
        assert len(set(streams)) == expect, (name, len(set(streams)))
        streams.clear()
        lp = _run(inp, 4096, entry="logprob", backend="cake", compact_rows=True)
        assert len(set(streams)) == expect, (name, "logprob", len(set(streams)))
        _check_dtypes(res, inp)
        out[name] = (res, lp)
    for on, off in (
        ("two_on", "two_off"),
        ("two_auto", "two_off"),
        ("three_auto", "three_off"),
    ):
        for which in (0, 1):
            for key in ("loss", "logp", "dX", "dW"):
                assert torch.equal(out[on][which][key], out[off][which][key]), (
                    on,
                    which,
                    key,
                )
    kw = dict(
        objective="ce",
        loss_div=three.loss_div,
        chunk_size=4096,
        backend="cake",
        compact_rows=True,
    )
    runner = prepare_lm_head_loss(three.X, three.W, three.labels, **kw)
    assert runner.plan.num_chunks == 3
    steps = {}
    for env, expect in (("0", 1), ("auto", 2), ("1", 2)):
        monkeypatch.setenv(cake_backend.DW_STREAM_ENV, env)
        streams.clear()
        runner.step()
        torch.cuda.synchronize()
        assert len(set(streams)) == expect, ("runner", env, len(set(streams)))
        steps[env] = (
            runner.loss.clone(),
            runner.logp.clone(),
            runner.dx_out.clone(),
            runner.dw_out.clone(),
        )
    for env in ("auto", "1"):
        for a, b in zip(steps[env], steps["0"], strict=True):
            assert torch.equal(a, b)
    assert torch.equal(runner.scatter(steps["auto"][2]), out["three_off"][0]["dX"])


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
            logp = chunked_lm_head_logprob(
                X, W, inp.labels, chunk_size=C, backend="cake"
            )
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
    assert "error" not in outcome, (
        f"launch from a fresh thread failed: {outcome.get('error')!r}"
    )
    for key in ("logp", "dX", "dW"):
        assert torch.equal(outcome["result"][key], main[key]), key
    # a fresh process: the same seeded inputs, the same entry, bitwise the same result
    root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    out = tmp_path / "fresh_process.pt"
    script = _FRESH_PROCESS_SCRIPT.format(
        root=root, T=T, seed=SEED + T, seed_w=SEED, C=C, out=str(out)
    )
    env = dict(
        os.environ, PYTHONPATH=root + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        cwd=root,
        timeout=1800,
    )
    assert proc.returncode == 0, (
        f"fresh process failed ({proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
    )
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
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=4096,
        backend="cake",
    )
    runner.step()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - base
    m = runner.memory
    assert m["vocab_rows_max"] == 4096 and m["num_chunks"] == 2
    assert m["temporary"]["logits"] == 4096 * DEFAULT_V * 2  # never the full [T, V]
    assert peak <= m["temporary_bytes"] + m["accumulator_bytes"] + m[
        "outputs_bytes"
    ] + (256 << 20)
    # the vocabulary-sized part of the peak is the reported chunk workspace (logits + statistics + K-slice slabs), never a
    # [T, V] buffer: the unchunked path at the contract's boundaries holds BF16 z plus its FP32 promotion (T * V * 6 bytes)
    assert peak - m["accumulator_bytes"] - m["outputs_bytes"] <= m[
        "temporary_bytes"
    ] + (256 << 20)
    assert (
        m["temporary_bytes"] < inp.T * DEFAULT_V * 6
        and m["temporary"]["logits"] < inp.T * DEFAULT_V * 2
    )
    # a step through the prepared runner allocates nothing more
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    runner.step()
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - base == 0


def test_device_fused_dw_cast_is_bitwise(glm_weight):
    _require_program(entry="loss")
    _require_program(entry="logprob")
    inp = _device_inputs(4097, W=glm_weight)
    g3 = torch.tensor(3.0, device=inp.X.device)
    for C in (
        4096,
        2048,
    ):  # one compacted chunk (no FP32 accumulator at all) / two chunks
        plain = _run(inp, C, backend="cake", fuse_dw_cast=False)
        fused = _run(inp, C, backend="cake", fuse_dw_cast=True)
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(plain[key], fused[key]), (C, key)
        kw = dict(
            objective="ce",
            loss_div=inp.loss_div,
            chunk_size=C,
            grad_weight_dtype=torch.float32,
            backend="cake",
        )
        fr = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, fuse_dw_cast=True, **kw
        )
        pr = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, fuse_dw_cast=False, **kw
        )
        assert (fr.dw_acc is None) == (
            fr.memory["num_chunks"] == 1
        ) and fr.dz_last is not None
        assert (
            fr.x_src is inp.X and fr.x_last is None
        )  # compacted: the chunk's rows are gathered in the backward
        # the saved rows are a view of the chunk buffer, the storage that stays alive until the backward: the report
        # counts that buffer (not the rows), and the first call of a binding retains no more than the later ones
        saved = int(fr.memory["saved_dz_bytes"])
        assert saved == min(C, int(fr.memory["valid_rows"])) * inp.V * 2
        assert int(fr.dz_last.untyped_storage().nbytes()) == saved
        assert int(fr.dz_last.numel()) * 2 <= saved
        fr2 = cake_backend.forward_loss(
            inp.X, inp.W, inp.labels, fuse_dw_cast=True, **kw
        )  # the remembered binding
        assert int(fr2.dz_last.untyped_storage().nbytes()) == saved
        assert int(fr2.memory["saved_dz_bytes"]) == saved
        del fr2
        if (
            fr.dw_acc is not None
        ):  # the accumulator holds the chunks before the last one
            assert not torch.equal(fr.dw_acc, pr.dw_acc)
        _, dw32 = fr.backward(g3, grad_weight_dtype=torch.float32)
        _, dw32p = pr.backward(g3, grad_weight_dtype=torch.float32)
        torch.cuda.synchronize()
        assert torch.equal(dw32, dw32p) and dw32.dtype == torch.float32
        assert torch.equal(
            dw32.to(torch.bfloat16), _run(inp, C, backend="cake", scale=3.0)["dW"]
        )
    plain = _run(inp, 2048, entry="logprob", backend="cake", fuse_dw_cast=False)
    fused = _run(inp, 2048, entry="logprob", backend="cake", fuse_dw_cast=True)
    for key in ("logp", "dX", "dW"):
        assert torch.equal(plain[key], fused[key]), key
    # the prepared runner: the deferred chunk's dz stays in the workspace between forward() and backward()
    workspace = torch.empty(
        cake_backend.lm_head_loss_workspace_size(
            inp.T, inp.V, 2048, inp.X.device, compact_rows=True
        ),
        dtype=torch.uint8,
        device=inp.X.device,
    )
    # the compacted size bounds every valid-row count: a tail chunk of any length may take more K slices (more FP32
    # slabs) than the full-T chunks, so a runner over a few valid rows must fit the same buffer
    sparse_labels = torch.full_like(inp.labels, IGNORE_INDEX)
    for keep in (1, 3, 129, 2047, 2049, min(inp.T, 3591)):
        sparse_labels[:keep] = inp.labels[:keep].clamp(min=0)  # valid rows only
        cake_backend.prepare_lm_head_loss(
            inp.X,
            inp.W,
            sparse_labels,
            objective="ce",
            loss_div=inp.loss_div,
            chunk_size=2048,
            workspace_buffer=workspace,
            backend="cake",
            compact_rows=True,
            fuse_dw_cast=True,
        )
    runner = cake_backend.prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        objective="ce",
        loss_div=inp.loss_div,
        chunk_size=2048,
        workspace_buffer=workspace,
        backend="cake",
        compact_rows=True,
        fuse_dw_cast=True,
    )
    assert (
        runner.plan.dw_deferred
        and runner.plan.num_chunks == 2
        and runner.backward_order
        == (
            ("scale_cast_bf16", "dx"),
            ("gather_rows", 1),
            ("gemm_dw_cast_bf16", 1),
        )
    )
    runner.step(g3)
    torch.cuda.synchronize()
    assert torch.equal(runner.dw_out, _run(inp, 2048, backend="cake", scale=3.0)["dW"])


def test_device_dx_finalize_is_bitwise(glm_weight, monkeypatch):
    """The fused dX finalize on the generated program: switch ``0`` (host ``add_`` chain, cast, zero fill,
    ``index_copy_``) and ``1`` (``slab_sum`` + ``scale_cast_scatter_bf16``) give bitwise the same ``loss`` / ``logp`` /
    ``dX`` / ``dW`` on both entries, with and without ignored rows (S >= 2: the chunk loops slice their dX GEMMs; S = 1:
    the switch is a no-op); the fused path binds the slab kernel for the sliced chunks and launches the scatter once per
    compacted dX output; the previous path binds neither."""
    _require_program(entry="loss")
    _require_program(entry="logprob")
    real_bind_all = cake_backend._bind_all
    bound = []

    def spy_bind_all(record, module_name, keys, values, device, geometry):
        bound.extend(k[0] for k in keys)
        return real_bind_all(record, module_name, keys, values, device, geometry)

    monkeypatch.setattr(cake_backend, "_bind_all", spy_bind_all)
    for ignore in (0.05, 0.0):
        inp = make_inputs(
            4097, seed=SEED + 49, device=CUDA, W=glm_weight, ignore_frac=ignore
        )
        compacted = int(inp.valid.sum()) < inp.T
        # whether this device's plan slices some dX GEMM of the 4096-row chunk loop (the plan decides from the SM count)
        probe = prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            objective="ce",
            loss_div=inp.loss_div,
            chunk_size=4096,
            backend="cake",
            compact_rows=True,
            dx_finalize=True,
        )
        expect_slabs = any(
            probe.plan.dx_slices_of(i) > 1 for i in range(probe.plan.num_chunks)
        )
        del probe
        out = {}
        for env in ("0", "1"):
            monkeypatch.setenv(cake_backend.DX_FINALIZE_ENV, env)
            with _cache(False):
                bound.clear()
                res = {
                    "loss": _run(inp, 4096, backend="cake", compact_rows=True),
                    "logprob": _run(
                        inp, 4096, entry="logprob", backend="cake", compact_rows=True
                    ),
                }
                fr = cake_backend.forward_loss(
                    inp.X,
                    inp.W,
                    inp.labels,
                    objective="ce",
                    loss_div=inp.loss_div,
                    chunk_size=1000,
                    backend="cake",
                    compact_rows=True,
                )
                assert (fr.row_index is not None) == compacted
                assert (fr.row_valid is not None) == (env == "1" and compacted)
                dx, dw = fr.backward(
                    torch.tensor(2.5, device=CUDA), grad_weight_dtype=torch.float32
                )
                res["pair"] = dict(dX=dx, dW=dw)
                torch.cuda.synchronize()
            slabs = bound.count("slab_sum")
            scatters = bound.count("scale_cast_scatter_bf16")
            if env == "0":
                assert slabs == 0 and scatters == 0, bound
            else:
                assert "dx_reduce" not in bound
                if expect_slabs:
                    assert slabs > 0, bound
                # one scatter per compacted dX output: entry (a) autograd, entry (b), the explicit pair
                assert scatters == (3 if compacted else 0), bound
            out[env] = res
        for entry in ("loss", "logprob"):
            for key in ("loss", "logp", "dX", "dW"):
                assert torch.equal(out["1"][entry][key], out["0"][entry][key]), (
                    ignore,
                    entry,
                    key,
                )
        for key in ("dX", "dW"):
            assert torch.equal(out["1"]["pair"][key], out["0"]["pair"][key]), (
                ignore,
                key,
            )
        assert torch.all(out["1"]["loss"]["dX"][~inp.valid] == 0)
    # the kernels against the torch forms: slab_sum over a prepared runner's workspace (the 4097-row loop: a one-row
    # tail chunk, K-sliced on every supported device), the scatter eagerly
    monkeypatch.setenv(cake_backend.DX_FINALIZE_ENV, "1")
    inp = make_inputs(4097, seed=SEED + 50, device=CUDA, W=glm_weight, ignore_frac=0.0)
    common = dict(
        objective="ce", loss_div=inp.loss_div, chunk_size=4096, backend="cake"
    )
    runner = prepare_lm_head_loss(inp.X, inp.W, inp.labels, dx_finalize=True, **common)
    plan = runner.plan
    sliced = [i for i in range(plan.num_chunks) if plan.dx_slices_of(i) > 1]
    assert sliced and all(("slab_sum", i) in runner.launches for i in sliced)
    assert all(("dx_reduce", i) not in runner.values for i in range(plan.num_chunks))
    assert "slab_sum" in runner.stages and plan.dx_finalize
    runner.forward()
    torch.cuda.synchronize()
    cast_vec = plan.geometry.cast_vec
    for i in sliced:
        v = runner.values[("slab_sum", i)]
        row0, rows_c = plan.chunks[i]
        assert v["dx"].data_ptr() == runner.dx_acc[row0].data_ptr()
        assert v["n_slabs"] == plan.dx_slices_of(i) - 1
        assert v["ws_slab"] == runner.tensors["dx_ws"].stride(0)
        assert v["num_vecs"] == rows_c * DEFAULT_H // cast_vec
        # the registry's grid rule is the kernel's: one CTA per 2048 vectors, at most 65535
        assert runner.launches[("slab_sum", i)].grid == (
            max(1, min(-(-v["num_vecs"] // 2048), 65535)),
            1,
            1,
        )
    plain = prepare_lm_head_loss(inp.X, inp.W, inp.labels, dx_finalize=False, **common)
    assert "slab_sum" not in plain.stages and all(
        ("dx_reduce", i) in plain.values for i in sliced
    )
    plain.forward()
    torch.cuda.synchronize()
    assert torch.equal(
        runner.dx_acc, plain.dx_acc
    )  # the kernel's fixed-order adds are the add_ chain's
    gen = torch.Generator(device="cuda").manual_seed(SEED)
    valid = make_inputs(
        4097, seed=SEED + 51, device=CUDA, W=glm_weight, ignore_frac=0.3
    ).valid
    T = int(valid.numel())
    idx = valid.nonzero().squeeze(1)
    scan = torch.cumsum(valid, 0, dtype=torch.int32)
    acc = torch.randn(int(idx.numel()), DEFAULT_H, device=CUDA, generator=gen)
    g = torch.tensor(0.75, device=CUDA)
    bound.clear()
    fused = cake_backend.scale_cast_scatter(acc, g, idx, scan, T, backend="cake")
    shipped = cake_backend.scatter_rows(
        cake_backend.scale_cast(acc, g, torch.bfloat16, backend="cake"), idx, T
    )
    torch.cuda.synchronize()
    assert torch.equal(fused, shipped) and torch.all(fused[~valid] == 0)
    assert torch.equal(
        fused,
        cake_backend.scale_cast_scatter(acc, g, idx, scan, T, backend="reference"),
    )
    assert bound == [
        "scale_cast_scatter_bf16",
        "scale_cast_bf16",
    ]  # the eager scatter kernel, then the shipped flat cast
