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
import inspect
import math
import os
import re
import subprocess
import sys
import threading
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

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
from tests.test_helpers.cake_package import cake_package

# Backend suite: the package is named once, through ``cake_package``; nothing else of flashinfer is imported here (a
# build that vendors the package under another import path runs this file unchanged after pointing the ``PACKAGES``
# entry at it).  The tests of the public entry points ``flashinfer.chunked_lm_head`` -- their experimental marking and
# the FLASHINFER_CAKE_LM_HEAD_LOSS_* environment variables they read -- live in ``test_cake_lm_head_loss_api.py``.
pkg = cake_package("cake_lm_head_loss")
cake_backend = cake_package("cake_lm_head_loss", "cake_backend")
cake_build = cake_package("cake_lm_head_loss", "cake_build")
cake_jit = cake_package("cake_lm_head_loss", "cake_jit")
ABI_CONTRACT = cake_backend.ABI_CONTRACT
COMMON_SCALARS = cake_backend.COMMON_SCALARS
COMMON_TENSORS = cake_backend.COMMON_TENSORS
CONTRACT_ALIASES = cake_backend.CONTRACT_ALIASES
MODE_CE = cake_backend.MODE_CE
MODE_NONE = cake_backend.MODE_NONE
MODE_POLICY = cake_backend.MODE_POLICY
STAGE_TENSORS = cake_backend.STAGE_TENSORS
SUPPORTED_ABIS = cake_backend.SUPPORTED_ABIS
SUPPORTED_COMPUTE_CAPABILITIES = cake_backend.SUPPORTED_COMPUTE_CAPABILITIES
WORKSPACE_ALIGN = cake_backend.WORKSPACE_ALIGN
BindingCache = cake_backend.BindingCache
Geometry = cake_backend.Geometry
bind_stage = cake_backend.bind_stage
cluster_resident = cake_backend.cluster_resident
forward_binding_key = cake_backend.forward_binding_key
generated_program_available = cake_backend.generated_program_available
grid_dims = cake_backend.grid_dims
lm_head_loss_workspace_size = cake_backend.lm_head_loss_workspace_size
logprob_backward_binding_key = cake_backend.logprob_backward_binding_key
memory_report = cake_backend.memory_report
plan_chunks = cake_backend.plan_chunks
prepare_lm_head_loss = cake_backend.prepare_lm_head_loss
recommended_k_slices = cake_backend.recommended_k_slices
record_abi = cake_backend.record_abi
stage_values = cake_backend.stage_values
stages_for_entry = cake_backend.stages_for_entry
validate_lm_head_inputs = cake_backend.validate_lm_head_inputs
wave_efficiency = cake_backend.wave_efficiency
workspace_layout = cake_backend.workspace_layout

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
_GEMM_VALUES = {"A", "B", "C", "STATS_OUT", "WS", *cake_backend.GEMM_SCALARS}
_STAGE_VALUES = {
    stage: set(tensors)
    | set(COMMON_TENSORS)
    | set(COMMON_SCALARS)
    | {"num_vecs"}
    | (_GEMM_VALUES if stage.startswith("gemm") else set())
    | ({"n_slabs"} if stage == "slab_sum" else set())
    | ({"row_vecs", "num_rows"} if stage == "scale_cast_scatter_bf16" else set())
    | (
        {"ld_x_words", "row_vecs", "num_rows"}
        if stage == cake_backend.HIDDEN_COUNT_STAGE
        else set()
    )
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


def _require_hidden_count():
    """Skip when the hidden valid-row count does not run on this device with the CUDA toolkit that built
    this checkout's libraries (``cake_jit.toolchain_runs_hidden_count``: sm_103a on CUDA 13.0 takes the
    host-count path)."""
    arch = cake_backend.arch_for(torch.device("cuda"))
    if not cake_jit.toolchain_runs_hidden_count(arch):
        pytest.skip(
            f"the hidden valid-row count does not run on {arch} with CUDA "
            f"{'.'.join(map(str, cake_build.record_cuda_version(arch)))}"
        )


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
    **switches,
):
    """One forward + backward through the backend's autograd entry points (``backend="cake"`` runs the generated
    program; the public module ``flashinfer.chunked_lm_head`` is a pass-through to these two functions, tested in
    ``test_cake_lm_head_loss_api.py``).  ``switches``: the remaining switch keywords (``dx_finalize``, ``dw_stream``,
    ``hidden_count``, ``gemm_tuning``; ``logprob_dz`` / ``logprob_lbl`` of the log-probability entry), passed
    through.  Returns ``loss`` / ``logp`` / ``dX`` / ``dW`` and the leaves."""
    X = inp.X.detach().requires_grad_(train_x)
    W = inp.W.detach().requires_grad_(train_w)
    if entry == "loss":
        loss, logp = cake_backend.chunked_lm_head_loss(
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
            **switches,
        )
        if train_x or train_w:
            (loss if scale is None else loss * scale).backward()
    else:
        logp = cake_backend.chunked_lm_head_logprob(
            X,
            W,
            inp.labels,
            chunk_size=C,
            backend=backend,
            compact_rows=compact_rows,
            fuse_dw_cast=fuse_dw_cast,
            **switches,
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


_KERNEL_SOURCE_RE = re.compile(
    r"^csrc/cake_lm_head_loss/cake_lm_head_loss_([0-9a-f]{20})_kernel\.cu$"
)
_PROGRAM_KERNEL_DEFINITION_RE = re.compile(
    r"^-DCAKE_PROGRAM_KERNEL=(kernel_cake_lm_head_loss_[0-9a-f]{20})$"
)
_PROGRAM_KERNEL_DEFAULT_RE = re.compile(
    r"^#ifndef CAKE_PROGRAM_KERNEL\n"
    r"#define CAKE_PROGRAM_KERNEL (kernel_cake_lm_head_loss_[0-9a-f]{20})\n"
    r"#endif$",
    re.MULTILINE,
)


def _kernel_source_symbol(entry, package):
    """The unit whose kernel translation unit a library compiles and the kernel symbol
    that source defines for it: the library's own unit spelling its symbol literally, or a
    shared unit whose source defaults ``CAKE_PROGRAM_KERNEL`` to its own symbol and takes
    a sibling form's symbol from the library's compile definitions."""
    kernel_unit = _KERNEL_SOURCE_RE.match(entry["sources"][0]).group(1)
    text = (package / entry["sources"][0]).read_text()
    own = f"kernel_cake_lm_head_loss_{kernel_unit}"
    defaults = _PROGRAM_KERNEL_DEFAULT_RE.findall(text)
    carried = [
        m.group(1)
        for m in map(_PROGRAM_KERNEL_DEFINITION_RE.match, entry["nvcc_flags"])
        if m is not None
    ]
    assert len(defaults) <= 1 and len(carried) <= 1, entry["name"]
    if not defaults:  # a unit of its own: the source spells its symbol
        assert not carried and own in text, entry["name"]
        return kernel_unit, own
    assert defaults[0] == own and "CAKE_PROGRAM_KERNEL(" in text, entry["name"]
    return kernel_unit, carried[0] if carried else own


def test_registry_records_are_well_formed():
    assert cake_jit.STAGES == (
        "gather_rows_bf16",
        "gemm_logits",
        "gemm_logits_mcnt",
        "gemm_logits_nostats",
        "gemm_logits_nostats_mcnt",
        "gemm_logits_dz",
        "gemm_logits_lbl",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx",
        "gemm_dx_s",
        "gemm_dx_tn256",
        "gemm_dx_s_tn256",
        "gemm_dx_st3",
        "gemm_dx_s_st3",
        "gemm_dx_split",
        "gemm_dx_s_split",
        "gemm_dx_st3_split",
        "gemm_dx_s_st3_split",
        "gemm_dx_swab",
        "gemm_dx_s_swab",
        "slab_sum",
        "gemm_dw_acc",
        "gemm_dw_acc_gn",
        "gemm_dw_cast_bf16",
        "gemm_dw_cast_bf16_tma",
        "gemm_dw_cast_f32",
        "gemm_dw_cast_f32_gn",
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
    # build.json (``cake_build``) is the only record of what to compile; the registry names its
    # entries ``<library>_<arch>`` per stage
    manifest = cake_build.manifest()
    libraries = {entry["name"]: entry for entry in manifest["libraries"]}
    assert len(libraries) == len(manifest["libraries"])  # names unique
    package = Path(cake_jit.__file__).resolve().parent
    for name, record in cake_jit.MODULES.items():
        assert record["arch"] in manifest["arch_flags"]
        assert record["arch"] in SUPPORTED_COMPUTE_CAPABILITIES.values()
        assert record_abi(record) in SUPPORTED_ABIS
        assert (
            cake_jit.select_module(record["arch"]) == name
        )  # one record per architecture
        geometry = Geometry.from_record(record)
        for op in (
            "logits",
            "dx",
            "dw",
        ):  # the launch-scalar defaults of the record's kernels
            assert getattr(geometry, f"{op}_group_m") >= 1
            assert getattr(geometry, f"{op}_item_cols") >= 1
        stages = cake_jit.registered_stages(name)
        assert stages and set(stages) <= set(cake_jit.STAGES)
        for stage in stages:
            physical = record[stage]
            assert physical["module"] and physical["ffi_entry"]
            unit = physical["module"].removeprefix("cake_lm_head_loss_")
            assert physical["library"] == cake_jit.library_stem(unit)
            entry = libraries[cake_jit.library_name(name, stage)]
            assert entry["arch"] == record["arch"]
            assert entry["path"] == f"aot/{entry['name']}.so"
            assert entry["entry"] == physical["ffi_entry"]
            assert isinstance(entry["nvcc_flags"], list)
            assert len(entry["closure_sha256"]) == 64
            assert len(entry["sources"]) == 2
            assert all(
                s.startswith("csrc/cake_lm_head_loss/") for s in entry["sources"]
            )
            assert set(entry["sha256"]) == set(entry["sources"])
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
            kernel, binding = (s.removeprefix("csrc/") for s in entry["sources"])
            # one architecture-neutral translation unit per kernel (no architecture directory);
            # a sibling form's library compiles the shared unit's file and names its own
            # kernel through CAKE_PROGRAM_KERNEL in its compile definitions
            kernel_unit, defined = _kernel_source_symbol(entry, package)
            assert (
                kernel == f"cake_lm_head_loss/cake_lm_head_loss_{kernel_unit}_kernel.cu"
            )
            assert kernel_unit == unit or defined == f"kernel_cake_lm_head_loss_{unit}"
            assert re.fullmatch(
                r"cake_lm_head_loss/cake_lm_head_loss_[0-9a-f]{20}_binding\.cu", binding
            )
            csrc = package / "csrc"
            assert (csrc / kernel).is_file() and (csrc / binding).is_file()
            symbol = f"kernel_{physical['module']}"
            assert defined == symbol
            # the binding launches the stage's kernel: spelled literally, or (a binding several
            # programs share) through CAKE_KERNEL_SYMBOL, which the library's compile definitions
            # or the template's own default set
            text = (csrc / binding).read_text()
            assert "TVM_FFI_DLL_EXPORT_TYPED_FUNC(run," in text
            named = [
                flag.removeprefix("-DCAKE_KERNEL_SYMBOL=")
                for flag in entry["nvcc_flags"]
                if flag.startswith("-DCAKE_KERNEL_SYMBOL=")
            ]
            default = re.findall(
                r"^#define CAKE_KERNEL_SYMBOL (\S+)$", text, re.MULTILINE
            )
            bound = (named or default or [None])[0]
            assert bound == symbol if bound is not None else symbol in text
            assert len(physical.get("grid", ["rows_c", 1, 1])) == 3
            launch = physical.get("launch")
            if launch is not None:
                assert len(launch["block"]) == 3 and all(
                    int(b) >= 1 for b in launch["block"]
                )
                assert len(launch["cluster"]) == 3 and all(
                    int(c) >= 1 for c in launch["cluster"]
                )

    assert (
        cake_jit._expand_registry(
            cake_jit._REGISTRY, cake_jit._ARG_PLANS, cake_jit._LAUNCHES
        )
        == cake_jit.MODULES
    )
    # every binding file the manifest compiles exists, and no other binding ships
    used = {
        libraries[cake_jit.library_name(name, stage)]["sources"][1]
        for name, record in cake_jit.MODULES.items()
        for stage in record["stages"]
    }
    if cake_jit.MODULES:
        unit_dir = package / "csrc" / "cake_lm_head_loss"
        assert sorted(
            f"csrc/cake_lm_head_loss/{p.name}" for p in unit_dir.glob("*_binding.cu")
        ) == sorted(used)


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
    # the default form is the documented default
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
            ["max(1, m_tiles//4*24)*4", "rows_c/8", "num_vecs"], scalars, 148
        ) == (768, 513, 9)
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
        and g.cluster_ctas_of("gemm_dx_s") == 4
        and g.cluster_ctas_of("gemm_dx_s_tn256") == 4
        and g.cluster_ctas_of("gemm_dw_acc") == 2
        and g.cluster_ctas_of("gemm_dw_acc_gn") == 2
        and g.cluster_ctas_of("gemm_logits_mcnt") == 2
        and g.cluster_ctas_of("gemm_dw_cast_f32_gn") == 2
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
    # the per-GEMM work-item columns and default raster heights (launch scalars of the generated kernels): the
    # reference path's defaults are one 256-column tile per item; a record declares its kernels' widths
    assert (
        default.logits_item_cols,
        default.dx_item_cols,
        default.dw_item_cols,
    ) == (256, 256, 256)
    assert (default.logits_group_m, default.dx_group_m, default.dw_group_m) == (
        32,
        16,
        8,
    )
    declared = Geometry.from_record(
        {"geometry": {"dx_item_cols": 512, "dw_item_cols": 512, "dw_group_m": 2}}
    )
    assert (declared.dx_item_cols, declared.dw_item_cols, declared.dw_group_m) == (
        512,
        512,
        2,
    )
    assert (
        cake_backend.gemm_op("gemm_logits_nostats_mcnt"),
        cake_backend.gemm_op("gemm_dx_s_st3"),
        cake_backend.gemm_op("gemm_dw_cast_bf16"),
    ) == ("logits", "dx", "dw")
    with pytest.raises(ValueError):
        cake_backend.gemm_op("row_grad")
    for bad in ({"dx_cluster_ctas": 0}, {"dw_item_cols": 0}, {"logits_group_m": 0}):
        with pytest.raises(ValueError):
            Geometry.from_record({"geometry": bad})


# --------------------------------------------------------------------------- per-chunk instance variants


def test_stage_variant_grammar():
    bases = cake_jit.BASE_STAGES
    assert len(bases) == 16 and set(bases) <= set(cake_jit.STAGES)
    assert cake_backend.STAGE_KNOBS == (
        "count",
        "sliced",
        "tile_n",
        "stages",
        "split",
        "swab",
        "blocked",
        "epi_store",
    )
    for stage in cake_jit.STAGES:
        base, knobs = cake_backend.parse_stage(stage)
        assert base in bases and cake_backend.stage_variant(base, **knobs) == stage
        assert tuple(knobs) == cake_backend.STAGE_KNOBS
    # the device-count form of the logits GEMMs (the hidden valid-row count's chunk 0), first in the suffix order
    assert cake_backend.stage_variant("gemm_logits", count=True) == "gemm_logits_mcnt"
    assert (
        cake_backend.stage_variant("gemm_logits_nostats", count=True)
        == "gemm_logits_nostats_mcnt"
    )
    assert cake_backend.parse_stage("gemm_logits_mcnt") == (
        "gemm_logits",
        {
            "count": True,
            "sliced": False,
            "tile_n": None,
            "stages": None,
            "split": False,
            "swab": False,
            "blocked": False,
            "epi_store": None,
        },
    )
    assert cake_backend.base_stage("gemm_logits_nostats_mcnt") == "gemm_logits_nostats"
    # the K-sliced dX GEMM (the slice count is a launch scalar), the narrow tile, the 3-deep ring of the wide tile
    assert cake_backend.stage_variant("gemm_dx", sliced=True) == "gemm_dx_s"
    assert (
        cake_backend.stage_variant("gemm_dx", sliced=True, tile_n=256)
        == "gemm_dx_s_tn256"
    )
    assert cake_backend.stage_variant("gemm_dx", tile_n=512) == "gemm_dx"
    assert (
        cake_backend.stage_variant("gemm_dx", sliced=True, stages=3) == "gemm_dx_s_st3"
    )
    assert cake_backend.stage_variant("gemm_dx", stages=3, tile_n=512) == "gemm_dx_st3"
    assert (
        cake_backend.stage_variant("gemm_dx", stages=4) == "gemm_dx"
    )  # the table depth
    assert cake_backend.parse_stage("gemm_dx_s_st3") == (
        "gemm_dx",
        {
            "count": False,
            "sliced": True,
            "tile_n": None,
            "stages": 3,
            "split": False,
            "swab": False,
            "blocked": False,
            "epi_store": None,
        },
    )
    assert cake_backend.parse_stage("gemm_dx_tn256")[1]["tile_n"] == 256
    # the wide dX tile's split-tail form (its last wave's trailing items as half items)
    assert cake_backend.stage_variant("gemm_dx", split=True) == "gemm_dx_split"
    assert (
        cake_backend.stage_variant("gemm_dx", sliced=True, stages=3, split=True)
        == "gemm_dx_s_st3_split"
    )
    assert cake_backend.parse_stage("gemm_dx_st3_split")[1] == dict(
        count=False,
        sliced=False,
        tile_n=None,
        stages=3,
        split=True,
        swab=False,
        blocked=False,
        epi_store=None,
    )
    # the 2-D blocked weight-gradient raster (the block width is a launch scalar)
    assert cake_backend.stage_variant("gemm_dw_acc", blocked=True) == "gemm_dw_acc_gn"
    assert (
        cake_backend.stage_variant("gemm_dw_cast_f32", blocked=True)
        == "gemm_dw_cast_f32_gn"
    )
    assert cake_backend.parse_stage("gemm_dw_cast_f32_gn") == (
        "gemm_dw_cast_f32",
        dict(
            count=False,
            sliced=False,
            tile_n=None,
            stages=None,
            split=False,
            swab=False,
            blocked=True,
            epi_store=None,
        ),
    )
    assert cake_backend.stage_variant("gemm_dw_acc", epi_store="redsm") == "gemm_dw_acc"
    assert (
        cake_backend.stage_variant("gemm_dw_acc", blocked=True, epi_store="tma")
        == "gemm_dw_acc_gn_tma"
    )  # grammar only: no rule selects the accumulate's reduce-add and no record registers it
    # the TMA-store form of the bf16 fused cast (dw_cast_tma_rule) and the swapped wide dX tile (dx_swab_rule)
    assert (
        cake_backend.stage_variant("gemm_dw_cast_bf16", epi_store="tma")
        == "gemm_dw_cast_bf16_tma"
    )
    assert cake_backend.parse_stage("gemm_dw_cast_bf16_tma") == (
        "gemm_dw_cast_bf16",
        dict(
            count=False,
            sliced=False,
            tile_n=None,
            stages=None,
            split=False,
            swab=False,
            blocked=False,
            epi_store="tma",
        ),
    )
    assert cake_backend.stage_variant("gemm_dx", swab=True) == "gemm_dx_swab"
    assert (
        cake_backend.stage_variant("gemm_dx", sliced=True, swab=True)
        == "gemm_dx_s_swab"
    )
    assert (
        cake_backend.stage_variant("gemm_dx", stages=4, tile_n=512, swab=True)
        == "gemm_dx_swab"
    )  # the table depth and the wide tile are the defaults
    assert cake_backend.parse_stage("gemm_dx_s_swab") == (
        "gemm_dx",
        dict(
            count=False,
            sliced=True,
            tile_n=None,
            stages=None,
            split=False,
            swab=True,
            blocked=False,
            epi_store=None,
        ),
    )
    assert (
        cake_backend.dx_stage(2) == "gemm_dx_s"
        and cake_backend.dx_stage(4) == "gemm_dx_s"
        and cake_backend.dx_stage(1, 256) == "gemm_dx_tn256"
        and cake_backend.dx_stage(3, 512, 3) == "gemm_dx_s_st3"
        and cake_backend.dx_stage(1) == "gemm_dx"
        and cake_backend.dx_stage(1, split=True) == "gemm_dx_split"
        and cake_backend.dx_stage(2, None, 3, split=True) == "gemm_dx_s_st3_split"
        and cake_backend.dx_stage(1, swab=True) == "gemm_dx_swab"
        and cake_backend.dx_stage(3, swab=True) == "gemm_dx_s_swab"
    )
    for bad_k in (0, 5):
        with pytest.raises(ValueError):
            cake_backend.dx_stage(bad_k)
    for bad in (
        dict(base="gemm_dx", tile_n=256, stages=3),  # the narrow tile keeps its depth
        dict(base="gemm_dx", stages=5),
        dict(base="gemm_dw_acc", stages=3),
        dict(
            base="gemm_dw_cast_bf16", blocked=True
        ),  # the bf16 cast keeps the 1-D raster
        dict(base="gemm_logits", blocked=True),
        dict(base="gemm_dx", blocked=True),
        dict(
            base="gemm_dx", count=True
        ),  # the device-count form exists for the logits GEMMs only
        dict(base="gemm_dw_acc", count=True),
        dict(base="row_finalize", count=True),
        dict(
            base="gemm_logits", sliced=True
        ),  # the K-sliced form exists for the dX GEMM only
        dict(base="gemm_dw_acc", sliced=True),
        dict(base="gemm_dx", tile_n=128),
        dict(
            base="gemm_dw_cast_f32", epi_store="tma"
        ),  # the TMA store is the bf16 cast's
        dict(base="gemm_dw_cast_f32", blocked=True, epi_store="tma"),
        dict(base="gemm_logits", epi_store="tma"),
        dict(base="gemm_dw_acc", epi_store="cast_bf16"),
        dict(base="row_grad", tile_n=256),
        dict(base="other"),
        dict(
            base="gemm_dx", tile_n=256, split=True
        ),  # the split-tail form exists for the wide tile only
        dict(base="gemm_logits", split=True),
        dict(
            base="gemm_dx", split=True, swab=True
        ),  # the two wide-tile forms exclude each other
        dict(base="gemm_dx", sliced=True, split=True, swab=True),
        dict(
            base="gemm_dx", stages=3, swab=True
        ),  # the swapped tile runs the table ring depth only
        dict(base="gemm_dx", tile_n=256, swab=True),  # ... on the wide tile only
        dict(base="gemm_dw_acc", swab=True),
        dict(base="gemm_logits", swab=True),
    ):
        with pytest.raises(ValueError):
            cake_backend.stage_variant(bad.pop("base"), **bad)
    for bad_name in (
        "gemm_dw_cast_bf16_gn",
        "gemm_dx_tn256_st3",
        "gemm_dx_gn",
        "gemm_dx_mcnt",
        "gemm_logits_s",
        "gemm_logits_mcnt_s",
        "gemm_dx_s2",  # the slice count is a launch scalar, not a name
        "gemm_dx_s3_tn256",
        "gemm_logits_g16",  # so is the raster height
        "gemm_dw_acc_g32",
        "gemm_dw_acc_gn12",  # and the block width
        "gemm_dw_acc_tma_gn",  # the suffix order
        "gemm_dx_st3_s",
        "gemm_dx_tn128",
        "row_grad_gn",
        "scale_cast",
        "gemm_dw_cast_f32_tma",  # the TMA store is the bf16 cast's
        "gemm_dw_cast_bf16_tma_gn",  # the suffix order
        "gemm_dx_split_st3",  # the suffix order
        "gemm_dx_swab_split",
        "gemm_dx_swab_s",
        "gemm_dx_split_swab",  # the two wide-tile forms exclude each other
        "gemm_dx_st3_swab",  # the table ring depth only
        "gemm_dx_tn256_swab",  # the wide tile only
        "gemm_dx_tn256_split",
    ):
        with pytest.raises(ValueError):
            cake_backend.parse_stage(bad_name)
    with pytest.raises(ValueError):
        cake_backend.base_stage("gemm_dx_s9")


def test_gemm_scalars_mirror_the_kernel_contract():
    """``gemm_scalars``: the launch scalars of a GEMM form from the record geometry, the call geometry and the rules'
    outputs -- the logits GEMM stores (first_chunk 0) with K = H, the dX GEMM stores its rows (first_chunk 1) with K = V
    split into ``k_slices`` slices of ``k_slice_iters`` K steps, the weight-gradient GEMM runs K = rows_c and takes the
    chunk's ``first_chunk``; ``group_m`` defaults to the GEMM's height, ``group_n`` to the whole row of column items (the
    blocked form takes the block width in column tiles)."""
    g = Geometry.from_record(
        {
            "geometry": {
                "logits_item_cols": 256,
                "dx_item_cols": 512,
                "dw_item_cols": 512,
                "logits_group_m": 32,
                "dx_group_m": 16,
                "dw_group_m": 2,
            }
        }
    )
    gs = cake_backend.gemm_scalars
    H, V = 6144, 154880
    logits = gs("gemm_logits", g, hidden=H, vocab=V, rows_c=4096)
    assert tuple(logits) == cake_backend.GEMM_SCALARS
    assert logits == dict(
        M=4096,
        m_tiles=32,
        k_iters=96,
        first_chunk=0,
        ws_slab=0,
        ldc=V,
        k_slices=1,
        k_slice_iters=96,
        group_m=32,
        group_n=605,
    )
    assert gs(
        "gemm_logits_mcnt", g, hidden=H, vocab=V, rows_c=4096, group_m=16
    ) == dict(logits, group_m=16)
    assert gs(
        "gemm_logits_nostats", g, hidden=H, vocab=V, rows_c=3943, first_chunk=1
    ) == dict(logits, M=3943, first_chunk=0)  # the logits GEMM never accumulates
    dx = gs(
        "gemm_dx_s", g, hidden=H, vocab=V, rows_c=4096, k_slices=4, ws_slab=4096 * H
    )
    assert dx == dict(
        M=4096,
        m_tiles=32,
        k_iters=2420,
        first_chunk=1,
        ws_slab=4096 * H,
        ldc=H,
        k_slices=4,
        k_slice_iters=605,
        group_m=16,
        group_n=12,
    )
    assert gs("gemm_dx_tn256", g, hidden=H, vocab=V, rows_c=1) == dict(
        dx, M=1, m_tiles=2, ws_slab=0, k_slices=1, k_slice_iters=2420
    )
    assert gs("gemm_dx_s_st3", g, hidden=H, vocab=V, rows_c=8192, k_slices=3)[
        "k_slice_iters"
    ] == -(-2420 // 3)
    dw = gs("gemm_dw_acc", g, hidden=H, vocab=V, rows_c=4096, first_chunk=0)
    assert dw == dict(
        M=V,
        m_tiles=1210,
        k_iters=64,
        first_chunk=0,
        ws_slab=0,
        ldc=H,
        k_slices=1,
        k_slice_iters=64,
        group_m=2,
        group_n=12,
    )
    assert gs(
        "gemm_dw_acc_gn",
        g,
        hidden=H,
        vocab=V,
        rows_c=4096,
        group_m=16,
        group_n=12,
        first_chunk=0,
    ) == dict(dw, group_m=16, group_n=6)  # 12 column tiles = 6 two-tile work items
    assert gs(
        "gemm_dw_cast_f32_gn",
        g,
        hidden=H,
        vocab=V,
        rows_c=3943,
        group_n=12,
        first_chunk=True,
    ) == dict(dw, k_iters=62, k_slice_iters=62, first_chunk=1, group_n=6)
    assert gs(
        "gemm_dw_cast_bf16", g, hidden=H, vocab=V, rows_c=1, first_chunk=1
    ) == dict(dw, k_iters=1, k_slice_iters=1, first_chunk=1)
    # a 4-CTA dX cluster rounds m_tiles up to it
    g4 = Geometry.from_record({"geometry": {"dx_cluster_ctas": 4, "dx_item_cols": 512}})
    assert gs("gemm_dx", g4, hidden=H, vocab=V, rows_c=4097)["m_tiles"] == 36
    for bad in (
        dict(stage="gemm_dx", k_slices=2),  # the plain form runs one slice
        dict(stage="gemm_dx_s", k_slices=1),  # the K-sliced form more than one
        dict(stage="gemm_dx_s", k_slices=5),
        dict(stage="gemm_logits", k_slices=2),
        dict(stage="gemm_dw_acc", k_slices=2),
        dict(stage="gemm_dw_acc", group_n=12),  # the 1-D raster takes no block width
        dict(stage="gemm_dw_acc_gn"),  # the blocked form needs one
        dict(stage="gemm_dw_acc_gn", group_n=3),  # whole two-tile work items
        dict(stage="gemm_dw_acc_gn", group_n=10),  # must divide the 12 column items
        dict(stage="gemm_dw_acc_gn", group_n=48),  # at most the row of items
        dict(stage="gemm_logits", group_m=0),
        dict(
            stage="gemm_dx_s", k_slices=4, hidden=7168
        ),  # H % 512 = 0 but V % 512 != 0 -> K steps OK, N OK
    ):
        kw = dict(hidden=H, vocab=V, rows_c=4096)
        kw.update(bad)
        stage = kw.pop("stage")
        if stage == "gemm_dx_s" and kw.get("hidden") == 7168:
            assert gs(stage, g, **kw)["ldc"] == 7168  # admissible: not an error
            continue
        with pytest.raises(ValueError):
            gs(stage, g, **kw)
    with pytest.raises(ValueError):
        gs("gemm_dx", g, hidden=6144 + 256, vocab=V, rows_c=4096)  # H % dx_item_cols
    with pytest.raises(ValueError):
        gs(
            "gemm_logits", g, hidden=H, vocab=V + 128, rows_c=4096
        )  # V % logits_item_cols
    with pytest.raises(ValueError):
        gs(
            "gemm_dx_s", g, hidden=H, vocab=128, rows_c=4096, k_slices=4
        )  # 4 slices over 2 K steps
    with pytest.raises(ValueError):
        gs("row_grad", g, hidden=H, vocab=V, rows_c=4096)


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
    assert cake_backend.DX_SPLIT_ARCHES == ("sm_100a", "sm_103a")
    assert cake_backend.DX_SPLIT_LONG_RING_ARCHES == ("sm_103a",)
    assert cake_backend.DX_SPLIT_WHOLE_ARCHES == ("sm_103a",)
    assert cake_backend.DX_SPLIT_WHOLE_FEW_CHUNKS_ARCHES == ("sm_100a",)
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
    for i in range(len(plan.chunks)):
        v = plan.variants_of(i)
        assert (v.logits_group_m, v.dw_group_m, v.dw_epi_store) == (16, 32, None)
        # the raster heights are launch scalars: the base forms run every chunk
        assert plan.logits_stage(i) == "gemm_logits"
        assert plan.logits_stage(i, stats=False) == "gemm_logits_nostats"
    # H 7168 on SM100: four K slices everywhere; the whole 4096-token chunks keep their full wide items (whole chunks
    # split only below H 7168), the 3943-row tail runs its last wave's 8 wide items as half items
    assert [
        (plan.dx_stage_of(i), plan.variants_of(i).dx_split, plan.dx_slices_of(i))
        for i in range(4)
    ] == [
        ("gemm_dx_s", None, 4),
        ("gemm_dx_s", None, 4),
        ("gemm_dx_s", None, 4),
        ("gemm_dx_s_split", 8, 4),
    ]
    assert all(plan.dw_acc_stage(i) == "gemm_dw_acc" for i in range(3))
    assert plan.dw_cast_stage == "gemm_dw_cast_bf16"
    assert "gemm_dw_acc_gn" not in plan.stages and "gemm_logits_mcnt" not in plan.stages
    with pytest.raises(IndexError):
        plan.logits_stage(4)
    assert {
        "gemm_logits",
        "gemm_dw_acc",
        "gemm_dw_cast_bf16",
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
    assert first == ["gemm_logits", plan.dx_stage_of(0), "gemm_dw_acc"]
    assert cake_backend.cast_keys(plan)[-1] == ("gemm_dw_cast_bf16", 3)
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
        "gemm_logits_nostats",
        logprob.dx_stage_of(3),
        "gemm_dw_cast_bf16",
    ]
    assert logprob.variants_of(3).dw_group_m == 32
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
        tail.variants_of(0).logits_group_m == 16  # 32 row tiles: the logits raster
        and tail.variants_of(1).logits_group_m is None
        and tail.dw_acc_stage(0) == "gemm_dw_acc_gn"
        and tail.variants_of(0).dw_group_n == 12
    )
    # the 1-row last chunk of a 2-chunk call: the bf16 cast's TMA store (dw_cast_tma_rule)
    assert tail.dw_cast_stage == "gemm_dw_cast_bf16_tma"
    assert [tail.variants_of(i).dw_cast_tma for i in range(2)] == [False, True]
    assert (
        tail.dx_slices_of(0) == 4
        and tail.dx_stage_of(0) == "gemm_dx_s_split"
        and tail.variants_of(0).dx_split == 28
    )  # 16 x 12 x 4 = 768 wide items on 74 SM pairs: the last wave's 28 items run as half items
    assert (
        tail.dx_slices_of(1) == 3 and tail.dx_stage_of(1) == "gemm_dx_s_tn256"
    )  # 1 x 12 x 3 = 36 < 74 pairs
    assert tail.stages == (
        "gemm_logits",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dx_s_tn256",
        "gemm_dx_s_split",
        "gemm_dw_acc_gn",
        "gemm_dw_cast_bf16_tma",
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
        big.dw_acc_stage(0) == "gemm_dw_acc"
        and big.variants_of(0).dw_group_m == 32
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
        long_chunks.variants_of(0).dw_group_m == 16
        and long_chunks.variants_of(1).dw_group_m is None
        and long_chunks.dw_acc_stage(0) == long_chunks.dw_acc_stage(1) == "gemm_dw_acc"
    )
    assert (
        long_chunks.variants_of(0).logits_group_m == 16
        and long_chunks.variants_of(1).logits_group_m is None
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
        == "gemm_dw_acc_gn"
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
        == "gemm_dw_acc_gn"
    )  # SM100 as well: the 2-D blocked raster at the C 4096 chunk
    assert (
        cake_backend.make_plan(
            _problem(8193, 6144, 154880, 8192),
            need_dx=False,
            need_dw=True,
            geometry=g0,
            num_sms=148,
            arch="sm_100a",
        )
        .variants_of(0)
        .dw_group_m
        == 16
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
        "gemm_logits",
        "row_finalize",
        "loss_reduce",
        "row_grad",
        "gemm_dw_cast_bf16",
    )
    assert one.dw_cast_stage == "gemm_dw_cast_bf16"
    assert (one.variants_of(0).logits_group_m, one.variants_of(0).dw_group_m) == (
        16,
        16,
    )
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
        deferred_tail.dw_acc_stage(0) == "gemm_dw_acc"
        and deferred_tail.variants_of(0).dw_group_m == 16
        and deferred_tail.dw_cast_stage == "gemm_dw_cast_bf16"
        and deferred_tail.variants_of(1).dw_group_m == 16
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
        )
        .variants_of(0)
        .dw_group_m
        == 16
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
        # both chunks take the 3-deep ring of the wide tile; the long ring splits on sm_103a only, the whole 8192-token chunk too
        assert [
            (ring.dx_stage_of(i), ring.variants_of(i).dx_split, ring.dx_slices_of(i))
            for i in range(2)
        ] == {
            "sm_103a": [("gemm_dx_s_st3_split", 32, 3), ("gemm_dx_s_st3_split", 32, 3)],
            "sm_100a": [("gemm_dx_s_st3", None, 2), ("gemm_dx_s_st3", None, 2)],
        }[arch]
        for i in range(2):
            assert ring.variants_of(i).dx_stages == 3
            assert (
                ring.variants_of(i).dw_group_n is None
            )  # above 4096 rows: the raster rule, not the block
        assert ring.variants_of(0).dw_group_m == 16
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
        (torch.float32, "gemm_dw_cast_f32_gn"),
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
            assert all(blocked.dw_acc_stage(i) == "gemm_dw_acc_gn" for i in range(4))
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
            # the wide tile's last wave of a partial token tile: the split-tail form where the rule splits
            split = tail_stage == "gemm_dx" and cake_backend.dx_split_rule(
                3591, 6144, sms, 1, g0, arch, None
            )
            assert tail_plan.dx_stage_of(last) == (
                "gemm_dx_split" if split else tail_stage
            )
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


def test_dx_split_eligible_whole_chunks_by_architecture_and_call_length():
    # a chunk that is not a whole number of 512-token tiles may always take the split rule
    assert cake_backend.dx_split_eligible(3943, 6144, "sm_100a", 512, 4)
    assert cake_backend.dx_split_eligible(3943, 6144, "sm_103a", 512, 4)
    # a whole 4096-token chunk below H 7168: every call on sm_103a; on sm_100a only a call of fewer than
    # DW_STREAM_MIN_CHUNKS chunks, whose dW accumulate shares the dX stream (an unknown chunk count: no)
    assert cake_backend.DW_STREAM_MIN_CHUNKS == 3
    for n in (1, 2, 3, 16, None):
        assert cake_backend.dx_split_eligible(4096, 6144, "sm_103a", 512, n)
    assert cake_backend.dx_split_eligible(4096, 6144, "sm_100a", 512, 1)
    assert cake_backend.dx_split_eligible(4096, 6144, "sm_100a", 512, 2)
    assert not cake_backend.dx_split_eligible(4096, 6144, "sm_100a", 512, 3)
    assert not cake_backend.dx_split_eligible(4096, 6144, "sm_100a", 512, None)
    # from H 7168 on a whole chunk keeps its full items on both architectures
    for arch in ("sm_100a", "sm_103a"):
        assert not cake_backend.dx_split_eligible(4096, 7168, arch, 512, 1)
        assert not cake_backend.dx_split_eligible(8192, 8192, arch, 512, 2)
    # the 256-wide tile is never a whole-tile chunk; the reference path (no architecture) never splits a whole chunk
    assert cake_backend.dx_split_eligible(4096, 6144, "sm_100a", 256, 4)
    assert not cake_backend.dx_split_eligible(4096, 6144, None, 512, 1)


# The dX form of every chunk at the default geometries, written out rather than derived from the rule functions: the
# chunk loops of the reference rows (their valid-row counts at 5 % ignored labels) and edge shapes, on SM100 (148 SMs)
# and SM103 (152 SMs).  Key: (arch, H, V, rows of the chunk loop, chunk); value: runs of (chunks, rows of each chunk, dX
# stage, split count F or None, K slices).  The forms do not depend on the entry or the objective, except on SM103
# (DX_SWAB_ARCHES): a cross-entropy loss call of >= 3 chunks runs its whole 512-token-tile chunks as the swapped tile
# (_DX_FORMS_SM103_CE below).
_DX_FORMS = {
    ("sm_100a", 6144, 154880, 1, 4096): ((1, 1, "gemm_dx_s_tn256", None, 3),),
    ("sm_100a", 6144, 154880, 1946, 4096): ((1, 1946, "gemm_dx_s", None, 3),),
    ("sm_100a", 6144, 154880, 3891, 4096): ((1, 3891, "gemm_dx_s_split", 28, 4),),
    ("sm_100a", 6144, 154880, 4096, 4096): ((1, 4096, "gemm_dx_s_split", 28, 4),),
    ("sm_100a", 6144, 154880, 4097, 4096): (
        (1, 4096, "gemm_dx_s_split", 28, 4),
        (1, 1, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_100a", 6144, 154880, 7711, 4096): (
        (1, 4096, "gemm_dx_s_split", 28, 4),
        (1, 3615, "gemm_dx_tn256", None, 1),
    ),
    ("sm_100a", 6144, 154880, 8192, 4096): ((2, 4096, "gemm_dx_s_split", 28, 4),),
    ("sm_100a", 6144, 154880, 11362, 4096): (
        (2, 4096, "gemm_dx_s", None, 4),
        (1, 3170, "gemm_dx_s_split", 32, 4),
    ),
    ("sm_100a", 6144, 154880, 15363, 4096): (
        (3, 4096, "gemm_dx_s", None, 4),
        (1, 3075, "gemm_dx_s_split", 32, 4),
    ),
    ("sm_100a", 6144, 154880, 15419, 4096): (
        (3, 4096, "gemm_dx_s", None, 4),
        (1, 3131, "gemm_dx_s_split", 32, 4),
    ),
    ("sm_100a", 6144, 154880, 16231, 4096): (
        (3, 4096, "gemm_dx_s", None, 4),
        (1, 3943, "gemm_dx_s_split", 28, 4),
    ),
    ("sm_100a", 6144, 154880, 30840, 4096): (
        (7, 4096, "gemm_dx_s", None, 4),
        (1, 2168, "gemm_dx_tn256", None, 1),
    ),
    ("sm_100a", 6144, 154880, 61779, 4096): (
        (15, 4096, "gemm_dx_s", None, 4),
        (1, 339, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_100a", 6144, 154880, 15419, 1024): (
        (15, 1024, "gemm_dx_s", None, 3),
        (1, 59, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_100a", 6144, 154880, 15419, 2048): (
        (7, 2048, "gemm_dx_s", None, 3),
        (1, 1083, "gemm_dx_s_split", 32, 3),
    ),
    ("sm_100a", 6144, 154880, 15419, 8192): (
        (1, 8192, "gemm_dx_s_st3", None, 2),
        (1, 7227, "gemm_dx_s_st3", None, 2),
    ),
    ("sm_100a", 7168, 129280, 4097, 4096): (
        (1, 4096, "gemm_dx_s", None, 4),
        (1, 1, "gemm_dx_s_tn256", None, 2),
    ),
    ("sm_100a", 7168, 129280, 15419, 4096): (
        (3, 4096, "gemm_dx_s", None, 4),
        (1, 3131, "gemm_dx_split", 34, 1),
    ),
    ("sm_100a", 8192, 128256, 4097, 4096): (
        (1, 4096, "gemm_dx", None, 1),
        (1, 1, "gemm_dx_s_tn256", None, 2),
    ),
    ("sm_100a", 8192, 128256, 15419, 4096): (
        (3, 4096, "gemm_dx", None, 1),
        (1, 3131, "gemm_dx_s_split", 32, 3),
    ),
    ("sm_103a", 6144, 154880, 1, 4096): ((1, 1, "gemm_dx_s_tn256", None, 3),),
    ("sm_103a", 6144, 154880, 1946, 4096): ((1, 1946, "gemm_dx_s", None, 3),),
    ("sm_103a", 6144, 154880, 3891, 4096): ((1, 3891, "gemm_dx_s_split", 8, 4),),
    ("sm_103a", 6144, 154880, 4096, 4096): ((1, 4096, "gemm_dx_s_split", 8, 4),),
    ("sm_103a", 6144, 154880, 4097, 4096): (
        (1, 4096, "gemm_dx_s_split", 8, 4),
        (1, 1, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_103a", 6144, 154880, 7711, 4096): (
        (1, 4096, "gemm_dx_s_split", 8, 4),
        (1, 3615, "gemm_dx_s_split", 36, 4),
    ),
    ("sm_103a", 6144, 154880, 8192, 4096): ((2, 4096, "gemm_dx_s_split", 8, 4),),
    ("sm_103a", 6144, 154880, 11362, 4096): (
        (2, 4096, "gemm_dx_s_split", 8, 4),
        (1, 3170, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 15363, 4096): (
        (3, 4096, "gemm_dx_s_split", 8, 4),
        (1, 3075, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 15419, 4096): (
        (3, 4096, "gemm_dx_s_split", 8, 4),
        (1, 3131, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 16231, 4096): (
        (3, 4096, "gemm_dx_s_split", 8, 4),
        (1, 3943, "gemm_dx_s_split", 8, 4),
    ),
    ("sm_103a", 6144, 154880, 30840, 4096): (
        (7, 4096, "gemm_dx_s_split", 8, 4),
        (1, 2168, "gemm_dx_split", 32, 1),
    ),
    ("sm_103a", 6144, 154880, 61779, 4096): (
        (15, 4096, "gemm_dx_s_split", 8, 4),
        (1, 339, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_103a", 6144, 154880, 15419, 1024): (
        (15, 1024, "gemm_dx_s", None, 3),
        (1, 59, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_103a", 6144, 154880, 15419, 2048): (
        (7, 2048, "gemm_dx_s", None, 3),
        (1, 1083, "gemm_dx_s_split", 28, 3),
    ),
    ("sm_103a", 6144, 154880, 15419, 8192): (
        (1, 8192, "gemm_dx_s_st3_split", 12, 3),
        (1, 7227, "gemm_dx_s_st3", None, 3),
    ),
    ("sm_103a", 7168, 129280, 4097, 4096): (
        (1, 4096, "gemm_dx", None, 1),
        (1, 1, "gemm_dx_s_tn256", None, 2),
    ),
    ("sm_103a", 7168, 129280, 15419, 4096): (
        (3, 4096, "gemm_dx", None, 1),
        (1, 3131, "gemm_dx_split", 30, 1),
    ),
    ("sm_103a", 8192, 128256, 4097, 4096): (
        (1, 4096, "gemm_dx_s", None, 4),
        (1, 1, "gemm_dx_s_tn256", None, 2),
    ),
    ("sm_103a", 8192, 128256, 15419, 4096): (
        (3, 4096, "gemm_dx_s", None, 4),
        (1, 3131, "gemm_dx_s_split", 36, 2),
    ),
}


# The SM103 cross-entropy loss plans that differ from _DX_FORMS: the chunk loops of >= 3 chunks; every chunk of whole
# 512-token tiles on the wide tile runs the swapped tile (no split, the table ring depth, the same K slices), the
# other chunks keep their forms.  Calls of 1 or 2 chunks, the policy objective and the log-probability entry point keep
# _DX_FORMS.
_DX_FORMS_SM103_CE = {
    ("sm_103a", 6144, 154880, 11362, 4096): (
        (2, 4096, "gemm_dx_s_swab", None, 4),
        (1, 3170, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 15363, 4096): (
        (3, 4096, "gemm_dx_s_swab", None, 4),
        (1, 3075, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 15419, 4096): (
        (3, 4096, "gemm_dx_s_swab", None, 4),
        (1, 3131, "gemm_dx_s_split", 16, 4),
    ),
    ("sm_103a", 6144, 154880, 16231, 4096): (
        (3, 4096, "gemm_dx_s_swab", None, 4),
        (1, 3943, "gemm_dx_s_split", 8, 4),
    ),
    ("sm_103a", 6144, 154880, 30840, 4096): (
        (7, 4096, "gemm_dx_s_swab", None, 4),
        (1, 2168, "gemm_dx_split", 32, 1),
    ),
    ("sm_103a", 6144, 154880, 61779, 4096): (
        (15, 4096, "gemm_dx_s_swab", None, 4),
        (1, 339, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_103a", 6144, 154880, 15419, 1024): (
        (15, 1024, "gemm_dx_s_swab", None, 3),
        (1, 59, "gemm_dx_s_tn256", None, 3),
    ),
    ("sm_103a", 6144, 154880, 15419, 2048): (
        (7, 2048, "gemm_dx_s_swab", None, 3),
        (1, 1083, "gemm_dx_s_split", 28, 3),
    ),
    ("sm_103a", 7168, 129280, 15419, 4096): (
        (3, 4096, "gemm_dx_swab", None, 1),
        (1, 3131, "gemm_dx_split", 30, 1),
    ),
    ("sm_103a", 8192, 128256, 15419, 4096): (
        (3, 4096, "gemm_dx_s_swab", None, 4),
        (1, 3131, "gemm_dx_s_split", 36, 2),
    ),
}


@pytest.mark.parametrize("key", list(_DX_FORMS), ids=lambda k: "-".join(map(str, k)))
def test_make_plan_dx_forms_of_the_default_geometries(key):
    arch, H, V, rows, C = key
    sms = {"sm_100a": 148, "sm_103a": 152}[arch]
    g = Geometry.from_record({"geometry": {"hidden": H, "vocab": V}})
    assert set(_DX_FORMS_SM103_CE) <= set(_DX_FORMS)
    for objective, entry in (("ce", "loss"), ("policy", "loss"), ("ce", "logprob")):
        forms = _DX_FORMS[key]
        if (objective, entry) == ("ce", "loss"):
            forms = _DX_FORMS_SM103_CE.get(key, forms)
        want = [(r, s, f, k) for n, r, s, f, k in forms for _ in range(n)]
        plan = cake_backend.make_plan(
            _problem(rows, H, V, C, objective=objective, entry=entry),
            need_dx=True,
            need_dw=True,
            geometry=g,
            dx_max_slices=4,
            num_sms=sms,
            dx_resident=sms // 2,
            fuse_dw_cast=True,
            arch=arch,
        )
        got = [
            (
                rows_c,
                plan.dx_stage_of(i),
                plan.variants_of(i).dx_split,
                plan.dx_slices_of(i),
            )
            for i, (_, rows_c) in enumerate(plan.chunks)
        ]
        assert got == want, (objective, entry)


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
        and pinned.variants_of(0).logits_group_m is None
        and pinned.dw_acc_stage(0) == "gemm_dw_acc"
        and pinned.variants_of(0).dw_group_m is None
    )
    assert pinned.dw_cast_stage == "gemm_dw_cast_bf16"
    assert all(
        pinned.dx_stage_of(i) == "gemm_dx_s_tn256" and pinned.dx_slices_of(i) == 2
        for i in range(4)
    )
    partial = cake_backend.make_plan(problem, gemm_tuning={"dx": {"k_slices": 2}}, **kw)
    assert (
        partial.variants_of(0).logits_group_m == 16
    )  # the rules still decide the knobs not named
    assert partial.variants_of(0).dw_group_m == 32
    assert all(partial.dx_slices_of(i) == 2 for i in range(4)) and all(
        s.startswith("gemm_dx_s") for s in partial.stages if s.startswith("gemm_dx")
    )
    # a present ``null`` pins the default form -- one dX slice, the base ``gemm_dx`` stage -- like every other
    # pinned knob; an absent knob keeps the slice rule
    one = cake_backend.make_plan(problem, gemm_tuning={"dx": {"k_slices": None}}, **kw)
    ruled = cake_backend.make_plan(problem, gemm_tuning={"dx": {}}, **kw)
    unpinned = cake_backend.make_plan(problem, **kw)
    assert len(one.chunks) == 4
    # one pinned slice: 16 x 14 = 224 wide items per whole chunk on 74 SM pairs (r = 2) keep their full items; the
    # 3943-row tail splits its last wave's 2 items
    assert [
        (one.dx_stage_of(i), one.variants_of(i).dx_split, one.dx_slices_of(i))
        for i in range(4)
    ] == [
        ("gemm_dx", None, 1),
        ("gemm_dx", None, 1),
        ("gemm_dx", None, 1),
        ("gemm_dx_split", 2, 1),
    ]
    for i in range(4):
        assert ruled.dx_slices_of(i) == unpinned.dx_slices_of(i)
    # the dX split-tail form: a pin wins over the rule
    plain = cake_backend.make_plan(problem, gemm_tuning={"dx": {"split": None}}, **kw)
    assert all(plain.variants_of(i).dx_split is None for i in range(4))
    assert all(not s.endswith("_split") for s in plain.stages)
    halved = cake_backend.make_plan(
        problem, gemm_tuning={"dx": {"tile_n": 512, "split": 2}}, **kw
    )
    assert all(halved.variants_of(i).dx_split == 2 for i in range(4))
    assert all(halved.dx_stage_of(i).endswith("_split") for i in range(4))
    # the swapped dX tile's knob is ``swap_ab`` (validated as the other integer knobs)
    assert cake_backend._resolve_gemm_tuning(
        {"dx": {"tile_n": 512, "swap_ab": False}}
    ) == {"dx": {"tile_n": 512, "swap_ab": 0}}
    with pytest.raises(ValueError, match="must be a dict over the knobs"):
        cake_backend._resolve_gemm_tuning({"dx": {"swab": True}})
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
        )
        .variants_of(0)
        .dw_group_m
        is None
    )
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"epi_store": None}}, **long_kw
        )
        .variants_of(0)
        .dw_group_m
        == 16
    )
    # the 1-row last chunk of this 2-chunk call takes the bf16 cast's TMA store (dw_cast_tma_rule) unless a call pins
    # the weight-gradient epilogue: an explicit ``epi_store`` keeps the shipped cast (``tma`` = the accumulate's
    # reduce-add only)
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"group_m": None}}, **long_kw
        ).dw_cast_stage
        == "gemm_dw_cast_bf16_tma"
    )
    for epi in (None, "redsm", "tma"):
        pinned_epi = cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"epi_store": epi}}, **long_kw
        )
        assert pinned_epi.dw_cast_stage == "gemm_dw_cast_bf16", epi
        assert not pinned_epi.variants_of(1).dw_cast_tma
        assert pinned_epi.dw_acc_stage(0) == (
            "gemm_dw_acc_tma" if epi == "tma" else "gemm_dw_acc"
        )
    assert (
        cake_backend.make_plan(
            long_problem, gemm_tuning={"dw": {"group_m": 32}}, **long_kw
        )
        .variants_of(0)
        .dw_group_m
        == 32
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


def test_make_plan_explicit_dx_split_runs_1_to_w_wide_items():
    """An explicit dX ``split`` runs 1 .. W of the chunk's W wide items as half items (the ``_split`` grid carries
    ``W + F`` clusters): ``F = 1`` and ``F = W`` plan the split-tail form, a value below 1 or above W raises, ``None``
    and 0 plan the plain form."""
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
    problem = _problem(3943, 7168, 129280, 4096)  # one chunk of 3943 rows
    k = cake_backend.make_plan(problem, **kw).dx_slices_of(0)
    wide, _ = cake_backend.dx_wide_items(3943, 7168, 148, k, g)
    assert (
        k > 1 and wide == 16 * 14 * k
    )  # 16 row-tile pairs x 14 column tiles of 512 x k slices
    for split in (1, wide):
        plan = cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"split": split}}, **kw
        )
        assert plan.chunks == ((0, 3943),) and plan.variants_of(0).dx_split == split
        assert plan.dx_stage_of(0).endswith("_split")
    for split in (-1, wide + 1):
        with pytest.raises(ValueError, match=rf"split={split}:.* W = {wide}$"):
            cake_backend.make_plan(problem, gemm_tuning={"dx": {"split": split}}, **kw)
    for split in (None, 0):
        plain = cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"split": split}}, **kw
        )
        assert plain.variants_of(0).dx_split is None
        assert not plain.dx_stage_of(0).endswith("_split")
    # a plan without dX (the log-probability forward, a frozen X) runs no split form and does not check the knob
    # against a one-slice chunk's items: W of k > 1 slices exceeds them
    cake_backend.make_plan(
        problem, gemm_tuning={"dx": {"split": wide}}, **{**kw, "need_dx": False}
    )


def _plan_kw(arch, H=6144, V=154880):
    sms = {"sm_100a": 148, "sm_103a": 152}[arch]
    return dict(
        need_dx=True,
        need_dw=True,
        geometry=Geometry.from_record({"geometry": {"hidden": H, "vocab": V}}),
        dx_max_slices=4,
        num_sms=sms,
        dx_resident=sms // 2,
        fuse_dw_cast=True,
        arch=arch,
    )


_ENTRIES = (("ce", "loss"), ("policy", "loss"), ("ce", "logprob"))


def test_dw_cast_tma_rule():
    assert cake_backend.DW_CAST_TMA_MAX_ROWS == {"sm_100a": 1792, "sm_103a": 1792}
    for arch in ("sm_100a", "sm_103a"):
        for rows in (1, 59, 63, 64, 65, 339, 1083, 1536, 1792):
            assert cake_backend.dw_cast_tma_rule(2, rows, arch)
            assert cake_backend.dw_cast_tma_rule(16, rows, arch)
            assert not cake_backend.dw_cast_tma_rule(1, rows, arch)  # no old sum
        assert not cake_backend.dw_cast_tma_rule(2, 1793, arch)
        assert not cake_backend.dw_cast_tma_rule(2, 4096, arch)
        assert not cake_backend.dw_cast_tma_rule(2, 0, arch)
    assert not cake_backend.dw_cast_tma_rule(2, 1792, None)
    assert not cake_backend.dw_cast_tma_rule(2, 1792, "sm_90a")


@pytest.mark.parametrize("arch", ["sm_100a", "sm_103a"])
def test_make_plan_dw_cast_tma_runs_the_last_chunk_of_tails_up_to_1792(arch):
    """The bf16 fused cast of the deferred last chunk takes the TMA store when the call has >= 2 chunks and the last
    chunk has 1 .. 1792 rows, on both architectures and every entry / objective; a one-chunk call, a 1793-row tail,
    the fp32 cast, the unfused cast and the reference path keep their shipped forms; only the last chunk's variant
    changes."""
    kw = _plan_kw(arch)
    C = 2048
    for objective, entry in _ENTRIES:
        for tail, tma in (
            (1, True),
            (59, True),
            (1083, True),
            (1792, True),
            (1793, False),
            (C, False),
        ):
            for full in (1, 3):
                rows = full * C + tail
                plan = cake_backend.make_plan(
                    _problem(rows, 6144, 154880, C, objective, entry), **kw
                )
                assert plan.num_chunks == full + 1 and plan.last_chunk[1] == tail
                cast = "gemm_dw_cast_bf16_tma" if tma else "gemm_dw_cast_bf16"
                assert plan.dw_cast_stage == cast, (objective, entry, rows)
                assert cast in plan.stages
                assert ("gemm_dw_cast_bf16" in plan.stages) is not tma
                assert [plan.variants_of(i).dw_cast_tma for i in range(full + 1)] == [
                    False
                ] * full + [tma]
                assert all(
                    plan.variants_of(i).dw_epi_store is None for i in range(full + 1)
                )
        # a one-chunk call (no old FP32 sum) keeps the shipped cast at every row count
        for rows in (1, 59, 1792, 1793, C):
            one = cake_backend.make_plan(
                _problem(rows, 6144, 154880, C, objective, entry), **kw
            )
            assert one.num_chunks == 1 and one.dw_cast_stage == "gemm_dw_cast_bf16"
            assert not one.variants_of(0).dw_cast_tma
    # the fp32 fused cast, the unfused cast and the reference path keep their forms
    p32 = cake_backend.make_plan(
        _problem(C + 59, 6144, 154880, C, dtype=torch.float32), **kw
    )
    assert p32.dw_cast_stage.startswith("gemm_dw_cast_f32")
    assert not any(v.dw_cast_tma for v in p32.variants)
    unfused = cake_backend.make_plan(
        _problem(C + 59, 6144, 154880, C), **dict(kw, fuse_dw_cast=False)
    )
    assert unfused.dw_cast_stage == "scale_cast_bf16"
    assert not any(v.dw_cast_tma for v in unfused.variants)
    reference = cake_backend.make_plan(
        _problem(C + 59, 6144, 154880, C), **dict(kw, arch=None)
    )
    assert reference.dw_cast_stage == "gemm_dw_cast_bf16"


def test_make_plan_sm100_changes_only_the_last_cast():
    """On SM100 the r6 rules add only the last chunk's TMA cast: no chunk runs the swapped dX tile (no SM100 swab
    program), every variant other than the last chunk's ``dw_cast_tma`` is the one of a call that pins the shipped
    weight-gradient epilogue (``"dw": {"epi_store": None}`` -- the rules of the shipped plan), and the stage sets differ
    by the cast's name only."""
    for key in [k for k in _DX_FORMS if k[0] == "sm_100a"]:
        _, H, V, rows, C = key
        kw_k = _plan_kw("sm_100a", H, V)
        for objective, entry in _ENTRIES:
            problem = _problem(rows, H, V, C, objective, entry)
            plan = cake_backend.make_plan(problem, **kw_k)
            shipped = cake_backend.make_plan(
                problem, gemm_tuning={"dw": {"epi_store": None}}, **kw_k
            )
            assert not any(v.dx_swab for v in plan.variants)
            assert all("_swab" not in s for s in plan.stages)
            tma = cake_backend.dw_cast_tma_rule(
                plan.num_chunks, plan.last_chunk[1], "sm_100a"
            )
            assert plan.variants[:-1] == shipped.variants[:-1]
            assert plan.variants[-1] == replace(shipped.variants[-1], dw_cast_tma=tma)
            rename = {"gemm_dw_cast_bf16_tma": "gemm_dw_cast_bf16"}
            assert tuple(rename.get(s, s) for s in plan.stages) == shipped.stages
            assert (plan.dw_cast_stage == "gemm_dw_cast_bf16_tma") is tma


def test_dx_swab_rule():
    assert cake_backend.DX_SWAB_ARCHES == ("sm_103a",)
    rule = cake_backend.dx_swab_rule
    for rows in (512, 1024, 2048, 4096, 8192):
        for n in (3, 4, 16):
            assert rule(rows, 6144, "sm_103a", None, True, n)
            assert rule(rows, 6144, "sm_103a", 512, True, n)
    assert rule(4096, 7168, "sm_103a", None, True, 3) and rule(
        4096, 8192, "sm_103a", None, True, 3
    )
    for args in (
        (
            4096,
            6144,
            "sm_103a",
            None,
            True,
            2,
        ),  # fewer than DW_STREAM_MIN_CHUNKS chunks
        (4096, 6144, "sm_103a", None, True, 1),
        (4096, 6144, "sm_103a", None, True, None),  # an unknown chunk count
        (4096, 6144, "sm_103a", None, False, 4),  # not a cross-entropy loss call
        (4096, 6144, "sm_100a", None, True, 4),  # no SM100 swab program
        (4096, 6144, None, None, True, 4),
        (4096, 6144, "sm_103a", 256, True, 4),  # the narrow tile
        (4095, 6144, "sm_103a", None, True, 4),  # not whole 512-token tiles
        (3131, 6144, "sm_103a", None, True, 4),
        (4096, 2880, "sm_103a", None, True, 4),  # H not a multiple of the wide tile
    ):
        assert not rule(*args), args


def test_make_plan_swab_runs_whole_chunks_of_sm103_ce_calls_of_3_or_more_chunks():
    """SM103 cross-entropy loss calls of >= 3 chunks run every chunk of whole 512-token tiles as the swapped dX tile,
    ahead of the split-tail rule and the H >= 7168 forms (table ring depth, no split, the slice rule's K slices); the
    policy objective, the log-probability entry point, calls of 1 or 2 chunks and SM100 never do."""
    kw = _plan_kw("sm_103a")
    exact = cake_backend.make_plan(_problem(3 * 4096, 6144, 154880, 4096), **kw)
    assert exact.num_chunks == 3
    assert all(exact.variants_of(i).dx_swab for i in range(3))
    assert all(exact.dx_stage_of(i) == "gemm_dx_s_swab" for i in range(3))
    assert all(exact.variants_of(i).dx_split is None for i in range(3))
    assert {"gemm_dx_s_swab"} == {s for s in exact.stages if s.startswith("gemm_dx")}
    for objective, entry in _ENTRIES[1:]:
        other = cake_backend.make_plan(
            _problem(3 * 4096, 6144, 154880, 4096, objective, entry), **kw
        )
        assert not any(v.dx_swab for v in other.variants)
        assert all("_swab" not in s for s in other.stages)
    for rows in (4096, 8192, 8191):  # 1 and 2 chunks
        few = cake_backend.make_plan(_problem(rows, 6144, 154880, 4096), **kw)
        assert not any(v.dx_swab for v in few.variants), rows
    sm100 = cake_backend.make_plan(
        _problem(3 * 4096, 6144, 154880, 4096), **_plan_kw("sm_100a")
    )
    assert not any(v.dx_swab for v in sm100.variants)


# Amendment rows of the swab lever gate: their plans must not run the cast's TMA store (whole-call and 5 %-ignored
# valid-row chunk loops).
_SWAB_GATE_ROWS = {
    "t16231_c4096_ce": (6144, 154880, (16231, 15419)),
    "t16172_c4096_ce": (6144, 154880, (16172, 15363)),
    "t16231_c4096_ce_h8192v128256": (8192, 128256, (16231, 15419)),
}


@pytest.mark.parametrize("row", list(_SWAB_GATE_ROWS))
def test_make_plan_swab_gate_rows_have_no_tma_cast(row):
    H, V, row_counts = _SWAB_GATE_ROWS[row]
    for rows in row_counts:
        plan = cake_backend.make_plan(
            _problem(rows, H, V, 4096), **_plan_kw("sm_103a", H, V)
        )
        assert plan.num_chunks == 4 and plan.last_chunk[1] > 1792
        assert "gemm_dw_cast_bf16_tma" not in plan.stages
        assert plan.dw_cast_stage == "gemm_dw_cast_bf16"
        assert [plan.variants_of(i).dx_swab for i in range(4)] == [True] * 3 + [False]


def test_make_plan_explicit_dx_pins_and_the_swab_rule():
    """An explicit ``split`` (any value) or a ``stages`` other than the table depth keeps the pinned form and turns the
    swab rule off; ``stages`` 4 / ``None`` keep it on; ``swap_ab`` pins the swapped tile on or off on any chunk and
    objective; ``swap_ab=True`` with another ring depth or with a split raises."""
    kw = _plan_kw("sm_103a")
    problem = _problem(15419, 6144, 154880, 4096)  # 3 whole chunks + a 3131-row tail
    ruled = cake_backend.make_plan(problem, **kw)
    assert [ruled.dx_stage_of(i) for i in range(4)] == ["gemm_dx_s_swab"] * 3 + [
        "gemm_dx_s_split"
    ]
    split = cake_backend.make_plan(
        problem, gemm_tuning={"dx": {"tile_n": 512, "split": 2}}, **kw
    )
    assert all(
        split.dx_stage_of(i) == "gemm_dx_s_split" and split.variants_of(i).dx_split == 2
        for i in range(4)
    )
    assert not any(v.dx_swab for v in split.variants)
    plain = cake_backend.make_plan(problem, gemm_tuning={"dx": {"split": None}}, **kw)
    assert [plain.dx_stage_of(i) for i in range(4)] == ["gemm_dx_s"] * 4
    st3 = cake_backend.make_plan(problem, gemm_tuning={"dx": {"stages": 3}}, **kw)
    assert all(st3.dx_stage_of(i).startswith("gemm_dx_s_st3") for i in range(4))
    assert not any(v.dx_swab for v in st3.variants)
    for table_depth in (4, None):
        kept = cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"stages": table_depth}}, **kw
        )
        assert [kept.dx_stage_of(i) for i in range(4)] == [
            ruled.dx_stage_of(i) for i in range(4)
        ]
    off = cake_backend.make_plan(problem, gemm_tuning={"dx": {"swap_ab": False}}, **kw)
    none = cake_backend.make_plan(problem, gemm_tuning={"dx": {"swap_ab": None}}, **kw)
    policy = _problem(15419, 6144, 154880, 4096, "policy")
    assert (
        off.variants == none.variants == cake_backend.make_plan(policy, **kw).variants
    )
    assert [off.dx_stage_of(i) for i in range(4)] == ["gemm_dx_s_split"] * 4
    for p in (
        problem,
        policy,
    ):  # a pin runs every chunk swapped, the 3131-row tail and the policy objective included
        on = cake_backend.make_plan(p, gemm_tuning={"dx": {"swap_ab": True}}, **kw)
        assert [on.dx_stage_of(i) for i in range(4)] == ["gemm_dx_s_swab"] * 4
        on4 = cake_backend.make_plan(
            p, gemm_tuning={"dx": {"swap_ab": True, "stages": 4, "split": None}}, **kw
        )
        assert on4.variants == on.variants
    with pytest.raises(ValueError, match="swap_ab"):
        cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"swap_ab": True, "stages": 3}}, **kw
        )
    with pytest.raises(ValueError, match="exclude each other"):
        cake_backend.make_plan(
            problem,
            gemm_tuning={"dx": {"swap_ab": True, "tile_n": 512, "split": 2}},
            **kw,
        )
    with pytest.raises(ValueError):  # the swapped form exists for the wide tile only
        cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"swap_ab": True, "tile_n": 256}}, **kw
        ).dx_stage_of(0)


def test_make_plan_explicit_swap_ab_errors_early():
    """An explicit ``swap_ab=True`` applies to every chunk: ``make_plan`` raises a ``ValueError`` (not a missing stage at
    bind time) for a chunk whose dX tile the rule narrows (the message names the chunk's rows and tile and says to pin
    ``tile_n=512`` as well), for an architecture without the swapped program and for a ring depth other than the
    table's; with ``tile_n=512`` pinned too, every chunk -- the ragged tail included -- runs the swapped tile."""
    kw = _plan_kw("sm_103a")
    problem = _problem(3 * 2048 + 59, 6144, 154880, 2048)  # a ragged 59-row tail
    ruled = cake_backend.make_plan(problem, **kw)
    assert ruled.num_chunks == 4 and ruled.last_chunk[1] == 59
    narrow = [i for i in range(4) if ruled.variants_of(i).dx_tile_n is not None]
    assert narrow and all(ruled.variants_of(i).dx_tile_n == 256 for i in narrow)
    rows = ruled.chunks[narrow[0]][1]
    with pytest.raises(
        ValueError,
        match=rf"swap_ab=True: .* chunk of {rows} rows resolves tile_n=256 .*pin tile_n=512 as well",
    ):
        cake_backend.make_plan(problem, gemm_tuning={"dx": {"swap_ab": True}}, **kw)
    for objective in ("ce", "policy"):
        p = _problem(3 * 2048 + 59, 6144, 154880, 2048, objective)
        pinned = cake_backend.make_plan(
            p, gemm_tuning={"dx": {"swap_ab": True, "tile_n": 512}}, **kw
        )
        assert all(
            pinned.variants_of(i).dx_swab and pinned.variants_of(i).dx_tile_n is None
            for i in range(4)
        )
        assert all(pinned.dx_stage_of(i).endswith("_swab") for i in range(4))
    with pytest.raises(ValueError, match=r"4-deep table ring only, got stages=3"):
        cake_backend.make_plan(
            problem,
            gemm_tuning={"dx": {"swap_ab": True, "tile_n": 512, "stages": 3}},
            **kw,
        )
    kw100 = _plan_kw("sm_100a")
    for pin in ({"swap_ab": True}, {"swap_ab": True, "tile_n": 512}):
        with pytest.raises(ValueError, match="sm_100a has no swapped dX program"):
            cake_backend.make_plan(problem, gemm_tuning={"dx": pin}, **kw100)
    assert (
        cake_backend.make_plan(
            problem, gemm_tuning={"dx": {"swap_ab": False}}, **kw100
        ).variants
        == cake_backend.make_plan(problem, **kw100).variants
    )


def test_reachable_variants_r6_forms():
    bases = cake_jit.BASE_STAGES
    r103 = cake_backend.reachable_variants(bases, "sm_103a", 4)
    r100 = cake_backend.reachable_variants(bases, "sm_100a", 4)
    assert {"gemm_dx_swab", "gemm_dx_s_swab", "gemm_dw_cast_bf16_tma"} <= r103
    assert "gemm_dw_cast_bf16_tma" in r100
    assert not {s for s in r100 if "_swab" in s}


def test_gemm_tuning_default_is_no_knob():
    """``gemm_tuning`` is a keyword: ``None`` (the default) pins nothing and the backend reads no environment variable;
    the knobs are part of the binding key; an explicit knob plans the pinned variant on the reference path."""
    assert (
        cake_backend.gemm_tuning_default() == {}
        and cake_backend._resolve_gemm_tuning(None) == {}
    )
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
    default_key = forward_binding_key(
        X, W, labels, **common
    )  # the knobs are part of the binding
    assert default_key == forward_binding_key(X, W, labels, gemm_tuning={}, **common)
    assert default_key != forward_binding_key(
        X, W, labels, gemm_tuning={"dx": {"tile_n": 256}}, **common
    )
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


def test_toolchain_workaround_flags():
    """The CUDA 13.0 ptxas workaround (``-Xptxas -O0``) is build data: every sm_103a entry of ``build.json``
    carries it under ``nvcc_flags_if`` for CUDA 13.0 only, and no sm_100a entry carries it."""
    workaround = {"cuda": "==13.0", "flags": ["-Xptxas", "-O0"]}
    entries = cake_build.manifest()["libraries"]
    for arch in sorted({record["arch"] for record in cake_jit.MODULES.values()}):
        of_arch = [entry for entry in entries if entry["arch"] == arch]
        assert of_arch, arch
        for entry in of_arch:
            assert "-Xptxas" not in entry["nvcc_flags"], entry["name"]
            conditional = [
                item for item in entry["nvcc_flags_if"] if item["cuda"] == "==13.0"
            ]
            if arch == "sm_103a":
                assert conditional == [workaround], entry["name"]
            else:
                assert workaround not in entry["nvcc_flags_if"], entry["name"]


@pytest.fixture
def toolchain_check_uncached():
    """``cake_jit.toolchain_runs_hidden_count`` caches its answer per architecture for the process: a test that
    patches the toolkit version starts and ends with an empty cache, so no patched answer outlives it."""
    cake_jit.toolchain_runs_hidden_count.cache_clear()
    yield
    cake_jit.toolchain_runs_hidden_count.cache_clear()


def test_toolchain_runs_hidden_count(monkeypatch, toolchain_check_uncached):
    """On CUDA 13.0 the sm_103a calls take the host-count path; every other architecture / toolchain pair runs
    the hidden valid-row count.  The toolkit version is ``cake_build.record_cuda_version`` (the toolkit that
    built the prebuilt libraries, else the local one), read once per architecture and process."""
    for version, sm_103a in (
        ((13, 0), False),
        ((12, 9), True),
        ((13, 1), True),
        ((13, 4), True),
    ):
        monkeypatch.setattr(
            cake_build, "record_cuda_version", lambda arch, v=version: v
        )
        cake_jit.toolchain_runs_hidden_count.cache_clear()
        assert cake_jit.toolchain_runs_hidden_count("sm_103a") is sm_103a, version
        assert cake_jit.toolchain_runs_hidden_count("sm_100a") is True, version
    # the answer is cached: the manifest and the library files are read once per architecture and process
    calls = []

    def counted(arch):
        calls.append(arch)
        return (13, 4)

    monkeypatch.setattr(cake_build, "record_cuda_version", counted)
    cake_jit.toolchain_runs_hidden_count.cache_clear()
    for _ in range(3):
        assert cake_jit.toolchain_runs_hidden_count("sm_103a") is True
    assert calls == ["sm_103a"]


def test_select_module_one_record_per_arch(monkeypatch):
    def rec(arch):
        return {"arch": arch, "abi": ABI_CONTRACT, "geometry": {}, "stages": []}

    monkeypatch.setattr(
        cake_jit,
        "MODULES",
        {
            "cake_lm_head_loss_sm_100a": rec("sm_100a"),
            "cake_lm_head_loss_sm_103a": rec("sm_103a"),
        },
    )
    assert cake_jit.select_module("sm_100a") == "cake_lm_head_loss_sm_100a"
    assert cake_jit.select_module("sm_103a") == "cake_lm_head_loss_sm_103a"
    with pytest.raises(NotImplementedError):
        cake_jit.select_module("sm_90a")
    monkeypatch.setattr(cake_jit, "MODULES", {"a": rec("sm_100a"), "b": rec("sm_100a")})
    with pytest.raises(NotImplementedError, match="more than one"):
        cake_jit.select_module("sm_100a")


def test_reachable_variants_complete_the_rule_closure():
    bases = (
        set(stages_for_entry("loss"))
        | set(stages_for_entry("loss", grad_weight_dtype=torch.float32))
        | set(stages_for_entry("logprob"))
    )
    rv = cake_backend.reachable_variants
    dx_forms = {
        "gemm_dx",
        "gemm_dx_s",
        "gemm_dx_tn256",
        "gemm_dx_s_tn256",
        "gemm_dx_st3",
        "gemm_dx_s_st3",
        "gemm_dx_split",
        "gemm_dx_s_split",
        "gemm_dx_st3_split",
        "gemm_dx_s_st3_split",
    }
    dw_forms = {"gemm_dw_acc_gn", "gemm_dw_cast_f32_gn", "gemm_dw_cast_bf16_tma"}
    logits_forms = {"gemm_logits_mcnt", "gemm_logits_nostats_mcnt"}
    # the forms are structural: the same set at every geometry (H / V, the slice count, the raster height and the
    # block width are launch scalars of these kernels); SM103 adds the swapped dX tile (DX_SWAB_ARCHES)
    assert rv(bases, "sm_100a", 4) == dx_forms | dw_forms | logits_forms
    assert rv(bases, "sm_103a", 4) == rv(bases, "sm_100a", 4) | {
        "gemm_dx_swab",
        "gemm_dx_s_swab",
    }
    assert rv(bases, "sm_100a", 2) == rv(bases, "sm_100a", 4)
    assert rv(bases, "sm_103a", 1) == rv(bases, "sm_100a", 1) | {"gemm_dx_swab"}
    # a record without the K-sliced form serves one slice: the plain dX forms only
    assert (
        rv(bases, "sm_100a", 1)
        == {
            "gemm_dx",
            "gemm_dx_tn256",
            "gemm_dx_st3",
            "gemm_dx_split",
            "gemm_dx_st3_split",
        }
        | dw_forms
        | logits_forms
    )
    assert rv({"row_grad", "scale_cast_bf16"}, "sm_103a", 4) == set()
    # an architecture without the ring / block / split / swap rules reaches none of those forms
    assert (
        rv(bases, "sm_90a", 4)
        == {"gemm_dx", "gemm_dx_s", "gemm_dx_tn256", "gemm_dx_s_tn256"} | logits_forms
    )
    # every name of STAGES is a base stage or a form some record can reach
    assert set(cake_jit.STAGES) == (
        set(cake_jit.BASE_STAGES) | rv(bases, "sm_103a", 4) | rv(bases, "sm_100a", 4)
    )
    assert len(cake_jit.STAGES) == 32
    assert (
        cake_backend.dx_max_slices(cake_jit.STAGES) == cake_backend.DX_K_SLICES_MAX == 4
    )
    assert cake_backend.dx_max_slices(("gemm_dx", "gemm_dx_tn256")) == 1


def test_cluster_resident_rule():
    assert (
        cluster_resident(None, 2, 148) == 74
        and cluster_resident(None, 2, 152) == 76
        and cluster_resident(None, 1, 5) == 5
    )
    with pytest.raises(ValueError):  # only one- and two-CTA clusters are supported
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
            and v["k_iters"] == H_HOST // 64
        )
        # the launch scalars of the geometry-free kernels (``gemm_scalars``): the output's leading dimension, one K
        # slice, the default raster height and the whole row of column items
        assert (
            v["ldc"] == V_HOST
            and (v["k_slices"], v["k_slice_iters"]) == (1, v["k_iters"])
            and v["group_m"] == plan.geometry.logits_group_m
            and v["group_n"] == V_HOST // plan.geometry.logits_item_cols
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
        assert (
            (dx["k_iters"], dx["k_slices"], dx["k_slice_iters"])
            == (V_HOST // 64, 1, V_HOST // 64)
            and dx["ldc"] == H_HOST
            and dx["ws_slab"] == 0
            and (dx["group_m"], dx["group_n"])
            == (plan.geometry.dx_group_m, H_HOST // plan.geometry.dx_item_cols)
        )
        dw = stage_values("gemm_dw_acc", t, plan, index)
        assert (
            dw["M"] == V_HOST
            and dw["k_iters"] == -(-rows_c // 64)
            and dw["C"] is t["dw_acc"]
        )
        assert (
            dw["ldc"] == H_HOST
            and (dw["k_slices"], dw["k_slice_iters"]) == (1, dw["k_iters"])
            and (dw["group_m"], dw["group_n"])
            == (plan.geometry.dw_group_m, H_HOST // plan.geometry.dw_item_cols)
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
            "library": "cake_lm_head_loss_fake",
            "ffi_entry": "run",
            "arg_plan": arg_plan,
            "tma_workspace_bytes": 0,
            "workspace_bytes": 0,
            "grid": ["rows_c", 1, 1],
            "launch": {"block": [128, 1, 1], "cluster": [1, 1, 1]},
        },
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
        dict(hidden_count=True),
        dict(logprob_lbl=True),
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
        X,
        W,
        labels,
        lse,
        dlogp,
        chunk_size=4096,
        need_dx=True,
        need_dw=True,
        hidden_count=True,
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
        logprob_dz=True,
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


def test_binding_cache_configure():
    """The binding cache is configured through keywords and :meth:`BindingCache.configure` (the public module applies
    its two environment variables through it on its first backend use): a positive-integer capacity, evicting the least recently used
    bindings when it shrinks; ``enabled=False`` bypasses the cache and drops the remembered bindings."""
    assert BindingCache().capacity == cake_backend.BINDING_CACHE_DEFAULT_CAPACITY == 64
    assert BindingCache().enabled and cake_backend.BINDING_CACHE.capacity >= 1
    assert BindingCache(capacity=3).capacity == 3  # an explicit capacity wins
    cache = BindingCache(capacity=4)
    for i in range(4):
        cache.remember(("fwd", i), SimpleNamespace(owned_bytes=1))
    # a hit refreshes ("fwd", 0): ("fwd", 1) is now the least recently used
    assert cache.lookup(("fwd", 0)) is not None
    assert cache.configure(capacity=2) is cache and cache.capacity == 2
    assert len(cache) == 2 and list(cache._bindings) == [("fwd", 3), ("fwd", 0)]
    cache.configure(capacity=8)
    assert cache.capacity == 8 and len(cache) == 2  # growing evicts nothing
    cache.configure(enabled=False)
    assert not cache.enabled and len(cache) == 0 and cache.capacity == 8
    cache.configure(enabled=True)
    assert cache.enabled and cache.capacity == 8
    cache.configure()  # nothing changes
    assert cache.enabled and cache.capacity == 8
    for bad in (0, -1, "5", "many", 2.5, True, [5]):
        with pytest.raises(ValueError, match="positive integer"):
            cache.configure(capacity=bad)
    assert cache.capacity == 8
    with pytest.raises(ValueError, match="positive integer"):
        BindingCache(capacity=0)


def test_stage_names_and_gemm_scalars_return_fresh_dicts():
    """``parse_stage`` and ``gemm_scalars`` remember their results per argument tuple; every call still returns a dict
    of its own, so a caller that edits one changes nothing for the next call."""
    stage = "gemm_dx_s_tn256"
    base, knobs = cake_backend.parse_stage(stage)
    assert base == "gemm_dx" and knobs["sliced"] is True and knobs["tile_n"] == 256
    knobs["sliced"] = None
    assert cake_backend.parse_stage(stage)[1]["sliced"] is True
    with pytest.raises(ValueError, match="not a stage name"):
        cake_backend.parse_stage("gemm_dx_bogus")
    g = Geometry.from_record(None)
    scalars = cake_backend.gemm_scalars(
        "gemm_logits", g, hidden=DEFAULT_H, vocab=DEFAULT_V, rows_c=7
    )
    expect = dict(scalars)
    scalars["M"] = -1
    again = cake_backend.gemm_scalars(
        "gemm_logits", g, hidden=DEFAULT_H, vocab=DEFAULT_V, rows_c=7
    )
    assert again == expect and again["M"] == 7 and again is not scalars
    with pytest.raises(ValueError, match="group_m must be positive"):
        cake_backend.gemm_scalars(
            "gemm_logits", g, hidden=DEFAULT_H, vocab=DEFAULT_V, rows_c=7, group_m=0
        )


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
    with pytest.raises(ValueError, match="W.dtype"):
        cake_backend.chunked_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            loss_div=inp.loss_div,
            grad_weight_dtype=torch.float32,
            backend="reference",
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


def test_switch_defaults_are_constants():
    """Every switch is a keyword whose ``None`` selects a constant, documented default; the package reads no
    environment variable (the public module ``flashinfer.chunked_lm_head`` owns the A/B variables)."""
    assert cake_backend.compact_rows_default() is True
    assert cake_backend._resolve_compact(None) is True
    assert cake_backend._resolve_compact(False) is False
    assert cake_backend.fuse_dw_cast_default() is True
    assert cake_backend._resolve_fuse(None) is True
    assert cake_backend._resolve_fuse(False) is False
    assert "gemm_dw_cast_bf16" in stages_for_entry("loss")
    assert "gemm_dw_cast_bf16" not in stages_for_entry("loss", fuse_dw_cast=False)
    assert cake_backend.dx_finalize_default() is True
    assert cake_backend.DX_FINALIZE_STAGES == ("slab_sum", "scale_cast_scatter_bf16")
    assert cake_backend._resolve_dx_finalize(None) is True
    assert cake_backend._resolve_dx_finalize(False) is False
    assert cake_backend.hidden_count_default() is True
    assert cake_backend._resolve_hidden_count(None) is True
    assert cake_backend._resolve_hidden_count(False) is False
    assert cake_backend.HIDDEN_COUNT_STAGE == "gather_rows_bf16"
    assert cake_backend.HIDDEN_COUNT_STAGE in cake_jit.BASE_STAGES
    assert cake_backend.HIDDEN_COUNT_STAGE in cake_jit.ROW_STAGES
    assert STAGE_TENSORS[cake_backend.HIDDEN_COUNT_STAGE] == (
        "x",
        "idx_lo",
        "count",
        "out",
    )
    assert cake_backend.gemm_tuning_default() == {}
    for module in (cake_backend, cake_jit, cake_build, pkg):
        source = inspect.getsource(module)
        assert "os.environ" not in source and "getenv" not in source, module.__name__
    # the memory report and the stage list follow the same defaults
    assert memory_report(16231, DEFAULT_H, DEFAULT_V, 4096)["fuse_dw_cast"] is True
    assert memory_report(4097, DEFAULT_H, DEFAULT_V, 4096, valid_rows=3892)[
        "dx_finalize"
    ]


def test_switch_keywords_reach_the_autograd_entries():
    """The autograd entry points take every switch as a keyword and resolve it once: on the reference backend the
    non-default values of the inert switches give bitwise the default results, and an unknown ``dw_stream`` is a
    ``ValueError`` from both entries."""
    inp = _host_inputs(37)
    base = _run(inp, 16, compact_rows=True)
    other = _run(
        inp,
        16,
        compact_rows=True,
        dx_finalize=False,
        dw_stream=False,
        hidden_count=False,
        gemm_tuning={},
    )
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(base[key], other[key]), key
    lp = _run(inp, 16, entry="logprob", compact_rows=True)
    lp_other = _run(
        inp,
        16,
        entry="logprob",
        compact_rows=True,
        dx_finalize=False,
        dw_stream=True,
        hidden_count=False,
        gemm_tuning={},
    )
    for key in ("logp", "dX", "dW"):
        assert torch.equal(lp[key], lp_other[key]), key
    for entry in ("loss", "logprob"):
        with pytest.raises(ValueError, match="dw_stream"):
            _run(inp, 16, entry=entry, dw_stream="sometimes")


def test_compaction_index_matches_nonzero():
    """The device compaction (``labels >= 0`` -> inclusive int32 scan -> ``searchsorted``) gives the ascending valid rows
    in the first ``count`` slots (``== nonzero``) and ``T`` in every slot after them, for some / no / every row ignored
    and a one-row call; pre-allocated outputs and a longer cached constant give the same index."""
    for labels in (
        torch.tensor([3, IGNORE_INDEX, 0, IGNORE_INDEX, 7]),
        torch.tensor([IGNORE_INDEX] * 4),
        torch.tensor([1, 2, 3]),
        torch.tensor([5]),
        torch.tensor([IGNORE_INDEX]),
        torch.tensor([IGNORE_INDEX, IGNORE_INDEX, 9]),
    ):
        T = labels.numel()
        valid, pos, idx_full = cake_backend.compaction_index(labels)
        assert torch.equal(valid, labels >= 0) and valid.dtype == torch.bool
        assert pos.dtype == torch.int32 and torch.equal(
            pos, torch.cumsum(labels >= 0, 0, dtype=torch.int32)
        )
        count = int(pos[-1])
        assert idx_full.dtype == torch.int64 and tuple(idx_full.shape) == (T,)
        assert torch.equal(idx_full[:count], (labels >= 0).nonzero().reshape(-1))
        assert torch.all(idx_full[count:] == T)
        outs = dict(
            valid=torch.empty(T, dtype=torch.bool),
            pos=torch.empty(T, dtype=torch.int32),
            idx_full=torch.empty(T, dtype=torch.int64),
        )
        ar = torch.arange(1, T + 9, dtype=torch.int32)
        v2, p2, i2 = cake_backend.compaction_index(labels, ar=ar, **outs)
        assert v2 is outs["valid"] and p2 is outs["pos"] and i2 is outs["idx_full"]
        assert (
            torch.equal(v2, valid)
            and torch.equal(p2, pos)
            and torch.equal(i2, idx_full)
        )


def test_hidden_count_plan_keys():
    """A plan of the hidden valid-row count skips chunk 0's host gather and logits GEMM (both ran in device-count form
    before the count was known), records the device-count stage at the buffer extent ``min(C, T)`` and lists it with
    the gather kernel in its stages; the log-probability backward does the same with the recompute form; the plan
    needs the program's architecture."""
    g = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        dx_finalize=True,
    )
    problem = _problem(8700, 6144, 154880, 4096)
    plain = cake_backend.make_plan(problem, valid_rows=8265, arch="sm_100a", **common)
    hidden = cake_backend.make_plan(
        problem, valid_rows=8265, arch="sm_100a", hidden_count=True, **common
    )
    assert not plain.hidden_count and hidden.hidden_count and hidden.hidden_stats
    assert hidden.chunks == plain.chunks == ((0, 4096), (4096, 4096), (8192, 73))
    # the buffer extent min(C, T) = 4096 rows takes the long logits raster: the device-count form with group_m 16 as
    # a launch scalar
    assert hidden.hidden_logits_group_m == 16
    assert hidden.hidden_logits_stage() == "gemm_logits_mcnt"
    keys, plain_keys = (
        cake_backend.forward_keys(hidden),
        cake_backend.forward_keys(plain),
    )
    assert plain_keys[:3] == (
        ("gather_rows", 0),
        ("gemm_logits", 0),
        ("row_finalize", 0),
    )
    assert keys == plain_keys[2:]  # chunk 0's gather and logits GEMM are already queued
    assert ("gather_rows", 1) in keys and ("gemm_logits", 1) in keys
    assert set(hidden.stages) == set(plain.stages) | {
        "gather_rows_bf16",
        "gemm_logits_mcnt",
    }
    assert list(hidden.stages) == [s for s in cake_jit.STAGES if s in hidden.stages]
    assert cake_backend.cast_keys(hidden) == cake_backend.cast_keys(plain)
    # the log-probability backward: the recompute GEMM without the statistics, chunk 0's already queued
    lp = cake_backend.make_plan(
        _problem(8700, 6144, 154880, 4096, entry="logprob"),
        valid_rows=8265,
        arch="sm_100a",
        hidden_count=True,
        hidden_stats=False,
        **common,
    )
    rk = cake_backend.recompute_keys(lp)
    prk = cake_backend.recompute_keys(replace(lp, hidden_count=False))
    assert prk[:3] == (
        ("gather_rows", "d_in"),
        ("gather_rows", 0),
        ("gemm_logits_nostats", 0),
    )
    assert rk == prk[:1] + prk[3:]
    assert lp.hidden_logits_stage() == "gemm_logits_nostats_mcnt"
    assert "gemm_logits_nostats_mcnt" in lp.stages
    # every row valid: the uncompacted plan with chunk 0's GEMM skipped; 1500 rows take the default raster
    full = cake_backend.make_plan(
        _problem(1500, 6144, 154880, 4096), arch="sm_100a", hidden_count=True, **common
    )
    assert not full.compact and full.hidden_logits_stage() == "gemm_logits_mcnt"
    assert full.hidden_logits_group_m is None  # 1500 rows: the default raster height
    assert cake_backend.forward_keys(full)[0] == ("row_finalize", 0)
    assert cake_backend.forward_keys(replace(full, hidden_count=False))[0] == (
        "gemm_logits",
        0,
    )
    with pytest.raises(ValueError, match="arch"):
        cake_backend.make_plan(
            _problem(1500, 6144, 154880, 4096), hidden_count=True, **common
        )


def test_logprob_arm_plan_keys():
    """The log-probability logits GEMM arms: the forward's ``gemm_logits_lbl`` replaces ``gemm_logits`` and the
    backward's ``gemm_logits_dz`` replaces ``gemm_logits_nostats`` + ``row_grad`` on every chunk (the dW side stream
    forks behind it); a hidden valid-row count backward keeps chunk 0's queued device-count recompute and its
    ``row_grad``; the arms belong to the log-probability entry."""
    g = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    common = dict(
        need_dx=True,
        need_dw=True,
        geometry=g,
        dx_max_slices=4,
        num_sms=148,
        dx_resident=74,
        fuse_dw_cast=True,
        dx_finalize=True,
        arch="sm_100a",
    )
    problem = _problem(8700, 6144, 154880, 4096, entry="logprob")
    plain = cake_backend.make_plan(problem, valid_rows=8265, **common)
    arms = cake_backend.make_plan(
        problem, valid_rows=8265, logprob_dz=True, logprob_lbl=True, **common
    )
    assert arms.chunks == plain.chunks and arms.num_chunks == 3
    fwd, plain_fwd = cake_backend.forward_keys(arms), cake_backend.forward_keys(plain)
    assert fwd == tuple(
        ("gemm_logits_lbl", k[1]) if k[0] == "gemm_logits" else k for k in plain_fwd
    )
    bwd = cake_backend.recompute_keys(arms)
    assert not any(k[0] in ("row_grad", "gemm_logits_nostats") for k in bwd)
    assert [k for k in bwd if k[0] == "gemm_logits_dz"] == [
        ("gemm_logits_dz", i) for i in range(3)
    ]
    assert [k for k in bwd if k[0] != "gemm_logits_dz"] == [
        k
        for k in cake_backend.recompute_keys(plain)
        if k[0] not in ("row_grad", "gemm_logits_nostats")
    ]
    assert {"gemm_logits_dz", "gemm_logits_lbl"} <= set(arms.stages)
    assert not {"row_grad", "gemm_logits", "gemm_logits_nostats"} & set(arms.stages)
    assert list(arms.stages) == [s for s in cake_jit.STAGES if s in arms.stages]
    _, forks, sides = cake_backend.side_stream_schedule(arms, bwd, "1")
    assert sorted(forks) == [i for i, k in enumerate(bwd) if k[0] == "gemm_logits_dz"]
    assert len(sides) == 2 and all(p > min(forks) for p in sides)
    hidden = replace(arms, hidden_count=True, hidden_stats=False)
    rk = cake_backend.recompute_keys(hidden)
    assert rk[:2] == (("gather_rows", "d_in"), ("row_grad", 0))
    assert ("gemm_logits_dz", 0) not in rk and ("gemm_logits_dz", 1) in rk
    assert "row_grad" in hidden.stages
    with pytest.raises(ValueError, match="log-probability"):
        cake_backend.make_plan(
            _problem(8700, 6144, 154880, 4096), logprob_dz=True, **common
        )


def test_forward_early_keys():
    """The host-count path's early launches (``forward_early_keys``): chunk 0's row gather (compacted plans) and
    its forward logits GEMM -- the label-column arm on a log-probability plan that runs it, the only early launch
    that reads the call's labels -- in their ``forward_keys`` order; none on the hidden valid-row count path."""
    g = Geometry.from_record({"geometry": {"hidden": 6144, "vocab": 154880}})
    common = dict(geometry=g, num_sms=148, arch="sm_100a")
    early, reads = cake_backend.forward_early_keys, cake_backend._reads_labels
    loss = _problem(5000, 6144, 154880, 4096)
    policy = _problem(5000, 6144, 154880, 4096, objective="policy")
    logprob = _problem(5000, 6144, 154880, 4096, entry="logprob")
    gather, logits, lbl = ("gather_rows", 0), ("gemm_logits", 0), ("gemm_logits_lbl", 0)
    for problem, kw, want in (
        (loss, dict(valid_rows=4750), (gather, logits)),
        (loss, {}, (logits,)),
        (policy, dict(valid_rows=4750), (gather, logits)),
        (logprob, dict(valid_rows=4750), (gather, logits)),
        (logprob, dict(valid_rows=4750, logprob_lbl=True), (gather, lbl)),
        (logprob, dict(logprob_lbl=True), (lbl,)),
    ):
        train = problem.entry == "loss"
        plan = cake_backend.make_plan(
            problem, need_dx=train, need_dw=train, **common, **kw
        )
        keys = early(plan)
        assert keys == want, (problem.objective, problem.entry, kw)
        assert [k for k in cake_backend.forward_keys(plan) if k in keys] == list(keys)
        assert [reads(k[0]) for k in keys] == [k == lbl for k in keys]
    hidden = cake_backend.make_plan(
        loss, need_dx=True, need_dw=True, valid_rows=4750, hidden_count=True, **common
    )
    assert early(hidden) == ()
    assert reads("row_finalize") and reads("row_grad") and reads("gemm_logits_dz")
    assert not reads("gemm_logits_nostats") and not reads("gather_rows_bf16")


def test_logprob_arm_switches_belong_to_the_logprob_entry():
    """``logprob_dz`` / ``logprob_lbl`` are keywords of the log-probability entry points defaulting to ``None`` (=
    ``True``, the arms); a loss-entry runner rejects ``True`` and accepts ``False``; a plan without a registered
    program (the reference backend) keeps the stages the arms replace under any value."""
    inp = _host_inputs(37)
    for kw in (dict(logprob_dz=True), dict(logprob_lbl=True)):
        with pytest.raises(ValueError, match="log-probability"):
            prepare_lm_head_loss(
                inp.X,
                inp.W,
                inp.labels,
                loss_div=inp.loss_div,
                chunk_size=16,
                backend="reference",
                **kw,
            )
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        loss_div=inp.loss_div,
        chunk_size=16,
        backend="reference",
        logprob_dz=False,
        logprob_lbl=False,
    )
    assert not (runner.plan.logprob_dz or runner.plan.logprob_lbl)
    runner = prepare_lm_head_loss(
        inp.X,
        inp.W,
        inp.labels,
        chunk_size=16,
        entry="logprob",
        backend="reference",
        logprob_dz=True,
        logprob_lbl=True,
    )
    assert not (runner.plan.logprob_dz or runner.plan.logprob_lbl)
    assert "row_grad" in runner.stages and "gemm_logits" in runner.stages
    for name, fns in (
        (
            "logprob_lbl",
            (cake_backend.forward_logprob, cake_backend.chunked_lm_head_logprob),
        ),
        (
            "logprob_dz",
            (cake_backend.backward_logprob, cake_backend.chunked_lm_head_logprob),
        ),
    ):
        for fn in (*fns, prepare_lm_head_loss):
            param = inspect.signature(fn).parameters[name]
            assert param.kind is inspect.Parameter.KEYWORD_ONLY
            assert param.default is None, (fn.__name__, name)
    assert cake_backend.logprob_dz_default() is True
    assert cake_backend.logprob_lbl_default() is True


def test_hidden_count_eligibility_rules(monkeypatch, toolchain_check_uncached):
    """A compacted call runs the hidden valid-row count only with a registered program that carries the gather kernel
    and the device-count form of chunk 0's logits GEMM (its raster height at the buffer extent is a launch scalar), a
    call of three or more chunks
    (``HIDDEN_COUNT_MIN_CHUNKS``: one- and two-chunk calls cannot hide the path's fixed host cost behind chunk 0's GEMM
    and take the previous path), ``T > 0``, an ``X`` the GEMM reads in place (no contiguous copy, a 16-byte-aligned
    base) and a toolchain whose sm_103a code runs it (not nvcc 13.0.x)."""
    monkeypatch.setattr(cake_build, "record_cuda_version", lambda arch: (13, 4))
    assert cake_backend.HIDDEN_COUNT_MIN_CHUNKS == 3
    name = "cake_lm_head_loss_fake"

    def record(*stages):
        rec = _fake_record(_ROW_GRAD_PLAN)
        rec["stages"] = list(stages)
        for stage in stages:
            rec[stage] = rec["row_grad"]
        return rec

    elig = cake_backend.hidden_count_eligible
    full = record(
        "row_grad",
        "gather_rows_bf16",
        "gemm_logits_mcnt",
        "gemm_logits_nostats_mcnt",
    )
    monkeypatch.setitem(cake_jit.MODULES, name, full)
    X = torch.empty(
        16, 6144, dtype=torch.bfloat16
    )  # host tensor: the rule reads the pointer only
    p = _problem(12289, 6144, 154880, 4096)  # four chunks, buffer extent 4096 rows
    assert elig(p, X, full, name, stats=True)
    assert elig(p, X, full, name, stats=False)  # the recompute's device-count form
    short = _problem(
        3100, 6144, 154880, 1024
    )  # four chunks, buffer extent 1024 rows: the default raster height (a launch scalar of the same form)
    assert elig(short, X, full, name, stats=True)
    # the chunk rule: a call of fewer than three chunks takes the previous path whatever the program carries
    assert not elig(
        _problem(1500, 6144, 154880, 4096), X, full, name, stats=True
    )  # one chunk
    assert not elig(
        _problem(4097, 6144, 154880, 4096), X, full, name, stats=True
    )  # two chunks
    assert not elig(
        _problem(8192, 6144, 154880, 4096), X, full, name, stats=True
    )  # exactly two
    assert elig(
        _problem(8193, 6144, 154880, 4096), X, full, name, stats=True
    )  # three: the first served
    assert not elig(
        _problem(2048, 6144, 154880, 1024), X, full, name, stats=True
    )  # two chunks at C = 1024
    assert elig(_problem(2049, 6144, 154880, 1024), X, full, name, stats=True)
    assert not elig(p, X, None, None, stats=True)  # no registered program
    assert not elig(_problem(0, 6144, 154880, 4096), X[:0], full, name, stats=True)
    assert not elig(
        replace(p, x_copy=True), X, full, name, stats=True
    )  # X needs a contiguous copy
    n = 16 * 6144
    misaligned = torch.empty(n + 8, dtype=torch.bfloat16)[4 : 4 + n].view(16, 6144)
    assert misaligned.data_ptr() % 16 == 8
    assert not elig(p, misaligned, full, name, stats=True)
    aligned = torch.empty(n + 8, dtype=torch.bfloat16)[8 : 8 + n].view(16, 6144)
    assert aligned.data_ptr() % 16 == 0 and elig(p, aligned, full, name, stats=True)
    without_gather = record("row_grad", "gemm_logits_mcnt", "gemm_logits_nostats_mcnt")
    monkeypatch.setitem(cake_jit.MODULES, name, without_gather)
    assert not elig(p, X, without_gather, name, stats=True)
    stats_only = record("row_grad", "gather_rows_bf16", "gemm_logits_mcnt")
    monkeypatch.setitem(cake_jit.MODULES, name, stats_only)
    assert elig(p, X, stats_only, name, stats=True)
    assert not elig(
        p, X, stats_only, name, stats=False
    )  # the recompute's device-count form is missing
    sm_103a = dict(full, arch="sm_103a")
    monkeypatch.setitem(cake_jit.MODULES, name, sm_103a)
    assert elig(short, X, sm_103a, name, stats=True)  # the same rules on sm_103a
    monkeypatch.setattr(cake_build, "record_cuda_version", lambda arch: (13, 0))
    cake_jit.toolchain_runs_hidden_count.cache_clear()
    assert not elig(
        short, X, sm_103a, name, stats=True
    )  # CUDA 13.0: the sm_103a calls take the host-count path
    monkeypatch.setitem(cake_jit.MODULES, name, full)
    assert elig(short, X, full, name, stats=True)  # sm_100a is not affected


_HIDDEN_COUNT_HOST_CASES = [  # (T, C, objective, entry, ignored fraction): three chunks + tail, T < C, T = C + 1, T = C
    (37, 16, "ce", "loss", "five_percent"),
    (37, 16, "policy", "loss", "half"),
    (37, 16, "ce", "logprob", "five_percent"),
    (37, 16, "ce", "loss", "all"),
    (37, 16, "ce", "logprob", "none"),
    (10, 16, "ce", "loss", "half"),
    (17, 16, "ce", "logprob", "all_but_one"),
    (16, 16, "policy", "loss", "none"),
]


@pytest.mark.parametrize(
    "T, C, objective, entry, frac",
    _HIDDEN_COUNT_HOST_CASES,
    ids=[f"{t}_{c}_{o}_{e}_{f}" for t, c, o, e, f in _HIDDEN_COUNT_HOST_CASES],
)
def test_hidden_count_switch_is_inert_on_the_reference_backend(
    T, C, objective, entry, frac, monkeypatch
):
    """The reference backend has no device-count kernels: the switch never consults the registry (``record_for`` is not
    called) and ``loss`` / ``logp`` / ``dX`` / ``dW`` are bitwise the same with it on and off."""

    def refuse(*args, **kwargs):
        raise AssertionError(
            "the reference backend must not consult the registry for the hidden valid-row count"
        )

    monkeypatch.setattr(cake_backend, "record_for", refuse)
    ignore = {
        "none": 0.0,
        "five_percent": 0.05,
        "half": 0.5,
        "all_but_one": (T - 1) / T,
        "all": 1.0,
    }[frac]
    inp = _host_inputs(T, objective, ignore_frac=ignore)
    out = {}
    for env in ("0", "1"):
        out[env] = _run(inp, C, entry=entry, compact_rows=True, hidden_count=env == "1")
        _check_dtypes(out[env], inp)
        assert torch.all(out[env]["dX"][~inp.valid] == 0)
    for key in ("loss", "logp", "dX", "dW"):
        assert torch.equal(out["1"][key], out["0"][key]), key


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


def test_valid_rows_host_count_matches_the_device_count(monkeypatch):
    """A CUDA call of at most ``HOST_COUNT_MAX_ROWS`` rows forms the valid-row index on the host from the mask's
    readback, a longer one on the device: the same int64 index and bool mask (contiguous, on the labels' device) for
    every row valid, some ignored and every row ignored, either side of the limit."""
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    limit = cake_backend.HOST_COUNT_MAX_ROWS
    assert 1 <= limit < 2048
    gen = torch.Generator().manual_seed(SEED)
    cases = []
    for T in (1, 2, 16, 128, limit, limit + 1, 3000):
        labels = torch.randint(0, 1000, (T,), generator=gen)
        some = labels.clone()
        some[torch.randperm(T, generator=gen)[: max(1, T // 20)]] = IGNORE_INDEX
        cases += [labels, some, torch.full((T,), IGNORE_INDEX)]
    for labels in cases:
        on_device = labels.to(CUDA)
        out = {}
        # the device path everywhere, the host path everywhere
        for rows in (0, 1 << 20):
            monkeypatch.setattr(cake_backend, "HOST_COUNT_MAX_ROWS", rows)
            out[rows] = cake_backend.valid_rows(on_device, mask=True)
            assert cake_backend.valid_rows(on_device)[1] is None
        (device_idx, device_mask), (host_idx, host_mask) = out[0], out[1 << 20]
        assert (device_idx is None) == (host_idx is None) == bool((labels >= 0).all())
        if device_idx is None:
            assert device_mask is None and host_mask is None
            continue
        for d, h in ((device_idx, host_idx), (device_mask, host_mask)):
            assert torch.equal(d, h) and d.dtype == h.dtype and d.shape == h.shape
            assert h.device == on_device.device and h.is_contiguous()


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
def test_dx_finalize_switch_is_bitwise_on_the_reference_backend(objective, entry, frac):
    """The fused dX finalize off (``dx_finalize=False``: cast + zero fill + ``index_copy_``) and on (one-pass scatter) through the
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
        result = _run(inp, C, entry=entry, compact_rows=True, dx_finalize=env == "1")
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
                dx_finalize=env == "1",
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


def test_dw_stream_keyword_modes():
    """The dW side-stream keyword and its modes: ``None`` / ``"auto"`` = calls of :data:`DW_STREAM_MIN_CHUNKS` (three)
    or more chunks, ``True`` / ``"1"`` = every multi-chunk call, ``False`` / ``"0"`` = never; a one-chunk call never
    (its only accumulate is deferred to the backward); any other value is a ``ValueError``."""
    assert cake_backend.DW_STREAM_MIN_CHUNKS == 3
    assert cake_backend.DW_STREAM_MODES == ("0", "1", "auto")
    for value in (None, "auto", " AUTO "):
        assert cake_backend.dw_stream_mode(value) == "auto"
    assert cake_backend.dw_stream_mode() == "auto"
    assert [cake_backend.dw_side_stream(n) for n in (0, 1, 2, 3, 4, 8)] == [
        False,
        False,
        False,
        True,
        True,
        True,
    ]
    for value in (False, "0"):
        assert cake_backend.dw_stream_mode(value) == "0"
        assert not any(cake_backend.dw_side_stream(n, value) for n in (1, 2, 3, 8))
    for value in (True, "1"):
        assert cake_backend.dw_stream_mode(value) == "1"
        assert [cake_backend.dw_side_stream(n, value) for n in (1, 2, 3)] == [
            False,
            True,
            True,
        ]
    for mode in cake_backend.DW_STREAM_MODES:
        assert cake_backend.dw_stream_mode(mode) == mode  # idempotent on a mode
    for bad in ("on", "off", "yes", "no", "", "anything-else", 0, 1, 2.0, [True]):
        with pytest.raises(ValueError, match="dw_stream"):
            cake_backend.dw_stream_mode(bad)
        with pytest.raises(ValueError, match="dw_stream"):
            cake_backend.dw_side_stream(3, bad)


def test_dw_stream_schedule_keys():
    """The side-stream placement over the launch keys: joins at the first key of every chunk after the first and
    before the keys that follow the chunk loop, forks at each chunk's ``row_grad``, sides at the per-chunk
    accumulates only -- a deferred last chunk's cast GEMM and the flat casts stay on the caller's stream; ``None``
    for the loss entry's cast keys, for a plan without ``dW`` and whenever the ``dw_stream`` keyword or the rule says
    no.  The reference runner takes no part (host operators on one stream)."""
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
    _, forks_2, sides_2 = cake_backend.side_stream_schedule(
        two, cake_backend.forward_keys(two), dw_stream=True
    )
    assert len(sides_2) == 1 and len(forks_2) == 2
    assert cake_backend.side_stream_schedule(three, fwd, dw_stream=False) is None
    assert cake_backend.side_stream_schedule(three, fwd, "0") is None
    assert cake_backend.side_stream_schedule(three, fwd, "auto") == (
        joins,
        forks,
        sides,
    )
    inp = _host_inputs(37)
    common = dict(
        objective="ce", loss_div=inp.loss_div, chunk_size=16, backend="reference"
    )
    results = []
    for value in (False, True):
        runner = prepare_lm_head_loss(
            inp.X, inp.W, inp.labels, dw_stream=value, **common
        )
        assert runner.dw_stream == cake_backend.dw_stream_mode(value)
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
        and plain["hidden_count"] is False
    )
    # the hidden valid-row count: the chunk buffers and the gather buffer exist at the buffer extent min(C, T) before the
    # count is known -- on the compacted outcome and on the uncompacted one (which gathers chunk 0 all the same)
    rows0 = min(C, T)
    hidden = memory_report(T, H, V, C, valid_rows=T_v, hidden_count=True)
    assert (
        hidden["hidden_count"]
        and hidden["compact_rows"]
        and hidden["valid_rows"] == T_v
    )
    assert hidden["temporary"]["logits"] == rows0 * V * 2
    assert hidden["gather_bytes"] == hidden["temporary"]["x_c"] == rows0 * H * 2
    assert hidden["temporary_bytes"] == sum(hidden["temporary"].values())
    assert (
        hidden["outputs"] == plain["outputs"] and hidden["weights"] == plain["weights"]
    )
    full = memory_report(T, H, V, C, hidden_count=True)
    assert full["hidden_count"] and not full["compact_rows"]
    assert full["gather_bytes"] == full["temporary"]["x_c"] == rows0 * H * 2
    assert full["temporary"]["logits"] == plain["temporary"]["logits"] == rows0 * V * 2
    assert full["temporary_bytes"] == plain["temporary_bytes"] + rows0 * H * 2
    assert memory_report(T, H, V, C, valid_rows=0, hidden_count=True)["num_chunks"] == 0
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
    with pytest.raises(ValueError, match="W.dtype"):
        cake_backend.chunked_lm_head_loss(
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
    hidden = (
        cake_backend.hidden_count_default()
        and cake_jit.toolchain_runs_hidden_count(cake_backend.arch_for(inp.X.device))
        and cake_backend.HIDDEN_COUNT_MIN_CHUNKS <= (4097 + 4095) // 4096
    )  # the effective switch of this call: two chunks take the host-count path (the chunk rule), as does sm_103a on nvcc 13.0.x
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
            hidden_count=hidden,
            **kw,
        )
        binding = cache.peek(key)
        assert (
            binding is not None and binding.holds_no_tensor() and binding.plan.compact
        )
        assert binding.plan.hidden_count is hidden
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
    """The dW side stream (the ``dw_stream`` keyword) off, forced on and at the ``auto`` rule, through the autograd
    entry points and a prepared runner: ``loss`` / ``logp`` / ``dX`` / ``dW`` bitwise for both entries (only the
    launch stream of the per-chunk accumulate differs); the launches of a call with the side stream enter exactly two
    streams, those of a single-stream call one; the remembered binding serves every mode (the mode is a launch
    placement, not part of the binding)."""
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
    for name, inp, mode, expect in (
        ("two_off", two, False, 1),
        ("two_on", two, True, 2),
        ("two_auto", two, "auto", 1),
        ("three_off", three, "0", 1),
        ("three_auto", three, None, 2),
    ):
        streams.clear()
        res = _run(inp, 4096, backend="cake", compact_rows=True, dw_stream=mode)
        assert len(set(streams)) == expect, (name, len(set(streams)))
        streams.clear()
        lp = _run(
            inp,
            4096,
            entry="logprob",
            backend="cake",
            compact_rows=True,
            dw_stream=mode,
        )
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
    assert runner.plan.num_chunks == 3 and runner.dw_stream == "auto"
    steps = {}
    for env, expect in (("0", 1), ("auto", 2), ("1", 2)):
        runner.dw_stream = env  # the mode of the following launches (``dw_stream`` of prepare_lm_head_loss)
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
from tests.test_helpers.cake_lm_head_loss_reference import make_inputs
from tests.test_helpers.cake_package import cake_package
cake_backend = cake_package("cake_lm_head_loss", "cake_backend")
W = make_inputs(1, seed={seed_w}, device="cuda", ignore_frac=0.0).W  # the module fixture's weight
inp = make_inputs({T}, seed={seed}, device="cuda", W=W, ignore_frac=0.05)
X = inp.X.detach().requires_grad_(True)
W = inp.W.detach().requires_grad_(True)
logp = cake_backend.chunked_lm_head_logprob(X, W, inp.labels, chunk_size={C}, backend="cake")
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
        logp = cake_backend.chunked_lm_head_logprob(
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
        # the hidden valid-row count sizes the chunk buffer at min(C, T) before the count is known (the shipped path
        # at min(C, valid rows))
        buffer_rows = (
            min(C, inp.T)
            if fr.memory["hidden_count"]
            else min(C, int(fr.memory["valid_rows"]))
        )
        assert saved == buffer_rows * inp.V * 2
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
            inp.T, inp.V, 2048, inp.X.device, hidden=inp.H, compact_rows=True
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


def test_device_eager_casts_remember_their_launch(glm_weight):
    """The output-boundary casts bind once per form while the binding cache is enabled and slot each call's tensors into
    the remembered launch: bitwise the outputs of a fresh binding per call (cache disabled), one remembered form per
    cast, no tensor held, and ``clear`` / ``enabled=False`` drop them."""
    _require_program(entry="loss")
    cache = cake_backend.BINDING_CACHE
    gen = torch.Generator(device=CUDA).manual_seed(SEED + 71)
    accs = [torch.randn((5, DEFAULT_H), device=CUDA, generator=gen) for _ in range(2)]
    dzs = [
        torch.randn((3, DEFAULT_V), device=CUDA, generator=gen).to(torch.bfloat16)
        for _ in range(2)
    ]
    xs = [
        torch.randn((3, DEFAULT_H), device=CUDA, generator=gen).to(torch.bfloat16)
        for _ in range(2)
    ]
    grads = [torch.tensor(v, device=CUDA) for v in (1.5, -0.25)]

    def casts():
        out = []
        for acc, dz, x, g in zip(accs, dzs, xs, grads, strict=True):
            out.append(cake_backend.scale_cast(acc, g, torch.bfloat16))
            out.append(cake_backend.scale_cast(acc, g, torch.float32))
            out.append(cake_backend.dw_cast(dz, x, None, g, torch.bfloat16))
        torch.cuda.synchronize()
        return out

    with _cache(False):
        fresh = casts()
    with _cache(True):
        cache.clear()
        remembered = casts()
        forms = len(cache._eager)
        assert forms == 3, list(cache._eager)
        again = casts()
        assert len(cache._eager) == forms
        assert not any(
            isinstance(a, torch.Tensor)
            for _, launch in cache._eager.values()
            for a in launch.arguments
        )
        cache.clear()
        assert len(cache._eager) == 0
    for a, b, c in zip(fresh, remembered, again, strict=True):
        assert torch.equal(a, b) and torch.equal(a, c) and a.dtype == b.dtype


def _bits(t: torch.Tensor) -> torch.Tensor:
    """The bit pattern of a float tensor as integers (bf16 -> int16, fp32 -> int32): equality of these views is
    bit-level equality (``torch.equal`` on floats equates +0 and -0)."""
    t = t.detach().reshape(-1).contiguous()
    return t.view({torch.bfloat16: torch.int16, torch.float32: torch.int32}[t.dtype])


def _assert_same_bits(a, b, keys=("loss", "logp", "dX", "dW")):
    for key in keys:
        x, y = a[key], b[key]
        assert (x is None) == (y is None), key
        if x is not None:
            assert x.dtype == y.dtype and x.shape == y.shape, key
            assert torch.equal(_bits(x), _bits(y)), key


def test_device_swab_dx_is_bitwise_the_shipped_form(glm_weight):
    """The swapped dX tile (SM103: the rule on whole chunks of a cross-entropy call of >= 3 chunks, or an explicit
    ``swap_ab=True`` with ``tile_n=512`` on every chunk -- ragged last chunks included -- and objective) gives the bits
    of the shipped forms (``swap_ab=False``, the rule's tiles): loss, logp, dX and dW as integer views."""
    _require_program(entry="loss")
    module_name, record = cake_backend.record_for(CUDA)
    if record["arch"] not in cake_backend.DX_SWAB_ARCHES:
        pytest.skip(f"no swapped dX tile on {record['arch']}")
    assert {"gemm_dx_swab", "gemm_dx_s_swab"} <= set(
        cake_jit.registered_stages(module_name)
    )
    off = {"gemm_tuning": {"dx": {"swap_ab": False}}}
    # an explicit swap_ab applies to every chunk; a chunk the tile rule narrows needs the wide tile pinned as well
    pinned = {"gemm_tuning": {"dx": {"swap_ab": True, "tile_n": 512}}}
    for T, C, objective, pin in (
        (
            4096,
            1024,
            "ce",
            {},
        ),  # whole 1024-token chunks (3 whole + a tail when compacted)
        (3 * 4096 + 59, 4096, "ce", {}),
        (3 * 2048 + 1083, 2048, "ce", pinned),  # ragged last chunks
        (2 * 2048 + 339, 2048, "policy", pinned),
        (3 * 2048 + 59, 2048, "ce", pinned),
    ):
        inp = _device_inputs(T, objective, W=glm_weight)
        for compact in (None, False):
            swab = _run(inp, C, backend="cake", compact_rows=compact, **pin)
            shipped = _run(inp, C, backend="cake", compact_rows=compact, **off)
            _assert_same_bits(swab, shipped)
    if generated_program_available(CUDA, entry="logprob"):
        inp = _device_inputs(3 * 2048 + 1083, W=glm_weight)
        _assert_same_bits(
            _run(inp, 2048, entry="logprob", backend="cake", **pinned),
            _run(inp, 2048, entry="logprob", backend="cake", **off),
        )


@pytest.mark.parametrize("tail", [1, 59, 63, 64, 65, 339, 1083, 1536, 1792, 1793])
def test_device_dw_cast_tma_is_bitwise_the_shipped_cast(glm_weight, tail):
    """The bf16 fused cast's TMA store on the last chunk of a 2- or 3-chunk call (tails 1 .. 1792; 1793 keeps the
    shipped cast) gives the bits of the shipped cast (``"dw": {"epi_store": None}``) on every entry and objective."""
    _require_program(entry="loss")
    module_name, _ = cake_backend.record_for(CUDA)
    assert "gemm_dw_cast_bf16_tma" in cake_jit.registered_stages(module_name)
    C = 2048
    shipped = {"gemm_tuning": {"dw": {"epi_store": None}}}
    for full, objective, scale in ((1, "ce", None), (2, "policy", 2.5)):
        inp = _device_inputs(full * C + tail, objective, W=glm_weight)
        _assert_same_bits(
            _run(inp, C, backend="cake", compact_rows=False, scale=scale),
            _run(inp, C, backend="cake", compact_rows=False, scale=scale, **shipped),
        )
    if generated_program_available(CUDA, entry="logprob"):
        inp = _device_inputs(C + tail, W=glm_weight)
        _assert_same_bits(
            _run(inp, C, entry="logprob", backend="cake", compact_rows=False),
            _run(
                inp, C, entry="logprob", backend="cake", compact_rows=False, **shipped
            ),
        )


def test_device_dw_cast_tma_adversarial_old_sum_is_bitwise(glm_weight):
    """The eager fused cast with an old FP32 sum of subnormals, +-0, +-inf, NaN and large magnitudes (sums that
    overflow): the TMA store (rows 1 .. 1792, the default) and the shipped cast (an explicit ``epi_store``) give the
    same bf16 bits for every tail and upstream scale, and the old sum is only read."""
    _require_program(entry="loss")
    module_name, _ = cake_backend.record_for(CUDA)
    assert "gemm_dw_cast_bf16_tma" in cake_jit.registered_stages(module_name)
    H, V = DEFAULT_H, DEFAULT_V
    gen = torch.Generator(device=CUDA).manual_seed(SEED + 97)
    acc = torch.randn((V, H), device=CUDA, generator=gen) * 64
    special = torch.tensor(
        [0.0, -0.0, float("inf"), float("-inf"), float("nan"), 1e-45, -1e-45]
        + [1.1754942e-38, -1.1754942e-38, 3.0e38, -3.0e38, 6.0e4],
        device=CUDA,
    )
    acc.view(-1, 64)[:, : special.numel()] = special
    acc_sum = _bits(acc).to(torch.int64).sum()
    for rows in (1, 59, 63, 64, 65, 339, 1083, 1536, 1792):
        dz = (torch.randn((rows, V), device=CUDA, generator=gen) * 8).to(torch.bfloat16)
        x = torch.randn((rows, H), device=CUDA, generator=gen).to(torch.bfloat16)
        for g in (
            None,
            torch.tensor(2.5, device=CUDA),
            torch.tensor(-3.0, device=CUDA),
        ):
            tma = cake_backend.dw_cast(dz, x, acc, g, torch.bfloat16)
            shipped = cake_backend.dw_cast(
                dz, x, acc, g, torch.bfloat16, gemm_tuning={"dw": {"epi_store": None}}
            )
            torch.cuda.synchronize()
            assert torch.equal(_bits(tma), _bits(shipped)), (rows, g)
    assert _bits(acc).to(torch.int64).sum() == acc_sum


def test_device_dx_finalize_is_bitwise(glm_weight, monkeypatch):
    """The fused dX finalize on the generated program: ``dx_finalize=False`` (host ``add_`` chain, cast, zero fill,
    ``index_copy_``) and ``True`` (``slab_sum`` + ``scale_cast_scatter_bf16``) give bitwise the same ``loss`` / ``logp`` /
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
            switch = env == "1"
            with _cache(False):
                bound.clear()
                res = {
                    "loss": _run(
                        inp, 4096, backend="cake", compact_rows=True, dx_finalize=switch
                    ),
                    "logprob": _run(
                        inp,
                        4096,
                        entry="logprob",
                        backend="cake",
                        compact_rows=True,
                        dx_finalize=switch,
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
                    dx_finalize=switch,
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


def test_device_hidden_count_kernels_match_torch(glm_weight):
    """Chunk 0's row gather and device-count logits GEMM against the torch gather and the shipped host-bound GEMM over
    the same rows: the count arrives through the pinned cell, the gathered rows are ``X``'s valid rows in order (exact
    zeros after the count), and the logits and row statistics of the counted rows are bitwise the host-bound form's."""
    _require_program(entry="logprob")
    _require_hidden_count()
    name, record = cake_backend.record_for(CUDA)
    # calls of three chunks and more (the chunk rule): some rows ignored, none, most (a count below the buffer
    # extent), five chunks at half, all ignored
    cases = (
        (3100, 1024, 0.05),
        (3100, 1024, 0.0),
        (3100, 1024, 0.9),
        (4097, 1024, 0.5),
        (2200, 1024, 1.0),
    )
    for T, C, ignore in cases:
        inp = make_inputs(
            T, seed=SEED + 62 + T, device=CUDA, W=glm_weight, ignore_frac=ignore
        )
        problem = validate_lm_head_inputs(
            inp.X, inp.W, inp.labels, chunk_size=C, entry="logprob"
        )
        assert cake_backend.hidden_count_eligible(
            problem, inp.X, record, name, stats=True
        )
        hc = cake_backend._hidden_count_begin(
            problem,
            inp.X,
            inp.W,
            inp.labels,
            record=record,
            module_name=name,
            stats=True,
        )
        count = hc.finish()
        n_valid = int(inp.valid.sum())
        assert count == n_valid and hc.rows0 == min(C, T) and hc.count == count
        idx = inp.valid.nonzero().reshape(-1)
        assert torch.equal(hc.valid, inp.valid)
        assert torch.equal(hc.idx_full[:count], idx)
        assert torch.all(hc.idx_full[count:] == T)
        row_index = hc.row_index()
        if count == T:
            assert row_index is None
        else:
            assert torch.equal(row_index, idx)
        n0 = min(count, hc.rows0)
        assert torch.equal(hc.x_c[:n0], inp.X.index_select(0, idx[:n0]))
        assert torch.all(hc.x_c[n0:] == 0)
        if n0 == 0:
            continue
        # the shipped host-bound GEMM over chunk 0's rows (the log-probability forward of the stored-logits form,
        # ``logprob_lbl=False``, keeps the logits intact); an explicit index also when every row is valid: the call has
        # several chunks and only chunk 0 is compared
        runner = prepare_lm_head_loss(
            inp.X,
            inp.W,
            inp.labels,
            chunk_size=C,
            need_dx=False,
            need_dw=False,
            entry="logprob",
            backend="cake",
            compact_rows=idx[:n0],
            logprob_lbl=False,
        )
        assert runner.plan.num_chunks == 1 and runner.plan.chunks[0][1] == n0
        runner.forward()
        torch.cuda.synchronize()
        assert torch.equal(hc.logits[:n0], runner.tensors["logits"][:n0])
        assert torch.equal(hc.stats_buf[:n0], runner.tensors["stats"][:n0])
        assert hc.stage == "gemm_logits_mcnt"
        raster0 = cake_backend.raster_variant(DEFAULT_H, hc.rows0, record["arch"])
        assert hc.group_m == (
            runner.plan.variants_of(0).logits_group_m
            if n0 == hc.rows0
            else (raster0[0] if raster0 else None)
        )


def test_device_hidden_count_switch_is_bitwise(glm_weight, monkeypatch):
    """The hidden valid-row count on the generated program: ``hidden_count=False`` (the count and the index on the host
    before the first launch) and ``True`` (device compaction, chunk 0's gather kernel and device-count logits GEMM queued before
    the count is read) give bitwise the same ``loss`` / ``logp`` / ``dX`` / ``dW`` on both entries over the three
    outcomes (some rows ignored, none, all) and over one-, two-, three- and four-chunk calls -- the count serves three
    chunks and more (``HIDDEN_COUNT_MIN_CHUNKS``), the shorter calls take the host-count path on either value; with the
    switch on and a served call the count is read only after both launches are queued and the host count is never
    taken; the remembered binding serves the switch."""
    _require_program(entry="loss")
    _require_program(entry="logprob")
    _require_hidden_count()
    real_bind_all = cake_backend._bind_all
    real_finish = cake_backend._HiddenCount.finish
    real_stream = cake_backend._ffi_stream_context
    real_valid_rows = cake_backend.valid_rows
    log: list = []

    def spy_bind_all(record, module_name, keys, values, device, geometry):
        log.extend(("bind", k[0]) for k in keys)
        return real_bind_all(record, module_name, keys, values, device, geometry)

    def spy_stream(index):
        log.append(("launch", ""))
        return real_stream(index)

    def spy_finish(self):
        log.append(("finish", self.stage))
        return real_finish(self)

    def refuse_valid_rows(*args, **kwargs):
        raise AssertionError("the hidden valid-row count must not count on the host")

    monkeypatch.setattr(cake_backend, "_bind_all", spy_bind_all)
    monkeypatch.setattr(cake_backend, "_ffi_stream_context", spy_stream)
    monkeypatch.setattr(cake_backend._HiddenCount, "finish", spy_finish)
    gather = cake_backend.HIDDEN_COUNT_STAGE
    cases = [
        (1500, "ce", 0.0),
        (1500, "ce", 0.05),
        (1500, "policy", 0.05),
        (1500, "ce", (1500 - 1) / 1500),
        (1500, "policy", 1.0),
        (4097, "ce", 0.05),
        (4097, "ce", 0.5),
        (8700, "ce", 0.05),
        (8700, "policy", 0.5),
        (8700, "ce", 1.0),
        (12289, "ce", 0.05),
    ]
    for T, objective, ignore in cases:
        hidden = (
            T + 4095
        ) // 4096 >= cake_backend.HIDDEN_COUNT_MIN_CHUNKS  # the chunk rule
        inp = make_inputs(
            T,
            objective=objective,
            seed=SEED + 62 + T,
            device=CUDA,
            W=glm_weight,
            ignore_frac=ignore,
        )
        n_valid = int(inp.valid.sum())
        out = {}
        for env in ("0", "1"):
            switch = env == "1"
            monkeypatch.setattr(
                cake_backend,
                "valid_rows",
                refuse_valid_rows if switch and hidden else real_valid_rows,
            )
            with _cache(False):
                log.clear()
                res = {
                    "loss": _run(
                        inp,
                        4096,
                        backend="cake",
                        compact_rows=True,
                        hidden_count=switch,
                    )
                }
                if objective == "ce":
                    res["logprob"] = _run(
                        inp,
                        4096,
                        entry="logprob",
                        backend="cake",
                        compact_rows=True,
                        hidden_count=switch,
                    )
                fr = cake_backend.forward_loss(
                    inp.X,
                    inp.W,
                    inp.labels,
                    objective=objective,
                    loss_div=inp.loss_div if objective == "ce" else None,
                    infer_logp=inp.infer_logp,
                    loss_weights=inp.loss_weights,
                    chunk_size=4096,
                    backend="cake",
                    compact_rows=True,
                    hidden_count=switch,
                )
                torch.cuda.synchronize()
            binds = [s for kind, s in log if kind == "bind"]
            counted = [
                s
                for s in binds
                if s != gather
                and s not in cake_backend.HOST_STAGES
                and cake_backend.parse_stage(s)[1]["count"]
            ]
            assert (fr.row_index is None) == (n_valid == T)
            if env == "0" or not hidden:
                assert gather not in binds and not counted
                assert not any(kind == "finish" for kind, _ in log)
                assert fr.memory["hidden_count"] is False
            else:
                # the loss forward, the log-probability forward (ce; its backward takes the forward's row index and
                # counts nothing), the explicit forward: each queues the gather and chunk 0's device-count GEMM, then
                # reads the count
                expect = 3 if objective == "ce" else 2
                assert binds.count(gather) == expect, binds
                assert len(counted) == expect, binds
                finishes = [i for i, (kind, _) in enumerate(log) if kind == "finish"]
                assert len(finishes) == expect
                for i in finishes:
                    j = max(k for k in range(i) if log[k] == ("bind", gather))
                    assert any(log[k][0] == "launch" for k in range(j, i)), (
                        "the count was read before the launches were queued"
                    )
                if n_valid:
                    assert fr.memory["hidden_count"] is True
                    assert fr.memory["gather_bytes"] == min(4096, T) * DEFAULT_H * 2
                else:
                    assert fr.memory["num_chunks"] == 0
            if n_valid == 0:
                assert torch.all(res["loss"]["dX"] == 0) and torch.all(
                    res["loss"]["dW"] == 0
                )
            assert torch.all(res["loss"]["dX"][~inp.valid] == 0)
            res["explicit"] = dict(
                loss=fr.loss, logp=fr.logp, dX=fr.dx_acc, dW=fr.dw_acc
            )
            out[env] = res
        for entry in out["1"]:
            for key in ("loss", "logp", "dX", "dW"):
                a, b = out["1"][entry].get(key), out["0"][entry].get(key)
                assert _same(a, b), (T, objective, ignore, entry, key)
    monkeypatch.setattr(cake_backend, "valid_rows", real_valid_rows)
    # the remembered binding of the hidden-count path: a hit is bitwise the validating call
    inp = make_inputs(8193, seed=SEED + 63, device=CUDA, W=glm_weight, ignore_frac=0.05)
    kw = dict(backend="cake", compact_rows=True, hidden_count=True)
    with _cache(True) as cache:
        cache.clear()
        first = _run(inp, 4096, **kw)
        second = _run(inp, 4096, **kw)
        assert cache.hits >= 1 and len(cache) >= 1
        for key in ("loss", "logp", "dX", "dW"):
            assert torch.equal(first[key], second[key]), key
        lp1 = _run(inp, 4096, entry="logprob", **kw)
        lp2 = _run(inp, 4096, entry="logprob", **kw)
        for key in ("logp", "dX", "dW"):
            assert torch.equal(lp1[key], lp2[key]), key


def _logprob_steps(inp, C, **kw):
    """Two autograd log-probability steps through the binding cache (the first call binds, the second launches from
    the remembered binding) and the explicit pair: ``forward_logprob``, then ``backward_logprob`` with the forward's
    row index and with a recount."""
    steps = [_run(inp, C, entry="logprob", backend="cake", **kw) for _ in range(2)]
    fwd_kw = {k: v for k, v in kw.items() if k != "logprob_dz"}
    bwd_kw = {k: v for k, v in kw.items() if k != "logprob_lbl"}
    fr = cake_backend.forward_logprob(inp.X, inp.W, inp.labels, chunk_size=C, **fwd_kw)
    dlogp = torch.where(inp.valid, inp.dlogp, torch.zeros_like(inp.dlogp))
    out = dict(lse=fr.lse, logp_eager=fr.logp)
    for i, step in enumerate(steps):
        out.update({f"{k}{i}": step[k] for k in ("loss", "logp", "dX", "dW")})
    for tag, extra in (("index", dict(row_index=fr.row_index)), ("recount", {})):
        dx, dw = cake_backend.backward_logprob(
            inp.X, inp.W, inp.labels, fr.lse, dlogp, chunk_size=C, **bwd_kw, **extra
        )
        out[f"dX_{tag}"], out[f"dW_{tag}"] = dx, dw
    torch.cuda.synchronize()
    return out, steps[0]


def test_device_logprob_arms_are_bitwise(glm_weight):
    """The log-probability logits GEMM arms against the stages they replace on the generated program: ``logprob_dz``
    / ``logprob_lbl`` on (the default) and off give bitwise the same ``lse`` / ``logp`` / ``dX`` / ``dW`` through the
    autograd entry (first call and remembered binding) and the explicit pair (the backward on the forward's row index
    and on a recount), run to run, over one-, two- and many-chunk calls, compacted and uncompacted calls with ignored
    rows (``-100`` and other negative labels), every row valid, labels at and beyond the last vocabulary column
    (clamped), the hidden valid-row count on and off and an ``X`` with a padded leading stride (read in place and
    copied); the arms keep the accuracy gates."""
    _require_program(entry="logprob")
    name, _ = cake_backend.record_for(CUDA)
    if not {"gemm_logits_dz", "gemm_logits_lbl"} <= set(
        cake_backend.registered_stages(name)
    ):
        pytest.skip("the registered program has no log-probability logits GEMM arms")
    V = int(glm_weight.shape[0])
    edge_labels = [-1, -7, V - 1, 0, V, V + 5, 2 * V, 2**40]
    cases = [
        # (T, chunk, ignored fraction, switches, X pitch padding, label edits, accuracy gate)
        (1500, 4096, 0.05, {}, 0, False, True),
        (1500, 4096, 0.0, {}, 0, False, True),
        (1500, 4096, 0.05, dict(compact_rows=False), 0, False, True),
        (6000, 4096, 0.05, {}, 0, False, False),
        (6000, 4096, 0.05, dict(compact_rows=False), 0, False, False),
        (8700, 4096, 0.05, {}, 0, False, False),
        (8700, 4096, 0.05, dict(hidden_count=False), 0, False, False),
        (3100, 1024, 0.5, {}, 0, False, True),
        (1500, 4096, 0.05, {}, 8, False, False),
        (1500, 4096, 0.05, dict(compact_rows=False), 4, False, False),
        (2500, 2048, 0.05, {}, 0, True, False),
        (2500, 2048, 0.05, dict(compact_rows=False), 0, True, False),
        (5000, 2048, 0.05, {}, 0, True, False),
    ]
    for T, C, ignore, switches, ld_pad, edit, gate in cases:
        inp = make_inputs(
            T,
            seed=SEED + 70 + T,
            device=CUDA,
            W=glm_weight,
            ignore_frac=ignore,
            ld_pad=ld_pad,
        )
        if edit:
            labels = inp.labels.clone()
            labels[: len(edge_labels)] = torch.tensor(edge_labels, device=CUDA)
            inp = replace(inp, labels=labels)
        case = (T, C, ignore, switches, ld_pad, edit)
        with _cache(True) as cache:
            cache.clear()
            on, first = _logprob_steps(inp, C, **switches)
            off, _ = _logprob_steps(
                inp, C, logprob_dz=False, logprob_lbl=False, **switches
            )
            # the forward bindings run the label-column arm under its switch, the backward bindings the dz arm
            arms = {
                (key[0], b.plan.logprob_lbl if key[0] == "fwd" else b.plan.logprob_dz)
                for key, b in cache._bindings.items()
            }
        assert arms == {
            ("fwd", True),
            ("fwd", False),
            ("bwd", True),
            ("bwd", False),
        }, (case, arms)
        for key in on:
            assert _same(on[key], off[key]), (case, key)
        for key in ("logp", "dX", "dW"):
            assert torch.equal(on[f"{key}0"], on[f"{key}1"]), (case, key)
        assert torch.equal(on["logp_eager"], on["logp0"]), case
        for tag in ("index", "recount"):
            assert torch.equal(on[f"dX_{tag}"], on["dX0"]), (case, tag)
            assert torch.equal(on[f"dW_{tag}"], on["dW0"]), (case, tag)
        assert torch.all(on["logp0"][~inp.valid] == 0)
        assert torch.all(on["dX0"][~inp.valid] == 0)
        if gate:
            _check_against_references(first, inp, entry="logprob")
        del on, off, first


def test_device_host_count_early_launch_is_bitwise(glm_weight, monkeypatch):
    """The host-count path's early launch (one- and two-chunk calls, and longer calls without the hidden valid-row
    count): the remembered forward launches chunk 0's gather and logits GEMM right behind the count and binds the
    rest behind them.  Against the in-order sequence (no early keys) every output of both entries is bitwise the
    same -- compacted, every row valid, uncompacted with ignored rows, both objectives and the log-probability entry
    (whose label-column GEMM binds the labels first), the dW side stream on two chunks, the unfused dW cast and dX
    finalize, a padded ``X`` -- the remembered bindings record the split, and the early launch is queued before the
    labels are bound (after them for the label-column GEMM)."""
    _require_program(entry="loss")
    _require_program(entry="logprob")
    real_early = cake_backend.forward_early_keys
    real_labels = cake_backend._prepare_labels
    real_stream = cake_backend._ffi_stream_context
    lbl = cake_backend.LOGPROB_LBL_STAGE
    log: list = []

    def spy_labels(*args, **kwargs):
        log.append("labels")
        return real_labels(*args, **kwargs)

    def spy_stream(index):
        log.append("launch")
        return real_stream(index)

    monkeypatch.setattr(cake_backend, "_prepare_labels", spy_labels)
    monkeypatch.setattr(cake_backend, "_ffi_stream_context", spy_stream)
    cases = [
        (1500, "ce", 0.05, {}, 0),
        (1500, "policy", 0.05, {}, 0),
        (1500, "ce", 0.0, {}, 0),
        (1500, "policy", 0.05, dict(compact_rows=False), 0),
        (6000, "ce", 0.05, {}, 0),
        (6000, "policy", 0.05, dict(dw_stream=True), 0),
        (6000, "ce", 0.05, dict(fuse_dw_cast=False, dx_finalize=False), 0),
        (1500, "policy", 0.05, {}, 8),
        (8700, "ce", 0.05, dict(hidden_count=False), 0),
    ]
    for T, objective, ignore, switches, ld_pad in cases:
        inp = make_inputs(
            T,
            objective=objective,
            seed=SEED + 80 + T,
            device=CUDA,
            W=glm_weight,
            ignore_frac=ignore,
            ld_pad=ld_pad,
        )
        entries = ("loss", "logprob") if objective == "ce" else ("loss",)
        case = (T, objective, ignore, switches, ld_pad)
        out = {}
        for early in (True, False):
            monkeypatch.setattr(
                cake_backend,
                "forward_early_keys",
                real_early if early else (lambda plan: ()),
            )
            with _cache(True) as cache:
                cache.clear()
                out[early] = {
                    entry: [
                        _run(inp, 4096, entry=entry, backend="cake", **switches)
                        for _ in range(2)
                    ]
                    for entry in entries
                }
                bindings = [b for b in cache._bindings.values() if b.forward_order]
                for b in bindings:
                    split = (b.forward_early, b.forward_early_labels)
                    if early:
                        assert split[0] + split[1] == real_early(b.plan), case
                        assert bool(split[0] or split[1]) == (
                            not b.plan.hidden_count
                        ), case
                        assert split[1] == (
                            ((lbl, 0),) if b.plan.logprob_lbl else ()
                        ), case
                    else:
                        assert split == ((), ()), case
                # the remembered forward's order of host work: the early launch ahead of the labels
                for entry in entries:
                    fwd = (
                        cake_backend.forward_loss
                        if entry == "loss"
                        else cake_backend.forward_logprob
                    )
                    # the autograd calls' forward binding (the same keywords: a hit)
                    kw = (
                        dict(
                            objective=objective,
                            loss_div=inp.loss_div if objective == "ce" else None,
                            infer_logp=inp.infer_logp,
                            loss_weights=inp.loss_weights,
                            **switches,
                        )
                        if entry == "loss"
                        else {
                            k: v
                            for k, v in switches.items()
                            if k in ("compact_rows", "hidden_count")
                        }
                    )
                    log.clear()
                    fwd(inp.X, inp.W, inp.labels, chunk_size=4096, **kw)
                    torch.cuda.synchronize()
                    first = log.index("labels")
                    # the call's remembered forward binding: launches ahead of the labels exactly when its early
                    # launches start with keys that do not read them (an uncompacted label-column GEMM reads them)
                    bound = [
                        b
                        for key, b in cache._bindings.items()
                        if key[:2] == ("fwd", entry)
                    ]
                    assert len(bound) == 1, (case, entry)
                    expect = bool(bound[0].forward_early)
                    assert ("launch" in log[:first]) == expect, (case, entry, log)
        for entry in entries:
            for run in range(2):
                for key in ("loss", "logp", "dX", "dW"):
                    a = out[True][entry][run][key]
                    b = out[False][entry][run][key]
                    assert _same(a, b), (case, entry, run, key)
            for key in ("loss", "logp", "dX", "dW"):
                assert torch.equal(
                    out[True][entry][0][key], out[True][entry][1][key]
                ), (case, entry, key)


def test_device_logprob_backward_does_not_synchronize(glm_weight):
    """The log-probability backward plans its chunks on the forward's valid-row index: with the binding remembered,
    the autograd backward of one-, two- and three-chunk calls (compacted, every row valid, uncompacted) runs without
    a host synchronization; the forward's host count is one synchronization (``valid_rows``: ``nonzero`` returns the
    count together with the index), with ignored rows and without."""
    _require_program(entry="logprob")
    for T, C, ignore, compact in (
        (1500, 4096, 0.05, True),
        (1500, 4096, 0.0, True),
        (6000, 4096, 0.05, True),
        (6000, 4096, 0.05, False),
        (8700, 4096, 0.05, True),
    ):
        inp = make_inputs(
            T, seed=SEED + 90 + T, device=CUDA, W=glm_weight, ignore_frac=ignore
        )
        X = inp.X.detach().requires_grad_(True)
        W = inp.W.detach().requires_grad_(True)
        for i in range(2):
            logp = cake_backend.chunked_lm_head_logprob(
                X, W, inp.labels, chunk_size=C, backend="cake", compact_rows=compact
            )
            torch.cuda.synchronize()
            if i:
                torch.cuda.set_sync_debug_mode("error")
            try:
                logp.backward(inp.dlogp)
            finally:
                torch.cuda.set_sync_debug_mode("default")
            torch.cuda.synchronize()
            assert torch.all(X.grad[~inp.valid] == 0)
            X.grad = W.grad = None
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            torch.cuda.set_sync_debug_mode("warn")
            try:
                index, mask = cake_backend.valid_rows(inp.labels, mask=True)
            finally:
                torch.cuda.set_sync_debug_mode("default")
        syncs = [w for w in caught if "synchroniz" in str(w.message)]
        assert len(syncs) == 1, [str(w.message) for w in caught]
        if ignore:
            assert torch.equal(index, inp.valid.nonzero().reshape(-1))
            assert torch.equal(mask, inp.valid)
        else:
            assert index is None and mask is None
