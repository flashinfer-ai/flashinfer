# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The FROST runners: a plan's checked inputs bound to the kernel wrappers ``FrostGdnConsumer`` (decode) and
``FrostGdnVerify`` (verify, per-token state layout: ``journal=False``).

The wrappers compile per binding (their ``key``: step shape, strides, launch policy) and own their output buffers per
row count. A runner owns one wrapper, binds the caller's FP8 output (and BF16 output when asked) into the wrapper's
output slots before each launch, and shares the compiled kernels with every runner whose wrapper compiles the same
code: the wrappers' keys leave out the device, ``eps`` and the parameter dtypes, so those key the shared table.
"""

from __future__ import annotations

import threading
from typing import Any, Dict

import torch

from ...api import Qwen35GdnInputs, Qwen35GdnPlanSpec
from . import admission
from .gdn import FrostGdnConsumer
from .gdn_verify import THREADS, FrostGdnVerify

_COMPILED: Dict[tuple, Dict[tuple, Any]] = {}
_COMPILED_LOCK = threading.Lock()


def _shared_compiled(spec: Qwen35GdnPlanSpec) -> Dict[tuple, Any]:
    key = (spec.form, spec.device, spec.geometry.norm_eps, spec.param_dtypes)
    with _COMPILED_LOCK:
        return _COMPILED.setdefault(key, {})


def _not_compiled(form: str) -> RuntimeError:
    return RuntimeError(
        f"qwen_3_5_gdn_block (frost, {form}): no kernel is compiled for this binding (its strides); run the same "
        "call once eagerly before CUDA-graph capture"
    )


class FrostDecodeRunner:
    def __init__(self, spec: Qwen35GdnPlanSpec):
        self.spec = spec
        self.kernel = FrostGdnConsumer(spec.device, spec.geometry.norm_eps)
        self.kernel.compiled = _shared_compiled(spec)
        self.buffers = self.kernel.outputs(spec.rows)
        self.scratch_bf16 = self.buffers["out"]

    def _args(self, x: Qwen35GdnInputs) -> tuple:
        return (
            self.spec.rows,
            x.qkv,
            x.conv_state,
            x.conv_weight,
            x.A_log,
            x.dt_bias,
            x.ssm_state,
            x.state_indices,
            x.z,
            x.norm_weight,
            x.out_scale,
            x.hidden,
            x.ba_weight,
        )

    def prepare(self, inputs: Qwen35GdnInputs) -> None:
        self.kernel.compile(*self._args(inputs))

    def run(self, inputs: Qwen35GdnInputs) -> None:
        args = self._args(inputs)
        rows, x, cs, state, z = args[0], args[1], args[2], args[6], args[8]
        if self.kernel.key(rows, x, cs, state, z) not in self.kernel.compiled:
            raise _not_compiled("decode")
        self.buffers["qout"] = inputs.out
        self.buffers["out"] = (
            self.scratch_bf16 if inputs.out_bf16 is None else inputs.out_bf16
        )
        self.kernel(*args)


class FrostVerifyRunner:
    def __init__(self, spec: Qwen35GdnPlanSpec):
        self.spec = spec
        # The per-token state layout, the fused speculative conv, no programmatic dependent launch (the launch does
        # not wait on, or release, a neighbouring kernel); the one-way norm-sum exchange of the two-CTA heads.
        self.kernel = FrostGdnVerify(
            spec.device,
            spec.geometry.norm_eps,
            conv=True,
            release=False,
            journal=False,
            pdl=False,
            push=True,
        )
        self.kernel.compiled = _shared_compiled(spec)
        self.buffers = self.kernel.outputs(spec.rows)
        self.scratch_bf16 = self.buffers["out"]

    def _args(self, x: Qwen35GdnInputs) -> tuple:
        return (
            x.qkv,
            x.z,
            x.hidden,
            x.ba_weight,
            x.A_log,
            x.dt_bias,
            x.ssm_state,
            x.state_indices,
            x.cu_seqlens,
            x.num_accepted_tokens,
            x.norm_weight,
            x.out_scale,
            x.conv_state,
            x.conv_weight,
        )

    def prepare(self, inputs: Qwen35GdnInputs) -> None:
        if not self.kernel.fuses_conv(inputs.conv_state, inputs.state_indices):
            raise NotImplementedError(
                "qwen_3_5_gdn_block (frost, verify): the fused speculative conv needs conv_state channel-contiguous "
                "(stride(1) == 1, i.e. stored [num_slots, state_len, conv_dim]) with stride(0) and stride(2) "
                f"multiples of 8 and a 16-byte aligned base; got strides {list(inputs.conv_state.stride())}"
            )
        self.kernel.compile(self.spec.rows, *self._args(inputs))

    def run(self, inputs: Qwen35GdnInputs) -> None:
        args = self._args(inputs)
        qkv, z, state, indices, cs = args[0], args[1], args[6], args[7], args[12]
        if self.kernel.key(qkv, z, state, indices, cs) not in self.kernel.compiled:
            raise _not_compiled("verify")
        self.buffers["qout"] = inputs.out
        self.buffers["out"] = (
            self.scratch_bf16 if inputs.out_bf16 is None else inputs.out_bf16
        )
        self.kernel(self.spec.rows, self.spec.requests, *args)


def _check_mirrored(spec: Qwen35GdnPlanSpec, runner) -> None:
    """The step-shape rules ``admission`` mirrors, re-read from the wrappers: a drift fails the plan, loudly."""
    if THREADS != admission.THREADS:
        raise RuntimeError(
            f"frost admission mirrors THREADS={admission.THREADS}, gdn_verify has {THREADS}"
        )
    if spec.form == "verify":
        kernel = runner.kernel
        if kernel.ctas(spec.requests) != admission.verify_ctas(spec.requests):
            raise RuntimeError(
                "frost admission's CTAs-per-head rule differs from FrostGdnVerify.ctas"
            )


def build(spec: Qwen35GdnPlanSpec):
    """The FROST ``build_fn``."""
    if spec.device.type != "cuda":
        raise NotImplementedError("the FROST GDN kernels run on CUDA devices")
    with torch.cuda.device(spec.device):
        runner = (
            FrostDecodeRunner(spec)
            if spec.form == "decode"
            else FrostVerifyRunner(spec)
        )
    _check_mirrored(spec, runner)
    return runner
