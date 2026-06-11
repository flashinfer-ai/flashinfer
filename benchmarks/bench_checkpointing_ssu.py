# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
TODO: Better name for mtp-len, which is actually draft_len + 1
Standalone benchmark for checkpointing_ssu (CUDA) vs the Triton reference.

Suitable for nsight-compute (ncu) and nsight-systems (nsys) capture.

Fixed model config: NVIDIA-Nemotron-3-Super-120B-A12B at TP=8
  nheads=16, head_dim=64, d_state=128, ngroups=1

Baseline kernel (--baseline [triton|flashinfer]):
  Calls selective_state_update with T=mtp_len tokens and disable_state_update=True,
  matching the MTP scoring pass in mamba2_mixer.py exactly.

Example usage:
  # Basic sweep
  python bench_checkpointing_ssu.py \\
      --batch-sizes 1,2,4 --mtp-lengths 1,4,8 --warmup 5 --iters 20

  # With CUDA graph (default) and Triton baseline:
  python bench_checkpointing_ssu.py --baseline \\
      --batch-sizes 1,2,4 --mtp-lengths 5,10,20

  # nsys capture (NVTX ranges visible in timeline)
  nsys profile --capture-range=cudaProfilerApi \\
      python bench_checkpointing_ssu.py --profile

  # ncu capture
  ncu --target-processes all \\
      python bench_checkpointing_ssu.py --profile \\
          --batch-sizes 1 --mtp-lengths 4 --warmup 5 --iters 5
"""

import argparse
import bisect
import contextlib
import os
import re
import statistics
import sys
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Callable, Literal

import torch
from einops import repeat

# Add tests/mamba to path for triton_reference imports
sys.path.insert(0, str(Path(__file__).parent.parent / "tests" / "mamba"))

from triton_reference.checkpointing_state_update import checkpointing_state_update

# triton-replay uses the faithful, TMA-optimized standalone reference (the
# merged checkpointing_state_update copy dropped TMA + tuning during migration).
from triton_reference.replay_selective_state_update import (
    REPLAY_WORK_CACHE_BUF_IDX,
    REPLAY_WORK_CACHE_SLOT,
    REPLAY_WORK_ITEM_WIDTH,
    REPLAY_WORK_PNAT,
    REPLAY_WORK_POSITION_IN_DECODE_BATCH,
    replay_selective_state_update,
)
from triton_reference.selective_state_update import (
    selective_state_update_triton as selective_state_update,
)
from flashinfer.mamba import selective_state_update as flashinfer_selective_state_update
from flashinfer.mamba.checkpointing_ssu import (
    checkpointing_ssu as cuda_checkpointing_ssu,
)

# Optional conv1d import for combined (conv1d + SSU) timing.
# Standalone copy lives in tests/mamba/triton_reference/ (stripped of TRT-LLM deps).
_causal_conv1d_update = None
with contextlib.suppress(Exception):
    from triton_reference.causal_conv1d_triton import (
        causal_conv1d_update as _causal_conv1d_update,
    )

# ---------------------------------------------------------------------------
# Model config defaults (Nemotron-3-Super-120B full model)
# --tp-size divides nheads and ngroups to get the per-GPU slice.
#   TP=1: nheads=128, ngroups=8
#   TP=4: nheads=32,  ngroups=2
#   TP=8: nheads=16,  ngroups=1  (default)
# ---------------------------------------------------------------------------
NHEADS = 128
HEAD_DIM = 64
D_STATE = 128
NGROUPS = 8
TP_SIZE = 8  # default; overridden by --tp-size

# L2 flush buffer: ~128 MB — larger than L2 on A100/H100/B200
_L2_FLUSH_SIZE = 32 * 1024 * 1024  # float32 elements → 128 MB
_l2_flush: torch.Tensor | None = None


# Public state-dtype map shared between the CLI and the in-process bench
# (bench_ssu_checkpoint_mixed.py).  fp16/fp32 are kept as aliases for
# backward compat.
STATE_DTYPE_MAP: dict[str, torch.dtype] = {
    "f16": torch.float16,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "f32": torch.float32,
    "fp32": torch.float32,
    "int8": torch.int8,
    "i8": torch.int8,
    "fp8": torch.float8_e4m3fn,
    "f8": torch.float8_e4m3fn,
    "e4m3": torch.float8_e4m3fn,
}

# Dtypes for which Philox stochastic rounding on gmem state writeback is
# supported (cuda-incr and Triton incremental paths only).
_PHILOX_STATE_DTYPES = (torch.float16, torch.int8, torch.float8_e4m3fn)


def parse_state_spec(s: str) -> tuple[torch.dtype, int, str]:
    """Parse a state-dtype spec string into (dtype, philox_rounds, label).

    Plain names: 'f16', 'bf16', 'f32', 'fp16', 'fp32', 'int8', 'i8',
        'fp8', 'e4m3'  → philox_rounds = 0.
    Philox suffix: '<dtype>-philox-<N>' (valid for f16/fp16/int8/i8/fp8/e4m3)
        → Philox-<N> stochastic rounding on gmem state writeback.
        Example: 'fp16-philox-5', 'int8-philox-10', 'fp8-philox-5'.

    The label is the user-provided string verbatim so the table and CSV
    preserve it.
    """
    m = re.match(r"^([a-z0-9]+)(?:-philox-(\d+))?$", s)
    if not m or m.group(1) not in STATE_DTYPE_MAP:
        raise ValueError(
            f"invalid state-dtype spec: {s!r} "
            f"(expected one of {sorted(STATE_DTYPE_MAP)} optionally suffixed "
            f"with '-philox-<N>' for f16/fp16/int8/i8/fp8/e4m3)"
        )
    dtype = STATE_DTYPE_MAP[m.group(1)]
    philox = int(m.group(2)) if m.group(2) is not None else 0
    if philox > 0 and dtype not in _PHILOX_STATE_DTYPES:
        raise ValueError(
            f"state-dtype spec {s!r}: philox stochastic rounding only "
            f"supported for {_PHILOX_STATE_DTYPES} state, got {dtype}"
        )
    return dtype, philox, s


def _init_l2_flush() -> None:
    global _l2_flush
    _l2_flush = torch.empty(_L2_FLUSH_SIZE, dtype=torch.float32, device="cuda")


def _flush_l2() -> None:
    """Evict L2 by writing to a large buffer then synchronising."""
    if _l2_flush is None:
        _init_l2_flush()
    _l2_flush.fill_(0.0)
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Tensor construction helpers
# ---------------------------------------------------------------------------


def _build_tensors(
    batch: int,
    mtp_len: int,
    max_window: int,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    nheads: int,
    head_dim: int,
    d_state: int,
    ngroups: int,
):
    """
    Build all tensors for one benchmark configuration.

    nheads/ngroups are already TP-split (i.e. full_nheads // tp_size).

    Returns:
      state0                   : (batch, nheads, head_dim, d_state) – initial SSM state
      intermediate_update_inputs: (batch, mtp_len, nheads*head_dim + nheads + ngroups*d_state)
                                   packed [old_x | old_dt_base | old_B] for incremental kernel
      x, dt, B, C              : (batch, mtp_len, ...) – token inputs for both kernels
      A, dt_bias, D            : SSM parameters (float32, tie_hdim strides)
      prev_tokens              : (batch,)
      out_incr                 : pre-allocated output for incremental kernel (batch, mtp_len, nheads, head_dim)
      out_base                 : pre-allocated output for baseline kernel   (batch, mtp_len, nheads, head_dim)
      intermediate_states_buffer: for baseline kernel (batch, mtp_len, nheads, head_dim, d_state)
    """
    device = "cuda"

    torch.manual_seed(42)

    # --- SSM parameters (float32, tie_hdim strides) ---
    A_base = -torch.rand(nheads, device=device) - 0.5
    A = repeat(
        A_base, "h -> h p n", p=head_dim, n=d_state
    )  # stride(-1)=0, stride(-2)=0

    dt_bias_base = torch.randn(nheads, device=device, dtype=torch.float32)
    dt_bias = repeat(dt_bias_base, "h -> h p", p=head_dim)  # stride(-1)=0

    D_base = torch.randn(nheads, device=device, dtype=torch.float32)
    D = repeat(D_base, "h -> h p", p=head_dim)

    # --- SSM state ---
    _quantized_state_dtypes = (torch.int8, torch.float8_e4m3fn)
    is_quantized = state_dtype in _quantized_state_dtypes
    if is_quantized:
        # Quantize from fp32 source: state0 is int8/fp8, state_scale is fp32 decode_scale.
        # decode_scale shape: (batch, nheads, head_dim) — per-(head, dim) channel.
        quant_max = {torch.int8: 127.0, torch.float8_e4m3fn: 448.0}[state_dtype]
        state0_fp32 = torch.randn(
            batch, nheads, head_dim, d_state, device=device, dtype=torch.float32
        )
        amax = state0_fp32.abs().amax(dim=-1)  # (batch, nheads, head_dim)
        encode_scale = quant_max / amax.clamp(min=1e-30)
        state_scale = 1.0 / encode_scale  # decode_scale, fp32
        scaled = state0_fp32 * encode_scale.unsqueeze(-1)
        if state_dtype == torch.float8_e4m3fn:
            # Native fp8 cast does RN at the fp8 grid; an explicit `round()`
            # would destroy sub-integer precision.
            state0 = scaled.clamp(-quant_max, quant_max).to(state_dtype)
        else:
            state0 = scaled.round().clamp(-quant_max, quant_max).to(state_dtype)
    else:
        state0 = torch.randn(
            batch, nheads, head_dim, d_state, device=device, dtype=state_dtype
        )
        state_scale = None

    # --- Cache tensors for incremental kernel ---
    # T-axis = max_window (cache capacity), independent of mtp_len (per-step
    # speculation depth).  must_checkpoint = (prev_k + mtp_len > max_window),
    # so max_window > mtp_len leaves room for the no-checkpoint branch.
    # old_x: single-buffered (cache, max_window, nheads, dim)
    old_x = torch.randn(
        batch, max_window, nheads, head_dim, device=device, dtype=act_dtype
    )
    # old_B: double-buffered (cache, 2, max_window, ngroups, dstate)
    old_B = torch.randn(
        batch, 2, max_window, ngroups, d_state, device=device, dtype=act_dtype
    )
    # old_dt: double-buffered (cache, 2, nheads, max_window) fp32
    old_dt = torch.randn(
        batch, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    # old_cumAdt: double-buffered (cache, 2, nheads, max_window) fp32
    old_cumAdt = torch.randn(
        batch, 2, nheads, max_window, device=device, dtype=torch.float32
    )
    # cache_buf_idx: which buffer to read (0 or 1)
    cache_buf_idx = torch.zeros(batch, device=device, dtype=torch.int32)
    # Legacy packed tensor (kept for baseline kernel only) — uses mtp_len slice
    # of the larger cache so the baseline shape (which is mtp_len-bound) stays
    # unchanged.
    old_x_flat = old_x[:, :mtp_len].reshape(batch, mtp_len, nheads * head_dim)
    old_dt_base = torch.randn(batch, mtp_len, nheads, device=device, dtype=act_dtype)
    old_B_flat = old_B[:, 0, :mtp_len].reshape(batch, mtp_len, ngroups * d_state)
    intermediate_update_inputs = torch.cat(
        [old_x_flat, old_dt_base, old_B_flat], dim=-1
    ).contiguous()

    # --- Token inputs (used by both incremental and baseline kernels) ---
    x = torch.randn(batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype)
    # TODO: For now, dt has to match D (fp32) for flashifner, so always do that
    dt_base = torch.randn(batch, mtp_len, nheads, device=device, dtype=torch.float32)
    dt = repeat(dt_base, "b t h -> b t h p", p=head_dim)  # tie_hdim
    B = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)
    C = torch.randn(batch, mtp_len, ngroups, d_state, device=device, dtype=act_dtype)

    # prev_tokens placeholder — overwritten per-run
    prev_tokens = torch.zeros(batch, device=device, dtype=torch.int32)

    out_incr = torch.zeros(
        batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype
    )
    out_base = torch.zeros(
        batch, mtp_len, nheads, head_dim, device=device, dtype=act_dtype
    )

    intermediate_states_buffer = torch.zeros(
        batch, mtp_len, nheads, head_dim, d_state, device=device, dtype=state_dtype
    )

    # --- Conv1d tensors ---
    d_inner = nheads * head_dim
    conv_dim = d_inner + 2 * ngroups * d_state
    d_conv = 4  # conv kernel width for Nemotron/Mamba2

    # xbc_input: (batch, conv_dim, mtp_len) — match production layout from in_proj
    # Production: in_proj output is (batch*mtp_len, conv_dim) contiguous, then
    # .view(batch, mtp_len, conv_dim).transpose(1, 2) gives strides
    # (mtp_len*conv_dim, 1, conv_dim) — NOT standard 3D allocation.
    xbc_input_flat = torch.randn(
        batch * mtp_len, conv_dim, device=device, dtype=act_dtype
    )
    xbc_input = xbc_input_flat.view(batch, mtp_len, conv_dim).transpose(1, 2)
    # conv_state: (batch, conv_dim, d_conv) — "cold" cache
    conv_state = torch.randn(batch, conv_dim, d_conv, device=device, dtype=act_dtype)
    # conv_weight: (conv_dim, d_conv) — parameter
    conv_weight = torch.randn(conv_dim, d_conv, device=device, dtype=act_dtype)
    # conv_bias: (conv_dim,) — parameter
    conv_bias = torch.randn(conv_dim, device=device, dtype=act_dtype)

    return (
        state0,
        state_scale,
        intermediate_update_inputs,
        old_x,
        old_B,
        old_dt,
        old_cumAdt,
        cache_buf_idx,
        x,
        dt,
        B,
        C,
        A,
        dt_bias,
        D,
        prev_tokens,
        out_incr,
        out_base,
        intermediate_states_buffer,
        xbc_input,
        conv_state,
        conv_weight,
        conv_bias,
    )


# ---------------------------------------------------------------------------
# Public Python API
#
# Other scripts (e.g. bench_ssu_checkpoint_mixed.py) drive the benchmark
# in-process via build_kernel_inputs() + time_kernel().  The CLI path
# (_run_benchmark) reuses the same helpers.
# ---------------------------------------------------------------------------


KernelName = Literal[
    "cuda-incr",  # cuda_checkpointing_ssu
    "incremental",  # Triton checkpointing_state_update (old 4D path)
    "triton-replay",  # Triton replay_selective_state_update — persistent_dynamic
    "triton-replay-pm",  # Triton replay_selective_state_update — persistent_main
    "fi-dump",  # flashinfer_selective_state_update with dst_state_batch_indices
    "baseline-triton",  # Triton selective_state_update (disable_state_update)
    "baseline-flashinfer",  # flashinfer selective_state_update (disable_state_update)
]


@dataclass
class TimingOptions:
    """Knobs for the inner timing loop.

    Attribute names mirror the CLI flags so that _time_kernel*() can read
    them via the same `args.<name>` accesses used on the CLI path.
    """

    warmup: int = 20
    iters: int = 100
    cupti: bool = False
    cuda_graph: bool = True
    l2_flush: bool = True


@dataclass
class TritonAutotune:
    """Optional autotune overrides for the Triton incremental kernel.

    ``None`` for any field means: let the kernel pick.  These are passed
    straight through to ``checkpointing_state_update``.
    """

    block_size_m: int | None = None
    num_warps: int | None = None
    num_stages: int | None = None
    precompute_num_warps: int | None = None
    precompute_num_stages: int | None = None
    internal_pdl: bool = True


@dataclass
class KernelInputs:
    """Pre-allocated tensor bundle for one (batch, mtp_len, dtype) config.

    Each field whose name ends in ``0`` is a pristine copy used by ``reset()``
    to restore the matching ``*_work`` tensor before each timed iteration.
    The ``_work`` tensors are mutated by the kernels.
    """

    # Pristine copies (overwritten by `reset()`)
    state0: torch.Tensor
    state_scale0: torch.Tensor | None
    intermediate_update_inputs: torch.Tensor
    old_x0: torch.Tensor
    # 5-D double-buffered old_x for the triton-replay (persistent) path, built
    # from the 4-D old_x0 (active buffer = cache_buf_idx[slot]).  CUDA / old
    # Triton use the 4-D old_x0; only triton-replay reads the 5-D copy.
    old_x0_5d: torch.Tensor
    old_B0: torch.Tensor
    old_dt0: torch.Tensor
    old_cumAdt0: torch.Tensor
    cache_buf_idx0: torch.Tensor

    # Working copies (mutated by kernels)
    state_work: torch.Tensor
    state_scale_work: torch.Tensor | None
    interm_work: torch.Tensor
    old_x_work: torch.Tensor
    old_x_work_5d: torch.Tensor
    old_B_work: torch.Tensor
    old_dt_work: torch.Tensor
    old_cumAdt_work: torch.Tensor
    cache_buf_idx_work: torch.Tensor

    # Read-only inputs
    x: torch.Tensor
    dt: torch.Tensor
    B: torch.Tensor
    C: torch.Tensor
    A: torch.Tensor
    dt_bias: torch.Tensor
    D: torch.Tensor

    # Per-slot accepted-tokens vector.  ``prev_tokens_i32`` feeds the
    # cuda-incr / Triton incremental kernels.  Mutated by ``time_kernel`` to
    # the user-supplied vector before each timed batch.
    prev_tokens_i32: torch.Tensor

    # Outputs (zeroed at construction; mutated by kernels)
    out_incr: torch.Tensor
    out_base: torch.Tensor
    intermediate_states_buffer: torch.Tensor

    # Shape/config metadata (needed by run-closures)
    batch: int
    mtp_len: int
    max_window: int

    # Conv1d tensors (for --with-conv1d combined timing)
    xbc_input0: torch.Tensor | None = None  # (batch, conv_dim, mtp_len) pristine
    xbc_input_work: torch.Tensor | None = (
        None  # (batch, conv_dim, mtp_len) working copy
    )
    conv_state0: torch.Tensor | None = None  # (batch, conv_dim, d_conv) pristine
    conv_state_work: torch.Tensor | None = (
        None  # (batch, conv_dim, d_conv) working copy
    )
    conv_weight: torch.Tensor | None = None  # (conv_dim, d_conv) — parameter
    conv_bias: torch.Tensor | None = None  # (conv_dim,) — parameter
    d_inner: int = 0  # nheads * head_dim
    conv_dim: int = 0  # d_inner + 2 * ngroups * d_state

    # Two-kernel split scratch (None ⇒ monolithic).  The precompute overwrites
    # them each iteration — graph-safe, no reset needed (like out_incr).
    cb_scaled: torch.Tensor | None = None  # (batch, nheads, 32, 8) act_dtype
    cumAdt_vec: torch.Tensor | None = None  # (batch, nheads, T_pad) f32
    cb_old: torch.Tensor | None = None  # (batch, nheads, 32, K_old//2) act_dtype

    def reset(self) -> None:
        """Restore ``*_work`` tensors to their pristine state."""
        self.state_work.copy_(self.state0)
        if self.state_scale_work is not None:
            self.state_scale_work.copy_(self.state_scale0)
        self.interm_work.copy_(self.intermediate_update_inputs)
        self.old_x_work.copy_(self.old_x0)
        self.old_x_work_5d.copy_(self.old_x0_5d)
        self.old_B_work.copy_(self.old_B0)
        self.old_dt_work.copy_(self.old_dt0)
        self.old_cumAdt_work.copy_(self.old_cumAdt0)
        self.cache_buf_idx_work.copy_(self.cache_buf_idx0)

    def reset_with_conv1d(self) -> None:
        """Realistic reset for conv1d + SSU combined timing.

        Production pipeline: cold cache → L2 flush → hot in_proj output.
        1. Restore cold state (cache tensors, SSM state)
        2. L2 flush (evicts cold state from cache)
        3. Write hot xbc_input (simulates in_proj output landing in L2)
        """
        # 1. Cold state reset
        self.state_work.copy_(self.state0)
        if self.state_scale_work is not None:
            self.state_scale_work.copy_(self.state_scale0)
        self.interm_work.copy_(self.intermediate_update_inputs)
        self.old_x_work.copy_(self.old_x0)
        self.old_x_work_5d.copy_(self.old_x0_5d)
        self.old_B_work.copy_(self.old_B0)
        self.old_dt_work.copy_(self.old_dt0)
        self.old_cumAdt_work.copy_(self.old_cumAdt0)
        self.cache_buf_idx_work.copy_(self.cache_buf_idx0)
        self.conv_state_work.copy_(self.conv_state0)
        # 2. L2 flush
        if _l2_flush is not None:
            _l2_flush.fill_(0.0)
        # 3. Hot in_proj output
        self.xbc_input_work.copy_(self.xbc_input0)


def _old_x_to_5d(old_x_4d: torch.Tensor, cache_buf_idx: torch.Tensor) -> torch.Tensor:
    """Build the 5-D double-buffered ``old_x`` the persistent (triton-replay)
    path expects from the 4-D single-buffer ``old_x``; active buffer =
    ``cache_buf_idx[slot]`` (the inactive buffer is left zero)."""
    cache_size, window, nheads, dim = old_x_4d.shape
    old_x_5d = old_x_4d.new_zeros(cache_size, 2, window, nheads, dim)
    rows = torch.arange(cache_size, device=old_x_4d.device)
    old_x_5d[rows, cache_buf_idx.long()] = old_x_4d
    return old_x_5d


def _make_replay_work_items(prev_tokens, cache_buf_idx, mtp_len, max_window, batch):
    """Build (n_writes, replay_work_items) for the standalone persistent kernel,
    write-first sorted — matches mamba2_metadata._prepare_replay_work_items.

    The bench uses contiguous cache slots (state_batch_indices=None), so
    cache_slot == position_in_decode_batch.  persistent_dynamic ignores the
    ordering at runtime; persistent_main relies on the write-first split.
    """
    device = prev_tokens.device
    pos = torch.arange(batch, device=device, dtype=torch.int32)
    pnat = prev_tokens[:batch].to(torch.int32)
    active_buf = cache_buf_idx[:batch].to(torch.int32)
    write_mask = (pnat + mtp_len) > max_window
    n_writes = write_mask.sum().to(torch.int32).reshape(1)
    order = torch.argsort((~write_mask).to(torch.int32), stable=True).to(torch.long)
    work = torch.empty(batch, REPLAY_WORK_ITEM_WIDTH, device=device, dtype=torch.int32)
    work[:, REPLAY_WORK_POSITION_IN_DECODE_BATCH] = pos[order]
    work[:, REPLAY_WORK_CACHE_SLOT] = pos[order]
    work[:, REPLAY_WORK_PNAT] = pnat[order]
    work[:, REPLAY_WORK_CACHE_BUF_IDX] = active_buf[order]
    return n_writes, work.contiguous()


def build_kernel_inputs(
    *,
    batch: int,
    mtp_len: int,
    max_window: int,
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    nheads: int,
    head_dim: int,
    d_state: int,
    ngroups: int,
    two_kernel: bool = True,
) -> KernelInputs:
    """Build a fresh tensor bundle for one benchmark configuration."""
    (
        state0,
        state_scale0,
        intermediate_update_inputs,
        old_x0,
        old_B0,
        old_dt0,
        old_cumAdt0,
        cache_buf_idx0,
        x,
        dt,
        B,
        C,
        A,
        dt_bias,
        D,
        prev_tokens_i32,
        out_incr,
        out_base,
        intermediate_states_buffer,
        xbc_input,
        conv_state,
        conv_weight,
        conv_bias,
    ) = _build_tensors(
        batch,
        mtp_len,
        max_window,
        state_dtype,
        act_dtype,
        nheads,
        head_dim,
        d_state,
        ngroups,
    )
    old_x0_5d = _old_x_to_5d(old_x0, cache_buf_idx0)
    d_inner = nheads * head_dim
    _conv_dim = d_inner + 2 * ngroups * d_state

    # Two-kernel split scratch (graph-safe; overwritten by the precompute each
    # iter).  fragA-native: cb_scaled m16n8k16 (8 regs/lane); cb_old m16n8k{K_old}
    # (K_old/2 regs/lane), K_old = next_multiple_of<8>(max_window); cumAdt_vec f32
    # over T_pad = next_multiple_of<16>(mtp_len).  See .plans/ssu_split.md.
    cb_scaled = cumAdt_vec = cb_old = None
    if two_kernel:
        t_pad = ((mtp_len + 15) // 16) * 16
        k_old = ((max_window + 7) // 8) * 8
        cb_scaled = torch.empty(batch, nheads, 32, 8, device=x.device, dtype=act_dtype)
        cumAdt_vec = torch.empty(
            batch, nheads, t_pad, device=x.device, dtype=torch.float32
        )
        cb_old = torch.empty(
            batch, nheads, 32, k_old // 2, device=x.device, dtype=act_dtype
        )

    return KernelInputs(
        state0=state0,
        state_scale0=state_scale0,
        intermediate_update_inputs=intermediate_update_inputs,
        old_x0=old_x0,
        old_x0_5d=old_x0_5d,
        old_B0=old_B0,
        old_dt0=old_dt0,
        old_cumAdt0=old_cumAdt0,
        cache_buf_idx0=cache_buf_idx0,
        state_work=state0.clone(),
        state_scale_work=state_scale0.clone() if state_scale0 is not None else None,
        interm_work=intermediate_update_inputs.clone(),
        old_x_work=old_x0.clone(),
        old_x_work_5d=old_x0_5d.clone(),
        old_B_work=old_B0.clone(),
        old_dt_work=old_dt0.clone(),
        old_cumAdt_work=old_cumAdt0.clone(),
        cache_buf_idx_work=cache_buf_idx0.clone(),
        x=x,
        dt=dt,
        B=B,
        C=C,
        A=A,
        dt_bias=dt_bias,
        D=D,
        prev_tokens_i32=prev_tokens_i32,
        out_incr=out_incr,
        out_base=out_base,
        intermediate_states_buffer=intermediate_states_buffer,
        batch=batch,
        mtp_len=mtp_len,
        max_window=max_window,
        xbc_input0=xbc_input,
        xbc_input_work=xbc_input.clone(),
        conv_state0=conv_state,
        conv_state_work=conv_state.clone(),
        conv_weight=conv_weight,
        conv_bias=conv_bias,
        d_inner=d_inner,
        conv_dim=_conv_dim,
        cb_scaled=cb_scaled,
        cumAdt_vec=cumAdt_vec,
        cb_old=cb_old,
    )


def time_kernel(
    *,
    kernel: KernelName,
    inputs: KernelInputs,
    prev_tokens: torch.Tensor,
    timing: TimingOptions | None = None,
    tag: str = "",
    philox_rounds: int = 0,
    rand_seed: torch.Tensor | None = None,
    autotune: TritonAutotune | None = None,
) -> tuple[float, float, float]:
    """Time one kernel invocation against a (batch,) prev_tokens vector.

    Returns (median_us, p95_us, p99_us) across ``timing.iters`` measurements.

    ``prev_tokens`` must be a (batch,)-shaped int32 tensor on CUDA, with
    values in [0, inputs.max_window].  The internal ``prev_tokens_i32`` work
    tensor is overwritten with this vector before timing; subsequent
    ``inputs.reset()`` calls between iterations do NOT touch it, so the same
    vector is reused across all timed iterations.
    """
    if timing is None:
        timing = TimingOptions()
    if autotune is None:
        autotune = TritonAutotune()

    assert prev_tokens.device.type == "cuda", (
        f"prev_tokens must be on CUDA, got {prev_tokens.device}"
    )
    assert prev_tokens.shape == (inputs.batch,), (
        f"prev_tokens shape mismatch: got {tuple(prev_tokens.shape)}, "
        f"expected ({inputs.batch},)"
    )
    pt_max = int(prev_tokens.max().item()) if prev_tokens.numel() > 0 else 0
    pt_min = int(prev_tokens.min().item()) if prev_tokens.numel() > 0 else 0
    assert pt_min >= 0 and pt_max <= inputs.max_window, (
        f"prev_tokens values must be in [0, max_window={inputs.max_window}]; "
        f"got [{pt_min}, {pt_max}]"
    )

    assert prev_tokens.dtype == torch.int32, (
        f"prev_tokens dtype must be int32, got {prev_tokens.dtype}"
    )
    inputs.prev_tokens_i32.copy_(prev_tokens)

    run_fn = _make_run_closure(
        kernel=kernel,
        inputs=inputs,
        philox_rounds=philox_rounds,
        rand_seed=rand_seed,
        autotune=autotune,
    )
    return _time_kernel(timing, run_fn, inputs.reset, tag)


def time_kernel_with_conv1d(
    *,
    kernel: KernelName,
    inputs: KernelInputs,
    prev_tokens: torch.Tensor,
    timing: TimingOptions | None = None,
    tag: str = "",
    philox_rounds: int = 0,
    rand_seed: torch.Tensor | None = None,
    autotune: TritonAutotune | None = None,
    external_pdl: bool = True,
) -> tuple[float, float, float]:
    """Time conv1d + kernel combined, measuring total span.

    Same as time_kernel but prepends causal_conv1d_update and uses the
    realistic reset (cold cache → L2 flush → hot xbc_input).
    Captures PDL overlap between conv1d and SSU.
    """
    if _causal_conv1d_update is None:
        raise RuntimeError(
            "causal_conv1d_update not available — import from "
            "tests/mamba/triton_reference/causal_conv1d_triton.py failed"
        )
    if timing is None:
        timing = TimingOptions()
    if autotune is None:
        autotune = TritonAutotune()

    assert inputs.xbc_input_work is not None, (
        "conv1d tensors not allocated in KernelInputs"
    )
    assert prev_tokens.device.type == "cuda", (
        f"prev_tokens must be on CUDA, got {prev_tokens.device}"
    )
    assert prev_tokens.shape == (inputs.batch,), (
        f"prev_tokens shape mismatch: got {tuple(prev_tokens.shape)}, "
        f"expected ({inputs.batch},)"
    )
    pt_max = int(prev_tokens.max().item()) if prev_tokens.numel() > 0 else 0
    pt_min = int(prev_tokens.min().item()) if prev_tokens.numel() > 0 else 0
    assert pt_min >= 0 and pt_max <= inputs.max_window, (
        f"prev_tokens values must be in [0, max_window={inputs.max_window}]; "
        f"got [{pt_min}, {pt_max}]"
    )
    assert prev_tokens.dtype == torch.int32, (
        f"prev_tokens dtype must be int32, got {prev_tokens.dtype}"
    )
    inputs.prev_tokens_i32.copy_(prev_tokens)

    run_fn = _make_run_closure(
        kernel=kernel,
        inputs=inputs,
        philox_rounds=philox_rounds,
        rand_seed=rand_seed,
        autotune=autotune,
        with_conv1d=True,
        external_pdl=external_pdl,
    )
    # Disable the timing loop's own L2 flush — reset_with_conv1d already
    # does a realistic flush (cold state → L2 evict → hot xbc_input write).
    # Double-flushing would distort timings.
    timing_no_flush = replace(timing, l2_flush=False)
    return _time_kernel(timing_no_flush, run_fn, inputs.reset_with_conv1d, tag)


def _make_run_closure(
    *,
    kernel: KernelName,
    inputs: KernelInputs,
    philox_rounds: int,
    rand_seed: torch.Tensor | None,
    autotune: TritonAutotune,
    with_conv1d: bool = False,
    external_pdl: bool = True,
) -> Callable[[], None]:
    """Build the per-iteration kernel invocation closure for ``time_kernel``.

    When *with_conv1d* is True the closure prepends a causal_conv1d_update
    call and feeds the conv1d outputs (x, B, C views) directly to the SSU
    kernel — matching the production pipeline where conv1d and SSU overlap
    on the GPU via PDL.
    """
    mtp_len = inputs.mtp_len
    max_window = inputs.max_window

    # --- Conv1d split helper (used when with_conv1d=True) ---
    if with_conv1d:
        assert _causal_conv1d_update is not None, (
            "causal_conv1d_update not available — import from "
            "tests/mamba/triton_reference/causal_conv1d_triton.py failed"
        )
        assert inputs.xbc_input_work is not None, (
            "conv1d tensors not allocated in KernelInputs"
        )
        batch = inputs.batch
        d_inner = inputs.d_inner
        conv_dim = inputs.conv_dim
        nheads = inputs.x.shape[2]  # from x: (batch, mtp_len, nheads, head_dim)
        head_dim = inputs.x.shape[3]
        ngroups = inputs.B.shape[2]  # from B: (batch, mtp_len, ngroups, d_state)
        d_state = inputs.B.shape[3]

        def _conv1d_split():
            """Run conv1d update and split output into (x, B, C) views."""
            xbc_result = _causal_conv1d_update(
                inputs.xbc_input_work,
                inputs.conv_state_work,
                inputs.conv_weight,
                inputs.conv_bias,
                activation="silu",
                launch_dependent_kernels=external_pdl,
            )
            xbc_flat = xbc_result.transpose(1, 2).reshape(batch * mtp_len, conv_dim)
            x_flat, B_flat, C_flat = torch.split(
                xbc_flat, [d_inner, ngroups * d_state, ngroups * d_state], dim=-1
            )
            x_conv = x_flat.view(batch, mtp_len, nheads, head_dim)
            B_conv = B_flat.view(batch, mtp_len, ngroups, d_state)
            C_conv = C_flat.view(batch, mtp_len, ngroups, d_state)
            return x_conv, B_conv, C_conv

    if kernel == "cuda-incr":
        if with_conv1d:

            def _run():
                x_conv, B_conv, C_conv = _conv1d_split()
                cuda_checkpointing_ssu(
                    inputs.state_work,
                    inputs.old_x_work,
                    inputs.old_B_work,
                    inputs.old_dt_work,
                    inputs.old_cumAdt_work,
                    inputs.cache_buf_idx_work,
                    inputs.prev_tokens_i32,
                    x=x_conv,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=B_conv,
                    C=C_conv,
                    out=inputs.out_incr,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    state_scale=inputs.state_scale_work,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds,
                    enable_pdl=external_pdl,
                    cb_scaled=inputs.cb_scaled,
                    cumAdt_vec=inputs.cumAdt_vec,
                    cb_old=inputs.cb_old,
                )
        else:

            def _run():
                cuda_checkpointing_ssu(
                    inputs.state_work,
                    inputs.old_x_work,
                    inputs.old_B_work,
                    inputs.old_dt_work,
                    inputs.old_cumAdt_work,
                    inputs.cache_buf_idx_work,
                    inputs.prev_tokens_i32,
                    x=inputs.x,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=inputs.B,
                    C=inputs.C,
                    out=inputs.out_incr,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    state_scale=inputs.state_scale_work,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds,
                    cb_scaled=inputs.cb_scaled,
                    cumAdt_vec=inputs.cumAdt_vec,
                    cb_old=inputs.cb_old,
                )

        return _run

    if kernel in ("triton-replay", "triton-replay-pm"):
        # Faithful 5D persistent path (standalone, TMA-optimized).  Both modes:
        #   triton-replay     → persistent_dynamic (single launch, per-slot
        #                        runtime write decision).
        #   triton-replay-pm  → persistent_main (two launches: write half +
        #                        nowrite half, work items pre-sorted write-first).
        # Build the work-item plumbing once (write-first sorted) — pd ignores the
        # ordering at runtime, pm relies on the n_writes split.  mtp_len /
        # max_window are fixed for this call; prev_tokens_i32 is set by
        # time_kernel before this closure is built.  Tuning knobs default to
        # None → the standalone's _resolve_tuning picks the tuned (TMA) config.
        replay_mode = (
            "persistent_dynamic" if kernel == "triton-replay" else "persistent_main"
        )
        n_writes, replay_work_items = _make_replay_work_items(
            inputs.prev_tokens_i32,
            inputs.cache_buf_idx_work,
            mtp_len,
            max_window,
            inputs.batch,
        )

        if with_conv1d:

            def _run():
                x_conv, B_conv, C_conv = _conv1d_split()
                replay_selective_state_update(
                    inputs.state_work,
                    inputs.old_x_work_5d,
                    inputs.old_B_work,
                    inputs.old_dt_work,
                    inputs.old_cumAdt_work,
                    inputs.cache_buf_idx_work,
                    inputs.prev_tokens_i32,
                    x=x_conv,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=B_conv,
                    C=C_conv,
                    out=inputs.out_incr,
                    n_writes=n_writes,
                    replay_work_items=replay_work_items,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds,
                    state_scales=inputs.state_scale_work,
                    mode=replay_mode,
                    launch_with_pdl=external_pdl,
                )
        else:

            def _run():
                replay_selective_state_update(
                    inputs.state_work,
                    inputs.old_x_work_5d,
                    inputs.old_B_work,
                    inputs.old_dt_work,
                    inputs.old_cumAdt_work,
                    inputs.cache_buf_idx_work,
                    inputs.prev_tokens_i32,
                    x=inputs.x,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=inputs.B,
                    C=inputs.C,
                    out=inputs.out_incr,
                    n_writes=n_writes,
                    replay_work_items=replay_work_items,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=None,
                    rand_seed=rand_seed,
                    philox_rounds=philox_rounds,
                    state_scales=inputs.state_scale_work,
                    mode=replay_mode,
                )

        return _run

    if kernel == "incremental":
        # Triton's `write_checkpoint` is `tl.constexpr` — one kernel launch
        # = one branch.  For a heterogeneous batch, production must launch
        # both variants per step (one for no-write slots, one for write
        # slots); the Triton kernels early-exit slots that don't match the
        # launch's WRITE_CHECKPOINT, so each launch can run over the full
        # batch without corrupting unrelated slots.  See the early-exit
        # block in tests/mamba/triton_reference/checkpointing_state_update.py.
        #
        # Decision rule based on the user-supplied prev_tokens vector:
        #   uniform-write    → 1 launch with write_checkpoint=True
        #   uniform-nowrite  → 1 launch with write_checkpoint=False
        #   heterogeneous    → 2 launches (False then True)
        needs_write = (inputs.prev_tokens_i32 + mtp_len) > max_window
        any_write = bool(needs_write.any().item())
        all_write = bool(needs_write.all().item())
        launches: list[bool]
        if all_write:
            launches = [True]
        elif not any_write:
            launches = [False]
        else:
            launches = [False, True]

        def _one_launch(wc: bool, x_call, B_call, C_call, _launch_with_pdl=False):
            checkpointing_state_update(
                inputs.state_work,
                inputs.old_x_work,
                inputs.old_B_work,
                inputs.old_dt_work,
                inputs.old_cumAdt_work,
                inputs.cache_buf_idx_work,
                inputs.prev_tokens_i32,
                x=x_call,
                dt=inputs.dt,
                A=inputs.A,
                B=B_call,
                C=C_call,
                out=inputs.out_incr,
                D=inputs.D,
                dt_bias=inputs.dt_bias,
                dt_softplus=True,
                state_batch_indices=None,
                use_internal_pdl=autotune.internal_pdl,
                rand_seed=rand_seed,
                philox_rounds=philox_rounds,
                state_scales=inputs.state_scale_work,
                write_checkpoint=wc,
                launch_with_pdl=_launch_with_pdl,
                _block_size_m=autotune.block_size_m,
                _num_warps=autotune.num_warps,
                _num_stages=autotune.num_stages,
                _precompute_num_warps=autotune.precompute_num_warps,
                _precompute_num_stages=autotune.precompute_num_stages,
            )

        if with_conv1d:

            def _run(_launches=tuple(launches)):
                x_conv, B_conv, C_conv = _conv1d_split()
                for wc in _launches:
                    _one_launch(
                        wc, x_conv, B_conv, C_conv, _launch_with_pdl=external_pdl
                    )
        else:

            def _run(_launches=tuple(launches)):
                for wc in _launches:
                    _one_launch(wc, inputs.x, inputs.B, inputs.C)

        return _run

    if kernel == "fi-dump":
        # Write state only at the LAST in-window token for each slot.
        # The CLI used a single scalar prev_k for the whole batch; with a
        # heterogeneous vector we set dst_indices per slot.
        sbi = torch.arange(inputs.batch, dtype=torch.int32, device="cuda")
        out_dump = torch.zeros_like(inputs.out_incr)
        dst_indices = torch.full(
            (inputs.batch, mtp_len), -1, dtype=torch.int32, device="cuda"
        )
        # Per-slot last position: max(prev_k - 1, 0), clamped to mtp_len-1.
        last_pos = (inputs.prev_tokens_i32 - 1).clamp_(min=0, max=mtp_len - 1)
        row_ix = torch.arange(inputs.batch, dtype=torch.int64, device="cuda")
        dst_indices[row_ix, last_pos.to(torch.int64)] = sbi

        if with_conv1d:

            def _run():
                x_conv, B_conv, C_conv = _conv1d_split()
                flashinfer_selective_state_update(
                    state=inputs.state_work,
                    x=x_conv,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=B_conv,
                    C=C_conv,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=sbi,
                    dst_state_batch_indices=dst_indices,
                    pad_slot_id=-1,
                    state_scale=inputs.state_scale_work,
                    out=out_dump,
                    cache_steps=mtp_len,
                    algorithm="simple",
                )
        else:

            def _run():
                flashinfer_selective_state_update(
                    state=inputs.state_work,
                    x=inputs.x,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=inputs.B,
                    C=inputs.C,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    state_batch_indices=sbi,
                    dst_state_batch_indices=dst_indices,
                    pad_slot_id=-1,
                    state_scale=inputs.state_scale_work,
                    out=out_dump,
                    cache_steps=mtp_len,
                    algorithm="simple",
                )

        return _run

    if kernel in ("baseline-triton", "baseline-flashinfer"):
        is_fi = kernel == "baseline-flashinfer"
        if is_fi:
            baseline_fn = flashinfer_selective_state_update
        else:
            baseline_fn = selective_state_update

        if with_conv1d:

            def _run():
                x_conv, B_conv, C_conv = _conv1d_split()
                extra = {"algorithm": "simple"} if is_fi else {}
                if inputs.state_scale_work is not None:
                    extra["state_scale"] = inputs.state_scale_work
                baseline_fn(
                    inputs.state_work,
                    x=x_conv,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=B_conv,
                    C=C_conv,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    out=inputs.out_base,
                    disable_state_update=True,
                    intermediate_states_buffer=inputs.intermediate_states_buffer,
                    cache_steps=mtp_len,
                    **extra,
                )
        else:

            def _run():
                extra = {"algorithm": "simple"} if is_fi else {}
                if inputs.state_scale_work is not None:
                    extra["state_scale"] = inputs.state_scale_work
                baseline_fn(
                    inputs.state_work,
                    x=inputs.x,
                    dt=inputs.dt,
                    A=inputs.A,
                    B=inputs.B,
                    C=inputs.C,
                    D=inputs.D,
                    dt_bias=inputs.dt_bias,
                    dt_softplus=True,
                    out=inputs.out_base,
                    disable_state_update=True,
                    intermediate_states_buffer=inputs.intermediate_states_buffer,
                    cache_steps=mtp_len,
                    **extra,
                )

        return _run

    raise ValueError(f"unknown kernel name: {kernel!r}")


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _compute_stats(latencies_us: list[float]) -> tuple[float, float, float]:
    """Return (median_us, p95_us, p99_us) from a list of latencies."""
    median_us = statistics.median(latencies_us)
    s = sorted(latencies_us)
    p95_us = s[int(0.95 * len(s))]
    p99_us = s[int(0.99 * len(s))]
    return median_us, p95_us, p99_us


def _time_kernel_cuda_graph(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> tuple[float, float, float]:
    """
    All-in-one CUDA graph timing.

    Captures a single graph containing warmup iterations followed by timed
    iterations with per-iteration event pairs recorded inside the graph.
    One replay, one sync, then all timings are read.
    """
    warmup = args.warmup
    iters = args.iters

    start_events = [
        torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)
    ]
    end_events = [
        torch.cuda.Event(enable_timing=True, external=True) for _ in range(iters)
    ]

    # Eager warmup before graph capture (triggers Triton autotune if active)
    reset_fn()
    run_fn()
    torch.cuda.synchronize()

    reset_fn()
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        # Warmup iterations (unrolled into the graph)
        for _ in range(warmup):
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            run_fn()

        # Timed iterations with events inside the graph
        for i in range(iters):
            reset_fn()
            if args.l2_flush:
                _l2_flush.fill_(0.0)
            start_events[i].record()
            run_fn()
            end_events[i].record()

    torch.cuda.synchronize()

    # Single replay
    torch.cuda.nvtx.range_push(tag)
    g.replay()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_pop()

    latencies_us = [
        start_events[i].elapsed_time(end_events[i]) * 1000.0 for i in range(iters)
    ]
    return _compute_stats(latencies_us)


def _time_kernel_eager(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> tuple[float, float, float]:
    """Non-CUDA-graph timing path (for debugging, ncu, etc.)."""
    # Warmup
    for _ in range(args.warmup):
        reset_fn()
        run_fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies_us: list[float] = []
    torch.cuda.nvtx.range_push(tag)
    for _ in range(args.iters):
        reset_fn()
        if args.l2_flush:
            _flush_l2()  # includes synchronize
        start_event.record()
        run_fn()
        end_event.record()
        torch.cuda.synchronize()
        latencies_us.append(start_event.elapsed_time(end_event) * 1000.0)
    torch.cuda.nvtx.range_pop()

    return _compute_stats(latencies_us)


def _time_kernel_cupti(
    args,
    run_fn,
    reset_fn,
    tag: str,
) -> tuple[float, float, float]:
    """CUPTI hardware-level GPU kernel timing (requires cupti-python >= 13).
    Source: flashinfer/testing/utils.py bench_gpu_time_with_cupti"""
    from cupti import cupti

    # -- CUPTI buffer callbacks --
    def _buf_requested():
        return 8 * 1024 * 1024, 0  # buffer_size, max_num_records

    def _buf_completed(launches, kernels, activities):
        for a in activities:
            if a.kind in (
                cupti.ActivityKind.CONCURRENT_KERNEL,
                cupti.ActivityKind.MEMCPY,
                cupti.ActivityKind.MEMSET,
            ):
                # `name` is only present on CONCURRENT_KERNEL activities;
                # MEMCPY/MEMSET have no kernel name (guard avoids attr error).
                name = (
                    a.name if a.kind == cupti.ActivityKind.CONCURRENT_KERNEL else None
                )
                kernels.append((a.start, a.end, a.correlation_id, name))
            elif a.kind in (cupti.ActivityKind.RUNTIME, cupti.ActivityKind.DRIVER):
                launches.append((a.start, a.end, a.correlation_id))

    # Eager warmup (triggers Triton autotune etc.)
    reset_fn()
    run_fn()
    torch.cuda.synchronize()

    # Capture two separate graphs: reset+flush vs run
    g_reset = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_reset):
        reset_fn()
        if args.l2_flush:
            _l2_flush.fill_(0.0)
    torch.cuda.synchronize()

    g_run = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g_run):
        run_fn()
    torch.cuda.synchronize()

    # Warmup graph replays
    for _ in range(args.warmup):
        g_reset.replay()
        g_run.replay()
    torch.cuda.synchronize()

    # CUPTI measurement — only measure g_run replay
    launches: list[tuple[int, int, int]] = []
    kernels: list[tuple[int, int, int]] = []
    iter_timestamps: list[tuple[int, int]] = []

    cupti.activity_enable(cupti.ActivityKind.RUNTIME)
    cupti.activity_enable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_enable(cupti.ActivityKind.DRIVER)
    cupti.activity_enable(cupti.ActivityKind.MEMCPY)
    cupti.activity_enable(cupti.ActivityKind.MEMSET)
    cupti.activity_register_callbacks(
        _buf_requested, partial(_buf_completed, launches, kernels)
    )

    torch.cuda.nvtx.range_push(tag)
    for _ in range(args.iters):
        g_reset.replay()
        torch.cuda.synchronize()
        start_cpu = cupti.get_timestamp()
        g_run.replay()
        torch.cuda.synchronize()
        end_cpu = cupti.get_timestamp()
        iter_timestamps.append((start_cpu, end_cpu))
    torch.cuda.nvtx.range_pop()

    cupti.activity_flush_all(0)
    cupti.activity_disable(cupti.ActivityKind.RUNTIME)
    cupti.activity_disable(cupti.ActivityKind.CONCURRENT_KERNEL)
    cupti.activity_disable(cupti.ActivityKind.DRIVER)
    cupti.activity_disable(cupti.ActivityKind.MEMCPY)
    cupti.activity_disable(cupti.ActivityKind.MEMSET)
    cupti.finalize()

    # Build correlation_id → kernel list mapping
    corr_to_kernels: dict[int, list[tuple[int, int, int]]] = {}
    for k in kernels:
        corr_to_kernels.setdefault(k[2], []).append(k)

    # Sort launches by start for binary search
    sorted_launches = sorted(launches, key=lambda launch: launch[0])
    launch_starts = [launch[0] for launch in sorted_launches]

    latencies_us: list[float] = []
    for idx, (t0, t1) in enumerate(iter_timestamps):
        li = bisect.bisect_left(launch_starts, t0)
        ri = bisect.bisect_right(launch_starts, t1)
        corr_ids = {sorted_launches[i][2] for i in range(li, ri)}

        iter_kernels = []
        for cid in corr_ids:
            if cid in corr_to_kernels:
                iter_kernels.extend(corr_to_kernels[cid])

        if not iter_kernels:
            raise ValueError(f"No kernel activities recorded for iteration {idx}")

        min_start = min(k[0] for k in iter_kernels)
        max_end = max(k[1] for k in iter_kernels)
        latencies_us.append((max_end - min_start) / 1e3)  # ns → µs

        if os.environ.get("SSU_CUPTI_DEBUG") and idx == 0:
            print(
                f"\n[CUPTI-DEBUG] tag={tag} iter={idx} n_kernels={len(iter_kernels)}",
                file=sys.stderr,
            )
            for k in sorted(iter_kernels, key=lambda r: r[0]):
                nm = k[3] if len(k) > 3 and k[3] is not None else "<memcpy/memset>"
                print(
                    f"  start={(k[0] - min_start) / 1e3:8.3f}  "
                    f"end={(k[1] - min_start) / 1e3:8.3f} µs  {nm}",
                    file=sys.stderr,
                )

    return _compute_stats(latencies_us)


_CUPTI_UNAVAILABLE_WARNED = False


def _time_kernel(args, run_fn, reset_fn, tag: str) -> tuple[float, float, float]:
    """Dispatch to CUPTI, CUDA-graph, or eager timing path.

    When ``--cupti`` is requested but ``cupti-python`` is not installed, fall
    back to the CUDA-graph path with a one-time warning instead of hard-failing.
    The reset_fn pattern (per-iter state reinit) precludes routing through
    ``flashinfer.testing.bench_gpu_time`` directly.
    """
    # Lazy-init the L2-flush buffer BEFORE entering any CUDA graph capture
    # (the CUDA-graph and CUPTI timing paths call `_l2_flush.fill_(0.0)`
    # directly inside `torch.cuda.graph(...)` where allocations are illegal).
    if args.l2_flush and _l2_flush is None:
        _init_l2_flush()
    if args.cupti:
        try:
            import cupti  # noqa: F401
        except ImportError:
            global _CUPTI_UNAVAILABLE_WARNED
            if not _CUPTI_UNAVAILABLE_WARNED:
                print(
                    "# WARNING: --cupti requested but cupti-python not installed; "
                    "falling back to CUDA-graph timing.  Install with: pip install -U cupti-python",
                    file=sys.stderr,
                )
                _CUPTI_UNAVAILABLE_WARNED = True
        else:
            return _time_kernel_cupti(args, run_fn, reset_fn, tag)
    if args.cuda_graph:
        return _time_kernel_cuda_graph(args, run_fn, reset_fn, tag)
    return _time_kernel_eager(args, run_fn, reset_fn, tag)


# ---------------------------------------------------------------------------
# Per-config benchmark (consolidated baseline + incremental)
# ---------------------------------------------------------------------------


def _bench_config(
    args,
    batch: int,
    mtp_len: int,
    max_window: int,
    prev_ks: list[int],
    state_dtype: torch.dtype,
    act_dtype: torch.dtype,
    baseline_fn,
    philox_rounds: int = 0,
    state_label: str | None = None,
) -> None:
    """
    Benchmark one (batch, mtp_len, dtype) configuration.

    Runs the baseline kernel (if baseline_fn is not None) followed by the
    incremental kernel for each prev_k value.  Tensors are built once and
    shared across all runs in this config.

    When ``philox_rounds > 0`` (f16 state only), the supporting kernels
    (cuda-incr and the Triton incremental reference) receive a ``rand_seed``
    tensor and the rounds count.  Other kernels (baseline) are timed without
    rounding — their f16 path doesn't expose the option.
    """
    # Display label preserves the user's spec verbatim (e.g. "fp16-philox-5"),
    # so CSV/table rows are unambiguous when philox is on.
    state_dtype_name = state_label or str(state_dtype).split(".")[-1]
    act_dtype_name = str(act_dtype).split(".")[-1]
    rand_seed = (
        torch.tensor([0xDECAFBAD], device="cuda", dtype=torch.int64)
        if philox_rounds > 0
        else None
    )

    inputs = build_kernel_inputs(
        batch=batch,
        mtp_len=mtp_len,
        max_window=max_window,
        state_dtype=state_dtype,
        act_dtype=act_dtype,
        nheads=args.tp_nheads,
        head_dim=args.head_dim,
        d_state=args.d_state,
        ngroups=args.tp_ngroups,
        two_kernel=args.two_kernel,
    )

    # Reuse args directly as the TimingOptions duck — it carries .warmup,
    # .iters, .cupti, .cuda_graph, .l2_flush.
    timing = args

    show_kernel_col = (
        baseline_fn is not None
        or args.flashinfer_dump
        or args.cuda_incr
        or args.triton_replay
        or args.triton_replay_pm
    )

    pt_uniform_i32 = torch.zeros(batch, dtype=torch.int32, device="cuda")

    # --- Baseline (no prev_k dependence; one row total) ---
    if baseline_fn is not None:
        tag = f"base_b{batch}_mtp{mtp_len}_s{state_dtype_name}_a{act_dtype_name}"
        kernel_name = (
            "baseline-flashinfer"
            if args.baseline == "flashinfer"
            else "baseline-triton"
        )
        pt_uniform_i32.zero_()
        median_us, p95_us, p99_us = time_kernel(
            kernel=kernel_name,
            inputs=inputs,
            prev_tokens=pt_uniform_i32,
            timing=timing,
            tag=tag,
            philox_rounds=0,  # baselines don't expose philox
            rand_seed=None,
        )
        _print_row(
            show_kernel_col,
            args.baseline,
            batch,
            mtp_len,
            "N/A",
            state_dtype_name,
            act_dtype_name,
            median_us,
            p95_us,
            p99_us,
        )

    def _parse_sweep(val):
        if val is None:
            return [None]
        return [int(x) for x in val.split(",")]

    bsm_values = _parse_sweep(args.block_size_m)
    nw_values = _parse_sweep(args.num_warps)
    ns_values = _parse_sweep(args.num_stages)
    pnw_values = _parse_sweep(args.precompute_num_warps)
    pns_values = _parse_sweep(args.precompute_num_stages)

    # --- Triton incremental kernel, one row per prev_k × autotune-point ---
    for prev_k in prev_ks:
        pt_uniform_i32.fill_(prev_k)
        tag = (
            f"incr_b{batch}_mtp{mtp_len}_k{prev_k}"
            f"_s{state_dtype_name}_a{act_dtype_name}"
        )
        for bsm in bsm_values:
            for nw in nw_values:
                for ns in ns_values:
                    for pnw in pnw_values:
                        for pns in pns_values:
                            autotune = TritonAutotune(
                                block_size_m=bsm,
                                num_warps=nw,
                                num_stages=ns,
                                precompute_num_warps=pnw,
                                precompute_num_stages=pns,
                                internal_pdl=args.internal_pdl,
                            )
                            parts = []
                            if bsm is not None:
                                parts.append(f"M={bsm}")
                            if nw is not None:
                                parts.append(f"W={nw}")
                            if ns is not None:
                                parts.append(f"S={ns}")
                            if pnw is not None:
                                parts.append(f"pW={pnw}")
                            if pns is not None:
                                parts.append(f"pS={pns}")
                            sweep_suffix = (" " + ",".join(parts)) if parts else ""
                            sweep_tag = tag + sweep_suffix.replace(" ", "_").replace(
                                ",", "_"
                            )
                            median_us, p95_us, p99_us = time_kernel(
                                kernel="incremental",
                                inputs=inputs,
                                prev_tokens=pt_uniform_i32,
                                timing=timing,
                                tag=sweep_tag,
                                philox_rounds=philox_rounds,
                                rand_seed=rand_seed,
                                autotune=autotune,
                            )
                            _print_row(
                                show_kernel_col,
                                "incremental",
                                batch,
                                mtp_len,
                                prev_k,
                                state_dtype_name,
                                act_dtype_name,
                                median_us,
                                p95_us,
                                p99_us,
                                sweep_suffix,
                            )

    # --- CUDA checkpointing_ssu kernel ---
    if args.cuda_incr:
        for prev_k in prev_ks:
            pt_uniform_i32.fill_(prev_k)
            tag = (
                f"cuda_incr_b{batch}_mtp{mtp_len}_k{prev_k}"
                f"_s{state_dtype_name}_a{act_dtype_name}"
            )
            median_us, p95_us, p99_us = time_kernel(
                kernel="cuda-incr",
                inputs=inputs,
                prev_tokens=pt_uniform_i32,
                timing=timing,
                tag=tag,
                philox_rounds=philox_rounds,
                rand_seed=rand_seed,
            )
            _print_row(
                show_kernel_col,
                "cuda-incr",
                batch,
                mtp_len,
                prev_k,
                state_dtype_name,
                act_dtype_name,
                median_us,
                p95_us,
                p99_us,
            )

    # --- Triton replay (new 5D persistent path) ---
    if args.triton_replay:
        for prev_k in prev_ks:
            pt_uniform_i32.fill_(prev_k)
            tag = (
                f"triton_replay_b{batch}_mtp{mtp_len}_k{prev_k}"
                f"_s{state_dtype_name}_a{act_dtype_name}"
            )
            median_us, p95_us, p99_us = time_kernel(
                kernel="triton-replay",
                inputs=inputs,
                prev_tokens=pt_uniform_i32,
                timing=timing,
                tag=tag,
                philox_rounds=philox_rounds,
                rand_seed=rand_seed,
            )
            _print_row(
                show_kernel_col,
                "triton-replay",
                batch,
                mtp_len,
                prev_k,
                state_dtype_name,
                act_dtype_name,
                median_us,
                p95_us,
                p99_us,
            )

    # --- Triton replay, persistent_main mode (write/nowrite two-launch split) ---
    if args.triton_replay_pm:
        for prev_k in prev_ks:
            pt_uniform_i32.fill_(prev_k)
            tag = (
                f"triton_replay_pm_b{batch}_mtp{mtp_len}_k{prev_k}"
                f"_s{state_dtype_name}_a{act_dtype_name}"
            )
            median_us, p95_us, p99_us = time_kernel(
                kernel="triton-replay-pm",
                inputs=inputs,
                prev_tokens=pt_uniform_i32,
                timing=timing,
                tag=tag,
                philox_rounds=philox_rounds,
                rand_seed=rand_seed,
            )
            _print_row(
                show_kernel_col,
                "triton-replay-pm",
                batch,
                mtp_len,
                prev_k,
                state_dtype_name,
                act_dtype_name,
                median_us,
                p95_us,
                p99_us,
            )

    # --- FlashInfer dynamic-dump ---
    if args.flashinfer_dump:
        for prev_k in prev_ks:
            pt_uniform_i32.fill_(prev_k)
            tag = (
                f"fi_dump_b{batch}_mtp{mtp_len}_k{prev_k}"
                f"_s{state_dtype_name}_a{act_dtype_name}"
            )
            median_us, p95_us, p99_us = time_kernel(
                kernel="fi-dump",
                inputs=inputs,
                prev_tokens=pt_uniform_i32,
                timing=timing,
                tag=tag,
            )
            _print_row(
                show_kernel_col,
                "fi-dump",
                batch,
                mtp_len,
                prev_k,
                state_dtype_name,
                act_dtype_name,
                median_us,
                p95_us,
                p99_us,
            )


def _print_row(
    show_kernel_col,
    kernel_name,
    batch,
    mtp_len,
    prev_k,
    state_dtype_name,
    act_dtype_name,
    median_us,
    p95_us,
    p99_us,
    sweep_suffix="",
):
    kernel_col = f"{kernel_name:>11} | " if show_kernel_col else ""
    print(
        f"| {kernel_col}{batch:>5} | {mtp_len:>7} | {str(prev_k):>6} | "
        f"{state_dtype_name:>14} | {act_dtype_name:>9} | "
        f"{median_us:>9.2f} | {p95_us:>7.2f} | {p99_us:>7.2f} |"
        f"{sweep_suffix}"
    )


# ---------------------------------------------------------------------------
# Main benchmark loop
# ---------------------------------------------------------------------------


def _run_benchmark(args) -> None:
    assert args.nheads % args.tp_size == 0, (
        f"nheads ({args.nheads}) must be divisible by tp_size ({args.tp_size})"
    )
    assert args.ngroups % args.tp_size == 0, (
        f"ngroups ({args.ngroups}) must be divisible by tp_size ({args.tp_size})"
    )
    args.tp_nheads = args.nheads // args.tp_size
    args.tp_ngroups = args.ngroups // args.tp_size

    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]
    mtp_lengths = [int(x) for x in args.mtp_lengths.split(",")]
    max_window = args.max_window
    assert max_window >= max(mtp_lengths), (
        f"--max-window ({max_window}) must be >= max(mtp_lengths) "
        f"({max(mtp_lengths)}); the incremental kernel requires npredicted <= max_window"
    )

    state_specs = [parse_state_spec(s) for s in args.state_dtypes.split(",")]
    act_dtype_tokens = [t.strip() for t in args.act_dtypes.split(",")]
    _bad_act = [t for t in act_dtype_tokens if t not in STATE_DTYPE_MAP]
    if _bad_act:
        raise ValueError(
            f"invalid --act-dtypes token(s): {_bad_act!r} "
            f"(allowed: {sorted(STATE_DTYPE_MAP)})"
        )
    act_dtypes = [STATE_DTYPE_MAP[t] for t in act_dtype_tokens]

    # Resolve baseline function
    if args.baseline == "flashinfer":
        from flashinfer.mamba import selective_state_update as baseline_fn
    elif args.baseline == "triton":
        baseline_fn = selective_state_update
    else:
        baseline_fn = None

    if args.l2_flush:
        _init_l2_flush()

    if args.profile:
        torch.cuda.cudart().cudaProfilerStart()

    # Print header
    show_kernel_col = (
        baseline_fn is not None
        or args.flashinfer_dump
        or args.cuda_incr
        or args.triton_replay
        or args.triton_replay_pm
    )
    if show_kernel_col:
        print(
            f"| {'kernel':>11} | {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>14} | {'act_dtype':>9} | "
            f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 13}|{'-' * 7}|{'-' * 9}|{'-' * 8}|"
            f"{'-' * 16}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )
    else:
        print(
            f"| {'batch':>5} | {'mtp_len':>7} | {'prev_k':>6} | "
            f"{'state_dtype':>14} | {'act_dtype':>9} | "
            f"{'median_us':>9} | {'p95_us':>7} | {'p99_us':>7} |"
        )
        print(
            f"|{'-' * 7}|{'-' * 9}|{'-' * 8}|"
            f"{'-' * 16}|{'-' * 11}|{'-' * 11}|{'-' * 9}|{'-' * 9}|"
        )

    for batch in batch_sizes:
        for mtp_len in mtp_lengths:
            # Resolve prev_k fractions → clamped integers in [0, mtp_len]
            prev_ks = sorted(
                set(
                    min(mtp_len, max(0, round(f * mtp_len)))
                    for f in args.prev_tokens_fracs
                )
            )
            for state_dtype, philox_rounds, state_label in state_specs:
                for act_dtype in act_dtypes:
                    _bench_config(
                        args,
                        batch,
                        mtp_len,
                        max_window,
                        prev_ks,
                        state_dtype,
                        act_dtype,
                        baseline_fn,
                        philox_rounds=philox_rounds,
                        state_label=state_label,
                    )

    if args.profile:
        torch.cuda.cudart().cudaProfilerStop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark checkpointing_ssu CUDA kernel vs Triton reference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--nheads",
        type=int,
        default=NHEADS,
        help="Full-model nheads (divided by --tp-size for per-GPU slice)",
    )
    parser.add_argument(
        "--ngroups",
        type=int,
        default=NGROUPS,
        help="Full-model ngroups (divided by --tp-size for per-GPU slice)",
    )
    parser.add_argument(
        "--head-dim", type=int, default=HEAD_DIM, help="Head dimension (not TP-split)"
    )
    parser.add_argument(
        "--d-state",
        type=int,
        default=D_STATE,
        help="SSM state dimension (not TP-split)",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=TP_SIZE,
        help="Tensor parallel size; divides nheads and ngroups",
    )
    parser.add_argument(
        "--batch-sizes", default="1,2,4,8", help="Comma-separated decode batch sizes"
    )
    parser.add_argument(
        "--mtp-lengths",
        default="1,2,4,8",
        help="Comma-separated MTP speculation depths",
    )
    parser.add_argument(
        "--max-window",
        type=int,
        default=16,
        help="Cache capacity (T-axis size for old_x/old_B/old_dt/old_cumAdt). "
        "The CUDA kernel triggers must_checkpoint when prev_k + mtp_len > max_window; "
        "Triton receives a matching write_checkpoint flag for apples-to-apples timing. "
        "Must be >= max(mtp_lengths).",
    )
    parser.add_argument(
        "--state-dtypes",
        default="fp32",
        help="Comma-separated state dtypes: f16, bf16, f32, int8, fp8 "
        "(fp16/fp32/i8/e4m3 aliases).  int8 and fp8 use per-(head, dim) block "
        "scaling (state_scale tensor auto-created). "
        "Append '-philox-<N>' to f16/fp16/int8/i8/fp8/e4m3 to enable Philox-<N> "
        "stochastic rounding on gmem state writes (cuda-incr / Triton incr only). "
        "Example: 'fp16-philox-5', 'int8-philox-10', 'fp8-philox-5'.",
    )
    parser.add_argument(
        "--act-dtypes",
        default="bf16",
        help="Comma-separated activation dtypes for x/B/C/dt: fp32,bf16",
    )
    parser.add_argument(
        "--warmup", type=int, default=20, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--iters", type=int, default=100, help="Number of timed iterations"
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Wrap timed region in cudaProfilerStart/Stop "
        "(for ncu --target-processes all)",
    )
    parser.add_argument(
        "--l2-flush",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="L2 eviction between iterations",
    )
    parser.add_argument(
        "--two-kernel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run the cuda-incr path as the two-kernel split (precompute + "
        "main, caller-provided cb_scaled/cumAdt_vec/cb_old scratch). On by "
        "default; --no-two-kernel runs the monolithic kernel instead.",
    )
    parser.add_argument(
        "--cuda-graph",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture all warmup + timed iterations in a "
        "single CUDA graph with per-iteration events "
        "inside the graph, eliminating all host overhead.",
    )
    parser.add_argument(
        "--cupti",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use CUPTI hardware-level profiling for timing "
        "(requires cupti-python >= 13). Falls back to "
        "CUDA events if CUPTI is unavailable.",
    )
    parser.add_argument(
        "--prev-tokens-fracs",
        default="0,0.5,1.0",
        type=lambda s: [float(x) for x in s.split(",")],
        help="Fractions of mtp_len to use as prev_num_accepted_tokens "
        "for the incremental kernel sweep. Values are rounded "
        "and clamped to [0, mtp_len].",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        nargs="?",
        const="triton",
        choices=[None, "triton", "flashinfer"],
        help="Baseline to benchmark alongside the incremental kernel. "
        "'triton': native Triton selective_state_update. "
        "'flashinfer': flashinfer selective_state_update (same signature). "
        "Pass --baseline alone for 'triton'. Default: no baseline.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save results (file or directory). "
        "If a directory, writes bench_checkpointing_ssu_<timestamp>.txt inside it.",
    )
    parser.add_argument(
        "--block-size-m",
        type=str,
        default=None,
        help="Override BLOCK_SIZE_M: single value or comma-separated sweep (e.g. '4,8,16,32').",
    )
    parser.add_argument(
        "--num-warps",
        type=str,
        default=None,
        help="Override num_warps: single value or comma-separated sweep (e.g. '1,2,4').",
    )
    parser.add_argument(
        "--internal-pdl",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Internal PDL between precompute and main kernels (default: on).",
    )
    parser.add_argument(
        "--num-stages",
        type=str,
        default=None,
        help="Override num_stages for the main kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--precompute-num-warps",
        type=str,
        default=None,
        help="Override num_warps for precompute kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--precompute-num-stages",
        type=str,
        default=None,
        help="Override num_stages for precompute kernel (comma-separated sweep).",
    )
    parser.add_argument(
        "--cuda-incr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run CUDA checkpointing_ssu kernel (default: on).",
    )
    parser.add_argument(
        "--triton-replay",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Triton replay_selective_state_update, persistent_dynamic "
        "(5D persistent path) (default: on).",
    )
    parser.add_argument(
        "--triton-replay-pm",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run Triton replay_selective_state_update, persistent_main "
        "(write/nowrite two-launch split) (default: on).",
    )
    parser.add_argument(
        "--flashinfer-dump",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run FlashInfer dynamic-dump scenario: write state only at prev_k "
        "via dst_state_batch_indices (default: on).",
    )
    return parser.parse_args()


class _Tee:
    """Write to both stdout and a file simultaneously."""

    def __init__(self, path: str):
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._file = open(path, "w")  # noqa: SIM115
        self._stdout = sys.stdout

    def write(self, data):
        self._stdout.write(data)
        self._file.write(data)

    def flush(self):
        self._stdout.flush()
        self._file.flush()

    def close(self):
        self._file.close()


if __name__ == "__main__":
    _args = _parse_args()

    _out_path = None
    if _args.output != "-":
        _ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        _fname = f"bench_checkpointing_ssu_{_ts}.txt"
        if _args.output is None:
            _out_path = os.path.expanduser(f"~/nemo_logs/{_fname}")
        elif os.path.isdir(_args.output) or _args.output.endswith("/"):
            _out_path = os.path.join(_args.output, _fname)
        else:
            _out_path = _args.output

    _tee = None
    if _out_path:
        _tee = _Tee(_out_path)
        sys.stdout = _tee
        print(f"# bench_checkpointing_ssu  {datetime.now().isoformat()}")
        print(f"# cmd: {' '.join(sys.argv)}")

    try:
        _run_benchmark(_args)
    finally:
        if _tee is not None:
            sys.stdout = _tee._stdout
            _tee.close()
            print(f"\nResults saved to: {_out_path}")
