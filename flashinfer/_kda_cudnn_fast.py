# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cached native submission of the existing cuDNN KDA compiled chain."""

import itertools
import logging
import math
import os
from typing import Any

import torch

cuDNNFastKDA = None
_native_unavailable = False
_enabled = os.environ.get("FLASHINFER_CUDNN_KDA_NATIVE", "1") != "0"
_logger = logging.getLogger(__name__)


_entries: list[tuple[tuple, Any]] = []
_names = itertools.count()
_stats = {"hits": 0, "misses": 0, "prepared": 0}
_max_entries = 8
_scalar_types = (type(None), int, float, bool)


def stats():
    return dict(_stats, entries=len(_entries))


def clear():
    """Release cached executors after their streams finish."""
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError("Clear KDA executors outside CUDA Graph capture")
    if any(not executor.can_release() for _, executor in _entries):
        raise RuntimeError("A cached KDA stream is capturing")
    _entries.clear()


def try_execute(inputs, config, output_final_state):
    if not _enabled or cuDNNFastKDA is None or not torch.is_inference_mode_enabled():
        return None
    state = inputs[9]
    if not isinstance(state, torch.Tensor) or state.requires_grad:
        return None
    if (
        type(config[0]) not in _scalar_types
        or type(config[1]) not in _scalar_types
        or type(config[2]) is not bool
        or type(config[3]) is not bool
        or type(config[4]) is not bool
    ):
        return None
    for key, executor in reversed(_entries):
        if key == config and executor.execute(inputs):
            _stats["hits"] += 1
            torch.autograd.graph.increment_version(state)
            return inputs[6].reshape(inputs[2].shape), (
                state if output_final_state else None
            )
    _stats["misses"] += 1
    return None


def prepare(inputs, config):
    """Prepare after the ordinary API has executed and validated the call."""
    global _native_unavailable
    try:
        _prepare(inputs, config)
    except (AttributeError, ImportError, IndexError, TypeError, ValueError) as error:
        # Private frontend interfaces may differ across installed versions.
        _native_unavailable = True
        _logger.warning(
            "cuDNN KDA native plan unavailable; using original path: %s", error
        )


def _prepare(inputs, config):
    global cuDNNFastKDA, _native_unavailable
    if not _enabled or _native_unavailable or not torch.is_inference_mode_enabled():
        return
    if (
        type(config[0]) not in _scalar_types
        or type(config[1]) not in _scalar_types
        or type(config[2]) is not bool
        or type(config[3]) is not bool
        or type(config[4]) is not bool
    ):
        return
    if any(not isinstance(x, torch.Tensor) for x in inputs):
        return
    q, k, v, g, beta, cu, out, a_log, bias, state = inputs
    if state.requires_grad or not all(x.is_cuda and x.is_contiguous() for x in inputs):
        return
    if any(x.dtype != torch.bfloat16 for x in (q, k, v, g, beta, out, state)):
        return
    if (
        a_log.dtype != torch.float32
        or bias.dtype != torch.float32
        or cu.dtype != torch.int32
    ):
        return
    if not hasattr(type(q), "__dlpack_c_exchange_api__"):
        return
    if q.shape[-1] != 128 or q.device.index != torch.cuda.current_device():
        return
    if torch.cuda.is_current_stream_capturing():
        return
    scale, lower_bound, norm, gate, sigmoid = config
    if not (norm and gate and sigmoid):
        return

    from .cudnn import linear_attention as la
    from cudnn.frost.workspace import Workspace
    import tvm_ffi

    q3, k3, v3, g3 = (x.squeeze(0) if x.dim() == 4 else x for x in (q, k, v, g))
    b2 = beta.squeeze(0) if beta.dim() == 3 else beta
    o3 = out.squeeze(0) if out.dim() == 4 else out
    bias2 = bias.reshape(-1, 128)
    graph, _ = la._build_la_graph(
        "kda",
        q3,
        k3,
        v3,
        g3,
        b2,
        cu,
        o3,
        a_log=a_log,
        dt_bias=bias2,
        initial_state=state,
        final_state=state,
        scale=1.0 / math.sqrt(128) if scale is None else float(scale),
        use_qk_l2norm=norm,
        use_beta_sigmoid=sigmoid,
        safe_gate=gate,
        gate_lower_bound=lower_bound,
        batch_invariant=False,
        overwrite_initial_state=True,
    )
    if not getattr(graph, "_fi_la_overwrite", False) or not getattr(
        graph, "_fi_la_ordered", False
    ):
        return
    plans = getattr(graph, "_compiled_plans", None)
    if plans is None or not hasattr(graph, "_normalize_ordered"):
        return
    plan = plans[graph._plan_index]
    compiled = getattr(plan, "compiled", None)
    if (
        not getattr(compiled, "chain", False)
        or getattr(compiled, "chain_launch", None) is None
    ):
        return
    # Build only after the original plan confirms the supported chain contract.
    if cuDNNFastKDA is None:
        from .jit.cudnn_fast_kda import get_cudnn_fast_kda_module

        try:
            cuDNNFastKDA = get_cudnn_fast_kda_module().cuDNNFastKDA
        except (ImportError, OSError, RuntimeError) as error:
            _native_unavailable = True
            _logger.warning(
                "cuDNN KDA native helper unavailable; using original path: %s", error
            )
            return
    workspace = torch.empty(
        graph._fi_la_workspace_size, dtype=torch.uint8, device=q.device
    )
    buffers = (q3, k3, v3, g3, b2, cu, o3, a_log, bias2, state, state)
    pack = graph._normalize_ordered(
        buffers, graph._fi_la_uids, workspace, None, None, None
    )
    views = pack.operands(plan.indices)
    ws = Workspace.over(pack, compiled.workspace_size, type(compiled).__name__)
    operands = compiled.chain_buffers((*views, *ws.carve(compiled.carve), None))
    stream = torch.cuda.current_stream()
    frame = (
        *compiled.chain_scalars,
        *operands,
        compiled.stream_type(stream.cuda_stream),
    )
    source = {
        "q": 0,
        "k": 1,
        "v": 2,
        "gate": 3,
        "beta": 4,
        "cu_seqlens": 5,
        "o": 6,
        "a_log": 7,
        "dt_bias": 8,
        "seed": 9,
        "final_state": 9,
    }
    slots = [-1] * len(compiled.chain_scalars)
    slots += [source.get(name, -1) for name in compiled.chain_buffer_names]
    slots += [-1]
    owners = (graph, plan, compiled, workspace, pack, views, ws, operands, stream)
    name = f"flashinfer.cudnn_fast_kda.{next(_names)}"
    tvm_ffi.register_global_func(name, compiled.chain_launch.__tvm_ffi_object__())
    try:
        try:
            executor = cuDNNFastKDA(name, frame, inputs, slots, owners)
        except ValueError as error:
            _stats["last_prepare_rejection"] = str(error)
            return
    finally:
        tvm_ffi.remove_global_func(name)
    if len(_entries) == _max_entries:
        if not _entries[0][1].can_release():
            return
        _entries.pop(0)
    _entries.append((config, executor))
    _stats["prepared"] += 1
