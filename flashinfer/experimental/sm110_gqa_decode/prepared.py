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

# Prepared FP16 decode for exact SM110a (Thor), CUDA 13 or newer.

from __future__ import annotations

import functools
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .jit import _check_exact_sm110a, load_sm110_gqa_decode_module

_ROOT = Path(__file__).resolve().parent / "csrc" / "prepared"


@functools.cache
def _manifest() -> dict[str, Any]:
    manifest = json.loads((_ROOT / "manifest.json").read_text())
    if manifest["contract"]["architecture"] != "sm_110a":
        raise RuntimeError("unexpected prepared decode architecture")
    for artifact in manifest["files"]:
        relative = Path(artifact["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("invalid generated artifact path")
        payload = (_ROOT / relative).read_bytes()
        if hashlib.sha256(payload).hexdigest() != artifact["sha256"]:
            raise RuntimeError(f"generated artifact changed: {relative}")
    return manifest


@functools.cache
def _record(key: str) -> dict[str, Any]:
    records = [r for r in _manifest()["modules"] if r["route"] == key]
    if len(records) != 1:
        raise ValueError(f"route stage {key!r} is not uniquely exported")
    return records[0]


@functools.cache
def _module(key: str, device: Any) -> Any:
    from ...jit.core import gen_jit_spec, sm110a_nvcc_flags

    record = _record(key)
    if record["kind"] == "original":
        return load_sm110_gqa_decode_module(device=device)
    identity = hashlib.sha256(
        json.dumps(_manifest(), sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()[:20]
    return gen_jit_spec(
        name=f"sm110_gqa_decode_prepared_{key}_{identity}",
        sources=[_ROOT / path for path in record["sources"]],
        extra_cuda_cflags=[*sm110a_nvcc_flags, *record["compile_flags"]],
        extra_ldflags=["-lcuda"],
    ).build_and_load()


def _stage(key: str, bindings: dict[str, Any], grid: tuple[int, int, int]) -> tuple:
    record = _record(key)
    module = _module(key, bindings["Q"].device)
    values = {**bindings, "grid_x": grid[0], "grid_y": grid[1], "grid_z": grid[2]}
    arguments = tuple(values[name] for _kind, name in record["arg_plan"])
    return getattr(module, record["ffi_entry"]), arguments, record["kernel_symbol"]


def prepare_for_launch(
    inputs: dict[str, Any], num_splits: int | None = None
) -> dict[str, Any]:
    """Prepare a fixed shape without copying sequence lengths to the host.

    The caller supplies contiguous CUDA tensors ``Q`` [B,32,128], ``KV``
    [B,2,8,capacity,128], ``O`` [B,32,128] (all FP16), and int32
    ``sequence_lengths`` [B]. ``O`` must not share storage with any input.
    Optional ``q_scale`` is finite and positive, and defaults to 1.

    Every sequence length must be in [1, capacity] at every launch. Their
    GPU values may change between launches; this precondition is the caller's
    responsibility and is not checked by a host synchronization here.

    Default routes specialize B=4/capacity=256 and B=1/capacity=1024 or 4096.
    Other capacities above 64 use the original long route. ``num_splits=1``
    explicitly selects original long above capacity 64, including B=4/256;
    the two specialized long routes also accept their fixed split count 10.
    Capacities up to 64 always use original short. Unsupported split choices
    are rejected rather than generating a new kernel.

    Prepare and warm up outside Graph capture. Keep the returned object alive
    until all launches complete and any captured Graphs are retired. Its
    bindings, route, and workspace are opaque and must not be modified.
    """
    import torch

    q, kv, output, lengths = (inputs[k] for k in ("Q", "KV", "O", "sequence_lengths"))
    if q.ndim != 3 or tuple(q.shape[1:]) != (32, 128):
        raise ValueError("Q must have shape [B,32,128]")
    batch = int(q.shape[0])
    if batch < 1 or tuple(output.shape) != tuple(q.shape):
        raise ValueError("O must match nonempty Q")
    if kv.ndim != 5 or tuple(kv.shape[:3]) != (batch, 2, 8) or kv.shape[-1] != 128:
        raise ValueError("KV must have shape [B,2,8,capacity,128]")
    capacity = int(kv.shape[-2])
    if capacity < 1:
        raise ValueError("KV capacity must be positive")
    if any(t.dtype != torch.float16 for t in (q, kv, output)):
        raise TypeError("Q, KV and O must be FP16")
    if lengths.dtype != torch.int32 or tuple(lengths.shape) != (batch,):
        raise TypeError("sequence_lengths must be int32[B]")
    tensors = (q, kv, output, lengths)
    if q.device.type != "cuda" or any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must share one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("prepared tensors must be contiguous")
    if output.untyped_storage().data_ptr() in tuple(
        t.untyped_storage().data_ptr() for t in (q, kv, lengths)
    ):
        raise ValueError("O must not alias any input storage")
    _check_exact_sm110a(q.device)
    q_scale = float(inputs.get("q_scale", 1.0))
    if not math.isfinite(q_scale) or q_scale <= 0:
        raise ValueError("q_scale must be finite and positive")
    if num_splits is not None and (
        type(num_splits) is not int or num_splits not in (1, 2, 4, 8, 10, 16)
    ):
        raise ValueError("num_splits must be None or an integer in {1,2,4,8,10,16}")
    contract = _manifest()["contract"]
    single_routes = set(contract["single_routes"])
    fused_routes = contract["fused_routes"]
    route = contract["routes"].get(f"{batch}:{capacity}", "long")
    if capacity <= 64:
        route = "short"
    elif num_splits is not None:
        if route in fused_routes:
            if num_splits == 1:
                route = "long"
            elif num_splits != fused_routes[route]:
                raise ValueError("the exported fused route has a fixed split count")
        elif num_splits == 1 and route == "n32_b4_direct":
            # The default direct route has no scratch or merge. Preserve the
            # explicit one-split override on the original long implementation.
            route = "long"
        elif num_splits > 1:
            raise ValueError("shape does not select an exported split tile")
    if route not in single_routes and route not in fused_routes:
        raise ValueError(f"selected route {route!r} was not exported")

    bindings = {
        "Q": q.view(batch, 8, 4, 128).transpose(1, 2),
        "K": kv[:, 0],
        "V": kv[:, 1],
        "O": output,
        "sequence_lengths": lengths,
        "kv_capacity": capacity,
        "softmax_scale_log2": q_scale / math.sqrt(128) / math.log(2),
    }
    workspace: tuple[torch.Tensor, ...] = ()
    if route in single_routes:
        split_count = 1
        stages = (_stage(route, bindings, (batch * 8, 1, 1)),)
    else:
        split_count = fused_routes[route]
        partial_o = torch.empty(
            (batch, 32, split_count, 128), dtype=torch.float32, device=q.device
        )
        partial_max = torch.empty(
            (batch, 32, split_count), dtype=torch.float32, device=q.device
        )
        partial_sum = torch.empty_like(partial_max)
        # The last CTA resets its completion counter after merging. Ordered
        # launches and Graph replays need no additional reset kernel.
        completed = torch.zeros((batch * 8,), dtype=torch.uint32, device=q.device)
        workspace = (partial_o, partial_max, partial_sum, completed)
        bindings.update(
            partial_O=partial_o,
            partial_max=partial_max,
            partial_sum=partial_sum,
            completed=completed,
        )
        stages = (_stage(route, bindings, (batch * 8 * split_count, 1, 1)),)
    return {
        "route": route,
        "num_splits": split_count,
        "O": output,
        "bindings": bindings,
        "workspace": workspace,
        "stages": stages,
        "launch_names": [stage[2] for stage in stages],
        "workspace_bytes": sum(t.numel() * t.element_size() for t in workspace),
    }


def launch_prepared(prepared: dict[str, Any]) -> Any:
    """Launch on the current PyTorch CUDA stream and return the caller's O.

    Retain ``prepared`` until asynchronous work finishes, or throughout the
    lifetime of any CUDA Graph capturing this launch. Concurrent invocations
    require distinct prepared objects and output/workspace storage. Reuse on
    ordered streams, including event-ordered handoffs, is supported.
    """
    import tvm_ffi

    with tvm_ffi.use_torch_stream():
        for call, arguments, _name in prepared["stages"]:
            call(*arguments)
    return prepared["O"]
