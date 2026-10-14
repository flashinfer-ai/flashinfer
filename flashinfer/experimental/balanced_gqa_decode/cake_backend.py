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

Cake backend: on-device load-balanced BF16 paged GQA decode (SM100/SM103).

One persistent launch (one CTA per SM) serves a whole ragged decode batch.  A
scheduler warp reads the device ``seq_lens`` buffer, derives a chunk length and
four length buckets, and hands ``(query row tile, kv head, KV block range)``
work items to the attention warps through a self-resetting global ticket
counter; long requests are split into chunks whose FP32 partials are merged by
the last CTA to finish the tile.  Nothing about the plan is decided on the
host, so a runner captured once into a CUDA Graph replays correctly for any
KV-length distribution written into ``seq_lens`` later.  See ``README.md`` in
this package and flashinfer-ai/flashinfer#4832.

The kernels are the ``csrc/cake_fmha`` balanced components.  ``q_len_per_req``
1 runs ``decode_balanced_bf16`` (one 8-head query row per work item);
``q_len_per_req`` 3..8 (speculative / MTP verify) runs the packed-row programs
``decode_balanced_bf16_mtp_n32`` / ``_n64``: one ``8 * q_len``-row tile per
``(request, kv head)`` item so each KV chunk is streamed once per request,
with the same on-device scheduler and a distributed reduce (one to eight
reduce tickets per split tile).  This module validates the batch, carves the
caller-owned workspace and binds the production launch adapter
(``cake_paged_attention_decode`` of the module built by
``flashinfer.jit.cake_fmha.load_cake_fmha_decode_balanced_bf16_module``).
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import torch
import tvm_ffi

from ...cake_fmha import (
    cake_fmha_balanced_counter_bytes,
    cake_fmha_balanced_workspace_bytes,
)
from ...jit.cake_fmha import (
    cake_fmha_balanced_component_name,
    get_cake_fmha_manifest,
    load_cake_fmha_decode_balanced_bf16_module,
)
from .cake_bounds import (
    MAX_REQUESTS,
    MTP_COUNTERS_PER_TILE,
    MTP_MAX_Q_LEN,
    MTP_MIN_Q_LEN,
    MTP_STATS_PER_SLOT,
    PLAN_FACTS_WORDS,
    QUEUE_COUNTERS,
    ROW_COUNTERS_PER_TILE,
    ROW_STATS_PER_SLOT,
    serves_q_len,
    uses_packed_mtp,
    workspace_bounds,
)

HEAD_DIM = 128
PAGE_SIZE = 16
GROUP_RATIO = 8  # query heads per KV head served by one MMA tile
WORKSPACE_ALIGN = 256
# Compute capability -> cubin target of the Cake FMHA loaders.
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm100a", (10, 3): "sm103a"}


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------


def _device_index(device: Optional[torch.device] = None) -> int:
    if device is None:
        return int(torch.cuda.current_device())
    index = torch.device(device).index
    return int(torch.cuda.current_device() if index is None else index)


@functools.cache
def _multi_processor_count(device_index: int) -> int:
    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


@functools.cache
def _device_target(device_index: int) -> Optional[str]:
    """``sm100a`` / ``sm103a`` of a device, ``None`` for other capabilities."""
    return SUPPORTED_COMPUTE_CAPABILITIES.get(
        torch.cuda.get_device_capability(device_index)
    )


def num_persistent_ctas(device: Optional[torch.device] = None) -> int:
    """Grid size of the persistent launch: one CTA per SM (queried once per device)."""
    return _multi_processor_count(_device_index(device))


def _align(nbytes: int, alignment: int = WORKSPACE_ALIGN) -> int:
    return (nbytes + alignment - 1) // alignment * alignment


def workspace_layout(num_ctas: int) -> dict:
    """Byte ``(offset, size)`` of the two workspace regions plus ``"total"``.

    ``"partials"`` is the adapter's ``workspace_buffer`` (FP32 partial outputs
    and statistics of split items plus the reserved plan-facts slot) and
    ``"counters"`` its ``multi_ctas_kv_counter_buffer`` (tile counters, then
    the 16-byte-aligned queue counters).  Both are sized by the production
    bounds for the packed-row programs' 64-row slots and four counter words
    per tile, which also bound the row program: one buffer serves every
    ``q_len_per_req`` on the device, and the layout depends on the CTA count
    only.
    """
    if num_ctas <= 0:
        raise ValueError("num_ctas must be positive")
    sizes = (
        ("partials", cake_fmha_balanced_workspace_bytes(num_ctas, MTP_MAX_Q_LEN)),
        ("counters", cake_fmha_balanced_counter_bytes(num_ctas, MTP_MAX_Q_LEN)),
    )
    layout: dict = {}
    offset = 0
    for name, nbytes in sizes:
        layout[name] = (offset, nbytes)
        offset += _align(nbytes)
    layout["total"] = offset
    return layout


def balanced_gqa_decode_workspace_size(
    device: Optional[torch.device] = None, *, num_sms: Optional[int] = None
) -> int:
    """Workspace bytes for any batch and ``q_len_per_req`` on ``device`` (or ``num_sms`` CTAs)."""
    if num_sms is None:
        num_sms = num_persistent_ctas(device)
    return int(workspace_layout(num_sms)["total"])


def _region(flat: torch.Tensor, layout: dict, name: str) -> torch.Tensor:
    offset, nbytes = layout[name]
    return flat[offset : offset + nbytes]


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BalancedGQADecodeRunner:
    """Launch the prepared balanced decode.

    Calling the runner or ``launch()`` writes the caller-owned ``out`` with no
    CUDA allocation and no host synchronization and returns ``out``.  The
    kernel reads ``seq_lens`` and ``block_tables`` on device at every launch,
    so the same runner (or a CUDA Graph capturing it) stays valid when the
    caller writes new lengths or page ids into those buffers.  Prepare a new
    runner when shapes, dtypes or tensor bindings change.
    """

    component: str  # csrc/cake_fmha registry component serving q_len_per_req
    q_len_per_req: int
    num_ctas: int
    out: torch.Tensor
    tile_counters: torch.Tensor  # uint32 view; zero between launches
    queue_counters: torch.Tensor  # uint32[4] view: ticket, done CTAs, two unused
    plan_facts: torch.Tensor  # float32[2] view: (chunk pairs, total items)
    entry: Callable[..., Any]
    arguments: tuple

    def launch(self) -> torch.Tensor:
        # The adapter encodes the three TMA descriptors on the host and
        # launches on the TVM FFI environment stream: the current torch
        # stream, or the capturing stream under CUDA Graph capture.
        with tvm_ffi.use_torch_stream():
            self.entry(*self.arguments)
        return self.out

    __call__ = launch

    def device_plan(self) -> tuple[int, int]:
        """``(chunk_pairs, num_items)`` published by the last launch (syncs)."""
        chunk_pairs, num_items = self.plan_facts.tolist()
        return int(chunk_pairs), int(num_items)


def component_name(q_len_per_req: int) -> str:
    """``csrc/cake_fmha`` component serving ``q_len_per_req`` (``ValueError`` for 2 and above 8)."""
    if not serves_q_len(q_len_per_req):
        raise ValueError(
            "balanced GQA decode serves q_len_per_req 1 (row-tile program) or "
            f"{MTP_MIN_Q_LEN}..{MTP_MAX_Q_LEN} (packed-row MTP programs), got "
            f"{q_len_per_req}"
        )
    return cake_fmha_balanced_component_name(q_len_per_req, "bf16")


def generated_program_available(device: torch.device, q_len_per_req: int = 1) -> bool:
    """True when the balanced component serving ``q_len_per_req`` is registered for ``device``."""
    if _device_target(_device_index(device)) is None or not serves_q_len(q_len_per_req):
        return False
    return component_name(q_len_per_req) in get_cake_fmha_manifest()["components"]


# ---------------------------------------------------------------------------
# Validation and preparation
# ---------------------------------------------------------------------------


def _split_kv_cache(
    kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]], kv_layout: str
) -> tuple[torch.Tensor, torch.Tensor]:
    if kv_layout != "HND":
        raise ValueError(
            "balanced GQA decode reads pages as [num_kv_heads, page_size, head_dim] "
            "(kv_layout='HND')"
        )
    if isinstance(kv_cache, torch.Tensor):
        if kv_cache.ndim == 5 and kv_cache.shape[1] == 2:
            k_cache, v_cache = kv_cache[:, 0], kv_cache[:, 1]
            if not (k_cache.is_contiguous() and v_cache.is_contiguous()):
                raise ValueError(
                    "balanced GQA decode needs K and V pages in separate contiguous "
                    "tensors: pass kv_cache=(k_cache, v_cache), each "
                    f"[num_pages, num_kv_heads, {PAGE_SIZE}, {HEAD_DIM}]"
                )
            return k_cache, v_cache
        raise ValueError(
            "kv_cache must be a (k_cache, v_cache) tuple of "
            f"[num_pages, num_kv_heads, {PAGE_SIZE}, {HEAD_DIM}] tensors"
        )
    if len(kv_cache) != 2:
        raise ValueError("kv_cache tuple must hold exactly (k_cache, v_cache)")
    return kv_cache[0], kv_cache[1]


def validate_balanced_gqa_decode_inputs(
    query: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    *,
    q_len_per_req: int,
    out: Optional[torch.Tensor] = None,
) -> tuple[int, int, int, int]:
    """Shape / dtype validation shared by ``prepare``.

    Returns ``(batch, num_q_heads, num_kv_heads, num_pages)``.  Device placement
    and compute capability are checked separately so this runs on host tensors.
    """
    if query.ndim != 3 or query.shape[-1] != HEAD_DIM or query.dtype != torch.bfloat16:
        raise ValueError(
            f"query must be a bfloat16 [batch * q_len, num_q_heads, {HEAD_DIM}] tensor"
        )
    if (
        block_tables.ndim != 2
        or block_tables.dtype != torch.int32
        or block_tables.shape[1] <= 0
    ):
        raise ValueError("block_tables must be an int32 [batch, max_pages] tensor")
    batch = int(block_tables.shape[0])
    total_q, num_q_heads = int(query.shape[0]), int(query.shape[1])
    if not isinstance(q_len_per_req, int) or q_len_per_req <= 0:
        raise ValueError("q_len_per_req must be a positive integer")
    if not serves_q_len(q_len_per_req):
        component_name(q_len_per_req)  # raises with the served range
    if batch <= 0 or total_q != batch * q_len_per_req:
        raise ValueError(
            "query rows must equal batch * q_len_per_req "
            f"(got {total_q} rows for batch {batch}, q_len_per_req {q_len_per_req})"
        )
    if batch > MAX_REQUESTS:
        raise ValueError(
            f"balanced GQA decode plans at most {MAX_REQUESTS} requests per launch"
        )
    for name, cache in (("k_cache", k_cache), ("v_cache", v_cache)):
        if (
            cache.ndim != 4
            or cache.dtype != torch.bfloat16
            or tuple(cache.shape[2:]) != (PAGE_SIZE, HEAD_DIM)
        ):
            raise ValueError(
                f"{name} must be a bfloat16 [num_pages, num_kv_heads, {PAGE_SIZE}, "
                f"{HEAD_DIM}] tensor"
            )
    if tuple(k_cache.shape) != tuple(v_cache.shape):
        raise ValueError("k_cache and v_cache must have the same shape")
    num_pages, num_kv_heads = int(k_cache.shape[0]), int(k_cache.shape[1])
    if num_kv_heads <= 0 or num_q_heads != GROUP_RATIO * num_kv_heads:
        raise ValueError(
            f"balanced GQA decode serves exactly {GROUP_RATIO} query heads per KV head "
            f"(got {num_q_heads} query heads, {num_kv_heads} KV heads)"
        )
    if seq_lens.shape != (batch,) or seq_lens.dtype != torch.int32:
        raise ValueError("seq_lens must be an int32 [batch] tensor")
    if out is not None and (
        tuple(out.shape) != (total_q, num_q_heads, HEAD_DIM)
        or out.dtype != torch.bfloat16
    ):
        raise ValueError(
            f"out must be a bfloat16 [batch * q_len, num_q_heads, {HEAD_DIM}] tensor"
        )
    return batch, num_q_heads, num_kv_heads, num_pages


def prepare_balanced_batch_decode_with_kv_cache(
    query: torch.Tensor,
    kv_cache: Union[torch.Tensor, tuple[torch.Tensor, torch.Tensor]],
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    sm_scale: Optional[float] = None,
    q_len_per_req: int = 1,
    out: Optional[torch.Tensor] = None,
    kv_layout: str = "HND",
    backend: str = "cake",
) -> BalancedGQADecodeRunner:
    """Validate and bind one balanced BF16 paged GQA decode batch.

    Every allocation happens here (only the optional output); the returned
    runner launches with none.  No host copy of ``seq_lens`` is made: the work
    plan is derived on device.  The caller's block table is read in place at
    every launch (the kernels clamp page indices to each request's last page,
    whatever the table width), so the runner also follows page ids written
    into ``block_tables`` later.
    """
    if backend != "cake":
        raise ValueError("balanced GQA decode supports backend='cake'")
    k_cache, v_cache = _split_kv_cache(kv_cache, kv_layout)
    batch, num_q_heads, num_kv_heads, _ = validate_balanced_gqa_decode_inputs(
        query,
        k_cache,
        v_cache,
        block_tables,
        seq_lens,
        q_len_per_req=q_len_per_req,
        out=out,
    )
    tensors = [query, k_cache, v_cache, block_tables, seq_lens, workspace_buffer]
    if out is not None:
        tensors.append(out)
    device = query.device
    if not all(t.is_cuda and t.device == device for t in tensors):
        raise ValueError("Expected all tensors on one CUDA device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Expected contiguous tensors")
    target = _device_target(_device_index(device))
    if target is None:
        capability = torch.cuda.get_device_capability(device)
        raise ValueError(
            "balanced GQA decode requires compute capability 10.0 or 10.3 "
            f"(got {capability[0]}.{capability[1]})"
        )
    if sm_scale is None:
        sm_scale = HEAD_DIM**-0.5
    num_ctas = num_persistent_ctas(device)
    layout = workspace_layout(num_ctas)
    flat = workspace_buffer.view(-1).view(torch.uint8)
    if flat.numel() < layout["total"]:
        raise ValueError(
            f"workspace_buffer needs {layout['total']} bytes on this device "
            f"({num_ctas} CTAs), got {flat.numel()}"
        )
    if out is None:
        out = torch.empty(
            (batch * q_len_per_req, num_q_heads, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
        )

    packed = uses_packed_mtp(q_len_per_req)
    partials = _region(flat, layout, "partials")
    counters = _region(flat, layout, "counters")
    # The kernel resets its ticket, done and tile counters at the end of every
    # launch; they must start at zero once.  The partial slots need no initial
    # value (each is written before it is read).
    counters.zero_()
    counter_words = counters.view(torch.uint32)
    _, max_split_tiles = workspace_bounds(num_ctas)
    tile_words = max_split_tiles * (
        MTP_COUNTERS_PER_TILE if packed else ROW_COUNTERS_PER_TILE
    )
    queue_word = _align(tile_words * 4, 16) // 4
    tile_counters = counter_words[:tile_words]
    queue_counters = counter_words[queue_word : queue_word + QUEUE_COUNTERS]
    # Plan facts: the reserved statistics slot past the last split item, i.e.
    # the last slot of the adapter's partial region for this program.
    stats_bytes = (MTP_STATS_PER_SLOT if packed else ROW_STATS_PER_SLOT) * 4
    plan_offset = (
        cake_fmha_balanced_workspace_bytes(num_ctas, q_len_per_req) - stats_bytes
    )
    plan_facts = partials[plan_offset : plan_offset + PLAN_FACTS_WORDS * 4].view(
        torch.float32
    )

    component = component_name(q_len_per_req)
    module = load_cake_fmha_decode_balanced_bf16_module(target, q_len_per_req)
    # Positional arguments of ``cake_paged_attention_decode``, the trtllm-style
    # decode entry shared by every Cake FMHA module.  The balanced adapter
    # rejects every feature the route does not serve (sinks, window, LSE, output
    # scales, block sparsity), so those arguments are fixed here.
    arguments = (
        out,
        None,  # out_scale_factor
        query,
        k_cache,
        v_cache,
        partials,  # workspace_buffer: FP32 partials and the plan-facts slot
        counters,  # multi_ctas_kv_counter_buffer: self-resetting counters
        block_tables,
        seq_lens,
        q_len_per_req,  # max_q_len (the module's Q_LEN)
        int(block_tables.shape[1]) * PAGE_SIZE,  # max_kv_len: capacity, no length read
        float(
            sm_scale
        ),  # bmm1_scale; the adapter applies log2(e) for the packed kernels
        1.0,  # bmm2_scale
        -1.0,  # o_sf_scale
        -1,  # o_sf_vec_size
        0,  # o_sf_start_index
        batch,  # batch_size
        -1,  # window_left
        0,  # sparse_mla_top_k
        num_ctas,  # sm_count: one persistent CTA per SM
        False,  # enable_pdl
        partials.numel(),  # workspace_size
        None,  # attention_sinks
        None,  # cum_seq_lens_q
        None,  # key_block_scales
        None,  # value_block_scales
        None,  # skip_softmax_threshold_scale_factor
        True,  # uses_shared_paged_kv_idx
        None,  # lse
        0,  # lse_stride_tokens
        0,  # lse_stride_heads
        False,  # enable_block_sparse_attention
        None,  # sparse_mla_top_k_lens
    )
    return BalancedGQADecodeRunner(
        component,
        q_len_per_req,
        num_ctas,
        out,
        tile_counters,
        queue_counters,
        plan_facts,
        module.cake_paged_attention_decode,
        arguments,
    )
