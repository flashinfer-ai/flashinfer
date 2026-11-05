# SPDX-License-Identifier: Apache-2.0
"""Compact plans, direct tile geometry, and independent native partial workspaces."""

from dataclasses import dataclass
import math

import torch

from ...utils import (
    device_support_pdl,
    get_compute_capability,
    get_device_sm_count,
)
from .jit import get_module
from .geometry import (
    check_head_config,
    causal_mask_words,
    query_group_count,
    plan_capacity,
    plan_geometry,
    workspace_capacity,
)


def _tensor(tensor, name, *, dtype, shape, device, contiguous=True):
    if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}")
    if tensor.device != device or tensor.device.type != "cuda":
        raise ValueError(f"{name} must be on CUDA device {device}")
    if tuple(tensor.shape) != tuple(shape):
        raise ValueError(f"{name} must have shape {tuple(shape)}")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _byte_span(tensor):
    if not tensor.numel():
        return None
    # data_ptr() already includes the storage offset. Bound every addressed
    # element, conservatively including gaps in strided cache views.
    lower = upper = 0
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        offset = (size - 1) * stride
        lower += min(0, offset)
        upper += max(0, offset)
    start = tensor.data_ptr()
    itemsize = tensor.element_size()
    return start + lower * itemsize, start + (upper + 1) * itemsize


def _storage_checks(
    q,
    k,
    v,
    ks,
    vs,
    table,
    scratch,
    output,
    lengths,
    tasks,
    requests,
    spans,
    header,
    causal_mask,
    semaphores,
):
    # Real storage only: Dynamo/FakeTensor must not inspect pointers. Both the
    # raw eager path and registered CUDA implementation call this same check.
    for name, tensor in (
        ("query", q),
        ("k_cache", k),
        ("v_cache", v),
        ("k_sf_cache", ks),
        ("v_sf_cache", vs),
        ("scratch", scratch),
        ("out", output),
    ):
        if tensor.data_ptr() % 16:
            raise ValueError(f"{name} must have 16-byte aligned storage")
    readonly = [
        _byte_span(tensor)
        for tensor in (
            q,
            k,
            v,
            ks,
            vs,
            table,
            lengths,
            tasks,
            requests,
            spans,
            header,
            causal_mask,
        )
    ]
    writable = [
        (name, _byte_span(tensor))
        for name, tensor in (
            ("out", output),
            ("scratch", scratch),
            ("semaphores", semaphores),
        )
    ]
    for index, (name, span) in enumerate(writable):
        if span is None:
            continue
        if any(
            other is not None and span[0] < other[1] and other[0] < span[1]
            for other in readonly + [other for _, other in writable[index + 1 :]]
        ):
            raise ValueError(
                f"{name} must not alias query, cache, metadata or other writable buffers"
            )


def _check_scales(bmm1_scale, bmm2_scale):
    if not all(
        math.isfinite(float(scale))
        and abs(float(scale)) <= torch.finfo(torch.float32).max
        for scale in (bmm1_scale, bmm2_scale)
    ):
        raise ValueError("BMM1 and BMM2 scales must be finite float32 values")


def _host_lengths(lengths, batch):
    if not isinstance(lengths, torch.Tensor) or lengths.dtype != torch.int32:
        raise TypeError("host_seq_lens must be a CPU int32 tensor")
    if lengths.device.type != "cpu" or tuple(lengths.shape) != (batch,):
        raise ValueError(f"host_seq_lens must have CPU shape ({batch},)")
    if not lengths.is_contiguous():
        raise ValueError("host_seq_lens must be contiguous")


def _descriptor_views(storage, batch, ctas):
    capacity = ctas + batch - 1
    tasks_end = 4 * capacity
    requests_end = tasks_end + 2 * batch
    return (
        storage[:tasks_end].view(capacity, 4),
        storage[tasks_end:requests_end].view(batch, 2),
        storage[requests_end:-4].view(ctas, 2),
        storage[-4:],
    )


def _prepare(
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    ctas: int,
    aligned: bool,
    page_size: int,
    head_group_ratio: int,
    q_len_per_req: int,
    query_tile: int,
) -> None:
    module, _ = get_module(
        not aligned,
        lengths.device,
        page_size=page_size,
        head_group_ratio=head_group_ratio,
        multi_query=q_len_per_req > 1,
        query_tile=query_tile,
    )
    with torch.cuda.device(lengths.device):
        module.plan(lengths, tasks, requests, spans, header, ctas, aligned)


def _fake_prepare(
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    ctas: int,
    aligned: bool,
    page_size: int,
    head_group_ratio: int,
    q_len_per_req: int,
    query_tile: int,
) -> None:
    pass


def _compute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    causal_mask: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    ctas: int,
    aligned: bool,
    bmm1_scale: float,
    bmm2_scale: float,
    enable_pdl: bool,
    page_size: int,
    head_group_ratio: int,
    q_len_per_req: int,
    query_tile: int,
) -> None:
    _check_scales(bmm1_scale, bmm2_scale)
    _storage_checks(
        q,
        k,
        v,
        ks,
        vs,
        table,
        scratch,
        output,
        lengths,
        tasks,
        requests,
        spans,
        header,
        causal_mask,
        semaphores,
    )
    # Empty headers are the factory's explicit static direct-plan representation.
    direct = header.numel() == 0
    module, _ = get_module(
        not aligned and not direct,
        q.device,
        direct=direct,
        page_size=page_size,
        head_group_ratio=head_group_ratio,
        multi_query=q_len_per_req > 1,
        query_tile=query_tile,
    )
    with torch.cuda.device(q.device):
        module.run(
            q,
            k,
            v,
            ks,
            vs,
            table,
            lengths,
            tasks,
            requests,
            spans,
            header,
            causal_mask,
            semaphores,
            scratch,
            output,
            ctas,
            aligned,
            bmm1_scale,
            bmm2_scale,
            enable_pdl,
            q_len_per_req,
        )


def _fake_compute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    ks: torch.Tensor,
    vs: torch.Tensor,
    table: torch.Tensor,
    lengths: torch.Tensor,
    tasks: torch.Tensor,
    requests: torch.Tensor,
    spans: torch.Tensor,
    header: torch.Tensor,
    causal_mask: torch.Tensor,
    semaphores: torch.Tensor,
    scratch: torch.Tensor,
    output: torch.Tensor,
    ctas: int,
    aligned: bool,
    bmm1_scale: float,
    bmm2_scale: float,
    enable_pdl: bool,
    page_size: int,
    head_group_ratio: int,
    q_len_per_req: int,
    query_tile: int,
) -> None:
    pass


# Keep this FRAGMENT alive for the lifetime of the module. Shared FlashInfer
# decorators in some checkouts are no-ops, so register this boundary directly.
_library = torch.library.Library("flashinfer", "FRAGMENT")
_library.define(
    "experimental_xqa_bucket_plan(Tensor lengths, Tensor(a!) tasks, "
    "Tensor(b!) requests, Tensor(c!) spans, Tensor(d!) header, int ctas, "
    "bool aligned, int page_size, int head_group_ratio, int q_len_per_req, "
    "int query_tile) -> ()"
)
_library.define(
    "experimental_xqa_bucket_compute(Tensor q, Tensor k, Tensor v, Tensor ks, Tensor vs, "
    "Tensor table, Tensor lengths, Tensor tasks, Tensor requests, Tensor spans, Tensor header, "
    "Tensor causal_mask, Tensor(a!) semaphores, Tensor(b!) scratch, Tensor(c!) output, "
    "int ctas, bool aligned, float bmm1_scale, float bmm2_scale, bool enable_pdl, "
    "int page_size, int head_group_ratio, int q_len_per_req, int query_tile) -> ()"
)
_library.impl("experimental_xqa_bucket_plan", _prepare, "CUDA")
_library.impl("experimental_xqa_bucket_compute", _compute, "CUDA")
_register_fake = (
    torch.library.register_fake
    if hasattr(torch.library, "register_fake")
    else torch.library.impl_abstract
)
_register_fake("flashinfer::experimental_xqa_bucket_plan")(_fake_prepare)
_register_fake("flashinfer::experimental_xqa_bucket_compute")(_fake_compute)
_prepare_op = torch.ops.flashinfer.experimental_xqa_bucket_plan.default
_compute_op = torch.ops.flashinfer.experimental_xqa_bucket_compute.default


@dataclass(frozen=True)
class BucketWorkspace:
    """Mutable native partial/counter storage for ordered compute calls."""

    scratch: torch.Tensor
    semaphores: torch.Tensor


@dataclass(frozen=True)
class BucketPlan:
    """Fixed buffer addresses; refresh device contents once per forward.

    Direct plans use geometry inside compute; prepare() is a no-op and all four
    descriptor tensors are empty. They apply to B=1 or when the static table's
    total tile bound fits the per-head CTA budget. Both mode names select the same
    one-task-per-CTA specialization for direct plans. Otherwise call prepare()
    once per forward after updating lengths, before its compute consumers.
    Compatible layers share the same request ordering and lengths. Replays and
    refreshes must be ordered on the same stream or by explicit CUDA events.
    Separate concurrent computes require independent BucketWorkspace instances.

    Host plans retain caller-owned CPU host_seq_lens; callers keep its values
    identical to lengths without a device readback. Host prepare runs outside
    capture before each compute-only Graph replay. One pinned staging backing is
    reused only after its previous asynchronous H2D finishes. This protects the
    staging lifetime; stream ordering still protects the device descriptors.
    """

    lengths: torch.Tensor
    tasks: torch.Tensor
    requests: torch.Tensor
    spans: torch.Tensor
    header: torch.Tensor
    ctas: int
    table_width: int
    page_size: int
    num_kv_heads: int
    head_group_ratio: int
    q_len_per_req: int
    query_tile: int
    num_groups: int
    causal_mask: torch.Tensor
    grid_y_limit: int
    mode: str
    scratch_bytes: int
    resources: tuple
    enable_pdl: bool
    direct: bool
    planner: str
    host_seq_lens: torch.Tensor | None
    _host_descriptors: torch.Tensor | None
    _device_descriptors: torch.Tensor | None
    _copy_done: torch.cuda.Event | None

    def prepare(self):
        if self.direct:
            return self
        if torch.compiler.is_compiling():
            if self.planner == "host":
                raise RuntimeError("Host prepare must run outside compiled consumers")
            _prepare_op(
                self.lengths,
                self.tasks,
                self.requests,
                self.spans,
                self.header,
                self.ctas,
                self.mode == "aligned",
                self.page_size,
                self.head_group_ratio,
                self.q_len_per_req,
                self.query_tile,
            )
            return self
        with torch.cuda.device(self.lengths.device):
            if self.planner == "host":
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "Host prepare must run before Graph replay, outside capture"
                    )
                _host_lengths(self.host_seq_lens, self.lengths.numel())
                # A single bounded staging buffer; never overwrite an in-flight
                # DMA source. This waits only when the prior copy is unfinished.
                if not self._copy_done.query():
                    self._copy_done.synchronize()
                module, _ = get_module(
                    self.mode == "persistent",
                    self.lengths.device,
                    page_size=self.page_size,
                    head_group_ratio=self.head_group_ratio,
                    multi_query=self.q_len_per_req > 1,
                    query_tile=self.query_tile,
                )
                module.plan_host(
                    self.host_seq_lens,
                    self._host_descriptors,
                    self.ctas,
                    self.table_width,
                    self.mode == "aligned",
                    self.q_len_per_req,
                )
                self._device_descriptors.copy_(
                    self._host_descriptors, non_blocking=True
                )
                self._copy_done.record()
                return self
            _prepare(
                self.lengths,
                self.tasks,
                self.requests,
                self.spans,
                self.header,
                self.ctas,
                self.mode == "aligned",
                self.page_size,
                self.head_group_ratio,
                self.q_len_per_req,
                self.query_tile,
            )
        return self

    def create_workspace(self):
        """Allocate cold, before capture; sequential layers may reuse this workspace."""
        return BucketWorkspace(
            torch.empty(
                self.scratch_bytes, dtype=torch.uint8, device=self.lengths.device
            ),
            torch.zeros(
                self.lengths.numel() * self.num_groups,
                dtype=torch.int32,
                device=self.lengths.device,
            ),
        )


def create_plan(
    seq_lens,
    *,
    page_table_width,
    page_size=32,
    mode="persistent",
    planner="gpu",
    host_seq_lens=None,
    num_kv_heads=1,
    head_group_ratio=6,
    q_len_per_req=1,
    query_tile=16,
):
    check_head_config(num_kv_heads, head_group_ratio)
    groups = query_group_count(
        num_kv_heads, head_group_ratio, q_len_per_req, query_tile
    )
    if mode not in ("persistent", "aligned"):
        raise ValueError("mode must be persistent or aligned")
    if planner not in ("gpu", "host"):
        raise ValueError("planner must be gpu or host")
    if not isinstance(seq_lens, torch.Tensor) or seq_lens.ndim != 1:
        raise ValueError("seq_lens must be a 1D CUDA int32 tensor")
    device = seq_lens.device
    _tensor(
        seq_lens, "seq_lens", dtype=torch.int32, shape=seq_lens.shape, device=device
    )
    if get_compute_capability(device) != (12, 0):
        raise ValueError("XQA Decode buckets currently support SM120")
    batch = seq_lens.numel()
    if planner == "host":
        _host_lengths(host_seq_lens, batch)
    elif host_seq_lens is not None:
        raise ValueError("host_seq_lens is only used with planner='host'")
    ctas, direct = plan_geometry(
        batch,
        page_table_width,
        get_device_sm_count(device),
        page_size,
        num_kv_heads,
        head_group_ratio,
        q_len_per_req,
        query_tile,
    )
    capacity = plan_capacity(batch, ctas, page_table_width, page_size)
    total_capacity = workspace_capacity(
        batch, ctas, page_table_width, page_size, groups
    )
    if batch * num_kv_heads * head_group_ratio * q_len_per_req > (1 << 32) - 1:
        raise ValueError("Native query head indices must fit uint32")
    module, resources = get_module(
        mode == "persistent" and not direct,
        device,
        direct=direct,
        page_size=page_size,
        head_group_ratio=head_group_ratio,
        multi_query=q_len_per_req > 1,
        query_tile=query_tile,
    )
    grid_y_limit = module.grid_y_limit(device.index)
    if groups > grid_y_limit:
        raise ValueError(f"Query/head groups exceed device grid-Y limit {grid_y_limit}")
    scratch_bytes = module.scratch_bytes(total_capacity)
    if q_len_per_req > 1:
        # Cold constant suffix mask, packed exactly like native XQA. No mask
        # generation or length readback is part of prepare/compute/replay.
        causal_mask = torch.tensor(
            causal_mask_words(q_len_per_req), dtype=torch.int32, device="cpu"
        )
        causal_mask = (
            causal_mask.repeat(batch, 1)
            .view(torch.uint16)
            .reshape(batch, q_len_per_req, 2)
            .to(device)
        )
    else:
        causal_mask = torch.empty(0, dtype=torch.uint16, device=device)
    host_descriptors = device_descriptors = copy_done = None
    if planner == "host" and not direct:
        words = 6 * (ctas + batch)
        host_descriptors = torch.empty(
            words, dtype=torch.int32, device="cpu", pin_memory=True
        )
        device_descriptors = torch.empty(words, dtype=torch.int32, device=device)
        tasks, requests, spans, header = _descriptor_views(
            device_descriptors, batch, ctas
        )
        copy_done = torch.cuda.Event()
    else:
        tasks = torch.empty(
            (0 if direct else capacity, 4), dtype=torch.int32, device=device
        )
        requests = torch.empty(
            (0 if direct else batch, 2), dtype=torch.int32, device=device
        )
        spans = torch.empty(
            (0 if direct else ctas, 2), dtype=torch.int32, device=device
        )
        header = torch.empty(0 if direct else 4, dtype=torch.int32, device=device)
    plan = BucketPlan(
        seq_lens,
        tasks,
        requests,
        spans,
        header,
        ctas,
        page_table_width,
        page_size,
        num_kv_heads,
        head_group_ratio,
        q_len_per_req,
        query_tile,
        groups,
        causal_mask,
        grid_y_limit,
        mode,
        scratch_bytes,
        resources,
        device_support_pdl(device),
        direct,
        planner,
        host_seq_lens,
        host_descriptors,
        device_descriptors,
        copy_done,
    )
    # The initial generation is usable immediately. Later forwards refresh it explicitly.
    return plan.prepare()


def run(
    query,
    k_cache,
    v_cache,
    k_sf_cache,
    v_sf_cache,
    page_table,
    plan,
    workspace,
    *,
    bmm1_scale,
    bmm2_scale,
    out=None,
    enable_pdl=None,
):
    if not isinstance(plan, BucketPlan) or not isinstance(workspace, BucketWorkspace):
        raise TypeError("Expected a bucket plan and its caller-owned workspace")
    device, batch = plan.lengths.device, plan.lengths.numel()
    query_heads = plan.num_kv_heads * plan.head_group_ratio
    shape = (
        (batch, query_heads, 256)
        if plan.q_len_per_req == 1
        else (batch, plan.q_len_per_req, query_heads, 256)
    )
    _tensor(query, "query", dtype=torch.bfloat16, shape=shape, device=device)
    _tensor(
        page_table,
        "page_table",
        dtype=torch.int32,
        shape=(batch, plan.table_width),
        device=device,
    )
    if k_cache.ndim != 4:
        raise ValueError("Expected packed NHD NVFP4 cache")
    pages = k_cache.shape[0]
    for name, value, dtype, width, unit in (
        ("k_cache", k_cache, torch.uint8, 128, 128),
        ("v_cache", v_cache, torch.uint8, 128, 128),
        ("k_sf_cache", k_sf_cache, torch.float8_e4m3fn, 16, 16),
        ("v_sf_cache", v_sf_cache, torch.float8_e4m3fn, 16, 16),
    ):
        _tensor(
            value,
            name,
            dtype=dtype,
            shape=(pages, plan.page_size, plan.num_kv_heads, width),
            device=device,
            contiguous=False,
        )
        if value.stride(-1) != 1 or any(value.stride(i) % unit for i in range(3)):
            raise ValueError(f"{name} must preserve native packed-head byte strides")
    if (
        k_cache.stride() != v_cache.stride()
        or k_sf_cache.stride() != v_sf_cache.stride()
    ):
        raise ValueError("K/V and their block scales must use matching strides")
    _tensor(
        workspace.scratch,
        "scratch",
        dtype=torch.uint8,
        shape=(plan.scratch_bytes,),
        device=device,
    )
    _tensor(
        workspace.semaphores,
        "semaphores",
        dtype=torch.int32,
        shape=(batch * plan.num_groups,),
        device=device,
    )
    _check_scales(bmm1_scale, bmm2_scale)
    if out is None:
        out = torch.empty_like(query)
    _tensor(out, "out", dtype=torch.bfloat16, shape=shape, device=device)
    if enable_pdl is None:
        enable_pdl = plan.enable_pdl
    if type(enable_pdl) is not bool:
        raise TypeError("enable_pdl must be a boolean or None")
    compute = _compute_op if torch.compiler.is_compiling() else _compute
    compute(
        query,
        k_cache,
        v_cache,
        k_sf_cache,
        v_sf_cache,
        page_table,
        plan.lengths,
        plan.tasks,
        plan.requests,
        plan.spans,
        plan.header,
        plan.causal_mask,
        workspace.semaphores,
        workspace.scratch,
        out,
        plan.ctas,
        plan.mode == "aligned",
        float(bmm1_scale),
        float(bmm2_scale),
        enable_pdl,
        plan.page_size,
        plan.head_group_ratio,
        plan.q_len_per_req,
        plan.query_tile,
    )
    return out
