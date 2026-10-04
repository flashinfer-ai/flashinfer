"""Prepared paged FP8 MQA lightning-indexer logits on SM100a/SM103a.

Decode-side counterpart of :mod:`.dense_mqa` with DeepGEMM's paged contract:
``q [B, next_n, H, 128]`` E4M3, the fused KV cache ``[pages, block_kv, 1, 132]``
(``block_kv`` FP8 rows then ``block_kv`` FP32 scales per page, read in place),
FP32 ``weights [B * next_n, H]``, two-dimensional int32 ``context_lens
[B, next_n]`` (the schedule is sized from each request's last token, every
token masks with its own length), an int32 ``block_table [B, S]`` with unit
column stride (any row stride) and the schedule metadata of the catalog's
metadata program. Two programs per route: the one-warp metadata program
(``[num_sms + 1, 2]`` per-CTA walk bounds) and the persistent logits program,
both taking the CTA budget as the compile-line definition ``SM_COUNT``.

Routes are selected from host-known scalars only: head count, page size and
``next_n`` (``paged_route_name``). Output: FP32 ``[B * next_n,
paged_logits_stride(max_context_len)]`` with DeepGEMM's ``clean_logits=False``
semantics (inside every KV split a row touches, positions at or past the row's
length are ``-inf``; cells beyond the row's last split are untouched);
consume ``plan.logical_output`` (``[:, :max_context_len]``).
"""

from __future__ import annotations

from .dense_mqa import (
    HEAD_DIM,
    _catalog,
    _resolve_device,
    _submission,
)

FUSED_ROW_BYTES = HEAD_DIM + 4  # one FP8 row + its FP32 scale
LOGITS_DTYPES = ("float32",)


def paged_policy():
    """Catalog policy of the paged family (block sizes, SPLIT_KV, atom rule, batch ceiling)."""
    return _catalog()["policy"]["paged"]


def paged_block_sizes():
    """Exported page sizes (``block_kv``)."""
    return tuple(int(v) for v in paged_policy()["block_kv"])


def paged_heads():
    """Exported paged head counts (``policy["paged"]["heads"]``): independent of the dense ``heads()`` --
    the 64-head paged programs ship before the 64-head dense programs."""
    return tuple(int(v) for v in paged_policy()["heads"])


def paged_next_n_atoms(next_n):
    """``num_next_n_atoms`` of the metadata call for ``next_n`` (catalog rule; 1 for every exported
    ``next_n`` of the whole-request programs, which iterate the atoms in-kernel). Raises ``ValueError``
    for a ``next_n`` without an exported program (the engine uses 1, 2 and 4)."""
    rule = paged_policy()["next_n_atoms"]
    key = str(int(next_n))
    if key not in rule:
        raise ValueError(
            f"next_n = {next_n} has no exported paged program; exported: {sorted(rule)}"
        )
    return int(rule[key])


def paged_route_name(num_heads, block_kv, next_n, logits_dtype="float32"):
    """Logical paged route of a problem (host-known scalars only)."""
    if logits_dtype not in LOGITS_DTYPES:
        raise ValueError(f"logits_dtype must be one of {LOGITS_DTYPES}")
    suffix = "" if logits_dtype == "float32" else f":{logits_dtype}"
    return f"paged:fp8:h{int(num_heads)}:p{int(block_kv)}:n{int(next_n)}{suffix}"


def paged_route_available(num_heads, block_kv, next_n, logits_dtype="float32"):
    """True when the shipped catalog carries the paged route (host-only)."""
    try:
        if num_heads not in paged_heads() or block_kv not in paged_block_sizes():
            return False
        paged_next_n_atoms(next_n)
        route = paged_route_name(num_heads, block_kv, next_n, logits_dtype)
    except ValueError:
        return False
    return route in _catalog()["paged_routes"]


def paged_logits_stride(max_context_len):
    """FP32 row stride of the paged logits: align(align(max_context_len, 256), 256) (1024-byte rows)."""
    split_kv = int(paged_policy()["split_kv"])
    aligned = (max_context_len + split_kv - 1) // split_kv * split_kv
    return (aligned + 255) // 256 * 256


def metadata_shape(num_sms):
    """Shape of the schedule metadata: ``(num_sms + 1, 2)`` int32."""
    return (int(num_sms) + 1, 2)


def _check_context_lens(context_lens):
    import torch

    if (
        context_lens.ndim != 2
        or context_lens.dtype != torch.int32
        or not context_lens.is_contiguous()
        or context_lens.device.type != "cuda"
    ):
        raise ValueError("context_lens must be contiguous int32 [B, next_n] on CUDA")
    batch, next_n = (int(v) for v in context_lens.shape)
    if batch < 1 or next_n < 1:
        raise ValueError("context_lens must have at least one request and one token")
    if batch > int(paged_policy()["max_batch"]):
        raise ValueError(
            f"batch {batch} exceeds the metadata program's ceiling {paged_policy()['max_batch']}"
        )
    return batch, next_n


def _metadata_route(block_kv, next_n):
    """Any paged route with this (block_kv, next_n): the metadata program is head-count independent."""
    for num_heads in paged_heads():
        route = paged_route_name(num_heads, block_kv, next_n)
        if route in _catalog()["paged_routes"]:
            return _catalog()["paged_routes"][route]
    raise ValueError(
        f"no exported paged route for block_kv = {block_kv}, next_n = {next_n}"
    )


def metadata_bindings(context_lens, schedule_meta, *, block_kv, num_sms):
    """Argument plan of the metadata program (stage ``metadata`` of every paged route)."""
    batch, next_n = _check_context_lens(context_lens)
    return dict(
        context_lens=context_lens.view(-1),
        schedule_meta=schedule_meta.view(-1),
        batch_size=batch,
        next_n=next_n,
        num_next_n_atoms=paged_next_n_atoms(next_n),
        split_kv=int(paged_policy()["split_kv"]),
        num_sms=int(num_sms),
        is_context_lens_2d=1,
        grid_x=1,
        grid_y=1,
        grid_z=1,
    )


def logits_bindings(
    q, kv_cache, weights, context_lens, block_table, schedule_meta, output, *, num_sms
):
    """Argument plan of the logits program (stage ``logits`` of every paged route).

    The names are the launcher contract of the paged programs: Q rows
    ``[B * next_n * H, 128]`` u8, the fused cache as its byte rows ``[pages,
    block_kv * 132]`` with the FP32 scale tail aliased as ``[pages,
    block_kv * 33]`` f32, weights, logits, flattened context lengths, the
    block table as a flat view with its row stride, the metadata, and the
    scalar geometry.
    """
    import torch

    batch, next_n = (int(v) for v in context_lens.shape)
    pages, block_kv = int(kv_cache.shape[0]), int(kv_cache.shape[1])
    fused = kv_cache.view(torch.uint8).reshape(pages, block_kv * FUSED_ROW_BYTES)
    return dict(
        Q=q.view(torch.uint8).reshape(-1, HEAD_DIM),
        KV=fused,
        KV_scales=fused.view(torch.float32),
        Weights=weights,
        Logits=output,
        context_lens=context_lens.view(-1),
        block_table=block_table.as_strided(
            ((batch - 1) * block_table.stride(0) + int(block_table.shape[1]),), (1,)
        ),
        schedule_meta=schedule_meta.view(-1),
        batch_size=batch,
        next_n=next_n,
        seq_len=batch * next_n,
        max_context_len=int(output.shape[1]),
        stride_logits=int(output.stride(0)),
        block_table_stride=int(block_table.stride(0)),
        grid_x=int(num_sms),
        grid_y=1,
        grid_z=1,
    )


class PagedMqaPlan:
    """Bind the paged operands for repeated metadata-to-logits submissions.

    q E4M3 [B, next_n, H, 128] contiguous; kv_cache uint8 [pages, block_kv, 1,
    132] contiguous (block_kv FP8 rows then block_kv FP32 scales per page);
    weights FP32 [B * next_n, H] contiguous; context_lens int32 [B, next_n]
    contiguous; block_table int32 [B, S] with stride(1) == 1 (any row stride,
    e.g. an engine's ``[::next_n]`` view); 1 <= max_context_len <= S *
    block_kv. The route is (H, block_kv, next_n) and must be shipped
    (``paged_route_available``).

    ``schedule_meta`` (int32 [num_sms + 1, 2]) and ``output`` (FP32 [B *
    next_n, paged_logits_stride(max_context_len)]) are allocated when not
    supplied and stay bound to the plan; contents of every bound tensor may
    change between runs, including CUDA Graph replay. ``run()`` submits the
    metadata program then the logits program on the current stream and
    returns ``logical_output`` (``output[:, :max_context_len]``).
    """

    def __init__(
        self,
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        *,
        schedule_meta=None,
        output=None,
        sm_count=None,
        logits_dtype="float32",
    ):
        import torch

        arch, num_sms = _resolve_device(q, sm_count)
        batch, next_n = _check_context_lens(context_lens)
        if (
            q.ndim != 4
            or tuple(q.shape[:2]) != (batch, next_n)
            or int(q.shape[3]) != HEAD_DIM
        ):
            raise ValueError(
                "q must be [B, next_n, H, 128] matching context_lens [B, next_n]"
            )
        num_heads = int(q.shape[2])
        if num_heads not in paged_heads():
            raise ValueError(
                f"q has {num_heads} heads; exported paged head counts: {paged_heads()}"
            )
        if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
            raise ValueError("q must be contiguous E4M3")
        if kv_cache.ndim != 4 or kv_cache.dtype != torch.uint8:
            raise ValueError("kv_cache must be uint8 [pages, block_kv, 1, 132]")
        pages, block_kv, kv_heads, row_bytes = (int(v) for v in kv_cache.shape)
        if kv_heads != 1 or row_bytes != FUSED_ROW_BYTES:
            raise ValueError(
                "kv_cache must be [pages, block_kv, 1, 132] (fused FP8 rows + FP32 scales)"
            )
        if block_kv not in paged_block_sizes():
            raise ValueError(
                f"block_kv {block_kv} is not exported; exported page sizes: {paged_block_sizes()}"
            )
        if (
            int(kv_cache.stride(3)) != 1
            or int(kv_cache.stride(2)) != FUSED_ROW_BYTES
            or int(kv_cache.stride(1)) != FUSED_ROW_BYTES
            or int(kv_cache.stride(0)) != block_kv * FUSED_ROW_BYTES
        ):
            raise ValueError(
                "kv_cache pages must be contiguous [block_kv, 132] byte blocks"
            )
        if (
            weights.dtype != torch.float32
            or tuple(weights.shape) != (batch * next_n, num_heads)
            or not weights.is_contiguous()
        ):
            raise ValueError("weights must be contiguous FP32 [B * next_n, H]")
        if (
            block_table.ndim != 2
            or block_table.dtype != torch.int32
            or int(block_table.shape[0]) != batch
            or int(block_table.stride(1)) != 1
        ):
            raise ValueError("block_table must be int32 [B, S] with unit column stride")
        max_context_len = int(max_context_len)
        if (
            max_context_len < 1
            or max_context_len > int(block_table.shape[1]) * block_kv
        ):
            raise ValueError("max_context_len must be in 1..S * block_kv")
        if not paged_route_available(num_heads, block_kv, next_n, logits_dtype):
            raise ValueError(
                f"no exported paged MQA route for {num_heads} heads, page {block_kv}, next_n {next_n}, {logits_dtype}"
            )
        tensors = [q, kv_cache, weights, context_lens, block_table]
        if any(t.device != q.device for t in tensors):
            raise ValueError("all inputs must be on one CUDA device")
        self.arch, self.num_sms, self.num_heads = arch, num_sms, num_heads
        self.block_kv, self.next_n, self.batch = block_kv, next_n, batch
        self.max_context_len = max_context_len
        self.route_name = paged_route_name(num_heads, block_kv, next_n, logits_dtype)
        self.route = _catalog()["paged_routes"][self.route_name]
        stride = paged_logits_stride(max_context_len)
        if output is None:
            output = torch.empty(
                (batch * next_n, stride), dtype=torch.float32, device=q.device
            )
        if (
            output.dtype != torch.float32
            or tuple(output.shape) != (batch * next_n, stride)
            or output.device != q.device
            or not output.is_contiguous()
        ):
            raise ValueError(
                "output must be contiguous FP32 [B * next_n, paged_logits_stride(max_context_len)]"
            )
        if schedule_meta is None:
            schedule_meta = torch.empty(
                metadata_shape(num_sms), dtype=torch.int32, device=q.device
            )
        if (
            schedule_meta.dtype != torch.int32
            or tuple(schedule_meta.shape) != metadata_shape(num_sms)
            or schedule_meta.device != q.device
            or not schedule_meta.is_contiguous()
        ):
            raise ValueError(
                f"schedule_meta must be contiguous int32 {metadata_shape(num_sms)} on the device"
            )
        bindings = {
            "metadata": metadata_bindings(
                context_lens, schedule_meta, block_kv=block_kv, num_sms=num_sms
            ),
            "logits": logits_bindings(
                q,
                kv_cache,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                output,
                num_sms=num_sms,
            ),
        }
        self._submissions, self._programs = [], []
        self.program_names = [program for _stage, program in self.route["stages"]]
        for stage_name, program in self.route["stages"]:
            submit, loaded = _submission(
                arch, program, bindings, num_sms, stage=stage_name
            )
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.output, self.schedule_meta = output, schedule_meta
        self.logical_output = output[:, :max_context_len]
        self._retained = (*tensors, output, schedule_meta)

    @property
    def launch_count(self):
        return len(self._submissions)

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output

    def run_logits_only(self):
        """Submit the logits program alone (``schedule_meta`` already holds this plan's metadata)."""
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            entry, args = self._submissions[-1]
            entry(*args)
        return self.logical_output


def get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms, indices=None, *, out=None
):
    """Schedule metadata with DeepGEMM's ``get_paged_mqa_logits_metadata`` signature.

    ``context_lens`` int32 ``[B, next_n]``; ``block_kv`` an exported page size;
    ``num_sms`` the CTA budget the logits call will use; ``indices`` (DeepGEMM's
    variable-length request selection) must be ``None``. Returns (or fills
    ``out``) int32 ``[num_sms + 1, 2]`` produced by the catalog's metadata
    program. Not interchangeable with DeepGEMM's buffer: the atom geometry is
    the catalog's.
    """
    import torch
    import tvm_ffi

    if indices is not None:
        raise ValueError("indices (variable-length request selection) is not supported")
    arch, num_sms = _resolve_device(context_lens, num_sms)
    _check_context_lens(context_lens)
    if block_kv not in paged_block_sizes():
        raise ValueError(
            f"block_kv {block_kv} is not exported; exported page sizes: {paged_block_sizes()}"
        )
    route = _metadata_route(block_kv, int(context_lens.shape[1]))
    if out is None:
        out = torch.empty(
            metadata_shape(num_sms), dtype=torch.int32, device=context_lens.device
        )
    if (
        out.dtype != torch.int32
        or tuple(out.shape) != metadata_shape(num_sms)
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous int32 {metadata_shape(num_sms)}")
    bindings = {
        "metadata": metadata_bindings(
            context_lens, out, block_kv=block_kv, num_sms=num_sms
        )
    }
    program = dict(route["stages"])["metadata"]
    (entry, args), _loaded = _submission(
        arch, program, bindings, num_sms, stage="metadata"
    )
    with tvm_ffi.use_torch_stream():
        entry(*args)
    return out


def fp8_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    schedule_meta,
    max_context_len,
    clean_logits=False,
    indices=None,
):
    """One-shot paged logits with DeepGEMM's ``fp8_paged_mqa_logits`` signature.

    ``schedule_meta`` must come from :func:`get_paged_mqa_logits_metadata` for
    the same ``context_lens`` and CTA budget (its first dimension fixes the
    budget). ``clean_logits=True`` is rejected for two-dimensional context
    lengths exactly as DeepGEMM does; ``indices`` must be ``None``. Returns the
    FP32 ``[B * next_n, max_context_len]`` view of a freshly allocated
    row-padded buffer.
    """
    if clean_logits:
        raise ValueError(
            "clean_logits=True is not supported with [B, next_n] context_lens (DeepGEMM semantics)"
        )
    if indices is not None:
        raise ValueError("indices (variable-length request selection) is not supported")
    if schedule_meta.ndim != 2 or int(schedule_meta.shape[1]) != 2:
        raise ValueError(
            "schedule_meta must be int32 [num_sms + 1, 2] from get_paged_mqa_logits_metadata"
        )
    num_sms = int(schedule_meta.shape[0]) - 1
    plan = PagedMqaPlan(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        schedule_meta=schedule_meta,
        sm_count=num_sms,
    )
    return plan.run_logits_only()
