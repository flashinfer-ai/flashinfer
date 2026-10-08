"""Prepared paged FP8 MQA lightning-indexer logits on SM100a/SM103a.

Decode-side counterpart of :mod:`.dense_mqa` with DeepGEMM's paged contract:
``q [B, next_n, H, 128]`` E4M3, the fused KV cache ``[pages, block_kv, 1, 132]``
(``block_kv`` FP8 rows then ``block_kv`` FP32 scales per page, read in place),
FP32 ``weights [B * next_n, H]``, two-dimensional int32 ``context_lens
[B, next_n]`` (the schedule is sized from each request's last token, every
token masks with its own length) and an int32 ``block_table [B, S]`` with unit
column stride (any row stride). One persistent logits program per route: it
takes the CTA budget ``num_sms`` as a kernel argument and derives its
(request, KV split) walk in-kernel, so a call is ONE launch.
``get_paged_mqa_logits_metadata`` exists for DeepGEMM call-signature parity
only: it launches nothing and returns a zero ``[num_sms + 1, 2]`` placeholder
that no exported program reads (``schedule_meta`` is accepted and ignored).

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
    """Atom rule of the in-kernel schedule for ``next_n`` (catalog rule; 1 for every exported ``next_n``:
    the whole-request programs iterate the atoms in-kernel). Raises ``ValueError`` for a ``next_n``
    without an exported program (the engine uses 1, 2 and 4)."""
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


def paged_admission_rules(arch, route_name):
    """Admission rules of ``route_name`` on ``arch`` (``policy["paged"]["admission"]``): a list of
    ``(max_batch, max_context_len)`` pairs (``None`` = unbounded), or ``None`` when the route is unbounded on
    that architecture. Host-only."""
    rules = paged_policy().get("admission", {}).get(str(arch), {}).get(route_name)
    if rules is None:
        return None
    return [
        (
            None if max_batch is None else int(max_batch),
            None if max_ctx is None else int(max_ctx),
        )
        for max_batch, max_ctx in rules
    ]


def paged_route_admitted(arch, route_name, batch, max_context_len):
    """True when the catalog admits ``route_name`` on ``arch`` for a call with ``batch`` requests
    (``context_lens.shape[0]``) and ``max_context_len``: some rule has ``batch <= max_batch`` and
    ``max_context_len <= max_context_len`` (``None`` = unbounded); a route without rules is admitted."""
    rules = paged_admission_rules(arch, route_name)
    if rules is None:
        return True
    batch, max_context_len = int(batch), int(max_context_len)
    return any(
        (max_batch is None or batch <= max_batch)
        and (max_ctx is None or max_context_len <= max_ctx)
        for max_batch, max_ctx in rules
    )


def paged_route_available(
    num_heads,
    block_kv,
    next_n,
    logits_dtype="float32",
    *,
    arch=None,
    batch=None,
    max_context_len=None,
):
    """True when the shipped catalog carries the paged route (host-only). With ``arch`` the per-architecture
    admission rules are applied to the call's ``batch`` and ``max_context_len`` as well (both required then):
    False outside the admitted regions, where the caller keeps its stock path."""
    try:
        if num_heads not in paged_heads() or block_kv not in paged_block_sizes():
            return False
        paged_next_n_atoms(next_n)
        route = paged_route_name(num_heads, block_kv, next_n, logits_dtype)
    except ValueError:
        return False
    if route not in _catalog()["paged_routes"]:
        return False
    if arch is None:
        return True
    if batch is None or max_context_len is None:
        raise ValueError(
            "paged_route_available(arch=...) requires batch and max_context_len"
        )
    return paged_route_admitted(arch, route, batch, max_context_len)


def paged_logits_stride(max_context_len):
    """FP32 row stride of the paged logits: align(align(max_context_len, 256), 256) (1024-byte rows)."""
    split_kv = int(paged_policy()["split_kv"])
    aligned = (max_context_len + split_kv - 1) // split_kv * split_kv
    return (aligned + 255) // 256 * 256


def metadata_shape(num_sms):
    """Shape of the DeepGEMM-signature schedule placeholder: ``(num_sms + 1, 2)`` int32."""
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
            f"batch {batch} exceeds the paged programs' batch ceiling {paged_policy()['max_batch']}"
        )
    return batch, next_n


def _route_exists(block_kv, next_n):
    """True when some exported paged route has this (block_kv, next_n) (head-count independent)."""
    return any(
        paged_route_name(num_heads, block_kv, next_n) in _catalog()["paged_routes"]
        for num_heads in paged_heads()
    )


def logits_bindings(
    q, kv_cache, weights, context_lens, block_table, output, *, num_sms
):
    """Argument plan of the logits program (the single stage of every paged route).

    The names are the launcher contract of the paged programs: Q rows
    ``[B * next_n * H, 128]`` u8, the fused cache as its byte rows ``[pages,
    block_kv * 132]`` with the FP32 scale tail aliased as ``[pages,
    block_kv * 33]`` f32, weights, logits, flattened context lengths, the
    block table as a flat view with its row stride, the scalar geometry, and
    the CTA budget ``num_sms`` (the logits program partitions the (request, KV
    split) walk over this many CTAs in-kernel, so it is both the launch grid
    and a kernel argument). No schedule buffer: the walk is derived in-kernel.
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
        batch_size=batch,
        next_n=next_n,
        seq_len=batch * next_n,
        max_context_len=int(output.shape[1]),
        stride_logits=int(output.stride(0)),
        block_table_stride=int(block_table.stride(0)),
        num_sms=int(num_sms),
        grid_x=int(num_sms),
        grid_y=1,
        grid_z=1,
    )


class PagedMqaPlan:
    """Bind the paged operands for repeated single-launch logits submissions.

    q E4M3 [B, next_n, H, 128] contiguous; kv_cache uint8 [pages, block_kv, 1,
    132] contiguous (block_kv FP8 rows then block_kv FP32 scales per page);
    weights FP32 [B * next_n, H] contiguous; context_lens int32 [B, next_n]
    contiguous; block_table int32 [B, S] with stride(1) == 1 (any row stride,
    e.g. an engine's ``[::next_n]`` view); 1 <= max_context_len <= S *
    block_kv. The route is (H, block_kv, next_n) and must be shipped
    (``paged_route_available``).

    ``output`` (FP32 [B * next_n, paged_logits_stride(max_context_len)]) is
    allocated when not supplied and stays bound to the plan; contents of every
    bound tensor may change between runs, including CUDA Graph replay.
    ``run()`` submits the logits program (one launch) on the current stream
    and returns ``logical_output`` (``output[:, :max_context_len]``).
    ``schedule_meta`` is accepted for DeepGEMM signature parity and ignored
    (kept as ``plan.schedule_meta``; no program reads it).

    ``enforce_admission`` (default True) applies the catalog's per-architecture
    admission rules (``paged_route_admitted``) to the call's batch and
    ``max_context_len`` and raises ``ValueError`` outside them; the export
    protocol passes False to validate every exported shape.
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
        enforce_admission=True,
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
        route_name = paged_route_name(num_heads, block_kv, next_n, logits_dtype)
        if enforce_admission and not paged_route_admitted(
            arch, route_name, batch, max_context_len
        ):
            raise ValueError(
                f"paged MQA route {route_name} is not admitted on {arch} for batch {batch}, max_context_len "
                f"{max_context_len} (admission rules (max_batch, max_context_len): "
                f"{paged_admission_rules(arch, route_name)}); keep the stock path for this call"
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
        bindings = {
            "logits": logits_bindings(
                q, kv_cache, weights, context_lens, block_table, output, num_sms=num_sms
            ),
        }
        stages = list(self.route["stages"])
        if [stage for stage, _program in stages] != ["logits"]:
            raise ValueError(
                f"paged route {self.route_name} must be a single logits stage, catalog has {stages}"
            )
        self._submissions, self._programs = [], []
        self.program_names = [program for _stage, program in stages]
        for stage_name, program in stages:
            submit, loaded = _submission(
                arch, program, bindings, num_sms, stage=stage_name
            )
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.output, self.schedule_meta = output, schedule_meta
        self.logical_output = output[:, :max_context_len]
        self._retained = (*tensors, output)

    @property
    def launch_count(self):
        return len(self._submissions)

    def run(self):
        """Submit the logits program (one launch) on the current PyTorch stream."""
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output


def get_paged_mqa_logits_metadata(
    context_lens, block_kv, num_sms, indices=None, *, out=None
):
    """DeepGEMM's ``get_paged_mqa_logits_metadata`` signature, as a placeholder.

    The exported paged logits programs derive their (request, KV split) walk
    in-kernel from ``context_lens`` and the CTA budget, so there is no schedule
    buffer to produce: this entry validates the call like DeepGEMM's
    (``context_lens`` int32 ``[B, next_n]``, ``block_kv`` an exported page
    size with an exported ``next_n``, ``indices`` ``None``) and returns -- or
    zero-fills ``out`` -- an int32 ``[num_sms + 1, 2]`` placeholder WITHOUT
    launching a kernel. Pass it to :func:`fp8_paged_mqa_logits` for signature
    parity; no program reads it. Not interchangeable with DeepGEMM's buffer.
    """
    import torch

    if indices is not None:
        raise ValueError("indices (variable-length request selection) is not supported")
    _arch, num_sms = _resolve_device(context_lens, num_sms)
    _batch, next_n = _check_context_lens(context_lens)
    if block_kv not in paged_block_sizes():
        raise ValueError(
            f"block_kv {block_kv} is not exported; exported page sizes: {paged_block_sizes()}"
        )
    if not _route_exists(block_kv, next_n):
        raise ValueError(
            f"no exported paged route for block_kv = {block_kv}, next_n = {next_n}"
        )
    if out is None:
        return torch.zeros(
            metadata_shape(num_sms), dtype=torch.int32, device=context_lens.device
        )
    if (
        out.dtype != torch.int32
        or tuple(out.shape) != metadata_shape(num_sms)
        or not out.is_contiguous()
    ):
        raise ValueError(f"out must be contiguous int32 {metadata_shape(num_sms)}")
    return out.zero_()


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
    """One-shot paged logits with DeepGEMM's ``fp8_paged_mqa_logits`` signature: ONE launch.

    ``schedule_meta`` is accepted for signature parity: an int32 ``[num_sms +
    1, 2]`` buffer (from :func:`get_paged_mqa_logits_metadata`) fixes the CTA
    budget through its first dimension and is otherwise ignored; ``None``
    selects the device's SM count. ``clean_logits=True`` is rejected for
    two-dimensional context lengths exactly as DeepGEMM does; ``indices`` must
    be ``None``. Returns the FP32 ``[B * next_n, max_context_len]`` view of a
    freshly allocated row-padded buffer.
    """
    if clean_logits:
        raise ValueError(
            "clean_logits=True is not supported with [B, next_n] context_lens (DeepGEMM semantics)"
        )
    if indices is not None:
        raise ValueError("indices (variable-length request selection) is not supported")
    num_sms = None
    if schedule_meta is not None:
        if (
            schedule_meta.ndim != 2
            or int(schedule_meta.shape[1]) != 2
            or int(schedule_meta.shape[0]) < 2
        ):
            raise ValueError(
                "schedule_meta must be int32 [num_sms + 1, 2] (get_paged_mqa_logits_metadata) or None"
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
    return plan.run()
