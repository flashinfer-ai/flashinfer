"""Prepared paged FP8 MQA lightning-indexer logits on SM120a (GB202).

DeepSeek-V3.2 / V4.1-Flash decode-side "lightning indexer" over a block-table KV
cache, in DeepGEMM's paged contract: ``q [B, next_n, H, 128]`` E4M3, the fused
uint8 KV cache (``page_kv`` FP8 rows of 128 bytes then ``page_kv`` FP32 scales
per page, read in place), FP32 ``weights [B * next_n, H]``, two-dimensional
int32 ``context_lens [B, next_n]`` (the schedule is sized from each request's
last token, every token masks with its own length) and an int32 ``block_table
[B, S]`` with unit column stride.

A call is a real TWO-kernel sequence: the single-warp scheduler writes
``schedule_meta [num_sms + 1, 2]`` and the persistent logits program walks the
half-open ``(q_atom, kv_split)`` range it finds there.  Both kernels arrive as
ONE generated prepared-sequence binding, so ``plan.run()`` is a single FFI
submission with the per-stage programmatic-dependent-launch attributes intact.
``get_paged_mqa_logits_metadata`` launches the scheduler alone through its own
standalone program, so it returns a real schedule (unlike the SM100 dense
family's placeholder).

Routes are selected from host-known scalars only: head count, page size and
``next_n``.  Output: FP32 ``[B * next_n, paged_logits_stride(max_context_len)]``
with DeepGEMM's ``clean_logits=False`` semantics -- positions at or past a
token's own context length carry unspecified values and the consumer masks by
length.  Consume ``plan.logical_output`` (``[:, :max_context_len]``).
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

CATALOG_SCHEMA = "sm120_paged_mqa.v1"
CATALOG_FILE = "sm120_paged_mqa_catalog.json"
HEAD_DIM = 128
FUSED_ROW_BYTES = HEAD_DIM + 4  # one FP8 row plus its FP32 scale
LOGITS_DTYPES = ("float32",)

# Exported compute capabilities.  sm_121a (GB10) has the same instruction
# surface but a separate cubin; it joins when a device is measured.
_ARCHES = {(12, 0): "sm_120a"}
SUPPORTED_CAPABILITIES = tuple(sorted(_ARCHES))
# FlashInfer's CompilationContext normalises capability 12.0 to the ``f``
# target on CUDA >= 12.9 and to ``a`` on 12.8, so a capability counts as a
# build target under either suffix.  The module itself is always compiled with
# the exact ``sm120a`` flag set (an exact-architecture payload).
_TARGET_SUFFIXES = {(12, 0): ("0a", "0f")}


@functools.cache
def _catalog():
    catalog = json.loads(Path(__file__).with_name(CATALOG_FILE).read_text())
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(
            f"SM120 paged MQA catalog schema {catalog.get('schema')!r} is not {CATALOG_SCHEMA!r}; regenerate the export"
        )
    return catalog


def policy():
    """Kernel-fact policy of the shipped family (head counts, pages, schedule)."""
    return _catalog()["policy"]


def exported_heads():
    return tuple(int(v) for v in policy()["heads"])


def exported_page_sizes():
    return tuple(int(v) for v in policy()["page_kv"])


def exported_next_n():
    return tuple(int(v) for v in policy()["next_n"])


def split_kv():
    """KV rows per schedule segment; one constant for every exported program."""
    return int(policy()["split_kv"])


def max_batch():
    """Request ceiling of the scheduler's shared-memory prefix buffer."""
    return int(policy()["max_batch"])


def next_n_atoms(next_n):
    """``num_next_n_atoms`` of the Q-atom rule for ``next_n`` (catalog rule).

    Independent of the head count and the page size, so the metadata entry keeps
    DeepGEMM's ``(context_lens, block_kv, num_sms)`` signature.
    """
    rule = policy()["next_n_atoms"]
    key = str(int(next_n))
    if key not in rule:
        raise ValueError(
            f"next_n = {next_n} has no exported program; exported: {sorted(rule)}"
        )
    return int(rule[key])


def route_name(num_heads, page_kv, next_n, logits_dtype="float32"):
    """Logical route of a problem (host-known scalars only)."""
    if logits_dtype not in LOGITS_DTYPES:
        raise ValueError(f"logits_dtype must be one of {LOGITS_DTYPES}")
    return f"sm120:fp8:h{int(num_heads)}:p{int(page_kv)}:n{int(next_n)}"


def metadata_route_name():
    """Route of the standalone scheduler launch."""
    return str(policy()["metadata_route"])


def route_available(num_heads, page_kv, next_n, logits_dtype="float32"):
    """True when the shipped catalog carries this paged route (host-only)."""
    try:
        if num_heads not in exported_heads() or page_kv not in exported_page_sizes():
            return False
        next_n_atoms(next_n)
        route = route_name(num_heads, page_kv, next_n, logits_dtype)
    except ValueError:
        return False
    return route in _catalog()["routes"]


def paged_logits_stride(max_context_len):
    """FP32 row stride: ``align(align(max_context_len, 128), 256)`` cells."""
    first, second = (int(v) for v in policy()["logits_stride_alignment"])
    aligned = (int(max_context_len) + first - 1) // first * first
    return (aligned + second - 1) // second * second


def metadata_shape(num_sms):
    """Shape of the schedule buffer: ``(num_sms + 1, 2)`` int32."""
    return (int(num_sms) + 1, 2)


# ---------------------------------------------------------------------------
# Architecture targets and JIT
# ---------------------------------------------------------------------------


def supported_capabilities():
    """Admitted capabilities that are also FlashInfer build targets.

    Targets follow ``FLASHINFER_CUDA_ARCH_LIST`` (or the visible devices); this
    never probes nvcc or a device capability to choose them.
    """
    from ...compilation_context import CompilationContext

    targets = CompilationContext().TARGET_CUDA_ARCHS
    return tuple(
        capability
        for capability in SUPPORTED_CAPABILITIES
        if any(
            (capability[0], suffix) in targets
            for suffix in _TARGET_SUFFIXES[capability]
        )
    )


def require_supported_capability(capability):
    """Return ``capability`` when the family is built for it, else raise."""
    capability = (int(capability[0]), int(capability[1]))
    if capability not in _ARCHES:
        supported = " or ".join(f"{a}.{b}" for a, b in SUPPORTED_CAPABILITIES)
        raise RuntimeError(
            f"SM120 paged MQA logits requires compute capability {supported}, got {capability[0]}.{capability[1]}"
        )
    if capability not in supported_capabilities():
        raise RuntimeError(
            f"SM120 paged MQA logits is not a build target for compute "
            f"capability {capability[0]}.{capability[1]}; include "
            f"{capability[0]}.{capability[1]} in FLASHINFER_CUDA_ARCH_LIST"
        )
    return capability


@functools.cache
def device_facts(device_index):
    """``(arch, sm_count)`` of one CUDA device through FlashInfer's queries."""
    import torch
    from flashinfer.utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = _ARCHES.get(capability)
    catalogued = sorted(_catalog()["arches"])
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"SM120 paged MQA logits has no exported programs for compute "
            f"capability {capability}; catalogued architectures: {catalogued}"
        )
    require_supported_capability(capability)
    return arch, int(get_device_sm_count(device))


def device_arch(device):
    """Generated-program architecture for ``device`` (raises when none ships)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("SM120 paged MQA logits requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device_facts(index)[0]


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm120a_nvcc_flags

    return {"sm_120a": sm120a_nvcc_flags}[arch]


def program_names(route=None):
    """Catalogued program names (optionally only those a route launches)."""
    catalog = _catalog()
    if route is None:
        return sorted(catalog["programs"])
    record = catalog["routes"][route]
    if record.get("sequence"):
        return [record["sequence"]]
    return [program for _stage, program in record["stages"]]


def program_spec(arch, name):
    """FlashInfer JIT build specification of one generated program.

    The programs take no compile-line definitions: the CTA budget is the launch
    grid.  The architecture is part of the spec name so two architectures never
    share one cached library.
    """
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = _catalog()["programs"][name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    return gen_jit_spec(
        name=f"{name}_{arch}",
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[
            *_nvcc_flags(arch),
            *record["compile_flags"],
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,  # Only explicit stage flags select fast math.
    )


@functools.cache
def load_program(arch, name):
    record = _catalog()["programs"][name]
    spec = program_spec(arch, name)
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def _submission(arch, program, bindings, *, stage=None):
    """``((entry, args), (module, record))`` for one generated program.

    ``stage=None`` selects a prepared-sequence binding whose argument keys are
    ``"<stage>.<name>"``; a stage name selects a standalone program's binding.
    """
    module, record = load_program(arch, program)
    arguments = []
    for kind, key in record["arg_plan"]:
        if kind == "workspace":
            raise NotImplementedError(
                "SM120 paged MQA route unexpectedly requires descriptor storage"
            )
        if stage is None:
            selected, name = key.split(".", 1)
        else:
            selected, name = stage, key
        arguments.append(bindings[selected][name])
    return (module[record["ffi_entry"]], tuple(arguments)), (module, record)


def _resolve_device(tensor, sm_count):
    if tensor.device.type != "cuda":
        raise RuntimeError("SM120 paged MQA logits requires CUDA tensors")
    import torch

    index = (
        tensor.device.index
        if tensor.device.index is not None
        else torch.cuda.current_device()
    )
    arch, num_sms = device_facts(index)
    if sm_count is not None:
        # CTA budget override (tests, a restricted serving partition).  The
        # schedule is sized for this many CTAs and the logits grid launches
        # exactly that many; any positive count is a legal grid.
        if int(sm_count) < 1:
            raise ValueError("sm_count must be a positive CTA budget")
        num_sms = int(sm_count)
    return arch, num_sms


# ---------------------------------------------------------------------------
# Operand checks and argument plans
# ---------------------------------------------------------------------------


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
    if batch > max_batch():
        raise ValueError(
            f"batch {batch} exceeds the scheduler's request ceiling {max_batch()}"
        )
    return batch, next_n


def _check_schedule_meta(schedule_meta, num_sms, device):
    import torch

    shape = metadata_shape(num_sms)
    if (
        schedule_meta.dtype != torch.int32
        or tuple(schedule_meta.shape) != shape
        or not schedule_meta.is_contiguous()
        or schedule_meta.device != device
    ):
        raise ValueError(f"schedule_meta must be contiguous int32 {shape} on {device}")
    return schedule_meta


def fused_cache_rows(kv_cache, page_kv=None):
    """``(fused_2d, pages, page_kv, block_stride_bytes)`` of the fused uint8 KV cache.

    Accepts the engine's 4-D ``[pages, page_kv, 1, 132]`` cache -- contiguous,
    or the strided per-layer view a block-outermost allocation hands over
    (``stride(0)`` = the bytes between consecutive pages, larger than
    ``page_kv * 132`` when every layer's page shares one block or the page is
    padded) -- and the 2-D ``[pages, row_bytes]`` view (``row_bytes >= page_kv *
    132``, ``stride(0) >= row_bytes``), which requires ``page_kv`` explicitly.
    Block strides must be multiples of 16 bytes.  The kernel reads exactly
    ``page_kv * 132`` bytes of every page through a TMA descriptor that carries
    the physical block stride, so neither padding nor interleaving costs anything.
    """
    import torch

    if kv_cache.dtype != torch.uint8:
        raise ValueError("kv_cache must be the fused uint8 cache")
    if kv_cache.ndim == 4:
        pages, cached_page, kv_heads, row = (int(v) for v in kv_cache.shape)
        if kv_heads != 1 or row != FUSED_ROW_BYTES:
            raise ValueError(
                "kv_cache must be [pages, page_kv, 1, 132] (FP8 rows + FP32 scales)"
            )
        if page_kv is not None and int(page_kv) != cached_page:
            raise ValueError(
                f"page_kv = {page_kv} disagrees with kv_cache.shape[1] = {cached_page}"
            )
        if int(kv_cache.stride(3)) != 1 or (
            cached_page > 1 and int(kv_cache.stride(1)) != FUSED_ROW_BYTES
        ):
            raise ValueError(
                "the 4-D fused cache must hold dense 132-byte token rows "
                "(stride(3) == 1, stride(1) == 132)"
            )
        page_kv = cached_page
        row_bytes = page_kv * FUSED_ROW_BYTES
        # Keep the view's physical page stride: a per-layer view of a
        # block-outermost layout is not contiguous, and reshape would copy it.
        fused = kv_cache.as_strided((pages, row_bytes), (int(kv_cache.stride(0)), 1))
    elif kv_cache.ndim == 2:
        if page_kv is None:
            raise ValueError(
                "page_kv is required with a 2-D [pages, block_stride_bytes] cache"
            )
        pages, row_bytes = (int(v) for v in kv_cache.shape)
        page_kv = int(page_kv)
        fused = kv_cache
    else:
        raise ValueError(
            f"kv_cache must be 4-D [pages, page_kv, 1, 132] or 2-D [pages, block_stride_bytes], got {kv_cache.ndim}-D"
        )
    block_stride_bytes = int(fused.stride(0))
    if (
        row_bytes < page_kv * FUSED_ROW_BYTES
        or row_bytes % 16
        or int(fused.stride(1)) != 1
        or block_stride_bytes < row_bytes
        or block_stride_bytes % 16
    ):
        raise ValueError(
            "fused cache rows must be unit-stride, at least "
            f"{page_kv * FUSED_ROW_BYTES} bytes and 16-byte aligned, with a 16-byte-aligned "
            f"block stride >= the row; got row {row_bytes} B, block stride {block_stride_bytes} B"
        )
    return fused, pages, page_kv, block_stride_bytes


def metadata_bindings(context_lens, schedule_meta, *, num_sms):
    """Argument plan of the scheduler stage (one warp, grid (1, 1, 1))."""
    batch, next_n = (int(v) for v in context_lens.shape)
    return dict(
        context_lens=context_lens.reshape(-1),
        schedule_meta=schedule_meta.reshape(-1),
        batch_size=batch,
        next_n=next_n,
        num_next_n_atoms=next_n_atoms(next_n),
        split_kv=split_kv(),
        num_sms=int(num_sms),
        grid_x=1,
        grid_y=1,
        grid_z=1,
    )


def logits_bindings(
    q, fused, weights, context_lens, block_table, schedule_meta, output, *, num_sms
):
    """Argument plan of the persistent logits stage.

    Q rows ``[B * next_n * H, 128]`` uint8, the fused cache as its byte rows
    with the FP32 scale tail aliased as the same buffer viewed as float32, the
    weights, the logits, the flattened context lengths, the block table as a
    flat view with its row stride, the schedule the first stage wrote, and the
    row stride of the logits.  The CTA budget is the launch grid only.
    """
    import torch

    batch = int(context_lens.shape[0])
    width = int(block_table.shape[1])
    span = (batch - 1) * int(block_table.stride(0)) + width
    return dict(
        Q=q.view(torch.uint8).reshape(-1, HEAD_DIM),
        KV=fused,
        KV_scales=fused.view(torch.float32),
        Weights=weights,
        Logits=output,
        context_lens=context_lens.reshape(-1),
        block_table=block_table.as_strided((span,), (1,)),
        schedule_meta=schedule_meta.reshape(-1),
        logits_stride=int(output.stride(0)),
        block_table_stride=int(block_table.stride(0)),
        grid_x=int(num_sms),
        grid_y=1,
        grid_z=1,
    )


# ---------------------------------------------------------------------------
# Plans
# ---------------------------------------------------------------------------


class Sm120MetadataPlan:
    """Bind the scheduler operands for repeated standalone metadata launches.

    ``context_lens`` int32 ``[B, next_n]``; ``page_kv`` an exported page size
    with an exported ``next_n``; ``num_sms`` the CTA budget of the logits call
    the schedule is built for (``None`` = the device's SM count).  ``out`` is an
    int32 ``[num_sms + 1, 2]`` buffer, allocated when not supplied.  ``run()``
    submits one launch on the current PyTorch stream and returns it.
    """

    def __init__(self, context_lens, page_kv, num_sms=None, *, out=None):
        import torch

        arch, num_sms = _resolve_device(context_lens, num_sms)
        _batch, next_n = _check_context_lens(context_lens)
        if int(page_kv) not in exported_page_sizes():
            raise ValueError(
                f"page_kv {page_kv} is not exported; exported page sizes: {exported_page_sizes()}"
            )
        next_n_atoms(next_n)
        if out is None:
            out = torch.empty(
                metadata_shape(num_sms), dtype=torch.int32, device=context_lens.device
            )
        _check_schedule_meta(out, num_sms, context_lens.device)
        self.arch, self.num_sms = arch, num_sms
        self.page_kv, self.next_n = int(page_kv), next_n
        self.route_name = metadata_route_name()
        self.route = _catalog()["routes"][self.route_name]
        stages = list(self.route["stages"])
        if [stage for stage, _program in stages] != ["metadata"]:
            raise ValueError(
                f"metadata route must be a single metadata stage, catalog has {stages}"
            )
        if self.route.get("sequence"):
            raise ValueError("the metadata route must not be a prepared sequence")
        bindings = {
            "metadata": metadata_bindings(context_lens, out, num_sms=num_sms),
        }
        self._submissions, self._programs = [], []
        self.program_names = [program for _stage, program in stages]
        for stage_name, program in stages:
            submit, loaded = _submission(arch, program, bindings, stage=stage_name)
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.context_lens, self.schedule_meta = context_lens, out
        self._retained = (context_lens, out)

    @property
    def launch_count(self):
        """FFI submissions ``run()`` issues."""
        return len(self._submissions)

    def run(self):
        """Submit the scheduler (one launch) on the current PyTorch stream."""
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.schedule_meta


class Sm120PagedIndexerPlan:
    """Bind the paged operands for repeated two-kernel logits submissions.

    ``q`` E4M3 ``[B, next_n, H, 128]`` contiguous; ``kv_cache`` the fused uint8
    cache (4-D ``[pages, page_kv, 1, 132]`` or its padded 2-D
    ``[pages, block_stride_bytes]`` view with ``page_kv`` given); ``weights``
    FP32 ``[B * next_n, H]`` contiguous; ``context_lens`` int32
    ``[B, next_n]`` contiguous; ``block_table`` int32 ``[B, S]`` with
    ``stride(1) == 1`` (any row stride, e.g. an engine's ``[::next_n]`` view);
    ``1 <= max_context_len <= S * page_kv``.  The route is
    ``(H, page_kv, next_n)`` and must be shipped (:func:`route_available`).

    ``output`` (FP32 ``[B * next_n, paged_logits_stride(max_context_len)]``) and
    ``schedule_meta`` (int32 ``[num_sms + 1, 2]``) are allocated when not
    supplied and stay bound to the plan; contents of every bound tensor may
    change between runs, including CUDA Graph replay.  ``run()`` submits the
    prepared sequence (one FFI call, two kernel launches: the scheduler then
    the logits program) on the current stream and returns ``logical_output``.
    The schedule is rebuilt on every call, so a caller-supplied
    ``schedule_meta`` never goes stale.
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
        page_kv=None,
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
        if num_heads not in exported_heads():
            raise ValueError(
                f"q has {num_heads} heads; exported head counts: {exported_heads()}"
            )
        if q.dtype != torch.float8_e4m3fn or not q.is_contiguous():
            raise ValueError("q must be contiguous E4M3")
        fused, pages, page_kv, block_stride_bytes = fused_cache_rows(kv_cache, page_kv)
        if page_kv not in exported_page_sizes():
            raise ValueError(
                f"page_kv {page_kv} is not exported; exported page sizes: {exported_page_sizes()}"
            )
        rows = batch * next_n
        if (
            weights.dtype != torch.float32
            or tuple(weights.shape) != (rows, num_heads)
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
        if max_context_len < 1 or max_context_len > int(block_table.shape[1]) * page_kv:
            raise ValueError("max_context_len must be in 1..S * page_kv")
        if not route_available(num_heads, page_kv, next_n, logits_dtype):
            raise ValueError(
                f"no exported SM120 paged MQA route for {num_heads} heads, page "
                f"{page_kv}, next_n {next_n}, {logits_dtype}"
            )
        tensors = [q, kv_cache, weights, context_lens, block_table]
        if any(tensor.device != q.device for tensor in tensors):
            raise ValueError("all inputs must be on one CUDA device")
        stride = paged_logits_stride(max_context_len)
        if rows * stride >= 2**31:
            raise ValueError("the logits element index must fit in int32")
        if output is None:
            output = torch.empty((rows, stride), dtype=torch.float32, device=q.device)
        if (
            output.dtype != torch.float32
            or tuple(output.shape) != (rows, stride)
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
        _check_schedule_meta(schedule_meta, num_sms, q.device)
        self.arch, self.num_sms, self.num_heads = arch, num_sms, num_heads
        self.page_kv, self.next_n, self.batch = page_kv, next_n, batch
        self.pages, self.block_stride_bytes = pages, block_stride_bytes
        self.max_context_len = max_context_len
        self.route_name = route_name(num_heads, page_kv, next_n, logits_dtype)
        self.route = _catalog()["routes"][self.route_name]
        stages = list(self.route["stages"])
        if [stage for stage, _program in stages] != ["metadata", "logits"]:
            raise ValueError(
                f"route {self.route_name} must be the ordered (metadata, logits) pair, catalog has {stages}"
            )
        bindings = {
            "metadata": metadata_bindings(context_lens, schedule_meta, num_sms=num_sms),
            "logits": logits_bindings(
                q,
                fused,
                weights,
                context_lens,
                block_table,
                schedule_meta,
                output,
                num_sms=num_sms,
            ),
        }
        sequence = self.route.get("sequence")
        if not sequence:
            raise ValueError(
                f"route {self.route_name} must be delivered as one prepared sequence"
            )
        self._submissions, self._programs = [], []
        self.program_names = [sequence]
        submit, loaded = _submission(arch, sequence, bindings, stage=None)
        self._submissions.append(submit)
        self._programs.append(loaded)
        self.output, self.schedule_meta = output, schedule_meta
        self.logical_output = output[:, :max_context_len]
        self._retained = (*tensors, fused, output, schedule_meta)

    @property
    def launch_count(self):
        """FFI submissions ``run()`` issues (one; the sequence holds both kernels)."""
        return len(self._submissions)

    @property
    def kernel_launches(self):
        """Kernel launches per submission: the scheduler and the logits program."""
        return int(self.route["kernel_launches"])

    def run(self):
        """Submit the (metadata, logits) sequence on the current PyTorch stream."""
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output


# ---------------------------------------------------------------------------
# DeepGEMM-shaped entries
# ---------------------------------------------------------------------------


def get_paged_mqa_logits_metadata(
    context_lens, page_kv, num_sms, indices=None, *, out=None
):
    """DeepGEMM's ``get_paged_mqa_logits_metadata``: a REAL scheduler launch.

    ``context_lens`` int32 ``[B, next_n]``, ``page_kv`` an exported page size
    with an exported ``next_n``, ``num_sms`` the CTA budget of the logits call,
    ``indices`` must be ``None`` (variable-length request selection is not
    ported).  Returns -- or fills ``out`` with -- the int32 ``[num_sms + 1, 2]``
    schedule the exported logits programs read.  The buffer is not
    interchangeable with DeepGEMM's: it carries this port's ``(q_atom_idx,
    kv_split_idx)`` walk boundaries.
    """
    if indices is not None:
        raise ValueError("indices (variable-length request selection) is unsupported")
    return Sm120MetadataPlan(context_lens, page_kv, num_sms, out=out).run()


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
    *,
    page_kv=None,
):
    """One-shot paged logits with DeepGEMM's ``fp8_paged_mqa_logits`` signature.

    ``schedule_meta`` is the int32 ``[num_sms + 1, 2]`` buffer from
    :func:`get_paged_mqa_logits_metadata`: its first dimension fixes the CTA
    budget, and the call REBUILDS it in the same sequence before the logits
    program reads it (the schedule is a pure function of ``context_lens``,
    ``split_kv``, ``next_n`` and the budget, so a buffer produced by the
    metadata entry on the same inputs is rewritten with identical bytes).
    ``None`` selects the device's SM count and an internal buffer.
    ``clean_logits=True`` is rejected for two-dimensional context lengths
    exactly as DeepGEMM does; ``indices`` must be ``None``.  Returns the FP32
    ``[B * next_n, max_context_len]`` view of a freshly allocated row-padded
    buffer.
    """
    if clean_logits:
        raise ValueError(
            "clean_logits=True is not supported with [B, next_n] context_lens (DeepGEMM semantics)"
        )
    if indices is not None:
        raise ValueError("indices (variable-length request selection) is unsupported")
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
    plan = Sm120PagedIndexerPlan(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        page_kv=page_kv,
        schedule_meta=schedule_meta,
        sm_count=num_sms,
    )
    return plan.run()


def prepare_sm120_paged_mqa_logits(
    q,
    kv_cache,
    weights,
    context_lens,
    block_table,
    max_context_len,
    *,
    page_kv=None,
    schedule_meta=None,
    output=None,
    sm_count=None,
):
    """Prepare repeated paged MQA logits on caller-provided operands."""
    return Sm120PagedIndexerPlan(
        q,
        kv_cache,
        weights,
        context_lens,
        block_table,
        max_context_len,
        page_kv=page_kv,
        schedule_meta=schedule_meta,
        output=output,
        sm_count=sm_count,
    )
