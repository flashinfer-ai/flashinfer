"""Prepared dense FP4/FP8 MQA lightning-indexer logits on SM100a/SM103a.

Production runtime has no source compiler, quantizer or native oracle dependency.
Plans bind user tensors once; run() submits the route's generated programs on
the current PyTorch stream without allocating, graph capture, or a fallback.

Routes are selected from host-known scalars only (precision, query count, KV
length range); the generated programs take the query count and the KV length at
runtime, so one program per physical schedule serves every catalogued
architecture. Every program takes the launch grid of the logits consumers (the
device's SM count, or a plan's CTA-budget override) as the compile-line
definition ``SM_COUNT``: the per-SM cost partition and the metadata offsets are
compile-time literals in every build, and one source text per program serves
every SM count.
"""

from __future__ import annotations

import functools
import json
from pathlib import Path

_ARCHES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}
NUM_HEADS = 32
BLOCK_Q = 4
BLOCK_KV = 256


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("dense_mqa_catalog.json").read_text())


@functools.cache
def device_facts(device_index):
    """``(arch, sm_count)`` of one CUDA device through FlashInfer's cached device queries."""
    import torch
    from flashinfer.utils import get_compute_capability, get_device_sm_count

    device = torch.device("cuda", device_index)
    capability = get_compute_capability(device)
    arch = _ARCHES.get(capability)
    catalogued = sorted(_catalog()["arches"])
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"Dense MQA has no exported programs for compute capability "
            f"{capability}; catalogued architectures: {catalogued}"
        )
    return arch, int(get_device_sm_count(device))


def device_arch(device):
    """Generated-program architecture for ``device`` (raises when none is catalogued)."""
    import torch

    device = torch.device(device)
    if device.type != "cuda":
        raise RuntimeError("Dense MQA requires a CUDA device")
    index = device.index if device.index is not None else torch.cuda.current_device()
    return device_facts(index)[0]


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


def program_names(route=None):
    """Names of the catalogued programs (optionally only those a route can launch)."""
    catalog = _catalog()
    if route is None:
        return sorted(catalog["programs"])
    record = catalog["routes"][route]
    names = [program for _stage, program in record["stages"]]
    if record.get("sequence"):
        names.append(record["sequence"])
    return names


def program_definitions(record, num_sms):
    """Compile-line definitions of a program record: ``SM_COUNT`` is the launch grid of the logits
    consumers (the SM count, or a plan's CTA-budget override)."""
    values = {"SM_COUNT": int(num_sms)}
    unknown = sorted(set(record["definitions"]) - set(values))
    if unknown:
        raise RuntimeError(
            f"catalog program requires definitions this runtime cannot supply: {unknown}"
        )
    return {name: values[name] for name in record["definitions"]}


def program_spec(arch, name, num_sms):
    """FlashInfer JIT build specification of one generated program for ``arch`` and the launch grid ``num_sms``
    (a compile-line definition of every program; the name carries every supplied value)."""
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = _catalog()["programs"][name]
    if arch not in record["arches"]:
        raise RuntimeError(f"program {name} is not exported for {arch}")
    definitions = program_definitions(record, num_sms)
    suffix = "".join(
        f"_{key.lower()}{value}" for key, value in sorted(definitions.items())
    )
    return gen_jit_spec(
        name=f"{name}_{arch}{suffix}",
        sources=[
            env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]
        ],
        extra_cuda_cflags=[
            *_nvcc_flags(arch),
            *record["compile_flags"],
            *(f"-D{key}={value}" for key, value in sorted(definitions.items())),
            "--device-entity-has-hidden-visibility=false",
        ],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,  # Only explicit stage flags select fast math.
    )


@functools.cache
def load_program(arch, name, num_sms):
    record = _catalog()["programs"][name]
    spec = program_spec(arch, name, num_sms)
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_name(precision, queries, keys):
    """Logical route of a problem: host-known scalars only, KV length by range."""
    policy = _catalog()["policy"]
    if precision == "fp4":
        return "fp4:q1" if queries == 1 else f"fp4:{metadata_tier(queries)}"
    if precision != "fp8":
        raise ValueError("precision must be 'fp4' or 'fp8'")
    if queries == 1:
        return "fp8:q1:short" if keys <= policy["fused_q1_max_kv"] else "fp8:q1"
    if queries == 128 and keys <= policy["fused_q128_max_kv"]:
        return "fp8:q128:short"
    kind = "fp8:full" if queries % BLOCK_Q == 0 else "fp8:partial"
    return f"{kind}:{metadata_tier(queries)}"


def metadata_tier(queries):
    """Metadata program tier of a query count: the smallest ceiling that covers it."""
    for tier, max_queries_of_tier in _catalog()["policy"]["metadata_tiers"]:
        if queries <= max_queries_of_tier:
            return tier
    raise ValueError(f"queries must be in 1..{max_queries()}")


def max_queries():
    """Largest query count the generated metadata schedule accepts."""
    return int(_catalog()["policy"]["max_q_tokens"])


def logits_stride(num_kv_tokens):
    """Physical FP32 logits row stride: align(K + 256, 8) elements."""
    return (num_kv_tokens + BLOCK_KV + 7) // 8 * 8


def metadata_words(num_q_tokens, num_sms):
    """int32 words of the schedule metadata: per-SM headers, then two words per query block."""
    return (3 * num_sms + 1) // 2 * 2 + 2 * ((num_q_tokens + BLOCK_Q - 1) // BLOCK_Q)


def stage_bindings(
    precision,
    q,
    kv,
    weights,
    starts,
    ends,
    output,
    metadata,
    *,
    q_scales=None,
    kv_scales=None,
    num_sms,
):
    """Encode the public tensor contract into the two generated argument plans."""
    import torch

    queries, keys = starts.numel(), kv.shape[0]
    blocks = (queries + BLOCK_Q - 1) // BLOCK_Q
    producer = dict(
        Starts=starts.view(torch.uint32),
        Ends=ends.view(torch.uint32),
        Metadata=metadata.view(torch.uint32),
        num_q_tokens=queries,
        num_kv_tokens=keys,
        grid_x=1,
        grid_y=1,
        grid_z=1,
    )
    consumer = dict(
        Weights=weights,
        Logits=output,
        ScheduleMeta=metadata.view(torch.uint32),
        cu_seq_len_k_start=starts,
        cu_seq_len_k_end=ends,
        seq_len=queries,
        seq_len_kv=keys,
        stride_logits=output.stride(0),
        num_q_blocks=blocks,
        grid_x=num_sms,
        grid_y=1,
        grid_z=1,
    )
    if precision == "fp4":
        consumer.update(
            Q=q.view(torch.uint8).reshape(-1, 64),
            KV=kv.view(torch.uint8).reshape(keys, 64),
            SF_Q=q_scales.view(torch.uint8).reshape(-1, 16),
            SF_KV=kv_scales.view(torch.uint8).reshape(-1, 16),
        )
    elif precision == "fp8":
        consumer.update(
            Q=q.view(torch.uint8).reshape(-1, 128),
            # DeepGEMM retains the otherwise-unused FP8 Q-scale descriptor in the
            # physical ABI; the live KV-scale map is aliased exactly as the source route.
            Q_scales_alias=kv_scales.view(1, keys),
            KV=kv.view(torch.uint8).reshape(keys, 128),
            KV_scales=kv_scales.view(1, keys),
            CandidateValues=output,
            CandidateIndices=starts,
            CandidateCounts=ends,
            ScoreThresholds=output,
            num_kv_splits=1,
            candidate_capacity=1,
        )
    else:
        raise ValueError("precision must be 'fp4' or 'fp8'")
    return {"metadata": producer, "logits": consumer}


def _submission(arch, program, bindings, num_sms, *, stage=None):
    module, record = load_program(arch, program, num_sms)
    arguments = []
    for kind, key in record["arg_plan"]:
        if kind == "workspace":
            raise NotImplementedError(
                "Dense route unexpectedly requires external descriptor storage"
            )
        if stage is None:
            selected, name = key.split(".", 1)
        else:
            selected, name = stage, key
        arguments.append(bindings[selected][name])
    return (module[record["ffi_entry"]], tuple(arguments)), (module, record)


class DenseMqaPlan:
    """Bind packed operands for repeated metadata-to-logits submissions.

    FP4 q/kv contain packed E2M1 bytes [Q,32,64]/[K,64]; q_scales and kv_scales
    are contiguous UE8M0 bytes [Q,32,4]/[K,4]. FP8 q/kv use E4M3
    [Q_storage,32,128]/[K,128], where Q_storage >= max(4,Q); kv_scales are
    FP32[K]. weights are FP32[Q,32] (FP4) or FP32[Q_storage,32] (FP8).
    starts/ends are int32[Q] windows with 0 <= start <= end <= K; K is a
    multiple of 256 and 1 <= Q <= max_queries(). logits[q, k] =
    sum_h max(0, Q[q,h] . KV[k]) * weights[q,h] for start[q] <= k < end[q]
    and -inf elsewhere.

    Inputs, output and metadata stay bound to the plan. Updating their contents
    is supported, including CUDA Graph replay. One plan is not concurrently
    reusable across streams because output and metadata are mutable. Output is
    FP32 with physical row stride logits_stride(K); consume output[:, :K]
    (``logical_output``). Every cell of the padded output is written by each
    submission. FP4 output storage must include the final four-row query tile.
    """

    def __init__(
        self,
        precision,
        q,
        kv,
        weights,
        starts,
        ends,
        *,
        q_scales=None,
        kv_scales=None,
        output=None,
        metadata=None,
        sm_count=None,
    ):
        import torch

        if q.device.type != "cuda":
            raise RuntimeError("Dense MQA requires CUDA tensors")
        device_index = (
            q.device.index
            if q.device.index is not None
            else torch.cuda.current_device()
        )
        arch, num_sms = device_facts(device_index)
        if sm_count is not None:
            # CTA budget override (tests, restricted serving partitions): the schedule
            # partitions over this many CTAs, the FP8 indexer is built with this count
            # defined and the metadata is sized for it. Any positive count is a legal
            # grid; above the device's SM count the extra CTAs run as a second wave.
            if int(sm_count) < 1:
                raise ValueError("sm_count must be a positive CTA budget")
            num_sms = int(sm_count)
        if precision not in ("fp4", "fp8"):
            raise ValueError("precision must be 'fp4' or 'fp8'")
        queries, keys = starts.numel(), kv.shape[0]
        if queries < 1 or queries > max_queries():
            raise ValueError(f"Q must be in 1..{max_queries()}")
        if keys < 1 or keys % BLOCK_KV:
            raise ValueError("positive K divisible by 256 is required")
        if (
            starts.dtype != torch.int32
            or ends.dtype != torch.int32
            or starts.shape != ends.shape
            or starts.ndim != 1
        ):
            raise ValueError("starts and ends must be equal-shaped int32 vectors")
        if weights.dtype != torch.float32:
            raise ValueError("weights must be FP32")
        packed_dtype = q.dtype in (torch.uint8, torch.int8)
        if precision == "fp4":
            if (
                not packed_dtype
                or kv.dtype != q.dtype
                or tuple(q.shape) != (queries, NUM_HEADS, 64)
                or tuple(kv.shape) != (keys, 64)
            ):
                raise ValueError("FP4 q/kv must be packed int8/uint8[Q,32,64]/[K,64]")
            if (
                q_scales is None
                or kv_scales is None
                or q_scales.dtype != torch.uint8
                or kv_scales.dtype != torch.uint8
            ):
                raise ValueError("FP4 q_scales/kv_scales must be UE8M0 uint8 tensors")
            if tuple(q_scales.shape) != (queries, NUM_HEADS, 4) or tuple(
                kv_scales.shape
            ) != (keys, 4):
                raise ValueError("FP4 scales must have shape [Q,32,4]/[K,4]")
            q_rows = queries
        else:
            if (
                q.dtype != torch.float8_e4m3fn
                or kv.dtype != torch.float8_e4m3fn
                or tuple(q.shape[1:]) != (NUM_HEADS, 128)
                or tuple(kv.shape) != (keys, 128)
            ):
                raise ValueError("FP8 q/kv must be E4M3[Q_storage,32,128]/[K,128]")
            q_rows = q.shape[0]
            if (
                q_rows < max(BLOCK_Q, queries)
                or kv_scales is None
                or kv_scales.dtype != torch.float32
                or tuple(kv_scales.shape) != (keys,)
            ):
                raise ValueError(
                    "FP8 requires Q_storage >= max(4,Q) and FP32 KV scales[K]"
                )
        if tuple(weights.shape) != (q_rows, NUM_HEADS):
            raise ValueError("weights must match the physical Q rows and 32 heads")
        tensors = [q, kv, weights, starts, ends, kv_scales]
        if q_scales is not None:
            tensors.append(q_scales)
        if any(t.device != q.device or not t.is_contiguous() for t in tensors):
            raise ValueError("all inputs must be contiguous on one CUDA device")
        self.arch = arch
        self.num_sms = num_sms
        self.route_name = route_name(precision, queries, keys)
        self.route = _catalog()["routes"][self.route_name]
        self.config = dict(
            precision=precision,
            queries=queries,
            keys=keys,
            num_sms=num_sms,
            route=self.route_name,
        )
        stride = logits_stride(keys)
        padded_rows = (queries + BLOCK_Q - 1) // BLOCK_Q * BLOCK_Q
        if output is None:
            rows = padded_rows if precision == "fp4" else queries
            output = torch.empty((rows, stride), dtype=torch.float32, device=q.device)[
                :queries
            ]
        if (
            output.dtype != torch.float32
            or tuple(output.shape) != (queries, stride)
            or output.device != q.device
            or not output.is_contiguous()
        ):
            raise ValueError("output must be contiguous FP32[Q,logits_stride(K)]")
        # The FP4 route's final CTA addresses the padded query rows of its last tile.
        if precision == "fp4" and (
            output.untyped_storage().nbytes() - output.storage_offset() * 4
            < padded_rows * stride * 4
        ):
            raise ValueError(
                "FP4 output backing storage must include the final 4-row tile"
            )
        words = metadata_words(queries, num_sms)
        if metadata is None:
            metadata = torch.empty(words, dtype=torch.int32, device=q.device)
        if (
            metadata.dtype != torch.int32
            or tuple(metadata.shape) != (words,)
            or metadata.device != q.device
            or not metadata.is_contiguous()
        ):
            raise ValueError("metadata has the wrong device, dtype or physical extent")
        bindings = stage_bindings(
            precision,
            q,
            kv,
            weights,
            starts,
            ends,
            output,
            metadata,
            q_scales=q_scales,
            kv_scales=kv_scales,
            num_sms=num_sms,
        )
        self._submissions, self._programs = [], []
        sequence = self.route.get("sequence")
        selections = (
            [(None, sequence)]
            if sequence
            else [(stage, program) for stage, program in self.route["stages"]]
        )
        self.program_names = [program for _stage, program in selections]
        for stage_name, program in selections:
            submit, loaded = _submission(
                arch, program, bindings, num_sms, stage=stage_name
            )
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.output, self.metadata = output, metadata
        self.logical_output = output[:, :keys]
        self._retained = (*tensors, output, metadata)

    @property
    def launch_count(self):
        """Number of FFI submissions ``run()`` issues (1 for sequence and fused routes)."""
        return len(self._submissions)

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output
