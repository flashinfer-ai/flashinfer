"""Prepared dense FP4/FP8 MQA lightning-indexer logits on SM100a/SM103a.

Production runtime has no source compiler, quantizer or native oracle dependency.
Plans bind user tensors once; run() submits the metadata producer (where the
route has one) and the fused logits/cleanup kernel on the current PyTorch stream
without allocating, graph capture, or a native-library fallback.
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


def device_arch(device):
    """Exact generated-program architecture for ``device`` (raises when none is catalogued)."""
    import torch

    device = torch.device(device)
    catalogued = sorted(_catalog()["arches"])
    if device.type != "cuda":
        raise RuntimeError("Dense MQA requires a CUDA device")
    capability = tuple(torch.cuda.get_device_capability(device))
    arch = _ARCHES.get(capability)
    if arch is None or arch not in catalogued:
        raise RuntimeError(
            f"Dense MQA has no exported programs for compute capability "
            f"{capability}; catalogued architectures: {catalogued}"
        )
    return arch


def supported_num_sms(arch):
    """SM counts with catalogued routes for ``arch`` (the last route-key field)."""
    routes = _catalog()["arches"][arch]["routes"]
    return sorted({int(key.rsplit(":sm", 1)[1]) for key in routes})


def _nvcc_flags(arch):
    from flashinfer.jit.core import sm100a_nvcc_flags, sm103a_nvcc_flags

    return {"sm_100a": sm100a_nvcc_flags, "sm_103a": sm103a_nvcc_flags}[arch]


@functools.cache
def load_program(arch, name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec

    record = _catalog()["arches"][arch]["programs"][name]
    spec = gen_jit_spec(
        name=name,
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
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_key(config):
    return f"{config['precision']}:q{config['queries']}:k{config['keys']}:sm{config['num_sms']}"


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


def _submission(arch, program, bindings, *, stage=None):
    module, record = load_program(arch, program)
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
    multiple of 256. logits[q, k] = sum_h max(0, Q[q,h] . KV[k]) * weights[q,h]
    for start[q] <= k < end[q] and -inf elsewhere.

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
    ):
        import torch

        arch = device_arch(q.device)
        if precision not in ("fp4", "fp8"):
            raise ValueError("precision must be 'fp4' or 'fp8'")
        queries, keys = starts.numel(), kv.shape[0]
        if queries < 1 or keys < 1 or keys % BLOCK_KV:
            raise ValueError("positive Q and positive K divisible by 256 are required")
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
        self.num_sms = torch.cuda.get_device_properties(q.device).multi_processor_count
        self.config = dict(
            precision=precision, queries=queries, keys=keys, num_sms=self.num_sms
        )
        try:
            self.route = _catalog()["arches"][arch]["routes"][route_key(self.config)]
        except KeyError as error:
            raise NotImplementedError(
                f"No exported {arch} dense physical schedule for {self.config}"
            ) from error
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
        words = metadata_words(queries, self.num_sms)
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
            num_sms=self.num_sms,
        )
        self._submissions, self._programs = [], []
        sequence = self.route["sequence"]
        selections = (
            [(None, sequence)]
            if sequence
            else [(stage["name"], stage["program"]) for stage in self.route["stages"]]
        )
        for stage_name, program in selections:
            submit, loaded = _submission(arch, program, bindings, stage=stage_name)
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.output, self.metadata = output, metadata
        self.logical_output = output[:, :keys]
        self._retained = (*tensors, output, metadata)

    def run(self):
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.logical_output
