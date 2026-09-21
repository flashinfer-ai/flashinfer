"""Prepared compressed sparse MQA metadata and logits on SM103a.

Production runtime has no source compiler, quantizer or native oracle dependency.
Plans bind user tensors once; run() submits on the current PyTorch stream.
"""
from __future__ import annotations

import functools
import json
from pathlib import Path


@functools.cache
def _catalog():
    return json.loads(Path(__file__).with_name("sparse_mqa_catalog.json").read_text())


@functools.cache
def load_program(name):
    from flashinfer.jit import env
    from flashinfer.jit.core import gen_jit_spec, sm103a_nvcc_flags
    record = _catalog()["programs"][name]
    spec = gen_jit_spec(
        name=name,
        sources=[env.FLASHINFER_CSRC_DIR / p.removeprefix("csrc/") for p in record["sources"]],
        extra_cuda_cflags=[*sm103a_nvcc_flags, *record["compile_flags"],
                           "--device-entity-has-hidden-visibility=false"],
        extra_ldflags=["-lcuda"],
        extra_include_paths=[env.FLASHINFER_CSRC_DIR, env.FLASHINFER_INCLUDE_DIR],
        use_fast_math=False,  # Only explicit stage flags select fast math.
    )
    module = spec.build_and_load()
    return module, {**record, "library_path": str(spec.get_library_path())}


def route_key(config):
    return ":".join(str(config[key]) for key in
                    ("fmt", "paged", "num_max_sparse_blocks", "sparse_block_kv",
                     "page_kv", "use_unaligned_ks", "num_sms"))


def metadata_size_bytes(queries, capacity, *, fmt, sparse_block_kv, paged, num_sms):
    """Size the public packed split/schedule ABI, including its maximum extent."""
    blocks_per_split = (640 if fmt == "mxfp4" else 512) // sparse_block_kv
    ceildiv = lambda a, b: (a + b - 1) // b
    splits = (queries * ceildiv(capacity, blocks_per_split) if paged else
              ceildiv(queries, 2) * ceildiv(2 * capacity, blocks_per_split))
    entries = ceildiv(splits, num_sms) * num_sms
    return 16 + splits * (16 + blocks_per_split * 8) + entries * 16


def metadata_bindings(sparse_indices, metadata, workspace, *, starts=None, ends=None,
                      num_kv_tokens=0, context_lens=None, block_table=None,
                      request_indices=None, num_sms):
    import torch
    paged = context_lens is not None
    queries = sparse_indices.shape[0]
    placeholder = sparse_indices.view(torch.uint32)
    ctas = min(queries if paged else (queries + 1) // 2, num_sms * 4)
    return dict(Starts=starts.view(torch.uint32) if starts is not None else placeholder,
                Ends=ends.view(torch.uint32) if ends is not None else placeholder,
                Context=context_lens.view(torch.uint32) if paged else placeholder,
                BlockTable=block_table.view(torch.uint32) if paged else placeholder,
                Requests=request_indices.view(torch.uint32) if paged else placeholder,
                Sparse=placeholder, Metadata=metadata.view(torch.uint32),
                Workspace=workspace.view(torch.uint32), num_q_tokens=queries,
                num_kv_tokens=num_kv_tokens,
                block_table_stride=block_table.stride(0) if paged else 0,
                num_ctas=ctas, grid_x=ctas, grid_y=1, grid_z=1)


def logits_bindings(q, sf_q, kv, sf_kv, weights, metadata, output, *, fmt, paged, num_sms):
    import torch
    row_bytes = 64 if fmt == "mxfp4" else 128
    kv_tma = (kv if not paged else q).view(torch.uint8).reshape(-1, row_bytes)
    sf_tma = (sf_kv if not paged else sf_q).view(torch.int32).reshape(1, -1)
    return dict(Q=q.view(torch.uint8).reshape(-1, row_bytes),
                SF_Q=sf_q.view(torch.int32).reshape(-1, 32), Weights=weights.reshape(-1, 32),
                KV_TMA=kv_tma, SF_KV_TMA=sf_tma, KV=kv.view(torch.uint8),
                SF_KV=(sf_kv if not paged else sf_q).view(torch.uint32),
                Metadata=metadata.view(torch.uint32), Logits=output,
                logits_stride=output.stride(0), kv_page_stride_bytes=kv.stride(0) if paged else 0,
                num_sms=num_sms, grid_x=num_sms, grid_y=1, grid_z=1)


def _submission(program, bindings, *, stage=None):
    module, record = load_program(program)
    arguments = []
    for kind, key in record["arg_plan"]:
        if kind == "workspace":
            raise NotImplementedError("Sparse route unexpectedly requires external descriptor storage")
        if stage is None:
            selected, name = key.split(".", 1)
        else:
            selected, name = stage, key
        arguments.append(bindings[selected][name])
    return (module[record["ffi_entry"]], tuple(arguments)), (module, record)


def _check_tensors(tensors, device):
    if any(t.device != device or not t.is_contiguous() for t in tensors if t is not None):
        raise ValueError("inputs must be contiguous tensors on one CUDA device")


class SparseMetadataPlan:
    """Prepare an upstream-compatible metadata buffer for repeated generation.

    sparse_indices is sorted int32[Q,capacity], with source-format duplicate
    padding in unused slots. Contiguous inputs use starts/ends int32[Q] and
    num_kv_tokens. Paged inputs use context_lens and request_indices int32[Q],
    plus a per-query block table int32[Q,pages]. Packed metadata is uint8; a
    caller-provided int32 workspace of 96+2*Q words must initially be zero.
    The metadata kernel restores its three counters after each submission.

    run() returns the same metadata tensor. Split allocation order may vary;
    unused bytes and scheduler ordering are not a canonical serialization.
    Metadata consumers must use this ABI rather than compare raw buffers.
    """
    def __init__(self, sparse_indices, *, fmt="mxfp4", sparse_block_kv=8, page_kv=64,
                 use_unaligned_ks=False, starts=None, ends=None, num_kv_tokens=0,
                 context_lens=None, block_table=None, request_indices=None,
                 metadata=None, workspace=None):
        import torch
        if sparse_indices.device.type != "cuda" or torch.cuda.get_device_capability(sparse_indices.device) != (10, 3):
            raise RuntimeError("Sparse MQA requires the validated SM103a target")
        if sparse_indices.dtype != torch.int32 or sparse_indices.ndim != 2:
            raise ValueError("sparse_indices must be int32[Q,capacity]")
        queries, capacity = sparse_indices.shape
        if queries < 1 or capacity < 1:
            raise ValueError("positive Q and sparse capacity are required")
        paged = context_lens is not None
        vectors = (context_lens, request_indices) if paged else (starts, ends)
        if any(t is None or t.dtype != torch.int32 or tuple(t.shape) != (queries,) for t in vectors):
            raise ValueError("window/request vectors must be int32[Q]")
        if paged:
            if block_table is None or block_table.dtype != torch.int32 or block_table.ndim != 2 or block_table.shape[0] != queries:
                raise ValueError("paged block_table must be int32[Q,pages]")
        elif num_kv_tokens <= 0:
            raise ValueError("contiguous metadata requires positive num_kv_tokens")
        self.num_sms = torch.cuda.get_device_properties(sparse_indices.device).multi_processor_count
        self.config = dict(fmt=fmt, paged=paged, num_max_sparse_blocks=capacity,
                           sparse_block_kv=sparse_block_kv, page_kv=page_kv,
                           use_unaligned_ks=bool(use_unaligned_ks), num_sms=self.num_sms)
        try:
            self.route = _catalog()["routes"][route_key(self.config)]
        except KeyError as error:
            raise NotImplementedError(f"No exported sparse physical schedule for {self.config}") from error
        size = metadata_size_bytes(queries, capacity, fmt=fmt, sparse_block_kv=sparse_block_kv,
                                   paged=paged, num_sms=self.num_sms)
        if metadata is None:
            metadata = torch.empty(size, dtype=torch.uint8, device=sparse_indices.device)
        if workspace is None:
            workspace = torch.zeros(96 + queries * 2, dtype=torch.int32, device=sparse_indices.device)
        if metadata.dtype != torch.uint8 or tuple(metadata.shape) != (size,):
            raise ValueError("metadata must be a uint8 vector with the required packed extent")
        if workspace.dtype != torch.int32 or tuple(workspace.shape) != (96 + queries * 2,):
            raise ValueError("workspace must be an int32 vector with 96+2*Q words")
        self._retained = (sparse_indices, starts, ends, context_lens, block_table, request_indices, metadata, workspace)
        _check_tensors(self._retained, sparse_indices.device)
        self.bindings = metadata_bindings(sparse_indices, metadata, workspace, starts=starts, ends=ends,
            num_kv_tokens=num_kv_tokens, context_lens=context_lens, block_table=block_table,
            request_indices=request_indices, num_sms=self.num_sms)
        stage = next(stage for stage in self.route["stages"] if stage["name"] == "metadata")
        self._submission, self._program = _submission(stage["program"], {"metadata": self.bindings}, stage="metadata")
        self.metadata, self.workspace = metadata, workspace
        self.queries, self.capacity = queries, capacity

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            entry, args = self._submission
            entry(*args)
        return self.metadata


class SparseMqaPlan:
    """Bind quantized q/kv and emit compressed BF16 logits, metadata included.

    q is packed E2M1[Q,32,64] or E4M3[Q,32,128]. sf_q is int32[Q,32]
    containing four UE8M0 scale bytes per row; weights is BF16[Q,32]. Contiguous
    kv has packed rows [K,64/128] and sf_kv int32[K]. Paged kv is uint8[pages,
    stride], with each page's row bytes followed by its scale bytes and padding
    to 512 bytes; sf_kv is unused. Output is BF16[Q,capacity*sparse_block_kv].
    Only selected valid slots are written; invalid slots retain their contents.
    Call run() for the complete metadata + logits pipeline on the current stream.
    A plan's mutable buffers must not be used concurrently on different streams.
    """
    def __init__(self, q, sf_q, kv, sf_kv, weights, metadata_plan, *, output=None):
        import torch
        config = metadata_plan.config
        queries, capacity = metadata_plan.queries, metadata_plan.capacity
        row = 64 if config["fmt"] == "mxfp4" else 128
        if tuple(q.shape) != (queries, 32, row):
            raise ValueError("Q shape does not match metadata query count and precision")
        supported_q = (torch.uint8, torch.int8) if config["fmt"] == "mxfp4" else (torch.float8_e4m3fn, torch.uint8)
        if q.dtype not in supported_q or sf_q.dtype != torch.int32 or tuple(sf_q.shape) != (queries, 32):
            raise ValueError("Q/scales have the wrong precision or packed scale layout")
        if weights.dtype != torch.bfloat16 or tuple(weights.shape) != (queries, 32):
            raise ValueError("weights must be BF16[Q,32]")
        if config["paged"]:
            minimum = (config["page_kv"] * (row + 4) + 511) // 512 * 512
            if kv.dtype != torch.uint8 or kv.ndim != 2 or kv.shape[1] < minimum or kv.stride(0) % 512:
                raise ValueError("paged KV must use the packed row/scales layout and 512-byte stride")
        elif kv.dtype not in supported_q or kv.ndim != 2 or kv.shape[1] != row or sf_kv is None or sf_kv.dtype != torch.int32 or tuple(sf_kv.shape) != (kv.shape[0],):
            raise ValueError("contiguous KV/scales have the wrong packed row layout")
        if output is None:
            output = torch.empty((queries, capacity * config["sparse_block_kv"]), device=q.device, dtype=torch.bfloat16)
        if output.dtype != torch.bfloat16 or tuple(output.shape) != (queries, capacity * config["sparse_block_kv"]):
            raise ValueError("output must be BF16[Q,capacity*sparse_block_kv]")
        self._retained = (q, sf_q, kv, sf_kv, weights, output, metadata_plan.metadata, metadata_plan.workspace)
        _check_tensors(self._retained, q.device)
        bindings = {"metadata": metadata_plan.bindings, "logits": logits_bindings(q, sf_q, kv, sf_kv, weights,
            metadata_plan.metadata, output, fmt=config["fmt"], paged=config["paged"], num_sms=metadata_plan.num_sms)}
        self._submissions, self._programs = [], []
        route = metadata_plan.route
        selections = [(None, route["sequence"])] if route["sequence"] else [(stage["name"], stage["program"]) for stage in route["stages"]]
        for name, program in selections:
            submit, loaded = _submission(program, bindings, stage=name)
            self._submissions.append(submit)
            self._programs.append(loaded)
        self.metadata_plan, self.output = metadata_plan, output

    def run(self):
        import tvm_ffi
        with tvm_ffi.use_torch_stream():
            for entry, args in self._submissions:
                entry(*args)
        return self.output
