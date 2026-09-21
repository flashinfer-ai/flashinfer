# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Prepared MXFP8 × MXFP4 grouped FC1/activation and FC2 from frozen Frost templates.

Inputs are already routed. Tokens are E4M3, weights are packed E2M1 pairs,
and scales are E8M0 bytes with one
scale per 32 K elements, and output is BF16. This explicit experimental path
does not register unmeasured tactics in MoELayer's automatic candidates.
cuDNN is needed only by the exporter, never by this runtime.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path

import torch

from .. import runtime
from ..activations import ACTIVATIONS, is_gated

_FC2 = "block_scale_grouped_gemm2"
_OPS = {f"block_scale_grouped_gemm1_{name}" for name in ACTIVATIONS} | {_FC2}


@dataclass(frozen=True)
class Mxfp8Mxfp4Kernel:
    artifact_id: str
    arch: str
    source_path: Path
    source_sha256: str
    workspace_bytes: int
    contract: dict
    tactic_metadata: dict
    launch_tail: tuple[str, ...]
    op: str

    @property
    def gated(self):
        return self.fc1 and is_gated(self.activation)

    @property
    def fc1(self):
        return self.op != _FC2

    @property
    def activation(self):
        return self.contract["activation"]

    @property
    def swap_ab(self):
        return self.tactic_metadata.get("swap_ab", False)

    @property
    def tactic(self):
        return (
            "cudnn_frost-mxfp8_mxfp4-v1",
            self.artifact_id,
            runtime._tactic_digest(self.source_sha256),
        )


@functools.lru_cache(maxsize=8)
def discover(root: Path | None = None) -> tuple[Mxfp8Mxfp4Kernel, ...]:
    """Load explicitly selected MXFP8 × MXFP4 artifacts, separate from the BF16 pool."""
    root = runtime.artifact_root("mxfp8_mxfp4") if root is None else Path(root)
    path = root / runtime._MANIFEST
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2 or not isinstance(
        payload.get("kernels"), list
    ):
        raise RuntimeError(f"invalid MXFP8 × MXFP4 manifest: {path}")
    result, seen = [], set()
    for raw in payload["kernels"]:
        op, identity = raw.get("op"), raw.get("id")
        if op not in _OPS:
            raise RuntimeError(f"unsupported MXFP8 × MXFP4 op: {op}")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise RuntimeError(
                "MXFP8 × MXFP4 artifact ids must be non-empty and unique"
            )
        seen.add(identity)
        runtime._validate_abi(raw, op)
        contract = raw.get("contract", {})
        expected = dict(
            token_dtype="float8_e4m3fn",
            weight_dtype="float4_e2m1fn_x2",
            output_dtype="bfloat16",
            scale_dtype="float8_e8m0fnu",
            block_size=32,
            scale_layout="F8_128x4",
            token_scale_layout="segmented_F8_128x4",
            activation="identity"
            if op == _FC2
            else op.removeprefix("block_scale_grouped_gemm1_"),
        )
        if any(contract.get(k) != v for k, v in expected.items()):
            raise RuntimeError(f"invalid MXFP8 × MXFP4 numerical contract: {identity}")
        tail = tuple(raw.get("launch", {}).get("tail", ()))
        store = raw.get("tactic", {}).get("store_mode")
        expected_tail = {"output"} if op == _FC2 else {"output", "scale"}
        if expected["activation"] == "situ":
            expected_tail |= {"gate_scale", "linear_scale"}
        if (
            store not in ("stg", "tma")
            or set(tail) != expected_tail
            or len(tail) != len(expected_tail)
            or tail[-1 if store == "tma" else 0] != "output"
        ):
            raise RuntimeError(f"invalid MXFP8 × MXFP4 launch tail: {identity}")
        size = raw.get("workspace_bytes")
        if type(size) is not int or size <= 0 or size % 128:
            raise RuntimeError(
                "MXFP8 × MXFP4 workspace must be a positive multiple of 128"
            )
        source, digest = runtime._read_source(root, raw)
        result.append(
            Mxfp8Mxfp4Kernel(
                identity,
                raw["arch"],
                source,
                digest,
                size,
                contract,
                raw["tactic"],
                tail,
                op,
            )
        )
    return tuple(result)


def segmented_scale_rows(rows: int, groups: int) -> int:
    """Capacity for any partition into independently 128-row-padded groups."""
    if rows < 0 or groups < 1:
        raise ValueError("rows must be nonnegative and groups positive")
    active = min(rows, groups)
    return 128 * (active + (rows - active) // 128)


def _scale_bytes(scales):
    if scales.dtype not in (torch.uint8, torch.float8_e8m0fnu):
        raise ValueError("scales must contain E8M0 bytes (uint8 or float8_e8m0fnu)")
    return scales.view(torch.uint8)


def _blocked(scales):
    # F8_128x4: [row_tile, k_tile, row%32, row//32, k%4].
    rows, cols = scales.shape
    padded = torch.zeros(
        ((rows + 127) // 128 * 128, (cols + 3) // 4 * 4),
        dtype=torch.uint8,
        device=scales.device,
    )
    padded[:rows, :cols] = _scale_bytes(scales)
    return (
        padded.view(-1, 4, 32, padded.shape[1] // 4, 4)
        .permute(0, 3, 2, 1, 4)
        .contiguous()
        .flatten()
    )


def _offsets(offsets, rows):
    if offsets.ndim != 1 or offsets.dtype != torch.int32 or not offsets.numel():
        raise ValueError("offsets must be a nonempty int32 vector")
    starts = offsets.tolist()
    if (
        starts[0] != 0
        or starts[-1] > rows
        or any(a > b for a, b in zip(starts, starts[1:], strict=False))
    ):
        raise ValueError("offsets must start at zero and increase within [0,S]")
    return starts


def pack_token_scales(scales: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    """Prepare token-major E8M0 scales for already-grouped tokens.

    This preparation helper reads offsets on the host; call outside capture.
    If routing changes, repack scales for the new group boundaries before launch.
    The returned allocation has capacity for any partition with these S/G.
    """
    if scales.ndim != 2 or scales.shape[1] == 0:
        raise ValueError("token scales must have shape [S,K/32]")
    if scales.is_cuda:
        with torch.cuda.device(scales.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("pack token scales outside CUDA Graph capture")
    rows, cols = scales.shape
    starts = _offsets(offsets, rows)
    result = torch.zeros(
        segmented_scale_rows(rows, len(starts)) * ((cols + 3) // 4 * 4),
        dtype=torch.uint8,
        device=scales.device,
    )
    pos = 0
    for begin, end in zip(starts, starts[1:] + [rows], strict=False):
        block = _blocked(scales[begin:end])
        result[pos : pos + block.numel()] = block
        pos += block.numel()
    return result.view(torch.float8_e8m0fnu)


def pack_weight_scales(scales: torch.Tensor) -> torch.Tensor:
    """Pack logical E8M0 [E,N,K/32] scales, independently for every expert."""
    if scales.ndim != 3 or min(scales.shape) <= 0:
        raise ValueError("weight scales must have shape [E,N,K/32]")
    return torch.stack([_blocked(s) for s in scales]).view(torch.float8_e8m0fnu)


def _launch_arguments(
    kernel,
    tokens,
    weights,
    offsets,
    token_scales,
    weight_scales,
    output,
    workspace,
    scale,
    activation_scales=None,
):
    s, k = tokens.shape
    e, n, _ = weights[0].shape
    token = tokens.unsqueeze(0).permute(1, 2, 0)
    weight = [w.view(torch.float4_e2m1fn_x2).permute(1, 2, 0) for w in weights]
    sfa = token_scales.view(torch.float8_e8m0fnu).reshape(-1, 1, 1)
    sfb = [
        sf.view(torch.float8_e8m0fnu).reshape(e, -1, 1).permute(1, 2, 0)
        for sf in weight_scales
    ]
    out = output.unsqueeze(0).permute(1, 2, 0)
    operands = (*weight, token) if kernel.swap_ab else (token, *weight)
    scales = (*sfb, sfa) if kernel.swap_ab else (sfa, *sfb)
    if kernel.swap_ab:
        out = out.transpose(0, 1)
    problem = (
        n if kernel.swap_ab else s,
        s if kernel.swap_ab else n,
        k,
        e,
        offsets.numel(),
        *(v for t in operands for v in t.stride()),
        *out.stride(),
    )
    tail = {"output": out, "scale": scale}
    tail.update(activation_scales or {})
    return (
        problem,
        offsets,
        workspace.view(torch.int64),
        *operands,
        *scales,
        *(tail[name] for name in kernel.launch_tail),
    )


class PreparedMxfp8Mxfp4GroupedGemm:
    """Bind one exported grouped kernel outside capture; replay on the current stream.

    Gated FC1 takes ``weights=(gate, up)`` and corresponding scales. Non-gated
    FC1 and FC2 take a one-element tuple. Activation parameters use the defaults
    in ACTIVATIONS, including SiTU's gate/linear scales (4/25) and Step's limit
    (7). Each weight is contiguous packed uint8 [E,N,K/2]. Scale blobs must
    follow pack_token_scales/pack_weight_scales; E8M0 bytes are never cast
    numerically. Every concurrent stream must own its workspace and plan.
    """

    def __init__(
        self,
        kernel: Mxfp8Mxfp4Kernel,
        tokens: torch.Tensor,
        weights: tuple[torch.Tensor, ...],
        offsets: torch.Tensor,
        token_scales: torch.Tensor,
        weight_scales: tuple[torch.Tensor, ...],
        out: torch.Tensor,
        *,
        scale: torch.Tensor | None = None,
        workspace: torch.Tensor | None = None,
    ):
        if tokens.device.type != "cuda":
            raise ValueError("MXFP8 × MXFP4 grouped GEMM requires CUDA tensors")
        with torch.cuda.device(tokens.device):
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("prepare MXFP8 × MXFP4 grouped GEMM outside capture")
        count = 2 if kernel.gated else 1
        if len(weights) != count or len(weight_scales) != count:
            raise ValueError(
                f"this MXFP8 × MXFP4 kernel requires {count} weight/scale pairs"
            )
        if tokens.ndim != 2 or weights[0].ndim != 3:
            raise ValueError("expected tokens[S,K] and weights[E,N,K/2]")
        s, k = tokens.shape
        e, n, wk = weights[0].shape
        if min(s, k, e, n) <= 0 or n % 128 or k % 128 or 2 * wk != k:
            raise ValueError(
                "positive dimensions and N/K divisible by 128 are required"
            )
        if max(s, k, n) >= 2**31:
            raise ValueError("MXFP8 × MXFP4 grouped dimensions must fit int32")
        if any(w.shape != weights[0].shape for w in weights):
            raise ValueError("all weights must share packed shape [E,N,K/2]")
        if offsets.device != tokens.device or not offsets.is_contiguous():
            raise ValueError("offsets must be contiguous on the token device")
        starts = _offsets(offsets, s)
        geometry = dict(s=s, n=n, k=k, experts=e, groups=len(starts))
        if kernel.arch != runtime._arch_for(tokens.device) or not all(
            runtime._dimension_matches(v, kernel.contract.get(key))
            for key, v in geometry.items()
        ):
            raise ValueError("MXFP8 × MXFP4 artifact geometry/architecture mismatch")
        if tokens.dtype != torch.float8_e4m3fn or any(
            w.dtype != torch.uint8 for w in weights
        ):
            raise ValueError(
                "MXFP8 tokens require E4M3; MXFP4 weights require packed uint8"
            )
        if out.shape != (s, n) or out.dtype != torch.bfloat16:
            raise ValueError("output must be BF16 [S,N]")
        required_sfa = segmented_scale_rows(s, len(starts)) * (k // 32)
        required_sfb = e * n * (k // 32)
        for sf, size in (
            (token_scales, required_sfa),
            *((v, required_sfb) for v in weight_scales),
        ):
            _scale_bytes(sf)
            if sf.numel() != size:
                raise ValueError(f"MXFP8 × MXFP4 scale blob requires {size} bytes")
        if scale is not None and not kernel.fc1:
            raise ValueError("FC2 does not accept an output scale")
        if scale is None:
            scale = torch.ones((1, 1, 1), dtype=torch.float32, device=tokens.device)
        if scale.dtype != torch.float32 or tuple(scale.shape) != (1, 1, 1):
            raise ValueError("FC1 output scale must be float32 [1,1,1]")
        activation_scales = {}
        if kernel.activation == "situ":
            activation = ACTIVATIONS["situ"]()
            activation_scales = {
                name: torch.full(
                    (1, 1, 1),
                    getattr(activation, name),
                    dtype=torch.float32,
                    device=tokens.device,
                )
                for name in ("gate_scale", "linear_scale")
            }
        if workspace is None:
            workspace = torch.empty(
                kernel.workspace_bytes, dtype=torch.uint8, device=tokens.device
            )
        if (
            workspace.ndim != 1
            or workspace.dtype != torch.uint8
            or workspace.numel() < kernel.workspace_bytes
        ):
            raise ValueError(
                "MXFP8 × MXFP4 workspace must be a sufficiently large uint8 buffer"
            )
        tensors = (
            tokens,
            *weights,
            offsets,
            token_scales,
            *weight_scales,
            out,
            workspace,
            scale,
            *activation_scales.values(),
        )
        if any(t.device != tokens.device or not t.is_contiguous() for t in tensors):
            raise ValueError(
                "MXFP8 × MXFP4 tensors must be contiguous on one CUDA device"
            )
        if workspace.data_ptr() % 128 or any(
            t.data_ptr() % 16
            for t in (tokens, *weights, token_scales, *weight_scales, out)
        ):
            raise ValueError(
                "MXFP8 × MXFP4 data/scales require 16B and workspace 128B alignment"
            )
        writes = {
            out.untyped_storage().data_ptr(),
            workspace.untyped_storage().data_ptr(),
        }
        reads = {
            t.untyped_storage().data_ptr()
            for t in tensors
            if t is not out and t is not workspace
        }
        if len(writes) != 2 or writes & reads:
            raise ValueError(
                "MXFP8 × MXFP4 output/workspace must not alias inputs or each other"
            )
        self.device, self.output = tokens.device, out
        self.workspace = workspace[: kernel.workspace_bytes]
        self._launch = runtime._load_kernel(kernel, tokens.device)
        self._args = _launch_arguments(
            kernel,
            tokens,
            weights,
            offsets,
            token_scales,
            weight_scales,
            out,
            self.workspace,
            scale,
            activation_scales,
        )

    def __call__(self):
        with torch.cuda.device(self.device):
            self.workspace.zero_()
            self._launch(*self._args, runtime._current_custream(self.device))
        return self.output
