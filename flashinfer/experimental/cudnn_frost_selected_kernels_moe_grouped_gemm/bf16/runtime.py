"""Runtime for cuDNN Frost-generated grouped GEMM1 activation kernels.

cuDNN Frost generates standalone Python sources on the build box. At runtime, CuTe
DSL compiles them through FlashInfer's versioned JIT cache. No cuDNN frontend
or cuDNN Frost graph compiler is needed in the deployed process.
"""

from __future__ import annotations

import functools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from ....autotuner import OptimizationProfile, TunableRunner
from ..activations import ACTIVATIONS, contract_activation, is_gated

from ..runtime import (
    _MANIFEST as _MANIFEST,
    _arch_for as _arch_for,
    _clear_source_cache,
    _current_custream as _current_custream,
    _dimension_matches as _dimension_matches,
    _load_kernel as _load_kernel,
    _load_source as _load_source,
    _read_source as _read_source,
    _tactic_digest as _tactic_digest,
    _validate_abi as _validate_abi,
    artifact_root as artifact_root,
)

_artifact_cache_version = 0


@dataclass(frozen=True)
class CudnnFrostGroupedGemm1Kernel:
    artifact_id: str
    arch: str
    source_path: Path
    source_sha256: str
    workspace_bytes: int
    contract: dict[str, Any]
    tactic_metadata: dict[str, Any]
    launch_tail: tuple[str, ...]

    @property
    def activation(self):
        return contract_activation(self.contract)

    @property
    def gated(self):
        return is_gated(self.activation)

    @property
    def swap_ab(self) -> bool:
        return self.tactic_metadata.get("swap_ab", False)

    @property
    def tactic(self) -> tuple[str, str, str]:
        """Stable identity suitable for FlashInfer's persisted autotuner."""

        return (
            "cudnn_frost-grouped-swiglu-v2",
            self.artifact_id,
            _tactic_digest(self.source_sha256),
        )


def _artifact_roots() -> tuple[Path, ...]:
    packaged = artifact_root("bf16")
    return (packaged,) if packaged.is_dir() else ()


def _read_root(root: Path) -> list[CudnnFrostGroupedGemm1Kernel]:
    path = root / _MANIFEST
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    if payload.get("schema_version") != 2:
        raise RuntimeError(f"{path}: unsupported cuDNN Frost manifest schema")
    kernels = payload.get("kernels")
    if not isinstance(kernels, list):
        raise RuntimeError(f"{path}: 'kernels' must be a list")

    result: list[CudnnFrostGroupedGemm1Kernel] = []
    seen: set[str] = set()
    for raw in kernels:
        artifact_id = raw.get("id")
        if not isinstance(artifact_id, str) or not artifact_id or artifact_id in seen:
            raise RuntimeError(f"{path}: kernel ids must be non-empty and unique")
        seen.add(artifact_id)
        op = raw.get("op", "")
        if not op.startswith("grouped_gemm1_"):
            continue
        activation = contract_activation(raw.get("contract", {}))
        if activation not in ACTIVATIONS or op != f"grouped_gemm1_{activation}":
            raise RuntimeError(f"Invalid FC1 activation contract: {artifact_id}")
        _validate_abi(raw, op)
        source_path, actual = _read_source(root, raw)
        workspace_bytes = int(raw.get("workspace_bytes", -1))
        if workspace_bytes < 0 or workspace_bytes % 128:
            raise RuntimeError(
                f"{path}: kernel {artifact_id!r} workspace must be a nonnegative "
                "multiple of 128 bytes"
            )
        launch_tail = tuple(raw.get("launch", {}).get("tail", ()))
        expected_tail = (
            {"output", "scale", "gate_scale", "linear_scale"}
            if activation == "situ"
            else {"output", "scale"}
        )
        if set(launch_tail) != expected_tail or len(launch_tail) != len(expected_tail):
            raise RuntimeError(
                f"{path}: kernel {artifact_id!r} has unsupported launch tail "
                f"{launch_tail!r}"
            )
        result.append(
            CudnnFrostGroupedGemm1Kernel(
                artifact_id=artifact_id,
                arch=str(raw.get("arch", "")),
                source_path=source_path,
                source_sha256=actual,
                workspace_bytes=workspace_bytes,
                contract=dict(raw.get("contract", {})),
                tactic_metadata=dict(raw.get("tactic", {})),
                launch_tail=launch_tail,
            )
        )
    return result


@functools.lru_cache(maxsize=8)
def _discover(
    roots: tuple[Path, ...],
) -> tuple[CudnnFrostGroupedGemm1Kernel, ...]:
    kernels = [kernel for root in roots for kernel in _read_root(root)]
    identities = [kernel.artifact_id for kernel in kernels]
    if len(identities) != len(set(identities)):
        raise RuntimeError("duplicate cuDNN Frost kernel id across artifact roots")
    return tuple(kernels)


def clear_artifact_cache() -> None:
    global _artifact_cache_version

    from ..shortlist import _read
    from .fc2 import discover as discover_fc2
    from .moe import _selected_kernels_cached

    _artifact_cache_version += 1
    _discover.cache_clear()
    discover_fc2.cache_clear()
    _selected_kernels_cached.cache_clear()
    _clear_source_cache()
    _read.cache_clear()


def _validate_common(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None,
) -> tuple[int, int, int, int, int]:
    if grouped_tokens.ndim != 2:
        raise ValueError("grouped_tokens must have shape [S, K]")
    if gate_weights.ndim != 3 or up_weights.shape != gate_weights.shape:
        raise ValueError("gate_weights and up_weights must share shape [E, N, K]")
    s, k = map(int, grouped_tokens.shape)
    e, n, weight_k = map(int, gate_weights.shape)
    if weight_k != k:
        raise ValueError(f"token K={k} does not match weight K={weight_k}")
    if first_token_offset.ndim != 1 or first_token_offset.dtype != torch.int32:
        raise ValueError("first_token_offset must be a one-dimensional int32 tensor")
    groups = int(first_token_offset.numel())
    if groups == 0:
        raise ValueError("first_token_offset must contain at least one group")
    if scale.dtype != torch.float32 or scale.numel() != 1:
        raise ValueError("scale must contain one float32 value")
    if out is not None and (tuple(out.shape) != (s, n) or out.dtype != torch.bfloat16):
        raise ValueError(f"out must be BF16 with shape {(s, n)}")
    tensors = [
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
    ]
    if out is not None:
        tensors.append(out)
    if any(t.device != grouped_tokens.device for t in tensors):
        raise ValueError("all grouped GEMM1 tensors must be on the same device")
    if any(not t.is_contiguous() for t in tensors):
        raise ValueError("all grouped GEMM1 tensors must be contiguous")
    if any(t.dtype != torch.bfloat16 for t in tensors[:3]):
        raise ValueError("the v1 grouped GEMM1 ABI requires BF16 tokens and weights")
    return s, n, k, e, groups


def matching_kernels(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> tuple[CudnnFrostGroupedGemm1Kernel, ...]:
    s, n, k, e, groups = _validate_common(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        out,
    )
    values = {"s": s, "n": n, "k": k, "experts": e, "groups": groups}
    return tuple(
        kernel
        for kernel in _discover(_artifact_roots())
        if kernel.arch == _arch_for(grouped_tokens.device)
        and kernel.activation == "swiglu"
        and all(
            _dimension_matches(values[name], kernel.contract.get(name))
            for name in values
        )
    )


def workspace_size(
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor | None = None,
) -> int:
    kernels = matching_kernels(
        grouped_tokens,
        gate_weights,
        up_weights,
        first_token_offset,
        scale,
        out,
    )
    if not kernels:
        raise RuntimeError(
            "no generated cuDNN Frost grouped GEMM1 + SwiGLU kernel matches this call"
        )
    return max(kernel.workspace_bytes for kernel in kernels)


def _launch(
    kernel: CudnnFrostGroupedGemm1Kernel,
    grouped_tokens: torch.Tensor,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    first_token_offset: torch.Tensor,
    scale: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    if workspace.dtype != torch.uint8 or not workspace.is_contiguous():
        raise ValueError("workspace must be a contiguous uint8 tensor")
    if workspace.device != grouped_tokens.device:
        raise ValueError("workspace must be on the grouped token device")
    if workspace.numel() < kernel.workspace_bytes:
        raise ValueError(
            f"kernel requires {kernel.workspace_bytes} workspace bytes; "
            f"got {workspace.numel()}"
        )
    if workspace.data_ptr() % 128:
        raise ValueError("cuDNN Frost grouped GEMM workspace must be 128-byte aligned")

    # cuDNN Frost's in-process CompiledMoeGemm wrapper resets the persistent grouped
    # scheduler counter before every launch.  The exported object deliberately
    # contains only the host/kernel launchable, so the embedding runtime must
    # perform that initialization.  Clearing the complete small descriptor
    # workspace is ABI-safe and also avoids depending on cuDNN Frost at deployment.
    workspace[: kernel.workspace_bytes].zero_()
    launch = _load_kernel(kernel, grouped_tokens.device)
    token = grouped_tokens.unsqueeze(0).permute(1, 2, 0)
    gate = gate_weights.permute(1, 2, 0)
    up = up_weights.permute(1, 2, 0)
    output = out.unsqueeze(0).permute(1, 2, 0)
    s, k = map(int, grouped_tokens.shape)
    e, n, _ = map(int, gate_weights.shape)
    operands = (token, gate, up) if kernel.gated else (token, gate)
    if kernel.swap_ab:
        operands = (gate, up, token) if kernel.gated else (gate, token)
        output = output.transpose(0, 1)
    problem = (
        n if kernel.swap_ab else s,
        s if kernel.swap_ab else n,
        k,
        e,
        int(first_token_offset.numel()),
        *(int(stride) for operand in operands for stride in operand.stride()),
        *map(int, output.stride()),
    )
    workspace_i64 = workspace[: kernel.workspace_bytes].view(torch.int64)
    tail_tensors = {
        "output": output,
        "scale": scale[:1].reshape(1, 1, 1),
    }
    if kernel.activation == "situ":
        if scale.numel() != 3:
            raise ValueError("SiTU launch needs scale, gate_scale, linear_scale")
        tail_tensors.update(
            gate_scale=scale[1:2].reshape(1, 1, 1),
            linear_scale=scale[2:3].reshape(1, 1, 1),
        )
    launch(
        problem,
        first_token_offset,
        workspace_i64,
        *operands,
        *(tail_tensors[name] for name in kernel.launch_tail),
        # Exported TVM-FFI functions do not accept keyword arguments, even
        # though the in-process CuTe callable uses ``stream=``.
        _current_custream(grouped_tokens.device),
    )


class CudnnFrostGroupedGemm1SwiGLURunner(TunableRunner):
    """One FlashInfer runner over all matching generated cuDNN Frost tactics."""

    def get_valid_tactics(
        self, inputs: list[torch.Tensor], profile: OptimizationProfile
    ) -> list[Any]:
        del profile
        return [kernel.tactic for kernel in matching_kernels(*inputs[:6])]

    def _resolve(self, inputs: list[torch.Tensor], tactic: Any):
        kernels = matching_kernels(*inputs[:6])
        if not kernels:
            raise RuntimeError(
                "no matching cuDNN Frost grouped GEMM1 + SwiGLU artifact"
            )
        if tactic == -1:
            return kernels[0]
        for kernel in kernels:
            if kernel.tactic == tactic:
                return kernel
        raise ValueError(f"unknown or stale cuDNN Frost tactic: {tactic!r}")

    def validate_tactic(self, inputs: list[torch.Tensor], tactic: Any) -> bool:
        try:
            self._resolve(inputs, tactic)
            return True
        except (RuntimeError, ValueError):
            return False

    def forward(
        self,
        inputs: list[torch.Tensor],
        tactic: Any = -1,
        do_preparation: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        del do_preparation, kwargs
        kernel = self._resolve(inputs, tactic)
        _launch(kernel, *inputs)
        return inputs[5]


__all__ = [
    "CudnnFrostGroupedGemm1Kernel",
    "CudnnFrostGroupedGemm1SwiGLURunner",
    "clear_artifact_cache",
    "matching_kernels",
    "workspace_size",
]
