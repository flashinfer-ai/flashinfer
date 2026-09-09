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

Small, collective-local tuner shared by PCIe IPC AllGather and ReduceScatter.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
import torch.distributed as dist


PCIE_IPC_TUNE_BLOCKS: Tuple[int, ...] = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
)
PCIE_IPC_TUNE_THREADS: Tuple[int, ...] = (64, 128, 256, 512)
PCIE_IPC_TUNE_BATCHES: Tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
PCIE_IPC_TUNE_WARMUP = 10
PCIE_IPC_TUNE_REPEAT = 50

_CACHE_FORMAT_VERSION = 1
_PACK_BYTES = 16
_INIT_MAX_VALUE = 8


class _Config(Protocol):
    blocks: int
    threads: int
    variant: IntEnum


class _TunableWorkspace(Protocol):
    group: Any
    rank: int
    world_size: int
    device: torch.device
    dtype: torch.dtype
    element_size: int
    max_numel: int
    max_blocks: int
    profile: str
    ordered_4plus4: bool
    placement_fingerprint: str

    def _joint_check(self, local: dict, what: str) -> None: ...

    def launch_config(self, inp: torch.Tensor) -> Optional[_Config]: ...

    def rebind_stream(self) -> None: ...

    def _tuning_input_shape(self, batch: int, hidden: int) -> Tuple[int, int]: ...

    def _tuning_shard_numel(self, inp: torch.Tensor) -> int: ...

    def _tuning_output_shape(self, inp: torch.Tensor) -> Tuple[int, int]: ...

    def _tuning_reference(self, inp: torch.Tensor, out: torch.Tensor) -> None: ...

    def _launch(
        self, inp: torch.Tensor, out: torch.Tensor, config: _Config
    ) -> None: ...


@dataclass(frozen=True)
class PcieIpcCollectiveTuningSpec:
    cache_stem: str
    version: int
    config_type: type
    variant_type: type[IntEnum]
    grid_variants: Tuple[IntEnum, ...]
    fixed_configs: Tuple[_Config, ...]
    is_launchable: Callable[[int, _Config, int, bool], bool]


def default_cache_path(
    spec: PcieIpcCollectiveTuningSpec,
    world_size: int,
    dtype: torch.dtype,
    placement_fingerprint: str,
) -> str:
    override = os.getenv("FLASHINFER_AUTOTUNE_DIR")
    if override:
        base = Path(override)
    else:
        from ..jit.env import FLASHINFER_WORKSPACE_DIR

        base = FLASHINFER_WORKSPACE_DIR / "autotune"
    dtype_name = str(dtype).removeprefix("torch.")
    placement = placement_fingerprint or "unknown"
    return str(base / f"{spec.cache_stem}_ws{world_size}_{dtype_name}_{placement}.json")


def config_to_tactic(config: _Config) -> Tuple[int, int, int]:
    return int(config.variant), int(config.blocks), int(config.threads)


def tactic_to_config(
    spec: PcieIpcCollectiveTuningSpec, tactic: Sequence[int]
) -> _Config:
    if not isinstance(tactic, (list, tuple)) or len(tactic) != 3:
        raise ValueError(f"expected a three-element tactic, got {tactic!r}")
    variant, blocks, threads = (int(value) for value in tactic)
    return spec.config_type(blocks, threads, spec.variant_type(variant))


def candidate_configs(
    spec: PcieIpcCollectiveTuningSpec,
    *,
    world_size: int,
    shard_numel: int,
    element_size: int,
    max_blocks: int,
    ordered_4plus4: bool = False,
    blocks: Tuple[int, ...] = PCIE_IPC_TUNE_BLOCKS,
    threads: Tuple[int, ...] = PCIE_IPC_TUNE_THREADS,
) -> Tuple[_Config, ...]:
    """Enumerate useful, launchable configurations in deterministic order."""
    payload_bytes = shard_numel * element_size
    if (
        shard_numel <= 0
        or element_size <= 0
        or _PACK_BYTES % element_size != 0
        or payload_bytes % _PACK_BYTES != 0
    ):
        return ()

    shard_packs = payload_bytes // _PACK_BYTES
    configs: List[_Config] = []
    for variant in spec.grid_variants:
        for block_count in blocks:
            for thread_count in threads:
                useful_blocks = (shard_packs + thread_count - 1) // thread_count
                if block_count > useful_blocks:
                    continue
                config = spec.config_type(block_count, thread_count, variant)
                if spec.is_launchable(world_size, config, max_blocks, ordered_4plus4):
                    configs.append(config)
    configs.extend(
        config
        for config in spec.fixed_configs
        if spec.is_launchable(world_size, config, max_blocks, ordered_4plus4)
    )
    return tuple(dict.fromkeys(configs))


class PcieIpcCollectiveTuningState:
    """Exact-shape cache plus correctness-gated, slowest-rank tuning."""

    def __init__(
        self,
        workspace: _TunableWorkspace,
        spec: PcieIpcCollectiveTuningSpec,
        *,
        tune_batches: Sequence[int] = PCIE_IPC_TUNE_BATCHES,
        tune_cache: Optional[str] = None,
    ) -> None:
        self.workspace = workspace
        self.spec = spec
        self._entries: Dict[str, List[int]] = {}
        self._dirty_entries: Dict[str, List[int]] = {}
        self._resolved: Dict[Tuple[int, int], _Config] = {}

        error: Optional[str] = None
        try:
            batches = tuple(int(batch) for batch in tune_batches)
            if (
                not batches
                or any(batch <= 0 for batch in batches)
                or tuple(sorted(set(batches))) != batches
            ):
                raise ValueError(
                    "tune_batches must be unique, positive, and increasing"
                )
            cache_path = os.fspath(
                tune_cache
                or default_cache_path(
                    spec,
                    workspace.world_size,
                    workspace.dtype,
                    workspace.placement_fingerprint,
                )
            )
        except (TypeError, ValueError) as err:
            error = str(err)
            batches = ()
            cache_path = ""
        self.tune_batches = batches
        self.cache_path = cache_path
        workspace._joint_check(
            {
                "error": error,
                "tune_batches": batches,
                "tune_cache": cache_path,
            },
            "validating tuning arguments",
        )
        self._load()

    def _cache_key(self, inp: torch.Tensor) -> str:
        fields = (
            self.spec.version,
            self.spec.cache_stem,
            self.workspace.world_size,
            self.workspace.profile,
            self.workspace.ordered_4plus4,
            self.workspace.placement_fingerprint,
            self.workspace.max_blocks,
            self.workspace.max_numel,
            str(self.workspace.dtype),
            tuple(inp.shape),
        )
        return json.dumps(fields, separators=(",", ":"))

    def _load(self) -> None:
        error: Optional[str] = None
        digest = ""
        exists = os.path.isfile(self.cache_path)
        if exists:
            try:
                raw = Path(self.cache_path).read_bytes()
                digest = hashlib.sha256(raw).hexdigest()
                payload = json.loads(raw)
                if payload.get("format_version") != _CACHE_FORMAT_VERSION:
                    raise ValueError("unsupported PCIe IPC tuning cache format")
                entries = payload.get("entries")
                if not isinstance(entries, dict):
                    raise ValueError("tuning cache entries must be an object")
                self._entries = {
                    str(key): list(value) for key, value in entries.items()
                }
            except Exception as err:  # noqa: BLE001 - re-raised collectively
                error = f"{type(err).__name__}: {err}"
        self.workspace._joint_check(
            {"error": error, "cache_exists": exists, "cache_digest": digest},
            "loading the tuning cache",
        )

    def tuned_launch_config(self, inp: torch.Tensor) -> Optional[_Config]:
        seed = self.workspace.launch_config(inp)
        if seed is None:
            return None
        shape = (inp.shape[0], inp.shape[1])
        resolved = self._resolved.get(shape)
        if resolved is not None:
            return resolved
        tactic = self._entries.get(self._cache_key(inp))
        if tactic is None:
            self._resolved[shape] = seed
            return seed
        try:
            config = tactic_to_config(self.spec, tactic)
        except (TypeError, ValueError):
            self._resolved[shape] = seed
            return seed
        if not self.spec.is_launchable(
            self.workspace.world_size,
            config,
            self.workspace.max_blocks,
            self.workspace.ordered_4plus4,
        ):
            self._resolved[shape] = seed
            return seed
        self._resolved[shape] = config
        return config

    def _input(self, batch: int, hidden: int) -> torch.Tensor:
        shape = self.workspace._tuning_input_shape(batch, hidden)
        return torch.randint(
            0,
            _INIT_MAX_VALUE,
            shape,
            dtype=torch.int32,
            device=self.workspace.device,
        ).to(self.workspace.dtype)

    def _correct(
        self,
        inp: torch.Tensor,
        reference: torch.Tensor,
        config: _Config,
    ) -> bool:
        out = torch.empty(
            self.workspace._tuning_output_shape(inp),
            dtype=inp.dtype,
            device=inp.device,
        )
        out.fill_(float("nan"))
        failed = 0
        try:
            self.workspace._launch(inp, out, config)
            failed = int(not torch.equal(out, reference))
        except Exception:  # noqa: BLE001 - disqualify the tactic on every rank
            failed = 1
        verdict = torch.tensor([failed], dtype=torch.int32, device=inp.device)
        dist.all_reduce(verdict, op=dist.ReduceOp.MAX, group=self.workspace.group)
        return verdict.item() == 0

    def _latency_us(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        config: _Config,
        warmup: int,
        repeat: int,
    ) -> float:
        for _ in range(warmup):
            self.workspace._launch(inp, out, config)
        torch.cuda.synchronize(self.workspace.device)
        dist.barrier(group=self.workspace.group)
        torch.cuda.synchronize(self.workspace.device)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(self.workspace.device)
        start.record(stream)
        for _ in range(repeat):
            self.workspace._launch(inp, out, config)
        end.record(stream)
        end.synchronize()
        local_us = start.elapsed_time(end) * 1000.0 / repeat
        latency = torch.tensor([local_us], dtype=torch.float64, device=inp.device)
        dist.all_reduce(latency, op=dist.ReduceOp.MAX, group=self.workspace.group)
        return float(latency.item())

    def tune(
        self,
        hiddens: Sequence[int],
        *,
        cache: Optional[str] = None,
        warmup: int = PCIE_IPC_TUNE_WARMUP,
        repeat: int = PCIE_IPC_TUNE_REPEAT,
    ) -> Dict[Tuple[int, int], _Config]:
        """Tune all supported ``(hidden, batch)`` pairs collectively."""
        error: Optional[str] = None
        try:
            hidden_values = tuple(int(hidden) for hidden in hiddens)
            if not hidden_values or any(hidden <= 0 for hidden in hidden_values):
                raise ValueError("hiddens must contain positive integers")
            cache_path = os.fspath(cache or self.cache_path)
            warmup = int(warmup)
            repeat = int(repeat)
            if warmup < 0 or repeat <= 0:
                raise ValueError(
                    "warmup must be nonnegative and repeat must be positive"
                )
        except (TypeError, ValueError) as err:
            error = str(err)
            hidden_values = ()
            cache_path = ""
        if error is None:
            unsupported_hiddens = tuple(
                hidden
                for hidden in hidden_values
                if not any(
                    batch * hidden <= self.workspace.max_numel
                    and batch * hidden * self.workspace.element_size % _PACK_BYTES == 0
                    for batch in self.tune_batches
                )
            )
            if unsupported_hiddens:
                error = (
                    "no tune batch fits the workspace capacity and 16-byte "
                    f"packing for hiddens {unsupported_hiddens}"
                )
        self.workspace._joint_check(
            {
                "error": error,
                "hiddens": hidden_values,
                "cache": cache_path,
                "warmup": warmup,
                "repeat": repeat,
            },
            "starting tuning",
        )

        results: Dict[Tuple[int, int], _Config] = {}
        torch.cuda.synchronize(self.workspace.device)
        self.workspace.rebind_stream()
        for hidden in hidden_values:
            for batch in self.tune_batches:
                if batch * hidden > self.workspace.max_numel:
                    continue
                inp = self._input(batch, hidden)
                seed = self.workspace.launch_config(inp)
                if seed is None:
                    continue
                candidates = candidate_configs(
                    self.spec,
                    world_size=self.workspace.world_size,
                    shard_numel=self.workspace._tuning_shard_numel(inp),
                    element_size=self.workspace.element_size,
                    max_blocks=self.workspace.max_blocks,
                    ordered_4plus4=self.workspace.ordered_4plus4,
                )
                candidates = (seed,) + tuple(
                    config for config in candidates if config != seed
                )

                reference = torch.empty(
                    self.workspace._tuning_output_shape(inp),
                    dtype=inp.dtype,
                    device=inp.device,
                )
                self.workspace._tuning_reference(inp, reference)
                valid = [
                    config
                    for config in candidates
                    if self._correct(inp, reference, config)
                ]
                if seed not in valid:
                    raise RuntimeError(
                        f"{self.spec.cache_stem} seed {seed} failed correctness "
                        f"for shape {tuple(inp.shape)}"
                    )

                out = torch.empty_like(reference)
                timings = [
                    self._latency_us(inp, out, config, warmup, repeat)
                    for config in valid
                ]
                winner = valid[min(range(len(valid)), key=timings.__getitem__)]
                tactic = list(config_to_tactic(winner))
                self.workspace._joint_check(
                    {"error": None, "shape": tuple(inp.shape), "tactic": tactic},
                    "selecting a tuned configuration",
                )
                key = self._cache_key(inp)
                self._entries[key] = tactic
                self._dirty_entries[key] = tactic
                self._resolved[(inp.shape[0], inp.shape[1])] = winner
                results[(hidden, batch)] = winner

        if not results:
            raise ValueError("no requested tuning shape fits this workspace")
        self._save(cache_path)
        return results

    def _save(self, cache_path: str) -> None:
        error: Optional[str] = None
        if self.workspace.rank == 0:
            temporary = ""
            try:
                directory = os.path.dirname(cache_path) or "."
                os.makedirs(directory, exist_ok=True)
                entries: Dict[str, List[int]] = {}
                if os.path.isfile(cache_path):
                    existing = json.loads(Path(cache_path).read_bytes())
                    if existing.get("format_version") != _CACHE_FORMAT_VERSION:
                        raise ValueError("unsupported PCIe IPC tuning cache format")
                    existing_entries = existing.get("entries")
                    if not isinstance(existing_entries, dict):
                        raise ValueError("tuning cache entries must be an object")
                    entries.update(
                        {
                            str(key): list(value)
                            for key, value in existing_entries.items()
                        }
                    )
                entries.update(self._dirty_entries)
                payload = {
                    "format_version": _CACHE_FORMAT_VERSION,
                    "entries": dict(sorted(entries.items())),
                }
                with tempfile.NamedTemporaryFile(
                    mode="w", dir=directory, delete=False
                ) as output:
                    temporary = output.name
                    json.dump(payload, output, indent=2, sort_keys=True)
                    output.write("\n")
                os.replace(temporary, cache_path)
            except Exception as err:  # noqa: BLE001 - re-raised collectively
                error = f"{type(err).__name__}: {err}"
                if temporary:
                    with suppress(OSError):
                        os.unlink(temporary)
        self.workspace._joint_check({"error": error}, "saving the tuning cache")
        self._dirty_entries.clear()

    def destroy(self) -> None:
        self._entries.clear()
        self._dirty_entries.clear()
        self._resolved.clear()


__all__ = [
    "PCIE_IPC_TUNE_BATCHES",
    "PCIE_IPC_TUNE_BLOCKS",
    "PCIE_IPC_TUNE_REPEAT",
    "PCIE_IPC_TUNE_THREADS",
    "PCIE_IPC_TUNE_WARMUP",
    "PcieIpcCollectiveTuningSpec",
    "PcieIpcCollectiveTuningState",
    "candidate_configs",
    "config_to_tactic",
    "default_cache_path",
    "tactic_to_config",
]
