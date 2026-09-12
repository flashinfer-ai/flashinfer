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
"""

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from ..api_logging import flashinfer_api
from ._pcie_ipc_collective_tuning import (
    PCIE_IPC_TUNE_BATCHES,
    PCIE_IPC_TUNE_REPEAT,
    PCIE_IPC_TUNE_WARMUP,
    PcieIpcCollectiveTuningSpec,
    PcieIpcCollectiveTuningState,
)
from ._pcie_ipc_ag_rs_module import get_pcie_ipc_ag_rs_module
from ._pcie_ipc_workspace import PCIE_IPC_MAX_BLOCKS, _PcieIpcWorkspace
from .pcie_ipc_ag_policy import (
    PcieIpcAllGatherLaunchConfig,
    PcieIpcAllGatherVariant,
    _is_launchable,
    get_pcie_ipc_all_gather_launch_config,
)


_TUNING_SPEC = PcieIpcCollectiveTuningSpec(
    cache_stem="pcie_ipc_all_gather",
    version=3,
    config_type=PcieIpcAllGatherLaunchConfig,
    variant_type=PcieIpcAllGatherVariant,
    grid_variants=(
        PcieIpcAllGatherVariant.FLAT_PUSH,
        PcieIpcAllGatherVariant.RECURSIVE_DOUBLING,
    ),
    fixed_configs=(
        PcieIpcAllGatherLaunchConfig(1, 32, PcieIpcAllGatherVariant.COPY_ENGINE),
    ),
    is_launchable=_is_launchable,
)


class PcieIpcAllGatherWorkspace(_PcieIpcWorkspace):
    """Workspace for out-of-place PCIe IPC all-gather.

    ``max_numel`` is the largest rank-local **input** shard, not the gathered
    output size.  Input has shape ``[local_rows, hidden]`` and output is
    rank-major with shape ``[world_size * local_rows, hidden]``.  Input and a
    caller-provided output must be 16-byte aligned.

    Construction, calls, and destruction are collective.  Every rank must
    issue the same call sequence and launch configuration.  One workspace is
    bound to one ordered CUDA stream and owns a slab independent from every
    other PCIe IPC workspace.

    A captured CUDA graph keeps using this workspace's handle and IPC slab.
    Keep the workspace alive until every replay has finished, and never replay
    a graph concurrently with another call that uses the same workspace.

    BF16, FP16 and FP32 are supported. All variants move opaque 16-byte packs,
    so the input bit pattern is preserved exactly. The default launch uses a
    measured exact-shape cache when available and otherwise falls back to a
    conservative seed.
    """

    def __init__(
        self,
        group: ProcessGroup,
        max_numel: int,
        dtype: torch.dtype = torch.bfloat16,
        max_blocks: int = PCIE_IPC_MAX_BLOCKS,
        profile: Optional[str] = None,
        tune_batches: Sequence[int] = PCIE_IPC_TUNE_BATCHES,
        tune_cache: Optional[str] = None,
    ) -> None:
        super().__init__(
            group=group,
            max_numel=max_numel,
            dtype=dtype,
            max_blocks=max_blocks,
            profile=profile,
            collective_name="PCIe IPC all-gather",
            module_getter=get_pcie_ipc_ag_rs_module,
            workspace_size_name="all_gather_workspace_size",
            init_name="all_gather_init",
            dispose_name="all_gather_dispose",
            layout_uses_ordered_4plus4=True,
        )
        try:
            self._tuning_state = PcieIpcCollectiveTuningState(
                self,
                _TUNING_SPEC,
                tune_batches=tune_batches,
                tune_cache=tune_cache,
            )
        except Exception:
            self.destroy()
            raise

    def launch_config(
        self, inp: torch.Tensor
    ) -> Optional[PcieIpcAllGatherLaunchConfig]:
        """Return a deterministic config for ``inp``, or ``None`` if unsupported."""
        if not self._supports_input(inp):
            return None
        if inp.numel() > self.max_numel:
            return None
        return get_pcie_ipc_all_gather_launch_config(
            self.world_size,
            inp.numel(),
            self.max_blocks,
            self.element_size,
            self.ordered_4plus4,
        )

    def supports(self, inp: torch.Tensor) -> bool:
        """Whether this workspace can all-gather ``inp``."""
        return self.launch_config(inp) is not None

    def tuned_launch_config(
        self, inp: torch.Tensor
    ) -> Optional[PcieIpcAllGatherLaunchConfig]:
        """Return a measured config when cached, otherwise the seed config."""
        return self._tuning_state.tuned_launch_config(inp)

    def tune(
        self,
        hiddens: Sequence[int],
        *,
        cache: Optional[str] = None,
        warmup: int = PCIE_IPC_TUNE_WARMUP,
        repeat: int = PCIE_IPC_TUNE_REPEAT,
    ) -> Dict[Tuple[int, int], PcieIpcAllGatherLaunchConfig]:
        """Measure configs for ``hiddens`` over this workspace's tune batches."""
        return self._tuning_state.tune(
            hiddens,
            cache=cache,
            warmup=warmup,
            repeat=repeat,
        )

    @flashinfer_api
    def all_gather(
        self,
        inp: torch.Tensor,
        *,
        out: Optional[torch.Tensor] = None,
        config: Optional[PcieIpcAllGatherLaunchConfig] = None,
    ) -> torch.Tensor:
        """Gather one rank-local shard into a rank-major output tensor.

        Pass ``config`` only when every rank has agreed on it. An explicit
        config is validated against the workspace limits.
        """
        if config is None:
            config = self.tuned_launch_config(inp)
            if config is None:
                raise ValueError(
                    f"unsupported shape {tuple(inp.shape)} dtype {inp.dtype} "
                    f"at {self.world_size} ranks; check supports() first"
                )
        elif self.launch_config(inp) is None:
            raise ValueError(
                f"unsupported shape {tuple(inp.shape)} dtype {inp.dtype} "
                f"at {self.world_size} ranks; check supports() first"
            )
        elif not isinstance(config, PcieIpcAllGatherLaunchConfig):
            raise TypeError("config must be a PcieIpcAllGatherLaunchConfig")
        elif not _is_launchable(
            self.world_size,
            config,
            self.max_blocks,
            self.ordered_4plus4,
        ):
            raise ValueError(f"config {config} is not launchable for this workspace")

        expected_shape = (inp.shape[0] * self.world_size, inp.shape[1])
        if out is None:
            out = torch.empty(expected_shape, dtype=inp.dtype, device=inp.device)
        else:
            self._validate_output(out, expected_shape)
        self._validate_no_overlap(inp, out)

        self._check_stream("all_gather")
        self._launch(inp, out, config)
        return out

    def _launch(
        self,
        inp: torch.Tensor,
        out: torch.Tensor,
        config: PcieIpcAllGatherLaunchConfig,
    ) -> None:
        """Launch an already validated config; used by the tuner as well."""
        get_pcie_ipc_ag_rs_module().all_gather(
            self.handle,
            inp,
            out,
            config.blocks,
            config.threads,
            int(config.variant),
        )

    def _tuning_input_shape(self, batch: int, hidden: int) -> Tuple[int, int]:
        return batch, hidden

    def _tuning_shard_numel(self, inp: torch.Tensor) -> int:
        return inp.numel()

    def _tuning_output_shape(self, inp: torch.Tensor) -> Tuple[int, int]:
        return inp.shape[0] * self.world_size, inp.shape[1]

    def _tuning_reference(self, inp: torch.Tensor, out: torch.Tensor) -> None:
        dist.all_gather_into_tensor(out, inp, group=self.group)


__all__ = [
    "PcieIpcAllGatherLaunchConfig",
    "PcieIpcAllGatherVariant",
    "PcieIpcAllGatherWorkspace",
]
