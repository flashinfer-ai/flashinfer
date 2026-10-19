"""
Copyright (c) 2023 by FlashInfer team.

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

import threading
from typing import Any, Optional

import torch

from ..core import JitSpec, MissingJITCacheError, _HeterogeneousAOTModule


class _LazyPagedKVStrideModule:
    """Load an independent paged-KV-stride module on first eager use."""

    _module_name: str

    def __init__(self, spec: JitSpec) -> None:
        self._spec = spec
        self._lock = threading.Lock()
        self._module: Optional[Any] = None
        self._device_modules: dict[int, Any] = {}

    @property
    def is_loaded(self) -> bool:
        """Whether this holder already has an in-process loaded module."""
        return self._module is not None

    @staticmethod
    def _check_not_capturing() -> None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "The lazy independent paged-KV-stride module cannot be compiled "
                "or loaded during CUDA graph capture. Call "
                "prewarm_paged_kv_stride_variant('independent') after plan() "
                "and before capture."
            )

    def _loaded_for_device(self, device: Optional[torch.device]) -> Any:
        module = self._module
        if not isinstance(module, _HeterogeneousAOTModule):
            return module
        index = device.index if device is not None else None
        if index is None:
            index = torch.cuda.current_device()
        return self._device_modules.get(index)

    def get(self, device: Optional[torch.device] = None) -> Any:
        """Resolve a module for the input device, outside graph capture."""
        resolved = self._loaded_for_device(device)
        if resolved is not None:
            return resolved

        with torch.cuda.device(device):
            self._check_not_capturing()
            with self._lock:
                resolved = self._loaded_for_device(device)
                if resolved is not None:
                    return resolved
                self._check_not_capturing()
                try:
                    module = self._module
                    if module is None:
                        module = self._spec.build_and_load()
                    resolved = module
                    if isinstance(module, _HeterogeneousAOTModule):
                        resolved = module._resolve_for_current_device()
                        self._device_modules[torch.cuda.current_device()] = resolved
                except MissingJITCacheError as exc:
                    raise MissingJITCacheError(
                        "Unequal K/V data strides require FlashInfer's lazy "
                        f"independent {self._module_name} module, which is not included in the "
                        "default JIT cache. Use equal-stride K/V tensors, enable "
                        "local JIT and call "
                        "prewarm_paged_kv_stride_variant('independent') after "
                        "plan(), or install a compatible independent-module cache "
                        "package when one becomes available.",
                        spec=exc.spec,
                    ) from exc
                self._module = module
                return resolved

    def prewarm(self, device: Optional[torch.device] = None) -> None:
        """Resolve the independent module for later capture on this device."""
        self.get(device)
