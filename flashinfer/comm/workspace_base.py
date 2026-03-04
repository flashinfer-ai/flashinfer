"""
Copyright (c) 2025 by FlashInfer team.

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

from abc import ABC, abstractmethod
from typing import Optional, Any

import torch


class AllReduceFusionWorkspace(ABC):
    """Base class for AllReduce fusion workspaces."""

    # Explicit type annotations for mypy (needed due to __getattr__ in subclasses)
    world_size: int
    rank: int
    _destroyed: bool

    def __init__(self, world_size: int, rank: int):
        self.world_size = world_size
        self.rank = rank
        self._destroyed = False

    @property
    @abstractmethod
    def backend(self) -> str:
        """Return backend name."""
        pass

    @abstractmethod
    def destroy(self) -> None:
        """
        Destroy workspace and free resources.

        This should be called explicitly when done using the workspace.
        Prefer using AllReduceFusionContext context manager for automatic cleanup.
        """
        pass

    @abstractmethod
    def is_buffer_size_sufficient(
        self,
        tp_size: int,
        num_tokens: int,
        hidden_dim: int,
        dtype: torch.dtype,
        use_oneshot: Optional[Any] = None,
    ) -> bool:
        pass

    def __del__(self):
        """
        Destructor - safety net if destroy() wasn't called explicitly.

        Warns if cleanup wasn't done properly. Not recommended to rely on this
        as __del__ timing is non-deterministic and can cause issues with
        distributed/CUDA resources.
        """
        if not self._destroyed:
            import warnings

            warnings.warn(
                f"{self.__class__.__name__} was not explicitly destroyed. "
                f"Call workspace.destroy() or use AllReduceFusionContext to ensure "
                f"proper cleanup of distributed/CUDA resources.",
                ResourceWarning,
                stacklevel=2,
            )
            try:
                self.destroy()
            except Exception as e:
                # Can't raise in __del__, just warn
                warnings.warn(
                    f"Error during automatic cleanup of {self.__class__.__name__}: {e}",
                    ResourceWarning,
                    stacklevel=2,
                )
