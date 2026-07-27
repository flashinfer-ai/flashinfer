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

# FlashInfer experimental namespace.
#
# Code under this package implements experimental backends and
# backend-specific logic: support checks, heuristics, routing, compilation,
# caching, and kernels that are not yet ready for stable support. Public
# experimental APIs live in core, marked with @flashinfer_experimental_api;
# only thin entry points in core may hand off to this package.
#
# Everything here is gated behind FLASHINFER_ENABLE_EXPERIMENTAL_FEATURES=1,
# provides no compatibility guarantees, and may change or be removed without
# deprecation. Importing this module is always allowed (so tooling and docs
# can introspect it); the gate is enforced at call time.
#
# See README.md in this directory for the full policy, and CLAUDE.md for
# agent-facing contribution rules.

from ..api_logging import (
    ExperimentalWarning,
    flashinfer_experimental_api,
    is_experimental_enabled,
    require_experimental,
)

__all__ = [
    "ExperimentalWarning",
    "flashinfer_experimental_api",
    "is_experimental_enabled",
    "require_experimental",
]
