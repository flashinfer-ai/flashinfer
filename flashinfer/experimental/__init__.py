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
# Everything here provides no compatibility guarantees and may change or be
# removed without deprecation. Using it is an explicit opt-in: calling an
# @flashinfer_experimental_api function, or naming an experimental backend
# with backend="<name>". Only *automatic* selection (backend="auto") is gated,
# behind FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS=1. Importing this module
# is always allowed (so tooling and docs can introspect it).
#
# See README.md in this directory for the full policy, and CLAUDE.md for
# agent-facing contribution rules.

from ..api_logging import (
    ExperimentalWarning,
    experimental_auto_backends_allowed,
    flashinfer_experimental_api,
    require_experimental_auto_backends,
    warn_experimental_backend_once,
)
from ..utils import experimental_backend

__all__ = [
    "ExperimentalWarning",
    "experimental_auto_backends_allowed",
    "experimental_backend",
    "flashinfer_experimental_api",
    "require_experimental_auto_backends",
    "warn_experimental_backend_once",
]
