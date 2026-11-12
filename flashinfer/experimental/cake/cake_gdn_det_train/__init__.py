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

# Experimental Cake backend: deterministic GDN (Gated DeltaNet) chunked
# training (head dimension 128, chunk 64) on SM100 / SM103 / SM107, with FLA's
# ``chunk_gated_delta_rule`` signature and no context parallelism.  The
# public entry point is ``cake_gdn_det``; the package exports
# the names that carry ``det`` only.  The chunk and segment tables, the scan
# planner, the launch order and the autograd chain live in ``cake_chain``
# (which also exports the planner and table helpers), the split forward's
# host piece table and the forward tape levels in ``cake_plan``, the registry
# in ``cake_jit`` and the loader in ``cake_backend``.  Every plan pin is an
# explicit keyword argument; the package reads no environment variable.

from .cake_chain import (
    CakeGDNDetFunction,
    cake_gdn_det,
    cake_gdn_det_train_backward,
    cake_gdn_det_train_forward,
)

__all__ = [
    "CakeGDNDetFunction",
    "cake_gdn_det",
    "cake_gdn_det_train_backward",
    "cake_gdn_det_train_forward",
]
