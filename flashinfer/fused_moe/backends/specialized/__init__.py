# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The ``specialized`` backend of the unified MoE API
(:class:`flashinfer.fused_moe.SpecializedMoEConfig`, backend key ``"specialized"``).

One ``MoELayer`` backend for MoE kernels generated for one served configuration
each (a model's expert geometry, routing, quantization and token counts on one
GPU architecture). ``kernel.py`` defines the kernel contract and the shared kill
switch, ``registry.py`` finds the kernels (one self-contained subpackage of
``kernels/`` each), and ``runner.py`` holds the ``MoELayer`` runner that binds
the kernels serving a layer's configuration. :func:`stats` returns each
kernel's dispatch and build diagnostics.
"""

from .kernel import KILL_SWITCH_ENV
from .registry import discovery_errors, registered_kernels, stats

__all__ = ["KILL_SWITCH_ENV", "discovery_errors", "registered_kernels", "stats"]
