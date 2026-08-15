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

"""CuteDSL backend for the SM120 FP8 groupwise MoE GEMM, selected via backend="cutedsl"."""

from .runner import launch_cutedsl_fp8_moe as launch_cutedsl_fp8_moe

__all__ = ["launch_cutedsl_fp8_moe"]
