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
"""
MSA CuTe-DSL Kernels (internal)
===============================

Internal module containing CuTe-DSL MSA kernel implementations: the SM120/SM121
family, and ``sparse_decode_nvfp4_sm100``, which is one of the two kernels the
compute-capability 10.0/10.3 NVFP4 decode route dispatches between.
Import from ``flashinfer.msa_ops`` for the public API; wrappers import
kernel classes from the submodules directly (and lazily).
"""
