# Copyright (c) 2025 by FlashInfer team.
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

"""Top-K decode kernels for FlashInfer.

The CuTe-DSL kernel source (GVR and radix Top-K for Blackwell sm_100+) lives in
``flashinfer.topk_varlen.kernels``.  The public ``top_k_varlen`` and
``top_k_varlen_page_table_transform`` APIs are defined in
``flashinfer.topk_varlen.topk_varlen`` and re-exported from the top-level
``flashinfer`` namespace.
"""
