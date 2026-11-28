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

"""PrimTS attention variants, one package per feature.

Each package holds a feature's host frontend and, under ``kernel/``, its CuTe
DSL kernel. The public names are re-exported from
:mod:`flashinfer.attention.prims_ts`, which also keeps the packages importable
under their earlier paths (``flashinfer.attention.prims_ts.decode`` and so on).
"""
