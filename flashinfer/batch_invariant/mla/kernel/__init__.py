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

"""Kernel sources of the batch-invariant sparse MLA classes ``mla.sparse.<dtype>.d<576|512>``.

A frozen copy of the part of ``flashinfer/attention/prims_ts/kernels`` (commit ``25bc267eb84dd1b28997f62aa4c282a09184c952``) that the classes' pinned
schedule reaches: the two-CTA sparse MLA decode kernel and the helpers it names. Functions, classes and constants
were kept whole or left out; no function body was edited. The classes own these files: a change to the ``prims_ts``
kernels or to their schedule policy does not reach them, and an edit here changes the class fingerprint
(``flashinfer.batch_invariant.mla.spec``) and must reproduce the recorded digests (``cases.py``).
"""
