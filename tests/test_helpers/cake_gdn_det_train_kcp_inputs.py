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

Seeded operands of the packed GDN context-parallel (KCP) training tests:
``make_inputs(cu_seqlens, H_qk, HV, *, regime, seed, with_conv, token_range)``
-> CPU tensors ``q, k [1, T, H_qk, 128]`` BF16 (rows L2-normalised in FP32
before the store), ``v, dout [1, T, HV, 128]`` BF16, ``g, beta [1, T, HV]``
FP32, for the regimes ``normal`` and the adversarial ``A1``-``A4``.

Generation runs on the CPU with a ``torch.Generator`` seeded per (seed, heads,
document, 64-token block), so a token's operands do not depend on the
partition, the device or the host: every rank generates only its own
fragments, and a ``token_range`` slice is bitwise the slice of the global pack.

The generator is the seeded-operands section of
``cake_gdn_det_train_kcp_reference`` (the FP64 reference generates the whole
pack from the same code), so the distributed tests, the reference and the
accuracy budgets of ``tests/experimental/cake_gdn_det_train_kcp_g1_budgets.json``
see the same operands.
"""

from tests.test_helpers.cake_gdn_det_train_kcp_reference import (
    BLOCK,
    REGIMES,
    SEED,
    digests,
    doc_block,
    doc_range,
    doc_regime,
    head_params,
    make_inputs,
    tensor_sha256,
)

__all__ = [
    "BLOCK",
    "REGIMES",
    "SEED",
    "digests",
    "doc_block",
    "doc_range",
    "doc_regime",
    "head_params",
    "make_inputs",
    "tensor_sha256",
]
