# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant KDA class ``kda.bf16.c16`` behind ``flashinfer.recurrent_kda(..., batch_invariant=True)``
(``api``): prefill, chunked prefill and decode on an fp32 state pool, every row the pinned chunked-prefill kernel's.
Light imports only: the step kernel's JIT module and the CuTe-DSL kernel are loaded on first use."""
