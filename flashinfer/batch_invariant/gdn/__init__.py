# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GDN class ``gdn.bf16.c64`` behind ``flashinfer.chunk_gated_delta_rule(..., batch_invariant=True)``
(``api``): prefill, chunked prefill and decode on an fp32 state pool, every row the pinned chunked-prefill kernel's.
Light imports only: the CuTe-DSL prefill kernel and the decode step module are loaded on first use."""
