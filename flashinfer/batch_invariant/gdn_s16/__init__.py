# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant GDN class ``gdn.bf16.s16`` behind ``flashinfer.chunk_gated_delta_rule(..., batch_invariant=True)``
(``api``): prefill, chunked prefill, decode and multi-token steps on a bf16 state pool, one term for all of them.
Light imports only: the kernel module is built and loaded on first use."""
