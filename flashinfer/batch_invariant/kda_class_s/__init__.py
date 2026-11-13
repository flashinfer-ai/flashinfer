# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant KDA class ``kda.bf16.s16`` behind ``flashinfer.recurrent_kda(..., batch_invariant=True)``
(``api``): one calculation for prefill, chunked prefill and decode on a bf16 state pool. Light imports only: the
kernels' JIT module is loaded on first use."""
