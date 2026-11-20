# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The batch-invariant paged attention class ``attention.bf16`` (head dims 64 / 128 / 256) behind
``BatchDecodeWithPagedKVCacheWrapper``, ``BatchPrefillWithPagedKVCacheWrapper`` and ``BatchPrefillWithRaggedKVCacheWrapper``
under ``batch_invariant=True``: the specification (``spec``), the planner (``plan``), the kernel bodies (``blocks``: the row body;
``swaps``: the decode body), the split-KV fixup (``fixup``), the engine entry points (``engine``) and the wrapper delegates
(``wrappers``). The contract (R1-R4) is ``flashinfer/batch_invariant/README.md``.
"""
