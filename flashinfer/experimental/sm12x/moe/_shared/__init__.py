# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
"""Lowering shared by moe ops: the declarative execution model
(``execution``), fused expert kernels (``kernels``), offline tuning
registries (``tuning``), and the triton top-k router (``routing``).
Imports only _lib and itself.
"""
