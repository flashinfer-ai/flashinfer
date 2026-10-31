# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The FROST backend of the Qwen 3.5 GDN block: CuTe DSL kernels for SM100 (cc 10.0).

``admission`` answers the support question without importing the kernels; ``runner`` builds a plan's runner and
imports the kernels (``gdn.FrostGdnConsumer`` for decode, ``gdn_verify.FrostGdnVerify`` for verify) on first use.
"""
