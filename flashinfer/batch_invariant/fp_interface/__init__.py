# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The floating-point interface of the batch-invariant kernels, in its two backends.

* :mod:`.term`: symbolic terms that record which operations were applied in which association (fingerprints,
  canonical equivalence, the Lean cases);
* :mod:`.cute`: the CuTe-DSL primitives with explicit ``.rn`` rounding that the generated kernels are allowed to use,
  together with the dtype table they share.

Schedule code is written once against an ``fp`` object and runs under both.
"""
