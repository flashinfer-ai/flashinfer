# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CUTLASS DSL version gates shared by Prims-TS tests."""

from importlib.metadata import PackageNotFoundError, version

import pytest
from packaging.version import Version


def cutlass_dsl_before_4_8() -> bool:
    try:
        return Version(version("nvidia-cutlass-dsl")).release < (4, 8)
    except PackageNotFoundError:
        return True


# Block-scaled C output at a non-zero partition offset only runs in production
# on SM107 (partitioned MoE), which needs CUTLASS DSL 4.8+. On DSL 4.7 that
# epilogue writes wrong values.
requires_dsl_4_8_for_c_partition_offset = pytest.mark.skipif(
    cutlass_dsl_before_4_8(),
    reason="block-scaled C at a partition offset needs CUTLASS DSL 4.8+",
)
