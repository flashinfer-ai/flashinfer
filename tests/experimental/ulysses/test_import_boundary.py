# SPDX-License-Identifier: Apache-2.0
"""Stable imports and API discovery must not load optional kernel dependencies."""

import subprocess
import sys


def test_lazy_kernel_imports():
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import flashinfer; "
            "import flashinfer.comm.ulysses_experimental as api; "
            "assert api.prepare_ulysses_distributed_fa4.is_experimental; "
            "assert api.pack_ulysses_qkv_fp8.is_experimental; "
            "assert 'flashinfer.experimental.ulysses.sm100_kernel' not in sys.modules; "
            "assert 'flashinfer.experimental.ulysses.fp8' not in sys.modules; "
            "assert 'flash_attn.cute.flash_fwd_sm100_distributed_qo' not in sys.modules",
        ],
        check=True,
        timeout=60,
    )
