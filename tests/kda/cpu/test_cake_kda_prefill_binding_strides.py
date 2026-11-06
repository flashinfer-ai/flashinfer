"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

"""The BF16 KDA prefill bindings take g and beta at their token pitch.

A rendered binding drops the contiguity check on the raw ``g`` / ``beta``
pointer exactly when the kernel body consumes the plan's token-stride scalar
(``g_token_stride`` / ``beta_token_stride``) for that operand; the bodies that
read beta at the head-count pitch keep the contiguous contract.  CPU-only: the
proof reads the kernel sources and the rendered binding texts.
"""

import re
from string import Template

import pytest

from flashinfer.jit import cake_kda_prefill_binding, cake_kda_tf32

# (pitched, contiguous) module counts among the 46 BF16 bodies whose plan
# passes both token strides: 44 fused M128 and 2 fused M64 bodies.
_EXPECTED_COUNTS = {"g": (46, 0), "beta": (36, 10)}


def _bf16_modules_with_token_strides():
    for name in cake_kda_tf32._MODULE_ROWS:
        record = cake_kda_tf32.MODULES[name]
        plan = record["arg_plan"]
        if record["cache_name"].startswith("cake_kda_bf16_") and all(
            ["parameter", f"{operand}_token_stride"] in plan
            for operand in ("g", "beta")
        ):
            yield name, record


def _check_fragment(layout, operand):
    # The exact text the renderer emits for this layout kind and operand.
    return Template(cake_kda_prefill_binding._TEMPLATES[layout]).substitute(
        name=operand
    )


def _consumes_token_stride(kernel_text, operand):
    # The kernel signature declares the scalar; any further occurrence reads it.
    return len(re.findall(rf"\b{operand}_token_stride\b", kernel_text)) >= 2


@pytest.mark.cpu_only
@pytest.mark.parametrize("operand", ["g", "beta"])
def test_contiguity_checks_follow_the_kernel_token_stride_reads(operand):
    contiguous_check = _check_fragment("contiguous", operand)
    pitched_check = _check_fragment("pitched", operand)
    assert contiguous_check != pitched_check
    pitched = contiguous = 0
    for name, record in _bf16_modules_with_token_strides():
        kernel = cake_kda_tf32._source_path(record["sources"][0]).read_text(
            encoding="utf-8"
        )
        binding = cake_kda_tf32.render_binding(name)
        checks_contiguous = contiguous_check in binding
        checks_pitch = pitched_check in binding
        assert checks_contiguous != checks_pitch, name
        assert checks_pitch == _consumes_token_stride(kernel, operand), name
        pitched += checks_pitch
        contiguous += checks_contiguous
    assert (pitched, contiguous) == _EXPECTED_COUNTS[operand]
