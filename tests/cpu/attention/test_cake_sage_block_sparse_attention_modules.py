"""CPU checks of the SM120 Sage FP8 block-sparse attention module table.

No GPU and no kernel build: only the JIT loader is imported, and the tests check
the pure mapping from a specialization key to its module name and to the compile
definitions that select the key's text in the shared kernel source.
"""

import pytest

from flashinfer.jit.cake_sage_block_sparse_attention import (
    SM120_MODULES,
    sm120_compile_definitions,
)

# Flag macros of the shared kernel source, in key order.
_FLAG_MACROS = (
    "CAKE_SAGE_HAS_BLOCK_NUMS",
    "CAKE_SAGE_BLOCK_SIZES_MODE",
    "CAKE_SAGE_FULL_K64_TILES",
    "CAKE_SAGE_UNIFORM_NONEMPTY",
    "CAKE_SAGE_CONTIGUOUS_BLOCK_INDICES",
)
_KEYS = sorted(SM120_MODULES)


def _parse_definitions(definitions):
    parsed = {}
    for definition in definitions:
        assert definition.startswith("-D")
        macro, separator, value = definition[2:].partition("=")
        assert separator == "="
        assert macro not in parsed
        parsed[macro] = value
    return parsed


@pytest.mark.cpu_only
def test_sm120_module_names_are_distinct():
    # One JIT module (and so one build directory) per specialization.
    names = list(SM120_MODULES.values())
    assert len(set(names)) == len(names)
    assert all(name.startswith("cake_sage_block_sparse_attention_") for name in names)
    assert not any(name.endswith(".cu") for name in names)


@pytest.mark.cpu_only
@pytest.mark.parametrize("key", _KEYS, ids=["_".join(map(str, key)) for key in _KEYS])
def test_sm120_compile_definitions_encode_the_key(key):
    definitions = sm120_compile_definitions(key)
    assert definitions[0] == f"-DCAKE_SAGE_SM120_KERNEL=kernel_{SM120_MODULES[key]}"
    parsed = _parse_definitions(definitions)
    del parsed["CAKE_SAGE_SM120_KERNEL"]
    assert parsed == {
        macro: str(value) for macro, value in zip(_FLAG_MACROS, key, strict=True)
    }


@pytest.mark.cpu_only
def test_sm120_contiguous_rows_are_uniform():
    # Contiguous block indices are only specialized together with a uniform
    # non-empty block count; the dispatcher clears the flag otherwise.
    assert all(key[3] == 1 for key in _KEYS if key[4] == 1)


@pytest.mark.cpu_only
def test_flat_offset_guard_refuses_tensors_beyond_int32():
    """The kernels index tensors with 32-bit element offsets; the host refuses larger ones before any launch."""
    import torch

    from flashinfer.cake_sage_block_sparse_attention_backend import (
        MAX_FLAT_ELEMENTS,
        _check_flat_indexable,
        _check_tensor,
    )

    assert MAX_FLAT_ELEMENTS == 2**31 - 1
    rows = (
        MAX_FLAT_ELEMENTS // 128 + 1
    )  # 16,777,216 rows of 128 channels: 2**31 elements, one past the limit
    meta = torch.device("meta")
    too_big = torch.empty((1, rows, 1, 128), device=meta, dtype=torch.float8_e4m3fn)
    assert too_big.numel() == MAX_FLAT_ELEMENTS + 1
    with pytest.raises(
        ValueError, match=r"q has 2147483648 elements \(shape \(1, 16777216, 1, 128\)\)"
    ):
        _check_flat_indexable("q", too_big)
    with pytest.raises(ValueError, match="32-bit element offsets"):
        _check_tensor("q", too_big, meta, torch.float8_e4m3fn)
    largest = torch.empty((1, rows - 1, 1, 128), device=meta, dtype=torch.float8_e4m3fn)
    assert largest.numel() == MAX_FLAT_ELEMENTS - 127
    _check_flat_indexable("q", largest)
    _check_tensor("q", largest, meta, torch.float8_e4m3fn, (1, rows - 1, 1, 128))
    _check_flat_indexable(
        "out", torch.empty((2, 3, 4, 128), device=meta, dtype=torch.bfloat16)
    )
