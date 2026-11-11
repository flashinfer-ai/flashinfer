# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Batch-invariant mode: deterministic, batch-invariant kernels behind ``batch_invariant=True``.

The package holds the engine of the batch-invariant kernels (the canonical reduction plan, the floating-point interface
``fp_interface`` in its symbolic and device backends, the term language and its fingerprints) and the Lean project that
proves the engine's reduction orders; each kernel class lives next to its FlashInfer operator (``flashinfer/gemm/batch_invariant``) or, for the
operators without a package of their own, in a subpackage here (``quantization``, ``sampling``). ``README.md``
next to this file is the contract: the guarantee (R1-R4), the flag semantics, the tests and how to check the proofs.
"""

from __future__ import annotations

from typing import Callable, Dict, NamedTuple, Union

import torch

from .support import (
    BatchInvariantUnsupportedError,
    check_device,
    is_supported_device,
)


class ClassInfo(NamedTuple):
    """A consistency class behind a FlashInfer API: its name, the fixed dimensions its fingerprint takes, the
    function computing the fingerprint from them, and the optional dimensions (flags of the specification with a
    default, such as a bias) the function also accepts."""

    name: str
    dims: tuple
    fingerprint: Callable[..., str]
    optional: tuple = ()


def _mm_bf16_fingerprint(N: int, K: int, bias: bool = False) -> str:
    from ..gemm.batch_invariant.schedule import chunks_for
    from ..gemm.batch_invariant.spec import spec_fingerprint

    return spec_fingerprint(K, "bf16", chunks_for(N, K), bias=bool(bias))


def _fp8_fmt(dtype: Union[torch.dtype, str]) -> str:
    """The fp8 class of an input dtype: ``torch.float8_e4m3fn`` / ``"e4m3"`` -> ``"e4m3"``, likewise e5m2."""
    names = {
        torch.float8_e4m3fn: "e4m3",
        torch.float8_e5m2: "e5m2",
        "e4m3": "e4m3",
        "e5m2": "e5m2",
    }
    if dtype not in names:
        raise ValueError(
            f"fingerprint('bmm_fp8') takes dtype torch.float8_e4m3fn or torch.float8_e5m2 (or 'e4m3' / 'e5m2'), not {dtype!r}"
        )
    return names[dtype]


def _out_fmt(dtype: Union[torch.dtype, str], op: str = "bmm_fp8") -> str:
    names = {
        torch.bfloat16: "bf16",
        torch.float16: "f16",
        "bf16": "bf16",
        "f16": "f16",
        "fp16": "f16",
    }
    if dtype not in names:
        raise ValueError(
            f"fingerprint({op!r}) takes out_dtype torch.bfloat16 or torch.float16 (or 'bf16' / 'f16'), not {dtype!r}"
        )
    return names[dtype]


def _bmm_fp8_fingerprint(
    N: int, K: int, dtype: Union[torch.dtype, str], out_dtype: Union[torch.dtype, str]
) -> str:
    from ..gemm.batch_invariant.schedule import chunks_for
    from ..gemm.batch_invariant.spec import spec_fingerprint

    fmt = _fp8_fmt(dtype)
    return spec_fingerprint(K, _out_fmt(out_dtype), chunks_for(N, K, fmt), fmt)


def _gemm_fp8_nt_groupwise_fingerprint(
    K: int, out_dtype: Union[torch.dtype, str]
) -> str:
    from ..gemm.batch_invariant.spec import spec_fingerprint

    return spec_fingerprint(
        K, _out_fmt(out_dtype, "gemm_fp8_nt_groupwise"), 1, "e4m3_bs"
    )


def _attention_fingerprint(head_dim: int) -> str:
    from ..attention.batch_invariant.spec import spec_fingerprint

    return spec_fingerprint(head_dim)


def _mx_family(arch, op: str = "mm_mxfp8") -> str:
    """The architecture family behind ``arch``: a family name (``"sm_100"`` / ``"sm_103"`` / ``"sm_107"``), a compute
    capability such as ``(10, 7)``, a ``torch.device``, or None for the current CUDA device."""
    from ..gemm.batch_invariant.heuristic import PART_TABLE, family_of

    if arch is None:
        if not torch.cuda.is_available():
            raise ValueError(
                f"fingerprint({op!r}) needs arch= (an architecture family 'sm_100' / 'sm_103' / 'sm_107', a compute "
                "capability or a device) when no CUDA device is available: the block-scaled class is the architecture's"
            )
        arch = torch.device("cuda", torch.cuda.current_device())
    if isinstance(arch, torch.device):
        from .support import compute_capability

        return family_of(compute_capability(arch))
    if isinstance(arch, str):
        if arch not in PART_TABLE:
            raise ValueError(
                f"fingerprint({op!r}) takes arch as one of {sorted(PART_TABLE)}, a compute capability or a device, not {arch!r}"
            )
        return arch
    return family_of(tuple(arch))


def _mm_mxfp8_fingerprint(
    N: int, K: int, out_dtype: Union[torch.dtype, str], arch=None
) -> str:
    from ..gemm.batch_invariant.schedule import chunks_for, mx_class_for
    from ..gemm.batch_invariant.spec import spec_fingerprint

    fmt = mx_class_for("mxfp8", _mx_family(arch))
    return spec_fingerprint(
        K, _out_fmt(out_dtype, "mm_mxfp8"), chunks_for(N, K, fmt), fmt
    )


def _mm_fp4_fingerprint(
    N: int,
    K: int,
    out_dtype: Union[torch.dtype, str],
    use_nvfp4: bool = True,
    arch=None,
) -> str:
    from ..gemm.batch_invariant.schedule import chunks_for, fp4_class_for, mx_class_for
    from ..gemm.batch_invariant.spec import spec_fingerprint

    family = _mx_family(arch, "mm_fp4")
    fmt = fp4_class_for(N, K, family) if use_nvfp4 else mx_class_for("mxfp4", family)
    return spec_fingerprint(
        K, _out_fmt(out_dtype, "mm_fp4"), chunks_for(N, K, fmt), fmt
    )


#: The FlashInfer APIs with a batch-invariant class, by API name.
CLASSES: Dict[str, ClassInfo] = {
    "mm_bf16": ClassInfo("gemm.bf16", ("N", "K"), _mm_bf16_fingerprint, ("bias",)),
    # the fp8 input dtype is a dimension: e4m3 and e5m2 are two classes (gemm.e4m3.*, gemm.e5m2.*) of one API; the output
    # dtype (bf16 / fp16) is the chain's conversion; the batch is not a dimension (every element is one launch of the class)
    "bmm_fp8": ClassInfo(
        "gemm.fp8", ("N", "K", "dtype", "out_dtype"), _bmm_fp8_fingerprint
    ),
    # the block-scale fp8 class (DeepSeek-style 1x128 / 128x128 fp32 scales applied once per 128-element K block): K and the
    # output dtype are its dimensions (N enters neither the fold nor a chunk rule: the class has no split variant)
    "gemm_fp8_nt_groupwise": ClassInfo(
        "gemm.e4m3.kblock128", ("K", "out_dtype"), _gemm_fp8_nt_groupwise_fingerprint
    ),
    # the architecture is a dimension of the MXFP8 class: the sm_107 instruction folds 64 elements per step, the sm_100 /
    # sm_103 one 32 (gemm.mxfp8.*64 vs gemm.mxfp8.*); `arch` defaults to the current CUDA device's
    "mm_mxfp8": ClassInfo(
        "gemm.mxfp8", ("N", "K", "out_dtype"), _mm_mxfp8_fingerprint, ("arch",)
    ),
    # the 4-bit classes: `use_nvfp4` picks NVFP4 (e4m3 scales per 16, gemm.nvfp4.*) or MXFP4 (UE8M0 per 32, gemm.mxfp4.*);
    # the architecture is a dimension as for mm_mxfp8 (the sm_107 instruction folds 128 elements per step: on SM107 the
    # MXFP4 class is gemm.mxfp4.*128, the NVFP4 class gemm.nvfp4.*128 for the weight shapes its chunk rule leaves unsplit
    # and gemm.nvfp4.* otherwise); the caller's alpha is the chain's `mul.rn`, its value an input
    "mm_fp4": ClassInfo(
        "gemm.fp4",
        ("N", "K", "out_dtype"),
        _mm_fp4_fingerprint,
        ("use_nvfp4", "arch"),
    ),
    # the three attention wrappers compute one class per head dim: a token's output depends on its KV position, its head and the
    # window alone, so prefill and decode (and the paged and ragged KV layouts) are one class
    "BatchDecodeWithPagedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
    "BatchPrefillWithPagedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
    "BatchPrefillWithRaggedKVCacheWrapper": ClassInfo(
        "attention.bf16", ("head_dim",), _attention_fingerprint
    ),
}


def fingerprint(op: str, **dims) -> str:
    """The receipt of a batch-invariant output: the fingerprint of the consistency class the FlashInfer API ``op``
    computes under ``batch_invariant=True`` for the fixed dimensions ``dims``.

    The guarantee is scoped to ``(fingerprint, FlashInfer version, architecture)``: two outputs with the same
    fingerprint, the same inputs and the same environment are the same bits, whatever the batch, the configuration
    the dispatcher picked or the API path. A fingerprint change is a release-note item.

    ``mm_bf16`` takes the weight shape ``N, K``: the K steps of the fold and the chunk count (a function of the
    weight shape alone) are the specification; M, the configuration and the weight's layout do not enter it. The
    optional ``bias=True`` names the biased specification (the bias added once in fp32 to the reduced accumulator
    before the single rounding); without it, the unbiased one.
    ``bmm_fp8`` takes ``N, K``, the fp8 input ``dtype`` (``torch.float8_e4m3fn`` or ``torch.float8_e5m2``) and the
    output ``out_dtype`` (bf16 / fp16): the 32-wide fp8 step, the chain's ``mul.rn`` by the device product of the two
    per-tensor scales and its conversion are the specification; the scales' values, like the operands, are inputs, and
    the batch is not a dimension (every element is one launch of the class).
    ``gemm_fp8_nt_groupwise`` takes ``K`` and ``out_dtype`` (bf16 / fp16): the block-scale class ``gemm.e4m3.kblock128``
    (e4m3 operands, fp32 scales per 1x128 block of A and per 128x128 block of B) folds each 128-element K block with the
    32-wide fp8 step from +0 and the block partials in block order with ``fma.rn(mul.rn(s_a, s_b), P, acc)``; the scales'
    values, like the operands, are inputs, and N enters nothing (no split variant).

    ``mm_mxfp8`` takes ``N, K`` and ``out_dtype`` (bf16 / fp16), and optionally ``arch`` (an architecture family
    ``"sm_100"`` / ``"sm_103"`` / ``"sm_107"``, a compute capability or a device; the current CUDA device by default):
    the block-scaled step of that architecture's instruction (32 e4m3 elements with one UE8M0 scale per row on SM100 /
    SM103, 64 with two on SM107: two classes, ``gemm.mxfp8.*`` and ``gemm.mxfp8.*64``), the chunk count of the weight
    shape and the conversion are the specification; the block scales, like the operands, are inputs.
    ``mm_fp4`` takes ``N, K`` and ``out_dtype``, and optionally ``use_nvfp4`` (True by default: the NVFP4 classes with
    one e4m3 scale per 16 elements, ``gemm.nvfp4.*``; False: the MXFP4 classes with one UE8M0 scale per 32,
    ``gemm.mxfp4.*``) and ``arch`` as for ``mm_mxfp8``: the SM107 instruction folds 128 elements per step
    (``gemm.mxfp4.*128`` on SM107; ``gemm.nvfp4.*128`` on SM107 for the weight shapes its chunk rule leaves unsplit,
    ``gemm.nvfp4.*`` otherwise), the chunk count, the chain's ``mul.rn`` by the caller's alpha and the conversion are
    the specification; the alpha's value and the block scales are inputs.

    The attention wrappers take ``head_dim``: the tile and chunk grid, the online-softmax step and the chunk merge are
    functions of the head dim alone; the batch, the sequence lengths, the page size, the KV layout and the configuration
    do not enter it, and ``BatchPrefillWithPagedKVCacheWrapper``, ``BatchDecodeWithPagedKVCacheWrapper`` and
    ``BatchPrefillWithRaggedKVCacheWrapper`` share the fingerprint (their outputs for one token are the same bits). The
    sliding window (``window_left``) is a per-call input of the class, like the KV position: the receipt is the digest of the
    class's no-window sample row and covers windowed calls too.

    >>> fingerprint("mm_bf16", N=4096, K=4096)  # doctest: +SKIP
    'gemm-bf16-kfold-...'
    >>> fingerprint("mm_bf16", N=4096, K=4096, bias=True)  # doctest: +SKIP
    'gemm-bf16-kfold-bias-...'
    >>> fingerprint("bmm_fp8", N=4096, K=4096, dtype=torch.float8_e4m3fn, out_dtype=torch.bfloat16)  # doctest: +SKIP
    'gemm-e4m3-kfold-scaled-...'
    >>> fingerprint("gemm_fp8_nt_groupwise", K=7168, out_dtype=torch.bfloat16)  # doctest: +SKIP
    'gemm-e4m3-kblock128-...'
    >>> fingerprint("mm_mxfp8", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_107")  # doctest: +SKIP
    'gemm-mxfp8-k64-kfold-...'
    >>> fingerprint("mm_fp4", N=4096, K=4096, out_dtype=torch.bfloat16, arch="sm_100")  # doctest: +SKIP
    'gemm-nvfp4-kfold-...'
    """
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    unknown = sorted(set(dims) - set(info.dims) - set(info.optional))
    if unknown:
        raise TypeError(
            f"fingerprint({op!r}) takes the fixed dimensions {list(info.dims)} and the optional {list(info.optional)}, not {unknown}"
        )
    missing = [d for d in info.dims if d not in dims]
    if missing:
        raise TypeError(f"fingerprint({op!r}) needs the fixed dimensions {missing}")
    return info.fingerprint(**dims)


def class_name(op: str) -> str:
    """The consistency class (family) name behind the API ``op`` (``"gemm.bf16"``; ``"gemm.fp8"`` for the two fp8 classes
    ``gemm.e4m3`` / ``gemm.e5m2`` behind ``bmm_fp8``; ``"gemm.e4m3.kblock128"`` behind ``gemm_fp8_nt_groupwise``;
    ``"gemm.mxfp8"`` for the two MXFP8 classes behind ``mm_mxfp8``;
    ``"gemm.fp4"`` for the NVFP4 and MXFP4 classes behind ``mm_fp4``)."""
    info = CLASSES.get(op)
    if info is None:
        raise ValueError(
            f"no batch-invariant class behind {op!r}; the APIs with one are {sorted(CLASSES)}"
        )
    return info.name


__all__ = [
    "CLASSES",
    "ClassInfo",
    "BatchInvariantUnsupportedError",
    "check_device",
    "class_name",
    "fingerprint",
    "is_supported_device",
]

# The quantization and sampling classes register themselves here (their modules keep the mode's registry generic).
from .quantization.classes import register as _register_quantization  # noqa: E402
from .sampling.classes import register as _register_sampling  # noqa: E402

_register_quantization(CLASSES, ClassInfo)
_register_sampling(CLASSES, ClassInfo)
