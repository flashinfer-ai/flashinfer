# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Small-token NVFP4 FMA specialization using the shared Frost JIT cache."""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
import tempfile

from .. import runtime as common
from ..activations import ACTIVATIONS, is_gated

_TAG = "cudnn_frost-nvfp4-fma-v1"
_ROOT = Path(__file__).parent
_KERNELS = ("fma_fc1_o2.py", "fma_fc1_o4.py", "fma_fc2.py")


def supported(tokens, hidden, intermediate, experts, topk, activation):
    """Small-token geometries with existing activation-specific TC shortlists."""
    return activation in ACTIVATIONS and (
        (
            (experts, hidden, intermediate, topk) == (64, 2048, 1408, 6)
            and 1 <= tokens <= 4
        )
        or (
            (experts, hidden, intermediate, topk) == (12, 7168, 3072, 2) and tokens == 1
        )
    )


@functools.cache
def source_digest():
    # Include imported conversion helpers and the native launch ABI. A change
    # must invalidate both persistent compilation and autotuning results.
    paths = [
        *(_ROOT / name for name in _KERNELS),
        Path(__file__),
        _ROOT.parent / "fma_activations.py",
        _ROOT.parent / "activations.py",
        _ROOT.parents[3] / "cute_dsl" / "fp4_common.py",
        _ROOT.parent / "csrc" / "moe_nvfp4.cu",
    ]
    return hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()


def tactic(activation="swiglu"):
    return (_TAG, activation, common._tactic_digest(source_digest()))


@functools.cache
def _requirements():
    return tuple(
        (path, common._digest(path))
        for path in (
            *(_ROOT / name for name in _KERNELS),
            _ROOT.parent / "fma_activations.py",
        )
    )


def check_support(activation="swiglu"):
    from ..capabilities import require_compiler

    require_compiler("sm_107a", _requirements())
    if activation in ("geglu", "gelu"):
        from cutlass import cute

        if not callable(getattr(cute.math, "erf", None)):
            raise NotImplementedError("FMA GELU/GeGLU requires cute.math.erf")


_COMPILE = """
from cutlass.cute.runtime import make_fake_stream

frost_compile_options = "--enable-tvm-ffi --gpu-arch sm_107a"


def _fake(dtype, shape, alignment=16):
    strides = []
    size = 1
    for dim in reversed(shape):
        strides.insert(0, size)
        size *= dim
    return cute.runtime.make_fake_tensor(
        dtype, shape, stride=tuple(strides), assumed_align=alignment
    )


def compile():
    t, h, i, e, k, swizzled, first = _FMA_CONFIG
    s = t * k
    if first:
        args = [
            _fake(cutlass.Uint8, (t, h // 2)),
            _fake(cutlass.Uint8, (1, 128 * h // 16) if swizzled else (t, h // 16)),
            _fake(cutlass.Uint8, (e, (2 if _GATED else 1) * i, h // 2)),
            _fake(cutlass.Uint8, ((2 if _GATED else 1), e, i, h // 16)),
            _fake(cutlass.Float32, (e,), 4),
            _fake(cutlass.Int32, (t, k), 4),
            _fake(cutlass.Uint8, (s * i // 2,)),
            _fake(cutlass.Uint8, (128 * s * i // 16,)),
            _fake(cutlass.Float32, (1,), 4),
        ]
    else:
        args = [
            _fake(cutlass.Uint8, (s * i // 2,)),
            _fake(cutlass.Uint8, (128 * s * i // 16,)),
            _fake(cutlass.Uint8, (e, h, i // 2)),
            _fake(cutlass.Uint8, (e, h, i // 16)),
            _fake(cutlass.Float32, (e,), 4),
            _fake(cutlass.Int32, (t, k), 4),
            _fake(cutlass.Float32, (t, k), 4),
            _fake(cutlass.BFloat16, (t, h)),
        ]
    return cute.compile(
        _launch, *args, make_fake_stream(use_tvm_ffi_env_stream=False),
        options=frost_compile_options,
    )
"""


@functools.cache
def _source(
    tokens, hidden, intermediate, experts, topk, swizzled, first, activation="swiglu"
):
    from .....jit.env import FLASHINFER_GEN_SRC_DIR

    name = _KERNELS[0 if tokens == 1 else 1] if first else _KERNELS[2]
    config = (tokens, hidden, intermediate, experts, topk, swizzled, first)
    activation = activation if first else "identity"
    source = (
        (_ROOT / name).read_text()
        + f"\n# FMA source identity: {source_digest()}\n_FMA_CONFIG = {config!r}\n"
        + f"_ACTIVATION = {activation!r}\n_GATED = {is_gated(activation)!r}\n"
        + _COMPILE
    )
    digest = hashlib.sha256(source.encode()).hexdigest()
    directory = FLASHINFER_GEN_SRC_DIR / "cudnn_frost_nvfp4_fma" / digest
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "kernel.py"
    if not path.is_file() or common._digest(path) != digest:
        fd, temporary = tempfile.mkstemp(dir=directory, suffix=".py.tmp")
        try:
            with os.fdopen(fd, "w") as stream:
                stream.write(source)
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path, digest


def build(
    tokens, hidden, intermediate, experts, topk, device, swizzled, activation="swiglu"
):
    arch = common._arch_for(device)
    if arch != "sm_107a" or not supported(
        tokens, hidden, intermediate, experts, topk, activation
    ):
        raise ValueError("Unsupported NVFP4 FMA geometry/architecture")
    return tuple(
        common._load_source(
            *_source(
                tokens, hidden, intermediate, experts, topk, swizzled, first, activation
            ),
            arch,
            device.index,
        )
        for first in (True, False)
    )
