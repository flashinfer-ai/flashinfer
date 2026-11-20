from functools import cache
import hashlib
import json

import jinja2

from . import env
from .core import gen_jit_spec
from .utils import write_if_different


@cache
def gen_colonels_module(variant, parameters, activation, arch):
    """Compile one tile family; K, N and Blackwell pipeline depth remain dynamic."""
    if activation not in (0, 1, 2):
        raise ValueError("Unknown Colonels activation")
    if variant.startswith("w8a8") and activation != 0:
        raise ValueError("Colonels FP8 supports only SwiGLU")
    config = dict(parameters)
    if arch not in (90, 103):
        raise ValueError(f"Colonels does not support SM{arch}")
    if variant.startswith("bf16_sm100") != (arch == 103) and variant != "bf16_gemv":
        raise ValueError(f"{variant} does not support SM{arch}")
    template_dir = env.FLASHINFER_CSRC_DIR / "colonels"
    kernels = env.FLASHINFER_INCLUDE_DIR / "flashinfer/colonels"
    digest = hashlib.sha256()
    for path in sorted(
        (*template_dir.glob("*.jinja"), *kernels.glob("*.h"), *kernels.glob("*.cuh"))
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps((variant, parameters, activation, arch)).encode())
    uri = f"colonels_{variant}_{digest.hexdigest()[:20]}"
    directory = env.FLASHINFER_GEN_SRC_DIR / uri
    directory.mkdir(parents=True, exist_ok=True)
    template = jinja2.Environment(
        loader=jinja2.FileSystemLoader(template_dir), undefined=jinja2.StrictUndefined
    ).get_template(f"{variant}.jinja")
    config["ACTIVATION"] = ("SWIGLU", "RELU_SQ", "RELU_CLAMP_SQ")[activation]
    source = template.render(
        variant=variant,
        arch=arch,
        config=config,
        fp8=variant.startswith("w8a8"),
        ratio=2 if activation == 0 else 1,
        gmem="gmem" in variant,
        completion_supported=variant in ("bf16_persistent", "w8a8_persistent"),
    )
    path = directory / "kernel.cu"
    write_if_different(path, source)
    return gen_jit_spec(
        uri,
        [path],
        extra_include_paths=[kernels],
        extra_cuda_cflags=[
            "-std=c++20",
            f"-gencode=arch=compute_{arch}a,code=sm_{arch}a",
        ],
    )


def gen_colonels_aot_modules(arch):
    """Prebuild the complete configuration catalogue for the architecture."""
    from ..fused_moe.backends.colonels.tactics import get_colonels_tactics

    modules = {}
    for fp8 in (False, True) if arch == 90 else (False,):
        for tactic in get_colonels_tactics(arch, fp8=fp8):
            for activation in (0,) if fp8 else (0, 1, 2):
                spec = gen_colonels_module(
                    tactic.variant, tactic.compilation_parameters(), activation, arch
                )
                modules[
                    (tactic.variant, tactic.compilation_parameters(), activation)
                ] = spec
    return list(modules.values())
