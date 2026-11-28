"""JIT-only build; generated files live in FlashInfer's writable cache."""

import functools
from ...jit import env
from ...jit.core import gen_jit_spec
from ...jit.utils import write_if_different


@functools.cache
def get_module():
    name = "hpc_parallel_decode_sm100_tile128_bf16_gqa4_d128_v2"
    folder = env.FLASHINFER_GEN_SRC_DIR / name
    folder.mkdir(parents=True, exist_ok=True)
    source = folder / "hpc_flat_bin.cu"
    write_if_different(source, (env.FLASHINFER_CSRC_DIR / source.name).read_text())
    return gen_jit_spec(
        name, [source], extra_include_paths=[env.FLASHINFER_CSRC_DIR]
    ).build_and_load()
