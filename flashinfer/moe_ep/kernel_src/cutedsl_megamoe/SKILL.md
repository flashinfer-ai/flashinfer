# Updating the CuTeDSL kernel src

## Layout

```
kernel_src/cutedsl/
├── src/                    ← kernel team code (treat as read-only; replace whole dir on update)
│   ├── _bootstrap_paths.py ← adds src/ to sys.path; keep as-is
│   ├── common/
│   ├── src/                ← CuTeDSL core src (bootstrap, dispatch, sym_buffer, …)
│   ├── moe_mxfp8_glu/      ← MXFP8 kernel implementation
│   └── moe_nvfp4_swapab/   ← NVFP4 kernel implementation
├── __init__.py             ← public API re-exports (our shim; do not replace)
├── _util.py                ← resolve_gate_up_clamp helper (our shim)
├── nvfp4.py                ← NVFP4 shim called by backends/mega/kernel/nvfp4_cutedsl/
├── mxfp8.py                ← MXFP8 shim called by backends/mega/kernel/mxfp8_cutedsl/
├── megamoe_frontend/       ← internal FI shim implementation (our code)
├── correctness.py          ← standalone correctness runner (not used by moe_ep)
└── SKILL.md                ← this file
```

## When the kernel team drops a new version of src/

1. **Replace `src/`** with the new drop (preserve `_bootstrap_paths.py` or diff it):
   ```bash
   rm -rf flashinfer/moe_ep/kernel_src/cutedsl/src/{common,src,moe_mxfp8_glu,moe_nvfp4_swapab}
   cp -r <new_drop>/{common,src,moe_mxfp8_glu,moe_nvfp4_swapab} \
       flashinfer/moe_ep/kernel_src/cutedsl/src/
   ```

2. **Check `_bootstrap_paths.py`** — it must add `src/` (i.e. `dirname(__file__)`) to `sys.path`.
   If the new drop ships its own bootstrap, keep ours or merge carefully.

3. **Audit shim compatibility** — the shim files (`nvfp4.py`, `mxfp8.py`, `megamoe_frontend/`) call
   into `common`, `moe_nvfp4_swapab`, `moe_mxfp8_glu`, and `src` via sys.path imports.
   Check these entrypoints after updating src/:

   | Shim import | Kernel src file |
   |---|---|
   | `from common.megamoe_constants import Nvfp4BlockSize, Mxfp8BlockSize` | `src/common/megamoe_constants.py` |
   | `from moe_nvfp4_swapab.runner_common import _DataDtype, ceil_div, …` | `src/moe_nvfp4_swapab/runner_common.py` |
   | `from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel` | `src/moe_nvfp4_swapab/megamoe_kernel.py` |
   | `from moe_nvfp4_swapab.epilogue_refactor import SwapABSwigluFp4Epilogue` | `src/moe_nvfp4_swapab/epilogue_refactor.py` |
   | `from moe_mxfp8_glu.megamoe_kernel_mxfp8 import Sm100MegaMoEMxfp8Kernel` | `src/moe_mxfp8_glu/megamoe_kernel_mxfp8.py` |
   | `from src.sym_buffer import SymBufferHost` | `src/src/sym_buffer.py` |
   | `from src.bootstrap import finalize_dist_and_nvshmem` | `src/src/bootstrap.py` |

4. **Run the cutedsl tests** to confirm everything still works:
   ```bash
   # Blackwell-only; requires torchrun + 4+ GPUs
   torchrun --standalone --nproc_per_node=4 -m pytest \
       tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
       tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py \
       tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
       -x -v
   ```

## What NOT to update here

- `backends/mega/kernel/nvfp4_cutedsl/` and `mxfp8_cutedsl/` — those are the FI backend
  wrappers; they import from this shim but are not part of it.
- `core/runtime/bootstrap.py` — imports `bootstrap_paths` from `src/_bootstrap_paths.py`;
  only change if `_bootstrap_paths.py` itself is removed or renamed.
