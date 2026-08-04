# Updating the CuTeDSL MegaMoE kernel src

## Layout

```
kernel_src/cutedsl_megamoe/
├── src/                    ← VERBATIM kernel-team drop; NEVER edit or add files here
│   ├── common/
│   ├── src/                ← CuTeDSL core src (bootstrap, dispatch, sym_buffer, …)
│   ├── moe_mxfp8_glu/      ← MXFP8 kernel implementation
│   └── moe_nvfp4_swapab/   ← NVFP4 kernel implementation
├── __init__.py             ← public API for moe_ep; talks ONLY to shim/ (our code)
├── shim/                   ← thin adapters over src/ (our code) — ALL adaptation lives here
│   ├── _paths.py           ← adds sibling src/ to sys.path (bootstrap_paths); shim glue
│   ├── comm.py             ← dist bootstrap, sym heap, compile state, resolve_gate_up_clamp
│   ├── nvfp4.py            ← NVFP4 frontend + symm-buffer/launch wrappers (self-contained)
│   ├── mxfp8.py            ← MXFP8 frontend + symm-buffer/launch wrappers (self-contained)
│   ├── kernel_helpers.py   ← SINGLE re-export point for raw-kernel helpers/constants/
│   │                          reference the FI backend + tests need (drop-audit point)
│   ├── tuner.py            ← kernel tuning knobs (tactic enumeration + config apply);
│   │                          mirrors tester/solvers/inference_solver knob taxonomy
│   ├── autotune.py         ← online (warmup-time) COLLECTIVE knob autotuning:
│   │                          times a curated candidate set on the live problem,
│   │                          all-reduces (MAX) across ranks, applies the winner
│   │                          (backends trigger it via config knobs="auto")
│   └── correctness.py      ← standalone NVFP4 smoke runner (not used by moe_ep)
├── SKILL.md                ← this file (drop-update workflow)
└── TUNING.md               ← tuning surface + measured profiles + benchmarking
                               methodology (read before re-tuning or comparing
                               against deep_gemm / the kernel-repo tester)
```

Core principle: **`src/` is a verbatim copy of the kernel-team drop — no injected
files, no edits.** Every adaptation (path bootstrap, symbol re-exports, API
shims) lives in `shim/`. A new drop is a pure replace of `src/`; the only work
is updating `shim/` to whatever the new `src/` exposes.

Layering: `moe_ep` backends import from the package (`__init__.py`) only →
`__init__.py` re-exports from `shim/` → `shim/` imports the raw kernel packages
from `src/` via sys.path (`shim/_paths.bootstrap_paths`).

Layer isolation (enforce on every drop — grep before/after):
- `shim/` is the **only** layer that imports `src/` packages (`common`,
  `moe_nvfp4_swapab`, `moe_mxfp8_glu`, `src`).
- FI backends (`backends/mega/kernel/{nvfp4,mxfp8}_cutedsl/`) import kernel
  helpers/constants/launch entry points **only** from the package `__init__`,
  never from `src/` directly.
- `modes/` talk to backends only; `core/` touches the kernel drop solely via
  `core/runtime/bootstrap.py` → package `bootstrap_paths` (from `shim/_paths.py`).
- The cutedsl tests verify **only** through the package public API (constants +
  the MXFP8 torch reference are re-exported by `kernel_helpers.py`).

`kernel_helpers.py` is where any raw helper that a backend or test needs gets
re-exported, so a drop that renames a helper breaks in ONE file. Light
constants/helpers are eager; the `mega_runner`/`mega_reference` helpers pull
`cutlass`, so they are exposed lazily (module `__getattr__` here + package-level
`__getattr__` in `__init__`) to keep importing the package CPU-safe.

## When the kernel team drops a new version of src/

1. **Replace `src/` verbatim** with the drop's four kernel packages — no injected
   files, no edits (the drop is a full repo; copy only these four dirs):
   ```bash
   rm -rf flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/{common,src,moe_mxfp8_glu,moe_nvfp4_swapab}
   cp -r <new_drop>/{common,src,moe_mxfp8_glu,moe_nvfp4_swapab} \
       flashinfer/moe_ep/kernel_src/cutedsl_megamoe/src/
   ```
   Do NOT copy the drop's repo scaffolding (`ci/`, `tester/`, `tests/`, `scripts/`,
   `.git`, `pyproject.toml`, `dispatch_test.py`, `README.md`).

2. **Path bootstrap needs nothing** — it lives in `shim/_paths.py` and points at
   the sibling `src/` dir, so a verbatim drop just works. (Ignore any bootstrap
   the drop ships inside its packages.)

3. **Audit the kernel construct + launch signatures FIRST** — the highest-churn
   surface, and one a symbol-existence grep will NOT catch (the args change, not
   the names). `shim/{nvfp4,mxfp8}.py` `_ensure_mega_compiled` (constructor) and
   `_build_mega_runtime_kwargs` (the `cute.compile` / launch kwargs) must match
   `Sm100MegaMoE{,Mxfp8}Kernel.__init__` and `.__call__`. The authoritative
   templates to mirror are the training integration's drivers:
   `moe_ep_training/megamoe/forward_nvfp4.py` and `forward.py` (kernel construct,
   `output_activation`, workspace pointer vs cute-tensor handling, `combine_format`).
   Also re-check `tuner.py`'s knob value-sets against
   `tester/solvers/inference_solver.py` (`_correctness_knobs` / `_perf_knobs` /
   `filter_invalid`).

4. **Audit shim compatibility** — `shim/nvfp4.py` and `shim/mxfp8.py` call into `common`,
   `moe_nvfp4_swapab`, `moe_mxfp8_glu`, and `src` via sys.path imports. Check these
   entrypoints after updating src/:

   | Shim import | Kernel src file |
   |---|---|
   | `from common.megamoe_constants import Nvfp4BlockSize, Mxfp8BlockSize` | `src/common/megamoe_constants.py` |
   | `from moe_nvfp4_swapab.runner_common import _DataDtype, ceil_div, …` | `src/moe_nvfp4_swapab/runner_common.py` |
   | `from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel` | `src/moe_nvfp4_swapab/megamoe_kernel.py` |
   | `from moe_nvfp4_swapab.epilogue_refactor import SwapABSwigluFp4Epilogue` | `src/moe_nvfp4_swapab/epilogue_refactor.py` |
   | `from moe_mxfp8_glu.megamoe_kernel_mxfp8 import Sm100MegaMoEMxfp8Kernel` | `src/moe_mxfp8_glu/megamoe_kernel_mxfp8.py` |
   | `from src.sym_buffer import SymBufferHost` | `src/src/sym_buffer.py` |
   | `from src.bootstrap import finalize_dist_and_nvshmem` | `src/src/bootstrap.py` |

   `shim/kernel_helpers.py` (the backend/test helper boundary) additionally
   depends on these src symbols — audit them too:

   | `kernel_helpers.py` import | Kernel src file |
   |---|---|
   | `from common.megamoe_constants import Nvfp4BlockSize, Mxfp8BlockSize` | `src/common/megamoe_constants.py` |
   | `from common.host_utils import kind_data_dtype, mxfp8_quantize_per_block_32` | `src/common/host_utils.py` |
   | `from moe_nvfp4_swapab.runner_common import Mxfp8ScaleDtype, ceil_div, round_up, to_blocked, nvfp4_quantize_per_block_16` | `src/moe_nvfp4_swapab/runner_common.py` |
   | `from moe_nvfp4_swapab.mega_runner import _stack_byte_reinterpretable_tensors` (lazy) | `src/moe_nvfp4_swapab/mega_runner.py` |
   | `from moe_mxfp8_glu.mega_runner import _make_fp8_tensor, _make_e8m0_scale_tensor` (lazy) | `src/moe_mxfp8_glu/mega_runner.py` |
   | `from moe_mxfp8_glu.mega_reference_mxfp8 import compute_megamoe_reference_mxfp8` (lazy) | `src/moe_mxfp8_glu/mega_reference_mxfp8.py` |

5. **Run the cutedsl tests** to confirm everything still works:
   ```bash
   # Blackwell-only; requires torchrun + 4+ GPUs
   torchrun --standalone --nproc_per_node=4 -m pytest \
       tests/moe_ep/test_moe_ep_nvfp4_cutedsl_mega_multirank.py \
       tests/moe_ep/test_moe_ep_mxfp8_cutedsl_mega_multirank.py \
       tests/moe_ep/test_mxfp8_cutedsl_preprocess_vs_reference.py \
       -x -v
   ```

## What NOT to update here

- `__init__.py` / `shim/` — our adapter layer. The public surface moe_ep depends on
  is `__init__.py`; keep it stable across kernel drops.
- `backends/mega/kernel/nvfp4_cutedsl/` and `mxfp8_cutedsl/` — those are the FI backend
  wrappers; they import from the package (`__init__.py`) but are not part of this drop.
- `core/runtime/bootstrap.py` — imports `bootstrap_paths` from the package
  (`shim/_paths.py`); only change if that public name moves.
