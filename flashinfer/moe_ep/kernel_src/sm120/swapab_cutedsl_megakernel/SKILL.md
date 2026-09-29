# Updating the SM120 swap-AB CuTeDSL MegaMoE kernel src

## Layout

```
kernel_src/sm120/swapab_cutedsl_megakernel/
├── src/                    ← VERBATIM kernel-team drop; NEVER edit or add files here
│   ├── common/
│   ├── src/                ← CuTeDSL core src (bootstrap, sym_buffer, token_comm, …)
│   ├── moe_sm120_mxfp8_swapab/  ← the SM120 MXFP8 swap-AB kernel
│   ├── moe_mxfp8_glu/      ← generic MXFP8 torch reference (+ its runner helpers)
│   └── moe_nvfp4_swapab/   ← import dependency of moe_mxfp8_glu's reference
├── __init__.py             ← public API for moe_ep; talks ONLY to shim/ (our code)
├── shim/                   ← thin adapters over src/ (our code) — ALL adaptation here
│   ├── _paths.py           ← adds sibling src/ to sys.path; sibling-tree guard
│   ├── comm.py             ← dist bootstrap, sym heap, compile state,
│   │                          zero_local_counter_regions (SM120 pre-launch contract)
│   ├── mxfp8.py            ← SM120 MXFP8 frontend + symm-buffer/launch wrappers
│   └── kernel_helpers.py   ← SINGLE (all-lazy) re-export point for raw-kernel
│                              helpers/constants/reference (drop-audit point)
├── VENDOR.md               ← provenance + sync state of the current src/ drop
└── SKILL.md                ← this file (drop-update workflow)
```

Core principle: **`src/` is a verbatim copy of the kernel-team drop — no
injected files, no edits.** Every adaptation lives in `shim/`. A new drop is a
pure replace of `src/`; the only work is updating `shim/` to whatever the new
`src/` exposes.

Layer isolation (enforce on every drop — grep before/after): `shim/` is the
**only** layer importing `src/` packages; FI backends
(`backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/`) import ONLY the
package `__init__`; the tree is process-exclusive with the SM100/SM90 trees
(shared top-level module names — `shim/_paths.py` raises on collision).

## When the kernel team drops a new version of src/

1. **Replace `src/` verbatim** with the drop's five kernel packages:
   ```bash
   cd flashinfer/moe_ep/kernel_src/sm120/swapab_cutedsl_megakernel/src
   rm -rf common src moe_sm120_mxfp8_swapab moe_mxfp8_glu moe_nvfp4_swapab
   cp -r <new_drop>/{common,src,moe_sm120_mxfp8_swapab,moe_mxfp8_glu,moe_nvfp4_swapab} .
   ```
   Do NOT copy repo scaffolding (`ci/`, `tester/`, `tests/`, `scripts/`,
   `pyproject.toml`, `dispatch_test.py`, `README.md`, `core.python.*`).
   Update VENDOR.md (commit SHA, date, pending local diffs).

2. **Audit the kernel construct + launch signatures FIRST** — the highest-churn
   surface. `shim/mxfp8.py` `_ensure_mega_compiled` (constructor) and
   `_build_mega_runtime_kwargs` (launch kwargs) must match
   `Sm120MegaMoEMxfp8SwapABKernel.__init__` / `.__call__`
   (`src/moe_sm120_mxfp8_swapab/megamoe_kernel.py`). The authoritative driver
   to mirror is `src/moe_sm120_mxfp8_swapab/mega_runner.py` (`run_kernel` +
   `generate_inputs`): workspace allocation, combine_output shape, the
   topk-reduce second launch, and `_reset_local_counters` (the pre-launch
   counter-zero contract — check whether the new drop made the kernel
   tail-clean, which would let `zero_local_counter_regions` go away).

3. **Re-verify the mirrored ABI constants** in `shim/mxfp8.py`
   (`_MXFP8_BLOCK_SIZE`, `_SF_PADDING_BLOCK`, `_CTA_TOKEN_TILE`,
   `_SWAP_AB_INTERLEAVE`) — `_assert_mirrored_constants()` catches value
   drift at first compile, but renames/moves need a manual fix here.

4. **Audit shim + kernel_helpers imports** against the new drop:

   | Shim import | Kernel src file |
   |---|---|
   | `from moe_sm120_mxfp8_swapab.megamoe_kernel import Sm120MegaMoEMxfp8SwapABKernel` | `src/moe_sm120_mxfp8_swapab/megamoe_kernel.py` |
   | `from moe_sm120_mxfp8_swapab.topk_reduce import compile_topk_reduce, _to_cute_tensor` | `src/moe_sm120_mxfp8_swapab/topk_reduce.py` |
   | `from moe_sm120_mxfp8_swapab.sm120_mma import CTA_TOKEN_TILE, SWAP_AB_INTERLEAVE` | `src/moe_sm120_mxfp8_swapab/sm120_mma.py` |
   | `from common.megamoe_constants import Mxfp8BlockSize, SfPaddingBlock` | `src/common/megamoe_constants.py` |
   | `from common.host_utils import kind_data_dtype, mxfp8_quantize_per_block_32_row` | `src/common/host_utils.py` |
   | `from moe_sm120_mxfp8_swapab.runner_common import Mxfp8ScaleDtype, ceil_div, round_up, to_blocked, _stack_byte_reinterpretable_tensors` | `src/moe_sm120_mxfp8_swapab/runner_common.py` |
   | `from moe_sm120_mxfp8_swapab.mega_runner import _make_fp8_tensor, _make_e8m0_scale_tensor` (lazy) | `src/moe_sm120_mxfp8_swapab/mega_runner.py` |
   | `from moe_sm120_mxfp8_swapab.mega_reference import compute_megamoe_reference_mxfp8` (lazy) | `src/moe_sm120_mxfp8_swapab/mega_reference.py` |
   | `from src.sym_buffer import SymBufferHost` | `src/src/sym_buffer.py` |
   | `from src.bootstrap import init_dist_and_nvshmem, finalize_dist_and_nvshmem` | `src/src/bootstrap.py` |

5. **Run the sm120 tests** (requires torchrun + 4 SM120 GPUs):
   ```bash
   torchrun --standalone --nproc_per_node=4 -m pytest \
       tests/moe_ep/test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py -x -v
   ```

## What NOT to update here

- `__init__.py` / `shim/` — our adapter layer; keep the public surface stable
  across kernel drops.
- `backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/` — the FI backend
  wrapper; imports the package `__init__` only, not part of this drop.
