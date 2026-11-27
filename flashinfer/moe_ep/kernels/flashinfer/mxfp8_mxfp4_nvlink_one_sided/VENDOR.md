# Source provenance

The NVLinkOneSided MegaMoE (`sm103_mxfp8_mxfp4_bf16_nvlink_one_sided`) started
from FlashInfer's Blackwell GenPhase engine at
`f5d4db2c488d77331afa653c282d2e03c87ba517` (FlashInfer `main`, 2026-10-24).
`engine/` began as a copy of
`flashinfer/moe_ep/kernel_src/sm100/next_cutedsl_megamoe/src/blackwell_genphase_sources/`
at that revision, and `frontend/` as a copy of its shim modules
`block_scaled.py`, `comm.py`, `correctness.py`, `genphase.py` and
`kernel_helpers.py` from `.../next_cutedsl_megamoe/shim/`, with imports
rewritten to this package. The provider export carries that engine under
`kernels/cutedsl_megamoe/sources/`.

FlashInfer maintains this implementation here; it has diverged from the
GenPhase engine (MXFP4 weights, the Sentinel and Fence protocols over torch
symmetric memory, the fused shared expert, die-local weights, SiTU) and is not
refreshed from it. `comm/` holds the protocols' PTX primitives. The backend,
`backends/mega/kernel/sm103/mxfp8_mxfp4_bf16_nvlink_one_sided/`, calls
`frontend/` directly. These files do not have an independent provider export
to refresh.
