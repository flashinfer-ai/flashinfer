# Updating the SM120 swap-AB sources

```text
sm120_swapab/
├── common/
├── src/                    # original communication package name
├── moe_sm120_mxfp8_swapab/
├── moe_mxfp8_glu/
└── moe_nvfp4_swapab/
```

The adapter lives at `flashinfer/moe_ep/shims/sm120/swapab_cutedsl/`.
Backends import its package exports. The adapter owns path setup, communication,
workspace allocation and launch binding; `kernel_helpers.py` lazily exposes
provider helpers and references. This legacy drop shares raw module names with
SM90 and SM100, so those implementations require separate processes.

Replace the five source packages from the pinned release-prepared branch drop,
then reapply the changes recorded in VENDOR.md. Preserve the adapters and
vendoring records. Do not format generated sources or copy provider test and
benchmark scaffolding.

Check `Sm120MegaMoEMxfp8SwapABKernel.__init__` and `.__call__` against the
adapter's constructor and launch arguments. Compare the provider's
`moe_sm120_mxfp8_swapab/mega_runner.py` for workspace allocation, combine-output
shape, the separate top-k reducer and local-counter reset. Keep the reset until
the selected kernel revision provides the corresponding tail reset.

Check `_MXFP8_BLOCK_SIZE`, `_SF_PADDING_BLOCK`, `_CTA_TOKEN_TILE` and
`_SWAP_AB_INTERLEAVE` in the adapter. `_assert_mirrored_constants()` catches
value drift at compile time; renamed symbols still need an import update.
Verify the reference and quantizer imports in `kernel_helpers.py`, including
their dependencies on `common`, `src` and the two helper kernel packages.

Run host import/configuration checks, then the numerical and graph cases on
SM120:

```bash
torchrun --standalone --nproc_per_node=4 -m pytest \
    tests/moe_ep/test_moe_ep_sm120_mxfp8_cutedsl_mega_multirank.py -x -v
```

Backend dispatch remains under
`backends/mega/kernel/sm120/mxfp8_mxfp8_bf16_cutedsl/` and is maintained
separately from the provider snapshot.
