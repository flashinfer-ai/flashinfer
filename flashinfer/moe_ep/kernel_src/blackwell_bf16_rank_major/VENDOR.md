# Blackwell BF16 rank-major source drop

This directory contains one generated CUDA source bundle for the capacity-
bounded SM100 BF16 rank-major MoE-EP layer and the FlashInfer runtime adapter
that owns its workspace and launch sequence. Each launch accepts an active
prefix of 1 to 128 rows per rank; every rank must use the same active row count.

The `src/` directory is an immutable generated pair:

- `flashinfer_blackwell_moe_ep_layer_sm100.cu` contains the eleven device
  kernels in launch order.
- `manifest.json` pins the source checksum, capacity/workload constraints,
  symbols, launch geometry, cluster geometry, dynamic shared memory, and PDL
  policy.

Do not edit either generated file independently. Updates must replace the
source and manifest together, preserve the public ABI, pass the
eight-rank correctness gates, and show paired no-regression measurements
against the previous source bundle.
