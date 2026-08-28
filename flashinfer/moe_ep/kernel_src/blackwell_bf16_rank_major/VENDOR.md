# Blackwell BF16 rank-major source drop

This directory contains one generated CUDA source bundle for the fixed SM100
BF16 rank-major MoE-EP layer and the FlashInfer runtime adapter that owns its
workspace and launch sequence.

The `src/` directory is an immutable generated pair:

- `flashinfer_blackwell_moe_ep_layer_sm100.cu` contains the eleven device
  kernels in launch order.
- `manifest.json` pins the source checksum, fixed workload constraints,
  symbols, launch geometry, cluster geometry, dynamic shared memory, and PDL
  policy.

Do not edit either generated file independently. Updates must replace the
source and manifest together, preserve the fixed public ABI, pass the
eight-rank correctness gates, and show paired no-regression measurements
against the previous source bundle.
