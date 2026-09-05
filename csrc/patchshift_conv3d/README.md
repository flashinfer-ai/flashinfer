# PatchShift Conv3d host integration

This directory is the host-side boundary for the PatchShift compute core.

- `tensor_maps.cuh` constructs input and packed-weight TMA descriptors and
  returns CUDA Driver errors to its caller.
- `select_policy.inl` owns measured shape thresholds and route priority. It is
  intentionally separate from kernel implementations.
- `launcher.cu` owns the single kernel-instantiating CUDA translation unit and
  the stream-correct hot launch.
- `prepare_descriptors.inl` builds pointer-dependent TensorMaps during explicit
  cold setup and is included by `launcher.cu` to avoid duplicate device symbols.
- `pack_weights.cu` converts logical `[K,C,3,3,3]` weights into the three packed
  tile layouts used by M128, M64, and M32 routes.
- `binding.cu` provides the TVM-FFI boundary and validates the public contract.

The standalone source's allocation, command-line parsing, timing, reference
convolution, JSON output, and process-terminating CUDA macros are intentionally
not imported.

The binding accepts BF16 NDHWC input/output tensors, explicit prepacked weights,
and caller-owned descriptor workspace. Weight prepacking and descriptor setup
remain separate from the graph-capturable hot launch.
