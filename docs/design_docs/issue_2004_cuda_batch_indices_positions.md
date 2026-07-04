# Issue 2004: CUDA get_batch_indices_positions Implementation Plan

Status: implementation plan only. Do not treat the snippets in this document as an applied patch.

Issue: https://github.com/flashinfer-ai/flashinfer/issues/2004

## Summary

Issue #2004 asks for a CUDA version of `get_batch_indices_positions_kernel` so C++ users can call `append_paged_kv_cache` without depending on the current Triton-only helper. The existing append kernels already consume two plain index arrays:

- `batch_indices[i]`: batch row for appended token `i`
- `positions[i]`: final KV-cache position for appended token `i`

The first PR should add a framework-agnostic CUDA helper to `include/flashinfer/page.cuh` and expose a direct page-module entry point for testing, but it should not change the existing Python `flashinfer.get_batch_indices_positions()` wrapper. Replacing the Python wrapper's Triton launch with the CUDA helper should be a separate follow-up PR after the C++ path lands.

## Current State

Relevant files:

- `flashinfer/triton/page.py`: defines the current Triton `get_batch_indices_positions_kernel`.
- `flashinfer/page.py`: allocates or validates output tensors, then launches the Triton kernel.
- `include/flashinfer/page.cuh`: defines `AppendPagedKVCache` and `AppendPagedKVMlaCache` raw-pointer CUDA helpers.
- `csrc/page.cu`: wraps page helpers with TVM-FFI `TensorView` checks.
- `csrc/flashinfer_page_binding.cu`: exports `append_paged_kv_cache` and `append_paged_mla_kv_cache`.
- `flashinfer/jit/page.py`: builds the page module from `csrc/page.cu` and `csrc/flashinfer_page_binding.cu`.
- `tests/attention/test_page.py`: basic append test currently exercises the Python helper but does not assert the generated index arrays directly.

The Triton formula is:

```python
positions[offset] = offset + seq_len[batch_idx] - append_indptr[batch_idx + 1]
```

This is equivalent to:

```text
old_seq_len = seq_lens[batch_idx] - append_len
positions[batch_start + j] = old_seq_len + j
```

where `seq_lens` is the final sequence length after the append has been accounted for.

## Goals

1. Provide a C++ callable CUDA launcher for generating `batch_indices` and `positions`.
2. Keep the current Python public wrapper implementation unchanged in the first PR.
3. Add a direct page-module entry point so the CUDA helper can be tested from Python without redirecting the public API.
4. Reuse the existing page JIT module instead of adding a new module.
5. Preserve stream/device behavior through FlashInfer's existing TVM-FFI utilities.
6. Add focused tests for helper correctness, edge cases, and C++/FFI wiring.

## Non-Goals

1. Do not change `append_paged_kv_cache` or `append_paged_mla_kv_cache` signatures.
2. Do not redesign paged KV metadata.
3. Do not add CPU support. This is a CUDA helper for CUDA tensors.
4. Do not add Jinja code generation; this helper is dtype-simple metadata code.
5. Do not require a new AOT registration path; `gen_page_module()` is already included in AOT warmup.
6. Do not replace `flashinfer.get_batch_indices_positions()` or remove the Triton helper in the first PR.

## PR Phasing

### PR 1: Add the C++/CUDA Path

Scope:

- Add raw-pointer CUDA kernel and launcher in `include/flashinfer/page.cuh`.
- Add a TVM-FFI page-module wrapper for direct testing and optional advanced use.
- Export that wrapper from `csrc/flashinfer_page_binding.cu`.
- Add tests that call the new page-module entry point directly.
- Leave `flashinfer/page.py::get_batch_indices_positions()` launching the Triton kernel exactly as it does today.

Recommended FFI symbol name for PR 1:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_batch_indices_positions_cuda,
                              get_batch_indices_positions_cuda);
```

Using a distinct `*_cuda` name avoids implying that the public Python helper has already moved. The follow-up PR can either switch the public wrapper to this symbol or rename/internalize the symbol after the transition.

### PR 2: Switch the Python Helper

Scope:

- Update `flashinfer/page.py::get_batch_indices_positions()` to call the CUDA page-module helper.
- Preserve the current Python-facing behavior: keep int32 output allocation and preallocated-output validation, and normalize input index dtypes in Python before calling the int32-only FFI helper.
- Update docstring examples and validation tests to document the supported CUDA int32 metadata path.
- Leave `flashinfer/triton/page.py` in place. Any cleanup or removal of the Triton helper should be a separate maintainer-owned decision, not part of this PR.

## Proposed API Shape

### C++ Raw-Pointer API

Add to `include/flashinfer/page.cuh` near the append helpers:

```cpp
namespace flashinfer {

template <typename IdType>
cudaError_t GetBatchIndicesPositions(const IdType* append_indptr,
                                      const IdType* seq_lens,
                                      IdType* batch_indices,
                                      IdType* positions,
                                      uint32_t batch_size,
                                      uint32_t nnz,
                                      cudaStream_t stream = nullptr);

}  // namespace flashinfer
```

Rationale:

- `IdType` matches the existing append helpers' index template style.
- The Python FFI path should initially enforce `int32`, matching today's public outputs and the existing `csrc/page.cu` append wrapper.
- `batch_size` is `append_indptr.size(0) - 1`.
- `nnz` is the output length. It is mainly needed to skip zero-sized launches and to document the expected output extent.

### TVM-FFI Export

For PR 1, add a direct page-module wrapper to `csrc/page.cu` without changing the public Python wrapper:

```cpp
void get_batch_indices_positions_cuda(TensorView append_indptr,
                                      TensorView seq_lens,
                                      TensorView batch_indices,
                                      TensorView positions);
```

Export from `csrc/flashinfer_page_binding.cu`:

```cpp
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_batch_indices_positions_cuda,
                              get_batch_indices_positions_cuda);
```

The loaded page module will then expose a non-public direct entry point:

```python
get_page_module().get_batch_indices_positions_cuda(...)
```

### Python Public API

In PR 1, keep the existing public function and implementation unchanged:

```python
def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int,
    batch_indices: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    ...
```

That means PR 1 should not:

- change allocation or validation behavior in `flashinfer/page.py`
- change the Triton launch path
- remove or deprecate `flashinfer/triton/page.py`

The follow-up PR can change only the implementation while preserving the public signature:

- allocate `batch_indices` and `positions` exactly as today when not provided
- validate preallocated outputs as today
- call the page module's CUDA helper
- remove the import of `flashinfer.triton.page.get_batch_indices_positions_kernel` from this function

Recommended PR 2 compatibility policy:

- Keep the direct FFI helper strict: `append_indptr`, `seq_lens`, `batch_indices`, and `positions` must be CUDA, contiguous, same device, and `torch.int32`.
- Keep the public Python wrapper consistent with existing `flashinfer/page.py` metadata wrappers: normalize input index tensors to `torch.int32` before the FFI call, but do not silently move tensors across devices.

```python
append_indptr = append_indptr.int()
seq_lens = seq_lens.int()
```

This preserves the current public wrapper's int32 output behavior while making the int32 FFI contract explicit. Hidden CPU-to-GPU copies should be avoided in this hot metadata path, so CPU or different-device inputs should be rejected by validation instead of moved implicitly. Update the docstring example so `seq_lens` is created as a CUDA int32 tensor.

## Kernel Design

Use one CTA per batch row. Each CTA writes only the contiguous token range for that row, so no atomics or inter-block synchronization are required.

Illustrative kernel:

```cpp
template <typename IdType, int BLOCK_THREADS = 128>
__global__ void GetBatchIndicesPositionsKernel(const IdType* __restrict__ append_indptr,
                                               const IdType* __restrict__ seq_lens,
                                               IdType* __restrict__ batch_indices,
                                               IdType* __restrict__ positions) {
  const uint32_t batch_idx = blockIdx.x;
  const IdType batch_start = append_indptr[batch_idx];
  const IdType batch_end = append_indptr[batch_idx + 1];
  const IdType seq_len = seq_lens[batch_idx];

  for (IdType offset = batch_start + threadIdx.x; offset < batch_end;
       offset += BLOCK_THREADS) {
    batch_indices[offset] = static_cast<IdType>(batch_idx);
    positions[offset] = offset + seq_len - batch_end;
  }
}
```

Illustrative host launcher:

```cpp
template <typename IdType>
cudaError_t GetBatchIndicesPositions(const IdType* append_indptr,
                                      const IdType* seq_lens,
                                      IdType* batch_indices,
                                      IdType* positions,
                                      uint32_t batch_size,
                                      uint32_t nnz,
                                      cudaStream_t stream) {
  if (batch_size == 0 || nnz == 0) {
    return cudaSuccess;
  }

  constexpr int BLOCK_THREADS = 128;
  auto kernel = GetBatchIndicesPositionsKernel<IdType, BLOCK_THREADS>;
  void* args[] = {
      (void*)&append_indptr,
      (void*)&seq_lens,
      (void*)&batch_indices,
      (void*)&positions,
  };
  FLASHINFER_CUDA_CALL(cudaLaunchKernel(
      (void*)kernel, dim3(batch_size), dim3(BLOCK_THREADS), args, 0, stream));
  return cudaSuccess;
}
```

Notes:

- This mirrors the issue author's naive CUDA shape while placing it in the reusable page header.
- `positions[offset] = offset + seq_len - batch_end` exactly matches the Triton implementation.
- `nnz` is not used by the kernel loop because each row's bounds come from `append_indptr`; the launcher still needs it to skip empty launches.
- The implementation assumes caller invariants already required by the append path: monotonic `append_indptr`, `append_indptr[-1] == nnz`, and final `seq_lens[batch_idx] >= append_len`.

## FFI Wrapper Plan

For PR 1, add this shape to `csrc/page.cu`:

```cpp
void get_batch_indices_positions_cuda(TensorView append_indptr,
                                      TensorView seq_lens,
                                      TensorView batch_indices,
                                      TensorView positions) {
  CHECK_INPUT_AND_TYPE(append_indptr, dl_int32);
  CHECK_INPUT_AND_TYPE(seq_lens, dl_int32);
  CHECK_INPUT_AND_TYPE(batch_indices, dl_int32);
  CHECK_INPUT_AND_TYPE(positions, dl_int32);
  CHECK_DIM(1, append_indptr);
  CHECK_DIM(1, seq_lens);
  CHECK_DIM(1, batch_indices);
  CHECK_DIM(1, positions);

  const uint32_t batch_size = seq_lens.size(0);
  TVM_FFI_ICHECK_EQ(append_indptr.size(0), batch_size + 1);
  TVM_FFI_ICHECK_EQ(positions.size(0), batch_indices.size(0));
  CHECK_DEVICE(seq_lens, append_indptr);
  CHECK_DEVICE(batch_indices, append_indptr);
  CHECK_DEVICE(positions, append_indptr);

  ffi::CUDADeviceGuard device_guard(append_indptr.device().device_id);
  const cudaStream_t stream = get_stream(append_indptr.device());
  const uint32_t nnz = batch_indices.size(0);

  cudaError_t status = flashinfer::GetBatchIndicesPositions<int32_t>(
      static_cast<const int32_t*>(append_indptr.data_ptr()),
      static_cast<const int32_t*>(seq_lens.data_ptr()),
      static_cast<int32_t*>(batch_indices.data_ptr()),
      static_cast<int32_t*>(positions.data_ptr()),
      batch_size,
      nnz,
      stream);
  TVM_FFI_ICHECK(status == cudaSuccess)
      << "GetBatchIndicesPositions failed with error: " << cudaGetErrorString(status);
}
```

Do not copy `append_indptr[-1]` to host to check `nnz`; that would add a synchronization point to a small metadata helper. Let correctness tests cover valid inputs and document the invariant.

## Follow-Up Python Wrapper Plan

This section is for PR 2, not PR 1. Replace the Triton launch in `flashinfer/page.py` with the page module call only after the C++ path has landed and has direct tests.

Illustrative code:

```python
@flashinfer_api
def get_batch_indices_positions(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    nnz: int,
    batch_indices: Optional[torch.Tensor] = None,
    positions: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = append_indptr.size(0) - 1
    device = append_indptr.device
    dtype = torch.int32

    # Match the existing metadata-wrapper pattern: normalize dtype in Python
    # before calling the int32-only FFI helper, without moving devices.
    append_indptr = append_indptr.int()
    seq_lens = seq_lens.int()

    if batch_indices is None:
        batch_indices = torch.full((nnz,), -1, device=device, dtype=dtype)
    else:
        check_shape_dtype_device(batch_indices, (nnz,), dtype, device, "batch_indices")
        batch_indices.fill_(-1)

    if positions is None:
        positions = torch.zeros((nnz,), device=device, dtype=dtype)
    else:
        check_shape_dtype_device(positions, (nnz,), dtype, device, "positions")

    get_page_module().get_batch_indices_positions_cuda(
        append_indptr, seq_lens, batch_indices, positions
    )
    return batch_indices, positions
```

If `torch.compile()` friendliness is a priority, add a private custom op wrapper:

```python
@register_custom_op(
    "flashinfer::get_batch_indices_positions",
    mutates_args=("batch_indices", "positions"),
)
def _get_batch_indices_positions_kernel(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    get_page_module().get_batch_indices_positions_cuda(
        append_indptr, seq_lens, batch_indices, positions
    )


@register_fake_op("flashinfer::get_batch_indices_positions")
def _fake_get_batch_indices_positions_kernel(
    append_indptr: torch.Tensor,
    seq_lens: torch.Tensor,
    batch_indices: torch.Tensor,
    positions: torch.Tensor,
) -> None:
    pass
```

This is optional because the current Triton helper is not registered as a custom op, but it would match the style used by append kernels in `flashinfer/page.py`.

## C++ Caller Example

Illustrative C++ usage after the helper exists:

```cpp
#include <flashinfer/page.cuh>

void append_from_cpp(flashinfer::paged_kv_t<half, int32_t> paged_kv,
                     half* append_key,
                     half* append_value,
                     const int32_t* append_indptr,
                     const int32_t* seq_lens,
                     int32_t* batch_indices,
                     int32_t* positions,
                     uint32_t batch_size,
                     uint32_t nnz,
                     size_t append_k_stride_n,
                     size_t append_k_stride_h,
                     size_t append_v_stride_n,
                     size_t append_v_stride_h,
                     cudaStream_t stream) {
  FLASHINFER_CUDA_CALL(flashinfer::GetBatchIndicesPositions<int32_t>(
      append_indptr, seq_lens, batch_indices, positions, batch_size, nnz, stream));

  FLASHINFER_CUDA_CALL(flashinfer::AppendPagedKVCache(
      paged_kv,
      append_key,
      append_value,
      batch_indices,
      positions,
      nnz,
      append_k_stride_n,
      append_k_stride_h,
      append_v_stride_n,
      append_v_stride_h,
      stream));
}
```

The caller owns allocation of `batch_indices` and `positions`, each with length `nnz`.

## Python Usage Example

In PR 1, public Python usage remains unchanged and still uses the Triton-backed wrapper:

```python
append_lens = torch.tensor([45, 8, 0, 22], dtype=torch.int32, device="cuda")
append_indptr = torch.empty(append_lens.numel() + 1, dtype=torch.int32, device="cuda")
append_indptr[0] = 0
append_indptr[1:] = torch.cumsum(append_lens, dim=0)

seq_lens = flashinfer.get_seq_lens(kv_indptr, kv_last_page_len, page_size)
nnz = int(append_indptr[-1].item())
batch_indices, positions = flashinfer.get_batch_indices_positions(
    append_indptr, seq_lens, nnz
)

flashinfer.append_paged_kv_cache(
    append_key,
    append_value,
    batch_indices,
    positions,
    paged_kv_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
)
```

## PR 1 Implementation Steps

1. Add the CUDA kernel and host launcher to `include/flashinfer/page.cuh`.
2. Add `get_batch_indices_positions_cuda(...)` TVM-FFI wrapper function to `csrc/page.cu`.
3. Export `get_batch_indices_positions_cuda` from `csrc/flashinfer_page_binding.cu`.
4. Leave `flashinfer/page.py` unchanged; `flashinfer.get_batch_indices_positions()` should continue to use the Triton kernel in PR 1.
5. Leave `flashinfer/jit/page.py` unchanged unless adding a short comment; the existing `gen_page_module()` already compiles the changed sources.
6. Leave `flashinfer/aot.py` unchanged; it already includes `gen_page_module()`.
7. Add direct helper tests that call `get_page_module().get_batch_indices_positions_cuda(...)`.
8. Run the targeted PR 1 test plan below.

## PR 2 Implementation Steps

1. Update `flashinfer/page.py` so `get_batch_indices_positions()` launches `get_page_module().get_batch_indices_positions_cuda(...)`.
2. Keep Python-wrapper compatibility with the existing implementation: cast `append_indptr` and `seq_lens` to int32 before the FFI call, keep output allocation/validation unchanged, and update the docstring example to use CUDA int32 metadata tensors.
3. Add or update public wrapper tests.
4. Strengthen append integration tests if not already done in PR 1.
5. Run the PR 2 test plan below.

## Appendix: Optional Triton Helper Cleanup

Do not remove `flashinfer/triton/page.py` in PR 1 or PR 2. Even if the public Python wrapper no longer imports it after PR 2, removal is a compatibility and maintenance-policy decision for FlashInfer core maintainers.

If maintainers later decide the Triton helper has no remaining compatibility, testing, or debugging value, handle that as a separate cleanup PR with its own deprecation/removal rationale.

## Test Plan

### Unit Tests for Index Generation

For PR 1, add tests to `tests/attention/test_page.py` or a new `tests/attention/test_page_indices.py` that call the new page-module entry point directly.

Reference helper:

```python
def batch_indices_positions_ref(append_indptr, seq_lens):
    append_indptr_cpu = append_indptr.cpu()
    seq_lens_cpu = seq_lens.cpu()
    nnz = int(append_indptr_cpu[-1].item())
    batch_indices = torch.empty(nnz, dtype=torch.int32)
    positions = torch.empty(nnz, dtype=torch.int32)
    for b in range(seq_lens_cpu.numel()):
        start = int(append_indptr_cpu[b])
        end = int(append_indptr_cpu[b + 1])
        for offset in range(start, end):
            batch_indices[offset] = b
            positions[offset] = offset + int(seq_lens_cpu[b]) - end
    return batch_indices.cuda(), positions.cuda()
```

Cases:

- deterministic issue-style example: append lengths `[1, 2, 3, 4]`, uniform final seq lens
- mixed append lengths: `[45, 8, 25, 22]`
- zero-length rows: `[0, 3, 0, 5, 1]`
- single batch row
- many rows with small appends, for example 1024 rows and random lengths in `[0, 8]`
- `nnz == 0`, for example all append lengths are zero
- preallocated output tensors filled with sentinels before the call

PR 1 assertions:

```python
module = flashinfer.page.get_page_module()
out_batch = torch.empty((nnz,), dtype=torch.int32, device="cuda")
out_pos = torch.empty((nnz,), dtype=torch.int32, device="cuda")
module.get_batch_indices_positions_cuda(append_indptr, seq_lens, out_batch, out_pos)
ref_batch, ref_pos = batch_indices_positions_ref(append_indptr, seq_lens)
torch.testing.assert_close(out_batch, ref_batch)
torch.testing.assert_close(out_pos, ref_pos)
```

PR 2 should add public-wrapper assertions after switching the implementation:

```python
out_batch, out_pos = flashinfer.get_batch_indices_positions(
    append_indptr, seq_lens, nnz
)
```

The direct PR 1 tests catch symbol export and JIT build regressions separately from the public wrapper.

### Validation Tests

For PR 1, validation tests should target `get_batch_indices_positions_cuda(...)`.

Because the direct FFI wrapper is strict CUDA int32:

- non-contiguous `append_indptr` should raise
- `seq_lens` on a different device should raise
- `int64` inputs should raise with a clear message
- output buffers with the wrong shape should raise

After PR 2 switches the public Python wrapper:

- CUDA `int64` `append_indptr` and `seq_lens` should still produce int32 outputs through the Python-wrapper casts
- CPU or different-device `seq_lens` should be rejected rather than moved implicitly
- preallocated `batch_indices` should still be validated and filled with `-1` before launch
- preallocated `positions` should still be validated without the extra sentinel fill

### Append Integration Test

For PR 1, add a new integration path that generates `batch_indices` and `positions` through `get_page_module().get_batch_indices_positions_cuda(...)`, then passes them to `append_paged_kv_cache`. This proves the C++/CUDA helper produces arrays accepted by the existing append path without changing the public helper.

Separately, strengthen `test_append_paged_kv_cache` so it checks cache contents after append. The current test only calls the API.

Recommended reference:

1. Clone the original paged cache before append.
2. Run `append_paged_kv_cache`.
3. For each appended token `i`, compute:
   - `b = batch_indices[i]`
   - `pos = positions[i]`
   - `page_iter = kv_indptr[b] + pos // page_size`
   - `entry = pos % page_size`
   - `page = kv_indices[page_iter]`
4. Assert the cache slot equals `append_key[i]` and `append_value[i]`.
5. Include both `NHD` and `HND` layouts if the existing append test is expanded.

### PR 2 Trace/Reference Tests

Run existing trace reference tests in PR 2 to catch behavior drift after the public Python helper is switched:

```bash
pytest tests/trace/test_append_paged_kv_cache_reference_correctness.py -v
pytest tests/trace/test_append_paged_mla_kv_cache_reference_correctness.py -v
pytest tests/trace/test_rope_quantize_fp8_append_paged_kv_cache_reference_correctness.py -v
```

The helper does not have a trace template, so no new trace JSON should be required.

### PR 1 Build/JIT Tests

Run:

```bash
pytest tests/attention/test_page.py -v
pytest tests/utils/test_jit_warmup.py::test_warmpup_llama -v
pytest tests/test_jit_cpp_ext.py -v
```

The warmup test includes `flashinfer.page.gen_page_module()` and should catch missing symbols or source-list mistakes.

### Full Local Quality Gate

Before sending a PR:

```bash
pre-commit run -a
pytest tests/attention/test_page.py tests/trace/test_append_paged_kv_cache_reference_correctness.py -v
```

Run broader `pytest tests/` only if GPU time is available.

## Edge Cases and Invariants

- `batch_size == 0`: the launcher must return `cudaSuccess` without launching.
- `nnz == 0`: the launcher must return `cudaSuccess` without launching.
- zero append length for a row: that block performs no stores.
- `append_indptr` must be monotonic and have length `batch_size + 1`.
- `append_indptr[-1]` must equal `nnz`; avoid checking this in the FFI wrapper to prevent host synchronization.
- `seq_lens[b]` is the final sequence length after append. The previous length is `seq_lens[b] - (append_indptr[b + 1] - append_indptr[b])`.
- PR 1 FFI output dtype is `int32`; PR 2 should preserve the public Python API's int32 output behavior.
- values beyond int32 range are unsupported in the PR 1 FFI path and should remain unsupported if PR 2 keeps int32 outputs.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| PR 2 could accidentally expose the stricter FFI dtype contract or add hidden device copies | Keep input dtype normalization in the Python wrapper, keep CUDA/same-device validation explicit, and update the docstring example to show CUDA int32 metadata tensors. |
| Invalid `append_indptr[-1]` causes out-of-bounds writes | Keep the invariant documented; add debug-only validation only if a synchronization cost is acceptable. |
| Missing TVM-FFI export breaks direct module access | Add a direct `get_page_module().get_batch_indices_positions_cuda(...)` test. |
| CUDA helper diverges from Triton formula | Test against a Python reference across random ragged shapes and zero-length rows. |
| Empty inputs attempt an invalid zero-block launch | Explicitly return early for `batch_size == 0 || nnz == 0`. |

## Rollback Plan

PR 1 is isolated to the new C++/CUDA helper and direct page-module symbol. If regressions appear:

1. Remove the `get_batch_indices_positions_cuda` export and FFI wrapper.
2. Remove or fix the raw CUDA helper, depending on whether the regression is in FFI wiring or kernel logic.
3. No public Python wrapper rollback is needed for PR 1 because `flashinfer.get_batch_indices_positions()` remains unchanged.

For PR 2, rollback is straightforward: revert the Python wrapper to import and launch `flashinfer.triton.page.get_batch_indices_positions_kernel` while keeping the C++ helper if it remains useful for C++ callers.
