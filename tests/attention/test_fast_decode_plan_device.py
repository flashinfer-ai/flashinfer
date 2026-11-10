import pytest
import torch

import flashinfer

@pytest.mark.parametrize("metadata_device", ["cpu", "cuda:1"])
def test_fast_decode_plan_rejects_metadata_off_wrapper_device(metadata_device):
    # fast_decode_plan adopts indptr/indices/last_page_len as the wrapper's
    # buffers without copying, so metadata on any other device used to reach
    # the kernel as-is and fault at the next synchronization (#6391).
    if metadata_device == "cuda:1" and torch.cuda.device_count() < 2:
        pytest.skip("needs a second CUDA device")
    page_size = 16
    kv_lens = [40, 90, 64, 200]
    pages = [(n + page_size - 1) // page_size for n in kv_lens]
    kv_indptr = torch.tensor([0] + pages, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    kv_indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [(n - 1) % page_size + 1 for n in kv_lens], dtype=torch.int32
    )
    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", backend="fa2"
    )
    with pytest.raises(ValueError, match="must be on the wrapper's device"):
        flashinfer.fast_decode_plan(
            wrapper,
            kv_indptr.to(metadata_device),
            kv_indices.to(metadata_device),
            kv_last_page_len.to(metadata_device),
            8,
            2,
            128,
            page_size,
            q_data_type=torch.float16,
        )
