import pytest
import torch

import flashinfer
from flashinfer.utils import _to_device_packed

DEVICE = torch.device("cuda:0")


def _metadata(kv_lens, page_size):
    pages = [(n + page_size - 1) // page_size for n in kv_lens]
    kv_indptr = torch.tensor([0] + pages, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    kv_indices = torch.arange(int(kv_indptr[-1]), dtype=torch.int32)
    kv_last_page_len = torch.tensor(
        [(n - 1) % page_size + 1 for n in kv_lens], dtype=torch.int32
    )
    return kv_indptr, kv_indices, kv_last_page_len


def _follows(a, b):
    return b.data_ptr() == a.data_ptr() + a.numel() * a.element_size()


def test_to_device_packed_host_tensors_share_one_buffer():
    host = [
        torch.arange(n, dtype=torch.int32) + 10 * i for i, n in enumerate((3, 2, 5))
    ]
    out = _to_device_packed(host, DEVICE, False)
    for h, d in zip(host, out, strict=True):
        assert d.device == DEVICE and torch.equal(d.cpu(), h)
    assert _follows(out[0], out[1]) and _follows(out[1], out[2])


def test_to_device_packed_leaves_device_and_mixed_tensors_alone():
    on_device = torch.arange(4, dtype=torch.int32, device=DEVICE)
    host_a = torch.arange(3, dtype=torch.int32)
    host_b = torch.arange(5, dtype=torch.int32)
    out = _to_device_packed([host_a, on_device, host_b], DEVICE, False)
    # A tensor already on the device is returned as is, as Tensor.to does.
    assert out[1].data_ptr() == on_device.data_ptr()
    assert _follows(out[0], out[2])
    assert torch.equal(out[0].cpu(), host_a) and torch.equal(out[2].cpu(), host_b)
    # Host tensors of different dtypes cannot share a buffer.
    mixed = _to_device_packed([host_a, host_b.to(torch.int64)], DEVICE, False)
    assert mixed[0].dtype == torch.int32 and mixed[1].dtype == torch.int64
    assert torch.equal(mixed[1].cpu(), host_b.to(torch.int64))


@pytest.mark.parametrize("use_tensor_cores", [False, True])
def test_batch_decode_host_metadata_matches_device_metadata(use_tensor_cores):
    torch.manual_seed(0)
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 2, 128, 16
    kv_lens = [40, 90, 64, 200]
    metadata = _metadata(kv_lens, page_size)
    kv_data = torch.randn(
        len(metadata[1]),
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=torch.float16,
    )
    q = torch.randn(
        len(kv_lens), num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    outputs = []
    for metadata_device in ("cpu", DEVICE):
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace, "NHD", use_tensor_cores=use_tensor_cores, backend="fa2"
        )
        wrapper.plan(
            *(t.to(metadata_device) for t in metadata),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        if metadata_device == "cpu":
            # Host metadata is staged in one device buffer, per-request arrays first.
            assert _follows(
                wrapper._paged_kv_indptr_buf, wrapper._paged_kv_last_page_len_buf
            )
        outputs.append(wrapper.run(q, kv_data))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)


def test_batch_prefill_host_metadata_matches_device_metadata():
    torch.manual_seed(0)
    num_qo_heads, num_kv_heads, head_dim, page_size = 8, 2, 128, 16
    kv_lens = [40, 90, 64, 200]
    q_lens = [3, 1, 5, 2]
    metadata = _metadata(kv_lens, page_size)
    qo_indptr = torch.tensor([0] + q_lens, dtype=torch.int32).cumsum(
        0, dtype=torch.int32
    )
    kv_data = torch.randn(
        len(metadata[1]),
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=DEVICE,
        dtype=torch.float16,
    )
    q = torch.randn(
        sum(q_lens), num_qo_heads, head_dim, device=DEVICE, dtype=torch.float16
    )
    outputs = []
    for metadata_device in ("cpu", DEVICE):
        workspace = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device=DEVICE)
        wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            workspace, "NHD", backend="fa2"
        )
        wrapper.plan(
            qo_indptr.to(metadata_device),
            *(t.to(metadata_device) for t in metadata),
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            causal=True,
            q_data_type=torch.float16,
            kv_data_type=torch.float16,
        )
        if metadata_device == "cpu":
            assert _follows(
                wrapper._paged_kv_indptr_buf, wrapper._paged_kv_last_page_len_buf
            )
            assert _follows(wrapper._paged_kv_last_page_len_buf, wrapper._qo_indptr_buf)
        outputs.append(wrapper.run(q, kv_data))
    torch.testing.assert_close(outputs[0], outputs[1], rtol=0, atol=0)
