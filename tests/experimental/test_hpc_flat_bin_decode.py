"""GPU correctness and real task-map tests for the parallel scheduling port."""

import math
import pytest
import torch
from flashinfer.decode import BatchDecodeWithPagedKVCacheWrapper
from flashinfer.hpc_decode import create_hpc_flat_bin_decode_wrapper

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 0),
    reason="Requires SM100",
)


@pytest.mark.parametrize("heads", [1, 8])
@pytest.mark.parametrize("lengths", [[1], [65, 129, 1023], [512] * 31 + [8193]])
@pytest.mark.parametrize("cta_per_sm", [1, 2])
def test_decode_and_changed_lengths(heads, lengths, cta_per_sm):
    torch.manual_seed(123)
    B = len(lengths)
    pages = [(n + 63) // 64 for n in lengths]
    ptr = torch.tensor(
        [0] + list(torch.tensor(pages).cumsum(0).tolist()), dtype=torch.int32
    )
    ids_cpu = torch.randperm(int(ptr[-1]), dtype=torch.int32)
    ids = ids_cpu.cuda()
    q = torch.randn(B, heads * 4, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(int(ptr[-1]), 64, heads, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    native = BatchDecodeWithPagedKVCacheWrapper(
        torch.empty(64 * 1024**2, device="cuda", dtype=torch.uint8),
        "NHD",
        use_tensor_cores=False,
        backend="fa2",
    )
    port = create_hpc_flat_bin_decode_wrapper(
        cta_per_sm=cta_per_sm, min_process_len=128
    )
    for cur in (lengths, [max(1, n - 1) for n in lengths]):
        tail = torch.tensor(
            [(n - 1) % 64 + 1 for n in cur], device="cuda", dtype=torch.int32
        )
        # Adjust page ptr/ids too when reducing a page-boundary length.
        cur_pages = [(n + 63) // 64 for n in cur]
        cur_ptr = torch.tensor(
            [0] + list(torch.tensor(cur_pages).cumsum(0).tolist()), dtype=torch.int32
        )
        cur_ids = torch.cat(
            [ids[int(ptr[b]) : int(ptr[b]) + cur_pages[b]] for b in range(B)]
        )
        native.plan(
            cur_ptr,
            cur_ids,
            tail,
            heads * 4,
            heads,
            128,
            64,
            q_data_type=torch.bfloat16,
            kv_data_type=torch.bfloat16,
        )
        port.plan(cur_ptr, cur_ids, tail, heads * 4, heads, 128, 64)
        with pytest.raises(ValueError, match="phase"):
            port.assign_gpu(4)
        out = port.run(q, (k, v))
        torch.testing.assert_close(out, native.run(q, (k, v)), atol=0.002, rtol=0.02)
        for b, n in enumerate(cur):
            local_ids = cur_ids[int(cur_ptr[b]) : int(cur_ptr[b + 1])].long()
            kb = k[local_ids].reshape(-1, heads, 128)[:n].float()
            vb = v[local_ids].reshape(-1, heads, 128)[:n].float()
            scores = torch.einsum(
                "hgd,nhd->hgn", q[b].float().reshape(heads, 4, 128), kb
            ) / math.sqrt(128)
            ref = torch.einsum("hgn,nhd->hgd", scores.softmax(-1), vb).reshape(
                heads * 4, 128
            )
            torch.testing.assert_close(out[b].float(), ref, atol=0.003, rtol=0.02)
        raw = port.schedule()
        tasks = raw["tasks"]
        assert len(tasks) <= B * heads + port.pool
        assert sorted(t[4] for t in tasks) == list(range(len(tasks)))
        assert raw["cta_indptr"][-1] == len(tasks)
        for b, n in enumerate(cur):
            for h in range(heads):
                ranges = sorted((t[2], t[3]) for t in tasks if t[0] == b and t[1] == h)
                assert ranges[0][0] == 0 and ranges[-1][1] == n
                assert all(
                    a[1] == c[0] for a, c in zip(ranges, ranges[1:], strict=False)
                )


def test_explicit_opt_in_and_guards():
    import flashinfer

    assert create_hpc_flat_bin_decode_wrapper.is_experimental
    assert not hasattr(flashinfer, "create_hpc_flat_bin_decode_wrapper")
    port = create_hpc_flat_bin_decode_wrapper()
    with pytest.raises(RuntimeError, match="plan"):
        port.run(torch.empty(0), (torch.empty(0), torch.empty(0)))
    with pytest.raises(ValueError, match="head count"):
        port.plan(torch.tensor([0, 1], dtype=torch.int32), None, None, 0, 0, 128, 64)
    with pytest.raises(ValueError, match="positive"):
        port.plan(torch.tensor([0, 0], dtype=torch.int32), None, None, 4, 1, 128, 64)
