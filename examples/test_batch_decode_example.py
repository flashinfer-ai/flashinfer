import torch
import flashinfer

def verify_tensors(tensor1, tensor2, rtol=1e-3, atol=1e-3):
    
    for i in range(tensor1.shape[0]):
        for j in range(tensor1.shape[1]):
            if torch.abs(tensor1[i][j] - tensor2[i][j]) > atol + rtol * torch.abs(tensor2[i][j]):
                print(f"Error at {i}, {j}")
                print(f"Expected: {tensor2[i][j]}")
                print(f"Got: {tensor1[i][j]}")
                return False
    return True

def test_batch_decode_with_paged_kv_cache(
    batch_size,
    kv_len,
    page_size,
    num_kv_heads,
    num_qo_heads,
    head_dim,
    kv_layout,
    pos_encoding_mode,
    logits_soft_cap,
    return_lse,
    q_dtype,
    kv_dtype,
    contiguous_kv,
):
    q = torch.randn(batch_size, num_qo_heads, head_dim, device="cuda:0", dtype=q_dtype)
    num_pages_per_seq = (kv_len + page_size - 1) // page_size
    total_num_pages = num_pages_per_seq * batch_size
    result = 0
    if kv_layout == "HND":
        kv_shape = [total_num_pages, 2, num_kv_heads, page_size, head_dim]
    else:
        kv_shape = [total_num_pages, 2, page_size, num_kv_heads, head_dim]
    if not contiguous_kv:
        tmp = [kv_shape[0]]
        for v in kv_shape[1:]:
            tmp.append(2)
            tmp.append(v)
        kv_shape = tmp
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
        kv_data = kv_data[:, 1, :, 1, :, 1, :, 1, :]
        kv_data_fp32 = kv_data_fp32[:, 1, :, 1, :, 1, :, 1, :]
        # actual data is stored in non-contiguous memory
        assert (
            kv_data.stride(-4)
            != kv_data.shape[-3] * kv_data.shape[-2] * kv_data.shape[-1]
        )
    else:
        kv_data_fp32 = torch.randn(*kv_shape, dtype=torch.float32, device="cuda:0")
        kv_data = kv_data_fp32.to(kv_dtype)
    kv_indptr = (
        torch.arange(0, batch_size + 1, device="cuda:0", dtype=torch.int32)
        * num_pages_per_seq
    )
    kv_indices = torch.arange(0, total_num_pages, device="cuda:0", dtype=torch.int32)
    kv_last_page_len = torch.full(
        (batch_size,), (kv_len - 1) % page_size + 1, dtype=torch.int32, device="cuda:0"
    )

    workspace_buffer = torch.empty(32 * 1024 * 1024, dtype=torch.int8, device="cuda:0")
    wrapper = flashinfer.decode.BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, kv_layout
    )
    wrapper.plan(
        kv_indptr,
        kv_indices,
        kv_last_page_len,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        logits_soft_cap=logits_soft_cap,
        pos_encoding_mode=pos_encoding_mode,
        data_type=kv_dtype,
        q_data_type=q_dtype,
    )
    if return_lse:
        o, _ = wrapper.run(q, kv_data, return_lse=True)
    else:
        o = wrapper.run(q, kv_data)

    for i in range(batch_size):
        perm_dims = [0, 2, 1, 3] if kv_layout == "HND" else [0, 1, 2, 3]
        perm_dims_last = [1, 0, 2] if kv_layout == "HND" else [0, 1, 2]
        qi = q[i]
        ki = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 0]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 0, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 0, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        vi = torch.cat(
            [
                kv_data_fp32[kv_indptr[i] : kv_indptr[i + 1] - 1, 1]
                .permute(*perm_dims)
                .reshape(-1, num_kv_heads, head_dim),
                (
                    kv_data_fp32[kv_indptr[i + 1] - 1, 1, :, : kv_last_page_len[i]]
                    if kv_layout == "HND"
                    else kv_data_fp32[kv_indptr[i + 1] - 1, 1, : kv_last_page_len[i], :]
                )
                .permute(*perm_dims_last)
                .reshape(-1, num_kv_heads, head_dim),
            ],
            dim=0,
        ).to(kv_dtype)
        # print(qi.shape, ki.shape, vi.shape)
        o_ref_i = flashinfer.single_decode_with_kv_cache(
            qi,
            ki,
            vi)
        # torch.testing.assert_close(o[i], o_ref_i, rtol=1e-3, atol=1e-3)
        result += verify_tensors(o[i], o_ref_i, rtol=1e-3, atol=1e-3)

    # test user-allocated output
    o_buffer = torch.empty_like(o)
    wrapper.run(q, kv_data, out=o_buffer)
    torch.testing.assert_close(o, o_buffer, rtol=1e-3, atol=1e-3)

    if result == batch_size:
        print("PASS")
    else:
        print("FAIL")

if __name__ == "__main__":

    batch_size = 256
    page_size = 8   

    # # This configuration works
    num_qo_heads = 32 
    num_kv_heads = 4 
    head_dim = 256
    kv_len = 512

    # # This configuration fails
    # num_qo_heads = 8
    # num_kv_heads = 8
    # head_dim = 128
    # kv_len = 54
    
    kv_layout = "NHD"
    pos_encoding_mode = "NONE"
    logits_soft_cap = 0.0
    return_lse = False
    q_dtype = torch.float16
    kv_dtype = torch.float16
    contiguous_kv = True
    
    num_qo_heads = 32 
    num_kv_heads = 4 
    head_dim = 256
    kv_len = 512
    test_batch_decode_with_paged_kv_cache(
        batch_size,
        kv_len,
        page_size,
        num_kv_heads,
        num_qo_heads,
        head_dim,
        kv_layout,
        pos_encoding_mode,
        logits_soft_cap,
        return_lse,
        q_dtype,
        kv_dtype,
        contiguous_kv)
    
