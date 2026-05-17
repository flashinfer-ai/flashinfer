"""DLLM Block Extend Attention Precision Tests

Precision tests for Block Extend Attention used in Diffusion LLMs.

Mask rule: mask[q, k] = ((q_local + q_offset) // B) >= (k // B)

Test coverage:
  - Single/multi request precision vs custom_mask reference
  - Ragged and Paged KV cache wrappers
  - Heterogeneous prefix lengths
  - Cascade vs block-extend alignment
  - FA2 and FA3 backends (FA3 only on SM90+)
"""

import math
import torch
import flashinfer
from flashinfer import (
    BatchPrefillWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    merge_state,
    single_prefill_with_kv_cache,
)
from flashinfer.dllm import (
    BatchBlockExtendPagedOffsetWrapper,
    BatchBlockExtendRaggedOffsetWrapper,
    block_extend_attention_with_offset,
)


def get_available_backends(device=None):
    """Get available backends based on GPU architecture."""
    from flashinfer.utils import is_sm90a_supported

    if device is None:
        device = torch.device("cuda:0")

    if is_sm90a_supported(device):
        return ["fa2", "fa3"]
    else:
        return ["fa2"]


def compute_block_extend_reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dllm_block_size: int,
    q_offset: int = 0,
    sm_scale: float = None,
) -> torch.Tensor:
    """Reference implementation using custom_mask."""
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    if sm_scale is None:
        head_dim = q.shape[-1]
        sm_scale = 1.0 / math.sqrt(head_dim)

    q_pos = torch.arange(qo_len, device=q.device) + q_offset
    k_pos = torch.arange(kv_len, device=k.device)
    q_block = q_pos.unsqueeze(1) // dllm_block_size
    k_block = k_pos.unsqueeze(0) // dllm_block_size
    mask_2d = (q_block >= k_block).to(torch.uint8)

    return single_prefill_with_kv_cache(
        q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"
    )


def test_dllm_precision_vs_custom_mask_fa2(
    verbose: bool = True,
    test_dtypes: list = None,
):
    """DLLM precision test: compare with native Custom Mask FA2 implementation."""
    device = torch.device("cuda:0")
    available_backends = get_available_backends(device)
    has_fa3 = "fa3" in available_backends

    if test_dtypes is None:
        test_dtypes = [torch.float16, torch.bfloat16]

    dtype_tolerances = {
        torch.float16: 1e-2,
        torch.bfloat16: 2e-2,
    }
    dtype_names = {
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
    }

    # Core test configs covering key dimensions
    test_configs = [
        # Different block_size
        {"dllm_block_size": 16, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        # Different q_offset (incremental prefill steps)
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 192, "q_offset": 64, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        # GQA and MQA
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 1, "head_dim": 128},
        # Different head_dim
        {"dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 64},
        # Long sequence
        {"dllm_block_size": 32, "qo_len": 128, "kv_len": 2048, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        # Non-aligned boundary
        {"dllm_block_size": 64, "qo_len": 33, "kv_len": 97, "q_offset": 17, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
    ]

    # Multi-request test configs
    multi_req_configs = [
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 192, "q_offset": 64, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"num_requests": 4, "dllm_block_size": 32, "qo_len": 64, "kv_len": 128, "q_offset": 0, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},
    ]

    page_size = 16

    for dtype in test_dtypes:
        dtype_name = dtype_names[dtype]
        tol = dtype_tolerances[dtype]

        print(f"\n{'='*100}")
        print(f"DLLM Precision Test vs Custom Mask [{dtype_name.upper()}]  backends={available_backends}")
        print(f"{'='*100}")

        # ===== Part 1: Single-request =====
        single_req_results = []
        for cfg_idx, cfg in enumerate(test_configs):
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            mask_2d = ((q_pos.unsqueeze(1) // dllm_block_size) >= (k_pos.unsqueeze(0) // dllm_block_size)).to(torch.uint8)
            ref_output = single_prefill_with_kv_cache(q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2")

            result = {"config_idx": cfg_idx, **cfg}

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=device)

            for backend in available_backends:
                workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
                wrapper = BatchBlockExtendRaggedOffsetWrapper(
                    workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                wrapper.plan(
                    qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, q_data_type=dtype,
                    sm_scale=sm_scale, q_offsets=q_offset_tensor,
                )
                bbe_output = wrapper.run(q, k, v)
                bbe_diff = (bbe_output - ref_output).abs().max().item()
                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_pass"] = bbe_diff < tol
                del workspace, wrapper

            # V2 API
            v2_output = block_extend_attention_with_offset(
                q, k, v, dllm_block_size=dllm_block_size, q_offset=q_offset, sm_scale=sm_scale, backend="fa2"
            )
            v2_diff = (v2_output - ref_output).abs().max().item()
            result["v2_max_diff"] = v2_diff
            result["v2_pass"] = v2_diff < tol

            single_req_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                v2_status = "PASS" if result["v2_pass"] else "FAIL"
                msg = f"  [{cfg_idx:02d}] B={dllm_block_size:3d} qo={qo_len:4d} kv={kv_len:4d} off={q_offset:3d} h={num_heads}/{num_kv_heads} d={head_dim}"
                msg += f"  FA2:{fa2_status}({result['bbe_fa2_max_diff']:.6f})"
                if has_fa3:
                    fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                    msg += f"  FA3:{fa3_status}({result['bbe_fa3_max_diff']:.6f})"
                msg += f"  V2:{v2_status}({v2_diff:.6f})"
                print(msg)

            torch.cuda.empty_cache()

        # Single-request summary
        print(f"\n[Single-request Summary] [{dtype_name}]")
        for backend in available_backends:
            pass_count = sum(1 for r in single_req_results if r[f"bbe_{backend}_pass"])
            print(f"  BBE ({backend.upper()}): {pass_count}/{len(single_req_results)} PASS")
        v2_pass_count = sum(1 for r in single_req_results if r["v2_pass"])
        print(f"  V2: {v2_pass_count}/{len(single_req_results)} PASS")

        # ===== Part 2: Multi-request =====
        multi_req_results = []
        for cfg_idx, cfg in enumerate(multi_req_configs):
            num_requests = cfg["num_requests"]
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            q_batch = torch.randn(num_requests * qo_len, num_heads, head_dim, dtype=dtype, device=device)
            k_batch = torch.randn(num_requests * kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v_batch = torch.randn(num_requests * kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)

            qo_indptr = torch.tensor([i * qo_len for i in range(num_requests + 1)], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([i * kv_len for i in range(num_requests + 1)], dtype=torch.int32, device=device)
            q_offsets = torch.full((num_requests,), q_offset, dtype=torch.int32, device=device)

            # Reference: per-request custom_mask
            ref_outputs = []
            for req_idx in range(num_requests):
                q_req = q_batch[req_idx * qo_len:(req_idx + 1) * qo_len]
                k_req = k_batch[req_idx * kv_len:(req_idx + 1) * kv_len]
                v_req = v_batch[req_idx * kv_len:(req_idx + 1) * kv_len]
                q_pos = torch.arange(qo_len, device=device) + q_offset
                k_pos = torch.arange(kv_len, device=device)
                mask_2d = ((q_pos.unsqueeze(1) // dllm_block_size) >= (k_pos.unsqueeze(0) // dllm_block_size)).to(torch.uint8)
                ref_outputs.append(single_prefill_with_kv_cache(q_req, k_req, v_req, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"))
            ref_output = torch.cat(ref_outputs, dim=0)

            result = {"config_idx": cfg_idx, **cfg}

            for backend in available_backends:
                workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=device)
                bbe_wrapper = BatchBlockExtendRaggedOffsetWrapper(
                    workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                bbe_wrapper.plan(
                    qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, q_data_type=dtype,
                    sm_scale=sm_scale, q_offsets=q_offsets,
                )
                bbe_output = bbe_wrapper.run(q_batch, k_batch, v_batch)
                bbe_diff = (bbe_output - ref_output).abs().max().item()
                result[f"bbe_{backend}_max_diff"] = bbe_diff
                result[f"bbe_{backend}_pass"] = bbe_diff < tol
                del bbe_wrapper

            multi_req_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["bbe_fa2_pass"] else "FAIL"
                msg = f"  [{cfg_idx:02d}] reqs={num_requests} B={dllm_block_size:3d} qo={qo_len:4d} kv={kv_len:4d} off={q_offset:3d}"
                msg += f"  FA2:{fa2_status}({result['bbe_fa2_max_diff']:.6f})"
                if has_fa3:
                    fa3_status = "PASS" if result["bbe_fa3_pass"] else "FAIL"
                    msg += f"  FA3:{fa3_status}({result['bbe_fa3_max_diff']:.6f})"
                print(msg)

            torch.cuda.empty_cache()

        print(f"\n[Multi-request Summary] [{dtype_name}]")
        for backend in available_backends:
            pass_count = sum(1 for r in multi_req_results if r[f"bbe_{backend}_pass"])
            print(f"  BBE ({backend.upper()}): {pass_count}/{len(multi_req_results)} PASS")

        # ===== Part 3: Paged KV Cache =====
        paged_results = []
        for cfg_idx, cfg in enumerate(test_configs[:5]):  # subset for paged
            dllm_block_size = cfg["dllm_block_size"]
            qo_len = cfg["qo_len"]
            kv_len = cfg["kv_len"]
            q_offset = cfg["q_offset"]
            num_heads = cfg["num_heads"]
            num_kv_heads = cfg["num_kv_heads"]
            head_dim = cfg["head_dim"]
            sm_scale = 1.0 / math.sqrt(head_dim)

            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            num_pages = (kv_len + page_size - 1) // page_size
            kv_data = torch.randn(num_pages, 2, page_size, num_kv_heads, head_dim, dtype=dtype, device=device)

            paged_kv_indices = torch.arange(num_pages, dtype=torch.int32, device=device)
            paged_kv_indptr = torch.tensor([0, num_pages], dtype=torch.int32, device=device)
            last_page_len = kv_len - (num_pages - 1) * page_size if kv_len % page_size != 0 else page_size
            paged_kv_last_page_len = torch.tensor([last_page_len], dtype=torch.int32, device=device)

            k_continuous = kv_data[:, 0, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]
            v_continuous = kv_data[:, 1, :, :, :].reshape(-1, num_kv_heads, head_dim)[:kv_len]

            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            mask_2d = ((q_pos.unsqueeze(1) // dllm_block_size) >= (k_pos.unsqueeze(0) // dllm_block_size)).to(torch.uint8)
            ref_output = single_prefill_with_kv_cache(q, k_continuous, v_continuous, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2")

            result = {"config_idx": cfg_idx, **cfg}

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            q_offsets = torch.tensor([q_offset], dtype=torch.int32, device=device)

            for backend in available_backends:
                paged_wrapper = BatchBlockExtendPagedOffsetWrapper(
                    torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device),
                    kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend
                )
                paged_wrapper.plan(
                    qo_indptr=qo_indptr, paged_kv_indptr=paged_kv_indptr,
                    paged_kv_indices=paged_kv_indices, paged_kv_last_page_len=paged_kv_last_page_len,
                    num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, page_size=page_size,
                    q_data_type=dtype, sm_scale=sm_scale, q_offsets=q_offsets,
                )
                paged_output = paged_wrapper.run(q, kv_data)
                paged_diff = (paged_output - ref_output).abs().max().item()
                result[f"paged_{backend}_max_diff"] = paged_diff
                result[f"paged_{backend}_pass"] = paged_diff < tol
                del paged_wrapper

            paged_results.append(result)

            if verbose:
                fa2_status = "PASS" if result["paged_fa2_pass"] else "FAIL"
                msg = f"  [{cfg_idx:02d}] B={dllm_block_size:3d} qo={qo_len:4d} kv={kv_len:4d}  Paged-FA2:{fa2_status}({result['paged_fa2_max_diff']:.6f})"
                if has_fa3:
                    fa3_status = "PASS" if result["paged_fa3_pass"] else "FAIL"
                    msg += f"  Paged-FA3:{fa3_status}({result['paged_fa3_max_diff']:.6f})"
                print(msg)

            torch.cuda.empty_cache()

        print(f"\n[Paged KV Cache Summary] [{dtype_name}]")
        for backend in available_backends:
            pass_count = sum(1 for r in paged_results if r[f"paged_{backend}_pass"])
            print(f"  Paged ({backend.upper()}): {pass_count}/{len(paged_results)} PASS")

        # Overall result
        if has_fa3:
            all_single_pass = all(r["bbe_fa2_pass"] and r["bbe_fa3_pass"] and r["v2_pass"] for r in single_req_results)
            all_multi_pass = all(r["bbe_fa2_pass"] and r["bbe_fa3_pass"] for r in multi_req_results)
            all_paged_pass = all(r["paged_fa2_pass"] and r["paged_fa3_pass"] for r in paged_results)
        else:
            all_single_pass = all(r["bbe_fa2_pass"] and r["v2_pass"] for r in single_req_results)
            all_multi_pass = all(r["bbe_fa2_pass"] for r in multi_req_results)
            all_paged_pass = all(r["paged_fa2_pass"] for r in paged_results)

        overall_pass = all_single_pass and all_multi_pass and all_paged_pass
        status = "ALL PASS" if overall_pass else "SOME FAILED"
        print(f"\n  Overall: {status}")

    return {"overall_pass": overall_pass}


def test_heterogeneous_prefix_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """Heterogeneous prefix test: different requests have different prefix lengths."""
    from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper

    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2

    test_configs = [
        {"name": "Req0(has_prefix) + Req1(no_prefix)", "dllm_block_size": 32,
         "requests": [{"qo_len": 64, "kv_len": 128, "q_offset": 64}, {"qo_len": 32, "kv_len": 32, "q_offset": 0}]},
        {"name": "Req0(step_3) + Req1(step_0) + Req2(step_1)", "dllm_block_size": 32,
         "requests": [{"qo_len": 32, "kv_len": 128, "q_offset": 96}, {"qo_len": 32, "kv_len": 32, "q_offset": 0}, {"qo_len": 32, "kv_len": 64, "q_offset": 32}]},
        {"name": "Long_short_prompt_mix(512_vs_32)", "dllm_block_size": 32,
         "requests": [{"qo_len": 64, "kv_len": 512, "q_offset": 448}, {"qo_len": 32, "kv_len": 32, "q_offset": 0}]},
        {"name": "4_requests_mixed_scenario", "dllm_block_size": 32,
         "requests": [{"qo_len": 32, "kv_len": 128, "q_offset": 96}, {"qo_len": 32, "kv_len": 32, "q_offset": 0}, {"qo_len": 64, "kv_len": 256, "q_offset": 192}, {"qo_len": 32, "kv_len": 64, "q_offset": 32}]},
    ]

    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    results = []
    for cfg_idx, cfg in enumerate(test_configs):
        requests = cfg["requests"]
        dllm_block_size = cfg["dllm_block_size"]

        all_q, all_k, all_v = [], [], []
        ref_outputs = []
        qo_indptr = [0]
        kv_indptr = [0]
        q_offsets_list = []

        for req in requests:
            q = torch.randn(req["qo_len"], num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(req["kv_len"], num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(req["kv_len"], num_kv_heads, head_dim, dtype=dtype, device=device)
            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

            q_pos = torch.arange(req["qo_len"], device=device) + req["q_offset"]
            k_pos = torch.arange(req["kv_len"], device=device)
            mask_2d = ((q_pos.unsqueeze(1) // dllm_block_size) >= (k_pos.unsqueeze(0) // dllm_block_size)).to(torch.uint8)
            ref_outputs.append(single_prefill_with_kv_cache(q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"))

            qo_indptr.append(qo_indptr[-1] + req["qo_len"])
            kv_indptr.append(kv_indptr[-1] + req["kv_len"])
            q_offsets_list.append(req["q_offset"])

        q_batch = torch.cat(all_q, dim=0)
        k_batch = torch.cat(all_k, dim=0)
        v_batch = torch.cat(all_v, dim=0)
        ref_output = torch.cat(ref_outputs, dim=0)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchBlockExtendRaggedOffsetWrapper(workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend)
        wrapper.plan(
            qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32, device=device),
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=torch.tensor(q_offsets_list, dtype=torch.int32, device=device),
        )
        bbe_output = wrapper.run(q_batch, k_batch, v_batch)
        max_diff = (bbe_output - ref_output).abs().max().item()
        passed = max_diff < tol

        results.append({"config_idx": cfg_idx, "name": cfg["name"], "max_diff": max_diff, "passed": passed})
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{cfg_idx}] {cfg['name']}: max_diff={max_diff:.6f} [{status}]")

        torch.cuda.empty_cache()

    all_pass = all(r["passed"] for r in results)
    assert all_pass, f"Heterogeneous prefix test failed: {[r for r in results if not r['passed']]}"
    return results


def test_cascade_current_chunk_batch(
    verbose: bool = True,
    backend: str = "fa2",
):
    """Batch cascade test: current chunk only (no prefix merge needed)."""
    from flashinfer.dllm import BatchBlockExtendRaggedOffsetWrapper

    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2

    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)

    test_configs = [
        {"name": "Req0(has_prefix) + Req1(no_prefix)", "dllm_block_size": 32,
         "requests": [{"qo_len": 32, "kv_len": 128}, {"qo_len": 32, "kv_len": 32}]},
        {"name": "4_requests_mixed(step_0,1,2,3)", "dllm_block_size": 32,
         "requests": [{"qo_len": 32, "kv_len": 32}, {"qo_len": 32, "kv_len": 64}, {"qo_len": 32, "kv_len": 96}, {"qo_len": 32, "kv_len": 128}]},
        {"name": "Req0(chunk=64) + Req1(chunk=32)", "dllm_block_size": 32,
         "requests": [{"qo_len": 64, "kv_len": 128}, {"qo_len": 32, "kv_len": 64}]},
    ]

    results = []
    for cfg_idx, cfg in enumerate(test_configs):
        dllm_block_size = cfg["dllm_block_size"]
        requests = cfg["requests"]

        all_q, all_k, all_v = [], [], []
        ref_outputs = []
        qo_indptr = [0]
        kv_indptr = [0]
        q_offsets_list = []

        for req_idx, req in enumerate(requests):
            qo_len = req["qo_len"]
            kv_len = req["kv_len"]
            q_offset = kv_len - qo_len  # incremental step offset

            q = torch.randn(qo_len, num_heads, head_dim, dtype=dtype, device=device)
            k = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            v = torch.randn(kv_len, num_kv_heads, head_dim, dtype=dtype, device=device)
            all_q.append(q)
            all_k.append(k)
            all_v.append(v)

            q_pos = torch.arange(qo_len, device=device) + q_offset
            k_pos = torch.arange(kv_len, device=device)
            mask_2d = ((q_pos.unsqueeze(1) // dllm_block_size) >= (k_pos.unsqueeze(0) // dllm_block_size)).to(torch.uint8)
            ref_outputs.append(single_prefill_with_kv_cache(q, k, v, custom_mask=mask_2d, sm_scale=sm_scale, backend="fa2"))

            qo_indptr.append(qo_indptr[-1] + qo_len)
            kv_indptr.append(kv_indptr[-1] + kv_len)
            q_offsets_list.append(q_offset)

        q_batch = torch.cat(all_q, dim=0)
        k_batch = torch.cat(all_k, dim=0)
        v_batch = torch.cat(all_v, dim=0)
        ref_output = torch.cat(ref_outputs, dim=0)

        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        wrapper = BatchBlockExtendRaggedOffsetWrapper(workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend=backend)
        wrapper.plan(
            qo_indptr=torch.tensor(qo_indptr, dtype=torch.int32, device=device),
            kv_indptr=torch.tensor(kv_indptr, dtype=torch.int32, device=device),
            num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
            head_dim=head_dim, q_data_type=dtype,
            sm_scale=sm_scale,
            q_offsets=torch.tensor(q_offsets_list, dtype=torch.int32, device=device),
        )
        bbe_output = wrapper.run(q_batch, k_batch, v_batch)
        max_diff = (bbe_output - ref_output).abs().max().item()
        passed = max_diff < tol

        results.append({"config_idx": cfg_idx, "name": cfg["name"], "max_diff": max_diff, "passed": passed})
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{cfg_idx}] {cfg['name']}: max_diff={max_diff:.6f} [{status}]")

        torch.cuda.empty_cache()

    all_pass = all(r["passed"] for r in results)
    assert all_pass, f"Cascade current chunk test failed: {[r for r in results if not r['passed']]}"
    return results


def test_cascade_precision_alignment(
    verbose: bool = True,
):
    """Step-by-step incremental prefill precision alignment test."""
    device = torch.device("cuda:0")
    dtype = torch.float16
    tol = 1e-2

    test_configs = [
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 64, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 128},
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 4, "head_dim": 128},
        {"dllm_block_size": 32, "num_steps": 4, "num_heads": 32, "num_kv_heads": 8, "head_dim": 64},
    ]

    results = []
    for cfg_idx, cfg in enumerate(test_configs):
        dllm_block_size = cfg["dllm_block_size"]
        num_steps = cfg["num_steps"]
        num_heads = cfg["num_heads"]
        num_kv_heads = cfg["num_kv_heads"]
        head_dim = cfg["head_dim"]
        sm_scale = 1.0 / math.sqrt(head_dim)

        total_tokens = num_steps * dllm_block_size
        q_all = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
        k_all = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
        v_all = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

        step_diffs = []
        for step_idx in range(num_steps):
            q_offset = step_idx * dllm_block_size
            kv_len = (step_idx + 1) * dllm_block_size
            qo_len = dllm_block_size

            q_chunk = q_all[q_offset:q_offset + qo_len]
            k_cumul = k_all[:kv_len]
            v_cumul = v_all[:kv_len]

            ref_output = compute_block_extend_reference(q_chunk, k_cumul, v_cumul, dllm_block_size, q_offset=q_offset, sm_scale=sm_scale)

            qo_indptr = torch.tensor([0, qo_len], dtype=torch.int32, device=device)
            kv_indptr = torch.tensor([0, kv_len], dtype=torch.int32, device=device)
            q_offset_tensor = torch.tensor([q_offset], dtype=torch.int32, device=device)

            workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            wrapper = BatchBlockExtendRaggedOffsetWrapper(workspace, kv_layout="NHD", dllm_block_size=dllm_block_size, backend="fa2")
            wrapper.plan(
                qo_indptr=qo_indptr, kv_indptr=kv_indptr,
                num_qo_heads=num_heads, num_kv_heads=num_kv_heads,
                head_dim=head_dim, q_data_type=dtype,
                sm_scale=sm_scale, q_offsets=q_offset_tensor,
            )
            bbe_output = wrapper.run(q_chunk, k_cumul, v_cumul)
            step_diff = (bbe_output - ref_output).abs().max().item()
            step_diffs.append(step_diff)

        max_diff = max(step_diffs)
        passed = max_diff < tol

        results.append({"config_idx": cfg_idx, **cfg, "max_diff": max_diff, "step_diffs": step_diffs, "pass": passed})
        if verbose:
            status = "PASS" if passed else "FAIL"
            print(f"  [{cfg_idx}] B={dllm_block_size} steps={num_steps} h={num_heads}/{num_kv_heads} d={head_dim}: max_diff={max_diff:.6f} [{status}]")

        torch.cuda.empty_cache()

    overall_pass = all(r["pass"] for r in results)
    assert overall_pass, f"Cascade precision alignment failed: {[r for r in results if not r['pass']]}"
    return {"results": results, "overall_pass": overall_pass}


def test_sglang_vs_block_extend_cascade(
    verbose: bool = True,
):
    """SGLang Cascade vs Block Extend precision alignment."""
    device = torch.device("cuda:0")
    available_backends = get_available_backends(device)
    cascade_backend = "fa3" if "fa3" in available_backends else "fa2"
    dtype = torch.float16
    tol = 1e-2

    dllm_block_size = 32
    num_heads = 32
    num_kv_heads = 8
    head_dim = 128
    sm_scale = 1.0 / math.sqrt(head_dim)
    num_steps = 4

    total_tokens = num_steps * dllm_block_size
    q_all = torch.randn(total_tokens, num_heads, head_dim, dtype=dtype, device=device)
    k_all = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)
    v_all = torch.randn(total_tokens, num_kv_heads, head_dim, dtype=dtype, device=device)

    # Reference: block_extend step-by-step
    ref_outputs = []
    for step_idx in range(num_steps):
        q_offset = step_idx * dllm_block_size
        kv_len = (step_idx + 1) * dllm_block_size
        q_chunk = q_all[q_offset:q_offset + dllm_block_size]
        k_cumul = k_all[:kv_len]
        v_cumul = v_all[:kv_len]
        ref_outputs.append(compute_block_extend_reference(q_chunk, k_cumul, v_cumul, dllm_block_size, q_offset=q_offset, sm_scale=sm_scale))

    # SGLang Cascade: current + prefix + merge_state
    cascade_outputs = []
    for step_idx in range(num_steps):
        q_offset = step_idx * dllm_block_size
        kv_len = (step_idx + 1) * dllm_block_size
        q_chunk = q_all[q_offset:q_offset + dllm_block_size]

        # Current chunk
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
        current_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD", backend=cascade_backend)
        qo_indptr = torch.tensor([0, dllm_block_size], dtype=torch.int32, device=device)
        kv_indptr = torch.tensor([0, dllm_block_size], dtype=torch.int32, device=device)
        current_wrapper.plan(qo_indptr=qo_indptr, kv_indptr=kv_indptr, num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim_qk=head_dim, causal=False, sm_scale=sm_scale)
        current_out, current_lse = current_wrapper.run(q_chunk, k_all[q_offset:kv_len], v_all[q_offset:kv_len], return_lse=True)

        if step_idx == 0:
            cascade_outputs.append(current_out)
        else:
            prefix_len = q_offset
            workspace2 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=device)
            prefix_wrapper = BatchPrefillWithRaggedKVCacheWrapper(workspace2, kv_layout="NHD", backend=cascade_backend)
            kv_indptr_prefix = torch.tensor([0, prefix_len], dtype=torch.int32, device=device)
            prefix_wrapper.plan(qo_indptr=qo_indptr, kv_indptr=kv_indptr_prefix, num_qo_heads=num_heads, num_kv_heads=num_kv_heads, head_dim_qk=head_dim, causal=False, sm_scale=sm_scale)
            prefix_out, prefix_lse = prefix_wrapper.run(q_chunk, k_all[:prefix_len], v_all[:prefix_len], return_lse=True)
            merged_out, _ = merge_state(current_out, current_lse, prefix_out, prefix_lse)
            cascade_outputs.append(merged_out)

    step_diffs = [(ref_outputs[i] - cascade_outputs[i]).abs().max().item() for i in range(num_steps)]
    max_diff = max(step_diffs)
    precision_ok = max_diff < tol

    if verbose:
        for i, d in enumerate(step_diffs):
            print(f"  step {i}: max_diff = {d:.6f}")
        status = "PASS" if precision_ok else "FAIL"
        print(f"  Overall: max_diff={max_diff:.6f} [{status}]")

    assert precision_ok, f"SGLang vs Block Extend cascade precision failed: max_diff={max_diff}"
    return {"precision_ok": precision_ok, "max_diff": max_diff, "step_diffs": step_diffs}