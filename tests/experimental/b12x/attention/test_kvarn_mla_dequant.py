"""Decode-math parity probe for the KVarN dual-rate record arms.

Builds a paged kvarn cache with the b12x writer, then runs a minimal CuTe
kernel that stages selected records into smem exactly like ``io_issue_gather``
(scalar u32 copies + kv_sc page-tail Int64 addresses) and dequants them with
the production ``_kvarn_k_pair_bf16`` / ``_kvarn_v_pair_bf16`` primitives.
Compares against the pure-torch ``dequant_kvarn_mla_records`` helper (already
parity-verified against the vLLM reference codec).

Run: CUDA_VISIBLE_DEVICES=2 CUDA_HOME=/opt/cuda ~/lil/vllm/.venv/bin/python \
     tests/attention/test_kvarn_mla_dequant.py
"""

import os
import sys

import torch

REPO = os.environ.get("B12X_REPO", os.path.expanduser("~/.cache/b12x-kvarn-wt"))
VLLM = os.environ.get("VLLM_REPO", os.path.expanduser("~/lil/vllm"))
for p in (REPO, VLLM):
    if p not in sys.path:
        sys.path.insert(0, p)

import cutlass  # noqa: E402
from cutlass import Uint64  # noqa: E402
import cutlass.cute as cute  # noqa: E402
import cuda.bindings.driver as cuda  # noqa: E402
from cutlass import Float32, Int32, Int64, Uint32  # noqa: E402

from b12x._lib.intrinsics import (  # noqa: E402
    ld_global_nc_u32,
    ld_shared_u32,
    st_global_u32,
    st_shared_u32,
)
from b12x.attention._shared.mla.decode_math import (  # noqa: E402
    _kvarn_k_pair_bf16,
    _kvarn_v_pair_bf16,
)
from b12x.attention._shared.mla.decode_math import (  # noqa: E402
    _kvarn_code_u32,
    _ld_u16_zext as _ld_u16_probe,
)
from b12x.attention._shared.mla._kvarn_writer import (  # noqa: E402
    concat_and_cache_glm_next_mla_kvarn,
    dequant_kvarn_mla_records,
    kvarn_record_bytes,
)

GROUP = 256
LATENT = 512
NE = 8  # probed records (simulated top-k candidates)
THREADS = 256
DEVICE = "cuda"


class KvarnDequantProbe:
    """Stage NE records like the io gather, dequant with the math arms."""

    def __init__(self, k_bits: int, v_bits: int):
        self.k_bits = k_bits
        self.v_bits = v_bits
        self.record_bytes = kvarn_record_bytes(k_bits, v_bits)

    @cute.jit
    def __call__(
        self,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,  # [NE] int32 global record slots
        k_out: cute.Tensor,  # [NE, 256] int32 (bf16x2 pairs)
        v_out: cute.Tensor,
        block_stride: Int64,
        entry_stride: Int64,
        stream: cuda.CUstream,
    ):
        self.kernel(
            kv_cache,
            slot_mapping,
            k_out,
            v_out,
            block_stride,
            entry_stride,
        ).launch(grid=(1, 1, 1), block=[THREADS, 1, 1], stream=stream)

    @cute.kernel
    def kernel(
        self,
        kv_cache: cute.Tensor,
        slot_mapping: cute.Tensor,
        k_out: cute.Tensor,
        v_out: cute.Tensor,
        block_stride: Int64,
        entry_stride: Int64,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        tid = Int32(tidx)

        smem_ptr = cute.arch.alloc_smem(
            Float32, (NE * self.record_bytes + NE * 4 + 3) // 4
        )
        smem = smem_ptr.toint()
        rec_smem = smem  # NE records, record_bytes each (8B-aligned)
        tok_smem = smem + Int32(NE * self.record_bytes)  # NE int32 slots

        # -- stage records + slots (mirrors io_issue_gather kvarn). --
        total_words = (NE * self.record_bytes) // 4
        trips = (total_words + THREADS - 1) // THREADS
        for it in cutlass.range(Int32(trips)):
            idx = it * Int32(THREADS) + tid
            if idx < Int32(total_words):
                e = idx // Int32(self.record_bytes // 4)
                word = idx - e * Int32(self.record_bytes // 4)
                slot = Int64(slot_mapping[e])
                block_idx = slot // Int64(GROUP)
                local = slot - block_idx * Int64(GROUP)
                rec_addr = (
                    get_ptr_as_int64_fallback(kv_cache)
                    + block_idx * block_stride
                    + local * entry_stride
                )
                val = ld_global_nc_u32(rec_addr + Int64(word) * Int64(4))
                st_shared_u32(rec_smem + idx * Int32(4), val)
        if tid < Int32(NE):
            st_shared_u32(tok_smem + tid * Int32(4), Uint32(slot_mapping[tid]))
        cute.arch.fence_acq_rel_cta()
        cute.arch.barrier()

        # -- dequant: each thread owns pair columns (tid*2, tid*2+1) per entry. --
        cache_base = get_ptr_as_int64_fallback(kv_cache)
        for e in cutlass.range(Int32(NE), unroll=1):
            c0 = tid * Int32(2)
            entry_smem = rec_smem + e * Int32(self.record_bytes)
            slot = Int64(max(ld_shared_u32(tok_smem + e * Int32(4)), Int32(0)))
            kpair = _kvarn_k_pair_bf16(
                rec_smem,
                e,
                c0,
                bits=self.k_bits,
                record_bytes=self.record_bytes,
                k_code_bytes=(LATENT * self.k_bits + 7) // 8,
                v_code_bytes=(LATENT * self.v_bits + 7) // 8,
            )
            vpair = _kvarn_v_pair_bf16(
                rec_smem,
                e,
                c0,
                k_bits=self.k_bits,
                v_bits=self.v_bits,
                record_bytes=self.record_bytes,
                k_code_bytes=(LATENT * self.k_bits + 7) // 8,
                v_code_bytes=(LATENT * self.v_bits + 7) // 8,
            )
            out_word = e * Int32(256) + tid
            st_global_u32(
                get_ptr_as_int64_fallback(k_out) + Int64(out_word) * Int64(4), kpair
            )
            st_global_u32(
                get_ptr_as_int64_fallback(v_out) + Int64(out_word) * Int64(4), vpair
            )
            if e == Int32(1) and tid == Int32(0):
                rec_bytes = Int32(self.record_bytes)
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out)
                    + Int64(NE) * Int64(1024)
                    + Int64(20),
                    ld_shared_u32(entry_smem + rec_bytes - Int32(8)),
                )
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out)
                    + Int64(NE) * Int64(1024)
                    + Int64(24),
                    ld_shared_u32(entry_smem + rec_bytes - Int32(4)),
                )
            if e == Int32(0) and tid == Int32(0):
                # Debug probe: first record word + record gmem address.
                rec_addr = cache_base + slot * entry_stride
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out) + Int64(NE) * Int64(1024),
                    ld_global_nc_u32(rec_addr),
                )
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out)
                    + Int64(NE) * Int64(1024)
                    + Int64(8),
                    ld_global_nc_u32(rec_addr + Int64(4)),
                )
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out)
                    + Int64(NE) * Int64(1024)
                    + Int64(12),
                    Uint32(rec_addr.to(Uint64) & Uint64(0xFFFFFFFF)),
                )
                st_global_u32(
                    get_ptr_as_int64_fallback(k_out)
                    + Int64(NE) * Int64(1024)
                    + Int64(16),
                    Uint32((rec_addr.to(Uint64) >> Uint32(32)) & Uint64(0xFFFFFFFF)),
                )


def get_ptr_as_int64_fallback(tensor: cute.Tensor) -> Int64:
    from b12x._lib.intrinsics import get_ptr_as_int64

    return get_ptr_as_int64(tensor, Int64(0))


def run_variant(k_bits: int, v_bits: int) -> None:
    from b12x.attention._shared.mla._kvarn_writer import _to_kernel_tensor

    record_bytes = kvarn_record_bytes(k_bits, v_bits)
    pages, gen = 2, torch.Generator(device="cpu").manual_seed(11)
    latent = torch.randn(
        pages * GROUP, LATENT, generator=gen, dtype=torch.bfloat16
    ).cuda()
    slots = torch.arange(pages * GROUP, dtype=torch.int32).cuda()
    block_bytes = GROUP * record_bytes
    cache = torch.zeros(pages * block_bytes, dtype=torch.uint8, device=DEVICE)
    kv_cache = cache.as_strided(
        (pages, GROUP, record_bytes), (block_bytes, record_bytes, 1)
    )

    concat_and_cache_glm_next_mla_kvarn(
        latent, kv_cache, slots, k_bits=k_bits, v_bits=v_bits, group=GROUP
    )
    torch.cuda.synchronize()

    # Select NE spread records across both pages (simulated top-k).
    probe_slots = torch.tensor(
        [0, 1, 63, 100, 255, 256, 300, 511], dtype=torch.int32, device=DEVICE
    )
    k_out = torch.zeros(NE + 2, 256, dtype=torch.int32, device=DEVICE)
    v_out = torch.zeros(NE + 2, 256, dtype=torch.int32, device=DEVICE)

    probe = KvarnDequantProbe(k_bits, v_bits)
    args = (
        _to_kernel_tensor(kv_cache, assumed_align=16, leading_dim=2),
        _to_kernel_tensor(probe_slots, assumed_align=4, leading_dim=0),
        _to_kernel_tensor(k_out, assumed_align=4, leading_dim=1),
        _to_kernel_tensor(v_out, assumed_align=4, leading_dim=1),
        Int64(block_bytes),
        Int64(record_bytes),
    )
    from b12x._lib.utils import current_cuda_stream

    probe(*args, stream=current_cuda_stream())
    torch.cuda.synchronize()

    # Reference: dequant each probed record + its page tail (torch helper).
    flat = cache.view(-1)
    recs = []
    for s_val in probe_slots.cpu().tolist():
        page, local = s_val // GROUP, s_val % GROUP
        base = page * block_bytes
        recs.append(
            flat[base + local * record_bytes : base + (local + 1) * record_bytes]
        )
    k_refs, v_refs = [], []
    for rec_i in recs:
        ki, vi = dequant_kvarn_mla_records(rec_i.unsqueeze(0), k_bits, v_bits)
        k_refs.append(ki[0])
        v_refs.append(vi[0])
    ref_k = torch.stack(k_refs).cpu()
    ref_v = torch.stack(v_refs).cpu()
    got_k = k_out[:NE].view(torch.bfloat16).float().reshape(NE, 512).cpu()
    got_v = v_out[:NE].view(torch.bfloat16).float().reshape(NE, 512).cpu()

    per_slot_k = (got_k - ref_k).abs().max(dim=1).values
    per_slot_v = (got_v - ref_v).abs().max(dim=1).values
    k_err = per_slot_k.max().item()
    v_err = per_slot_v.max().item()
    print(
        f"k{k_bits}v{v_bits}: max_k_err={k_err:.4g} max_v_err={v_err:.4g} "
        f"ref_scale k={ref_k.abs().max().item():.3f} v={ref_v.abs().max().item():.3f}"
    )
    print(
        "  per-slot k err:",
        [
            f"{s_val}:{x:.3g}"
            for s_val, x in zip(probe_slots.cpu().tolist(), per_slot_k.tolist())
        ],
    )
    print(
        "  per-slot v err:",
        [
            f"{s_val}:{x:.3g}"
            for s_val, x in zip(probe_slots.cpu().tolist(), per_slot_v.tolist())
        ],
    )
    print("  got_k[0,:4]", got_k[0, :4].tolist(), "ref_k[0,:4]", ref_k[0, :4].tolist())
    print("  got_v[0,:4]", got_v[0, :4].tolist(), "ref_v[0,:4]", ref_v[0, :4].tolist())
    print("  got_v[1,:8]", [f"{x:.4g}" for x in got_v[1, :8].tolist()])
    print("  ref_v[1,:8]", [f"{x:.4g}" for x in ref_v[1, :8].tolist()])
    print("  got_k[1,:8]", [f"{x:.4g}" for x in got_k[1, :8].tolist()])
    print("  ref_k[1,:8]", [f"{x:.4g}" for x in ref_k[1, :8].tolist()])
    dbg = k_out.reshape(-1)[NE * 256 + 8 : NE * 256 + 12].tolist()
    for t in range(4):
        w = dbg[t] & 0xFFFFFFFF
        print(f"  e1 c{2 * t}: code={w & 0xFFFF} vscale_bits={hex((w >> 16) & 0xFFFF)}")
    smem_head = k_out.reshape(-1)[NE * 256 + 12 : NE * 256 + 16].tolist()
    print("  smem rec[1] words0-3:", [hex(x & 0xFFFFFFFF) for x in smem_head])
    host_r1 = cache[
        probe_slots[1].item() * record_bytes : probe_slots[1].item() * record_bytes + 16
    ]
    print(
        "  host rec[1] words0-3:",
        [
            hex(
                int.from_bytes(
                    bytes(host_r1[i * 4 : (i + 1) * 4].cpu().tolist()), "little"
                )
            )
            for i in range(4)
        ],
    )
    host_rec1 = cache[
        probe_slots[1].item() * record_bytes : probe_slots[1].item() * record_bytes
        + record_bytes
    ]
    print(
        "  host rec[1] tail-8B:",
        host_rec1[-8:-4].tolist(),
        host_rec1[-4:].tolist(),
        "->",
        hex(int.from_bytes(bytes(host_rec1[-8:-4].cpu().tolist()), "little")),
        hex(int.from_bytes(bytes(host_rec1[-4:].cpu().tolist()), "little")),
    )
    tol = 0.02 * max(ref_k.abs().max().item(), 1e-3)
    assert k_err <= tol, f"K dequant mismatch {k_err} > {tol}"
    vtol = 0.02 * max(ref_v.abs().max().item(), 1e-3)
    assert v_err <= vtol, f"V dequant mismatch {v_err} > {vtol}"


if __name__ == "__main__":
    for kb, vb in ((4, 4), (5, 5), (4, 2)):
        run_variant(kb, vb)
    print("ALL OK")
