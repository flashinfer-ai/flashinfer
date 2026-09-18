# SPDX-License-Identifier: Apache-2.0
"""Example-local adapters to existing FlashInfer kernels, not a new public API.

All modes receive BF16 BSHD tensors, compute only the valid prefix and return
BF16 BSHD. Quantization/layout conversion is part of run(), not setup/timing
exclusions. FP8 uses a fixed positive scalar dequantization factor shared by
all heads/ranks so head chunking does not change the quantization recipe.
"""

import math

import torch

BACKENDS = {
    "fa3-bf16": (9, 0),
    "fa3-fp8": (9, 0),
    "trtllm-bf16": (10, 0),
    "trtllm-fp8": (10, 0),
    "sm120-bf16": (12, 0),
    "sm120-fp8": (12, 0),
    "sm120-sage": (12, 0),
}


def validate_backend(backend, capability, sequence, used, heads, dim, fp8_scale):
    if backend not in BACKENDS or tuple(capability) != BACKENDS[backend]:
        raise ValueError(f"{backend} is not admitted on SM{capability}")
    if any(type(v) is not int or v <= 0 for v in (sequence, used, heads, dim)):
        raise ValueError("positive integer geometry is required")
    if used > sequence or dim != 128:
        raise ValueError("example requires D=128 and 0 < used <= physical sequence")
    if not math.isfinite(fp8_scale) or fp8_scale <= 0:
        raise ValueError("FP8 dequantization scale must be finite and positive")


class NativeAttention:
    """Prepared B=1 non-causal prefix example; returned storage is overwritten.

    No auto dispatch, external FlashAttention/Sage package, fallback, or process
    group. A separate instance is needed for each concurrently consumed chunk.
    Padding-query outputs are zeroed, not computed as a second packed sequence.
    """

    def __init__(self, backend, *, sequence, used, heads, dim=128, fp8_scale=1 / 32):
        validate_backend(
            backend,
            torch.cuda.get_device_capability(),
            sequence,
            used,
            heads,
            dim,
            fp8_scale,
        )
        self.backend, self.sequence, self.used = backend, sequence, used
        self.heads, self.dim, self.scale = heads, dim, fp8_scale
        self.device = torch.device("cuda", torch.cuda.current_device())
        self.output = torch.zeros(
            1, sequence, heads, dim, device=self.device, dtype=torch.bfloat16
        )
        self.indptr = torch.tensor([0, used], device=self.device, dtype=torch.int32)
        self.lengths = self.indptr[1:]
        self.stream_id = None
        if backend.startswith(("fa3", "trtllm")):
            self.workspace = torch.empty(
                128 << 20, device=self.device, dtype=torch.uint8
            )
        if backend.startswith("fa3"):
            from flashinfer import BatchPrefillWithRaggedKVCacheWrapper

            self.wrapper = BatchPrefillWithRaggedKVCacheWrapper(
                self.workspace, "NHD", backend="fa3"
            )
            dtype = torch.float8_e4m3fn if backend.endswith("fp8") else torch.bfloat16
            self.wrapper.plan(
                self.indptr,
                self.indptr,
                heads,
                heads,
                dim,
                causal=False,
                q_data_type=dtype,
                kv_data_type=dtype,
                o_data_type=torch.bfloat16,
            )
        if backend in ("sm120-bf16", "sm120-sage"):
            # All-active blocks: this is dense semantics, NOT sparse speedup.
            self.blocks = (used + 63) // 64
            self.indices = (
                torch.arange(self.blocks, device=self.device, dtype=torch.int32)
                .view(1, 1, 1, -1)
                .expand(1, heads, self.blocks, self.blocks)
                .contiguous()
            )
            self.block_sizes = torch.full(
                (self.blocks,), 64, device=self.device, dtype=torch.int32
            )
            self.block_sizes[-1] = used - (self.blocks - 1) * 64
        self.ready = torch.cuda.Event()
        self.ready.record()

    def __call__(self, q, k, v):
        expected = (1, self.sequence, self.heads, self.dim)
        for x in (q, k, v):
            if (
                x.shape != expected
                or x.dtype != torch.bfloat16
                or x.device != self.device
                or x.requires_grad
            ):
                raise ValueError("expected inference BF16 BSHD on the prepared device")
        stream = torch.cuda.current_stream(self.device)
        if self.stream_id is not None and stream.cuda_stream != self.stream_id:
            raise RuntimeError(
                "native attention example is bound to its first calling stream"
            )
        self.stream_id = stream.cuda_stream
        stream.wait_event(self.ready)
        q, k, v = [t[:, : self.used].contiguous() for t in (q, k, v)]
        output = self.output[:, : self.used]
        if self.backend.endswith("fp8"):
            q, k, v = [
                (t.float() / self.scale).clamp(-448, 448).to(torch.float8_e4m3fn)
                for t in (q, k, v)
            ]
        if self.backend.startswith("fa3"):
            scales = (self.scale,) * 3 if self.backend.endswith("fp8") else ()
            self.wrapper.run(q[0], k[0], v[0], *scales, out=output[0])
        elif self.backend.startswith("trtllm"):
            from flashinfer.prefill import trtllm_ragged_attention_deepseek

            scale = self.scale if self.backend.endswith("fp8") else 1.0
            trtllm_ragged_attention_deepseek(
                q[0],
                k[0],
                v[0],
                self.workspace,
                self.lengths,
                max_q_len=self.used,
                max_kv_len=self.used,
                bmm1_scale=scale * scale / math.sqrt(self.dim),
                bmm2_scale=scale,
                o_sf_scale=1.0,
                batch_size=1,
                window_left=-1,
                cum_seq_lens_q=self.indptr,
                cum_seq_lens_kv=self.indptr,
                enable_pdl=False,
                is_causal=False,
                return_lse=False,
                out=output[0],
                backend="trtllm-gen",
            )
        elif self.backend == "sm120-fp8":
            from flashinfer.attention.cute_dsl.sm120_fmha import (
                sm120_fmha_fp8_ragged_prefill,
            )

            sm120_fmha_fp8_ragged_prefill(
                q[0],
                k[0],
                v[0],
                output[0],
                self.indptr,
                self.indptr,
                max_seqlen_q=self.used,
                is_causal=False,
                sm_scale=self.scale * self.scale / math.sqrt(self.dim),
                v_scale=self.scale,
            )
        elif self.backend == "sm120-bf16":
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
                bsa_attn_sm120_blk64_fwd,
            )

            bsa_attn_sm120_blk64_fwd(
                q,
                k,
                v,
                self.indices,
                self.blocks,
                block_sizes=self.block_sizes,
                out=output,
            )
        else:
            from flashinfer.cute_dsl.sparse.bsa_attn_sm120 import (
                bsa_attn_sm120_blk64_sage_fwd,
            )
            from flashinfer.cute_dsl.sparse.bsa_utils.sage_quant_sm120 import (
                quantize_sage_qkv_sm120,
            )

            quantized = quantize_sage_qkv_sm120(
                *(t.transpose(1, 2).contiguous() for t in (q, k, v))
            )
            result = bsa_attn_sm120_blk64_sage_fwd(
                *quantized,
                self.indices,
                self.blocks,
                block_sizes=self.block_sizes,
                backend="cute_dsl",
            )
            output.copy_(result.transpose(1, 2))
        return self.output
