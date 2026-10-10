"""Isolated service experiment for fused block-scaled FP8 prefill.

Uses the existing preparation hook. Decode remains on the configured backend;
prefill is eager, so installing this after decode graph capture is intentional.
"""

import torch


def create_hook(config):
    from materialized_fp8_quantize import prepare
    import materialized_fp8_interface as candidate
    from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
    from sglang.srt.layers.attention.hybrid_attn_backend import HybridAttnBackend

    threshold = int(config.get("minimum_query_tokens", 1024))
    old_extend = FlashAttentionBackend.forward_extend

    def prepare_prefill_qkv(
        self, *, q, q_pe, kv_a, k_pe, positions, layer, forward_batch
    ):
        backend = self.prefill_backend
        assert isinstance(backend, FlashAttentionBackend) and backend.fa_impl_ver == 4
        assert not layer.use_dsa and layer.num_local_heads == 20
        assert (
            layer.qk_nope_head_dim == 192
            and layer.qk_rope_head_dim == 64
            and layer.v_head_dim == 256
        )
        assert q.dtype == torch.bfloat16
        if layer.rotary_emb is not None:
            q_pe, k_pe = layer.rotary_emb(positions, q_pe, k_pe)
        q[..., 192:] = q_pe
        # CUDA's existing cache setter does not use latent_cache.
        layer._set_mla_kv_buffer(None, kv_a, k_pe, forward_batch)
        if forward_batch.mha_one_shot and any(forward_batch.extend_prefix_lens_cpu):
            kv_a, k_pe = layer._get_mla_kv_buffer(
                forward_batch.fetch_mha_one_shot_kv_indices(), q.dtype, forward_batch
            )
        kv = layer.kv_b_proj(kv_a)[0].view(-1, 20, 448)
        k_nope, v = kv[..., :192], kv[..., 192:]
        eligible = (
            forward_batch.mha_one_shot
            and q.shape[0] >= threshold
            and q.shape[0] == forward_batch.extend_num_tokens
            and getattr(forward_batch, "spec_info", None) is None
        )
        if not eligible:
            return q, layer._concat_and_cast_mha_k(k_nope, k_pe, forward_batch), v
        metadata = backend.forward_metadata
        q8, k8, v8, scales = prepare(
            q,
            k_nope,
            k_pe,
            v,
            metadata.cu_seqlens_q,
            metadata.cu_seqlens_k,
            metadata.max_seq_len_q,
            metadata.max_seq_len_k,
            64,
            64,
        )
        forward_batch._sm120_union_prefill = (
            layer.layer_id,
            q8.shape[0],
            k8.shape[0],
            scales,
        )
        if layer.layer_id == 0:
            print("SM120_K32_PREFILL", q8.shape[0], k8.shape[0], flush=True)
        return q8, k8, v8

    def forward_extend(
        self, q, k, v, layer, forward_batch, save_kv_cache=True, **kwargs
    ):
        if q.dtype != torch.float8_e4m3fn:
            return old_extend(
                self, q, k, v, layer, forward_batch, save_kv_cache, **kwargs
            )
        assert not save_kv_cache and forward_batch.mha_one_shot
        assert (
            not forward_batch.mha_return_lse
            and forward_batch.attn_attend_prefix_cache is False
        )
        assert kwargs.get("q_rope") is None and kwargs.get("k_rope") is None
        layer_id, nq, nk, scales = forward_batch._sm120_union_prefill
        assert layer_id == layer.layer_id and q.shape[0] == nq and k.shape[0] == nk
        metadata = self.forward_metadata
        output = candidate._flash_attn_fwd(
            q.view(-1, 20, 256),
            k.view(-1, 20, 256),
            v.view(-1, 20, 256),
            cu_seqlens_q=metadata.cu_seqlens_q,
            cu_seqlens_k=metadata.cu_seqlens_k,
            max_seqlen_q=metadata.max_seq_len_q,
            max_seqlen_k=metadata.max_seq_len_k,
            softmax_scale=layer.scaling,
            causal=True,
            tile_mn=(64, 64),
            aux_tensors=scales,
        )[0]
        assert output.dtype == torch.bfloat16
        del forward_batch._sm120_union_prefill
        if getattr(forward_batch, "_attn_output", None) is not None:
            target = forward_batch._attn_output.view_as(output)
            assert target.dtype == torch.bfloat16
            target.copy_(output)
            return target
        return output

    HybridAttnBackend.prepare_prefill_qkv = prepare_prefill_qkv
    FlashAttentionBackend.forward_extend = forward_extend
    print("SM120_K32_PREFILL_INSTALLED", threshold, flush=True)

    def hook(module, inputs, output):
        pass

    return hook
