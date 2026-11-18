"""EP4 MegaMoE parameters of the GB200 perf guards, indexed by model, quantization and tokens per rank.

Every workload is the DeepSeek V4 Pro Rubin guard's contract (ci/deepseek_v4_pro_rubin_nvfp4/best_params.py
BASE_PARAMS: EP4, balanced routing, BF16 combine, fused pull dispatch, seed 45494) with the model's expert
shape and activation:

| model | experts / hidden / intermediate / topk | activation |
|---|---|---|
| deepseek_v4_pro | 384 / 7168 / 3072 / 6 | SwiGLU, gate/up clamp 10 |
| deepseek_v4_1_flash | 384 / 5120 / 2304 / 6 | SwiGLU, gate/up clamp 10 |
| kimi_k3 | 896 / 3584 / 3072 / 16 | SiTU (beta 4, linear beta 25), sigmoid on tanh.approx |

The per-token settings start from the GB200 configurations used for the EP4 performance tables so far
(DeepSeek V4 Pro and V4.1 Flash) and from the VR200 Kimi K3 guard's (Kimi K3). A configuration from a search on gb200nvl72_preprod over the settings the
unmodified main runner supports replaced one only where it measured faster in an A/B on four nodes
(README.md). Use get_params() to see the complete arguments for one case. Importing this module needs no
CUDA.
"""
from .rubin_pro_best_params import BASE_PARAMS as RUBIN_V4_PRO_BASE_PARAMS

MODELS = ('deepseek_v4_pro', 'deepseek_v4_1_flash', 'kimi_k3')
QUANTS = ('nvfp4', 'mxfp4_mxfp8')
# Shared workload and default kernel arguments per model. None disables an optional flag.
WORKLOADS = {
    "deepseek_v4_pro": dict(RUBIN_V4_PRO_BASE_PARAMS),
    "deepseek_v4_1_flash": dict(RUBIN_V4_PRO_BASE_PARAMS, hidden=5120, intermediate=2304),
    # SiTU with its sigmoid on tanh.approx (use_tanh_sigmoid), as the VR200 Kimi K3 guard runs it.
    "kimi_k3": dict(RUBIN_V4_PRO_BASE_PARAMS, hidden=3584, intermediate=3072, experts=896, topk=16,
                    gateup_clamp="situ_b4_lb25", use_tanh_sigmoid=True),
}


def _cfg(tile, cluster, schedule, fc2_tma_stages, epi_flag_batches, token_in_flag_batch):
    """One token count's kernel settings; 4-CTA clusters fall back to 2-CTA ones."""
    return dict(mma_tiler_mnk=tile, cluster_shape_mnk=cluster,
                fallback_cluster_shape_mnk="2,1,1" if cluster == "4,1,1" else None,
                schedule_policy=schedule, fc2_tma_stages=fc2_tma_stages,
                epi_flag_batches=epi_flag_batches, token_in_flag_batch=token_in_flag_batch)


# tile, cluster, schedule policy, FC2 TMA stages, epilogue flag batches, token-in flag batch
BEST_PARAMS = {
    "deepseek_v4_pro": {
        "nvfp4": {
            128: _cfg("256,64,128", "4,1,1", "phase_interleave,4", 1, "1,4", 1),
            256: _cfg("256,128,256", "2,1,1", "phase_interleave,4", 2, "1,4", 1),
            512: _cfg("256,64,128", "4,1,1", "phase_interleave,4", 1, "1,4", 1),
            1024: _cfg("256,128,128", "4,1,1", "phase_interleave,4", 1, "1,4", 1),
            2048: _cfg("256,256,256", "4,1,1", "phase_interleave,3", 1, "1,4", 2),
            4096: _cfg("256,256,256", "4,1,1", "phase_interleave", 1, "1,1", 4),
            8192: _cfg("256,256,256", "4,1,1", "phase_interleave,3", 1, "2,4", 1),
            16384: _cfg("256,256,256", "4,1,1", "phase_interleave,3", 1, "2,4", 4),
            32768: _cfg("256,256,256", "4,1,1", "phase_interleave", 1, "1,4", 4),
        },
        "mxfp4_mxfp8": {
            128: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "2,4", 1),
            256: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "2,4", 1),
            512: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "1,4", 1),
            1024: _cfg("256,128,128", "2,1,1", "phase_interleave,4", 1, "1,1", 1),
            2048: _cfg("256,256,128", "4,1,1", "phase_interleave,3", 1, "1,2", 1),
            4096: _cfg("256,256,128", "4,1,1", "phase_interleave,2", 1, "1,4", 1),
            8192: _cfg("256,256,128", "4,1,1", "phase_interleave,3", 1, "1,1", 4),
            16384: _cfg("256,256,128", "4,1,1", "phase_interleave,3", 1, "1,4", 2),
            32768: _cfg("256,256,128", "4,1,1", "phase_interleave,2", 1, "1,1", 2),
        },
    },
    "deepseek_v4_1_flash": {
        "nvfp4": {
            128: _cfg("256,128,256", "2,1,1", "phase_interleave,4", 1, "1,2", 1),
            256: _cfg("256,128,128", "4,1,1", "phase_interleave,4", 2, "1,1", 1),
            512: _cfg("256,128,128", "4,1,1", "phase_interleave,4", 2, "1,1", 1),
            1024: _cfg("256,128,256", "4,1,1", "phase_interleave,4", 1, "1,1", 1),
            2048: _cfg("256,128,256", "4,1,1", "phase_interleave,3", 1, "2,4", 1),
            4096: _cfg("256,256,256", "4,1,1", "phase_interleave,2", 1, "1,4", 1),
            8192: _cfg("256,256,256", "4,1,1", "phase_interleave,2", 1, "2,4", 1),
            16384: _cfg("256,256,256", "4,1,1", "phase_interleave,3", 1, "1,1", 2),
            32768: _cfg("256,256,256", "4,1,1", "phase_interleave,2", 1, "1,2", 1),
        },
        "mxfp4_mxfp8": {
            128: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "2,4", 1),
            256: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "2,4", 1),
            512: _cfg("256,64,128", "2,1,1", "phase_interleave,4", 1, "1,1", 1),
            1024: _cfg("256,128,128", "2,1,1", "phase_interleave,4", 1, "1,2", 1),
            2048: _cfg("256,128,128", "4,1,1", "phase_interleave,2", 1, "1,4", 1),
            4096: _cfg("256,256,128", "4,1,1", "phase_interleave,3", 1, "1,2", 4),
            8192: _cfg("256,256,128", "4,1,1", "phase_interleave", 2, "1,1", 4),
            16384: _cfg("256,256,128", "4,1,1", "phase_interleave", 1, "1,4", 2),
            32768: _cfg("256,256,128", "4,1,1", "phase_interleave,2", 1, "1,1", 4),
        },
    },
    "kimi_k3": {
        "nvfp4": {
            128: _cfg("256,128,256", "4,1,1", "phase_interleave,4", 1, "1,4", 1),
            256: _cfg("256,128,256", "4,1,1", "phase_interleave,4", 1, "1,1", 1),
            512: _cfg("256,128,256", "4,1,1", "phase_interleave,4", 1, "1,4", 1),
            1024: _cfg("256,128,256", "4,1,1", "phase_interleave,3", 1, "1,2", 1),
            2048: _cfg("256,256,256", "4,1,1", "phase_interleave", 1, "2,4", 1),
            4096: _cfg("256,128,256", "4,1,1", "phase_interleave,4", 1, "1,2", 1),
            8192: _cfg("256,256,256", "4,1,1", "phase_interleave", 2, "1,1", 2),
            16384: _cfg("256,256,256", "4,1,1", "phase_interleave,3", 2, "1,1", 1),
            32768: _cfg("256,256,256", "4,1,1", "phase_interleave,4", 2, "1,4", 1),
        },
        "mxfp4_mxfp8": {
            128: _cfg("256,64,128", "4,1,1", "phase_interleave,4", 1, "2,4", 1),
            256: _cfg("256,64,128", "4,1,1", "phase_interleave,4", 1, "2,4", 1),
            512: _cfg("256,64,128", "4,1,1", "phase_interleave", 1, "2,4", 1),
            1024: _cfg("256,128,128", "4,1,1", "phase_interleave", 1, "2,4", 1),
            2048: _cfg("256,256,128", "4,1,1", "phase_interleave,4", 1, "1,2", 1),
            4096: _cfg("256,256,128", "4,1,1", "phase_interleave", 1, "1,4", 1),
            8192: _cfg("256,256,128", "4,1,1", "phase_interleave", 1, "1,2", 4),
            16384: _cfg("256,256,128", "4,1,1", "phase_interleave", 1, "1,1", 4),
            32768: _cfg("256,256,128", "4,1,1", "phase_interleave,3", 2, "1,1", 2),
        },
    },
}


def get_params(model, quant_kind, tokens_per_rank):
    """Return a fresh dictionary with the complete runner arguments for one case."""
    return (dict(WORKLOADS[model], max_tokens_per_rank=tokens_per_rank, quant_kind=quant_kind)
            | BEST_PARAMS[model][quant_kind][tokens_per_rank])
