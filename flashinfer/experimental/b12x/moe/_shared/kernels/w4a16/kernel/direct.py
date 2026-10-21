"""W4A16 small-M direct MoE kernels."""

from __future__ import annotations

from b12x.moe._shared.kernels.micro.kernel import MoEMicroKernelBackend
from b12x.moe._shared.kernels.w4a16.kernel.config import _W4A16_SMALL_M_DIRECT_MAX_M


class MoEMicroKernelW4A16SmallMDirect(MoEMicroKernelBackend):
    """Decode-sized W4A16 specialization using the native ModelOpt layout."""

    _SUPPORTED_M = tuple(range(1, _W4A16_SMALL_M_DIRECT_MAX_M + 1))

    @classmethod
    def is_supported(
        cls,
        *,
        m: int,
        hidden_size: int,
        intermediate_size: int,
        topk: int,
        num_experts: int,
        scale_format: str = "e4m3_k16",
    ) -> bool:
        # E8M0 reads the shared packed scale grid. Both block axes must be /32.
        # The FC2 chunking (256 intermediate values per chunk, fc2_n_chunks)
        # masks a tail inside a single chunk correctly (I=128 oracle-covered),
        # but multi-chunk shards with a partial last chunk (352, 384) index
        # scale columns past the e8m0 grid and corrupt decode outputs — the
        # e4m3_k16 path masks this fine; e8m0 must cover multi-chunk exactly.
        if scale_format == "e8m0_k32":
            fc2_n_chunks = ((int(intermediate_size) // 2) + 127) // 128
            scale_block_ok = (
                int(hidden_size) % 32 == 0
                and int(intermediate_size) % 32 == 0
                and (fc2_n_chunks == 1 or int(intermediate_size) % 256 == 0)
            )
        else:
            scale_block_ok = True
        return (
            int(m) in cls._SUPPORTED_M
            and int(m) <= _W4A16_SMALL_M_DIRECT_MAX_M
            and int(hidden_size) > 0
            and int(hidden_size) % 128 == 0
            and int(intermediate_size) > 0
            and int(intermediate_size) % 16 == 0
            and scale_block_ok
            and 0 < int(topk) <= 32
            and int(num_experts) > 0
            and MoEMicroKernelBackend.is_supported(
                int(m),
                int(hidden_size),
                int(intermediate_size),
                int(topk),
                int(num_experts),
            )
        )

    def __init__(
        self,
        *,
        activation: str,
        fast_math: bool,
        share_input_across_experts: bool,
        share_expert_scales: bool,
        single_token: bool,
        scale_format: str = "e4m3_k16",
        swiglu_limit: float | None = None,
        swiglu_clamp_after_silu: bool = False,
        swiglu_alpha: float | None = None,
        swiglu_beta: float | None = None,
        w13_layout: str = "w13",
        compile_time_phase: int = 0,
    ):
        super().__init__(
            sf_vec_size=16,
            mma_tiler_mn=(64, 128),
            output_tile_count_n=1,
            fast_math=fast_math,
            activation=activation,
            share_input_across_experts=share_input_across_experts,
            share_expert_scales=share_expert_scales,
            single_token=single_token,
            dynamic_down_scale=False,
            w4a16_mode=True,
            scale_format=scale_format,
            e4m3_scale_layout="modelopt",
            swiglu_limit=swiglu_limit,
            swiglu_clamp_after_silu=swiglu_clamp_after_silu,
            swiglu_alpha=swiglu_alpha,
            swiglu_beta=swiglu_beta,
            w13_layout=w13_layout,
            compile_time_phase=compile_time_phase,
        )
