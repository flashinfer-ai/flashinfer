import os
from contextlib import contextmanager
from pathlib import Path

from .generator_utils import (
    InputLayout,
    encode_name,
    enumerate_hmma_flash_kernels,
    enumerate_qmma_flash_kernels,
    generate_files,
)


@contextmanager
def working_directory(path: Path):
    """Context manager to temporarily change working directory."""
    original_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(original_dir)


def _setup_output_directory(src_target: Path, gen_dir: Path):
    """Setup output directory with symlinks to TensorRT-LLM source directories."""
    gen_dir.mkdir(parents=True, exist_ok=True)

    # Create symlink to csrc/fmha_v2/ directory
    src_link = gen_dir / "src"

    if src_link.is_symlink() or src_link.exists():
        src_link.unlink()
    src_link.symlink_to(src_target, target_is_directory=True)

    (gen_dir / "generated").mkdir(exist_ok=True)
    (gen_dir / "bin").mkdir(exist_ok=True)


def enumerate_kernels(src_target: Path, gen_dir: Path):
    # Setup output directory with symlinks to source headers
    _setup_output_directory(src_target, gen_dir)

    # Enumerate kernels, emit to generated/ directory
    with working_directory(gen_dir):
        specs: list = []
        enumerate_hmma_flash_kernels(specs, sm=120, dtype="bf16", head_size_v=128)
        enumerate_qmma_flash_kernels(specs, sm=120, dtype="e4m3_fp32", head_sizes=[192])
        enumerate_qmma_flash_kernels(
            specs, sm=120, dtype="e4m3_fp32", head_sizes=[192], output_dtype="bf16"
        )

        # Expand the cartesian product of the list fields "seq_len" and "head_size".
        specs_expanded = []
        list_like = lambda x: isinstance(x, (list, tuple))
        for kspec in specs:
            tmp_s = kspec.seq_len
            tmp_d = kspec.head_size
            tmp_dtype = kspec.dtype
            tmp_exp = (
                [kspec._replace(seq_len=s) for s in tmp_s]
                if list_like(tmp_s)
                else [kspec]
            )
            tmp_exp = (
                [tmp_ks._replace(head_size=d) for d in tmp_d for tmp_ks in tmp_exp]
                if list_like(tmp_d)
                else tmp_exp
            )
            tmp_exp = (
                [tmp_ks._replace(dtype=dt) for dt in tmp_dtype for tmp_ks in tmp_exp]
                if list_like(tmp_dtype)
                else tmp_exp
            )
            specs_expanded.extend(tmp_exp)

        # Sanitize kernel specs
        specs_expanded = [kspec for kspec in specs_expanded if kspec.sm >= kspec.sm_mma]

        specs_names = [
            (kspec, *encode_name(kspec))
            for kspec in specs_expanded
            # Volta is deprecated in TRT-LLM.
            if (
                kspec.sm >= 80
                and kspec.dtype in ["fp16", "bf16", "fp16_fp32", "e4m3", "e4m3_fp32"]
                and kspec.head_size <= 256
                and kspec.head_size_v == 0
                and kspec.sage_block_sizes is None
                and kspec.version == 2
                and not kspec.cross_mha
                and kspec.flash_attention
                and kspec.input_layout != InputLayout.SEPARATE_Q_K_V
                or (
                    kspec.sm == 90
                    and kspec.dtype in ["fp16", "bf16", "fp16_fp32"]
                    and kspec.head_size <= 256
                    and kspec.ldgsts_q
                    and kspec.version == 2
                    and not kspec.cross_mha
                    and not kspec.flash_attention
                )
                # Clip/SigLip support.
                or (
                    kspec.sm == 100
                    and kspec.dtype
                    in ["fp16", "bf16", "fp16_fp32", "e4m3", "e4m3_fp32"]
                    and kspec.head_size == 80
                    and kspec.head_size_v == 0
                    and kspec.sage_block_sizes is None
                    and kspec.version == 2
                    and not kspec.cross_mha
                    and kspec.flash_attention
                    and kspec.input_layout != InputLayout.SEPARATE_Q_K_V
                )
                # Deepseek MLA (generation 576/512 paged)
                or (
                    kspec.sm in [90, 100, 120]
                    and kspec.dtype in ["bf16", "e4m3_fp32"]
                    and kspec.head_size == 576
                    and kspec.head_size_v == 512
                    and kspec.input_layout == InputLayout.Q_PAGED_KV
                    and kspec.sage_block_sizes is None
                    and kspec.version == 2
                    and not kspec.cross_mha
                    and kspec.flash_attention
                    and not kspec.warp_specialization
                    and kspec.tiled
                )
                # Deepseek MLA (context 192/128 separate-q-k-v)
                or (
                    kspec.sm in [90, 100, 120]
                    and kspec.dtype in ["bf16", "e4m3", "e4m3_fp32"]
                    and kspec.head_size == 192
                    and kspec.head_size_v == 128
                    and kspec.input_layout == InputLayout.SEPARATE_Q_K_V
                    and kspec.sage_block_sizes is None
                    and kspec.version == 2
                    and not kspec.cross_mha
                    and kspec.flash_attention
                    and (
                        (kspec.warp_specialization and not kspec.alibi)  # sm90
                        or (not kspec.warp_specialization and kspec.tiled)
                    )  # non-sm90
                    and not kspec.enable_attn_logit_softcapping
                )
                # SageAttention (warp_spec, head_size in (80, 128), packed QKV, padding mask)
                or (
                    kspec.sm == 90
                    and kspec.head_size in [80, 128]
                    and kspec.version == 2
                    and kspec.sage_block_sizes in [(64, 64, 256)]
                    and not kspec.cross_mha
                    and kspec.flash_attention
                    and kspec.warp_specialization
                    and kspec.input_layout == InputLayout.PACKED_QKV
                    and not kspec.alibi
                    and not kspec.enable_attn_logit_softcapping
                )
                # SageAttention on Ada (head_size in (80, 128), packed QKV, padding mask)
                or (
                    kspec.sm == 89
                    and kspec.head_size in [80, 128]
                    and kspec.sage_block_sizes in [(64, 32, 32)]
                    and kspec.output_dtype in ["fp16", "bf16"]
                    and kspec.version == 2
                    and not kspec.cross_mha
                    and kspec.flash_attention
                    and not kspec.warp_specialization
                    and kspec.input_layout == InputLayout.PACKED_QKV
                )
            )
            # only generate head_size = 128/256 for attn_logit_softcapping operation.
            and (
                kspec.head_size == 128
                or kspec.head_size == 256
                or not kspec.enable_attn_logit_softcapping
            )
        ]
        # yapf: enable

        generate_files(specs_names)
