"""JIT wiring for the generated SM100/SM103 Ulysses implementation."""

from . import env as jit_env
from .core import (
    current_compilation_context,
    gen_jit_spec,
    sm100a_nvcc_flags,
    sm103a_nvcc_flags,
)


def generated_ulysses_spec():
    # Other compilation targets retain the existing portable NVLink kernel.
    targets = current_compilation_context.TARGET_CUDA_ARCHS
    if targets == {(10, "0a")}:
        name, arch_flags = "sm100a", sm100a_nvcc_flags
    elif targets == {(10, "3a")}:
        name, arch_flags = "sm103a", sm103a_nvcc_flags
    else:
        return None
    sources = [
        "ulysses_all_to_all.cu",
        "cake_ulysses_dispatch.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_060d10a295b39a821b1c_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_060d10a295b39a821b1c_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_0a5f70d2046773fe2222_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_0a5f70d2046773fe2222_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_3997ed9b5eff01a95e7b_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_3997ed9b5eff01a95e7b_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_3e15799aefc2a6dd67fd_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_3e15799aefc2a6dd67fd_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_424cc8b9443592d967af_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_424cc8b9443592d967af_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_529ae577ff5c50923a78_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_529ae577ff5c50923a78_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_54deb052ac6ae2b15662_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_54deb052ac6ae2b15662_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_5f26453ba41d53a97ac1_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_5f26453ba41d53a97ac1_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_73c8c0e7219ae91ecfe6_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_73c8c0e7219ae91ecfe6_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_81d93e801f61b0d2b4e0_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_81d93e801f61b0d2b4e0_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_81eb3da8eb3723c0e75a_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_81eb3da8eb3723c0e75a_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_9d0ea74ee5d92c570a9f_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_9d0ea74ee5d92c570a9f_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_9d71bccb13996aa8f2e5_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_9d71bccb13996aa8f2e5_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_aa2652f6707abed77ee5_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_aa2652f6707abed77ee5_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b22486e951c99364ee15_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b22486e951c99364ee15_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b4d01efa1606195e00e8_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b4d01efa1606195e00e8_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b538db2a3106c4836810_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_b538db2a3106c4836810_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_db6770e67a1e623ddc47_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_db6770e67a1e623ddc47_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_e5068bdd090c78c81537_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_e5068bdd090c78c81537_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_e7639882cc46546b168f_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_e7639882cc46546b168f_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_ef0d2d5321e43f0d4d84_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_ef0d2d5321e43f0d4d84_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f492358b1a26be5cc57e_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f492358b1a26be5cc57e_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f5f6d5e72126f2d6800b_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f5f6d5e72126f2d6800b_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f612b1f375cd6096585f_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_f612b1f375cd6096585f_kernel.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_00774cea4486a72f7a6f_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_1cdda316532c38f0e221_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_2707ccbf8200057be50d_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_2f560a23b81cc15c47be_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_396ec3c387b3d9a5ca52_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_7baed92ad8ec7e65005e_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_8e2cbc444730c80a8e9f_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_b3a241c1a14301cec3b7_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_c4412fd74980ec7da889_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_cf34839748bfea20225c_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_eb420dd5e582de094bd2_binding.cu",
        "generated/ulysses/sm_100a/cake_ulysses_a2a_seq_fa7672e2c0597c2a23d3_binding.cu",
    ]
    return gen_jit_spec(
        "ulysses_a2a_" + name,
        [jit_env.FLASHINFER_CSRC_DIR / path for path in sources],
        extra_cuda_cflags=arch_flags + [] + ["-DFLASHINFER_ULYSSES_GENERATED=1"],
    )
