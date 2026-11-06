"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Program selection by token count of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (SM100 / SM103 / SM107), rendered from the registry
of this package (:mod:`.cake_jit`) by its exporter; do not edit.  ``PROGRAM_WINDOWS``
is the table ``cake_jit.program_windows`` derives from the registry's ``MODULES`` and
``ROUTES`` at call time; ``EDGES`` the measured outcome at every window edge; ``NOTES``
the contract of the outputs across the windows; ``SECTIONS`` the text the host module
appends to the forward and backward docstrings.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 2048, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 2048, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (8192, 40960, "cake_rmsnorm_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_train_1388ab779a35c4e41efb"),
            (114688, 196608, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (196608, 327680, "cake_rmsnorm_train_ed9483d8c356affee55d"),
            (327680, None, "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (8192, 40960, "cake_rmsnorm_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_train_b0299f6e0881fb35f27b"),
            (114688, 196608, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (196608, 327680, "cake_rmsnorm_train_d0a656f053396a58c49a"),
            (327680, None, "cake_rmsnorm_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 24576, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (24576, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 24576, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (24576, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 16384, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (16384, None, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 768, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (768, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 768, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (768, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": (
            (0, 20480, "cake_rmsnorm_train_195c4317b51bb4ea1137"),
            (20480, 92160, "cake_rmsnorm_train_172a47a15d4c1a76c061"),
            (92160, 524288, "cake_rmsnorm_train_083ea823c691e0631ad3"),
            (524288, None, "cake_rmsnorm_train_172a47a15d4c1a76c061"),
        ),
        "bwd_w32_h256": (
            (0, 20480, "cake_rmsnorm_train_8de0131445eed6eeabb7"),
            (20480, 92160, "cake_rmsnorm_train_7c2aa4b429538da06a2a"),
            (92160, 524288, "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b"),
            (524288, None, "cake_rmsnorm_train_7c2aa4b429538da06a2a"),
        ),
        "bwd_h512": (
            (0, 227, "cake_rmsnorm_train_5f3e3fccaea795d0a806"),
            (227, 1024, "cake_rmsnorm_train_09601d42628de24f6f87"),
            (1024, 1280, "cake_rmsnorm_train_123db0b2f7fb38cfab03"),
            (1280, None, "cake_rmsnorm_train_d0bde38aa443d42a5f04"),
        ),
        "bwd_w32_h512": (
            (0, 227, "cake_rmsnorm_train_942503b80a21f22cd392"),
            (227, 292, "cake_rmsnorm_train_aa7c5a5407cba063b9d5"),
            (292, 960, "cake_rmsnorm_train_9f07cea1d13b923ed91e"),
            (960, None, "cake_rmsnorm_train_fd08c6f436eb6071fa22"),
        ),
        "bwd_h1536": (
            (0, 64, "cake_rmsnorm_train_38e972a94141a1961c90"),
            (64, 384, "cake_rmsnorm_train_dc478ac1195882aedee6"),
            (384, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_train_c1a9efdf06c3087b4efe"),
            (4096, 32768, "cake_rmsnorm_train_21c8bb55c8d672b63952"),
            (32768, 36864, "cake_rmsnorm_train_b457c4d134e44476dc6f"),
            (36864, 78784, "cake_rmsnorm_train_ae610afdf77ff6714a5b"),
            (78784, None, "cake_rmsnorm_train_b42efd6bf06148d75151"),
        ),
        "bwd_w32_h1536": (
            (0, 64, "cake_rmsnorm_train_d5a621af7a24e403b5f3"),
            (64, 384, "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e"),
            (384, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_train_4b6981c10fd40a9d8aec"),
            (4096, 32768, "cake_rmsnorm_train_4bf450b6af0314ed03be"),
            (32768, 75776, "cake_rmsnorm_train_4b6981c10fd40a9d8aec"),
            (75776, None, "cake_rmsnorm_train_e2b99fee32764cfd1295"),
        ),
        "bwd_h2048": (
            (0, 64, "cake_rmsnorm_train_ef052c0918725ffa68d4"),
            (64, 336, "cake_rmsnorm_train_64e6aa82bb2c6d4ea978"),
            (336, 3072, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (3072, 4096, "cake_rmsnorm_train_9ac8d21e46a658e7913f"),
            (4096, 7168, "cake_rmsnorm_train_c73074cf44f07fcf3bb6"),
            (7168, 11264, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
            (11264, 13608, "cake_rmsnorm_train_c73074cf44f07fcf3bb6"),
            (13608, 45056, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
            (45056, 58880, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
            (58880, 65536, "cake_rmsnorm_train_0fa4f255caf709d09f66"),
            (65536, None, "cake_rmsnorm_train_83f905ffd00c0ef641e2"),
        ),
        "bwd_w32_h2048": (
            (0, 64, "cake_rmsnorm_train_382e45788ace3ff40ba1"),
            (64, 336, "cake_rmsnorm_train_17493054fed5409b6df9"),
            (336, 1024, "cake_rmsnorm_train_17493054fed5409b6df9"),
            (1024, 4096, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (4096, 15403, "cake_rmsnorm_train_b187e5989302def22b2d"),
            (15403, 45056, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (45056, 58880, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (58880, 65536, "cake_rmsnorm_train_ba7dbf6a07744268705e"),
            (65536, None, "cake_rmsnorm_train_9f775c72c1335f4b08ac"),
        ),
        "bwd_h4096": (
            (0, 64, "cake_rmsnorm_train_231e01eca0459c2c0bf5"),
            (64, 1152, "cake_rmsnorm_train_4822793ea0c954c11af6"),
            (1152, 4096, "cake_rmsnorm_train_9a830956447d4e304914"),
            (4096, 4608, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (4608, 36864, "cake_rmsnorm_train_15e157d2ee9080d206a9"),
            (36864, 44032, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (44032, 65536, "cake_rmsnorm_train_15e157d2ee9080d206a9"),
            (65536, None, "cake_rmsnorm_train_c67fc8b9d1b1edb8d7da"),
        ),
        "bwd_w32_h4096": (
            (0, 64, "cake_rmsnorm_train_ce1f7d216ecc0a1573b9"),
            (64, 1792, "cake_rmsnorm_train_8616070567821eaa3e35"),
            (1792, 3584, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
            (3584, 36864, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
            (36864, 44032, "cake_rmsnorm_train_626aed743358c27513f1"),
            (44032, 65536, "cake_rmsnorm_train_f7573bcb748c0784f3ec"),
            (65536, None, "cake_rmsnorm_train_9ebd0232954f1b1842f7"),
        ),
        "bwd_h5120": (
            (0, 64, "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c"),
            (64, 896, "cake_rmsnorm_train_080e0a6c3b18e9bad790"),
            (896, 6144, "cake_rmsnorm_train_b482c84dbffcd75a4a4c"),
            (6144, 56256, "cake_rmsnorm_train_c145ef195d7aeb04f242"),
            (56256, None, "cake_rmsnorm_train_ade9f6a4f10b069ba8a3"),
        ),
        "bwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_train_244b7d841d62d9721c99"),
            (64, 1792, "cake_rmsnorm_train_68f6e584c687c30c0084"),
            (1792, 6144, "cake_rmsnorm_train_de8f6091077dc7ac5947"),
            (6144, 56256, "cake_rmsnorm_train_9479b1298a1b19d4d996"),
            (56256, None, "cake_rmsnorm_train_fc2a52a2c9bf8c5be9af"),
        ),
        "bwd_h6144": (
            (0, 64, "cake_rmsnorm_train_d70f2562a592151ec4f9"),
            (64, 1152, "cake_rmsnorm_train_44508df7dab2ab688f85"),
            (1152, 4096, "cake_rmsnorm_train_5c804aed7ad14b99e1d0"),
            (4096, 5632, "cake_rmsnorm_train_b93bf060d4d3defc8150"),
            (5632, 65536, "cake_rmsnorm_train_9433061da18b2db4299d"),
            (65536, None, "cake_rmsnorm_train_10a41132a42d594ce6dd"),
        ),
        "bwd_residual_h6144": (
            (0, 64, "cake_rmsnorm_train_7b0f5a6a8769473c322e"),
            (64, 1408, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (1408, 65536, "cake_rmsnorm_train_3fbfdd750cbeb47032b5"),
            (65536, None, "cake_rmsnorm_train_1609f7969c50fba7f2c7"),
        ),
        "bwd_w32_h6144": (
            (0, 64, "cake_rmsnorm_train_9b37923697ee8ca52a9e"),
            (64, 1168, "cake_rmsnorm_train_b3e6531e46667901ab91"),
            (1168, 3584, "cake_rmsnorm_train_41b80b6af97974feb181"),
            (3584, 65536, "cake_rmsnorm_train_a562656cef5bf5dda160"),
            (65536, None, "cake_rmsnorm_train_45cf60e3cd8603c2abeb"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 64, "cake_rmsnorm_train_ac9d69f94aabfc68aa40"),
            (64, 1408, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (1408, 65536, "cake_rmsnorm_train_a640d0b799889f785ad5"),
            (65536, None, "cake_rmsnorm_train_d3d674373d17aaf425c0"),
        ),
    },
    "sm_103a": {
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_c5a8099e177e463daab9"),
            (1024, 4096, "cake_rmsnorm_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_01f0ebf3db9fa2c0884e"),
            (1024, 4096, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_train_0489c784c13bb25499f6"),
            (8192, 20480, "cake_rmsnorm_train_bbb05314930d192dfe7b"),
            (20480, 40960, "cake_rmsnorm_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (114688, 196608, "cake_rmsnorm_train_86d51eff64fc9fe57cf0"),
            (196608, 327680, "cake_rmsnorm_train_ed9483d8c356affee55d"),
            (327680, 368640, "cake_rmsnorm_train_37f663183cbff23c2893"),
            (368640, None, "cake_rmsnorm_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_train_f2e721db4c30900f12fe"),
            (8192, 20480, "cake_rmsnorm_train_a8fffd93682c5e382dc5"),
            (20480, 40960, "cake_rmsnorm_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (114688, 196608, "cake_rmsnorm_train_42d82d26e97c36651dae"),
            (196608, 327680, "cake_rmsnorm_train_d0a656f053396a58c49a"),
            (327680, 368640, "cake_rmsnorm_train_18e3c0a5204006b3a56f"),
            (368640, None, "cake_rmsnorm_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 3072, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (3072, 6144, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 3072, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (3072, 6144, "cake_rmsnorm_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (640, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, None, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (6144, None, "cake_rmsnorm_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": (
            (0, 1080, "cake_rmsnorm_train_195c4317b51bb4ea1137"),
            (1080, 5120, "cake_rmsnorm_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (12288, 40960, "cake_rmsnorm_train_af005462cd6a301f31c4"),
            (40960, 97280, "cake_rmsnorm_train_a385ae1681a2c078344e"),
            (97280, 720896, "cake_rmsnorm_train_083ea823c691e0631ad3"),
            (720896, 1572864, "cake_rmsnorm_train_0291949ba535c646c3a8"),
            (1572864, None, "cake_rmsnorm_train_c18297d75c0d424978fd"),
        ),
        "bwd_w32_h256": (
            (0, 1080, "cake_rmsnorm_train_8de0131445eed6eeabb7"),
            (1080, 5120, "cake_rmsnorm_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (12288, 40960, "cake_rmsnorm_train_b7d9577ff884de98ea06"),
            (40960, 97280, "cake_rmsnorm_train_420788e663a2c3a810f6"),
            (97280, 720896, "cake_rmsnorm_train_c4c89f0d04b7b3a38d0b"),
            (720896, 1572864, "cake_rmsnorm_train_89505762fa1cbf33cb8f"),
            (1572864, None, "cake_rmsnorm_train_f788ef57163552825d27"),
        ),
        "bwd_h512": (
            (0, 26, "cake_rmsnorm_train_dd5c218dacb8aa174790"),
            (26, 1024, "cake_rmsnorm_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_train_8e7f4d811e55a351a22e"),
            (4096, 32768, "cake_rmsnorm_train_274b0055aace183f2367"),
            (32768, None, "cake_rmsnorm_train_cfaae3cec98f8501832b"),
        ),
        "bwd_w32_h512": (
            (0, 292, "cake_rmsnorm_train_4724ad6abf88e40c2151"),
            (292, 960, "cake_rmsnorm_train_9f07cea1d13b923ed91e"),
            (960, 32768, "cake_rmsnorm_train_56ee8e66479625fd078a"),
            (32768, None, "cake_rmsnorm_train_a546f410ec8cf03ef20d"),
        ),
        "bwd_h1536": (
            (0, 64, "cake_rmsnorm_train_38e972a94141a1961c90"),
            (64, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_train_c1a9efdf06c3087b4efe"),
            (4096, 14928, "cake_rmsnorm_train_1df7db17277d363b5168"),
            (14928, 15566, "cake_rmsnorm_train_4df08ce51fb01ba1193f"),
            (15566, 46592, "cake_rmsnorm_train_f522f5b8d4343d96ca04"),
            (46592, None, "cake_rmsnorm_train_327bfeb4ab8fefc2731d"),
        ),
        "bwd_w32_h1536": (
            (0, 64, "cake_rmsnorm_train_d5a621af7a24e403b5f3"),
            (64, 320, "cake_rmsnorm_train_963a7ca8d03f78bbe6f9"),
            (320, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_train_17d9cdfd157ba4b86c2b"),
            (4096, 14928, "cake_rmsnorm_train_a99e2b380514c4d86d60"),
            (14928, 15566, "cake_rmsnorm_train_7a72e97bd6cb4dc0efc2"),
            (15566, 37888, "cake_rmsnorm_train_109f19bb177c8f20cadc"),
            (37888, None, "cake_rmsnorm_train_963a7ca8d03f78bbe6f9"),
        ),
        "bwd_h2048": (
            (0, 64, "cake_rmsnorm_train_ef052c0918725ffa68d4"),
            (64, 3072, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (3072, 4096, "cake_rmsnorm_train_9ac8d21e46a658e7913f"),
            (4096, 16384, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (16384, 32768, "cake_rmsnorm_train_caf6978ed0a17fbf65b9"),
            (32768, 58496, "cake_rmsnorm_train_302f64db6abe79e5259e"),
            (58496, 61952, "cake_rmsnorm_train_018f0cd723106f709f11"),
            (61952, None, "cake_rmsnorm_train_8bd89a4c834f80db8a34"),
        ),
        "bwd_w32_h2048": (
            (0, 64, "cake_rmsnorm_train_382e45788ace3ff40ba1"),
            (64, 3072, "cake_rmsnorm_train_07f9f2fbea7163639c27"),
            (3072, 4096, "cake_rmsnorm_train_d96a14fe8882bbd9ec2e"),
            (4096, 16384, "cake_rmsnorm_train_07f9f2fbea7163639c27"),
            (16384, 32768, "cake_rmsnorm_train_07f9f2fbea7163639c27"),
            (32768, 58496, "cake_rmsnorm_train_e98cb156abfe1666c76f"),
            (58496, 61952, "cake_rmsnorm_train_ca8ca4674a3dbd89ad16"),
            (61952, None, "cake_rmsnorm_train_88c6c6b2af1922f4c1a9"),
        ),
        "bwd_h4096": (
            (0, 32, "cake_rmsnorm_train_3dbab2323fa42435a72d"),
            (32, 256, "cake_rmsnorm_train_50044ee3bd1ca54bd821"),
            (256, 1024, "cake_rmsnorm_train_ad2bb26ea5916baef963"),
            (1024, 4096, "cake_rmsnorm_train_24643edc7e1143da24e0"),
            (4096, 50176, "cake_rmsnorm_train_35a0b0ca6b2060313255"),
            (50176, 262144, "cake_rmsnorm_train_8250eb924d925c0db0d9"),
            (262144, None, "cake_rmsnorm_train_8c8bebecc5b08454b6a0"),
        ),
        "bwd_w32_h4096": (
            (0, 32, "cake_rmsnorm_train_c1b9ee6b9f10766cd537"),
            (32, 256, "cake_rmsnorm_train_3fd77c8ce97e8a523abb"),
            (256, 1024, "cake_rmsnorm_train_16545056022785f2a7d8"),
            (1024, 4096, "cake_rmsnorm_train_78c26a53e83c9803bcbf"),
            (4096, 9216, "cake_rmsnorm_train_626aed743358c27513f1"),
            (9216, 262144, "cake_rmsnorm_train_2911faf7a786144495a6"),
            (262144, None, "cake_rmsnorm_train_131a495c5897206b6874"),
        ),
        "bwd_h5120": (
            (0, 64, "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c"),
            (64, 5120, "cake_rmsnorm_train_2fc4932fbde0bc848b32"),
            (5120, 61952, "cake_rmsnorm_train_7ffb443a63a92ab6b0e0"),
            (61952, 393216, "cake_rmsnorm_train_b523a764b029e5ca1aaf"),
            (393216, None, "cake_rmsnorm_train_9e4c0825df92af02074f"),
        ),
        "bwd_w32_h5120": (
            (0, 48, "cake_rmsnorm_train_244b7d841d62d9721c99"),
            (48, 5120, "cake_rmsnorm_train_79811a76205d7d7dd197"),
            (5120, 61952, "cake_rmsnorm_train_f9f7d4cd75ef91cc8374"),
            (61952, 196608, "cake_rmsnorm_train_90598c14b4e396f5c8b0"),
            (196608, None, "cake_rmsnorm_train_f165933caadda3434828"),
        ),
        "bwd_h6144": (
            (0, 32, "cake_rmsnorm_train_d70f2562a592151ec4f9"),
            (32, 4096, "cake_rmsnorm_train_5c804aed7ad14b99e1d0"),
            (4096, 49152, "cake_rmsnorm_train_322574af704b51aef0bb"),
            (49152, None, "cake_rmsnorm_train_582b2a20f9be648af8ec"),
        ),
        "bwd_residual_h6144": (
            (0, 48, "cake_rmsnorm_train_7b0f5a6a8769473c322e"),
            (48, 56, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (56, 304, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (304, 1920, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (1920, 2816, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (2816, 4608, "cake_rmsnorm_train_115dcc719a6de4b079f1"),
            (4608, 5632, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (5632, 32768, "cake_rmsnorm_train_115dcc719a6de4b079f1"),
            (32768, None, "cake_rmsnorm_train_52831494c1b62db758fd"),
        ),
        "bwd_w32_h6144": (
            (0, 48, "cake_rmsnorm_train_9b37923697ee8ca52a9e"),
            (48, 320, "cake_rmsnorm_train_41b80b6af97974feb181"),
            (320, 448, "cake_rmsnorm_train_b3e6531e46667901ab91"),
            (448, 4096, "cake_rmsnorm_train_41b80b6af97974feb181"),
            (4096, 49152, "cake_rmsnorm_train_9a8c0396edd85d4d545e"),
            (49152, None, "cake_rmsnorm_train_4f9a76e68671f213ab7c"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 48, "cake_rmsnorm_train_ac9d69f94aabfc68aa40"),
            (48, 320, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (320, 1824, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (1824, 3584, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (3584, 4608, "cake_rmsnorm_train_3c390339fa8c82311a0a"),
            (4608, 5632, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (5632, 15872, "cake_rmsnorm_train_3c390339fa8c82311a0a"),
            (15872, None, "cake_rmsnorm_train_56c46a201c168050aa60"),
        ),
    },
    "sm_107a": {
        "fwd_h1536": (
            (0, 2048, "cake_rmsnorm_train_1ea88523077fc340e55f"),
            (2048, 12288, "cake_rmsnorm_train_4cdc4c495b3f59ff5069"),
            (12288, None, "cake_rmsnorm_train_682675bda578f5b7a312"),
        ),
        "fwd_w32_h1536": (
            (0, 2048, "cake_rmsnorm_train_21d3416baa877808371e"),
            (2048, 12288, "cake_rmsnorm_train_92cf7b87f1ebffb1b12b"),
            (12288, None, "cake_rmsnorm_train_a30e5a3b69b7c183992f"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_train_b44107e05796c2922187"),
            (1024, 8192, "cake_rmsnorm_train_a4da85136fba4588239f"),
            (8192, 12288, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (12288, 16384, "cake_rmsnorm_train_0ad4d53c4a0df501a99d"),
            (16384, None, "cake_rmsnorm_train_bde56bd5a68c493f07a2"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_train_0da8e18d4af432c6927a"),
            (1024, 8192, "cake_rmsnorm_train_a72c6ece1900f748eaa2"),
            (8192, 12288, "cake_rmsnorm_train_595f506004038a65cc63"),
            (12288, 16384, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
            (16384, None, "cake_rmsnorm_train_40210950f2e7ee48bc3e"),
        ),
        "fwd_h256": (
            (0, 6144, "cake_rmsnorm_train_c5ca409f44e169148301"),
            (6144, 32768, "cake_rmsnorm_train_bbb05314930d192dfe7b"),
            (32768, 98304, "cake_rmsnorm_train_466331ff3cd2d6c62738"),
            (98304, None, "cake_rmsnorm_train_37f663183cbff23c2893"),
        ),
        "fwd_w32_h256": (
            (0, 6144, "cake_rmsnorm_train_8f2756937eaac6b02502"),
            (6144, 32768, "cake_rmsnorm_train_a8fffd93682c5e382dc5"),
            (32768, 98304, "cake_rmsnorm_train_32bebe5a7782311d60a0"),
            (98304, None, "cake_rmsnorm_train_18e3c0a5204006b3a56f"),
        ),
        "fwd_h4096": (
            (0, 512, "cake_rmsnorm_train_8392d1a008d58a006bba"),
            (512, 1024, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
            (1024, 4096, "cake_rmsnorm_train_d447b7f6398b60fe5daa"),
            (4096, 20480, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (20480, 49152, "cake_rmsnorm_train_bf933e1660cab8e5787b"),
            (49152, None, "cake_rmsnorm_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 512, "cake_rmsnorm_train_4371270e4db4b985998f"),
            (512, 1024, "cake_rmsnorm_train_f0db3068d0d05442a59d"),
            (1024, 4096, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (4096, 20480, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (20480, 49152, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
            (49152, None, "cake_rmsnorm_train_f9f3423be0bd8ff8c794"),
        ),
        "fwd_h512": (
            (0, 256, "cake_rmsnorm_train_7673978dc698790054d2"),
            (256, 6144, "cake_rmsnorm_train_2a42e5c02b240f8709b1"),
            (6144, None, "cake_rmsnorm_train_60f47c5d00570ecf7740"),
        ),
        "fwd_w32_h512": (
            (0, 256, "cake_rmsnorm_train_e2eabcb3b0915d916a13"),
            (256, 6144, "cake_rmsnorm_train_79fecc7016e44e940733"),
            (6144, None, "cake_rmsnorm_train_61accbcc02891615f286"),
        ),
        "fwd_h5120": (
            (0, 512, "cake_rmsnorm_train_fd1097c45c1e328871fd"),
            (512, 1152, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
            (1152, 6144, "cake_rmsnorm_train_b5e173f5ecc736d91109"),
            (6144, 20480, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (20480, 49152, "cake_rmsnorm_train_044d4a2aa0553c16db06"),
            (49152, None, "cake_rmsnorm_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 512, "cake_rmsnorm_train_ba28d2c0c2c81a2c714e"),
            (512, 1152, "cake_rmsnorm_train_1defffdf2712cae36784"),
            (1152, 6144, "cake_rmsnorm_train_9ecb62c2640d571e50a8"),
            (6144, 20480, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (20480, 49152, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
            (49152, None, "cake_rmsnorm_train_b49de9b47372fa4e1051"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_train_bad4db9153cc903b7ace"),
            (1024, 4096, "cake_rmsnorm_train_cf1d05546951bdbd9f5d"),
            (4096, None, "cake_rmsnorm_train_e7f4221a7c26b9dfdd23"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_train_2763cd5ebcc501c9ddb6"),
            (1024, 4096, "cake_rmsnorm_train_3d619cd1e3ae638cce3d"),
            (4096, None, "cake_rmsnorm_train_fdfa500c83d95df0f5c1"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_e646bbdbd852884953da"),
            (256, 1024, "cake_rmsnorm_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_train_6325c474ad9955a319e9"),
            (256, 1024, "cake_rmsnorm_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_train_6325c474ad9955a319e9"),
        ),
        "bwd_h256": (
            (0, 1056, "cake_rmsnorm_train_195c4317b51bb4ea1137"),
            (1056, 3584, "cake_rmsnorm_train_5450d39836b26cfd3256"),
            (3584, 36864, "cake_rmsnorm_train_195c4317b51bb4ea1137"),
            (36864, None, "cake_rmsnorm_train_172a47a15d4c1a76c061"),
        ),
        "bwd_w32_h256": (
            (0, 36864, "cake_rmsnorm_train_8de0131445eed6eeabb7"),
            (36864, None, "cake_rmsnorm_train_7c2aa4b429538da06a2a"),
        ),
        "bwd_h512": (
            (0, 77, "cake_rmsnorm_train_5f3e3fccaea795d0a806"),
            (77, 1024, "cake_rmsnorm_train_3feed5026e1156d2fba2"),
            (1024, 4096, "cake_rmsnorm_train_3feed5026e1156d2fba2"),
            (4096, 16384, "cake_rmsnorm_train_dfaf378aa8ad93e68bbd"),
            (16384, 49152, "cake_rmsnorm_train_eb141cc9e0a93c6f0ce3"),
            (49152, None, "cake_rmsnorm_train_30501ccec48cc058333a"),
        ),
        "bwd_w32_h512": (
            (0, 229, "cake_rmsnorm_train_942503b80a21f22cd392"),
            (229, 1024, "cake_rmsnorm_train_b841d64bbcaefff5ce7b"),
            (1024, 4096, "cake_rmsnorm_train_fd08c6f436eb6071fa22"),
            (4096, 16384, "cake_rmsnorm_train_383f709cd5c4fadd9436"),
            (16384, 49152, "cake_rmsnorm_train_03efa681eb38f59b00c2"),
            (49152, None, "cake_rmsnorm_train_26fbe8e6f07a658e26f6"),
        ),
        "bwd_h1536": (
            (0, 64, "cake_rmsnorm_train_38e972a94141a1961c90"),
            (64, 1024, "cake_rmsnorm_train_7dc29629ff21625607f8"),
            (1024, 16384, "cake_rmsnorm_train_6a428da2227972ae18ae"),
            (16384, 53248, "cake_rmsnorm_train_56d26996239835327501"),
            (53248, 65536, "cake_rmsnorm_train_3f75f4a81f2fb8269389"),
            (65536, None, "cake_rmsnorm_train_dca25ace64c075a81625"),
        ),
        "bwd_w32_h1536": (
            (0, 64, "cake_rmsnorm_train_d5a621af7a24e403b5f3"),
            (64, 256, "cake_rmsnorm_train_3ddc5ca5ad86c7725d7e"),
            (256, 1024, "cake_rmsnorm_train_baff9ecbafe7de1e6030"),
            (1024, 16384, "cake_rmsnorm_train_7c0dd63f0b011049603e"),
            (16384, 53248, "cake_rmsnorm_train_de8e07d4cb312b06dbc4"),
            (53248, 65536, "cake_rmsnorm_train_7c0dd63f0b011049603e"),
            (65536, None, "cake_rmsnorm_train_ccd94d87e9036a0b2faf"),
        ),
        "bwd_h2048": (
            (0, 64, "cake_rmsnorm_train_ef052c0918725ffa68d4"),
            (64, 256, "cake_rmsnorm_train_84efa2af7907877dfb56"),
            (256, 1024, "cake_rmsnorm_train_845cdcd8145ada631461"),
            (1024, 16384, "cake_rmsnorm_train_81c3d8ab0a482e3af043"),
            (16384, 50176, "cake_rmsnorm_train_845cdcd8145ada631461"),
            (50176, None, "cake_rmsnorm_train_1a3df5ac020055477581"),
        ),
        "bwd_w32_h2048": (
            (0, 64, "cake_rmsnorm_train_382e45788ace3ff40ba1"),
            (64, 127, "cake_rmsnorm_train_7a3e5eb75314492eebdb"),
            (127, 256, "cake_rmsnorm_train_6b13aa49e0c0414d9041"),
            (256, 1024, "cake_rmsnorm_train_838a9a4054731f7942af"),
            (1024, 16384, "cake_rmsnorm_train_810989bdea464df479ce"),
            (16384, 50176, "cake_rmsnorm_train_838a9a4054731f7942af"),
            (50176, None, "cake_rmsnorm_train_810989bdea464df479ce"),
        ),
        "bwd_h4096": (
            (0, 64, "cake_rmsnorm_train_231e01eca0459c2c0bf5"),
            (64, 256, "cake_rmsnorm_train_0da5dd4b80ad9b5426ce"),
            (256, 1024, "cake_rmsnorm_train_a647fccc3d9fa9134733"),
            (1024, 4096, "cake_rmsnorm_train_d41a514f5210e9adc39f"),
            (4096, 36864, "cake_rmsnorm_train_ee4a9053bb25b8d49fcf"),
            (36864, 65536, "cake_rmsnorm_train_f11a86167db4f8e9a5da"),
            (65536, None, "cake_rmsnorm_train_5e9d7baa36814bb7aaf6"),
        ),
        "bwd_w32_h4096": (
            (0, 64, "cake_rmsnorm_train_ce1f7d216ecc0a1573b9"),
            (64, 256, "cake_rmsnorm_train_0af20ef9ee6868c5569d"),
            (256, 1024, "cake_rmsnorm_train_dc439f8335327e482025"),
            (1024, 4096, "cake_rmsnorm_train_b08633208e60b8ce94fd"),
            (4096, 24064, "cake_rmsnorm_train_0f1859b9fe9e47668a1c"),
            (24064, 65536, "cake_rmsnorm_train_a1f537fc94bde175bb3c"),
            (65536, None, "cake_rmsnorm_train_23cee954b648495ceef1"),
        ),
        "bwd_h5120": (
            (0, 64, "cake_rmsnorm_train_cc082d1fc7f42b4b3f2c"),
            (64, 4096, "cake_rmsnorm_train_838c077df3a6d35d55b7"),
            (4096, 19968, "cake_rmsnorm_train_6816fb80e97380f1bcbc"),
            (19968, 49152, "cake_rmsnorm_train_b8b0913b40f035f43b76"),
            (49152, None, "cake_rmsnorm_train_56634f53e6456b34f76f"),
        ),
        "bwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_train_244b7d841d62d9721c99"),
            (64, 4096, "cake_rmsnorm_train_cd612de56c0985bb4caa"),
            (4096, 19968, "cake_rmsnorm_train_dd795e849940a49d0d29"),
            (19968, 49152, "cake_rmsnorm_train_7b9e10b21b75d4615135"),
            (49152, None, "cake_rmsnorm_train_6cedccdf3a58f23f260f"),
        ),
        "bwd_h6144": (
            (0, 64, "cake_rmsnorm_train_d70f2562a592151ec4f9"),
            (64, 19456, "cake_rmsnorm_train_5d292172df772adb17ae"),
            (19456, 30720, "cake_rmsnorm_train_19e18371f9243bd1a91a"),
            (30720, None, "cake_rmsnorm_train_083b5f2e2a0d1b4f1c82"),
        ),
        "bwd_residual_h6144": (
            (0, 64, "cake_rmsnorm_train_7b0f5a6a8769473c322e"),
            (64, 448, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (448, 3584, "cake_rmsnorm_train_864e84f3d874f850c091"),
            (3584, 7168, "cake_rmsnorm_train_0dcdc563be377247044e"),
            (7168, None, "cake_rmsnorm_train_c07ab5f16934c0cc8421"),
        ),
        "bwd_w32_h6144": (
            (0, 64, "cake_rmsnorm_train_9b37923697ee8ca52a9e"),
            (64, 2304, "cake_rmsnorm_train_44c5b870a7dcbb8de924"),
            (2304, 19456, "cake_rmsnorm_train_e77387f14390ba486055"),
            (19456, 28672, "cake_rmsnorm_train_b83f8b3d546630a60675"),
            (28672, None, "cake_rmsnorm_train_25f9b466fa418dd2afc4"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 64, "cake_rmsnorm_train_ac9d69f94aabfc68aa40"),
            (64, 96, "cake_rmsnorm_train_c8bc10e76c624f5f4534"),
            (96, 448, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (448, 3584, "cake_rmsnorm_train_71d36617003ab7742a77"),
            (3584, 7168, "cake_rmsnorm_train_56f988dde54db37f09d9"),
            (7168, None, "cake_rmsnorm_train_e655055ead76f8df4d24"),
        ),
    },
}

# Per architecture and route: the upper edge of every window but the last and the measured
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (the
# forward's ulp differences of the notes), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "ulp"), (2048, "ulp"), (4096, "ulp")),
        "fwd_h256": (
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_w32_h256": (
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (6144, "same")),
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (24576, "same")),
        "fwd_h5120": ((640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((640, "same"), (6144, "same"), (16384, "same")),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": ((768, "same"), (1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h256": ((20480, "same"), (92160, "same"), (524288, "same")),
        "bwd_w32_h256": ((20480, "same"), (92160, "same"), (524288, "same")),
        "bwd_h512": ((227, "same"), (1024, "same"), (1280, "same")),
        "bwd_w32_h512": ((227, "same"), (292, "same"), (960, "same")),
        "bwd_h1536": (
            (64, "same"),
            (384, "same"),
            (1024, "same"),
            (4096, "same"),
            (32768, "same"),
            (36864, "same"),
            (78784, "same"),
        ),
        "bwd_w32_h1536": (
            (64, "same"),
            (384, "same"),
            (1024, "same"),
            (4096, "same"),
            (32768, "same"),
            (75776, "same"),
        ),
        "bwd_h2048": (
            (64, "same"),
            (336, "same"),
            (3072, "same"),
            (4096, "same"),
            (7168, "same"),
            (11264, "same"),
            (13608, "same"),
            (45056, "same"),
            (58880, "same"),
            (65536, "same"),
        ),
        "bwd_w32_h2048": (
            (64, "same"),
            (336, "same"),
            (1024, "same"),
            (4096, "same"),
            (15403, "same"),
            (45056, "same"),
            (58880, "same"),
            (65536, "same"),
        ),
        "bwd_h4096": (
            (64, "same"),
            (1152, "same"),
            (4096, "same"),
            (4608, "same"),
            (36864, "same"),
            (44032, "same"),
            (65536, "same"),
        ),
        "bwd_w32_h4096": (
            (64, "same"),
            (1792, "same"),
            (3584, "same"),
            (36864, "same"),
            (44032, "same"),
            (65536, "same"),
        ),
        "bwd_h5120": ((64, "same"), (896, "same"), (6144, "same"), (56256, "same")),
        "bwd_w32_h5120": (
            (64, "same"),
            (1792, "same"),
            (6144, "same"),
            (56256, "same"),
        ),
        "bwd_h6144": (
            (64, "same"),
            (1152, "same"),
            (4096, "same"),
            (5632, "same"),
            (65536, "same"),
        ),
        "bwd_residual_h6144": ((64, "same"), (1408, "same"), (65536, "same")),
        "bwd_w32_h6144": (
            (64, "same"),
            (1168, "same"),
            (3584, "same"),
            (65536, "same"),
        ),
        "bwd_w32_residual_h6144": ((64, "same"), (1408, "same"), (65536, "same")),
    },
    "sm_103a": {
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_h256": (
            (8192, "same"),
            (20480, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
            (368640, "same"),
        ),
        "fwd_w32_h256": (
            (8192, "same"),
            (20480, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
            (368640, "same"),
        ),
        "fwd_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_w32_h4096": ((640, "ulp"), (1024, "same"), (3072, "same"), (6144, "same")),
        "fwd_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_w32_h512": ((1024, "ulp"), (4096, "same"), (16384, "same")),
        "fwd_h5120": ((640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((640, "same"), (6144, "same")),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
        "bwd_h256": (
            (1080, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (97280, "same"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (1080, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (97280, "same"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_h512": ((26, "same"), (1024, "same"), (4096, "same"), (32768, "same")),
        "bwd_w32_h512": ((292, "same"), (960, "same"), (32768, "same")),
        "bwd_h1536": (
            (64, "same"),
            (1024, "same"),
            (4096, "same"),
            (14928, "same"),
            (15566, "same"),
            (46592, "same"),
        ),
        "bwd_w32_h1536": (
            (64, "same"),
            (320, "same"),
            (1024, "same"),
            (4096, "same"),
            (14928, "same"),
            (15566, "same"),
            (37888, "same"),
        ),
        "bwd_h2048": (
            (64, "same"),
            (3072, "same"),
            (4096, "same"),
            (16384, "same"),
            (32768, "same"),
            (58496, "same"),
            (61952, "same"),
        ),
        "bwd_w32_h2048": (
            (64, "same"),
            (3072, "same"),
            (4096, "same"),
            (16384, "same"),
            (32768, "same"),
            (58496, "same"),
            (61952, "same"),
        ),
        "bwd_h4096": (
            (32, "same"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (50176, "same"),
            (262144, "same"),
        ),
        "bwd_w32_h4096": (
            (32, "same"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (9216, "same"),
            (262144, "same"),
        ),
        "bwd_h5120": ((64, "same"), (5120, "same"), (61952, "same"), (393216, "same")),
        "bwd_w32_h5120": (
            (48, "same"),
            (5120, "same"),
            (61952, "same"),
            (196608, "same"),
        ),
        "bwd_h6144": ((32, "same"), (4096, "same"), (49152, "same")),
        "bwd_residual_h6144": (
            (48, "same"),
            (56, "same"),
            (304, "same"),
            (1920, "same"),
            (2816, "same"),
            (4608, "same"),
            (5632, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h6144": (
            (48, "same"),
            (320, "same"),
            (448, "same"),
            (4096, "same"),
            (49152, "same"),
        ),
        "bwd_w32_residual_h6144": (
            (48, "same"),
            (320, "same"),
            (1824, "same"),
            (3584, "same"),
            (4608, "same"),
            (5632, "same"),
            (15872, "same"),
        ),
    },
    "sm_107a": {
        "fwd_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_w32_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_h2048": ((1024, "ulp"), (8192, "same"), (12288, "same"), (16384, "same")),
        "fwd_w32_h2048": (
            (1024, "ulp"),
            (8192, "same"),
            (12288, "same"),
            (16384, "same"),
        ),
        "fwd_h256": ((6144, "same"), (32768, "same"), (98304, "same")),
        "fwd_w32_h256": ((6144, "same"), (32768, "same"), (98304, "same")),
        "fwd_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_h512": ((256, "same"), (6144, "same")),
        "fwd_w32_h512": ((256, "same"), (6144, "same")),
        "fwd_h5120": (
            (512, "same"),
            (1152, "same"),
            (6144, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h5120": (
            (512, "same"),
            (1152, "same"),
            (6144, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_h6144": ((1024, "same"), (4096, "same")),
        "fwd_w32_h6144": ((1024, "same"), (4096, "same")),
        "fwd_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "bwd_h256": ((1056, "same"), (3584, "same"), (36864, "same")),
        "bwd_w32_h256": ((36864, "same"),),
        "bwd_h512": (
            (77, "same"),
            (1024, "same"),
            (4096, "same"),
            (16384, "same"),
            (49152, "same"),
        ),
        "bwd_w32_h512": (
            (229, "same"),
            (1024, "same"),
            (4096, "same"),
            (16384, "same"),
            (49152, "same"),
        ),
        "bwd_h1536": (
            (64, "same"),
            (1024, "same"),
            (16384, "same"),
            (53248, "same"),
            (65536, "same"),
        ),
        "bwd_w32_h1536": (
            (64, "same"),
            (256, "same"),
            (1024, "same"),
            (16384, "same"),
            (53248, "same"),
            (65536, "same"),
        ),
        "bwd_h2048": (
            (64, "same"),
            (256, "same"),
            (1024, "same"),
            (16384, "same"),
            (50176, "same"),
        ),
        "bwd_w32_h2048": (
            (64, "same"),
            (127, "same"),
            (256, "same"),
            (1024, "same"),
            (16384, "same"),
            (50176, "same"),
        ),
        "bwd_h4096": (
            (64, "same"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (36864, "same"),
            (65536, "same"),
        ),
        "bwd_w32_h4096": (
            (64, "same"),
            (256, "same"),
            (1024, "same"),
            (4096, "same"),
            (24064, "same"),
            (65536, "same"),
        ),
        "bwd_h5120": ((64, "same"), (4096, "same"), (19968, "same"), (49152, "same")),
        "bwd_w32_h5120": (
            (64, "same"),
            (4096, "same"),
            (19968, "same"),
            (49152, "same"),
        ),
        "bwd_h6144": ((64, "same"), (19456, "same"), (30720, "same")),
        "bwd_residual_h6144": (
            (64, "same"),
            (448, "same"),
            (3584, "same"),
            (7168, "same"),
        ),
        "bwd_w32_h6144": (
            (64, "same"),
            (2304, "same"),
            (19456, "same"),
            (28672, "same"),
        ),
        "bwd_w32_residual_h6144": (
            (64, "same"),
            (96, "same"),
            (448, "same"),
            (3584, "same"),
            (7168, "same"),
        ),
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

# Per architecture: the backward routes that run the deterministic pair in every window, each
# with the measured reason; every other backward route ships the single kernel in at least one
# window (the exporter's ``DET_PAIR_ROUTES``; the registry gate test reads this table).
DET_PAIR_ROUTES: dict[str, dict[str, str]] = {
    "sm_100a": {},
    "sm_103a": {},
    "sm_107a": {},
}

# The shared-memory sizes (KB per CTA of a kernel resident on every SM) the co-residence of the
# backward windows is measured beside, and per architecture and route the windows measured:
# ``(above, at_most, KB)``, KB the smallest size beside which the window waits on any measured
# board of the architecture (None: it starts beside every measured size on every board); the
# exporter's ``SMEM_WAITS``, checked by the stall test (no board waits below KB).
SMEM_SIDE_KB = (100, 132)
SMEM_WAITS: dict[str, dict[str, tuple[tuple[int, int | None, int | None], ...]]] = {
    "sm_100a": {
        "bwd_h1536": (
            (0, 64, None),
            (64, 384, None),
            (384, 1024, 132),
            (1024, 4096, 100),
            (4096, 32768, 100),
            (32768, 36864, 132),
            (36864, 78784, None),
            (78784, None, None),
        ),
        "bwd_h2048": (
            (0, 64, None),
            (64, 336, None),
            (336, 3072, 100),
            (3072, 4096, 100),
            (4096, 7168, 100),
            (7168, 11264, 132),
            (11264, 13608, 100),
            (13608, 45056, 132),
            (45056, 58880, 132),
            (58880, 65536, 132),
            (65536, None, 132),
        ),
        "bwd_h256": (
            (0, 20480, None),
            (20480, 92160, None),
            (92160, 524288, None),
            (524288, None, None),
        ),
        "bwd_h4096": (
            (0, 64, None),
            (64, 1152, None),
            (1152, 4096, 132),
            (4096, 4608, 132),
            (4608, 36864, None),
            (36864, 44032, 132),
            (44032, 65536, None),
            (65536, None, None),
        ),
        "bwd_h512": (
            (0, 227, None),
            (227, 1024, None),
            (1024, 1280, 132),
            (1280, None, None),
        ),
        "bwd_h5120": (
            (0, 64, 100),
            (64, 896, None),
            (896, 6144, None),
            (6144, 56256, None),
            (56256, None, 132),
        ),
        "bwd_h6144": (
            (0, 64, None),
            (64, 1152, 132),
            (1152, 4096, 132),
            (4096, 5632, 132),
            (5632, 65536, 100),
            (65536, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 64, None),
            (64, 1408, 100),
            (1408, 65536, 100),
            (65536, None, 100),
        ),
        "bwd_w32_h1536": (
            (0, 64, None),
            (64, 384, None),
            (384, 1024, 132),
            (1024, 4096, None),
            (4096, 32768, 100),
            (32768, 75776, None),
            (75776, None, None),
        ),
        "bwd_w32_h2048": (
            (0, 64, None),
            (64, 336, None),
            (336, 1024, None),
            (1024, 4096, 132),
            (4096, 15403, 100),
            (15403, 45056, 132),
            (45056, 58880, 132),
            (58880, 65536, 132),
            (65536, None, 132),
        ),
        "bwd_w32_h256": (
            (0, 20480, None),
            (20480, 92160, None),
            (92160, 524288, None),
            (524288, None, None),
        ),
        "bwd_w32_h4096": (
            (0, 64, None),
            (64, 1792, None),
            (1792, 3584, None),
            (3584, 36864, None),
            (36864, 44032, 132),
            (44032, 65536, None),
            (65536, None, None),
        ),
        "bwd_w32_h512": (
            (0, 227, None),
            (227, 292, None),
            (292, 960, None),
            (960, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 64, 100),
            (64, 1792, None),
            (1792, 6144, 132),
            (6144, 56256, None),
            (56256, None, 132),
        ),
        "bwd_w32_h6144": (
            (0, 64, None),
            (64, 1168, 132),
            (1168, 3584, 132),
            (3584, 65536, 100),
            (65536, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 64, None),
            (64, 1408, 100),
            (1408, 65536, 100),
            (65536, None, 100),
        ),
    },
    "sm_103a": {
        "bwd_h1536": (
            (0, 64, None),
            (64, 1024, 132),
            (1024, 4096, 100),
            (4096, 14928, 132),
            (14928, 15566, 100),
            (15566, 46592, 100),
            (46592, None, None),
        ),
        "bwd_h2048": (
            (0, 64, None),
            (64, 3072, 100),
            (3072, 4096, 100),
            (4096, 16384, 100),
            (16384, 32768, 100),
            (32768, 58496, 100),
            (58496, 61952, 100),
            (61952, None, 132),
        ),
        "bwd_h256": (
            (0, 1080, None),
            (1080, 5120, None),
            (5120, 12288, None),
            (12288, 40960, None),
            (40960, 97280, None),
            (97280, 720896, None),
            (720896, 1572864, None),
            (1572864, None, None),
        ),
        "bwd_h4096": (
            (0, 32, None),
            (32, 256, None),
            (256, 1024, None),
            (1024, 4096, 132),
            (4096, 50176, 132),
            (50176, 262144, None),
            (262144, None, None),
        ),
        "bwd_h512": (
            (0, 26, None),
            (26, 1024, None),
            (1024, 4096, None),
            (4096, 32768, None),
            (32768, None, None),
        ),
        "bwd_h5120": (
            (0, 64, 100),
            (64, 5120, 132),
            (5120, 61952, 132),
            (61952, 393216, 132),
            (393216, None, 132),
        ),
        "bwd_h6144": (
            (0, 32, None),
            (32, 4096, 132),
            (4096, 49152, 132),
            (49152, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 48, None),
            (48, 56, 100),
            (56, 304, 100),
            (304, 1920, 100),
            (1920, 2816, 100),
            (2816, 4608, 100),
            (4608, 5632, 100),
            (5632, 32768, 100),
            (32768, None, 100),
        ),
        "bwd_w32_h1536": (
            (0, 64, None),
            (64, 320, None),
            (320, 1024, 132),
            (1024, 4096, 100),
            (4096, 14928, 132),
            (14928, 15566, 100),
            (15566, 37888, 100),
            (37888, None, None),
        ),
        "bwd_w32_h2048": (
            (0, 64, None),
            (64, 3072, 100),
            (3072, 4096, 100),
            (4096, 16384, 100),
            (16384, 32768, 100),
            (32768, 58496, 100),
            (58496, 61952, 100),
            (61952, None, 132),
        ),
        "bwd_w32_h256": (
            (0, 1080, None),
            (1080, 5120, None),
            (5120, 12288, None),
            (12288, 40960, None),
            (40960, 97280, None),
            (97280, 720896, None),
            (720896, 1572864, None),
            (1572864, None, None),
        ),
        "bwd_w32_h4096": (
            (0, 32, None),
            (32, 256, None),
            (256, 1024, None),
            (1024, 4096, 132),
            (4096, 9216, 132),
            (9216, 262144, None),
            (262144, None, None),
        ),
        "bwd_w32_h512": (
            (0, 292, None),
            (292, 960, None),
            (960, 32768, None),
            (32768, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 48, 100),
            (48, 5120, 132),
            (5120, 61952, 132),
            (61952, 196608, 132),
            (196608, None, 132),
        ),
        "bwd_w32_h6144": (
            (0, 48, None),
            (48, 320, 132),
            (320, 448, 132),
            (448, 4096, 132),
            (4096, 49152, 132),
            (49152, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 48, None),
            (48, 320, 100),
            (320, 1824, 100),
            (1824, 3584, 100),
            (3584, 4608, 100),
            (4608, 5632, 100),
            (5632, 15872, 100),
            (15872, None, 100),
        ),
    },
    "sm_107a": {
        "bwd_h1536": (
            (0, 64, None),
            (64, 1024, None),
            (1024, 16384, 100),
            (16384, 53248, 100),
            (53248, 65536, None),
            (65536, None, None),
        ),
        "bwd_h2048": (
            (0, 64, None),
            (64, 256, None),
            (256, 1024, 100),
            (1024, 16384, 100),
            (16384, 50176, 100),
            (50176, None, None),
        ),
        "bwd_h256": (
            (0, 1056, None),
            (1056, 3584, None),
            (3584, 36864, None),
            (36864, None, None),
        ),
        "bwd_h4096": (
            (0, 64, None),
            (64, 256, 100),
            (256, 1024, 100),
            (1024, 4096, 100),
            (4096, 36864, 100),
            (36864, 65536, 132),
            (65536, None, 132),
        ),
        "bwd_h512": (
            (0, 77, None),
            (77, 1024, None),
            (1024, 4096, 100),
            (4096, 16384, 100),
            (16384, 49152, 100),
            (49152, None, None),
        ),
        "bwd_h5120": (
            (0, 64, 100),
            (64, 4096, 100),
            (4096, 19968, 100),
            (19968, 49152, 100),
            (49152, None, 132),
        ),
        "bwd_h6144": (
            (0, 64, None),
            (64, 19456, 100),
            (19456, 30720, 100),
            (30720, None, 100),
        ),
        "bwd_residual_h6144": (
            (0, 64, None),
            (64, 448, 100),
            (448, 3584, 100),
            (3584, 7168, 100),
            (7168, None, 100),
        ),
        "bwd_w32_h1536": (
            (0, 64, None),
            (64, 256, None),
            (256, 1024, None),
            (1024, 16384, None),
            (16384, 53248, 100),
            (53248, 65536, None),
            (65536, None, None),
        ),
        "bwd_w32_h2048": (
            (0, 64, None),
            (64, 127, None),
            (127, 256, None),
            (256, 1024, 100),
            (1024, 16384, None),
            (16384, 50176, 100),
            (50176, None, None),
        ),
        "bwd_w32_h256": ((0, 36864, None), (36864, None, None)),
        "bwd_w32_h4096": (
            (0, 64, None),
            (64, 256, None),
            (256, 1024, 100),
            (1024, 4096, 100),
            (4096, 24064, 100),
            (24064, 65536, 132),
            (65536, None, 132),
        ),
        "bwd_w32_h512": (
            (0, 229, None),
            (229, 1024, None),
            (1024, 4096, None),
            (4096, 16384, 100),
            (16384, 49152, 100),
            (49152, None, None),
        ),
        "bwd_w32_h5120": (
            (0, 64, 100),
            (64, 4096, 100),
            (4096, 19968, 132),
            (19968, 49152, 100),
            (49152, None, 132),
        ),
        "bwd_w32_h6144": (
            (0, 64, None),
            (64, 2304, 132),
            (2304, 19456, 100),
            (19456, 28672, 100),
            (28672, None, 100),
        ),
        "bwd_w32_residual_h6144": (
            (0, 64, None),
            (64, 96, 100),
            (96, 448, 100),
            (448, 3584, 100),
            (3584, 7168, 100),
            (7168, None, 100),
        ),
    },
}

# Per architecture and route: the token-count ranges ``(above, at_most)`` whose program needs a
# whole SM and waits beside any kernel resident on every SM, whatever its carveout: the
# deterministic package's whole-SM programs in this package's det-pair windows and forward
# (the exporter's ``DET_WHOLE_SM``; no single-kernel window); ``WHOLE_SM_REMEDY`` the
# deterministic facade's remedy.  The stall tests count these as not measured (det-inherited).
WHOLE_SM: dict[str, dict[str, tuple[tuple[int, int | None], ...]]] = {
    "sm_103a": {
        "bwd_residual_h6144": ((56, 304), (1920, 2816), (4608, 5632)),
        "bwd_w32_residual_h6144": ((48, 320), (1824, 3584), (4608, 5632)),
    },
    "sm_107a": {
        "fwd_h6144": ((4096, None),),
        "fwd_w32_h6144": ((4096, None),),
        "bwd_h1536": ((1024, 53248),),
        "bwd_w32_h1536": ((16384, 53248),),
        "bwd_h2048": ((256, 50176),),
        "bwd_w32_h2048": ((256, 1024), (16384, 50176)),
        "bwd_h4096": ((1024, 36864),),
        "bwd_w32_h4096": ((1024, 24064),),
        "bwd_residual_h6144": ((64, 448), (3584, 7168)),
        "bwd_w32_residual_h6144": ((96, 448), (3584, 7168)),
    },
}
WHOLE_SM_REMEDY = "Beside those, leave some SMs free of the resident kernel."

NOTES = """
Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and at a forward edge between programs of different per-row thread groups ``rstd``
moves by at most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and
``y`` by at most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300
and R200).  A backward window (``deterministic=False``) runs one of two forms.  The
single kernel computes ``dx`` row by row (bitwise reproducible, within one BF16 ulp of
``max(|dx|, rms(dx))`` of the deterministic backward's ``dx``, not always bitwise equal to
it) and adds the ``dw`` sums of its CTAs into FP64 column sums in a
scheduling-dependent order: its ``dw`` is not bitwise reproducible, even for a fixed
``T``.  The top window of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` and on sm_107a of ``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144``
and ``bwd_residual_h6144`` (and of their FP32-weight routes), the windows above T 49152
of sm_107a ``bwd_h512``, above T 65536 of sm_107a ``bwd_h4096`` and of sm_100a
``bwd_h2048``, ``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144``, above T 56256
of sm_100a ``bwd_h5120`` (both weight dtypes) and above T 78784 of sm_100a
``bwd_h1536`` and above T 75776 of ``bwd_w32_h1536``, the lowest window of each of
those routes that runs its top window's program, the lowest window on sm_103a of
``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_w32_h5120``, the lowest window of
``bwd_w32_h512`` on sm_103a and its window from T 227 to 292 on sm_100a, and the
sm_100a ``bwd_h256`` window from 20480 to 92160 rows, run a twosum single kernel: each
thread keeps its ``dw`` column sums exact (error-free FP32 products, TwoSum
compensation) and adds them to the FP64 sums once, so the ``dw`` error does not grow
with ``T``; its ``dw`` is not bitwise reproducible either.  On sm_103a the window of
``bwd_h4096`` and ``bwd_h5120`` below the top window and on sm_107a the top window of
``bwd_h5120`` (and of their FP32-weight routes) run a kblock single kernel: each thread
adds the ``dw`` products of a block of 8 rows in FP32 and keeps the block sums as FP32
TwoSum pairs, which it adds to the FP64 sums once, so the ``dw`` error grows slowly
with ``T`` (at most 0.81 of the per-element budget at the sampled ``T`` up to 131072 on
GB300 and R200; on sm_103a it exceeds the budget at larger ``T``, so its windows there
end at T 262144 on ``bwd_w32_h4096``, T 196608 on ``bwd_w32_h5120``, T 262144 on
``bwd_h4096``, T 393216 on ``bwd_h5120``, the largest sampled ``T`` at or below which
it stays within the budget); its ``dw`` is not bitwise reproducible either.  A window
marked ``(det pair)`` runs the deterministic package's main kernel and fixed-order dw
reduce, rendered into this package; on ``bwd_h5120`` (both weight dtypes) the det pair
windows above T 6144 on sm_100a, 5120 on sm_103a and 19968 on sm_107a run the
deterministic package's compensated programs (compensated per-chunk ``dw`` sums) up to
T 56256 on sm_100a, up to T 61952 on sm_103a and up to T 49152 on
sm_107a, the single kernel above.  On sm_103a ``bwd_h256`` (both weight dtypes) every
window above 1080 rows runs a det pair.  Their main kernels are those the
deterministic package runs for those rows on a device with 152 or more SMs, except
from 1080 to 2048 rows, where it is the main kernel that package runs from 2048 to
5120 rows, and from 97280 to 99840 rows, where it is the main kernel that package runs
from 99840 to 720896 rows (the pair below 99840 rows exceeds 0.85 of the ``dw`` budget
above 97280 rows); their fixed-order dw reduce is that package's, except from 5120 to
12288 and from 720896 to 1572864 rows, where it is the dw reduce that package ran
there earlier.  On ``bwd_h256`` (both weight dtypes) the det pair the deterministic
package runs above 80896 rows on sm_100a also runs from 92160 to 524288 rows on
sm_100a; the twosum single kernel runs above, where that pair's worst ``dw`` element
exceeds 0.85 of the budget at some sampled row counts.  On sm_103a
``bwd_w32_h512`` runs from T 960 to 32768 the det pair the deterministic package runs
from T 4096, for ``dw`` accuracy.  For ``dw`` accuracy, on sm_100a ``bwd_w32_h1536``
from T 1024 to 4096 runs the single kernel of its window from T 32768 to 75776,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 the
single kernel whose twosum form runs their top window, ``bwd_w32_h2048`` from T 336 to
1024 the column-task kernel of its window from T 64 to 336 (its CTAs first sum whole
columns of ``dw`` over every row) and ``bwd_w32_h5120`` from T 1792 to 6144 the single
kernel whose twosum form runs its top window; on sm_107a ``bwd_w32_h6144`` from T 19456
to 28672 runs the det pair the deterministic package runs above T 19456,
``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest token counts,
``bwd_w32_h5120`` from T 4096 to 19968 a row single kernel, and ``bwd_w32_h1536`` from
T 1024 to 16384 and ``bwd_w32_h6144`` from T 2304 to 19456 the single kernel whose
twosum form runs their top window; each replaces a pair whose ``dw`` error exceeds the
FP32 eager reduction's at some sampled ``T`` of that pair's window.  On sm_100a
``bwd_h2048`` and ``bwd_w32_h2048`` from T 45056 to 58880 run the single kernel of
their neighbouring windows, within 2 % of the det pair's time (worst ``dw`` element
over both weight dtypes: 0.969 of the budget for the det pair, 0.800 for the single
kernel; every 64th ``T``, 120 inputs per family).  On sm_107a ``bwd_h6144`` from
T 19456 to 30720 runs, for now, the det pair the deterministic package runs there; the
pair of its window below stays within the ``dw`` budget there and takes 0.97x to 0.99x
the det pair's time.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T 227 run the
deterministic package's column-task program of that window, which is faster there.  On
18 backward routes the lowest window, from T 1 up to a token count from 64 to 229 (1056
to 1080 rows on ``bwd_h256`` and its FP32-weight route and 36864 rows on sm_107a
``bwd_w32_h256``), runs a single kernel for ``dw`` accuracy: there, over 60 inputs of
each of three families, the det pair's median ``dw`` error, or on the FP32-weight,
``bwd_h5120`` and ``bwd_h256`` routes its worst ``dw`` error, exceeds the FP32 eager
reduction's at some sampled ``T`` (measured on GB300 and R200 with 212 SMs).
On ``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the lowest window, up to T 64 (on
sm_103a up to T 32 on ``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_h6144``, up to T 48 on
``bwd_w32_h5120``, ``bwd_w32_h6144``, ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``), runs the fixed-claim form of the twosum column-task
kernel: each CTA takes its rows and its column tasks by its index instead of by claim
tickets, so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of
the ticket form.
"""

FORWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and at a forward edge between programs of different per-row thread groups ``rstd``
moves by at most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and
``y`` by at most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300
and R200).  A backward window (``deterministic=False``) runs one of two forms.  The
single kernel computes ``dx`` row by row (bitwise reproducible, within one BF16 ulp of
``max(|dx|, rms(dx))`` of the deterministic backward's ``dx``, not always bitwise equal to
it) and adds the ``dw`` sums of its CTAs into FP64 column sums in a
scheduling-dependent order: its ``dw`` is not bitwise reproducible, even for a fixed
``T``.  The top window of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` and on sm_107a of ``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144``
and ``bwd_residual_h6144`` (and of their FP32-weight routes), the windows above T 49152
of sm_107a ``bwd_h512``, above T 65536 of sm_107a ``bwd_h4096`` and of sm_100a
``bwd_h2048``, ``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144``, above T 56256
of sm_100a ``bwd_h5120`` (both weight dtypes) and above T 78784 of sm_100a
``bwd_h1536`` and above T 75776 of ``bwd_w32_h1536``, the lowest window of each of
those routes that runs its top window's program, the lowest window on sm_103a of
``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_w32_h5120``, the lowest window of
``bwd_w32_h512`` on sm_103a and its window from T 227 to 292 on sm_100a, and the
sm_100a ``bwd_h256`` window from 20480 to 92160 rows, run a twosum single kernel: each
thread keeps its ``dw`` column sums exact (error-free FP32 products, TwoSum
compensation) and adds them to the FP64 sums once, so the ``dw`` error does not grow
with ``T``; its ``dw`` is not bitwise reproducible either.  On sm_103a the window of
``bwd_h4096`` and ``bwd_h5120`` below the top window and on sm_107a the top window of
``bwd_h5120`` (and of their FP32-weight routes) run a kblock single kernel: each thread
adds the ``dw`` products of a block of 8 rows in FP32 and keeps the block sums as FP32
TwoSum pairs, which it adds to the FP64 sums once, so the ``dw`` error grows slowly
with ``T`` (at most 0.81 of the per-element budget at the sampled ``T`` up to 131072 on
GB300 and R200; on sm_103a it exceeds the budget at larger ``T``, so its windows there
end at T 262144 on ``bwd_w32_h4096``, T 196608 on ``bwd_w32_h5120``, T 262144 on
``bwd_h4096``, T 393216 on ``bwd_h5120``, the largest sampled ``T`` at or below which
it stays within the budget); its ``dw`` is not bitwise reproducible either.  A window
marked ``(det pair)`` runs the deterministic package's main kernel and fixed-order dw
reduce, rendered into this package; on ``bwd_h5120`` (both weight dtypes) the det pair
windows above T 6144 on sm_100a, 5120 on sm_103a and 19968 on sm_107a run the
deterministic package's compensated programs (compensated per-chunk ``dw`` sums) up to
T 56256 on sm_100a, up to T 61952 on sm_103a and up to T 49152 on
sm_107a, the single kernel above.  On sm_103a ``bwd_h256`` (both weight dtypes) every
window above 1080 rows runs a det pair.  Their main kernels are those the
deterministic package runs for those rows on a device with 152 or more SMs, except
from 1080 to 2048 rows, where it is the main kernel that package runs from 2048 to
5120 rows, and from 97280 to 99840 rows, where it is the main kernel that package runs
from 99840 to 720896 rows (the pair below 99840 rows exceeds 0.85 of the ``dw`` budget
above 97280 rows); their fixed-order dw reduce is that package's, except from 5120 to
12288 and from 720896 to 1572864 rows, where it is the dw reduce that package ran
there earlier.  On ``bwd_h256`` (both weight dtypes) the det pair the deterministic
package runs above 80896 rows on sm_100a also runs from 92160 to 524288 rows on
sm_100a; the twosum single kernel runs above, where that pair's worst ``dw`` element
exceeds 0.85 of the budget at some sampled row counts.  On sm_103a
``bwd_w32_h512`` runs from T 960 to 32768 the det pair the deterministic package runs
from T 4096, for ``dw`` accuracy.  For ``dw`` accuracy, on sm_100a ``bwd_w32_h1536``
from T 1024 to 4096 runs the single kernel of its window from T 32768 to 75776,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 the
single kernel whose twosum form runs their top window, ``bwd_w32_h2048`` from T 336 to
1024 the column-task kernel of its window from T 64 to 336 (its CTAs first sum whole
columns of ``dw`` over every row) and ``bwd_w32_h5120`` from T 1792 to 6144 the single
kernel whose twosum form runs its top window; on sm_107a ``bwd_w32_h6144`` from T 19456
to 28672 runs the det pair the deterministic package runs above T 19456,
``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest token counts,
``bwd_w32_h5120`` from T 4096 to 19968 a row single kernel, and ``bwd_w32_h1536`` from
T 1024 to 16384 and ``bwd_w32_h6144`` from T 2304 to 19456 the single kernel whose
twosum form runs their top window; each replaces a pair whose ``dw`` error exceeds the
FP32 eager reduction's at some sampled ``T`` of that pair's window.  On sm_100a
``bwd_h2048`` and ``bwd_w32_h2048`` from T 45056 to 58880 run the single kernel of
their neighbouring windows, within 2 % of the det pair's time (worst ``dw`` element
over both weight dtypes: 0.969 of the budget for the det pair, 0.800 for the single
kernel; every 64th ``T``, 120 inputs per family).  On sm_107a ``bwd_h6144`` from
T 19456 to 30720 runs, for now, the det pair the deterministic package runs there; the
pair of its window below stays within the ``dw`` budget there and takes 0.97x to 0.99x
the det pair's time.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T 227 run the
deterministic package's column-task program of that window, which is faster there.  On
18 backward routes the lowest window, from T 1 up to a token count from 64 to 229 (1056
to 1080 rows on ``bwd_h256`` and its FP32-weight route and 36864 rows on sm_107a
``bwd_w32_h256``), runs a single kernel for ``dw`` accuracy: there, over 60 inputs of
each of three families, the det pair's median ``dw`` error, or on the FP32-weight,
``bwd_h5120`` and ``bwd_h256`` routes its worst ``dw`` error, exceeds the FP32 eager
reduction's at some sampled ``T`` (measured on GB300 and R200 with 212 SMs).
On ``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the lowest window, up to T 64 (on
sm_103a up to T 32 on ``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_h6144``, up to T 48 on
``bwd_w32_h5120``, ``bwd_w32_h6144``, ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``), runs the fixed-claim form of the twosum column-task
kernel: each CTA takes its rows and its column tasks by its index instead of by claim
tickets, so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of
the ticket form.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical outputs on the common rows across it, ``~``
the forward's ulp differences (see above), ``?`` not yet measured.

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

Programs that need a whole SM wait beside any kernel resident on every SM, whatever
its carveout: the deterministic package's whole-SM programs where this package runs
them (``WHOLE_SM``: det-pair windows and forward programs; no single-kernel window).
Beside those, leave some SMs free of the resident kernel.

* ``sm_107a`` ``fwd_h6144``: (4096, inf)
* ``sm_107a`` ``fwd_w32_h6144``: (4096, inf)

* ``sm_100a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_100a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 2048] 780b9833 ~
  (2048, 4096] b44107e0 ~ (4096, inf) 780b9833
* ``sm_100a`` ``fwd_h256``: (0, 8192] 0489c784 = (8192, 40960] 1b284bae =
  (40960, 81920] 86d51eff = (81920, 114688] 1388ab77 = (114688, 196608] 466331ff =
  (196608, 327680] ed9483d8 = (327680, inf) b8f32dd4
* ``sm_100a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_100a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 24576] 3d20ab4a = (24576, inf) 3430e5d9
* ``sm_100a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 =
  (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_w32_h5120``: (0, 640] ba28d2c0 = (640, 6144] 9ecb62c2 =
  (6144, 16384] b49de9b4 = (16384, inf) 9ecb62c2
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e = (768, 1024] 2763cd5e =
  (1024, 6144] 3d619cd1 = (6144, inf) 5f89e5fd
* ``sm_100a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
* ``sm_103a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_103a`` ``fwd_h2048``: (0, 1024] c5a8099e = (1024, 4096] b44107e0 ~
  (4096, inf) 780b9833
* ``sm_103a`` ``fwd_h256``: (0, 8192] 0489c784 = (8192, 20480] bbb05314 =
  (20480, 40960] 1b284bae = (40960, 81920] 86d51eff = (81920, 114688] 466331ff =
  (114688, 196608] 86d51eff = (196608, 327680] ed9483d8 = (327680, 368640] 37f66318 =
  (368640, inf) b8f32dd4
* ``sm_103a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 3072] d447b7f6 = (3072, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_103a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 16384] 3d20ab4a = (16384, inf) 3430e5d9
* ``sm_103a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 =
  (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 640] 1defffdf = (640, 6144] 9ecb62c2 =
  (6144, inf) 9ecb62c2
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e = (1024, 6144] 2763cd5e =
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
* ``sm_107a`` ``fwd_h1536``: (0, 2048] 1ea88523 ~ (2048, 12288] 4cdc4c49 =
  (12288, inf) 682675bd
* ``sm_107a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 8192] a4da8513 =
  (8192, 12288] 0ad4d53c = (12288, 16384] 0ad4d53c = (16384, inf) bde56bd5
* ``sm_107a`` ``fwd_w32_h2048``: (0, 1024] 0da8e18d ~ (1024, 8192] a72c6ece =
  (8192, 12288] 595f5060 = (12288, 16384] 40210950 = (16384, inf) 40210950
* ``sm_107a`` ``fwd_h256``: (0, 6144] c5ca409f = (6144, 32768] bbb05314 =
  (32768, 98304] 466331ff = (98304, inf) 37f66318
* ``sm_107a`` ``fwd_h4096``: (0, 512] 8392d1a0 ~ (512, 1024] fd913ecd =
  (1024, 4096] d447b7f6 = (4096, 20480] bf933e16 = (20480, 49152] bf933e16 =
  (49152, inf) fd913ecd
* ``sm_107a`` ``fwd_w32_h4096``: (0, 512] 4371270e ~ (512, 1024] f0db3068 =
  (1024, 4096] f9f3423b = (4096, 20480] f9f3423b = (20480, 49152] f9f3423b =
  (49152, inf) f9f3423b
* ``sm_107a`` ``fwd_h512``: (0, 256] 7673978d = (256, 6144] 2a42e5c0 =
  (6144, inf) 60f47c5d
* ``sm_107a`` ``fwd_h5120``: (0, 512] fd1097c4 = (512, 1152] 5eef28cd =
  (1152, 6144] b5e173f5 = (6144, 20480] 044d4a2a = (20480, 49152] 044d4a2a =
  (49152, inf) 5eef28cd
* ``sm_107a`` ``fwd_w32_h5120``: (0, 512] ba28d2c0 = (512, 1152] 1defffdf =
  (1152, 6144] 9ecb62c2 = (6144, 20480] b49de9b4 = (20480, 49152] b49de9b4 =
  (49152, inf) b49de9b4
* ``sm_107a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, 4096] cf1d0554 =
  (4096, inf) e7f4221a
* ``sm_107a`` ``fwd_residual_h6144``: (0, 256] e646bbdb ~ (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
"""

BACKWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships one or more programs
and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  The forward programs are the deterministic package's,
with its windows: within one window a row's ``y`` and ``rstd`` do not depend on ``T``,
and at a forward edge between programs of different per-row thread groups ``rstd``
moves by at most two FP32 ulps (three on R200 at the residual H = 6144 edge 1024) and
``y`` by at most one BF16 ulp, on up to about 32 % of the rows (measured on B200, GB300
and R200).  A backward window (``deterministic=False``) runs one of two forms.  The
single kernel computes ``dx`` row by row (bitwise reproducible, within one BF16 ulp of
``max(|dx|, rms(dx))`` of the deterministic backward's ``dx``, not always bitwise equal to
it) and adds the ``dw`` sums of its CTAs into FP64 column sums in a
scheduling-dependent order: its ``dw`` is not bitwise reproducible, even for a fixed
``T``.  The top window of ``bwd_h256`` on sm_100a and sm_107a, on sm_103a of
``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` and on sm_107a of ``bwd_h1536``, ``bwd_h2048``, ``bwd_h6144``
and ``bwd_residual_h6144`` (and of their FP32-weight routes), the windows above T 49152
of sm_107a ``bwd_h512``, above T 65536 of sm_107a ``bwd_h4096`` and of sm_100a
``bwd_h2048``, ``bwd_h4096``, ``bwd_h6144`` and ``bwd_residual_h6144``, above T 56256
of sm_100a ``bwd_h5120`` (both weight dtypes) and above T 78784 of sm_100a
``bwd_h1536`` and above T 75776 of ``bwd_w32_h1536``, the lowest window of each of
those routes that runs its top window's program, the lowest window on sm_103a of
``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_w32_h5120``, the lowest window of
``bwd_w32_h512`` on sm_103a and its window from T 227 to 292 on sm_100a, and the
sm_100a ``bwd_h256`` window from 20480 to 92160 rows, run a twosum single kernel: each
thread keeps its ``dw`` column sums exact (error-free FP32 products, TwoSum
compensation) and adds them to the FP64 sums once, so the ``dw`` error does not grow
with ``T``; its ``dw`` is not bitwise reproducible either.  On sm_103a the window of
``bwd_h4096`` and ``bwd_h5120`` below the top window and on sm_107a the top window of
``bwd_h5120`` (and of their FP32-weight routes) run a kblock single kernel: each thread
adds the ``dw`` products of a block of 8 rows in FP32 and keeps the block sums as FP32
TwoSum pairs, which it adds to the FP64 sums once, so the ``dw`` error grows slowly
with ``T`` (at most 0.81 of the per-element budget at the sampled ``T`` up to 131072 on
GB300 and R200; on sm_103a it exceeds the budget at larger ``T``, so its windows there
end at T 262144 on ``bwd_w32_h4096``, T 196608 on ``bwd_w32_h5120``, T 262144 on
``bwd_h4096``, T 393216 on ``bwd_h5120``, the largest sampled ``T`` at or below which
it stays within the budget); its ``dw`` is not bitwise reproducible either.  A window
marked ``(det pair)`` runs the deterministic package's main kernel and fixed-order dw
reduce, rendered into this package; on ``bwd_h5120`` (both weight dtypes) the det pair
windows above T 6144 on sm_100a, 5120 on sm_103a and 19968 on sm_107a run the
deterministic package's compensated programs (compensated per-chunk ``dw`` sums) up to
T 56256 on sm_100a, up to T 61952 on sm_103a and up to T 49152 on
sm_107a, the single kernel above.  On sm_103a ``bwd_h256`` (both weight dtypes) every
window above 1080 rows runs a det pair.  Their main kernels are those the
deterministic package runs for those rows on a device with 152 or more SMs, except
from 1080 to 2048 rows, where it is the main kernel that package runs from 2048 to
5120 rows, and from 97280 to 99840 rows, where it is the main kernel that package runs
from 99840 to 720896 rows (the pair below 99840 rows exceeds 0.85 of the ``dw`` budget
above 97280 rows); their fixed-order dw reduce is that package's, except from 5120 to
12288 and from 720896 to 1572864 rows, where it is the dw reduce that package ran
there earlier.  On ``bwd_h256`` (both weight dtypes) the det pair the deterministic
package runs above 80896 rows on sm_100a also runs from 92160 to 524288 rows on
sm_100a; the twosum single kernel runs above, where that pair's worst ``dw`` element
exceeds 0.85 of the budget at some sampled row counts.  On sm_103a
``bwd_w32_h512`` runs from T 960 to 32768 the det pair the deterministic package runs
from T 4096, for ``dw`` accuracy.  For ``dw`` accuracy, on sm_100a ``bwd_w32_h1536``
from T 1024 to 4096 runs the single kernel of its window from T 32768 to 75776,
``bwd_w32_h2048`` from T 1024 to 4096 and ``bwd_w32_h4096`` from T 1792 to 3584 the
single kernel whose twosum form runs their top window, ``bwd_w32_h2048`` from T 336 to
1024 the column-task kernel of its window from T 64 to 336 (its CTAs first sum whole
columns of ``dw`` over every row) and ``bwd_w32_h5120`` from T 1792 to 6144 the single
kernel whose twosum form runs its top window; on sm_107a ``bwd_w32_h6144`` from T 19456
to 28672 runs the det pair the deterministic package runs above T 19456,
``bwd_w32_h2048`` from T 1024 to 16384 the single kernel of its largest token counts,
``bwd_w32_h5120`` from T 4096 to 19968 a row single kernel, and ``bwd_w32_h1536`` from
T 1024 to 16384 and ``bwd_w32_h6144`` from T 2304 to 19456 the single kernel whose
twosum form runs their top window; each replaces a pair whose ``dw`` error exceeds the
FP32 eager reduction's at some sampled ``T`` of that pair's window.  On sm_100a
``bwd_h2048`` and ``bwd_w32_h2048`` from T 45056 to 58880 run the single kernel of
their neighbouring windows, within 2 % of the det pair's time (worst ``dw`` element
over both weight dtypes: 0.969 of the budget for the det pair, 0.800 for the single
kernel; every 64th ``T``, 120 inputs per family).  On sm_107a ``bwd_h6144`` from
T 19456 to 30720 runs, for now, the det pair the deterministic package runs there; the
pair of its window below stays within the ``dw`` budget there and takes 0.97x to 0.99x
the det pair's time.  On sm_100a ``bwd_h512`` and ``bwd_w32_h512`` up to T 227 run the
deterministic package's column-task program of that window, which is faster there.  On
18 backward routes the lowest window, from T 1 up to a token count from 64 to 229 (1056
to 1080 rows on ``bwd_h256`` and its FP32-weight route and 36864 rows on sm_107a
``bwd_w32_h256``), runs a single kernel for ``dw`` accuracy: there, over 60 inputs of
each of three families, the det pair's median ``dw`` error, or on the FP32-weight,
``bwd_h5120`` and ``bwd_h256`` routes its worst ``dw`` error, exceeds the FP32 eager
reduction's at some sampled ``T`` (measured on GB300 and R200 with 212 SMs).
On ``bwd_h1536``, ``bwd_h2048``, ``bwd_h4096``, ``bwd_h5120``, ``bwd_h6144`` and
``bwd_residual_h6144`` (and their FP32-weight routes) the lowest window, up to T 64 (on
sm_103a up to T 32 on ``bwd_h4096``, ``bwd_w32_h4096`` and ``bwd_h6144``, up to T 48 on
``bwd_w32_h5120``, ``bwd_w32_h6144``, ``bwd_residual_h6144`` and
``bwd_w32_residual_h6144``), runs the fixed-claim form of the twosum column-task
kernel: each CTA takes its rows and its column tasks by its index instead of by claim
tickets, so no CTA waits for another, and its ``dx`` and ``dw`` are bitwise those of
the ticket form.

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge: ``=`` bitwise-identical ``dx`` on the common rows across it (``dw``
sums every row and is not bitwise reproducible), ``~`` the forward's ulp differences
(see above), ``?`` not yet measured.

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows and are not listed; a route re-cut for the FP32
weight (another twin in a window) is listed with its own windows.

A backward window marked ``(det pair)`` runs the deterministic package's main kernel
(the program named) and its dw reduce; the others run the single kernel.

Beside a kernel that holds shared memory on every SM (one CTA per SM with a large
dynamic shared-memory allocation), a backward window waits until that kernel exits
when the shared memory it leaves is too small for the window's program.  Beside
those, leave some SMs free of the resident kernel.  Measured with
100 / 132 KB per CTA (``SMEM_WAITS``): ``>= K`` after a window means it waits beside
K KB or more, ``-`` that it starts beside every measured size; an architecture or
route not listed is unmeasured.

* ``sm_100a`` ``bwd_h1536``: (0, 64] -, (64, 384] -, (384, 1024] >= 132, (1024, 4096] >= 100, (4096, 32768] >= 100, (32768, 36864] >= 132, (36864, 78784] -, (78784, inf) -
* ``sm_100a`` ``bwd_h2048``: (0, 64] -, (64, 336] -, (336, 3072] >= 100, (3072, 4096] >= 100, (4096, 7168] >= 100, (7168, 11264] >= 132, (11264, 13608] >= 100, (13608, 45056] >= 132, (45056, 58880] >= 132, (58880, 65536] >= 132, (65536, inf) >= 132
* ``sm_100a`` ``bwd_h256``: (0, 20480] -, (20480, 92160] -, (92160, 524288] -, (524288, inf) -
* ``sm_100a`` ``bwd_h4096``: (0, 64] -, (64, 1152] -, (1152, 4096] >= 132, (4096, 4608] >= 132, (4608, 36864] -, (36864, 44032] >= 132, (44032, 65536] -, (65536, inf) -
* ``sm_100a`` ``bwd_h512``: (0, 227] -, (227, 1024] -, (1024, 1280] >= 132, (1280, inf) -
* ``sm_100a`` ``bwd_h5120``: (0, 64] >= 100, (64, 896] -, (896, 6144] -, (6144, 56256] -, (56256, inf) >= 132
* ``sm_100a`` ``bwd_h6144``: (0, 64] -, (64, 1152] >= 132, (1152, 4096] >= 132, (4096, 5632] >= 132, (5632, 65536] >= 100, (65536, inf) >= 100
* ``sm_100a`` ``bwd_residual_h6144``: (0, 64] -, (64, 1408] >= 100, (1408, 65536] >= 100, (65536, inf) >= 100
* ``sm_100a`` ``bwd_w32_h1536``: (0, 64] -, (64, 384] -, (384, 1024] >= 132, (1024, 4096] -, (4096, 32768] >= 100, (32768, 75776] -, (75776, inf) -
* ``sm_100a`` ``bwd_w32_h2048``: (0, 64] -, (64, 336] -, (336, 1024] -, (1024, 4096] >= 132, (4096, 15403] >= 100, (15403, 45056] >= 132, (45056, 58880] >= 132, (58880, 65536] >= 132, (65536, inf) >= 132
* ``sm_100a`` ``bwd_w32_h256``: (0, 20480] -, (20480, 92160] -, (92160, 524288] -, (524288, inf) -
* ``sm_100a`` ``bwd_w32_h4096``: (0, 64] -, (64, 1792] -, (1792, 3584] -, (3584, 36864] -, (36864, 44032] >= 132, (44032, 65536] -, (65536, inf) -
* ``sm_100a`` ``bwd_w32_h512``: (0, 227] -, (227, 292] -, (292, 960] -, (960, inf) -
* ``sm_100a`` ``bwd_w32_h5120``: (0, 64] >= 100, (64, 1792] -, (1792, 6144] >= 132, (6144, 56256] -, (56256, inf) >= 132
* ``sm_100a`` ``bwd_w32_h6144``: (0, 64] -, (64, 1168] >= 132, (1168, 3584] >= 132, (3584, 65536] >= 100, (65536, inf) >= 100
* ``sm_100a`` ``bwd_w32_residual_h6144``: (0, 64] -, (64, 1408] >= 100, (1408, 65536] >= 100, (65536, inf) >= 100
* ``sm_103a`` ``bwd_h1536``: (0, 64] -, (64, 1024] >= 132, (1024, 4096] >= 100, (4096, 14928] >= 132, (14928, 15566] >= 100, (15566, 46592] >= 100, (46592, inf) -
* ``sm_103a`` ``bwd_h2048``: (0, 64] -, (64, 3072] >= 100, (3072, 4096] >= 100, (4096, 16384] >= 100, (16384, 32768] >= 100, (32768, 58496] >= 100, (58496, 61952] >= 100, (61952, inf) >= 132
* ``sm_103a`` ``bwd_h256``: (0, 1080] -, (1080, 5120] -, (5120, 12288] -, (12288, 40960] -, (40960, 97280] -, (97280, 720896] -, (720896, 1572864] -, (1572864, inf) -
* ``sm_103a`` ``bwd_h4096``: (0, 32] -, (32, 256] -, (256, 1024] -, (1024, 4096] >= 132, (4096, 50176] >= 132, (50176, 262144] -, (262144, inf) -
* ``sm_103a`` ``bwd_h512``: (0, 26] -, (26, 1024] -, (1024, 4096] -, (4096, 32768] -, (32768, inf) -
* ``sm_103a`` ``bwd_h5120``: (0, 64] >= 100, (64, 5120] >= 132, (5120, 61952] >= 132, (61952, 393216] >= 132, (393216, inf) >= 132
* ``sm_103a`` ``bwd_h6144``: (0, 32] -, (32, 4096] >= 132, (4096, 49152] >= 132, (49152, inf) >= 100
* ``sm_103a`` ``bwd_residual_h6144``: (0, 48] -, (48, 56] >= 100, (56, 304] >= 100, (304, 1920] >= 100, (1920, 2816] >= 100, (2816, 4608] >= 100, (4608, 5632] >= 100, (5632, 32768] >= 100, (32768, inf) >= 100
* ``sm_103a`` ``bwd_w32_h1536``: (0, 64] -, (64, 320] -, (320, 1024] >= 132, (1024, 4096] >= 100, (4096, 14928] >= 132, (14928, 15566] >= 100, (15566, 37888] >= 100, (37888, inf) -
* ``sm_103a`` ``bwd_w32_h2048``: (0, 64] -, (64, 3072] >= 100, (3072, 4096] >= 100, (4096, 16384] >= 100, (16384, 32768] >= 100, (32768, 58496] >= 100, (58496, 61952] >= 100, (61952, inf) >= 132
* ``sm_103a`` ``bwd_w32_h256``: (0, 1080] -, (1080, 5120] -, (5120, 12288] -, (12288, 40960] -, (40960, 97280] -, (97280, 720896] -, (720896, 1572864] -, (1572864, inf) -
* ``sm_103a`` ``bwd_w32_h4096``: (0, 32] -, (32, 256] -, (256, 1024] -, (1024, 4096] >= 132, (4096, 9216] >= 132, (9216, 262144] -, (262144, inf) -
* ``sm_103a`` ``bwd_w32_h512``: (0, 292] -, (292, 960] -, (960, 32768] -, (32768, inf) -
* ``sm_103a`` ``bwd_w32_h5120``: (0, 48] >= 100, (48, 5120] >= 132, (5120, 61952] >= 132, (61952, 196608] >= 132, (196608, inf) >= 132
* ``sm_103a`` ``bwd_w32_h6144``: (0, 48] -, (48, 320] >= 132, (320, 448] >= 132, (448, 4096] >= 132, (4096, 49152] >= 132, (49152, inf) >= 100
* ``sm_103a`` ``bwd_w32_residual_h6144``: (0, 48] -, (48, 320] >= 100, (320, 1824] >= 100, (1824, 3584] >= 100, (3584, 4608] >= 100, (4608, 5632] >= 100, (5632, 15872] >= 100, (15872, inf) >= 100
* ``sm_107a`` ``bwd_h1536``: (0, 64] -, (64, 1024] -, (1024, 16384] >= 100, (16384, 53248] >= 100, (53248, 65536] -, (65536, inf) -
* ``sm_107a`` ``bwd_h2048``: (0, 64] -, (64, 256] -, (256, 1024] >= 100, (1024, 16384] >= 100, (16384, 50176] >= 100, (50176, inf) -
* ``sm_107a`` ``bwd_h256``: (0, 1056] -, (1056, 3584] -, (3584, 36864] -, (36864, inf) -
* ``sm_107a`` ``bwd_h4096``: (0, 64] -, (64, 256] >= 100, (256, 1024] >= 100, (1024, 4096] >= 100, (4096, 36864] >= 100, (36864, 65536] >= 132, (65536, inf) >= 132
* ``sm_107a`` ``bwd_h512``: (0, 77] -, (77, 1024] -, (1024, 4096] >= 100, (4096, 16384] >= 100, (16384, 49152] >= 100, (49152, inf) -
* ``sm_107a`` ``bwd_h5120``: (0, 64] >= 100, (64, 4096] >= 100, (4096, 19968] >= 100, (19968, 49152] >= 100, (49152, inf) >= 132
* ``sm_107a`` ``bwd_h6144``: (0, 64] -, (64, 19456] >= 100, (19456, 30720] >= 100, (30720, inf) >= 100
* ``sm_107a`` ``bwd_residual_h6144``: (0, 64] -, (64, 448] >= 100, (448, 3584] >= 100, (3584, 7168] >= 100, (7168, inf) >= 100
* ``sm_107a`` ``bwd_w32_h1536``: (0, 64] -, (64, 256] -, (256, 1024] -, (1024, 16384] -, (16384, 53248] >= 100, (53248, 65536] -, (65536, inf) -
* ``sm_107a`` ``bwd_w32_h2048``: (0, 64] -, (64, 127] -, (127, 256] -, (256, 1024] >= 100, (1024, 16384] -, (16384, 50176] >= 100, (50176, inf) -
* ``sm_107a`` ``bwd_w32_h256``: (0, 36864] -, (36864, inf) -
* ``sm_107a`` ``bwd_w32_h4096``: (0, 64] -, (64, 256] -, (256, 1024] >= 100, (1024, 4096] >= 100, (4096, 24064] >= 100, (24064, 65536] >= 132, (65536, inf) >= 132
* ``sm_107a`` ``bwd_w32_h512``: (0, 229] -, (229, 1024] -, (1024, 4096] -, (4096, 16384] >= 100, (16384, 49152] >= 100, (49152, inf) -
* ``sm_107a`` ``bwd_w32_h5120``: (0, 64] >= 100, (64, 4096] >= 100, (4096, 19968] >= 132, (19968, 49152] >= 100, (49152, inf) >= 132
* ``sm_107a`` ``bwd_w32_h6144``: (0, 64] -, (64, 2304] >= 132, (2304, 19456] >= 100, (19456, 28672] >= 100, (28672, inf) >= 100
* ``sm_107a`` ``bwd_w32_residual_h6144``: (0, 64] -, (64, 96] >= 100, (96, 448] >= 100, (448, 3584] >= 100, (3584, 7168] >= 100, (7168, inf) >= 100

Programs that need a whole SM wait beside any kernel resident on every SM, whatever
its carveout: the deterministic package's whole-SM programs where this package runs
them (``WHOLE_SM``: det-pair windows and forward programs; no single-kernel window).
Beside those, leave some SMs free of the resident kernel.

* ``sm_103a`` ``bwd_residual_h6144``: (56, 304], (1920, 2816], (4608, 5632]
* ``sm_103a`` ``bwd_w32_residual_h6144``: (48, 320], (1824, 3584], (4608, 5632]
* ``sm_107a`` ``bwd_h1536``: (1024, 53248]
* ``sm_107a`` ``bwd_w32_h1536``: (16384, 53248]
* ``sm_107a`` ``bwd_h2048``: (256, 50176]
* ``sm_107a`` ``bwd_w32_h2048``: (256, 1024], (16384, 50176]
* ``sm_107a`` ``bwd_h4096``: (1024, 36864]
* ``sm_107a`` ``bwd_w32_h4096``: (1024, 24064]
* ``sm_107a`` ``bwd_residual_h6144``: (64, 448], (3584, 7168]
* ``sm_107a`` ``bwd_w32_residual_h6144``: (96, 448], (3584, 7168]

* ``sm_100a`` ``bwd_h256``: (0, 20480] 195c4317 = (20480, 92160] 172a47a1 =
  (92160, 524288] 083ea823 (det pair) = (524288, inf) 172a47a1
* ``sm_100a`` ``bwd_h512``: (0, 227] 5f3e3fcc = (227, 1024] 09601d42 (det pair) =
  (1024, 1280] 123db0b2 (det pair) = (1280, inf) d0bde38a
* ``sm_100a`` ``bwd_w32_h512``: (0, 227] 942503b8 = (227, 292] aa7c5a54 =
  (292, 960] 9f07cea1 (det pair) = (960, inf) fd08c6f4
* ``sm_100a`` ``bwd_h1536``: (0, 64] 38e972a9 = (64, 384] dc478ac1 =
  (384, 1024] 7dc29629 (det pair) = (1024, 4096] c1a9efdf (det pair) =
  (4096, 32768] 21c8bb55 (det pair) = (32768, 36864] b457c4d1 (det pair) =
  (36864, 78784] ae610afd = (78784, inf) b42efd6b
* ``sm_100a`` ``bwd_w32_h1536``: (0, 64] d5a621af = (64, 384] 3ddc5ca5 =
  (384, 1024] baff9ecb (det pair) = (1024, 4096] 4b6981c1 =
  (4096, 32768] 4bf450b6 (det pair) = (32768, 75776] 4b6981c1 = (75776, inf) e2b99fee
* ``sm_100a`` ``bwd_h2048``: (0, 64] ef052c09 = (64, 336] 64e6aa82 =
  (336, 3072] caf6978e (det pair) = (3072, 4096] 9ac8d21e (det pair) =
  (4096, 7168] c73074cf (det pair) = (7168, 11264] 0fa4f255 =
  (11264, 13608] c73074cf (det pair) = (13608, 45056] 0fa4f255 = (45056, 58880] 0fa4f255
  = (58880, 65536] 0fa4f255 = (65536, inf) 83f905ff
* ``sm_100a`` ``bwd_w32_h2048``: (0, 64] 382e4578 = (64, 336] 17493054 =
  (336, 1024] 17493054 = (1024, 4096] ba7dbf6a = (4096, 15403] b187e598 (det pair) =
  (15403, 45056] ba7dbf6a = (45056, 58880] ba7dbf6a = (58880, 65536] ba7dbf6a =
  (65536, inf) 9f775c72
* ``sm_100a`` ``bwd_h4096``: (0, 64] 231e01ec = (64, 1152] 4822793e =
  (1152, 4096] 9a830956 (det pair) = (4096, 4608] 35a0b0ca (det pair) =
  (4608, 36864] 15e157d2 = (36864, 44032] 35a0b0ca (det pair) = (44032, 65536] 15e157d2
  = (65536, inf) c67fc8b9
* ``sm_100a`` ``bwd_w32_h4096``: (0, 64] ce1f7d21 = (64, 1792] 86160705 =
  (1792, 3584] f7573bcb = (3584, 36864] f7573bcb = (36864, 44032] 626aed74 (det pair) =
  (44032, 65536] f7573bcb = (65536, inf) 9ebd0232
* ``sm_100a`` ``bwd_h5120``: (0, 64] cc082d1f = (64, 896] 080e0a6c =
  (896, 6144] b482c84d (det pair) = (6144, 56256] c145ef19 (det pair) =
  (56256, inf) ade9f6a4
* ``sm_100a`` ``bwd_w32_h5120``: (0, 64] 244b7d84 = (64, 1792] 68f6e584 =
  (1792, 6144] de8f6091 = (6144, 56256] 9479b129 (det pair) = (56256, inf) fc2a52a2
* ``sm_100a`` ``bwd_h6144``: (0, 64] d70f2562 = (64, 1152] 44508df7 =
  (1152, 4096] 5c804aed (det pair) = (4096, 5632] b93bf060 (det pair) =
  (5632, 65536] 9433061d = (65536, inf) 10a41132
* ``sm_100a`` ``bwd_residual_h6144``: (0, 64] 7b0f5a6a = (64, 1408] 864e84f3 =
  (1408, 65536] 3fbfdd75 = (65536, inf) 1609f796
* ``sm_100a`` ``bwd_w32_h6144``: (0, 64] 9b379236 = (64, 1168] b3e6531e =
  (1168, 3584] 41b80b6a (det pair) = (3584, 65536] a562656c = (65536, inf) 45cf60e3
* ``sm_103a`` ``bwd_h256``: (0, 1080] 195c4317 = (1080, 5120] af005462 (det pair) =
  (5120, 12288] a385ae16 (det pair) = (12288, 40960] af005462 (det pair) =
  (40960, 97280] a385ae16 (det pair) = (97280, 720896] 083ea823 (det pair) =
  (720896, 1572864] 0291949b (det pair) = (1572864, inf) c18297d7 (det pair)
* ``sm_103a`` ``bwd_h512``: (0, 26] dd5c218d = (26, 1024] 09601d42 (det pair) =
  (1024, 4096] 8e7f4d81 (det pair) = (4096, 32768] 274b0055 (det pair) =
  (32768, inf) cfaae3ce (det pair)
* ``sm_103a`` ``bwd_w32_h512``: (0, 292] 4724ad6a = (292, 960] 9f07cea1 (det pair) =
  (960, 32768] 56ee8e66 (det pair) = (32768, inf) a546f410 (det pair)
* ``sm_103a`` ``bwd_h1536``: (0, 64] 38e972a9 = (64, 1024] 7dc29629 (det pair) =
  (1024, 4096] c1a9efdf (det pair) = (4096, 14928] 1df7db17 (det pair) =
  (14928, 15566] 4df08ce5 (det pair) = (15566, 46592] f522f5b8 (det pair) =
  (46592, inf) 327bfeb4
* ``sm_103a`` ``bwd_w32_h1536``: (0, 64] d5a621af = (64, 320] 963a7ca8 =
  (320, 1024] baff9ecb (det pair) = (1024, 4096] 17d9cdfd (det pair) =
  (4096, 14928] a99e2b38 (det pair) = (14928, 15566] 7a72e97b (det pair) =
  (15566, 37888] 109f19bb (det pair) = (37888, inf) 963a7ca8
* ``sm_103a`` ``bwd_h2048``: (0, 64] ef052c09 = (64, 3072] caf6978e (det pair) =
  (3072, 4096] 9ac8d21e (det pair) = (4096, 16384] caf6978e (det pair) =
  (16384, 32768] caf6978e (det pair) = (32768, 58496] 302f64db (det pair) =
  (58496, 61952] 018f0cd7 (det pair) = (61952, inf) 8bd89a4c
* ``sm_103a`` ``bwd_h4096``: (0, 32] 3dbab232 = (32, 256] 50044ee3 (det pair) =
  (256, 1024] ad2bb26e (det pair) = (1024, 4096] 24643edc (det pair) =
  (4096, 50176] 35a0b0ca (det pair) = (50176, 262144] 8250eb92 = (262144, inf) 8c8bebec
* ``sm_103a`` ``bwd_w32_h4096``: (0, 32] c1b9ee6b = (32, 256] 3fd77c8c (det pair) =
  (256, 1024] 16545056 (det pair) = (1024, 4096] 78c26a53 (det pair) =
  (4096, 9216] 626aed74 (det pair) = (9216, 262144] 2911faf7 = (262144, inf) 131a495c
* ``sm_103a`` ``bwd_h5120``: (0, 64] cc082d1f = (64, 5120] 2fc4932f (det pair) =
  (5120, 61952] 7ffb443a (det pair) = (61952, 393216] b523a764 = (393216, inf) 9e4c0825
* ``sm_103a`` ``bwd_w32_h5120``: (0, 48] 244b7d84 = (48, 5120] 79811a76 (det pair) =
  (5120, 61952] f9f7d4cd (det pair) = (61952, 196608] 90598c14 = (196608, inf) f165933c
* ``sm_103a`` ``bwd_h6144``: (0, 32] d70f2562 = (32, 4096] 5c804aed (det pair) =
  (4096, 49152] 322574af (det pair) = (49152, inf) 582b2a20
* ``sm_103a`` ``bwd_residual_h6144``: (0, 48] 7b0f5a6a = (48, 56] 864e84f3 =
  (56, 304] 0dcdc563 (det pair) = (304, 1920] 864e84f3 =
  (1920, 2816] 0dcdc563 (det pair) = (2816, 4608] 115dcc71 =
  (4608, 5632] 0dcdc563 (det pair) = (5632, 32768] 115dcc71 = (32768, inf) 52831494
* ``sm_103a`` ``bwd_w32_h6144``: (0, 48] 9b379236 = (48, 320] 41b80b6a (det pair) =
  (320, 448] b3e6531e = (448, 4096] 41b80b6a (det pair) =
  (4096, 49152] 9a8c0396 (det pair) = (49152, inf) 4f9a76e6
* ``sm_103a`` ``bwd_w32_residual_h6144``: (0, 48] ac9d69f9 =
  (48, 320] 56f988dd (det pair) = (320, 1824] 71d36617 =
  (1824, 3584] 56f988dd (det pair) = (3584, 4608] 3c390339 =
  (4608, 5632] 56f988dd (det pair) = (5632, 15872] 3c390339 = (15872, inf) 56c46a20
* ``sm_107a`` ``bwd_h256``: (0, 1056] 195c4317 = (1056, 3584] 5450d398 (det pair) =
  (3584, 36864] 195c4317 = (36864, inf) 172a47a1
* ``sm_107a`` ``bwd_w32_h256``: (0, 36864] 8de01314 = (36864, inf) 7c2aa4b4
* ``sm_107a`` ``bwd_h512``: (0, 77] 5f3e3fcc = (77, 1024] 3feed502 (det pair) =
  (1024, 4096] 3feed502 (det pair) = (4096, 16384] dfaf378a (det pair) =
  (16384, 49152] eb141cc9 (det pair) = (49152, inf) 30501cce
* ``sm_107a`` ``bwd_w32_h512``: (0, 229] 942503b8 = (229, 1024] b841d64b (det pair) =
  (1024, 4096] fd08c6f4 = (4096, 16384] 383f709c (det pair) =
  (16384, 49152] 03efa681 (det pair) = (49152, inf) 26fbe8e6
* ``sm_107a`` ``bwd_h1536``: (0, 64] 38e972a9 = (64, 1024] 7dc29629 (det pair) =
  (1024, 16384] 6a428da2 (det pair) = (16384, 53248] 56d26996 (det pair) =
  (53248, 65536] 3f75f4a8 = (65536, inf) dca25ace
* ``sm_107a`` ``bwd_w32_h1536``: (0, 64] d5a621af = (64, 256] 3ddc5ca5 =
  (256, 1024] baff9ecb (det pair) = (1024, 16384] 7c0dd63f =
  (16384, 53248] de8e07d4 (det pair) = (53248, 65536] 7c0dd63f = (65536, inf) ccd94d87
* ``sm_107a`` ``bwd_h2048``: (0, 64] ef052c09 = (64, 256] 84efa2af (det pair) =
  (256, 1024] 845cdcd8 (det pair) = (1024, 16384] 81c3d8ab (det pair) =
  (16384, 50176] 845cdcd8 (det pair) = (50176, inf) 1a3df5ac
* ``sm_107a`` ``bwd_w32_h2048``: (0, 64] 382e4578 = (64, 127] 7a3e5eb7 =
  (127, 256] 6b13aa49 (det pair) = (256, 1024] 838a9a40 (det pair) =
  (1024, 16384] 810989bd = (16384, 50176] 838a9a40 (det pair) = (50176, inf) 810989bd
* ``sm_107a`` ``bwd_h4096``: (0, 64] 231e01ec = (64, 256] 0da5dd4b (det pair) =
  (256, 1024] a647fccc (det pair) = (1024, 4096] d41a514f (det pair) =
  (4096, 36864] ee4a9053 (det pair) = (36864, 65536] f11a8616 = (65536, inf) 5e9d7baa
* ``sm_107a`` ``bwd_w32_h4096``: (0, 64] ce1f7d21 = (64, 256] 0af20ef9 =
  (256, 1024] dc439f83 (det pair) = (1024, 4096] b0863320 (det pair) =
  (4096, 24064] 0f1859b9 (det pair) = (24064, 65536] a1f537fc = (65536, inf) 23cee954
* ``sm_107a`` ``bwd_h5120``: (0, 64] cc082d1f = (64, 4096] 838c077d (det pair) =
  (4096, 19968] 6816fb80 (det pair) = (19968, 49152] b8b0913b (det pair) =
  (49152, inf) 56634f53
* ``sm_107a`` ``bwd_w32_h5120``: (0, 64] 244b7d84 = (64, 4096] cd612de5 (det pair) =
  (4096, 19968] dd795e84 = (19968, 49152] 7b9e10b2 (det pair) = (49152, inf) 6cedccdf
* ``sm_107a`` ``bwd_h6144``: (0, 64] d70f2562 = (64, 19456] 5d292172 (det pair) =
  (19456, 30720] 19e18371 (det pair) = (30720, inf) 083b5f2e
* ``sm_107a`` ``bwd_residual_h6144``: (0, 64] 7b0f5a6a = (64, 448] 0dcdc563 (det pair) =
  (448, 3584] 864e84f3 = (3584, 7168] 0dcdc563 (det pair) = (7168, inf) c07ab5f1
* ``sm_107a`` ``bwd_w32_h6144``: (0, 64] 9b379236 = (64, 2304] 44c5b870 =
  (2304, 19456] e77387f1 = (19456, 28672] b83f8b3d (det pair) = (28672, inf) 25f9b466
* ``sm_107a`` ``bwd_w32_residual_h6144``: (0, 64] ac9d69f9 = (64, 96] c8bc10e7 =
  (96, 448] 56f988dd (det pair) = (448, 3584] 71d36617 =
  (3584, 7168] 56f988dd (det pair) = (7168, inf) e655055e
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "BACKWARD_SECTION",
    "DET_PAIR_ROUTES",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
    "SMEM_SIDE_KB",
    "SMEM_WAITS",
    "WHOLE_SM",
    "WHOLE_SM_REMEDY",
]
