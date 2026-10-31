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

Program selection by token count of the Cake BF16 RMSNorm training kernels
(SM100 / SM103 / SM107), rendered from the registry of this package
(:mod:`.cake_jit`) by its exporter; do not edit.  ``PROGRAM_WINDOWS`` is the
table ``cake_jit.program_windows`` derives from the registry's ``MODULES`` and
``ROUTES`` at call time; ``EDGES`` the measured outcome at every window edge;
``NOTES`` the contract of the outputs across the windows; ``SECTIONS`` the text
the host module appends to the forward and backward docstrings.  The package
tests hold the table to the registry and the edges to the outputs.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "bwd_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (4096, 32768, "cake_rmsnorm_det_train_21c8bb55c8d672b63952"),
            (32768, None, "cake_rmsnorm_det_train_b457c4d134e44476dc6f"),
        ),
        "bwd_w32_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (4096, 32768, "cake_rmsnorm_det_train_4bf450b6af0314ed03be"),
            (32768, None, "cake_rmsnorm_det_train_8777b1c8538527c9aaf7"),
        ),
        "bwd_h2048": (
            (0, 3072, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (3072, 4096, "cake_rmsnorm_det_train_9ac8d21e46a658e7913f"),
            (4096, 16384, "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6"),
            (16384, None, "cake_rmsnorm_det_train_ceeef043ac64f37fc529"),
        ),
        "bwd_w32_h2048": (
            (0, 3072, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (3072, 4096, "cake_rmsnorm_det_train_d96a14fe8882bbd9ec2e"),
            (4096, 16384, "cake_rmsnorm_det_train_b187e5989302def22b2d"),
            (16384, None, "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9"),
        ),
        "bwd_h256": (
            (0, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 5120, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (12288, 40960, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (40960, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 5120, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (12288, 40960, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (40960, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 256, "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9"),
            (256, 1024, "cake_rmsnorm_det_train_59270fb249dcba1ff679"),
            (1024, 4096, "cake_rmsnorm_det_train_9a830956447d4e304914"),
            (4096, None, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
        ),
        "bwd_w32_h4096": (
            (0, 256, "cake_rmsnorm_det_train_20c64c610d22c6db4c05"),
            (256, 1024, "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1"),
            (1024, 4096, "cake_rmsnorm_det_train_0c84999caf2b86c2e09e"),
            (4096, None, "cake_rmsnorm_det_train_626aed743358c27513f1"),
        ),
        "bwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_123db0b2f7fb38cfab03"),
            (4096, 32768, "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036"),
            (32768, None, "cake_rmsnorm_det_train_8877322a3002e3a4fefd"),
        ),
        "bwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_632ca6ff01848a71f9f2"),
            (4096, 32768, "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441"),
            (32768, None, "cake_rmsnorm_det_train_659bfc53e5372850ae2e"),
        ),
        "bwd_h5120": (
            (0, 256, "cake_rmsnorm_det_train_db01cec1153cd7fb2759"),
            (256, None, "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c"),
        ),
        "bwd_w32_h5120": (
            (0, 256, "cake_rmsnorm_det_train_6042554b427634669739"),
            (256, None, "cake_rmsnorm_det_train_40227b2a2b2122628d58"),
        ),
        "bwd_h6144": (
            (0, 256, "cake_rmsnorm_det_train_b93bf060d4d3defc8150"),
            (256, 4096, "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0"),
            (4096, None, "cake_rmsnorm_det_train_b93bf060d4d3defc8150"),
        ),
        "bwd_w32_h6144": (
            (0, 256, "cake_rmsnorm_det_train_62beb52fa50cc8624c9f"),
            (256, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, None, "cake_rmsnorm_det_train_62beb52fa50cc8624c9f"),
        ),
        "bwd_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_0dcdc563be377247044e"),
        ),
        "bwd_w32_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_56f988dde54db37f09d9"),
        ),
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_det_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_det_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (1024, 2048, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (1024, 2048, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
            (8192, 40960, "cake_rmsnorm_det_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_det_train_1388ab779a35c4e41efb"),
            (114688, 196608, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (196608, 327680, "cake_rmsnorm_det_train_ed9483d8c356affee55d"),
            (327680, None, "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
            (8192, 40960, "cake_rmsnorm_det_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_det_train_b0299f6e0881fb35f27b"),
            (114688, 196608, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (196608, 327680, "cake_rmsnorm_det_train_d0a656f053396a58c49a"),
            (327680, None, "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 6144, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 6144, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (4096, 24576, "cake_rmsnorm_det_train_3d20ab4a432bddf7349a"),
            (24576, None, "cake_rmsnorm_det_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (4096, 24576, "cake_rmsnorm_det_train_7da36ff557bb2bac8d89"),
            (24576, None, "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e"),
            (640, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, 16384, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (16384, None, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 768, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (768, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_det_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 768, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (768, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d"),
            (6144, None, "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
    "sm_103a": {
        "bwd_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (4096, 14928, "cake_rmsnorm_det_train_1df7db17277d363b5168"),
            (14928, 15566, "cake_rmsnorm_det_train_4df08ce51fb01ba1193f"),
            (15566, None, "cake_rmsnorm_det_train_f522f5b8d4343d96ca04"),
        ),
        "bwd_w32_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (4096, 14928, "cake_rmsnorm_det_train_a99e2b380514c4d86d60"),
            (14928, 15566, "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2"),
            (15566, None, "cake_rmsnorm_det_train_109f19bb177c8f20cadc"),
        ),
        "bwd_h2048": (
            (0, 3072, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (3072, 4096, "cake_rmsnorm_det_train_9ac8d21e46a658e7913f"),
            (4096, 16384, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (16384, 32768, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (32768, None, "cake_rmsnorm_det_train_302f64db6abe79e5259e"),
        ),
        "bwd_w32_h2048": (
            (0, 3072, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (3072, 4096, "cake_rmsnorm_det_train_d96a14fe8882bbd9ec2e"),
            (4096, 16384, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (16384, 32768, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (32768, None, "cake_rmsnorm_det_train_e98cb156abfe1666c76f"),
        ),
        "bwd_h256": (
            (0, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 5120, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (12288, 40960, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (40960, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 5120, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (12288, 40960, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (40960, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 256, "cake_rmsnorm_det_train_50044ee3bd1ca54bd821"),
            (256, 1024, "cake_rmsnorm_det_train_ad2bb26ea5916baef963"),
            (1024, 4096, "cake_rmsnorm_det_train_24643edc7e1143da24e0"),
            (4096, None, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
        ),
        "bwd_w32_h4096": (
            (0, 256, "cake_rmsnorm_det_train_3fd77c8ce97e8a523abb"),
            (256, 1024, "cake_rmsnorm_det_train_16545056022785f2a7d8"),
            (1024, 4096, "cake_rmsnorm_det_train_78c26a53e83c9803bcbf"),
            (4096, None, "cake_rmsnorm_det_train_626aed743358c27513f1"),
        ),
        "bwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_8e7f4d811e55a351a22e"),
            (4096, 32768, "cake_rmsnorm_det_train_274b0055aace183f2367"),
            (32768, None, "cake_rmsnorm_det_train_cfaae3cec98f8501832b"),
        ),
        "bwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_dfcbd32cdb8108643fba"),
            (4096, 32768, "cake_rmsnorm_det_train_56ee8e66479625fd078a"),
            (32768, None, "cake_rmsnorm_det_train_a546f410ec8cf03ef20d"),
        ),
        "bwd_h5120": ((0, None, "cake_rmsnorm_det_train_2fc4932fbde0bc848b32"),),
        "bwd_w32_h5120": ((0, None, "cake_rmsnorm_det_train_79811a76205d7d7dd197"),),
        "bwd_h6144": (
            (0, 4096, "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0"),
            (4096, None, "cake_rmsnorm_det_train_322574af704b51aef0bb"),
        ),
        "bwd_w32_h6144": (
            (0, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, None, "cake_rmsnorm_det_train_9a8c0396edd85d4d545e"),
        ),
        "bwd_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_0dcdc563be377247044e"),
        ),
        "bwd_w32_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_56f988dde54db37f09d9"),
        ),
        "fwd_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_det_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1536, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1536, 3584, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
            (3584, 6144, "cake_rmsnorm_det_train_1974c5c86cee63f33370"),
            (6144, None, "cake_rmsnorm_det_train_29d0ae48de6b86aa4a41"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_c5a8099e177e463daab9"),
            (1024, 4096, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_01f0ebf3db9fa2c0884e"),
            (1024, 4096, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
            (8192, 20480, "cake_rmsnorm_det_train_bbb05314930d192dfe7b"),
            (20480, 40960, "cake_rmsnorm_det_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (114688, 196608, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (196608, 327680, "cake_rmsnorm_det_train_ed9483d8c356affee55d"),
            (327680, 368640, "cake_rmsnorm_det_train_37f663183cbff23c2893"),
            (368640, None, "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
            (8192, 20480, "cake_rmsnorm_det_train_a8fffd93682c5e382dc5"),
            (20480, 40960, "cake_rmsnorm_det_train_77456435c9fbd1d18fcf"),
            (40960, 81920, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (81920, 114688, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (114688, 196608, "cake_rmsnorm_det_train_42d82d26e97c36651dae"),
            (196608, 327680, "cake_rmsnorm_det_train_d0a656f053396a58c49a"),
            (327680, 368640, "cake_rmsnorm_det_train_18e3c0a5204006b3a56f"),
            (368640, None, "cake_rmsnorm_det_train_670bfb3eed0ddf6ba461"),
        ),
        "fwd_h4096": (
            (0, 640, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (640, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 3072, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (3072, 6144, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (6144, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 640, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (640, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 3072, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (3072, 6144, "cake_rmsnorm_det_train_bdc524a233d2bc6e9571"),
            (6144, None, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
        ),
        "fwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_8757f6da86abf54363d5"),
            (1024, 4096, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_det_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_det_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_6769e393999bc2ba1926"),
            (1024, 4096, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_det_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 640, "cake_rmsnorm_det_train_1defffdf2712cae36784"),
            (640, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, None, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, None, "cake_rmsnorm_det_train_288d7b1d51c340236cb7"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 6144, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (6144, None, "cake_rmsnorm_det_train_5f89e5fd2f8a6057997e"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_308da18230017dadc02d"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_8e3407e059cfc005f421"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
    "sm_107a": {
        "bwd_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 16384, "cake_rmsnorm_det_train_6a428da2227972ae18ae"),
            (16384, None, "cake_rmsnorm_det_train_56d26996239835327501"),
        ),
        "bwd_w32_h1536": (
            (0, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 16384, "cake_rmsnorm_det_train_e63beae98c236ce26335"),
            (16384, None, "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4"),
        ),
        "bwd_h2048": (
            (0, 256, "cake_rmsnorm_det_train_84efa2af7907877dfb56"),
            (256, 1024, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
            (1024, 16384, "cake_rmsnorm_det_train_81c3d8ab0a482e3af043"),
            (16384, None, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
        ),
        "bwd_w32_h2048": (
            (0, 256, "cake_rmsnorm_det_train_6b13aa49e0c0414d9041"),
            (256, 1024, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
            (1024, 16384, "cake_rmsnorm_det_train_9e5ec75a15ac2bff0e4b"),
            (16384, None, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
        ),
        "bwd_h256": (
            (0, 65536, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (65536, None, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
        ),
        "bwd_w32_h256": (
            (0, 65536, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (65536, None, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
        ),
        "bwd_h4096": (
            (0, 256, "cake_rmsnorm_det_train_0da5dd4b80ad9b5426ce"),
            (256, 1024, "cake_rmsnorm_det_train_a647fccc3d9fa9134733"),
            (1024, 4096, "cake_rmsnorm_det_train_d41a514f5210e9adc39f"),
            (4096, None, "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf"),
        ),
        "bwd_w32_h4096": (
            (0, 256, "cake_rmsnorm_det_train_0c0fa63d0ff2cdc484a4"),
            (256, 1024, "cake_rmsnorm_det_train_dc439f8335327e482025"),
            (1024, 4096, "cake_rmsnorm_det_train_b08633208e60b8ce94fd"),
            (4096, None, "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c"),
        ),
        "bwd_h512": (
            (0, 1024, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (1024, 4096, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (4096, 16384, "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd"),
            (16384, None, "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3"),
        ),
        "bwd_w32_h512": (
            (0, 1024, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (1024, 4096, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (4096, 16384, "cake_rmsnorm_det_train_383f709cd5c4fadd9436"),
            (16384, None, "cake_rmsnorm_det_train_03efa681eb38f59b00c2"),
        ),
        "bwd_h5120": (
            (0, 4096, "cake_rmsnorm_det_train_838c077df3a6d35d55b7"),
            (4096, None, "cake_rmsnorm_det_train_6816fb80e97380f1bcbc"),
        ),
        "bwd_w32_h5120": (
            (0, 4096, "cake_rmsnorm_det_train_cd612de56c0985bb4caa"),
            (4096, None, "cake_rmsnorm_det_train_468cc4d1b8128187b4b3"),
        ),
        "bwd_h6144": ((0, None, "cake_rmsnorm_det_train_5d292172df772adb17ae"),),
        "bwd_w32_h6144": ((0, None, "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb"),),
        "bwd_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_0dcdc563be377247044e"),
        ),
        "bwd_w32_residual_h6144": (
            (0, None, "cake_rmsnorm_det_train_56f988dde54db37f09d9"),
        ),
        "fwd_h1536": (
            (0, 2048, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (2048, 12288, "cake_rmsnorm_det_train_4cdc4c495b3f59ff5069"),
            (12288, None, "cake_rmsnorm_det_train_682675bda578f5b7a312"),
        ),
        "fwd_w32_h1536": (
            (0, 2048, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (2048, 12288, "cake_rmsnorm_det_train_92cf7b87f1ebffb1b12b"),
            (12288, None, "cake_rmsnorm_det_train_a30e5a3b69b7c183992f"),
        ),
        "fwd_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (1024, 8192, "cake_rmsnorm_det_train_a4da85136fba4588239f"),
            (8192, 12288, "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d"),
            (12288, 16384, "cake_rmsnorm_det_train_0ad4d53c4a0df501a99d"),
            (16384, None, "cake_rmsnorm_det_train_bde56bd5a68c493f07a2"),
        ),
        "fwd_w32_h2048": (
            (0, 1024, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (1024, 8192, "cake_rmsnorm_det_train_a72c6ece1900f748eaa2"),
            (8192, 12288, "cake_rmsnorm_det_train_595f506004038a65cc63"),
            (12288, 16384, "cake_rmsnorm_det_train_40210950f2e7ee48bc3e"),
            (16384, None, "cake_rmsnorm_det_train_40210950f2e7ee48bc3e"),
        ),
        "fwd_h256": (
            (0, 6144, "cake_rmsnorm_det_train_c5ca409f44e169148301"),
            (6144, 32768, "cake_rmsnorm_det_train_bbb05314930d192dfe7b"),
            (32768, 98304, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (98304, None, "cake_rmsnorm_det_train_37f663183cbff23c2893"),
        ),
        "fwd_w32_h256": (
            (0, 6144, "cake_rmsnorm_det_train_8f2756937eaac6b02502"),
            (6144, 32768, "cake_rmsnorm_det_train_a8fffd93682c5e382dc5"),
            (32768, 98304, "cake_rmsnorm_det_train_32bebe5a7782311d60a0"),
            (98304, None, "cake_rmsnorm_det_train_18e3c0a5204006b3a56f"),
        ),
        "fwd_h4096": (
            (0, 512, "cake_rmsnorm_det_train_8392d1a008d58a006bba"),
            (512, 1024, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
            (1024, 4096, "cake_rmsnorm_det_train_d447b7f6398b60fe5daa"),
            (4096, 20480, "cake_rmsnorm_det_train_bf933e1660cab8e5787b"),
            (20480, 49152, "cake_rmsnorm_det_train_bf933e1660cab8e5787b"),
            (49152, None, "cake_rmsnorm_det_train_fd913ecd254a6bba4b18"),
        ),
        "fwd_w32_h4096": (
            (0, 512, "cake_rmsnorm_det_train_4371270e4db4b985998f"),
            (512, 1024, "cake_rmsnorm_det_train_f0db3068d0d05442a59d"),
            (1024, 4096, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (4096, 20480, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (20480, 49152, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
            (49152, None, "cake_rmsnorm_det_train_f9f3423be0bd8ff8c794"),
        ),
        "fwd_h512": (
            (0, 256, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (256, 6144, "cake_rmsnorm_det_train_2a42e5c02b240f8709b1"),
            (6144, None, "cake_rmsnorm_det_train_60f47c5d00570ecf7740"),
        ),
        "fwd_w32_h512": (
            (0, 256, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (256, 6144, "cake_rmsnorm_det_train_79fecc7016e44e940733"),
            (6144, None, "cake_rmsnorm_det_train_61accbcc02891615f286"),
        ),
        "fwd_h5120": (
            (0, 512, "cake_rmsnorm_det_train_fd1097c45c1e328871fd"),
            (512, 1152, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (1152, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, 20480, "cake_rmsnorm_det_train_044d4a2aa0553c16db06"),
            (20480, 49152, "cake_rmsnorm_det_train_044d4a2aa0553c16db06"),
            (49152, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 512, "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e"),
            (512, 1152, "cake_rmsnorm_det_train_1defffdf2712cae36784"),
            (1152, 6144, "cake_rmsnorm_det_train_9ecb62c2640d571e50a8"),
            (6144, 20480, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (20480, 49152, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
            (49152, None, "cake_rmsnorm_det_train_b49de9b47372fa4e1051"),
        ),
        "fwd_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_bad4db9153cc903b7ace"),
            (1024, 4096, "cake_rmsnorm_det_train_cf1d05546951bdbd9f5d"),
            (4096, None, "cake_rmsnorm_det_train_e7f4221a7c26b9dfdd23"),
        ),
        "fwd_w32_h6144": (
            (0, 1024, "cake_rmsnorm_det_train_2763cd5ebcc501c9ddb6"),
            (1024, 4096, "cake_rmsnorm_det_train_3d619cd1e3ae638cce3d"),
            (4096, None, "cake_rmsnorm_det_train_fdfa500c83d95df0f5c1"),
        ),
        "fwd_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
}

# Per architecture and route: the upper edge of every window but the last and the measured
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (the
# forward's one-ulp differences), "dx" (the backward kernel switch), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "bwd_h1536": ((1024, "same"), (4096, "same"), (32768, "same")),
        "bwd_w32_h1536": ((1024, "same"), (4096, "same"), (32768, "same")),
        "bwd_h2048": ((3072, "same"), (4096, "ulp"), (16384, "same")),
        "bwd_w32_h2048": ((3072, "same"), (4096, "ulp"), (16384, "same")),
        "bwd_h256": ((2048, "same"), (5120, "same"), (12288, "same"), (40960, "same")),
        "bwd_w32_h256": (
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
        ),
        "bwd_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_w32_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_h512": ((1024, "ulp"), (4096, "same"), (32768, "same")),
        "bwd_w32_h512": ((1024, "ulp"), (4096, "same"), (32768, "same")),
        "bwd_h5120": ((256, "unmeasured"),),
        "bwd_w32_h5120": ((256, "unmeasured"),),
        "bwd_h6144": ((256, "same"), (4096, "same")),
        "bwd_w32_h6144": ((256, "same"), (4096, "same")),
        "bwd_residual_h6144": (),
        "bwd_w32_residual_h6144": (),
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
        "fwd_h5120": ((640, "same"), (6144, "unmeasured")),
        "fwd_w32_h5120": (
            (640, "unmeasured"),
            (6144, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": (
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (6144, "unmeasured"),
        ),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
    },
    "sm_103a": {
        "bwd_h1536": (
            (1024, "same"),
            (4096, "same"),
            (14928, "unmeasured"),
            (15566, "unmeasured"),
        ),
        "bwd_w32_h1536": (
            (1024, "same"),
            (4096, "same"),
            (14928, "unmeasured"),
            (15566, "unmeasured"),
        ),
        "bwd_h2048": ((3072, "same"), (4096, "ulp"), (16384, "same"), (32768, "same")),
        "bwd_w32_h2048": (
            (3072, "same"),
            (4096, "ulp"),
            (16384, "same"),
            (32768, "same"),
        ),
        "bwd_h256": ((2048, "same"), (5120, "same"), (12288, "same"), (40960, "same")),
        "bwd_w32_h256": (
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
        ),
        "bwd_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_w32_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_h512": ((1024, "ulp"), (4096, "same"), (32768, "same")),
        "bwd_w32_h512": ((1024, "ulp"), (4096, "same"), (32768, "same")),
        "bwd_h5120": (),
        "bwd_w32_h5120": (),
        "bwd_h6144": ((4096, "same"),),
        "bwd_w32_h6144": ((4096, "same"),),
        "bwd_residual_h6144": (),
        "bwd_w32_residual_h6144": (),
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
        "fwd_h5120": ((640, "same"), (6144, "unmeasured")),
        "fwd_w32_h5120": ((640, "unmeasured"), (6144, "unmeasured")),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "unmeasured"), (6144, "unmeasured")),
        "fwd_residual_h6144": ((256, "same"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "same"), (1024, "ulp")),
    },
    "sm_107a": {
        "bwd_h1536": ((1024, "same"), (16384, "same")),
        "bwd_w32_h1536": ((1024, "same"), (16384, "same")),
        "bwd_h2048": ((256, "same"), (1024, "ulp"), (16384, "same")),
        "bwd_w32_h2048": ((256, "same"), (1024, "ulp"), (16384, "same")),
        "bwd_h256": ((65536, "unmeasured"),),
        "bwd_w32_h256": ((65536, "unmeasured"),),
        "bwd_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_w32_h4096": ((256, "same"), (1024, "same"), (4096, "same")),
        "bwd_h512": ((1024, "same"), (4096, "same"), (16384, "unmeasured")),
        "bwd_w32_h512": ((1024, "same"), (4096, "same"), (16384, "unmeasured")),
        "bwd_h5120": ((4096, "unmeasured"),),
        "bwd_w32_h5120": ((4096, "unmeasured"),),
        "bwd_h6144": (),
        "bwd_w32_h6144": (),
        "bwd_residual_h6144": (),
        "bwd_w32_residual_h6144": (),
        "fwd_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_w32_h1536": ((2048, "ulp"), (12288, "same")),
        "fwd_h2048": ((1024, "ulp"), (8192, "same"), (12288, "same"), (16384, "same")),
        "fwd_w32_h2048": (
            (1024, "unmeasured"),
            (8192, "unmeasured"),
            (12288, "unmeasured"),
            (16384, "unmeasured"),
        ),
        "fwd_h256": (
            (6144, "unmeasured"),
            (32768, "unmeasured"),
            (98304, "unmeasured"),
        ),
        "fwd_w32_h256": (
            (6144, "unmeasured"),
            (32768, "unmeasured"),
            (98304, "unmeasured"),
        ),
        "fwd_h4096": (
            (512, "ulp"),
            (1024, "same"),
            (4096, "same"),
            (20480, "same"),
            (49152, "same"),
        ),
        "fwd_w32_h4096": (
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_h512": ((256, "same"), (6144, "same")),
        "fwd_w32_h512": ((256, "same"), (6144, "same")),
        "fwd_h5120": (
            (512, "unmeasured"),
            (1152, "unmeasured"),
            (6144, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_w32_h5120": (
            (512, "unmeasured"),
            (1152, "unmeasured"),
            (6144, "unmeasured"),
            (20480, "unmeasured"),
            (49152, "unmeasured"),
        ),
        "fwd_h6144": ((1024, "same"), (4096, "same")),
        "fwd_w32_h6144": ((1024, "same"), (4096, "same")),
        "fwd_residual_h6144": ((256, "ulp"), (1024, "ulp")),
        "fwd_w32_residual_h6144": ((256, "ulp"), (1024, "ulp")),
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

NOTES = """
Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order, so ``rstd`` moves by one FP32 ulp (two
on H = 1536) on about 11-15 % of the rows (25-30 % on the residual route) and ``y`` by
one BF16 ulp on at most 1.6 % of them, and ``dx`` follows on those rows; across the
other forward edges nothing changes.  Across a backward-only edge ``dx`` is bitwise
identical: the H = 6144 route, which switched between the base backward kernel and the
bulk-copy one with ``dx`` differing on about 1-2 % of the rows, now runs the bulk-copy
kernel at every ``T`` on every architecture.  ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` on one architecture it is bitwise reproducible.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).
"""

FORWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order, so ``rstd`` moves by one FP32 ulp (two
on H = 1536) on about 11-15 % of the rows (25-30 % on the residual route) and ``y`` by
one BF16 ulp on at most 1.6 % of them, and ``dx`` follows on those rows; across the
other forward edges nothing changes.  Across a backward-only edge ``dx`` is bitwise
identical: the H = 6144 route, which switched between the base backward kernel and the
bulk-copy one with ``dx`` differing on about 1-2 % of the rows, now runs the bulk-copy
kernel at every ``T`` on every architecture.  ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` on one architecture it is bitwise reproducible.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge, which the package tests hold: ``=`` bitwise-identical outputs on the
common rows across it, ``~`` the forward's one-ulp differences, ``dx`` the backward
kernel switch, ``?`` not yet measured (the tests bound the differences there).

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

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
* ``sm_100a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 ?
  (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_w32_h5120``: (0, 640] ba28d2c0 ? (640, 6144] 9ecb62c2 ?
  (6144, 16384] b49de9b4 ? (16384, inf) 9ecb62c2
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e ? (768, 1024] 2763cd5e ?
  (1024, 6144] 3d619cd1 ? (6144, inf) 5f89e5fd
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
* ``sm_103a`` ``fwd_h5120``: (0, 640] 5eef28cd = (640, 6144] b5e173f5 ?
  (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 640] 1defffdf ? (640, 6144] 9ecb62c2 ?
  (6144, inf) 9ecb62c2
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e ? (1024, 6144] 2763cd5e ?
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 256] 308da182 = (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
* ``sm_107a`` ``fwd_h1536``: (0, 2048] 1ea88523 ~ (2048, 12288] 4cdc4c49 =
  (12288, inf) 682675bd
* ``sm_107a`` ``fwd_h2048``: (0, 1024] b44107e0 ~ (1024, 8192] a4da8513 =
  (8192, 12288] 0ad4d53c = (12288, 16384] 0ad4d53c = (16384, inf) bde56bd5
* ``sm_107a`` ``fwd_w32_h2048``: (0, 1024] 0da8e18d ? (1024, 8192] a72c6ece ?
  (8192, 12288] 595f5060 ? (12288, 16384] 40210950 ? (16384, inf) 40210950
* ``sm_107a`` ``fwd_h256``: (0, 6144] c5ca409f ? (6144, 32768] bbb05314 ?
  (32768, 98304] 466331ff ? (98304, inf) 37f66318
* ``sm_107a`` ``fwd_h4096``: (0, 512] 8392d1a0 ~ (512, 1024] fd913ecd =
  (1024, 4096] d447b7f6 = (4096, 20480] bf933e16 = (20480, 49152] bf933e16 =
  (49152, inf) fd913ecd
* ``sm_107a`` ``fwd_w32_h4096``: (0, 512] 4371270e ? (512, 1024] f0db3068 ?
  (1024, 4096] f9f3423b ? (4096, 20480] f9f3423b ? (20480, 49152] f9f3423b ?
  (49152, inf) f9f3423b
* ``sm_107a`` ``fwd_h512``: (0, 256] 7673978d = (256, 6144] 2a42e5c0 =
  (6144, inf) 60f47c5d
* ``sm_107a`` ``fwd_h5120``: (0, 512] fd1097c4 ? (512, 1152] 5eef28cd ?
  (1152, 6144] b5e173f5 ? (6144, 20480] 044d4a2a ? (20480, 49152] 044d4a2a ?
  (49152, inf) 5eef28cd
* ``sm_107a`` ``fwd_w32_h5120``: (0, 512] ba28d2c0 ? (512, 1152] 1defffdf ?
  (1152, 6144] 9ecb62c2 ? (6144, 20480] b49de9b4 ? (20480, 49152] b49de9b4 ?
  (49152, inf) b49de9b4
* ``sm_107a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, 4096] cf1d0554 =
  (4096, inf) e7f4221a
* ``sm_107a`` ``fwd_residual_h6144``: (0, 256] e646bbdb ~ (256, 1024] 03bdf566 ~
  (1024, inf) e646bbdb
"""

BACKWARD_SECTION = """
Program selection by token count

Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order, so ``rstd`` moves by one FP32 ulp (two
on H = 1536) on about 11-15 % of the rows (25-30 % on the residual route) and ``y`` by
one BF16 ulp on at most 1.6 % of them, and ``dx`` follows on those rows; across the
other forward edges nothing changes.  Across a backward-only edge ``dx`` is bitwise
identical: the H = 6144 route, which switched between the base backward kernel and the
bulk-copy one with ``dx`` differing on about 1-2 % of the rows, now runs the bulk-copy
kernel at every ``T`` on every architecture.  ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` on one architecture it is bitwise reproducible.
Measured on B200 (sm_100a), GB300 (sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge, which the package tests hold: ``=`` bitwise-identical outputs on the
common rows across it, ``~`` the forward's one-ulp differences, ``dx`` the backward
kernel switch, ``?`` not yet measured (the tests bound the differences there).

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

* ``sm_100a`` ``bwd_h1536``: (0, 1024] 7dc29629 = (1024, 4096] c1a9efdf =
  (4096, 32768] 21c8bb55 = (32768, inf) b457c4d1
* ``sm_100a`` ``bwd_h2048``: (0, 3072] caf6978e = (3072, 4096] 9ac8d21e ~
  (4096, 16384] c73074cf = (16384, inf) ceeef043
* ``sm_100a`` ``bwd_h256``: (0, 2048] a385ae16 = (2048, 5120] af005462 =
  (5120, 12288] a385ae16 = (12288, 40960] af005462 = (40960, inf) a385ae16
* ``sm_100a`` ``bwd_h4096``: (0, 256] 3e644ea1 = (256, 1024] 59270fb2 =
  (1024, 4096] 9a830956 = (4096, inf) 35a0b0ca
* ``sm_100a`` ``bwd_h512``: (0, 1024] 09601d42 ~ (1024, 4096] 123db0b2 =
  (4096, 32768] ad1a5a3c = (32768, inf) 8877322a
* ``sm_100a`` ``bwd_h5120``: (0, 256] db01cec1 ? (256, inf) b482c84d
* ``sm_100a`` ``bwd_h6144``: (0, 256] b93bf060 = (256, 4096] 5c804aed =
  (4096, inf) b93bf060
* ``sm_100a`` ``bwd_residual_h6144``: (0, inf) 0dcdc563
* ``sm_103a`` ``bwd_h1536``: (0, 1024] 7dc29629 = (1024, 4096] c1a9efdf =
  (4096, 14928] 1df7db17 ? (14928, 15566] 4df08ce5 ? (15566, inf) f522f5b8
* ``sm_103a`` ``bwd_h2048``: (0, 3072] caf6978e = (3072, 4096] 9ac8d21e ~
  (4096, 16384] caf6978e = (16384, 32768] caf6978e = (32768, inf) 302f64db
* ``sm_103a`` ``bwd_h256``: (0, 2048] a385ae16 = (2048, 5120] af005462 =
  (5120, 12288] a385ae16 = (12288, 40960] af005462 = (40960, inf) a385ae16
* ``sm_103a`` ``bwd_h4096``: (0, 256] 50044ee3 = (256, 1024] ad2bb26e =
  (1024, 4096] 24643edc = (4096, inf) 35a0b0ca
* ``sm_103a`` ``bwd_h512``: (0, 1024] 09601d42 ~ (1024, 4096] 8e7f4d81 =
  (4096, 32768] 274b0055 = (32768, inf) cfaae3ce
* ``sm_103a`` ``bwd_h5120``: (0, inf) 2fc4932f
* ``sm_103a`` ``bwd_h6144``: (0, 4096] 5c804aed = (4096, inf) 322574af
* ``sm_103a`` ``bwd_residual_h6144``: (0, inf) 0dcdc563
* ``sm_107a`` ``bwd_h1536``: (0, 1024] 7dc29629 = (1024, 16384] 6a428da2 =
  (16384, inf) 56d26996
* ``sm_107a`` ``bwd_h2048``: (0, 256] 84efa2af = (256, 1024] 845cdcd8 ~
  (1024, 16384] 81c3d8ab = (16384, inf) 845cdcd8
* ``sm_107a`` ``bwd_h256``: (0, 65536] af005462 ? (65536, inf) a385ae16
* ``sm_107a`` ``bwd_h4096``: (0, 256] 0da5dd4b = (256, 1024] a647fccc =
  (1024, 4096] d41a514f = (4096, inf) ee4a9053
* ``sm_107a`` ``bwd_h512``: (0, 1024] 3feed502 = (1024, 4096] 3feed502 =
  (4096, 16384] dfaf378a ? (16384, inf) eb141cc9
* ``sm_107a`` ``bwd_h5120``: (0, 4096] 838c077d ? (4096, inf) 6816fb80
* ``sm_107a`` ``bwd_h6144``: (0, inf) 5d292172
* ``sm_107a`` ``bwd_residual_h6144``: (0, inf) 0dcdc563
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "BACKWARD_SECTION",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
]
