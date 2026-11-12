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
the host module appends to the forward and backward docstrings;
``ACCURACY_MIN_SMS`` and ``ACCURACY_EDGES`` the backward accuracy guard (see
``NOTES``).  The package tests hold the table to the registry and the edges to
the outputs.
"""

from __future__ import annotations

# Per architecture and route (``cake_jit.route_key``): the windows ``above < T <= at_most``
# (``at_most`` None: unbounded) and the program each launches, from the smallest rows up.
PROGRAM_WINDOWS: dict[str, dict[str, tuple[tuple[int, int | None, str], ...]]] = {
    "sm_100a": {
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_273b578bec562eee3879"),
            (1, 768, "cake_rmsnorm_det_train_8d4ec25545d36490cdeb"),
            (768, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (4096, 32768, "cake_rmsnorm_det_train_21c8bb55c8d672b63952"),
            (32768, 36352, "cake_rmsnorm_det_train_b457c4d134e44476dc6f"),
            (36352, None, "cake_rmsnorm_det_train_b5bdabba0b80fb8afcc5"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_bdb11d00f96428c898c7"),
            (1, 768, "cake_rmsnorm_det_train_4b4079f72c8e93f716d9"),
            (768, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (4096, 32768, "cake_rmsnorm_det_train_4bf450b6af0314ed03be"),
            (32768, 36352, "cake_rmsnorm_det_train_8777b1c8538527c9aaf7"),
            (36352, None, "cake_rmsnorm_det_train_25cae5062df01a2da92f"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_61a2d9fcb0cf1bc9c36c"),
            (1, 768, "cake_rmsnorm_det_train_39d33412ac8ec6b17a35"),
            (768, 1024, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (1024, 2047, "cake_rmsnorm_det_train_c73074cf44f07fcf3bb6"),
            (2047, 22528, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (22528, 58880, "cake_rmsnorm_det_train_ceeef043ac64f37fc529"),
            (58880, None, "cake_rmsnorm_det_train_2fe43a951c2d72c1ffd5"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_5f20b731afe2c0b0619b"),
            (1, 768, "cake_rmsnorm_det_train_12875cc887603ca86fce"),
            (768, 1024, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (1024, 16384, "cake_rmsnorm_det_train_b187e5989302def22b2d"),
            (16384, 58880, "cake_rmsnorm_det_train_7e6a0b631aea3edf51f9"),
            (58880, None, "cake_rmsnorm_det_train_e6347c71a6831e3c615b"),
        ),
        "bwd_h256": (
            (0, 87, "cake_rmsnorm_det_train_c0ca220c25fd472a5128"),
            (87, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 5120, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (12288, 40960, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (40960, 80896, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (80896, 1572864, "cake_rmsnorm_det_train_083ea823c691e0631ad3"),
            (1572864, None, "cake_rmsnorm_det_train_c18297d75c0d424978fd"),
        ),
        "bwd_w32_h256": (
            (0, 87, "cake_rmsnorm_det_train_e5eaf0fd38e702d0c68a"),
            (87, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 5120, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (12288, 40960, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (40960, 80896, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (80896, 1572864, "cake_rmsnorm_det_train_c4c89f0d04b7b3a38d0b"),
            (1572864, None, "cake_rmsnorm_det_train_f788ef57163552825d27"),
        ),
        "bwd_h4096": (
            (0, 203, "cake_rmsnorm_det_train_bbc80180faee6bad6da3"),
            (203, 256, "cake_rmsnorm_det_train_3e644ea19b7d14e61ab9"),
            (256, 1024, "cake_rmsnorm_det_train_59270fb249dcba1ff679"),
            (1024, 22528, "cake_rmsnorm_det_train_5572b2671862fe670ef4"),
            (22528, 32768, "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a"),
            (32768, 44032, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
            (44032, None, "cake_rmsnorm_det_train_7122dbcd697b3fbe1ee0"),
        ),
        "bwd_w32_h4096": (
            (0, 203, "cake_rmsnorm_det_train_0af20ef9ee6868c5569d"),
            (203, 256, "cake_rmsnorm_det_train_20c64c610d22c6db4c05"),
            (256, 1024, "cake_rmsnorm_det_train_c9f55c0ae4b198c859d1"),
            (1024, 22528, "cake_rmsnorm_det_train_47a01388444960da7aae"),
            (22528, 32768, "cake_rmsnorm_det_train_a8eb754739a1589de3ec"),
            (32768, 44032, "cake_rmsnorm_det_train_626aed743358c27513f1"),
            (44032, None, "cake_rmsnorm_det_train_5ef5f1f13ed8f2b994c4"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_43d6b60bffc7acab55f8"),
            (1, 227, "cake_rmsnorm_det_train_5f3e3fccaea795d0a806"),
            (227, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_123db0b2f7fb38cfab03"),
            (4096, 32768, "cake_rmsnorm_det_train_ad1a5a3c5f88e3d10036"),
            (32768, None, "cake_rmsnorm_det_train_8877322a3002e3a4fefd"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_e97e46f27f9139442b55"),
            (1, 227, "cake_rmsnorm_det_train_942503b80a21f22cd392"),
            (227, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_632ca6ff01848a71f9f2"),
            (4096, 32768, "cake_rmsnorm_det_train_1f628cf4fd17b4ea9441"),
            (32768, None, "cake_rmsnorm_det_train_659bfc53e5372850ae2e"),
        ),
        "bwd_h5120": (
            (0, 1, "cake_rmsnorm_det_train_6a1cfc92df862d3000e7"),
            (1, 256, "cake_rmsnorm_det_train_baa071a215cdf8b946bf"),
            (256, 6144, "cake_rmsnorm_det_train_b482c84dbffcd75a4a4c"),
            (6144, 50688, "cake_rmsnorm_det_train_c145ef195d7aeb04f242"),
            (50688, None, "cake_rmsnorm_det_train_c145ef195d7aeb04f242"),
        ),
        "bwd_w32_h5120": (
            (0, 1, "cake_rmsnorm_det_train_9560032be3774471322b"),
            (1, 256, "cake_rmsnorm_det_train_9560032be3774471322b"),
            (256, 6144, "cake_rmsnorm_det_train_40227b2a2b2122628d58"),
            (6144, 50688, "cake_rmsnorm_det_train_9479b1298a1b19d4d996"),
            (50688, None, "cake_rmsnorm_det_train_9479b1298a1b19d4d996"),
        ),
        "bwd_h6144": (
            (0, 384, "cake_rmsnorm_det_train_7100a5f169836fd246e6"),
            (384, 767, "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0"),
            (767, 4096, "cake_rmsnorm_det_train_61421bc72e57d14d8a32"),
            (4096, 42496, "cake_rmsnorm_det_train_b93bf060d4d3defc8150"),
            (42496, None, "cake_rmsnorm_det_train_c440e57b9cd55c3e22e9"),
        ),
        "bwd_w32_h6144": (
            (0, 384, "cake_rmsnorm_det_train_76a09fc04aced9f40868"),
            (384, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, 42496, "cake_rmsnorm_det_train_62beb52fa50cc8624c9f"),
            (42496, None, "cake_rmsnorm_det_train_0ae6eb51bc0b61084cff"),
        ),
        "bwd_residual_h6144": (
            (0, 1152, "cake_rmsnorm_det_train_d2749326f8c9b856bb01"),
            (1152, 42496, "cake_rmsnorm_det_train_c1a3be946e94e94233af"),
            (42496, None, "cake_rmsnorm_det_train_84385468c8e8ef19f3dc"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 1152, "cake_rmsnorm_det_train_c8bc10e76c624f5f4534"),
            (1152, 42496, "cake_rmsnorm_det_train_ab0e20feefa296ffff3c"),
            (42496, None, "cake_rmsnorm_det_train_486424b132fa3256e8ef"),
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
            (0, 1, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (1, 1024, "cake_rmsnorm_det_train_a5f323db12acfb2cd565"),
            (1024, 2048, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
            (2048, 4096, "cake_rmsnorm_det_train_b44107e05796c2922187"),
            (4096, None, "cake_rmsnorm_det_train_780b9833fd0a236ef0e4"),
        ),
        "fwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (1, 1024, "cake_rmsnorm_det_train_27fd540856828056457a"),
            (1024, 2048, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
            (2048, 4096, "cake_rmsnorm_det_train_0da8e18d4af432c6927a"),
            (4096, None, "cake_rmsnorm_det_train_a4d3f82b3e776f4fc49b"),
        ),
        "fwd_h256": (
            (0, 1024, "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8"),
            (1024, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
            (8192, 40960, "cake_rmsnorm_det_train_1b284baec9663d6de1b9"),
            (40960, 81920, "cake_rmsnorm_det_train_86d51eff64fc9fe57cf0"),
            (81920, 114688, "cake_rmsnorm_det_train_1388ab779a35c4e41efb"),
            (114688, 196608, "cake_rmsnorm_det_train_466331ff3cd2d6c62738"),
            (196608, 327680, "cake_rmsnorm_det_train_ed9483d8c356affee55d"),
            (327680, None, "cake_rmsnorm_det_train_b8f32dd4e0a8ca9ab2a5"),
        ),
        "fwd_w32_h256": (
            (0, 1024, "cake_rmsnorm_det_train_f8087de00e0409186d93"),
            (1024, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
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
            (0, 64, "cake_rmsnorm_det_train_6e12e07353014329ebad"),
            (64, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_c1819da87ab45ebb8d56"),
            (64, 640, "cake_rmsnorm_det_train_ba28d2c0c2c81a2c714e"),
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
            (0, 256, "cake_rmsnorm_det_train_8b4730338cac98ec272d"),
            (256, 1024, "cake_rmsnorm_det_train_03bdf5667404141e1eaf"),
            (1024, 4096, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
            (4096, None, "cake_rmsnorm_det_train_41805c5801f6956b4ce4"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 256, "cake_rmsnorm_det_train_3e55e44ce21eff7d8931"),
            (256, 1024, "cake_rmsnorm_det_train_e5a3aa2548fc41ee0485"),
            (1024, 4096, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
            (4096, None, "cake_rmsnorm_det_train_4be7a14e78ff59fa96e2"),
        ),
    },
    "sm_103a": {
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_0eaab6475ecda03ee034"),
            (1, 31, "cake_rmsnorm_det_train_ad547a3cd25e6cd640a3"),
            (31, 214, "cake_rmsnorm_det_train_53285b9598170177d3e0"),
            (214, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 4096, "cake_rmsnorm_det_train_c1a9efdf06c3087b4efe"),
            (4096, 14928, "cake_rmsnorm_det_train_1df7db17277d363b5168"),
            (14928, 15566, "cake_rmsnorm_det_train_4df08ce51fb01ba1193f"),
            (15566, 40448, "cake_rmsnorm_det_train_f522f5b8d4343d96ca04"),
            (40448, None, "cake_rmsnorm_det_train_cbeae168d70edd196417"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_5f6f9f3c32410cc1575a"),
            (1, 31, "cake_rmsnorm_det_train_775d9130f3ce2d46f118"),
            (31, 214, "cake_rmsnorm_det_train_5c5be9a6aeac20fd2776"),
            (214, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 4096, "cake_rmsnorm_det_train_17d9cdfd157ba4b86c2b"),
            (4096, 14928, "cake_rmsnorm_det_train_a99e2b380514c4d86d60"),
            (14928, 15566, "cake_rmsnorm_det_train_7a72e97bd6cb4dc0efc2"),
            (15566, 40448, "cake_rmsnorm_det_train_109f19bb177c8f20cadc"),
            (40448, None, "cake_rmsnorm_det_train_ca6211f3ea2ce8c86a05"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_56635d2cf6f4e9a9f44a"),
            (1, 2, "cake_rmsnorm_det_train_ce65889518ffd6340c59"),
            (2, 171, "cake_rmsnorm_det_train_2b4c0f5b0e9c312fa43e"),
            (171, 1024, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (1024, 16384, "cake_rmsnorm_det_train_cc26e316ae40d683a118"),
            (16384, 32768, "cake_rmsnorm_det_train_caf6978ed0a17fbf65b9"),
            (32768, 55296, "cake_rmsnorm_det_train_302f64db6abe79e5259e"),
            (55296, None, "cake_rmsnorm_det_train_018f0cd723106f709f11"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_02f98bf9651f7200d064"),
            (1, 2, "cake_rmsnorm_det_train_28c46db39116f7f718a0"),
            (2, 171, "cake_rmsnorm_det_train_2e9171b7753650cda515"),
            (171, 1024, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (1024, 16384, "cake_rmsnorm_det_train_c83c314c6dcf28ad7369"),
            (16384, 32768, "cake_rmsnorm_det_train_07f9f2fbea7163639c27"),
            (32768, 55296, "cake_rmsnorm_det_train_e98cb156abfe1666c76f"),
            (55296, None, "cake_rmsnorm_det_train_ca8ca4674a3dbd89ad16"),
        ),
        "bwd_h256": (
            (0, 1680, "cake_rmsnorm_det_train_083ea823c691e0631ad3"),
            (1680, 2048, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (2048, 5120, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (5120, 12288, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (12288, 40960, "cake_rmsnorm_det_train_af005462cd6a301f31c4"),
            (40960, 99840, "cake_rmsnorm_det_train_a385ae1681a2c078344e"),
            (99840, 720896, "cake_rmsnorm_det_train_083ea823c691e0631ad3"),
            (720896, 1572864, "cake_rmsnorm_det_train_0291949ba535c646c3a8"),
            (1572864, None, "cake_rmsnorm_det_train_c18297d75c0d424978fd"),
        ),
        "bwd_w32_h256": (
            (0, 1680, "cake_rmsnorm_det_train_c4c89f0d04b7b3a38d0b"),
            (1680, 2048, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (2048, 5120, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (5120, 12288, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (12288, 40960, "cake_rmsnorm_det_train_b7d9577ff884de98ea06"),
            (40960, 99840, "cake_rmsnorm_det_train_420788e663a2c3a810f6"),
            (99840, 720896, "cake_rmsnorm_det_train_c4c89f0d04b7b3a38d0b"),
            (720896, 1572864, "cake_rmsnorm_det_train_89505762fa1cbf33cb8f"),
            (1572864, None, "cake_rmsnorm_det_train_f788ef57163552825d27"),
        ),
        "bwd_h4096": (
            (0, 512, "cake_rmsnorm_det_train_10f7bdf21790269a977c"),
            (512, 1024, "cake_rmsnorm_det_train_ad2bb26ea5916baef963"),
            (1024, 24576, "cake_rmsnorm_det_train_5572b2671862fe670ef4"),
            (24576, 32768, "cake_rmsnorm_det_train_5abcdc29d54ac214bd4a"),
            (32768, 48640, "cake_rmsnorm_det_train_35a0b0ca6b2060313255"),
            (48640, 51200, "cake_rmsnorm_det_train_850eec3aff01f019a392"),
            (51200, None, "cake_rmsnorm_det_train_e813cc307923546f0064"),
        ),
        "bwd_w32_h4096": (
            (0, 1, "cake_rmsnorm_det_train_11820fe48711775d5685"),
            (1, 512, "cake_rmsnorm_det_train_f03fb1c6ef9e54f54240"),
            (512, 1024, "cake_rmsnorm_det_train_16545056022785f2a7d8"),
            (1024, 24576, "cake_rmsnorm_det_train_47a01388444960da7aae"),
            (24576, 32768, "cake_rmsnorm_det_train_a8eb754739a1589de3ec"),
            (32768, 48640, "cake_rmsnorm_det_train_626aed743358c27513f1"),
            (48640, 51200, "cake_rmsnorm_det_train_cb164a27c0157ed0712d"),
            (51200, None, "cake_rmsnorm_det_train_cc019a164f0e91c4b583"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_9dc76b2260f10acc2926"),
            (1, 4, "cake_rmsnorm_det_train_117a66e423695c124202"),
            (4, 227, "cake_rmsnorm_det_train_877751ba915bbc261064"),
            (227, 1024, "cake_rmsnorm_det_train_09601d42628de24f6f87"),
            (1024, 4096, "cake_rmsnorm_det_train_8e7f4d811e55a351a22e"),
            (4096, 32768, "cake_rmsnorm_det_train_274b0055aace183f2367"),
            (32768, None, "cake_rmsnorm_det_train_cfaae3cec98f8501832b"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_2633b05c48af196c8bbe"),
            (1, 4, "cake_rmsnorm_det_train_df3b2da5a4185e19ebbf"),
            (4, 227, "cake_rmsnorm_det_train_81d9adbf31a26a4ebc02"),
            (227, 1024, "cake_rmsnorm_det_train_9f07cea1d13b923ed91e"),
            (1024, 4096, "cake_rmsnorm_det_train_dfcbd32cdb8108643fba"),
            (4096, 32768, "cake_rmsnorm_det_train_56ee8e66479625fd078a"),
            (32768, None, "cake_rmsnorm_det_train_a546f410ec8cf03ef20d"),
        ),
        "bwd_h5120": (
            (0, 64, "cake_rmsnorm_det_train_e73ed9335334417fb063"),
            (64, 5120, "cake_rmsnorm_det_train_2fc4932fbde0bc848b32"),
            (5120, None, "cake_rmsnorm_det_train_7ffb443a63a92ab6b0e0"),
        ),
        "bwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_33b1b76e986f135bb2f5"),
            (64, 5120, "cake_rmsnorm_det_train_79811a76205d7d7dd197"),
            (5120, None, "cake_rmsnorm_det_train_f9f7d4cd75ef91cc8374"),
        ),
        "bwd_h6144": (
            (0, 256, "cake_rmsnorm_det_train_5d6088e6e1bf16d6c29c"),
            (256, 767, "cake_rmsnorm_det_train_5c804aed7ad14b99e1d0"),
            (767, 4096, "cake_rmsnorm_det_train_61421bc72e57d14d8a32"),
            (4096, 48640, "cake_rmsnorm_det_train_322574af704b51aef0bb"),
            (48640, None, "cake_rmsnorm_det_train_289dc9ab6377fbf87332"),
        ),
        "bwd_w32_h6144": (
            (0, 1, "cake_rmsnorm_det_train_017091c7827bdc8c95d5"),
            (1, 256, "cake_rmsnorm_det_train_f9c641ac48eaa4a50908"),
            (256, 4096, "cake_rmsnorm_det_train_41b80b6af97974feb181"),
            (4096, 48640, "cake_rmsnorm_det_train_9a8c0396edd85d4d545e"),
            (48640, None, "cake_rmsnorm_det_train_aa46aa07d86900ad832f"),
        ),
        "bwd_residual_h6144": (
            (0, 1216, "cake_rmsnorm_det_train_07135550fae8b07bb82c"),
            (1216, 48640, "cake_rmsnorm_det_train_04ebec87cbee3f0ce093"),
            (48640, None, "cake_rmsnorm_det_train_21b5e791889c85c956e3"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 1216, "cake_rmsnorm_det_train_570eec506ed94f36f062"),
            (1216, 48640, "cake_rmsnorm_det_train_47890c9ae28e3e4b6759"),
            (48640, None, "cake_rmsnorm_det_train_83043d3ae46d9d20c42c"),
        ),
        "fwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1, 256, "cake_rmsnorm_det_train_d52385477d517c951e87"),
            (256, 1536, "cake_rmsnorm_det_train_1ea88523077fc340e55f"),
            (1536, 3584, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
            (3584, 6144, "cake_rmsnorm_det_train_7de1e51089d432dbc26b"),
            (6144, None, "cake_rmsnorm_det_train_2fdbab2af48db42cd2a9"),
        ),
        "fwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_21d3416baa877808371e"),
            (1, 256, "cake_rmsnorm_det_train_8e49f86a65049578dee1"),
            (256, 1536, "cake_rmsnorm_det_train_21d3416baa877808371e"),
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
            (0, 24, "cake_rmsnorm_det_train_76b9c06cc9ec083c24b8"),
            (24, 8192, "cake_rmsnorm_det_train_0489c784c13bb25499f6"),
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
            (0, 24, "cake_rmsnorm_det_train_f8087de00e0409186d93"),
            (24, 8192, "cake_rmsnorm_det_train_f2e721db4c30900f12fe"),
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
            (0, 4, "cake_rmsnorm_det_train_8757f6da86abf54363d5"),
            (4, 1025, "cake_rmsnorm_det_train_def532441f08b90a9903"),
            (1025, 4096, "cake_rmsnorm_det_train_7673978dc698790054d2"),
            (4096, 16384, "cake_rmsnorm_det_train_3d20ab4a432bddf7349a"),
            (16384, None, "cake_rmsnorm_det_train_3430e5d9c053309b0323"),
        ),
        "fwd_w32_h512": (
            (0, 4, "cake_rmsnorm_det_train_6769e393999bc2ba1926"),
            (4, 1025, "cake_rmsnorm_det_train_20ebc686cc4099b9d026"),
            (1025, 4096, "cake_rmsnorm_det_train_e2eabcb3b0915d916a13"),
            (4096, 16384, "cake_rmsnorm_det_train_7da36ff557bb2bac8d89"),
            (16384, None, "cake_rmsnorm_det_train_3c1e1f571f15d42e8d57"),
        ),
        "fwd_h5120": (
            (0, 64, "cake_rmsnorm_det_train_6e12e07353014329ebad"),
            (64, 640, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
            (640, 6144, "cake_rmsnorm_det_train_b5e173f5ecc736d91109"),
            (6144, None, "cake_rmsnorm_det_train_5eef28cd35498e9abfa5"),
        ),
        "fwd_w32_h5120": (
            (0, 64, "cake_rmsnorm_det_train_c1819da87ab45ebb8d56"),
            (64, 640, "cake_rmsnorm_det_train_1defffdf2712cae36784"),
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
            (0, 4096, "cake_rmsnorm_det_train_8b4730338cac98ec272d"),
            (4096, None, "cake_rmsnorm_det_train_e646bbdbd852884953da"),
        ),
        "fwd_w32_residual_h6144": (
            (0, 4096, "cake_rmsnorm_det_train_3e55e44ce21eff7d8931"),
            (4096, None, "cake_rmsnorm_det_train_6325c474ad9955a319e9"),
        ),
    },
    "sm_107a": {
        "bwd_h1536": (
            (0, 1, "cake_rmsnorm_det_train_33551729d38fdc94cde7"),
            (1, 576, "cake_rmsnorm_det_train_9e71ed0d9ad7a7739002"),
            (576, 1024, "cake_rmsnorm_det_train_7dc29629ff21625607f8"),
            (1024, 16384, "cake_rmsnorm_det_train_6a428da2227972ae18ae"),
            (16384, 45056, "cake_rmsnorm_det_train_56d26996239835327501"),
            (45056, None, "cake_rmsnorm_det_train_e694b0a25ea01d1f6cdc"),
        ),
        "bwd_w32_h1536": (
            (0, 1, "cake_rmsnorm_det_train_d259a22ec493a07e5e02"),
            (1, 576, "cake_rmsnorm_det_train_db8d986d719b909621d0"),
            (576, 1024, "cake_rmsnorm_det_train_baff9ecbafe7de1e6030"),
            (1024, 16384, "cake_rmsnorm_det_train_e63beae98c236ce26335"),
            (16384, 45056, "cake_rmsnorm_det_train_de8e07d4cb312b06dbc4"),
            (45056, None, "cake_rmsnorm_det_train_0f0b782f1e538caad3ce"),
        ),
        "bwd_h2048": (
            (0, 1, "cake_rmsnorm_det_train_bc9d86b3b63d513a1aac"),
            (1, 256, "cake_rmsnorm_det_train_32eb12e27be1360b890d"),
            (256, 1024, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
            (1024, 16384, "cake_rmsnorm_det_train_81c3d8ab0a482e3af043"),
            (16384, 50688, "cake_rmsnorm_det_train_845cdcd8145ada631461"),
            (50688, None, "cake_rmsnorm_det_train_694decfb79ba96468f62"),
        ),
        "bwd_w32_h2048": (
            (0, 1, "cake_rmsnorm_det_train_0ff8a6d52d561fb5a952"),
            (1, 256, "cake_rmsnorm_det_train_22944704b77a80a04834"),
            (256, 1024, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
            (1024, 16384, "cake_rmsnorm_det_train_9e5ec75a15ac2bff0e4b"),
            (16384, 50688, "cake_rmsnorm_det_train_838a9a4054731f7942af"),
            (50688, None, "cake_rmsnorm_det_train_723710a92b1ff8356f66"),
        ),
        "bwd_h256": (
            (0, 1572864, "cake_rmsnorm_det_train_083ea823c691e0631ad3"),
            (1572864, None, "cake_rmsnorm_det_train_c18297d75c0d424978fd"),
        ),
        "bwd_w32_h256": (
            (0, 1572864, "cake_rmsnorm_det_train_c4c89f0d04b7b3a38d0b"),
            (1572864, None, "cake_rmsnorm_det_train_f788ef57163552825d27"),
        ),
        "bwd_h4096": (
            (0, 256, "cake_rmsnorm_det_train_44511fa02465a6d7283d"),
            (256, 1024, "cake_rmsnorm_det_train_a647fccc3d9fa9134733"),
            (1024, 4096, "cake_rmsnorm_det_train_d41a514f5210e9adc39f"),
            (4096, 24064, "cake_rmsnorm_det_train_ee4a9053bb25b8d49fcf"),
            (24064, 50176, "cake_rmsnorm_det_train_99a01f2a850874718c3f"),
            (50176, None, "cake_rmsnorm_det_train_9daa9bc7c5a75816e3b6"),
        ),
        "bwd_w32_h4096": (
            (0, 256, "cake_rmsnorm_det_train_440bd6f2a114414ea9db"),
            (256, 1024, "cake_rmsnorm_det_train_dc439f8335327e482025"),
            (1024, 4096, "cake_rmsnorm_det_train_b08633208e60b8ce94fd"),
            (4096, 24064, "cake_rmsnorm_det_train_0f1859b9fe9e47668a1c"),
            (24064, 50176, "cake_rmsnorm_det_train_fa3d4fc5dbd76348b31a"),
            (50176, None, "cake_rmsnorm_det_train_42aa82e30162448ef08c"),
        ),
        "bwd_h512": (
            (0, 1, "cake_rmsnorm_det_train_9dc76b2260f10acc2926"),
            (1, 768, "cake_rmsnorm_det_train_3c06f3e2f267255a1989"),
            (768, 1024, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (1024, 4096, "cake_rmsnorm_det_train_3feed5026e1156d2fba2"),
            (4096, 16384, "cake_rmsnorm_det_train_dfaf378aa8ad93e68bbd"),
            (16384, None, "cake_rmsnorm_det_train_eb141cc9e0a93c6f0ce3"),
        ),
        "bwd_w32_h512": (
            (0, 1, "cake_rmsnorm_det_train_2633b05c48af196c8bbe"),
            (1, 768, "cake_rmsnorm_det_train_f8786d8e382af9b8d02e"),
            (768, 1024, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (1024, 4096, "cake_rmsnorm_det_train_b841d64bbcaefff5ce7b"),
            (4096, 16384, "cake_rmsnorm_det_train_383f709cd5c4fadd9436"),
            (16384, None, "cake_rmsnorm_det_train_03efa681eb38f59b00c2"),
        ),
        "bwd_h5120": (
            (0, 1, "cake_rmsnorm_det_train_505283a5225ac45bc095"),
            (1, 256, "cake_rmsnorm_det_train_e73ed9335334417fb063"),
            (256, 1280, "cake_rmsnorm_det_train_505283a5225ac45bc095"),
            (1280, 4096, "cake_rmsnorm_det_train_838c077df3a6d35d55b7"),
            (4096, 19968, "cake_rmsnorm_det_train_6816fb80e97380f1bcbc"),
            (19968, None, "cake_rmsnorm_det_train_47da668e7e96652f840b"),
        ),
        "bwd_w32_h5120": (
            (0, 1, "cake_rmsnorm_det_train_3a7bd162c370762719b3"),
            (1, 256, "cake_rmsnorm_det_train_33b1b76e986f135bb2f5"),
            (256, 1280, "cake_rmsnorm_det_train_3a7bd162c370762719b3"),
            (1280, 4096, "cake_rmsnorm_det_train_cd612de56c0985bb4caa"),
            (4096, 19968, "cake_rmsnorm_det_train_468cc4d1b8128187b4b3"),
            (19968, None, "cake_rmsnorm_det_train_de0cf457cd240f9c6422"),
        ),
        "bwd_h6144": (
            (0, 1, "cake_rmsnorm_det_train_57d45549f8ed34e34e65"),
            (1, 256, "cake_rmsnorm_det_train_dab76189611ec3bb733b"),
            (256, 1664, "cake_rmsnorm_det_train_57d45549f8ed34e34e65"),
            (1664, 19456, "cake_rmsnorm_det_train_5d292172df772adb17ae"),
            (19456, None, "cake_rmsnorm_det_train_19e18371f9243bd1a91a"),
        ),
        "bwd_w32_h6144": (
            (0, 1, "cake_rmsnorm_det_train_a501b31feb94fa326722"),
            (1, 256, "cake_rmsnorm_det_train_6faede3656da9fc0409b"),
            (256, 1664, "cake_rmsnorm_det_train_a501b31feb94fa326722"),
            (1664, 19456, "cake_rmsnorm_det_train_7d2f6707bfa3d2cd60bb"),
            (19456, None, "cake_rmsnorm_det_train_b83f8b3d546630a60675"),
        ),
        "bwd_residual_h6144": (
            (0, 368, "cake_rmsnorm_det_train_c15868b87bfb2c21645e"),
            (368, 19456, "cake_rmsnorm_det_train_36cad444bc992ccc732d"),
            (19456, None, "cake_rmsnorm_det_train_b73a6086490dd9e1855c"),
        ),
        "bwd_w32_residual_h6144": (
            (0, 368, "cake_rmsnorm_det_train_a1c12b0ee6454391a37f"),
            (368, 19456, "cake_rmsnorm_det_train_b9d848cdbcfaeea0325c"),
            (19456, None, "cake_rmsnorm_det_train_f5703907295a466b7cf7"),
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
# class of the outputs across it: "same" (bitwise identical on the common rows), "ulp" (y
# moves by at most one BF16 ulp, rstd by at most two FP32 ulps, three on R200 (216 SMs) at
# the residual route's edge 1024), "dx" (the backward kernel switch), "unmeasured".
EDGES: dict[str, dict[str, tuple[tuple[int, str], ...]]] = {
    "sm_100a": {
        "bwd_h1536": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (32768, "same"),
            (36352, "same"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (32768, "same"),
            (36352, "same"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (2047, "unmeasured"),
            (22528, "unmeasured"),
            (58880, "unmeasured"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "unmeasured"),
            (16384, "same"),
            (58880, "unmeasured"),
        ),
        "bwd_h256": (
            (87, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (80896, "same"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (87, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (80896, "same"),
            (1572864, "same"),
        ),
        "bwd_h4096": (
            (203, "unmeasured"),
            (256, "same"),
            (1024, "unmeasured"),
            (22528, "unmeasured"),
            (32768, "unmeasured"),
            (44032, "same"),
        ),
        "bwd_w32_h4096": (
            (203, "unmeasured"),
            (256, "same"),
            (1024, "unmeasured"),
            (22528, "unmeasured"),
            (32768, "unmeasured"),
            (44032, "same"),
        ),
        "bwd_h512": (
            (1, "unmeasured"),
            (227, "unmeasured"),
            (1024, "ulp"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (227, "unmeasured"),
            (1024, "ulp"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (6144, "same"),
            (50688, "same"),
        ),
        "bwd_w32_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (6144, "unmeasured"),
            (50688, "unmeasured"),
        ),
        "bwd_h6144": (
            (384, "unmeasured"),
            (767, "unmeasured"),
            (4096, "unmeasured"),
            (42496, "same"),
        ),
        "bwd_w32_h6144": ((384, "unmeasured"), (4096, "same"), (42496, "same")),
        "bwd_residual_h6144": ((1152, "unmeasured"), (42496, "unmeasured")),
        "bwd_w32_residual_h6144": ((1152, "unmeasured"), (42496, "unmeasured")),
        "fwd_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_w32_h1536": ((1536, "ulp"), (3584, "same"), (6144, "same")),
        "fwd_h2048": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "ulp"),
            (4096, "ulp"),
        ),
        "fwd_w32_h2048": (
            (1, "unmeasured"),
            (1024, "unmeasured"),
            (2048, "ulp"),
            (4096, "ulp"),
        ),
        "fwd_h256": (
            (1024, "unmeasured"),
            (8192, "same"),
            (40960, "same"),
            (81920, "same"),
            (114688, "same"),
            (196608, "same"),
            (327680, "same"),
        ),
        "fwd_w32_h256": (
            (1024, "unmeasured"),
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
        "fwd_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_w32_h5120": (
            (64, "unmeasured"),
            (640, "same"),
            (6144, "same"),
            (16384, "same"),
        ),
        "fwd_h6144": ((768, "same"), (1024, "same")),
        "fwd_w32_h6144": ((768, "same"), (1024, "same"), (6144, "same")),
        "fwd_residual_h6144": (
            (256, "unmeasured"),
            (1024, "ulp"),
            (4096, "unmeasured"),
        ),
        "fwd_w32_residual_h6144": (
            (256, "unmeasured"),
            (1024, "ulp"),
            (4096, "unmeasured"),
        ),
    },
    "sm_103a": {
        "bwd_h1536": (
            (1, "unmeasured"),
            (31, "unmeasured"),
            (214, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (14928, "same"),
            (15566, "same"),
            (40448, "same"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (31, "unmeasured"),
            (214, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (14928, "same"),
            (15566, "same"),
            (40448, "same"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (2, "unmeasured"),
            (171, "unmeasured"),
            (1024, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (55296, "same"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (2, "unmeasured"),
            (171, "unmeasured"),
            (1024, "unmeasured"),
            (16384, "unmeasured"),
            (32768, "same"),
            (55296, "same"),
        ),
        "bwd_h256": (
            (1680, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (99840, "same"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_w32_h256": (
            (1680, "unmeasured"),
            (2048, "same"),
            (5120, "same"),
            (12288, "same"),
            (40960, "same"),
            (99840, "same"),
            (720896, "same"),
            (1572864, "same"),
        ),
        "bwd_h4096": (
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
            (48640, "same"),
            (51200, "same"),
        ),
        "bwd_w32_h4096": (
            (1, "unmeasured"),
            (512, "unmeasured"),
            (1024, "unmeasured"),
            (24576, "unmeasured"),
            (32768, "unmeasured"),
            (48640, "unmeasured"),
            (51200, "unmeasured"),
        ),
        "bwd_h512": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (227, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (4, "unmeasured"),
            (227, "unmeasured"),
            (1024, "unmeasured"),
            (4096, "same"),
            (32768, "same"),
        ),
        "bwd_h5120": ((64, "unmeasured"), (5120, "same")),
        "bwd_w32_h5120": ((64, "unmeasured"), (5120, "same")),
        "bwd_h6144": (
            (256, "unmeasured"),
            (767, "unmeasured"),
            (4096, "unmeasured"),
            (48640, "same"),
        ),
        "bwd_w32_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (4096, "same"),
            (48640, "same"),
        ),
        "bwd_residual_h6144": ((1216, "unmeasured"), (48640, "unmeasured")),
        "bwd_w32_residual_h6144": ((1216, "unmeasured"), (48640, "unmeasured")),
        "fwd_h1536": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1536, "ulp"),
            (3584, "same"),
            (6144, "same"),
        ),
        "fwd_w32_h1536": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1536, "ulp"),
            (3584, "same"),
            (6144, "same"),
        ),
        "fwd_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_w32_h2048": ((1024, "same"), (4096, "ulp")),
        "fwd_h256": (
            (24, "unmeasured"),
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
            (24, "unmeasured"),
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
        "fwd_h512": (
            (4, "unmeasured"),
            (1025, "unmeasured"),
            (4096, "same"),
            (16384, "same"),
        ),
        "fwd_w32_h512": (
            (4, "unmeasured"),
            (1025, "unmeasured"),
            (4096, "same"),
            (16384, "same"),
        ),
        "fwd_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_w32_h5120": ((64, "unmeasured"), (640, "same"), (6144, "same")),
        "fwd_h6144": ((1024, "same"),),
        "fwd_w32_h6144": ((1024, "same"), (6144, "same")),
        "fwd_residual_h6144": ((4096, "unmeasured"),),
        "fwd_w32_residual_h6144": ((4096, "unmeasured"),),
    },
    "sm_107a": {
        "bwd_h1536": (
            (1, "unmeasured"),
            (576, "unmeasured"),
            (1024, "same"),
            (16384, "same"),
            (45056, "same"),
        ),
        "bwd_w32_h1536": (
            (1, "unmeasured"),
            (576, "unmeasured"),
            (1024, "same"),
            (16384, "same"),
            (45056, "same"),
        ),
        "bwd_h2048": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1024, "ulp"),
            (16384, "same"),
            (50688, "same"),
        ),
        "bwd_w32_h2048": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1024, "ulp"),
            (16384, "same"),
            (50688, "same"),
        ),
        "bwd_h256": ((1572864, "same"),),
        "bwd_w32_h256": ((1572864, "same"),),
        "bwd_h4096": (
            (256, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (24064, "same"),
            (50176, "same"),
        ),
        "bwd_w32_h4096": (
            (256, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (24064, "same"),
            (50176, "same"),
        ),
        "bwd_h512": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (16384, "unmeasured"),
        ),
        "bwd_w32_h512": (
            (1, "unmeasured"),
            (768, "unmeasured"),
            (1024, "same"),
            (4096, "same"),
            (16384, "unmeasured"),
        ),
        "bwd_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1280, "unmeasured"),
            (4096, "unmeasured"),
            (19968, "unmeasured"),
        ),
        "bwd_w32_h5120": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1280, "unmeasured"),
            (4096, "unmeasured"),
            (19968, "unmeasured"),
        ),
        "bwd_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1664, "unmeasured"),
            (19456, "same"),
        ),
        "bwd_w32_h6144": (
            (1, "unmeasured"),
            (256, "unmeasured"),
            (1664, "unmeasured"),
            (19456, "same"),
        ),
        "bwd_residual_h6144": ((368, "unmeasured"), (19456, "unmeasured")),
        "bwd_w32_residual_h6144": ((368, "unmeasured"), (19456, "unmeasured")),
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
    },
}

EDGE_CLASSES = ["same", "ulp", "dx", "unmeasured"]

# The backward accuracy guard (see NOTES): per architecture the smallest SM count the accuracy
# edges were measured on, and per route the edge above which it runs compensated programs.  On
# a device with fewer SMs a backward call of such a route selects its programs as if ``T`` were
# the edge plus one (its launch keeps ``T``).
ACCURACY_MIN_SMS: dict[str, int] = {"sm_100a": 148, "sm_103a": 152, "sm_107a": 212}
ACCURACY_EDGES: dict[str, dict[str, int]] = {
    "sm_100a": {
        "bwd_h1536": 36352,
        "bwd_w32_h1536": 36352,
        "bwd_h2048": 58880,
        "bwd_w32_h2048": 58880,
        "bwd_h256": 80896,
        "bwd_w32_h256": 80896,
        "bwd_h4096": 44032,
        "bwd_w32_h4096": 44032,
        "bwd_h5120": 6144,
        "bwd_w32_h5120": 6144,
        "bwd_h6144": 42496,
        "bwd_w32_h6144": 42496,
        "bwd_residual_h6144": 42496,
        "bwd_w32_residual_h6144": 42496,
    },
    "sm_103a": {
        "bwd_h1536": 40448,
        "bwd_w32_h1536": 40448,
        "bwd_h2048": 55296,
        "bwd_w32_h2048": 55296,
        "bwd_h256": 99840,
        "bwd_w32_h256": 99840,
        "bwd_h4096": 48640,
        "bwd_w32_h4096": 48640,
        "bwd_h5120": 5120,
        "bwd_w32_h5120": 5120,
        "bwd_h6144": 48640,
        "bwd_w32_h6144": 48640,
        "bwd_residual_h6144": 48640,
        "bwd_w32_residual_h6144": 48640,
    },
    "sm_107a": {
        "bwd_h1536": 45056,
        "bwd_w32_h1536": 45056,
        "bwd_h2048": 50688,
        "bwd_w32_h2048": 50688,
        "bwd_h256": 0,
        "bwd_w32_h256": 0,
        "bwd_h4096": 24064,
        "bwd_w32_h4096": 24064,
        "bwd_h5120": 19968,
        "bwd_w32_h5120": 19968,
        "bwd_h6144": 19456,
        "bwd_w32_h6144": 19456,
        "bwd_residual_h6144": 19456,
        "bwd_w32_residual_h6144": 19456,
    },
}

NOTES = """
Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical: the H = 6144 route,
which switched between the base backward kernel and the bulk-copy one with ``dx``
differing on about 1-2 % of the rows, now runs the bulk-copy kernel at every ``T`` on
every architecture.  The one-row programs at ``T`` = 1 (H = 512, 1536 and 2048) fold the
row's dot product in their own fixed order, so ``dx`` may change bits at their edge;
``EDGES`` marks the edges not yet measured. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows) runs column-task programs instead: one CTA sums
each column of ``dw`` over every row in a fixed order (exact products and TwoSum, one
FP32 rounding), so there ``dw`` does not depend on the SM count.  Every backward route
but H = 512 and the sm_107a residual H = 6144 switches from plain to compensated ``dw``
sums at an accuracy edge (``ACCURACY_EDGES``) measured on B200 (148 SMs), GB300
(152 SMs) and R200 (212 SMs; the H = 5120 and H = 256 edges also on 216 SMs).  A device
with fewer SMs than ``ACCURACY_MIN_SMS`` gives each CTA more rows to sum at the same
``T``; there these routes select their programs as if ``T`` were the edge plus one, so
their compensated programs run at every ``T``. Measured on B200 (sm_100a), GB300
(sm_103a) and R200 (sm_107a).
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
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical: the H = 6144 route,
which switched between the base backward kernel and the bulk-copy one with ``dx``
differing on about 1-2 % of the rows, now runs the bulk-copy kernel at every ``T`` on
every architecture.  The one-row programs at ``T`` = 1 (H = 512, 1536 and 2048) fold the
row's dot product in their own fixed order, so ``dx`` may change bits at their edge;
``EDGES`` marks the edges not yet measured. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows) runs column-task programs instead: one CTA sums
each column of ``dw`` over every row in a fixed order (exact products and TwoSum, one
FP32 rounding), so there ``dw`` does not depend on the SM count.  Every backward route
but H = 512 and the sm_107a residual H = 6144 switches from plain to compensated ``dw``
sums at an accuracy edge (``ACCURACY_EDGES``) measured on B200 (148 SMs), GB300
(152 SMs) and R200 (212 SMs; the H = 5120 and H = 256 edges also on 216 SMs).  A device
with fewer SMs than ``ACCURACY_MIN_SMS`` gives each CTA more rows to sum at the same
``T``; there these routes select their programs as if ``T`` were the edge plus one, so
their compensated programs run at every ``T``. Measured on B200 (sm_100a), GB300
(sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge, which the package tests hold: ``=`` bitwise-identical outputs on the
common rows across it, ``~`` the forward's measured differences (``y`` by at most one
BF16 ulp, ``rstd`` by at most two FP32 ulps, three on R200 (216 SMs) at the residual
route's edge 1024), ``dx`` the backward kernel switch, ``?`` not yet measured (the tests
bound the differences there).

The FP32-weight routes (``fwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

* ``sm_100a`` ``fwd_h1536``: (0, 1536] 1ea88523 ~ (1536, 3584] 2fdbab2a =
  (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_100a`` ``fwd_h2048``: (0, 1] b44107e0 ? (1, 1024] a5f323db ?
  (1024, 2048] 780b9833 ~ (2048, 4096] b44107e0 ~ (4096, inf) 780b9833
* ``sm_100a`` ``fwd_h256``: (0, 1024] 76b9c06c ? (1024, 8192] 0489c784 =
  (8192, 40960] 1b284bae = (40960, 81920] 86d51eff = (81920, 114688] 1388ab77 =
  (114688, 196608] 466331ff = (196608, 327680] ed9483d8 = (327680, inf) b8f32dd4
* ``sm_100a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_100a`` ``fwd_h512``: (0, 1024] 8757f6da ~ (1024, 4096] 7673978d =
  (4096, 24576] 3d20ab4a = (24576, inf) 3430e5d9
* ``sm_100a`` ``fwd_h5120``: (0, 64] 6e12e073 ? (64, 640] 5eef28cd =
  (640, 6144] b5e173f5 = (6144, inf) 5eef28cd
* ``sm_100a`` ``fwd_w32_h5120``: (0, 64] c1819da8 ? (64, 640] ba28d2c0 =
  (640, 6144] 9ecb62c2 = (6144, 16384] b49de9b4 = (16384, inf) 9ecb62c2
* ``sm_100a`` ``fwd_h6144``: (0, 768] bad4db91 = (768, 1024] bad4db91 =
  (1024, inf) 288d7b1d
* ``sm_100a`` ``fwd_w32_h6144``: (0, 768] 2763cd5e = (768, 1024] 2763cd5e =
  (1024, 6144] 3d619cd1 = (6144, inf) 5f89e5fd
* ``sm_100a`` ``fwd_residual_h6144``: (0, 256] 8b473033 ? (256, 1024] 03bdf566 ~
  (1024, 4096] e646bbdb ? (4096, inf) 41805c58
* ``sm_103a`` ``fwd_h1536``: (0, 1] 1ea88523 ? (1, 256] d5238547 ? (256, 1536] 1ea88523
  ~ (1536, 3584] 2fdbab2a = (3584, 6144] 7de1e510 = (6144, inf) 2fdbab2a
* ``sm_103a`` ``fwd_h2048``: (0, 1024] c5a8099e = (1024, 4096] b44107e0 ~
  (4096, inf) 780b9833
* ``sm_103a`` ``fwd_h256``: (0, 24] 76b9c06c ? (24, 8192] 0489c784 =
  (8192, 20480] bbb05314 = (20480, 40960] 1b284bae = (40960, 81920] 86d51eff =
  (81920, 114688] 466331ff = (114688, 196608] 86d51eff = (196608, 327680] ed9483d8 =
  (327680, 368640] 37f66318 = (368640, inf) b8f32dd4
* ``sm_103a`` ``fwd_h4096``: (0, 640] 8392d1a0 ~ (640, 1024] fd913ecd =
  (1024, 3072] d447b7f6 = (3072, 6144] d447b7f6 = (6144, inf) fd913ecd
* ``sm_103a`` ``fwd_h512``: (0, 4] 8757f6da ? (4, 1025] def53244 ? (1025, 4096] 7673978d
  = (4096, 16384] 3d20ab4a = (16384, inf) 3430e5d9
* ``sm_103a`` ``fwd_h5120``: (0, 64] 6e12e073 ? (64, 640] 5eef28cd =
  (640, 6144] b5e173f5 = (6144, inf) 5eef28cd
* ``sm_103a`` ``fwd_w32_h5120``: (0, 64] c1819da8 ? (64, 640] 1defffdf =
  (640, 6144] 9ecb62c2 = (6144, inf) 9ecb62c2
* ``sm_103a`` ``fwd_h6144``: (0, 1024] bad4db91 = (1024, inf) 288d7b1d
* ``sm_103a`` ``fwd_w32_h6144``: (0, 1024] 2763cd5e = (1024, 6144] 2763cd5e =
  (6144, inf) 5f89e5fd
* ``sm_103a`` ``fwd_residual_h6144``: (0, 4096] 8b473033 ? (4096, inf) e646bbdb
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

Each route -- one architecture, hidden size and direction -- ships several programs of
one kernel and selects one per call from the token count ``T``: the program whose window
``above < T <= at_most`` admits it (``cake_jit.program_windows`` returns the windows;
the table below lists them).  Within one window a row's ``y``, ``rstd`` and ``dx`` do
not depend on ``T``: two calls whose token counts fall in the same forward window (and,
for the backward, the same backward window) return bitwise-identical results on their
common rows.  Across a forward edge between programs of different per-row thread groups
the sum of squares is reduced in another order.  On the rows both programs compute,
``rstd`` then moves by at most two FP32 ulps (three on SM107 at the residual route's
edge 1024) on 10.8-18.2 % of the rows (24.0-31.6 % on the residual route); ``y`` moves
by one BF16 ulp, only on rows whose ``rstd`` moved, on at most 2.34 % of the rows
(4.30 % on the residual route); ``dx`` follows on those rows and the residual output
``h_new`` does not change.  Across every other measured forward edge, every measured
edge of H = 256 and H = 5120 among them, the outputs are bitwise identical.  The
programs of the smallest token counts that form ``rstd`` with the approximate reciprocal
square root (``rsqrt.approx``) give other ``rstd`` and ``y`` bits than the programs
beside them, so the outputs can change bits at their edges; ``EDGES`` marks those edges
as not yet measured.  Measured over three random inputs per edge, with BF16 and FP32
weights: on B200 (sm_100a) at most two ulps on up to 18.0 % of the rows (27.3 % on the
residual route), ``y`` on up to 2.03 % (2.83 %); on GB300 (sm_103a) at most two ulps on
up to 17.3 % (27.8 %), ``y`` on up to 2.03 % (2.34 %); on R200 (sm_107a, 212- and 216-SM
boards) at most three ulps on up to 18.2 % (31.6 %), ``y`` on up to 2.34 % (4.30 %).
Across a measured backward-only edge ``dx`` is bitwise identical: the H = 6144 route,
which switched between the base backward kernel and the bulk-copy one with ``dx``
differing on about 1-2 % of the rows, now runs the bulk-copy kernel at every ``T`` on
every architecture.  The one-row programs at ``T`` = 1 (H = 512, 1536 and 2048) fold the
row's dot product in their own fixed order, so ``dx`` may change bits at their edge;
``EDGES`` marks the edges not yet measured. ``dw`` sums every row and depends on ``T``
by definition; for a fixed ``T`` it is bitwise reproducible on devices with the same SM
count.  The backward splits the rows over ``ctas_per_sm`` x SM count CTAs, so devices of
one architecture with different SM counts (B200 with 148 and GB200 with 152 SMs, R200
boards with 212 and 216 SMs) cut different chunk tables and can give different ``dw``
bits.  In the windows of its smallest token counts a backward route other than H = 256
(on sm_100a H = 256 too, to 87 rows) runs column-task programs instead: one CTA sums
each column of ``dw`` over every row in a fixed order (exact products and TwoSum, one
FP32 rounding), so there ``dw`` does not depend on the SM count.  Every backward route
but H = 512 and the sm_107a residual H = 6144 switches from plain to compensated ``dw``
sums at an accuracy edge (``ACCURACY_EDGES``) measured on B200 (148 SMs), GB300
(152 SMs) and R200 (212 SMs; the H = 5120 and H = 256 edges also on 216 SMs).  A device
with fewer SMs than ``ACCURACY_MIN_SMS`` gives each CTA more rows to sum at the same
``T``; there these routes select their programs as if ``T`` were the edge plus one, so
their compensated programs run at every ``T``. Measured on B200 (sm_100a), GB300
(sm_103a) and R200 (sm_107a).

The windows per architecture and route, ``(above, at_most]`` and the program (the first
8 hex digits of its name).  The mark after a window names the measured outcome
at its upper edge, which the package tests hold: ``=`` bitwise-identical outputs on the
common rows across it, ``~`` the forward's measured differences (``y`` by at most one
BF16 ulp, ``rstd`` by at most two FP32 ulps, three on R200 (216 SMs) at the residual
route's edge 1024), ``dx`` the backward kernel switch, ``?`` not yet measured (the tests
bound the differences there).

The FP32-weight routes (``bwd_w32_...``) select among the FP32-weight twins of the
same programs over the same windows, with the same outcome at every edge, and are not
listed; a route re-cut for the FP32 weight (another twin in a window) is listed with
its own windows.

* ``sm_100a`` ``bwd_h1536``: (0, 1] 273b578b ? (1, 768] 8d4ec255 ? (768, 1024] 7dc29629
  = (1024, 4096] c1a9efdf = (4096, 32768] 21c8bb55 = (32768, 36352] b457c4d1 =
  (36352, inf) b5bdabba
* ``sm_100a`` ``bwd_h2048``: (0, 1] 61a2d9fc ? (1, 768] 39d33412 ? (768, 1024] cc26e316
  ? (1024, 2047] c73074cf ? (2047, 22528] cc26e316 ? (22528, 58880] ceeef043 ?
  (58880, inf) 2fe43a95
* ``sm_100a`` ``bwd_w32_h2048``: (0, 1] 5f20b731 ? (1, 768] 12875cc8 ?
  (768, 1024] 07f9f2fb ? (1024, 16384] b187e598 = (16384, 58880] 7e6a0b63 ?
  (58880, inf) e6347c71
* ``sm_100a`` ``bwd_h256``: (0, 87] c0ca220c ? (87, 2048] a385ae16 =
  (2048, 5120] af005462 = (5120, 12288] a385ae16 = (12288, 40960] af005462 =
  (40960, 80896] a385ae16 = (80896, 1572864] 083ea823 = (1572864, inf) c18297d7
* ``sm_100a`` ``bwd_h4096``: (0, 203] bbc80180 ? (203, 256] 3e644ea1 =
  (256, 1024] 59270fb2 ? (1024, 22528] 5572b267 ? (22528, 32768] 5abcdc29 ?
  (32768, 44032] 35a0b0ca = (44032, inf) 7122dbcd
* ``sm_100a`` ``bwd_h512``: (0, 1] 43d6b60b ? (1, 227] 5f3e3fcc ? (227, 1024] 09601d42 ~
  (1024, 4096] 123db0b2 = (4096, 32768] ad1a5a3c = (32768, inf) 8877322a
* ``sm_100a`` ``bwd_h5120``: (0, 1] 6a1cfc92 ? (1, 256] baa071a2 ? (256, 6144] b482c84d
  = (6144, 50688] c145ef19 = (50688, inf) c145ef19
* ``sm_100a`` ``bwd_w32_h5120``: (0, 1] 9560032b ? (1, 256] 9560032b ?
  (256, 6144] 40227b2a ? (6144, 50688] 9479b129 ? (50688, inf) 9479b129
* ``sm_100a`` ``bwd_h6144``: (0, 384] 7100a5f1 ? (384, 767] 5c804aed ?
  (767, 4096] 61421bc7 ? (4096, 42496] b93bf060 = (42496, inf) c440e57b
* ``sm_100a`` ``bwd_w32_h6144``: (0, 384] 76a09fc0 ? (384, 4096] 41b80b6a =
  (4096, 42496] 62beb52f = (42496, inf) 0ae6eb51
* ``sm_100a`` ``bwd_residual_h6144``: (0, 1152] d2749326 ? (1152, 42496] c1a3be94 ?
  (42496, inf) 84385468
* ``sm_103a`` ``bwd_h1536``: (0, 1] 0eaab647 ? (1, 31] ad547a3c ? (31, 214] 53285b95 ?
  (214, 1024] 7dc29629 = (1024, 4096] c1a9efdf = (4096, 14928] 1df7db17 =
  (14928, 15566] 4df08ce5 = (15566, 40448] f522f5b8 = (40448, inf) cbeae168
* ``sm_103a`` ``bwd_w32_h1536``: (0, 1] 5f6f9f3c ? (1, 31] 775d9130 ? (31, 214] 5c5be9a6
  ? (214, 1024] baff9ecb = (1024, 4096] 17d9cdfd = (4096, 14928] a99e2b38 =
  (14928, 15566] 7a72e97b = (15566, 40448] 109f19bb = (40448, inf) ca6211f3
* ``sm_103a`` ``bwd_h2048``: (0, 1] 56635d2c ? (1, 2] ce658895 ? (2, 171] 2b4c0f5b ?
  (171, 1024] caf6978e ? (1024, 16384] cc26e316 ? (16384, 32768] caf6978e =
  (32768, 55296] 302f64db = (55296, inf) 018f0cd7
* ``sm_103a`` ``bwd_h256``: (0, 1680] 083ea823 ? (1680, 2048] a385ae16 =
  (2048, 5120] af005462 = (5120, 12288] a385ae16 = (12288, 40960] af005462 =
  (40960, 99840] a385ae16 = (99840, 720896] 083ea823 = (720896, 1572864] 0291949b =
  (1572864, inf) c18297d7
* ``sm_103a`` ``bwd_h4096``: (0, 512] 10f7bdf2 ? (512, 1024] ad2bb26e ?
  (1024, 24576] 5572b267 ? (24576, 32768] 5abcdc29 ? (32768, 48640] 35a0b0ca =
  (48640, 51200] 850eec3a = (51200, inf) e813cc30
* ``sm_103a`` ``bwd_w32_h4096``: (0, 1] 11820fe4 ? (1, 512] f03fb1c6 ?
  (512, 1024] 16545056 ? (1024, 24576] 47a01388 ? (24576, 32768] a8eb7547 ?
  (32768, 48640] 626aed74 ? (48640, 51200] cb164a27 ? (51200, inf) cc019a16
* ``sm_103a`` ``bwd_h512``: (0, 1] 9dc76b22 ? (1, 4] 117a66e4 ? (4, 227] 877751ba ?
  (227, 1024] 09601d42 ? (1024, 4096] 8e7f4d81 = (4096, 32768] 274b0055 =
  (32768, inf) cfaae3ce
* ``sm_103a`` ``bwd_h5120``: (0, 64] e73ed933 ? (64, 5120] 2fc4932f =
  (5120, inf) 7ffb443a
* ``sm_103a`` ``bwd_h6144``: (0, 256] 5d6088e6 ? (256, 767] 5c804aed ?
  (767, 4096] 61421bc7 ? (4096, 48640] 322574af = (48640, inf) 289dc9ab
* ``sm_103a`` ``bwd_w32_h6144``: (0, 1] 017091c7 ? (1, 256] f9c641ac ?
  (256, 4096] 41b80b6a = (4096, 48640] 9a8c0396 = (48640, inf) aa46aa07
* ``sm_103a`` ``bwd_residual_h6144``: (0, 1216] 07135550 ? (1216, 48640] 04ebec87 ?
  (48640, inf) 21b5e791
* ``sm_107a`` ``bwd_h1536``: (0, 1] 33551729 ? (1, 576] 9e71ed0d ? (576, 1024] 7dc29629
  = (1024, 16384] 6a428da2 = (16384, 45056] 56d26996 = (45056, inf) e694b0a2
* ``sm_107a`` ``bwd_h2048``: (0, 1] bc9d86b3 ? (1, 256] 32eb12e2 ? (256, 1024] 845cdcd8
  ~ (1024, 16384] 81c3d8ab = (16384, 50688] 845cdcd8 = (50688, inf) 694decfb
* ``sm_107a`` ``bwd_h256``: (0, 1572864] 083ea823 = (1572864, inf) c18297d7
* ``sm_107a`` ``bwd_h4096``: (0, 256] 44511fa0 ? (256, 1024] a647fccc =
  (1024, 4096] d41a514f = (4096, 24064] ee4a9053 = (24064, 50176] 99a01f2a =
  (50176, inf) 9daa9bc7
* ``sm_107a`` ``bwd_h512``: (0, 1] 9dc76b22 ? (1, 768] 3c06f3e2 ? (768, 1024] 3feed502 =
  (1024, 4096] 3feed502 = (4096, 16384] dfaf378a ? (16384, inf) eb141cc9
* ``sm_107a`` ``bwd_h5120``: (0, 1] 505283a5 ? (1, 256] e73ed933 ? (256, 1280] 505283a5
  ? (1280, 4096] 838c077d ? (4096, 19968] 6816fb80 ? (19968, inf) 47da668e
* ``sm_107a`` ``bwd_h6144``: (0, 1] 57d45549 ? (1, 256] dab76189 ? (256, 1664] 57d45549
  ? (1664, 19456] 5d292172 = (19456, inf) 19e18371
* ``sm_107a`` ``bwd_residual_h6144``: (0, 368] c15868b8 ? (368, 19456] 36cad444 ?
  (19456, inf) b73a6086
"""

SECTIONS: dict[str, str] = {"fwd": FORWARD_SECTION, "bwd": BACKWARD_SECTION}

__all__ = [
    "ACCURACY_EDGES",
    "ACCURACY_MIN_SMS",
    "BACKWARD_SECTION",
    "EDGES",
    "EDGE_CLASSES",
    "FORWARD_SECTION",
    "NOTES",
    "PROGRAM_WINDOWS",
    "SECTIONS",
]
