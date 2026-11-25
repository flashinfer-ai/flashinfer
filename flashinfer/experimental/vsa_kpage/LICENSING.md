# Licensing

cudnn-frontend is distributed primarily under the **Apache License 2.0**
(see [LICENSE.txt](LICENSE.txt)). Some files retain **MIT** or **BSD-3-Clause**
terms for pre-existing code. Every source file carries an SPDX
`SPDX-License-Identifier:` expression declaring the applicable licenses.

To find the license of any file, read its SPDX tag, e.g.:

```
SPDX-License-Identifier: Apache-2.0
SPDX-License-Identifier: MIT
SPDX-License-Identifier: Apache-2.0 AND MIT
SPDX-License-Identifier: Apache-2.0 AND BSD-3-Clause
```

Third-party attributions are in [THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt).

## Apache-2.0 modifications to existing licensed code

The OSS warning cleanup licenses its NVIDIA modifications under Apache-2.0,
including changes to files with retained MIT or BSD-3-Clause code. Their headers
identify the scope of the modifications and use an SPDX `AND` expression to
preserve the pre-existing terms. This is not an alternative-license choice:
both sets of applicable terms must be respected. See the
[SPDX license-expression specification](https://spdx.github.io/spdx-spec/v2.3/SPDX-license-expressions/#d43-conjunctive-and-operator).

Original copyright notices, provenance comments, and embedded license texts
remain intact. MIT terms are reproduced in [LICENSE-MIT.txt](LICENSE-MIT.txt),
and upstream license texts and attributions are reproduced in
[THIRD_PARTY_LICENSES.txt](THIRD_PARTY_LICENSES.txt). Where an existing file
carries both an MIT declaration and a BSD notice, its expression retains both.
The adapted CUTLASS static scheduler in `python/cudnn/_cutlass_helpers/` likewise
retains BSD-3-Clause for the upstream code and applies Apache-2.0 to local changes.

An Apache-2.0 modification does not assert that upstream authors relicensed
their contributions. Files whose complete NVIDIA ownership is established can
use Apache-2.0 alone; `_fp8_quant.py` in `block_sparse_attention/` is one such file.

## Why some files remain under MIT

The repository's original license was MIT. During the MIT → Apache 2.0
relicensing ([PR #408](https://github.com/NVIDIA/cudnn-frontend/pull/408),
sign-off [issue #431](https://github.com/NVIDIA/cudnn-frontend/issues/431)),
files were converted to Apache 2.0 **only** when all of their surviving code is
owned by NVIDIA. Files are kept under MIT in two cases:

1. **Pending external-contributor consent** — the file contains code
   contributed by a non-NVIDIA contributor whose consent to relicense has not
   (yet) been obtained. Determined by `git blame` on `develop`: a file stays
   MIT if any external contributor's lines survive in it.
2. **Third-party-derived code** — the file is derived from external open
   source (FlashAttention, QuACK, or CUTLASS) and carries the original author's
   copyright.

If/when a listed external contributor grants consent, the files attributed to
them below can be moved to Apache 2.0 by flipping their SPDX tag.

**Consent received so far** (see [issue #431](https://github.com/NVIDIA/cudnn-frontend/issues/431)):
@take-cheeze, @fallintoplace, @zianglih, @JackRao123, @zkyue, @Hyaloid, @haowen-han, @junaire, @szluyu99, @dimitar-asenov — their files have already been moved to Apache-2.0.
A file still appears below if *another* contributor who has not yet consented
also has surviving lines in it.

**Cleared by NVIDIA employment** (not by issue #431 consent): @HollowMan6, @hxbai — commits under a personal
email address but is an NVIDIA employee, so those contributions are covered by
employment and their files are Apache-2.0.

The **Introducing commit(s)** column links the exact commit that introduced the
surviving external line(s) in each file (blame on `origin/develop`).

## Category 1 — MIT pending external-contributor consent (51 files)

| File | External contributor(s) | Introducing commit(s) |
|------|-------------------------|-----------------------|
| `CMakeLists.txt` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `cudnn_frontend-config.cmake.in` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `dlpack_version.txt` | Emilien Macchi (@EmilienM) | [`1669048`](https://github.com/NVIDIA/cudnn-frontend/commit/1669048643384e2dab17aa69db196c09d67880e3) (#165) |
| `include/cudnn_frontend/graph_properties.h` | Subhobrata Dey (@sbcd90) | [`6943af9`](https://github.com/NVIDIA/cudnn-frontend/commit/6943af90f98c6f8a726283b61240eacfca7fcea1) (#214) |
| `include/cudnn_frontend/node/conv_dgrad.h` | DrDirk (@DrDirk) | [`5f680bc`](https://github.com/NVIDIA/cudnn-frontend/commit/5f680bc0c6271c527d6ecf815ffb544b8c89a45d) (#423) |
| `include/cudnn_frontend/node/conv_fprop.h` | DrDirk (@DrDirk) | [`5f680bc`](https://github.com/NVIDIA/cudnn-frontend/commit/5f680bc0c6271c527d6ecf815ffb544b8c89a45d) (#423) |
| `include/cudnn_frontend/node/conv_wgrad.h` | DrDirk (@DrDirk) | [`5f680bc`](https://github.com/NVIDIA/cudnn-frontend/commit/5f680bc0c6271c527d6ecf815ffb544b8c89a45d) (#423) |
| `include/cudnn_frontend/node/matmul.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node/pointwise.h` | DrDirk (@DrDirk)<br>James Y Knight (@jyknight) | [`5f680bc`](https://github.com/NVIDIA/cudnn-frontend/commit/5f680bc0c6271c527d6ecf815ffb544b8c89a45d) (#423)<br>[`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node/reduction.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node/reshape.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node/rng.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node/softmax.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend/node_interface.h` | James Y Knight (@jyknight) | [`af9bc9e`](https://github.com/NVIDIA/cudnn-frontend/commit/af9bc9e88be17693d5841876e8f2f69a279c7ff7) (#56) |
| `include/cudnn_frontend_utils.h` | Martin Valgur (@valgur) | [`31b2c5d`](https://github.com/NVIDIA/cudnn-frontend/commit/31b2c5dfcd1a4d50340819e7dfc2c8671b8ad0c2) (#154) |
| `python/CMakeLists.txt` | Connor Baker (@ConnorBaker)<br>Emilien Macchi (@EmilienM) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125)<br>[`1669048`](https://github.com/NVIDIA/cudnn-frontend/commit/1669048643384e2dab17aa69db196c09d67880e3) (#165) |
| `samples/cpp/CMakeLists.txt` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/convolution/dgrads.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/convolution/fp8_fprop.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/convolution/fprop.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/convolution/int8_fprop.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/convolution/wgrads.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/matmul/fp8_matmul.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/matmul/int8_matmul.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/matmul/matmuls.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/matmul/mixed_matmul.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/misc/pointwise.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/misc/resample.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/misc/serialization.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/misc/slice.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/misc/sm_carveout.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/norm/batchnorm.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/norm/layernorm.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/norm/rmsnorm.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_bwd.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_bwd_with_cudagraphs.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_bwd_with_flexible_graphs.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_cached.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd_paged_decode_and_prefill.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd_with_cudagraphs.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd_with_custom_dropout.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd_with_flexible_graphs.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp16_fwd_with_paged_caches.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp8_bwd.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp8_bwd_bottom_right_causal_mask.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp8_fwd.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/cpp/sdpa/fp8_fwd_bottom_right_causal_mask.cpp` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `samples/legacy_samples/CMakeLists.txt` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |
| `setup.py` | Emilien Macchi (@EmilienM) | [`e0449f6`](https://github.com/NVIDIA/cudnn-frontend/commit/e0449f6e7de80a520dd1aeabe1772a1318179963) (#163) |
| `test/cpp/CMakeLists.txt` | Connor Baker (@ConnorBaker) | [`0f828cf`](https://github.com/NVIDIA/cudnn-frontend/commit/0f828cf169b39c5d3afc9bb2aeb042f275e69320) (#125) |

> Note: `dlpack_version.txt` is a plain version-string file that cannot carry
> a header comment; it is listed here and governed by MIT via this manifest.

## Category 2 — Third-party-derived files retaining MIT terms (38 files)

Derived from FlashAttention or CUTLASS (both BSD-3-Clause) and/or QuACK
(Apache-2.0); they retain their original authors' copyright notices. See
THIRD_PARTY_LICENSES.txt. Files with Apache-2.0 modifications retain MIT in their
compound SPDX expression. The commit link(s) are the NVIDIA import commits that
introduced the surviving derived lines. This historical list does not enumerate
every later import; the source headers and third-party notices also cover those.

| File | Import commit(s) |
|------|------------------|
| `python/cudnn/block_sparse_attention/_interface.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/bwd/bsa_bwd_postprocess.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333)<br>[`5b9c94a`](https://github.com/NVIDIA/cudnn-frontend/commit/5b9c94a67763d4d6ebf4158b49edca4aad2d81e7) (#349) |
| `python/cudnn/block_sparse_attention/csrc/bwd/bsa_bwd_preprocess.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk128/bsa_bwd_sm100.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333)<br>[`0041a2b`](https://github.com/NVIDIA/cudnn-frontend/commit/0041a2bb917f886a46388b0b59781ee185ed8f11) (#382)<br>[`d380fab`](https://github.com/NVIDIA/cudnn-frontend/commit/d380fabf3ae9b7002f0511143d1e27a29341bb2f) (#350) |
| `python/cudnn/block_sparse_attention/csrc/bwd/sm100_blk128/bsa_bwd_sm100_kpage.py` | derived from `bsa_bwd_sm100.py` above (KPage contribution) |
| `python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_combine.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/fwd/sm100_blk64/bsa_fwd_helpers.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/block_info.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/block_sparse_tile_scheduler.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/kpage_tile_scheduler.py` | derived from `block_sparse_tile_scheduler.py` above (KPage contribution) |
| `python/cudnn/block_sparse_attention/csrc/utils/copy_utils.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333)<br>[`0041a2b`](https://github.com/NVIDIA/cudnn-frontend/commit/0041a2bb917f886a46388b0b59781ee185ed8f11) (#382) |
| `python/cudnn/block_sparse_attention/csrc/utils/cute_dsl_utils.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/kernel_utils.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/layout_utils.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/mma_sm100_desc.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/named_barrier.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/pack_gqa.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/pipeline.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/sm90_utils.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/softmax.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333)<br>[`52119ee`](https://github.com/NVIDIA/cudnn-frontend/commit/52119ee60e4b82e104a7a04ab36022cee2608195) (#341) |
| `python/cudnn/block_sparse_attention/csrc/utils/tcgen05_mma_helpers.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/block_sparse_attention/csrc/utils/tile_scheduler.py` | [`afd575d`](https://github.com/NVIDIA/cudnn-frontend/commit/afd575d6ecf9cb86bc35325b9a961547f0d4d272) (#333) |
| `python/cudnn/deepseek_sparse_attention/score_recompute/dense_score_recompute_sm90.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241)<br>[`3e2b8ec`](https://github.com/NVIDIA/cudnn-frontend/commit/3e2b8ecddec1a9d03d02e7ed1f38dd62358177d7) (#263)<br>[`7016b04`](https://github.com/NVIDIA/cudnn-frontend/commit/7016b04077c4c53ee00ba39a86ab1067a7542e8c) (#316)<br>[`28462c3`](https://github.com/NVIDIA/cudnn-frontend/commit/28462c3f684b8f72785a3de84811180481a7fc8d) (#273) |
| `python/cudnn/deepseek_sparse_attention/score_recompute/sparse_score_recompute_sm90.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241) |
| `python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm100.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241)<br>[`7016b04`](https://github.com/NVIDIA/cudnn-frontend/commit/7016b04077c4c53ee00ba39a86ab1067a7542e8c) (#316)<br>[`3e2b8ec`](https://github.com/NVIDIA/cudnn-frontend/commit/3e2b8ecddec1a9d03d02e7ed1f38dd62358177d7) (#263)<br>[`cfee724`](https://github.com/NVIDIA/cudnn-frontend/commit/cfee7248b6f32a9aaf2421649a71668d68d80bfe) (#318) |
| `python/cudnn/deepseek_sparse_attention/sparse_attention_backward/_interface_sm90.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241)<br>[`3e2b8ec`](https://github.com/NVIDIA/cudnn-frontend/commit/3e2b8ecddec1a9d03d02e7ed1f38dd62358177d7) (#263)<br>[`f3ee97b`](https://github.com/NVIDIA/cudnn-frontend/commit/f3ee97b58ea208110f161807111f7755fcd2ec3e) (#388) |
| `python/cudnn/deepseek_sparse_attention/utils/copy.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241) |
| `python/cudnn/deepseek_sparse_attention/utils/sm90/bwd_barriers.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241) |
| `python/cudnn/deepseek_sparse_attention/utils/sm90/bwd_tile_scheduler.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241) |
| `python/cudnn/deepseek_sparse_attention/utils/sm90/mma.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241)<br>[`74efc0d`](https://github.com/NVIDIA/cudnn-frontend/commit/74efc0d44a11005326e797201b58adb1399b4733) (#321) |
| `python/cudnn/deepseek_sparse_attention/utils/sm90/primitives.py` | [`c4a9762`](https://github.com/NVIDIA/cudnn-frontend/commit/c4a97621eca52fa0c3a1862a411a16be580b25c6) (#241)<br>[`7016b04`](https://github.com/NVIDIA/cudnn-frontend/commit/7016b04077c4c53ee00ba39a86ab1067a7542e8c) (#316) |
| `python/cudnn/hstu/hstu_attention/_kernels/blackwell_helpers.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/block_info.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/fast_math.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/mask.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/mma_sm100_desc.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/tile_scheduler.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
| `python/cudnn/hstu/hstu_attention/_kernels/utils.py` | [`c5471cb`](https://github.com/NVIDIA/cudnn-frontend/commit/c5471cbf68aaea8f232e94c3b77168f8185ad6f2) |
