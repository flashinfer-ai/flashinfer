# Changelog

## [0.2.0.post1](https://github.com/flashinfer-ai/flashinfer/compare/v0.2.0...v0.2.0.post1) (2024-12-22)

### Bug Fixes

* bug fix on determine_attention_backend condition ([#688](https://github.com/flashinfer-ai/flashinfer/pull/688)) ([bcf7a3e](https://github.com/flashinfer-ai/flashinfer/commit/bcf7a3ee0d919eca45d2f07241479b5776975bc3))
* accelerate plan speed of fa3 template ([#690](https://github.com/flashinfer-ai/flashinfer/pull/690)) ([db8f04d](https://github.com/flashinfer-ai/flashinfer/commit/db8f04d30989f57acef3fbde41cbd3ce373727f1))

## [0.2.0](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.6...v0.2.0) (2024-12-17)

### Release Blog

[FlashInfer 0.2 - Efficient and Customizable Kernels for LLM Inference Serving](https://flashinfer.ai/2024/12/16/flashinfer-v02-release.html)

### Features

* add `rotary_dim` argument to rope APIs for partial apply rope ([#599](https://github.com/flashinfer-ai/flashinfer/issues/599)) ([eb9bc71](https://github.com/flashinfer-ai/flashinfer/commit/eb9bc710ce875dd276109b6b62745fc1282f1541))
* add a `use_softmax` field in variant class ([#533](https://github.com/flashinfer-ai/flashinfer/issues/533)) ([d81af97](https://github.com/flashinfer-ai/flashinfer/commit/d81af9775e56bb30152b17770e804823cddfc279))
* add an option `non_blocking` to plan function ([#622](https://github.com/flashinfer-ai/flashinfer/issues/622)) ([560af6f](https://github.com/flashinfer-ai/flashinfer/commit/560af6f687524a2415eb94ad333b65b9461a47b1))
* add gemma_rmsnorm and gemma_fused_add_rmsnorm ([#477](https://github.com/flashinfer-ai/flashinfer/issues/477)) ([1a6b17e](https://github.com/flashinfer-ai/flashinfer/commit/1a6b17e2b78fc811d50030b9326a4d01f1ff956f))
* add group size 3 to GQA decode dispatch ([#558](https://github.com/flashinfer-ai/flashinfer/issues/558)) ([6227562](https://github.com/flashinfer-ai/flashinfer/commit/62275625f9332e40a69789467835cbb376f2940d))
* add JIT compilation support for FA3 templates ([#672](https://github.com/flashinfer-ai/flashinfer/issues/672)) ([d4e8d79](https://github.com/flashinfer-ai/flashinfer/commit/d4e8d79b340589633943bebd827da17b3f4c29ad))
* allow the cascade kernels to be executed using varying sequence lengths ([#627](https://github.com/flashinfer-ai/flashinfer/issues/627)) ([92ac440](https://github.com/flashinfer-ai/flashinfer/commit/92ac4401d434e988ec8aeb769ecf3ff575c32983))
* CUDAGraph compatibility of multi-level cascade inference APIs ([#586](https://github.com/flashinfer-ai/flashinfer/issues/586)) ([2332e8a](https://github.com/flashinfer-ai/flashinfer/commit/2332e8ae477656b2be060465b30c30b5dee389b9))
* fix the maximal grid dimension in prefill planning with CUDA graphs ([#639](https://github.com/flashinfer-ai/flashinfer/issues/639)) ([86ca89a](https://github.com/flashinfer-ai/flashinfer/commit/86ca89a60f1bf1eb566cb9e45d21e4c8f174c251))
* improve the precision of the FusedAddRMSNormKernel function ([#587](https://github.com/flashinfer-ai/flashinfer/issues/587)) ([c7dc921](https://github.com/flashinfer-ai/flashinfer/commit/c7dc921f9323d2f767fd8e9d9d0ab4c1d95ad1b5))
* JIT compilation ([#507](https://github.com/flashinfer-ai/flashinfer/issues/507)) ([3613a5b](https://github.com/flashinfer-ai/flashinfer/commit/3613a5bd829234863a96bc23e3bd2a1da345a592))
* modify group-gemm stage number ([#497](https://github.com/flashinfer-ai/flashinfer/issues/497)) ([52dab1d](https://github.com/flashinfer-ai/flashinfer/commit/52dab1d4a4d7e5d910a8c695de911d979d6f2038))
* non-contiguous query with paged kv cache ([#553](https://github.com/flashinfer-ai/flashinfer/issues/553)) ([89f2c4a](https://github.com/flashinfer-ai/flashinfer/commit/89f2c4a816ff133e09cb9fc1d7c3de43d4431ffd))
* pass a dynamic token count to the cascade kernels ([#635](https://github.com/flashinfer-ai/flashinfer/issues/635)) ([5fe9f7d](https://github.com/flashinfer-ai/flashinfer/commit/5fe9f7d1d1ab8aa13cb6073a6447e383ad52b484))
* simplify prefill JIT compilation ([#605](https://github.com/flashinfer-ai/flashinfer/issues/605)) ([fe4f898](https://github.com/flashinfer-ai/flashinfer/commit/fe4f8980223a92cc918f2e6041df854fcebefbc9))
* specify gemm backend ([#648](https://github.com/flashinfer-ai/flashinfer/issues/648)) ([0cc1a51](https://github.com/flashinfer-ai/flashinfer/commit/0cc1a51757e73a4f4a1be9f2e7ac0e0f2c156056))
* support cached cos/sin in rope APIs ([#585](https://github.com/flashinfer-ai/flashinfer/issues/585)) ([83e541d](https://github.com/flashinfer-ai/flashinfer/commit/83e541d8fa2b15ff23c8c68c136fa5023e2c977d))
* support huggingface transformer style rope interface ([#568](https://github.com/flashinfer-ai/flashinfer/issues/568)) ([4f40420](https://github.com/flashinfer-ai/flashinfer/commit/4f40420e24d65cabd8be731e12f96a5ef0795a4b))
* support sm90 cutlass group gemm ([#509](https://github.com/flashinfer-ai/flashinfer/issues/509)) ([794bdda](https://github.com/flashinfer-ai/flashinfer/commit/794bdda1ea2d62d4d2c0e858553058ad890ee5e3))
* torch custom_op fix for rope ([#569](https://github.com/flashinfer-ai/flashinfer/issues/569)) ([3e104bc](https://github.com/flashinfer-ai/flashinfer/commit/3e104bc7769735af83ffc709fe1f7a641f2471da))
* torch custom_op support: norm ([#552](https://github.com/flashinfer-ai/flashinfer/issues/552)) ([f6e0010](https://github.com/flashinfer-ai/flashinfer/commit/f6e0010833f54a5b8181a9232588649f0b3c182e))
* torch.compile and custom_op support ([#554](https://github.com/flashinfer-ai/flashinfer/issues/554)) ([9bf916f](https://github.com/flashinfer-ai/flashinfer/commit/9bf916f236139f5b6410e298615d0db152e82409))
* warmup for jit kernel tests ([#629](https://github.com/flashinfer-ai/flashinfer/issues/629)) ([8f5f349](https://github.com/flashinfer-ai/flashinfer/commit/8f5f3491c523f5c43623d3cd3eaa42854f47ad76))


### Bug Fixes

* AOT compiler flags on non-sm90 ([#522](https://github.com/flashinfer-ai/flashinfer/issues/522)) ([0aa4726](https://github.com/flashinfer-ai/flashinfer/commit/0aa47269f9f06f20e4a15662931972c9a2de482f))
* batch decode kernel redundant store output to gmem ([#505](https://github.com/flashinfer-ai/flashinfer/issues/505)) ([90e42a7](https://github.com/flashinfer-ai/flashinfer/commit/90e42a7307dad08bc1f800efb3d73a3bd22a0824))
* compatible with torch 2.2 ([#478](https://github.com/flashinfer-ai/flashinfer/issues/478)) ([ac41d1b](https://github.com/flashinfer-ai/flashinfer/commit/ac41d1bdc72ed4614c9eafb8644d45b234260005))
* https://github.com/flashinfer-ai/flashinfer/issues/452 ([b53a46f](https://github.com/flashinfer-ai/flashinfer/commit/b53a46f8b073e66fbc8fe888e87517b3aea8bd2d))
* remove redundant load ([#495](https://github.com/flashinfer-ai/flashinfer/issues/495)) ([2de16b0](https://github.com/flashinfer-ai/flashinfer/commit/2de16b0f4afbb9d3c5725187ee2f14ef08fa364f))
* update bmm fp8 test ([#487](https://github.com/flashinfer-ai/flashinfer/issues/487)) ([45eac04](https://github.com/flashinfer-ai/flashinfer/commit/45eac04f9420b2372737d16d51f4d07bf928d293))


### Performance Improvements

* accelerate JIT compilation speed ([#618](https://github.com/flashinfer-ai/flashinfer/issues/618)) ([eaf73fd](https://github.com/flashinfer-ai/flashinfer/commit/eaf73fd0246f32f214f1db6ed8143bf8a503aae4))
* Dense and sparse customizable flashattention-3 template ([#667](https://github.com/flashinfer-ai/flashinfer/issues/667)) ([51236c9](https://github.com/flashinfer-ai/flashinfer/commit/51236c913107f2f6098ac039a4aaa4841a443c25))
* fix prefill kernel performance degradation (step 1) ([#602](https://github.com/flashinfer-ai/flashinfer/issues/602)) ([595cf60](https://github.com/flashinfer-ai/flashinfer/commit/595cf602e73688d2f96f8cf1aad7cb2fce689d41))
* fix the performance issue of `append_paged_kv_cache` ([#588](https://github.com/flashinfer-ai/flashinfer/issues/588)) ([e15f7c9](https://github.com/flashinfer-ai/flashinfer/commit/e15f7c984bc4152c0b65cfec916ace37c98668cd))
* improve parallelism in RoPE with pos_ids ([#609](https://github.com/flashinfer-ai/flashinfer/issues/609)) ([ff05155](https://github.com/flashinfer-ai/flashinfer/commit/ff05155581f5e085b573f803aed398434859e22f))
* improve plan performance by using non-blocking memcpy ([#547](https://github.com/flashinfer-ai/flashinfer/issues/547)) ([41ebe6d](https://github.com/flashinfer-ai/flashinfer/commit/41ebe6dce7c505801853a27246feea2e06500620))
* reduce the read and write of shared memory in the FusedAddRMSNormKernel ([#592](https://github.com/flashinfer-ai/flashinfer/issues/592)) ([2043ca2](https://github.com/flashinfer-ai/flashinfer/commit/2043ca2181d1e9119a1fb8b86a739c245be5b536))
* reduce total_num_tiles_q by one ([#644](https://github.com/flashinfer-ai/flashinfer/issues/644)) ([553ace5](https://github.com/flashinfer-ai/flashinfer/commit/553ace5eb91fc07681fa9edf8b6c09827a72617a))
* remove unnecessary contiguous operation in block sparse attention ([#561](https://github.com/flashinfer-ai/flashinfer/issues/561)) ([7a7ad46](https://github.com/flashinfer-ai/flashinfer/commit/7a7ad4659a7b7e1a78eebbb9bb8af6c21130f14e))
* speedup jit compilation of prefill attention kernels ([#632](https://github.com/flashinfer-ai/flashinfer/issues/632)) ([a059586](https://github.com/flashinfer-ai/flashinfer/commit/a0595866db384b4a782c1ec70df72251b17de287))
* use cuda-core implementation for io-bound block-sparse attention ([#560](https://github.com/flashinfer-ai/flashinfer/issues/560)) ([3fbf028](https://github.com/flashinfer-ai/flashinfer/commit/3fbf02800e6166d2bf9e1de1cfa6ac826fa4618d))

## [0.1.6](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.5...v0.1.6) (2024-08-27)

### SM75 Support

Starting from [0.1.6](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.5...v0.1.6), our pre-built wheels include experimental support sm75 (Turing architecture GPUs such as Tesla T4, Quadro RTX 6000 and RTX 2080).

### API Changes

#### `plan`/`run`

Since [0.1.6](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.5...v0.1.6) on, `begin_forward`/`forward`/`end_forward` APIs are replaced with the new `plan`/`run` API.
- `forward` is renamed to `run`, which is more precise and consistent with the naming convention of cutlass's python API.
- `begin_forward` is renamed to `plan`, which is consistent with the naming convention of nvmath API.
- `end_forward` is deprecated and has no effect after this PR.

There is some slight difference between the old `forward` and the new `run` API:
- All extra arguments such as `causal` and `logits_soft_cap` will be provided in `plan` (previously `begin_forward`) API, and cached until next `plan` call, and we only need to provide query and KV-Cache tensors in `run` API.

The old `begin_forward`/`forward`/`end_forward` APIs are still functional, but we will gradually deprecate them in future releases.

Check [#466](https://github.com/flashinfer-ai/flashinfer/pull/466) for more details.

#### `MultiLevelCascadeAttentionWrapper`

Since [0.1.6](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.5...v0.1.6) on, we introduce a new `MultiLevelCascadeAttentionWrapper` API for cascade inference,
which supports multi-level cascade inference where all levels' KV-Cache can be managed in a unified Paged KV-Cache.

See [documentation](https://docs.flashinfer.ai/api/python/cascade.html#flashinfer.cascade.MultiLevelCascadeAttentionWrapper) and [tutorial](https://docs.flashinfer.ai/tutorials/kv_layout.html#multi-level-cascade-inference-data-layout) on API usage and layout explanation.

The old `BatchDecodeWithSharedPrefixPagedKVCacheWrapper` and `BatchPrefillWithSharedPrefixPagedKVCacheWrapper` will be deprecated in future releases.

### Features

* sm75 support ([#448](https://github.com/flashinfer-ai/flashinfer/pull/448), [#449](https://github.com/flashinfer-ai/flashinfer/pull/449))
* add `MultiLevelCascadeAttentionWrapper` API ([#462](https://github.com/flashinfer-ai/flashinfer/issues/462)) ([1e37989](https://github.com/flashinfer-ai/flashinfer/commit/1e379898a589cdd4ff18a4621fcbe18d63501545))
* add accept num, emit num metric for ChainSpeculativeSampling ([#450](https://github.com/flashinfer-ai/flashinfer/issues/450)) ([fa38b5e](https://github.com/flashinfer-ai/flashinfer/commit/fa38b5e34b9591bd5ab07186bea229ea95307755))
* support bmm fp8 ([#469](https://github.com/flashinfer-ai/flashinfer/issues/469)) ([f1c0b68](https://github.com/flashinfer-ai/flashinfer/commit/f1c0b68d0f4a77ff3bf705307b3529b996fc9826))

### Refactor

* refactor: replace `begin_forward`/`forward`/`end_forward` with `plan`/`run` [#466](https://github.com/flashinfer-ai/flashinfer/pull/466)

### Misc

* misc: improve error handling of sampling kernels ([#456](https://github.com/flashinfer-ai/flashinfer/pull/456)) ([0dce178](https://github.com/flashinfer-ai/flashinfer/commit/0dce178389e5e85b1d40212b1d12d1754304e46))

### Performance Improvements

* slight optimization on f16-&gt;f8 fragment layout swizzling ([#453](https://github.com/flashinfer-ai/flashinfer/issues/453)) ([0d61871](https://github.com/flashinfer-ai/flashinfer/commit/0d618712faff20a84bbd513d02ac01e16be19306))
* slight optimization on fragment layout swizzle ([#458](https://github.com/flashinfer-ai/flashinfer/issues/458)) ([7c397cb](https://github.com/flashinfer-ai/flashinfer/commit/7c397cbd81d4fa5da8aef9f105576dbe67f6c22b))
* use persistent kernel for merging attention states ([#459](https://github.com/flashinfer-ai/flashinfer/issues/459)) ([be6bf5b](https://github.com/flashinfer-ai/flashinfer/commit/be6bf5bb26f1f1b3edf094d903544600c574ee09))

### Acknowledgement

We thank [@LiuXiaoxuanPKU](https://github.com/LiuXiaoxuanPKU) on enhance of speculative sampling operator, [@merrymercy](https://github.com/merrymercy) on API change suggestion and [@zhyncs](https://github.com/zhyncs) on integrating fp8 BMM cublas implementation.

## [0.1.5](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.4...v0.1.5) (2024-08-13)


### Bugfix

* resolve cu121 compile wired issue ([#446](https://github.com/flashinfer-ai/flashinfer/issues/446)) ([5f0159e](https://github.com/flashinfer-ai/flashinfer/commit/5f0159e6abeb7308d965bb1b9aef05547b8a57b3))
* Fix PagedPrefill python api and some typos ([#441](https://github.com/flashinfer-ai/flashinfer/pull/441)) ([3fff008](https://github.com/flashinfer-ai/flashinfer/commit/3fff008dc9af56c325d9c487bddf69ff014f3989))
* fix prefill kernels' lse result for empty kv-cache ([#440](https://github.com/flashinfer-ai/flashinfer/pull/440)) ([6ac28f4](https://github.com/flashinfer-ai/flashinfer/commit/6ac28f4dd3a9a34a2b4abcbe0a815fc59a2d74ad))

### Features

* decouple float and int workspace buffer ([#442](https://github.com/flashinfer-ai/flashinfer/issues/442)) ([a7ee566](https://github.com/flashinfer-ai/flashinfer/commit/a7ee5662bf967ab1ee16910c73761d326fbeb9a0))


### Performance Improvements

* faster fp8-&gt;fp16 dequantization for pre sm_90 arch ([#439](https://github.com/flashinfer-ai/flashinfer/issues/439)) ([c93f647](https://github.com/flashinfer-ai/flashinfer/commit/c93f647a0dd6b58c9ac20b39438316202358463c))

### Acknowledgement

We thank contributions and feedbacks from the community: [@comaniac](https://github.com/comaniac), [@hnyls2002](https://github.com/hnyls2002), [@jianfei-wangg](https://github.com/jianfei-wangg), [@Yard1](https://github.com/Yard1).



## [0.1.4](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.3...v0.1.4) (2024-08-09)


### Features

* append attention kernels for fp8 kv-cache ([#420](https://github.com/flashinfer-ai/flashinfer/issues/420)) ([906c2f5](https://github.com/flashinfer-ai/flashinfer/commit/906c2f5df3b35df45a4fb2614815308b662099ea))
* support min_p sampling ([#422](https://github.com/flashinfer-ai/flashinfer/pull/422)) ([d52f2da](https://github.com/flashinfer-ai/flashinfer/commit/d52f2da6825f0fd7f614bf3a2db3b75c8fef961b))
* deterministic sampling ([#417](https://github.com/flashinfer-ai/flashinfer/issues/417)) ([0dd801d](https://github.com/flashinfer-ai/flashinfer/commit/0dd801d2027af89f3603cbbf68a76e9503bb2f57))
* more sampling operator options ([#431](https://github.com/flashinfer-ai/flashinfer/issues/431)) ([68df9c4](https://github.com/flashinfer-ai/flashinfer/commit/68df9c487e672b4a4ea3be97aed63a48aac5945b))
* support fused add rmsnorm ([#419](https://github.com/flashinfer-ai/flashinfer/issues/419)) ([b781513](https://github.com/flashinfer-ai/flashinfer/commit/b78151383d4a75094195cba29aba45d694d5fdb7))
* support fused silu mul ([#427](https://github.com/flashinfer-ai/flashinfer/issues/427)) ([ea0ba9a](https://github.com/flashinfer-ai/flashinfer/commit/ea0ba9a51238597bd7863b6e3c9bfda574df4df5))

### Bug Fixes

* fix dispatch fp16 type when enable fp8 ([#430](https://github.com/flashinfer-ai/flashinfer/pull/430)) ([daa5566](https://github.com/flashinfer-ai/flashinfer/commit/daa556697fed849810745f0aae0015d8e4460050))
* improve numerical stability of sampling kernels ([#429](https://github.com/flashinfer-ai/flashinfer/pull/429)) ([898d8ea](https://github.com/flashinfer-ai/flashinfer/commit/898d8ea8a21f5850288bc4a860399678131a2d30))

### Other improvements

* break up `_kernels` into multiple modules ([#428](https://github.com/flashinfer-ai/flashinfer/pull/428)) ([8e482d9](https://github.com/flashinfer-ai/flashinfer/commit/8e482d92cb0ad046ec5f57509f9473e76bd668fe))

### Acknowledgement

We thank contributions and feedbacks from the community: [@comaniac](https://github.com/comaniac), [@esmeetu](https://github.com/esmeetu), [@LiuXiaoxuanPKU](https://github.com/LiuXiaoxuanPKU), [@peng1999](https://github.com/peng1999), [@xslingcn](https://github.com/xslingcn), [@Yard1](https://github.com/Yard1), [@zhyncs](https://github.com/zhyncs).


## [0.1.3](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.2...v0.1.3) (2024-07-31)

### Bugfix

* bugfix: Fix cudagraph mode of BatchPrefillWithRaggedKVCacheWrapper ([#412](https://github.com/flashinfer-ai/flashinfer/pull/412)) ([9907bc](https://github.com/flashinfer-ai/flashinfer/commit/9907bc163eec7677870014b6ed5bb1789cc584f0))
* fix cu118 cub usage for sampling kernels ([#410](https://github.com/flashinfer-ai/flashinfer/pull/410)) ([58d359](https://github.com/flashinfer-ai/flashinfer/commit/58d35930740083f27e65c9818ab857f9f4880aff))

### MiscBreak up _kernels into multiple modules

* enhance allocator error info and add shape check for prefill begin forward functions ([#413](https://github.com/flashinfer-ai/flashinfer/pull/413)) ([5e36c5](https://github.com/flashinfer-ai/flashinfer/commit/5e36c527bb10c9331a17d4ecd609120406280979))

## [0.1.2](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.1...v0.1.2) (2024-07-29)

### Bugfix
* Fix the sampling kernel bug for cu118 ([#386](https://github.com/flashinfer-ai/flashinfer/pull/386), [#387](https://github.com/flashinfer-ai/flashinfer/pull/387)) ([0cd499](https://github.com/flashinfer-ai/flashinfer/commit/0cd49949e6c05a0c8f63d050ff96c8f6168cf914), [dc3f18](https://github.com/flashinfer-ai/flashinfer/commit/dc3f184eda83b9feb5c901606b3d8aede23a4a5f))

### Features

* add llama 3.1 style rope ([#401](https://github.com/flashinfer-ai/flashinfer/issues/401)) ([4c89dec](https://github.com/flashinfer-ai/flashinfer/commit/4c89decadc8ae9f261cae97c350064156e66bc09))
* non-inplace rope operators ([#405](https://github.com/flashinfer-ai/flashinfer/issues/405)) ([74ffba1](https://github.com/flashinfer-ai/flashinfer/commit/74ffba1d1b946fcd3536b7637a4e1a999e5a5d3e))
* sliding window attention ([#406](https://github.com/flashinfer-ai/flashinfer/issues/406)) ([28cffd3](https://github.com/flashinfer-ai/flashinfer/commit/28cffd366888649a1e9d871efec32e67b88070cb))
* support non-contiguous (packed) input for prefill kernels ([#404](https://github.com/flashinfer-ai/flashinfer/issues/404)) ([68c3719](https://github.com/flashinfer-ai/flashinfer/commit/68c3719113f90bed5bf1a5d4990f8e2c0b0f5fd3))


### Performance Improvements

* slight optimization on merge states ([#313](https://github.com/flashinfer-ai/flashinfer/issues/313)) ([701c813](https://github.com/flashinfer-ai/flashinfer/commit/701c813cb1266f8dd2b93d17978d35fd6fb975dd))

## [0.1.1](https://github.com/flashinfer-ai/flashinfer/compare/v0.1.0...v0.1.1) (2024-07-20)

### Bugfix

* fix the invalid kernel configuration for architectures with small shared memory size ([#385](https://github.com/flashinfer-ai/flashinfer/pull/385)) ([cdac57](https://github.com/flashinfer-ai/flashinfer/commit/cdac577011e8ab50aa26dfef0cecf77d92d2f804))

### Features

* expose decoupled kv-cache to pytorch api ([#383](https://github.com/flashinfer-ai/flashinfer/issues/383)) ([457a0ae](https://github.com/flashinfer-ai/flashinfer/commit/457a0ae0c8a43bd95a803167e28be19555a2ebf8))


### Performance Improvements

* use stmatrix in epilogue for sm90+ ([#380](https://github.com/flashinfer-ai/flashinfer/issues/380)) ([c6f20d1](https://github.com/flashinfer-ai/flashinfer/commit/c6f20d1406a3a8c4f134c4a764d16e157a184338))

## [0.1.0](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.9...v0.1.0) (2024-07-17)


### Features

* Add mask to `merge_state_in_place` ([#372](https://github.com/flashinfer-ai/flashinfer/issues/372)) ([e14fa81](https://github.com/flashinfer-ai/flashinfer/commit/e14fa8194cfc09c271e6f2c102060698f18297a9))
* expose pytorch api for block sparse attention ([#375](https://github.com/flashinfer-ai/flashinfer/issues/375)) ([4bba6fa](https://github.com/flashinfer-ai/flashinfer/commit/4bba6fa3aa848d2e43248bca8d959fd58a27cfa4))
* Fused GPU sampling kernel for joint top-k & top-p sampling ([#374](https://github.com/flashinfer-ai/flashinfer/issues/374)) ([6e028eb](https://github.com/flashinfer-ai/flashinfer/commit/6e028eb997173658832a66c7480cc9224d637a15))

## [0.0.9](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.8...v0.0.9) (2024-07-12)

### Bugfix

* fix the decode kernel segfault in cudagraph mode ([#368](https://github.com/flashinfer-ai/flashinfer/pull/368))([c69cfa](https://github.com/flashinfer-ai/flashinfer/commit/c69cfabc540e4a7edd991713df10d575ff3b0c21))
- fix decode kernels output for empty kv cache ([#363](https://github.com/flashinfer-ai/flashinfer/pull/363))([ac72b1](https://github.com/flashinfer-ai/flashinfer/commit/ac72b1cc14a6474d601f371c8d69e2600ac28d2f))
- check gpu id in PyTorch APIs and use input tensor's gpu default stream ([#361](https://github.com/flashinfer-ai/flashinfer/pull/361))([1b84fa](https://github.com/flashinfer-ai/flashinfer/commit/1b84fab3e4f53fb4fa26952fdb46fa8018634057))

### Performance Improvements

* accelerate alibi ([#365](https://github.com/flashinfer-ai/flashinfer/issues/365)) ([4f0a9f9](https://github.com/flashinfer-ai/flashinfer/commit/4f0a9f987ad2036f3c466257459de823be85fcc6))
* accelerate gqa performance ([#356](https://github.com/flashinfer-ai/flashinfer/issues/356)) ([e56ddad](https://github.com/flashinfer-ai/flashinfer/commit/e56ddadf4bdbb164c3f1a03f9f69cb8a25621ef5))
* Optimize tensor conversions in C++ code to avoid unnecessary copies ([#366](https://github.com/flashinfer-ai/flashinfer/issues/366)) ([1116237](https://github.com/flashinfer-ai/flashinfer/commit/1116237ac1e5690cf404841327b58b1d268d9951))

### Acknowledgement

We thank [@Yard1](https://github.com/Yard1), [@Ying1123](https://github.com/Ying1123) and [@zhyncs](https://github.com/zhyncs) for their contributions.

## [0.0.8](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.7...v0.0.8) (2024-07-03)

### Bugfix

* fix prefill/append kernel behavior for empty kv-cache ([#353](https://github.com/flashinfer-ai/flashinfer/pull/353)) ([7adc8c](https://github.com/flashinfer-ai/flashinfer/commit/7adc8cf01a029645307c321a7754d0b0a4f0f4de))
* fix decode attention kernel with logits cap ([#350](https://github.com/flashinfer-ai/flashinfer/pull/350)) ([f5f7a2](https://github.com/flashinfer-ai/flashinfer/commit/f5f7a2a23249fd0be5b30fd8fb3957ac3bb527ca))


## [0.0.7](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.6...v0.0.7) (2024-06-28)

### Breaking Changes
* `batch_decode_with_padded_kv_cache` was removed, we encourage user to use `BatchDecodeWithPagedKVCacheWrapper` instead. ([#343](https://github.com/flashinfer-ai/flashinfer/pull/343))

### Bugfix

* fix the `forward_return_lse` function in `BatchPrefillWithRaggedKVCache` class ([#337](https://github.com/flashinfer-ai/flashinfer/pull/337))
* fix the scheduler behavior of large page size ([#333](https://github.com/flashinfer-ai/flashinfer/pull/333))

### Features

* customize `logits_soft_cap` value ([#339](https://github.com/flashinfer-ai/flashinfer/issues/339)) ([a2498f5](https://github.com/flashinfer-ai/flashinfer/commit/a2498f511b354ce049bda6be320a24b73c719be3))


### Performance Improvements

* change minimal `kv_chunk_size` back to 128 ([#329](https://github.com/flashinfer-ai/flashinfer/issues/329)) ([f237f5f](https://github.com/flashinfer-ai/flashinfer/commit/f237f5f80199e2c433fcca750713c6e774693b58))
* more options for kv tile size ([#336](https://github.com/flashinfer-ai/flashinfer/issues/336)) ([bf2a6c7](https://github.com/flashinfer-ai/flashinfer/commit/bf2a6c7c05a82e0ee0ea04381d04b84327355b69))

## [0.0.6](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.5...v0.0.6) (2024-06-21)

### Bugfix

Fix some bug in v0.0.5 that might lead to crashes and instable performance.

### Performance Improvements

* use 1x4 warp layout for small query length ([#322](https://github.com/flashinfer-ai/flashinfer/issues/322)) ([4e89b4d](https://github.com/flashinfer-ai/flashinfer/commit/4e89b4dfdeb0c07b290ace9f82edf31e63136cfd))

## [0.0.5](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.4...v0.0.5) (2024-06-20)

### Highlights

* Support any GQA group size support for tensor-cores kernels.
* Support any page size support for tensor-cores kernels.
* Support CUDA-Graph for prefill/decode APIs.
* Add an option to accelerate decode kernels with Tensor Cores.
* Support custom attention mask. (https://docs.flashinfer.ai/tutorials/kv_layout.html#mask-layout-2d-ragged-tensor)
* Support logits cap in Grok-1 models.
* Fused GPU-sampling kernels: top-p, top-k, speculative verification. (https://docs.flashinfer.ai/api/python/sampling.html)
* PyTorch wrapper of group-gemm cutlass kernels. (https://docs.flashinfer.ai/api/python/group_gemm.html)

### Acknowledgement

We thank [@ibsidorenko](https://github.com/ibsidorenko), [@LiuXiaoxuanPKU](https://github.com/LiuXiaoxuanPKU), [@Yard1](https://github.com/Yard1) [@AgrawalAmey](https://github.com/AgrawalAmey), [@xuzhenqi](https://github.com/xuzhenqi), [@mgerstgrasser](https://github.com/mgerstgrasser), [@esmeetu](https://github.com/esmeetu), [@yz-tang](https://github.com/yz-tang), [@HSQ79815](https://github.com/HSQ79815), [@Qubitium](https://github.com/Qubitium), [@shreygupta2809](https://github.com/shreygupta2809), [@sighingnow](https://github.com/sighingnow), [@vinx13](https://github.com/vinx13),
[@tqchen](https://github.com/tqchen), [@merrymercy](https://github.com/merrymercy), [@comaniac](https://github.com/comaniac) and many others for their contributions and helpful discussions for 0.0.5 release.

### Refactor

* support any GQA group size for tensor-cores kernels ([#301](https://github.com/flashinfer-ai/flashinfer/pull/301)) ([c111ca](https://github.com/flashinfer-ai/flashinfer/commit/c111ca630d57bc4c301fff2599253a5d782a95c8))
* support any page size for tensor-cores kernels ([#306](https://github.com/flashinfer-ai/flashinfer/pull/306)) ([82fd8c](https://github.com/flashinfer-ai/flashinfer/commit/82fd8c7ee2d569b1876d547f73c7ad4b085a771e))


### Features

* add `use_tensor_cores` option to decode kernels to accelerate GQA ([#317](https://github.com/flashinfer-ai/flashinfer/issues/317)) ([3b50dd5](https://github.com/flashinfer-ai/flashinfer/commit/3b50dd59b0e1f23905e583d5af069e43ff5e15a4))
* add group gemm operators ([#282](https://github.com/flashinfer-ai/flashinfer/issues/282)) ([e08ba42](https://github.com/flashinfer-ai/flashinfer/commit/e08ba4226f694d5469cce4233f1854c965f05197))
* initial support of distributed operators ([#289](https://github.com/flashinfer-ai/flashinfer/issues/289)) ([03553da](https://github.com/flashinfer-ai/flashinfer/commit/03553dac1dffff9a6867be0d5676d69d6eeae18c))
* initial support of logits hook ([#298](https://github.com/flashinfer-ai/flashinfer/issues/298)) ([ab1e2ad](https://github.com/flashinfer-ai/flashinfer/commit/ab1e2ad89f27319f5b4874c5e8b526c1cae43598))
* Separate Q and KV dtypes for decode ([#286](https://github.com/flashinfer-ai/flashinfer/issues/286)) ([5602659](https://github.com/flashinfer-ai/flashinfer/commit/5602659d8cd0616ec8214d056ea5c4078b21342b))
* support cuda graph for batched multi-query(prefill/append) attention ([#275](https://github.com/flashinfer-ai/flashinfer/issues/275)) ([83ceb67](https://github.com/flashinfer-ai/flashinfer/commit/83ceb67a5773b0447f5f0344411abfdbc53cf5f4))
* support cuda graph for batched multi-query(prefill/append) attention ([#277](https://github.com/flashinfer-ai/flashinfer/issues/277)) ([24cc583](https://github.com/flashinfer-ai/flashinfer/commit/24cc583cb6b1a205aa8aad53f56472305b73f5f4))
* support custom attention mask in prefill/append attention kernels ([#266](https://github.com/flashinfer-ai/flashinfer/issues/266)) ([7304282](https://github.com/flashinfer-ai/flashinfer/commit/7304282a8068942100f8e59adff533ce28f4d3e5))
* fused speculative sampilng kernels ([#259](https://github.com/flashinfer-ai/flashinfer/pull/259)) ([cea2bb](https://github.com/flashinfer-ai/flashinfer/commit/cea2bb9a836ba6d34d6667b8983ad79fa35cf933))
* expose sampling APIs in pytorch ([#238](https://github.com/flashinfer-ai/flashinfer/pull/238)) ([092902](https://github.com/flashinfer-ai/flashinfer/commit/0929023e5325a30357750eacec27b0d3a20d1254))


### Performance Improvements

* initial cuda graph support ([#256](https://github.com/flashinfer-ai/flashinfer/issues/256)) ([7e9cc7f](https://github.com/flashinfer-ai/flashinfer/commit/7e9cc7ff42ca283c317061a877305d09a395fad2))
* split kv-cache for prefill/append kernels ([#310](https://github.com/flashinfer-ai/flashinfer/issues/310)) ([f0bb0a3](https://github.com/flashinfer-ai/flashinfer/commit/f0bb0a3a723cbe1a138c604680e6b573d877f210))
* use packed bit array for attention mask ([#308](https://github.com/flashinfer-ai/flashinfer/issues/308)) ([3d43dc9](https://github.com/flashinfer-ai/flashinfer/commit/3d43dc9dc1a2ae804eaa7e40b4555e471fd03fe3))

## [0.0.4](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.3...v0.0.4) (2024-05-01)


### Features

* pytorch 2.3 support
* gpu sampling kernels (top-p, top-k)
* more gqa group sizes
* add mma instructions for fp8 ([#179](https://github.com/flashinfer-ai/flashinfer/issues/179)) ([d305798](https://github.com/flashinfer-ai/flashinfer/commit/d3057983e6d47e857ec3956de94eb11f62d9d83e))
* mma rowsum for fp8 ([#180](https://github.com/flashinfer-ai/flashinfer/issues/180)) ([5af935c](https://github.com/flashinfer-ai/flashinfer/commit/5af935ca783d3487034110902c6406089c31acbc))
* support any num_heads for get_alibi_slope ([#200](https://github.com/flashinfer-ai/flashinfer/issues/200)) ([b217a6f](https://github.com/flashinfer-ai/flashinfer/commit/b217a6fefb7bd091469467d32b8aedde4a25cad7))

### Bug Fixes

* fix python package dispatch error message ([#182](https://github.com/flashinfer-ai/flashinfer/issues/182)) ([8eed01c](https://github.com/flashinfer-ai/flashinfer/commit/8eed01c094ceb47375a1d4da8748c43a2947e959))

## [0.0.3](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.2...v0.0.3) (2024-03-08)


### Features

* adding `sm_scale` field for all attention APIs ([#145](https://github.com/flashinfer-ai/flashinfer/issues/145)) ([85d4018](https://github.com/flashinfer-ai/flashinfer/commit/85d4018de4766dafd1be60cf6d953cd9236a4058))
* enable `head_dim=256` for attention kernels ([#132](https://github.com/flashinfer-ai/flashinfer/issues/132)) ([0372acc](https://github.com/flashinfer-ai/flashinfer/commit/0372acc44d0d393af7fd9fb3dcef0ff25953d4e1))
* pytorch api of fp8 kv-cache ([#156](https://github.com/flashinfer-ai/flashinfer/issues/156)) ([66ee066](https://github.com/flashinfer-ai/flashinfer/commit/66ee06683eaea7efe724c46df528ae47aa75eca2))
* support ALiBi ([#146](https://github.com/flashinfer-ai/flashinfer/issues/146)) ([383518b](https://github.com/flashinfer-ai/flashinfer/commit/383518bdf1824f68d33a2eaafd72a780f195bdd4))


### Bug Fixes

* bugfix to pr 135 ([#136](https://github.com/flashinfer-ai/flashinfer/issues/136)) ([3d55c71](https://github.com/flashinfer-ai/flashinfer/commit/3d55c71a62052c590c130897d3a3db49b14fcc34))
* fix bugs introduced in [#132](https://github.com/flashinfer-ai/flashinfer/issues/132) ([#135](https://github.com/flashinfer-ai/flashinfer/issues/135)) ([9b7b0b9](https://github.com/flashinfer-ai/flashinfer/commit/9b7b0b913e1fbef7aac6351109911c7ac08a8904))
* fix FindThrust.cmake ([#161](https://github.com/flashinfer-ai/flashinfer/issues/161)) ([30fa584](https://github.com/flashinfer-ai/flashinfer/commit/30fa5843aeb1ac48816967a63db140cff6044e13))


### Misc
* add stream argument in BeginForwardFunction of TVMWrapper ([#164](https://github.com/flashinfer-ai/flashinfer/pull/164)) ([fabfcb5](https://github.com/flashinfer-ai/flashinfer/tree/fabfcb5751dcc003137a5a7d2d5514f3afe2e302))


### Performance Improvements

* multiple q by sm_scale in decode kernels ([#144](https://github.com/flashinfer-ai/flashinfer/issues/144)) ([660c559](https://github.com/flashinfer-ai/flashinfer/commit/660c559348ba9710d0d81b53f710f7e4951eee2b))

## [0.0.2](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.1...v0.0.2) (2024-02-17)


### Bug Fixes

* add python 3.9 wheels to ci/cd ([#114](https://github.com/flashinfer-ai/flashinfer/issues/114)) ([2d8807d](https://github.com/flashinfer-ai/flashinfer/commit/2d8807d1fb3359ace8a03b73c92bd0679b9d4b33))
* version names cannot include multiple `+` ([#118](https://github.com/flashinfer-ai/flashinfer/issues/118)) ([af6bd10](https://github.com/flashinfer-ai/flashinfer/commit/af6bd10db03fa1353699631f6b31eee52d343569))
* version naming issue ([#117](https://github.com/flashinfer-ai/flashinfer/issues/117)) ([c849a90](https://github.com/flashinfer-ai/flashinfer/commit/c849a90e6b6756a2ca87733782607796d8c7b85a))
