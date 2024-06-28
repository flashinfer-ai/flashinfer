# Changelog

## [0.0.7](https://github.com/flashinfer-ai/flashinfer/compare/v0.0.6...v0.0.7) (2024-06-28)


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
* PyTorch wrapper of group-gemm cutlass kernels. (https://docs.flashinfer.ai/api/python/sampling.html)

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
