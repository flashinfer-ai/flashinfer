# Changelog

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
