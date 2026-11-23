# Acknowledgements

The SM90 push communication kernels and their Python integration are developed
by the FlashInfer project and distributed under Apache-2.0.

The private FP8 grouped-GEMM implementation builds on the DeepGEMM execution
model and TensorRT-LLM CUDA support code shipped in FlashInfer's
`csrc/nv_internal/tensorrt_llm` tree. The fused FC1 source retains the
DeepSeek MIT, NVIDIA Apache-2.0, and FlashInfer Apache-2.0 notices that apply to
its respective contributions. The remaining source files retain their own
copyright and SPDX notices.

DeepGEMM is maintained at <https://github.com/deepseek-ai/DeepGEMM> and
TensorRT-LLM at <https://github.com/NVIDIA/TensorRT-LLM>.
