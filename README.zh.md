<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-black-background.png?raw=true">
    <img alt="FlashInfer" src="https://github.com/flashinfer-ai/web-data/blob/main/logo/FlashInfer-white-background.png?raw=true" width=55%>
  </picture>
</p>
<h1 align="center">
面向大模型推理的高性能 GPU 算子库
</h1>

<p align="center">
| <a href="https://docs.flashinfer.ai"><b>官方文档</b></a> | <a href="https://github.com/flashinfer-ai/flashinfer/releases/latest"><b>最新发布</b></a> | <a href="https://flashinfer.ai"><b>技术博客</b></a> | <a href="https://join.slack.com/t/flashinfer/shared_invite/zt-379wct3hc-D5jR~1ZKQcU00WHsXhgvtA"><b>Slack 社区</b></a> | <a href="https://github.com/orgs/flashinfer-ai/discussions"><b>讨论论坛</b></a> |
</p>

<p align="center">
  <a href="README.md">English</a> | <b>简体中文</b>
</p>

[![Build Status](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/badge/icon)](https://ci.tlcpack.ai/job/flashinfer-ci/job/main/)
[![Documentation](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml/badge.svg)](https://github.com/flashinfer-ai/flashinfer/actions/workflows/build-doc.yml)

**FlashInfer** 是一个专为大语言模型（LLM）推理打造的高性能算子库（Library）与算子生成器（Kernel Generator），能够在多样化的 GPU 架构上提供业界领先（SOTA）的推理加速性能。它为注意力机制（Attention）、矩阵乘法（GEMM）以及混合专家模型（MoE）运算提供了统一的 API 接口，并深度集成了包括 FlashAttention-2/3、cuDNN、CUTLASS 以及 TensorRT-LLM 在内的多种高性能后端实现。

## 为什么选择 FlashInfer？

- **业界领先的性能表现**：针对预填充（Prefill）、解码（Decode）以及混合批处理（Mixed Batching）等核心推理场景进行了深度算子优化
- **多后端智能调度**：根据您的目标硬件架构与实际工作负载特征，自动选择最优的底层计算后端
- **现代 GPU 架构全覆盖**：全面支持 SM75（Turing 架构）及更高版本（直至最新一代 Blackwell 架构）
- **低精度量化计算**：支持面向 Attention、GEMM 与 MoE 算子的 FP8 及 FP4 低精度量化计算
- **生产级就绪**：全面兼容 CUDAGraph 与 `torch.compile`，满足低延迟在线推理服务需求

## 核心特性

### 注意力算子 (Attention Kernels)
- **分页与不规则 KV-Cache (Paged & Ragged KV-Cache)**：面向动态批处理（Dynamic Batch）在线推理服务的高效显存管理
- **Decode、Prefill 与 Append**：针对注意力计算全阶段深度优化的专用算子
- **MLA 注意力机制**：原生支持 DeepSeek 的多头潜在注意力（Multi-Latent Attention）
- **级联注意力 (Cascade Attention)**：面向共享前缀（Shared Prefixes）场景的高显存效率分层 KV-Cache 计算
- **稀疏注意力 (Sparse Attention)**：支持块稀疏（Block-Sparse）与可变块稀疏（Variable Block-Sparse）模式
- **POD-Attention**：面向混合批处理的融合预填充+解码（Fused Prefill+Decode）算子

### 矩阵乘法与线性运算 (GEMM & Linear Operations)
- **BF16 GEMM**：支持 SM10.0+ GPU 的 BF16 矩阵乘法
- **FP8 GEMM**：支持逐张量（Per-Tensor）与分组（Groupwise）缩放量化
- **FP4 GEMM**：支持 Blackwell 架构 GPU 的 NVFP4 与 MXFP4 矩阵乘法
- **分组矩阵乘法 (Grouped GEMM)**：面向 LoRA 微调适配与多专家路由场景的高效批处理矩阵运算

### 混合专家模型 (Mixture of Experts, MoE)
- **融合 MoE 算子 (Fused MoE Kernels)**
- **多样化路由策略**：支持 DeepSeek-V3、Llama-4 以及标准 Top-K 路由算法
- **量化 MoE (Quantized MoE)**：支持带块级缩放（Block-wise Scaling）的 FP8 与 FP4 专家权重推理

### 采样与解码 (Sampling & Decoding)
- **免排序采样 (Sorting-Free Sampling)**：无需排序即可高效执行 Top-K、Top-P 与 Min-P 采样
- **投机解码 (Speculative Decoding)**：支持链式投机采样（Chain Speculative Sampling）加速

### 分布式通信 (Communication)
- **AllReduce**：提供针对低延迟推理深度定制的 AllReduce 实现
- **多节点 NVLink (Multi-Node NVLink)**：支持用于多机分布式推理的 MNNVL 通信加速
- **NVSHMEM 集成**：支持高效的分布式显存直通操作

### 其他算子 (Other Operators)
- **RoPE**：支持 LLaMA 风格旋转位置编码（含 LLaMA 3.1 扩展格式）
- **归一化 (Normalization)**：支持 RMSNorm、LayerNorm 以及 Gemma 风格的融合归一化算子
- **激活函数 (Activations)**：支持带融合门控（Fused Gating）的 SiLU 与 GELU 算子

## GPU 硬件支持

| 架构 (Architecture) | 计算能力 (Compute Capability) | 代表 GPU 型号 (Example GPUs) |
|--------------|-------------------|------|
| Turing | SM 7.5 | T4, RTX 20 系列 |
| Ampere | SM 8.0, 8.6 | A100, A10, RTX 30 系列 |
| Ada Lovelace | SM 8.9 | L4, L40, RTX 40 系列 |
| Hopper | SM 9.0 | H100, H200 |
| Blackwell | SM 10.0, 10.3 | B200, B300 |
| Blackwell | SM 11.0 | Jetson Thor |
| Blackwell | SM 12.0, 12.1 | RTX 50 系列, DGX Spark |

> **注意：** 并非所有功能特性均在全部计算能力（Compute Capabilities）级别上受支持。

## 最新动态 (News)

最新版本：[![GitHub Release](https://img.shields.io/github/v/release/flashinfer-ai/flashinfer)](https://github.com/flashinfer-ai/flashinfer/releases/latest)

重要更新：
- [2025-10-08] 在 [v0.4.0](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.4.0) 版本中正式新增 Blackwell 架构支持
- [2025-03-10] 发布技术博客 [《面向 LLM 采样的免排序 GPU 算子》](https://flashinfer.ai/2025/03/10/sampling.html)，深入解析 FlashInfer 中采样算子的底层架构设计。

## 快速上手 (Getting Started)

### 安装指南

**快速安装：**

```bash
pip install flashinfer-python
```

**软件包选项：**

- **flashinfer-python**：核心软件包，支持在首次调用时按需即时编译（JIT）或下载算子
- **flashinfer-cubin**：涵盖所有受支持 GPU 架构的预编译算子二进制文件（cubin）
- **flashinfer-jit-cache**：特定于 CUDA 版本的适配包（Shim），用于安装针对特定硬件架构预构建的 JIT 缓存提供程序

**为了获得更快的初始化速度或支持离线环境部署**，建议一并安装可选预编译包以预置绝大多数算子：

```bash
pip install flashinfer-python
flashinfer install-cubin-wheel
flashinfer install-jit-cache-wheel
```

**对于 Blackwell (SM100+) CuTe DSL 算子**，请附加安装 CUDA 13 扩展依赖以启用专为 Blackwell 优化的算子：

```bash
pip install flashinfer-python[cu13]
```

### 验证安装

```bash
flashinfer show-config
```

### 基础用法示例

```python
import torch
import flashinfer

# 单请求 Decode 注意力计算 (Single decode attention)
q = torch.randn(32, 128, device="cuda", dtype=torch.float16)  # [num_qo_heads, head_dim]
k = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)  # [kv_len, num_kv_heads, head_dim]
v = torch.randn(2048, 32, 128, device="cuda", dtype=torch.float16)

output = flashinfer.single_decode_with_kv_cache(q, k, v)
```

有关完整的 API 参考手册与使用教程，请参阅[官方文档](https://docs.flashinfer.ai/)。

### 从源码编译安装

```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
python -m pip install -v .
```

**面向本地开发**，建议使用可编辑模式（Editable Mode）安装：

```bash
python -m pip install --no-build-isolation -e . -v
```

> **注意：** 当使用 `--no-build-isolation` 标志时，`pip` 不会自动安装构建依赖项。FlashInfer 要求 `setuptools>=77`。如果您遇到类似 `AttributeError: module 'setuptools.build_meta' has no attribute 'prepare_metadata_for_build_editable'` 的报错，请先升级 `pip` 与 `setuptools`：
> ```bash
> python -m pip install --upgrade pip setuptools
> ```

构建可选加速包：

```bash
# 构建 flashinfer-cubin
python -m build --no-isolation --wheel flashinfer-cubin
python -m pip install flashinfer-cubin/dist/*.whl
```

为目标 GPU 构建对应的 JIT-cache 提供程序包（Provider），随后构建依赖该提供程序的适配包（Shim）。以下示例展示了如何构建 SM90a 架构的提供程序；请注意两个 Wheel 包必须使用相同的版本配置：

```bash
export FLASHINFER_JIT_CACHE_PROVIDER_ARCH=9.0a
python -m build --no-isolation --wheel flashinfer-jit-cache-provider

export FLASHINFER_JIT_CACHE_PROVIDER_ARCHS="9.0a"
python -m build --no-isolation --wheel flashinfer-jit-cache

python -m pip install \
  flashinfer-jit-cache-provider/dist/*.whl \
  flashinfer-jit-cache/dist/*.whl
```

更多详情请参阅[从源码安装文档](https://docs.flashinfer.ai/installation.html#install-from-source)。

### 每日构建版本 (Nightly Builds)

```bash
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps
pip install flashinfer-python  # 从 PyPI 安装常规依赖项
flashinfer install-cubin-wheel --nightly
flashinfer install-jit-cache-wheel --nightly
```

### CLI 命令行工具

FlashInfer 提供了一系列实用的 CLI 命令，用于环境配置检查、模块状态管理以及辅助开发：

```bash
# 验证安装状态并查看当前环境配置
flashinfer show-config

# 列出并检查算子模块状态
flashinfer list-modules
flashinfer module-status

# 管理算子构建产物与本地缓存
flashinfer download-cubin
flashinfer install-cubin-wheel
flashinfer install-jit-cache-wheel
flashinfer download-kernels
flashinfer clear-cache

# 面向开发者：生成 compile_commands.json 以便集成至 IDE
flashinfer export-compile-commands [output_path]
```

有关完整命令说明，请参阅 [CLI 参考文档](https://docs.flashinfer.ai/cli.html)。

## API 日志调试 (API Logging)

FlashInfer 提供了详尽的 API 日志记录功能以便于调试排障。您可以通过设置环境变量来启用：

```bash
# 启用日志级别（0=关闭 (默认), 1=基础信息, 3=详细日志, 5=性能统计）
export FLASHINFER_LOGLEVEL=3

# 设置日志输出目标（stdout (默认), stderr, 或指定文件路径）
export FLASHINFER_LOGDEST=stdout
```

有关日志级别、配置选项及高级特性的详细说明，请参阅文档中的[日志指南 (Logging)](https://docs.flashinfer.ai/logging.html)。

## 自定义注意力变体 (Custom Attention Variants)

用户可以借助附加参数灵活定制专属的注意力算子变体。更多详情请参考我们的 [JIT 扩展示例](https://github.com/flashinfer-ai/flashinfer/blob/main/tests/utils/test_jit_example.py)。

## CUDA 版本支持

**受支持的 CUDA 版本：** 12.9、13.0 以及 13.4（PyTorch Nightly）

> **注意：** FlashInfer 致力于紧跟 PyTorch 官方支持的 CUDA 版本以及最新的 CUDA 发行版。
> CUDA 13.4 Wheel 包基于预览版工具链与 PyTorch Nightly 构建；当前的运行时 CI 测试主要覆盖 CUDA 12.9 与 13.0。

## 生态采用与集成 (Adoption)

FlashInfer 已为以下主流大模型推理框架与引擎提供底层算子加速支持：

- [SGLang](https://github.com/sgl-project/sglang)
- [vLLM](https://github.com/vllm-project/vllm)
- [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)
- [TGI (Text Generation Inference)](https://github.com/huggingface/text-generation-inference)
- [MLC-LLM](https://github.com/mlc-ai/mlc-llm)
- [LightLLM](https://github.com/ModelTC/lightllm)
- [lorax](https://github.com/predibase/lorax)
- [ScaleLLM](https://github.com/vectorch-ai/ScaleLLM)

## 参与贡献 (Contributing)

请参阅 [CONTRIBUTING.md](CONTRIBUTING.md) 了解开发环境搭建、Pull Request 提交规范（包括 PR 模板、性能优化需附带的性能基准数据要求）以及 CI 触发方式。编程智能体（Coding Agents）会自动从 [CLAUDE.md](CLAUDE.md) / [AGENTS.md](AGENTS.md) 读取相同的工程规范；人工代码审查指南请参阅 [docs/code_review_guidance_human.md](docs/code_review_guidance_human.md)。

## 致谢 (Acknowledgement)

FlashInfer 的设计受到了 [FlashAttention](https://github.com/dao-AILab/flash-attention/)、[vLLM](https://github.com/vllm-project/vllm)、[stream-K](https://arxiv.org/abs/2301.03598)、[CUTLASS](https://github.com/nvidia/cutlass) 以及 [AITemplate](https://github.com/facebookincubator/AITemplate) 等优秀项目的启发。

## 引用 (Citation)

如果您在项目或学术研究中觉得 FlashInfer 有所帮助，欢迎引用我们的[论文](https://arxiv.org/abs/2501.01005)：

```bibtex
@article{ye2025flashinfer,
    title = {FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving},
    author = {
      Ye, Zihao and
      Chen, Lequn and
      Lai, Ruihang and
      Lin, Wuwei and
      Zhang, Yineng and
      Wang, Stephanie and
      Chen, Tianqi and
      Kasikci, Baris and
      Grover, Vinod and
      Krishnamurthy, Arvind and
      Ceze, Luis
    },
    journal = {arXiv preprint arXiv:2501.01005},
    year = {2025},
    url = {https://arxiv.org/abs/2501.01005}
}
```

---

> _注：本简体中文文档由社区志愿者翻译维护。若与英文原版存在差异，请以 [README.md](README.md) 为准。_
