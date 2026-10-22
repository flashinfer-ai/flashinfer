在本次 SM120、GLM-4.7-Flash 20 heads 的实验中，选择 FlashInfer 作为基线有两个具体原因：用户最初指定研究 `BatchMLAPagedAttentionKernel`；这条 FA2 kernel 已在本机实际跑通，并完成 BF16/FP16 数值校验。这不构成 FlashInfer 对所有硬件和形状都快于 FlashMLA 的结论。

| 比较项 | FlashInfer MLA | FlashMLA |
|---|---|---|
| 项目范围 | FlashInfer 提供多种 attention、量化、GEMM 等推理算子，MLA 是其中一部分 | 主要围绕 MLA / 稀疏 attention 的专门实现 |
| 计算语义 | 使用压缩 KV 和吸收后的 Q，输出 latent attention 结果 | 同样可以处理压缩 KV 的 MLA |
| 共同优化基础 | 分块 QK/PV、online softmax、分页 KV、沿 KV 分段 | 同样利用这些基本方法，具体调度和硬件指令取决于实现 |
| kernel 选择 | 同一库中有 FA2、FA3、CUTLASS、XQA 等路径；它们的能力不能互相替代 | 支持范围同样取决于架构、dense/sparse、decode/prefill 和版本 |
| 本机实际 SM120 路径 | 目标 FA2 kernel 可运行，BF16/FP16 20 heads 已验证 | 已安装的 `sgl_kernel.flash_mla` 扩展在 metadata 调用时就报无匹配 kernel image |
| FP8 KV | 目标 FA2 wrapper 当前拒绝 SM120 FP8 KV；独立 XQA MLA 的 128-head 限制另需适配 | 本地存在 FP8 API，但不能据此认定本机 SM120 可执行 |
| 在当前 SGLang 中的集成 | `FlashInferMLAAttnBackend` 提供 MLA 相关执行路径 | `FlashMLABackend` 继承前者，prefill 复用 FlashInfer 路径，decode 等阶段才切换 FlashMLA |

直接证据：

- [SGLang FlashMLABackend](/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashmla_backend.py:58) 的继承关系和注释写明 EXTEND/prefill 使用 FlashInfer 父类。
- [FlashMLA 构建配置](/sgl-workspace/sglang/python/sglang/kernels/aot/cmake/flashmla.cmake:24) 添加了 `sm_90a`、`sm_100a` 和条件性的 `sm_103a`，没有 `sm_120` 构建目标。
- [本机调用记录](flashmla_sm120_probe.json)：`get_mla_metadata` 返回 CUDA 错误 `no kernel image is available for execution on the device`。扩展直接结束了独立测试进程，尚未进入主 attention kernel。

SM100 与 SM120 虽然都属于 Blackwell，但不能共用所有架构专用 kernel。Hopper 的 WGMMA 路径、SM100 的 TCGEN05/TMEM 路径与 SM120 的实现条件不同；某个库在 H100/B200 上支持 MLA 或 FP8，不自动意味着它能运行在 RTX PRO 的 SM120 上。

还应区分框架中的后端名称和 profiler 中真正运行的 kernel。当前工作区另有 `flash_mla_with_kvcache_sm120`，其主要针对特定稀疏 MLA 格式并可转到 FlashInfer/Triton；它并不能证明原生 FlashMLA dense decode 已支持本次 GLM-4.7-Flash。库名、框架 backend 名、最终 CUDA kernel 是三个不同层次。

前期 FP8 性能实验使用独立的 Triton 融合 kernel，与原 FlashInfer BF16 kernel 对照；随后新增了基于 FlashInfer CUDA 的原生 FP8×FP8 分支，decode/prefill 结果见 [native 报告](native_fp8/README.zh.md)。它们是实验实现，不是通过打开 FlashInfer/FlashMLA 的现成开关得到的性能。工作区 SGLang 源码还存在其他本地修改，实验不修改其服务配置，也不把那些修改当作官方库的能力。

如果部署到支持原生 FlashMLA 的 GPU，应以相同模型 head 数、page size、Q/KV dtype、batch、上下文长度以及精度要求，分别测 decode 和 prefill。对本机而言，先验证架构兼容性，再谈谁更快。
