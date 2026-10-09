本次已在 **RTX PRO 5000 72GB Blackwell（SM120）** 上运行 FlashInfer `BatchMLAPagedAttentionKernel`，使用 GLM-4.7-Flash 的实际 attention head 配置。当前安装版本 `flashinfer-python==0.6.15.post1` 的这条 FA2 路径支持 **BF16 / FP16 输入与输出，FP32 累加**；不支持在这条路径直接使用 FP8 / FP4 KV cache。

这里的“精度支持”同时检查了数据类型支持和算子数值误差。实验使用合成输入，不加载模型权重，也不等同于 GLM 整模型的任务准确率评估。

**运行记录。** 2026-10-09，PyTorch `2.11.0+cu129`、CUDA Toolkit 12.9、驱动 580.126.20。GPU 有 110 个 SM，CUDA 查询到每 SM 可用 shared memory 为 102400 B，每 block opt-in 上限为 101376 B。本机已有 SGLang 进程占用大部分显存，实验使用独立的小张量和 128 MiB workspace。

可复现文件：

- [demo.py](demo.py)：最小的 20-head BF16 decode 示例，已运行。
- [run_mla.py](run_mla.py)：10 组数值校验、数据类型拒绝检查和 profiler。
- [results.json](results.json)：各测试的形状、误差、计划信息和完整 kernel 名。
- [profile_summary.json](profile_summary.json)：GPU kernel 的 launch 参数、寄存器与 shared memory 信息。
- [probe_xqa.py](probe_xqa.py) / [xqa_results.json](xqa_results.json)：另一个 SM120 MLA 后端的 20-head 兼容性检查。

```bash
python demo.py
python run_mla.py
python probe_xqa.py
```

运行使用当前安装包及其可用的编译缓存，没有重新安装 FlashInfer。目标 CUDA 源码、scheduler、MMA helper、JIT 生成器均与安装包 RECORD 哈希一致。Python `_core.py` 在其他后端区域存在本地修改，但本次使用的 `BatchMLAPagedAttentionWrapper` 与经 RECORD 校验的原始备份一致。详情见 [source_provenance.json](source_provenance.json)。本结论针对这个版本及这些文件。

**先对齐 GLM-4.7-Flash 的 head。** 数据来自 [官方 config.json](https://huggingface.co/zai-org/GLM-4.7-Flash/blob/main/config.json)，本地留存于 [glm47_flash_config.json](glm47_flash_config.json)。

| 配置项 | 值 | 对 kernel 的意义 |
|---|---:|---|
| `num_attention_heads` | 20 | TP1 时 `num_heads=20` |
| `num_key_value_heads` | 20 | 展开后的 head 数；压缩 cache 不存 20 份 latent |
| `qk_nope_head_dim` | 192 | 吸收前 Q/K 的非 RoPE 部分 |
| `qk_rope_head_dim` | 64 | kernel 的 `head_dim_kpe=64` |
| `kv_lora_rank` | 512 | kernel 的 `head_dim_ckv=512` |
| `v_head_dim` | 256 | 每个 head 最终的 V 空间维度，位于 latent 输出投影之后 |
| `q_lora_rank` | 768 | Q 投影的低秩中间维度，不是 kernel 的 Q head 维度 |
| `hidden_size` | 2048 | 最终模型 hidden state 维度 |
| `dtype` | bfloat16 | 官方模型配置的默认精度 |

因此要区分两个阶段：

```text
展开的普通 attention：Q/K 每 head = 192 + 64 = 256，V 每 head = 256
吸收后的 MLA attention：Q/K 每 head = 512 + 64 = 576，latent 输出每 head = 512
```

下面用行向量写吸收过程。对于 head h，令 `c_j` 是第 j 个 token 的 512 维 latent，`W_UK,h` 的形状为 `[512,192]`，`W_UV,h` 为 `[512,256]`：

```text
原始 K_nope[j,h] = c_j @ W_UK,h
原始 V[j,h]      = c_j @ W_UV,h

吸收后的 Q_nope[i,h] = 原始 Q_nope[i,h] @ W_UK,h.T   # 192 -> 512

score[i,h,j] = (Q_nope[i,h] · c_j + Q_pe[i,h] · K_pe[j]) / sqrt(192+64)
p[i,h,:]    = softmax(score[i,h,:] + causal_mask)
latent_o[i,h] = sum_j p[i,h,j] * c_j                   # 512 维

head_o[i,h]  = latent_o[i,h] @ W_UV,h                  # 512 -> 256
model_o[i]   = concat_20_heads(head_o[i]) @ W_O        # 5120 -> 2048
```

`BatchMLAPagedAttentionKernel` 只计算上述 score、softmax、latent_o。Q/K 投影、RoPE 旋转和输出投影在它外面完成。它的 `q_nope` 已经吸收过 K 的上投影，不能直接传原始 `[T,20,192]` 的 Q_nope。

**softmax scale 必须保留原始维度：`1/sqrt(192+64)=1/16=0.0625`。** 把它改为 `1/sqrt(512+64)=1/24` 会改变模型的注意力分布。官方配置 `rope_scaling=null`，这里不需要额外的 YaRN 缩放；SGLang 中也从原始 `qk_head_dim` 计算 scaling，见 [deepseek_v2.py](/sgl-workspace/sglang/python/sglang/srt/models/deepseek_v2.py:1757)。

kernel 输入/输出的实际形状如下，`T` 是 batch 内所有 query token 的总数，decode 时 `T=B`：

| 张量 | TP1 形状 |
|---|---|
| `q_nope` | `[T,20,512]` |
| `q_pe` | `[T,20,64]` |
| `ckv_cache` | `[num_pages,page_size,512]` |
| `kpe_cache` | `[num_pages,page_size,64]` |
| `out` | `[T,20,512]` |
| `lse` | `[T,20]`，FP32 |

cache 没有 20-head 这一维，也没有另一份独立的 V cache：`ckv` 同时参与 QK 与 PV。BF16/FP16 下每层每 token 存储 `576*2=1152 B`；若完全展开并分别存储 20 个 K/V heads，则为 `20*(256+256)*2=20480 B`，约为前者的 17.78 倍。这是张量载荷比较，不包括页表、padding 等额外开销。

**从 Python 走到你关心的 CUDA 函数。** 建议按这个顺序读源码：

1. [BatchMLAPagedAttentionWrapper.plan](/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py:1566)：确定 dtype、维度、后端，并生成调度数据。SM120 上 `auto` 选择 FA2。
2. [gen_batch_mla_module](/usr/local/lib/python3.12/dist-packages/flashinfer/jit/attention/modules.py:112)：把 Q/KV/O dtype 和 512/64 维度实例化为 C++ 模板；head 数由运行时参数传入。
3. [MLAPlan](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/scheduler.cuh:1557)：按 query tile 与 KV 段分配任务，生成 `work_indptr`、`kv_start/end` 和 merge 信息。
4. [BatchMLAPagedAttentionWrapper.run](/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py:1740) → [BatchMLAPagedAttentionRun](/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/batch_mla_run.cu:30)：传入张量指针、stride、分页索引和 scale。
5. [BatchMLAPagedAttention](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh:1096)：根据 shared memory 选择模板配置，并 cooperative launch。
6. [BatchMLAPagedAttentionKernel](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh:853)：执行 attention 与 split-KV 结果合并。

kernel 的主干可以压缩为：

```text
遍历当前 CTA/调度组的 work_indptr 任务
  load_q：将 (query token, head) 打包后的 Q 载入 shared memory
  按 page table 加载 KV tile：逻辑 token -> 物理 page + 页内偏移
  对每个 KV tile：
    compute_mla_qk：先 Q_pe @ K_pe.T，再加 Q_nope @ C_KV.T
    logits_mask_：处理 KV 尾部与 causal 边界
    update_mdo_states_：更新 online-softmax 的 m、d，并重缩放输出累积
    compute_mla_pv：P @ C_KV
  normalize_d_ / write_o：写最终输出或 split-KV partial output + LSE
grid.sync()
DevicePersistentMergeStates：合并不同 KV 段的结果
```

分页参数的单位容易混淆：`qo_indptr` 是 query token 的偏移，`kv_indptr` 是 `kv_indices` 中 page 条目的偏移，`kv_len_arr` 是 token 数。`kv_indices` 存物理 page 编号，`page_size` 才是每页 token 数。本次数值测试随机打乱了物理页顺序，并用非整页长度验证了尾部 mask。

使用 split-KV 时，每段分别得到归一化输出 `o_s` 和 LSE `L_s`，再按 `exp(L_s-logsumexp_s(L_s))` 加权合并。实现内部使用 base-2 的等价公式；默认返回的 LSE 也是 base-2。本次与 PyTorch 比较时显式设置 `return_lse_base_on_e=True`。

**20 heads 在 SM120 上怎样分块。** Profiler 捕获到的 BF16 模板为：

```cpp
KernelTraits<
    false,          // CAUSAL：本条 trace 为 decode
    1,              // NUM_STAGES
    false,          // QK_SHARD
    512, 64,        // HEAD_DIM_CKV, HEAD_DIM_KPE
    64, 16,         // CTA_TILE_Q, CTA_TILE_KV
    __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, int
>
```

| 项目 | 本机观测或源码确定值 |
|---|---|
| block | `(32,4,2)`，256 threads，8 warps |
| decode grid | `(1,110,1)` |
| Q tile | 64 行，每行是一个 `(token,head)` |
| KV tile | 16 tokens；与 page size 是两个概念 |
| pipeline stage | 1 |
| shared memory / CTA | 92672 B = 90.5 KiB |
| registers / thread | 本次编译产物为 196 |
| 资源允许的常驻 CTA | 至多 1 个 / SM |

shared memory 分支见 [DISPATCH_SMEM_CONFIG](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh:1072)。本机 102400 B 落入 `>=92672` 的分支，因此选择 1 stage、KV tile 16、`QK_SHARD=false`。92672 B 可拆成：Q 的 `64*576*2=73728 B`，KV 的 `16*576*2=18432 B`，以及 512 B 的跨组状态存储；输出 shared memory 与这些区域通过 union 复用。

`load_q` 使用 `divmod(packed_row, num_heads)` 找到 token 和 head。单 token 的 20 heads 占前 20 行，剩余行通过边界条件处理，调用者无需显式 pad 到 32/64。因此 TP1 的 Q 行有效占比是 `20/64=31.25%`；只考虑每卡单 token 时，TP2 为 `10/64=15.625%`，TP4 为 `5/64=7.8125%`。这些是 tile 行占比，不能当成实测 GPU 利用率或端到端加速比。10/5 heads 的测试只是单卡上的对应形状测试，没有启动分布式 TP。

这里也不是“一个 warp 负责一个 head”：每个 warp 处理多个打包行。两个由 4 warps 组成的组在 `QK_SHARD=false` 时重复计算相同 QK，各负责一半 latent 输出维度（256）。源码使用 `mma.sync`；代码里出现 warpgroup 这个词，不代表使用 Hopper 的 WGMMA。

planner 用 split-KV 补充并行度，但并不保证所有 launch 的 CTA 都有 attention 工作。例如本次 `B=1,H=20,KV=1024`：源码的 `f(ceil(1024/110))=64` 得到每段最多 64 KV tokens，因此产生 16 个 attention 分段，每段又遍历 4 个 16-token KV tiles。grid 仍是 110 个 CTA，其中部分 CTA 没有 attention 分段，随后统一进行 grid barrier 和输出合并。不能仅凭 grid 大小判断算力已经跑满。

增量 prefill 可将多个 query token 的 heads 打包：例如 3 个 token 是 60 行。planner 在平均 `query_len*heads >64` 时使用两个 CTA 的逻辑调度组。本次 `[3,5]` query lengths 对应平均 80 行，实测 grid 为 `(2,55,1)`。这里的调度组不要与硬件 thread-block cluster 指令混淆。

**实测数值结果。** 每种 dtype 都跑了以下 5 组，共 10 组全部通过：

| 场景 | heads | query lengths | KV lengths | page size | causal |
|---|---:|---|---|---:|---|
| 单请求 decode | 20 | `[1]` | `[1024]` | 1 | false |
| ragged decode | 20 | `[1,1,1,1]` | `[63,127,1024,4097]` | 16 | false |
| 增量 prefill | 20 | `[3,5]` | `[129,257]` | 16 | true |
| TP2 对应形状 | 10 | `[1]` | `[1024]` | 16 | false |
| TP4 对应形状 | 5 | `[1]` | `[1024]` | 16 | false |

参考实现将输入中已经存储的 BF16/FP16 数值转为 **CPU FP64**，恢复物理页的逻辑顺序，完整计算 QK、mask、softmax 和 PV。误差包含低精度 P、partial/output 舍入等算子误差，但不包含从原始模型 FP32 激活量化到这些输入的误差。

| 输入 dtype | 20-head decode 最大绝对误差 | 该例相对 L2 误差 | 全部 5 例中的最大绝对误差 |
|---|---:|---:|---:|
| BF16 | 0.00141149 | 0.00228003 | 0.00593150 |
| FP16 | 0.00036348 | 0.00028943 | 0.00061147 |

相对 L2 为 `norm(out-ref)/norm(ref)`。BF16 校验容差为 `atol=0.008,rtol=0.02`，FP16 为 `atol=0.002,rtol=0.005`，LSE 为 `atol=0.005,rtol=0.001`。两种 dtype 使用各自生成的随机张量，上表不构成整模型 BF16/FP16 精度优劣结论。

Profiler 单次捕获的上述 decode kernel 时长：BF16 约 16.064 μs，FP16 约 16.127 μs。`results.json` 还记录了逐次 Python 调用间的 CUDA event 时间，但该方法可能包含 GPU 等待主机提交的间隔。已有其他进程、时钟状态和 profiler 开销也会影响结果；这些时间仅为运行记录，不用来排名性能或预测模型 tokens/s。

**SM120 上的数据类型支持边界。** 下表专指本次 `BatchMLAPagedAttentionWrapper(backend='fa2')` → `BatchMLAPagedAttentionKernel`：

| 输入 Q / KV | 输出 | 支持结论 | 证据 |
|---|---|---|---|
| BF16 / BF16 | BF16 | 支持 | 5 组实测通过 |
| FP16 / FP16 | FP16 | 支持 | 5 组实测通过 |
| FP32 / FP32（包括期待 TF32 计算） | — | 不支持 | `plan()` 拒绝 FP32 KV |
| BF16 / FP8 E4M3 | — | SM120 的这条路径不支持 | `plan()` 要求 FA3 + SM90 |
| FP8 E4M3 / FP8 E4M3 | — | 不支持 | FA2 路径拒绝 FP8 KV |
| BF16 / FP8 E5M2 | — | 不支持 | 不在 KV dtype allowlist |
| FP4 / NVFP4 cache | — | 不支持直接传入 | 无 FP4 布局/scale 接口，`uint8` 存储被拒绝 |
| BF16 / INT8 cache | — | 不支持 | `plan()` 拒绝 INT8 KV |

这里的 FP32 支持主要体现在**累加与统计**：QK 的 `s_frag`、PV 的 `o_frag`、online softmax 的 `m/d`、LSE 都用 FP32。进入 PV Tensor Core 运算前，P 会转回 BF16/FP16；split-KV partial output 和最终 output 也会按输出 dtype 存储。因此“FP32 accumulation”不等于整条 attention 都用 FP32 存储和乘法。对应实现见 [compute_mla_pv](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/attention/mla.cuh:499) 与 [mma helper](/usr/local/lib/python3.12/dist-packages/flashinfer/data/include/flashinfer/mma.cuh:318)，后者实际发出：

```text
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

Q 和 KV 在这条路径应使用同一种 16-bit dtype。虽然 `plan()` 有两个独立 dtype 参数，FA2 的 QK helper 并没有提供 BF16/FP16 交叉转换；不能据此假定混合输入受支持。

精确的 FP8 guard 在 [_core.py](/usr/local/lib/python3.12/dist-packages/flashinfer/mla/_core.py:1615)：只允许 FA3、SM90、BF16 Q、FP8 E4M3 KV、512/64 维度。在本机实测手动指定 `backend='fa3'` 后仍报 `FP8 kv_data_type for MLA requires an SM90 (Hopper) device, got SM120.`。不能通过把后端名字改成 FA3 来获得 SM120 FP8 支持。wrapper 的 CUTLASS MLA 分支也面向 SM100/SM110，不能因它们同属 Blackwell 就直接套用 SM120。

**不要把单个 kernel 的限制扩大到整个 GPU 或整个 FlashInfer。** SM120 具备更低精度 Tensor Core 能力，当前 FlashInfer 还提供独立的 XQA MLA 路径，支持 `(BF16,BF16)` 或 `(FP8 E4M3,FP8 E4M3)`。不过本版本 [xqa.py](/usr/local/lib/python3.12/dist-packages/flashinfer/xqa.py:523) 和 [JIT 配置](/usr/local/lib/python3.12/dist-packages/flashinfer/jit/xqa.py:139) 要求 **128 个 query heads**。本次实际传入 GLM 的 20 heads，两种 dtype 均被明确拒绝。它不构成 GLM-4.7-Flash 20-head FP8 decode 的直接替代方案；为了对接而额外补 heads 属于另一项适配和性能验证工作。

模型权重采用 FP8/NVFP4、KV cache 的存储精度、attention kernel 的乘法精度是不同的选择。即使模型权重经过低位量化，调用这个 kernel 时仍可使用 BF16 Q/KV。SGLang 当前 [FlashInfer MLA decode 路径](/sgl-workspace/sglang/python/sglang/srt/layers/attention/flashinfer_mla_backend.py:660) 也有 `get_key_buffer(...).to(q.dtype)` 的显式转换；因此单看 KV 存储配置不能确定 kernel 真正接收到的 dtype。本次实验针对算子本身，不代表已验证所有量化模型部署组合。

学习时可以先运行 `demo.py`，同时打开 `mla.cuh` 的 kernel 主循环，再依次阅读 `load_q`、`compute_mla_qk`、`update_mdo_states_`、`compute_mla_pv` 和 merge。把 **20 heads、512+64 输入、512 latent 输出、1/16 scale、64×16 tile** 这五个数对应到源码，便能把 GLM 的模型配置与 SM120 上实际执行的工作联系起来。

**补充：为什么不直接做 FP8×FP8？** 可以做。SM120 上已有 FP8 MMA，20 heads 也不是硬件障碍。为验证这一点，[fp8_math_probe.py](fp8_math_probe.py) 已在本机实际运行两次 FP8 E4M3 GEMM：`Q8 @ K8.T` 与 `P8 @ C8`，都使用 FP32 累加/输出，softmax 使用 FP32。该实验把 20 heads 补为 32 行，KV 长度为 1024，QK 维度为 576、latent 输出维度为 512；没有补成 128 heads。

```bash
python fp8_math_probe.py
```

Profiler 确认使用了 cuBLAS 的 `e4m3...tensor16x8x32` GEMM。产物中的 kernel 符号名含 `sm89`，这是所选实现的命名，本次运行设备仍是实际查询的 SM120。实验使用分开的 GEMM、物化 score/P，证明 FP8 矩阵计算可行，不代表已经实现融合分页 attention，也不能用来推断融合实现的速度。

此次采用每个张量 `absmax/448` 的简单量化。与相同 BF16 来源输入的 CPU FP64 attention 参考相比，只引入 Q/K 量化后的输出相对 L2 误差为 5.57%，两次 GEMM 都使用 FP8 后为 6.29%，最大绝对误差为 0.04195。完整数据见 [fp8_math_results.json](fp8_math_results.json)。这只是一个合成样本和一种缩放策略的结果，不能换算成 GLM 模型准确率下降；它说明还需要验证量化尺度和误差传播。

现有 FA2 kernel 使用 16-bit load/MMA/P 转换，FP8 改造需要协同调整 MMA 的 K 维分块（此处从 16 到 32）、字节布局、KV 转置和边界、Q/K/P 的量化尺度，以及 online-softmax 与 split-KV 的缩放一致性。只把 dtype 改为 FP8 或移除 Python guard 不会补齐这些操作。尤其 PV 的 P 来自 FP32 softmax，必须另外量化；SM120 XQA 源码就在 [mla_sm120.cu](/usr/local/lib/python3.12/dist-packages/flashinfer/data/csrc/xqa/mla_sm120.cu:794) 将减去局部最大值后的指数权重乘以 448、转换为 E4M3，并在求和和输出归一化中处理该尺度。

因此面向 GLM 的合理实现路线是：为 20 heads 使用有边界处理的 32 行 tile，使用 FP8 Q/K 与 FP8 P/C 的 MMA，保留 FP32 累加、softmax 状态和 merge，最终输出 BF16，并与原有 BF16 kernel 做数值及真实 decode 性能比较。20-head padding 到 32 的方案是可行起点，最佳 tile、scale 粒度和性能收益仍需要融合实现后的实测。
