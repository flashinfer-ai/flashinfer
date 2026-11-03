# Legacy NVFP4 NCCL GIN：接口、设计与验证

本 PR 将 legacy MegaMoE 的 NVFP4 NCCL GIN 路径接入 FlashInfer 的正式
`MoEEpLayer` API，使一个 MoE 层能够跨两个独立 NVLink domain 执行。
该能力必须显式启用，当前只开放下表中的固定配置。已有 intranode 默认行为保留。
固定范围的正确性验证已完成；off 性能门槛全部通过，internode 性能仍有明确的验收缺口。

本文面向 API、通信和 kernel reviewer，说明最终行为与修改理由。
[迁移审计](migration-audit.md)保留来源与逐文件记录；
[性能报告](performance-fix-blackwell.md)保留每个测量范围、原始证据引用及失败历史。

## 1. 接口变化与调用约定

`Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig` 新增两个仅限关键字的参数：

| 参数 | 默认值 | 启用 GIN 后的含义 |
| --- | --- | --- |
| `internode` | `False` | 显式选择 legacy NCCL GIN；不自动切换 next provider |
| `ranks_per_node` | `None` | 每个 NVLink domain 的 rank 数；当前必须为 `2` |

现有构造调用可以继续省略这两个参数。`kernel_name` 必须是
`sm100_nvfp4_nvfp4_bf16_cutedsl`；与 `_cutedsl_next` 一起设置 `internode=True`
会被拒绝。on 模式不会推断 `ranks_per_node`，省略或保留 `None` 会被拒绝；
off 模式不使用该参数。本 PR 没有改变 next provider 的实现或默认选择策略。

### 1.1 当前开放范围

| 项目 | 必须满足的配置 |
| --- | --- |
| EP / 拓扑 | 默认 process group，world size `4`，两个独立 NVLink domain，每个 domain 两个连续 rank |
| 容量 / hidden | `max_tokens_per_rank=32`，`token_hidden_size=7168`；每 rank 独立满足 `0 ≤ T ≤ 32`，包括空 rank |
| 专家 / top-k | 全局 `48` 个专家，每 rank `12` 个，`top_k=4` |
| intermediate | 公共 `intermediate_size=1536`；FC1 gate-up 宽度为 `3072`，FC2 输入宽度为 `1536` |
| 数值路径 | NVFP4×NVFP4，标准 SwiGLU，early routing weights，BF16 combine，单独 TopkReduce |
| finalization | `FinalizeMode.AUTO` 或 `DEFAULT`，都解析为 DEFAULT 的独立 TopkReduce；不启用 in-kernel FC2 reduce |
| 其他配置 | 默认 fast math、`input_norm_const=1.0`、`knobs=None`；不允许自定义 tile/cluster/schedule |

不开放 subgroup、其他拓扑/shape、internode CUDA Graph、autotune、自定义 activation、
epilogue scale、FC1 per-token scale、量化 combine 或自定义 finalizer。
这些限制针对新增 on 路径；原有 off 路径保留其支持的 activation、scale、finalizer 和 graph。

### 1.2 API 使用示例

每个进程先绑定本 rank 的 CUDA device，并初始化默认 process group。
为控制校验提供 CPU Gloo backend 可降低公共调用开销：

```python
torch.cuda.set_device(local_rank)
dist.init_process_group(backend="cpu:gloo,cuda:nccl")

backend = MegaConfig(
    megakernel=Sm100_Nvfp4_Nvfp4_Bf16_Cutedsl_MegaMoeConfig(
        intermediate_size=1536,
        top_k=4,
        internode=True,
        ranks_per_node=2,
    ),
    preprocess_weights=True,
    quantize_input=True,
)
layer = MoEEpLayer(
    bootstrap=BootstrapConfig(
        world_size=4, rank=dist.get_rank(), auto_bootstrap=False, device=local_rank
    ),
    fleet_params=FleetParams(
        num_experts=48, max_tokens_per_rank=32, token_hidden_size=7168
    ),
    weights=expert_weights,
    backend=backend,
)
y = layer.forward(
    MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
)
torch.cuda.synchronize()
# 仅在正常结束时销毁；致命错误由 launcher 终止整个作业。
layer.destroy()
dist.destroy_process_group()
```

此处 `x` 是连续 CUDA BF16 `[T,7168]`；`ids` 是连续 CUDA int32/int64 `[T,4]`；
`weights` 是连续 CUDA FP32 `[T,4]`。expert ID 使用全局编号 `0..47`；`-1`
表示无效 route，对应 payload/score 不应参与结果。`quantize_input=False` 时，调用方
提供打包的 `torch.float4_e2m1fn_x2` activation，以及 `float8_e4m3fn` SF；
固定 hidden 下 packed activation shape 是 `[T,3584]`；SF 是连续的
二维 `[T,S]`，`S ≥ 448`，按每16个 hidden 元素一个 SF 的原始行布局提供，
不是 weight SF 的 blocked/swizzled 布局。校验入口是 backend 的
`validate_forward` 和 `staging.validate_nvfp4_forward_inputs`。
两种输入路径都已通过公共 API 测试。

示例中的 `expert_weights` 使用 BF16 `MoEWeightPack`：
`w13=[12,3072,7168]`（gate 与 up 先后拼接），`w2=[12,7168,1536]`，
位于本 rank CUDA device；`preprocess_weights=True` 将其量化并转换为 kernel layout。
已有预量化权重入口保持不变；其 packed/logical shape、SF shape、gate/up interleave
和 K-major layout 必须满足现有 `nvfp4_nvfp4_bf16_cutedsl/weights.py` 与 transformed-weight
validator，不能只凭相同 dtype 复用其他 kernel 的权重。本次 example 的权重准备采用 BF16 路径。

[完整可执行示例](../../../examples/nvfp4_gin.py)包含 weights、fleet/bootstrap、
imports 和销毁流程。在两个独立 domain 上分别设置 `NODE_RANK=0/1`：

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 examples/nvfp4_gin.py
```

所有默认组 rank 必须按相同顺序进入初始化、forward/staging 和资源生命周期。
这是集体调用契约，不能只让有 token 的 rank 调用。两个主机本身并不能证明是两个
NVLink domain：运行时仍验证 NCCL device API、railed GIN/RDMA、LSA team 数与本地 peer 可达性。
NCCL-only 默认组仍可使用原有控制协议；CPU Gloo 是已测的优化路径，并非额外正确性前提。

底层 shim 的 `get_symm_buffer_for_mega_moe` 同时增加仅限关键字的
`internode=False`、`ranks_per_node=None`、`process_group=None`；它们将配置传给
GIN session。正常集成应使用上面的层 API，不能绕过公共 allowlist 声称支持更多配置。
正常结束时先调用 `layer.destroy()`，再由调用方销毁默认 process group。workspace-backed output view 和 launch thunk
不能超过所属 workspace/session 的生命周期；需要长期保存结果时使用拥有独立存储的输出。

## 2. 设计与实现

### 2.1 在正式 backend 内接入，保留 vendor 边界

调用链为 `MoEEpLayer → MoEEpMegaLayer → Nvfp4CutedslMegaKernelBackend → package exports
→ NVFP4 shim / GinSession → legacy device kernel`。core 不直接导入 vendor `src/`。
新增能力位 `MegaKernelBackend.requires_collective_forward_validation` 默认为 `False`；
只有启用 GIN 的 backend 将其设为 `True`。core 因而可以在 rank-local 输入检查和 staging
之前进行集体校验，同时保留其他 backend 的现有热路径。

新增路径使用一个独立的 NCCL communicator/session，借用默认 Torch process group
进行 bootstrap/control，不拥有或销毁调用方的 process group。on 路径不依赖
NVSHMEM peer-pointer 初始化；off 的已有 runtime 依赖保持原样。

### 2.2 一个注册 root，统一 peer 地址语义

`shim/gin.py` 在一个完整的连续 CUDA uint8 allocation 中划分 128-byte 对齐区域：

```text
root
 ├─ gin_transport（offset 0，固定 donor 的 window wire layout）
 ├─ packed activation
 ├─ activation SF
 ├─ top-k weights
 └─ compiled-kernel shared workspace
```

NCCL 注册整个 root；所有需要远端访问的 view 都从这个 root 派生。这样本地 peer
base 的差值才能用于这些 view，不能将独立 allocation 混入同一组 peer offsets。
transport 位于 offset 0，使 donor wire offset 与 registered-window offset 保持一致。
shape、dtype、storage extent、对齐、不重叠和边界均在 host 校验。

session 通过默认组广播 UID，创建 communicator，确认 rail/LSA 能力，创建 device
communicator 和 registered window，再查询同 domain 的 peer 地址。
`GinSession` 创建并持有 root、window、device communicator 和 communicator；
workspace/views/thunk 借用这些资源。权重和非 peer-visible 的本地状态不需要搬入
这个注册 root。输出 view 属于 workspace，后续调用可能覆写，不能在 workspace 销毁后使用。

### 2.3 通信、归约与代际状态

迁移保留 legacy dispatch、FC1/activation/FC2 和 token-back 的协作调度，增加
跨 domain relay 与返回路径。通信的固定 wire layout 从固定 donor 引入；
最终按既定 BF16 term/节点顺序执行 combine，再由单独 TopkReduce 形成输出。
bitwise protocol oracle 与独立 Torch math oracle 分开验证，后者的容差不能掩盖通信错误。

迁移期间定位并修复了 token-back generation race：较快的 relay producer 可能先推进
`combine_issue_counter`，较慢 consumer 随后取 `counter+1` 就会等待下一次 launch，形成
循环等待。修正后的等待目标来自 `internode_barrier_counter+1`；该 generation 在
token-back 退休之后才推进，因此本次 launch 内目标稳定。此局部 device 修改在
VENDOR 中登记为 pending-upstream diff，不声称是原 donor 的逐字副本。

### 2.4 集体错误处理和资源生命周期

初始化在 communicator、allocation、registration、peer mapping 等资源边界汇总
rank-local 错误，避免某个 rank 提前退出、其他 rank 继续进入设备通信。
公共 forward 和独立 `stage_inputs` 每次都校验 metadata、route 值与 bound device。
预检同时比较 phase、配置 signature 和 session identity。

默认组有 `cpu:gloo` 时，正常路径收集固定大小 CPU header，使用 128-bit BLAKE2b
指纹比较 phase/signature；任何错误或不一致让所有 rank 回到详细 object-report 路径。
header all-gather 完成后，每个 rank 读取同一组 header，并共同决定是否进入
object fallback；不会一部分 rank 继续 header、一部分 rank 单独进入 object。
每个边界使用相同 header 协议，避免不同 phase 选择不同控制通信而挂起。
NCCL-only 组保留 object-report 协议。该优化没有移除错误传播和拓扑检查。

已汇总的 rank-local 输入错误表现为 `GinPreflightError`（`RuntimeError` 子类），
在设备通信前于所有 rank 被拒绝，修正后可以继续调用。不能把所有 `RuntimeError`
都视为可恢复：资源/control collective 失败可能包装为 `GinFatalError`，
原始 CUDA/NCCL 异常同样要求按下面的致命错误流程处理。
控制集体通信、NCCL 或 device 通信失败则视为整个作业失败：停止复用 session，
依靠全 rank launcher/timeout 退出；不承诺错误后的 communicator recovery。
正常销毁先同步设备，再按 window → device communicator → communicator → root
释放。同步或释放失败时保留其依赖，避免提前释放注册内存。

同一 session 上只支持有序、串行的公共调用；host 调用不并发，前一次设备工作
也必须在复用共享 workspace 前完成，或通过 stream/event 依赖建立先后关系。
独立 session 可共存并已测试。
允许单个 rank 在有序调用之间切换 eager CUDA stream，thunk 在执行时绑定当前 stream；
这不提供跨 stream 的自动生产者依赖，调用方仍须保证输入和前次调用已正确排序。
CUDA device 不能在 session 生命周期内更换。internode capture 被显式拒绝。

### 2.5 控制开销与 off 路径

准备好的 launch 参数按 workspace、weight identity、compiled callable 缓存。
eager GIN 的 key 不包含 rank-local stream，以免仅一个 rank cache miss 后进入
`launch-prepare`、其他 rank 却进入 `launch`。off/graph 的缓存仍按 stream 区分。
去掉重复校验后，稳态 GIN 公共 forward 保留两次 collective agreement，原先为五次；
首次编译和 cache miss 仍有额外准备步骤。

普通 off case 编译 donor 已有的 `launch_without_gin` entry，避免传递无用 GIN 参数。
持久化 finalizer 和 FC1 per-token scale 等 FI 扩展仍走原完整 entry。
性能修复阶段没有更改 vendor device 算术或通信协议，也没有改变 tactic 或放宽容差。
这与前述迁移阶段的 generation correctness patch 是两件不同的修改。

## 3. 代码修改范围

以下路径相对于 repository root；kernel 路径前缀为
`flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/`。

| 范围 | 主要修改 | reviewer 关注点 |
| --- | --- | --- |
| `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/{config,backend}.py`（位于 `flashinfer/moe_ep/`） | 新参数、固定 allowlist、集体校验、workspace/launch cache、capture 拒绝 | 新开关不会自动选 next；错误在 staging 前传播 |
| `flashinfer/moe_ep/core/kernel/base.py`、`modes/mega_layer.py` | 通用 collective-validation 能力和调用顺序 | 其他 backend 默认能力为 false；避免 rank-local 提前退出 |
| `shim/gin.py`、`shim/gin_debug.py` | 注册 arena、NCCL/LSA session、错误边界和诊断 | 地址/生命周期；诊断不绕过 capability checks |
| `shim/nvfp4.py`、`shim/autotune.py`、package exports | frontend/runtime 接线、非共享 GIN session、lean off entry、tuning 拒绝 | thunk 的 session/device/stream 归属 |
| `src/src/`、`src/common/`、`src/moe_nvfp4_swapab/` | 17 个 device 文件的针对性 import/merge | wire ABI、generation、barrier 和 FI 既有差异；详见 hunk ledger |
| `tests/cpu/moe_ep/`、`tests/moe_ep/test_nvfp4_cutedsl_gin_*.py` | host contracts、真实 N2R2 公共 API/强 oracle、off finalizer 回归 | 拓扑 guard、全 rank 结果和无效 route |
| `examples/nvfp4_gin.py`、`benchmarks/moe_ep/*legacy_nvfp4_gin*.py` | 可执行示例、基线断言、计时与报告 | 独立的计时范围、payload provenance 和失败留存 |
| `scripts/audit_legacy_nvfp4_gin.py`、本目录 | 固定源审计、设计与验证记录 | 部分迁移，不将整个 vendor snapshot 改标为 donor |

原 vendor snapshot 为 `fa662f345980f75776ae0c85d2e5c188918ded26`。
此次 donor 固定于 `f863d7540614bd546a42867c21adbf51d573061a`，接收基线
F_BASE 固定于 `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34`。
这不是整包替换：FI 的 sparse decode、masked reduction、per-token scale、custom
activation、singleton TMA 以及 BF16/MXFP8 兼容行为保留。逐文件差异见
[device hunk ledger](evidence/device-hunk-ledger.json)；后续同步仍需保留这些本地修改。

## 4. 正确性验证与可复现性

| 验证层 | 已完成结果 |
| --- | --- |
| Host contracts | 16 项：arena/wire、LSA offsets、集体错误和生命周期 |
| 固定 N2R2 强 oracle | 每 rank 24 次 exact protocol 调用，全作业合计 96 项 transport/L1 byte 检查；包含 partial/empty rank、sparse holes 与 NaN invalid scores |
| 独立 math | 原容差不变；最大 relative L2 为 `0.0026485608`，阈值 `0.02` |
| 最终公共/强测试 | 每 rank 17 项通过，无 skips；覆盖 packed/BF16、rank-local 错误恢复、双 session/重建、capture 拒绝、单 rank stream 切换和 device mismatch |
| 最终 off 兼容性 | 每 rank 4 项 finalizer 和 1 项 workspace/cache/graph 测试通过；更早 EP1/2/4、BF16/MXFP8 回归结果保留 |
| 源码一致性 | GPU 验收时检查的 111 个包/测试文件 hash 与 launch snapshot 一致（包含 Python、脚本与元数据）；17 个 device 文件与迁移 ledger 一致 |

最终 GPU correctness source 为 `74c0362878cb6dd5d0c90c346c7b420848563f1d`；
后续修改只涉及文档、vendor 状态说明和归档回执，Python runtime/test 字节未变。
更早的 EP1/2/4、BF16/MXFP8 完整矩阵是历史验证，并未在最终源码重跑全部组合；
最终源码重跑的是本表明确列出的17项与4+1项；每 rank 的24次 protocol 调用
包含在17项测试中的 matrix/balanced 两个参数化 oracle case 内（各12次），不是额外一轮。失败的中间作业不计作通过。
普通单机四 rank CI 不能替代两 domain 的 N2R2 验证。多节点测试需要正确的 torchrun
geometry 和真实 rail/LSA 能力；跳过 topology-specific 测试不算验证成功。

复现命令及环境见 [validation](validation.md)。login 环境没有 Torch/GPU，
不能用 host contracts 代替真实 CUDA/NCCL 结果；文档变更不重新触发完整 GPU 矩阵。

## 5. Blackwell 性能

### 5.1 基线和计时范围

GPU 为 GB300（SM103，152 SM/device），Torch `2.13.0a0+9186a08b2c.nv26.07`、
CUDA `13.3`、CuTe DSL `4.6.2`、nccl4py `0.3.1`、NCCL `2.30.7`、driver `580.173.10`。
测量使用独占 GPU，未锁定 clocks；每对实现使用同一次 Slurm 作业资源分配（相同物理 GPU）和相同输入；
这里的 allocation 不是指两个实现共享 CUDA 内存。

- **Off baseline：**上述 F_BASE 的 legacy 实现。
- **On baseline：**固定 donor 的原生 `pre_unification/moe_nvfp4_swapab/mega_runner.py`，
  来自实际测量；不是 next，也不是推算出的迁移前数字。
- **Raw pipeline：**CUDA events 包围 mega kernel、独立 TopkReduce 及其间 launch gap；
  不能将其称为单个 fused kernel 的耗时。profiler 单独记录各 kernel。
- **Full operator：**完成同步的 wall time，含公开调用中的校验、staging、allocation 和 launch。
  native donor 是 prepared adapter，缺少等价的 FI 公共校验，因此 on 的 full ratio 范围不同。

每组五轮交替 source、每轮五次 warmup 和二十次采样，每个 source 共 100 样本；逐 iteration
使用最慢 rank。warm/cold 分开，cold 在计时外驱逐 512 MiB/rank。
所有接收的样本均通过 bit/payload 检查。门槛为 median 回退不超过 5%，但只有
baseline round-median CV 不超过 5% 时才给出通过/失败，否则记为 inconclusive。
on fixture 使用固定种子的 random top-k，不是严格均衡路由；所有实现共享同一 payload。

### 5.2 结果

Off shape 为 H2048、公共 I1024、E8、K4。下表是 warm raw median，单位 µs；
EP1/2/4 × T64/1024 × warm/cold × raw/full 的完整 **24 项门槛均通过**。每项包含两个 source 各100个样本，合计4800样本。

| EP | T/rank | F_BASE legacy | Receiver |
| --- | --- | ---: | ---: |
| 1 | 64 | 91.44 | 87.50 |
| 1 | 1024 | 132.40 | 127.07 |
| 2 | 64 | 106.98 | 101.52 |
| 2 | 1024 | 152.82 | 146.02 |
| 4 | 64 | 129.73 | 124.43 |
| 4 | 1024 | 184.62 | 181.89 |

On shape 为 N2R2、T32、H7168、公共 I1536、E48、K4。
同一 allocation 的 host 修复对照测得 FI public full-call **6155.76 → 2208.60 µs**；
该次 before/after 四个 rank 的 PTX/cubin 完全一致，支持 host overhead 的归因。
最终两次 agreement 版本在另一 allocation 的 standard warm full median 为
`1630.15 µs`；不能把跨 allocation 的 `6155.76 → 1630.15` 当作配对加速比。

| 最终 standard raw | Native legacy donor µs | Receiver µs | Baseline round CV | 结论 |
| --- | ---: | ---: | ---: | --- |
| Warm | 350.37 | 349.31 | 6.38% | Inconclusive |
| Cold | 363.98 | 356.43 | 6.39% | Inconclusive |

该次 native prepared full-call warm median 为 `402.74 µs`，FI public 为 `1630.15 µs`。
FI 仍有真实的公共校验成本，不能声称已达到 native adapter 的完整调用延迟；
同样不能用二者 full ratio 宣称 device kernel 回退四倍。

EP4 控制实验表明，GPU-local/disjoint CPU placement 可以显著降低同一 baseline 的
launch 方差；但它没有消除所有 N2R2 波动。旧 raw +18.8% 回退在其他 allocation
发生方向反转，最终 baseline 仍未达到稳定性门槛。结论是 off 修复已验收、GIN host
开销已降低；internode 原始流水线没有足够稳定的基线来正式验收，也不宣称普遍加速。

完整 cold/full、独立 conditioned protocol、分布、binding 和失败记录见
[性能报告](performance-fix-blackwell.md)，统计见
[off summary](evidence/blackwell-fix-off-summary.json)及
[on summary](evidence/blackwell-fix-on-summary.json)。

## 6. 合并评审边界

本 PR 的可评审交付是默认关闭、固定范围的 legacy GIN 集成及上述验证结果。
扩大 shape/topology、internode graphs、去掉公共校验、改变 reduce 精度/顺序、
kernel autotuning 和正式 internode 性能 promotion 都需要后续工作。
当前 PR 不以 baseline 波动为理由删除不利样本，也不将 correctness 通过等同于性能通过。
