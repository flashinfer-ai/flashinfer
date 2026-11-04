# Legacy NVFP4 NCCL GIN：接口、设计与验证

本 PR 将 legacy MegaMoE 的 NVFP4 NCCL GIN 路径接入 FlashInfer 的正式
`MoEEpLayer` API，使一个 MoE 层能够跨两个独立 NVLink domain 执行。
该能力必须显式启用，当前只开放下表中的固定配置。已有 intranode 默认行为保留。
已执行的数值/公共 API 测试通过；off 性能门槛全部通过，internode 性能仍有验收缺口，
新增的 CUDA Graph 路径要求预先 warmup，并约束固定输入地址、shape 与资源生命周期。

本文保留接口、设计、修改范围、验证和性能结论。来源与本地设备差异登记在
[VENDOR.md](../../flashinfer/moe_ep/kernel_src/sm100/cutedsl_megamoe/VENDOR.md)。
调查过程、日志和统计文件保存在独立归档中，不作为代码依赖。

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

`internode` 是本次port的总开关，默认关闭，在创建配置/layer时选择。
`False` 走FI原有intranode路径，忽略GIN专用的`ranks_per_node`，
不进入GIN预检、协商或session创建；
`True` 才进入新增GIN路径，并要求relay和NCCL GIN同时启用。
不在已存活的session/Graph上原地切换开关。

本次只移植GIN功能：撤回新增但未接线的`f4cvt_use_manual`显式参数、diagnostic字段
和`_f4man`命名token。非GIN的FP4 decoder继续由既有`MEGA_F4CVT_USE_MANUAL`
环境变量控制，其转换/归约算法保持FI基线；GIN仍仅使用BF16 reduce。

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

不开放 subgroup、其他拓扑/shape、autotune、自定义 activation、
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

[完整可执行示例](../../examples/nvfp4_gin.py)包含 weights、fleet/bootstrap、
imports 和销毁流程。在两个独立 domain 上分别设置 `NODE_RANK=0/1`：

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 examples/nvfp4_gin.py
```

所有默认组 rank 必须按相同顺序进入初始化、forward/staging 和资源生命周期。
这是集体调用契约，不能只让有 token 的 rank 调用。两个主机本身并不能证明是两个
NVLink domain：运行时仍验证 NCCL device API、railed GIN/RDMA、LSA team 数与本地 peer 可达性。
GIN 初始化要求 PyTorch 的 Gloo backend 可用（不增加 off 路径依赖）。
默认组有 CPU/Gloo 时直接复用；NCCL-only 默认组在 GIN 初始化时集体创建并缓存
一个 Gloo 控制组。它随默认 PG 销毁，不在 capture 或 replay 中创建。

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
workspace factory 和直接 session 初始化都会在创建资源前共同拒绝 capture 初始化。
session 创建后，view、routes/output、epilogue 向量和workspace对象构建由同一阶段保护；
所有rank完成后才返回workspace。局部分配失败时共同拒绝并释放session；清理结果也经协商，
任一rank清理失败都升级为全作业`GinFatalError`，保留失败资源的依赖。

GIN仅支持默认`compile_tokens_per_rank=None`的固定workspace容量；显式bucket（包括32）
不增加支持。超出bucket、各rank bucket不一致及不支持的显式bucket在既有输入agreement内
共同拒绝，修正为默认None后可恢复；不增加普通forward或Graph replay的控制通信轮数。
eager 的公共 forward 和独立 `stage_inputs` 每次都校验 metadata、route 值与 bound device；
capture 只校验 metadata 和绑定状态，route 值是每次 replay 的调用方前提。
预检同时比较 phase、配置 signature 和 session identity。

默认组有 `cpu:gloo` 时，正常路径收集固定大小 CPU header，使用 128-bit BLAKE2b
指纹比较 phase/signature；任何错误或不一致让所有 rank 回到详细 object-report 路径。
header all-gather 完成后，每个 rank 读取同一组 header，并共同决定是否进入
object fallback；不会一部分 rank 继续 header、一部分 rank 单独进入 object。
每个边界使用相同 header 协议，避免不同 phase 选择不同控制通信而挂起。
NCCL-only 默认组的 GIN 校验也使用预创建的 CPU 控制组。详细错误报告始终走
CPU 通道；该路径不会在 capture 中意外提交 NCCL/CUDA collective。

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
CUDA device 不能在 session 生命周期内更换。Graph capture 和 eager 使用相同的
已准备 thunk；调用时绑定 capture stream，并将当前模式纳入 CPU agreement，
在 staging/launch 前共同拒绝混合 eager/capture。

### 2.5 控制开销与 off 路径

准备好的 launch 参数按 workspace、weight identity、compiled callable 缓存。
GIN 的 key 不包含 rank-local stream，以免仅一个 rank cache miss 后进入
`launch-prepare`、其他 rank 却进入 `launch`。off 路径（含其 Graph）的缓存仍按 stream 区分。
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
| `backends/mega/kernel/sm100/nvfp4_nvfp4_bf16_cutedsl/{config,backend}.py`（位于 `flashinfer/moe_ep/`） | 新参数、固定 allowlist、集体校验、workspace/launch cache、预热后的 capture | 新开关不会自动选 next；错误在 staging 前传播 |
| `flashinfer/moe_ep/core/kernel/base.py`、`modes/mega_layer.py` | 通用 collective-validation 能力和调用顺序 | 其他 backend 默认能力为 false；避免 rank-local 提前退出 |
| `shim/gin.py`、`shim/gin_debug.py` | 注册 arena、NCCL/LSA session、错误边界和诊断 | 地址/生命周期；诊断不绕过 capability checks |
| `shim/nvfp4.py`、`shim/autotune.py`、package exports | frontend/runtime 接线、非共享 GIN session、lean off entry、tuning 拒绝 | thunk 的 session/device/stream 归属 |
| `src/src/`、`src/common/`、`src/moe_nvfp4_swapab/` | 17 个 device 文件的针对性 import/merge | wire ABI、generation、barrier 和 FI 既有差异；见 VENDOR 的 targeted-import 记录 |
| `tests/cpu/moe_ep/`、`tests/moe_ep/test_nvfp4_cutedsl_gin_*.py` | host contracts、真实 N2R2 公共 API/强 oracle、off finalizer 回归 | 拓扑 guard、全 rank 结果和无效 route |
| `examples/nvfp4_gin.py`、`benchmarks/moe_ep/bench_legacy_nvfp4_gin.py` | 可执行示例、基线断言与可复现计时 | 区分 raw/full；逐 rank 输出 payload 与结果校验 |
| 本文、package `VENDOR.md` | 设计、支持范围、验证与 targeted-import 来源 | 不将整个 vendor snapshot 改标为 donor |

原 vendor snapshot 为 `fa662f345980f75776ae0c85d2e5c188918ded26`。
此次 donor 固定于 `f863d7540614bd546a42867c21adbf51d573061a`，接收基线
F_BASE 固定于 `e74a1677ab95a1b97aa1c6b6a80b3cf5c0ebac34`。
这不是整包替换：FI 的 sparse decode、masked reduction、per-token scale、custom
activation、singleton TMA 以及 BF16/MXFP8 兼容行为保留。VENDOR 列出针对性 import 的文件与待上游修复；后续同步仍需保留这些本地修改。

## 4. 正确性验证与可复现性

| 验证层 | 已完成结果 |
| --- | --- |
| Host contracts | 25 项：arena/wire、LSA offsets、集体错误、控制组复用/PG 重建、session/workspace 构建清理与清理失败升级 |
| 固定 N2R2 强 oracle | 每 rank 24 次 exact protocol 调用，全作业合计 96 项 transport/L1 byte 检查；包含 partial/empty rank、sparse holes 与 NaN invalid scores |
| 独立 math | 原容差不变；最大 relative L2 为 `0.0026485608`，阈值 `0.02` |
| 公共/强测试 | 每 rank 40 项通过，无 skips；包含原38项及relay配置限制、缓存非空时新权重绑定的capture-ready拒绝/恢复 |
| NCCL-only 默认组 | 每 rank 21 项通过，包含新增review用例；确认只在capture外创建一次CPU/Gloo控制组并复用 |
| 当前 off 兼容性 | 每 rank 10 项通过：原6项、2项FP4环境变量数值/PTX检查、2项禁用GIN入口的off eager/Graph隔离检查（默认及残留GIN拓扑参数） |

下列GPU验收source为`a0ff9eb5e`。GB200任务`3595270`（N2R2，`lyris0120/0129`）
COMPLETED，exit0：先通过2项新review用例/rank，再通过40项strong/public及21项NCCL-only/rank。
任务`3595271`（N1R4，`lyris0131`）COMPLETED，exit0：10项off检查/rank通过。
两次运行源码相同，100个Python文件哈希与该次验收source一致。默认off测试将GIN预检和session
入口替换为调用即失败，仍完成eager/Graph；原FP4环境变量两种分支均通过16种code的bitwise
归约和PTX指令检查。非GIN reducer的5个核心方法AST与固定FI基线一致。
本轮原始文件只调整GIN构造限制及撤回无效接口，未改非GIN转换/归约算法；不新增性能结论。

后续session初始化清理修复通过25项CPU契约：初始化本地步骤共同拒绝后，所有rank
清理并再次协商；任一清理失败或清理协商失败均升级为`GinFatalError`。新增3项测试
调用实际factory，以host doubles注入LSA mapping、清理和控制面失败；修复前均失败，
修复后均通过。本次未重跑CUDA/NCCL多节点测试，上述GPU记录保留原source。

此前错误协议修复source为`6a32476ae`，job`3594770`在GB200 `lyris0144/0238`
上COMPLETED，exit0：先独立通过14项新增错误路径/rank，再通过当时38+19+6项/rank。
分配失败通过单rank注入`torch.OutOfMemoryError`模拟，没有实际耗尽显存；检查view、routes、
output和最后epilogue四处失败后的window/device-comm/comm/root/views清理与session重建。
清理失败的全rank致命升级由CPU contract覆盖，不注入破坏性NCCL设备故障。
该次99个Python源码哈希与运行一致，未改device文件；5节保留各历史source/计时范围。

此前 Graph runtime source 为 `3bb3a307b`，job `3578913` 在两台 GB200/SM100
上 COMPLETED，exit0（CUDA13.3、Torch2.13dev、DSL4.6.2、NCCL2.30.7、driver580.196）。
每种 Graph case 比较三份 Graph、12组变化输入，共36次 replay/rank；覆盖实际
capture stream、输出污染、上游 clone 和下游 GPU 运算，profiler 断言 mega/reducer
各一次。预初始化后的裸 compiled-call 实验 `3578784` 也通过每 rank16次 replay；
它使用旧接收worktree，其17个device文件与当时接收版本`8cea9b3d2`一致；
不代表原生donor验收，也不表示与当前构造/接口调整后的源码逐字一致。

后续 `19bd994fd`/`613421079` 仅增强测试：再连续排队12次 replay，并在每次
replay 前污染 staged activation/SF/routes/weights 和输出，排除复用 eager 遗留输入的
假通过。用预分配快照在一次最终同步后逐次比较，runtime 字节未变。最终补充 job
`3579044` 在两台 GB200 上 COMPLETED，exit0：4个增强 Graph case/rank
和5项 NCCL-only 检查/rank通过。每个 case 共48次精确 replay 比较，其中12次
连续排队，无每轮主机同步/控制通信/分配；仅在末尾统一同步并校验预分配快照。
更早 EP1/2/EP4、BF16/MXFP8 的全矩阵为历史证据；本轮只重新运行上表声明的范围。
每 rank24次 protocol 调用包含在 matrix/balanced 两个 oracle case 中（各12次），
93个非空批次的独立 math 最大 relative L2 为0.0026485608，阈值保持0.02。
中间任务 `3578798` 虽通过24+5项测试，但因 launcher 重复销毁 PG 而 FAILED，
不能算整轮验收通过；修复 launcher 后的 `3578913` 才是完整成功任务。
普通单机四 rank CI 不能替代两个真实 domain；跳过拓扑测试不计为通过。

Host contracts 不依赖 Torch/GPU；不能代替真实 CUDA/NCCL 结果：

```bash
python -m unittest discover -s tests/cpu/moe_ep -p test_legacy_nvfp4_gin_contracts.py -v
```

在两个真实独立 domain 上，使用前述 torchrun geometry 和下面两个测试文件执行
strong/public 验证；普通单机4rank不等价：

```bash
torchrun --nnodes=2 --nproc-per-node=2 --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29500 -m pytest \
  tests/moe_ep/test_nvfp4_cutedsl_gin_oracles.py \
  tests/moe_ep/test_nvfp4_cutedsl_gin_multinode.py -v -x
```

## 5. Blackwell 性能

### 5.1 基线和计时范围

GPU 为 GB300（SM103，152 SM/device），Torch `2.13.0a0+9186a08b2c.nv26.07`、
CUDA `13.3`、CuTe DSL `4.6.2`、nccl4py `0.3.1`、NCCL `2.30.7`、driver `580.173.10`。
本节 5.1–5.2 为此前 eager 性能测量；新增 Graph 对比见 5.3。
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

Off shape 为 H2048、公共 I1024、E8、K4。下表为 median µs，均为 F_BASE legacy → receiver；
EP1/2/4 × T64/1024 × warm/cold × raw/full 的完整 **24 项门槛均通过**。每项包含两个 source 各100个样本，合计4800样本。

| EP | T/rank | Warm raw | Cold raw | Warm full | Cold full |
| --- | --- | --- | --- | --- | --- |
| 1 | 64 | 91.44 → 87.50 | 95.68 → 90.91 | 211.91 → 207.97 | 202.06 → 199.99 |
| 1 | 1024 | 132.40 → 127.07 | 136.98 → 131.42 | 237.33 → 234.87 | 238.56 → 231.43 |
| 2 | 64 | 106.98 → 101.52 | 106.46 → 102.11 | 226.71 → 217.63 | 217.39 → 212.37 |
| 2 | 1024 | 152.82 → 146.02 | 156.82 → 152.78 | 264.48 → 256.95 | 259.99 → 254.90 |
| 4 | 64 | 129.73 → 124.43 | 130.90 → 124.03 | 249.62 → 241.95 | 246.21 → 240.35 |
| 4 | 1024 | 184.62 → 181.89 | 186.27 → 180.62 | 299.97 → 293.81 | 296.12 → 287.70 |

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

性能入口为 [bench_legacy_nvfp4_gin.py](../../benchmarks/moe_ep/bench_legacy_nvfp4_gin.py)，
使用相同 fixture、独立进程和指定 source checkout 运行，输出逐 rank 的样本和 manifest：

- Off：分别从固定 F_BASE 与 receiver 导入 FlashInfer，使用 `--mode off --impl base/port`，
  对 EP1/2/4 执行五轮；脚本覆盖 T64/1024。双方必须共享 `--root` fixture。
- On：先在 receiver 的 N2R2 launcher 中用 `--mode on --impl port --prepare` 生成 fixture；
  再分别运行 `--impl port` 与 `--impl donor`，donor 进程的 `PYTHONPATH` 指向固定 donor
  checkout，避免导入 FI；每次使用同一 `--root` 和不同 `--round`。
- 使用 `--skip-profiles` 将 primary samples 与 profiler 分开；全部 primary 完成后以
  `--profile-only` 采集。baseline/receiver 轮次顺序交替，warm/cold 不能混合汇总。

benchmark 的独立进程协议与最终 resident experiment 的 controller 不相同；
最终 resident 数据、conditioned protocol、旧负面结果和失败记录随调查归档保留，
不把未运行过的新协议结果代入本文数字。

### 5.3 CUDA Graph：按相同 GPU 工作量对比

此前 job `3593453` 的公共full完成wall，warm BF16为`1505.30 → 308.11 µs`、
packed为`1803.99 → 359.94 µs`。这组4.9–5.0×是**带逐次校验的公共入口与replay**
的差异，不能作为相同GPU工作量的kernel加速比。Eager含4个路由检查GPU kernel、
1次`.item()`同步、2次CPU控制协商，并默认申请owned输出；replay不重复这些工作。
历史native donor约350µs属于prepared raw范围，不能与上述1.5–1.8ms公共full直接比较。

为对齐口径，job `3593508` 在两台GB200/SM100（`lyris0057/0126`）补测，
COMPLETED，exit0。源码`200576d1c`与前一验收的runtime相同；CUDA13.3、Torch2.13dev、
DSL4.6.2、NCCL2.30.7、driver580.196。固定N2R2、T32/rank、H7168、公共I1536、E48、K4，
同allocation、layer、权重、输入和预分配输出；基线为当前legacy eager，不使用next。

- **Raw：**只执行预准备的mega+独立reduce，双方均不含staging/公共校验/输出copy。
- **Prepared GPU：**双方均执行相同staging+mega+reduce+输出copy，输入值在计时前验证；
  这是计时adapter，不更改公共eager API的逐次校验契约。
- **Checked public：**`forward`逐次校验仍保留，仅预分配输出，用来定位公共入口成本。

下表为warm cache连续提交32次、末尾统一同步后的每次调用完成wall中位数（µs）。
3轮交替执行模式，每轮2个warmup block和12个采样block，共36样本/模式；
逐样本取最慢rank，再汇总。初始化、编译、capture和入口CPU/Gloo barrier在计时外。

| 输入 | 对齐范围 | Eager µs | Graph µs | Eager / Graph |
| --- | --- | ---: | ---: | ---: |
| BF16 | Raw mega+reduce | 231.26 | 231.63 | 0.998× |
| Packed | Raw mega+reduce | 222.75 | 222.28 | 1.002× |
| BF16 | Prepared GPU含staging/copy | 239.39 | 234.08 | 1.023× |
| Packed | Prepared GPU含staging/copy | 241.62 | 237.30 | 1.018× |

同allocation的预分配checked public为BF16`1511.28 → 234.12 µs`、
packed`1523.54 → 236.01 µs`。预分配输出后公共入口差距仍然存在；不能把大比值归因于
输出分配，也不把它当作mega算术加速。本次raw连续提交中位数基本相同；
prepared GPU差约2%，不宣称稳定的kernel算术加速或通过native donor的5%迁移门槛。

另外测量单次提交，3轮×60样本/模式，每轮10次warmup：raw完成wall分别为
BF16`397.95 → 327.07 µs`、packed`388.11 → 297.95 µs`；prepared GPU分别为
`470.63 → 342.63 µs`和`489.88 → 359.12 µs`。这些区间仍包含host提交空隙和跨rank等待，
不是单个kernel时间；不能把单次与32次连续提交的数字混在一起计算加速比。
原始单次长尾、逐rank样本、CV/P95及各轮中位数全部保留。

每个rank的trace确认prepared eager/replay与公共replay的**计算kernel序列相同**，
raw双方均为1个mega和1个reduce；BF16输出D2D copy在Graph中可表现为`memcpy32_post`，
因此不能只按profiler的kernel数量判断是否缺少copy。Packed双方均有5个D2D copy。
Profiler启动后先做CPU barrier，再提交被测操作；profile与主要计时分开。
所有2592条样本/rank完整对齐，污染staged输入/输出、所有路径和计时block的结果均bitwise通过。
原job `3593453` 同allocation每rank24项strong/public测试也全部通过；93个非空math批次
最大relative L2为0.0026485608，原阈值0.02未改。实验adapter、完整trace及原始数据保留在
独立实验目录；该次口径审计仅修正文档。随后host错误协议修复的验收见4节，device代码未改。

## 6. 合并评审边界

本 PR 的可评审交付是默认关闭、固定范围的 legacy GIN 集成及上述验证结果。
扩大 shape/topology、去掉 eager 公共校验、改变 reduce 精度/顺序、
kernel autotuning 和正式 internode 性能 promotion 都需要后续工作。
当前 PR 不以 baseline 波动为理由删除不利样本，也不将 correctness 通过等同于性能通过。

CUDA/PyTorch 由调用方管理 Graph；FI 提供可被捕获的 staging/launch 接口；
GIN kernel 在设备端执行通信、计算和轮次推进，不自行创建 Graph。

| 阶段 | CPU／Gloo | GPU |
| --- | --- | --- |
| Warmup（每个 session/workspace/权重准备状态） | 确认拓扑、初始化/注册、编译、准备参数、检查当前 route 值 | 初始化并执行一次 eager |
| Capture（每份 Graph） | 执行 metadata、phase/session 和 capture-mode 的 CPU/Gloo 协商 | 录制 staging、mega/reduce 及相邻 GPU 操作 |
| Replay（每批） | 不调用 FI Python、CPU/Gloo 预检 | 执行录制好的操作，更新设备通信状态 |

输入或输出地址、shape/bucket 变化后需要使用相应输入重新 warmup/capture；
不能让旧 Graph 自动改用新地址。只更新固定 buffer 内的数据不需要重新确认拓扑。
初始化和 warmup 在 capture 外执行，并在开始 capture 前完成设备同步。

使用现有 `MoEEpLayer`，由调用方通过 PyTorch 发起 capture：

```python
static = MoEEpTensors(hidden_states=x, topk_ids=ids, topk_weights=weights)
layer.warmup(static)  # 所有 rank：初始化、注册、编译和 launch 参数准备
# 当前输入的 route 值需要合法；也可在 Graph 外调用 stage_inputs 做 eager 校验。
torch.cuda.synchronize()
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    y = layer.forward(static)  # 同时录制量化/staging、GIN mega、TopkReduce、output copy
# 所有 rank 完成 capture 后，按相同顺序 replay。
dist.barrier(group=dist.group.WORLD)
graph.replay()
```

Capture 时执行 CPU metadata/phase/session 校验，但不读取 GPU route 值到 CPU；
replay 不重跑捕获时的 Python forward，也不重新执行 CPU 预检。每个 rank 的 shape/地址在一份 Graph
内固定；可以原位更新 activation、SF、route IDs/weights。调用方必须保证每次输入
的 route ID 都在 `-1..47`，不能期待 replay 通过 CPU 抛出非法 route 值错误，
也不能依赖设备端安全拒绝越界输入；违反此条件的执行不在支持范围内。
输入更新与 replay 使用同一 stream，或由调用方使用 event/wait_stream 建立先后关系。
同一 session 的多份 Graph 与 eager 可以串行交替使用，不能并发复用其 workspace；
必须在下一次复用前消费输出，或将结果复制到独立存储。
同一 workspace 的 Graph/eager 执行及共享输出的读取应使用同一 stream，或用
event/wait_stream 保证前次执行与输出读取完成后再复用；不同 stream 上仅有 Python
提交顺序并不构成设备执行顺序。
这里的参与者是同一 GIN session 的全部默认 PG ranks；capture 的调用顺序/次数也应一致。
所有参与 rank 必须按相同顺序、次数 replay，即使本 rank 的 live token 为零。
固定 shape 为零的 rank 已有独立 bucket 测试；一份 Graph 内不改变实际 tensor shape。

layer 持有准备后的量化权重；本次不承诺 Graph 中动态替换权重，修改原始 BF16
weight pack 也不会自动更新已准备权重。必须让 layer、workspace/session、准备后的
权重和 Graph 输入/输出保持有效，直至 Graph 不再使用且所有在途 replay 已完成；
销毁 layer 会使捕获的设备指针失效。初始化、编译、参数重建和资源
销毁仍在 Graph 外进行。metadata、未 warmup 或混合模式的 CPU 共同拒绝可修正后重新 capture；
部分 Graph 应丢弃。设备/NCCL 错误仍按作业致命错误处理。

低层调用方可在 capture 外集体调用 `nvfp4_mega_launch_thunk(w1, w2, workspace)`
准备 launcher，`w1/w2` 为 kernel-ready 的 `(weight, scale)` 元组；
再在目标 stream 的 capture 中调用它，GIN thunk 绑定当时的当前 stream。
参数、权重或 workspace 生命周期改变后重新准备，不在 capture 中编译或注册。
正式集成优先使用上述 `MoEEpLayer`；不扩展普通低层 `nvfp4_mega_moe` 的 capture 契约。


此前迁移与 eager 性能的调查归档位于内部 `qiangf/agent-trace` 仓库，commit
`2cf0a06da12654a3ab4089224523822f93de9807`，路径
`cases/legacy-nvfp4-gin-blackwell-01a19f6e-r6`。它保留调查记录、报告和证据摘要；
大型 tensor、profiler trace 和二进制仍保留在原始实验位置。该冻结归档不包含本轮
新增 Graph 验证；本轮以本文 job IDs、对应测试源码及本地实验日志为准。
归档不是运行本 PR 的前提。
