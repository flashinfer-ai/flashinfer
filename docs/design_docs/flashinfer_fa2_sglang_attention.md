# FlashInfer FA2 Attention API 与 SGLang 接入方式

本文梳理 FlashInfer attention 的 FA2 backend API，重点讨论 paged KV cache
与 ragged KV tensor 两种 batch prefill 形式，并说明 SGLang 的 FlashInfer
attention backend 如何选择它们、如何构造输入，以及 SGLang 的 token pool
布局如何与 FlashInfer API 对齐。

调研基于以下源码快照：

- FlashInfer: `17bdd718fc56d6d3e3c194372e8c75737462eb4b`
- SGLang (`../sglang`): `07c3048d8a8c350198e37aefc4fc427d9075c052`

## 1. 结论摘要

1. FlashInfer 的 FA2 batch prefill module 同时导出 `ragged_run` 和
   `paged_run`。Python 层分别用
   `BatchPrefillWithRaggedKVCacheWrapper` 和
   `BatchPrefillWithPagedKVCacheWrapper` 管理它们。两种 wrapper 都采用
   `plan(...)` + `run(...)` 两阶段 API：batch 的序列结构在 `plan` 阶段
   给出，逐层变化的 Q/K/V 或 KV cache 在 `run` 阶段给出。
2. ragged 输入是 packed contiguous tensor：Q、K、V 的 token 维分别由
   `qo_indptr`、`kv_indptr` 分段。paged 输入则是物理 KV page 数组加
   CSR 风格 page table：`paged_kv_indptr`、`paged_kv_indices` 和
   `paged_kv_last_page_len`。
3. SGLang 的 paged prefill 和 decode wrapper 显式指定
   `backend="fa2"`。SGLang 的 ragged wrapper 没有固定为 FA2：一般传
   `backend="auto"`，在部分 SM100 配置下传 `"cutlass"`。因此：
   - SGLang 的 paged FlashInfer 路径确定使用 FA2；
   - SGLang 的 ragged 路径只有在 FlashInfer `auto` 最终选中 FA2 时才
     使用 FA2。例如 Ampere/Ada 上通常是 FA2，而 Hopper 上满足条件时
     `auto` 会选 FA3。
4. SGLang 普通 extend 默认不是在 paged 和 ragged 中二选一。存在缓存
   前缀时，它把 attention 拆成：
   - ragged：当前 extend token 之间的 causal attention；
   - paged：当前 Q 对缓存前缀的 non-causal attention；
   - 最后用两部分的 output 和 LSE 做数值稳定的 `merge_state`。
   没有缓存前缀时只需 ragged。显式选择 paged 时，SGLang 先把当前
   K/V 写入 token pool，再用一个 causal paged prefill 覆盖完整
   `prefix + extend`。
5. SGLang 调 FlashInfer paged API 时固定传 `page_size=1`。其
   `req_to_token` 中的 token slot id 直接成为 FlashInfer 的 page id，
   `kv_indptr` 因而既是 token indptr 也是 page indptr，
   `kv_last_page_len` 恒为 1。SGLang 的 3-D KV buffer
   `[num_slots, Hkv, D]` 会被 FlashInfer 自动扩成
   `[num_pages, 1, Hkv, D]`，无需复制 KV 数据。

## 2. FlashInfer FA2 API

### 2.1 两阶段 wrapper API

三个相关的公共 wrapper 是：

| 场景 | Wrapper | 计划阶段 | 执行阶段 |
| --- | --- | --- | --- |
| batch prefill，paged KV | `BatchPrefillWithPagedKVCacheWrapper` | `plan(qo_indptr, paged_kv_indptr, paged_kv_indices, paged_kv_last_page_len, ...)` | `run(q, paged_kv_cache, ...)` |
| batch prefill，ragged KV | `BatchPrefillWithRaggedKVCacheWrapper` | `plan(qo_indptr, kv_indptr, ...)` | `run(q, k, v, ...)` |
| batch decode，paged KV | `BatchDecodeWithPagedKVCacheWrapper` | `plan(indptr, indices, last_page_len, ...)` | `run(q, paged_kv_cache, ...)` |

`plan` 根据 batch size、各请求 Q/KV 长度、head 数、head dim、page size、
mask/position encoding 等信息生成并缓存调度所需的辅助数据。
`run` 可以在不同 Transformer layer 上复用同一个 plan，只替换该层的
Q/K/V 数据。FA2 JIT specialization 还包含 Q/KV/output dtype、index
dtype、head dim、position encoding、是否 sliding window、是否 soft
cap、是否使用 FP16 QK reduction 等参数。

FlashInfer 仍保留了旧名称：

```python
begin_forward = plan
```

wrapper 的 `forward(...)` 则是一个 deprecated 兼容入口，它更新少量
runtime 配置后调用 `run(...)`。当前 SGLang 使用的正是
`begin_forward(...)` 和 `forward(...)` 这组旧名称；其语义仍分别等于
plan 和 run。

FA2 的 JIT module 不是为 paged 和 ragged 分别编译两个完全独立的
Python module。`get_batch_prefill_module(...)` 从同一个 module 中取出
`plan`、`ragged_run` 和 `paged_run`；FA2 codegen 同时实例化 paged 与
ragged kernel。

### 2.2 Paged prefill

核心 API 形式：

```python
wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    backend="fa2",
)

wrapper.plan(
    qo_indptr,                 # [B + 1]
    paged_kv_indptr,           # [B + 1]，按 page 计数
    paged_kv_indices,          # [paged_kv_indptr[-1]]
    paged_kv_last_page_len,    # [B]
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    page_size,
    head_dim_vo=None,
    causal=False,
    custom_mask=None,
    pos_encoding_mode="NONE",
    window_left=-1,
    logits_soft_cap=None,
    q_data_type=torch.float16,
    kv_data_type=None,
)

o = wrapper.run(q, (k_cache, v_cache))
# 或
o, lse = wrapper.run(q, (k_cache, v_cache), return_lse=True)
```

其中：

- `q` 的 NHD 形状是 `[qo_indptr[-1], Hq, Dqk]`。
- `qo_indptr[i+1] - qo_indptr[i]` 是请求 `i` 的 Q token 数。
- `paged_kv_indptr[i+1] - paged_kv_indptr[i]` 是请求 `i` 使用的 page 数，
  不是 KV token 数。
- 请求 `i` 的有效 KV 长度为：

  ```text
  (num_pages_i - 1) * page_size + paged_kv_last_page_len[i]
  ```

- NHD 下，分离的 K/V cache 形状分别是：
  `[max_num_pages, page_size, Hkv, Dqk]` 和
  `[max_num_pages, page_size, Hkv, Dvo]`。
- 也可传一个合并 tensor
  `[max_num_pages, 2, page_size, Hkv, D]`，其中下标 0/1 分别为 K/V。
- `paged_kv_indices[paged_kv_indptr[i]:paged_kv_indptr[i+1]]`
  给出请求 `i` 的逻辑 KV 序列依次映射到哪些物理 page。
- output 形状为 `[qo_indptr[-1], Hq, Dvo]`；LSE 形状为
  `[qo_indptr[-1], Hq]`。

当 page size 为 1 时，FlashInfer 允许分离 K/V 直接使用 3-D tensor
`[max_num_pages, Hkv, D]`；`_unpack_paged_kv_cache` 会用 `unsqueeze`
创建 `[max_num_pages, 1, Hkv, D]` view。

### 2.3 Ragged prefill

核心 API 形式：

```python
wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    backend="fa2",
)

wrapper.plan(
    qo_indptr,       # [B + 1]
    kv_indptr,       # [B + 1]，按 token 计数
    num_qo_heads,
    num_kv_heads,
    head_dim_qk,
    head_dim_vo=None,
    causal=False,
    custom_mask=None,
    pos_encoding_mode="NONE",
    window_left=-1,
    logits_soft_cap=None,
    q_data_type=torch.float16,
    kv_data_type=None,
)

o = wrapper.run(q, k, v)
# 或
o, lse = wrapper.run(q, k, v, return_lse=True)
```

ragged 的 “ragged” 指 batch 中每个请求的 Q/KV 长度可以不同，但 Q、K、V
本身仍是沿 token 维拼接的连续 packed tensor：

- `q`: `[qo_indptr[-1], Hq, Dqk]`
- `k`: `[kv_indptr[-1], Hkv, Dqk]`
- `v`: `[kv_indptr[-1], Hkv, Dvo]`
- output: `[qo_indptr[-1], Hq, Dvo]`
- LSE: `[qo_indptr[-1], Hq]`

请求 `i` 的切片为：

```python
q_i = q[qo_indptr[i] : qo_indptr[i + 1]]
k_i = k[kv_indptr[i] : kv_indptr[i + 1]]
v_i = v[kv_indptr[i] : kv_indptr[i + 1]]
```

与 paged 相比，ragged 不需要 `indices` 和 `last_page_len`，因为物理 token
顺序已经就是逻辑序列顺序。`qo_indptr` 和 `kv_indptr` 可以不同；当
Q/K/V 是同一批新 token 时，两者通常相同。

### 2.4 Paged decode

decode 只有 paged wrapper：

```python
wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
    workspace,
    kv_layout="NHD",
    backend="fa2",
    use_tensor_cores=...,
)

wrapper.plan(
    indptr,          # [B + 1]
    indices,         # [indptr[-1]]
    last_page_len,   # [B]
    num_qo_heads,
    num_kv_heads,
    head_dim,
    page_size,
    q_data_type=...,
    kv_data_type=...,
)

o = wrapper.run(q, (k_cache, v_cache))
```

通常 `q` 为 `[B, Hq, D]`，即每个请求一个 decode token。FA2 tensor-core
decode 也支持在 plan 中指定 `q_len_per_req > 1`，此时 Q 的 token 维为
`B * q_len_per_req`。

### 2.5 Mask 和 causal 对齐

两种 prefill wrapper 的 causal 对齐规则相同。对请求 `i`，设 Q 长度为
`Lq`、KV 长度为 `Lkv`，causal mask 的对角线偏移是 `Lkv - Lq`。因此
append 场景中 Q 可以只包含序列尾部的新 token，而 KV 可以包含
`prefix + new tokens`。

也可在 plan 中传 flattened custom mask。其总元素数为：

```text
sum(Lq[i] * Lkv[i] for i in range(B))
```

FlashInfer 会为每个请求分段 pack bits。若同时提供 custom mask 和
`causal=True`，custom mask 优先。

## 3. SGLang 如何使用这些 API

### 3.1 Wrapper 初始化和实际 backend

`FlashInferAttnBackend` 初始化时固定：

```python
self.prefill_backend = "fa2"
self.decode_backend = "fa2"
```

并创建：

```python
BatchPrefillWithPagedKVCacheWrapper(..., "NHD", backend="fa2")
BatchDecodeWithPagedKVCacheWrapper(..., "NHD", backend="fa2")
```

ragged wrapper 单独处理：

```python
fmha_backend = "auto"
if is_sm100_supported() and not tc_piecewise_cuda_graph:
    fmha_backend = "cutlass"

BatchPrefillWithRaggedKVCacheWrapper(
    workspace, "NHD", backend=fmha_backend
)
```

所以不能把 SGLang 中所有 `prefill_wrapper_ragged` 调用都称作 FA2。
FlashInfer 的 `auto` 在支持 FA3 的 Hopper 配置上会优先选 FA3，否则
回退 FA2；指定 `"cutlass"` 时也显然不是 FA2。

### 3.2 什么时候用 paged，什么时候用 ragged

SGLang 的高层选择可以概括如下：

| SGLang 场景 | 使用形式 |
| --- | --- |
| decode / idle | paged decode |
| speculative target verify | paged prefill |
| 普通 extend，默认配置 | ragged 新 token；若有 prefix，再加 paged prefix 并 merge |
| 普通 extend，`SGLANG_FLASHINFER_USE_PAGED=true` | 纯 paged prefill |
| deterministic inference | 纯 paged prefill |
| TC piecewise CUDA graph | 纯 paged prefill |
| multimodal 或 multi-item scoring | 纯 paged prefill |
| full prefill CUDA graph 的普通 extend | 纯 paged prefill |
| draft extend v2 CUDA graph | 纯 paged prefill |

环境变量 `SGLANG_FLASHINFER_USE_PAGED` 默认是 `False`。普通非 multimodal、
非 multi-item extend 的选择条件是：

```python
use_ragged = (
    not enable_deterministic
    and not is_in_tc_piecewise_cuda_graph()
    and not SGLANG_FLASHINFER_USE_PAGED
)
```

这里的 `use_ragged=True` 实际表示允许 hybrid ragged/paged 路径：

```text
                         ┌─ 当前 extend K/V ─ ragged causal ─ (o1, lse1)
Q（当前 extend tokens） ─┤
                         └─ 缓存 prefix ───── paged non-causal ─ (o2, lse2)

                         (o1, lse1), (o2, lse2)
                                      │
                                  merge_state
                                      │
                                      o
```

- `prefix_len == 0`：只执行 ragged 分支。
- `prefix_len > 0`：执行两支并用 `_safe_merge_state` 合并。
- pure paged：先把当前 K/V 写入 KV pool，再对完整 KV 序列执行一次
  causal paged prefill。

这种拆分避免了为了做 chunk prefill/cache reuse 而把缓存 prefix
重新 materialize 成 contiguous K/V，同时让新 token 间的 K/V 保持
连续输入。

### 3.3 SGLang 输入到 FlashInfer paged 输入的映射

SGLang 为每个 request 保存一张：

```text
req_to_token[request_pool_id, logical_token_position] = physical_token_slot
```

`create_flashinfer_kv_indices_triton` 按 request 取出指定逻辑区间，将
physical slot id 拼成 FlashInfer 的 `paged_kv_indices`。

设：

```text
S_i = seq_lens[i]                 # prefix + 本次 extend
P_i = prefix_lens[i]
E_i = S_i - P_i                   # 本次 extend 长度
```

普通 extend 的 metadata 映射如下：

| FlashInfer 参数 | SGLang pure paged | SGLang hybrid 的 paged prefix |
| --- | --- | --- |
| `qo_indptr` | `exclusive_cumsum(E_i)` | `exclusive_cumsum(E_i)` |
| `paged_kv_indptr` | `exclusive_cumsum(S_i)` | `exclusive_cumsum(P_i)` |
| `paged_kv_indices` | 拼接 `req_to_token[req_i, 0:S_i]` | 拼接 `req_to_token[req_i, 0:P_i]` |
| `paged_kv_last_page_len` | 全 1 | 全 1 |
| `page_size` | 1 | 1 |
| mask | causal | non-causal（prefix 对所有当前 Q 可见） |

decode 的映射同理，只是没有 `qo_indptr`：

- `indptr = exclusive_cumsum(seq_lens)`；
- `indices` 是各请求 `[0:seq_len]` 的 physical slot id 拼接；
- `last_page_len` 全 1；
- `page_size=1`；
- Q 通常为 `[B, Hq, D]`。

SGLang 的默认 NHD token pool 每层返回：

```python
k_cache: [num_slots, Hkv, Dqk]
v_cache: [num_slots, Hkv, Dvo]
```

传入 FlashInfer 后，由 page-size-1 兼容逻辑扩为：

```python
k_cache: [num_slots, 1, Hkv, Dqk]
v_cache: [num_slots, 1, Hkv, Dvo]
```

因此一个 SGLang physical token slot 正好对应一个 FlashInfer physical
page。`paged_kv_indices` 中无需做除法或 page 内 offset 计算，slot id
就是 page id。

### 3.4 SGLang 输入到 ragged 输入的映射

SGLang 的 model forward 已将本批 extend token 打包在 token 维：

```python
q: [sum(E_i), Hq, D]
k: [sum(E_i), Hkv, D]
v: [sum(E_i), Hkv, D]
```

metadata updater 构造：

```python
qo_indptr = exclusive_cumsum(E_i)
kv_indptr = qo_indptr
```

实际调用等价于：

```python
ragged_wrapper.plan(
    qo_indptr,
    qo_indptr,
    num_qo_heads,
    num_kv_heads,
    head_dim,
    q_data_type=...,
)

o1, lse1 = ragged_wrapper.run(
    q.view(-1, Hq, D),
    k.view(-1, Hkv, D),
    v.view(-1, Hkv, D),
    return_lse=True,
)
```

因为 ragged K/V 只包含当前 extend 段，所以该分支使用 causal mask。
缓存 prefix 由另一 paged 分支读取，并使用相同的 `qo_indptr`，从而两个
分支产生完全相同的 output/LSE shape，可以直接 `merge_state`。

如果不存在缓存 prefix，SGLang 直接使用 ragged output，不请求 LSE，
然后把当前 K/V 写入 token pool。如果存在 prefix，则在两支计算与
merge 完成后写入当前 K/V；因此 paged prefix 分支不会意外读到当前
extend token。

### 3.5 plan/run 生命周期在 SGLang 中的对应

每次 forward batch 开始时，indices updater：

1. 根据 `seq_lens`、`prefix_lens`、`req_to_token` 生成 indptr 和 indices；
2. 调 wrapper 的 `begin_forward(...)`，即 FlashInfer `plan(...)`；
3. 将 plan 后的 wrapper 放入 `forward_metadata`。

随后每一层 attention：

1. 取得该层的 Q/K/V 或 KV pool view；
2. 调相同 wrapper 的 `forward(...)`，即 FlashInfer `run(...)`；
3. 跨层复用 batch metadata 和 workspace。

CUDA graph 场景会在 wrapper 构造时提供固定的 indptr、indices 和
last-page-len buffer。部分路径还用 SGLang 的 `fast_prefill_plan` /
`fast_decode_plan` 替代标准 `begin_forward`，绕过标准 plan 中的
device-to-host metadata copy；其最终传入 FA2 module 的问题描述与普通
plan 保持一致。

## 4. 容易混淆的点

### 4.1 “paged” 不等于 SGLang 的 allocator page size

这条 SGLang FlashInfer MHA 路径固定向 wrapper 传 `page_size=1`。
FlashInfer page table 实际是 token-slot table。即使 SGLang 其他内存
管理组件存在更大的 allocator page granularity，也不能据此认为这里
传给 FA2 的 `page_size` 是同一个值。

### 4.2 `use_ragged=True` 通常仍会 plan paged wrapper

SGLang 的 ragged 路径只覆盖当前连续 extend K/V。存在缓存 prefix 时，
paged wrapper 仍然是必需的。代码中的 `use_ragged` 更准确的含义是
“是否把当前 extend 段从完整 paged attention 中拆出来”。

### 4.3 SGLang ragged wrapper 不保证是 FA2

paged prefill 和 decode wrapper 都显式写死 FA2；ragged wrapper 使用
auto/CUTLASS 选择。分析 kernel 性能、抓取 kernel 名称或比较 FA2
paged/ragged 时，需要先确认 `prefill_wrapper_ragged._backend` 的最终
值和目标 GPU。

### 4.4 `qo_indptr` 与 `kv_indptr` 的单位不同于物理地址

- ragged `qo_indptr` / `kv_indptr`：packed tensor 中的 token offset；
- paged `qo_indptr`：packed Q tensor 中的 token offset；
- paged `paged_kv_indptr`：page-index 数组中的 page offset；
- paged `paged_kv_indices`：物理 page id。

SGLang 因为使用 `page_size=1`，paged KV 的 page 数恰好等于 token 数，
这只是该接入方式下的特例。

## 5. 主要源码定位

FlashInfer：

- `flashinfer/prefill.py:450-465`：一个 batch prefill module 同时暴露
  `ragged_run` / `paged_run`。
- `flashinfer/prefill.py:1492-1760`：paged prefill wrapper 构造。
- `flashinfer/prefill.py:2068-2491`：paged prefill `plan`。
- `flashinfer/prefill.py:2560-2662`：paged prefill `run` API 与 tensor shape。
- `flashinfer/prefill.py:2947-3166`：ragged prefill wrapper 构造。
- `flashinfer/prefill.py:3197-3660`：ragged prefill `plan`。
- `flashinfer/prefill.py:3720-3781`：ragged prefill `run` API 与 tensor shape。
- `flashinfer/decode.py:710-788`：paged decode wrapper。
- `flashinfer/decode.py:1239-1325`：paged decode `plan`。
- `flashinfer/decode.py:1810-1912`：paged decode `run`。
- `flashinfer/utils.py:76-109,186-200`：page size 1 时自动扩展 KV cache 维度。
- `flashinfer/utils.py:522-577`：`auto` 的 FA3/FA2 选择。
- `flashinfer/jit/attention/modules.py:962-1032`：FA2 batch prefill
  specialization。
- `flashinfer/jit/attention/modules.py:1650-1696`：同时生成 paged/ragged
  FA2 kernel 实例。

SGLang：

- `../sglang/python/sglang/srt/layers/attention/flashinfer_backend.py:300-313`：
  paged prefill/decode backend 固定为 FA2。
- 同文件 `:430-517`：workspace 和三类 wrapper 初始化。
- 同文件 `:916-1010`：forward mode 与 paged/ragged 选择。
- 同文件 `:1252-1407`：pure paged 与 hybrid ragged/paged forward。
- 同文件 `:1410-1466`：paged decode forward。
- 同文件 `:1642-1742`：decode page table 构造及 `page_size=1` plan。
- 同文件 `:2034-2213`：prefill indptr/page table 构造及两种 plan。
- `../sglang/python/sglang/kernels/ops/kvcache/kv_indices.py:8-44`：
  从 `req_to_token` 生成 FlashInfer `kv_indices`。
- `../sglang/python/sglang/srt/mem_cache/memory_pool.py:1988-1999`：
  默认 NHD KV pool shape。
- `../sglang/python/sglang/srt/environ.py:645-649`：
  `SGLANG_FLASHINFER_USE_PAGED` 默认值。
