# VSA (Video Sparse Attention) 集成说明

> 分支:`hsr/vsa_integrate`
> 相关 commit:`6748365b` "add vsa to block sparse api"(新增约 3,979 行,6 个文件)

## 核心实现 — CuTe DSL 稀疏注意力(新增模块)

`flashinfer/cute_dsl/sparse/`

| 文件 | 行数 | 说明 |
|------|------|------|
| `__init__.py` | 114 | 公开 API:`dsl_block_sparse_attn_forward()` |
| `fwd.py` | 2,497 | 主 kernel:`VideoSparseAttentionForwardGroup2QInterleaveKV`,Q/KV 交错的 Blackwell SM100a kernel |
| `scheduler.py` | 209 | 稀疏 block 的静态 persistent 调度器 |
| `ptx.py` | 28 | 内联 PTX 辅助函数 |

## Python API 集成

`flashinfer/sparse.py` — 通过 `BlockSparseAttentionWrapper` 暴露,`backend="vsa_blackwell"`

- `_bsr_to_vsa_index()` (L43) — BSR 矩阵 → VSA 索引格式
- `_block_mask_to_vsa_index()` (L88) — per-head 布尔 block mask → VSA 索引
- `plan()` (L391-465) — VSA 后端校验与规划
- `run()` (L722-760) — VSA 运行时执行路径

## 测试

`tests/attention/test_vsa_block_sparse.py` (663 行) — 完整测试套件:

- `test_vsa_accuracy` / `test_vsa_accuracy_vs_dense` — 精度验证
- `test_vsa_return_lse` / `test_vsa_preallocated_lse` — LSE 处理
- `test_vsa_preallocated_output` — 输出张量复用
- `test_vsa_sm_scale` — softmax scale 校验
- `test_vsa_per_head_mask_correctness` / `test_vsa_per_head_mask_differs_across_heads` — per-head 掩码
- `test_vsa_vs_flashinfer_default_backend` / `test_vsa_vs_auto_80k` — 跨后端对比
- `test_vsa_performance_vs_dense` — 性能基准

## VSA Blackwell 后端约束

- 固定 block size R=C=64,head_dim=128
- 硬件要求:sm100a(Blackwell GPU)
- 不支持:GQA、causal mask、per-element block mask、position encoding、logits soft cap

## 两种输入模式

1. **BSR 格式**(`indptr`/`indices`):head-independent 稀疏模式,经 `_bsr_to_vsa_index()`
2. **Block Mask 格式**(`block_mask`):per-head 布尔掩码,经 `_block_mask_to_vsa_index()`
