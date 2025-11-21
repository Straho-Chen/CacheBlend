# vLLM Decode 机制详解

本文档详细解释 vLLM 在 decode（生成）阶段的工作原理，包括 CUDA Graph 优化、PagedAttention KV Cache 机制等核心概念。

## 目录

1. [CUDA Graph 优化机制](#cuda-graph-优化机制)
2. [为什么 Decode 阶段看不到日志](#为什么-decode-阶段看不到日志)
3. [PagedAttention KV Cache 机制](#pagedattention-kv-cache-机制)
4. [Cache Shape 为什么不变](#cache-shape-为什么不变)
5. [如何验证 Decode 行为](#如何验证-decode-行为)

---

## CUDA Graph 优化机制

### 什么是 CUDA Graph？

CUDA Graph 是 NVIDIA CUDA 提供的一种优化技术，它可以将一系列 GPU 操作（kernel launches）预先录制下来，然后在后续执行时直接回放，避免 Python 解释器的开销。

### vLLM 中的 CUDA Graph 工作流程

#### 阶段 1: Graph 捕获（Capture Phase）

在第一次 decode 时，vLLM 会捕获执行模式：

```python
# 在 model_runner.py 中
with torch.cuda.graph(self._graph, pool=memory_pool):
    hidden_states = self.model(
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        **kwargs,
    )
```

**关键点：**
- 这个阶段会执行一次完整的模型前向传播
- 所有的 GPU kernel launches 被**录制**下来
- Python 代码（包括日志）会正常执行
- 这是你看到 `[Decode Triggered]` 日志的唯一时机

#### 阶段 2: Graph 回放（Replay Phase）

在后续的 decode 步骤中：

```python
# 在 CUDAGraphRunner.forward() 中
self.graph.replay()  # 直接回放 GPU kernels，跳过 Python 代码
```

**关键点：**
- `graph.replay()` 直接执行 GPU kernels
- **完全绕过 Python 解释器**
- 日志语句不会被执行
- 性能大幅提升（避免 Python 开销）

### 执行路径对比

```
正常执行（无 CUDA Graph）:
┌─────────────────────────────────────┐
│ Python 代码执行                     │
│   ↓                                 │
│ logger.info("Decode triggered")     │ ← 每次都会执行
│   ↓                                 │
│ PagedAttention.forward_decode()     │ ← 每次都会执行
│   ↓                                 │
│ GPU Kernels                         │
└─────────────────────────────────────┘

CUDA Graph 执行:
┌─────────────────────────────────────┐
│ 捕获阶段（一次）:                  │
│   logger.info() ← 执行并记录        │
│   GPU Kernels ← 记录               │
└─────────────────────────────────────┘
           ↓
┌─────────────────────────────────────┐
│ 回放阶段（多次）:                   │
│   graph.replay()                    │
│   ↓                                 │
│   GPU Kernels ← 直接执行            │
│   logger.info() ← 跳过！            │
└─────────────────────────────────────┘
```

---

## 为什么 Decode 阶段看不到日志

### 问题现象

在 decode 阶段，即使设置了 `max_tokens=512`，你也只会看到：
- **一次** decode 日志（在 CUDA Graph 捕获时）
- 后续 511 次 decode 步骤**完全没有日志**

### 原因分析

1. **CUDA Graph 回放绕过 Python**
   - `graph.replay()` 直接执行 GPU kernels
   - Python 代码（包括 `logger.info()`）不会被执行

2. **这是预期行为**
   - CUDA Graph 的设计目的就是避免 Python 开销
   - 如果每次 decode 都执行 Python 代码，性能会大幅下降

3. **验证方法**
   - 虽然看不到日志，但 decode 确实在执行
   - 可以通过输出结果、TTFT 指标等验证

### 如何看到所有 Decode 日志？

禁用 CUDA Graph：

```python
llm = LLM(
    model=test_model,
    gpu_memory_utilization=0.95,
    dtype=torch.bfloat16,
    max_model_len=20000,
    enforce_eager=True,  # ← 添加这个参数
)
```

**注意：** 这会显著降低性能，仅用于调试。

---

## PagedAttention KV Cache 机制

### 标准 Transformers vs vLLM

#### 标准 Transformers（拼接方式）

```python
# 每次 decode 都拼接新的 KV
kv_cache = torch.cat([kv_cache, new_kv], dim=1)
# Shape 变化: [batch, 1, ...] → [batch, 2, ...] → [batch, 3, ...]
```

**特点：**
- Cache shape 每次都会增长
- 需要动态内存分配
- 内存碎片化问题

#### vLLM PagedAttention（固定大小分页缓存）

```python
# 预分配固定大小的缓存
key_cache.shape = [num_blocks, num_kv_heads, head_size//x, block_size, x]
# 例如: [4991, 8, 16, 16, 8] - 固定大小！

# 新 token 写入特定 slot
PagedAttention.write_to_paged_cache(
    key, value, key_cache, value_cache,
    slot_mapping,  # ← 指定写入位置
    ...
)
```

**特点：**
- Cache shape **永远不变**
- 预分配，避免动态分配
- 高效的内存管理

### Cache 结构详解

#### Cache Shape 组成

```python
key_cache.shape = [4991, 8, 16, 16, 8]
#                  [num_blocks, num_kv_heads, head_size//x, block_size, x]

value_cache.shape = [4991, 8, 128, 16]
#                   [num_blocks, num_kv_heads, head_size, block_size]
```

**参数说明：**
- `num_blocks = 4991`: 总块数（预分配）
- `num_kv_heads = 8`: KV 头数
- `head_size = 128`: 每个头的维度
- `block_size = 16`: 每个块存储的 token 数
- `x = 8`: 内存对齐参数

#### 容量计算

```
总容量 = num_blocks × block_size
       = 4991 × 16
       = 79,856 tokens
```

### Slot Mapping 机制

每个 token 通过 `slot_mapping` 映射到特定的缓存位置：

```python
# slot_mapping 示例: [35, 2, 17]
# 如果 block_size = 16:
# - Token 0 → Block 2, Slot 3 (35 = 2*16 + 3)
# - Token 1 → Block 0, Slot 2 (2 = 0*16 + 2)
# - Token 2 → Block 1, Slot 1 (17 = 1*16 + 1)
```

### Context Length 跟踪

虽然 cache shape 不变，但 `context_lens` 会跟踪实际存储的 token 数：

```python
# Decode 步骤 1: context_lens = [1]
# Decode 步骤 2: context_lens = [2]
# Decode 步骤 3: context_lens = [3]
# ...
# Decode 步骤 6265: context_lens = [6265]
```

---

## Cache Shape 为什么不变

### 核心原因

1. **预分配策略**
   - vLLM 在初始化时根据 GPU 内存预分配所有块
   - 块的数量在运行时不会改变

2. **固定 Tensor Shape**
   - PyTorch tensor 的 shape 是固定的
   - 不能像列表那样动态增长

3. **性能优化**
   - 避免动态内存分配的开销
   - 减少内存碎片化
   - 提高缓存命中率

### 实际使用情况

假设 `context_lens = 6265`：

```
需要的块数 = ceil(6265 / 16) = 392 blocks

实际分配: 4991 blocks
已使用:   392 blocks (7.9%)
未使用:   4599 blocks (92.1%)
```

**可视化：**

```
┌─────────────────────────────────────────────────────────────┐
│ Block 0  │ Block 1  │ ... │ Block 391 │ Block 392 │ ... │ Block 4990 │
│ [16 tok] │ [16 tok] │     │ [16 tok]  │ [empty]   │     │ [empty]    │
└─────────────────────────────────────────────────────────────┘
     ↑ 已使用 (6265 tokens)              ↑ 未使用 (73,591 token 容量)
```

### 为什么分配这么多块？

块的数量由以下因素决定：

1. **GPU 内存大小**
2. **gpu_memory_utilization** (默认 0.90-0.95)
3. **模型大小和层数**
4. **最大序列长度**

vLLM 会预分配尽可能多的块，以支持更长的序列和更多的并发请求。

---

## 如何验证 Decode 行为

### 方法 1: 禁用 CUDA Graph

```python
llm = LLM(
    model=test_model,
    enforce_eager=True,  # 禁用 CUDA Graph
    ...
)
```

**结果：**
- 每次 decode 都会执行 Python 代码
- 可以看到所有 decode 日志
- 性能会显著下降

### 方法 2: 添加详细日志

在 `xformers.py` 中添加：

```python
if decode_meta := attn_metadata.decode_metadata:
    logger.info(f"[Decode Triggered] Key Shape: {key.shape}, "
                f"Value Shape: {value.shape}, Query Shape: {query.shape}")
    logger.info(f"[Decode Info] Key Cache Shape: {key_cache.shape}, "
                f"Value Cache Shape: {value_cache.shape}")
    logger.info(f"context_lens: {decode_meta.context_lens}, "
                f"max_context_len: {decode_meta.max_context_len}")
    
    # 计算实际使用的块数
    block_size = value_cache.shape[3]
    blocks_used = (decode_meta.context_lens[0].item() + block_size - 1) // block_size
    total_blocks = value_cache.shape[0]
    logger.info(f"[Block Usage] blocks_used: {blocks_used}/{total_blocks}, "
                f"capacity: {total_blocks * block_size} tokens, "
                f"used: {decode_meta.context_lens[0].item()} tokens")
```

### 方法 3: 监控 GPU 活动

```python
import torch

# 在生成前
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize()

# 运行生成
output = llm.generate(...)

torch.cuda.synchronize()
print(f"GPU 内存使用: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
```

### 方法 4: 检查输出结果

最简单的方法：检查生成结果是否正确。

```python
output = llm.generate(prompts, sampling_params)
print(f"生成结果: {output[0].outputs[0].text}")
print(f"生成 token 数: {len(output[0].outputs[0].token_ids)}")
```

---

## 常见问题解答

### Q1: 为什么 decode 日志只出现一次？

**A:** 因为 CUDA Graph 回放阶段绕过了 Python 代码执行。只有捕获阶段会执行 Python 代码。

### Q2: 为什么 cache shape 不变？

**A:** vLLM 使用预分配的固定大小缓存，通过 `slot_mapping` 和 `context_lens` 跟踪实际使用情况。

### Q3: 4991 和 6265 的关系是什么？

**A:** 
- `4991` 是块的数量（blocks）
- `6265` 是 token 的数量
- 每个块可以存储 16 个 tokens
- 总容量 = 4991 × 16 = 79,856 tokens
- 当前使用 = 6265 tokens，需要约 392 个块

### Q4: 如何确认 decode 真的在执行？

**A:** 
1. 检查生成输出是否正确
2. 监控 `context_lens` 是否递增
3. 使用 `enforce_eager=True` 查看所有日志
4. 检查 GPU 使用率和内存变化

### Q5: CUDA Graph 会影响正确性吗？

**A:** 不会。CUDA Graph 只是优化执行方式，不改变计算逻辑。GPU kernels 的执行结果与正常执行完全相同。

---

## 总结

1. **CUDA Graph 优化**
   - 捕获阶段：执行一次，记录 GPU kernels
   - 回放阶段：直接执行 GPU kernels，跳过 Python
   - 这是为什么看不到 decode 日志的原因

2. **PagedAttention KV Cache**
   - 使用预分配的固定大小缓存
   - Cache shape 不变，但 `context_lens` 会递增
   - 通过 `slot_mapping` 管理 token 存储位置

3. **验证方法**
   - 禁用 CUDA Graph (`enforce_eager=True`)
   - 监控 `context_lens` 变化
   - 检查生成结果

4. **性能 vs 可观测性**
   - CUDA Graph 提供最佳性能
   - 但会牺牲可观测性（日志）
   - 调试时可以使用 `enforce_eager=True`

---

## 参考资料

- [vLLM PagedAttention 论文](https://arxiv.org/abs/2309.06180)
- [CUDA Graph 文档](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
- [vLLM 源码](https://github.com/vllm-project/vllm)

---

*最后更新: 2024-11-19*

