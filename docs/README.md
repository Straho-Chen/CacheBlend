# CacheBlend 中文文档

本目录包含 CacheBlend 项目的中文技术文档。

## 文档列表

### [vLLM Decode 机制详解](./vllm_decode_mechanism.md)

详细解释 vLLM 在 decode（生成）阶段的工作原理，包括：

- **CUDA Graph 优化机制**: 解释为什么 decode 阶段看不到日志
- **PagedAttention KV Cache**: 说明为什么 cache shape 不变
- **验证方法**: 如何确认 decode 确实在执行
- **常见问题**: FAQ 和解答

**适用场景：**
- 理解 vLLM decode 阶段的工作原理
- 调试 decode 相关问题
- 优化 decode 性能

---

## 快速链接

- [主项目 README](../README.md)
- [英文文档](../docs/) (如果有)

---

*如有问题或建议，请提交 Issue 或 PR*

