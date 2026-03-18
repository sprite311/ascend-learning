# P5C20 — vLLM 在昇腾上的适配：开源推理引擎的移植

## 引子

vLLM 是目前最流行的开源大模型推理引擎，由加州大学伯克利分校开发。它因提出 PagedAttention 技术而闻名，在 NVIDIA GPU 上有极佳的性能。

很多开发者已经熟悉 vLLM 的使用方式，如果能在昇腾上也使用 vLLM，就能大幅降低迁移成本。**vLLM-Ascend** 就是社区为此做的适配项目。

---

## 1. vLLM 的核心设计

### 1.1 PagedAttention

vLLM 的核心创新是 **PagedAttention**——把 KV Cache 拆成固定大小的"页"（Page），像操作系统管理虚拟内存一样动态分配和回收。

```
传统 KV Cache:
[请求A 预分配 2048 Token 空间] [请求B 预分配 2048 空间]
  实际只用了 512              实际只用了 128
  浪费 75%                    浪费 94%

PagedAttention:
[A页1][A页2][B页1][A页3][空闲][空闲]
  按需分配，用多少占多少
  浪费 < 5%
```

### 1.2 Continuous Batching

vLLM 同样支持连续批处理（和 MindIE 类似），请求随时加入和退出。

### 1.3 OpenAI 兼容 API

vLLM 提供了和 OpenAI API 格式完全兼容的 HTTP 接口。已有的客户端代码几乎不需要修改即可对接。

---

## 2. vLLM-Ascend 适配原理

### 2.1 适配层次

vLLM 原始代码中，和硬件相关的部分主要集中在：
- **注意力计算后端**：调用 CUDA kernel
- **内存管理**：调用 CUDA 内存 API
- **通信**：调用 NCCL

vLLM-Ascend 的工作是把这些部分替换为昇腾对应的实现：

| 原始 (CUDA) | 昇腾适配 |
|------------|---------|
| CUDA Attention kernels | CANN FlashAttention 算子 |
| torch.cuda.* | torch.npu.* (torch_npu) |
| NCCL | HCCL |
| cudaMalloc | aclrtMalloc |

### 2.2 适配方式

vLLM 的架构设计有较好的后端抽象。适配主要通过：

1. **注册 NPU 后端**：在 vLLM 的 Platform 层注册昇腾设备
2. **实现 Attention 后端**：用 CANN 的 FlashAttention/PagedAttention 算子
3. **适配 Worker**：处理多卡通信和任务分发

---

## 3. MindIE vs vLLM-Ascend：如何选择

| 维度 | MindIE | vLLM-Ascend |
|------|--------|-------------|
| 开发方 | 华为官方 | 社区适配 |
| 性能 | 通常更优（深度集成 CANN） | 良好（持续优化中） |
| 生态兼容 | 华为生态闭环 | 对接 HuggingFace、OpenAI API 生态 |
| 模型支持 | 需要逐模型适配 | 继承 vLLM 社区的模型支持 |
| 学习成本 | 有独立的 API 和配置方式 | 和 GPU 上使用 vLLM 一致 |
| 更新速度 | 随 CANN 版本更新 | 跟随 vLLM 社区主线 |

**选择建议**：
- **追求最优性能 + 华为技术支持** → MindIE
- **已有 vLLM 经验 + 追求迁移便利** → vLLM-Ascend
- **需要最新开源模型支持** → vLLM-Ascend（社区模型支持通常更快）

---

## 4. 使用 vLLM-Ascend 的基本步骤

```bash
# 1. 安装（在已配好 CANN + torch_npu 的环境中）
pip install vllm  # 安装 vLLM
pip install vllm-ascend  # 安装昇腾适配插件

# 2. 启动服务
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --device npu \
    --tensor-parallel-size 1 \
    --max-model-len 4096

# 3. 调用（和 OpenAI API 完全相同）
curl http://localhost:8000/v1/chat/completions \
    -d '{"model":"Qwen/Qwen2.5-7B-Instruct","messages":[{"role":"user","content":"你好"}]}'
```

---

## 本章小结

| 要点 | 内容 |
|------|------|
| vLLM 核心 | PagedAttention + Continuous Batching |
| 昇腾适配 | 替换 CUDA → CANN，NCCL → HCCL |
| vs MindIE | MindIE 性能更优，vLLM 生态更便利 |
| 使用方式 | 和 GPU 上使用 vLLM 几乎完全一致 |

---

*下一章：推理优化技术全景——量化、KV Cache 管理、算子融合的原理与实践。*
