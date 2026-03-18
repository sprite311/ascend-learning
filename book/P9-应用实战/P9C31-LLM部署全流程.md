# P9C31 — 场景一：大语言模型部署全流程

## 引子

本章以 Qwen2.5-7B 为例，串联全书知识，展示在昇腾上部署 LLM 的端到端流程。代码示例将放在本章供参考。

---

## 1. 部署流程总览

```
步骤 1: 环境准备 → Docker 镜像 + 版本矩阵确认
步骤 2: 模型获取 → HuggingFace 下载权重
步骤 3: 方案选择 → MindIE 或 vLLM-Ascend
步骤 4: 部署启动 → 配置参数、启动服务
步骤 5: 验证测试 → 功能验证、性能测试
步骤 6: 生产优化 → 量化、调参、监控
```

---

## 2. 环境准备

### 2.1 使用 Docker

```bash
# 拉取官方镜像（含 CANN + torch_npu）
docker pull ascend-hub.huawei.com/public-ascendhub/pytorch:latest

# 启动容器（映射 NPU 设备）
docker run -it --privileged \
    --device=/dev/davinci0 \
    --device=/dev/davinci_manager \
    --device=/dev/devmm_svm \
    --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -p 8000:8000 \
    ascend-hub.huawei.com/public-ascendhub/pytorch:latest
```

### 2.2 验证环境

```python
import torch
import torch_npu
print(f"NPU 可用: {torch.npu.is_available()}")
print(f"NPU 设备数: {torch.npu.device_count()}")
```

---

## 3. 使用 vLLM-Ascend 部署

### 3.1 安装

```bash
pip install vllm vllm-ascend
```

### 3.2 启动推理服务

```bash
vllm serve Qwen/Qwen2.5-7B-Instruct \
    --device npu \
    --dtype float16 \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.9 \
    --port 8000
```

关键参数说明：
- `--device npu`：使用昇腾 NPU
- `--tensor-parallel-size`：张量并行度（多卡时增大）
- `--max-model-len`：最大序列长度（影响 KV Cache 内存）
- `--gpu-memory-utilization`：显存使用比例

### 3.3 调用服务

```bash
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-7B-Instruct",
        "messages": [{"role": "user", "content": "请介绍一下华为昇腾芯片"}],
        "max_tokens": 512
    }'
```

---

## 4. 性能验证

### 4.1 基准测试

```bash
# 使用 vLLM 自带的 benchmark
python -m vllm.entrypoints.openai.api_server &
python benchmark_serving.py \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-prompts 100 \
    --request-rate 10
```

### 4.2 关键指标

| 指标 | 7B FP16 单卡参考值 |
|------|-------------------|
| TTFT (首 Token 延迟) | < 200ms |
| Token/s (单请求) | 30-60 |
| 吞吐量 (并发) | 500-1000 Token/s |

---

## 5. 生产优化

| 优化项 | 做法 | 效果 |
|--------|------|------|
| 量化 | W8A8 量化 | 速度提升 ~1.5x，内存减半 |
| 多卡 | tensor-parallel-size=2 | 吞吐翻倍 |
| 调整 max-model-len | 根据实际需求设置 | 释放 KV Cache 空间给更多并发 |

---

## 本章小结

本章展示了从零开始在昇腾上部署 LLM 的完整流程。核心步骤：Docker 环境 → 安装 vLLM → 启动服务 → 验证 → 优化。

---

*下一章：计算机视觉模型部署。*
