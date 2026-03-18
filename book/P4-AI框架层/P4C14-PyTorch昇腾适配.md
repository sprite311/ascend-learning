# P4C14 — PyTorch 在昇腾上跑：torch_npu 是如何工作的

## 引子：不想换框架，能直接在昇腾上跑 PyTorch 吗？

答案是：**可以。** 这就是 torch_npu 的价值。

torch_npu（正式名称 Ascend Extension for PyTorch）是一个 PyTorch 插件，它让你只需要最小程度的代码修改，就能把 PyTorch 模型从 CUDA 迁移到昇腾 NPU 上运行。

---

## 1. torch_npu 的工作原理

### 1.1 一句话原理

torch_npu 通过 PyTorch 的**设备扩展机制**（PrivateUse1 后端），注册了一个叫 `npu` 的新设备类型。当你把张量放到 `npu` 设备上后，所有对该张量的运算都会被路由到 CANN 的 ACL 接口执行。

### 1.2 调用链路

```python
# 你的代码
x = torch.randn(4, 768).npu()  # 数据放到 NPU 上
y = torch.matmul(x, w)          # 调用矩阵乘法
```

内部发生了什么：

```
PyTorch: torch.matmul 调度器
    │
    ├─ 检查输入张量的设备类型 → npu
    │
    └─→ torch_npu: 注册在 npu 后端的 matmul 实现
            │
            └─→ CANN ACL: aclnnMatMul(...)
                    │
                    └─→ NPU 硬件: Cube 单元执行
```

### 1.3 支持范围

torch_npu 覆盖了 PyTorch 绝大多数的标准操作（约 **1500+ 个算子**）。常用的操作如 `matmul`、`softmax`、`linear`、`conv2d`、`embedding` 等全部支持。

但一些不太常见的操作，或 PyTorch 较新版本中新增的操作，可能还没有适配。遇到不支持的算子会报 `RuntimeError: not supported on NPU` 。

---

## 2. 迁移步骤

把一个 CUDA PyTorch 项目迁移到昇腾，核心步骤很简洁：

### 2.1 代码修改

**改动一：导入 torch_npu**
```python
import torch
import torch_npu  # 新增这一行
```

**改动二：替换设备名**
```python
# 之前
device = torch.device('cuda:0')
# 之后
device = torch.device('npu:0')
```

**改动三：移除 CUDA 专属操作**
```python
# 删除这类代码
torch.cuda.synchronize()
torch.cuda.empty_cache()

# 改为
torch.npu.synchronize()
torch.npu.empty_cache()
```

### 2.2 环境搭建

迁移的代码改动很少，但**环境搭建是最耗时的部分**：

1. 安装匹配的 NPU 驱动和固件
2. 安装匹配版本的 CANN Toolkit
3. 安装匹配版本的 PyTorch
4. 安装匹配版本的 torch_npu

**强烈建议使用华为提供的 Docker 镜像**——里面已经把所有依赖装好并验证过兼容性。

---

## 3. 常见迁移问题

### 3.1 算子不支持

**现象**：运行时报 `RuntimeError: xxx is not supported on NPU`

**解决思路**：
1. 检查 CANN 和 torch_npu 版本是否最新
2. 用 PyTorch 标准 API 替代自定义操作
3. 如果是模型中的自定义 CUDA kernel，需要用 Ascend C 重写或回退到 CPU 执行

### 3.2 精度差异

**现象**：同一个模型，GPU 上结果正确，NPU 上出现 NaN 或结果偏差大

**常见原因**：
- FP16 精度下数值溢出（昇腾和 GPU 的 FP16 行为在边界值附近可能不同）
- 某些算子的实现使用了不同的计算顺序

**解决思路**：
1. 先用 FP32 跑一遍，排除是精度问题还是逻辑错误
2. 开启混合精度中的"黑名单"机制，把敏感算子强制留在 FP32
3. 使用 CANN 提供的精度对比工具定位出问题的算子

### 3.3 性能不如预期

**现象**：迁移后能跑，但速度没有明显提升，甚至更慢

**常见原因与对策**：

| 可能原因 | 排查方法 | 解决方案 |
|---------|---------|---------|
| CPU-NPU 数据来回拷贝 | Profiling 看 MemCpy 占比 | 确保数据全程留在 NPU |
| 大量小算子 | Profiling 看算子数量和间隙 | 使用 torch.compile 或图模式 |
| 同步点过多 | 代码中检查 synchronize 调用 | 减少显式同步 |
| Batch 太小 | 检查 batch size | 增大 batch，提高 Cube 利用率 |

---

## 4. torch_npu 的高级特性

### 4.1 与 torch.compile 集成

PyTorch 2.0+ 的 `torch.compile` 可以和 torch_npu 配合使用，通过图编译获得更好的性能：

```python
import torch
import torch_npu

model = MyModel().npu()
compiled_model = torch.compile(model, backend='npu')
output = compiled_model(input.npu())
```

### 4.2 混合精度训练

torch_npu 完整支持 PyTorch 的 AMP（Automatic Mixed Precision）：

```python
scaler = torch.npu.amp.GradScaler()
with torch.npu.amp.autocast():
    output = model(input)
    loss = criterion(output, target)
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### 4.3 分布式训练

torch_npu 适配了 PyTorch 的 `torch.distributed` 模块，底层使用 HCCL 通信库：

```python
torch.distributed.init_process_group(backend='hccl')
```

和 GPU 上使用 NCCL 的方式完全对称。

---

## 本章小结

| 要点 | 内容 |
|------|------|
| torch_npu 作用 | 让 PyTorch 代码在昇腾 NPU 上运行 |
| 工作原理 | 通过设备扩展机制将算子调用路由到 CANN |
| 迁移工作量 | 代码改动很小（改设备名），环境搭建是主要工作 |
| 常见问题 | 算子不支持、精度差异、性能不达标 |
| 最佳实践 | 用 Docker、查版本矩阵、先 FP32 验证再优化 |

---

*下一章：在昇腾上训练模型——从单卡到集群的训练流程。*
