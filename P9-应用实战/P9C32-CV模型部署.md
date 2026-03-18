# P9C32 — 场景二：计算机视觉模型部署

## 引子

虽然大模型是当前的热点，但计算机视觉（CV）模型仍然是昇腾最广泛的应用场景之一——从安防监控到工业质检到自动驾驶。

CV 模型的部署流程和 LLM 有一个关键区别：CV 模型通常走 **ONNX → ATC → .om** 的静态编译路径，而非推理引擎动态加载。

---

## 1. 典型流程

```
PyTorch CV 模型 (如 YOLOv8)
      │
      ▼  torch.onnx.export()
  model.onnx
      │
      ▼  ATC 编译
  model.om              ← 包含所有图优化
      │
      ▼  ACL 推理程序
  高性能推理服务
```

---

## 2. 模型导出

```python
import torch
model = load_yolov8_model()
model.eval()

dummy_input = torch.randn(1, 3, 640, 640)
torch.onnx.export(
    model, dummy_input, "yolov8.onnx",
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={"images": {0: "batch"}}  # 支持动态 Batch
)
```

---

## 3. ATC 编译

```bash
atc --model=yolov8.onnx \
    --framework=5 \           # 5 = ONNX
    --output=yolov8 \         # 输出 yolov8.om
    --input_shape="images:1,3,640,640" \
    --soc_version=Ascend910B
```

ATC 会自动执行算子融合、数据布局转换等优化。

---

## 4. ACL 推理

使用 AscendCL Python 接口进行推理：

```python
import acl
# 初始化设备
acl.init()
acl.rt.set_device(0)

# 加载模型
model_id = acl.mdl.load_from_file("yolov8.om")

# 创建输入/输出数据集
# 执行推理
acl.mdl.execute(model_id, input_dataset, output_dataset)

# 后处理（NMS 等）
results = postprocess(output_data)
```

---

## 5. CV vs LLM 部署差异

| 维度 | CV 模型 | LLM |
|------|---------|-----|
| 编译路径 | ONNX → .om 静态编译 | 推理引擎动态加载 |
| 输入形状 | 通常固定 | 动态（不同 prompt 长度） |
| 计算模式 | 一次前向得到结果 | 自回归多次前向 |
| 典型延迟 | < 10ms / 张图 | 数百毫秒到数秒 |
| 部署形态 | 常在边缘设备（Atlas 200/300I） | 常在数据中心（Atlas 800） |

---

## 本章小结

| 要点 | 内容 |
|------|------|
| CV 部署路径 | PyTorch → ONNX → ATC → .om → ACL 推理 |
| 和 LLM 区别 | 静态编译、固定输入、低延迟、适合边缘 |
| 性能关键 | ATC 的图优化质量决定推理性能 |
