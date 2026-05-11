# Hermes 用户培训指南

**版本**: 1.0  
**适用对象**: AI算法工程师  
**阅读时间**: 约30分钟

---

## 🚀 快速入门

### 1. 环境准备

```bash
# 设置环境变量
export HERMES_API_URL=http://hermes.example.com/v1
export HERMES_TENANT_ID=your-tenant-id

# 安装CLI（可选）
pip install hermes-cosmos
```

### 2. 提交第一个作业

```python
import httpx

response = httpx.post(
    f"{HERMES_API_URL}/jobs",
    json={
        "name": "my-first-training-job",
        "tenant_id": HERMES_TENANT_ID,
        "user_id": "your-user-id",
        "priority": "NORMAL",
        "requirements": {
            "gpu_count": 8,
            "gpu_type": "nvidia-h100",
            "memory_gb": 512,
            "cpu_cores": 32,
        },
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "command": "python train.py --epochs 10",
    }
)

job = response.json()
print(f"作业ID: {job['job']['id']}")
```

### 3. 监控作业状态

```bash
# 查询作业状态
curl "$HERMES_API_URL/jobs/{job-id}"

# 查询集群资源
curl "$HERMES_API_URL/resources/cluster/summary"
```

---

## 📋 作业配置详解

### 作业优先级

```python
# 优先级从高到低
"priority": "CRITICAL"  # 最高优先级，可抢占其他作业
"priority": "HIGH"      # 高优先级
"priority": "NORMAL"    # 默认
"priority": "LOW"       # 低优先级，可能被抢占
```

### 资源需求

```python
"requirements": {
    "gpu_count": 8,          # GPU数量
    "gpu_type": "nvidia-h100", # GPU型号
    "memory_gb": 512,        # 内存
    "cpu_cores": 32,         # CPU核心数
    "storage_gb": 1000,      # 存储空间
    "network_bandwidth_gbps": 100,  # 网络带宽
    "max_duration_hours": 72, # 最大运行时间
    "checkpoint_interval_seconds": 300, # Checkpoint间隔
}
```

### 调度约束

```python
"constraints": {
    "regions": ["eu-west"],  # 仅限欧盟区域（数据不离境）
    "data_locality": true,   # 要求数据本地性
    "carbon_aware": true,    # 碳感知调度
    "max_carbon_intensity": 200,  # 最大碳强度
}
```

---

## 💾 Checkpoint 最佳实践

### 启用自动Checkpoint

```python
"checkpoint_enabled": true,
"requirements": {
    "checkpoint_interval_seconds": 300,  # 每5分钟保存一次
}
```

### 手动触发Checkpoint

```bash
curl -X POST "$HERMES_API_URL/jobs/{job-id}/checkpoint"
```

### 从Checkpoint恢复

```bash
curl -X POST "$HERMES_API_URL/checkpoints/{checkpoint-id}/restore"
```

### 代码层面的容错

```python
import torch
from hermes.checkpoint import HermesCheckpointer

checkpointer = HermesCheckpointer(job_id="your-job-id")

# 训练循环
for epoch in range(epochs):
    # 尝试从Checkpoint恢复
    try:
        state = checkpointer.load()
        start_epoch = state["epoch"]
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
    except Exception:
        start_epoch = 0
    
    for batch in dataloader:
        # 训练逻辑
        loss = train_step(batch)
        
        # 定期保存Checkpoint（可选，Hermes会自动保存）
        if batch_idx % 100 == 0:
            checkpointer.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "loss": loss.item(),
            })
```

---

## 💰 成本优化

### 查询成本预估

```bash
# 获取作业成本预估
curl "$HERMES_API_URL/jobs/{job-id}/cost-estimate"

# 获取租户成本统计
curl "$HERMES_API_URL/tenants/{tenant-id}/cost-summary"
```

### 成本优化策略

| 策略 | 方法 | 预期节省 |
|------|------|----------|
| 批量调度 | 将多个小作业打包到同一节点 | GPU利用率+15% |
| 分时调度 | 在电价低谷时段运行 | 成本-30% |
| 竞价实例 | 使用Spot实例运行非关键作业 | 成本-40% |

### 使用碳感知调度

```python
"constraints": {
    "carbon_aware": true,           # 启用碳感知
    "max_carbon_intensity": 200,    # 限制碳强度
}
```

---

## 🚨 故障处理

### 常见问题

| 问题 | 现象 | 解决方法 |
|------|------|----------|
| 作业排队 | status=QUEUED | 检查资源是否充足 |
| 调度失败 | status=FAILED | 检查资源需求是否合理 |
| Checkpoint失败 | 日志显示错误 | 检查存储配置 |
| GPU故障 | 作业被中断 | 等待自动恢复（<5秒） |

### 手动恢复

```bash
# 查看故障预测
curl "$HERMES_API_URL/agents/{agent-id}/predictions"

# 手动迁移作业
curl -X POST "$HERMES_API_URL/jobs/{job-id}/migrate?target_region=us-west"
```

### 联系支持

- **紧急问题**: #hermes-support Slack频道
- **非紧急问题**: hermes-support@example.com
- **文档**: docs.hermes.example.com

---

## 📊 性能监控

### 关键指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 调度延迟 | 提交到分配的时间 | < 500ms |
| GPU利用率 | 训练期间的GPU使用率 | > 80% |
| Checkpoint时间 | 保存一次Checkpoint的时间 | < 1秒 |
| 故障恢复时间 | 从故障到恢复的时间 | < 5秒 |

### 查看指标

```bash
# Prometheus查询示例
# 调度延迟P99
rate(hermes_scheduler_scheduling_latency_seconds[5m])

# GPU利用率
avg(hermes_agent_gpu_utilization)

# Checkpoint成功率
sum(hermes_checkpoint_total{status="success"}) / sum(hermes_checkpoint_total)
```

---

## 🔒 安全注意事项

### 数据保护

- 敏感数据必须使用 `data_locality` 约束
- 训练数据应加密传输和存储
- 模型权重应使用TEE保护

### 权限管理

- 仅授权用户可提交作业
- 作业只能访问本租户的资源
- 管理员操作需要二次验证

---

## 📝 示例：完整作业提交

```python
import httpx

# 提交一个BERT训练作业
response = httpx.post(
    "http://hermes.example.com/v1/jobs",
    json={
        "name": "bert-fine-tuning",
        "tenant_id": "nlp-team",
        "user_id": "john-doe",
        "priority": "HIGH",
        "requirements": {
            "gpu_count": 16,
            "gpu_type": "nvidia-h100",
            "memory_gb": 1024,
            "cpu_cores": 64,
            "storage_gb": 2000,
            "checkpoint_interval_seconds": 180,
        },
        "constraints": {
            "regions": ["us-east"],
            "carbon_aware": true,
        },
        "image": "pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime",
        "command": """
            python run_pretraining.py \
                --model_type bert \
                --model_name_or_path bert-base-uncased \
                --do_train \
                --train_file train.txt \
                --num_train_epochs 3 \
                --per_device_train_batch_size 32
        """,
        "environment": {
            "WANDB_PROJECT": "nlp-bert",
            "TOKENIZERS_PARALLELISM": "false",
        },
        "volumes": [
            "nlp-datasets:/data",
        ],
        "checkpoint_enabled": true,
        "metadata": {
            "experiment_name": "bert-v2-fine-tune",
            "team": "nlp",
        },
    }
)

print("作业提交成功！")
print(f"作业ID: {response.json()['job']['id']}")
```

---

## 📚 参考资源

| 资源 | 链接 |
|------|------|
| API文档 | https://hermes.example.com/docs |
| 完整示例 | https://github.com/hermes-cosmos/examples |
| 最佳实践 | https://docs.hermes.example.com/best-practices |
| 故障排查 | https://docs.hermes.example.com/troubleshooting |
