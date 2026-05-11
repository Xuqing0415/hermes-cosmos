# Hermes 用户迁移指南

**版本**: 1.0  
**目标**: 让用户在1小时内完成迁移并看到效果

---

## 📝 私信模板

### 标准版
> “Hi [名字]，我最近写了一个轻量级工具，能让K8s上的训练作业在Pod意外删除后自动恢复，实测恢复时间3-4秒。我看你之前分享过训练中断的烦恼，想不想免费试一下？我可以帮你迁移一个作业，全程不需要你改代码（除了加两行Checkpoint保存）。如果你有兴趣，回复‘试试’。”

### 简化版（适合私信）
> “嘿，我做了个工具能让训练作业故障自动恢复，3秒搞定，要不要试试？”

---

## 🚀 极简迁移流程

### 前置条件
- [ ] 用户的作业已在K8s上运行
- [ ] 用户能提供Docker镜像

### 步骤

#### 1. 准备镜像（10分钟）
```bash
# 在用户镜像中添加Hermes Checkpoint客户端
docker build -t hermes-user-job --build-arg BASE_IMAGE=user-image .
```

#### 2. 修改训练脚本（5分钟）
```python
# 在训练循环中添加Checkpoint保存
from hermes.checkpoint import HermesCheckpointer

checkpointer = HermesCheckpointer(job_id="my-job")

for epoch in range(epochs):
    for batch in dataloader:
        # 训练逻辑...
        
        # 每100个batch保存一次Checkpoint
        if batch_idx % 100 == 0:
            checkpointer.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict()
            })
```

#### 3. 配置Deployment（5分钟）
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: user-training-job
spec:
  replicas: 1
  selector:
    matchLabels:
      app: user-training-job
  template:
    metadata:
      labels:
        app: user-training-job
    spec:
      restartPolicy: Never  # 让Hermes接管恢复
      containers:
      - name: training
        image: hermes-user-job
        command: ["python", "train.py"]
        resources:
          limits:
            nvidia.com/gpu: 1
```

#### 4. 提交到Hermes（5分钟）
```bash
# 使用Hermes CLI提交
hermes job submit \
  --name "user-training-job" \
  --gpus 1 \
  --image hermes-user-job \
  --checkpoint-interval 300
```

#### 5. 演示故障恢复（10分钟）
```bash
# 让用户自己删除Pod
kubectl delete pod <pod-name>

# 观察新Pod在3秒内启动
watch kubectl get pods

# 查看训练日志继续输出
kubectl logs <new-pod-name> -f
```

---

## 📊 预期效果

| 指标 | 预期值 |
|------|--------|
| 恢复时间 | 3-4秒 |
| Checkpoint保存 | < 1秒 |
| 数据损失 | < 100个batch |

---

## 💬 反馈收集

### 话术
> “你觉得这个工具对你有帮助吗？方便给一句评价吗？我想写到文档里。”

### 评价示例
- “恢复速度很快，再也不用担心训练中断了！”
- “用起来很简单，5分钟就迁移完了。”
- “比手动重启快太多了，省了不少时间。”
