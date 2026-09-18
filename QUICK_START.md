# Hermes 快速入门指南

**面向**: AI算法工程师  
**阅读时间**: 5分钟  
**更新日期**: 2026年5月

---

## 🚀 快速开始

### 1. 安装CLI

```bash
pip install hermes-cosmos
```

### 2. 配置环境

```bash
export HERMES_API_URL=http://hermes.example.com/v1
export HERMES_TENANT_ID=your-tenant-id
```

### 3. 提交作业

```bash
# 训练作业
hermes job submit \
  --name "my-training-job" \
  --gpus 8 \
  --image "pytorch/pytorch:2.2.0" \
  --command "python train.py"

# 推理服务
hermes service create \
  --name "my-inference-service" \
  --model "llama-2-13b" \
  --replicas 2
```

---

## 📋 作业配置要点

### 资源需求

| 参数 | 说明 | 示例 |
|------|------|------|
| --gpus | GPU数量 | 8 |
| --gpu-type | GPU型号 | nvidia-h100 |
| --memory | 内存(GB) | 512 |
| --cpu | CPU核心数 | 32 |

### 优先级

```bash
--priority CRITICAL  # 最高，可抢占其他作业
--priority HIGH
--priority NORMAL    # 默认
--priority LOW       # 可被抢占
```

### 约束

```bash
--regions us-east    # 限定Region
--carbon-aware       # 碳感知调度
--tee-enabled        # 启用TEE加密
```

---

## 📊 查看状态

```bash
# 查看作业列表
hermes job list

# 查看作业详情
hermes job get <job-id>

# 查看日志
hermes job logs <job-id> -f

# 查看集群状态
hermes cluster status
```

---

## 💾 Checkpoint

```bash
# 手动保存Checkpoint
hermes checkpoint save <job-id>

# 从Checkpoint恢复
hermes checkpoint restore <checkpoint-id>
```

---

## 📈 成本预估

```bash
# 查看作业成本预估
hermes job cost <job-id>

# 查看租户成本统计
hermes tenant cost
```

---

## ❓ 常见问题

### Q: 作业排队太久怎么办？
A: 检查资源配额和优先级，联系管理员申请更多资源。

### Q: 如何查看GPU利用率？
A: 使用 `hermes job metrics <job-id>` 或查看Grafana仪表板。

### Q: 作业失败如何排查？
A: 使用 `hermes job logs <job-id>` 查看日志，或联系支持团队。

---

## 📞 支持

| 渠道 | 响应时间 |
|------|----------|
| Slack: #hermes-support | 2小时内 |
| 邮箱: hermes-support@example.com | 4小时内 |
| 紧急: PagerDuty | 30分钟内 |

---

**更多文档**: docs.hermes.example.com
