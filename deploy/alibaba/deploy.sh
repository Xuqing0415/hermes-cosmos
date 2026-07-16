#!/bin/bash
# Hermes Parameter Server Alibaba Cloud ECS Deployment Script

set -e

echo " Hermes Parameter Server Alibaba Cloud Deployment"
echo "==================================================="

# Configuration
REGION="cn-hangzhou"
INSTANCE_TYPE="ecs.g6.large"
IMAGE_ID="ubuntu_22_04_x64_20G_alibase_20230510.vhd"
SECURITY_GROUP="hermes-sg"
INSTANCE_NAME_PREFIX="hermes"

# Create security group
echo " Creating security group..."
SECURITY_GROUP_ID=$(aliyun ecs CreateSecurityGroup \
    --RegionId $REGION \
    --SecurityGroupName $SECURITY_GROUP \
    --Description "Hermes Parameter Server Security Group" \
    --query 'SecurityGroupId' \
    --output text)

echo " Security Group ID: $SECURITY_GROUP_ID"

# Allow inbound traffic
echo " Configuring security rules..."
aliyun ecs AuthorizeSecurityGroup \
    --RegionId $REGION \
    --SecurityGroupId $SECURITY_GROUP_ID \
    --IpProtocol tcp \
    --PortRange 5000/5000 \
    --SourceCidrIp 0.0.0.0/0

aliyun ecs AuthorizeSecurityGroup \
    --RegionId $REGION \
    --SecurityGroupId $SECURITY_GROUP_ID \
    --IpProtocol tcp \
    --PortRange 22/22 \
    --SourceCidrIp 0.0.0.0/0

# Launch server instance
echo " Launching Parameter Server instance..."
SERVER_INSTANCE=$(aliyun ecs RunInstances \
    --RegionId $REGION \
    --InstanceType $INSTANCE_TYPE \
    --ImageId $IMAGE_ID \
    --SecurityGroupId $SECURITY_GROUP_ID \
    --InstanceName "${INSTANCE_NAME_PREFIX}-server" \
    --Amount 1 \
    --InternetMaxBandwidthOut 100 \
    --query 'InstanceIdSets.InstanceIdSet[0]' \
    --output text)

echo " Server instance ID: $SERVER_INSTANCE"

# Wait for instance to be ready
echo "⏳ Waiting for server to initialize..."
sleep 60

SERVER_IP=$(aliyun ecs DescribeInstances \
    --RegionId $REGION \
    --InstanceIds "[\"$SERVER_INSTANCE\"]" \
    --query 'Instances.Instance[0].PublicIpAddress.IpAddress[0]' \
    --output text)

echo " Server public IP: $SERVER_IP"

# Launch worker instances
echo "‍ Launching worker instances..."
WORKER_COUNT=4
WORKER_INSTANCES=""
WORKER_IPS=""

for i in $(seq 1 $WORKER_COUNT); do
    WORKER_INSTANCE=$(aliyun ecs RunInstances \
        --RegionId $REGION \
        --InstanceType $INSTANCE_TYPE \
        --ImageId $IMAGE_ID \
        --SecurityGroupId $SECURITY_GROUP_ID \
        --InstanceName "${INSTANCE_NAME_PREFIX}-worker-$i" \
        --Amount 1 \
        --InternetMaxBandwidthOut 100 \
        --query 'InstanceIdSets.InstanceIdSet[0]' \
        --output text)
    echo " Worker $i instance ID: $WORKER_INSTANCE"
    WORKER_INSTANCES="$WORKER_INSTANCES $WORKER_INSTANCE"
done

sleep 60

for INSTANCE_ID in $WORKER_INSTANCES; do
    IP=$(aliyun ecs DescribeInstances \
        --RegionId $REGION \
        --InstanceIds "[\"$INSTANCE_ID\"]" \
        --query 'Instances.Instance[0].PublicIpAddress.IpAddress[0]' \
        --output text)
    WORKER_IPS="$WORKER_IPS $IP"
done

echo " Worker IPs: $WORKER_IPS"

# Save configuration
echo " Saving deployment configuration..."
cat > deploy_config.sh << EOF
export SERVER_IP="$SERVER_IP"
export WORKER_IPS="$WORKER_IPS"
export REGION="$REGION"
EOF

echo ""
echo " Deployment complete!"
echo "==================================================="
echo "Server IP: $SERVER_IP"
echo "Worker IPs: $WORKER_IPS"
echo ""
echo "Next steps:"
echo "1. SSH into server: ssh root@$SERVER_IP"
echo "2. Install dependencies and start server"
echo "3. SSH into workers and start worker processes"