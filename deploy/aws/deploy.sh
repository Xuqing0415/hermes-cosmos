#!/bin/bash
# Hermes Parameter Server AWS EC2 Deployment Script

set -e

echo " Hermes Parameter Server AWS Deployment"
echo "========================================="

# Configuration
REGION="us-west-2"
INSTANCE_TYPE="c5.large"
AMI_ID="ami-0c55b159cbfafe1f0"
KEY_NAME="hermes-key"
SECURITY_GROUP="hermes-sg"

# Create security group
echo " Creating security group..."
aws ec2 create-security-group \
    --group-name $SECURITY_GROUP \
    --description "Hermes Parameter Server Security Group" \
    --region $REGION

# Allow inbound traffic
echo " Configuring security rules..."
aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 5000 \
    --cidr 0.0.0.0/0 \
    --region $REGION

aws ec2 authorize-security-group-ingress \
    --group-name $SECURITY_GROUP \
    --protocol tcp \
    --port 22 \
    --cidr 0.0.0.0/0 \
    --region $REGION

# Launch server instance
echo " Launching Parameter Server instance..."
SERVER_INSTANCE=$(aws ec2 run-instances \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --region $REGION \
    --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=hermes-server}]' \
    --query 'Instances[0].InstanceId' \
    --output text)

echo " Server instance ID: $SERVER_INSTANCE"

# Wait for instance to be ready
echo "⏳ Waiting for server to initialize..."
aws ec2 wait instance-running --instance-ids $SERVER_INSTANCE --region $REGION
SERVER_IP=$(aws ec2 describe-instances \
    --instance-ids $SERVER_INSTANCE \
    --region $REGION \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo " Server public IP: $SERVER_IP"

# Launch worker instances
echo "‍ Launching worker instances..."
WORKER_COUNT=4
for i in $(seq 1 $WORKER_COUNT); do
    WORKER_INSTANCE=$(aws ec2 run-instances \
        --image-id $AMI_ID \
        --instance-type $INSTANCE_TYPE \
        --key-name $KEY_NAME \
        --security-groups $SECURITY_GROUP \
        --region $REGION \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=hermes-worker-$i}]" \
        --query 'Instances[0].InstanceId' \
        --output text)
    echo " Worker $i instance ID: $WORKER_INSTANCE"
    WORKER_IPS="$WORKER_IPS $(aws ec2 describe-instances \
        --instance-ids $WORKER_INSTANCE \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)"
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
echo "========================================="
echo "Server IP: $SERVER_IP"
echo "Worker IPs: $WORKER_IPS"
echo ""
echo "Next steps:"
echo "1. SSH into server: ssh -i $KEY_NAME.pem ubuntu@$SERVER_IP"
echo "2. Install dependencies and start server"
echo "3. SSH into workers and start worker processes"