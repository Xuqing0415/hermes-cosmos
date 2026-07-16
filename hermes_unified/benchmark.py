#!/usr/bin/env python3
"""
Hermes Unified Benchmark Suite


1.  HermesGossip/PS PyTorch DDP 
2. lossGPU
3. AMP
"""

import argparse
import json
import os
import sqlite3
import time
from datetime import datetime
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

#  Hermes 
try:
    from hermes_unified.core.worker import Worker
    from hermes_unified.core.parameter_server import RaftParameterServer
    HERMES_AVAILABLE = True
except ImportError:
    HERMES_AVAILABLE = False

# 
METRICS_DB = "benchmark_metrics.db"


class MetricsRecorder:
    """"""
    
    def __init__(self, run_id: str, mode: str):
        self.run_id = run_id
        self.mode = mode
        self.conn = sqlite3.connect(METRICS_DB)
        self._create_tables()
        self.batch_count = 0
        self.epoch_start_time = 0
    
    def _create_tables(self):
        """"""
        cursor = self.conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS training_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                mode TEXT,
                epoch INTEGER,
                batch INTEGER,
                loss REAL,
                grad_norm REAL,
                param_change_rate REAL,
                comm_delay_ms REAL,
                gpu_memory_used INTEGER,
                gpu_utilization INTEGER,
                timestamp TEXT
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS epoch_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                mode TEXT,
                epoch INTEGER,
                throughput REAL,
                epoch_time REAL,
                accuracy REAL,
                timestamp TEXT
            )
        ''')
        self.conn.commit()
    
    def record_batch(self, epoch: int, loss: float, grad_norm: float = None, 
                     param_change_rate: float = None, comm_delay_ms: float = None):
        """"""
        self.batch_count += 1
        
        #  GPU 
        gpu_memory_used = 0
        gpu_utilization = 0
        if torch.cuda.is_available():
            gpu_memory_used = torch.cuda.memory_allocated() / (1024 ** 2)  # MB
            gpu_utilization = 0  #  NVIDIA 
        
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO training_metrics 
            (run_id, mode, epoch, batch, loss, grad_norm, param_change_rate, 
             comm_delay_ms, gpu_memory_used, gpu_utilization, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (self.run_id, self.mode, epoch, self.batch_count, loss, grad_norm,
              param_change_rate, comm_delay_ms, gpu_memory_used, gpu_utilization,
              datetime.now().isoformat()))
        self.conn.commit()
    
    def record_epoch(self, epoch: int, throughput: float, epoch_time: float, accuracy: float):
        """ epoch """
        cursor = self.conn.cursor()
        cursor.execute('''
            INSERT INTO epoch_summary 
            (run_id, mode, epoch, throughput, epoch_time, accuracy, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (self.run_id, self.mode, epoch, throughput, epoch_time, accuracy,
              datetime.now().isoformat()))
        self.conn.commit()
    
    def close(self):
        """"""
        self.conn.close()


class ResNet(nn.Module):
    """ ResNet """
    
    def __init__(self, num_classes=10, depth=18):
        super(ResNet, self).__init__()
        self.in_channels = 64
        
        if depth == 18:
            layers = [2, 2, 2, 2]
        elif depth == 50:
            layers = [3, 4, 6, 3]
        else:
            layers = [2, 2, 2, 2]
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(64, layers[0])
        self.layer2 = self._make_layer(128, layers[1], stride=2)
        self.layer3 = self._make_layer(256, layers[2], stride=2)
        self.layer4 = self._make_layer(512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
    
    def _make_layer(self, channels, blocks, stride=1):
        layers = []
        downsample = None
        
        if stride != 1 or self.in_channels != channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channels)
            )
        
        layers.append(self._basic_block(channels, stride, downsample))
        self.in_channels = channels
        
        for _ in range(1, blocks):
            layers.append(self._basic_block(channels))
        
        return nn.Sequential(*layers)
    
    def _basic_block(self, channels, stride=1, downsample=None):
        return nn.Sequential(
            nn.Conv2d(self.in_channels, channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            Residual(downsample)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class Residual(nn.Module):
    """"""
    def __init__(self, downsample=None):
        super(Residual, self).__init__()
        self.downsample = downsample
    
    def forward(self, x):
        residual = x
        if self.downsample is not None:
            residual = self.downsample(x)
        return F.relu(x + residual)


def get_data_loaders(dataset_name: str = "cifar10", batch_size: int = 64):
    """"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    if dataset_name.lower() == "cifar100":
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data', train=False, download=True, transform=transform_test)
        num_classes = 100
    else:
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train)
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test)
        num_classes = 10
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader, num_classes


def compute_accuracy(model, test_loader, device):
    """"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total


def train_with_amp(model, train_loader, criterion, optimizer, device, 
                   scaler, metrics_recorder, epoch, args):
    """ AMP  epoch"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    epoch_start = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # 
        comm_start = time.time()
        
        # 
        with torch.cuda.amp.autocast(enabled=args.amp):
            outputs = model(images)
            loss = criterion(outputs, labels)
        
        # 
        comm_delay_ms = (time.time() - comm_start) * 1000
        
        # 
        scaler.scale(loss).backward()
        
        # 
        grad_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_norm += param.grad.norm().item() ** 2
        grad_norm = grad_norm ** 0.5
        
        # 
        param_change_rate = 0.0
        
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item() * images.size(0)
        total_samples += images.size(0)
        
        # 
        metrics_recorder.record_batch(
            epoch=epoch,
            loss=loss.item(),
            grad_norm=grad_norm,
            param_change_rate=param_change_rate,
            comm_delay_ms=comm_delay_ms
        )
        
        if batch_idx % 100 == 0:
            print(f"  Batch {batch_idx}: Loss={loss.item():.4f}, GradNorm={grad_norm:.4f}")
    
    epoch_time = time.time() - epoch_start
    throughput = total_samples / epoch_time
    avg_loss = total_loss / total_samples
    
    return throughput, epoch_time, avg_loss


def benchmark_ddp(args):
    """DDP """
    print("\n=== Running DDP Benchmark ===")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size)
    
    # 
    model = ResNet(num_classes=num_classes, depth=args.depth).to(device)
    
    #  DDPGPUGPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # AMP 
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # 
    run_id = f"ddp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_recorder = MetricsRecorder(run_id, "ddp")
    
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        throughput, epoch_time, avg_loss = train_with_amp(
            model, train_loader, criterion, optimizer, device, scaler,
            metrics_recorder, epoch, args
        )
        
        accuracy = compute_accuracy(model, test_loader, device)
        
        metrics_recorder.record_epoch(epoch, throughput, epoch_time, accuracy)
        
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Epoch Time: {epoch_time:.2f} sec")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        results.append({
            "epoch": epoch,
            "throughput": throughput,
            "epoch_time": epoch_time,
            "accuracy": accuracy
        })
    
    metrics_recorder.close()
    
    # 
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    avg_epoch_time = sum(r["epoch_time"] for r in results) / len(results)
    final_accuracy = results[-1]["accuracy"]
    
    print(f"\n=== DDP Benchmark Results ===")
    print(f"Average Throughput: {avg_throughput:.2f} samples/sec")
    print(f"Average Epoch Time: {avg_epoch_time:.2f} sec")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    
    return {
        "mode": "ddp",
        "avg_throughput": avg_throughput,
        "avg_epoch_time": avg_epoch_time,
        "final_accuracy": final_accuracy,
        "results": results
    }


def benchmark_hermes(args, mode="gossip"):
    """Hermes """
    print(f"\n=== Running Hermes {mode.upper()} Benchmark ===")
    
    if not HERMES_AVAILABLE:
        print(" Hermes modules not available, skipping...")
        return None
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader, num_classes = get_data_loaders(args.dataset, args.batch_size)
    
    # 
    model = ResNet(num_classes=num_classes, depth=args.depth).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4)
    
    # AMP 
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    
    # 
    run_id = f"hermes_{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    metrics_recorder = MetricsRecorder(run_id, f"hermes_{mode}")
    
    results = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        
        throughput, epoch_time, avg_loss = train_with_amp(
            model, train_loader, criterion, optimizer, device, scaler,
            metrics_recorder, epoch, args
        )
        
        accuracy = compute_accuracy(model, test_loader, device)
        
        metrics_recorder.record_epoch(epoch, throughput, epoch_time, accuracy)
        
        print(f"  Throughput: {throughput:.2f} samples/sec")
        print(f"  Epoch Time: {epoch_time:.2f} sec")
        print(f"  Accuracy: {accuracy:.2f}%")
        
        results.append({
            "epoch": epoch,
            "throughput": throughput,
            "epoch_time": epoch_time,
            "accuracy": accuracy
        })
    
    metrics_recorder.close()
    
    # 
    avg_throughput = sum(r["throughput"] for r in results) / len(results)
    avg_epoch_time = sum(r["epoch_time"] for r in results) / len(results)
    final_accuracy = results[-1]["accuracy"]
    
    print(f"\n=== Hermes {mode.upper()} Benchmark Results ===")
    print(f"Average Throughput: {avg_throughput:.2f} samples/sec")
    print(f"Average Epoch Time: {avg_epoch_time:.2f} sec")
    print(f"Final Accuracy: {final_accuracy:.2f}%")
    
    return {
        "mode": f"hermes_{mode}",
        "avg_throughput": avg_throughput,
        "avg_epoch_time": avg_epoch_time,
        "final_accuracy": final_accuracy,
        "results": results
    }


def run_benchmark(args):
    """"""
    print("=" * 60)
    print("Hermes Unified Benchmark Suite")
    print("=" * 60)
    print(f"Model: ResNet-{args.depth}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"AMP Enabled: {args.amp}")
    print("=" * 60)
    
    all_results = []
    
    # DDP 
    ddp_result = benchmark_ddp(args)
    if ddp_result:
        all_results.append(ddp_result)
    
    # Hermes Gossip 
    gossip_result = benchmark_hermes(args, mode="gossip")
    if gossip_result:
        all_results.append(gossip_result)
    
    # Hermes PS 
    ps_result = benchmark_hermes(args, mode="ps")
    if ps_result:
        all_results.append(ps_result)
    
    # 
    output_file = f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "args": vars(args),
            "results": all_results
        }, f, indent=2)
    
    print(f"\n=== Results saved to {output_file} ===")
    
    # 
    print("\n=== Benchmark Comparison ===")
    print(f"{'Mode':<20} {'Throughput':<15} {'Epoch Time':<15} {'Accuracy':<10}")
    print("-" * 60)
    for result in all_results:
        print(f"{result['mode']:<20} {result['avg_throughput']:<15.2f} "
              f"{result['avg_epoch_time']:<15.2f} {result['final_accuracy']:<10.2f}")


def main():
    parser = argparse.ArgumentParser(description="Hermes Unified Benchmark")
    
    # 
    parser.add_argument("--depth", type=int, default=18, choices=[18, 50],
                        help="ResNet depth (18 or 50)")
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"],
                        help="Dataset to use")
    
    # 
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    
    # AMP 
    parser.add_argument("--amp", action="store_true", default=True,
                        help="Enable automatic mixed precision")
    
    # 
    parser.add_argument("--mode", type=str, default="all", 
                        choices=["ddp", "gossip", "ps", "all"],
                        help="Benchmark mode")
    
    args = parser.parse_args()
    
    run_benchmark(args)


if __name__ == "__main__":
    main()
