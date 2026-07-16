"""
Hermes Unified Launcher

"""

import argparse
import sys
import os
import time
import socket

def main():
    parser = argparse.ArgumentParser(description='Hermes Unified Launcher')
    
    # 
    parser.add_argument('--mode', type=str, default='train', 
                        choices=['train', 'tune', 'benchmark', 'ps', 'worker'],
                        help='')
    
    # 
    parser.add_argument('--epochs', type=int, default=10, help='')
    parser.add_argument('--batch-size', type=int, default=64, help='')
    parser.add_argument('--lr', type=float, default=0.01, help='')
    parser.add_argument('--compression', type=float, default=0.1, help='')
    
    # 
    parser.add_argument('--tune-calls', type=int, default=20, help='')
    parser.add_argument('--tune-workers', type=int, default=4, help='Worker')
    
    # 
    parser.add_argument('--gpu-workers', type=int, default=2, help='GPU Worker')
    parser.add_argument('--cpu-workers', type=int, default=2, help='CPU Worker')
    
    # 
    parser.add_argument('--bind', type=str, default='0.0.0.0', help='')
    parser.add_argument('--port', type=int, default=8888, help='')
    parser.add_argument('--server-ip', type=str, default='172.20.0.10', help='IP')
    parser.add_argument('--worker-id', type=int, default=0, help='Worker ID')
    
    args = parser.parse_args()
    
    print(f" Hermes Launcher starting in {args.mode} mode")
    
    if args.mode == 'train':
        run_training(args)
    elif args.mode == 'tune':
        run_tuning(args)
    elif args.mode == 'benchmark':
        run_benchmark(args)
    elif args.mode == 'ps':
        run_parameter_server(args)
    elif args.mode == 'worker':
        run_worker(args)

def run_parameter_server(args):
    """"""
    print(f"\n Starting Parameter Server on {args.bind}:{args.port}")
    
    import socket
    import threading
    
    # socket
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind((args.bind, args.port))
    server_socket.listen(10)
    
    print(f" Parameter Server listening on {args.bind}:{args.port}")
    
    # 
    params = {'value': 1.0}
    workers = {}
    worker_id_counter = 0
    
    def handle_worker(conn, addr):
        nonlocal worker_id_counter
        print(f" Worker connected from {addr}")
        
        worker_id = worker_id_counter
        worker_id_counter += 1
        workers[worker_id] = {'conn': conn, 'last_heartbeat': time.time()}
        
        try:
            while True:
                data = conn.recv(1024)
                if not data:
                    break
                
                # 
                message = data.decode('utf-8')
                if message.startswith('get_params'):
                    # 
                    conn.sendall(f"params:{params['value']}".encode('utf-8'))
                elif message.startswith('update:'):
                    # 
                    _, delta = message.split(':')
                    params['value'] -= float(delta) * 0.1
                    print(f" Parameter updated: {params['value']:.4f}")
                elif message.startswith('heartbeat'):
                    # 
                    workers[worker_id]['last_heartbeat'] = time.time()
        except Exception as e:
            print(f" Worker {worker_id} disconnected: {e}")
        finally:
            del workers[worker_id]
            conn.close()
    
    # 
    def heartbeat_checker():
        while True:
            now = time.time()
            dead_workers = []
            for wid, info in workers.items():
                if now - info['last_heartbeat'] > 10:
                    dead_workers.append(wid)
                    print(f" Worker {wid} timeout, removing from cluster")
            
            for wid in dead_workers:
                del workers[wid]
            
            time.sleep(5)
    
    # 
    threading.Thread(target=heartbeat_checker, daemon=True).start()
    
    # 
    while True:
        conn, addr = server_socket.accept()
        threading.Thread(target=handle_worker, args=(conn, addr), daemon=True).start()

def run_worker(args):
    """Worker"""
    print(f"\n‍ Starting Worker {args.worker_id} connecting to {args.server_ip}:{args.port}")
    
    import socket
    import time
    
    while True:
        try:
            # 
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((args.server_ip, args.port))
            print(f" Worker {args.worker_id} connected to server")
            
            # 
            iteration = 0
            while True:
                # 
                sock.sendall(b'get_params')
                response = sock.recv(1024).decode('utf-8')
                
                if response.startswith('params:'):
                    param = float(response.split(':')[1])
                    
                    # 
                    grad = 2 * param - 4  # : param = 2
                    print(f" Worker {args.worker_id} Iteration {iteration}: param={param:.4f}, grad={grad:.4f}")
                    
                    # 
                    sock.sendall(f'update:{grad}'.encode('utf-8'))
                    
                    # 
                    sock.sendall(b'heartbeat')
                
                iteration += 1
                time.sleep(1)
                
        except Exception as e:
            print(f" Worker {args.worker_id} connection lost: {e}")
            print(" Retrying in 5 seconds...")
            time.sleep(5)

def run_training(args):
    """"""
    from mnist_training import run_mnist_training
    print(f"\n Training with lr={args.lr}, batch_size={args.batch_size}, compression={args.compression}")
    run_mnist_training()

def run_tuning(args):
    """"""
    from hermes_unified.auto_tuner.bayesian_optimizer import BayesianOptimizer
    from hermes_unified.scheduler.hardware_aware import HardwareAwareScheduler, WorkerInfo
    
    print(f"\n Hyperparameter Tuning with {args.tune_calls} trials")
    
    # Worker
    workers = []
    for i in range(args.gpu_workers):
        workers.append(WorkerInfo(i, 'gpu', speed=10.0))
    for i in range(args.cpu_workers):
        workers.append(WorkerInfo(args.gpu_workers + i, 'cpu', speed=2.0))
    
    scheduler = HardwareAwareScheduler(workers)
    
    # 
    search_space = [
        ('lr', 'real', 1e-4, 1e-1),
        ('batch_size', 'integer', 16, 128),
        ('compression', 'real', 0.01, 0.5)
    ]
    
    # 
    def objective(params):
        target_lr = 0.05
        target_batch = 32
        target_compression = 0.1
        
        lr_error = (params['lr'] - target_lr) ** 2
        batch_error = (params['batch_size'] - target_batch) ** 2 / 100
        comp_error = (params['compression'] - target_compression) ** 2
        
        noise = np.random.normal(0, 0.01)
        
        return lr_error + batch_error + comp_error + noise
    
    optimizer = BayesianOptimizer(objective, search_space, n_calls=args.tune_calls)
    best_params, best_loss = optimizer.run()
    
    print(f"\n Best parameters found:")
    print(f"   lr: {best_params['lr']:.6f}")
    print(f"   batch_size: {best_params['batch_size']}")
    print(f"   compression: {best_params['compression']:.4f}")
    print(f"   loss: {best_loss:.6f}")
    
    if best_loss < 0.05:
        print(" Hyperparameter tuning successful!")
    else:
        print(" Hyperparameter tuning needs more iterations")

def run_benchmark(args):
    """"""
    from performance_benchmark import run_benchmark
    print("\n Running Performance Benchmark")
    run_benchmark()

if __name__ == "__main__":
    import numpy as np
    main()