#!/usr/bin/env python3
"""
Performance benchmark script for Barlow Twins implementation.
"""

import torch
import time
import argparse
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
import torchvision.transforms as T

from main import BarlowTwin, DataBunch, BarlowLoss
from performance_utils import (
    optimize_torch_settings, print_system_info, print_model_info,
    memory_profiler, timer, warmup_model
)


def benchmark_dataloader(dataloader, num_batches=10, device='cuda'):
    """Benchmark DataLoader performance."""
    print(f"Benchmarking DataLoader (first {num_batches} batches)...")
    
    times = []
    data_transfer_times = []
    
    for i, (data, target) in enumerate(dataloader):
        if i >= num_batches:
            break
            
        start_time = time.time()
        
        # Measure data transfer time
        transfer_start = time.time()
        if isinstance(data, tuple):  # DoubleTransform case
            data = tuple(d.to(device, non_blocking=True) for d in data)
        else:
            data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        transfer_time = time.time() - transfer_start
        total_time = time.time() - start_time
        
        times.append(total_time)
        data_transfer_times.append(transfer_time)
    
    avg_time = sum(times) / len(times)
    avg_transfer_time = sum(data_transfer_times) / len(data_transfer_times)
    
    print(f"  - Average batch time: {avg_time:.4f}s")
    print(f"  - Average transfer time: {avg_transfer_time:.4f}s")
    print(f"  - Batches per second: {1/avg_time:.2f}")
    
    return avg_time, avg_transfer_time


def benchmark_model_forward(model, input_shape, device='cuda', num_iterations=100):
    """Benchmark model forward pass."""
    print(f"Benchmarking model forward pass ({num_iterations} iterations)...")
    
    model.eval()
    dummy_input = torch.randn(input_shape, device=device)
    
    # Warmup
    warmup_model(model, input_shape, device, num_warmup=10)
    
    times = []
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.time()
            _ = model(dummy_input)
            
            if device == 'cuda':
                torch.cuda.synchronize()
            
            times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = input_shape[0] / avg_time  # samples per second
    
    print(f"  - Average forward time: {avg_time:.4f}s")
    print(f"  - Throughput: {throughput:.2f} samples/sec")
    
    return avg_time, throughput


def benchmark_training_step(model, criterion, optimizer, scaler, input_shape, device='cuda', num_iterations=50):
    """Benchmark a complete training step."""
    print(f"Benchmarking training step ({num_iterations} iterations)...")
    
    model.train()
    times = []
    
    for _ in range(num_iterations):
        # Generate dummy data
        y_a = torch.randn(input_shape, device=device)
        y_b = torch.randn(input_shape, device=device)
        
        start_time = time.time()
        
        with torch.cuda.amp.autocast():
            _, z_a = model(y_a)
            _, z_b = model(y_b)
            loss = criterion(z_a, z_b)
        
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        
        times.append(time.time() - start_time)
    
    avg_time = sum(times) / len(times)
    throughput = input_shape[0] / avg_time  # samples per second
    
    print(f"  - Average training step time: {avg_time:.4f}s")
    print(f"  - Training throughput: {throughput:.2f} samples/sec")
    
    return avg_time, throughput


def run_full_benchmark():
    """Run comprehensive performance benchmark."""
    print("=" * 60)
    print("BARLOW TWINS PERFORMANCE BENCHMARK")
    print("=" * 60)
    
    # System info
    optimize_torch_settings()
    print_system_info()
    print()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print()
    
    # Model setup
    model = BarlowTwin(dim=128).to(device)
    print_model_info(model)
    print()
    
    # Try to compile model
    try:
        if hasattr(torch, 'compile'):
            compiled_model = torch.compile(model, mode='reduce-overhead')
            print("Model compiled successfully!")
        else:
            compiled_model = model
            print("torch.compile not available")
    except Exception as e:
        print(f"Model compilation failed: {e}")
        compiled_model = model
    
    criterion = BarlowLoss(llambda=0.005)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()
    
    # Benchmark data loading
    print("\n" + "="*40)
    print("DATA LOADING BENCHMARK")
    print("="*40)
    
    db = DataBunch()
    
    print("\nTraining DataLoader:")
    train_time, train_transfer = benchmark_dataloader(db.train, num_batches=20, device=device)
    
    print("\nTest DataLoader:")
    test_time, test_transfer = benchmark_dataloader(db.test, num_batches=10, device=device)
    
    # Benchmark model performance
    print("\n" + "="*40)
    print("MODEL PERFORMANCE BENCHMARK")
    print("="*40)
    
    batch_size = 256
    input_shape = (batch_size, 3, 32, 32)
    
    print("\nOriginal Model:")
    with memory_profiler(device):
        orig_forward_time, orig_throughput = benchmark_model_forward(
            model, input_shape, device, num_iterations=100
        )
    
    print("\nCompiled Model:")
    with memory_profiler(device):
        comp_forward_time, comp_throughput = benchmark_model_forward(
            compiled_model, input_shape, device, num_iterations=100
        )
    
    # Benchmark training performance
    print("\n" + "="*40)
    print("TRAINING PERFORMANCE BENCHMARK")
    print("="*40)
    
    print("\nTraining Step Performance:")
    with memory_profiler(device):
        train_step_time, train_throughput = benchmark_training_step(
            compiled_model, criterion, optimizer, scaler, input_shape, device, num_iterations=30
        )
    
    # Summary
    print("\n" + "="*60)
    print("BENCHMARK SUMMARY")
    print("="*60)
    
    print(f"Data Loading:")
    print(f"  - Training batch time: {train_time:.4f}s ({1/train_time:.1f} batches/sec)")
    print(f"  - Test batch time: {test_time:.4f}s ({1/test_time:.1f} batches/sec)")
    
    print(f"\nModel Forward Pass:")
    print(f"  - Original model: {orig_forward_time:.4f}s ({orig_throughput:.1f} samples/sec)")
    print(f"  - Compiled model: {comp_forward_time:.4f}s ({comp_throughput:.1f} samples/sec)")
    if comp_forward_time < orig_forward_time:
        speedup = orig_forward_time / comp_forward_time
        print(f"  - Compilation speedup: {speedup:.2f}x")
    
    print(f"\nTraining Performance:")
    print(f"  - Training step time: {train_step_time:.4f}s")
    print(f"  - Training throughput: {train_throughput:.1f} samples/sec")
    
    # Estimate training time
    total_samples = 50000  # CIFAR-10 training samples
    samples_per_epoch = total_samples
    time_per_epoch = samples_per_epoch / train_throughput
    total_training_time = time_per_epoch * 1000 / 3600  # 1000 epochs in hours
    
    print(f"\nTraining Time Estimates:")
    print(f"  - Time per epoch: {time_per_epoch:.1f}s ({time_per_epoch/60:.1f} min)")
    print(f"  - Total training time (1000 epochs): {total_training_time:.1f} hours")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark Barlow Twins performance')
    parser.add_argument('--quick', action='store_true', 
                       help='Run quick benchmark with fewer iterations')
    args = parser.parse_args()
    
    if args.quick:
        print("Running quick benchmark...")
        # Reduce iterations for quick test
        globals()['NUM_ITERATIONS'] = 20
    
    run_full_benchmark()