"""
Performance optimization utilities for Barlow Twins training.
"""

import torch
import torch.nn as nn
import time
import psutil
import os
from contextlib import contextmanager
from typing import Dict, Any, Optional


def optimize_torch_settings():
    """Apply optimal PyTorch settings for performance."""
    # Enable optimized attention (if available)
    if hasattr(torch.backends.cuda, 'flash_sdp_enabled'):
        torch.backends.cuda.flash_sdp_enabled(True)
    
    # Enable memory efficient attention
    if hasattr(torch.backends.cuda, 'mem_efficient_sdp_enabled'):
        torch.backends.cuda.mem_efficient_sdp_enabled(True)
    
    # Optimize for inference/training
    torch.jit.set_fusion_strategy([('STATIC', 20), ('DYNAMIC', 20)])
    
    # Set optimal number of threads
    if torch.get_num_threads() != min(8, os.cpu_count()):
        torch.set_num_threads(min(8, os.cpu_count()))
    
    print(f"Optimized torch settings:")
    print(f"  - Threads: {torch.get_num_threads()}")
    print(f"  - CUDNN benchmark: {torch.backends.cudnn.benchmark}")
    print(f"  - CUDNN deterministic: {torch.backends.cudnn.deterministic}")


def get_model_size(model: nn.Module) -> Dict[str, float]:
    """Calculate model size and parameter count."""
    param_count = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    model_size_mb = (param_size + buffer_size) / (1024 ** 2)
    
    return {
        'total_params': param_count,
        'trainable_params': trainable_params,
        'model_size_mb': model_size_mb
    }


def print_model_info(model: nn.Module):
    """Print detailed model information."""
    info = get_model_size(model)
    print(f"Model Information:")
    print(f"  - Total parameters: {info['total_params']:,}")
    print(f"  - Trainable parameters: {info['trainable_params']:,}")
    print(f"  - Model size: {info['model_size_mb']:.2f} MB")


@contextmanager
def memory_profiler(device: torch.device):
    """Context manager for GPU memory profiling."""
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        start_memory = torch.cuda.memory_allocated(device)
        start_peak = torch.cuda.max_memory_allocated(device)
        
        yield
        
        end_memory = torch.cuda.memory_allocated(device)
        end_peak = torch.cuda.max_memory_allocated(device)
        
        print(f"Memory Usage:")
        print(f"  - Start: {start_memory / 1024**2:.2f} MB")
        print(f"  - End: {end_memory / 1024**2:.2f} MB")
        print(f"  - Peak: {end_peak / 1024**2:.2f} MB")
        print(f"  - Difference: {(end_memory - start_memory) / 1024**2:.2f} MB")
    else:
        yield


@contextmanager
def timer(description: str = "Operation"):
    """Context manager for timing operations."""
    start_time = time.time()
    yield
    end_time = time.time()
    print(f"{description} took: {end_time - start_time:.4f} seconds")


def get_system_info():
    """Get system information for performance analysis."""
    info = {
        'cpu_count': os.cpu_count(),
        'cpu_percent': psutil.cpu_percent(interval=1),
        'memory_percent': psutil.virtual_memory().percent,
        'available_memory_gb': psutil.virtual_memory().available / (1024**3),
    }
    
    if torch.cuda.is_available():
        info.update({
            'cuda_device_count': torch.cuda.device_count(),
            'cuda_device_name': torch.cuda.get_device_name(0),
            'cuda_memory_total_gb': torch.cuda.get_device_properties(0).total_memory / (1024**3),
        })
    
    return info


def print_system_info():
    """Print system information."""
    info = get_system_info()
    print("System Information:")
    print(f"  - CPU cores: {info['cpu_count']}")
    print(f"  - CPU usage: {info['cpu_percent']:.1f}%")
    print(f"  - Memory usage: {info['memory_percent']:.1f}%")
    print(f"  - Available memory: {info['available_memory_gb']:.2f} GB")
    
    if 'cuda_device_count' in info:
        print(f"  - CUDA devices: {info['cuda_device_count']}")
        print(f"  - GPU: {info['cuda_device_name']}")
        print(f"  - GPU memory: {info['cuda_memory_total_gb']:.2f} GB")


def optimize_dataloader_workers(dataset_size: int, batch_size: int) -> int:
    """Calculate optimal number of DataLoader workers."""
    # Rule of thumb: 2-4 workers per CPU core, but not more than needed
    max_workers = min(8, os.cpu_count())
    
    # For small datasets, fewer workers might be better
    if dataset_size < 10000:
        return min(2, max_workers)
    elif dataset_size < 50000:
        return min(4, max_workers)
    else:
        return max_workers


def compile_model_if_available(model: nn.Module) -> nn.Module:
    """Compile model with torch.compile if available (PyTorch 2.0+)."""
    if hasattr(torch, 'compile'):
        try:
            print("Compiling model with torch.compile for better performance...")
            # Use 'reduce-overhead' mode for training
            compiled_model = torch.compile(model, mode='reduce-overhead')
            print("Model compilation successful!")
            return compiled_model
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Continuing with uncompiled model...")
            return model
    else:
        print("torch.compile not available (requires PyTorch 2.0+)")
        return model


def warmup_model(model: nn.Module, input_shape: tuple, device: torch.device, num_warmup: int = 10):
    """Warm up the model for consistent performance measurements."""
    print(f"Warming up model with {num_warmup} iterations...")
    model.eval()
    
    with torch.no_grad():
        dummy_input = torch.randn(input_shape, device=device)
        
        # Warmup iterations
        for _ in range(num_warmup):
            _ = model(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
    
    print("Model warmup completed!")


if __name__ == "__main__":
    # Example usage
    print("Performance Utilities Test")
    print("=" * 50)
    
    optimize_torch_settings()
    print()
    print_system_info()