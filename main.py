import torch
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.models import resnet18
from tqdm import tqdm
import os

from knn_test import test
from performance_utils import (
    optimize_torch_settings, print_system_info, print_model_info,
    compile_model_if_available, memory_profiler, timer
)

# Optimize CUDA settings for better performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set optimal thread count
torch.set_num_threads(min(8, os.cpu_count()))


class DoubleTransform:
    def __init__(self):
        self.tfm = T.Compose([
            T.RandomResizedCrop(32),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            T.RandomGrayscale(p=0.2),
            T.ToTensor(),
            T.Normalize(
                [0.4914, 0.4822, 0.4465],
                [0.2023, 0.1994, 0.2010]
            )
        ])

    def __call__(self, x):
        return self.tfm(x), self.tfm(x)


test_tfm = T.Compose([
    T.ToTensor(),
    T.Normalize(
        [0.4914, 0.4822, 0.4465],
        [0.2023, 0.1994, 0.2010]
    )
])


class DataBunch:
    # Optimize DataLoader settings for better performance
    train = DataLoader(
        CIFAR10(root='data', train=True, download=True, transform=DoubleTransform()),
        shuffle=True,
        batch_size=256,  # Increased batch size for better GPU utilization
        num_workers=min(8, os.cpu_count()),  # Adaptive worker count
        pin_memory=True,
        persistent_workers=True,  # Keep workers alive between epochs
        prefetch_factor=2,  # Prefetch batches
        drop_last=True,  # Ensure consistent batch sizes
    )
    memory = DataLoader(
        CIFAR10(root='data', train=True, download=True, transform=test_tfm),
        shuffle=False,  # No need to shuffle for memory bank
        batch_size=256,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    test = DataLoader(
        CIFAR10(root='data', train=False, download=True, transform=test_tfm),
        shuffle=False,  # No need to shuffle for testing
        batch_size=256,
        num_workers=min(8, os.cpu_count()),
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )


class BarlowTwin(nn.Module):
    def __init__(self, dim=128):
        super(BarlowTwin, self).__init__()
        self.encoder = resnet18()
        # Optimize for CIFAR-10: smaller conv kernel and no maxpool
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        # Fixed projector architecture - was incorrectly 512->2048
        self.projector = nn.Sequential(
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, dim, bias=True),  # Fixed: was 2048, should be 512
        )

    def forward(self, x):
        feature = self.encoder(x)
        out = self.projector(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class BarlowLoss(nn.Module):
    def __init__(self, llambda):
        super(BarlowLoss, self).__init__()
        self.llambda = llambda

    def __call__(self, z_a, z_b):
        # More numerically stable normalization
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + 1e-8)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + 1e-8)

        n_batch, n_dim = z_a.shape
        
        # More efficient cross-correlation computation
        c = torch.mm(z_a_norm.T, z_b_norm) / n_batch
        
        # Create identity matrix on the same device
        eye = torch.eye(n_dim, device=c.device, dtype=c.dtype)
        c_diff = (c - eye) ** 2
        
        # Apply lambda to off-diagonal elements
        c_diff[~eye.bool()] *= self.llambda
        return c_diff.sum()


if __name__ == "__main__":
    # Apply performance optimizations
    optimize_torch_settings()
    print_system_info()
    print()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BarlowTwin(dim=128).to(device)
    print_model_info(model)
    
    # Compile model for better performance (PyTorch 2.0+)
    model = compile_model_if_available(model)
    criterion = BarlowLoss(llambda=0.005)
    
    # Optimized optimizer settings
    optimizer = torch.optim.AdamW(  # AdamW often performs better than Adam
        params=model.parameters(), 
        lr=1e-3, 
        weight_decay=1e-4,  # Slightly increased weight decay
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000, eta_min=1e-6)
    
    # Mixed precision training with gradient clipping
    scaler = GradScaler()
    max_grad_norm = 1.0  # Gradient clipping threshold

    epochs = 1000
    best_score = 0.
    db = DataBunch()
    
    # Create checkpoint directory
    os.makedirs("checkpoints", exist_ok=True)
    
    for epoch in range(epochs):
        model.train()
        loss_avg = 0.
        batch_load = tqdm(db.train, desc=f"Epoch {epoch:3d}")
        
        for idx, ((y_a, y_b), _) in enumerate(batch_load, start=1):
            # More efficient data transfer
            y_a = y_a.to(device, non_blocking=True)
            y_b = y_b.to(device, non_blocking=True)
            
            with autocast():
                _, z_a = model(y_a)
                _, z_b = model(y_b)
                loss = criterion(z_a, z_b)

            # Zero gradients more efficiently
            optimizer.zero_grad(set_to_none=True)
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Gradient clipping for stability
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(optimizer)
            scaler.update()

            # Update learning rate
            scheduler.step(epoch + idx / len(db.train))

            loss_avg += (loss.item() - loss_avg) / idx
            batch_load.set_description(f"Epoch {epoch:3d}, loss={loss_avg:.3f}, lr={scheduler.get_last_lr()[0]:.2e}")

        # Test every 10 epochs instead of 5 to reduce overhead
        if epoch % 10 == 0 or epoch == epochs - 1:
            test_acc_1, test_acc_5 = test(model, db)
            if test_acc_1 > best_score:
                best_score = test_acc_1
                # Save checkpoint with more information
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'best_score': best_score,
                    'scaler_state_dict': scaler.state_dict(),
                }
                torch.save(checkpoint, "checkpoints/best_model.pt")
                print(f"Saving checkpoint at epoch {epoch} with Acc@1: {test_acc_1:.2f}%")
