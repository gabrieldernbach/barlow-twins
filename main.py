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
from contextlib import nullcontext
import os

from knn_test import test

# Enable cuDNN autotune
torch.backends.cudnn.benchmark = True


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
    def __init__(self, device: torch.device, batch_size: int = 128):
        is_cuda = device.type == "cuda"
        num_workers_default = max(1, (os.cpu_count() or 1) - 1)
        num_workers = int(os.environ.get("NUM_WORKERS", num_workers_default))
        num_workers = max(0, num_workers)

        train_dataset = CIFAR10(root='data', train=True, download=True, transform=DoubleTransform())
        memory_dataset = CIFAR10(root='data', train=True, download=True, transform=test_tfm)
        test_dataset = CIFAR10(root='data', train=False, download=True, transform=test_tfm)

        common_loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers)
        if is_cuda:
            common_loader_kwargs.update(dict(pin_memory=True))
        if num_workers > 0:
            # Only set when workers are used
            common_loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

        self.train = DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            **common_loader_kwargs,
        )
        self.memory = DataLoader(
            memory_dataset,
            shuffle=False,
            **common_loader_kwargs,
        )
        self.test = DataLoader(
            test_dataset,
            shuffle=False,
            **common_loader_kwargs,
        )


class BarlowTwin(nn.Module):
    def __init__(self, dim=128):
        super(BarlowTwin, self).__init__()
        self.encoder = resnet18()
        # was conv 7x7 stride 2, but CIFAR images are small already
        self.encoder.conv1 = nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.encoder.maxpool = nn.Identity()
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(512, 2048, bias=False),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, dim, bias=True),
        )

    def forward(self, x):
        feature = self.encoder(x)
        out = self.projector(feature)
        return F.normalize(feature, dim=-1), F.normalize(out, dim=-1)


class BarlowLoss(nn.Module):
    def __init__(self, llambda: float, eps: float = 1e-12):
        super(BarlowLoss, self).__init__()
        self.llambda = llambda
        self.eps = eps

    @staticmethod
    def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
        n, m = x.shape
        assert n == m
        # Return a flattened view of off-diagonal elements
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def forward(self, z_a: torch.Tensor, z_b: torch.Tensor) -> torch.Tensor:
        # Normalize batch-wise to zero mean and unit variance with numerical stability
        z_a_norm = (z_a - z_a.mean(dim=0)) / (z_a.std(dim=0) + self.eps)
        z_b_norm = (z_b - z_b.mean(dim=0)) / (z_b.std(dim=0) + self.eps)

        n_batch, n_dim = z_a.shape
        c = torch.mm(z_a_norm.T, z_b_norm) / n_batch

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self._off_diagonal(c).pow_(2).sum()
        return on_diag + self.llambda * off_diag


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Allow TF32 on CUDA (Ampere+) for faster matmul/conv
    if device.type == "cuda":
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

    model = BarlowTwin(dim=128)
    if device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)

    # Optional: torch.compile opt-in via env var TORCH_COMPILE=1
    if os.environ.get("TORCH_COMPILE", "0") == "1" and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    criterion = BarlowLoss(llambda=0.005)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-6)

    use_amp = device.type == "cuda" and os.environ.get("USE_AMP", "0") == "1"
    scaler = GradScaler() if use_amp else None

    epochs = 1000
    best_score = 0.0
    db = DataBunch(device=device)

    # Optional: improve matmul precision/perf on recent PyTorch
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    for epoch in range(epochs):
        model.train()
        loss_avg = 0.0
        batch_load = tqdm(db.train)
        for idx, ((y_a, y_b), _) in enumerate(batch_load, start=1):
            if device.type == "cuda":
                y_a = y_a.cuda(non_blocking=True).to(memory_format=torch.channels_last)
                y_b = y_b.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            else:
                y_a = y_a.to(device)
                y_b = y_b.to(device)

            amp_context = autocast() if use_amp else nullcontext()
            with amp_context:
                _, z_a = model(y_a)
                _, z_b = model(y_b)
                loss = criterion(z_a, z_b)

            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            loss_avg += (loss.item() - loss_avg) / idx
            batch_load.set_description(f"{epoch:3d}, loss={loss_avg:.3f}")

        if epoch % 5 == 0:
            test_acc_1, test_acc_5 = test(model, db)
            if test_acc_1 > best_score:
                best_score = test_acc_1
                torch.save(model.state_dict(), "ckpt.pt")
                print("saving checkpoint")
