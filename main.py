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

from knn_test import test

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
    train = DataLoader(
        CIFAR10(root='data', train=True, download=True, transform=DoubleTransform()),
        shuffle=True,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
    )
    memory = DataLoader(
        CIFAR10(root='data', train=True, download=True, transform=test_tfm),
        shuffle=True,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
    )
    test = DataLoader(
        CIFAR10(root='data', train=False, download=True, transform=test_tfm),
        shuffle=True,
        batch_size=128,
        num_workers=16,
        pin_memory=True,
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
            nn.Linear(512, 512, bias=False),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Linear(512, dim, bias=True),
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
        z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)

        n_batch, n_dim, = z_a.shape
        c = torch.mm(z_a_norm.T, z_b_norm) / n_batch
        c_diff = (c - torch.eye(n_dim, device="cuda")) ** 2
        c_diff[~torch.eye(n_dim, dtype=torch.bool)] *= self.llambda
        return c_diff.sum()


if __name__ == "__main__":
    model = BarlowTwin(dim=128).cuda()
    criterion = BarlowLoss(llambda=0.005)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=1e-3, weight_decay=1e-6)
    scaler = GradScaler()

    epochs = 1000
    best_score = 0.
    db = DataBunch()
    for epoch in range(epochs):
        loss_avg = 0.
        batch_load = tqdm(db.train)
        for idx, ((y_a, y_b), _) in enumerate(batch_load, start=1):
            with autocast():
                _, z_a = model(y_a.cuda(non_blocking=True))
                _, z_b = model(y_b.cuda(non_blocking=True))
                loss = criterion(z_a, z_b)

            model.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            loss_avg += (loss.item() - loss_avg) / idx
            batch_load.set_description(f"{epoch:3d}, loss={loss_avg:.3f}")

        if epoch % 5 == 0:
            test_acc_1, test_acc_5 = test(model, db)
            if test_acc_1 > best_score:
                best_score = test_acc_1
                torch.save(model.state_dict(), "ckpt.pt")
                print("saving checkpoint")
