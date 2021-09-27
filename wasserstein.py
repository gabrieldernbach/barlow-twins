import torch
import torch.nn as nn

def sqrtm(X):
    # even with eps this is unstable?
    # eps = torch.eye(len(X), device="cuda") * 1e-3
    # vals, vecs = torch.symeig(X + eps, eigenvectors=True)
    u, s, v = torch.svd(X)
    s = torch.diag(torch.sqrt(s))
    return u @ s @ v.T

def wasserstein(A, B):
    B12 = sqrtm(B)
    C = sqrtm(B12 @ A @ B12)
    return torch.sqrt(torch.trace(A + B - 2 * C))

class WassersteinBarlow(nn.Module):
    def __init__(self, llambda):
        super().__init__()
        self.llambda = llambda

    def __call__(self, z_a, z_b):
        z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        A = torch.mm(z_a_norm.T, z_a_norm)
        B = torch.mm(z_b_norm.T, z_b_norm)

        n_batch, n_dim, = z_a.shape
        C = torch.mm(z_a_norm.T, z_b_norm) / n_batch
        C[~torch.eye(n_dim, dtype=torch.bool)] *= self.llambda

        loss_ab = wasserstein(A, B)
        loss_c = wasserstein(C, torch.eye(n_dim, device="cuda"))
        return loss_ab + loss_c
