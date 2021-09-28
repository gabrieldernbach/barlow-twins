import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
from torch.autograd import Function


class MatrixSquareRoot(Function):
    "https://github.com/steveli/pytorch-sqrtm/blob/master/sqrtm.py"
    """Square root of a positive definite matrix.
    NOTE: matrix square root is not differentiable for matrices with
          zero eigenvalues.
    """

    @staticmethod
    def forward(ctx, input):
        m = input.detach().cpu().numpy().astype(np.float_)
        sqrtm = torch.from_numpy(scipy.linalg.sqrtm(m).real).to(input)
        ctx.save_for_backward(sqrtm)
        return sqrtm

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = None
        if ctx.needs_input_grad[0]:
            sqrtm, = ctx.saved_tensors
            sqrtm = sqrtm.data.cpu().numpy().astype(np.float_)
            gm = grad_output.data.cpu().numpy().astype(np.float_)

            # Given a positive semi-definite matrix X,
            # since X = X^{1/2}X^{1/2}, we can compute the gradient of the
            # matrix square root dX^{1/2} by solving the Sylvester equation:
            # dX = (d(X^{1/2})X^{1/2} + X^{1/2}(dX^{1/2}).
            grad_sqrtm = scipy.linalg.solve_sylvester(sqrtm, sqrtm, gm)

            grad_input = torch.from_numpy(grad_sqrtm).to(grad_output)
        return grad_input


sqrtm = MatrixSquareRoot.apply


# def sqrtm(X):
#     vals, vecs = torch.symeig(X, eigenvectors=True)
#     svals = torch.diag(torch.sqrt(vals))
#     sq = vecs @ svals @ vecs.T
#     return (sq + sq.T)/2
#
# def sqrtm(X):
#     u, s, v = torch.linalg.svd(X)
#     s = torch.diag(torch.sqrt(s))
#     return u @ s @ v.T


def wasserstein(A, B):
    eps = torch.eye(len(A), device="cuda") * 0.001
    A = A + eps
    B = B + eps
    B12 = sqrtm(B)
    C = sqrtm(B12 @ A @ B12)
    trace = torch.trace(A + B - 2 * C)
    return trace


class WassersteinCov(nn.Module):
    def __init__(self, norm: bool):
        super(WassersteinCov, self).__init__()
        self.norm = norm

    def forward(self, z_a, z_b):
        if self.norm:
            z_a = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
            z_b = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        A = torch.mm(z_a.T, z_a)
        B = torch.mm(z_b.T, z_b)
        return wasserstein(A, B)


class Riemann(nn.Module):
    def __init__(self, llambda):
        super(Riemann, self).__init__()
        self.llambda = llambda

    def forward(self, z_a, z_b):
        z_a_norm = (z_a - z_a.mean(dim=0)) / z_a.std(dim=0)
        z_b_norm = (z_b - z_b.mean(dim=0)) / z_b.std(dim=0)
        A = torch.mm(z_a_norm.T, z_a_norm).detach().cpu().numpy()
        B = torch.mm(z_b_norm.T, z_b_norm).detach().cpu().numpy()
        C = np.linalg.eigvalsh(A, B)
        breakpoint()
        return None

# def riemann(A, B):
#     return np.sqrt((np.log(np.linalg.eigvalsh(A, B))**2).sum())
