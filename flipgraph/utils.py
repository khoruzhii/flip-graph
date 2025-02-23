import torch
import numpy as np

def build_tensor(n1, n2, n3):
    i, j, k = np.indices((n1, n2, n3))
    T = torch.zeros((n1 * n3, n1 * n2, n2 * n3), dtype=int)
    T[i * n3 + k, i * n2 + j, j * n3 + k] = 1
    return T

def generate_triads(n1, n2, n3):
    I, J, K = np.indices((n1, n2, n3)).reshape(3, -1)
    return (
        torch.eye(n1 * n3, dtype=torch.int8)[I * n3 + K],
        torch.eye(n1 * n2, dtype=torch.int8)[I * n2 + J],
        torch.eye(n2 * n3, dtype=torch.int8)[J * n3 + K]
    )

def reconstruct(U, V, W):
    return torch.einsum('bri, brj, brk -> bijk', U.half(), V.half(), W.half()) % 2

def int2bin(tensor, nbit=64):
    return (tensor[..., None] >> torch.arange(nbit, device=tensor.device, dtype=tensor.dtype)) & 1

def bin2int(tensor, nbit=64, dim=-1):
    return (tensor * (1 << torch.arange(nbit, device=tensor.device))).sum(dim=dim)

import torch
from .utils import generate_triads, bin2int

def generate_triads_binary(n1, n2, n3, dims, uvw_dtype):
    UVW = generate_triads(n1, n2, n3)
    UVW_binary = torch.stack([bin2int(M, d).to(uvw_dtype) for (M, d) in zip(UVW, dims)])
    return UVW_binary

def check_uvw(UVW, T, dims):
    return (reconstruct(*(int2bin(M, d) for (M, d) in zip(UVW.permute(1, 0, 2), dims))) == T).view(UVW.size(0), -1).all(dim=-1)