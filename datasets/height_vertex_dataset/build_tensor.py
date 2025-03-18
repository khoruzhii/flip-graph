import torch

def build_tensor(VUW, shape):
    """
    Recovers the original tensor from its decomposition
    !!! code is not optimal, use only for tests !!!

    Args:
        VUW (torch.Tensor): Input tensor of dimension (N, 3, r).
        shape (tuple): The dimensions of the original matrices.
    
     Returns:
        torch.Tensor: The original tensor of dimension (N, n1, n2, n3).
    """
    N, _, r = VUW.shape
    d1, d2, d3 = shape[0] * shape[1], shape[1] * shape[2], shape[2] * shape[0]
    result = torch.zeros(N, d1, d2, d3, dtype=torch.int32)
    for tensor_numb in range(N):
        for i in range(d1):
            for j in range(d2):
                for k in range(d3):
                    for g in range(r):
                        result[tensor_numb, i, j, k] ^= (((VUW[tensor_numb, 0, g] >> i) & (VUW[tensor_numb, 1, g] >> j) & (VUW[tensor_numb, 2, g] >> k)) & 1)
    return result
