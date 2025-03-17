import torch

def generate(shape, N, k):
    """
    Generates random triads of integers in tensor form.
    
    Args:
        shape (tuple): A tuple of three integers specifying the dimensions of the original matrices.
        N (int): The number of triads.
        k (int): The number of values in each triad.
    
    Returns:
        torch.Tensor: Three-dimensional tensor of dimension (N, 3, k).
    """
    tensors = []
    for i in range(3):
        tensor = torch.randint(0, (1 << (shape[i] * shape[(i + 1) % 3])), (N, k), dtype=torch.int32)
        tensors.append(tensor)
    result_tensor = torch.stack(tensors, dim=1)
    return result_tensor