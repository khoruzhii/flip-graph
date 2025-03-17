import torch

def split(VUW, shape):
    """
    Splits one triple into two: (v, u, w) -> (v, u, r), (v, u, w ^ r)
    
    Args:
        VUW (torch.Tensor): Input tensor of dimension (N, 3, r).
        shape (tuple): The dimensions of the original matrices.
    
    Returns:
        torch.Tensor: The modified tensor with additional values.
    """
    ranges = [
        1 << (shape[0] * shape[1]),
        1 << (shape[1] * shape[2]),
        1 << (shape[2] * shape[0])
    ]
    N, _, r = VUW.shape
    c = torch.randint(0, 3, (N,))
    pos = torch.randint(0, r, (N,))
    mask = torch.zeros_like(VUW, dtype=torch.bool)
    mask[torch.arange(N), c, pos] = True
    random_numbers = torch.zeros(N, dtype=torch.int32)
    for i in range(3):
        mask = (c == i)
        if mask.any():
            random_numbers[mask] = torch.randint(0, ranges[i], (mask.sum(),), dtype=torch.int32)
    result = VUW.clone()
    result[torch.arange(N), c, pos] ^= random_numbers
    old_triads = result[torch.arange(N), :, pos]
    old_triads = old_triads.unsqueeze(2)
    result[torch.arange(N), c, pos] = random_numbers
    result = torch.cat((result, old_triads), dim=2)
    return result
