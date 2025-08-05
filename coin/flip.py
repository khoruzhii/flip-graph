import torch

def flippable(UVW):
    val, _ = torch.sort(UVW, dim=2)
    return (val[..., 1:] == val[..., :-1]).view(UVW.size(0), -1).any(dim=1)

def flip(UVW):
    """
    In-place flip of N decompositions with r triads (u, v, w).
    For each state, for a random channel with a duplicate pair, performs a flip in ℤ₂.
    """
    # init variables
    N, _, r = UVW.shape
    device = UVW.device
    ar_N = torch.arange(N, device=device)
    
    # sort to find pairs with equal vectors in O(r ln(r))
    val, idx = torch.sort(UVW, dim=2)
    mask = val[..., 1:] == val[..., :-1]
    
    # choose random pair through all triad vectors
    flat_pos = torch.argmax((
        (1+torch.rand((N,3,r-1), dtype=torch.float, device=device)) * mask
    ).view(N, -1), dim=1)
    channel, pos = flat_pos // (r - 1), flat_pos % (r - 1)
    idx_sel = idx[ar_N, channel]
    j1, j2 = idx_sel[ar_N, pos], idx_sel[ar_N, pos+1]
    c1 = (channel + 1 + torch.randint(0, 1+1, (N,), device=device)) % 3
    c2 = 3 - c1 - channel
    
    # inplace flip in ℤ₂
    UVW[ar_N, c1, j1] ^= UVW[ar_N, c1, j2]
    UVW[ar_N, c2, j2] ^= UVW[ar_N, c2, j1]
    
    return mask.view(N, -1).sum(dim=1)