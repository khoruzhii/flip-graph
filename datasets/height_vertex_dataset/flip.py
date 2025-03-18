import torch

def flip(VUW):
    """
    Changes two triples: (v, u1, w1), (v, u2, w2) -> (v, u1 ^ u2, w1), (v, u2, w1 ^ w2)
    If no change is possible, it does nothing. 
    
    Args:
        VUW (torch.Tensor): Input tensor of dimension (N, 3, r).
    
    Returns:
        torch.Tensor: The modified tensor after the flip operation.
    """
    N, _, r = VUW.shape
    val, idx = torch.sort(VUW, dim=2)
    mask = val[..., 1:] == val[..., :-1]
    has_duplicates = mask.view(N, -1).any(dim=1)
    result = VUW.clone()
    if has_duplicates.any():
        flat_pos = torch.argmax((
            (1 + torch.rand((N, 3, r-1), dtype=torch.float)) * mask
        ).view(N, -1), dim=1)
        channel, pos = flat_pos // (r - 1), flat_pos % (r - 1)
        idx_sel = idx[has_duplicates, channel[has_duplicates]]
        pos_selected = pos[has_duplicates]
        j1 = idx_sel[torch.arange(idx_sel.size(0)), pos_selected]
        j2 = idx_sel[torch.arange(idx_sel.size(0)), (pos_selected + 1) % r]

        c1 = (channel[has_duplicates] + 1 + torch.randint(0, 1+1, (has_duplicates.sum(),))) % 3
        c2 = 3 - c1 - channel[has_duplicates]

        result[has_duplicates, c1, j1] ^= result[has_duplicates, c1, j2]
        result[has_duplicates, c2, j2] ^= result[has_duplicates, c2, j1]
    return result
    
    
