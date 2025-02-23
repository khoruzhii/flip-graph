import torch

def reducible(UVW):
    N, _, r = UVW.shape
    mask = torch.zeros(N, dtype=torch.bool, device=UVW.device)
    for a, b in ((0,1), (0,2), (1,2)):
        keys = (UVW[:, a, :].to(torch.int32) << 16) ^ UVW[:, b, :].to(torch.int32)
        sorted_keys, _ = torch.sort(keys, dim=1)
        mask |= (sorted_keys[:, 1:] == sorted_keys[:, :-1]).any(dim=1)
    return mask

def reduce(UVW):
    # READ
    N, _, r = UVW.shape
    device = UVW.device
    ar_N = torch.arange(N, device=device)
    
    # Find a duplicate pair for each state over channel pairs (0,1), (0,2), (1,2)
    cand_i = torch.empty(N, dtype=torch.long, device=device)
    cand_j = torch.empty(N, dtype=torch.long, device=device)
    cand_ab = torch.empty((N, 2), dtype=torch.long, device=device)
    found = torch.zeros(N, dtype=torch.bool, device=device)
    for a, b in ((0, 1), (0, 2), (1, 2)):
        dup = torch.triu((UVW[:, a, :][:, :, None] == UVW[:, a, :][:, None, :]) &
                         (UVW[:, b, :][:, :, None] == UVW[:, b, :][:, None, :]), diagonal=1)
        dup_flat = dup.view(N, -1)
        has_dup = dup_flat.any(dim=1)
        cand = (~found) & has_dup
        if cand.any():
            idx = cand.nonzero(as_tuple=True)[0]
            pos = dup_flat[idx].float().argmax(dim=1)
            cand_i[idx] = pos // r
            cand_j[idx] = pos % r
            cand_ab[idx, :] = torch.tensor([a, b], device=device)
            found[idx] = True

    # Determine remaining channel: c = 3 - (a+b)
    rem = 3 - (cand_ab[:, 0] + cand_ab[:, 1])
    
    # Remove duplicate triad j from each state.
    idx_all = torch.arange(r, device=device).unsqueeze(0).expand(N, r)
    mask = idx_all != cand_j.unsqueeze(1)
    new_idx = idx_all[mask].view(N, r-1)
    UVW_red = torch.gather(UVW, 2, new_idx.unsqueeze(1).expand(N, 3, r-1))
    
    # Merging: update remaining channel at merged index.
    merge_idx = torch.where(cand_i < cand_j, cand_i, cand_i - 1)
    UVW_red[ar_N, rem, merge_idx] = UVW[ar_N, rem, cand_i] ^ UVW[ar_N, rem, cand_j]
    
    return UVW_red