import torch

def plus_transition(VUW, shape):
    """
    Performs the following operation on two triples (v, u, w) for all N decompositions of
    (v1, u1, w1) (v1 ^ v2, u1, w1)
    (v2, u2, w2) -> (v2, u2, w1 ^ w2)
                    (v2, u2 ^ u1, w2)
    
    Args:
        VUW (torch.Tensor): Input tensor of dimension (N, 3, r).
        shape (tuple): The dimensions of the original matrices.
    
    Returns:
        torch.Tensor: The modified tensor after the plus-transition.
    """
    N, _, r = VUW.shape

    pos1 = torch.randint(0, r, (N,))
    pos2 = torch.randint(0, r - 1, (N, ))
    pos2 = torch.where(pos2 >= pos1, pos2 + 1, pos2)

    v1, u1, w1 = VUW[torch.arange(N), 0, pos1], VUW[torch.arange(N), 1, pos1], VUW[torch.arange(N), 1, pos1]
    v2, u2, w2 = VUW[torch.arange(N), 0, pos2], VUW[torch.arange(N), 1, pos2], VUW[torch.arange(N), 1, pos2]

    result = VUW.clone()
    result[torch.arange(N), 0, pos1] = v1 ^ v2
    result[torch.arange(N), 1, pos2] = u1
    result[torch.arange(N), 2, pos2] = w1 ^ w2

    new_triads = torch.stack((v2, u2 ^ u1, w2), dim=1).unsqueeze(2)
    result = torch.cat((result, new_triads), dim=2)

    return result
