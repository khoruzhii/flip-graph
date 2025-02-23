# FlipGraph: Tensor Decomposition with Random Walks

This repository reproduces the results of [arXiv:2212.01175](https://arxiv.org/abs/2212.01175), but implemented in **PyTorch** with support for **GPU** parallel computing.

## Overview  
The method uses **random walk search with flipping and reduction operations** to find optimal decomposition schemes for matrix multiplication tensors. It successfully finds **optimal multiplication schemes for matrices of size** `n × n` for `n = 2, 3, 4`.

## Example Usage  
Example notebooks demonstrating the method can be found in the **`notebooks/`** directory.  