import torch

# Small tensors: single-threaded torch is fastest and avoids oversubscription with Numba.
torch.set_num_threads(1)
