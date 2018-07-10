import numpy as np
import torch
import collections
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

def one_hot(num, num_total):
    arr = np.zeros(num_total)
    arr[num] = 1
    arr = torch.tensor(arr, dtype=torch.float, device=device)
    return arr
