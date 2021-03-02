import random
import numpy as np
import torch


def seed_all(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)

<<<<<<< HEAD

def get_mean(lis):
    return sum(lis) / len(lis)
=======
def get_mean(lis):
    return sum(lis) / len(lis)
>>>>>>> 8e24c8c08c5ceac6af5a475eb92aaaf3188f7b2a
