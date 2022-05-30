import torch

def exceptEvery(nth, a):
    print(type(a.size(0)))
    m = a.size(0) // nth * nth
    return torch.cat((a[:m].reshape(-1,nth)[:,:nth-1].reshape(-1), a[m:m+nth-1]))

print(exceptEvery(2, torch.arange(11)))