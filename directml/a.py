import torch_directml

import torch

import time

print(torch.cuda.is_available())

print(torch_directml.device(0))

a = torch.randn(40000000, 1).cpu()

b = torch.randn(40000000, 1).cpu()

t0 = time.time()

c = a * b

print(c)

print(time.time() - t0)

a = a.to(dtype=torch.float32, device=torch_directml.device(0))
b = b.to(dtype=torch.float32, device=torch_directml.device(0))

for i in range(1, 2):
    d = a * b

t1 = time.time()

print(d)

print(time.time() - t1)
