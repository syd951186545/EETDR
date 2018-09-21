import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker
import time

TensorX = np.load("E:/PYworkspace/EETDR/data/preprodata/TensorX1000.npy")

a = time.time()
core, factors = tl.decomposition.tucker(TensorX, rank=[96, 96, 65], n_iter_max=20, verbose=True)
# core1, factors1 = tl.decomposition.tucker(TensorX1, rank=[96, 96, 64], n_iter_max=20, verbose=True)
b = time.time()
print(b - a)
np.save("E:/PYworkspace/EETDR/data/preprodata/core1000", core)
np.save("E:/PYworkspace/EETDR/data/preprodata/initU1000", factors[0])
np.save("E:/PYworkspace/EETDR/data/preprodata/initI1000", factors[1])
np.save("E:/PYworkspace/EETDR/data/preprodata/initA1000", factors[2])
TensorX_approximation = tl.tucker_to_tensor(core,factors)
print(TensorX_approximation[1][0][104])
print(TensorX[1][0][104])
print(TensorX_approximation[2][0][104])
print(TensorX[2][0][104])
print(TensorX_approximation[3][0][104])
print(TensorX[3][0][104])
print(TensorX_approximation[5][0][104])
print(TensorX[5][0][104])