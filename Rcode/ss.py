"""
初始化数据时尝试使用tucker分解一次作为初始值
"""
import tensorly as tl
import numpy as np
from tensorly.decomposition import tucker
import time

TensorX = np.load("E:/PYworkspace/EETDR/data/preprodata/TensorX300.npy")

a = time.time()
core, factors = tl.decomposition.tucker(TensorX, rank=[128, 128, 96], n_iter_max=100, verbose=True)
# core1, factors1 = tl.decomposition.tucker(TensorX1, rank=[96, 96, 64], n_iter_max=20, verbose=True)
b = time.time()
print(b - a)
TensorX_approximation = tl.tucker_to_tensor(core,factors)
print(TensorX_approximation[1][0][104])
print(TensorX[1][0][104])
print(TensorX_approximation[2][0][104])
print(TensorX[2][0][104])
print(TensorX_approximation[3][0][104])
print(TensorX[3][0][104])
print(TensorX_approximation[5][0][104])
print(TensorX[5][0][104])
print(TensorX.sum()-TensorX_approximation.sum())
