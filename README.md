# cuDMRG
Testing cupy-based DMRG implementation.

python >= 3.7, numpy >= 1.19, conda install -c conda-forge cupy cutensor cudatoolkit=11

If Cupy is missing, this project automatically switches to Numpy.

To get some sense of how cuda compares with numpy, it is recommended to run benchmarks in https://gist.github.com/fukatani/4702aa05aed255cd25f42e77d0a22e37

In general, two types of operations are most common in DMRG: (1) tensor multiplication; (2) SVD or Eigen decomposition. For CUDA, the first is super fast and efficient (this project can switch between "transpose + matrix multiplication + transpose", or the cutensor implementation of https://github.com/springer13/tcl). However, SVD or Eigen decomposition does not benefit as much from a GPU, and it comes with big overhead. For dense SVD and Eigen solvers, GPU's speedup against CPU is only visible when matrix size is in the thousands, but this speedup grows with matrix size. Hopefully more sparse solvers can be added in the future, which will likely benefit the block sparse structure of quantum-number DMRG with the massive parallelism of GPU.

Latest benchmark of the current code on a EVGA RTX 3090 FTW3 Ultra:
```
[21:01:54 cuDMRG.apps.dmrgINFO] sweep = 0, E = -43.14847950275731, max_dim = 4
[21:01:54 cuDMRG.apps.dmrgINFO] sweep = 1, E = -44.09681238309175, max_dim = 16
[21:01:55 cuDMRG.apps.dmrgINFO] sweep = 2, E = -44.12544610855322, max_dim = 64
[21:01:59 cuDMRG.apps.dmrgINFO] sweep = 3, E = -44.12734668170454, max_dim = 256
[21:02:17 cuDMRG.apps.dmrgINFO] sweep = 4, E = -44.12767996796869, max_dim = 600
[21:02:57 cuDMRG.apps.dmrgINFO] sweep = 5, E = -44.12773208020941, max_dim = 912
[21:03:51 cuDMRG.apps.dmrgINFO] sweep = 6, E = -44.12773892512647, max_dim = 1000
[21:04:46 cuDMRG.apps.dmrgINFO] sweep = 7, E = -44.12773980075967, max_dim = 1000
[21:05:32 cuDMRG.apps.dmrgINFO] sweep = 8, E = -44.12773988714249, max_dim = 1000
[21:06:07 cuDMRG.apps.dmrgINFO] sweep = 9, E = -44.12773989317154, max_dim = 1000
```
This is actually faster than on CPU with ITensor (built with parallel Intel MKL fully enabled on AMD Ryzen 5600X).
