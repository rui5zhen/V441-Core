#!/bin/bash
# 获取当前 conda 环境的路径
CONDA_PREFIX_PATH=$(echo $CONDA_PREFIX)

# 设置环境变量指向 Conda 内部的 CUDA
export CUDA_HOME=$CONDA_PREFIX_PATH
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH

# 再次验证 nvcc 是否显示为 11.8
nvcc --version
