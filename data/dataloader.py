# Necessary packages
import numpy as np
import pandas as pd
from keras.datasets import mnist

# 将完整数据变为缺失数据
def binary_sampler(p, rows, cols,seed=None):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    np.random.seed(seed)
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix
# 加载数据
def data_loader(data, miss_rate,seed=None):
    '''Loads datasets and introduce missingness.

    Args:
      - data_name: letter, spam, or mnist
      - miss_rate: the probability of missing components

    Returns:
      data_x: original data
      miss_data_x: data with missing values
      data_m: indicator matrix for missing components
    '''

    # Load data

    # Parameters
    no, dim = data.shape

    # Introduce missing data
    data_m = binary_sampler(1 - miss_rate, no, dim,seed)
    miss_data_x = data.copy()
    miss_data_x[data_m == 0] = np.nan

    return data, miss_data_x, data_m

# # 读取加载数据
# df = pd.read_csv("./data/letter.csv")
# ori_data_x, miss_data_x, data_m = data_loader(df)
