import numpy as np
import math
import tensorflow as tf
import pandas as pd
tf=tf.compat.v1
tf.disable_v2_behavior()

def min_max_scale(x):
    cols = x.shape[1]
    min_record = []
    max_record = []

    for col in range(cols):
        min_val= np.min(x[:,col])
        max_val = np.max(x[:,col])
        x[:,col] = (x[:,col] - min_val)/(max_val - min_val)
        min_record.append(min_val)
        max_record.append(max_val)

    return x, min_record,max_record

def min_max_recover(X, min_vec, max_vec):
    cols = X.shape[1]
    for col in range(cols):
        X[:,col] = X[:,col]*(max_vec[col]-min_vec[col])+min_vec[col]
    return X

def RMSE(original, filled):
    from sklearn.metrics import mean_squared_error
    score = np.sqrt(mean_squared_error(original, filled))
    return score





# 计算RMSE均方根误差
def rmse_loss(ori_data, imputed_data, data_m):
    '''Compute RMSE loss between ori_data and imputed_data

    Args:
      - ori_data: original data without missing values
      - imputed_data: imputed data
      - data_m: indicator matrix for missingness

    Returns:
      - rmse: Root Mean Squared Error
    '''

    ori_data, norm_parameters = normalization(ori_data)
    imputed_data, _ = normalization(imputed_data, norm_parameters)

    # Only for missing values
    nominator = np.sum(((1 - data_m) * ori_data - (1 - data_m) * imputed_data) ** 2)
    denominator = np.sum(1 - data_m)

    rmse = np.sqrt(nominator / float(denominator))

    return rmse


def normalization(data, parameters=None):
    '''Normalize data in [0, 1] range.

    Args:
      - data: original data

    Returns:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization
    '''

    # Parameters
    _, dim = data.shape
    norm_data = data.copy()

    if parameters is None:

        # MixMax normalization
        min_val = np.zeros(dim)
        max_val = np.zeros(dim)

        # For each dimension
        for i in range(dim):
            min_val[i] = np.nanmin(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] - np.nanmin(norm_data[:, i])
            max_val[i] = np.nanmax(norm_data[:, i])
            norm_data[:, i] = norm_data[:, i] / (np.nanmax(norm_data[:, i]) + 1e-6)

            # Return norm_parameters for renormalization
        norm_parameters = {'min_val': min_val,
                           'max_val': max_val}

    else:
        min_val = parameters['min_val']
        max_val = parameters['max_val']

        # For each dimension
        for i in range(dim):
            norm_data[:, i] = norm_data[:, i] - min_val[i]
            norm_data[:, i] = norm_data[:, i] / (max_val[i] + 1e-6)

        norm_parameters = parameters

    return norm_data, norm_parameters


# 求相关系数
def correlation(x, y):
    stdev_x = standard_deviation(x)
    stdev_y = standard_deviation(y)
    if stdev_x > 0 and stdev_y > 0:
        return abs(covariance(x, y) / stdev_x / stdev_y)
    else:
        return 0


# 求标准差
def standard_deviation(x):
    return math.sqrt(variance(x))


# 求协方差
def covariance(x, y):
    n = len(x)
    return dot(de_mean(x), de_mean(y)) / (n - 1)


# 计算平均值
def mean(x):
    return sum(x) / len(x)


# 计算每一项数据与均值的差
def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]


# 辅助计算函数 dot product 、sum_of_squares
def dot(v, w):
    return sum(v_i * w_i for v_i, w_i in zip(v, w))


def sum_of_squares(v):
    return dot(v, v)


# 方差
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations) / (n - 1)

# def weight(view1_corr,view2_corr,count1,count2):
#     view1 = view1_corr/count1
#     view2 = view2_corr/count2
#     weight = view1/view1+view2
#     return weight
# 将非数值属性转换为数值属性
'''
# 遍历数据的列
for colum in X.iteritems():
    # 对非数值型列进行处理
    if colum[1].dtype == np.object_:
        # 拆分出列名和数据
        feature_name, data = colum

        # 将该列转换成字典
        colum = data.map(lambda x: {feature_name: x})
        colum = dv.fit_transform(colum)
        features = dv.get_feature_names_out()

        # 将新创建的列添加进去
        X[features] = colum

        # 删除当前列
        X = X.drop([feature_name], axis=1)

        # 如果原先值是空，则吧所以新添加的列设置为nan
        if list(features).__contains__(feature_name):
            features = list(features)
            features.remove(feature_name)
            features = np.array(features)

        mask = X[features].sum(axis=1) == 0
        X.loc[mask, features] = np.nan
'''



# 平均绝对误差（MAE）
def mae_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mae -- MAE 评价指标
    """

    n = len(y_true)
    mae = sum(np.abs(y_true - y_pred)) / n
    return mae




# 均方误差（MSE）：均方根误差（RMSE）
def mse_value(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mse -- MSE 评价指标
    """

    n = len(y_true)
    mse = sum(np.square(y_true - y_pred)) / n
    return mse

# 平均绝对百分比误差 MAPE
def mape(y_true, y_pred):
    """
    参数:
    y_true -- 测试集目标真实值
    y_pred -- 测试集目标预测值

    返回:
    mape -- MAPE 评价指标
    """

    n = len(y_true)
    mape = sum(np.abs((y_true - y_pred) / y_true)) / n * 100
    return mape

def infogain(x,y):
    def _entropy(values):
        counts = np.bincount(values)
        probs = counts[np.nonzero(counts)] / float(len(values))
        return - np.sum(probs * np.log(probs))

    def _information_gain(feature, y):
        feature_set_indices = np.nonzero(feature)[1]
        feature_not_set_indices = [i for i in feature_range if i not in feature_set_indices]
        entropy_x_set = _entropy(y[feature_set_indices])
        entropy_x_not_set = _entropy(y[feature_not_set_indices])

        return entropy_before - (((len(feature_set_indices) / float(feature_size)) * entropy_x_set)
                                 + ((len(feature_not_set_indices) / float(feature_size)) * entropy_x_not_set))

    feature_size = x.shape[0]
    feature_range = range(0, feature_size)
    entropy_before = _entropy(y)
    # information_gain_scores = []
    score = _information_gain(x, y)
    # for feature in x.T:
    #     information_gain_scores.append(_information_gain(feature, y))
    return score





def renormalization(norm_data, norm_parameters):
    '''Renormalize data from [0, 1] range to the original range.

    Args:
      - norm_data: normalized data
      - norm_parameters: min_val, max_val for each feature for renormalization

    Returns:
      - renorm_data: renormalized original data
    '''

    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape
    renorm_data = norm_data.copy()

    for i in range(dim):
        renorm_data[:, i] = renorm_data[:, i] * (max_val[i] + 1e-6)
        renorm_data[:, i] = renorm_data[:, i] + min_val[i]

    return renorm_data


def rounding(imputed_data, data_x):
    '''Round imputed data for categorical variables.

    Args:
      - imputed_data: imputed data
      - data_x: original data with missing values

    Returns:
      - rounded_data: rounded imputed data
    '''

    _, dim = data_x.shape
    rounded_data = imputed_data.copy()

    for i in range(dim):
        temp = data_x[~np.isnan(data_x[:, i]), i]
        # Only for the categorical variable
        if len(np.unique(temp)) < 20:
            rounded_data[:, i] = np.round(rounded_data[:, i])

    return rounded_data




def xavier_init(size):
    '''Xavier initialization.

    Args:
      - size: vector size

    Returns:
      - initialized random vector.
    '''
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def binary_sampler(p, rows, cols):
    '''Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    '''
    unif_random_matrix = np.random.uniform(0., 1., size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix


def uniform_sampler(low, high, rows, cols):
    '''Sample uniform random variables.

    Args:
      - low: low limit
      - high: high limit
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - uniform_random_matrix: generated uniform random matrix.
    '''
    return np.random.uniform(low, high, size=[rows, cols])


def sample_batch_index(total, batch_size):
    '''Sample index of the mini-batch.

    Args:
      - total: total number of samples
      - batch_size: batch size

    Returns:
      - batch_idx: batch index
    '''
    total_idx = np.random.permutation(total)
    batch_idx = total_idx[:batch_size]
    return batch_idx





