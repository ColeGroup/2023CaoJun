from sklearn.impute import SimpleImputer
from utils import rmse_loss
import numpy as np

# 定义 SimpleImputer
# 0
def fill_0(ori_data_x,missing_data,data_m):
    imputer_0 = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0).fit_transform(missing_data)
    rmse_0 = round(rmse_loss(ori_data_x.values, imputer_0, data_m), 5)
    print('0:', rmse_0)
    return rmse_0
# mean
def fill_mean(ori_data_x,missing_data,data_m):
    imputer_mean = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(missing_data)
    rmse_mean = round(rmse_loss(ori_data_x.values, imputer_mean, data_m),5)
    print('mean:', rmse_mean)
    return rmse_mean
# median
def fill_median(ori_data_x,missing_data,data_m):
    imputer_median = SimpleImputer(missing_values=np.nan, strategy='median').fit_transform(missing_data)
    rmse_median = round(rmse_loss(ori_data_x.values, imputer_median, data_m),5)
    print('median:', rmse_median)
    return rmse_median
# most_frequent
def fill_most_frequent(ori_data_x,missing_data,data_m):
    imputer_most_frequent = SimpleImputer(missing_values=np.nan, strategy='most_frequent').fit_transform(missing_data)
    rmse_most_frequent = round(rmse_loss(ori_data_x.values, imputer_most_frequent, data_m),5)
    print('most_frequent:', rmse_most_frequent)
    return rmse_most_frequent


