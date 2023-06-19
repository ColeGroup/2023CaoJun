# coding=UTF8
# import warnings
# warnings.filterwarnings('ignore')

from model.RF_3_0 import *
import pylab as pl
from data.dataloader_2 import *

'''比较随机森林回归模型参数变化对填补误差的影响。（参数包括n_eatimator、criterion等）'''

RMSE_2 = []
RMSE_4 = []
RMSE_6 = []
RMSE_8 = []

TREE = []
for miss in range(2, 10, 2):
    RMSE_rf_3 = []
    tree=[]
    print('miss:', miss / 10)
    for i in range(10, 300, 10):
        tree.append(i)
        rf3 = 0
        count = 0
        for epoch in range(0, 5):
            ori_data_x, miss_data_x, data_m \
                , miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6 = multi_view_data(
                miss=miss)
            # Load data and introduce missing ness

            count = count + 1
            '''rf 3.0 tree'''
            imputed_RF_3 = miss_data_x.copy()
            rmse_rf_3 = RF_3(ori_data_x, imputed_RF_3, data_m, tree=i)
            rf3 = rf3 + rmse_rf_3
        RMSE_rf_3.append(round(rf3 / count, 5))
    TREE = tree
    if miss == 2:
        RMSE_2 = RMSE_rf_3
    elif miss == 4:
        RMSE_4 = RMSE_rf_3
    elif miss == 6:
        RMSE_6 = RMSE_rf_3
    elif miss == 8:
        RMSE_8 = RMSE_rf_3
# 可视化
print('tree:', TREE)
print('RMSE_2:', RMSE_2)
print('RMSE_4:', RMSE_4)
print('RMSE_6:', RMSE_6)
print('RMSE_8:', RMSE_8)
pl.plot(TREE, RMSE_2, 'r-', label='2')
pl.plot(TREE, RMSE_4, 'b-', label='4')
pl.plot(TREE, RMSE_6, 'y-', label='6')
pl.plot(TREE, RMSE_8, 'd-', label='8')
pl.title('TREE-RMSE-MISS')
pl.xlabel('count_tree')
pl.ylabel('NRMSE')
pl.legend()
pl.show()
