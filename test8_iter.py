'''对比各方法对真实企业金融数据的填补效果'''
# coding=UTF8
from model.RF_3_0 import *
from model.RF_4 import *
import pylab as pl
from data.dataloader_2 import *
from pylab import *

mpl.rcParams['font.sans-serif'] = ['SimHei']
seed = 11
missing_rate = []
RMSE_rf_3 = []
RMSE_rf_4 = []
rf3 = 0
rf4 = 0
count = 0

for i in range(2, 10, 2):
    # 加载数据和缺失率
    missing_rate.append(i / 10)
    print('missing rate:', i / 10)
    # 加载数据
    for epoch in range(0, 1):
        ori_data_x, miss_data_x, data_m \
            , miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, \
        miss_data_x5, miss_data_x6 = multi_view_data(i)
        count = count + 1

        '''rf 3.0 '''
        # 多视角加权随机森林填补
        imputed_RF_3 = miss_data_x.copy()
        rmse_rf_3 = RF_3(ori_data_x, imputed_RF_3, data_m, tree=100)
        rf3 = rf3 + rmse_rf_3

        # 多视角加权随机森林填补迭代
        imputed_RF_4 = miss_data_x.copy()
        rmse_rf_4 = RF_4(ori_data_x, imputed_RF_3, data_m, tree=100)
        rf4 = rf4 + rmse_rf_3

    RMSE_rf_3.append(round(rf3 / count, 5))
    RMSE_rf_4.append(round(rf3 / count, 5))

print('WMVRF:', RMSE_rf_3)
print('WMVRF_iter:', RMSE_rf_4)

# 可视化

pl.plot(missing_rate, RMSE_rf_3, 'b', label='WMVRF')
pl.plot(missing_rate, RMSE_rf_4, 'r', label='WMVRF_iter')
pl.title('WMVRF迭代前后填补效果对比')
pl.xlabel('缺失率')
pl.ylabel('NRMSE')
pl.legend()
# pl.savefig('./result.png')
pl.show()