# coding=UTF8
from model.RF_1 import RF_1
from model.RF_2 import RF_2
from model.RF_3_0 import RF_3,RF_3_no
import pylab as pl
from data.dataloader_2 import *
'''对比各方法对mydata填补效果'''
missing_rate = []
RMSE_rf_1 = []
RMSE_rf_2 = []
RMSE_rf_3 = []
RMSE_rf_4 = []
rf1=0
rf2=0
rf3=0
rf4=0
count=0


for i in range(1, 10, 1):
    # Load data and introduce missing ness
    missing_rate.append(i / 10)
    print('missing rate:', i / 10)
    # 加载数据
    ori_data_x, miss_data_x, data_m \
        , miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6 = multi_view_data(i)
    for epoch in range(0, 1):
        count = count + 1

        '''rf 1.0 '''
        # 多视角特征拼接后使用缺失森林填补
        imputed_RF_1 = miss_data_x.copy()
        rmse_rf_1 = RF_1(ori_data_x, imputed_RF_1, data_m)
        rf1 = rf1 + rmse_rf_1
        '''rf 2.0 '''
        # 对每个单独的视角使用缺失森林进行填补
        imputed_RF_2 = miss_data_x.copy()
        rmse_rf_2 = RF_2(ori_data_x, miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4,
                         miss_data_x5, miss_data_x6, data_m)
        rf2 = rf2 + rmse_rf_2
        '''rf 3.0 '''
        imputed_RF_3 = miss_data_x.copy()
        rmse_rf_3 = RF_3(ori_data_x, imputed_RF_3, data_m,tree=100)
        rf3 = rf3 + rmse_rf_3
        '''rf 4.0 '''
        imputed_RF_4 = miss_data_x.copy()
        rmse_rf_4 = RF_3_no(ori_data_x, imputed_RF_4, data_m,tree=100)
        rf4 = rf4 + rmse_rf_4

    RMSE_rf_1.append(round(rf1 / count, 5))
    RMSE_rf_2.append(round(rf2 / count, 5))
    RMSE_rf_3.append(round(rf3 / count, 5))
    RMSE_rf_4.append(round(rf4 / count, 5))


# 可视化
pl.plot(missing_rate, RMSE_rf_1, 'g', label='RF1')
pl.plot(missing_rate, RMSE_rf_2, 'b', label='RF2')
pl.plot(missing_rate, RMSE_rf_3, 'r', label='RF-weight')
pl.plot(missing_rate, RMSE_rf_4, 'b', label='RF-mean')
pl.title('不同RF多视角填补策略对填补结果的影响')
pl.xlabel('缺失率')
pl.ylabel('NRMSE')
pl.legend()
pl.show()