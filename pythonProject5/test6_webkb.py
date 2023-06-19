# coding=UTF8
from model.gain import gain
from model.SimpleImputer import *
from model.KNN import *
from model.RF_1 import *
from model.RF_2 import *
from model.RF_3_0 import *
from model.RF_4 import *
import pylab as pl
from data.dataloader_2 import *

'''对比各方法对webkb填补效果'''
missing_rate = []
RMSE_rf_1 = []
RMSE_rf_2 = []
RMSE_rf_3 = []
RMSE_knn = []
RMSE_gain = []
RMSE_mean = []
RMSE_mice=[]
RMSE_em=[]
rf1 = 0
rf2 = 0
rf3 = 0
mean = 0
count = 0
knn_count = 0
gain_count = 0
em_count=0
mice_count=0
for i in range(1, 10, 1):
    # Load data and introduce missing ness
    missing_rate.append(i / 10)
    print('missing rate:', i / 10)
    # 加载数据
    ori_data_x, miss_data_x, data_m \
        , miss_data_x1, miss_data_x2,miss_data_x3 = multi_view_data_webkb(i)
    for epoch in range(0, 1):
        count = count + 1
        '''rf 1.0 '''
        # 多视角特征拼接后使用缺失森林填补
        imputed_RF_1 = miss_data_x.copy()
        rmse_rf_1 = RF_1_r(ori_data_x, imputed_RF_1, data_m,tree=100)
        rf1 = rf1 + rmse_rf_1
        # 对每个单独的视角使用缺失森林进行填补
        '''rf 2.0'''
        # imputed_RF_2 = miss_data_x.copy()
        # rmse_rf_2 = RF_2_webkb(ori_data_x, miss_data_x1, miss_data_x2, data_m)
        # rf2 = rf2 + rmse_rf_2
        # 基于特征相关性对每个视角构建随机森林预测缺失值
        # 然后基于特征重要性加权集成每个视角的缺失值得到最终得缺失值
        '''rf 3.0 '''
        imputed_RF_3 = miss_data_x.copy()
        rmse_rf_3 = RF_3_webkb(ori_data_x, imputed_RF_3, data_m)
        rf3 = rf3 + rmse_rf_3

        '''KNN'''
        # KNN
        imputed_knn = miss_data_x.copy()
        rmse_knn = knn(ori_data_x, imputed_knn.values, data_m, n_neighbors=5)
        knn_count = knn_count + rmse_knn
        '''gain'''
        imputed_gain = miss_data_x.copy()
        gain_parameters = {'batch_size': 20,
                           'hint_rate': 0.9,
                           'alpha': 100,
                           'iterations': 10000}
        rmse_gain = gain(ori_data_x, imputed_gain.values, data_m, gain_parameters)
        gain_count = gain_count + rmse_gain
        # '''mean'''
        imputed_mean = miss_data_x.copy()
        rmse_mean = fill_mean(ori_data_x, imputed_mean, data_m)
        mean = mean + rmse_mean
        # # '''mice'''
        # imputed_mice = miss_data_x.copy()
        # rmse_mice = MICE(ori_data_x, imputed_mice.values, data_m)
        # mice_count = mice_count + rmse_mice
        # '''em'''
        # imputed_em = miss_data_x.copy()
        # rmse_em = EM(ori_data_x, imputed_em.values, data_m)
        # em_count = em_count + rmse_em
    print('RMSE_rf_1:', round(rf1 / count, 5))
    print('RMSE_rf_3', round(rf3 / count, 5))
    print('RMSE_knn:', round(knn_count / count, 5))
    print('RMSE_gain:', round(gain_count / count, 5))
    print('RMSE_mean', round(mean / count, 5))
    RMSE_rf_1.append(round(rf1 / count, 5))
    # RMSE_rf_2.append(round(rf2 / count, 5))
    RMSE_rf_3.append(round(rf3 / count, 5))
    # RMSE_rf_4.append(round(rf4 / count, 5))

    RMSE_knn.append(round(knn_count / count, 5))
    RMSE_gain.append(round(gain_count / count, 5))
    RMSE_mean.append(round(mean / count, 5))
    # RMSE_em.append(round(em_count / count, 5))
print('RMSE_rf_1:',RMSE_rf_1)
# print('RMSE_rf_2:',RMSE_rf_2)
print('RMSE_rf_3',RMSE_rf_3)
print('RMSE_knn:',RMSE_knn)
print('RMSE_gain:',RMSE_gain)
print('RMSE_mean',RMSE_mean)
# print(RMSE_em)
# print(RMSE_mice)
# 可视化
pl.plot(missing_rate, RMSE_rf_1, 'g', label='RF')
# pl.plot(missing_rate, RMSE_rf_2, 'g--', label='RF2')
pl.plot(missing_rate, RMSE_rf_3, 'r', label='MVRF')
pl.plot(missing_rate, RMSE_knn, 'c', label='knn')
pl.plot(missing_rate, RMSE_gain, 'y', label='gain')
pl.plot(missing_rate, RMSE_mean, 'b', label='mean')
# pl.plot(missing_rate, RMSE_em, 'b--', label='em')
# pl.plot(missing_rate, RMSE_mice, 'y--', label='mice')
pl.title('webkb')
pl.xlabel('missing_rate')
pl.ylabel('rmse')
pl.legend()
pl.show()