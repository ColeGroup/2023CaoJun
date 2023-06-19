'''对比各方法对真实企业金融数据的填补效果'''
 # coding=UTF8
from model.gain import gain
from model.SimpleImputer import *
from model.KNN import *
from model.RF_1 import *
from model.RF_3_0 import *
import pylab as pl
from data.dataloader_2 import *
from pylab import *
from model.IterativeImputer import *
from model.EM import *
from model.pcgain import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
from model.RF_2 import RF_2
seed=11
missing_rate = []
RMSE_rf_1 = []
RMSE_rf_2 = []
RMSE_rf_3 = []
RMSE_rf_4 = []
RMSE_knn = []
RMSE_gain = []
RMSE_mean = []
RMSE_mice=[]
RMSE_em=[]
RMSE_pcgain = []
rf1 = 0
rf2 = 0
rf3 = 0
rf4 = 0
mean = 0
count = 0
knn_count = 0
gain_count = 0
pcgain_count = 0
em_count=0
mice_count=0
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
        # 多视角加权随机森林填补
        imputed_RF_3 = miss_data_x.copy()
        rmse_rf_3 = RF_3(ori_data_x, imputed_RF_3, data_m,tree=100)
        rf3 = rf3 + rmse_rf_3
        '''KNN'''
        imputed_knn = miss_data_x.copy()
        rmse_knn = knn(ori_data_x, imputed_knn.values, data_m,n_neighbors=5)
        knn_count =knn_count+rmse_knn
        '''gain'''
        batch_size={5,10,20,100}
        alpha={100,200}
        iterations={10000,20000}
        imputed_gain = miss_data_x.copy()
        gain_parameters = {'batch_size':9,
                           'hint_rate': 0.2,
                           'alpha': 60,
                           'iterations': 20000}
        rmse_gain = gain(ori_data_x, imputed_gain.values, data_m, gain_parameters)
        gain_count =gain_count+rmse_gain
        '''pc-gain'''
        imputed_pcgain = miss_data_x.copy()
        pc_gain_parameters = {'batch_size': 5, #64
                           'hint_rate': 0.4,  #0.9
                           'alpha': 30,
                           'beta': 70,  #20
                           'lambda_': 0.4,
                           'k': 6,
                           'iterations': 10000,
                           'cluster_species': 'KM'}
        rmse_pcgain = PC_GAIN(ori_data_x, imputed_pcgain.values, data_m, pc_gain_parameters)
        pcgain_count = pcgain_count + rmse_pcgain

        '''mean'''
        imputed_mean = miss_data_x.copy()
        rmse_mean = fill_mean(ori_data_x, imputed_mean, data_m)
        mean = mean + rmse_mean
        '''mice'''
        imputed_mice = miss_data_x.copy()
        rmse_mice = MICE(ori_data_x, imputed_mice.values, data_m)
        mice_count = mice_count + rmse_mice
        '''em'''
        imputed_em = miss_data_x.copy()
        rmse_em = EM(ori_data_x, imputed_em.values, data_m)
        em_count = em_count + rmse_em

    RMSE_rf_1.append(round(rf1 / count, 5))
    RMSE_rf_2.append(round(rf2 / count, 5))
    RMSE_rf_3.append(round(rf3 / count, 5))
    RMSE_knn.append(round(knn_count / count, 5))
    RMSE_gain.append(round(gain_count / count, 5))
    RMSE_pcgain.append(np.round(pcgain_count / count, 5))
    RMSE_mean.append(round(mean / count, 5))
    RMSE_em.append(round(em_count / count, 5))
    RMSE_mice.append(round(mice_count / count, 5))
print('RF1',RMSE_rf_1)
print('RF2:',RMSE_rf_2)
print('WMVRF:',RMSE_rf_3)
print('KNN:',RMSE_knn)
print('GAIN:',RMSE_gain)
print('PCGAIN:',RMSE_pcgain)
print('MEAN:',RMSE_mean)
print('EM:',RMSE_em)
print('MICE',RMSE_mice)

# 可视化
pl.plot(missing_rate, RMSE_rf_1, 'g', label='RF1')
pl.plot(missing_rate, RMSE_rf_2, 'b', label='RF2')
pl.plot(missing_rate, RMSE_rf_3, 'r', label='WMVRF')
pl.plot(missing_rate, RMSE_knn, 'c', label='KNN')
pl.plot(missing_rate, RMSE_gain, 'm', label='GAIN')
pl.plot(missing_rate, RMSE_pcgain, 'm--', label='PC-GAIN')
pl.plot(missing_rate, RMSE_mean, 'y', label='Mean')
pl.plot(missing_rate, RMSE_em, 'k', label='EM')
pl.plot(missing_rate, RMSE_mice, 'k--', label='MICE')
pl.title('多视角随机森林在不同缺失率下填补效果对比')
pl.xlabel('missing_rate')
pl.ylabel('rmse')
# 设置数字标签
# for a, b in zip(missing_rate, RMSE_rf_1):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_rf_2):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_rf_3):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_rf_4):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_knn):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_mean):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
# for a, b in zip(missing_rate, RMSE_gain):
#     pl.text(a, b, b, ha='center', va='bottom', fontsize=8)
pl.legend()
# pl.savefig('./result.png')
pl.show()