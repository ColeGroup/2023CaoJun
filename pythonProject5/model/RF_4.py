import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from utils import *
from tqdm import tqdm
from data.dataloader import data_loader
import pandas as pd
from pylab import *
mpl.rcParams['font.sans-serif'] = ['SimHei']
feature_c = ['PERSON_PMS_002','COM_CICI_035','COM_CICI_037','TAX_TW_015','TAX_TW_016','TAX_TW_017','TAX_TW_018','TAX_TW_025','TAX_TW_019','TAX_TW_020','TAX_TW_027','TAX_TW_022','TAX_TW_023','DWSBJN_002','DEBT_DMP_021','DEBT_DMP_025','NSDJ_016','TAX_TW_009','TAX_TW_028','JUSTICE_JNP_023','JUSTICE_JNP_024','JUSTICE_JNP_025','JUSTICE_JNP_026','JUSTICE_JNP_027','JUSTICE_JNP_028','JUSTICE_JNP_029','JUSTICE_JNP_030','JUSTICE_JNP_031','JNSHJXYHB_002','JNSHJXYYBMD_002','JUSTICE_JE_009','JUSTICE_JNP_022','SHTZXY_002','COM_CIBI_033','XWQY_001','TAX_TW_029','TAX_TW_030']
def RF_4(ori_data_x, miss_data_x, data_m,tree,miss):
    view1 = ['PERSON_PMS_002', 'COM_CICI_035', 'COM_CICI_037', 'LRB_020', 'LRB_007']
    view2 = ['LRB_001', 'LRB_022', 'LRB_011', 'LRB_013', 'LRB_009', 'TAX_TW_015', 'TAX_TW_016', 'TAX_TW_017',
             'TAX_TW_001', 'TAX_TW_024', 'TAX_TW_004', 'TAX_TW_003', 'TAX_TW_018', 'TAX_TW_025', 'TAX_TW_019',
             'TAX_TW_020', 'TAX_TW_027', 'TAX_TW_022', 'TAX_TW_023', 'OPT_ARCSS_001', 'DWSBJN_002', 'DEBT_DMP_021',
             'DEBT_DMP_025']
    view3 = ['NSDJ_016', 'TAX_TW_009', 'TAX_TW_028', 'JUSTICE_JNP_023', 'JUSTICE_JNP_024', 'JUSTICE_JNP_025',
             'JUSTICE_JNP_026', 'JUSTICE_JNP_027', 'JUSTICE_JNP_028', 'JUSTICE_JNP_029', 'JUSTICE_JNP_030',
             'JUSTICE_JNP_031', 'JNSHJXYHB_002', 'JNSHJXYYBMD_002', 'JUSTICE_JE_009', 'JUSTICE_JNP_022',
             'SHTZXY_002']
    view4 = ['LRB_002', 'LRB_004', 'LRB_021', 'LRB_023']
    view5 = ['COM_CIBI_006', 'COM_CIBI_033', 'XWQY_001', 'TAX_TW_029', 'TAX_TW_030', 'COM_CIBI_002']
    view6 = ['LRB_024', 'LRB_025', 'LRB_026', 'ZCFZB_034', 'TAX_TW_031', 'DL_002']
    iterable_count = [1]
    rmse_iter = []
    r=[]
    X_missing_fill_nan_fi = miss_data_x.copy()
    # 按照当前列缺失值的数量进行升序排列
    sort_index = X_missing_fill_nan_fi.isnull().sum().sort_values().index
    for j in tqdm(sort_index):
        if X_missing_fill_nan_fi.loc[:, j].isnull().sum() != 0:
            # 将当前列作为目标值
            fill = X_missing_fill_nan_fi.loc[:, j]
            # 将其余列作为特征值（包括目标值）
            df0 = X_missing_fill_nan_fi.loc[:, X_missing_fill_nan_fi.columns != j]
            # 分割视角
            v1 = []
            v2 = []
            v3 = []
            v4 = []
            v5 = []
            v6 = []

            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)
                elif j in view3:
                    v3.append(j)
                elif j in view4:
                    v4.append(j)
                elif j in view5:
                    v5.append(j)
                elif j in view6:
                    v6.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]
            df_view3 = df0.loc[:, v3]
            df_view4 = df0.loc[:, v4]
            df_view5 = df0.loc[:, v5]
            df_view6 = df0.loc[:, v6]

            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)
            df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view3)
            df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view4)
            df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view5)
            df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view6)

            # 将fill中非空的样本作为训练数据
            Y_train = fill[fill.notnull()]
            Y_test = fill[fill.isnull()]
            # 视角1
            X_view1_train = df_view1_0[Y_train.index, :]
            X_view1_test = df_view1_0[Y_test.index, :]
            # 视角2
            X_view2_train = df_view2_0[Y_train.index, :]
            X_view2_test = df_view2_0[Y_test.index, :]
            # 视角3
            X_view3_train = df_view3_0[Y_train.index, :]
            X_view3_test = df_view3_0[Y_test.index, :]
            # 视角4
            X_view4_train = df_view4_0[Y_train.index, :]
            X_view4_test = df_view4_0[Y_test.index, :]
            # 视角5
            X_view5_train = df_view5_0[Y_train.index, :]
            X_view5_test = df_view5_0[Y_test.index, :]
            # 视角6
            X_view6_train = df_view6_0[Y_train.index, :]
            X_view6_test = df_view6_0[Y_test.index, :]

            if j in feature_c:
                rfc1 = RandomForestClassifier(n_estimators=tree)
                rfc1 = rfc1.fit(X_view1_train, Y_train)
                Y_predict_view1 = rfc1.predict(X_view1_test)
                rfc2 = RandomForestClassifier(n_estimators=tree)
                rfc2 = rfc2.fit(X_view2_train, Y_train)
                Y_predict_view2 = rfc2.predict(X_view2_test)
                rfc3 = RandomForestClassifier(n_estimators=tree)
                rfc3 = rfc3.fit(X_view3_train, Y_train)
                Y_predict_view3 = rfc3.predict(X_view3_test)
                rfc4 = RandomForestClassifier(n_estimators=tree)
                rfc4 = rfc4.fit(X_view4_train, Y_train)
                Y_predict_view4 = rfc4.predict(X_view4_test)
                rfc5 = RandomForestClassifier( n_estimators=tree)
                rfc5 = rfc5.fit(X_view5_train, Y_train)
                Y_predict_view5 = rfc5.predict(X_view5_test)
                rfc6 = RandomForestClassifier(n_estimators=tree)
                rfc6 = rfc6.fit(X_view6_train, Y_train)
                Y_predict_view6 = rfc6.predict(X_view6_test)
            else:
                rfc1 = RandomForestRegressor( n_estimators=tree)
                rfc1 = rfc1.fit(X_view1_train, Y_train)
                Y_predict_view1 = rfc1.predict(X_view1_test)
                rfc2 = RandomForestRegressor( n_estimators=tree)
                rfc2 = rfc2.fit(X_view2_train, Y_train)
                Y_predict_view2 = rfc2.predict(X_view2_test)
                rfc3 = RandomForestRegressor( n_estimators=tree)
                rfc3 = rfc3.fit(X_view3_train, Y_train)
                Y_predict_view3 = rfc3.predict(X_view3_test)
                rfc4 = RandomForestRegressor(n_estimators=tree)
                rfc4 = rfc4.fit(X_view4_train, Y_train)
                Y_predict_view4 = rfc4.predict(X_view4_test)
                rfc5 = RandomForestRegressor(n_estimators=tree)
                rfc5 = rfc5.fit(X_view5_train, Y_train)
                Y_predict_view5 = rfc5.predict(X_view5_test)
                rfc6 = RandomForestRegressor(n_estimators=tree)
                rfc6 = rfc6.fit(X_view6_train, Y_train)
                Y_predict_view6 = rfc6.predict(X_view6_test)

            # 对合并的多视角数据构建随机森林计算特征重要性
            if j in feature_c:
                X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
                                          X_view6_train))
                rfc = RandomForestClassifier(n_estimators=tree)
                rfc = rfc.fit(X_view_train, Y_train)
                imp = rfc.feature_importances_
            else:
                X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
                                          X_view6_train))
                rfc = RandomForestRegressor( n_estimators=tree)
                rfc = rfc.fit(X_view_train, Y_train)
                imp = rfc.feature_importances_
            # 计算各视角预测权重
            a = 0
            b = []
            X = [X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1],
                 X_view4_train.shape[1], X_view5_train.shape[1], X_view6_train.shape[1]]
            for x in X:
                a = a + x
                b.append(a)

            weight1 = sum(imp[0:b[0]])
            weight2 = sum(imp[b[0]:b[1]])
            weight3 = sum(imp[b[1]:b[2]])
            weight4 = sum(imp[b[2]:b[3]])
            weight5 = sum(imp[b[3]:b[4]])
            weight6 = sum(imp[b[4]:b[5]])
            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 + \
                        Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS

    iter_fill = X_missing_fill_nan_fi.copy()
    rmse0 = rmse_loss(ori_data_x.values, iter_fill.values, data_m)
    rmse_iter.append(rmse0)
    print('iter_first:', round(rmse0, 5))
    # for time in range(0,5):

    for iterable in range(0, 21):
        rmse_time=0
        count_time=0

        for time in range(0, 5):
            X_missing_fill_nan_ = miss_data_x.copy()
            sort_index = X_missing_fill_nan_.isnull().sum().sort_values().index
            count_time=count_time+1
            for j in tqdm(sort_index):
                if X_missing_fill_nan_.loc[:, j].isnull().sum() != 0:
                    # 将当前列作为目标值
                    fill = X_missing_fill_nan_.loc[:, j]
                    # 将其余列作为特征值（包括目标值）
                    df0 = X_missing_fill_nan_.loc[:, X_missing_fill_nan_.columns != j]
                    df1 = iter_fill.loc[:, iter_fill.columns != j]

                    # 将其余列继续分割为多个视角，方便对每个视角构建缺失森林
                    v1 = []
                    v2 = []
                    v3 = []
                    v4 = []
                    v5 = []
                    v6 = []

                    for j in df0.columns:
                        if j in view1:
                            v1.append(j)
                        elif j in view2:
                            v2.append(j)
                        elif j in view3:
                            v3.append(j)
                        elif j in view4:
                            v4.append(j)
                        elif j in view5:
                            v5.append(j)
                        elif j in view6:
                            v6.append(j)

                    df1_view1 = df1.loc[:, v1].values
                    df1_view2 = df1.loc[:, v2].values
                    df1_view3 = df1.loc[:, v3].values
                    df1_view4 = df1.loc[:, v4].values
                    df1_view5 = df1.loc[:, v5].values
                    df1_view6 = df1.loc[:, v6].values

                    Y_train = fill[fill.notnull()]
                    Y_test = fill[fill.isnull()]
                    # 视角1
                    X_view1_train = df1_view1[Y_train.index, :]
                    X_view1_test = df1_view1[Y_test.index, :]
                    # 视角2
                    X_view2_train = df1_view2[Y_train.index, :]
                    X_view2_test = df1_view2[Y_test.index, :]
                    # 视角3
                    X_view3_train = df1_view3[Y_train.index, :]
                    X_view3_test = df1_view3[Y_test.index, :]
                    # 视角4
                    X_view4_train = df1_view4[Y_train.index, :]
                    X_view4_test = df1_view4[Y_test.index, :]
                    # 视角5
                    X_view5_train = df1_view5[Y_train.index, :]
                    X_view5_test = df1_view5[Y_test.index, :]
                    # 视角6
                    X_view6_train = df1_view6[Y_train.index, :]
                    X_view6_test = df1_view6[Y_test.index, :]

                    # '''
                    # person corrlation
                    # '''
                    # corr_1 = []
                    # corr_2 = []
                    # corr_3 = []
                    # corr_4 = []
                    # corr_5 = []
                    # corr_6 = []
                    # corr_7 = []
                    # index1 = []
                    # index2 = []
                    # index3 = []
                    # index4 = []
                    # index5 = []
                    # index6 = []
                    # index7 = []

                    # for k in range(0, X_view1_train.shape[1]):
                    #     score = correlation(X_view1_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_1.append(score)
                    #         index1.append(k)
                    # X_view1_train_new = X_view1_train[:, index1]
                    # X_view1_test_new = X_view1_test[:, index1]
                    # # from sklearn.feature_selection import SelectKBest
                    #
                    # for k in range(0, X_view2_train.shape[1]):
                    #     score = correlation(X_view2_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_2.append(score)
                    #         index2.append(k)
                    # X_view2_train_new = X_view2_train[:, index2]
                    # X_view2_test_new = X_view2_test[:, index2]
                    # for k in range(0, X_view3_train.shape[1]):
                    #     score = correlation(X_view3_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_3.append(score)
                    #         index3.append(k)
                    # X_view3_train_new = X_view3_train[:, index3]
                    # X_view3_test_new = X_view3_test[:, index3]
                    # for k in range(0, X_view4_train.shape[1]):
                    #     score = correlation(X_view4_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_4.append(score)
                    #         index4.append(k)
                    # X_view4_train_new = X_view4_train[:, index4]
                    # X_view4_test_new = X_view4_test[:, index4]
                    # for k in range(0, X_view5_train.shape[1]):
                    #     score = correlation(X_view5_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_5.append(score)
                    #         index5.append(k)
                    # X_view5_train_new = X_view5_train[:, index5]
                    # X_view5_test_new = X_view5_test[:, index5]
                    # for k in range(0, X_view6_train.shape[1]):
                    #     score = correlation(X_view6_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_6.append(score)
                    #         index6.append(k)
                    # X_view6_train_new = X_view6_train[:, index6]
                    # X_view6_test_new = X_view6_test[:, index6]
                    # for k in range(0, X_view7_train.shape[1]):
                    #     score = correlation(X_view7_train[:, k], Y_train)
                    #     if score > 0.001:
                    #         corr_7.append(score)
                    #         index7.append(k)
                    # X_view7_train_new = X_view7_train[:, index7]
                    # X_view7_test_new = X_view7_test[:, index7]

                    '''
                    利用特征重要性计算多视角权重
                    '''

                    # 对各视角建立随机森林回归树进行训练
                    if j in feature_c:
                        rfc1 = RandomForestClassifier( n_estimators=tree)
                        rfc1 = rfc1.fit(X_view1_train, Y_train)
                        Y_predict_view1 = rfc1.predict(X_view1_test)
                        rfc2 = RandomForestClassifier(n_estimators=tree)
                        rfc2 = rfc2.fit(X_view2_train, Y_train)
                        Y_predict_view2 = rfc2.predict(X_view2_test)
                        rfc3 = RandomForestClassifier( n_estimators=tree)
                        rfc3 = rfc3.fit(X_view3_train, Y_train)
                        Y_predict_view3 = rfc3.predict(X_view3_test)
                        rfc4 = RandomForestClassifier( n_estimators=tree)
                        rfc4 = rfc4.fit(X_view4_train, Y_train)
                        Y_predict_view4 = rfc4.predict(X_view4_test)
                        rfc5 = RandomForestClassifier(n_estimators=tree)
                        rfc5 = rfc5.fit(X_view5_train, Y_train)
                        Y_predict_view5 = rfc5.predict(X_view5_test)
                        rfc6 = RandomForestClassifier(n_estimators=tree)
                        rfc6 = rfc6.fit(X_view6_train, Y_train)
                        Y_predict_view6 = rfc6.predict(X_view6_test)
                    else:
                        rfc1 = RandomForestRegressor( n_estimators=tree)
                        rfc1 = rfc1.fit(X_view1_train, Y_train)
                        Y_predict_view1 = rfc1.predict(X_view1_test)
                        rfc2 = RandomForestRegressor(n_estimators=tree)
                        rfc2 = rfc2.fit(X_view2_train, Y_train)
                        Y_predict_view2 = rfc2.predict(X_view2_test)
                        rfc3 = RandomForestRegressor(n_estimators=tree)
                        rfc3 = rfc3.fit(X_view3_train, Y_train)
                        Y_predict_view3 = rfc3.predict(X_view3_test)
                        rfc4 = RandomForestRegressor(n_estimators=tree)
                        rfc4 = rfc4.fit(X_view4_train, Y_train)
                        Y_predict_view4 = rfc4.predict(X_view4_test)
                        rfc5 = RandomForestRegressor(n_estimators=tree)
                        rfc5 = rfc5.fit(X_view5_train, Y_train)
                        Y_predict_view5 = rfc5.predict(X_view5_test)
                        rfc6 = RandomForestRegressor(n_estimators=tree)
                        rfc6 = rfc6.fit(X_view6_train, Y_train)
                        Y_predict_view6 = rfc6.predict(X_view6_test)

                    # 对合并的多视角数据构建随机森林计算特征重要性
                    if j in feature_c:
                        X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
                                                  X_view6_train))
                        rfc = RandomForestClassifier(n_estimators=tree)
                        rfc = rfc.fit(X_view_train, Y_train)
                        imp = rfc.feature_importances_
                    else:
                        X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
                                                  X_view6_train))
                        rfc = RandomForestRegressor(n_estimators=tree)
                        rfc = rfc.fit(X_view_train, Y_train)
                        imp = rfc.feature_importances_
                    # 计算各视角预测权重 5 23 17 4 6 5
                    a = 0
                    b = []
                    X = [X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1],
                         X_view4_train.shape[1], X_view5_train.shape[1], X_view6_train.shape[1]]
                    for x in X:
                        a = a + x
                        b.append(a)

                    weight1 = sum(imp[0:b[0]])
                    weight2 = sum(imp[b[0]:b[1]])
                    weight3 = sum(imp[b[1]:b[2]])
                    weight4 = sum(imp[b[2]:b[3]])
                    weight5 = sum(imp[b[3]:b[4]])
                    weight6 = sum(imp[b[4]:b[5]])
                    # 多视角集成
                    # 加权得到最终预测值
                    Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 \
                                + Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
                    # 填充缺失值至预测标签
                    X_missing_fill_nan_.loc[Y_test.index, Y_test.name] = Y_predict
            iter_fill = X_missing_fill_nan_
            rmse = rmse_loss(ori_data_x.values, iter_fill.values, data_m)
            rmse_time=rmse_time+rmse
        loss = rmse_time/count_time
        rmse_iter.append(loss)
        iterable_count.append(int(iterable + 2))
        print('iter_'+str(iterable+1)+':', round(loss, 5))
        rmse_iter0 = rmse_iter[:-1]
        rmse_iter1 = rmse_iter[1:]
        r = abs(np.array(rmse_iter1) - np.array(rmse_iter0))
        if r[-1]<=0.0001:
            print('在第',iterable+1,'次结束循环，此时差值为',r[-1])
        else:
            print('迭代了',iterable+1,'次')
            print('当前的差值为',r[-1])
    rmse = rmse_loss(ori_data_x.values, iter_fill.values, data_m)
    print('iter_end:', round(rmse, 5))
    rmse_iter0 = rmse_iter[:-1]
    rmse_iter1 = rmse_iter[1:]
    r = abs(np.array(rmse_iter1)-np.array(rmse_iter0))
    import pylab as pl
    pl.plot(iterable_count, rmse_iter, 'r', label='iter')
    pl.title('缺失率：'+str(miss))
    pl.xlabel('iter')
    pl.ylabel('rmse')
    pl.legend()
    pl.show()
    return rmse_iter,iterable_count,r
