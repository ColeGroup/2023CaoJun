from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from utils import *
from tqdm import tqdm
from sklearn.impute import KNNImputer
from data.dataloader import data_loader
import pandas as pd
# feature_c = ['NSDJ_016','TAX_TW_009','COM_CIBI_003']
feature_c = ['PERSON_PMS_002','COM_CICI_035','COM_CICI_037','TAX_TW_015','TAX_TW_016','TAX_TW_017','TAX_TW_018','TAX_TW_025','TAX_TW_019','TAX_TW_020','TAX_TW_027','TAX_TW_022','TAX_TW_023','DWSBJN_002','DEBT_DMP_021','DEBT_DMP_025','NSDJ_016','TAX_TW_009','TAX_TW_028','JUSTICE_JNP_023','JUSTICE_JNP_024','JUSTICE_JNP_025','JUSTICE_JNP_026','JUSTICE_JNP_027','JUSTICE_JNP_028','JUSTICE_JNP_029','JUSTICE_JNP_030','JUSTICE_JNP_031','JNSHJXYHB_002','JNSHJXYYBMD_002','JUSTICE_JE_009','JUSTICE_JNP_022','SHTZXY_002','COM_CIBI_033','XWQY_001','TAX_TW_029','TAX_TW_030']
def KNN(missing_data):
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(missing_data)
    impute_knn = imputer.transform(missing_data)
    return impute_knn
def RF_3(ori_data_x, miss_data_x, data_m, c='mse',tree=100):
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


            # 使用knn填充其余列缺失值
            df_view1_0 = KNN(df_view1)
            df_view2_0 = KNN(df_view2)
            df_view3_0 = KNN(df_view3)
            df_view4_0 = KNN(df_view4)
            df_view5_0 = KNN(df_view5)
            df_view6_0 = KNN(df_view6)

            # df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view1)
            # df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view2)
            # df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view3)
            # df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view4)
            # df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view5)
            # df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view6)


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

            # 对各视角建立随机森林回归树进行训练
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
                rfc5 = RandomForestClassifier(n_estimators=tree)
                rfc5 = rfc5.fit(X_view5_train, Y_train)
                Y_predict_view5 = rfc5.predict(X_view5_test)
                rfc6 = RandomForestClassifier(n_estimators=tree)
                rfc6 = rfc6.fit(X_view6_train, Y_train)
                Y_predict_view6 = rfc6.predict(X_view6_test)
            else:
                rfc1 = RandomForestRegressor( n_estimators=tree,criterion=c)
                rfc1 = rfc1.fit(X_view1_train, Y_train)
                Y_predict_view1 = rfc1.predict(X_view1_test)
                rfc2 = RandomForestRegressor(n_estimators=tree,criterion=c)
                rfc2 = rfc2.fit(X_view2_train, Y_train)
                Y_predict_view2 = rfc2.predict(X_view2_test)
                rfc3 = RandomForestRegressor(n_estimators=tree,criterion=c)
                rfc3 = rfc3.fit(X_view3_train, Y_train)
                Y_predict_view3 = rfc3.predict(X_view3_test)
                rfc4 = RandomForestRegressor(n_estimators=tree,criterion=c)
                rfc4 = rfc4.fit(X_view4_train, Y_train)
                Y_predict_view4 = rfc4.predict(X_view4_test)
                rfc5 = RandomForestRegressor( n_estimators=tree,criterion=c)
                rfc5 = rfc5.fit(X_view5_train, Y_train)
                Y_predict_view5 = rfc5.predict(X_view5_test)
                rfc6 = RandomForestRegressor(n_estimators=tree,criterion=c)
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
                rfc = RandomForestRegressor( n_estimators=tree,criterion=c)
                rfc = rfc.fit(X_view_train, Y_train)
                imp = rfc.feature_importances_
            # 计算各视角预测权重 5 23 17 4 6 5
            a=0
            b=[]
            X=[X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1],
               X_view4_train.shape[1], X_view5_train.shape[1],X_view6_train.shape[1]]
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
            # Y_predict = (Y_predict_view1  + Y_predict_view2 + Y_predict_view3 + Y_predict_view4 + Y_predict_view4  + Y_predict_view5 + Y_predict_view6)/6
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 + \
                        Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_3 = rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m)
    print('RF_3:', round(rmse_3,5))

    return rmse_3
# def RF_3_fan(ori_data_x, miss_data_x, data_m, tree):
#     view1 = ['PERSON_PMS_002', 'COM_CICI_035', 'COM_CICI_037', 'LRB_020', 'LRB_007']
#     view2 = ['LRB_001', 'LRB_022', 'LRB_011', 'LRB_013', 'LRB_009', 'TAX_TW_015', 'TAX_TW_016', 'TAX_TW_017',
#              'TAX_TW_001', 'TAX_TW_024', 'TAX_TW_004', 'TAX_TW_003', 'TAX_TW_018', 'TAX_TW_025', 'TAX_TW_019',
#              'TAX_TW_020', 'TAX_TW_027', 'TAX_TW_022', 'TAX_TW_023', 'OPT_ARCSS_001', 'DWSBJN_002', 'DEBT_DMP_021',
#              'DEBT_DMP_025']
#     view3 = ['NSDJ_016', 'TAX_TW_009', 'TAX_TW_028', 'JUSTICE_JNP_023', 'JUSTICE_JNP_024', 'JUSTICE_JNP_025',
#              'JUSTICE_JNP_026', 'JUSTICE_JNP_027', 'JUSTICE_JNP_028', 'JUSTICE_JNP_029', 'JUSTICE_JNP_030',
#              'JUSTICE_JNP_031', 'JNSHJXYHB_002', 'JNSHJXYYBMD_002', 'JUSTICE_JE_009', 'JUSTICE_JNP_022',
#              'SHTZXY_002']
#     view4 = ['LRB_002', 'LRB_004', 'LRB_021', 'LRB_023']
#     view5 = ['COM_CIBI_006', 'COM_CIBI_033', 'XWQY_001', 'TAX_TW_029', 'TAX_TW_030', 'COM_CIBI_002']
#     # view6 = ['ZJTX_002', 'CZBZ_002', 'GXJS_002']
#     view6 = ['LRB_024', 'LRB_025', 'LRB_026', 'ZCFZB_034', 'TAX_TW_031', 'DL_002']
#     X_missing_fill_nan_fi = miss_data_x.copy()
#     # 按照当前列缺失值的数量进行升序排列
#     sort_index = X_missing_fill_nan_fi.isnull().sum().sort_values(ascending=False).index
#     # a = X_missing_fill_nan_fi.isnull().sum().sort_values(ascending=False)
#     # print(a)
#     for j in tqdm(sort_index):
#         if X_missing_fill_nan_fi.loc[:, j].isnull().sum() != 0:
#             # 将当前列作为目标值
#             fill = X_missing_fill_nan_fi.loc[:, j]
#             # 将其余列作为特征值（包括目标值）
#             df0 = X_missing_fill_nan_fi.loc[:, X_missing_fill_nan_fi.columns != j]
#             # 分割视角
#             v1 = []
#             v2 = []
#             v3 = []
#             v4 = []
#             v5 = []
#             v6 = []
#
#             for j in df0.columns:
#                 if j in view1:
#                     v1.append(j)
#                 elif j in view2:
#                     v2.append(j)
#                 elif j in view3:
#                     v3.append(j)
#                 elif j in view4:
#                     v4.append(j)
#                 elif j in view5:
#                     v5.append(j)
#                 elif j in view6:
#                     v6.append(j)
#
#             df_view1 = df0.loc[:, v1]
#             df_view2 = df0.loc[:, v2]
#             df_view3 = df0.loc[:, v3]
#             df_view4 = df0.loc[:, v4]
#             df_view5 = df0.loc[:, v5]
#             df_view6 = df0.loc[:, v6]
#
#             # 使用0填补/均值填充其余列缺失值
#             df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view1)
#             df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view2)
#             df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view3)
#             df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view4)
#             df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view5)
#             df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view6)
#
#             # 将fill中非空的样本作为训练数据
#             Y_train = fill[fill.notnull()]
#             Y_test = fill[fill.isnull()]
#             # 视角1
#             X_view1_train = df_view1_0[Y_train.index, :]
#             X_view1_test = df_view1_0[Y_test.index, :]
#             # 视角2
#             X_view2_train = df_view2_0[Y_train.index, :]
#             X_view2_test = df_view2_0[Y_test.index, :]
#             # 视角3
#             X_view3_train = df_view3_0[Y_train.index, :]
#             X_view3_test = df_view3_0[Y_test.index, :]
#             # 视角4
#             X_view4_train = df_view4_0[Y_train.index, :]
#             X_view4_test = df_view4_0[Y_test.index, :]
#             # 视角5
#             X_view5_train = df_view5_0[Y_train.index, :]
#             X_view5_test = df_view5_0[Y_test.index, :]
#             # 视角6
#             X_view6_train = df_view6_0[Y_train.index, :]
#             X_view6_test = df_view6_0[Y_test.index, :]
#
#             # 对各视角建立随机森林回归树进行训练
#
#             if j in feature_c:
#                 rfc1 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc1 = rfc1.fit(X_view1_train, Y_train)
#                 Y_predict_view1 = rfc1.predict(X_view1_test)
#                 rfc2 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc2 = rfc2.fit(X_view2_train, Y_train)
#                 Y_predict_view2 = rfc2.predict(X_view2_test)
#                 rfc3 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc3 = rfc3.fit(X_view3_train, Y_train)
#                 Y_predict_view3 = rfc3.predict(X_view3_test)
#                 rfc4 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc4 = rfc4.fit(X_view4_train, Y_train)
#                 Y_predict_view4 = rfc4.predict(X_view4_test)
#                 rfc5 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc5 = rfc5.fit(X_view5_train, Y_train)
#                 Y_predict_view5 = rfc5.predict(X_view5_test)
#                 rfc6 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc6 = rfc6.fit(X_view6_train, Y_train)
#                 Y_predict_view6 = rfc6.predict(X_view6_test)
#             else:
#                 rfc1 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc1 = rfc1.fit(X_view1_train, Y_train)
#                 Y_predict_view1 = rfc1.predict(X_view1_test)
#                 rfc2 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc2 = rfc2.fit(X_view2_train, Y_train)
#                 Y_predict_view2 = rfc2.predict(X_view2_test)
#                 rfc3 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc3 = rfc3.fit(X_view3_train, Y_train)
#                 Y_predict_view3 = rfc3.predict(X_view3_test)
#                 rfc4 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc4 = rfc4.fit(X_view4_train, Y_train)
#                 Y_predict_view4 = rfc4.predict(X_view4_test)
#                 rfc5 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc5 = rfc5.fit(X_view5_train, Y_train)
#                 Y_predict_view5 = rfc5.predict(X_view5_test)
#                 rfc6 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc6 = rfc6.fit(X_view6_train, Y_train)
#                 Y_predict_view6 = rfc6.predict(X_view6_test)
#             # 对合并的多视角数据构建随机森林计算特征重要性
#             if j in feature_c:
#                 X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
#                                           X_view6_train))
#                 rfc = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc = rfc.fit(X_view_train, Y_train)
#                 imp = rfc.feature_importances_
#             else:
#                 X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
#                                           X_view6_train))
#                 rfc = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc = rfc.fit(X_view_train, Y_train)
#                 imp = rfc.feature_importances_
#             # 计算各视角预测权重 5 23 17 4 6 5
#             a=0
#             b=[]
#             X=[X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1],
#                X_view4_train.shape[1], X_view5_train.shape[1],X_view6_train.shape[1]]
#             for x in X:
#                 a = a + x
#                 b.append(a)
#
#             weight1 = sum(imp[0:b[0]])
#             weight2 = sum(imp[b[0]:b[1]])
#             weight3 = sum(imp[b[1]:b[2]])
#             weight4 = sum(imp[b[2]:b[3]])
#             weight5 = sum(imp[b[3]:b[4]])
#             weight6 = sum(imp[b[4]:b[5]])
#
#             # 多视角集成
#             # 加权得到最终预测值
#             # Y_predict = (Y_predict_view1  + Y_predict_view2 + Y_predict_view3 + Y_predict_view4 + Y_predict_view4  + Y_predict_view5 + Y_predict_view6)/6
#             Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 + \
#                         Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
#             # 填充缺失值至预测标签
#             X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
#     # 计算填补前后的均方根误差REMS
#     rmse_3 = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
#     print('RF_3:', rmse_3)
#
#     return rmse_3
# def RF_3_rand(ori_data_x, miss_data_x, data_m, tree):
#     view1 = ['PERSON_PMS_002', 'COM_CICI_035', 'COM_CICI_037', 'LRB_020', 'LRB_007']
#     view2 = ['LRB_001', 'LRB_022', 'LRB_011', 'LRB_013', 'LRB_009', 'TAX_TW_015', 'TAX_TW_016', 'TAX_TW_017',
#              'TAX_TW_001', 'TAX_TW_024', 'TAX_TW_004', 'TAX_TW_003', 'TAX_TW_018', 'TAX_TW_025', 'TAX_TW_019',
#              'TAX_TW_020', 'TAX_TW_027', 'TAX_TW_022', 'TAX_TW_023', 'OPT_ARCSS_001', 'DWSBJN_002', 'DEBT_DMP_021',
#              'DEBT_DMP_025']
#     view3 = ['NSDJ_016', 'TAX_TW_009', 'TAX_TW_028', 'JUSTICE_JNP_023', 'JUSTICE_JNP_024', 'JUSTICE_JNP_025',
#              'JUSTICE_JNP_026', 'JUSTICE_JNP_027', 'JUSTICE_JNP_028', 'JUSTICE_JNP_029', 'JUSTICE_JNP_030',
#              'JUSTICE_JNP_031', 'JNSHJXYHB_002', 'JNSHJXYYBMD_002', 'JUSTICE_JE_009', 'JUSTICE_JNP_022',
#              'SHTZXY_002']
#     view4 = ['LRB_002', 'LRB_004', 'LRB_021', 'LRB_023']
#     view5 = ['COM_CIBI_006', 'COM_CIBI_033', 'XWQY_001', 'TAX_TW_029', 'TAX_TW_030', 'COM_CIBI_002']
#     # view6 = ['ZJTX_002', 'CZBZ_002', 'GXJS_002']
#     view6 = ['LRB_024', 'LRB_025', 'LRB_026', 'ZCFZB_034', 'TAX_TW_031', 'DL_002']
#     X_missing_fill_nan_fi = miss_data_x.copy()
#     sort_index = X_missing_fill_nan_fi.isnull().sum().index
#     # a=X_missing_fill_nan_fi.isnull().sum()
#     # print(a)
#     for j in tqdm(sort_index):
#         if X_missing_fill_nan_fi.loc[:, j].isnull().sum() != 0:
#             # 将当前列作为目标值
#             fill = X_missing_fill_nan_fi.loc[:, j]
#             # 将其余列作为特征值（包括目标值）
#             df0 = X_missing_fill_nan_fi.loc[:, X_missing_fill_nan_fi.columns != j]
#             # 分割视角
#             v1 = []
#             v2 = []
#             v3 = []
#             v4 = []
#             v5 = []
#             v6 = []
#
#             for j in df0.columns:
#                 if j in view1:
#                     v1.append(j)
#                 elif j in view2:
#                     v2.append(j)
#                 elif j in view3:
#                     v3.append(j)
#                 elif j in view4:
#                     v4.append(j)
#                 elif j in view5:
#                     v5.append(j)
#                 elif j in view6:
#                     v6.append(j)
#
#             df_view1 = df0.loc[:, v1]
#             df_view2 = df0.loc[:, v2]
#             df_view3 = df0.loc[:, v3]
#             df_view4 = df0.loc[:, v4]
#             df_view5 = df0.loc[:, v5]
#             df_view6 = df0.loc[:, v6]
#
#             # 使用0填补/均值填充其余列缺失值
#             df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view1)
#             df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view2)
#             df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view3)
#             df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view4)
#             df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view5)
#             df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#                 df_view6)
#
#             # 将fill中非空的样本作为训练数据
#             Y_train = fill[fill.notnull()]
#             Y_test = fill[fill.isnull()]
#             # 视角1
#             X_view1_train = df_view1_0[Y_train.index, :]
#             X_view1_test = df_view1_0[Y_test.index, :]
#             # 视角2
#             X_view2_train = df_view2_0[Y_train.index, :]
#             X_view2_test = df_view2_0[Y_test.index, :]
#             # 视角3
#             X_view3_train = df_view3_0[Y_train.index, :]
#             X_view3_test = df_view3_0[Y_test.index, :]
#             # 视角4
#             X_view4_train = df_view4_0[Y_train.index, :]
#             X_view4_test = df_view4_0[Y_test.index, :]
#             # 视角5
#             X_view5_train = df_view5_0[Y_train.index, :]
#             X_view5_test = df_view5_0[Y_test.index, :]
#             # 视角6
#             X_view6_train = df_view6_0[Y_train.index, :]
#             X_view6_test = df_view6_0[Y_test.index, :]
#
#             # 对各视角建立随机森林回归树进行训练
#             if j in feature_c:
#                 rfc1 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc1 = rfc1.fit(X_view1_train, Y_train)
#                 Y_predict_view1 = rfc1.predict(X_view1_test)
#                 rfc2 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc2 = rfc2.fit(X_view2_train, Y_train)
#                 Y_predict_view2 = rfc2.predict(X_view2_test)
#                 rfc3 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc3 = rfc3.fit(X_view3_train, Y_train)
#                 Y_predict_view3 = rfc3.predict(X_view3_test)
#                 rfc4 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc4 = rfc4.fit(X_view4_train, Y_train)
#                 Y_predict_view4 = rfc4.predict(X_view4_test)
#                 rfc5 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc5 = rfc5.fit(X_view5_train, Y_train)
#                 Y_predict_view5 = rfc5.predict(X_view5_test)
#                 rfc6 = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc6 = rfc6.fit(X_view6_train, Y_train)
#                 Y_predict_view6 = rfc6.predict(X_view6_test)
#             else:
#                 rfc1 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc1 = rfc1.fit(X_view1_train, Y_train)
#                 Y_predict_view1 = rfc1.predict(X_view1_test)
#                 rfc2 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc2 = rfc2.fit(X_view2_train, Y_train)
#                 Y_predict_view2 = rfc2.predict(X_view2_test)
#                 rfc3 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc3 = rfc3.fit(X_view3_train, Y_train)
#                 Y_predict_view3 = rfc3.predict(X_view3_test)
#                 rfc4 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc4 = rfc4.fit(X_view4_train, Y_train)
#                 Y_predict_view4 = rfc4.predict(X_view4_test)
#                 rfc5 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc5 = rfc5.fit(X_view5_train, Y_train)
#                 Y_predict_view5 = rfc5.predict(X_view5_test)
#                 rfc6 = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc6 = rfc6.fit(X_view6_train, Y_train)
#                 Y_predict_view6 = rfc6.predict(X_view6_test)
#             # 对合并的多视角数据构建随机森林计算特征重要性
#             if j in feature_c:
#                 X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
#                                           X_view6_train))
#                 rfc = RandomForestClassifier(random_state=0, n_estimators=tree)
#                 rfc = rfc.fit(X_view_train, Y_train)
#                 imp = rfc.feature_importances_
#             else:
#                 X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train, X_view4_train, X_view5_train,
#                                           X_view6_train))
#                 rfc = RandomForestRegressor(random_state=0, n_estimators=tree)
#                 rfc = rfc.fit(X_view_train, Y_train)
#                 imp = rfc.feature_importances_
#             # 计算各视角预测权重 5 23 17 4 6 5
#             a=0
#             b=[]
#             X=[X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1],
#                X_view4_train.shape[1], X_view5_train.shape[1],X_view6_train.shape[1]]
#             for x in X:
#                 a = a + x
#                 b.append(a)
#
#             weight1 = sum(imp[0:b[0]])
#             weight2 = sum(imp[b[0]:b[1]])
#             weight3 = sum(imp[b[1]:b[2]])
#             weight4 = sum(imp[b[2]:b[3]])
#             weight5 = sum(imp[b[3]:b[4]])
#             weight6 = sum(imp[b[4]:b[5]])
#
#             # 多视角集成
#             # 加权得到最终预测值
#             # Y_predict = (Y_predict_view1  + Y_predict_view2 + Y_predict_view3 + Y_predict_view4 + Y_predict_view4  + Y_predict_view5 + Y_predict_view6)/6
#             Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 + \
#                         Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
#             # 填充缺失值至预测标签
#             X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
#     # 计算填补前后的均方根误差REMS
#     rmse_3 = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
#     print('RF_3:', rmse_3)
#
#     return rmse_3
#
def RF_3_no(ori_data_x, miss_data_x, data_m, tree):
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
    # view6 = ['ZJTX_002', 'CZBZ_002', 'GXJS_002']
    view6 = ['LRB_024', 'LRB_025', 'LRB_026', 'ZCFZB_034', 'TAX_TW_031', 'DL_002']
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
            # df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view1)
            # df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view2)
            # df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view3)
            # df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view4)
            # df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view5)
            # df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view6)
            df_view1_0 = KNN(df_view1)
            df_view2_0 = KNN(df_view2)
            df_view3_0 = KNN(df_view3)
            df_view4_0 = KNN(df_view4)
            df_view5_0 = KNN(df_view5)
            df_view6_0 = KNN(df_view6)

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

            # 对各视角建立随机森林回归树进行训练
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
                rfc5 = RandomForestClassifier(n_estimators=tree)
                rfc5 = rfc5.fit(X_view5_train, Y_train)
                Y_predict_view5 = rfc5.predict(X_view5_test)
                rfc6 = RandomForestClassifier(n_estimators=tree)
                rfc6 = rfc6.fit(X_view6_train, Y_train)
                Y_predict_view6 = rfc6.predict(X_view6_test)
            else:
                rfc1 = RandomForestRegressor(n_estimators=tree)
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
            Y_predict = (Y_predict_view1  + Y_predict_view2 + Y_predict_view3 +
                         Y_predict_view4 + Y_predict_view4  + Y_predict_view5 + Y_predict_view6)/6
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_3 = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF_3_no_weight:', rmse_3)

    return rmse_3

def RF_3_100leaves(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []
    view3 = []
    for v1 in range(0, 64):
        view1.append('a' + str(v1))
    for v2 in range(0, 64):
        view2.append('b' + str(v2))
    for v3 in range(0, 64):
        view3.append('c' + str(v3))
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

            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)
                elif j in view3:
                    v3.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]
            df_view3 = df0.loc[:, v3]

            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)
            df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view3)
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

            # R1 = []
            # R2 = []
            # R3 = []
            # Sum1 = Sum2 = Sum3 = 0
            # rank1 = []
            # rank2 = []
            # rank3 = []
            # import random
            # def f(a, b, id):
            #     y = []
            #     index = []
            #     for i in range(0, b.shape[0]):
            #         if b[i] < a:
            #             y.append(b[i])
            #             index.append(id[i])
            #         else:
            #             y.append(b[i])
            #             index.append(id[i])
            #             break
            #     return y, index
            #     # view1
            #
            # for k in range(0, X_view1_train.shape[1]):
            #     rank = correlation(X_view1_train[:, k], Y_train)
            #     rank1.append(rank)
            # rank1 = sorted(rank1, reverse=True)
            # index1 = sorted(range(len(rank1)), key=lambda x: rank1[x], reverse=True)
            # for k in range(0, X_view1_train.shape[1]):
            #     if rank1[k] > 0:
            #         Sum1 = Sum1 + rank1[k]
            #         R1.append(Sum1)
            # a = random.uniform(0, Sum1)
            # y, index = f(a, np.array(R1), index1)
            # X_view1_train_new = X_view1_train[:, index]
            # X_view1_test_new = X_view1_test[:, index]
            # # view2
            # for k in range(0, X_view2_train.shape[1]):
            #     rank = correlation(X_view2_train[:, k], Y_train)
            #     rank2.append(rank)
            # rank2 = sorted(rank2, reverse=True)
            # index2 = sorted(range(len(rank2)), key=lambda x: rank2[x], reverse=True)
            # for k in range(0, X_view2_train.shape[1]):
            #     if rank2[k] > 0:
            #         Sum2 = Sum2 + rank2[k]
            #         R2.append(Sum2)
            # a = random.uniform(0, Sum2)
            # y, index = f(a, np.array(R2), index2)
            # X_view2_train_new = X_view2_train[:, index]
            # X_view2_test_new = X_view2_test[:, index]
            # # view3
            # for k in range(0, X_view3_train.shape[1]):
            #     rank = correlation(X_view3_train[:, k], Y_train)
            #     rank3.append(rank)
            # rank3 = sorted(rank3, reverse=True)
            # index3 = sorted(range(len(rank3)), key=lambda x: rank3[x], reverse=True)
            # for k in range(0, X_view3_train.shape[1]):
            #     if rank3[k] > 0:
            #         Sum3 = Sum3 + rank3[k]
            #         R3.append(Sum3)
            # a = random.uniform(0, Sum3)
            # y, index = f(a, np.array(R3), index3)
            # X_view3_train_new = X_view3_train[:, index]
            # X_view3_test_new = X_view3_test[:, index]

            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor(n_estimators=100)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor(n_estimators=100)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)
            rfc3 = RandomForestRegressor(n_estimators=100)
            rfc3 = rfc3.fit(X_view3_train, Y_train)
            Y_predict_view3 = rfc3.predict(X_view3_test)

            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train))
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算各视角预测权重
            weight1 = sum(imp[0:64])
            weight2 = sum(imp[64:128])
            weight3 = sum(imp[128:191])
            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('rmse_FI:', rmse_fi)

    return rmse_fi


def RF_3_handwritten(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []
    view3 = []
    view4 = []
    view5 = []
    view6 = []
    for v1 in range(0, 250):
        view1.append('a' + str(v1))
    for v2 in range(0, 76):
        view2.append('b' + str(v2))
    for v3 in range(0, 216):
        view3.append('c' + str(v3))
    for v4 in range(0, 47):
        view4.append('d' + str(v4))
    for v5 in range(0, 64):
        view5.append('e' + str(v5))
    for v6 in range(0, 6):
        view6.append('f' + str(v6))
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
            # df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view1)
            # df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view2)
            # df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view3)
            # df_view4_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view4)
            # df_view5_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view5)
            # df_view6_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
            #     df_view6)
            df_view1_0 = KNN(df_view1)
            df_view2_0 = KNN(df_view2)
            df_view3_0 = KNN(df_view3)
            df_view4_0 = KNN(df_view4)
            df_view5_0 = KNN(df_view5)
            df_view6_0 = KNN(df_view6)
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

            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)
            rfc3 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc3 = rfc3.fit(X_view3_train, Y_train)
            Y_predict_view3 = rfc3.predict(X_view3_test)
            rfc4 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc4 = rfc4.fit(X_view4_train, Y_train)
            Y_predict_view4 = rfc4.predict(X_view4_test)
            rfc5 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc5 = rfc5.fit(X_view5_train, Y_train)
            Y_predict_view5 = rfc5.predict(X_view5_test)
            rfc6 = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc6 = rfc6.fit(X_view6_train, Y_train)
            Y_predict_view6 = rfc6.predict(X_view6_test)

            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train))
            rfc = RandomForestRegressor(random_state=0,n_estimators=10)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算多视角权重
            weight1 = sum(imp[0:240])
            weight2 = sum(imp[240:316])
            weight3 = sum(imp[316:532])
            weight4 = sum(imp[532:579])
            weight5 = sum(imp[579:643])
            weight6 = sum(imp[643:648])
            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3 + \
                        Y_predict_view4 * weight4 + Y_predict_view5 * weight5 + Y_predict_view6 * weight6
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('rmse_3:', rmse_fi)

    return rmse_fi


def RF_3_3sources(ori_data_x, miss_data_x, data_m, tree=100):
    view1 = []
    view2 = []
    view3 = []

    # for v1 in range(0, 3560):
    #     view1.append('a' + str(v1))
    # for v2 in range(0, 3631):
    #     view2.append('b' + str(v2))
    # for v3 in range(0, 3068):
    #     view3.append('c' + str(v3))
    for v1 in range(0, 100):
        view1.append('a' + str(v1))
    for v2 in range(0, 100):
        view2.append('b' + str(v2))
    for v3 in range(0, 100):
        view3.append('c' + str(v3))
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

            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)
                elif j in view3:
                    v3.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]
            df_view3 = df0.loc[:, v3]

            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)
            df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view3)

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

            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestClassifier(n_estimators=tree)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestClassifier(n_estimators=tree)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)
            rfc3 = RandomForestClassifier(n_estimators=tree)
            rfc3 = rfc3.fit(X_view3_train, Y_train)
            Y_predict_view3 = rfc3.predict(X_view3_test)
            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train))
            rfc = RandomForestClassifier(n_estimators=tree)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算多视角权重
            # weight1 = sum(imp[0:3559])
            # weight2 = sum(imp[3560:7190])
            # weight3 = sum(imp[7191:10258])
            weight1 = sum(imp[0:100])
            weight2 = sum(imp[100:200])
            weight3 = sum(imp[200:300])
            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF_3_3sources:', rmse_fi)

    return rmse_fi


def RF_3_wiki(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []

    for v1 in range(0, 128):
        view1.append('a' + str(v1))
    for v2 in range(0, 10):
        view2.append('b' + str(v2))

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

            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]

            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)

            # 将fill中非空的样本作为训练数据
            Y_train = fill[fill.notnull()]
            Y_test = fill[fill.isnull()]
            # 视角1
            X_view1_train = df_view1_0[Y_train.index, :]
            X_view1_test = df_view1_0[Y_test.index, :]
            # 视角2
            X_view2_train = df_view2_0[Y_train.index, :]
            X_view2_test = df_view2_0[Y_test.index, :]

            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor(n_estimators=10)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor(n_estimators=10)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)

            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train))
            rfc = RandomForestRegressor(n_estimators=10)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算多视角权重
            weight1 = sum(imp[0:128])
            weight2 = sum(imp[128:])

            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF_3_wiki:', rmse_fi)

    return rmse_fi


# def RF_3_webkb(ori_data_x, miss_data_x, data_m):
#     view1 = []
#     view2 = []
#
#     for v1 in range(0, 195):
#         view1.append('a' + str(v1))
#     for v2 in range(0, 1703):
#         view2.append('b' + str(v2))
#
#     X_missing_fill_nan_fi = miss_data_x.copy()
#     # 按照当前列缺失值的数量进行升序排列
#     sort_index = X_missing_fill_nan_fi.isnull().sum().sort_values().index
#     for j in tqdm(sort_index):
#         if X_missing_fill_nan_fi.loc[:, j].isnull().sum() != 0:
#             # 将当前列作为目标值
#             fill = X_missing_fill_nan_fi.loc[:, j]
#             # 将其余列作为特征值（包括目标值）
#             df0 = X_missing_fill_nan_fi.loc[:, X_missing_fill_nan_fi.columns != j]
#             # 分割视角
#             v1 = []
#             v2 = []
#
#             for j in df0.columns:
#                 if j in view1:
#                     v1.append(j)
#                 elif j in view2:
#                     v2.append(j)
#
#             df_view1 = df0.loc[:, v1]
#             df_view2 = df0.loc[:, v2]
#
#             # 使用0填补/均值填充其余列缺失值
#             # df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#             #     df_view1)
#             # df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
#             #     df_view2)
#             df_view1_0 = KNN(df_view1)
#             df_view2_0 = KNN(df_view2)
#
#             # 将fill中非空的样本作为训练数据
#             Y_train = fill[fill.notnull()]
#             Y_test = fill[fill.isnull()]
#             # 视角1
#             X_view1_train = df_view1_0[Y_train.index, :]
#             X_view1_test = df_view1_0[Y_test.index, :]
#             # 视角2
#             X_view2_train = df_view2_0[Y_train.index, :]
#             X_view2_test = df_view2_0[Y_test.index, :]
#
#             # 对各视角建立随机森林回归树进行训练
#             rfc1 = RandomForestRegressor(n_estimators=100)
#             rfc1 = rfc1.fit(X_view1_train, Y_train)
#             Y_predict_view1 = rfc1.predict(X_view1_test)
#             rfc2 = RandomForestRegressor(n_estimators=100)
#             rfc2 = rfc2.fit(X_view2_train, Y_train)
#             Y_predict_view2 = rfc2.predict(X_view2_test)
#
#             # 对合并的多视角数据构建随机森林计算特征重要性
#             X_view_train = np.hstack((X_view1_train, X_view2_train))
#             rfc = RandomForestRegressor(n_estimators=100)
#             rfc = rfc.fit(X_view_train, Y_train)
#             imp = rfc.feature_importances_
#             # 计算多视角权重
#             weight1 = sum(imp[0:195])
#             weight2 = sum(imp[195:])
#
#             # 多视角集成
#             # 加权得到最终预测值
#             Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2
#             # 填充缺失值至预测标签
#             X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
#     # 计算填补前后的均方根误差REMS
#     rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
#     print('RF_3_webkb:', rmse_fi)
#
#     return rmse_fi

def RF_3_webkb(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []
    view3 = []
    for v1 in range(0, 1703):
        view1.append('a' + str(v1))
    for v2 in range(0, 230):
        view2.append('b' + str(v2))
    for v3 in range(0, 230):
        view3.append('c' + str(v3))
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
            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)
                elif j in view3:
                    v3.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]
            df_view3 = df0.loc[:, v3]
            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)
            df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view3)
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
            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor(n_estimators=100)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor(n_estimators=100)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)
            rfc3 = RandomForestRegressor(n_estimators=100)
            rfc3 = rfc3.fit(X_view3_train, Y_train)
            Y_predict_view3 = rfc3.predict(X_view3_test)
            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train))
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算多视角权重
            a = 0
            b = []
            X = [X_view1_train.shape[1], X_view2_train.shape[1], X_view3_train.shape[1]]
            for x in X:
                a += x
                b.append(a)
            weight1 = sum(imp[0:b[0]])
            weight2 = sum(imp[b[0]:b[1]])
            weight3 = sum(imp[b[1]:b[2]])

            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF_3_webkb:', rmse_fi)

    return rmse_fi

def RF_3_BBCsport(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []

    # for v1 in range(0, 3183):
    #     view1.append('a' + str(v1))
    # for v2 in range(0, 3203):
    #     view2.append('b' + str(v2))
    for v1 in range(0, 100):
        view1.append('a' + str(v1))
    for v2 in range(0, 100):
        view2.append('b' + str(v2))

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

            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]

            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)

            # 将fill中非空的样本作为训练数据
            Y_train = fill[fill.notnull()]
            Y_test = fill[fill.isnull()]
            # 视角1
            X_view1_train = df_view1_0[Y_train.index, :]
            X_view1_test = df_view1_0[Y_test.index, :]
            # 视角2
            X_view2_train = df_view2_0[Y_train.index, :]
            X_view2_test = df_view2_0[Y_test.index, :]

            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor(n_estimators=100)
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor(n_estimators=100)
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)

            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train))
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算各视角预测权重
            # weight1 = sum(imp[0:3183])
            # weight2 = sum(imp[3183:6386])
            weight1 = sum(imp[0:100])
            weight2 = sum(imp[100:200])

            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF3:', rmse_fi)

    return rmse_fi


def RF_3_news(ori_data_x, miss_data_x, data_m):
    view1 = []
    view2 = []
    view3 = []

    for v1 in range(0, 2000):
        view1.append('a' + str(v1))
    for v2 in range(0, 2000):
        view2.append('b' + str(v2))
    for v3 in range(0, 2000):
        view3.append('c' + str(v3))

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
            for j in df0.columns:
                if j in view1:
                    v1.append(j)
                elif j in view2:
                    v2.append(j)
                elif j in view3:
                    v3.append(j)

            df_view1 = df0.loc[:, v1]
            df_view2 = df0.loc[:, v2]
            df_view3 = df0.loc[:, v3]
            # 使用0填补/均值填充其余列缺失值
            df_view1_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view1)
            df_view2_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view2)
            df_view3_0 = SimpleImputer(missing_values=np.nan, strategy='mean').fit_transform(
                df_view3)
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
            # 对各视角建立随机森林回归树进行训练
            rfc1 = RandomForestRegressor()
            rfc1 = rfc1.fit(X_view1_train, Y_train)
            Y_predict_view1 = rfc1.predict(X_view1_test)
            rfc2 = RandomForestRegressor()
            rfc2 = rfc2.fit(X_view2_train, Y_train)
            Y_predict_view2 = rfc2.predict(X_view2_test)
            rfc3 = RandomForestRegressor()
            rfc3 = rfc3.fit(X_view3_train, Y_train)
            Y_predict_view3 = rfc3.predict(X_view3_test)
            # 对合并的多视角数据构建随机森林计算特征重要性
            X_view_train = np.hstack((X_view1_train, X_view2_train, X_view3_train))
            rfc = RandomForestRegressor()
            rfc = rfc.fit(X_view_train, Y_train)
            imp = rfc.feature_importances_
            # 计算多视角权重
            weight1 = sum(imp[0:2000])
            weight2 = sum(imp[2000:4000])
            weight3 = sum(imp[4000:6000])

            # 多视角集成
            # 加权得到最终预测值
            Y_predict = Y_predict_view1 * weight1 + Y_predict_view2 * weight2 + Y_predict_view3 * weight3
            # 填充缺失值至预测标签
            X_missing_fill_nan_fi.loc[Y_test.index, Y_test.name] = Y_predict
    # 计算填补前后的均方根误差REMS
    rmse_fi = round(rmse_loss(ori_data_x.values, X_missing_fill_nan_fi.values, data_m), 5)
    print('RF_3_news:', rmse_fi)

    return rmse_fi
