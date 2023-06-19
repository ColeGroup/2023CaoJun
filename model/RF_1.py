from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from utils import *
from tqdm import tqdm
from sklearn.impute import KNNImputer
# feature_c = ['NSDJ_016','TAX_TW_009','COM_CIBI_003']
feature_c = ['PERSON_PMS_002','COM_CICI_035','COM_CICI_037','TAX_TW_015','TAX_TW_016','TAX_TW_017','TAX_TW_018','TAX_TW_025','TAX_TW_019','TAX_TW_020','TAX_TW_027','TAX_TW_022','TAX_TW_023','DWSBJN_002','DEBT_DMP_021','DEBT_DMP_025','NSDJ_016','TAX_TW_009','TAX_TW_028','JUSTICE_JNP_023','JUSTICE_JNP_024','JUSTICE_JNP_025','JUSTICE_JNP_026','JUSTICE_JNP_027','JUSTICE_JNP_028','JUSTICE_JNP_029','JUSTICE_JNP_030','JUSTICE_JNP_031','JNSHJXYHB_002','JNSHJXYYBMD_002','JUSTICE_JE_009','JUSTICE_JNP_022','SHTZXY_002','COM_CIBI_033','XWQY_001','TAX_TW_029','TAX_TW_030']

# feature_r = ['LRB_020','LRB_007','LRB_001','LRB_022','LRB_011','LRB_013','LRB_009','TAX_TW_024','TAX_TW_001','TAX_TW_004','TAX_TW_003','OPT_ARCSS_001','LRB_002',LRB_004,LRB_021,LRB_023,COM_CIBI_006,LRB_024,LRB_025,LRB_026,ZCFZB_034,TAX_TW_031]

def KNN(missing_data):
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(missing_data)
    impute_knn = imputer.transform(missing_data)
    return impute_knn

def RF_1(ori_data,data_missing,data_m,tree=100):
    X_missing_fillna = data_missing.copy()
    # 按照当前列缺失值的数量进行升序排列
    sortindex = X_missing_fillna.isnull().sum().sort_values().index
    for i in tqdm(sortindex):
        if X_missing_fillna.loc[:, i].isnull().sum() != 0:
            df = X_missing_fillna
            # 将当前列作为目标值
            fillc = df.loc[:, i]
            # 将其余列作为特征值（包括目标值）
            df = df.loc[:, df.columns != i]
            # 使用均值填充其余列缺失值
            # df_0 = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(df)
            df_0 = KNN(df)
            # 将fillc中非空的样本作为训练数据
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            # 建立随机森林回归树进行训练
            if i in feature_c:
                rfc = RandomForestClassifier(n_estimators=tree)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)

            else:
                rfc = RandomForestRegressor(n_estimators=tree)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
            # 填充缺失值
            X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
    rmse = rmse_loss(ori_data.values, X_missing_fillna.values, data_m)
    print('RF_1:', round(rmse,5))
    return rmse

def RF_1_r(ori_data,data_missing,data_m,tree=100):
    X_missing_fillna = data_missing.copy()
    # 按照当前列缺失值的数量进行升序排列
    sortindex = X_missing_fillna.isnull().sum().sort_values().index
    for i in tqdm(sortindex):
        if X_missing_fillna.loc[:, i].isnull().sum() != 0:
            df = X_missing_fillna
            # 将当前列作为目标值
            fillc = df.loc[:, i]
            # 将其余列作为特征值（包括目标值）
            df = df.loc[:, df.columns != i]
            # 使用均值填充其余列缺失值
            # df_0 = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(df)
            df_0 = KNN(df)
            # 将fillc中非空的样本作为训练数据
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            # 建立随机森林回归树进行训练

            rfc = RandomForestRegressor(n_estimators=tree)
            rfc = rfc.fit(Xtrain, Ytrain)
            # 对缺失值进行预测
            Ypredict = rfc.predict(Xtest)

            # 填充缺失值
            X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
    rmse = rmse_loss(ori_data.values, X_missing_fillna.values, data_m)
    print('RF_1:', round(rmse,5))
    return rmse
def RF_1_3source(ori_data,data_missing,data_m,tree=100):
    X_missing_fillna = data_missing.copy()
    # 按照当前列缺失值的数量进行升序排列
    sortindex = X_missing_fillna.isnull().sum().sort_values().index
    for i in tqdm(sortindex):
        if X_missing_fillna.loc[:, i].isnull().sum() != 0:
            df = X_missing_fillna
            # 将当前列作为目标值
            fillc = df.loc[:, i]
            # 将其余列作为特征值（包括目标值）
            df = df.loc[:, df.columns != i]
            # 使用均值填充其余列缺失值
            df_0 = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(df)

            # 将fillc中非空的样本作为训练数据
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            # 建立随机森林回归树进行训练
            rfc = RandomForestClassifier(n_estimators=tree)
            rfc = rfc.fit(Xtrain, Ytrain)
            # 对缺失值进行预测
            Ypredict = rfc.predict(Xtest)
            # 填充缺失值
            X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
    rmse = rmse_loss(ori_data.values, X_missing_fillna.values, data_m)
    print('RF_1:', round(rmse,5))
    return rmse

def RF_1_BBCsport(ori_data,data_missing,data_m):
    X_missing_fillna = data_missing.copy()
    # 按照当前列缺失值的数量进行升序排列
    sortindex = X_missing_fillna.isnull().sum().sort_values().index
    for i in tqdm(sortindex):
        if X_missing_fillna.loc[:, i].isnull().sum() != 0:
            df = X_missing_fillna
            # 将当前列作为目标值
            fillc = df.loc[:, i]
            # 将其余列作为特征值（包括目标值）
            df = df.loc[:, df.columns != i]
            # 使用均值填充其余列缺失值
            df_0 = SimpleImputer(missing_values=np.nan,strategy='mean').fit_transform(df)

            # 将fillc中非空的样本作为训练数据
            Ytrain = fillc[fillc.notnull()]
            Ytest = fillc[fillc.isnull()]
            Xtrain = df_0[Ytrain.index, :]
            Xtest = df_0[Ytest.index, :]
            # 建立随机森林回归树进行训练
            rfc = RandomForestRegressor(n_estimators=100)
            rfc = rfc.fit(Xtrain, Ytrain)
            # 对缺失值进行预测
            Ypredict = rfc.predict(Xtest)
            # 填充缺失值
            X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
    rmse = round(rmse_loss(ori_data.values, X_missing_fillna.values, data_m), 5)
    print('RF_1:', rmse)
    return rmse