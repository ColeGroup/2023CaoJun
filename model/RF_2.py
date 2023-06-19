from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
from sklearn.impute import SimpleImputer
from utils import *
from tqdm import tqdm
from data.dataloader import data_loader
import pandas as pd
from sklearn.impute import KNNImputer
def KNN(missing_data):
    imputer = KNNImputer(n_neighbors=5)
    imputer.fit(missing_data)
    impute_knn = imputer.transform(missing_data)
    return impute_knn
feature_c = ['PERSON_PMS_002','COM_CICI_035','COM_CICI_037','TAX_TW_015','TAX_TW_016','TAX_TW_017','TAX_TW_018','TAX_TW_025','TAX_TW_019','TAX_TW_020','TAX_TW_027','TAX_TW_022','TAX_TW_023','DWSBJN_002','DEBT_DMP_021','DEBT_DMP_025','NSDJ_016','TAX_TW_009','TAX_TW_028','JUSTICE_JNP_023','JUSTICE_JNP_024','JUSTICE_JNP_025','JUSTICE_JNP_026','JUSTICE_JNP_027','JUSTICE_JNP_028','JUSTICE_JNP_029','JUSTICE_JNP_030','JUSTICE_JNP_031','JNSHJXYHB_002','JNSHJXYYBMD_002','JUSTICE_JE_009','JUSTICE_JNP_022','SHTZXY_002','COM_CIBI_033','XWQY_001','TAX_TW_029','TAX_TW_030']
def RF_2(ori_data_x,miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6,data_m):
    def RF(data_missing):
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

                rfc = RandomForestRegressor(n_estimators=100)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)
    fill_3 = RF(miss_data_x3)
    fill_4 = RF(miss_data_x4)
    fill_5 = RF(miss_data_x5)
    fill_6 = RF(miss_data_x6)


    fill_all = pd.concat([fill_1, fill_2, fill_3, fill_4, fill_5, fill_6], axis=1)
    rmse_fk = rmse_loss(ori_data_x.values, fill_all.values, data_m)
    print('RF_2:', round(rmse_fk,5))



    return rmse_fk

def RF_2_100leaves(ori_data_x,miss_data_x1, miss_data_x2, miss_data_x3,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor(random_state=0,n_estimators=10)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)
    fill_3 = RF(miss_data_x3)

    fill_all = pd.concat([fill_1, fill_2, fill_3], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2:', rmse_fk)



    return rmse_fk

def RF_2_handwritten(ori_data_x,miss_data_x1, miss_data_x2, miss_data_x3,miss_data_x4, miss_data_x5, miss_data_x6,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor(random_state=0)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)
    fill_3 = RF(miss_data_x3)
    fill_4 = RF(miss_data_x4)
    fill_5 = RF(miss_data_x5)
    fill_6 = RF(miss_data_x6)

    fill_all = pd.concat([fill_1, fill_2, fill_3,fill_4, fill_5, fill_6], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2:', rmse_fk)



    return rmse_fk

def RF_2_3sources(ori_data_x,miss_data_x1, miss_data_x2, miss_data_x3,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor()
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)
    fill_3 = RF(miss_data_x3)


    fill_all = pd.concat([fill_1, fill_2, fill_3], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2_3sources:', rmse_fk)


    return rmse_fk

def RF_2_wiki(ori_data_x,miss_data_x1, miss_data_x2,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor()
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)

    fill_all = pd.concat([fill_1, fill_2], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2_wiki:', rmse_fk)

    return rmse_fk

def RF_2_webkb(ori_data_x,miss_data_x1, miss_data_x2,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor(n_estimators=10)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)

    fill_all = pd.concat([fill_1, fill_2], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2_webkb:', rmse_fk)

    return rmse_fk

def RF_2_BBCsport(ori_data_x,miss_data_x1, miss_data_x2,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor(n_estimators=10)
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)


    fill_all = pd.concat([fill_1, fill_2], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2:', rmse_fk)



    return rmse_fk

def RF_2_news(ori_data_x,miss_data_x1, miss_data_x2,miss_data_x3,data_m):
    def RF(data_missing):
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
                rfc = RandomForestRegressor()
                rfc = rfc.fit(Xtrain, Ytrain)
                # 对缺失值进行预测
                Ypredict = rfc.predict(Xtest)
                # 填充缺失值
                X_missing_fillna.loc[Ytest.index, Ytest.name] = Ypredict
        return X_missing_fillna

    fill_1 = RF(miss_data_x1)
    fill_2 = RF(miss_data_x2)
    fill_3 = RF(miss_data_x3)

    fill_all = pd.concat([fill_1, fill_2,fill_3], axis=1)
    rmse_fk = round(rmse_loss(ori_data_x.values, fill_all.values, data_m), 5)
    print('RF_2_news:', rmse_fk)

    return rmse_fk