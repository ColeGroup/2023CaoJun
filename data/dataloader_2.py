import numpy as np
import pandas as pd
from data.dataloader import data_loader


def missing_data_generate(data, missing_columns, missing_rate=0.2,seed=11):
    # if missing_columns:
        X_full = data.loc[:, missing_columns]
        n_samples = X_full.shape[0]
        n_features = X_full.shape[1]
        n_missing_samples = n_samples * n_features * missing_rate
        n_missing_samples = int(round(n_missing_samples))

        rng = np.random.RandomState(seed=seed)
        missing_row_index = rng.randint(0, n_samples, n_missing_samples)
        missing_col_index = rng.randint(0, n_features, n_missing_samples)
        for i in range(n_missing_samples):
            X_full.iloc[missing_row_index[i], missing_col_index[i]] = np.nan

        data_missing = data.copy()
        data_missing[X_full.columns] = X_full
        return data_missing
    # else:
    #     return data


def generate_mask(missdata):
    mask = np.zeros((missdata.shape[0], missdata.shape[1]))
    # 遍历矩阵每个元素
    for row in range(missdata.shape[0]):
        for col in range(missdata.shape[1]):
            if np.isnan(missdata[row][col]):
                mask[row][col] = 0
            else:
                mask[row][col] = 1
    return mask

def sapm_data(miss,seed=11):
    df = pd.read_csv("./data/spam.csv")

    ori_data_x = df
    miss_data_x = missing_data_generate(ori_data_x,missing_columns=ori_data_x.columns, missing_rate=miss / 10,seed=seed)
    data_m = generate_mask(np.array(miss_data_x))

    return ori_data_x, miss_data_x, data_m


def multi_view_data(miss,seed=11):
    df = pd.read_csv("./data/mydata1.csv")

    # 完整视角属性
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

    # 部分缺失属性
    imp1 = ['LRB_020', 'LRB_007']
    imp2 = ['LRB_001', 'LRB_022', 'LRB_011', 'LRB_013', 'LRB_009', 'TAX_TW_015', 'TAX_TW_016', 'TAX_TW_017',
            'TAX_TW_001', 'TAX_TW_024', 'TAX_TW_004', 'TAX_TW_003', 'TAX_TW_018', 'TAX_TW_025', 'TAX_TW_019',
            'TAX_TW_020', 'TAX_TW_027', 'TAX_TW_022', 'TAX_TW_023', 'OPT_ARCSS_001', 'DWSBJN_002']
    imp3 = ['NSDJ_016', 'TAX_TW_009', 'TAX_TW_028']
    imp4 = ['LRB_002', 'LRB_004', 'LRB_021', 'LRB_023']
    imp5 = ['COM_CIBI_006', 'COM_CIBI_033', 'TAX_TW_029', 'TAX_TW_030', 'COM_CIBI_002']
    # imp6 = []
    imp6 = ['LRB_024', 'LRB_025', 'LRB_026', 'ZCFZB_034', 'TAX_TW_031']

    ori_data_x1 = df[view1]
    miss_data_x1 = missing_data_generate(ori_data_x1, missing_columns=imp1, missing_rate=miss / 10)
    data_m1 = generate_mask(np.array(miss_data_x1))
    ori_data_x2 = df[view2]
    miss_data_x2 = missing_data_generate(ori_data_x2, missing_columns=imp2, missing_rate=miss / 10)
    data_m2 = generate_mask(np.array(miss_data_x2))
    ori_data_x3 = df[view3]
    miss_data_x3 = missing_data_generate(ori_data_x3, missing_columns=imp3, missing_rate=miss / 10)
    data_m3 = generate_mask(np.array(miss_data_x3))
    ori_data_x4 = df[view4]
    miss_data_x4 = missing_data_generate(ori_data_x4, missing_columns=imp4, missing_rate=miss / 10)
    data_m4 = generate_mask(np.array(miss_data_x4))
    ori_data_x5 = df[view5]
    miss_data_x5 = missing_data_generate(ori_data_x5, missing_columns=imp5, missing_rate=miss / 10)
    data_m5 = generate_mask(np.array(miss_data_x5))
    ori_data_x6 = df[view6]
    miss_data_x6 = missing_data_generate(ori_data_x6, missing_columns=imp6, missing_rate=miss / 10)
    data_m6 = generate_mask(np.array(miss_data_x6))


    # 合并
    ori_data_x = pd.concat(
        [ori_data_x1, ori_data_x2, ori_data_x3, ori_data_x4, ori_data_x5, ori_data_x6], axis=1)
    miss_data_x = pd.concat(
        [miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6], axis=1)
    data_m = np.hstack((data_m1, data_m2, data_m3, data_m4, data_m5, data_m6))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6


def multi_view_data_100leaves(miss):
    df = pd.read_csv("./data/100leaves.csv")
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:63], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 64:127], miss_rate=miss / 10)
    ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 128:191], miss_rate=miss / 10)

    # ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:63], miss_rate=miss / 10)
    # ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 64:127], miss_rate=miss / 10)
    # ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 128:191], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2, ori_data_x3], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2, miss_data_x3], axis=1)
    data_m = np.hstack((data_m1, data_m2, data_m3))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2, miss_data_x3


def multi_view_data_handwritten(miss):
    df = pd.read_csv("./data/handwritten.csv")
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:239], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 240:315], miss_rate=miss / 10)
    ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 316:531], miss_rate=miss / 10)
    ori_data_x4, miss_data_x4, data_m4 = data_loader(df.iloc[:, 532:578], miss_rate=miss / 10)
    ori_data_x5, miss_data_x5, data_m5 = data_loader(df.iloc[:, 579:642], miss_rate=miss / 10)
    ori_data_x6, miss_data_x6, data_m6 = data_loader(df.iloc[:, 643:648], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2, ori_data_x3, ori_data_x4, ori_data_x5, ori_data_x6], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6],
                            axis=1)
    data_m = np.hstack((data_m1, data_m2, data_m3, data_m4, data_m5, data_m6))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2, miss_data_x3, miss_data_x4, miss_data_x5, miss_data_x6


def multi_view_data_3sources(miss):
    df = pd.read_csv("./data/3-sources-2.csv")
    # ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:3559], miss_rate=miss / 10)
    # ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 3560:7190], miss_rate=miss / 10)
    # ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 7191:10258], miss_rate=miss / 10)
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:100], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 100:200], miss_rate=miss / 10)
    ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 200:300], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2, ori_data_x3], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2, miss_data_x3], axis=1)
    data_m = np.hstack((data_m1, data_m2, data_m3))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2, miss_data_x3


def multi_view_data_wiki(miss):
    df = pd.read_csv("./data/wiki_fea.csv")
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:128], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 128:138], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2], axis=1)
    data_m = np.hstack((data_m1, data_m2))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2


def multi_view_data_webkb(miss):
    df = pd.read_csv("./data/webkb2.csv")
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:1703], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 1703:1933], miss_rate=miss / 10)
    ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 1933:2163], miss_rate=miss / 10)
    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2,ori_data_x3], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2,miss_data_x3], axis=1)
    data_m = np.hstack((data_m1, data_m2,data_m3))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2,miss_data_x3


def multi_view_data_BBCSport(miss):
    df = pd.read_csv("./data/BBCSport3.csv")
    # ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:3183], miss_rate=miss / 10,seed=3)
    # ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 3183:6386], miss_rate=miss / 10,seed=3)
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:100], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 100:200], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2], axis=1)
    data_m = np.hstack((data_m1, data_m2))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2


def multi_view_data_news(miss):
    df = pd.read_csv("./data/newsgroups.csv")
    ori_data_x1, miss_data_x1, data_m1 = data_loader(df.iloc[:, 0:2000], miss_rate=miss / 10)
    ori_data_x2, miss_data_x2, data_m2 = data_loader(df.iloc[:, 2000:4000], miss_rate=miss / 10)
    ori_data_x3, miss_data_x3, data_m3 = data_loader(df.iloc[:, 4000:6000], miss_rate=miss / 10)

    # 合并
    ori_data_x = pd.concat([ori_data_x1, ori_data_x2, ori_data_x3], axis=1)
    miss_data_x = pd.concat([miss_data_x1, miss_data_x2, miss_data_x3], axis=1)
    data_m = np.hstack((data_m1, data_m2, data_m3))

    return ori_data_x, miss_data_x, data_m, miss_data_x1, miss_data_x2, miss_data_x3
