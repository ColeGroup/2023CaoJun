# kNN分类器
# kNN数据空值填充
from sklearn.impute import KNNImputer
from utils import rmse_loss


# KNN
def knn(ori_data_x,missing_data,data_m,n_neighbors):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputer.fit(missing_data)
    impute_knn = imputer.transform(missing_data)
    # RMSE
    rmse_knn = round(rmse_loss(ori_data_x.values, impute_knn, data_m),5)
    print('KNN:',rmse_knn)
    return rmse_knn

