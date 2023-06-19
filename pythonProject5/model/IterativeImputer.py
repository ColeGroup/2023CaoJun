from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from utils import rmse_loss
import warnings
warnings.filterwarnings('ignore')

# mice
def MICE(ori_data_x,missing_data,data_m):
    imputer = IterativeImputer(max_iter=1)
    imputer.fit(missing_data)
    impute_knn = imputer.transform(missing_data)
    # RMSE
    rmse_mice = round(rmse_loss(ori_data_x.values, impute_knn, data_m),5)
    print('MICE:',rmse_mice)
    return rmse_mice