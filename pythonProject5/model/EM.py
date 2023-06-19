# from impyute import em
import impyute
from utils import rmse_loss

def EM(ori_data_x,missing_data,data_m):
    result = impyute.em(missing_data,loops=10)
    rmse_em = round(rmse_loss(ori_data_x.values, result, data_m), 5)
    print('em:', rmse_em)
    return rmse_em







