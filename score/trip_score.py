from tdrive_etl_utils import load_fule_truck_agg_data, load_fule_truck_stats_data
# from etl_utils import load_tdrive_data, load_fuel_truck_data
from pca.reduce_by_pca import pca_reduce
import scipy.stats as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def entropy_weight_score(data0):
    '''
    Entropy weight based feature importance and overall score
    :param data0:
    :return:
    '''
    #返回每个样本的指数
    #样本数，指标个数
    n,m=np.shape(data0)
    #一行一个样本，一列一个指标
    #下面是归一化
    maxium=np.max(data0,axis=0)
    minium=np.min(data0,axis=0)
    data= (data0-minium)*1.0/(maxium-minium)
    ##计算第j项指标，第i个样本占该指标的比重
    sumzb=np.sum(data,axis=0)
    data=data/sumzb
    #对ln0处理
    a=data*1.0
    a[np.where(data==0)]=0.0001
    #    #计算每个指标的熵
    e=(-1.0/np.log(n))*np.sum(data*np.log(a),axis=0)
    #    #计算权重
    w=(1-e)/np.sum(1-e)
    recodes=np.sum(data*w,axis=1)  # 计算得分
    return recodes


def exp_cpd_score(data, init_weights):
    '''
    Exponential cumulative probability density based score
    :param data:
    :return:
    # 名称	含义
    # beta	beta分布
    # f	F分布
    # gamma	gam分布
    # poisson	泊松分布
    # hypergeom	超几何分布
    # lognorm	对数正态分布
    # binom	二项分布
    # uniform	均匀分布
    # chi2	卡方分布
    # cauchy	柯西分布
    # laplace	拉普拉斯分布
    # rayleigh	瑞利分布
    # t	学生T分布
    # norm	正态分布
    # expon	指数分布
    '''
    distribution = {}
    for col in agg_data.columns[1:]:
        loc, scale = st.expon.fit(agg_data[col])
        distribution[col] = st.expon.cdf(agg_data[col], loc, scale)
    distribution = pd.DataFrame(distribution)
    agg_score = np.dot(distribution, agg_weights.reshape([-1,1]))
    distribution['agg_score'] = agg_score*100
    return distribution


agg_data = load_fule_truck_agg_data()
print(agg_data.head(5))
agg_weights = np.ones(agg_data.shape[1]-1) / (agg_data.shape[1]-1)
# distribution score
distribution = {}
for col in agg_data.columns[1:]:
    loc, scale = st.expon.fit(agg_data[col])
    distribution[col] = st.expon.cdf(agg_data[col], loc, scale)
distribution = pd.DataFrame(distribution)
agg_score = np.dot(distribution, agg_weights.reshape([-1,1]))
distribution['agg_score'] = agg_score*100

sns.distplot(agg_score)
plt.show()

# dimension reduction
# pca_data = pca_reduced()
