import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import scipy.stats as st
np.random.seed(1234)
pd.set_option('display.max_columns', None)
n_rows = 5000
n_cols = 5

avg_scale = [50, 60, 5, 1, 0.5]
std = [10, 5] # for the first two data sets
seed = 1234

def toy_data(n_rows):
    X = pd.DataFrame()
    X['0']=5*np.random.randn(n_rows) + avg_scale[0]
    X['1']=3*np.random.randn(n_rows)+ avg_scale[1]
    X['2'] = np.random.exponential(avg_scale[2], n_rows)
    X['3'] = np.random.exponential(avg_scale[3], n_rows)
    X['4'] = np.random.exponential(avg_scale[4], n_rows)
    return X

X=toy_data(n_rows)
print(X.head(5))

P = pd.DataFrame()
for j in range(2):
    args = st.norm.fit(X[str(j)])
    P[str(j)] = (1-st.norm.cdf(X[str(j)].values, *args)) * 100
for j in range(2, 5):
    args = st.expon.fit(X[str(j)])
    P[str(j)] = (1-st.expon.cdf(X[str(j)],*args))*100 # [st.expon.cdf(x,args) * 100 for x in X[str(j)]]
print(P.head(5))
print(P.describe())

import math
def EW(x, n, maxval, minval):
    x = (x - minval)/(maxval - minval)
    s = np.sum(x)
    tmp = np.sum([e/s * math.log(e/s+ 1e-6) for e in x])
    p = - tmp / math.log(n)
    d = 1 - p
    return d

col_max = X.max(axis=0).values
col_min = X.min(axis=0).values

D = [EW(X[str(j)], n_rows, col_max[j], col_min[j]) for j in range(n_cols)]

def scale(D):
    s = np.sum(D)
    D /= s
    return D

w = scale(D)
print(w, np.sum(w))

import seaborn as sns
ew_score = np.dot(P, w)

sns.distplot(ew_score, label='weighted sum score', color='b', kde=True)
plt.show()

from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score


def get_score(name, estimator, clusterInput, compareInput=None):
    estimator.fit(clusterInput)
    labels = estimator.predict(clusterInput)
    if compareInput is None:
        silhouette_avg = silhouette_score(clusterInput, labels, metric='euclidean')
        ch = calinski_harabasz_score(clusterInput, labels)
        dbs = davies_bouldin_score(clusterInput, labels)
    else:
        silhouette_avg = silhouette_score(compareInput, labels, metric='euclidean')
        ch = calinski_harabasz_score(compareInput, labels)
        dbs = davies_bouldin_score(compareInput, labels)
    print("%s\t%.3f\t%.3f\t%.3f" %(name, silhouette_avg, ch, dbs))
    return labels


def cluster_validation(n_digit, clusterInput, compareInput, score_vec=None):
    print("methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score")
    # # Fit a kmeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=n_digit, max_iter=3000, tol=1e-4, n_init=10, random_state=0)
    labels = get_score("kmeans", kmeans, clusterInput, compareInput)
    if score_vec is not None:
        tmp = pd.DataFrame({'label':labels, 'score':score_vec})
        print(tmp.groupby('label')['score'].describe())
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4, covariance_type='spherical', random_state=0)
    labels = get_score("GMM", gmm, clusterInput, compareInput)
    if score_vec is not None:
        tmp = pd.DataFrame({'label':labels, 'score':score_vec})
        print(tmp.groupby('label')['score'].describe())
    # # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4,
                                            covariance_type='spherical', random_state=0)
    labels = get_score("DPGMM", dpgmm, clusterInput, compareInput)
    if score_vec is not None:
        tmp = pd.DataFrame({'label':labels, 'score':score_vec})
        print(tmp.groupby('label')['score'].describe())

## -------------- without score----------------------
# cluster_validation(3, X, X, ew_score)

# from sklearn.preprocessing import StandardScaler
# input = StandardScaler().fit_transform(X)
# cluster_validation(3, input, X, ew_score)
#
# from sklearn.preprocessing import MinMaxScaler
# input = MinMaxScaler().fit_transform(X)
# cluster_validation(3, input, X, ew_score)

# --------------add score as additional feature--------------------
X_hat = X

# X_hat['score'] = ew_score
# cluster_validation(3, X_hat, X, ew_score)

from sklearn.preprocessing import MinMaxScaler
input = MinMaxScaler().fit_transform(X_hat)
cluster_validation(3, input, X, ew_score)


##################without socre without normalization#########################
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# kmeans	0.299	2461.639	1.100
# GMM	0.293	2387.765	1.116
# DPGMM	0.293	2380.529	1.117

# count       mean        std  ...        50%        75%        max
# label                                ...
# 0      2154.0  47.092245  15.082097  ...  46.820663  57.729210  89.982776
# 1       696.0  63.310415  13.034603  ...  63.687236  72.723787  92.680000
# 2      2150.0  48.397211  14.996510  ...  48.538944  59.157578  87.953355
#
# [3 rows x 8 columns]

# count       mean        std  ...        50%        75%        max
# label                                ...
# 0      2118.0  46.960927  15.077286  ...  46.681365  57.624602  89.982776
# 1       751.0  62.974766  13.176220  ...  63.599677  72.548531  92.680000
# 2      2131.0  48.239069  14.919422  ...  48.424217  58.861069  87.953355
#
# [3 rows x 8 columns]

# count       mean        std  ...        50%        75%        max
# label                                ...
# 0      2112.0  46.941076  15.075040  ...  46.658138  57.528025  89.982776
# 1       756.0  62.925643  13.203451  ...  63.578014  72.473905  92.680000
# 2      2132.0  48.237997  14.916003  ...  48.420034  58.860210  87.953355

##################without socre with Maxmin #########################
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# kmeans	0.180	1050.390	1.875
# GMM	0.022	220.166	4.133
# DPGMM	0.038	338.220	3.438
# count       mean        std       min        25%        50% \
#                                              label
# 0      1656.0  49.798884  16.001441  9.458060  38.491328  49.090845
# 1      1628.0  48.645617  15.688920  6.802507  37.385371  48.725907
# 2      1716.0  51.219540  15.430307  8.410860  40.973271  51.162227
#
# 75%       max
# label
# 0      60.751546  92.68000
# 1      59.948032  91.24258
# 2      62.344639  91.18168

# count       mean        std        min        25%        50% \
#                                               label
# 0      1557.0  44.013666  13.615873   6.802507  34.669725  44.422433
# 1      1864.0  58.288603  15.087623   7.952542  47.952578  59.056808
# 2      1579.0  45.836298  14.114794  10.087282  35.790604  45.928575
#
# 75%        max
# label
# 0      53.239773  77.979531
# 1      68.926501  92.680000
# 2      56.144432  82.316338

# count       mean        std       min        25%        50% \
#                                              label
# 0      1646.0  44.963524  13.727326  6.802507  35.455329  45.023117
# 1      2016.0  57.909396  15.142980  7.952542  47.952578  58.633223
# 2      1338.0  43.945772  13.652694  9.458060  34.320473  44.279491
#
# 75%        max
# label
# 0      54.975762  78.696961
# 1      68.527847  92.680000
# 2      53.561358  80.376146


##################with socre without Normalization #########################
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# kmeans	0.350	5390.635	0.936
# GMM	0.342	5249.909	0.944
# DPGMM	0.342	5244.028	0.944

# count       mean       std        min        25%        50% \
#                                              label
# 0      1449.0  68.721550  7.434329  48.714041  62.814449  67.113092
# 1      1336.0  30.381273  7.129241   6.802507  26.014319  31.907284
# 2      2215.0  49.385030  5.515780  38.489812  44.825675  49.289398
#
# 75%        max
# label
# 0      73.354337  92.680000
# 1      36.163563  40.343266
# 2      54.154646  60.187536

# count       mean       std        min        25%        50% \
#                                              label
# 0      1243.0  29.711518  6.940372   6.802507  25.576461  31.138831
# 1      1726.0  66.854462  8.120962  42.174256  60.508679  65.415960
# 2      2031.0  47.874214  5.130948  38.377634  43.593256  47.776322
#
# 75%        max
# label
# 0      35.402553  39.178830
# 1      71.916110  92.680000
# 2      52.113900  58.158349

# count       mean       std        min        25%        50% \
#                                              label
# 0      1235.0  29.654106  6.925919   6.802507  25.497093  31.083876
# 1      1745.0  66.750644  8.137694  42.174256  60.437001  65.320747
# 2      2020.0  47.748541  5.093924  38.377634  43.514511  47.584154
#
# 75%        max
# label
# 0      35.325511  39.178830
# 1      71.852548  92.680000
# 2      51.887807  57.937363

##################with socre with maxmin #########################
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# kmeans	0.222	3342.488	1.207
# GMM	0.212	3242.888	1.316
# DPGMM	0.224	3278.226	1.223

# count       mean       std        min        25%        50% \
#                                              label
# 0      1583.0  50.702019  8.358724  24.481271  45.186918  50.798020
# 1      1877.0  35.483302  9.719122   6.802507  28.856616  36.255120
# 2      1540.0  66.682658  9.030239  43.132300  59.911393  65.907797
#
# 75%        max
# label
# 0      56.289564  79.666746
# 1      42.999275  55.019970
# 2      72.819018  92.680000

# count       mean        std        min        25%        50% \
#                                               label
# 0      1567.0  32.755339   8.500520   6.802507  26.932899  33.783721
# 1      1794.0  64.277625  10.813393  33.959295  56.607094  64.832889
# 2      1639.0  50.587597   7.122232  27.256837  45.418080  50.386710
#
# 75%        max
# label
# 0      39.165906  50.959155
# 1      71.646652  92.680000
# 2      55.861422  68.463188

# count       mean        std        min        25%        50% \
#                                               label
# 0      1259.0  30.708607   7.935483   6.802507  25.682940  31.396781
# 1      1876.0  64.011250  10.858274  33.076234  56.765125  64.568244
# 2      1865.0  48.690351   7.245289  24.481271  43.678843  48.634234
#
# 75%        max
# label
# 0      36.601295  46.640283
# 1      71.151097  92.680000
# 2      54.146092  66.246180



