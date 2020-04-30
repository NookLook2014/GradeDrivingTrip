import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import mixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from ftruck_etl_utils import get_trip_vectors
from score.trip_score import exp_cpd_score, get_weight

np.random.seed(1234)

# key_metric_fetures = ['vid', 'harsh_acc', 'harsh_dec', 'harsh_turn', 'idleRate', 'over_speed', 'tripMileKm']

def cluster_validation(n_digit, X):
    print("methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score")
    # # Fit a kmeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=n_digit, max_iter=3000, tol=1e-4, n_init=10, random_state=0)
    get_score("kmeans", kmeans, X)
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4, covariance_type='spherical', random_state=0)
    get_score("GMM", gmm, X)
    # # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4,
                                        covariance_type='spherical', random_state=0)
    get_score("DPGMM", dpgmm, X)


def get_score(name, estimator, X):
    estimator.fit(X)
    labels = estimator.predict(X)
    silhouette_avg  = silhouette_score(X, labels, metric='euclidean')
    ch = calinski_harabasz_score(X, labels)
    dbs = davies_bouldin_score(X, labels)
    print("%s\t%.3f\t%.3f\t%.3f" %(name, silhouette_avg, ch, dbs))


def predict_cluster_labels(n_digit, X, input_name, compare_input):
    # Fit a kmeans clustering model
    kmeans = KMeans(init='k-means++', n_clusters=n_digit, max_iter=3000, tol=1e-4, n_init=10, random_state=0)
    # Fit a Gaussian mixture with EM using five components
    gmm = mixture.GaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4, covariance_type='spherical', random_state=0)
    # # Fit a Dirichlet process Gaussian mixture using five components
    dpgmm = mixture.BayesianGaussianMixture(n_components=n_digit, max_iter=3000, tol=1e-4,covariance_type='spherical', random_state=0)
    estimators = {"Kmeans": kmeans, "GMM": gmm, "BGM": dpgmm}
    output = {}
    print("methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score")
    for k in estimators.keys():
        name = k+"_"+input_name
        model_file = Path(name)
        if model_file.is_file():
            with open(name, 'rb') as fr:
                estimator = pickle.load(fr)
        else:
            # print(k + 'Clustering ...')
            estimator = estimators[k]
            estimator.fit(X)
            # with open(name, 'wb') as fw:
            #     pickle.dump(estimator, fw)
        labels = estimator.predict(X)
        silhouette_avg = silhouette_score(compare_input, labels, metric='euclidean')
        ch = calinski_harabasz_score(X, labels)
        dbs = davies_bouldin_score(X, labels)
        print("%s\t%.3f\t%.3f\t%.3f" %(k, silhouette_avg, ch, dbs))
        output[name] = labels
    return output


def aggregation(data, by):
    tmps = data.groupby(by).sum()
    tmpm = data.groupby(by).max()
    return tmps.div(tmpm)


def cluster_driver_vectors(data, score_features):
    if score_features is not None:
        data["score"] = score_features
    # else:
    #     data["score"] = np.zeros(data.shape[0])
    # print(data.shape)
    driver_vectors = aggregation(data, "vid")
    compare_input = driver_vectors
    if score_features is not None:
        compare_input.drop(columns="score", inplace=True)

    print("Without any standard")
    predict_cluster_labels(3, driver_vectors, "", compare_input)
    print("With standard")
    X = StandardScaler().fit_transform(driver_vectors)
    predict_cluster_labels(3, X, "", compare_input)
    print("With MaxMin")
    X = MinMaxScaler().fit_transform(driver_vectors)
    output = predict_cluster_labels(3, X, "", compare_input)

    driver_vectors["label"] = output["GMM_"]
    # print(driver_vectors.groupby("label")["score"].describe())

    return driver_vectors


if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    vid, data = get_trip_vectors()
    data.drop(columns='Unnamed: 0', inplace=True)

    performance = exp_cpd_score(data)
    sub_w = np.array([0.2, 0.2, 0.2, 0.15, 0.1, 0.15])

    data["vid"] = vid
    cluster_driver_vectors(data, None)

    print("##########################Trip level  Avg score #########################")
    weights = get_weight(data, method='average')
    overall_score1 = np.dot(performance, weights) # * 99 + np.random.rand() #add some noise
    data["vid"] = vid
    del vid
    cluster_driver_vectors(data, overall_score1)

    # print("##########################Trip level  EW score #########################.drop(columns='overall_score')")
    # weights = get_weight(data, method='entropy_weight')
    # overall_score2 = np.dot(performance, weights)# * 99 + np.random.rand() #add some noise
    # tmp = pd.DataFrame({"vid":vid, "score":overall_score2})
    # tmp = tmp.groupby("vid")["score"].mean()
    # data["vid"] = vid
    # del vid
    # driver_vectors = cluster_driver_vectors(data, overall_score2)
    # driver_vectors["avg_score"] = tmp
    # tmp = pd.concat([driver_vectors, tmp], join="inner")
    # print(driver_vectors.groupby("label")["avg_score"].describe())
    #
    # print("##########################Trip level  Avg*SW score #########################.drop(columns='overall_score')")
    # weights = get_weight(data, method='average', sub_weights=sub_w)
    # overall_score3 = np.dot(performance, weights) # * 99 + np.random.rand() #add some noise
    # tmp = pd.DataFrame({"vid":vid, "score":overall_score3})
    # tmp = tmp.groupby("vid")["score"].mean()
    # data["vid"] = vid
    # del vid
    # driver_vectors = cluster_driver_vectors(data, overall_score3)
    # driver_vectors["avg_score"] = tmp
    # tmp = pd.concat([driver_vectors, tmp], join="inner")
    # print(driver_vectors.groupby("label")["avg_score"].describe())
    #
    # print("##########################Trip level  EW*SW score #########################.drop(columns='overall_score')")
    # weights = get_weight(data, method='entropy_weight', sub_weights=sub_w)
    # overall_score4 = np.dot(performance, weights) #* 99 + np.random.rand()
    # data["vid"] = vid
    # del vid
    # cluster_driver_vectors(data, overall_score4)


    # files = ["on_Metric_use_AvgScore", "on_Metric_use_EWScore", "on_Metric_use_SWEWScore"]
    # best_clustering_method = "Kmeans_"
    # for f in files:
    #     print(f)
    #     data = pd.read_csv("cahed_res/"+f+"_res.csv")
    #     print(data["score"].describe())

    # cluster_trip_vectors()
    # a = [10, 9, 8, 7]
    # args = st.expon.fit(a)
    # print(st.expon.cdf(5, *args) )   # cluster_driver_vectors()
    # print(st.expon.cdf(5, 6, 1.5) )
    # print(st.expon.cdf(5, 1.5, 6) )
    # print(st.expon.cdf(10, *args) )
    # vid, data = get_trip_vectors()
    # feature_score = exp_cpd_score(data)
    # score_stats = driver_score_stats(vid, feature_score)
    # cluster_driver_vectors(score_stats)


#######################on metric features only#############################
# Step1. without standardization
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# GMMClustering ...
# GMM	0.319	0.000	0.000
# BGMClustering ...
# BGM	0.215	0.000	0.000
# KmeansClustering ...
# Kmeans	0.280	0.000	0.000
# Step2. with MinMaxScaler
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# GMMClustering ...
# GMM	0.299	0.000	0.000
# BGMClustering ...
# BGM	0.355	0.000	0.000
# KmeansClustering ...
# Kmeans	0.269	0.000	0.000
# Step2. with StandardScaler
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# BGMClustering ...
# BGM	0.286	0.000	0.000
# GMMClustering ...
# GMM	0.274	0.000	0.000
# KmeansClustering ...
# Kmeans	0.225	0.000	0.000


##########################Trip level  Avg score #########################
# Without any standard
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# Kmeans	0.138	0.000	0.000
# GMM	0.123	0.000	0.000
# BGM	0.114	0.000	0.000
# With standard
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# Kmeans	0.281	0.000	0.000
# GMM	0.315	0.000	0.000
# BGM	0.342	0.000	0.000
# With MaxMin
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# Kmeans	0.210	24.995	1.425
# GMM	0.342	16.063	0.775
# BGM	0.365	26.113	0.931

##########################Trip level  EW score #########################.drop(columns='overall_score')
# Without any standard
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# GMM	0.128	0.000	0.000
# Kmeans	0.156	0.000	0.000
# BGM	0.112	0.000	0.000
# Without standard
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# GMM	0.299	0.000	0.000
# Kmeans	0.281	0.000	0.000
# BGM	0.333	0.000	0.000
# Without MaxMin
# methods  silhouette_score  calinski_harabasz_score  davies_bouldin_score
# Kmeans	0.215	24.962	1.420
# GMM	0.342	16.026	0.776
# BGM	0.365	25.992	0.933
