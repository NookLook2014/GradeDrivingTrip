from ftruck_etl_utils import load_fuel_truck_dag_data
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabaz_score,silhouette_samples, silhouette_score
from keras.models import load_model
import pandas as pd
from time import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn import metrics

freq_model_path = '../bak/encoder_ftruck_freq_dag.model'
time_model_path = '../bak/encoder_ftruck_time_dag.model'

num_freq_features = 27 * 27


def bench_k_means(estimator, name, data, labels):
    t0 = time()
    estimator.fit(data)
    print('%-9s\t%.2fs\t%i\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_, metric='euclidean')))  # the higher the better


def cluster_with_only_freq_vector(data):
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    # data = load_fuel_truck_dag_data().iloc[:,:num_freq_features+1]
    freq_encoder = load_model(freq_model_path)
    # data = data.sample(5000)
    labels = data.iloc[:,0]
    n_digits = len(np.unique(labels))
    input = freq_encoder.predict(data.iloc[:,1:])
    # input = pd.DataFrame(res)
    # input['label'] = labels
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="k-means++", data=input, labels=labels)
    # clusterer  = KMeans(n_clusters=n_digits, random_state=1234)
    # cluster_labels = clusterer.fit_predict(res)
    # silhouette_avg = silhouette_score(res, cluster_labels)
    # print(input)


def visulaize(data, n_digits):
    # #############################################################################
    # Visualize the results on PCA-reduced data

    reduced_data = PCA(n_components=3).fit_transform(data)
    kmeans = KMeans(init='k-means++', n_clusters=n_digits, n_init=10)
    kmeans.fit(reduced_data)

    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .02     # point in the mesh [x_min, x_max]x[y_min, y_max].

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    plt.plot(reduced_data[:, 0], reduced_data[:, 1], 'k.', markersize=2)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=3,
                color='w', zorder=10)
    plt.title('K-means clustering on the digits dataset (PCA-reduced data)\n'
              'Centroids are marked with white cross')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks(())
    plt.yticks(())
    plt.show()


if __name__ == '__main__':
    data = load_fuel_truck_dag_data(path='D:/data/fueltruck/transformed/dag1/part-00000') #.iloc[:,:9*9*1+1]
    # data = scale(data)
    # data = data.sample(5000)
    n_samples, n_features = data.shape
    print(data.shape)
    labels = data.iloc[:,0]
    n_digits = len(np.unique(labels))
    num_states = 27
    feature_dim = num_states * num_states

    freq_encoder = load_model(freq_model_path)
    encoded_freq_input = freq_encoder.predict(data.iloc[:,1:feature_dim+1])
    n_reduced_freq_features = len(encoded_freq_input[0])

    time_encoder = load_model(time_model_path)
    encoded_time_input = time_encoder.predict(data.iloc[:,feature_dim+1:])

    encoded_all_input = np.concatenate((encoded_freq_input, encoded_time_input), axis=1)
    print(len(encoded_all_input[0]))

    pca_input = PCA(n_components=n_reduced_freq_features).fit_transform(data.iloc[:,1:])
    pca_input10 = PCA(n_components=10).fit_transform(data.iloc[:,1:])

    print("n_digits: %d, \t n_samples %d, \t n_features %d,\t n_reduced_features %d" % (n_digits, n_samples, n_features, n_reduced_freq_features))
    print(82 * '_')
    print('init\t\ttime\tinertia\thomo\tcompl\tv-meas\tARI\tAMI\tsilhouette')
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="ae_freq_k-means++", data=encoded_freq_input, labels=labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="ae_time_k-means++", data=encoded_time_input, labels=labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="ae_all_k-means++", data=encoded_all_input, labels=labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
              name="pca_5_k-means++", data=pca_input, labels=labels)
    bench_k_means(KMeans(init='k-means++', n_clusters=n_digits, n_init=10),
                  name="pca_10_k-means++", data=pca_input10, labels=labels)

    # cluster_with_only_freq_vector()