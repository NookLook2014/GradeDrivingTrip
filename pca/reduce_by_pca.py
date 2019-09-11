import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.covariance import ShrunkCovariance, LedoitWolf
from sklearn.decomposition import PCA, SparsePCA, FactorAnalysis, NMF
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from ftruck_etl_utils import load_fuel_truck_dag_data

'''
This script first helps find the best dimensionality given input data and param grid with PCA and FA.
Reasons: The two methods themselves provide score methods.
'''


def compute_scores(X, n_components):
    pca = PCA(svd_solver='auto')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))
    return pca_scores, fa_scores


def shrunk_cov_score(X):
    shrinkages = np.logspace(-2, 0, 30)
    cv = GridSearchCV(ShrunkCovariance(), {'shrinkage': shrinkages}, cv=5)
    return np.mean(cross_val_score(cv.fit(X).best_estimator_, X, cv=5))


def lw_score(X):
    return np.mean(cross_val_score(LedoitWolf(), X, cv=5))


def best_components_PCA_FA(X):
    '''
    aims to find the best and minimum number of components
    :return:
    '''
    # #############################################################################
    # Fit the models
    n_components = [3, 5, 7, 9, 15, 25, 35, 45, 55, 65, 75, 85, 95, 100, 150, 200]#np.arange(5, 201, 10)  # options for n_components
    pca_scores, fa_scores = compute_scores(X, n_components)
    n_components_pca = n_components[np.argmax(pca_scores)]
    n_components_fa = n_components[np.argmax(fa_scores)]

    print("best n_components by PCA CV = %d" % n_components_pca)
    print("best n_components by FactorAnalysis CV = %d" % n_components_fa)

    plt.figure()
    plt.plot(n_components, pca_scores, 'b', label='PCA scores')
    plt.plot(n_components, fa_scores, 'r', label='FA scores')
    # plt.axvline(rank, color='g', label='TRUTH: %d' % rank, linestyle='-')
    plt.axvline(n_components_pca, color='b',
                label='PCA CV: %d' % n_components_pca, linestyle='--')
    plt.axvline(n_components_fa, color='r',
                label='FactorAnalysis CV: %d' % n_components_fa,
                linestyle='--')
    # plt.axvline(n_components_pca_mle, color='k',
    #             label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

    # compare with other covariance estimators
    plt.axhline(shrunk_cov_score(X), color='violet',
                label='Shrunk Covariance MLE', linestyle='-.')
    plt.axhline(lw_score(X), color='orange',
                label='LedoitWolf MLE % n_components_pca_mle', linestyle='-.')

    plt.xlabel('nb of components')
    plt.ylabel('CV scores')
    plt.legend(loc='lower right')
    plt.title('Automatic Choice of Reduced Dimensionality')
    plt.show()


def pca_reduce(data, n_components):
    # pca = SparsePCA(n_components=n_components, normalize_components=True, random_state=0)
    pca = SparsePCA(n_components=n_components, random_state=0)
    output = pca.fit_transform(data)
    return output


def test_NMF(data):
    model = NMF(init='random', random_state=0)
    n_components = np.arange(5, 100, 10)  # options for n_components
    errors = []
    for component in n_components:
        model.n_components_ = component
        model.fit(data)
        errors.append(model.reconstruction_err_)
    plt.plot(x=n_components, y=errors)
    plt.xlabel('number of components')
    plt.ylabel('reconstruction error (F-Norm)')
    plt.show()


def plot_cev(X, title):
    pca = PCA().fit(X)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.title(title)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    ticks = [x for x in range(X.shape[1] + 1) if x % 40 == 0]  #np.linspace(0,X.shape[1],5)
    plt.xticks(ticks)
    plt.grid()
    plt.show()


def plot_recons_error(X, components):
    for n in components:
        pca = PCA(n)#.fit(X)
        trans = pca.fit_transform(X)
        X_hat = pca.inverse_transform(trans)

        pca.score()



if __name__ == '__main__':
    num_features_freq = 27 * 27
    # X = load_fuel_truck_dag_data().iloc[:,1:]
    # plot_cev(X, 'cumulative explained variance ratio of all')
    # X = load_fuel_truck_dag_data().iloc[:, 1:num_features_freq+1]
    # plot_cev(X, 'cumulative explained variance ratio of freq features')
    X = load_fuel_truck_dag_data().iloc[:, num_features_freq+1:]
    plot_cev(X, 'cumulative explained variance ratio of duration features')

    # best_components_PCA_FA(X)  #best n_components by PCA CV = 55 FA = 5
    # test_NMF(X)