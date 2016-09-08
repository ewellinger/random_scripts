import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
sns.set_style('white')


def two_vectors(arr, idx0, idx1):
    '''listwise deletion for array having two columns from passed array, returns two 1d arrays of equal length
    arr  : numpy array
    idx0 : column index
    idx1 : column index
    '''
    arr = arr[:, [idx0, idx1]]
    arr = arr[~np.isnan(arr).any(axis=1)]
    return arr[:, 0], arr[:, 1]


def line_fit(x0, x1):
    '''return predicted values for bivariate regression of x1 on x0
    x0 : 1d numpy array, independent variable
    x1 : 1d numpy array, dependent variable
    '''
    x0 = x0.reshape(len(x0), 1)
    mod = LinearRegression()
    return mod.fit(x0, x1).predict(x0)


def bivariate_association_matrix(arr, file_path, labels=None, line_color='k'):
    '''histograms on the diagonal, scatter with line of best fit in the lower triangle, upper triangle suppressed
    arr        : numpy array of features
    file_path  : path to write the image
    labels     : feature names for the array
    line_color : color of the line of best fit
    '''
    dim = arr.shape[1]
    fig_size = dim * 0.75, dim * 0.75
    num_bins = arr.shape[0] / 15
    fig, ax = plt.subplots(dim, dim, figsize=fig_size)
    for i in xrange(dim):
        for j in xrange(dim):
            if j <= i:
                if i == j:
                    ax[i, j].hist(arr[:, i], bins=num_bins, color='k', alpha=0.3)
                else:
                    x, y = two_vectors(arr, i, j)
                    ax[i, j].scatter(x, y, color='k', alpha=0.05)
                    ax[i, j].plot(x, line_fit(x, y), color=line_color, linewidth=1)
                ax[i, j].set_xticklabels([])
                ax[i, j].set_yticklabels([])
                if labels is not None:
                    if j == 0:
                        ax[i, j].set_ylabel(labels[i], labelpad=8, fontsize=8)
                    if i == (dim - 1):
                        ax[i, j].set_xlabel(labels[j], labelpad=8, fontsize=8)
            else:
                ax[i, j].axis('off')
    fig.subplots_adjust(left=0.05, right=0.975, wspace=0.2, hspace=0.1)
    plt.savefig(file_path)
    plt.close()


if __name__ == '__main__':
    data = load_diabetes()
    X, y = data.data, data.target
    cols = ['column' + str(i) for i in xrange(X.shape[1])]
    bivariate_association_matrix(X, file_path='corrs.png')
    bivariate_association_matrix(X, file_path='corrs_labels.png', labels=cols, line_color='b')
