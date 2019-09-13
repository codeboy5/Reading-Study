import warnings
from sklearn import svm
from matplotlib.colors import ListedColormap
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale=1.2)


df = pd.read_csv('newData/newThree.csv')

X = df[['averageFixationDuration', 'averageRegressionDuration']].values
y = df['target'].values

# X_transformed = StandardScaler().fit_transform(X)

model = SVC(C=1000, gamma=0.0001, kernel='poly')
# model = SVC(kernel='linear')
model.fit(X, y)


def plot_svc_decision_function(model, ax=None, plot_support=True):
    """Plot the decision function for a 2D SVC"""
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # create grid to evaluate model
    x = np.linspace(xlim[0], xlim[1], 30)
    y = np.linspace(ylim[0], ylim[1], 30)
    Y, X = np.meshgrid(y, x)
    xy = np.vstack([X.ravel(), Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)

    # plot decision boundary and margins
    ax.contour(X, Y, P, colors='k',
               levels=[-1, 0, 1], alpha=0.5,
               linestyles=['--', '-', '--'])

    # plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:, 0],
                   model.support_vectors_[:, 1],
                   s=300, linewidth=1, facecolors='none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='autumn')
plot_svc_decision_function(model)
plt.show()


# param_grid = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
# }

# grid = GridSearchCV(SVC(), param_grid, verbose=3)
# grid.fit(X, y)


# sns.lmplot('averageFixationDuration', 'averageRegressionDuration', data=df, hue='target',
#            palette='Set1', fit_reg=False, scatter_kws={"s": 70})
# plt.show()

# sns.scatterplot(x=fixPerLine, y=regPerLine)
# plt.show()


sns.set(font_scale=1.2)

# Import packages to do the classifying


def versiontuple(v):
    return tuple(map(int, (v.split("."))))


def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('blue', 'lightgreen', 'red', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np._version_) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=55, label='test set')


DataSet = pd.read_csv('Dataset.csv')

sns.lmplot('FixationPerLine', 'AverageFixDuration', data=DataSet, hue='TYPE',
           palette='Set1', fit_reg=False, scatter_kws={"s": 70})
plt.show()
features = DataSet[['FixationPerLine', 'AverageFixDuration']].as_matrix()
type_label = np.where(DataSet['TYPE'] == 'easy', 0, 1)

clf = SVC(C=1, kernel='rbf', gamma=0.001)
model = clf.fit(features, type_label)

plot_decision_regions(features, type_label, classifier=clf)
#plt.legend(loc='upper right')
plt.tight_layout()
