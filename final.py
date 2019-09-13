import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, train_test_split

# Plotting
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

df = pd.read_csv('newData/finalNew.csv')


cols = ['averageFixationDuration', 'averageRegressionDuration',
        'averageSaccadeDuration', 'averageSaccadeVelocity', 'target', 'FixDurationPerSaccAmp', 'SaccadeAmpPerLine', 'FixDuration*SaccAmpPerLine']


# * 2D Scatter Plot
# sns.pairplot(data=df,
#              x_vars=["averageRegressionDuration"],
#              y_vars=["FixDurationPerSaccAmp"],
#              height=4.5,
#              hue="target",  # <== 😀 Look here!
#              palette='Set1',
#              plot_kws=dict(edgecolor="k", linewidth=0.5))
# plt.show()

# cols = ['averageFixationDuration', 'averageRegressionDuration',
#         'averageSaccadeDuration', 'averageSaccadeVelocity', 'target']

# pp = sns.pairplot(data=df[cols],
#                   hue='target',  # <== 😀 Look here!
#                   height=1.8, aspect=1.8,
#                   palette='Set1',
#                   plot_kws=dict(edgecolor="black", linewidth=0.5))
# fig = pp.fig
# fig.subplots_adjust(top=0.93, wspace=0.3)
# fig.suptitle('Wine Attributes Pairwise Plots', fontsize=14)
# plt.show()

# * 3d Plot
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
x = 'averageRegressionDuration'
y = 'FixDurationPerSaccAmp'
z = 'SaccadeAmpPerLine'
xs = df[x]
ys = df[y]
zs = df[z]
ax.scatter(xs, ys, zs, c=df['target'],
           cmap='coolwarm', s=50, alpha=0.6, edgecolors='w')
ax.set_xlabel(x)
ax.set_ylabel(y)
ax.set_zlabel(z)
plt.show()
# X = df[['averageFixationDuration', 'averageRegressionDuration',
#         'averageSaccadeDuration']].values
# Y = df['target'].values

# X_train, X_test, y_train, y_test = train_test_split(
#     X, Y, test_size=0.33, shuffle=True)

# svc = SVC(kernel='poly', C=1000, gamma=0.0001)
# svc.fit(X_train, y_train)

# print("train score:", svc.score(X_train, y_train))
# print("test score:", svc.score(X_test, y_test))


# * Grid Search
# param_grid = {
#     'C': [0.1, 1, 10, 100, 1000],
#     'gamma': [1, 0.1, 0.01, 0.001, 0.0001]
# }
# grid = GridSearchCV(SVC(), param_grid, verbose=3)
# grid.fit(X, Y)


# def z(x, y): return (-svc.intercept_[0]-svc.coef_[0]
#                      [0]*x-svc.coef_[0][1]*y) / svc.coef_[0][2]

# tmp = np.linspace(-2, 2, 51)
# x, y = np.meshgrid(tmp, tmp)
# # Plot stuff.
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(x, y, z(x, y))
# ax.plot3D(X[Y == 0, 0], X[Y == 0, 1], X[Y == 0, 2], 'ob')
# ax.plot3D(X[Y == 1, 0], X[Y == 1, 1], X[Y == 1, 2], 'sr')
# plt.show()

# * Correlation Matrix
# corr = df.corr()
# print(corr)
