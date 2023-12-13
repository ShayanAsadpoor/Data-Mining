import numpy as np
import pandas as pd
from numpy.linalg import svd
from numpy.linalg import eig
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import plotly.plotly
import plotly.graph_objs as go
import scipy.stats as stats
from plotly.offline import *

X = pd.read_csv("data.txt", delim_whitespace=True, header=None)

# center the data
X_std = StandardScaler().fit_transform(X)
def center():
    mu = np.mean(X_std, axis=0)
    X_norm = X_std - mu
    return X_norm


# Compute the covariance matrix of the data
X_norm = center()


def covar():
    m = X_norm.shape[0]
    cov_mat = (X_norm.T @ X_norm) / m
    return cov_mat


cov_mat = covar()
cov_mat.shape
print(cov_mat)

U, S, _ = svd(
    cov_mat
)  # The V vector is not important here, so just use _ instead of that
print(U)  # U is the Eigenvectors
print(U.shape)
print("----------------")
print(S)  # S is the Eigenvalues
print(S.shape)

eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print("Eigenvectors \n%s" % eig_vecs)
print("\nEigenvalues \n%s" % eig_vals)

# Sort the eigenvalues and eigenvectors
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]

eig_pairs.sort()
eig_pairs.reverse()

print("Eigenvalues in descending order:")
for i in eig_pairs:
    print(i[0])


# number of principal components
m, n = X_std.shape


def prin_comp_num():
    for r in range(1, n + 1):
        total_var = np.sum(S[:r]) / np.sum(S)
        print("r = {:d}, explained variance = {:.3f}".format(r, total_var))
        if total_var >= 0.90:
            break


prin_comp_num()


tot = sum(eig_vals)
var_exp = [(i / tot) * 100 for i in sorted(eig_vals, reverse=True)]
cum_var_exp = np.cumsum(var_exp)
trace1 = dict(
    type="bar", x=["PC %s" % i for i in range(1, 4)], y=var_exp, showlegend=False
)

trace2 = dict(
    type="scatter",
    x=["PC %s" % i for i in range(1, 4)],
    y=cum_var_exp,
    name="cumulative explained variance",
)


data = [trace1, trace2]

layout = dict(
    title="Explained Variance by different Principal Components",
    yaxis=dict(title="Explained variance in percent"),
    annotations=list(
        [
            dict(
                x=1.16,
                y=1.05,
                xref="paper",
                yref="paper",
                text="Explained Variance",
                showarrow=False,
            )
        ]
    ),
)

fig = dict(data=data, layout=layout)
plotly.offline.plot(fig, filename="PCs.html")


# Compute the reduced-dimension data matrix A
A = X_norm @ U[:, :2]
print(A.shape)
print(A)

# plot matrix A
plt.scatter(A[:, 0], A[:, 1])
plt.title("Scatter plot of Matrix A")
plt.axis("equal")
plt.show()


# PCA implementation with the help of sklearn library
pca = PCA(n_components=3).fit(X_norm)
PCA(n_components=2)
print("PCA_explained_variance_ratio=", pca.explained_variance_ratio_)
print("PCA_singular_values=", pca.singular_values_)
print("----------------------------")

U = pca.components_
S = pca.explained_variance_

print("1st Principal Component: {} ({:.2f})".format(U[0], S[0]))
print("2nd Principal Component: {} ({:.2f})".format(U[1], S[1]))
print("3rd Principal Component: {} ({:.2f})".format(U[2], S[2]))
print("---------------------------")
pca = PCA(0.9)  # keep 90% of variance
X_proj = pca.fit_transform(
    X_norm
)  # the first and second columns are exactly the same as matrix A which was computed above

print("X_norm_shape=", X.shape)
print("X_proj_shape=", X_proj.shape)
print("---------------------------")
print("X_projection:", X_proj)
