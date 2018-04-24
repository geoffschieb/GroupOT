import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D 
from sklearn.cluster import KMeans

#%% parameters and data generation
n_source = 30 # nb samples
n_target = 40 # nb samples

mu_source = np.array([[0, 0], [0, 5]])
cov_source = np.array([[1, 0], [0, 1]])

mu_target = np.array([[10, 1], [10, 5], [10, 10]])
cov_target = np.array([[1, 0], [0, 1]])

num_clust_source = mu_source.shape[0]
num_clust_target = mu_target.shape[0]

xs = np.vstack([ot.datasets.get_2D_samples_gauss(n_source/num_clust_source,  mu_source[i,:], cov_source) for i in range(num_clust_source)])
xt = np.vstack([ot.datasets.get_2D_samples_gauss(n_target/num_clust_target,  mu_target[i,:], cov_target) for i in range(num_clust_target)])

ind_clust_source = np.vstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
ind_clust_target = np.vstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

#%% individual OT

# uniform distribution on samples
n1 = len(xs)
n2 = len(xt)
a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

# loss matrix
M = ot.dist(xs, xt)
M /= M.max()

#OT
#G_ind = ot.emd(a, b, M)
lambd = 1e-3
G_ind = ot.sinkhorn(a, b, M, lambd)

## plot OT transformation for samples
#ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples')

#%% clustered OT

#find CM of clusters
kmeans = KMeans(n_clusters=num_clust_source, random_state=0).fit(xs)
CM_source = kmeans.cluster_centers_
kmeans = KMeans(n_clusters=num_clust_target, random_state=0).fit(xt)
CM_target = kmeans.cluster_centers_

# loss matrix
M_clust = ot.dist(CM_source, CM_target)
M_clust /= M_clust.max()

# uniform distribution on CMs
n1_clust = len(CM_source)
n2_clust = len(CM_target)
a_clust, b_clust = np.ones((n1_clust,)) / n1_clust, np.ones((n2_clust,)) / n2_clust # uniform distribution on samples

#OT
G_clust = ot.emd(a_clust, b_clust, M_clust)

## plot OT transformation for CMs
#ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
#pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples, CM')

#%% OT figures
f, axs = pl.subplots(1,2,figsize=(10,4))

sub=pl.subplot(121)
ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by samples')

sub=pl.subplot(122)
ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by CMs')
