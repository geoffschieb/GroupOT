#%%

from __future__ import division
from sklearn import manifold, datasets, decomposition
import matplotlib
from scipy.interpolate import griddata
import numpy as np
from importlib import reload

import sklearn
from scipy.spatial.distance import cdist

import ot

import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D 

# import networkx as nx
from sklearn.neighbors import kneighbors_graph

import random

from sklearn.cluster import KMeans
from numpy import linalg as LA
from sklearn.neighbors import NearestNeighbors

import OT_fixed_k as hot
import Convert_MNN2017 as conv

#%% synthetic biological data (for batch effect in single cell sequencing data)

# parameters and data generation
ns = 200 # number of samples in first experiment
nt = 160 # number of samples in second experiment
num_dim = 100 # number of dimensions of single cell data
proj_mat = np.random.normal(0,1,(2,num_dim)) # projection matrix from 2d to num_dim d
# rand_vec1 = np.random.normal(0,1,(num_dim,1)) # bias vector for first experiment
# rand_vec2 = np.random.normal(28,1,(num_dim,1)) # bias vector for second experiment
# rand_vec2 = np.random.normal(0,1,(num_dim,1)) # bias vector for second experiment
batch_effect = np.random.normal(0, 1, (3, num_dim))

# parameters for mixture of gaussians representing low-d mixture of cell types
# mu_s1 = np.array([0, 0])
# cov_s1 = np.array([[1, 0], [0, 1]])
# mu_s2 = np.array([5, 9])
# cov_s2 = np.array([[1, 0], [0, 1]])
# mu_s3 = np.array([11, 7])
# cov_s3 = np.array([[1, 0], [0, 1]])
mu_s1 = np.array([0, 5])
cov_s1 = np.array([[1, 0], [0, 1]])
mu_s2 = np.array([5, 0])
cov_s2 = np.array([[1, 0], [0, 1]])
mu_s3 = np.array([5, 5])
cov_s3 = np.array([[1, 0], [0, 1]])

# sampling single cells for first experiment
xs1 = ot.datasets.get_2D_samples_gauss(int(ns*0.2), mu_s1, cov_s1)
xs2 = ot.datasets.get_2D_samples_gauss(int(ns*0.3), mu_s2, cov_s1)
xs3 = ot.datasets.get_2D_samples_gauss(int(ns*0.5), mu_s3, cov_s1)
xs = np.vstack((xs1,xs2,xs3))
labs = np.hstack((np.ones(int(ns*0.2)),2*np.ones(int(ns*0.3)),3*np.ones(int(ns*0.5)))).astype(int) # cell type labels
# datas = np.dot(xs,proj_mat) + rand_vec1.T # single cell data 
datas = np.dot(xs,proj_mat) + np.random.normal(0, 1, (xs.shape[0], num_dim)) # single cell data 

# sampling single cells for second experiment
xt1 = ot.datasets.get_2D_samples_gauss(int(nt*0.05), mu_s1, cov_s1)
xt2 = ot.datasets.get_2D_samples_gauss(int(nt*0.65), mu_s2, cov_s1)
xt3 = ot.datasets.get_2D_samples_gauss(int(nt*0.3), mu_s3, cov_s1)
xt = np.vstack((xt1,xt2,xt3))
labt = np.hstack((np.ones(int(nt*0.05)),2*np.ones(int(nt*0.65)),3*np.ones(int(nt*0.3)))).astype(int) # cell type labels
# datat = np.dot(xt,proj_mat) + rand_vec2.T # single cell data 
datat = np.dot(xt,proj_mat) + np.random.normal(0, 1, (xt.shape[0], num_dim)) + np.random.normal(0, 1, (1, num_dim))# single cell data 
# for i in range(len(labt_ind)):
#     datat[labt_ind[i],:] += batch_effect[i,:]

(datas, datat, labs, labt) = conv.convert_data()
labs_ind = [np.where(labs == i)[0] for i in range(1, 4)]
labt_ind = [np.where(labt == i)[0] for i in range(1, 4)]
ns = 1000
nt = 1000

n1 = datas.shape[0]
n2 = datat.shape[0]
a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 # uniform distribution on samples

# distance matrix
M = ot.dist(datas, datat)
M /= M.max()

##embed data for visualization
emb_dat0 = decomposition.PCA(n_components=2).fit_transform(np.vstack((datas,datat)))

#% plot embedded samples
# emb_dat0 = manifold.TSNE(n_components=2).fit_transform(np.vstack((datas,datat)))
pl.plot(emb_dat0[0:ns, 0], emb_dat0[0:ns, 1], '+b', label='Source samples')
pl.plot(emb_dat0[ns:, 0], emb_dat0[ns:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
pl.title('Source and target distributions')
pl.show()

#OT
G0 = ot.emd(a, b, M)
#lambd = 1e-3
#G0 = ot.sinkhorn(a, b, M, lambd)

labt_pred = hot.classify_ot(G0, labs_ind)


mat_results = np.zeros((3,2))
for i in range(1,4):
    mat_results[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G0[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    mat_results[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G0[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))

# print(mat_results)

class_ot = hot.class_err(labt, labt_pred, labt_ind)
print(class_ot)

# mutual nearest neighbors
all_data = np.vstack((datas,datat))
all_data = np.divide(all_data,np.tile(LA.norm(all_data,axis=1),(num_dim,1)).T)

temp_dist = cdist(all_data,all_data, 'euclidean')
temp_dist[0:ns-1,0:ns-1] = np.nan
temp_dist[(ns-1):len(temp_dist),(ns-1):len(temp_dist)] = np.nan
temp_ord = np.argsort(temp_dist,axis=1)

knn = 5
temp_dist1 = np.zeros((len(temp_dist),len(temp_dist)))
for i in range(len(temp_dist)):
    temp_dist1[i,temp_ord[i,0:knn]]=1

G_mnn = np.multiply(temp_dist1,temp_dist1.T)[0:ns,(ns):len(temp_dist)]    

labt_pred_mnn = hot.classify_ot(G_mnn, labs_ind)
mat_results_mnn = np.zeros((3,2))
# for i in range(1,4):
    # mat_results_mnn[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G_mnn[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    # mat_results_mnn[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G0[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G_mnn[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))


# print(mat_results_mnn)
class_mnn = hot.class_err(labt, labt_pred_mnn, labt_ind)
print(class_mnn)

# correlation + majority vote
temp_dist_mm = cdist(all_data,all_data, 'correlation')
temp_dist_mm[0:ns-1,0:ns-1] = np.nan
temp_dist_mm[(ns-1):len(temp_dist),(ns-1):len(temp_dist)] = np.nan

G_mm = temp_dist_mm[0:ns,(ns):len(temp_dist)]
labt_pred_mm = hot.classify_ot(G_mm, labs_ind)

# mat_results_mm = np.zeros((3,2))
class_mm = np.zeros((3,2))
# for i in range(1,4):
    # mat_results_mm[i-1,0] = (1/len(np.where(labs==i)[0]))*((np.sum(G_mm[np.where(labs==i)[0][:, None],np.where(labt==i)[0]]))/(np.sum(G_mm[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt==i)[0])/len(labt))
    # mat_results_mm[i-1,1] = (1/len(np.where(labs==i)[0]))*((np.sum(G_mm[np.where(labs==i)[0][:, None],np.where(labt!=i)[0]]))/(np.sum(G_mm[np.where(labs==i)[0][:, None],:])))/(len(np.where(labt!=i)[0])/len(labt))

# print(mat_results_mm)
class_mm = hot.class_err(labt, labt_pred_mm, labt_ind)
print(class_mm)

# hub OT
k = 10
(zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot.cluster_ot(a, b, datas, datat, k, k, [1.0, 1.0, 1.0], 1e1, debug = True, relax_outside = [1e2, 1e2], warm_start = True)
# (zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot.cluster_ot(a, b, datas, datat, k, k, [1.0, 1.0, 1.0], 1e1, debug = True, relax_inside = [1e4, 1e4], warm_start = True)
# (zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot.cluster_ot(a, b, datas, datat, k, k, [1.0, 1.0, 1.0], 1e1, debug = True, relax_outside = [np.inf, np.inf], warm_start = True)
labt_pred_cluster_ot = hot.classify_cluster_ot(gammas, labs_ind)
class_cluster_ot = hot.class_err(labt, labt_pred_cluster_ot, labt_ind)
print(class_cluster_ot)
emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack([datas, datat, zs1_l[1], zs2_l[1]]))

# pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt_pred, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[(ns+nt):(ns+nt+k),0], emb_dat[(ns+nt):(ns+nt+k), 1], marker = '<', label='zs1')
# pl.scatter(emb_dat[(ns+nt+k):,0], emb_dat[(ns+nt+k):, 1], marker = '>', label='zs1')
# pl.show()

##% plot hub OT
#pl.imshow(G0, interpolation='nearest')
pl.subplot(122)
emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((datas,datat,zs1,zs2)))
colors = ['red','green','blue']
ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[(ns+nt):(ns+nt+k), :], gammas[0], c=[.5, .5, 1])
ot.plot.plot2D_samples_mat(emb_dat[(ns+nt):(ns+nt+k),:], emb_dat[(ns+nt+k):,:], gammas[1], c=[.5, .5, .5])
ot.plot.plot2D_samples_mat(emb_dat[(ns+nt+k):,:], emb_dat[ns:(ns+nt),:], gammas[2], c=[1, .5, .5])
pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
pl.legend()
pl.title('OT matrix with samples')

##% plot OT from source to target
#pl.imshow(G0, interpolation='nearest')
pl.subplot(121)
colors = ['red','green','blue']
ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[ns:(ns+nt), :], G0, c=[.5, .5, 1])
pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
pl.legend()
pl.title('OT matrix with samples')
pl.show()

# # k barycenter
# k = 10
# (zs, gammas) = hot.kbarycenter(a, b, datas, datat, k, [1.0, 1.0], 1e1, warm_start = True, relax_outside = [1e1, 1e1], max_iter = 100)
# # (zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot.cluster_ot(a, b, datas, datat, k, k, [1.0, 1.0, 1.0], 1e1, debug = True, relax_inside = [1e4, 1e4], warm_start = True)
# # (zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot.cluster_ot(a, b, datas, datat, k, k, [1.0, 1.0, 1.0], 1e1, debug = True, relax_outside = [np.inf, np.inf], warm_start = True)
# labt_pred_kbary = hot.classify_kbary(gammas, labs_ind)
# class_cluster_ot = hot.class_err(labt, labt_pred_kbary, labt_ind)
# print(class_cluster_ot)

# ##% plot k barycenter
# #pl.imshow(G0, interpolation='nearest')
# pl.subplot(122)
# emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((datas,datat,zs)))
# colors = ['red','green','blue']
# ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[(ns+nt):, :], gammas[0], c=[.5, .5, 1])
# ot.plot.plot2D_samples_mat(emb_dat[(ns+nt):,:], emb_dat[ns:(ns+nt),:], gammas[1], c=[1, .5, .5])
# pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
# pl.legend()
# pl.title('OT matrix with samples')

# ##% plot OT source to target
# #pl.imshow(G0, interpolation='nearest')
# pl.subplot(121)
# colors = ['red','green','blue']
# ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[ns:(ns+nt), :], G0, c=[.5, .5, 1])
# pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
# pl.legend()
# pl.title('OT matrix with samples')
# pl.show()

# # fixed weights
# k = 8
# (zs1, zs2, gammas) = hot.reweighted_clusters(datas, datat, k, [1.0, 1.0, 1.0], 1e2)

# labt_pred_cluster_ot = hot.classify_cluster_ot(gammas, labs_ind)
# class_cluster_ot = hot.class_err(labt, labt_pred_cluster_ot, labt_ind)
# print(class_cluster_ot)
# # emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack([datas, datat, zs1_l[1], zs2_l[1]]))

# # pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# # pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt_pred, cmap=matplotlib.colors.ListedColormap(colors))
# # pl.scatter(emb_dat[(ns+nt):(ns+nt+k),0], emb_dat[(ns+nt):(ns+nt+k), 1], marker = '<', label='zs1')
# # pl.scatter(emb_dat[(ns+nt+k):,0], emb_dat[(ns+nt+k):, 1], marker = '>', label='zs1')
# # pl.show()

# ##% plot hub OT
# #pl.imshow(G0, interpolation='nearest')
# pl.subplot(122)
# emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((datas,datat,zs1,zs2)))
# colors = ['red','green','blue']
# ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[(ns+nt):(ns+nt+k), :], gammas[0], c=[.5, .5, 1])
# ot.plot.plot2D_samples_mat(emb_dat[(ns+nt):(ns+nt+k),:], emb_dat[(ns+nt+k):,:], gammas[1], c=[.5, .5, .5])
# ot.plot.plot2D_samples_mat(emb_dat[(ns+nt+k):,:], emb_dat[ns:(ns+nt),:], gammas[2], c=[1, .5, .5])
# pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
# pl.legend()
# pl.title('OT matrix with samples')

# ##% plot OT from source to target
# #pl.imshow(G0, interpolation='nearest')
# pl.subplot(121)
# colors = ['red','green','blue']
# ot.plot.plot2D_samples_mat(emb_dat[0:ns, :], emb_dat[ns:(ns+nt), :], G0, c=[.5, .5, 1])
# pl.scatter(emb_dat[0:ns, 0], emb_dat[0:ns, 1], marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors))
# pl.scatter(emb_dat[ns:(ns+nt), 0], emb_dat[ns:(ns+nt), 1], marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors))
# pl.legend()
# pl.title('OT matrix with samples')
# pl.show()

