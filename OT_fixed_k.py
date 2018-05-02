import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import axes3d 
from sklearn.cluster import KMeans

def barycenter_bregman(b1, b2, M1, M2, w1, w2, entr_reg,
        verbose = False,
        tol = 1e-7,
        max_iter = 500
        ):

    K1 = np.exp(-M1/entr_reg)
    K2 = np.exp(-M2/entr_reg)
    K1t = K1.T.copy()
    K2t = K2.T.copy()
    n = K1.shape[0]
    v1 = np.ones(K1.shape[1])
    v2 = np.ones(K2.shape[1])
    v1_old = v1.copy()
    v2_old = v2.copy()
    u1 = np.ones(K1.shape[0])
    u2 = np.ones(K2.shape[0])
    u1_old = u1.copy()
    u2_old = u2.copy()

    converged = False
    its = 0

    while not converged and its < max_iter: 
        its += 1
        if verbose:
            print("Barycenter weights iteration: {}".format(its))

        v1 = np.divide(b1, np.dot(K1t, u1))
        v2 = np.divide(b2, np.dot(K2t, u2))
        Kv1 = np.dot(K1, v1)
        Kv2 = np.dot(K2, v2)
        uKv1 = u1 * Kv1
        uKv2 = u2 * Kv2
        p = np.exp(np.log(uKv1) * w1 + np.log(uKv2) * w2)
        u1 = np.divide(u1 * p, uKv1)
        u2 = np.divide(u2 * p, uKv2)

#         Kv = np.dot(K, v)
#         p = np.dot(u, Kv)
#         u = np.divide(p, Kv)
#         v = np.divide(b, np.dot(Kt, u))

        # Kv = np.dot(K, v)
        # u = np.divide(a, Kv)
        # v = np.divide(b, np.dot(Kt, u))

        err = np.linalg.norm(u1 - u1_old)/np.linalg.norm(u1) + np.linalg.norm(v1 - v1_old) / np.linalg.norm(v1)
        err += np.linalg.norm(u2 - u2_old)/np.linalg.norm(u2) + np.linalg.norm(v2 - v2_old) / np.linalg.norm(v2)
        # print(err)
        # if its == 1:
        #     print(u1)
        # print(v1)
        u1_old = u1.copy()
        v1_old = v1.copy()
        u2_old = u2.copy()
        v2_old = v2.copy()
        if err < tol:
            converged = True

    gamma1 = np.dot(np.diag(u1), np.dot(K1, np.diag(v1)))
    gamma2 = np.dot(np.diag(u2), np.dot(K2, np.diag(v2)))
    # return (np.dot(gamma1, np.ones(K1.shape[1])), np.dot(gamma2, np.ones(K2.shape[1])))
    # return np.dot(gamma1, np.ones(K1.shape[1]))
    return (gamma1, gamma2)

def barycenter_free(b1, b2, xs1, xs2, w1, w2, entr_reg, k,
        tol = 1e-5,
        max_iter = 100,
        verbose = False
        ):

    d = xs1.shape[1]
    ys = np.random.normal(size = (k, d))
    cost_old = np.float("Inf")
    its = 0
    converged = False

    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Barycenter points iteration: {}".format(its))

        if its > 1:
            ys = (w1 * np.matmul(gamma1, xs1) + w2 * np.matmul(gamma2, xs2))/(np.dot((w1 * np.sum(gamma1, axis = 1) + w2 * np.sum(gamma2, axis = 1)).reshape((k, 1)), np.ones((1, d))))

        M1 = ot.dist(ys, xs1)
        M2 = ot.dist(ys, xs2)
        (gamma1, gamma2) = barycenter_bregman(b1, b2, M1, M2, w1, w2, entr_reg)
        cost = (w1 * np.sum(gamma1 * M1) + w2 * np.sum(gamma2 * M2)) + entr_reg * (w1 * stats.entropy(gamma1.reshape((-1,1))) + w2 * stats.entropy(gamma2.reshape((-1,1))))
        # TODO: Calculate the entropy safely
        # cost = w1 * np.sum(gamma1 * M1) + w2 * np.sum(gamma2 * M2)

        err = abs(cost - cost_old)/max(abs(cost), 1e-12)
        cost_old = cost.copy()
        if verbose:
            print("Relative change in points iteration: {}".format(err))
        if err < tol:
            converged = True

    return (ys, np.dot(gamma1, np.ones(gamma1.shape[1])), cost, gamma1, gamma2)

def cluster_ot(b1, b2, xs1, xs2, k1, k2,
        source_reg, target_reg, entr_reg,
        tol = 1e-4, max_iter = 100,
        verbose = True):

    converged = False

    d = xs1.shape[1]
    zs1  = np.random.normal(size = (k1, d))
    zs2  = np.random.normal(size = (k2, d))
    a1 = np.ones(k1)/k1
    a2 = np.ones(k2)/k2

    w_left = 1 + source_reg
    w_right = 1 + target_reg
    w_1 = source_reg/w_left
    w_mid_1 = 1/w_left
    w_mid_2 = 1/w_right
    w_2 = target_reg/w_right

    its = 0

    cost_old = np.float("Inf")
    
    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Alternating barycenter iteration: {}".format(its))
        # Alternate between computing barycenters
        (zs1, a1, cost1, mid_left_source, mid_left_mid_right) = barycenter_free(b1, a2, xs1, zs2, w_1, w_mid_1, entr_reg, k1)
        (zs2, a2, cost2, mid_right_mid_left, mid_right_target) = barycenter_free(a1, b2, zs1, xs2, w_mid_2, w_2, entr_reg, k2)
        cost = w_left * cost1 + w_right * cost2

        err = abs(cost - cost_old)/max(abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs1, zs2, a1, a2, mid_left_source, mid_right_mid_left, mid_right_target)

### Barycenter histogram test

# lambd = 1e-2
# # n = 20
# # xs = np.linspace(0, 10, n)
# # xs = xs.reshape((xs.shape[0], 1))
# # ys = xs.copy()
# xm = np.vstack((xm, np.array([0, -10])))
# M1 = ot.dist(xs, xm)
# M1 /= M1.max()
# M2 = ot.dist(xt, xm)
# M2 /= M2.max()
# n1 = xs.shape[0]
# n2 = xt.shape[0]
# b1 = np.ones(n1)/n1
# b2 = np.ones(n2)/n2
# (gamma1, gamma2) = barycenter_bregman(b1, b2, M1.T, M2.T, lambd)
# # print(ot.sinkhorn2(a, b, M, lambd))
# x_combined = np.vstack((xs, xt, xm))
# b_combined = np.zeros((x_combined.shape[0], 2))
# b_combined[0:b1.shape[0], 0] = b1
# b_combined[b1.shape[0]:(b1.shape[0]+b2.shape[0]), 1] = b2
# b_combined += 1e-5
# b_combined[:,0] / np.sum(b_combined[:,0])
# b_combined[:,1] / np.sum(b_combined[:,1])
# M = ot.dist(x_combined, x_combined)
# a_ot = ot.bregman.barycenter(b_combined, M, lambd)

# ### Barycenter fixed point test

# lambd = 0.1
# cov1 = np.array([[1, 0], [0, 1]])
# cov2 = np.array([[1, 0], [0, 1]])
# mu1 = np.zeros(2)
# mu2 = np.zeros(2)
# n = 400
# k = 100
# xs1 = ot.datasets.get_2D_samples_gauss(n, mu1, cov1)
# xs2 = ot.datasets.get_2D_samples_gauss(n, mu2, cov2)
# xs1 /= np.dot(np.linalg.norm(xs1, axis = 1).reshape(n, 1), np.ones((1, 2)))
# xs2 /= np.dot(np.linalg.norm(xs2, axis = 1).reshape(n, 1), np.ones((1, 2)))
# xs1[:,0] *= 5
# xs2[:,1] *= 5
# b1 = np.ones(n)/n
# b2 = b1.copy()
# (ys, weights, cost) = barycenter_free(b1, b2, xs1, xs2, 0.1, 0.9, lambd, k)
# print("Cost = {}".format(cost))
# pl.plot(xs1[:,0], xs1[:,1], 'xb')
# pl.plot(xs2[:,0], xs2[:,1], 'xr')
# pl.plot(ys[:,0], ys[:,1], 'xg')
# pl.show()

#%% Test run

#%% parameters and data generation
n_source = 40 # nb samples
n_target = 30 # nb samples

mu_source = np.array([[0, 0], [0, 5]])
cov_source = np.array([[1, 0], [0, 1]])

mu_target = np.array([[10, 1], [10, 5], [10, 10]])
cov_target = np.array([[1, 0], [0, 1]])

num_clust_source = mu_source.shape[0]
num_clust_target = mu_target.shape[0]

xs = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_source/num_clust_source).astype(int),  mu_source[i,:], cov_source) for i in range(num_clust_source)])
xt = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_target/num_clust_target).astype(int),  mu_target[i,:], cov_target) for i in range(num_clust_target)])

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

sub=pl.subplot(131)
ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by samples')

sub=pl.subplot(132)
ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.legend(loc=0)
sub.set_title('OT by CMs')

# uniform distribution on samples
n1 = len(xs)
n2 = len(xt)
b1, b2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

(zs1, zs2, a1, a2, mid_left_source, mid_right_mid_left, mid_right_target) = cluster_ot(b1, b2, xs, xt, 2, 3, 1.0, 1.0, 1, verbose = True)

sub = pl.subplot(133)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
ot.plot.plot2D_samples_mat(zs1, xs, mid_left_source, c=[.5, .5, 1])
ot.plot.plot2D_samples_mat(zs2, zs1, mid_right_mid_left, c=[.5, .5, .5])
ot.plot.plot2D_samples_mat(zs2, xt, mid_right_target, c=[1, .5, .5])
pl.legend(loc=0)
sub.set_title("OT, regularized")
