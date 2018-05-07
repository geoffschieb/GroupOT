import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
import copy
import time
from mpl_toolkits.mplot3d import axes3d 
from sklearn.cluster import KMeans

def barycenter_bregman_chain(b_left, b_right, Ms, lambdas, entr_reg_final,
        verbose = False,
        tol = 1e-7,
        max_iter = 500,
        rebalance_thresh = 1e10,
        warm_start = False
        ):

    entr_reg = 1000.0 if warm_start else entr_reg_final

    alphas = [[np.zeros(M.shape[i]) for i in range(2)] for M in Ms]

    Ks = [np.exp(-M/entr_reg) for M in Ms]
    Kts = [K.T.copy() for K in Ks]

    us = [[np.ones(K.shape[i]) for i in range(2)] for K in Ks]
    us_old = copy.deepcopy(us)

    weights = [[float(lambdas[i+j])/(lambdas[i]+lambdas[i+1]) for j in range(2)] for i in range(len(lambdas)-1)]

    # print(lambdas)

    converged = False
    its = 0

    def rebalance(i):
        for j in range(2):
            alphas[i][j] += entr_reg * np.log(us[i][j])
            us[i][j] = np.ones(Ks[i].shape[j])

    def update_K(i):
        Ks[i] = np.exp(-(Ms[i] - alphas[i][0].reshape(-1, 1) - alphas[i][1].reshape(1, -1))/entr_reg)
        Kts[i] = Ks[i].T.copy()

    while not converged and its < max_iter: 
        its += 1
        if verbose:
            print("Barycenter weights iteration: {}".format(its))

        for i in range(len(us)):
            if max([np.max(np.abs(us[i][j])) for j in range(2)]) > rebalance_thresh:
                if verbose:
                    print("Rebalancing pair {}".format(i))
                rebalance(i)
                update_K(i)

        us[0][0] = np.divide(b_left, np.dot(Ks[0], us[0][1]))
        us[-1][1] = np.divide(b_right, np.dot(Kts[-1], us[-1][0]))

        for i in range(len(Ks) - 1):
            Ku1 = np.dot(Kts[i], us[i][0]) 
            Kv2 = np.dot(Ks[i+1], us[i+1][1])
            vKu1 = us[i][1] * Ku1
            uKv2 = us[i+1][0] * Kv2
            p = np.exp(weights[i][0] * np.log(vKu1) + weights[i][1] * np.log(uKv2))
            us[i][1] = np.divide(us[i][1] * p, vKu1)
            us[i+1][0] = np.divide(us[i+1][0] * p, uKv2)

        err = sum([np.linalg.norm(us[i][j] - us_old[i][j])/np.linalg.norm(us[i][j]) for i in range(len(us)) for j in range(2)])
        if verbose:
            print("Relative error: {}".format(err))
        # if its == 1:
        #     print(u1)
        # print(v1)

        us_old = copy.deepcopy(us)

        if err < tol:
            if not warm_start or abs(entr_reg - entr_reg_final) < 1e-12:
                converged = True
            else:
                for i in range(len(us)):
                    rebalance(i)

                entr_reg = max(entr_reg/2, entr_reg_final)
                if verbose:
                    print("New regularization parameter: {}".format(entr_reg))

                for i in range(len(us)):
                    update_K(i)

    gammas = [us[i][0].reshape(-1, 1) * Ks[i] * us[i][1].reshape(1, -1) for i in range(len(us))]
    return gammas


def update_points(xs, xt, zs1, zs2, gammas, lambdas,
        tol = 1e-10,
        max_iter = 50,
        verbose = False):

    converged = False
    its = 0

    weights = [(lambdas[0] * np.sum(gammas[0], axis = 0) + lambdas[1] * np.sum(gammas[1], axis = 1)),
            (lambdas[1] * np.sum(gammas[1], axis = 0) + lambdas[2] * np.sum(gammas[2], axis = 1))]

    point_avgs = [lambdas[0] * np.dot(xs.T, gammas[0]).T, lambdas[2] * np.dot(gammas[2], xt)]

    while not converged and its < max_iter:
        its += 1
        zs1_old = zs1.copy()
        zs2_old = zs2.copy()
        zs1 = (point_avgs[0] + lambdas[1] * np.dot(gammas[1], zs2))/weights[0].reshape(-1, 1)
        zs2 = (lambdas[1] * np.dot(zs1.T, gammas[1]).T + point_avgs[1])/weights[1].reshape(-1, 1)
        err = sum([np.linalg.norm(z-z_old, "fro")/np.linalg.norm(z, "fro") for (z, z_old) in [(zs1, zs1_old), (zs2, zs2_old)]])

        if err < tol:
            converged = True
            print("Total inner iterations: {}".format(its))

    return (zs1, zs2)


def cluster_ot(b1, b2, xs, xt, k1, k2,
        lambdas, entr_reg,
        tol = 1e-4, max_iter = 1000,
        verbose = True):

    converged = False

    d = xs.shape[1]
    zs1  = np.random.normal(size = (k1, d))
    zs2  = np.random.normal(size = (k2, d))

    its = 0

    cost_old = np.float("Inf")
    
    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Alternating barycenter iteration: {}".format(its))
        if its > 1:
            (zs1, zs2) = update_points(xs, xt, zs1, zs2, gammas, lambdas, verbose = False)

        Ms = [ot.dist(xs, zs1), ot.dist(zs1, zs2), ot.dist(zs2, xt)]
        gammas = barycenter_bregman_chain(b1, b2, Ms, lambdas, entr_reg, verbose = False, tol = 1e-5, max_iter = 300)
        cost = np.sum([lambdas[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * stats.entropy(gammas[i].reshape(-1))) for i in range(3)])

        err = np.abs(cost - cost_old)/max(np.abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs1, zs2, np.sum(gammas[1], axis = 1), np.sum(gammas[1], axis = 0), gammas)

def update_points_map(xs, xt, zs1, zs2, gammas, lambdas,
        tol = 1e-8,
        max_iter = 50,
        verbose = False):

    converged = False
    its = 0

    while not converged and its < max_iter:
        its += 1
        zs1_old = zs1.copy()
        zs2_old = zs2.copy()
        alpha = np.sum(gammas[0], axis = 0)
        weights = [(lambdas[0] + lambdas[1]) * alpha,
                (lambdas[1] + lambdas[2]) * alpha]
        point_avgs = [lambdas[0] * np.dot(xs.T, gammas[0]).T,
                lambdas[2] * np.dot(gammas[1], xt)]
        zs1 = (point_avgs[0] + lambdas[1] * alpha.reshape(-1, 1) * zs2)/weights[0].reshape(-1, 1)
        zs2 = (lambdas[1] * alpha.reshape(-1, 1) * zs1 + point_avgs[1])/weights[1].reshape(-1, 1)
        err = sum([np.linalg.norm(z-z_old, "fro")/np.linalg.norm(z, "fro") for (z, z_old) in [(zs1, zs1_old), (zs2, zs2_old)]])

        if err < tol:
            converged = True
            print("Total inner iterations: {}".format(its))

    return (zs1, zs2)

def cluster_ot_map(b1, b2, xs, xt, k,
        lambdas, entr_reg,
        tol = 1e-4, max_iter = 1000,
        verbose = True
        ):
    
    converged = False

    d = xs.shape[1]
    zs1 = np.random.normal(size = (k, d))
    zs2 = np.random.normal(size = (k, d))
    lambdas_barycenter = [lambdas[0], lambdas[2]]
    dist_weight = lambdas[1]/lambdas[0]

    its = 0
    cost_old = np.float("Inf")

    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Alternating barycenter iteration: {}".format(its))
        if its > 1:
            (zs1, zs2) = update_points_map(xs, xt, zs1, zs2, gammas, lambdas, verbose = False)

        Ms = [ot.dist(xs, zs1) + dist_weight*np.sum((zs2 - zs1)**2, axis = 1).reshape(1,-1), ot.dist(zs2, xt)]
        gammas = barycenter_bregman_chain(b1, b2, Ms, lambdas_barycenter, entr_reg, verbose = False)
        cost = np.sum([lambdas_barycenter[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * stats.entropy(gammas[i].reshape(-1))) for i in range(2)])

        err = np.abs(cost - cost_old)/max(np.abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs1, zs2, np.sum(gammas[1], axis = 1), gammas)


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
n_target = 60 # nb samples

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
# M /= M.max()

#OT
#G_ind = ot.emd(a, b, M)
lambd = 1
M = ot.dist(xs, xt)
t0 = time.time()
G_ind = ot.sinkhorn(a, b, M, lambd)
print("Time: {}".format(time.time() - t0))

## plot OT transformation for samples
#ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples')

# #%% clustered OT

# #find CM of clusters
# kmeans = KMeans(n_clusters=num_clust_source, random_state=0).fit(xs)
# CM_source = kmeans.cluster_centers_
# kmeans = KMeans(n_clusters=num_clust_target, random_state=0).fit(xt)
# CM_target = kmeans.cluster_centers_

# # loss matrix
# M_clust = ot.dist(CM_source, CM_target)
# M_clust /= M_clust.max()

# # uniform distribution on CMs
# n1_clust = len(CM_source)
# n2_clust = len(CM_target)
# a_clust, b_clust = np.ones((n1_clust,)) / n1_clust, np.ones((n2_clust,)) / n2_clust # uniform distribution on samples

# #OT
# G_clust = ot.emd(a_clust, b_clust, M_clust)

## plot OT transformation for CMs
#ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
#pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
#pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#pl.legend(loc=0)
#pl.title('OT matrix with samples, CM')

# #%% OT figures
# f, axs = pl.subplots(1,2,figsize=(10,4))

# sub=pl.subplot(131)
# ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# sub.set_title('OT by samples')

# sub=pl.subplot(132)
# ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
# pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
# pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
# pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
# pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
# pl.legend(loc=0)
# sub.set_title('OT by CMs')

# uniform distribution on samples
n1 = len(xs)
n2 = len(xt)
b1, b2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

# t0 = time.time()
# (zs1, zs2, a1, a2, mid_left_source, mid_right_mid_left, mid_right_target) = cluster_ot(b1, b2, xs, xt, 2, 3, 1.0, 1.0, 1, verbose = True)
# print("Time: {}".format(time.time() - t0))
# t0 = time.time()
# (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, xs, xt, 3, 3, [1.0, 1.0, 1.0], 1.0, verbose = True)
# print("Time: {}".format(time.time() - t0))

t0 = time.time()
(zs1, zs2, a, gammas) = cluster_ot_map(b1, b2, xs, xt, 3, [0.5, 1.0, 0.5], 1.0, verbose = True, tol = 1e-5)
print("Time: {}".format(time.time() - t0))

# sub = pl.subplot(133)
pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
ot.plot.plot2D_samples_mat(zs1, zs2, np.diag(np.sum(gammas[0], axis = 0)), c=[.5, .5, .5])
ot.plot.plot2D_samples_mat(zs2, xt, gammas[1], c=[1, .5, .5])
# ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
# ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
# ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
pl.legend(loc=0)
sub.set_title("OT, regularized")

# #%% Test new barycenter

# #%% parameters and data generation
# n_source = 10000 # nb samples
# n_target = 30000 # nb samples

# mu_source = np.array([[0, 0], [0, 5]])
# cov_source = np.array([[1, 0], [0, 1]])

# mu_target = np.array([[10, 1], [10, 5], [10, 10]])
# cov_target = np.array([[1, 0], [0, 1]])

# num_clust_source = mu_source.shape[0]
# num_clust_target = mu_target.shape[0]

# xs = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_source/num_clust_source).astype(int),  mu_source[i,:], cov_source) for i in range(num_clust_source)])
# xt = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_target/num_clust_target).astype(int),  mu_target[i,:], cov_target) for i in range(num_clust_target)])

# ind_clust_source = np.vstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
# ind_clust_target = np.vstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

# zs_left = np.random.normal(size=(100, 2))
# zs_right = np.random.normal(size=(200, 2))

# M1 = ot.dist(xs, zs_left)
# M2 = ot.dist(zs_left, zs_right)
# M3 = ot.dist(zs_right, xt)

# # (gamma1, gamma2) = barycenter_bregman(np.ones(n_source)/n_source, np.ones(6)/6, M1.T, M2, 0.5, 0.5, 0.01, tol = 1e-6)
# # (gamma1_ws, gamma2_ws, gamma3_ws) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 0.01, tol = 1e-6, verbose = True, max_iter = 1000, warm_start = True)
# # t0 = time.time()
# # (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 1, tol = 1e-6, verbose = True, max_iter = 1000, warm_start = False)
# # print("Time: {}".format(time.time() - t0))
# t0 = time.time()
# (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 1, tol = 1e-6, verbose = False, max_iter = 1000, warm_start = True, swap_order = True)
# print("Time: {}".format(time.time() - t0))
# t0 = time.time()
# gammas = barycenter_bregman_chain(np.ones(n_source)/n_source, np.ones(n_target)/n_target, [M1, M2, M3], [1, 1, 1], 1, tol=1e-6, verbose = False, max_iter = 1000, warm_start = True)
# print("Time: {}".format(time.time() - t0))
# # (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 0.2, 0.2, 0.2, 1000, tol = 1e-4, verbose = True, max_iter = 10)
