import numpy as np
import scipy.stats as stats
from scipy.io import loadmat, savemat
import matplotlib
import ot
import scipy as sp
from scipy.special import kl_div, entr
from sklearn import manifold, datasets, decomposition
import matplotlib.pylab as pl
import copy
import time
from mpl_toolkits.mplot3d import axes3d 
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
import itertools
import warnings
import pickle
import os
from sklearn import preprocessing

#%% Utility functions

def opt_grid(fun, *args):
    results = np.empty(tuple(map(len, args)))
    for comb_ind in itertools.product(*map(lambda x: range(len(x)), args)):
        results[comb_ind] = fun([args[i][comb_ind[i]] for i in range(len(args))])
    multi_opt = np.unravel_index(np.argmin(results), results.shape)
    return [args[i][multi_opt[i]] for i in range(len(args))]

#%% OT functions

def kl_div_vec(x, y):
    return np.sum(kl_div(x, y))

def entr_vec(x):
    return np.sum(entr(x))

# @profile
def barycenter_bregman_chain(b_left, b_right, Ms, lambdas, entr_reg_final,
        relax_outside = [np.float("Inf"), np.float("Inf")],
        relax_inside = [np.float("Inf"), np.float("Inf")], # TODO This should adapt to the size of Ms
        verbose = False,
        tol = 1e-7,
        max_iter = 500,
        rebalance_thresh = 1e10,
        warm_start = False,
        entr_reg_start = 1000.0
        ):

    warnings.filterwarnings("error")
    try:
        entr_reg = entr_reg_start if warm_start else entr_reg_final
        np.seterr(all="warn")

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

            if relax_outside[0] == np.float("Inf"):
                us[0][0] = np.divide(b_left, np.dot(Ks[0], us[0][1]))
            else:
                us[0][0] = np.exp(relax_outside[0]/(relax_outside[0] + entr_reg) * np.log(np.divide(b_left, np.dot(Ks[0], us[0][1]))))
            if relax_outside[1] == np.float("Inf"):
                us[-1][1] = np.divide(b_right, np.dot(Kts[-1], us[-1][0]))
            else:
                us[-1][1] = np.exp(relax_outside[1]/(relax_outside[1] + entr_reg) * np.log(np.divide(b_right, np.dot(Kts[-1], us[-1][0]))))

            for i in range(len(Ks) - 1):
                Ku1 = np.dot(Kts[i], us[i][0]) 
                Kv2 = np.dot(Ks[i+1], us[i+1][1])
                vKu1 = us[i][1] * Ku1
                uKv2 = us[i+1][0] * Kv2
                if relax_inside[i] == np.float("Inf"):
                    p = np.exp(weights[i][0] * np.log(vKu1) + weights[i][1] * np.log(uKv2))
                    us[i][1] = np.divide(us[i][1] * p, vKu1)
                    us[i+1][0] = np.divide(us[i+1][0] * p, uKv2)
                else:
                    p = weights[i][0] * np.exp(entr_reg/(entr_reg + relax_inside[i]) * np.log(vKu1)) + \
                            weights[i][1] * np.exp(entr_reg/(entr_reg + relax_inside[i]) * np.log(uKv2))
                    p = np.exp((entr_reg + relax_inside[i])/entr_reg * np.log(p))
                    us[i][1] = np.exp(relax_inside[i]/(relax_inside[i] + entr_reg) * np.log(np.divide(us[i][1] * p, vKu1)))
                    us[i+1][0] = np.exp(relax_inside[i]/(relax_inside[i] + entr_reg) * np.log(np.divide(us[i+1][0] * p, uKv2)))

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

                    entr_reg = max(entr_reg/5, entr_reg_final)
                    if verbose:
                    # if True:
                        print("New regularization parameter: {} at iteration {}".format(entr_reg, its))

                    for i in range(len(us)):
                        update_K(i)
    except RuntimeWarning:
        print("Warning issued!")
        return None
    finally:
        warnings.filterwarnings("default")

    gammas = [us[i][0].reshape(-1, 1) * Ks[i] * us[i][1].reshape(1, -1) for i in range(len(us))]
    if verbose:
        print("Bregman iterations: {}".format(its))
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
            # print("Total inner iterations: {}".format(its))

    return (zs1, zs2)

def plot_transport_map(xs, xt):
    for i in range(xs.shape[0]):
        pl.plot([xs[i, 0], xt[i, 0]], [xs[i, 1], xt[i, 1]], 'k', alpha = 0.5)

# @profile
def cluster_ot(b1, b2, xs, xt, k1, k2,
        lambdas, entr_reg,
        relax_outside = [np.float("Inf"), np.float("Inf")],
        relax_inside = [np.float("Inf"), np.float("Inf")],
        tol = 1e-4,
        inner_tol = 1e-5,
        max_iter = 1000,
        inner_max_iter = 1000,
        verbose = False,
        debug = False,
        warm_start = False,
        # iter_mult = 10,
        # iter_inc = 10
        reduced_inner_tol = False,
        inner_tol_fact = 0.1,
        inner_tol_start = 1e-3,
        inner_tol_dec_every = 20,
        entr_reg_start = 1000.0
        ):

    converged = False
    np.seterr(all="warn")

    d = xs.shape[1]
    # zs1  = 100 * np.random.normal(size = (k1, d))
    # zs2  = 100 * np.random.normal(size = (k2, d))
    zs1 = KMeans(n_clusters = k1).fit(xs).cluster_centers_
    zs2 = KMeans(n_clusters = k2).fit(xt).cluster_centers_

    if debug:
        zs1_l = [zs1]
        zs2_l = [zs2]

    its = 0
    # inner_its = iter_mult
    inner_tol_actual = inner_tol if not reduced_inner_tol else inner_tol_start

    cost_old = np.float("Inf")
    
    while not converged and its < max_iter:
        its += 1
        if reduced_inner_tol and its % inner_tol_dec_every == 0:
            inner_tol_actual *= inner_tol_fact
            inner_tol_actual = max(inner_tol, inner_tol_actual)
        if verbose:
            print("Alternating barycenter iteration: {}".format(its))
        if its > 1:
            (zs1, zs2) = update_points(xs, xt, zs1, zs2, gammas, lambdas, verbose = False)
            if debug:
                zs1_l.append(zs1)
                zs2_l.append(zs2)

        Ms = [ot.dist(xs, zs1), ot.dist(zs1, zs2), ot.dist(zs2, xt)]
        gammas = barycenter_bregman_chain(b1, b2, Ms, lambdas, entr_reg,
                verbose = False,
                tol = inner_tol_actual,
                max_iter = inner_max_iter,
                relax_outside = relax_outside,
                relax_inside = relax_inside,
                warm_start = warm_start,
                entr_reg_start = entr_reg_start
                )
        if gammas is None:
            return None 
        cost = np.sum([lambdas[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * entr_vec(gammas[i].reshape(-1))) for i in range(3)])
        if relax_outside[0] != np.float("Inf"):
            cost += lambdas[0] * relax_outside[0] * kl_div_vec(np.sum(gammas[0], axis = 1), b1)
        if relax_outside[1] != np.float("Inf"):
            cost += lambdas[2] * relax_outside[1] * kl_div_vec(np.sum(gammas[2], axis = 0), b2)
        # TODO: Incorporate loss, but then I need to incorporate an iteration for the barycenter
        # if relax_inside[0] != np.float("Inf"):
        #     cost += lambdas[0] * relax_inside[0] * kl_div_vec(np.sum(gammas[0], axis = 0, 

        err = np.abs(cost - cost_old)/max(np.abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            if reduced_inner_tol and inner_tol_actual > inner_tol:
                inner_tol_actual = max(inner_tol, inner_tol_actual * inner_tol_fact)
            else:
                converged = True

    if debug:
        return (zs1, zs2, np.sum(gammas[1], axis = 1), np.sum(gammas[1], axis = 0), gammas, zs1_l, zs2_l)
    else:
        return (zs1, zs2, np.sum(gammas[1], axis = 1), np.sum(gammas[1], axis = 0), gammas)

def constraint_ot(b1, lb, M, entr_reg,
        tol = 1e-4,
        max_iter = 1000):
    converged = False
    its = 0

    # K = exp(-M/entr_reg)
    # u = np.ones(K.shape[0])
    # v = np.ones(K.shape[1])
    # gamma1 = K.copy()
    gamma1 = np.exp(-M/entr_reg)
    # gamma2 = gamma.copy()
    q1 = np.ones(gamma1.shape)
    q2 = np.ones(gamma1.shape)

    while not converged:
        its += 1
        arg = gamma1 * q1
        # arg = np.exp(np.log(gamma1) + np.log(q1))
        # mult = np.exp(np.log(b1) - np.log(np.sum(arg, axis = 1)))
        mult = b1 / np.sum(arg, axis = 1)
        gamma2 = mult.reshape(-1, 1) * arg
        # gamma2 = np.exp(np.log(mult.reshape(-1, 1)) + np.log(arg))
        # print("gamma2", gamma2)
        q1 *= gamma1/gamma2
        # q1 = np.exp(np.log(q1) - np.log(gamma2) + np.log(gamma1))
        # print("q1", q1)

        arg = gamma2 * q2
        # arg = np.exp(np.log(gamma2) + np.log(q2))
        mult = lb / np.sum(arg, axis = 0)
        # mult = np.exp(np.log(lb) - np.log(np.sum(arg, axis = 0)))
        mult[mult < 1] = 1
        # print(mult.shape)
        # print(arg.shape)
        # print(1, np.log(arg))
        # print(2, np.log(mult))
        # print(3, np.log(mult.reshape(1, -1)) + np.log(arg))
        gamma1 = mult.reshape(1, -1) * arg
        # gamma1 = np.exp(np.log(mult.reshape(1, -1)) + np.log(arg))
        # print("gamma1", gamma1)
        q2 *= gamma2 / gamma1
        # q2 = np.exp(np.log(q2) + np.log(gamma2) - np.log(gamma1))
        # print("q2", q2)

        err = np.linalg.norm(gamma1 - gamma2, "fro")/max(np.linalg.norm(gamma1, "fro"), 1e-12)
        print(its, err)
        if err < tol or its > max_iter:
            converged = True

    return gamma2

def test_constraint_ot():
    global xs, ys
    xs = np.vstack([range(4), np.zeros(4)]).T
    ys = np.vstack([range(4), np.ones(4)]).T
    M = ot.dist(xs, ys)
    a = np.array([0.4, 0.4, 0.1, 0.1])
    lb = 0.2
    gamma = constraint_ot(a, lb, M, 1e0, max_iter = 1000)
    print(np.sum(gamma, axis = 0))

def update_points_barycenter(xs, xt, zs, gammas, lambdas,
        verbose = False):

    zs = (lambdas[0] * np.dot(xs.T, gammas[0]).T + lambdas[1] * np.dot(gammas[1], xt)) / \
        (lambdas[0] * np.sum(gammas[0], axis = 0) + lambdas[1] * np.sum(gammas[1], axis = 1)).reshape(-1,1)

    return zs

def kbarycenter(b1, b2, xs, xt, k,
        lambdas, entr_reg,
        relax_outside = [np.float("Inf"), np.float("Inf")],
        relax_inside = [np.float("Inf")],
        tol = 1e-4,
        inner_tol = 1e-5,
        inner_max_iter = 1000,
        max_iter = 1000,
        warm_start = False,
        verbose = False,
        reduced_inner_tol = False,
        inner_tol_fact = 0.1,
        inner_tol_start = 1e-3,
        inner_tol_dec_every = 20,
        entr_reg_start = 1000.0
        ):

    converged = False

    d = xs.shape[1]
    # zs = np.random.normal(size = (k, d))
    source_means = KMeans(n_clusters = k).fit(xs).cluster_centers_
    # target_means = KMeans(n_clusters = k).fit(xt).cluster_centers_
    # match = ot.emd(np.ones(k)/k, np.ones(k)/k, ot.dist(source_means, target_means))
    # optmap = np.argmax(match, axis = 1)
    # zs = 0.5 * source_means + 0.5 * target_means[optmap]
    zs = source_means

    its = 0
    inner_tol_actual = inner_tol if not reduced_inner_tol else inner_tol_start

    cost_old = np.float("Inf")
    
    while not converged and its < max_iter:
        its += 1
        if its % inner_tol_dec_every == 0:
            inner_tol_actual *= inner_tol_fact
            inner_tol_actual = max(inner_tol_actual, inner_tol)
        if verbose:
            print("k barycenter iteration: {}".format(its))
        if its > 1:
            zs = update_points_barycenter(xs, xt, zs, gammas, lambdas)

        Ms = [ot.dist(xs, zs), ot.dist(zs, xt)]
        gammas = barycenter_bregman_chain(b1, b2, Ms, lambdas, entr_reg,
                verbose = False,
                tol = inner_tol_actual,
                max_iter = inner_max_iter,
                warm_start = warm_start,
                relax_outside = relax_outside,
                entr_reg_start = entr_reg_start)

        # Check for nans
        if gammas is None:
            return None

        cost = np.sum([lambdas[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * entr_vec(gammas[i])) for i in range(2)])

        err = np.abs(cost - cost_old)/max(np.abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs, gammas)

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
            # print("Total inner iterations: {}".format(its))

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
    dist_weight = float(lambdas[1])/lambdas[0]

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

def nearest_points(M):
    gamma = np.zeros_like(M)
    min_ind = np.argmin(M, axis = 1)
    gamma[(range(gamma.shape[0]), min_ind)] = 1.0/gamma.shape[0]

    return gamma

def nearest_points_reg(a, M, entr_reg):
    K = np.exp(-M/entr_reg)
    return np.divide(a, np.sum(K, axis = 1)).reshape(-1, 1) * K

def reweighted_clusters(xs, xt, k, lambdas, entr_reg,
        tol = 1e-10,
        lb = 0):
    converged = False

    d = xs.shape[1]
    # zs1 = np.random.normal(size = (k, d))
    # zs2 = np.random.normal(size = (k, d))
    zs1 = KMeans(n_clusters = k).fit(xs).cluster_centers_
    zs2 = KMeans(n_clusters = k).fit(xt).cluster_centers_
    its = 0
    unif = np.ones(k)/k
    cost_old = np.float("Inf")

    while not converged:
        its += 1
        if its > 1:
            (zs1, zs2) = update_points(xs, xt, zs1, zs2, gammas, lambdas, verbose = False)
        Ms = [ot.dist(xs, zs1), ot.dist(zs1, zs2), ot.dist(zs2, xt)]
        # gamma_left = nearest_points(Ms[0])
        # gamma_right = nearest_points(Ms[2].T).T
        if lb == 0:
            gamma_left = nearest_points_reg(np.ones(Ms[0].shape[0])/Ms[0].shape[0], Ms[0], entr_reg)
            gamma_right = nearest_points_reg(np.ones(Ms[2].shape[1])/Ms[2].shape[1], Ms[2].T, entr_reg).T
        else:
            gamma_left = constraint_ot(np.ones(Ms[0].shape[0])/Ms[0].shape[0], lb, Ms[0], entr_reg)
            gamma_right = constraint_ot(np.ones(Ms[2].shape[1])/Ms[2].shape[1], lb, Ms[2].T, entr_reg).T
        gamma_middle = ot.sinkhorn(unif, unif, Ms[1], entr_reg)
        gammas = [gamma_left, gamma_middle, gamma_right]

        # cost = np.sum([lambdas[i] * np.sum(gammas[i] * Ms[i]) for i in range(3)]) + lambdas[1] * entr_reg * stats.entropy(gammas[1].reshape(-1,1))
        cost = np.sum([lambdas[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * entr_vec(gammas[i])) for i in range(3)])
        err = abs(cost - cost_old)/max(abs(cost), 1e-12)

        if err < tol:
            converged = True

        cost_old = cost

    return (zs1, zs2, gammas)

def kmeans_transport(xs, xt, k, entr_reg):
    clust_sources = KMeans(n_clusters = k).fit(xs)
    clust_targets = KMeans(n_clusters = k).fit(xt)
    zs1 = clust_sources.cluster_centers_
    zs2 = clust_targets.cluster_centers_
    a = np.zeros(zs1.shape[0])
    b = np.zeros(zs2.shape[0])

    for i in range(a.shape[0]):
        a[i] += np.sum(clust_sources.labels_ == i)
    a /= xs.shape[0]
    for i in range(b.shape[0]):
        b[i] += np.sum(clust_targets.labels_ == i)
    b /= xt.shape[0]
    # gamma_clust = ot.sinkhorn(a, b, ot.dist(zs1, zs2), entr_reg)
    gamma_clust = ot.emd(a, b, ot.dist(zs1, zs2))
    gamma_source = np.zeros((xs.shape[0], zs1.shape[0]))
    gamma_source[range(gamma_source.shape[0]), clust_sources.labels_] = 1
    gamma_source /= gamma_source.shape[0]
    gamma_target = np.zeros((xt.shape[0], zs2.shape[0]))
    gamma_target[range(gamma_target.shape[0]), clust_targets.labels_] = 1
    gamma_target /= gamma_target.shape[0]
    return np.dot(np.dot(gamma_source, gamma_clust), gamma_target.T)

def bary_map(gamma, xs):
    # print(np.sum(gamma, axis = 0).shape)
    # print(gamma.T.shape)
    return np.dot(gamma.T / np.sum(gamma, axis = 0).reshape(-1, 1), xs)

def find_centers(xs, xt, gammas):
    if len(gammas) == 3:
        centers1 = np.dot(gammas[0].T, xs)/np.sum(gammas[0], axis = 0).reshape(-1, 1)
        centers2 = np.dot(gammas[2], xt)/np.sum(gammas[2], axis = 1).reshape(-1, 1)
    elif len(gammas) == 2:
        centers1 = np.dot(gammas[0].T, xs)/np.sum(gammas[0], axis = 0).reshape(-1, 1)
        centers2 = np.dot(gammas[1], xt)/np.sum(gammas[1], axis = 1).reshape(-1, 1)

    return (centers1, centers2)

def map_from_clusters(xs, xt, gammas):
    """
    Compute target to source mapping using clusters
    """
    (centers1, centers2) = find_centers(xs, xt, gammas)
    if len(gammas) == 3:
        offsets = np.dot((gammas[1]/np.sum(gammas[1], axis = 0)).T, centers1) - centers2
        return xt + np.dot((gammas[2]/np.sum(gammas[2], axis = 0)).T, offsets)
    else:
        offsets = centers1 - centers2
        return xt + np.dot((gammas[1]/np.sum(gammas[1], axis = 0)).T, offsets)

def classify_1nn(xs, xt, labs):
    # print(np.argmin(ot.dist(xt, xs), axis = 1).shape)
    return labs[np.argmin(ot.dist(xt, xs), axis = 1)]

def classify_knn(xs, xt, labs, k):
    return KNeighborsClassifier(n_neighbors = k).fit(xs, labs).predict(xt)

#%% Estimation/rounding functions

def estimate_w2_cluster(xs, xt, gammas, b0 = None):
    if b0 is None:
        b0 = np.ones(xs.shape[0])/xs.shape[0]

    gammas_thresh = copy.deepcopy(gammas)
    for i in range(len(gammas_thresh)):
        gammas_thresh[i][np.abs(gammas[i]) < 1e-10] = 0

    (centers1, centers2) = find_centers(xs, xt, gammas)
    if len(gammas) == 3:
        costs = np.sum(gammas[1] * ot.dist(centers1, centers2), axis = 1)/np.sum(gammas[1], axis = 1)
    elif len(gammas) == 2:
        costs = np.sum((centers1 - centers2)**2, axis = 1)
    # print(centers1)
    # print(centers2)
    # print(costs)
    total_cost = np.sum((gammas[0] / np.sum(gammas[0], axis = 1).reshape(-1, 1) * b0.reshape(-1, 1)) * costs)
    # total_cost = np.sum((gammas[0] * costs))
    return total_cost

def estimate_distances(b1, b2, xs, xt, ks, entr_reg, lambdas = [1.0, 1.0, 1.0]):
    results = np.zeros(len(ks))
    for (i, k) in enumerate(ks):
        (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, xs, xt, k, k, [1.0, 1.0, 1.0], entr_reg, verbose = True)
        results[i] = estimate_w2_cluster(xs, xt, gammas, b1)
    return results

def calc_lab_ind(lab):
    lab_ind = [np.where(lab == i)[0] for i in np.sort(np.unique(lab))]
    return lab_ind

def classify_ot(G, labs_ind, labs = None):
    odds = np.vstack([np.sum(G[labs_ind[i], :], axis = 0) for i in range(len(labs_ind))]).T
    if labs is None:
        labt_pred = np.argmax(odds, axis = 1)
    else:
        lab_nums = np.sort(np.unique(labs))
        labt_pred = lab_nums[np.argmax(odds, axis = 1)]
    return labt_pred

def total_gamma(gammas):
    if len(gammas) == 3:
        return np.dot(np.dot(gammas[0], gammas[1]/np.sum(gammas[1], axis = 1).reshape(-1, 1)), gammas[2]/np.sum(gammas[2], axis = 1).reshape(-1, 1))
    else:
        return np.dot(gammas[0], gammas[1]/np.sum(gammas[1], axis = 1).reshape(-1, 1))

def classify_cluster_ot(gammas, labs_ind):
    total_gamma = np.dot(np.dot(gammas[0], gammas[1]), gammas[2])
    total_gamma /= np.sum(total_gamma, axis = 0).reshape(1, -1)
    return classify_ot(total_gamma, labs_ind)

def classify_kbary(gammas, labs_ind):
    total_gamma = np.dot(gammas[0], gammas[1])
    total_gamma /= np.sum(total_gamma, axis = 0).reshape(1, -1)
    return classify_ot(total_gamma, labs_ind)
    
def class_err(labt, labt_pred, labt_ind = None):
    if labt_ind is None:
        labt_ind = calc_lab_ind(labt)
    ret = np.zeros((len(labt_ind), 2))
    for i in range(ret.shape[0]):
        ret[i, 0] = np.mean(labt_pred[labt_ind[i]] == labt[labt_ind[i]])
        ret[i, 1] = np.mean(labt_pred[labt_ind[i]] != labt[labt_ind[i]])
    return ret

def class_err_combined(labt, labt_pred):
    return np.mean(labt != labt_pred)

# def bootstrap_param(eval_proc, params, data, coverage, its):
#     gt = eval_proc(data, params[-1])
#     # results = 
#     variances = np.zeros(len(params))
#     biases = np.zeros(len(params))

#     for param in params:
#         results = np.zeros(its)
#         for i in range(its):
#             subsampled_data = []
#             for d in data:
#                 subsampled_data.append(d[np.random.choice(d.shape[0],int(np.floor(d.shape[0] * coverage), :])))
#             results[i] = eval_proc()


def test_gaussian_mixture():
    global xs, xt

    n_cluster = 4
    k = 12
    # prp = np.random.uniform(size = n_cluster)
    # prp /= np.sum(prp)
    prp = np.array([0.1, 0.2, 0.3, 0.4])
    d = 50
    ns = 1000
    nt = 1000
    spread = 1.0
    variance = 1.0
    entr_reg = 10.0

    # # psd matrix
    # A = np.random.normal(0, 1, (d, d))
    # A = A.dot(A.T)
    # print(A)
    # def trafo(x):
    #     return A.dot(x)

    # # Random direction
    # direction = np.random.normal(0, 10/np.sqrt(d), d)
    # def trafo(x):
    #     return x + direction

    # Scaling
    scale = 0.1
    def trafo(x):
        return scale * x

    cluster_centers = np.random.normal(0, spread, (n_cluster, d))
    samples_target = []
    samples_source = []
    cluster_ids = list(range(n_cluster))
    for i in range(ns):
        c = np.random.choice(cluster_ids, p = prp)
        samples_source.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))
    for i in range(nt):
        c = np.random.choice(range(n_cluster), p = prp)
        samples_target.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))

    xs = np.vstack([samples for samples in samples_target])
    xt = np.vstack([samples for samples in samples_source])
    xt = np.apply_along_axis(trafo, 1, xt)

    emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((xs,xt)))
    xs_emb = emb_dat[:xs.shape[0],:]
    xt_emb = emb_dat[xs.shape[0]:,:]
    # xs_emb = xs
    # xt_emb = xt

    pl.plot(*xs_emb.T, 'xr', label='Source samples')
    pl.plot(*xt_emb.T, '+b', label='Target samples')
    pl.show()
    
    ground_truth = np.sum(prp.reshape(-1,1) * (np.apply_along_axis(trafo, 1, cluster_centers) - cluster_centers)**2)
    print("Ground truth: {}".format(ground_truth))

    b1 = np.ones(xs.shape[0])/xs.shape[0]
    b2 = np.ones(xt.shape[0])/xt.shape[0]
    M = ot.dist(xs, xt)
    gamma_ot = ot.sinkhorn(b1, b2, M, entr_reg)
    cost_sinkhorn = np.sum(gamma_ot * M)
    print("Sinkhorn cost: {}".format(cost_sinkhorn))

    (zs, gammas) = kbarycenter(b1, b2, xs, xt, k, [1.0, 1.0], 1, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf],
            max_iter = 100)
    # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
    bary_cost = estimate_w2_cluster(xs, xt, gammas)
    print("Barycenter cost: {}".format(bary_cost))

def test_split_data_uniform_vark():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 100
    n = 10*d
    samples = 20
    results_vanilla = np.empty((samples, len(ks)))
    results_kbary = np.empty((samples, len(ks)))
    for (k_ind, k) in enumerate(ks):
        for sample in range(samples):
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, k_ind] = cost
            print("Sinkhorn: {}".format(cost))

            xs = samples_source
            xt = samples_target

            (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 1, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            results_kbary[sample, k_ind] = bary_cost
            print("Barycenter cost: {}".format(bary_cost))

    print(results_vanilla)
    print(results_kbary)

    with open(os.path.join("uniform_results", "vark.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "n": n,
            "ks": ks,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary
            }, f)

def test_split_data_uniform_varn():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 10
    ns = (np.array([10.0])**np.linspace(1.7, 3, 20)).astype(int)
    samples = 20
    k = 10
    results_vanilla = np.empty((samples, len(ns)))
    results_kbary = np.empty((samples, len(ns)))
    for (n_ind, n) in enumerate(ns):
        for sample in range(samples):
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            cost = ot.emd2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, n_ind] = cost
            print("Sinkhorn: {}".format(cost))

            xs = samples_source
            xt = samples_target

            (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 0.1, verbose = True,
                    warm_start = True, relax_outside = [np.inf, np.inf],
                    tol = 1e-4,
                    inner_tol = 1e-5,
                    max_iter = 500)
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            results_kbary[sample, n_ind] = bary_cost
            print("Barycenter cost: {}".format(bary_cost))

    print(results_vanilla)
    print(results_kbary)

    with open(os.path.join("uniform_results", "varn6.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "ns": ns,
            "k": k,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary
            }, f)

def test_split_data_uniform_visual():
    global ds, results_vanilla, results_kbary
    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 20
    # ns = (np.array([10.0])**np.linspace(1.7, 3, 20)).astype(int)
    n = 100
    k = 4
    figsize = (6,4)

    samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)

    emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((samples_source,samples_target)))
    xs_emb = emb_dat[0:samples_source.shape[0],:]
    xt_emb = emb_dat[samples_source.shape[0]:,:]
    pl.figure(figsize=figsize)
    pl.plot(*xs_emb.T, '+b', label='Source samples')
    pl.plot(*xt_emb.T, 'xr', label='Target samples')

    b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
    b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
    # cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
    M = ot.dist(samples_source, samples_target)
    gamma_ot = ot.sinkhorn(b1, b2, M, 1)
    # ot.plot.plot2D_samples_mat(xs_emb, xt_emb, gamma_ot, c=[.5, .5, 1])
    cost = np.sum(gamma_ot * M)
    print("Sinkhorn: {}".format(cost))
    # pl.show()
    # pl.savefig(os.path.join("Figures","Hypercube_OT.png"), dpi = 300)

    xs = samples_source
    xt = samples_target

    (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 1, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
    # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
    bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
    print("Barycenter cost: {}".format(bary_cost))
    newsource = map_from_clusters(xs_emb, xt_emb, gammas)
    pl.figure(figsize=figsize)
    pl.plot(*xs_emb.T, '+b', label='Source samples')
    pl.plot(*xt_emb.T, 'xr', label='Target samples')
    plot_transport_map(xt_emb, newsource)
    pl.savefig(os.path.join("Figures","Hypercube_kOT.png"), dpi = 300)

def test_split_data_uniform_vard():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    k = 40
    samples = 10
    ds = (np.array([2])**np.linspace(2, 8, 10)).astype(int)
    ns = 10*ds
    results_vanilla = np.empty((samples, len(ds)))
    results_kbary = np.empty((samples, len(ds)))

    for (d_ind, d) in enumerate(ds):
        for sample in range(samples):
            # d = 2
            n = ns[d_ind]
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)

            emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((samples_target,samples_source)))
            # pl.plot(emb_dat[0:samples_target.shape[0], 0], emb_dat[0:samples_target.shape[0], 1], '+b', label='Target samples')
            # pl.plot(emb_dat[samples_target.shape[0]:, 0], emb_dat[samples_target.shape[0]:, 1], 'xr', label='Source samples')
            # pl.show()
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, d_ind] = cost
            print("Sinkhorn: {}".format(cost))

            # ks = range(1, 100)
            # results = estimate_distances(b1, b2, samples_source, samples_target, ks, 1)
            # pl.plot(ks, results)
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 8, 16, [1.0, 1.0, 1.0], 1, verbose = True)
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 3, 3, [1.0, 1.0, 1.0], 1, verbose = True, relax_outside = [1e+5, 1e+5])
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 8, 8, [1.0, 1.0, 1.0], 1, verbose = True, relax_inside = [1e0, 1e0])
            # (zs1, zs2, gammas) = reweighted_clusters(samples_source, samples_target, 8, [1.0, 0.5, 1.0], 5e-1, lb = float(1)/10)
            # cluster_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            # print("Cluster cost: {}".format(cluster_cost))

            xs = samples_source
            xt = samples_target

            # print(zs1)
            # print(zs2)

        #     # hubOT plot
        #     pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
        #     pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
        #     pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
        #     pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
        #     ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
        #     ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
        #     ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
        #     # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
        #     pl.show()

            (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 1, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            results_kbary[sample, d_ind] = bary_cost
            print("Barycenter cost: {}".format(bary_cost))

    #         # Barycenter plot
    #         pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    #         pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    #         pl.plot(zs[:, 0], zs[:, 1], '<c', label='Mid')
    #         # ot.plot.plot2D_samples_mat(xs, zs, gammas[0], c=[.5, .5, 1])
    #         # ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[.5, .5, .5])
    #         plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
    #         pl.show()

    with open(os.path.join("uniform_results", "vard.bin"), "wb") as f:
        pickle.dump({
            "ds": ds,
            "ns": ns,
            "k": k,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary
            }, f)

def gen_rot2d(deg):
    phi = deg/360*2*np.pi
    return np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])

def gen_moons(n_each, y_shift = 0.75, x_shift = 1, radius = 2, noise = 0):
    X = np.zeros((2*n_each, 2))
    for i in range(2):
        angles = np.random.uniform(high = np.pi, size = n_each)
        X[(i * n_each):((i+1)*n_each),:] = radius * np.vstack((np.cos(angles), (1 - 2*i) * np.sin(angles))).T - (1 - 2*i) * np.array([x_shift, y_shift])

    X += np.random.normal(0, noise, size = X.shape)
    y = np.hstack((np.zeros(n_each), np.ones(n_each))).T.astype(int)
    return (X, y)

def gen_moons_data(ns, nt, angle, d, noise, samples):
    data = []
    for sample in range(samples):
        # Generate two moon samples
        (xs, labs) = gen_moons(int(ns/2), noise = 0)
        (xt, labt) = gen_moons(int(nt/2), noise = 0)
        # Rotate target samples
        rho = gen_rot2d(angle)
        xt = np.dot(xt, rho.T)

        # Embed in higher dim
        if d > 2:
            proj = np.random.normal(size = (2, d))
        else:
            proj = np.eye(d)
        xs = np.dot(xs, proj) + noise * np.random.normal(size = (ns, d))
        xt = np.dot(xt, proj) + noise * np.random.normal(size = (nt, d))

        # labs_ind =  calc_lab_ind(labs)
        # data.append((xs, xt, labs, labt, labs_ind))
        data.append((xs, xt, labs, labt))
    return data

def prep_synth_clustering_data(ns, nt, d, prop1, prop2):
    # TODO
    #%% synthetic biological data (for batch effect in single cell sequencing data)
    proj_mat = np.random.normal(0,1,(2,num_dim)) # projection matrix from 2d to num_dim d
    # rand_vec1 = np.random.normal(0,1,(num_dim,1)) # bias vector for first experiment
    # rand_vec2 = np.random.normal(28,1,(num_dim,1)) # bias vector for second experiment
    # rand_vec2 = np.random.normal(0,1,(num_dim,1)) # bias vector for second experiment
    # batch_effect = np.random.normal(0, 1, (3, num_dim))

    # parameters for mixture of gaussians representing low-d mixture of cell types
    mu_s1 = np.array([0, 5])
    cov_s1 = np.array([[1, 0], [0, 1]])
    mu_s2 = np.array([5, 0])
    cov_s2 = np.array([[1, 0], [0, 1]])
    mu_s3 = np.array([5, 5])
    cov_s3 = np.array([[1, 0], [0, 1]])

    # sampling single cells for first experiment
    xs1 = ot.datasets.get_2D_samples_gauss(int(ns*prop1[0]), mu_s1, cov_s1)
    xs2 = ot.datasets.get_2D_samples_gauss(int(ns*prop1[1]), mu_s2, cov_s1)
    xs3 = ot.datasets.get_2D_samples_gauss(int(ns*prop1[2]), mu_s3, cov_s1)
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

def test_synth_clustering_data():
    # TODO
    # parameters and data generation
    ns = 200 # number of samples in first experiment
    nt = 160 # number of samples in second experiment
    num_dim = 100 # number of dimensions of single cell data

def prep_satija_data():
    print("Loading Satija data")
    full_data = np.loadtxt("satija_data.txt", skiprows = 1)
    source_ind = np.where(full_data[:, 1] == 1)[0]
    target_ind = np.where(full_data[:, 1] == 1)[0]
    labs = full_data[source_ind, 0]
    labt = full_data[target_ind, 1]
    xs = full_data[source_ind, 2:]
    xt = full_data[target_ind, 2:]

    return (xs, xt, labs, labt)

def subsample_data(xs, xt, labs, labt, nperclass, samples):
    data = []
    for sample in range(samples):
        inds = []
        indt = []
        for c in np.unique(labs):
            c_ind = np.argwhere(labs == c).ravel()
            np.random.shuffle(c_ind)
            inds.extend(c_ind[:min(nperclass, len(c_ind))])
        for c in np.unique(labt):
            c_ind = np.argwhere(labt == c).ravel()
            np.random.shuffle(c_ind)
            indt.extend(c_ind[:min(nperclass, len(c_ind))])

        labs_ind = calc_lab_ind(labs[inds])
        data.append((xs[inds,:], xt[indt,:], labs[inds], labt[indt], labs_ind))
    return data

def test_moons_kplot():
    noise = 1
    d = 100
    # k = 10
    angle = 40
    ns = 600
    nt = 600
    samples = 10
    ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4)])
    entr_reg = 10

    data = gen_moons_data(ns, nt, angle, d, noise, samples)

    # hubOT
    def hot_eval(k):
        print("Running hubOT for {}".format(k))
        err = 0.0
        w2_estimate = 0.0
        for sample in range(samples):
            print("Sample: {}".format(sample))
            # Compute hub OT
            (xs, xt, labs, labt, labs_ind) = data[sample]
            a = np.ones(ns)/ns
            b = np.ones(nt)/nt
            hot_ret = cluster_ot(a, b, xs, xt, k, k, [1.0, 1.0, 1.0], entr_reg,
                    relax_outside = [np.inf, np.inf],
                    warm_start = False,
                    inner_tol = 1e-4,
                    tol = 1e-4,
                    reduced_inner_tol = True,
                    inner_tol_start = 1e0,
                    max_iter = 300,
                    verbose = False
                    )
            if hot_ret is None:
                err += np.inf
                break
            else:
                (zs1, zs2, a1, a2, gammas) = hot_ret
                # labt_pred_hot = classify_cluster_ot(gammas_k, labs_ind)
                labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas), xs), labs)
                labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas), labs)
                # print(class_err_combined(labt, labt_pred_hot))
                # print(class_err_combined(labt, labt_pred_hot2))
                # print()
                err += class_err_combined(labt, labt_pred_hot2)
                w2_estimate += estimate_w2_cluster(xs, xt, gammas)

        err /= samples
        w2_estimate /= samples
        return (err, w2_estimate)

    errs = np.zeros(len(ks))
    w2s = np.zeros(len(ks))

    for (i, k) in enumerate(ks):
        (err, w2) = hot_eval(k)
        errs[i] = err
        w2s[i] = w2

    pl.subplot(211)
    pl.plot(ks, errs)
    pl.subplot(212)
    pl.plot(ks, w2s)
    pl.show()

def test_moons():
    samples = {"train": 1, "test": 1}

    # entr_regs = np.array([10.0])**range(-3, 5)
    # gl_params = np.array([10.0])**range(-3, 5)
    # ks = np.array([2])**range(1, 7)
    entr_regs = np.array([10.0, 100.0])
    gl_params = np.array([10.0])**range(-3, 5)
    ks = np.array([2])**range(5, 6)

    estimators = {
            # "ot_gl": {
            #     "function": "ot_gl",
            #     "parameter_ranges": [entr_regs, gl_params]
            #     },
            # "ot": {
            #     "function": "ot",
            #     "parameter_ranges": []
            #     },
            # "ot_entr": {
            #     "function": "ot_entr",
            #     "parameter_ranges": [entr_regs]
            #     },
            # "ot_kmeans": {
            #     "function": "ot_kmeans",
            #     "parameter_ranges": [entr_regs, ks]
            #     },
            "ot_2kbary": {
                "function": "ot_2kbary",
                "parameter_ranges": [entr_regs, ks]
                },
            # "ot_kbary": {
            #     "function": "ot_kbary",
            #     "parameter_ranges": [entr_regs, ks]
            # }
            }

    noise = 1
    d = 100
    # k = 10
    angle = 40
    ns = 600
    nt = 600

    simulation_params = {
            "noise": noise,
            "d": d,
            "angle": angle,
            "ns": ns,
            "nt": nt,
            "entr_regs": entr_regs,
            "gl_params": gl_params,
            "ks": ks,
            "samples_test": samples["test"],
            "samples_train": samples["train"],
            "outfile": "two_moons_results.bin",
            "estimators": estimators
            }

    train_data = gen_moons_data(ns, nt, angle, d, noise, samples["train"])
    test_data = gen_moons_data(ns, nt, angle, d, noise, samples["test"])

    def get_data(train, i):
        if train:
            return train_data[i]
        else:
            return test_data[i]

    test_domain_adaptation(simulation_params, get_data)

def test_satija():
    entr_regs = np.array([10.0])**range(-3, 5)
    gl_params = np.array([100.0])**range(-3, 5)
    ks = np.array([2])**range(1, 8)
    samples_test = 10
    samples_train = 10

    nperclass = 30

    simulation_params = {
            "nperclass": nperclass,

            "entr_regs": entr_regs,
            "gl_params": gl_params,
            "ks": ks,
            "samples_test": samples_test,
            "samples_train": samples_train,
            "outfile": "two_moons_results.bin"
            }

    (xs, xt, labs, labt) = prep_satija_data()
    print("Subsampling")
    train_data = subsample_data(xs, xt, labs, labt, nperclass, samples_train)
    test_data = subsample_data(xs, xt, labs, labt, nperclass, samples_test)

    def get_data(train, i):
        if train:
            return train_data[i]
        else:
            return test_data[i]

    test_domain_adaptation(simulation_params, get_data)

def test_domain_adaptation(sim_params, get_data):
    global est_train_results, est_test_results
    
    # Extract data from parameters
    entr_regs = sim_params["entr_regs"]
    ks = sim_params["ks"]
    gl_params = sim_params["gl_params"]
    samples_test = sim_params["samples_test"]
    samples_train = sim_params["samples_train"]
    estimators = sim_params["estimators"]

    # Determine best parameters

    opt_params = {}

    # Regular OT
    def ot_err(data, params):
        print("Running OT for {}".format(params))
        (xs, xt, labs, labt) = data
        # Compute OT mapping
        gamma_ot = ot.da.EMDTransport().fit(Xs = xs, Xt = xt).coupling_
        # if np.any(np.isnan(gamma_ot)):
        #     return(np.inf)
        # else:
        # labt_pred = classify_ot(gamma_ot, labs_ind, labs)
        labt_pred = classify_1nn(xs, bary_map(gamma_ot, xs), labs)
        # labt_pred = classify_knn(xs, bary_map(gamma_ot, xs), labs, 50)
        return(class_err_combined(labt, labt_pred))

    # Entropically regularized OT
    def ot_entr_err(data, params):
        entr_reg = params[0]
        print("Running entr reg OT for {}".format(params))
        (xs, xt, labs, labt) = data
        # Compute OT mapping
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        try:
            warnings.filterwarnings("error")
            gamma_ot = ot.sinkhorn(a, b, ot.dist(xs, xt), entr_reg)
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")
        # if np.any(np.isnan(gamma_ot)):
        #     return(np.inf)
        # else:
        # labt_pred = classify_ot(gamma_ot, labs_ind, labs)
        labt_pred = classify_1nn(xs, bary_map(gamma_ot, xs), labs)
        expected_err = np.sum(gamma_ot * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
        # labt_pred = classify_knn(xs, bary_map(gamma_ot, xs), labs, 50)
        return (expected_err, class_err_combined(labt, labt_pred))

    # k means + OT
    def kmeans_ot_err(data, params):
        (entr_reg, k) = params
        print("Running kmeans + OT for {}".format(params))
        (xs, xt, labs, labt) = data
        gamma_km = kmeans_transport(xs, xt, k, entr_reg)
        if np.any(np.isnan(gamma_km)):
            return(np.inf)
        else:
            # labt_pred_km = classify_ot(gamma_km, labs_ind, labs)
            labt_pred_km = classify_1nn(xs, bary_map(gamma_km, xs), labs)
            return class_err_combined(labt, labt_pred_km)
    
    # hubOT
    def hot_err(data, params):
        (entr_reg, k) = params
        print("Running hubOT for {}".format(params))
        # Compute hub OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        hot_ret = cluster_ot(a, b, xs, xt, k, k, [1.0, 1.0, 1.0], entr_reg,
                relax_outside = [np.inf, np.inf],
                warm_start = False,
                inner_tol = 1e-5,
                tol = 1e-4,
                reduced_inner_tol = True,
                inner_tol_start = 1e0,
                max_iter = 300,
                verbose = False,
                entr_reg_start = 10000.0
                )
        if hot_ret is None:
            return np.inf
        else:
            (zs1, zs2, a1, a2, gammas_k) = hot_ret
            # labt_pred_hot = classify_cluster_ot(gammas_k, labs_ind)
            gamma_total = total_gamma(gammas_k)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas_k), xs), labs)
            labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas_k), labs)
            # print(class_err_combined(labt, labt_pred_hot))
            # print(class_err_combined(labt, labt_pred_hot2))
            # print()
            return [expected_err, class_err_combined(labt, labt_pred_hot), class_err_combined(labt, labt_pred_hot2)]
    
    # k barycenter
    def kbary_err(data, params):
        (entr_reg, k) = params
        print("Running k barycenter for {}".format(params))
        # k barycenter OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        kbary_ret = kbarycenter(a, b, xs, xt, k, [1.0, 1.0],
                entr_reg,
                warm_start = False,
                tol = 1e-4,
                inner_tol = 1e-5,
                reduced_inner_tol = True,
                inner_tol_start = 1e0,
                max_iter = 300,
                verbose = False,
                entr_reg_start = 10000.0
                )
        if kbary_ret is None:
            return np.inf
        else:
            (zs, gammas) = kbary_ret
            # labt_pred_kb = classify_kbary(gammas, labs_ind)
            labt_pred_kb = classify_1nn(xs, bary_map(total_gamma(gammas), xs), labs)
            labt_pred_kb2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas), labs)
            gamma_total = total_gamma(gammas)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # print(class_err_combined(labt, labt_pred_kb))
            # print(class_err_combined(labt, labt_pred_kb2))
            # print()
            return [expected_err, class_err_combined(labt, labt_pred_kb), class_err_combined(labt, labt_pred_kb2)]
    
    # Group lasso
    def gl_ot_err(data, params):
        (xs, xt, labs, labt) = data
        # (entr_reg, eta, samples, train) = params
        (entr_reg, eta) = params
        print("Running group lasso for {}".format(params))
        try:
            warnings.filterwarnings("error")
            gamma_gl = ot.da.SinkhornL1l2Transport(reg_e=entr_reg, reg_cl=eta).fit(Xs = xs, ys = labs, Xt = xt).coupling_
            # labt_pred_gl = classify_ot(gamma_gl, labs_ind, labs)
            labt_pred_gl = classify_1nn(xs, bary_map(gamma_gl, xs), labs)
            return class_err_combined(labt, labt_pred_gl)
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")

    def no_adjust_err(data, params):
        (xs, xt, labs, labt) = data
        print("Running 1NN classifier")
        labt_pred = classify_1nn(xs, xt, labs)
        return class_err_combined(labt, labt_pred)

    def subspace_align_err(data, params):
        (xs, xt, labs, labt) = data
        d = params[0]
        print("Running subspace alignment with {}".format(d))
        if d > max(xs.shape[0], xt.shape[0]):
            return np.inf
        # Subspace Alignment, described in:
        # Unsupervised Visual Domain Adaptation Using Subspace Alignment, 2013,
        # Fernando et al.
        from sklearn.decomposition import PCA
        pcaS = PCA(d).fit(xs)
        pcaT = PCA(d).fit(xt)
        XS = np.transpose(pcaS.components_)[:, :d]  # source subspace matrix
        XT = np.transpose(pcaT.components_)[:, :d]  # target subspace matrix
        Xa = XS.dot(np.transpose(XS)).dot(XT)  # align source subspace
        sourceAdapted = xs.dot(Xa)  # project source in aligned subspace
        targetAdapted = xt.dot(XT)  # project target in target subspace
        labt_pred = classify_1nn(sourceAdapted, targetAdapted, labs)
        return class_err_combined(labt, labt_pred)

    def tca_err(data, params):
        (xs, xt, labs, labt) = data
        d = params[0]
        print("Running Transfer Component Analysis with {}".format(d))
        if d > max(xs.shape[0], xt.shape[0]):
            return np.inf
        # Domain adaptation via transfer component analysis. IEEE TNN 2011
        Ns = xs.shape[0]
        Nt = xt.shape[0]
        L_ss = (1. / (Ns * Ns)) * np.full((Ns, Ns), 1)
        L_st = (-1. / (Ns * Nt)) * np.full((Ns, Nt), 1)
        L_ts = (-1. / (Nt * Ns)) * np.full((Nt, Ns), 1)
        L_tt = (1. / (Nt * Nt)) * np.full((Nt, Nt), 1)
        L_up = np.hstack((L_ss, L_st))
        L_down = np.hstack((L_ts, L_tt))
        L = np.vstack((L_up, L_down))
        X = np.vstack((xs, xt))
        K = np.dot(X, X.T)  # linear kernel
        H = (np.identity(Ns+Nt)-1./(Ns+Nt)*np.ones((Ns + Nt, 1)) *
             np.ones((Ns + Nt, 1)).T)
        inv = np.linalg.pinv(np.identity(Ns + Nt) + K.dot(L).dot(K))
        D, W = np.linalg.eigh(inv.dot(K).dot(H).dot(K))
        W = W[:, np.argsort(-D)[:d]]  # eigenvectors of d highest eigenvalues
        sourceAdapted = np.dot(K[:Ns, :], W)  # project source
        targetAdapted = np.dot(K[Ns:, :], W)  # project target
        labt_pred = classify_1nn(sourceAdapted, targetAdapted, labs)
        return class_err_combined(labt, labt_pred)

    def coral_err(data, params):
        (xs, xt, labs, labt) = data
        print("Running CORAL")
        # Return of Frustratingly Easy Domain Adaptation. AAAI 2016
        Cs = np.cov(xs, rowvar=False) + np.eye(xs.shape[1])
        Ct = np.cov(xt, rowvar=False) + np.eye(xt.shape[1])
        Cs = Cs + Cs.T
        Ct = Ct + Ct.T
        svdCs = np.linalg.svd(Cs)
        svdCt = np.linalg.svd(Ct)
        Ds = xs.dot((svdCs[0] * (svdCs[1]**(-1/2)).reshape(1, -1)).dot(svdCs[2]))
        D2 = (svdCt[0] * (svdCt[1]**(-1/2)).reshape(1, -1)).dot(svdCt[2])
        Ds = Ds.dot(D2)  # re-coloring with target covariance
        sourceAdapted = Ds
        targetAdapted = xt
        labt_pred = classify_1nn(sourceAdapted, targetAdapted, labs)
        return class_err_combined(labt, labt_pred)

    estimator_functions = {
            "ot": ot_err,
            "ot_entr": ot_entr_err,
            "ot_kmeans": kmeans_ot_err,
            "ot_2kbary": hot_err,
            "ot_kbary": kbary_err,
            "ot_gl": gl_ot_err,
            "noadj": no_adjust_err,
            "sa": subspace_align_err,
            "tca": tca_err,
            "coral": coral_err
            }


    estimator_outlen = {
            "ot_2kbary": 3,
            "ot_kbary": 3,
            "ot_entr": 2
            }

    # Training
    est_train_results = {}
    est_test_results = {}
    skip_entr_reg = {}

    # Prepare arrays
    for (est_name, est_params) in estimators.items():
        parameters = est_params["parameter_ranges"]
        param_lens = list(map(len, parameters))
        outlen = estimator_outlen.get(est_params["function"], None)
        if outlen is not None:
            est_train_results[est_name] = np.empty((*param_lens, outlen, samples_train))
            est_test_results[est_name] = np.empty((outlen, samples_test))
        else:
            est_train_results[est_name] = np.empty((*param_lens, samples_train))
            est_test_results[est_name] = np.empty(samples_test)
        skip_entr_reg[est_name] = []

    # Run training
    for sample in range(samples_train):
        t0 = time.time()
        print("Training sample: {}/{}".format(sample+1, samples_train))
        (xs, xt, labs, labt) = get_data(True, sample)
        for (est_name, est_params) in estimators.items():
            parameters = est_params["parameter_ranges"]
            param_lens = list(map(len, parameters))
            if len(param_lens) == 0:
                continue
            est_fun = estimator_functions[est_params["function"]]
            for comb_ind in itertools.product(*map(range, param_lens)):
                cur_params = [parameters[i][comb_ind[i]] for i in range(len(parameters))]
                if len(cur_params) > 0 and cur_params[0] in skip_entr_reg[est_name]:
                    continue
                err = est_fun((xs, xt, labs, labt), cur_params)
                outlen = estimator_outlen.get(est_params["function"], None)
                if err == np.inf:
                    print("Skipping entropy parameter: {}".format(cur_params[0]))
                    skip_entr_reg[est_name].append(cur_params[0])
                    est_train_results[est_name][comb_ind[0],...] = np.inf
                else:
                    if outlen is not None:
                        est_train_results[est_name][(*comb_ind, slice(None), sample)] = err
                    else:
                        est_train_results[est_name][(*comb_ind, sample)] = err
        print("Time: {:.2f}s".format(time.time()-t0))

    # Pick best parameters
    est_train_err = {}
    opt_params = {}
    for (est_name, est_params) in estimators.items():
        parameters = est_params["parameter_ranges"]
        param_lens = list(map(len, parameters))
        if len(param_lens) == 0:
            opt_params[est_name] = []
            continue
        outlen = estimator_outlen.get(est_params["function"], None)
        if outlen is not None:
            est_train_err[est_name] = np.mean(est_train_results[est_name], axis = -1)
            opt_params[est_name] = []
            for j in range(outlen):
                err_mat = est_train_err[est_name][...,j]
                opt_ind = np.unravel_index(np.argmin(err_mat), err_mat.shape)
                opt_params[est_name].append([est_params["parameter_ranges"][i][opt_ind[i]] for i in range(len(opt_ind))])
        else:
            est_train_err[est_name] = np.mean(est_train_results[est_name], axis = -1)
            opt_ind = np.unravel_index(np.argmin(est_train_err[est_name]), est_train_err[est_name].shape)
            opt_params[est_name] = [est_params["parameter_ranges"][i][opt_ind[i]] for i in range(len(opt_ind))]

    # Run tests
    for sample in range(samples_test):
        t0 = time.time()
        print("Testing sample: {}/{}".format(sample+1, samples_test))
        (xs, xt, labs, labt) = get_data(False, sample)
        for (est_name, est_params) in estimators.items():
            outlen = estimator_outlen.get(est_params["function"], None)
            est_fun = estimator_functions[est_params["function"]]
            cur_params = opt_params[est_name]
            if outlen is not None:
                for j in range(outlen):
                    err = est_fun((xs, xt, labs, labt), cur_params[j])
                    if err != np.inf:
                        err = err[j]
                    est_test_results[est_name][j, sample] = err
            else:
                err = est_fun((xs, xt, labs, labt), cur_params)
                est_test_results[est_name][sample] = err
        print("Time for sample: {:.2f}".format(time.time() - t0))

    # # Change to testing
    # for var in opt_params.values():
    #     var[-2] = samples_test
    #     var[-1] = False

    # error_dict = {}
    # if "ot" in estimators:
    #     error_dict["ot"] = ot_err(opt_params["ot"])
    # if "kmeans_ot" in estimators:
    #     error_dict["kmeans_ot"] = kmeans_ot_err(opt_params["kmeans_ot"])
    # if "hot" in estimators:
    #     error_dict["hot"] = hot_err(opt_params["hot"])
    # if "kbary" in estimators:
    #     error_dict["kbary"] = kbary_err(opt_params["kbary"])
    # if "gl_ot" in estimators:
    #     error_dict["gl_ot"] = gl_ot_err(opt_params["gl_ot"])
    # print(error_dict)

    with open(sim_params["outfile"], 'wb') as outfile:
        pickle.dump({
            "params": sim_params,
            "train": est_train_results,
            "test": est_test_results,
            "opt_params": opt_params
            } , outfile)

def test_opt_grid():
    global ret, a, b
    a = np.array([7, 9, 11, -10])
    b = np.array([11, 2, 3, -10, -20])
    def fun(x):
        return x[0] + x[1]
    ret = opt_grid(fun, a, b)

#     # hubOT plot
#     pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
#     pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
#     ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
#     ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
#     ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
#     pl.show()

#     # k bary plot
#     pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     pl.plot(zs[:, 0], zs[:, 1], '<c', label='Mid 1')
#     ot.plot.plot2D_samples_mat(xs, zs, gammas[0], c=[.5, .5, 1])
#     ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[.5, .5, .5])
#     pl.show()

def test_bio_data():
    global data_ind, labels, xs, xt, labs, labt

    perclass = {"source": 20, "target": 20}
    samples = {"train": 1, "test": 1}
    label_samples = [874, 262, 92, 57]
    outfile = "pancreas.bin"

    entr_regs = np.array([10.0])**range(-3, 4)
    # entr_regs = np.array([10.0])**range(-2, 0)
    # entr_regs = np.array([10.0])**range(-1, 0)
    gl_params = np.array([10.0])**range(-3, 5)
    # gl_params = np.array([10.0])**range(-3, 0)
    # gl_params = np.array([0.1])
    # ks = np.array([2])**range(1, 8)
    # ks = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    ks = np.array([5, 10, 15, 20, 25, 30])

    # entr_regs = np.array([10.0])
    # gl_params = np.array([10.0])**range(4, 5)
    # ks = np.array([2])**range(5, 6)

    estimators = {
            "ot_gl": {
                "function": "ot_gl",
                "parameter_ranges": [entr_regs, gl_params]
                },
            # "ot": {
            #     "function": "ot",
            #     "parameter_ranges": []
            #     },
            "ot_entr": {
                "function": "ot_entr",
                "parameter_ranges": [entr_regs]
                },
            # "ot_kmeans": {
            #     "function": "ot_kmeans",
            #     "parameter_ranges": [entr_regs, ks]
            #     },
            "ot_2kbary": {
                "function": "ot_2kbary",
                "parameter_ranges": [entr_regs, ks]
                },
            "ot_kbary": {
                "function": "ot_kbary",
                "parameter_ranges": [entr_regs, ks]
            },
            "noadj": {
                "function": "noadj",
                "parameter_ranges": []
                },
            "sa": {
                "function": "sa",
                "parameter_ranges": [ks]
                },
            "tca": {
                "function": "tca",
                "parameter_ranges": [ks]
                },
            "coral": {
                "function": "coral",
                "parameter_ranges": []
                }
            }

    domain_data = {}

    print("-"*30)
    print("Running tests for bio data")
    print("-"*30)

    # data = loadmat(os.path.join(".", "MNN_haem_data.mat"))
    data = loadmat(os.path.join(".", "pancreas.mat"))
    xs = data['x2'].astype(float)
    xt = data['x3'].astype(float)
    labs = data['lab2'].ravel().astype(int)
    labt = data['lab3'].ravel().astype(int)

    # Prepare data splits
    data_ind = {"train": {}, "test": {}}
    features = {"source": xs,
            "target": xt}
    labels = {"source": labs,
            "target": labt}
    labels_unique = {k: np.unique(v) for (k, v) in labels.items()}

    for data_type in ["train", "test"]:
        for dataset in ["source", "target"]:
            data_ind[data_type][dataset] = []
            for sample in range(samples[data_type]):
                ind_list = []
                data_ind[data_type][dataset].append(ind_list)
                lab = labels[dataset]
                for c in sorted(labels_unique[dataset]):
                    ind = np.argwhere(lab == c).ravel()
                    np.random.shuffle(ind)
                    # ind_list.extend(ind[:min(perclass[dataset], len(ind)])
                    ind_list.extend(ind[:label_samples[int(c)]])

    def get_data(train, sample):
        trainstr = "train" if train else "test"
        xs = features["source"][data_ind[trainstr]["source"][sample], :]
        xt = features["target"][data_ind[trainstr]["target"][sample], :]
        labs = labels["source"][data_ind[trainstr]["source"][sample]]
        labt = labels["target"][data_ind[trainstr]["target"][sample]]
        # labs_ind =  calc_lab_ind(labs)
        # return (xs, xt, labs, labt, labs_ind)
        return (xs, xt, labs, labt)

    simulation_params = {
            "entr_regs": entr_regs,
            "gl_params": gl_params,
            "ks": ks,
            "samples_test": samples["test"],
            "samples_train": samples["train"],
            "outfile": outfile,
            "estimators": estimators
            }

    test_domain_adaptation(simulation_params, get_data)

def test_bio_diag():
    global gammas, gamma_total

    data = loadmat(os.path.join(".", "MNN_haem_data.mat"))
    xs = data['xs'].astype(float)
    xt = data['xt'].astype(float)
    labs = data['labs'].ravel().astype(float)
    labt = data['labt'].ravel().astype(float)

    data_ind = {"train": {}, "test": {}}
    features = {"source": xs,
            "target": xt}
    labels = {"source": labs,
            "target": labt}
    labels_unique = {k: np.unique(v) for (k, v) in labels.items()}

    perclass = {"source": 50, "target": 50}
    samples = {"train": 1, "test": 1}

    for data_type in ["train", "test"]:
        for dataset in ["source", "target"]:
            data_ind[data_type][dataset] = []
            for sample in range(samples[data_type]):
                ind_list = []
                data_ind[data_type][dataset].append(ind_list)
                lab = labels[dataset]
                for c in labels_unique[dataset]:
                    ind = np.argwhere(lab == c).ravel()
                    np.random.shuffle(ind)
                    ind_list.extend(ind[:min(perclass[dataset], len(ind))])

    def get_data(train, sample):
        trainstr = "train" if train else "test"
        xs = features["source"][data_ind[trainstr]["source"][sample], :]
        xt = features["target"][data_ind[trainstr]["target"][sample], :]
        labs = labels["source"][data_ind[trainstr]["source"][sample]]
        labt = labels["target"][data_ind[trainstr]["target"][sample]]
        # labs_ind =  calc_lab_ind(labs)
        # return (xs, xt, labs, labt, labs_ind)
        return (xs, xt, labs, labt)

    (xs, xt, labs, labt) = get_data(True, 0)

    k = 20
    entr_reg = 0.1
    a = np.ones(xs.shape[0])/xs.shape[0]
    b = np.ones(xt.shape[0])/xt.shape[0]
    gamma_ot = ot.sinkhorn(a, b, ot.dist(xs, xt), entr_reg)
    hot_ret = cluster_ot(a, b, xs, xt, k, k, [1.0, 1.0, 1.0], entr_reg,
            relax_outside = [np.inf, np.inf],
            warm_start = False,
            inner_tol = 1e-5,
            tol = 1e-4,
            reduced_inner_tol = True,
            inner_tol_start = 1e0,
            max_iter = 300,
            verbose = False,
            entr_reg_start = 10000.0
            )
    (zs1, zs2, a1, a2, gammas) = hot_ret
    gamma_total = total_gamma(gammas)
    expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
    labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas), xs), labs)
    labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas), labs)
    print(expected_err)
    print(class_err_combined(labt, labt_pred_hot))
    print(class_err_combined(labt, labt_pred_hot2))

    savemat("bio_diag.mat", {
        "xs": xs,
        "xt": xt,
        "labs": labs,
        "labt": labt,
        "zs1": zs1,
        "zs2": zs2,
        "gammas": gammas,
        "gamma_ot": gamma_ot,
        "gamma_total": gamma_total
        })

def test_caltech_office():
    global domain_data, data_ind, labels

    # Adjust
    tasks = [{
        "source": "amazon", 
        "target": "caltech10",
        "features": "GoogleNet1024",
        "perclass": {"source": 50, "target": 50},
        "samples": {"train": 10, "test": 10}
        }]
    features_todo = ["GoogleNet1024"]
    outfiles = ["caltech_google_ama_to_cal.bin"]

    # Do all pairs, all features
    domain_names = ["amazon", "caltech10", "dslr", "webcam"]
    feature_names = ["GoogleNet1024", "CaffeNet4096", "surf"]

    tasks = []
    for source_name in domain_names:
        for target_name in domain_names:
            if source_name != target_name:
                for feature_name in feature_names:
                    tasks.append({
                        "source": source_name,
                        "target": target_name,
                        "features": feature_name,
                        "perclass": {"source": 40 if source_name != "dslr" else 8, "target": 40 if source_name != "dslr" else 8},
                        "samples": {"train": 10, "test": 10},
                        "outfile": "caltech_" + source_name + "_to_" + target_name + "_" + feature_name + ".bin"
                        })

    # entr_regs = np.array([10.0])**range(-3, 5)
    entr_regs = np.array([10.0])**range(-3, 4)
    gl_params = np.array([10.0])**range(-3, 5)
    # ks = np.array([2])**range(1, 8)
    ks = np.array([5, 10, 15, 20, 30, 40, 50, 60, 70, 80])

    # entr_regs = np.array([10.0])
    # gl_params = np.array([10.0])**range(4, 5)
    # ks = np.array([2])**range(5, 6)

    estimators = {
            "ot_gl": {
                "function": "ot_gl",
                "parameter_ranges": [entr_regs, gl_params]
                },
            "ot": {
                "function": "ot",
                "parameter_ranges": []
                },
            "ot_entr": {
                "function": "ot_entr",
                "parameter_ranges": [entr_regs]
                },
            "ot_kmeans": {
                "function": "ot_kmeans",
                "parameter_ranges": [entr_regs, ks]
                },
            "ot_2kbary": {
                "function": "ot_2kbary",
                "parameter_ranges": [entr_regs, ks]
                },
            "ot_kbary": {
                "function": "ot_kbary",
                "parameter_ranges": [entr_regs, ks]
            },
            "noadj": {
                "function": "noadj",
                "parameter_ranges": []
                },
            "sa": {
                "function": "sa",
                "parameter_ranges": [ks]
                },
            "tca": {
                "function": "tca",
                "parameter_ranges": [ks]
                },
            # "coral": {
            #     "function": "coral",
            #     "parameter_ranges": []
            #     }
            }

    domain_data = {}

    for task in tasks:
        source_name = task["source"]
        target_name = task["target"]
        features_name = task["features"]
        perclass = task["perclass"]
        samples = task["samples"]
        outfile = task["outfile"]

        print("-"*30)
        print("Running {} to {} using {}".format(source_name, target_name, features_name))
        print("-"*30)

        for name in domain_names:
            data = loadmat(os.path.join(".", "features", features_name, name + ".mat"))
            features = data['fts'].astype(float)
            if features_name == "surf":
                features = features / np.sum(features, 1).reshape(-1, 1)
            features = preprocessing.scale(features)
            labels = data['labels'].ravel()
            domain_data[name] = {"features": features, "labels": labels}

        # Prepare data splits
        data_ind = {"train": {}, "test": {}}
        features = {"source": domain_data[source_name]["features"],
                "target": domain_data[target_name]["features"]}
        labels = {"source": domain_data[source_name]["labels"],
                "target": domain_data[target_name]["labels"]}
        labels_unique = {k: np.unique(v) for (k, v) in labels.items()}

        for data_type in ["train", "test"]:
            for dataset in ["source", "target"]:
                data_ind[data_type][dataset] = []
                for sample in range(samples[data_type]):
                    ind_list = []
                    data_ind[data_type][dataset].append(ind_list)
                    lab = labels[dataset]
                    for c in labels_unique[dataset]:
                        ind = np.argwhere(lab == c).ravel()
                        np.random.shuffle(ind)
                        ind_list.extend(ind[:min(perclass[dataset], len(ind))])

        def get_data(train, sample):
            trainstr = "train" if train else "test"
            xs = features["source"][data_ind[trainstr]["source"][sample], :]
            xt = features["target"][data_ind[trainstr]["target"][sample], :]
            labs = labels["source"][data_ind[trainstr]["source"][sample]]
            labt = labels["target"][data_ind[trainstr]["target"][sample]]
            # labs_ind =  calc_lab_ind(labs)
            # return (xs, xt, labs, labt, labs_ind)
            return (xs, xt, labs, labt)

        simulation_params = {
                "entr_regs": entr_regs,
                "gl_params": gl_params,
                "ks": ks,
                "samples_test": samples["test"],
                "samples_train": samples["train"],
                "outfile": outfile,
                "estimators": estimators
                }

        try:
            test_domain_adaptation(simulation_params, get_data)
        except KeyboardInterrupt:
            print("Canceled")
            break
        except:
            print("An error occurred, contuinuing with next data set!")


if __name__ == "__main__":
    # test_gaussian_mixture()
    # test_split_data_uniform_vard()
    # test_split_data_uniform_vark()
    # test_split_data_uniform_varn()
    # test_split_data_uniform_visual()
    # test_constraint_ot()
    # t0 = time.time()
    # test_moons()
    # print("Time for moons: {}".format(time.time() - t0))
    # test_opt_grid()
    # test_moons_kplot()
    # test_satija()
    # test_caltech_office()
    test_bio_data()
    # test_bio_diag()

#     ### Barycenter histogram test

#     # lambd = 1e-2
#     # # n = 20
#     # # xs = np.linspace(0, 10, n)
#     # # xs = xs.reshape((xs.shape[0], 1))
#     # # ys = xs.copy()
#     # xm = np.vstack((xm, np.array([0, -10])))
#     # M1 = ot.dist(xs, xm)
#     # M1 /= M1.max()
#     # M2 = ot.dist(xt, xm)
#     # M2 /= M2.max()
#     # n1 = xs.shape[0]
#     # n2 = xt.shape[0]
#     # b1 = np.ones(n1)/n1
#     # b2 = np.ones(n2)/n2
#     # (gamma1, gamma2) = barycenter_bregman(b1, b2, M1.T, M2.T, lambd)
#     # # print(ot.sinkhorn2(a, b, M, lambd))
#     # x_combined = np.vstack((xs, xt, xm))
#     # b_combined = np.zeros((x_combined.shape[0], 2))
#     # b_combined[0:b1.shape[0], 0] = b1
#     # b_combined[b1.shape[0]:(b1.shape[0]+b2.shape[0]), 1] = b2
#     # b_combined += 1e-5
#     # b_combined[:,0] / np.sum(b_combined[:,0])
#     # b_combined[:,1] / np.sum(b_combined[:,1])
#     # M = ot.dist(x_combined, x_combined)
#     # a_ot = ot.bregman.barycenter(b_combined, M, lambd)

#     # ### Barycenter fixed point test

#     # lambd = 0.1
#     # cov1 = np.array([[1, 0], [0, 1]])
#     # cov2 = np.array([[1, 0], [0, 1]])
#     # mu1 = np.zeros(2)
#     # mu2 = np.zeros(2)
#     # n = 400
#     # k = 100
#     # xs1 = ot.datasets.get_2D_samples_gauss(n, mu1, cov1)
#     # xs2 = ot.datasets.get_2D_samples_gauss(n, mu2, cov2)
#     # xs1 /= np.dot(np.linalg.norm(xs1, axis = 1).reshape(n, 1), np.ones((1, 2)))
#     # xs2 /= np.dot(np.linalg.norm(xs2, axis = 1).reshape(n, 1), np.ones((1, 2)))
#     # xs1[:,0] *= 5
#     # xs2[:,1] *= 5
#     # b1 = np.ones(n)/n
#     # b2 = b1.copy()
#     # (ys, weights, cost) = barycenter_free(b1, b2, xs1, xs2, 0.1, 0.9, lambd, k)
#     # print("Cost = {}".format(cost))
#     # pl.plot(xs1[:,0], xs1[:,1], 'xb')
#     # pl.plot(xs2[:,0], xs2[:,1], 'xr')
#     # pl.plot(ys[:,0], ys[:,1], 'xg')
#     # pl.show()

#     #%% Test run

#     # #%% parameters and data generation
#     # n_source = 40 # nb samples
#     # n_target = 60 # nb samples

#     # mu_source = np.array([[0, 0], [0, 5]])
#     # cov_source = np.array([[1, 0], [0, 1]])

#     # mu_target = np.array([[10, 1], [10, 5], [10, 10]])
#     # cov_target = np.array([[1, 0], [0, 1]])

#     # num_clust_source = mu_source.shape[0]
#     # num_clust_target = mu_target.shape[0]

#     # xs = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_source/num_clust_source).astype(int),  mu_source[i,:], cov_source) for i in range(num_clust_source)])
#     # xt = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_target/num_clust_target).astype(int),  mu_target[i,:], cov_target) for i in range(num_clust_target)])

#     # ind_clust_source = np.vstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
#     # ind_clust_target = np.vstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

#     #%% Unbalanced samples
#     n_source = 100
#     n_target = 100

#     mu_source = np.array([[-5, -5], [-5, 5]])
#     cov_source = np.array([[1, 0], [0, 1]])

#     mu_target = np.array([[5, -5], [5, 5]])
#     cov_target = np.array([[1, 0], [0, 1]])

#     num_clust_source = mu_source.shape[0]
#     num_clust_target = mu_target.shape[0]

#     weights = [1.0/2, 1.0/2]

#     xs = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(weights[i] * n_source/num_clust_source).astype(int),  mu_source[i,:], cov_source) for i in range(num_clust_source)])
#     xt = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(weights[1-i] * n_target/num_clust_target).astype(int),  mu_target[i,:], cov_target) for i in range(num_clust_target)])

#     ind_clust_source = np.vstack([i*np.ones(int(n_source/num_clust_source)) for i in range(num_clust_source)])
#     ind_clust_target = np.vstack([i*np.ones(int(n_target/num_clust_target)) for i in range(num_clust_target)])

#     #%% individual OT

#     # uniform distribution on samples
#     n1 = len(xs)
#     n2 = len(xt)
#     a, b = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

#     # loss matrix
#     # M /= M.max()

#     #OT
#     #G_ind = ot.emd(a, b, M)
#     lambd = 1
#     M = ot.dist(xs, xt)
#     t0 = time.time()
#     G_ind = ot.sinkhorn(a, b, M, lambd)
#     vanilla_cost = ot.sinkhorn2(a, b, M, lambd)
#     print("Time: {}".format(time.time() - t0))

#     ## plot OT transformation for samples
#     #ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
#     #pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     #pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     #pl.legend(loc=0)
#     #pl.title('OT matrix with samples')

#     # #%% clustered OT

#     # #find CM of clusters
#     # kmeans = KMeans(n_clusters=num_clust_source, random_state=0).fit(xs)
#     # CM_source = kmeans.cluster_centers_
#     # kmeans = KMeans(n_clusters=num_clust_target, random_state=0).fit(xt)
#     # CM_target = kmeans.cluster_centers_

#     # # loss matrix
#     # M_clust = ot.dist(CM_source, CM_target)
#     # M_clust /= M_clust.max()

#     # # uniform distribution on CMs
#     # n1_clust = len(CM_source)
#     # n2_clust = len(CM_target)
#     # a_clust, b_clust = np.ones((n1_clust,)) / n1_clust, np.ones((n2_clust,)) / n2_clust # uniform distribution on samples

#     # #OT
#     # G_clust = ot.emd(a_clust, b_clust, M_clust)

#     ## plot OT transformation for CMs
#     #ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
#     #pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
#     #pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
#     #pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     #pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     #pl.legend(loc=0)
#     #pl.title('OT matrix with samples, CM')

#     # #%% OT figures
#     # f, axs = pl.subplots(1,2,figsize=(10,4))

#     # sub=pl.subplot(131)
#     # ot.plot.plot2D_samples_mat(xs, xt, G_ind, c=[.5, .5, 1])
#     # pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     # pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     # pl.legend(loc=0)
#     # sub.set_title('OT by samples')

#     # sub=pl.subplot(132)
#     # ot.plot.plot2D_samples_mat(CM_source, CM_target, G_clust, c=[.5, .5, 1])
#     # pl.plot(CM_source[:, 0], CM_source[:, 1], 'ok', markersize=10,fillstyle='full')
#     # pl.plot(CM_target[:, 0], CM_target[:, 1], 'ok', markersize=10,fillstyle='full')
#     # pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     # pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     # pl.legend(loc=0)
#     # sub.set_title('OT by CMs')

#     # uniform distribution on samples
#     n1 = len(xs)
#     n2 = len(xt)
#     b1, b2 = np.ones((n1,)) / n1, np.ones((n2,)) / n2 

#     # t0 = time.time()
#     # (zs1, zs2, a1, a2, mid_left_source, mid_right_mid_left, mid_right_target) = cluster_ot(b1, b2, xs, xt, 2, 3, 1.0, 1.0, 1, verbose = True)
#     # print("Time: {}".format(time.time() - t0))
#     # t0 = time.time()
#     # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, xs, xt, 2, 2, [1.0, 1.0, 1.0], 1.0, verbose = True, relax_inside = [1e0, 1e0], max_iter = 30)
#     # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, xs, xt, 10, 10, [1.0, 1.0, 1.0], 1.0, verbose = True, relax_outside = [1e2, 1e2], max_iter = 30)
#     # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, xs, xt, 5, 5, [1.0, 1.0, 1.0], 1.0, verbose = True, max_iter = 30)
#     # print("Time: {}".format(time.time() - t0))

#     # cluster_cost = estimate_w2_cluster(xs, xt, gammas)
#     # print(vanilla_cost)
#     # print(cluster_cost)

#     # (zs1, zs2, gammas) = reweighted_clusters(xs, xt, 4, [1.0, 1.0, 1.0], 1e+0)

#     # t0 = time.time()
#     # (zs1, zs2, a, gammas) = cluster_ot_map(b1, b2, xs, xt, 3, [10, 1.0, 10.0], 1.0, verbose = True, tol = 1e-5)
#     # print("Time: {}".format(time.time() - t0))

#     # # sub = pl.subplot(133)
#     # pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
#     # pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
#     # pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
#     # pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
#     # # ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
#     # # ot.plot.plot2D_samples_mat(zs1, zs2, np.diag(np.sum(gammas[0], axis = 0)), c=[.5, .5, .5])
#     # # ot.plot.plot2D_samples_mat(zs2, xt, gammas[1], c=[1, .5, .5])
#     # ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
#     # ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
#     # ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
#     # pl.legend(loc=0)
#     # # sub.set_title("OT, regularized")

#     # #%% Test new barycenter

#     # #%% parameters and data generation
#     # n_source = 10000 # nb samples
#     # n_target = 30000 # nb samples

#     # mu_source = np.array([[0, 0], [0, 5]])
#     # cov_source = np.array([[1, 0], [0, 1]])

#     # mu_target = np.array([[10, 1], [10, 5], [10, 10]])
#     # cov_target = np.array([[1, 0], [0, 1]])

#     # num_clust_source = mu_source.shape[0]
#     # num_clust_target = mu_target.shape[0]

#     # xs = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_source/num_clust_source).astype(int),  mu_source[i,:], cov_source) for i in range(num_clust_source)])
#     # xt = np.vstack([ot.datasets.get_2D_samples_gauss(np.floor(n_target/num_clust_target).astype(int),  mu_target[i,:], cov_target) for i in range(num_clust_target)])

#     # ind_clust_source = np.vstack([i*np.ones(n_source/num_clust_source) for i in range(num_clust_source)])
#     # ind_clust_target = np.vstack([i*np.ones(n_target/num_clust_target) for i in range(num_clust_target)])

#     # zs_left = np.random.normal(size=(100, 2))
#     # zs_right = np.random.normal(size=(200, 2))

#     # M1 = ot.dist(xs, zs_left)
#     # M2 = ot.dist(zs_left, zs_right)
#     # M3 = ot.dist(zs_right, xt)

#     # # (gamma1, gamma2) = barycenter_bregman(np.ones(n_source)/n_source, np.ones(6)/6, M1.T, M2, 0.5, 0.5, 0.01, tol = 1e-6)
#     # # (gamma1_ws, gamma2_ws, gamma3_ws) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 0.01, tol = 1e-6, verbose = True, max_iter = 1000, warm_start = True)
#     # # t0 = time.time()
#     # # (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 1, tol = 1e-6, verbose = True, max_iter = 1000, warm_start = False)
#     # # print("Time: {}".format(time.time() - t0))
#     # t0 = time.time()
#     # (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 1, 1, 1, 1, tol = 1e-6, verbose = False, max_iter = 1000, warm_start = True, swap_order = True)
#     # print("Time: {}".format(time.time() - t0))
#     # t0 = time.time()
#     # gammas = barycenter_bregman_chain(np.ones(n_source)/n_source, np.ones(n_target)/n_target, [M1, M2, M3], [1, 1, 1], 1, tol=1e-6, verbose = False, max_iter = 1000, warm_start = True)
#     # print("Time: {}".format(time.time() - t0))
#     # # (gamma1, gamma2, gamma3) = barycenter_bregman2(np.ones(n_source)/n_source, np.ones(n_target)/n_target, M1, M2, M3, 0.2, 0.2, 0.2, 1000, tol = 1e-4, verbose = True, max_iter = 10)
