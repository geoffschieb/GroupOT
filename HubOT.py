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
import errno
import subprocess
from sklearn import preprocessing

#%% Utility functions

def opt_grid(fun, *args):
    results = np.empty(tuple(map(len, args)))
    for comb_ind in itertools.product(*map(lambda x: range(len(x)), args)):
        results[comb_ind] = fun([args[i][comb_ind[i]] for i in range(len(args))])
    multi_opt = np.unravel_index(np.argmin(results), results.shape)
    return [args[i][multi_opt[i]] for i in range(len(args))]

def mkdir(path):
    try:
        os.mkdir(path)
    except IOError as e:
        if e.errno != errno.EEXIST:
            raise
        pass

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
        max_iter = 1000,
        rebalance_thresh = 1e3,
        warm_start = False,
        entr_reg_start = 1000.0
        ):

    warnings.filterwarnings("error")
    try:
        if not hasattr(entr_reg_final, '__len__'):
            entr_reg_final = np.repeat(entr_reg_final, len(Ms))
        if not hasattr(entr_reg_start, '__len__'):
            entr_reg_start = np.repeat(entr_reg_start, len(Ms))

        entr_reg = entr_reg_start if warm_start else entr_reg_final
        np.seterr(all="warn")

        alphas = [[np.zeros(M.shape[i]) for i in range(2)] for M in Ms]

        Ks = [np.exp(-M/entr_reg[i]) for (i, M) in enumerate(Ms)]
        Kts = [K.T.copy() for K in Ks]

        us = [[np.ones(K.shape[i]) for i in range(2)] for K in Ks]
        us_old = copy.deepcopy(us)

        weights = [[float(lambdas[i+j])/(lambdas[i]+lambdas[i+1]) for j in range(2)] for i in range(len(lambdas)-1)]

        # print(lambdas)

        converged = False
        its = 0

        def rebalance(i):
            for j in range(2):
                alphas[i][j] += entr_reg[i] * np.log(us[i][j])
                us[i][j] = np.ones(Ks[i].shape[j])

        def update_K(i):
            Ks[i] = np.exp(-(Ms[i] - alphas[i][0].reshape(-1, 1) - alphas[i][1].reshape(1, -1))/entr_reg[i])
            Kts[i] = Ks[i].T.copy()

        while not converged and its < max_iter: 
            its += 1
            # if True:
            if verbose:
                print("Barycenter weights iteration: {}".format(its))

            for i in range(len(us)):
                # if i == 1:
                #     print("dual variable:")
                #     print(max([np.max(np.abs(us[i][j])) for j in range(2)]))
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
                if not warm_start or max(abs(entr_reg - entr_reg_final)) < 1e-12:
                    converged = True
                else:
                    for i in range(len(us)):
                        rebalance(i)

                    entr_reg = np.maximum(entr_reg/2, entr_reg_final)
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
        verbose = False,
        no_iteration = False
        ):

    converged = False
    its = 0

    weights = [(lambdas[0] * np.sum(gammas[0], axis = 0) + lambdas[1] * np.sum(gammas[1], axis = 1)),
                (lambdas[1] * np.sum(gammas[1], axis = 0) + lambdas[2] * np.sum(gammas[2], axis = 1))]
    point_avgs = [lambdas[0] * np.dot(xs.T, gammas[0]).T, lambdas[2] * np.dot(gammas[2], xt)]

    if no_iteration:
        (k, d) = zs1.shape
        A = np.zeros(shape = (2*k*d, 2*k*d))
        A[0:k*d, 0:k*d] = np.kron(np.diag(weights[0]), np.eye(d))
        A[0:k*d, k*d:] = -np.kron(np.diag(lambdas[1] * np.sum(gammas[1], axis = 1)), np.eye(d))
        A[k*d:, k*d:] = np.kron(np.diag(weights[1]), np.eye(d))
        A[k*d:, 0:k*d] = -np.kron(np.diag(lambdas[1] * np.sum(gammas[1], axis = 0)), np.eye(d))
        b = np.vstack([point_avgs[0].reshape(k*d, 1), point_avgs[1].reshape(k*d, 1)])
        zs_new = np.linalg.solve(A, b).reshape((d, 2*k), order = 'F')
        zs1 = zs_new[:,0:k].T
        zs2 = zs_new[:,k:].T

    else:
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

def plot_transport_map(xs, xt, **kwds):
    for i in range(xs.shape[0]):
        pl.plot([xs[i, 0], xt[i, 0]], [xs[i, 1], xt[i, 1]], 'k', alpha = 0.5, **kwds)

def plot_transport_map_arrows(xs, xt, **kwds):
    for i in range(xs.shape[0]):
        pl.arrow(xs[i, 0], xs[i, 1], xt[i, 0] - xs[i, 0], xt[i, 1] - xs[i, 1], color = 'k', alpha = 0.5, **kwds)

# @profile
def cluster_ot(b1, b2, xs, xt, k1, k2,
        lambdas, entr_reg,
        relax_outside = [np.float("Inf"), np.float("Inf")],
        relax_inside = [np.float("Inf"), np.float("Inf")],
        tol = 1e-4,
        inner_tol = 1e-5,
        max_iter = 1000,
        inner_max_iter = 2000,
        verbose = False,
        debug = False,
        warm_start = False,
        # iter_mult = 10,
        # iter_inc = 10
        reduced_inner_tol = False,
        inner_tol_fact = 0.1,
        inner_tol_start = 1e-3,
        inner_tol_dec_every = 20,
        entr_reg_start = 1000.0,
        zs1_start = None,
        zs2_start = None,
        gammas_start = None,
        plot_iterations = False
        ):

    converged = False
    np.seterr(all="warn")

    d = xs.shape[1]
    # zs1  = 1 * np.random.normal(size = (k1, d))
    # zs2  = 1 * np.random.normal(size = (k2, d))
    if zs1_start is None:
        zs1 = KMeans(n_clusters = k1).fit(xs).cluster_centers_
    else:
        zs1 = zs1_start
    if zs2_start is None:
        zs2 = KMeans(n_clusters = k2).fit(xt).cluster_centers_
    else:
        zs2 = zs2_start
    gammas = gammas_start if gammas_start is not None else None


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
        if its > 1 or gammas is not None:
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

        # hubOT plot
        if plot_iterations:
            fig = pl.figure(figsize = (6,4))
            pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
            pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
            pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
            pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
            ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
            ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
            ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
            # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
            mkdir(os.path.join("Figures","iterations"))
            pl.savefig(os.path.join("Figures","iterations","{}.png".format(its)), dpi = 300)
            pl.close(fig)

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
        # return (zs1, zs2, np.sum(gammas[1], axis = 1), np.sum(gammas[1], axis = 0), gammas)
        return (zs1, zs2, gammas)

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

def alternating_min(a, b, xs, xt, k,
        entr_reg = 1.0,
        tol = 1e-8,
        max_iters = 10000):
    converged = False

    its = 0
    n = xs.shape[0]
    A = np.zeros((n, k))
    B = np.zeros((k, n))
    A[:k,:k] = np.eye(k)
    B[:k,:k] = np.eye(k)
    C = ot.dist(xs, xt)

    while not converged and its < max_iters:
        print("Iteration: {}".format(its))
        its += 1
        A_old = copy.copy(A)
        B_old = copy.copy(B)
        M = np.exp(-np.dot(A.T, C)/entr_reg)
        # print(np.linalg.lstsq(A, a)[0].shape)
        # print(np.sum(M, axis = 1).shape)
        u = np.linalg.lstsq(A, a)[0]/np.sum(M, axis = 1)
        u[u < 0] = 0.0 
        B = u.reshape(-1, 1) * M

        M = np.exp(-np.dot(C, B.T)/entr_reg)
        v = np.linalg.lstsq(B.T, b)[0]/np.sum(M, axis = 0)
        v[v < 0] = 0.0
        A = v.reshape(1, -1) * M

        err = (np.linalg.norm(A - A_old) + np.linalg.norm(B - B_old))
        print("Change: {}".format(err))

        if err < tol:
            converged = True

    return (A, B)

def test_alternating_min():
    global A, B
    xs = np.random.randn(20,2)
    xt = np.random.randn(20,2)
    a = np.ones(20)/20
    b = np.ones(20)/20

    (A, B) = alternating_min(a, b, xs, xt, 5)

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
        entr_reg_start = 1000.0,
        debug = False
        ):

    converged = False

    d = xs.shape[1]
    # zs = np.random.normal(size = (k, d))
    source_means = KMeans(n_clusters = k).fit(xs).cluster_centers_
    # target_means = KMeans(n_clusters = k).fit(xt).cluster_centers_
    # match = ot.emd(np.ones(k)/k, np.ones(k)/k, ot.dist(source_means, target_means))
    # optmap = np.argmax(match, axis = 1)
    # zs = 0.5 * source_means + 0.5 * target_means[optmap]
    # combined_means = KMeans(n_clusters = k).fit(np.vstack([xs, xt])).cluster_centers_
    zs = source_means
    # zs = combined_means

    if debug:
        zs_l = [zs]

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
            if debug:
                zs_l.append(zs)

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

    if not debug:
        return (zs, gammas)
    else:
        return (zs, gammas, zs_l)

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
        verbose = True,
        warm_start = True
        ):
    
    converged = False

    d = xs.shape[1]
    # zs1 = np.random.normal(size = (k, d))
    # zs2 = np.random.normal(size = (k, d))
    zs1 = KMeans(n_clusters = k).fit(xs).cluster_centers_
    zs2 = KMeans(n_clusters = k).fit(xt).cluster_centers_
    assignment = ot.emd(np.ones(k), np.ones(k), ot.dist(zs1, zs2))
    zs2 = zs2[np.nonzero(assignment > 0.5)[1]]
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
        gammas = barycenter_bregman_chain(b1, b2, Ms, lambdas_barycenter, entr_reg, verbose = False, warm_start = warm_start)
        if gammas is None:
            return None
        cost = np.sum([lambdas_barycenter[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * stats.entropy(gammas[i].reshape(-1))) for i in range(2)])

        err = np.abs(cost - cost_old)/max(np.abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs1, zs2, gammas)

def nearest_points(M):
    gamma = np.zeros_like(M)
    min_ind = np.argmin(M, axis = 1)
    gamma[(range(gamma.shape[0]), min_ind)] = 1.0/gamma.shape[0]

    return gamma

def nearest_points_reg(a, M, entr_reg):
    K = np.exp(-M/entr_reg)
    return np.divide(a, np.sum(K, axis = 1)).reshape(-1, 1) * K

def reweighted_clusters(b1, b2, xs, xt, k, lambdas, entr_reg,
        tol = 1e-10,
        lb = 0,
        equal_weights = False,
        max_iter = 1000
        ):

    converged = False

    d = xs.shape[1]
    # zs1 = np.random.normal(size = (k, d))
    # zs2 = np.random.normal(size = (k, d))
    zs1 = KMeans(n_clusters = k).fit(xs).cluster_centers_
    zs2 = KMeans(n_clusters = k).fit(xt).cluster_centers_
    its = 0
    unif_middle = np.ones(k)/k
    # unif_left = np.ones(xs.shape[0])/xs.shape[0]
    # unif_right = np.ones(xt.shape[0])/xt.shape[0]
    cost_old = np.float("Inf")

    try:
        warnings.filterwarnings("error")
        while not converged and its < max_iter:
            its += 1
            if its > 1:
                (zs1, zs2) = update_points(xs, xt, zs1, zs2, gammas, lambdas, verbose = False, no_iteration = False)
            Ms = [ot.dist(xs, zs1), ot.dist(zs1, zs2), ot.dist(zs2, xt)]
            # gamma_left = nearest_points(Ms[0])
            # gamma_right = nearest_points(Ms[2].T).T
            if equal_weights:
                gamma_left = ot.sinkhorn(b1, unif_middle, Ms[0], entr_reg)
                gamma_right = ot.sinkhorn(unif_middle, b2, Ms[2], entr_reg)
            elif lb == 0:
                gamma_left = nearest_points_reg(b1, Ms[0], entr_reg)
                gamma_right = nearest_points_reg(b2, Ms[2].T, entr_reg).T
            else:
                gamma_left = constraint_ot(b1, lb, Ms[0], entr_reg)
                gamma_right = constraint_ot(b2, lb, Ms[2].T, entr_reg).T
            # gamma_middle = ot.sinkhorn(unif, unif, Ms[1], entr_reg)
            gamma_middle = ot.emd(unif_middle, unif_middle, Ms[1])
            gammas = [gamma_left, gamma_middle, gamma_right]

            # cost = np.sum([lambdas[i] * np.sum(gammas[i] * Ms[i]) for i in range(3)]) + lambdas[1] * entr_reg * stats.entropy(gammas[1].reshape(-1,1))
            cost = np.sum([lambdas[i] * (np.sum(gammas[i] * Ms[i]) + entr_reg * entr_vec(gammas[i])) for i in range(3)])
            err = abs(cost - cost_old)/max(abs(cost), 1e-12)

            if err < tol:
                converged = True

            cost_old = cost

        return (zs1, zs2, gammas)
    except RuntimeWarning:
        return None
    finally:
        warnings.filterwarnings("default")

def kmeans_transport(xs, xt, k):
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
    gammas = [gamma_source, gamma_clust, gamma_target.T]
    return (zs1, zs2, [gamma_source, gamma_clust, gamma_target.T])

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

# def classify_knn(xs, xt, labs, k):
#     return KNeighborsClassifier(n_neighbors = k).fit(xs, labs).predict(xt)

def classify_knn(xs, xt, labs, k):
    return KNeighborsClassifier(n_neighbors = k, algorithm='brute').fit(xs, labs).predict(xt)

def classify_knn(xs, xt, labs, k):
    M = ot.dist(xt, xs)
    inds = np.argsort(M, axis = 1)[:,1:k]
    labs[inds]
    return 

#%% Estimation/rounding functions

def estimate_w2_cluster(xs, xt, gammas, b0 = None):
    if b0 is None:
        b0 = np.ones(xs.shape[0])/xs.shape[0]

    gammas_thresh = copy.deepcopy(gammas)
    for i in range(len(gammas_thresh)):
        gammas_thresh[i][np.abs(gammas[i]) < 1e-10] = 0

    (centers1, centers2) = find_centers(xs, xt, gammas)

    a1 = np.sum(gammas[0], axis = 0)
    a2 = np.sum(gammas[-1], axis = 1)
    # print(a1)
    # print(a2)
    total_cost = ot.emd2(a1, a2, ot.dist(centers1, centers2))

    # if len(gammas) == 3:
    #     costs = np.sum(gammas[1] * ot.dist(centers1, centers2), axis = 1)/np.sum(gammas[1], axis = 1)
    # elif len(gammas) == 2:
    #     costs = np.sum((centers1 - centers2)**2, axis = 1)
    # total_cost = np.sum((gammas[0] / np.sum(gammas[0], axis = 1).reshape(-1, 1) * b0.reshape(-1, 1)) * costs)
    # total_cost = np.sum((gammas[0] * costs))

    return total_cost

def estimate_w2_middle(xs, xt, zs1, zs2, gammas, bias_correction = False, b0 = None):
    if b0 is None:
        b0 = np.ones(xs.shape[0])/xs.shape[0]

    gammas_thresh = copy.deepcopy(gammas)
    for i in range(len(gammas_thresh)):
        gammas_thresh[i][np.abs(gammas[i]) < 1e-10] = 0

    n, d = xs.shape
    k1, d = zs1.shape
    k2, d = zs2.shape

    if bias_correction:
        dists1 = xs.reshape(n, d, 1) - zs1.T.reshape(1, d, k1)
        weights1 = gammas[0] / np.sum(gammas[0], axis = 0).reshape(1, -1)
        avg1 = np.einsum("ijk,ik->kj", dists1, weights1)
        dists2 = xt.reshape(n, d, 1) - zs2.T.reshape(1, d, k2)
        weights2 = gammas[2].T / np.sum(gammas[2], axis = 1).reshape(1, -1)
        avg2 = np.einsum("ijk,ik->kj", dists2, weights2)
        centers1 = zs1 + avg1
        # pl.plot(zs1[0,0], zs1[0,1], 'xr')
        # pl.plot(avg1[:,0], avg1[:,1], 'og')
        # pl.plot(xs[:,0], xs[:,1], 'xk')
        # pl.plot(dists1[:,0,0], dists1[:,1,0], '+b')
        # pl.show()
        # print(zs1)
        # print(centers1)
        centers2 = zs2 + avg2
        M = ot.dist(centers1, centers2)
        total_cost = np.sum(gammas[1] * M)

#         a1 = np.sum(gammas[0], axis = 0)
#         a2 = np.sum(gammas[-1], axis = 1)
#         # print(a1)
#         # print(a2)
#         total_cost = ot.emd2(a1, a2, M)
    else:
        M = ot.dist(zs1, zs2)
        total_cost = np.sum(gammas[1] * M)

        # a1 = np.sum(gammas[0], axis = 0)
        # a2 = np.sum(gammas[-1], axis = 1)
        # # print(a1)
        # # print(a2)
        # total_cost = ot.emd2(a1, a2, M)

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
    k = 4
    # prp = np.random.uniform(size = n_cluster)
    # prp /= np.sum(prp)
    # prp = np.array([0.1, 0.2, 0.3, 0.4])
    prp = np.array([0.25, 0.25, 0.25, 0.25])
    d = 30
    ns = 100
    nt = 100
    spread = 3.0
    variance = 1.0
    entr_reg = 10.0

    np.random.seed(42)

    # # psd matrix
    # A = np.random.normal(0, 1, (d, d))
    # A = A.dot(A.T)
    # print(A)
    # def trafo(x):
    #     return A.dot(x)

    # Random direction
    direction = np.random.normal(0, 70/d, d)
    def trafo(x):
        return x + direction

    # # Scaling
    # scale = 0.1
    # def trafo(x):
    #     return scale * x

    cluster_centers = np.random.normal(0, spread, (n_cluster, d))
    samples_target = []
    samples_source = []
    cluster_ids = list(range(n_cluster))
    c = np.random.choice(cluster_ids, p = prp, size = ns)
    xs = cluster_centers[c,:] + np.random.normal(0, variance, size = (ns, d))
    c = np.random.choice(cluster_ids, p = prp, size = nt)
    xt = cluster_centers[c,:] + np.random.normal(0, variance, size = (nt, d))
    xt = np.apply_along_axis(trafo, 1, xt)
    # for i in range(ns):
    #     c = np.random.choice(cluster_ids, p = prp)
    #     samples_source.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))
    # for i in range(nt):
    #     c = np.random.choice(range(n_cluster), p = prp)
    #     samples_target.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))

    # xs = np.vstack([samples for samples in samples_target])
    # xt = np.vstack([samples for samples in samples_source])
    # xt = np.apply_along_axis(trafo, 1, xt)

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

def test_gaussian_mixture_varn():
    ns = (np.array([10.0])**np.linspace(1.0, 2.7, 20)).astype(int)
    # ns = (np.array([10.0])**np.linspace(2.5, 2.7, 20)).astype(int)
    samples = 20

    n_cluster = 4
    k = 4
    # prp = np.random.uniform(size = n_cluster)
    # prp /= np.sum(prp)
    # prp = np.array([0.1, 0.2, 0.3, 0.4])
    prp = np.array([0.25, 0.25, 0.25, 0.25])
    d = 30
    spread = 3.0
    variance = 1.0
    entr_reg = 10.0

    np.random.seed(42)

    # # psd matrix
    # A = np.random.normal(0, 1, (d, d))
    # A = A.dot(A.T)
    # print(A)
    # def trafo(x):
    #     return A.dot(x)

    samples_target = []
    samples_source = []
    cluster_ids = list(range(n_cluster))

    ground_truth = np.sum(prp.reshape(-1,1) * (np.apply_along_axis(trafo, 1, cluster_centers) - cluster_centers)**2)
    print("Ground truth: {}".format(ground_truth))

    results_vanilla = np.empty((samples, len(ns)))
    results_kbary = np.empty((samples, len(ns)))
    results_kmeans = np.empty((samples, len(ns)))

    for sample in range(samples):
        print("Sample: {}".format(sample))

        print("n_ind = {}, n = {}".format(n_ind, n))
        # Random direction
        direction = np.random.normal(0, 70/d, d)
        def trafo(x):
            return x + direction

        cluster_centers = np.random.normal(0, spread, (n_cluster, d))
        # # Scaling
        # scale = 0.1
        # def trafo(x):
        #     return scale * x

        for (n_ind, n) in enumerate(ns):
            c = np.random.choice(cluster_ids, p = prp, size = n)
            xs = cluster_centers[c,:] + np.random.normal(0, variance, size = (n, d))
            c = np.random.choice(cluster_ids, p = prp, size = n)
            xt = cluster_centers[c,:] + np.random.normal(0, variance, size = (n, d))
            xt = np.apply_along_axis(trafo, 1, xt)

            b1 = np.ones(xs.shape[0])/xs.shape[0]
            b2 = np.ones(xt.shape[0])/xt.shape[0]
            M = ot.dist(xs, xt)

            cost = ot.sinkhorn2(b1, b2, ot.dist(xs, xt), entr_reg)
            print("Sinkhorn cost: {}".format(cost))
            results_vanilla[sample, n_ind] = cost

            (zs, gammas) = kbarycenter(b1, b2, xs, xt, k, [1.0, 1.0], entr_reg, verbose = True,
                    warm_start = True, relax_outside = [np.inf, np.inf],
                    tol = 1e-3,
                    inner_tol = 1e-6,
                    max_iter = 200)
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(xs, xt, gammas)
            print("Barycenter cost: {}".format(bary_cost))
            results_kbary[sample, n_ind] = bary_cost

            gammas_kmeans = kmeans_transport(xs, xt, k)
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, n_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))

    with open(os.path.join("gaussian_results", "varn1.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "ns": ns,
            "k": k,
            "n_cluster": n_cluster,
            "spread": spread,
            "variance": variance,
            "entr_reg": entr_reg,
            "ground_truth": ground_truth,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary,
            "results_kmeans": results_kmeans,
            }, f)

def test_gaussian_mixture_vard():
    ds = (np.array([2])**np.linspace(2, 8, 10)).astype(int)
    ns = 10*ds
    samples = 20
    n_cluster = 4
    k = 4
    # prp = np.random.uniform(size = n_cluster)
    # prp /= np.sum(prp)
    # prp = np.array([0.1, 0.2, 0.3, 0.4])
    prp = np.array([0.25, 0.25, 0.25, 0.25])
    spread = 3.0
    variance = 1.0
    entr_reg = 10.0


    np.random.seed(42)

    # # psd matrix
    # A = np.random.normal(0, 1, (d, d))
    # A = A.dot(A.T)
    # print(A)
    # def trafo(x):
    #     return A.dot(x)


    # # Scaling
    # scale = 0.1
    # def trafo(x):
    #     return scale * x

    results_vanilla = np.empty((samples, len(ds)))
    results_kbary = np.empty((samples, len(ds)))
    results_kmeans = np.empty((samples, len(ds)))
    ground_truths = np.empty((samples, len(ds)))

    for (d_ind, d) in enumerate(ds):
        n = ns[d_ind]
        print("d = {}".format(d))
        for sample in range(samples):
            print("Sample: {}".format(sample))
            # Random direction
            direction = np.random.normal(0, 70/d, d)
            def trafo(x):
                return x + direction
            cluster_ids = list(range(n_cluster))
            cluster_centers = np.random.normal(0, spread, (n_cluster, d))
            c = np.random.choice(cluster_ids, p = prp, size = n)
            xs = cluster_centers[c,:] + np.random.normal(0, variance, size = (n, d))
            c = np.random.choice(cluster_ids, p = prp, size = n)
            xt = cluster_centers[c,:] + np.random.normal(0, variance, size = (n, d))
            xt = np.apply_along_axis(trafo, 1, xt)

            ground_truth = np.sum(prp.reshape(-1,1) * (np.apply_along_axis(trafo, 1, cluster_centers) - cluster_centers)**2)
            print("Ground truth: {}".format(ground_truth))
            ground_truths[sample, d_ind] = ground_truth


            b1 = np.ones(xs.shape[0])/xs.shape[0]
            b2 = np.ones(xt.shape[0])/xt.shape[0]
            M = ot.dist(xs, xt)

            cost = ot.sinkhorn2(b1, b2, ot.dist(xs, xt), entr_reg)
            print("Sinkhorn cost: {}".format(cost))
            results_vanilla[sample, d_ind] = cost

            (zs, gammas) = kbarycenter(b1, b2, xs, xt, k, [1.0, 1.0], entr_reg, verbose = True,
                    warm_start = True, relax_outside = [np.inf, np.inf],
                    tol = 1e-3,
                    inner_tol = 1e-6,
                    max_iter = 200)
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(xs, xt, gammas)
            print("Barycenter cost: {}".format(bary_cost))
            results_kbary[sample, d_ind] = bary_cost

            gammas_kmeans = kmeans_transport(xs, xt, k)
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, d_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))

    with open(os.path.join("gaussian_results", "vard.bin"), "wb") as f:
        pickle.dump({
            "ds": ds,
            "ns": ns,
            "k": k,
            "n_cluster": n_cluster,
            "spread": spread,
            "variance": variance,
            "entr_reg": entr_reg,
            "ground_truths": ground_truths,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary,
            "results_kmeans": results_kmeans,
            }, f)

def test_gaussian_mixture_vark():
    d = 30
    n = 10*d
    samples = 20
    n_cluster = 4
    ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # prp = np.random.uniform(size = n_cluster)
    # prp /= np.sum(prp)
    # prp = np.array([0.1, 0.2, 0.3, 0.4])
    prp = np.array([0.25, 0.25, 0.25, 0.25])
    d = 30
    spread = 3.0
    variance = 1.0
    entr_reg = 5.0


    np.random.seed(42)

    # # psd matrix
    # A = np.random.normal(0, 1, (d, d))
    # A = A.dot(A.T)
    # print(A)
    # def trafo(x):
    #     return A.dot(x)

    # Random direction
    direction = np.random.normal(0, 70/d, d)
    def trafo(x):
        return x + direction

    # # Scaling
    # scale = 0.1
    # def trafo(x):
    #     return scale * x

    cluster_centers = np.random.normal(0, spread, (n_cluster, d))
    samples_target = []
    samples_source = []
    cluster_ids = list(range(n_cluster))

    ground_truth = np.sum(prp.reshape(-1,1) * (np.apply_along_axis(trafo, 1, cluster_centers) - cluster_centers)**2)
    print("Ground truth: {}".format(ground_truth))

    results_vanilla = np.empty((samples))
    results_kbary = np.empty((samples, len(ks)))
    results_kmeans = np.empty((samples, len(ks)))

    for sample in range(samples):
        print("Sample: {}".format(sample))
        for i in range(n):
            c = np.random.choice(cluster_ids, p = prp)
            samples_source.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))
        for i in range(n):
            c = np.random.choice(range(n_cluster), p = prp)
            samples_target.append(cluster_centers[c,:] + np.random.normal(0, variance, (1, d)))

        xs = np.vstack([samples for samples in samples_target])
        xt = np.vstack([samples for samples in samples_source])
        xt = np.apply_along_axis(trafo, 1, xt)

        b1 = np.ones(xs.shape[0])/xs.shape[0]
        b2 = np.ones(xt.shape[0])/xt.shape[0]
        M = ot.dist(xs, xt)

        cost = ot.sinkhorn2(b1, b2, ot.dist(xs, xt), entr_reg)
        print("Sinkhorn cost: {}".format(cost))
        results_vanilla[sample] = cost

        for (k_ind, k) in enumerate(ks):

            (zs, gammas) = kbarycenter(b1, b2, xs, xt, k, [1.0, 1.0], entr_reg, verbose = True,
                    warm_start = True, relax_outside = [np.inf, np.inf],
                    tol = 1e-4,
                    inner_tol = 1e-6,
                    max_iter = 200)
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(xs, xt, gammas)
            print("Barycenter cost: {}".format(bary_cost))
            results_kbary[sample, k_ind] = bary_cost

            gammas_kmeans = kmeans_transport(xs, xt, k)
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, k_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))

    with open(os.path.join("gaussian_results", "vark1.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "n": n,
            "ks": ks,
            "n_cluster": n_cluster,
            "spread": spread,
            "variance": variance,
            "entr_reg": entr_reg,
            "ground_truth": ground_truth,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary,
            "results_kmeans": results_kmeans,
            }, f)


def test_split_data_uniform_all():
    prefix = "split_cube_results"

    d = 30
    n = 1000
    ks = np.hstack([range(1,11), range(15,51,5)]).astype(int)
    # ks = [10]
    entropies_l = [0.1, 1.0]
    middle_params_l = [0.5, 1.0, 2.0]
    samples = 20

    # Varying k
    for entropy in entropies_l:
        for middle_param in middle_params_l:
            ds = np.repeat(d, len(ks))
            ns = np.repeat(n, len(ks))
            middle_params = np.repeat(middle_param, len(ks))
            entropies = np.repeat(entropy, len(ks))
            filename = "vark_d_{}_n_{}_middle_{:.2e}_entropy_{:.2e}.bin".format(d, n, middle_param, entropy)
            test_split_data_uniform(ks, ds, ns, middle_params, entropies, samples, prefix, filename)

    # # Pictures
    # # ks = np.hstack([range(1,11), range(15,51,5)]).astype(int)
    # # ks = np.hstack([range(1,11)]).astype(int)
    # ks = [20]
    # ds = np.repeat(2, len(ks))
    # ns = np.repeat(100, len(ks))
    # middle_params = np.repeat(1.0, len(ks))
    # entropies = np.repeat(0.1, len(ks))
    # samples = 1
    # filename = "dummy.bin"
    # test_annulus(ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True)

def test_annulus_all():
    prefix = "annulus_results"

    d = 2
    n = 100
    # ks = np.hstack([range(1,11), range(15,51,5)]).astype(int)
    ks = [3]
    # entropies_l = [0.05]
    # entropies_l = [1.0, 10.0]
    entropies_l = [0.1]
    # middle_params_l = [0.5, 1.0, 2.0]
    middle_params_l = [1.0]
    samples = 1

    # Varying k
    for entropy in entropies_l:
        for middle_param in middle_params_l:
            ds = np.repeat(d, len(ks))
            ns = np.repeat(n, len(ks))
            middle_params = np.repeat(middle_param, len(ks))
            entropies = np.repeat(entropy, len(ks))
            filename = "vark_d_{}_n_{}_middle_{:.2e}_entropy_{:.2e}.bin".format(d, n, middle_param, entropy)
            # test_runs(generate_annulus_data, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True, visual_filename = "Annulus")
            # test_runs(generate_split_uniform_data, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True, visual_filename = "Cube")
            test_runs(generate_cluster_data, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True, visual_filename = "Clusters")
            # test_runs(generate_split_uniform_data, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True)

#     # Pictures
#     # ks = np.hstack([range(1,11), range(15,51,5)]).astype(int)
#     # ks = np.hstack([range(1,11)]).astype(int)
#     ks = [10]
#     ds = np.repeat(100, len(ks))
#     ns = np.repeat(100, len(ks))
#     middle_params = np.repeat(1.0, len(ks))
#     entropies = np.repeat(1.0, len(ks))
#     samples = 1
#     filename = "dummy.bin"
#     test_annulus(ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True)

def generate_split_uniform_data(d, n):
    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
    
    b1 = np.ones(samples_source.shape[0])/samples_source.shape[0]
    b2 = np.ones(samples_target.shape[0])/samples_target.shape[0]

    return (samples_source, samples_target, b1, b2)

def generate_cluster_data(d, n):
    centers_source = np.array([
            [-2, 2],
            [0, 0],
            [-2, -1]
            ])

    centers_target = np.array([
            [2, 3],
            [1, 0.5],
            [0, -1]
            ])

    xs = 0.3*np.random.randn(n, 2)
    xt = 0.3*np.random.randn(n, 2)
    labels_s = np.random.randint(3, size = (n,))
    labels_t = np.random.randint(3, size = (n,))
    xs += centers_source[labels_s,:]
    xt += centers_target[labels_t,:]
    b1 = np.ones(xs.shape[0])/xs.shape[0]
    b2 = np.ones(xt.shape[0])/xt.shape[0]

    return (xs, xt, b1, b2)


def generate_annulus_data(d, n):
    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = x[0:2]
        direction[0:2] /= np.linalg.norm(direction[0:2])
        return x + length * direction

    angles = np.random.uniform(0, 2*np.pi, size=n)
    radiuses = np.sqrt(np.random.uniform(0, 1, size=n))
    samples_source_2d = radiuses * np.vstack([np.sin(angles), np.cos(angles)])
    samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_source[:,0:2] = samples_source_2d.T

    # Generate targets
    angles = np.random.uniform(0, 2*np.pi, size=n)
    radiuses = np.sqrt(np.random.uniform(0, 1, size=n))
    samples_target_2d = radiuses * np.vstack([np.sin(angles), np.cos(angles)])
    samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target[:, 0:2] = samples_target_2d.T
    samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)

    xs = samples_source
    xt = samples_target
    b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
    b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]

    return (xs, xt, b1, b2)

def test_runs(data_generator,
        ks, ds, ns,
        middle_params, entropies, samples, prefix,
        filename, visual = False, figsize = (4,4), visual_filename = "Annulus",
        ):

    # k = 4
    # ks = np.hstack([range(1,11), range(15,31,5)]).astype(int)
    # ks = [100]
    # samples = 1
    # ds = (np.array([10.0])**np.linspace(1, 2, 10)).astype(int)
    # ds = np.array([5])
    # d = 20
    # ns = 500 * ds
    # n = 2000
    results_vanilla = np.empty((samples, len(ds)))
    results_kbary = np.empty((samples, len(ds)))
    results_kmeans = np.empty((samples, len(ds)))
    results_cluster_ot = np.empty((samples, len(ds)))
    results_cluster_ot_nocor = np.empty((samples, len(ds)))
    results_cluster_ot_map = np.empty((samples, len(ds)))
    results_cluster_ot_map_nocor = np.empty((samples, len(ds)))
    results_cluster_ot_fixed = np.empty((samples, len(ds)))
    results_cluster_ot_fixed_nocor = np.empty((samples, len(ds)))

    # for (d_ind, d) in enumerate(ds):
    for (k_ind, k) in enumerate(ks):
        for sample in range(samples):
            # Generate samples
            n = ns[k_ind]
            d = ds[k_ind]
            entropy = entropies[k_ind]
            middle_param = middle_params[k_ind]
            
            (samples_source, samples_target, b1, b2) = data_generator(d, n)
            xs = samples_source
            xt = samples_target

            # emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((samples_target,samples_source)))

            # pl.plot(emb_dat[0:samples_target.shape[0], 0], emb_dat[0:samples_target.shape[0], 1], '+b', label='Target samples')
            # pl.plot(emb_dat[samples_target.shape[0]:, 0], emb_dat[samples_target.shape[0]:, 1], 'xr', label='Source samples')
            # pl.show()
            
            # Run Sinkhorn
            cost = ot.emd2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, k_ind] = cost
            print("Sinkhorn: {}".format(cost))

            if visual:
                transport_plan = ot.emd(b1, b2, ot.dist(samples_source, samples_target))
                fig = pl.figure(figsize = figsize)
                ax = pl.axes(aspect = 1.0)
                ot.plot.plot2D_samples_mat(xs, xt, transport_plan, c=[.5, .5, .5])
                pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
                pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
                # pl.plot(zs[:, 0], zs[:, 1], '<c', label='Mid')
                # ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[1, .5, .5])
                # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
                pl.savefig(os.path.join("Figures", "{}_ot.png".format(visual_filename)), dpi = 300)
                pl.close(fig)

            # results = estimate_distances(b1, b2, samples_source, samples_target, ks, 1)
            # pl.plot(ks, results)
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 8, 16, [1.0, 1.0, 1.0], 1, verbose = True)
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 3, 3, [1.0, 1.0, 1.0], 1, verbose = True, relax_outside = [1e+5, 1e+5])
            # (zs1, zs2, a1, a2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, 8, 8, [1.0, 1.0, 1.0], 1, verbose = True, relax_inside = [1e0, 1e0])
            # (zs1, zs2, gammas) = reweighted_clusters(samples_source, samples_target, 8, [1.0, 0.5, 1.0], 5e-1, lb = float(1)/10)
            # cluster_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            # print("Cluster cost: {}".format(cluster_cost))

            # # Fixed weights
            # ret = reweighted_clusters(b1, b2, samples_source, samples_target, k, [1.0, middle_param, 1.0], entropy, equal_weights = True, tol = 1e-7)
            # if ret is not None:
            #     (zs1, zs2, gammas) = ret
            #     # cluster_cost_equal = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = True)
            #     cluster_cost_equal = estimate_w2_cluster(samples_source, samples_target, gammas)
            #     cluster_cost_equal_nocor = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = False)
            #     print("Fixed weights cost: {}".format(cluster_cost_equal))
            #     print("Fixed weights cost, no correction: {}".format(cluster_cost_equal_nocor))
            #     results_cluster_ot_fixed[sample, k_ind] = cluster_cost_equal
            #     results_cluster_ot_fixed_nocor[sample, k_ind] = cluster_cost_equal_nocor

            #     if visual:
            #     # if True:
            #         # fixed weights plot
            #         fig = pl.figure(figsize = figsize)
            #         pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
            #         pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
            #         pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
            #         pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
            #         ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
            #         ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
            #         ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
            #         # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
            #         pl.savefig(os.path.join("Figures","Annulus_fixed.png"), dpi = 300)
            #         pl.close(fig)
            # else:
            #     results_cluster_ot_fixed[sample, k_ind] = np.nan
            #     results_cluster_ot_fixed_nocor[sample, k_ind] = np.nan
            #     print("NaN encountered")

            # # Run hubot
            # # (zs1, zs2, gammas) = cluster_ot(b1, b2,
            # #         samples_source, samples_target, k, k,
            # #         [1.0, middle_param, 1.0],
            # #         np.array([entropy, entropy, entropy]),
            # #         verbose = True,
            # #         warm_start = True,
            # #         relax_outside = [np.inf, np.inf],
            # #         inner_tol = 1e-7, tol = 1e-6,
            # #         zs1_start = zs1, zs2_start = zs2, gammas_start = gammas,
            # #         plot_iterations = False)
            # ret = cluster_ot(b1, b2,
            #         samples_source, samples_target, k, k,
            #         [1.0, middle_param, 1.0],
            #         np.array([entropy, entropy, entropy]),
            #         verbose = True,
            #         warm_start = True,
            #         relax_outside = [np.inf, np.inf],
            #         inner_tol = 1e-7, tol = 1e-6,
            #         plot_iterations = False)
            # if ret is not None:
            #     (zs1, zs2, gammas) = ret
            #     cluster_cost = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = True)
            #     cluster_cost_no_correction = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = False)
            #     cluster_cost_old = estimate_w2_cluster(samples_source, samples_target, gammas)
            #     results_cluster_ot[sample, k_ind] = cluster_cost
            #     results_cluster_ot_nocor[sample, k_ind] = cluster_cost_no_correction
            #     print("Double cluster cost: {}".format(cluster_cost))
            #     print("Double cluster cost, old way: {}".format(cluster_cost_old))
            #     print("Double cluster cost, no correction: {}".format(cluster_cost_no_correction))


            #     # print(zs1)
            #     # print(zs2)

            #     if visual:
            #         # hubOT plot
            #         fig = pl.figure(figsize = figsize)
            #         pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
            #         pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
            #         pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
            #         pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
            #         ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
            #         ot.plot.plot2D_samples_mat(zs1, zs2, gammas[1], c=[.5, .5, .5])
            #         ot.plot.plot2D_samples_mat(zs2, xt, gammas[2], c=[1, .5, .5])
            #         # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
            #         pl.savefig(os.path.join("Figures","Annulus_hubot.png"), dpi = 300)
            #         pl.close(fig)
            # else:
            #     results_cluster_ot[sample, k_ind] = np.nan
            #     results_cluster_ot_nocor[sample, k_ind] = np.nan
            #     print("NaN encountered")


            # # Run hubot with map middle transport
            # ret = cluster_ot_map(b1, b2, samples_source, samples_target, k, [1.0, middle_param, 1.0], entropy, verbose = True, tol = 1e-6, warm_start = True)
            # if ret is not None:
            #     (zs1, zs2, gammas) = ret
            #     cluster_cost_map = estimate_w2_middle(samples_source, samples_target, zs1, zs2, [gammas[0], np.eye(k)/k, gammas[1]], bias_correction = True)
            #     cluster_cost_map_nocor = estimate_w2_middle(samples_source, samples_target, zs1, zs2, [gammas[0], np.eye(k)/k, gammas[1]], bias_correction = False)
            #     print("Map cluster cost: {}".format(cluster_cost_map))
            #     print("Map cluster cost, no correction: {}".format(cluster_cost_map_nocor))
            #     results_cluster_ot_map[sample, k_ind] = cluster_cost_map
            #     results_cluster_ot_map_nocor[sample, k_ind] = cluster_cost_map_nocor

            #     if visual:
            #         # map plot
            #         fig = pl.figure(figsize = figsize)
            #         pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
            #         pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
            #         pl.plot(zs1[:, 0], zs1[:, 1], '<c', label='Mid 1')
            #         pl.plot(zs2[:, 0], zs2[:, 1], '>m', label='Mid 2')
            #         ot.plot.plot2D_samples_mat(xs, zs1, gammas[0], c=[.5, .5, 1])
            #         ot.plot.plot2D_samples_mat(zs1, zs2, np.eye(k), c=[.5, .5, .5])
            #         ot.plot.plot2D_samples_mat(zs2, xt, gammas[1], c=[1, .5, .5])
            #         # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
            #         pl.savefig(os.path.join("Figures","Annulus_map.png"), dpi = 300)
            #         pl.close(fig)
            # else:
            #     results_cluster_ot_map[sample, k_ind] = np.nan
            #     results_cluster_ot_map_nocor[sample, k_ind] = np.nan
            #     print("NaN encountered")

            # Run HubOT
            ret = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf], tol = 1e-6, inner_tol = 1e-7)
            if ret is not None:
                (zs, gammas) = ret
                # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
                bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
                results_kbary[sample, k_ind] = bary_cost
                print("Barycenter cost: {}".format(bary_cost))

                if visual:
                # if True:
                    # Barycenter plot
                    fig = pl.figure(figsize = figsize)
                    ax = pl.axes(aspect = 1.0)
                    ot.plot.plot2D_samples_mat(xs, zs, gammas[0], c=[.5, .5, 1])
                    ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[1, .5, .5])
                    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
                    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
                    pl.plot(zs[:, 0], zs[:, 1], 'dc', label='Mid')
                    # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
                    pl.savefig(os.path.join("Figures", "{}_kbary.png".format(visual_filename)), dpi = 300)
                    pl.close(fig)

                    # Barycenter plot, straight
                    fig = pl.figure(figsize = figsize)
                    ax = pl.axes(aspect = 1.0)
                    # newtarget = map_from_clusters(xt, xs, [gamma.T for gamma in reversed(gammas)])
                    newsource = map_from_clusters(xs, xt, gammas)
                    plot_transport_map(xt, newsource, c = [.5, .5, .5])
                    pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
                    pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
                    # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
                    pl.savefig(os.path.join("Figures", "{}_kbary_straight.png".format(visual_filename)), dpi = 300)
                    pl.close(fig)

            else:
                results_kbary[sample, k_ind] = np.nan
                print("NaN encountered")

            gammas_kmeans = kmeans_transport(xs, xt, k)[2]
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, k_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))

    if not visual:
        mkdir(prefix)
        with open(os.path.join(prefix, filename), "wb") as f:
            pickle.dump({
                "ds": ds,
                "ns": ns,
                "ks": ks,
                "results_vanilla": results_vanilla,
                "results_kbary": results_kbary,
                "results_kmeans" : results_kmeans,
                "results_hubot" : results_cluster_ot,
                "results_hubot_nocor" : results_cluster_ot_nocor,
                "results_hubot_map" : results_cluster_ot_map,
                "results_hubot_map_nocor" : results_cluster_ot_map_nocor,
                "results_hubot_fixed" : results_cluster_ot_fixed,
                "results_hubot_fixed_nocor" : results_cluster_ot_fixed_nocor,
                }, f)

def test_split_data_uniform_vard():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    k = 4
    samples = 20
    ds = (np.array([10.0])**np.linspace(1, 2, 10)).astype(int)
    ns = 10 * ds
    results_vanilla = np.empty((samples, len(ds)))
    results_kbary = np.empty((samples, len(ds)))
    results_kmeans = np.empty((samples, len(ds)))

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

            gammas_kmeans = kmeans_transport(xs, xt, k)
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, d_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))

    #         # Barycenter plot
    #         pl.plot(xs[:, 0], xs[:, 1], '+b', label='Source samples')
    #         pl.plot(xt[:, 0], xt[:, 1], 'xr', label='Target samples')
    #         pl.plot(zs[:, 0], zs[:, 1], '<c', label='Mid')
    #         # ot.plot.plot2D_samples_mat(xs, zs, gammas[0], c=[.5, .5, 1])
    #         # ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[.5, .5, .5])
    #         plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
    #         pl.show()

    with open(os.path.join("uniform_results", "vard2.bin"), "wb") as f:
        pickle.dump({
            "ds": ds,
            "ns": ns,
            "k": k,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary,
            "results_kmeans" : results_kmeans
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
    centroid_ks = sim_params["centroid_ks"]
    gl_params = sim_params["gl_params"]
    samples_test = sim_params["samples_test"]
    samples_train = sim_params["samples_train"]
    estimators = sim_params["estimators"]
    nn_ks = sim_params["nn_ks"]

    # Determine best parameters

    opt_params = {}

    # Regular OT
    def ot_err(data, params, classifiers):
        print("Running OT for {}".format(params))
        (xs, xt, labs, labt) = data
        # Compute OT mapping
        gamma_ot = ot.da.EMDTransport().fit(Xs = xs, Xt = xt).coupling_
        # if np.any(np.isnan(gamma_ot)):
        #     return(np.inf)
        # else:
        # labt_pred = classify_ot(gamma_ot, labs_ind, labs)
        # labt_pred = classify_1nn(xs, bary_map(gamma_ot, xs), labs)
        labt_preds = [c.predict(bary_map(gamma_ot, xs)) for c in classifiers]
        # labt_pred = classify_knn(xs, bary_map(gamma_ot, xs), labs, 50)
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    def mnn_err(data, params, classifiers):
        print("Running MNN for {}".format(params))
        (xs, xt, labs, labt) = data

        savemat(os.path.join(".", "in.mat"), {
            "A" : xs,
            "B" : xt})
        # os.system("Rscript simple_MNN.R | /dev/null")
        with open(os.devnull, 'w') as devnull:
            subprocess.run(["Rscript", "simple_MNN.R"], stdout=devnull, stderr=devnull)
        mnn_data = loadmat(os.path.join(".", "out.mat"))
        os.remove("out.mat")
        os.remove("in.mat")
        xt_adj = mnn_data["B_corr"]

        labt_preds = [c.predict(xt_adj) for c in classifiers]
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    # Entropically regularized OT
    def ot_entr_err(data, params, classifiers):
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
        # labt_pred = classify_1nn(xs, bary_map(gamma_ot, xs), labs)
        labt_preds = [c.predict(bary_map(gamma_ot, xs)) for c in classifiers]
        expected_err = np.sum(gamma_ot * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
        # labt_pred = classify_knn(xs, bary_map(gamma_ot, xs), labs, 50)
        return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    # k means + OT
    def kmeans_ot_err(data, params, classifiers):
        k = params[0]
        print("Running kmeans + OT for {}".format(params))
        (xs, xt, labs, labt) = data
        (zs1, zs2, gammas) = kmeans_transport(xs, xt, k)
        if any([np.any(np.isnan(gamma)) for gamma in gammas]):
            return(np.inf)
        else:
            # labt_pred_km = classify_ot(gamma_km, labs_ind, labs)
            gamma_tot = total_gamma(gammas)
            expected_err = np.sum(gamma_tot * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            labt_preds_bary = [c.predict(bary_map(gamma_tot, xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas)) for c in classifiers]
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # hubOT
    def hot_err(data, params, classifiers):
        (entr_reg, k, middle_param) = params
        print("Running hubOT for {}".format(params))
        # Compute hub OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        hot_ret = cluster_ot(a, b, xs, xt, k, k, [1.0, middle_param, 1.0], entr_reg,
                relax_outside = [np.inf, np.inf],
                warm_start = True,
                inner_tol = 1e-6,
                tol = 1e-5,
                reduced_inner_tol = False,
                inner_tol_start = 1e-1,
                max_iter = 500,
                verbose = False,
                entr_reg_start = 1000.0
                )
        if hot_ret is None:
            return np.inf
        else:
            (zs1, zs2, gammas_k) = hot_ret
            # labt_pred_hot = classify_cluster_ot(gammas_k, labs_ind)
            gamma_total = total_gamma(gammas_k)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas_k), xs), labs)
            # labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas_k), labs)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas_k), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas_k)) for c in classifiers]
            # print(class_err_combined(labt, labt_pred_hot))
            # print(class_err_combined(labt, labt_pred_hot2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # hubOT, fixed weights
    def hot_fixed_err(data, params, classifiers):
        (entr_reg, k, middle_param) = params
        print("Running hubOT (fixed weights) for {}".format(params))
        # Compute hub OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        hot_ret = reweighted_clusters(a, b, xs, xt,
                k, [1.0, middle_param, 1.0], entr_reg,
                tol = 1e-6,
                max_iter = 500,
                equal_weights = True
                )
        if hot_ret is None:
            return np.inf
        else:
            (zs1, zs2, gammas_k) = hot_ret
            # labt_pred_hot = classify_cluster_ot(gammas_k, labs_ind)
            gamma_total = total_gamma(gammas_k)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas_k), xs), labs)
            # labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas_k), labs)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas_k), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas_k)) for c in classifiers]
            # print(class_err_combined(labt, labt_pred_hot))
            # print(class_err_combined(labt, labt_pred_hot2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # k barycenter
    def kbary_err(data, params, classifiers):
        (entr_reg, k) = params
        print("Running k barycenter for {}".format(params))
        # k barycenter OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        kbary_ret = kbarycenter(a, b, xs, xt, k, [1.0, 1.0],
                entr_reg,
                warm_start = True,
                tol = 1e-5,
                inner_tol = 1e-6,
                reduced_inner_tol = False,
                inner_tol_start = 1e-1,
                max_iter = 500,
                verbose = False,
                entr_reg_start = 1000.0
                )
        if kbary_ret is None:
            return np.inf
        else:
            (zs, gammas) = kbary_ret
            # labt_pred_kb = classify_kbary(gammas, labs_ind)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas)) for c in classifiers]
            gamma_total = total_gamma(gammas)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # print(class_err_combined(labt, labt_pred_kb))
            # print(class_err_combined(labt, labt_pred_kb2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # Group lasso
    def gl_ot_err(data, params, classifiers):
        (xs, xt, labs, labt) = data
        # (entr_reg, eta, samples, train) = params
        (entr_reg, eta) = params
        print("Running group lasso for {}".format(params))
        try:
            warnings.filterwarnings("error")
            gamma_gl = ot.da.SinkhornL1l2Transport(reg_e=entr_reg, reg_cl=eta).fit(Xs = xs, ys = labs, Xt = xt).coupling_
            # labt_pred_gl = classify_ot(gamma_gl, labs_ind, labs)
            labt_preds = [c.predict(bary_map(gamma_gl, xs)) for c in classifiers]
            return [class_err_combined(labt, labt_pred) for labt_pred  in labt_preds]
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")

    # hubOT, map
    def hot_map_err(data, params, classifiers):
        (entr_reg, k, middle_param) = params
        print("Running hubOT (restricted to map) for {}".format(params))
        # Compute hub OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        hot_ret = cluster_ot_map(a, b, xs, xt,
                k, [1.0, middle_param, 1.0], entr_reg,
                tol = 1e-6,
                max_iter = 500,
                verbose = False,
                warm_start = True
                )
        if hot_ret is None:
            return np.inf
        else:
            (zs1, zs2, gammas_k) = hot_ret
            # labt_pred_hot = classify_cluster_ot(gammas_k, labs_ind)
            gamma_total = total_gamma(gammas_k)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas_k), xs), labs)
            # labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas_k), labs)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas_k), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas_k)) for c in classifiers]
            # print(class_err_combined(labt, labt_pred_hot))
            # print(class_err_combined(labt, labt_pred_hot2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # k barycenter
    def kbary_err(data, params, classifiers):
        (entr_reg, k) = params
        print("Running k barycenter for {}".format(params))
        # k barycenter OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        kbary_ret = kbarycenter(a, b, xs, xt, k, [1.0, 1.0],
                entr_reg,
                warm_start = True,
                tol = 1e-6,
                inner_tol = 1e-7,
                reduced_inner_tol = False,
                inner_tol_start = 1e-1,
                max_iter = 500,
                verbose = False,
                entr_reg_start = 1000.0
                )
        if kbary_ret is None:
            return np.inf
        else:
            (zs, gammas) = kbary_ret
            # labt_pred_kb = classify_kbary(gammas, labs_ind)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas)) for c in classifiers]
            gamma_total = total_gamma(gammas)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # print(class_err_combined(labt, labt_pred_kb))
            # print(class_err_combined(labt, labt_pred_kb2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # Group lasso
    def gl_ot_err(data, params, classifiers):
        (xs, xt, labs, labt) = data
        # (entr_reg, eta, samples, train) = params
        (entr_reg, eta) = params
        print("Running group lasso for {}".format(params))
        try:
            warnings.filterwarnings("error")
            gamma_gl = ot.da.SinkhornL1l2Transport(reg_e=entr_reg, reg_cl=eta).fit(Xs = xs, ys = labs, Xt = xt).coupling_
            # labt_pred_gl = classify_ot(gamma_gl, labs_ind, labs)
            labt_preds = [c.predict(bary_map(gamma_gl, xs)) for c in classifiers]
            return [class_err_combined(labt, labt_pred) for labt_pred  in labt_preds]
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")

    # k barycenter
    def kbary_err(data, params, classifiers):
        (entr_reg, k) = params
        print("Running k barycenter for {}".format(params))
        # k barycenter OT
        (xs, xt, labs, labt) = data
        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        kbary_ret = kbarycenter(a, b, xs, xt, k, [1.0, 1.0],
                entr_reg,
                warm_start = True,
                tol = 1e-6,
                inner_tol = 1e-7,
                reduced_inner_tol = False,
                inner_tol_start = 1e-1,
                max_iter = 500,
                verbose = False,
                entr_reg_start = 1000.0
                )
        if kbary_ret is None:
            return np.inf
        else:
            (zs, gammas) = kbary_ret
            # labt_pred_kb = classify_kbary(gammas, labs_ind)
            labt_preds_bary = [c.predict(bary_map(total_gamma(gammas), xs)) for c in classifiers]
            labt_preds_map = [c.predict(map_from_clusters(xs, xt, gammas)) for c in classifiers]
            gamma_total = total_gamma(gammas)
            expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
            # print(class_err_combined(labt, labt_pred_kb))
            # print(class_err_combined(labt, labt_pred_kb2))
            # print()
            return [expected_err] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_bary] + [class_err_combined(labt, labt_pred) for labt_pred in labt_preds_map]
    
    # Group lasso
    def gl_ot_err(data, params, classifiers):
        (xs, xt, labs, labt) = data
        # (entr_reg, eta, samples, train) = params
        (entr_reg, eta) = params
        print("Running group lasso for {}".format(params))
        try:
            warnings.filterwarnings("error")
            gamma_gl = ot.da.SinkhornL1l2Transport(reg_e=entr_reg, reg_cl=eta).fit(Xs = xs, ys = labs, Xt = xt).coupling_
            # labt_pred_gl = classify_ot(gamma_gl, labs_ind, labs)
            labt_preds = [c.predict(bary_map(gamma_gl, xs)) for c in classifiers]
            return [class_err_combined(labt, labt_pred) for labt_pred  in labt_preds]
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")

    # Regularized OT
    def ot_map_err(data, params):
        (xs, xt, labs, labt) = data
        # (entr_reg, eta, samples, train) = params
        (mu, eta) = params
        print("Running regularized transport for {}".format(params))
        try:
            warnings.filterwarnings("error")
            # MappingTransport with linear kernel
            ot_mapping_linear = ot.da.MappingTransport(
                kernel="linear", mu=mu, eta=eta, bias=True,
                max_iter=20, verbose=True).fit(Xs = xs, Xt = xt)
            # labt_pred_gl = classify_ot(gamma_gl, labs_ind, labs)
            labt_preds = [c.predict(ot_mapping_linear.transform(Xs = xs)) for c in classifiers]
            return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]
        except RuntimeWarning:
            return np.inf
        finally:
            warnings.filterwarnings("default")

    def no_adjust_err(data, params, classifiers):
        (xs, xt, labs, labt) = data
        print("Running 1NN classifier")
        labt_preds = [c.predict(xt) for c in classifiers]
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    def subspace_align_err(data, params, classifiers):
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
        labt_preds = [KNeighborsClassifier(n_neighbors = k, algorithm='brute').fit(sourceAdapted, labs).predict(targetAdapted) for k in nn_ks]
        # labt_pred = [classify_1nn(sourceAdapted, targetAdapted, labs)
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    def tca_err(data, params, classifiers):
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
        # labt_pred = classify_1nn(sourceAdapted, targetAdapted, labs)
        labt_preds = [KNeighborsClassifier(n_neighbors = k, algorithm='brute').fit(sourceAdapted, labs).predict(targetAdapted) for k in nn_ks]
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    def coral_err(data, params, classifiers):
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
        # labt_pred = classify_1nn(sourceAdapted, targetAdapted, labs)
        # return class_err_combined(labt, labt_pred)
        labt_preds = [KNeighborsClassifier(n_neighbors = k, algorithm='brute').fit(sourceAdapted, labs).predict(targetAdapted) for k in nn_ks]
        return [class_err_combined(labt, labt_pred) for labt_pred in labt_preds]

    estimator_functions = {
            "ot": ot_err,
            "ot_entr": ot_entr_err,
            "ot_kmeans": kmeans_ot_err,
            "ot_2kbary": hot_err,
            "ot_2kbary_fixed": hot_fixed_err,
            "ot_2kbary_map": hot_map_err,
            "ot_kbary": kbary_err,
            "ot_gl": gl_ot_err,
            "ot_map": ot_map_err,
            "noadj": no_adjust_err,
            "sa": subspace_align_err,
            "tca": tca_err,
            "coral": coral_err,
            "mnn": mnn_err
            }

    estimator_outlen = {
            "ot": len(nn_ks),
            "ot_entr": 1 + len(nn_ks),
            "ot_kmeans": 1 + 2*len(nn_ks),
            "ot_2kbary": 1 + 2*len(nn_ks),
            "ot_kbary": 1 + 2*len(nn_ks),
            "ot_2kbary_fixed": 1 + 2*len(nn_ks),
            "ot_2kbary_map": 1 + 2*len(nn_ks),
            "ot_gl": len(nn_ks),
            "ot_map": len(nn_ks),
            "noadj": len(nn_ks),
            "sa": len(nn_ks),
            "tca": len(nn_ks),
            "coral": len(nn_ks),
            "mnn": len(nn_ks)
            }

    # Training
    est_train_results = {}
    est_test_results = {}
    skip_entr_reg = {}

    # Prepare arrays
    for (est_name, est_params) in estimators.items():
        parameters = est_params["parameter_ranges"]
        param_lens = list(map(len, parameters))
        outlen = estimator_outlen.get(est_params["function"], 1)
        if outlen != 1:
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
        classifiers = [KNeighborsClassifier(n_neighbors = k, algorithm='brute').fit(xs, labs) for k in nn_ks]
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
                err = est_fun((xs, xt, labs, labt), cur_params, classifiers)
                outlen = estimator_outlen.get(est_params["function"], 1)
                if err == np.inf:
                    print("Skipping entropy parameter: {}".format(cur_params[0]))
                    skip_entr_reg[est_name].append(cur_params[0])
                    est_train_results[est_name][comb_ind[0],...] = np.inf
                else:
                    if outlen != 1:
                        est_train_results[est_name][(*comb_ind, slice(None), sample)] = err
                    else:
                        est_train_results[est_name][(*comb_ind, sample)] = err[0]
        print("Time: {:.2f}s".format(time.time()-t0))

    # Pick best parameters
    est_train_err = {}
    opt_params = {}
    for (est_name, est_params) in estimators.items():
        parameters = est_params["parameter_ranges"]
        param_lens = list(map(len, parameters))
        outlen = estimator_outlen.get(est_params["function"], 1)
        if len(param_lens) == 0:
            opt_params[est_name] = [] if outlen == 1 else [[]]*outlen
            continue
        if outlen != 1:
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
        classifiers = [KNeighborsClassifier(n_neighbors = k, algorithm='kd_tree').fit(xs, labs) for k in nn_ks]
        for (est_name, est_params) in estimators.items():
            outlen = estimator_outlen.get(est_params["function"], 1)
            est_fun = estimator_functions[est_params["function"]]
            cur_params = opt_params[est_name]
            parameters = est_params["parameter_ranges"]
            param_lens = list(map(len, parameters))
            if outlen != 1:
                if len(param_lens) == 0:
                    err = est_fun((xs, xt, labs, labt), [], classifiers)
                    est_test_results[est_name][:,sample] = err
                else:
                    for j in range(outlen):
                        err = est_fun((xs, xt, labs, labt), cur_params[j], classifiers)
                        if err != np.inf:
                            err = err[j]
                        est_test_results[est_name][j, sample] = err
            else:
                err = est_fun((xs, xt, labs, labt), cur_params, classifiers)
                est_test_results[est_name][sample] = err[0]
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
    global data_ind, labels, xs, xt, labs, labt, features

    # np.random.seed(42*42)
    np.random.seed(42**3)

    # perclass = {"source": 100, "target": 100}
    samples = {"train": 20, "test": 20}
    # samples = {"train": 1, "test": 1}
    # label_samples = [20, 20, 20]
    label_samples = [100, 100, 100]
    # label_samples = [10, 10, 10, 10]
    # outfile = "pancreas.bin"
    outfile = "haem3_new_uncentered.bin"

    # entr_regs = np.array([10.0])**np.linspace(-3, 0, 7)
    entr_regs = np.array([10.0])**np.linspace(-3, -1, 5)
    # entr_regs = np.array([10.0])**range(-2, 0)
    # entr_regs = np.array([10.0])**range(-1, 0)
    # gl_params = np.array([10.0])**range(-3, 3)
    gl_params = np.array([10.0])**range(-3, 1)
    # map_params1 = np.array([10.0])**range(-1,1)
    # map_params2 = np.array([10.0])**range(-1,1)
    # gl_params = np.array([10.0])**range(-3, 0)
    # gl_params = np.array([0.1])
    # ks = np.array([2])**range(1, 8)
    # ks = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    # ks = np.array([5, 10, 15, 20, 25, 30])
    # ks = np.array([3, 5, 10, 20, 30, 50])
    # ks = np.array([3, 5, 10, 20, 30, 50])
    ks = np.array([3, 6, 9, 12, 20, 30])
    ds = np.array([10, 20, 30, 40, 50, 60, 70])
    middle_params = [0.5, 1.0, 2.0]
    # middle_params = [1.0]

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
                "parameter_ranges": [ks]
                },
            # "ot_2kbary": {
            #     "function": "ot_2kbary",
            #     "parameter_ranges": [entr_regs, ks, middle_params]
            #     },
            # "ot_2kbary_fixed": {
            #     "function": "ot_2kbary_fixed",
            #     "parameter_ranges": [entr_regs, ks, middle_params]
            #     },
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
                "parameter_ranges": [ds]
                },
            "tca": {
                "function": "tca",
                "parameter_ranges": [ds]
                },
            "mnn": {
                "function": "mnn",
                "parameter_ranges": []
            }
            # Exclude
            # "coral": {
            #     "function": "coral",
            #     "parameter_ranges": []
            #     }
            # Exclude
            # "ot_2kbary_map": {
            #     "function": "ot_2kbary_map",
            #     "parameter_ranges": [entr_regs, ks, middle_params]
            #     },
            # Exclude
            # "ot_map": {
            #     "function": "ot_map",
            #     "parameter_ranges": [map_params1, map_params2]
            #     },
            }

    domain_data = {}

    print("-"*30)
    print("Running tests for bio data")
    print("-"*30)

    data = loadmat(os.path.join(".", "MNN_haem_data.mat"))
    # data = loadmat(os.path.join(".", "pancreas.mat"))
    # xs = data['x2'].astype(float)
    # xt = data['x3'].astype(float)
    # labs = data['lab2'].ravel().astype(int)
    # labt = data['lab3'].ravel().astype(int)
    xs = data['xs'].astype(float)
    xt = data['xt'].astype(float)
    labs = data['labs'].ravel().astype(int)
    labt = data['labt'].ravel().astype(int)

    # Print label proportions:
    for c in np.unique(labs):
        print("Cluster {}".format(c))
        print("source: {}".format(np.sum(labs == c)))
        print("target: {}".format(np.sum(labt == c)))

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
                    # ind_list.extend(ind[:min(perclass[dataset], len(ind))])
                    ind_list.extend(ind[:label_samples[int(c)]])

    def get_data(train, sample):
        trainstr = "train" if train else "test"
        xs = features["source"][data_ind[trainstr]["source"][sample], :]
        xt = features["target"][data_ind[trainstr]["target"][sample], :]
        labs = labels["source"][data_ind[trainstr]["source"][sample]]
        labt = labels["target"][data_ind[trainstr]["target"][sample]]
        # xs = xs - np.mean(xs, axis = 0)
        # xt = xt - np.mean(xt, axis = 0)
        # labs_ind =  calc_lab_ind(labs)
        # return (xs, xt, labs, labt, labs_ind)
        print(xs.shape)
        print(xt.shape)
        return (xs, xt, labs, labt)

    simulation_params = {
            "entr_regs": entr_regs,
            "gl_params": gl_params,
            "centroid_ks": ks,
            "samples_test": samples["test"],
            "samples_train": samples["train"],
            "outfile": outfile,
            "estimators": estimators,
            "nn_ks": [1, 10, 20]
            }

    test_domain_adaptation(simulation_params, get_data)

def test_pancreas_data():
    global data_ind, labels, xs, xt, labs, labt

    perclass = {"source": 20, "target": 20}
    samples = {"train": 10, "test": 10}
    tasks = []

    for source in range(4):
        for target in range(4):
            if source != target:
                tasks.append({"source": source, "target": target})

    entr_regs = np.array([10.0])**range(-3, 2)
    # entr_regs = np.array([10.0])**range(-2, 0)
    # entr_regs = np.array([10.0])**range(-1, 0)
    gl_params = np.array([10.0])**range(-3, 3)
    # gl_params = np.array([10.0])**range(-3, 0)
    # gl_params = np.array([0.1])
    # ks = np.array([2])**range(1, 8)
    ks = np.array([10, 20, 30, 40, 50, 60, 70, 80])
    # ks = np.array([5, 10, 15, 20, 25, 30])

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

    print("-"*30)
    print("Running tests for pancreas data")
    print("-"*30)

    for task in tasks:
        print("-"*30)
        print("Running tests for pancreas data")
        print(str(task["source"]) + " to " + str(task["target"]))
        print("-"*30)
        outfile = "pancreas" + str(task["source"]) + "to" + str(task["target"]) + ".bin"
        data = loadmat(os.path.join(".", "pancreas.mat"))
        xs = data['x'+str(task["source"])].astype(float)
        xt = data['x'+str(task["target"])].astype(float)
        labs = data['lab'+str(task["source"])].ravel().astype(int)
        labt = data['lab'+str(task["target"])].ravel().astype(int)

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
                        len_s = np.sum(labs == c).astype(int)
                        len_t = np.sum(labt == c).astype(int)
                        np.random.shuffle(ind)
                        # ind_list.extend(ind[:min(perclass[dataset], len(ind)])
                        # ind_list.extend(ind[:label_samples[int(c)]])
                        ind_list.extend(ind[:min(len_s, len_t)])

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
    global gammas, gamma_total, zs1, zs2, zs1_l, zs2_l, xs, xt, labs, labt

    # np.random.seed(43)

    samples_to_run = 20

    # ls = np.array(range(1,21))
    ls = np.array([1, 10, 20, 30])
    # ls = np.array([1])

    kbary_1nc_bary_errs = np.empty(samples_to_run)
    kbary_1nc_map_errs = np.empty(samples_to_run)
    kbary_expected_errs = np.empty(samples_to_run)
    kbary_knn_bary_errs = np.empty((samples_to_run, len(ls)))
    kbary_knn_map_errs = np.empty((samples_to_run, len(ls)))

    ot_1nc_bary_errs = np.empty(samples_to_run)
    ot_knn_bary_errs = np.empty((samples_to_run, len(ls)))
    ot_expected_errs = np.empty(samples_to_run)

    bary_mismatches = []
    map_mismatches = []

    for sample_to_run in range(samples_to_run):
        print("Running sample {}".format(sample_to_run))
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

        perclass = {"source": 300, "target": 300}
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
        classifiers = [KNeighborsClassifier(n_neighbors = l, algorithm='brute').fit(xs, labs) for l in ls]

        n1 = xs.shape[0]
        n2 = xt.shape[0]
        # emb_dat2 = decomposition.PCA(n_components=2).fit_transform(np.vstack((xs,xt)))
        # xs = emb_dat2[:n1,:]
        # xt = emb_dat2[n1:n1+n2,:]

        k = 3
        entr_reg = 0.05

        xs_centr = []
        labs_centr = []
        for c in np.unique(labs):
            xs_centr.append(np.mean(xs[labs == c,:], axis = 0))
            labs_centr.append(c)
        xs_centr = np.array(xs_centr)
        labs_centr = np.array(labs_centr).astype(int)

        a = np.ones(xs.shape[0])/xs.shape[0]
        b = np.ones(xt.shape[0])/xt.shape[0]
        gamma_ot = ot.sinkhorn(a, b, ot.dist(xs, xt), entr_reg)
        # hot_ret = cluster_ot(a, b, xs, xt, k, k, [1.0, 1.0, 1.0], entr_reg,
        #         relax_outside = [np.inf, np.inf],
        #         warm_start = True,
        #         inner_tol = 1e-4,
        #         tol = 1e-5,
        #         reduced_inner_tol = True,
        #         inner_tol_start = 1e0,
        #         max_iter = 300,
        #         verbose = False,
        #         entr_reg_start = 100.0,
        #         debug = True,
        #         )

        # # gamma_ot = ot.sinkhorn(a, b, ot.dist(xs, xt), entr_reg)
        # gamma_ot = ot.emd(a, b, ot.dist(xs, xt))
        bary_map_v = bary_map(gamma_ot, xs)
        ot_expected_errs[sample_to_run] = np.sum(gamma_ot * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
        ot_knn_bary_errs[sample_to_run,:] = [class_err_combined(labt, classifiers[l_ind].predict(bary_map_v)) for l_ind in range(len(ls))]
        ot_1nc_bary_errs[sample_to_run] = class_err_combined(labt, classify_1nn(xs_centr, bary_map(gamma_ot, xs), labs_centr))
        labt_ot = [classify_knn(xs, bary_map_v, labs, l) for l in ls]

        # (zs1, zs2, a1, a2, gammas, zs1_l, zs2_l) = hot_ret
        # gamma_total = total_gamma(gammas)
        # expected_err = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
        # labt_pred_hot = classify_1nn(xs, bary_map(total_gamma(gammas), xs), labs)
        # labt_pred_hot2 = classify_1nn(xs, map_from_clusters(xs, xt, gammas), labs)

        kbary_ret = kbarycenter(a, b, xs, xt, k, [1.0, 1.0],
                    entr_reg,
                    warm_start = True,
                    tol = 1e-4,
                    inner_tol = 1e-5,
                    reduced_inner_tol = False,
                    inner_tol_start = 1e0,
                    max_iter = 100,
                    verbose = True,
                    entr_reg_start = 100.0,
                    debug = True
                    )
        (zs, gammas_kbary, zs_l) = kbary_ret

        gamma_total = total_gamma(gammas_kbary)
        cluster_map = map_from_clusters(xs, xt, gammas_kbary)
        bary_map_v = bary_map(total_gamma(gammas_kbary), xs)

        kbary_expected_errs[sample_to_run] = np.sum(gamma_total * (labs.reshape(-1, 1) != labt.reshape(1, -1)))
        kbary_knn_bary_errs[sample_to_run, :] = [class_err_combined(labt, classifiers[l_ind].predict(bary_map_v)) for l_ind in range(len(ls))]
        # labt_kbary_bary = [classify_knn(xs, bary_map_v, labs, l) for l in ls]
        kbary_knn_map_errs[sample_to_run, :] = [class_err_combined(labt, classifiers[l_ind].predict(cluster_map)) for l_ind in range(len(ls))]
        # labt_kbary_map = [classify_knn(xs, cluster_map, labs, l) for l in ls]
        xsnew = map_from_clusters(xs, xt, gammas_kbary)
        # xsnew = bary_map(total_gamma(gammas_kbary), xs)
        kbary_1nc_bary_errs[sample_to_run] = class_err_combined(labt, classify_1nn(xs_centr, bary_map_v, labs_centr))
        kbary_1nc_bary_errs[sample_to_run] = class_err_combined(labt, classify_1nn(xs_centr, cluster_map, labs_centr))

        # # Check mislabeled points
        # for (l_ind, l) in enumerate(ls):
        #     labt_ot_mismatch = labt_ot[l_ind] != labt
        #     labt_kbary_map_mismatch = labt_kbary_map[l_ind] != labt
        #     labt_kbary_bary_mismatch = labt_kbary_bary[l_ind] != labt
        #     base1 = np.union1d(np.argwhere(labt_ot_mismatch).ravel(), np.argwhere(labt_kbary_map_mismatch).ravel())
        #     base2 = np.union1d(np.argwhere(labt_ot_mismatch).ravel(), np.argwhere(labt_kbary_bary_mismatch).ravel())

        #     map_mismatches.append(np.mean((labt_ot_mismatch * labt_kbary_map_mismatch)[base1]))
        #     bary_mismatches.append(np.mean((labt_ot_mismatch * labt_kbary_bary_mismatch)[base2]))


    mean_mat = lambda x: np.mean(x, axis = 0)
    std_mat = lambda x: np.std(x, axis = 0)

    # print(expected_err)
    print("KBARY:")
    print("1NC, bary:      {} +- {}".format(*[fun(kbary_1nc_bary_errs) for fun in [np.mean, np.std]]))
    print("1NC, map:       {} +- {}".format(*[fun(kbary_1nc_map_errs) for fun in [np.mean, np.std]]))
    print("Expected error: {} +- {}".format(*[fun(kbary_expected_errs) for fun in [np.mean, np.std]]))
    print("kNN, bary:      {} +- {}".format(*[fun(kbary_knn_bary_errs) for fun in [mean_mat, std_mat]]))
    print("kNN, map:       {} +- {}".format(*[fun(kbary_knn_map_errs) for fun in [mean_mat, std_mat]]))
    print()
    print("ENTR OT")
    print("1NC, bary:      {} +- {}".format(*[fun(ot_1nc_bary_errs) for fun in [np.mean, np.std]]))
    print("Expected error: {} +- {}".format(*[fun(ot_expected_errs) for fun in [np.mean, np.std]]))
    print("kNN, bary:      {} +- {}".format(*[fun(ot_knn_bary_errs) for fun in [mean_mat, std_mat]]))
    # print()
    # print("Map mismatch:   {} += {}".format(*[fun(map_mismatches) for fun in [np.mean, np.std]]))
    # print("Bary mismatch:   {} += {}".format(*[fun(bary_mismatches) for fun in [np.mean, np.std]]))

    # savemat("bio_diag.mat", {
    #     "xs": xs,
    #     "xt": xt,
    #     "labs": labs,
    #     "labt": labt,
    #     "zs1": zs1,
    #     "zs2": zs2,
    #     "zs": zs,
    #     "zs1_l": zs1_l,
    #     "zs2_l": zs2_l,
    #     "zs_l": zs_l,
    #     "gammas": gammas,
    #     "gammas_kbary": gammas_kbary,
    #     "gamma_ot": gamma_ot,
    #     "gamma_total": gamma_total
    #     })

    # ##% plot OT from source to target
    # n1 = xs.shape[0]
    # n2 = xt.shape[0]
    # # nz = zs1.shape[0]
    # # nz = zs.shape[0]
    # nz = xsnew.shape[0]
    # nc = xs_centr.shape[0]
    # # emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((xs,xt,zs1,zs2)))
    # # emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((xs,xt,zs)))

    # # emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((xs,xt,xsnew,xs_centr)))
    # emb_dat = np.vstack((xs,xt,xsnew,xs_centr))

    # label_mask = labs.reshape(-1, 1) == labt.reshape(1, -1)
    # gamma_ot_cor = label_mask * gamma_ot
    # gamma_ot_wrong = np.logical_not(label_mask) * gamma_ot

    # colors = ['red','green','blue']
    # mark_size = 40
    # mark_hub = 8
    # pl.scatter(*emb_dat[:n1,:].T, marker = 'o', label='Source samples', c=labs, cmap=matplotlib.colors.ListedColormap(colors), s=mark_size)
    # pl.scatter(*emb_dat[n1:n1+n2,:].T, marker = '^', label='Target samples', c=labt, cmap=matplotlib.colors.ListedColormap(colors), s=mark_size)
    # pl.scatter(*emb_dat[n1+n2+nz:n1+n2+nz+nc,:].T, marker = 'x', label='Centroids', c=labs_centr, cmap=matplotlib.colors.ListedColormap(colors), s=200)
    # plot_transport_map(emb_dat[n1:n1+n2,:], emb_dat[n1+n2:,:])
    # # pl.plot(*emb_dat[n1+n2:n1+n2+nz,:].T, 'xk', markersize=mark_hub,fillstyle='full')
    # # ot.plot.plot2D_samples_mat(emb_dat[:n1,:], emb_dat[n1+n2:n1+n2+nz,:], gammas_kbary[0])
    # # ot.plot.plot2D_samples_mat(emb_dat[n1+n2:n1+n2+nz,:], emb_dat[n1:n1+n2,:], gammas_kbary[1])
    # # pl.plot(*emb_dat[n1+n2+nz:,:].T, 'ob', markersize=mark_hub,fillstyle='full')
    # # ot.plot.plot2D_samples_mat(emb_dat[:n1,:], emb_dat[n1:n1+n2,:], gamma_ot_cor, c = [.5, .5, 1.0])
    # # ot.plot.plot2D_samples_mat(emb_dat[:n1,:], emb_dat[n1:n1+n2,:], gamma_ot_wrong, c = [1.0, .5, .5])
    # pl.legend()
    # pl.title('OT matrix with samples')
    # pl.legend(loc=0)
    # pl.show()

def test_bio_diag2():
    global xs, xt, labs, labt, emb

    samples = {"train": 1, "test": 1}
    label_samples = [874, 262, 92, 57]

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

    print([np.sum(labs == i) for i in range(4)])
    print([np.sum(labt == i) for i in range(4)])

    (xs, xt, labs, labt) = get_data(True, 0)
    ns = xs.shape[0]
    emb = manifold.TSNE().fit_transform(np.vstack([xs, xt]))
    # emb = decomposition.PCA(n_components = 2).fit_transform(np.vstack([xs, xt]))
    colors = ["r", "b", "g", "k"]
    pl.scatter(*emb[:ns,:].T, c = labs, cmap = pl.get_cmap("jet"))
    pl.scatter(*emb[ns:,:].T, c = labt, cmap = pl.get_cmap("jet"))
    pl.show()

def test_bio_diag3():
    global xs, xt, labs, labt, emb, xt_adj

    samples = {"train": 1, "test": 1}
    perclass = {"source": 50, "target": 50}
    # label_samples = [874, 262, 92, 57]

    data = loadmat(os.path.join(".", "MNN_haem_data.mat"))
    xs = data['xs'].astype(float)
    xt = data['xt'].astype(float)
    labs = data['labs'].ravel().astype(int)
    labt = data['labt'].ravel().astype(int)

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
                    ind_list.extend(ind[:min(perclass[dataset], len(ind))])
                    # ind_list.extend(ind[:label_samples[int(c)]])

    def get_data(train, sample):
        trainstr = "train" if train else "test"
        xs = features["source"][data_ind[trainstr]["source"][sample], :]
        xt = features["target"][data_ind[trainstr]["target"][sample], :]
        labs = labels["source"][data_ind[trainstr]["source"][sample]]
        labt = labels["target"][data_ind[trainstr]["target"][sample]]
        # labs_ind =  calc_lab_ind(labs)
        # return (xs, xt, labs, labt, labs_ind)
        return (xs, xt, labs, labt)

    print([np.sum(labs == i) for i in range(4)])
    print([np.sum(labt == i) for i in range(4)])

    (xs, xt, labs, labt) = get_data(True, 0)

    savemat(os.path.join(".", "in.mat"), {
        "A" : xs,
        "B" : xt})
    # os.system("Rscript simple_MNN.R | /dev/null")
    devnull = open(os.devnull, 'w')
    subprocess.run(["Rscript", "simple_MNN.R"], stdout=devnull, stderr=devnull)
    mnn_data = loadmat(os.path.join(".", "out.mat"))
    os.remove("out.mat")
    os.remove("in.mat")
    xt_adj = mnn_data["B_corr"]

    ns = xs.shape[0]
    nt = xt.shape[0]
    # emb = manifold.TSNE().fit_transform(np.vstack([xs, xt]))
    emb = decomposition.PCA(n_components = 2).fit_transform(np.vstack([xs, xt, xt_adj]))
    # colors = ["r", "b", "g", "k"]
    colors = ['red','green','blue']
    pl.scatter(*emb[:ns,:].T, c = labs, cmap = matplotlib.colors.ListedColormap(colors))
    pl.scatter(*emb[ns:ns+nt,:].T, marker="^", c = labt, cmap = matplotlib.colors.ListedColormap(colors))
    pl.scatter(*emb[ns+nt:,:].T, marker=">", c = labt, cmap = matplotlib.colors.ListedColormap(colors))
    pl.show()

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
    # test_gaussian_mixture_varn()
    # test_gaussian_mixture_vard()
    # test_gaussian_mixture_vark()


    # test_constraint_ot()
    # t0 = time.time()
    # test_moons()
    # print("Time for moons: {}".format(time.time() - t0))
    # test_opt_grid()
    # test_moons_kplot()
    # test_satija()
    # test_caltech_office()
    # test_bio_diag()
    # test_bio_diag2()
    # test_bio_diag3()
    # test_pancreas_data()

    # test_annulus_all()
    # test_split_data_uniform_all()
    test_bio_data()
    # test_alternating_min()
