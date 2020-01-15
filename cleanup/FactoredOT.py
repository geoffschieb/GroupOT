import numpy as np
import scipy.stats as stats
from scipy.io import loadmat, savemat
import matplotlib
import ot
import ot.plot
import scipy as sp
from scipy.special import kl_div, entr
from sklearn import manifold, datasets, decomposition
import matplotlib.pyplot as pl
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

def set_fig_params(case):
    # pl.subplots_adjust(**{'left': 0.19271045301649306, 'right': 0.963, 'bottom': 0.17794444444444446, 'top': 0.8991666666666667, 'wspace': 0.2, 'hspace': 0.2})
    # pl.subplots_adjust(**{'left': .1, 'right': 0.95, 'bottom': 0.17794444444444446, 'top': 0.8991666666666667, 'wspace': 0.2, 'hspace': 0.2})
    if case == 'a':
        pl.subplots_adjust(**{'left': 0.19271045301649306, 'right': 0.963, 'bottom': 0.17794444444444446, 'top': 0.8991666666666667, 'wspace': 0.2, 'hspace': 0.2})
    elif case == 'b':
        pl.subplots_adjust(**{'left': 0.19271045301649306, 'right': 0.963, 'bottom': 0.17794444444444446, 'top': 0.8991666666666667, 'wspace': 0.2, 'hspace': 0.2})
    elif case == 'c':
        pl.subplots_adjust(**{'left' : 0.17255555555555555, 'right': 0.963, 'bottom': 0.17069444444444445, 'top': 0.8991666666666667, 'wspace': 0.2, 'hspace': 0.2})

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

def plot_transport_map(xs, xt, **kwds):
    for i in range(xs.shape[0]):
        pl.plot([xs[i, 0], xt[i, 0]], [xs[i, 1], xt[i, 1]], 'k', alpha = 0.5, **kwds)

def update_points(xs, xt, zs, gammas, lambdas,
        verbose = False):

    zs = (lambdas[0] * np.dot(xs.T, gammas[0]).T + lambdas[1] * np.dot(gammas[1], xt)) / \
        (lambdas[0] * np.sum(gammas[0], axis = 0) + lambdas[1] * np.sum(gammas[1], axis = 1)).reshape(-1,1)

    return zs

def factored_ot(b1, b2, xs, xt, k,
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
    source_means = KMeans(n_clusters = k).fit(xs).cluster_centers_
    zs = source_means

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
            print("FactoredOT iteration: {}".format(its))
        if its > 1:
            zs = update_points(xs, xt, zs, gammas, lambdas)
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

    return total_cost

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

def synthetic_benchmark(
        gen_fun = generate_split_uniform_data,
        prefix = "results",
        fig_prefix = "cube",
        ns = 100,
        ds = 2**6,
        ks = 4,
        entropy = 0.1,
        middle_param = 1.0,
        samples = 20,
        task = "vark"
        ):


    # Varying k
    if task == "vark":
        if not hasattr(ds, '__len__'):
            ds = np.repeat(ds, len(ks))
        if not hasattr(ns, '__len__'):
            ns = np.repeat(ns, len(ks))
        middle_params = np.repeat(middle_param, len(ks))
        entropies = np.repeat(entropy, len(ks))
        filename = "vark.bin"
        test_runs(gen_fun, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = False, visual_filename = fig_prefix)

    # Varying n
    if task == "varn":
        if not hasattr(ds, '__len__'):
            ds = np.repeat(ds, len(ns))
        if not hasattr(ks, '__len__'):
            ks = np.repeat(ks, len(ns))
        middle_params = np.repeat(middle_param, len(ks))
        entropies = np.repeat(entropy, len(ks))
        filename = "varn.bin"
        test_runs(gen_fun, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = False, visual_filename = fig_prefix)

    if task == "vard":
        # Varying d
        if not hasattr(ks, '__len__'):
            ks = np.repeat(ks, len(ds))
        if not hasattr(ns, '__len__'):
            ns = np.repeat(ns, len(ds))
        middle_params = np.repeat(middle_param, len(ks))
        entropies = np.repeat(entropy, len(ks))
        filename = "vard.bin"
        test_runs(gen_fun, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = False, visual_filename = fig_prefix)

    if task == "figures":

        filename = "figures.bin"
        middle_params = np.repeat(middle_param, len(ks))
        entropies = np.repeat(entropy, len(ks))
        test_runs(gen_fun, ks, ds, ns, middle_params, entropies, samples, prefix, filename, visual = True, visual_filename = fig_prefix)

def test_runs(data_generator,
        ks, ds, ns,
        middle_params, entropies, samples,
        prefix = "results",
        filename = "results.bin",
        visual = False, figsize = (5,4),
        visual_filename = "vis",
        ):

    results_vanilla = np.empty((samples, len(ds)))
    results_factored_ot = np.empty((samples, len(ds)))
    results_kmeans = np.empty((samples, len(ds)))

    for (k_ind, k) in enumerate(ks):
        for sample in range(samples):
            # Generate samples
            n = ns[k_ind]
            d = ds[k_ind]

            print(30*'-')
            print("k = {}, n = {}, d = {}".format(k, n, d))
            print(30*'-')
            entropy = entropies[k_ind]
            middle_param = middle_params[k_ind]
            
            (samples_source, samples_target, b1, b2) = data_generator(d, n)
            xs = samples_source
            xt = samples_target

            
            # Run Sinkhorn
            dist = ot.emd2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, k_ind] = dist
            print("Sinkhorn: {}".format(dist))

            if visual:
                plotting_params = {"markersize" : 9, "markeredgewidth": 2.0}
                transport_plan = ot.emd(b1, b2, ot.dist(samples_source, samples_target))
                fig = pl.figure(figsize = figsize)
                # ax = pl.axes(aspect = 1.0)
                ot.plot.plot2D_samples_mat(xs, xt, transport_plan, c=[.6, .6, .6])
                pl.plot(xs[:, 0], xs[:, 1], '+r', label='Source samples', **plotting_params)
                pl.plot(xt[:, 0], xt[:, 1], 'xb', label='Target samples', **plotting_params)
                set_fig_params('a')
                pl.tight_layout()
                os.makedirs("Figures", exist_ok = True)
                pl.savefig(os.path.join("Figures", "{}_ot.png".format(visual_filename)), dpi = 600)
                pl.close(fig)

            # Run FactoredOT
            ret = factored_ot(b1, b2, samples_source, samples_target, k, [1.0, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf], tol = 1e-6, inner_tol = 1e-7, max_iter = 300)
            if ret is not None:
                (zs, gammas) = ret
                factored_ot_dist = estimate_w2_cluster(samples_source, samples_target, gammas)
                results_factored_ot[sample, k_ind] = factored_ot_dist
                print("Hubot distance: {}".format(factored_ot_dist))

                if visual:
                # if True:
                    # Barycenter plot
                    fig = pl.figure(figsize = figsize)
                    # ax = pl.axes(aspect = 1.0)
                    ot.plot.plot2D_samples_mat(xs, zs, gammas[0], c=[1, .5, .5])
                    ot.plot.plot2D_samples_mat(zs, xt, gammas[1], c=[.5, .5, 1])
                    pl.plot(xs[:, 0], xs[:, 1], '+r', label='Source samples', **plotting_params)
                    pl.plot(xt[:, 0], xt[:, 1], 'xb', label='Target samples', **plotting_params)
                    pl.plot(zs[:, 0], zs[:, 1], '^k', label='Mid', **plotting_params)
                    # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
                    set_fig_params('b')
                    pl.tight_layout()
                    os.makedirs("Figures", exist_ok = True)
                    pl.savefig(os.path.join("Figures", "{}_hubot.png".format(visual_filename)), dpi = 600)
                    pl.close(fig)

                    # Barycenter plot, straight
                    fig = pl.figure(figsize = figsize)
                    # ax = pl.axes(aspect = 1.0)
                    # newtarget = map_from_clusters(xt, xs, [gamma.T for gamma in reversed(gammas)])
                    newsource = map_from_clusters(xs, xt, gammas)
                    plot_transport_map(xt, newsource, c = [.5, .5, .5])
                    # plot_transport_map(newtarget, xs, c = [.5, .5, .5])
                    pl.plot(xs[:, 0], xs[:, 1], '+r', label='Source samples', **plotting_params)
                    pl.plot(xt[:, 0], xt[:, 1], 'xb', label='Target samples', **plotting_params)
                    # plot_transport_map(xt, map_from_clusters(xs, xt, gammas))
                    set_fig_params('c')
                    pl.tight_layout()
                    pl.savefig(os.path.join("Figures", "{}_hubot_straight.png".format(visual_filename)), dpi = 600)
                    pl.close(fig)

            else:
                results_factored_ot[sample, k_ind] = np.nan
                print("NaN encountered")

            # K-Means
            gammas_kmeans = kmeans_transport(xs, xt, k)[2]
            kmeans_dist = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, k_ind] = kmeans_dist
            print("Kmeans distance: {}".format(kmeans_dist))

    if not visual:
        mkdir(prefix)
        with open(os.path.join(prefix, filename), "wb") as f:
            pickle.dump({
                "ds": ds,
                "ns": ns,
                "ks": ks,
                "results_vanilla": results_vanilla,
                "results_factored_ot": results_factored_ot,
                "results_kmeans" : results_kmeans,
                }, f)

def run_cube():
    prefix = "results_cube"
    samples = 2

    synthetic_benchmark(
        gen_fun = generate_split_uniform_data,
        prefix = prefix,
        ds = 30,
        ns = 1000,
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        ks = np.array(range(1, 10, 2)).astype(int),
        entropy = [0.1],
        middle_param = [1.0],
        # samples = samples,
        samples = 1,

        task = "vark"
    )

    synthetic_benchmark(
        gen_fun = generate_split_uniform_data,
        prefix = prefix,
        ns = (np.array([10.0])**np.linspace(1.0, 2.7, 20)).astype(int),
        ds = 30,
        ks = 10,
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "varn"
    )

    ds = (np.array([2])**np.linspace(2, 8, 10)).astype(int)
    synthetic_benchmark(
        gen_fun = generate_split_uniform_data,
        prefix = prefix,
        ds = ds,
        ns = 10 * ds,
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        ks = 10,
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "vard"
    )

    synthetic_benchmark(
        gen_fun = generate_split_uniform_data,
        prefix = prefix,
        ds = [30],
        ns = [100],
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        ks = [4],
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "figures"
    )

def run_annulus():
    prefix = "results_annulus"
    samples = 2

    synthetic_benchmark(
        gen_fun = generate_annulus_data,
        prefix = prefix,
        ds = 30,
        ns = 1000,
        ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        entropy = 0.1,
        middle_param = 1.0,
        # samples = samples,
        samples = 1,

        task = "vark"
    )

    synthetic_benchmark(
        gen_fun = generate_annulus_data,
        prefix = prefix,
        ns = (np.array([10.0])**np.linspace(1.0, 2.7, 20)).astype(int),
        ds = 30,
        ks = 10,
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "varn"
    )

    ds = (np.array([2])**np.linspace(6, 8, 4)).astype(int)
    synthetic_benchmark(
        gen_fun = generate_annulus_data,
        prefix = prefix,
        ds = ds,
        ns = 10 * ds,
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        ks = 10,
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "vard"
    )

    synthetic_benchmark(
        gen_fun = generate_annulus_data,
        prefix = prefix,
        ds = [30],
        ns = [100],
        # ks = np.hstack([range(1,11), range(15,31,5), range(31, 201, 20)]).astype(int),
        ks = [10],
        entropy = 0.1,
        middle_param = 1.0,
        # samples = 20,
        samples = samples,

        task = "figures"
    )

if __name__ == "__main__":
    run_cube()
    run_annulus()
