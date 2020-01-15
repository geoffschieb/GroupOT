import numpy as np
import scipy.stats as stats
import HubOT as hot
import pickle
import matplotlib
import matplotlib.pyplot as pl
import ot
import os
from argparse import Namespace
from os.path import join
from sklearn import manifold, datasets, decomposition

dpi = 600
figsize = (5,4)

fig_folder = "Figures"

new_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728',
              '#9467bd', '#8c564b', '#e377c2', '#7f7f7f',
              '#bcbd22', '#17becf']

def unif_vark_plot():
    global dat

    os.makedirs(fig_folder, exist_ok = True)
    with open(os.path.join("results_cube", "vark.bin"), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    fig = pl.figure(figsize = figsize)
    pl.plot(dat.ks, np.mean(dat.results_factored_ot, axis = 0), "o-", label = "FactoredOT")
    pl.plot(dat.ks, np.array([8]).repeat(len(dat.ks)), "-", label = "ground truth")
    pl.xlabel("$k$")
    pl.ylabel(r"$\hat W_2^2(\hat P, \hat Q)$")
    pl.title("Varying $k$, $d = {}$, n = {}".format(dat.ds[0], dat.ns[0]))
    pl.legend(loc = "lower right")
    pl.tight_layout()
    pl.savefig(join(fig_folder, "uniform_vark.png"), dpi = dpi)
    print("(c)")
    print(fig.subplotpars.__dict__)
    pl.close(fig)

def unif_vard_plot():
    global dat
    os.makedirs(fig_folder, exist_ok = True)
    with open(os.path.join("results_cube", "vard.bin"), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    fig = pl.figure(figsize = figsize)
    pl.loglog(dat.ds, np.mean(np.abs(dat.results_vanilla - 8), axis = 0), "x-", label="OT", c = new_colors[0])
    pl.loglog(dat.ds, np.mean(np.abs(dat.results_factored_ot - 8), axis = 0), "o-", label = "FactoredOT", c = new_colors[1])
    pl.loglog(dat.ds, np.mean(np.abs(dat.results_kmeans - 8), axis = 0), "^-", label = "k-means & OT", c = new_colors[4])
    pl.xlabel("$d$")
    pl.ylabel(r"$|\hat W_2^2(\hat P, \hat Q) - W_2^2(P, Q)|$")
    pl.title("Varying d, $n = 10d, k = {}$".format(dat.ks[0]))
    pl.legend(loc = "upper left")
    pl.tight_layout()
    pl.savefig(join(fig_folder, "uniform_vard.png"), dpi = dpi)
    print("(b)")
    print(fig.subplotpars.__dict__)
    pl.close(fig)

def unif_varn_plot():
    global dat
    os.makedirs(fig_folder, exist_ok = True)
    with open(os.path.join("results_cube", "varn.bin"), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    results_vanilla = np.mean(np.abs(dat.results_vanilla - 8), axis = 0)
    results_kbary = np.mean(np.abs(dat.results_factored_ot - 8), axis = 0)
    results_kmeans = np.mean(np.abs(dat.results_kmeans - 8), axis = 0)
    (slope_v, intercept_v, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_vanilla))
    (slope_k, intercept_k, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_kbary))
    (slope_m, intercept_m, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_kmeans))

    fig = pl.figure(figsize = figsize)
    pl.loglog(dat.ns, results_vanilla,  "x-", c = new_colors[0], label = "OT")
    pl.loglog(dat.ns, results_kbary, "o-", c = new_colors[1], label = "FactoredOT")
    pl.loglog(dat.ns, results_kmeans, "^-", c = new_colors[4], label = "k-means & OT")
    pl.loglog(dat.ns, np.exp(intercept_v) * dat.ns**slope_v, c = new_colors[9])
    pl.loglog(dat.ns, np.exp(intercept_k) * dat.ns**slope_k, c = new_colors[3])
    pl.loglog(dat.ns, np.exp(intercept_m) * dat.ns**slope_m, c = new_colors[6])
    pl.xlabel(r"n")
    pl.ylabel(r"$|\hat W_2^2(\hat P, \hat Q) - W_2^2(P, Q)|$")
    pl.legend()
    pl.title("Varying $n$, $d = {}$, $k = {}$".format(dat.ds[0], dat.ks[0]))
    pl.tight_layout()
    pl.savefig(join(fig_folder,"uniform_varn.png"), dpi = dpi)
    print("(a)")
    print(fig.subplotpars.__dict__)
    pl.close(fig)

def annulus_vard_plot():
    global dat

    d = 30
    n = 1000

    os.makedirs(fig_folder, exist_ok = True)
    filename = "vard.bin"
    with open(os.path.join("results_annulus", filename), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    fig = pl.figure(figsize = figsize)

    pl.loglog(dat.ds, np.mean(np.abs(dat.results_vanilla - 4), axis = 0), "x-", label="OT", c = new_colors[0])
    pl.loglog(dat.ds, np.mean(np.abs(dat.results_factored_ot - 4), axis = 0), "o-", label = "FactoredOT", c = new_colors[1])
    pl.loglog(dat.ds, np.mean(np.abs(dat.results_kmeans - 4), axis = 0), "^-", label = "k-means & OT", c = new_colors[4])
    pl.xlabel("$d$")
    pl.ylabel(r"$|\hat W_2^2(\hat P_0, \hat P_1) - W_2^2(P_0, P_1)$")
    pl.title(r"Varying $d$, $k = {}$, n = 10d".format(dat.ks[0]))
    pl.tight_layout()
    pl.legend(loc = "upper left")
    pl.savefig(join(fig_folder, "annulus_vard.png"), dpi = dpi)
    pl.close(fig)

def annulus_varn_plot():
    global dat, fig, ax

    d = 30
    n = 1000

    os.makedirs(fig_folder, exist_ok = True)
    filename = "varn.bin"
    with open(os.path.join("results_annulus", filename), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    results_vanilla = np.mean(np.abs(dat.results_vanilla - 4), axis = 0)
    results_kbary = np.mean(np.abs(dat.results_factored_ot - 4), axis = 0)
    results_kmeans = np.mean(np.abs(dat.results_kmeans - 4), axis = 0)

    (slope_v, intercept_v, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_vanilla))
    (slope_k, intercept_k, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_kbary))
    (slope_m, intercept_m, _, _, _) = stats.linregress(np.log(dat.ns), np.log(results_kmeans))
    print(slope_k)
    print(slope_v)
    print(slope_m)

    fig = pl.figure(figsize = figsize)
    ax = pl.axes()
    pl.loglog(dat.ns, results_vanilla,  "x-", c = new_colors[0], label = "OT")
    pl.loglog(dat.ns, results_kbary, "o-", c = new_colors[1], label = "FactoredOT")
    pl.loglog(dat.ns, results_kmeans, "^-", c = new_colors[4], label = "k-means & OT")
    pl.loglog(dat.ns, np.exp(intercept_v) * dat.ns**slope_v, c = new_colors[9])
    pl.loglog(dat.ns, np.exp(intercept_k) * dat.ns**slope_k, c = new_colors[3])
    pl.loglog(dat.ns, np.exp(intercept_m) * dat.ns**slope_m, c = new_colors[6])
    pl.xlabel(r"n")
    pl.ylabel(r"$|\hat W_2^2(\hat P_0, \hat P_1) - W_2^2(P_0, P_1)|$")
    pl.legend()
    pl.title("Varying $n$, $d = {}$, $k = {}$".format(dat.ds[0], dat.ks[0]))
    pl.tight_layout()
    pl.savefig(join(fig_folder,"annulus_varn.png"), dpi = dpi)
    print(fig.subplotpars.__dict__)

def annulus_vark_plot():
    global dat

    d = 30
    n = 1000

    os.makedirs(fig_folder, exist_ok = True)
    filename = "vark.bin"
    with open(os.path.join("results_annulus", filename), "rb") as f:
        dat = pickle.load(f)
    dat = Namespace(**dat)

    fig = pl.figure(figsize = figsize)

    pl.plot(dat.ks, np.array([4]).repeat(len(dat.ks)), "-", label = "ground truth", c = new_colors[3])
    pl.plot(dat.ks, np.mean(dat.results_factored_ot, axis = 0), "o-", label = "FactoredOT", c = new_colors[1])
    pl.xlabel("$k$")
    pl.ylabel(r"$\hat W_2^2(\hat P_0, \hat P_1)$")
    pl.title(r"Varying $k$, $d = {}$, $n = {}$".format(dat.ds[0], dat.ns[0]))
    pl.legend(loc = "lower right")
    pl.tight_layout()
    pl.savefig(join(fig_folder, "annulus_vark.png"), dpi = dpi)
    pl.close(fig)

if __name__ == "__main__":
    unif_vark_plot()
    unif_varn_plot()
    unif_vard_plot()
    annulus_vark_plot()
    annulus_vard_plot()
    annulus_varn_plot()
