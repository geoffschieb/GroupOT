import numpy as np
from scipy.io import loadmat
from argparse import Namespace

saved_data = loadmat("bio_diag.mat")
dat = Namespace(**saved_data)

# Contains (prefix with dat.)
# xs : source data
# xt : target data
# labs : source labels
# labt: target labels
# zs1: cluster points left
# zs2: cluster points right
# gammas: length three list with the transports: source to cluster, cluster to cluster, cluster to target
# gamma_ot: vanilla optimal transport map
# gamma_total: source to target transport map through clusters

