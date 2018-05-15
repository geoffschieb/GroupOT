import rpy2.robjects as robj
import numpy as np

def convert_data():
    robj.r['load']("../MNN2017/Simulations/Sim.RData")
    labs_str = np.asarray(robj.r['clust1'])
    labt_str = np.asarray(robj.r['clust2i'])
    # print(labs_str)
    # labs = np.zeros_like(labs_str).astype(int)
    # labt = np.zeros_like(labs_str).astype(int)
    labs = np.zeros_like(labs_str)
    labt = np.zeros_like(labs_str)
    labs[labs_str == 'blue'] = 1
    labs[labs_str == 'brown1'] = 2
    labs[labs_str == 'gold2'] = 3
    labt[labt_str == 'blue'] = 1
    labt[labt_str == 'brown1'] = 2
    labt[labt_str == 'gold2'] = 3
    labs = labs.astype(int)
    labt = labt.astype(int)

    return (np.asarray(robj.r['B1']).T, np.asarray(robj.r['B2i']).T, labs, labt)

