import numpy as np
import pandas as pd
from scipy.io import savemat

infile = "pancreas_data.txt"
outfile = "pancreas.mat"

# with open(infile, "r") as f:
#     bigmat = np.loadtxt(f, skiprows = 1)
df = pd.read_csv(infile, ",")
bigmat = df.as_matrix()

protocols = ["CELseq", "SmartSeq2"]
studies = ["GSE81076","GSE85241","GSE86473","E-MTAB-5061"]

labels = ["Alpha", "Beta", "Gamma", "Delta"]

lab_ind = np.hstack([np.argwhere(bigmat[:, 1] == l).ravel() for l in labels])
bigmat = bigmat[lab_ind, :]
all_lab = np.empty(bigmat.shape[0]).astype(int)
for (cind, c) in enumerate(labels):
    ind = np.argwhere(bigmat[:, 1] == c).ravel()
    all_lab[ind] = cind

# studies = np.unique(bigmat[:,3])
x = []
lab = []
for study in studies:
    ind_cur = np.argwhere(bigmat[:,3] == study).ravel()
    print(ind_cur)
    x.append(bigmat[ind_cur, 4:].astype(float))
    lab.append(all_lab[ind_cur].astype(int))
    
for l in lab:
    for c in np.unique(all_lab):
        print(np.mean(l == c))
    print()

outdict = {"x"+str(i): x[i] for i in range(len(x))}
outdict.update({"lab"+str(i): lab[i] for i in range(len(lab))})
savemat(outfile, outdict)
