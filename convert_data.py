import numpy as np
import pandas as pd
from scipy.io import savemat

infile = "MNN_haem_data.txt"
outfile = "MNN_haem_data.mat"

# with open(infile, "r") as f:
#     bigmat = np.loadtxt(f, skiprows = 1)
df = pd.read_csv(infile, "\t")
bigmat = df.as_matrix()

lab = np.empty(bigmat.shape[0]).astype(int)
for (cind, c) in enumerate(np.unique(bigmat[:, 0])):
    ind = np.argwhere(bigmat[:, 0] == c).ravel()
    lab[ind] = cind

inds = np.argwhere(bigmat[:, 1] == 1).ravel()
indt = np.argwhere(bigmat[:, 1] == 2).ravel()
xs = bigmat[inds, 2:].astype(float)
xt = bigmat[indt, 2:].astype(float)
labs = lab[inds].astype(int)
labt = lab[indt].astype(int)

for c in range(3):
    print(np.mean(labs == c))

print()
for c in range(3):
    print(np.mean(labt == c))

savemat(outfile,
        {
            "xs": xs,
            "xt": xt,
            "labs": labs,
            "labt": labt
            })
