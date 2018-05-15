import pickle
import numpy as np

infile =  "caltech_google_ama_to_cal.bin"
with open(infile, "rb") as f:
    l = pickle.load(f)

for (k, v) in l["test"].items():
    print("{}:".format(k))
    if len(v.shape) == 1:
        print(np.mean(v))
    else:
        print(np.mean(v, axis = 1))
