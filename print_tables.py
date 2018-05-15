import pickle
import numpy as np

infile =  "caltech_google_ama_to_cal.bin"
with open(infile, "rb") as f:
    l = pickle.load(f)

for (k, v) in l["test"]:
    print("{}:".format(k))
    print(np.mean(v, axis = 1))
