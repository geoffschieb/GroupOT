import pickle
import numpy as np

# ftype = "caltech"
ftype = "bio"

if ftype == "caltech":
    domain_names = ["amazon", "caltech10", "dslr", "webcam"]
    feature_names = ["GoogleNet1024", "CaffeNet4096", "surf"]

    for source_name in domain_names:
        for target_name in domain_names:
            if source_name != target_name:
                for feature_name in feature_names:
                    infile = "caltech_" + source_name + "_to_" + target_name + "_" + feature_name + ".bin"

                    print("-" * 30)
                    print("Results for:")
                    print("Source: {}".format(source_name))
                    print("Target: {}".format(target_name))
                    print("Features: {}".format(feature_name))
                        
                    with open(infile, "rb") as f:
                        l = pickle.load(f)

                    for (k, v) in l["test"].items():
                        print("{}:".format(k))
                        if len(v.shape) == 1:
                            print(np.mean(v))
                        else:
                            print(np.mean(v, axis = 1))
                        # print(v)
elif ftype == "bio":
    infile = "pancreas.bin"

    print("-" * 30)
    print("Results for bio data:")
        
    with open(infile, "rb") as f:
        l = pickle.load(f)

    for (k, v) in l["test"].items():
        print("{}:".format(k))
        if hasattr(v, "shape"):
            if len(v.shape) == 1:
                print(np.mean(v))
            else:
                print(np.mean(v, axis = 1))
        # if True:
        #     print(v)
