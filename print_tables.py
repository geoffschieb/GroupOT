import pickle
import numpy as np

# ftype = "caltech"
ftype = "bio"
# ftype = "pancreas"

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

if ftype == "pancreas":
    domains = range(4)

    for source in domains:
        for target in domains:
            if source != target:
                infile = "pancreas" + str(source) + "to" + str(target) + ".bin"

                print("-" * 30)
                print("Results for:")
                print("Source: {}".format(source))
                print("Target: {}".format(target))
                    
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
    # infile = "haem3.bin"
    infile = "haem_small_new.bin"

    print("-" * 30)
    print("Results for bio data:")
        
    with open(infile, "rb") as f:
        l = pickle.load(f)

    # for (k, v) in l["test"].items():
    #     print("{}:".format(k))
    #     if hasattr(v, "shape"):
    #         if len(v.shape) == 1:
    #             print(np.mean(v))
    #         else:
    #             # v = v[v != np.inf]
    #             v_cor = v[-1,:]
    #             v_cor = (v_cor[v_cor != np.inf])
    #             # print(v_cor)
    #             # print("{} +- {}".format(np.mean(v[-1,:], axis = 0), np.std(v[-1,:], axis = 0)))
    #             print("{} +-".format(np.mean(v_cor)))
    #     # if True:
    #     #     print(v)

    # print(" & & ".join(l["test"].keys()) + "\\\\")

    vals = []
    stds = []
    for v in l["test"].values():
        vals.append(np.mean(v[-1,:]))
        stds.append(np.std(v[-1,:]))

    outvals = []
    print(" & ".join(l["test"].keys()))
    for val in vals:
        outvals.append("{:.2f}".format(100 * val))
    print("Mean acc & " + " & ".join(outvals) + "\\\\")
    outvals = []
    for std in stds:
        outvals.append("{:.2f}".format(100 * std))
    print("Std & " + " & ".join(outvals) + "\\\\")

    # for (k, v) in l["test"].items():
    #     print("{}:".format(k))
    #     if hasattr(v, "shape"):
    #         if len(v.shape) == 1:
    #             print(np.mean(v))
    #         else:
    #             # v = v[v != np.inf]
    #             v_cor = v[-1,:]
    #             v_cor = (v_cor[v_cor != np.inf])
    #             # print(v_cor)
    #             # print("{} +- {}".format(np.mean(v[-1,:], axis = 0), np.std(v[-1,:], axis = 0)))
    #             print("{} +-".format(np.mean(v_cor)))
    #     if True:
    #         print(v)

    # for (k, v) in l["test"].items():
    #     print(30*'-')
    #     print(k)
    #     for errs in v:
    #         print(np.mean(errs))
