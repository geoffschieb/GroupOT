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
    # infile = "haem_small_new.bin"
    infile = "haem3_new_centered.bin"
    # infile = "haem3_new_uncentered.bin"

    print("-" * 30)
    print("Results for bio data:")
        
    with open(infile, "rb") as f:
        l = pickle.load(f)

    # ## Latex table
    name_dict = {
            "ot" : "OT",
            "ot_entr" : "OT-ER",
            "sa" : "SA",
            "ot_gl" : "OT-L1L2",
            "ot_kmeans" : "k-means OT",
            "ot_kbary" : "NAME",
            "noadj" : "NN",
            "tca" : "TCA",
            "mnn" : "MNN"
            }

    # order = ["ot_kbary", "mnn", "ot", "ot_entr", "ot_gl", "ot_kmeans", "sa", "tca", "noadj"]
    # order = ["sa", "ot"]
    order = ["mnn"]

    for i in range(3):
        vals = []
        stds = []
        print(30 * '-')
        for elem in order:
            v = l["test"][elem]
            ind = 6-i if v.shape[0] == 7 else 2-i
            vals.append(np.mean(v[ind,:]))
            stds.append(np.std(v[ind,:]))
        outvals = []
        print(" & ".join([r"\mc{" + name_dict[elem] + r"}" for elem in order]))
        for val in vals:
            outvals.append("{:.2f}".format(100 * val))
        print("Mean acc & " + " & ".join(outvals) + "\\\\")
        outvals = []
        for std in stds:
            outvals.append("{:.2f}".format(100 * std))
        print("Std & " + " & ".join(outvals) + "\\\\")

    ## All entries
    for (k, v) in l["test"].items():
        print(30*'-')
        print(k)
        print(v.shape)
        for errs in v:
            print(np.mean(errs))

    # Old
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

