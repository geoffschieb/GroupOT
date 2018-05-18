import OT_fixed_k as hot # hub-OT functions
import matplotlib.pyplot as plt
import numpy as np
import pickle
import time
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# sections to run
load_data = False #True #load original MNIST data (needed to refit models)
refit_models = False #True
make_error_plot = True
make_data_plot = False


np.random.seed(1)

num_samples = 500 # number of images in one training or testing batch


if load_data:
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data')




    # consider preprocessing, using z-scores








    num_clean_samples = 6000 # number of images loaded without noise, for all train/test runs
    num_noisy_samples = num_clean_samples


    clean_data = mnist.train.next_batch(num_clean_samples)
    future_noisy_data = mnist.train.next_batch(num_noisy_samples)

    dimensions = clean_data[0].shape[1]


    # Prepare noisy_data by adding Gaussian noise at different levels

    # alphas to run
    #alphas = 0.03*np.array(range(21))
    alphas = np.array([0.0, 0.3, 0.6])
    
    noisy_data = [((1-alpha)*future_noisy_data[0] + alpha* np.random.randn(num_noisy_samples, dimensions), future_noisy_data[1], alpha) for alpha in alphas]


    # Classify at each noise level using clean data

    samples_test = 1
    samples_train = samples_test
    entr_regs = np.array([10.0])**range(0,1)#(-3,5)
    gl_params = np.array([10.0])**range(-3,5)
    centroid_ks =  np.array([10, 20, 30, 40, 50, 60, 70, 80])
    nn_ks = np.array([1,3,5,10,20])

    
    estimators = {
        "ot_gl": {
            "function": "ot_gl",
            "parameter_ranges": [entr_regs, gl_params]
        },
        "ot": {
            "function": "ot",
            "parameter_ranges": []
        },
        "ot_entr": {
            "function": "ot_entr",
            "parameter_ranges": [entr_regs]
        },
        "ot_kmeans": {
            "function": "ot_kmeans",
            "parameter_ranges": [entr_regs, centroid_ks]
        },
        "ot_2kbary": {
            "function": "ot_2kbary",
            "parameter_ranges": [entr_regs, centroid_ks]
        },
        "ot_kbary": {
            "function": "ot_kbary",
            "parameter_ranges": [entr_regs, centroid_ks]
        },
        "noadj": {
            "function": "noadj",
            "parameter_ranges": []
        },
        "sa": {
            "function": "sa",
            "parameter_ranges": [centroid_ks]
        },
        "tca": {
            "function": "tca",
            "parameter_ranges": [centroid_ks]
        },
                                    #"coral": {
                                    #"function": "coral",
                                    #"parameter_ranges": []
        #}
    } 

    sim_params = {
        "entr_regs": entr_regs,
        "gl_params": gl_params,
        "centroid_ks": centroid_ks,
        "nn_ks": nn_ks,
        "samples_test": samples_test,
        "samples_train": samples_train,
        "estimators": estimators,
        "outfile": "mnist_results.bin"}

    if refit_models:
        for target in noisy_data:
            sim_params["outfile"] = "mnist_results_{:.2f}.bin".format(target[2])
            
            print("Noise level: alpha = {}".format(target[2]))
            def get_data(train, i):
                start_index = i*num_samples*2
                if train:
                    start_index = start_index + num_samples
                    
                    
                xs = clean_data[0][start_index:(start_index+num_samples)]
                xt = target[0][start_index:(start_index+num_samples)]
                labs = clean_data[1][start_index:(start_index+num_samples)]
                labt = target[1][start_index:(start_index+num_samples)]
                    
                    
                return (xs, xt, labs, labt)
            
            hot.test_domain_adaptation(sim_params, get_data)














if make_error_plot:
    


        

    # alphas to load

    #first plot version
    #alphas = np.array([0.0,0.05,0.1,0.2,0.3,0.4,0.5])
    
    #alphas = 0.05*np.array(range(13))
#    alphas = 0.03*np.array(range(21))
    alphas = np.array([0.0, 0.3, 0.6])
    nn_ks = np.array([1,3,5,10,20])

    # load results
    results = {}
    for alpha in alphas:
        file = open('mnist_results_{:.2f}.bin'.format(alpha),'rb')
        results[alpha] = pickle.load(file)
        file.close()


        # summarize
        results[alpha]['average_test'] = {}
        for key in results[alpha]['test'].keys():
            results[alpha]['average_test'][key] = results[alpha]["test"][key] 

        print('alpha = ' + str(alpha))
        print(results[alpha]['average_test'])








    
    # make plots

    # rearrange average test results into plottable vectors

    result_curves = {}
    for estimator in ['sa','ot_entr']:#[results[alphas[0]]['average_test'].keys():
        for kindex in range(len(nn_ks)):
            k = nn_ks[kindex]
            if (estimator == 'ot_kbary') | (estimator == 'ot_2kbary') :
                #bary_map is never any good
                #result_curves[estimator + '_bary_map ' + str(k) + 'nn'] = np.array([results[alpha]['average_test'][estimator][kindex+1] for alpha in alphas])
                result_curves[estimator + '_map_from_clusters ' + str(k) + 'nn'] = np.array([results[alpha]['average_test'][estimator][kindex+1+len(nn_ks)] for alpha in alphas])
            else:
                result_curves[estimator + ' ' + str(k) + 'nn'] = np.array([results[alpha]['average_test'][estimator][kindex] for alpha in alphas])
        



    plot_handles = []
    plot_labels = []
    for estimator in result_curves.keys():
        current_plot, =  plt.plot(alphas, result_curves[estimator], label = estimator)
        plot_handles.append(current_plot)
        plot_labels.append(estimator)


    #print(plot_handles)
    #print(plot_labels)
    
    plt.legend(plot_handles, plot_labels, loc = 'upper left', bbox_to_anchor = (0,1))
    plt.ylabel('Classification error')
    plt.xlabel('Noise level')
    plt.show()







if make_data_plot:

    alphas = np.array([0.0,0.3,0.5])

    if load_data:
        noisy_data = [((1-alpha)*future_noisy_data[0] + alpha* np.random.randn(num_noisy_samples, dimensions), future_noisy_data[1], alpha) for alpha in alphas]

        all_points = np.vstack([noisy_data[i][0][0:num_samples] for i in range(len(noisy_data))])
        all_points = np.vstack([clean_data[0][0:num_samples],all_points])
        point_classes = np.hstack([noisy_data[i][1][0:num_samples] for i in range(len(noisy_data))])
        point_classes = np.hstack([clean_data[1][0:num_samples],point_classes])

        label_outfile = open('mnist_labels.bin','wb')
        pickle.dump(point_classes, label_outfile)
        label_outfile.close()
    else:
        label_outfile = open('mnist_labels.bin','rb')
        point_classes = pickle.load(label_outfile)
        label_outfile.close()
        


        
    point_alphas = np.hstack([np.zeros(num_samples)] + [alpha*np.ones(num_samples) for alpha in alphas])


    # reduce to 50 dim before t-SNE
    t1 = time.time()
    pca = PCA(n_components = 50)
    post_pca_points = pca.fit_transform(all_points)

    t2 = time.time()
    print('PCA time: {:.2f}'.format(t2-t1))
    
    tsne = TSNE()
    tsne_points = tsne.fit_transform(post_pca_points)
    t3 = time.time()
    print('t-SNE time: {:.2f}'.format(t3-t2))

#    print(tsne_points.shape)

#    tsne_outfile = open('mnist_tsne.bin', 'rb')
#    pickle.dump(tsne_points, tsne_outfile)
#    tsne_points = pickle.load(tsne_outfile)
#    tsne_outfile.close()
    
                        
    plot_handles = []
    plot_labels = []

    colors = ['green', 'cyan','blue','red']

    fig, ax = plt.subplots()
    ax.set_color_cycle(colors)

    for i in range(len(alphas)+1):
       
        if i == 0:
            alpha = 0.0
        else:
            alpha =  alphas[i-1]
        # restrict to this noise level
        current_points = tsne_points[(i*num_samples):((i+1)*num_samples),:]
        for label in range(10):
            # select points with the right label
            current_points_label = current_points[point_classes[(i*num_samples):((i+1)*num_samples)] == label, :]
            for pt in current_points_label:
                plt.text(pt[0], pt[1], str(label), color = colors[i], fontsize = 10)
        current_plot =  plt.scatter(current_points[:,0], current_points[:,1], s = 1)
        plot_handles.append(current_plot)
        plot_labels.append('({})'.format(str(alpha)))


    plt.legend(plot_handles, plot_labels, loc = 'upper left', bbox_to_anchor = (0,1))
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.show()
