import OT_fixed_k as hot # hub-OT functions
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data')

np.random.seed(1)





num_clean_samples = 6000 # number of images in dataset without noise
num_noisy_samples = num_clean_samples

clean_data = mnist.train.next_batch(num_clean_samples)
future_noisy_data = mnist.train.next_batch(num_noisy_samples)

dimensions = clean_data[0].shape[1]


# Prepare noisy_data by adding Gaussian noise at different levels

alphas = np.linspace(0,1,6)

noisy_data = [((1-alpha)*future_noisy_data[0] + alpha* np.random.randn(num_noisy_samples, dimensions), future_noisy_data[1], alpha) for alpha in alphas]


# Classify at each noise level using clean data

samples_test = 5
samples_train = samples_test
entr_regs = np.array([10.0])**(-3,5)
gl_params = np.array([10.0])**(-3,5)
ks =  np.array([2])**range(1,8)

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
                                    "parameter_ranges": [entr_regs, ks]
                                    },
                "ot_2kbary": {
                                    "function": "ot_2kbary",
                                    "parameter_ranges": [entr_regs, ks]
                                    },
                "ot_kbary": {
                                    "function": "ot_kbary",
                                    "parameter_ranges": [entr_regs, ks]
                                },
                "noadj": {
                                    "function": "noadj",
                                    "parameter_ranges": []
                                    },
                "sa": {
                                    "function": "sa",
                                    "parameter_ranges": [ks]
                                    },
                "tca": {
                                    "function": "tca",
                                    "parameter_ranges": [ks]
                                    },
                "coral": {
                                    "function": "coral",
                                    "parameter_ranges": []
                                    }
                } 

sim_params = {
    "entr_regs": entr_regs,
    "gl_params": gl_params,
    "ks": ks,
    "samples_test": samples_test,
    "samples_train": samples_train,
    "estimators": estimators,
    "outfile": "mnist_results.bin"}


for target in noisy_data:
    sim_params["outfile"] = "mnist_results_" + str(target[2]) + ".bin"

    print("Noise level: alpha = {}".format(target[2]))
    def get_data(train, i):
        num_samples = 500
        start_index = i*num_samples*2
        if train:
            start_index = start_index + num_samples


        xs = clean_data[0][start_index:(start_index+num_samples)]
        xt = target[0][start_index:(start_index+num_samples)]
        labs = clean_data[1][start_index:(start_index+num_samples)]
        labt = target[1][start_index:(start_index+num_samples)]


        return (xs, xt, labs, labt)

    hot.test_domain_adaptation(sim_params, get_data)
