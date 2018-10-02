import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import axes3d 
from sklearn.cluster import KMeans

def check_gradient():
    (loss, log) = ot.sinkhorn2(a, b, M, lambd, log = True)
    subgrad = lambd * np.log(log["u"])
    subgrad = subgrad.reshape((subgrad.shape[0],))
    subgrad -= np.mean(subgrad)
    eps = 1e-4
    grad = np.zeros(a.shape[0])
    for i in range(a.shape[0]):
        direction = np.zeros(a.shape[0])
        direction[i] = 1
        direction -= np.mean(direction)
        grad[i] = (ot.sinkhorn2(a + eps * direction, b, M, lambd) - loss)/eps

    print(a.shape)
    print(subgrad.shape)
    loss2 = ot.sinkhorn2(a - 0.1 * subgrad, b, M, lambd)
    # disturbed_subgrad = subgrad + 0.01 * np.random.normal(size = (subgrad.shape[0],))
    # disturbed_subgrad -= np.mean(disturbed_subgrad)
    # loss3 = ot.sinkhorn2(a - 0.01 * disturbed_subgrad, b, M, lambd)
    print(loss)
    print(loss2)
    # print(loss3)
    return (grad, subgrad)

def min_a(b, M, lambd):
    n = M.shape[0]
    a_tilde = np.ones(n)/n
    a_hat = a_tilde.copy()
    a_hat_old = a_hat.copy()
    converged = False
    t = 0.5
    tol = 1e-5
    its = 0
    gamma = 1e-2

    while not converged:
        # its += 1
        # print("Iteration: {}".format(its))
        # beta = (t+1)/2
        # a = (1 - 1/beta)*a_hat + 1/beta * a_tilde
        # (_, log) = ot.sinkhorn2(a, b, M, lambd, log = True)
        # u = log["u"].reshape(n)
        # alpha = lambd * np.log(u)
        # alpha -= np.mean(alpha)
        # # a_tilde *= u**(-t * beta * lambd)
        # a_tilde *= np.exp(-t * beta * alpha)
        # a_tilde /= np.sum(a_tilde)
        # a_hat_old = a_hat.copy()
        # a_hat = (1 - 1/beta) * a_hat + 1/beta * a_tilde
        # t += 1

        its += 1
        print("Iteration: {}".format(its))
        beta = (t+1)/2
        a = (1 - 1/beta)*a_hat + 1/beta * a_tilde
        (_, log) = ot.sinkhorn2(a, b, M, lambd, log = True)
        u = log["u"].reshape(n)
        alpha = lambd * np.log(u)
        alpha -= np.mean(alpha)
        # a_tilde *= u**(-t * beta * lambd)
        a_hat_old = a_hat.copy()
        a_hat -= gamma * alpha
        a_hat[a_hat < 0] = 0
        a_hat /= np.sum(a_hat)

        print(a_hat_old)
        print(a_hat)

        if np.linalg.norm(a_hat - a_hat_old) < tol:
            converged = True

    return a_hat

def barycenter_bregman(b1, b2, M1, M2, w1, w2, entr_reg,
        verbose = False,
        tol = 1e-7,
        max_iter = 500
        ):

    K1 = np.exp(-M1/entr_reg)
    K2 = np.exp(-M2/entr_reg)
    K1t = K1.T.copy()
    K2t = K2.T.copy()
    n = K1.shape[0]
    v1 = np.ones(K1.shape[1])
    v2 = np.ones(K2.shape[1])
    v1_old = v1.copy()
    v2_old = v2.copy()
    u1 = np.ones(K1.shape[0])
    u2 = np.ones(K2.shape[0])
    u1_old = u1.copy()
    u2_old = u2.copy()

    converged = False
    its = 0

    while not converged and its < max_iter: 
        its += 1
        if verbose:
            print("Barycenter weights iteration: {}".format(its))

        v1 = np.divide(b1, np.dot(K1t, u1))
        v2 = np.divide(b2, np.dot(K2t, u2))
        Kv1 = np.dot(K1, v1)
        Kv2 = np.dot(K2, v2)
        uKv1 = u1 * Kv1
        uKv2 = u2 * Kv2
        # print(uKv1)
        # print(uKv2)
        p = np.exp(np.log(uKv1) * w1 + np.log(uKv2) * w2)
        u1 = np.divide(u1 * p, uKv1)
        u2 = np.divide(u2 * p, uKv2)

#         Kv = np.dot(K, v)
#         p = np.dot(u, Kv)
#         u = np.divide(p, Kv)
#         v = np.divide(b, np.dot(Kt, u))

        # Kv = np.dot(K, v)
        # u = np.divide(a, Kv)
        # v = np.divide(b, np.dot(Kt, u))

        err = np.linalg.norm(u1 - u1_old)/np.linalg.norm(u1) + np.linalg.norm(v1 - v1_old) / np.linalg.norm(v1)
        err += np.linalg.norm(u2 - u2_old)/np.linalg.norm(u2) + np.linalg.norm(v2 - v2_old) / np.linalg.norm(v2)
        # print(err)
        # if its == 1:
        #     print(u1)
        # print(v1)
        u1_old = u1.copy()
        v1_old = v1.copy()
        u2_old = u2.copy()
        v2_old = v2.copy()
        if err < tol:
            converged = True

    gamma1 = np.dot(np.diag(u1), np.dot(K1, np.diag(v1)))
    gamma2 = np.dot(np.diag(u2), np.dot(K2, np.diag(v2)))
    # return (np.dot(gamma1, np.ones(K1.shape[1])), np.dot(gamma2, np.ones(K2.shape[1])))
    # return np.dot(gamma1, np.ones(K1.shape[1]))
    return (gamma1, gamma2)

def barycenter_bregman2(b_source, b_target, M_source_left, M_left_right, M_right_target, lambda_left, lambda_mid, lambda_right, entr_reg_final,
        verbose = False,
        tol = 1e-7,
        max_iter = 500,
        rebalance_thresh = 1e10,
        warm_start = False,
        swap_order = False
        ):

    entr_reg = 1000.0 if warm_start else entr_reg_final

    alpha_sl = np.zeros_like(M_source_left.shape[0])
    beta_sl = np.zeros_like(M_source_left.shape[1])
    alpha_lr = np.zeros_like(M_left_right.shape[0])
    beta_lr = np.zeros_like(M_left_right.shape[1])
    alpha_rt = np.zeros_like(M_right_target.shape[0])
    beta_rt = np.zeros_like(M_right_target.shape[1])

    K_source_left = np.exp(-(M_source_left - alpha_sl.reshape(-1, 1) - beta_sl.reshape(1, -1))/entr_reg)
    Kt_source_left = K_source_left.T.copy()
    K_left_right = np.exp(-(M_left_right - alpha_lr.reshape(-1, 1) - beta_lr.reshape(1, -1))/entr_reg)
    Kt_left_right = K_left_right.T.copy()
    K_right_target = np.exp(-(M_right_target - alpha_rt.reshape(-1, 1) - beta_rt.reshape(1, -1))/entr_reg)
    Kt_right_target = K_right_target.T.copy()

    u_sl = np.ones(K_source_left.shape[0])
    v_sl = np.ones(K_source_left.shape[1])
    u_lr = np.ones(K_left_right.shape[0])
    v_lr = np.ones(K_left_right.shape[1])
    u_rt = np.ones(K_right_target.shape[0])
    v_rt = np.ones(K_right_target.shape[1])

    alpha_sl = np.zeros_like(u_sl)
    beta_sl = np.zeros_like(v_sl)
    alpha_lr = np.zeros_like(u_lr)
    beta_lr = np.zeros_like(v_lr)
    alpha_rt = np.zeros_like(u_rt)
    beta_rt = np.zeros_like(v_rt)

    u_sl_old = u_sl.copy()
    v_sl_old = v_sl.copy()
    u_lr_old = u_lr.copy()
    v_lr_old = v_lr.copy()
    u_rt_old = u_rt.copy()
    v_rt_old = v_rt.copy()

    lambda_left_total = np.float(lambda_left + lambda_mid)
    lambda_right_total = np.float(lambda_mid + lambda_right)
    lambda_left_left = lambda_left/lambda_left_total
    lambda_left_mid = lambda_mid/lambda_left_total
    lambda_right_mid = lambda_mid/lambda_right_total
    lambda_right_right = lambda_right/lambda_right_total

    print(lambda_left_left, lambda_left_mid, lambda_right_mid, lambda_right_right)

    converged = False
    its = 0

    while not converged and its < max_iter: 
        its += 1
        if verbose:
            print("Barycenter weights iteration: {}".format(its))

        if max(np.max(np.abs(u_sl)), np.max(np.abs(v_sl))) > rebalance_thresh:
            print("Rebalance sl")
            alpha_sl += entr_reg * np.log(u_sl)
            beta_sl += entr_reg * np.log(v_sl)
            u_sl = np.ones(K_source_left.shape[0])
            v_sl = np.ones(K_source_left.shape[1])
            u_sl_old = u_sl.copy()
            v_sl_old = v_sl.copy()
            K_source_left = np.exp(-(M_source_left - alpha_sl.reshape(-1, 1) - beta_sl.reshape(1, -1))/entr_reg)
            Kt_source_left = K_source_left.T.copy()
        if max(np.max(np.abs(u_lr)), np.max(np.abs(v_lr))) > rebalance_thresh:
            print("Rebalance lr")
            alpha_lr += entr_reg * np.log(u_lr)
            beta_lr += entr_reg * np.log(v_lr)
            u_lr = np.ones(K_left_right.shape[0])
            v_lr = np.ones(K_left_right.shape[1])
            u_lr_old = u_lr.copy()
            v_lr_old = v_lr.copy()
            K_left_right = np.exp(-(M_left_right - alpha_lr.reshape(-1, 1) - beta_lr.reshape(1, -1))/entr_reg)
            Kt_left_right = K_left_right.T.copy()
        if max(np.max(np.abs(u_rt)), np.max(np.abs(v_rt))) > rebalance_thresh:
            print("Rebalance rt")
            alpha_rt += entr_reg * np.log(u_rt)
            beta_rt += entr_reg * np.log(v_rt)
            u_rt = np.ones(K_right_target.shape[0])
            v_rt = np.ones(K_right_target.shape[1])
            u_rt_old = u_rt.copy()
            v_rt_old = v_rt.copy()
            K_right_target = np.exp(-(M_right_target - alpha_rt.reshape(-1, 1) - beta_rt.reshape(1, -1))/entr_reg)
            Kt_right_target = K_right_target.T.copy()

        u_sl = np.divide(b_source, np.dot(K_source_left, v_sl))
        if swap_order:
            v_rt = np.divide(b_target, np.dot(Kt_right_target, u_rt))

        Ku_sl = np.dot(Kt_source_left, u_sl)
        Kv_lr = np.dot(K_left_right, v_lr)
        vKu_sl = v_sl * Ku_sl
        uKv_lr = u_lr * Kv_lr
        p1 = np.exp(lambda_left_left * np.log(vKu_sl) + lambda_left_mid * np.log(uKv_lr))
        v_sl = np.divide(v_sl * p1, vKu_sl)
        u_lr = np.divide(u_lr * p1, uKv_lr)
        # print(u_lr)
        # print(v_sl)

        Ku_lr = np.dot(Kt_left_right, u_lr)
        Kv_rt = np.dot(K_right_target, v_rt)
        vKu_lr = v_lr * Ku_lr
        uKv_rt = u_rt * Kv_rt
        p2 = np.exp(lambda_right_mid * np.log(vKu_lr) + lambda_right_right * np.log(uKv_rt))
        v_lr = np.divide(v_lr * p2, vKu_lr)
        u_rt = np.divide(u_rt * p2, uKv_rt)

        if not swap_order:
            v_rt = np.divide(b_target, np.dot(Kt_right_target, u_rt))

        err = np.linalg.norm(u_sl - u_sl_old)/np.linalg.norm(u_sl) + np.linalg.norm(v_sl - v_sl_old)/np.linalg.norm(v_sl)
        err += np.linalg.norm(u_lr - u_lr_old)/np.linalg.norm(u_lr) + np.linalg.norm(v_lr - v_lr_old)/np.linalg.norm(v_lr)
        err += np.linalg.norm(u_rt - u_rt_old)/np.linalg.norm(u_rt) + np.linalg.norm(v_rt - v_rt_old)/np.linalg.norm(v_rt)
        print(err)
        # if its == 1:
        #     print(u1)
        # print(v1)

        u_sl_old = u_sl.copy()
        v_sl_old = v_sl.copy()
        u_lr_old = u_lr.copy()
        v_lr_old = v_lr.copy()
        u_rt_old = u_rt.copy()
        v_rt_old = v_rt.copy()

        if err < tol:
            if not warm_start or abs(entr_reg - entr_reg_final) < 1e-12:
                converged = True
            else:
                alpha_sl += entr_reg * np.log(u_sl)
                beta_sl += entr_reg * np.log(v_sl)
                u_sl = np.ones(K_source_left.shape[0])
                v_sl = np.ones(K_source_left.shape[1])
                u_sl_old = u_sl.copy()
                v_sl_old = v_sl.copy()
                alpha_lr += entr_reg * np.log(u_lr)
                beta_lr += entr_reg * np.log(v_lr)
                u_lr = np.ones(K_left_right.shape[0])
                v_lr = np.ones(K_left_right.shape[1])
                u_lr_old = u_lr.copy()
                v_lr_old = v_lr.copy()
                alpha_rt += entr_reg * np.log(u_rt)
                beta_rt += entr_reg * np.log(v_rt)
                u_rt = np.ones(K_right_target.shape[0])
                v_rt = np.ones(K_right_target.shape[1])
                u_rt_old = u_rt.copy()
                v_rt_old = v_rt.copy()

                entr_reg = max(entr_reg/2, entr_reg_final)
                print("New regularization parameter: {}".format(entr_reg))
                K_source_left = np.exp(-(M_source_left - alpha_sl.reshape(-1, 1) - beta_sl.reshape(1, -1))/entr_reg)
                Kt_source_left = K_source_left.T.copy()
                K_left_right = np.exp(-(M_left_right - alpha_lr.reshape(-1, 1) - beta_lr.reshape(1, -1))/entr_reg)
                Kt_left_right = K_left_right.T.copy()
                K_right_target = np.exp(-(M_right_target - alpha_rt.reshape(-1, 1) - beta_rt.reshape(1, -1))/entr_reg)
                Kt_right_target = K_right_target.T.copy()

    gamma_source_left = u_sl.reshape(-1, 1) * K_source_left * v_sl.reshape(1, -1)
    gamma_left_right = u_lr.reshape(-1, 1) * K_left_right * v_lr.reshape(1, -1)
    gamma_right_target = u_rt.reshape(-1, 1) * K_right_target * v_rt.reshape(1, -1)
    return (gamma_source_left, gamma_left_right, gamma_right_target)

def barycenter_free(b1, b2, xs1, xs2, w1, w2, entr_reg, k,
        tol = 1e-5,
        max_iter = 100,
        verbose = False
        ):

    d = xs1.shape[1]
    ys = np.random.normal(size = (k, d))
    cost_old = np.float("Inf")
    its = 0
    converged = False

    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Barycenter points iteration: {}".format(its))

        if its > 1:
            ys = (w1 * np.matmul(gamma1, xs1) + w2 * np.matmul(gamma2, xs2))/(np.dot((w1 * np.sum(gamma1, axis = 1) + w2 * np.sum(gamma2, axis = 1)).reshape((k, 1)), np.ones((1, d))))

        M1 = ot.dist(ys, xs1)
        M2 = ot.dist(ys, xs2)
        (gamma1, gamma2) = barycenter_bregman(b1, b2, M1, M2, w1, w2, entr_reg, max_iter = 500)
        cost = (w1 * np.sum(gamma1 * M1) + w2 * np.sum(gamma2 * M2)) + entr_reg * (w1 * stats.entropy(gamma1.reshape((-1,1))) + w2 * stats.entropy(gamma2.reshape((-1,1))))
        # TODO: Calculate the entropy safely
        # cost = w1 * np.sum(gamma1 * M1) + w2 * np.sum(gamma2 * M2)

        err = abs(cost - cost_old)/max(abs(cost), 1e-12)
        cost_old = cost.copy()
        if verbose:
            print("Relative change in points iteration: {}".format(err))
        if err < tol:
            converged = True

    return (ys, np.dot(gamma1, np.ones(gamma1.shape[1])), cost, gamma1, gamma2)

def cluster_ot(b1, b2, xs1, xs2, k1, k2,
        source_reg, target_reg, entr_reg,
        tol = 1e-4, max_iter = 1000,
        verbose = True):

    converged = False

    d = xs1.shape[1]
    zs1  = np.random.normal(size = (k1, d))
    zs2  = np.random.normal(size = (k2, d))
    a1 = np.ones(k1)/k1
    a2 = np.ones(k2)/k2

    w_left = 1 + source_reg
    w_right = 1 + target_reg
    w_1 = source_reg/w_left
    w_mid_1 = 1/w_left
    w_mid_2 = 1/w_right
    w_2 = target_reg/w_right

    its = 0

    cost_old = np.float("Inf")
    
    while not converged and its < max_iter:
        its += 1
        if verbose:
            print("Alternating barycenter iteration: {}".format(its))
        # Alternate between computing barycenters
        (zs1, a1, cost1, mid_left_source, mid_left_mid_right) = barycenter_free(b1, a2, xs1, zs2, w_1, w_mid_1, entr_reg, k1, max_iter = 10)
        (zs2, a2, cost2, mid_right_mid_left, mid_right_target) = barycenter_free(a1, b2, zs1, xs2, w_mid_2, w_2, entr_reg, k2, max_iter = 10)
        cost = w_left * cost1 + w_right * cost2

        err = abs(cost - cost_old)/max(abs(cost), 1e-12)
        if verbose:
            print("Alternating barycenter relative error: {}".format(err))
        cost_old = cost.copy()
        if err < tol:
            converged = True

    return (zs1, zs2, a1, a2, mid_left_source, mid_right_mid_left, mid_right_target)

def test_split_data_uniform_vark():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ks = np.hstack([range(10,21,2)])
    ks = [50]
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 30
    n = 10*d
    samples = 1
    middle_param = 0.1
    entropy = 0.5
    results_vanilla = np.empty((samples, len(ks)))
    results_kbary = np.empty((samples, len(ks)))
    results_cluster = np.empty((samples, len(ks)))
    results_cluster_nocor = np.empty((samples, len(ks)))

    for (k_ind, k) in enumerate(ks):
        print()
        print("k = {}".format(k))
        print()
        for sample in range(samples):
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, k_ind] = cost
            print("Sinkhorn: {}".format(cost))

            xs = samples_source
            xt = samples_target

            (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            results_kbary[sample, k_ind] = bary_cost
            print("Barycenter cost: {}".format(bary_cost))

            (zs1, zs2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, k, k, [1.0, middle_param, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf], inner_tol = 1e-12, tol = 1e-5)
            cluster_cost = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = True)
            cluster_cost_no_correction = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = False)
            print("Double cluster cost: {}".format(cluster_cost))
            print("Double cluster cost, no correction: {}".format(cluster_cost_no_correction))

    print(results_vanilla)
    print(results_kbary)

    with open(os.path.join("uniform_results", "vark_trial.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "n": n,
            "ks": ks,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary
            }, f)

def test_split_data_uniform_varn():
    global ds, results_vanilla, results_kbary

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 30
    ns = (np.array([10.0])**np.linspace(1.7, 3, 20)).astype(int)
    samples = 20
    k = 10
    results_vanilla = np.empty((samples, len(ns)))
    results_kbary = np.empty((samples, len(ns)))
    results_kmeans = np.empty((samples, len(ns)))

    for (n_ind, n) in enumerate(ns):
        for sample in range(samples):
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            # cost = ot.emd2(b1, b2, ot.dist(samples_source, samples_target), 1)
            # results_vanilla[sample, n_ind] = cost
            # print("Sinkhorn: {}".format(cost))

            xs = samples_source
            xt = samples_target

            # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 0.1, verbose = True,
            #         warm_start = True, relax_outside = [np.inf, np.inf],
            #         tol = 1e-4,
            #         inner_tol = 1e-5,
            #         max_iter = 500)
            # # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            # bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            # results_kbary[sample, n_ind] = bary_cost
            # print("Barycenter cost: {}".format(bary_cost))

            gammas_kmeans = kmeans_transport(xs, xt, k)
            kmeans_cost = estimate_w2_cluster(xs, xt, gammas_kmeans)
            results_kmeans[sample, n_ind] = kmeans_cost
            print("Kmeans cost: {}".format(kmeans_cost))



    print(results_vanilla)
    print(results_kbary)
    print(results_kmeans)

    # with open(os.path.join("uniform_results", "varn6.bin"), "wb") as f:
    #     pickle.dump({
    #         "d": d,
    #         "ns": ns,
    #         "k": k,
    #         "results_vanilla": results_vanilla,
    #         "results_kbary": results_kbary
    #         }, f)

    with open(os.path.join("uniform_results", "varn_kmeans.bin"), "wb") as f:
        pickle.dump({
            "d": d,
            "ns": ns,
            "k": k,
            "results_kmeans": results_kmeans,
            }, f)

def test_split_data_uniform_visual():
    global ds, results_vanilla, results_kbary
    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    # ks = np.hstack([range(1,11), range(12,31,2), range(34, 71, 4), range(70, 10, 101)]).astype(int)
    # ds = (np.array([2])**np.linspace(2, 8, 15)).astype(int)
    d = 30
    # ns = (np.array([10.0])**np.linspace(1.7, 3, 20)).astype(int)
    n = 100
    k = 4
    figsize = (6,4)

    samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
    samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)

    emb_dat = decomposition.PCA(n_components=2).fit_transform(np.vstack((samples_source,samples_target)))
    xs_emb = emb_dat[0:samples_source.shape[0],:]
    xt_emb = emb_dat[samples_source.shape[0]:,:]
    pl.figure(figsize=figsize)
    pl.plot(*xs_emb.T, '+b', label='Source samples')
    pl.plot(*xt_emb.T, 'xr', label='Target samples')

    b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
    b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
    # cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
    M = ot.dist(samples_source, samples_target)
    gamma_ot = ot.sinkhorn(b1, b2, M, 1)
    # ot.plot.plot2D_samples_mat(xs_emb, xt_emb, gamma_ot, c=[.5, .5, 1])
    cost = np.sum(gamma_ot * M)
    print("Sinkhorn: {}".format(cost))
    # pl.show()
    # pl.savefig(os.path.join("Figures","Hypercube_OT.png"), dpi = 300)

    xs = samples_source
    xt = samples_target

    (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], 1, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
    # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
    bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
    print("Barycenter cost: {}".format(bary_cost))
    newsource = map_from_clusters(xs_emb, xt_emb, gammas)
    pl.figure(figsize=figsize)
    pl.plot(*xs_emb.T, '+b', label='Source samples')
    pl.plot(*xt_emb.T, 'xr', label='Target samples')
    plot_transport_map(xt_emb, newsource)
    pl.savefig(os.path.join("Figures","Hypercube_kOT.png"), dpi = 300)
    pl.close()

def test_split_data_uniform(ks, ds, ns, middle_params, entropies, samples, prefix, filename):

    def transport_map(x, length = 1):
        direction = np.zeros_like(x)
        direction[0:2] = np.sign(x[0:2])
        return x + length * direction

    results_vanilla = np.empty((samples, len(ks)))
    results_kbary = np.empty((samples, len(ks)))
    results_cluster_ot = np.empty((samples, len(ds)))
    results_cluster_ot_nocor = np.empty((samples, len(ds)))
    results_cluster_ot_map = np.empty((samples, len(ds)))
    results_cluster_ot_map_nocor = np.empty((samples, len(ds)))
    results_cluster_ot_fixed = np.empty((samples, len(ds)))
    results_cluster_ot_fixed_nocor = np.empty((samples, len(ds)))

    for (ind, k) in enumerate(ks):
        d = ds[ind]
        n = ns[ind]
        entropy = entropies[ind]
        middle_param = middle_params[ind]
        print()
        print("k = {}, d = {}, n = {}, entropy = {}, middle_param = {}".format(k, d, n, entropy, middle_param))
        print()
        for sample in range(samples):
            samples_source = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.random.uniform(low=-1, high=1, size=(n, d))
            samples_target = np.apply_along_axis(lambda x: transport_map(x, 2), 1, samples_target)
            
            b1 = np.ones(samples_target.shape[0])/samples_target.shape[0]
            b2 = np.ones(samples_source.shape[0])/samples_source.shape[0]
            cost = ot.sinkhorn2(b1, b2, ot.dist(samples_source, samples_target), 1)
            results_vanilla[sample, ind] = cost
            print("Sinkhorn: {}".format(cost))

            xs = samples_source
            xt = samples_target

            (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, k, [1.0, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf])
            # # (zs, gammas) = kbarycenter(b1, b2, samples_source, samples_target, 16, [1.0, 1.0], 1, verbose = True, warm_start = True)
            bary_cost = estimate_w2_cluster(samples_source, samples_target, gammas)
            results_kbary[sample, ind] = bary_cost
            print("Barycenter cost: {}".format(bary_cost))

            (zs1, zs2, gammas) = cluster_ot(b1, b2, samples_source, samples_target, k, k, [1.0, middle_param, 1.0], entropy, verbose = True, warm_start = True, relax_outside = [np.inf, np.inf], inner_tol = 1e-12, tol = 1e-5)
            cluster_cost = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = True)
            cluster_cost_no_correction = estimate_w2_middle(samples_source, samples_target, zs1, zs2, gammas, bias_correction = False)
            print("Double cluster cost: {}".format(cluster_cost))
            print("Double cluster cost, no correction: {}".format(cluster_cost_no_correction))

    print(results_vanilla)
    print(results_kbary)

    with open(os.path.join(prefix, filename), "wb") as f:
        pickle.dump({
            "ds": ds,
            "ns": ns,
            "ks": ks,
            "entropies": entropies,
            "middle_params": middle_params,
            "results_vanilla": results_vanilla,
            "results_kbary": results_kbary
            }, f)
