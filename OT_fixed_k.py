import numpy as np
import scipy.stats as stats
import matplotlib
import ot
import scipy as sp
import matplotlib.pylab as pl
from mpl_toolkits.mplot3d import Axes3D 
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

def barycenter_bregman(b1, b2, M1, M2, lambd):
    tol = 1e-7

    K1 = np.exp(-M1/lambd)
    K2 = np.exp(-M2/lambd)
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

    while not converged and its < 500: 
        its += 1
        # print("Iteration: {}".format(its))

        v1 = np.divide(b1, np.dot(K1t, u1))
        v2 = np.divide(b2, np.dot(K2t, u2))
        Kv1 = np.dot(K1, v1)
        Kv2 = np.dot(K2, v2)
        uKv1 = u1 * Kv1
        uKv2 = u2 * Kv2
        p = np.exp(np.log(uKv1) * 1/2 + np.log(uKv2) * 1/2)
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
        # print(u1)
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

def barycenter_free(b1, b2, xs1, xs2, lambd, k):
    tol = 1e-5

    d = xs1.shape[1]
    ys = np.random.normal(size = (k, d))
    cost_old = 0
    its = 0
    converged = False

    while not converged:
        its += 1
        print("Outer iteration: {}".format(its))

        M1 = ot.dist(ys, xs1)
        M2 = ot.dist(ys, xs2)
        (gamma1, gamma2) = barycenter_bregman(b1, b2, M1, M2, lambd)
        # cost = 0.5 * (np.sum(gamma1 * M1) + np.sum(gamma2 * M2)) - lambd * (np.sum(gamma1 * np.log(gamma1)) + np.sum(gamma2 * np.log(gamma2)))
        # TODO: Calculate the entropy safely
        cost = 0.5 * (np.sum(gamma1 * M1) + np.sum(gamma2 * M2))
        ys = (np.matmul(gamma1, xs1) + np.matmul(gamma2, xs2))/(np.dot((np.sum(gamma1, axis = 1) + np.sum(gamma2, axis = 1)).reshape((k, 1)), np.ones((1, d))))

        err = abs(cost - cost_old)
        cost_old = cost.copy()
        print(err)
        if err < tol:
            converged = True

    return (ys, np.dot(gamma1, np.ones(gamma1.shape[1])))


# Barycenter histogram test

# lambd = 1e-2
# # n = 20
# # xs = np.linspace(0, 10, n)
# # xs = xs.reshape((xs.shape[0], 1))
# # ys = xs.copy()
# xm = np.vstack((xm, np.array([0, -10])))
# M1 = ot.dist(xs, xm)
# M1 /= M1.max()
# M2 = ot.dist(xt, xm)
# M2 /= M2.max()
# n1 = xs.shape[0]
# n2 = xt.shape[0]
# b1 = np.ones(n1)/n1
# b2 = np.ones(n2)/n2
# (gamma1, gamma2) = barycenter_bregman(b1, b2, M1.T, M2.T, lambd)
# # print(ot.sinkhorn2(a, b, M, lambd))
# x_combined = np.vstack((xs, xt, xm))
# b_combined = np.zeros((x_combined.shape[0], 2))
# b_combined[0:b1.shape[0], 0] = b1
# b_combined[b1.shape[0]:(b1.shape[0]+b2.shape[0]), 1] = b2
# b_combined += 1e-5
# b_combined[:,0] / np.sum(b_combined[:,0])
# b_combined[:,1] / np.sum(b_combined[:,1])
# M = ot.dist(x_combined, x_combined)
# a_ot = ot.bregman.barycenter(b_combined, M, lambd)

# Barycenter fixed point test

lambd = 0.1
cov1 = np.array([[1, 0], [0, 1]])
cov2 = np.array([[1, 0], [0, 1]])
mu1 = np.zeros(2)
mu2 = np.zeros(2)
n = 400
k = 100
xs1 = ot.datasets.get_2D_samples_gauss(n, mu1, cov1)
xs2 = ot.datasets.get_2D_samples_gauss(n, mu2, cov2)
xs1 /= np.dot(np.linalg.norm(xs1, axis = 1).reshape(n, 1), np.ones((1, 2)))
xs2 /= np.dot(np.linalg.norm(xs2, axis = 1).reshape(n, 1), np.ones((1, 2)))
xs1[:,0] *= 5
xs2[:,1] *= 5
b1 = np.ones(n)/n
b2 = b1.copy()
(ys, weights) = barycenter_free(b1, b2, xs1, xs2, lambd, k)
pl.plot(xs1[:,0], xs1[:,1], 'xb')
pl.plot(xs2[:,0], xs2[:,1], 'xr')
pl.plot(ys[:,0], ys[:,1], 'xg')
pl.show()
