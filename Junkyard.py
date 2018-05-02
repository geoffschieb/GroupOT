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
