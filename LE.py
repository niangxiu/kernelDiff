# the no-propagate algorithm for finite T, section 3.1 in paper
from __future__ import division
import shutil
import sys
import os
import time
import pickle
import itertools
import numpy as np
from numpy import newaxis, sin, cos, pi, tanh, cosh
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multiprocessing import Pool, current_process
from pdb import set_trace
from misc import nanarray

plt.rc('axes', labelsize='xx-large',  labelpad=12)
plt.rc('xtick', labelsize='xx-large')
plt.rc('ytick', labelsize='xx-large')
plt.rc('legend', fontsize='xx-large')
plt.rc('font', family='sans-serif')
plt.rc('axes', titlesize='xx-large')

n_repeat = 10
n_thread = min(5, os.cpu_count()-1)
n_layer = 50
n_dim = 9

J = 4*np.array([[-0.13516168, -0.29783588, -0.08243096,  0.41499251, -0.12447637,
        -0.32565383,  0.37992217, -0.12526514,  0.48796698],
       [-0.39935415, -0.3879358 , -0.3633098 ,  0.15289905,  0.4803628 ,
         0.14838584, -0.03952375, -0.28411444, -0.31846027],
       [-0.14859367, -0.16202381, -0.32922162, -0.36403019, -0.20393094,
        -0.23715312, -0.36719849, -0.01883415, -0.09450373],
       [-0.19575303, -0.06414615,  0.21856889,  0.49696407,  0.01734091,
         0.2173913 , -0.19816477, -0.10919189,  0.27635875],
       [ 0.2001853 , -0.32060724, -0.12938035, -0.25272391,  0.37228039,
         0.37210707, -0.41365685, -0.11224939,  0.05360169],
       [-0.44369768,  0.00692191, -0.34641336, -0.07080262,  0.10880086,
         0.31721268,  0.15363462,  0.00152954, -0.00597713],
       [-0.04479272, -0.07330322, -0.18167448,  0.13342434, -0.20624915,
        -0.39450329, -0.35140716,  0.01846531, -0.46114955],
       [ 0.15950353,  0.21597017,  0.1835388 ,  0.23963441, -0.01410735,
         0.00963944,  0.27477256,  0.30589367, -0.06963555],
       [ 0.29566231, -0.48732178, -0.09285783,  0.00191314,  0.31114643,
        -0.08012442,  0.1069531 ,  0.01424637, -0.31959618]])


def nop(ga=3, sig=0.5, L=93):
    np.random.seed()

    def ffga(x, y):
        xnew = J @ tanh(x + ga) + y
        fga = J @ (1 / cosh(x+ga)**2)
        return xnew, fga

    def fx(x):
        return J / cosh(x+ga)**2

    def phi_func(x):
        return x.sum()
    
    # return dp/p
    if sig == 0:
        def dpp(y): 
            return np.zeros(n_dim)
    else:
        def dpp(y): 
            return (-y) / sig**2

    phiT, S = nanarray([2, L])
    xT = nanarray([L, n_dim])
    LE = np.zeros([L, n_dim],dtype=complex)
    for l in range(L):
        x = nanarray([n_layer+1, n_dim])
        I = nanarray([n_layer+1,])
        x[0] = np.random.normal(scale=1, size=n_dim)

        D = np.eye(n_dim)
        for m in range(n_layer):
            y = np.random.normal(scale=sig, size=n_dim)
            x[m+1], fga = ffga(x[m], y)
            I[m] = fga @ dpp(y)
            D = fx(x[m]) @ D
        
        xT[l] = x[-1]
        phiT[l] = phi_func(xT[l])
        S[l] = - I[:-1].sum()
        LE[l], _ = np.linalg.eig(D)

    phiTavg = phiT.mean()
    phiT_central = phiT - phiTavg # centralize phiT
    grad = np.mean(S * phiT_central)
    print("{: .2f}, {:9d}, {: .2e}, {: .2e}".format(ga, L, phiTavg, grad))
    print("average max LE", np.abs(LE).max(axis=1).mean())

    set_trace()
    return phiTavg, grad

nop(0.01)


def change_ga():
    NN = 10 # number of steps in parameters
    galeft = -1
    garight = 1
    A = (galeft-garight)/(NN-1)/2.5 # step size in the plot
    gas = np.linspace(galeft,garight,NN)
    phiavgs = nanarray(NN)
    phiavgs_sig0 = nanarray(NN)
    grads = nanarray(NN)
    try:
        gas, phiavgs, grads, phiavgs_sig0 = pickle.load(open("change_ga.p", "rb"))
    except FileNotFoundError:
        for i, ga in enumerate(gas):
            phiavgs[i], grads[i] = nop(ga)
            phiavgs_sig0[i], _ = nop(ga,sig=0)
        pickle.dump((gas, phiavgs, grads, phiavgs_sig0), open("change_ga.p", "wb"))
    plt.figure(figsize=[11,5])
    plt.plot(gas, phiavgs, 'k.', markersize=6)
    plt.plot(gas, phiavgs_sig0, 'r-', markersize=6)
    for ga, phiavg, grad in zip(gas, phiavgs, grads):
        plt.plot([ga-A, ga+A], [phiavg-grad*A, phiavg+grad*A], color='grey', linestyle='-')
    plt.ylabel('$\Phi_{avg} $')
    plt.xlabel('$\gamma$')
    # plt.ylim(0.455,0.53)
    plt.tight_layout()
    plt.savefig("change_ga.png")
    plt.close()


def wrap_L(L): return nop(L=L)

def change_L():
    # gradients for different trajectory length L
    Ls = np.array([1, 2, 5, 1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3], dtype=int) * 1000
    arguments = [(L,) for L in np.repeat(Ls, n_repeat)]
    phiavgs, grads = nanarray([2, Ls.shape[0], n_thread])
    try:
        phiavgs, grads, Ls = pickle.load( open("change_L.p", "rb"))
    except FileNotFoundError:
        if n_thread == 1:
            results = [wrap_L(*arguments[0])]
        else:
            with Pool(processes=n_thread) as pool:
                results = pool.starmap(wrap_L, arguments)
        phiavgs, grads = zip(*results)
        pickle.dump((phiavgs, grads, Ls), open("change_L.p", "wb"))

    plt.semilogx(arguments, grads, 'k.')
    plt.xlabel('$L$')
    plt.ylabel('$\delta\Phi_{avg}$')
    plt.tight_layout()
    plt.savefig('L_grad.png')
    plt.close()

    grads = np.array(grads).reshape(Ls.shape[0], -1)
    plt.loglog(Ls, np.std(grads, axis=1), 'k.')
    x = np.array([Ls[0], Ls[-1]])
    plt.loglog(x, x**-0.5, 'k--')
    plt.xlabel('$L$')
    plt.ylabel('std $\delta\Phi_{avg}$')
    plt.tight_layout()
    plt.savefig('L_std.png')
    plt.close()


def wrap_W(W): return nop(W=W, L=100000)

def change_W():
    Ws = np.arange(10)
    arguments = [(W,) for W in np.repeat(Ws, n_repeat)]
    phiavgs, grads = nanarray([2, Ws.shape[0], n_thread])
   
    try:
        phiavgs, grads, Ws = pickle.load( open("change_W.p", "rb"))
    except FileNotFoundError:
        if n_thread == 1:
            results = [wrap_W(*arguments[0])]
        else:
            with Pool(processes=n_thread) as pool:
                results = pool.starmap(wrap_W, arguments)
        phiavgs, grads = zip(*results)
        pickle.dump((phiavgs, grads, Ws), open("change_W.p", "wb"))

    plt.plot(arguments, grads, 'k.')
    plt.ylabel('$\delta\Phi{avg}$')
    plt.xlabel('$W$')
    plt.tight_layout()
    plt.savefig('W_grad.png')  
    plt.close()


def change_W_std():
    # standard deviation to different W
    Ws = np.array([1e1, 2e1, 5e1, 1e2, 2e2, 5e2, 1e3, 2e3, 5e3], dtype=int) 
    arguments = [(W,) for W in np.repeat(Ws, n_repeat)]
    phiavgs, grads = nanarray([2, Ws.shape[0], n_thread])
   
    try:
        phiavgs, grads, Ws = pickle.load( open("change_W_std.p", "rb"))
    except FileNotFoundError:
        if n_thread == 1:
            results = [wrap_W(*arguments[0])]
        else:
            with Pool(processes=n_thread) as pool:
                results = pool.starmap(wrap_W, arguments)
        phiavgs, grads = zip(*results)
        pickle.dump((phiavgs, grads, Ws), open("change_W_std.p", "wb"))

    grads = np.array(grads).reshape(Ws.shape[0], -1) 
    plt.loglog(Ws, np.std(grads, axis=1), 'k.')
    x = np.array([Ws[0], Ws[-1]])
    plt.loglog(x, 0.001*x**0.5, 'k--')
    plt.xlabel('$W$')
    plt.ylabel('std $\delta\Phi_{avg}$')
    plt.tight_layout()
    plt.savefig('W_std.png')
    plt.close()


# change_ga()
# change_L()
# change_W()
# change_W_std()
