# the no-propagate algorithm for finite T, section 3.1 in paper
from __future__ import division
import shutil
import sys
import os
import time
import pickle
import itertools
import numpy as np
from numpy import newaxis, sin, cos, pi, tanh, cosh, sqrt
# from math import 
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

J = 4*np.array([
       [-0.13516168, -0.29783588, -0.08243096,  0.41499251, -0.12447637,
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


def nop(ga=3, sig=1.5, noise_dim=n_dim, L=10000):
    np.random.seed()
    starttime = time.time()
    one = np.ones(n_dim)

    if sig == 0:
        def xnewI(x):
            xnew = J @ tanh(x) + ga
            return xnew, 0 

    elif noise_dim == 1:
        def xnewI(x):
            y = np.random.normal(scale=sig)
            xnew = J @ tanh(x) + ga + y * one / sqrt(n_dim)
            I = sqrt(n_dim) * (-y) / sig**2
            return xnew, I

    elif noise_dim == n_dim:
        def xnewI(x):
            y = np.random.normal(scale=sig, size=noise_dim)
            xnew = J @ tanh(x) + ga + y
            fga = one
            dpp = (-y) / sig**2
            I = fga @ dpp
            return xnew, I

    else:
        pass


    def phi_func(x):
        return x.sum() - n_dim * ga

    # def ffga(x, y):
        # xnew = J @ tanh(x) + y
        # fga = J @ (1 / cosh(x+ga)**2)
        # return xnew, fga

    # # return dp/p
    # if sig == 0:
        # def dpp(y): 
            # return np.zeros(n_dim)
    # else:
        # def dpp(y): 
            # return (-y) / sig**2


    phiT, S = nanarray([2, L])
    I02 = nanarray(L)
    xT = nanarray([L, n_dim])
    for l in range(L):
        x = nanarray([n_layer+1, n_dim])
        I = nanarray([n_layer+1,])
        # x[0] = np.random.normal(scale=1, size=n_dim) 
        x[0] = np.random.normal(scale=1, size=n_dim) + ga

        for m in range(n_layer):
            # y = np.random.normal(scale=sig, size=n_dim)
            # x[m+1], fga = ffga(x[m], y)
            # I[m] = fga @ dpp(y)
            x[m+1], I[m] = xnewI(x[m])
        xT[l] = x[-1]
        phiT[l] = phi_func(xT[l])
        S[l] = - I[:-1].sum() + (x[0]-ga).sum()
        I02[l] = I[-2]**2

    phiTavg = phiT.mean()
    phiT_central = phiT - phiTavg # centralize phiT
    grad = np.mean(S * phiT_central) - 9
    print("\n{: .2f}, {:9d}, {: .2e}, {: .2e}".format(ga, L, phiTavg, grad))
    print("sig = ", sig)
    print("I0 square = ", I02.mean() )
    print("S mean = ", S.mean() )
    endtime = time.time()
    print('time spent (seconds):', endtime-starttime)

    return phiTavg, grad


def change_ga():
    NN = 13 # number of steps in parameters
    galeft = -1
    garight = 1
    sig = 1.5
    A = (galeft-garight)/(NN-1)/2.5 # step size in the plot
    gas = np.linspace(galeft,garight,NN)
    phiavgs_sig0 = nanarray(NN)
    phiavgs_1dimnoise = nanarray(NN)
    phiavgs_Mdimnoise = nanarray(NN)
    grads_1dimnoise = nanarray(NN)
    grads_Mdimnoise = nanarray(NN)
    try:
        gas, phiavgs_sig0, phiavgs_1dimnoise, grads_1dimnoise, phiavgs_Mdimnoise, grads_Mdimnoise = pickle.load(open("change_ga_sig"+"{}".format(sig)+".p", "rb"))
    except FileNotFoundError:
        for i, ga in enumerate(gas):
            phiavgs_sig0[i], _ = nop(ga,sig=0)
            phiavgs_1dimnoise[i], grads_1dimnoise[i] = nop(ga, sig=sig, noise_dim=1)
            phiavgs_Mdimnoise[i], grads_Mdimnoise[i] = nop(ga, sig=sig, noise_dim=n_dim)
        pickle.dump((gas, phiavgs_sig0, phiavgs_1dimnoise, grads_1dimnoise, phiavgs_Mdimnoise, grads_Mdimnoise), open("change_ga_sig"+"{}".format(sig)+".p", "wb"))
    plt.figure(figsize=[11,5])

    plt.plot(gas, phiavgs_sig0, marker='o', color='black', linestyle='None', markersize=7, label = 'no noise')
    # plt.plot(gas, phiavgs_sig0, 'k--', markersize=6)

    plt.plot(gas, phiavgs_1dimnoise, marker='s', color='red', linestyle='None', markersize=6, label = '1 dim noise')
    for ga, phiavg, grad in zip(gas, phiavgs_1dimnoise, grads_1dimnoise):
        plt.plot([ga-A, ga+A], [phiavg-grad*A, phiavg+grad*A], color='red', linestyle='-')

    plt.plot(gas, phiavgs_Mdimnoise, marker='^', color='blue', linestyle='None', markersize=6, label = 'M dim noise')
    for ga, phiavg, grad in zip(gas, phiavgs_Mdimnoise, grads_Mdimnoise):
        plt.plot([ga-A, ga+A], [phiavg-grad*A, phiavg+grad*A], color='blue', linestyle='-')
    
    plt.ylabel('$\Phi_{avg} $')
    plt.xlabel('$\gamma$')
    plt.legend()
    # plt.ylim(0.455,0.53)
    plt.tight_layout()
    plt.savefig("change_ga_sig"+"{}".format(sig)+".png")
    plt.close()


def wrap_L(L): return nop(L=L)

def change_L():
    # gradients for different trajectory length L
    Ls = np.array([1, 2, 5, 1e1, 2e1, 5e1], dtype=int) * 1000
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


change_ga()
# change_L()
