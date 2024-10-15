from __future__ import division
import shutil
import sys
import os
import time
import pickle
import itertools
import numpy as np
from numpy import newaxis, sin, cos, pi
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


def nop(ga=3, sig=0.1, L=100000, W=7):
    
    nPre = 100
    np.random.seed()

    def ffga(x, y):
        if 0 <= x <= 0.5:
            f = ga*x 
            fga = x
        else:
            f = ga*(1-x)
            fga = 1-x
        xnew =np.array( (f+y) %1) 
        return xnew, fga

    def phi_func(x):
        return x
    
    # return dp/p
    if sig == 0:
        def dpp(y): 
            return 0
    else:
        def dpp(y): 
            return (-y) / sig**2

    x0 = 0.5
    for m in range(nPre):
        y = np.random.normal(scale=sig)
        x0, _ = ffga(x0, y)

    x, I, phi = nanarray([3, W+L+1])
    x[0] = x0
    phi[0] = phi_func(x0)
    for m in range(W+L):
        y = np.random.normal(scale=sig)
        x[m+1], fga = ffga(x[m], y)
        I[m+1] = fga * dpp(y)
        phi[m+1] = phi_func(x[m+1])
    phiavg = phi.mean()
    phi -= phiavg # centralize phi

    phiS = np.zeros(L+1)
    for n in range(0, W+1):
        phiS += phi[n:n+L+1]
    grad = - np.mean(phiS[1:L+1] * I[1:L+1])
    print("{}, {:9d}, {}, {:.2e}, {:.2e}".format(ga, L, W, phiavg, grad))
    return phiavg, grad


def change_ga():
    NN = 30 # number of steps in parameters
    galeft = 1
    garight = 5
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
            results = [wrap_L(L[0]) for L in arguments]
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
            results = [wrap_W(W[0]) for W in arguments]
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
            results = [wrap_W(W[0]) for W in arguments]
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


change_ga()
change_L()
change_W()
change_W_std()
