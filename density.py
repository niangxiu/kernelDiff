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

nbins = 30

def nop(ga=2,sig=0.1):

    W = 4
    L = 10000000
    nPre = 1000

    def ffga(x, y):
        if 0 <= x <= 0.5:
            f = ga*x 
        else:
            f = ga*(1-x)
        xnew =np.array( (f+y) %1) 
        return xnew

    x0 = 0.5
    for m in range(nPre):
        y = np.random.normal(scale=sig)
        x0 = ffga(x0, y)

    x = nanarray([W+L+1,])
    x[0] = np.random.rand()
    for m in range(W+L):
        y = np.random.normal(scale=sig)
        x[m+1] = ffga(x[m], y)

    # plt.hist(x, 30, density=True, histtype='step', alpha=0.75, label="".format(sig))
    den = np.histogram(x, bins=nbins, range=(0,1), density=True, weights=None)

    return den[0], x.mean()#sin(2*np.pi*x).mean()


def change_sig():

    starttime = time.time()
    # sigs = np.array([0,0.05,0.1,0.2,0.3,0.5,0.01,0.02,0.03,0.04])
    sigs = np.array([0,0.05,0.1,0.2,0.3,0.5])
    dens = nanarray([sigs.shape[0],nbins])
    Phiavgs = nanarray([sigs.shape[0]])

    x = np.linspace(0,1,nbins+1)
    x = (x[1:]+x[:-1])/2
   
    try:
        dens, Phiavgs = pickle.load( open("dens.p", "rb"))
    except FileNotFoundError:
        for i, sig in enumerate(sigs):
            print(i)
            dens[i], Phiavgs[i] = nop(ga=3,sig=sig)
    pickle.dump((dens, Phiavgs), open("dens.p", "wb"))

    endtime = time.time()
    print('time elapsed in seconds:', endtime-starttime)

    x = np.linspace(0,1,nbins+1)
    x = (x[1:]+x[:-1])/2
    fig = plt.figure(figsize=(8,6.5))
    linestyles = ['-', '--', '-.', ':','.', '.-' ]
    for i, sig in enumerate(sigs[:6]):
        plt.plot(x, dens[i], linestyles[i], label="$\sigma$={}".format(sig) )
    plt.xlabel('x')
    plt.ylabel('density')
    plt.legend()
    plt.grid(True)
    plt.savefig('densities.png')
    plt.close()

    fig = plt.figure(figsize=(10,8))
    plt.plot(sigs, Phiavgs, '.')
    plt.xlabel('$\sigma$')
    plt.ylabel('$\Phi_{avg}$')
    plt.tight_layout()
    plt.savefig('Phi_sig_converge.png')
    plt.close()

change_sig()
