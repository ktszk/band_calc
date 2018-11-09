#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy as sc
import scipy.linalg as sclin
import matplotlib.pyplot as plt
from input_ham import import_out

fname='000AsP.input'
N=100
mu=3.0832*3.2
mass=1. #3.2

alatt=sc.array([3.96*sc.sqrt(2.),3.96*sc.sqrt(2.),13.02*0.5])
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]]

xlabel=['$\Gamma$','X','M','$\Gamma$']

spectrum=True

def get_ham(k):
    phase=sc.array([sc.sum(r*k) for r in rvec])
    expk=(sc.cos(phase)-1j*sc.sin(phase))/ndegen
    ham=sc.array([[sc.sum(hr*expk) for hr in hmr] for hmr in ham_r])
    return ham

def mk_klist(k_list,N):
    klist=[]
    splen=[]
    maxsplen=0
    xticks=[]
    for ks,ke in zip(k_list,k_list[1:]):
        dkv=sc.array(ke)-sc.array(ks)
        dkv_length=sc.sqrt(sc.sum((dkv*alatt)**2))
        tmp=[2.*sc.pi*(dkv/N*i+ks) for i in range(N)]
        tmp2=sc.linspace(0,dkv_length,N)+maxsplen*N/(N-1)
        maxsplen=tmp2.max()
        xticks=xticks+[tmp2[0]]
        klist=klist+tmp
        splen=splen+list(tmp2)
    klist=klist+[2*sc.pi*sc.array(k_list[-1])]
    splen=splen+[maxsplen+dkv_length/N]
    xticks=xticks+[splen[-1]]
    return sc.array(klist),sc.array(splen),xticks

def plot_band(eig,spl,xticks,uni):
    for e,cl in zip(eig,uni):
        clist=sc.array([abs(cl[0])*abs(cl[0]),
                        0.5*(abs(cl[1])*abs(cl[1])
                             +abs(cl[2])*abs(cl[2])),
                        abs(cl[3])*abs(cl[3])]).T
        plt.scatter(spl,e,s=5,c=clist)
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.xlim(0,spl.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

def plot_spectrum(ham,klen,mu,de=100,eta0=5.e-2,smesh=200):
    etmp=[sclin.eigh(h) for h in ham]
    eigtmp=sc.array([eg[0] for eg in etmp])
    emax=(eigtmp.max()-mu)*1.1
    emin=(eigtmp.min()-mu)*1.1
    w=sc.linspace(emin,emax,de)
    #eta=w*0+eta0
    etamax=4.0e0
    eta=etamax*w*w/(emax*emax)+eta0
    G=sc.array([[-sclin.inv((ww+mu+et*1j)*sc.identity(no)-h) for h in ham] for ww,et in zip(w,eta)])
    trG=sc.array([[sc.trace(gg).imag/(no*no) for gg in g] for g in G])
    sp,w=sc.meshgrid(klen,w)
    plt.hot()
    plt.contourf(sp,w,trG,smesh)
    plt.colorbar()
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.xlim(0,klen.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()


if __name__=="__main__":
    rvec,ndegen,ham_r,no,nr=import_out(fname,False)
    klist,spa_length,xticks=mk_klist(k_list,N)
    ham=sc.array([get_ham(k) for k in klist])
    if spectrum:
        plot_spectrum(ham,spa_length,mu)
    else:
        etmp=[sclin.eigh(h) for h in ham]
        eigtmp=sc.array([eg[0] for eg in etmp])
        eig=eigtmp.T/mass-mu
        uni=sc.array([eg[1] for eg in etmp]).T
        plot_band(eig,spa_length,xticks,uni)

__license__="""Copyright (c) 2018 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
