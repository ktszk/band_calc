#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy as sc
import scipy.linalg as sclin
import matplotlib.pyplot as plt
from input_ham import import_out

fname='000AsP.input'
N=100
mu=3.0832
mass=3.2

alatt=sc.array([3.96*sc.sqrt(2.),3.96*sc.sqrt(2.),13.02*0.5])
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]]

xlabel=['$\Gamma$','X','M','$\Gamma$']

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
        xticks=xticks+[tmp2[0]]
        klist=klist+tmp
        splen=splen+list(tmp2)
        maxsplen=sc.amax(splen)
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
    plt.xlim(0,sc.amax(spl))
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

if __name__=="__main__":
    rvec,ndegen,ham_r,no,nr=import_out(fname,False)
    klist,spa_length,xticks=mk_klist(k_list,N)
    ham=sc.array([get_ham(k) for k in klist])
    etmp=[sclin.eigh(h) for h in ham]
    eigtmp=sc.array([eg[0] for eg in etmp])
    eig=eigtmp.T/mass-mu
    uni=sc.array([eg[1] for eg in etmp]).T
    plot_band(eig,spa_length,xticks,uni)

__license__="""Copyright (c) 2018 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
