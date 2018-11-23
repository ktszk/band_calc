#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy as sc
import scipy.linalg as sclin
import matplotlib.pyplot as plt
import input_ham

fname='000AsP.input'
N=100
mu=9.8
mass=1.0

alatt=sc.array([3.96*sc.sqrt(2.),3.96*sc.sqrt(2.),13.02*0.5])
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]]
xlabel=['$\Gamma$','X','M','$\Gamma$']

FSmesh=100
eta=1.0e-1

sw_inp=0
spectrum=False
sw_FS=True
sw_plot_veloc=True

def get_ham(k,rvec,ham_r,ndegen,out_phase=False):
    def gen_phase(k,rvec,ndegen):
        phase=sc.array([sc.sum(r*k) for r in rvec])
        expk=(sc.cos(phase)-1j*sc.sin(phase))/ndegen
        return expk
    expk=gen_phase(k,rvec,ndegen)
    ham=sc.array([[sc.sum(hr*expk) for hr in hmr] for hmr in ham_r])
    if(out_phase):
        return ham,expk
    else:
        return ham

def get_vec(k,rvec,ham_r,ndegen):
    ham,expk=get_ham(k,rvec,ham_r,ndegen,out_phase=True)
    uni=sclin.eigh(ham)[1]
    vec0=sc.array([[[sc.sum(-1j*r*hr*expk) for r in rvec.T] for hr in hmr] for hmr in ham_r])
    vec=sc.array([sc.diag(sc.conjugate(uni.T).dot(v0.T).dot(uni))for v0 in vec0.T]).T
    return vec

def gen_eig(ham,mass,mu,sw):
    etmp=[sclin.eigh(h) for h in ham]
    eigtmp=sc.array([eg[0] for eg in etmp])
    if sw:
        eig=eigtmp.T/mass-mu
        uni=sc.array([eg[1] for eg in etmp]).T
        return eig,uni
    else:
        return (eigtmp.max()/mass-mu),(eigtmp.min()/mass-mu)

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
                        abs(cl[1])*abs(cl[1]),
                        abs(cl[2])*abs(cl[2])]).T
        plt.scatter(spl,e,s=5,c=clist)
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.xlim(0,spl.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

def plot_spectrum(ham,klen,mu,de=100,eta0=5.e-2,smesh=200):
    emax,emin=gen_eig(ham,mass,mu,False)
    w=sc.linspace(emin*1.1,emax*1.1,de)
    #eta=w*0+eta0
    etamax=4.0e0
    eta=etamax*w*w/min(emax*emax,emin*emin)+eta0
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

def gen_ksq(mesh):
    x=sc.linspace(-sc.pi,sc.pi,mesh)
    sqmesh=mesh*mesh
    X,Y=sc.meshgrid(x,x)
    return sc.array([X.reshape(1,sqmesh),Y.reshape(1,sqmesh)*0.0,Y.reshape(1,sqmesh)]).T,X,Y

def plot_FS(eig,X,Y):
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    for en in eig:
        if(en.max()*en.min()<0.0):
            plt.contour(X,Y,en.reshape(FSmesh,FSmesh),levels=[0.],color='black')
    plt.show()

def plot_vec(veloc,eig,X,Y):
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    for v,en in zip(veloc,eig):
        plt.contourf(X,Y,v[2].reshape(FSmesh,FSmesh).real,100)
        plt.colorbar()
        if(en.max()*en.min()<0.0):
            plt.contour(X,Y,en.reshape(FSmesh,FSmesh),levels=[0.])
        plt.show()

def plot_FSsp(ham,mu,X,Y,eta=5.0e-2,smesh=50):
    G=sc.array([-sclin.inv((0.+mu+eta*1j)*sc.identity(no)-h) for h in ham])
    trG=sc.array([sc.trace(gg).imag/(no*no) for gg in G]).reshape(FSmesh,FSmesh)
    #trG=sc.array([(gg[4,4]+gg[9,9]).imag/(no*no) for gg in G]).reshape(FSmesh,FSmesh)

    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    ax.set_xticks([-sc.pi,0,sc.pi])
    ax.set_xticklabels(['-$\pi$','0','$\pi$'])
    ax.set_yticks([-sc.pi,0,sc.pi])
    ax.set_yticklabels(['-$\pi$','0','$\pi$'])
    cont=ax.contourf(X,Y,trG,smesh,cmap=plt.jet())
    fig.colorbar(cont)
    plt.show()

if __name__=="__main__":
    if sw_inp==0: #.input file
        rvec,ndegen,ham_r,no,nr=input_ham.import_out(fname,False)
    elif sw_inp==1: #rvec.txt, ham_r.txt, ndegen.txt files
        rvec,ndegen,ham_r,no,nr=input_ham.import_hop(fname,True,False)
    else: #Hopping.dat file
        rvec,ndegen,ham_r,no,nr,axis=input_ham.import_Hopping(False)

    if sw_FS:
        klist,X,Y=gen_ksq(FSmesh)
    else:
        klist,spa_length,xticks=mk_klist(k_list,N)
    ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    if sw_plot_veloc:
        veloc=sc.array([get_vec(k,rvec,ham_r,ndegen) for k in klist])
        abs_veloc=sc.array([[sc.sqrt(sum(v**2)) for v in vv] for vv in veloc]).T
        veloc=sc.array([get_vec(k,rvec,ham_r,ndegen).T for k in klist]).T
    if spectrum:
        if sw_FS:
            plot_FSsp(ham,mu,X,Y,eta)
        else:
            plot_spectrum(ham,spa_length,mu,eta)
    else:
        eig,uni=gen_eig(ham,mass,mu,True)
        if sw_FS:
            if sw_plot_veloc:
                plot_vec(veloc,eig,X,Y)
                #plot_vec(abs_veloc,eig,X,Y)
            else:
                plot_FS(eig,X,Y)
        else:
            plot_band(eig,spa_length,xticks,uni)

__license__="""Copyright (c) 2018 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
