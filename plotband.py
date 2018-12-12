#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy as sc
import scipy.linalg as sclin
import scipy.constants as scconst
import matplotlib.pyplot as plt
import input_ham

fname='000AsP.input'
sw_inp=0
mu=9.8
mass=1.0

alatt=sc.array([3.96*sc.sqrt(2.),3.96*sc.sqrt(2.),13.02*0.5])
Arot=sc.array([[ .5,-.5, .5],[ .5, .5, .5],[-.5,-.5, .5]])
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]]
xlabel=['$\Gamma$','X','M','$\Gamma$']
olist=[0,2,3]

N=100
FSmesh=20
eta=1.0e-1
sw_dec_axis=False #transform Cartesian axis
sw_color=True #plot band or FS with orbital weight

option=2
"""
option: switch calculation modes
0:band plot
1: write Fermi surface at kz=0
2: write 3D Fermi surface
3: write Fermi velocity with Fermi surface
4: plot spectrum like band plot
5: plot spectrum at E=EF
6: plot 3D Fermi velocity with Fermi surface
"""

spectrum=(True if option in (4,5) else False)
sw_FS=(True if option in (1,3,5) else False)
sw_plot_veloc=(True if option in (3,6) else False)
sw_3dfs=(True if option in (2,6) else False)

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
    hbar=scconst.physical_constants['Planck constant over 2 pi in eV s'][0]*1.0e10
    ham,expk=get_ham(k,rvec,ham_r,ndegen,out_phase=True)
    uni=sclin.eigh(ham)[1]
    vec0=sc.array([[[-1j*a*sc.sum(r*hr*expk)/hbar 
                      for a,r in zip(alatt,rvec.T)] for hr in hmr] for hmr in ham_r])
    vec=sc.array([sc.diag(sc.conjugate(uni.T).dot(v0.T).dot(uni)) for v0 in vec0.T]).T
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

def gen_uni(ham,blist):
    uni=sc.array([[sclin.eigh(h)[1][:,b] for h in hh] for hh,b in zip(ham,blist)])
    return uni

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

def plot_band(eig,spl,xticks,uni,ol):
    for e,cl in zip(eig,uni):
        clist=sc.array([sc.round_(abs(cl[ol[0]])*abs(cl[ol[0]]),4),
                        sc.round_(abs(cl[ol[1]])*abs(cl[ol[1]]),4),
                        sc.round_(abs(cl[ol[2]])*abs(cl[ol[2]]),4)]).T
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
    x=sc.linspace(-sc.pi,sc.pi,mesh,True)
    sqmesh=mesh*mesh
    X,Y=sc.meshgrid(x,x)
    return sc.array([X.reshape(1,sqmesh),Y.reshape(1,sqmesh),Y.reshape(1,sqmesh)*0.0]).T,X,Y

def mk_kf3d(mesh,sw_bnum):
    import skimage as sk
    from mpl_toolkits.mplot3d import axes3d
    km=sc.linspace(-sc.pi,sc.pi,mesh+1,True)
    cumesh=(mesh+1)*(mesh+1)*(mesh+1)
    x,y,z=sc.meshgrid(km,km,km)
    klist=sc.array([x.reshape(1,cumesh),y.reshape(1,cumesh),z.reshape(1,cumesh)]).T
    ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig,uni=gen_eig(ham,mass,mu,True)
    v2=[]
    if sw_bnum:
        fsband=[]
    for i,e in enumerate(eig):
        if(e.max()*e.min() < 0. ):
            vertices,faces,normals,values=sk.measure.marching_cubes_lewiner(e.reshape(mesh+1,mesh+1,mesh+1),0)
            if sw_bnum:
                fsband.append(i)
                v2.append((vertices-mesh/2)*2*sc.pi/mesh)
                #v3.append(faces)
            else:
                v2.extend((vertices-mesh/2)[faces])
    if sw_bnum:
        return v2,fsband
    else:
        return sc.array(v2)

def mk_kf2d(mesh,sw_bnum):
    import skimage as sk
    from mpl_toolkits.mplot3d import axes3d
    km=sc.linspace(-sc.pi,sc.pi,mesh+1,True)
    sqmesh=(mesh+1)*(mesh+1)
    x,y=sc.meshgrid(km,km)
    klist=sc.array([x.reshape(1,sqmesh),y.reshape(1,sqmesh),y.reshape(1,sqmesh)*0.0]).T
    ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig,uni=gen_eig(ham,mass,mu,True)
    v2=[]
    if sw_bnum:
        fsband=[]
    for i,e in enumerate(eig):
        if(e.max()*e.min() < 0. ):
            cont=sk.measure.find_contours(e.reshape(mesh+1,mesh+1),0)
            ct0=[]
            for c in cont:
                ct0.extend(c)
            ct=(sc.array([[c[0],c[1],mesh/2] for c in ct0])-mesh/2)*2*sc.pi/mesh
            if sw_bnum:
                fsband.append(i)
                v2.append(ct)
            else:
                v2.extend(ct)
    if sw_bnum:
        return v2,fsband
    else:
        return sc.array(v2)

def gen_3d_fs_plot(mesh):
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vert=mk_kf3d(mesh,False)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    #x,y,z=zip(*vert-mesh/2)
    #fs=ax.scatter(x,y,z,s=1.0)
    m = Poly3DCollection(vert)
    ax.add_collection3d(m)
    ax.set_xlim(-mesh/2, mesh/2)
    ax.set_ylim(-mesh/2, mesh/2)
    ax.set_zlim(-mesh/2, mesh/2)
    plt.tight_layout()
    plt.show()

def plot_veloc_FS(vfs,kfs):
    from mpl_toolkits.mplot3d import axes3d
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    vf,kf=[],[]
    for v,k in zip(vfs,kfs):
        vf.extend(v)
        kf.extend(k)
    x,y,z=zip(*sc.array(kf))
    ave=sc.array([sum(abs(v)) for v in vf])
    fs=ax.scatter(x,y,z,c=ave)
    ax.set_xlim(-mesh/2, mesh/2)
    ax.set_ylim(-mesh/2, mesh/2)
    ax.set_zlim(-mesh/2, mesh/2)
    plt.colorbar(fs)
    plt.show()

def plot_vec2(veloc,klist):
    v=[]
    k=[]
    for vv,kk in zip(veloc,klist):
        v0=sc.array([sc.sqrt(sum(abs(v0)*abs(v0))) for v0 in vv])
        v.extend(v0)
        k.extend(kk)
    v=sc.array(v)
    k=sc.array(k)
    plt.scatter(k[:,0],k[:,1],s=1.0,c=v)
    plt.jet()
    plt.xlim(-sc.pi,sc.pi)
    plt.ylim(-sc.pi,sc.pi)
    plt.xticks([-sc.pi,0,sc.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-sc.pi,0,sc.pi],['-$\pi$','0','$\pi$'])
    plt.colorbar(format='%.3e')
    plt.show()

def plot_FS(uni,klist,ol,eig,X,Y,sw_color):
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    if sw_color:
        for kk,cl in zip(klist,uni):
            clist=sc.array([[round(abs(c[ol[0]])*abs(c[ol[0]]),4),
                             round(abs(c[ol[1]])*abs(c[ol[1]]),4),
                             round(abs(c[ol[2]])*abs(c[ol[2]]),4)] for c in cl])
            plt.scatter(kk[:,0],kk[:,1],s=1.0,c=clist)
    else:
        for en in eig:
            if(en.max()*en.min()<0.0):
                plt.contour(X,Y,en.reshape(FSmesh,FSmesh),levels=[0.],color='black')
    plt.xlim(-sc.pi,sc.pi)
    plt.ylim(-sc.pi,sc.pi)
    plt.xticks([-sc.pi,0,sc.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-sc.pi,0,sc.pi],['-$\pi$','0','$\pi$'])
    plt.show()

def plot_vec(veloc,eig,X,Y):
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    for v,en in zip(veloc,eig):
        plt.contourf(X,Y,v.reshape(FSmesh,FSmesh).real,100)
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
    elif sw_inp==2:
        rvec,ndegen,ham_r,no,nr=input_ham.import_hr(fname,False)
    else: #Hopping.dat file
        rvec,ndegen,ham_r,no,nr,axis=input_ham.import_Hopping(False)

    if sw_dec_axis:
        rvec1=sc.array([Arot.T.dot(r) for r in rvec])
        rvec=rvec1
    if sw_3dfs:
        if sw_plot_veloc:
            klist,blist=mk_kf3d(FSmesh,True)
            veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
            plot_veloc_FS(veloc,klist)
        else:
            gen_3d_fs_plot(FSmesh)
    else:
        if sw_FS:
            if sw_plot_veloc:
                klist,blist=mk_kf2d(FSmesh,True)
            else:
                klist,X,Y=gen_ksq(FSmesh)
                klist1,blist=mk_kf2d(FSmesh,True)
                ham1=sc.array([[get_ham(k,rvec,ham_r,ndegen) for k in kk] for kk in klist1])
        else:
            klist,spa_length,xticks=mk_klist(k_list,N)
        if sw_plot_veloc:
            if sw_FS:
                veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
            else:
                veloc=sc.array([get_vec(k,rvec,ham_r,ndegen) for k in klist])
                abs_veloc=sc.array([[sc.sqrt(sum(v**2)) for v in vv] for vv in veloc]).T
                veloc=sc.array([get_vec(k,rvec,ham_r,ndegen).T for k in klist]).T
        else:
            ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
        if spectrum:
            if sw_FS:
                plot_FSsp(ham,mu,X,Y,eta)
            else:
                plot_spectrum(ham,spa_length,mu,eta)
        else:
            if sw_FS:
                if sw_plot_veloc:
                    plot_vec2(veloc,klist)
                else:
                    eig,uni=gen_eig(ham,mass,mu,True)
                    uni=gen_uni(ham1,blist)
                    plot_FS(uni,klist1,olist,eig,X,Y,sw_color)
            else:
                eig,uni=gen_eig(ham,mass,mu,True)
                plot_band(eig,spa_length,xticks,uni,olist)

__license__="""Copyright (c) 2018 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
