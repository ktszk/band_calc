#!/usr/bin/env python
#-*- coding:utf-8 -*-
import scipy as sc

fname='000AsP.input' #hamiltonian file name
mu=9.8               #chemical potential
mass=1.0             #effective mass
sw_inp=0             #input hamiltonian format
"""
sw_inp: switch input hamiltonian's format
0: .input file
1: ham_r.txt, irvec.txt, ndegen.txt
2: {case}_hr.dat file (wannier90 default hopping file)
else: Hopping.dat file (ecalj hopping file)
"""

sw_calc_mu =True
fill=3.05

alatt=sc.array([3.96*sc.sqrt(2.),3.96*sc.sqrt(2.),13.02*0.5]) #Bravais lattice parameter a,b,c
Arot=sc.array([[ .5,-.5, .5],[ .5, .5, .5],[-.5,-.5, .5]]) #rotation matrix for dec. to primitive vector
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]] #coordinate of sym. points
xlabel=['$\Gamma$','X','M','$\Gamma$'] #sym. points name

olist=[1,2,3]        #orbital number with color plot [R,G,B] if you merge some orbitals input orbital list in elements
N=100                #kmesh btween symmetry points
FSmesh=100           #kmesh for option in {1,2,3,5,6}
eta=1.0e-1           #eta for green function
sw_dec_axis=False    #transform Cartesian axis
sw_color=True        #plot band or FS with orbital weight
kz=sc.pi*0.
with_spin=False #use only with soc hamiltonian

option=1
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
#----------import modules without scipy-------------
import scipy.linalg as sclin
import scipy.optimize as scopt
import scipy.constants as scconst
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import input_ham
#----------------define functions-------------------
def get_ham(k,rvec,ham_r,ndegen,out_phase=False):
    """
    This function generates hamiltonian from hopping parameters.
    arguments:
    k: k-point coordinate
    rvec: real space coodinate for hoppings
    ham_r: values of hoppings
    ndegen: weight for hoppings
    out_phase: output or not phase array (optional, default=False)
    return values:
    ham: wave-number space hamiltonian in k
    expk: phase in k
    """
    phase=(rvec*k).sum(axis=1)
    expk=(sc.cos(phase)-1j*sc.sin(phase))/ndegen
    no,nr=len(ham_r),len(expk)
    ham=(ham_r.reshape(no*no,nr)*expk).sum(axis=1).reshape(no,no)
    if out_phase:
        return ham, expk,no,nr
    else:
        return ham

def get_mu(fill,rvec,ham_r,ndegen,temp=1.0e-3,mesh=40):
    """
    This function calculates chemical potential.
    arguments:
    fill: band filling (number of particles in band)
    rvec: real space coodinate for hoppings
    ham_r: values of hoppings
    ndegen: weight for hoppings
    temp: temperature (optional, default=1.0e-3)
    mesh: k-points mesh (optional, default=40)
    return value:
    mu: chemical potential
    """
    km=sc.linspace(-sc.pi,sc.pi,mesh+1,True)
    x,y,z=sc.meshgrid(km,km,km)
    klist=sc.array([x.ravel(),y.ravel(),z.ravel()]).T
    ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=sc.array([sclin.eigvalsh(h) for h in ham]).T
    f=lambda mu: 2.*fill*(mesh**3)+(sc.tanh(0.5*(eig-mu)/temp)-1.).sum()
    #mu=scopt.brentq(f,eig.min(),eig.max())
    mu=scopt.newton(f,0.5*(eig.min()+eig.max()))
    print('chemical potential = %6.3f'%mu)
    return mu

def get_vec(k,rvec,ham_r,ndegen):
    """
    This function generates velocities from hopping parameters.
    arguments:
    k: k-point coordinate
    rvec: real space coodinate for hoppings
    ham_r: values of hoppings
    ndegen: weight for hoppings
    return values:
    vec: velocity in k for each bands
    """
    ihbar=1./scconst.physical_constants['Planck constant over 2 pi in eV s'][0]*1.0e-10
    ham,expk,no,nr=get_ham(k,rvec,ham_r,ndegen,out_phase=True)
    uni=sclin.eigh(ham)[1]
    vec0=sc.array([-1j*ihbar*(ham_r.reshape(no*no,nr)*(r*expk)).sum(axis=1).reshape(no,no)
                    for r in (alatt*rvec).T])
    vec=sc.array([(uni.conjugate().T.dot(v0).dot(uni)).diagonal() for v0 in vec0]).T
    return vec

def gen_eig(ham,mass,mu,sw):
    """
    This function generates eigenvalue and eigenvectors or max and min energy values of hamiltonian.
    arguments:
    ham: wave-number space hamiltonian in k
    mass: effective mass
    mu: chemical potential
    sw: switch for retrun enegyes or max/min energy
    return values:
    eig: eigenvalues of hamiltonian
    uni: eigenvectors of hamiltonian
    """
    if sw:
        etmp=[sclin.eigh(h) for h in ham]
        eigtmp=sc.array([eg[0] for eg in etmp])
        eig=eigtmp.T/mass-mu
        uni=sc.array([eg[1] for eg in etmp]).T
        return eig,uni
    else:
        eigtmp=sc.array([sclin.eigvalsh(h) for h in ham])
        return (eigtmp.max()/mass-mu),(eigtmp.min()/mass-mu)

def mk_klist(k_list,N):
    """
    This function generates klist of spaghetti.
    arguments:
    k_list: name and coordinate of sym. points
    N: k-mesh between sym. points
    return values:
    klist: klist of spaghetti
    splen: length of hrizontal axis
    xticks: xlabels
    """
    klist=[]
    splen=[]
    maxsplen=0
    xticks=[]
    for ks,ke in zip(k_list,k_list[1:]):
        dkv=sc.array(ke)-sc.array(ks)
        dkv_length=sc.sqrt(((dkv*alatt)*(dkv*alatt)).sum())
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
    """
    This function plot spaghetti.
    arguments:
    eig: energy array
    spl: coordinates of horizontal axis
    xticks: xlabels
    uni: weight of orbitals
    ol: plot orbital list
    """
    for e,cl in zip(eig,uni):
        c1=((abs(cl[ol[0]])*abs(cl[ol[0]])).round(4) if isinstance(ol[0],int)
            else (abs(cl[ol[0]])*abs(cl[ol[0]])).sum(axis=0).round(4))
        c2=((abs(cl[ol[1]])*abs(cl[ol[1]])).round(4) if isinstance(ol[1],int)
            else (abs(cl[ol[1]])*abs(cl[ol[1]])).sum(axis=0).round(4))
        c3=((abs(cl[ol[2]])*abs(cl[ol[2]])).round(4) if isinstance(ol[2],int)
            else (abs(cl[ol[2]])*abs(cl[ol[2]])).sum(axis=0).round(4))
        clist=sc.array([c1,c2,c3]).T
        plt.scatter(spl,e,s=5,c=clist)
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.xlim(0,spl.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

def plot_spectrum(ham,klen,mu,de=100,eta0=5.e-2,smesh=200):
    """
    This function plot spaghetti like spectrum.
    arguments:
    ham: hamiltonian array
    klen: coordinates of horizontal axis
    mu: chemical potential
    de: energy mesh (optional, default=100)
    eta: eta for green function (optional, default=5e-2)
    smesh: contor mesh (optional, default=200)
    """
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

def gen_ksq(mesh,kz):
    """
    This function generates square k mesh for 2D spectrum plot
    arguments:
    mesh: k-mesh grid size
    kz: kz of plotting FS plane
    """
    x=sc.linspace(-sc.pi,sc.pi,mesh,True)
    sqmesh=mesh*mesh
    X,Y=sc.meshgrid(x,x)
    return sc.array([X.ravel(),Y.ravel(),Y.ravel()*0.0+kz]).T,X,Y

def mk_kf(mesh,sw_bnum,dim,kz=0):
    """
    This function generates k-list on Fermi surfaces
    arguments:
    mesh: initial k-mesh grid size
    sw_bnum:switch output format
    dim: output dimension
    kz: kz of plotting FS plane use only dim=2 (optional,default=0)
    return values:
    v2: klist on Fermi surface
    fsband: band number crossing Fermi energy
    """
    import skimage as sk
    from mpl_toolkits.mplot3d import axes3d
    km=sc.linspace(-sc.pi,sc.pi,mesh+1,True)
    if dim==2:
        x,y=sc.meshgrid(km,km)
        z=y*0.0+kz
    elif dim==3:
        x,y,z=sc.meshgrid(km,km,km)
    klist=sc.array([x.ravel(),y.ravel(),z.ravel()]).T
    ham=sc.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=sc.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    v2=[]
    if sw_bnum:
        fsband=[]
    for i,e in enumerate(eig):
        if(e.max()*e.min() < 0. ):
            if dim==2:
                cont=sk.measure.find_contours(e.reshape(mesh+1,mesh+1),0)
                ct0=[]
                for c in cont:
                    ct0.extend(c)
                ct=(sc.array([[c[0],c[1],+mesh/2] for c in ct0])-mesh/2)*2*sc.pi/mesh
                ct[:,2]=kz
                if sw_bnum:
                    fsband.append(i)
                    v2.append(ct)
                else:
                    v2.extend(ct)
            elif dim==3:
                vertices,faces,normals,values=sk.measure.marching_cubes_lewiner(e.reshape(mesh+1,mesh+1,mesh+1),0)
                if sw_bnum:
                    fsband.append(i)
                    v2.append((vertices-mesh/2)*2*sc.pi/mesh)
                    #v3.append(faces)
                else:
                    v2.extend((2*sc.pi*(vertices-mesh/2)/mesh)[faces])
    if sw_bnum:
        return v2,fsband
    else:
        return sc.array(v2)

def gen_3d_fs_plot(mesh):
    """
    This function plot 3D Fermi Surface
    argument:
    mesh: k-grid mesh size
    """
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vert=mk_kf(mesh,False,3)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    m = Poly3DCollection(vert)
    ax.add_collection3d(m)
    ax.set_xlim(-sc.pi, sc.pi)
    ax.set_ylim(-sc.pi, sc.pi)
    ax.set_zlim(-sc.pi, sc.pi)
    plt.tight_layout()
    plt.show()

def plot_veloc_FS(vfs,kfs):
    """
    This function plot 3D Fermi velocities
    argument:
    mesh: k-grid mesh size
    """
    from mpl_toolkits.mplot3d import axes3d
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    vf,kf=[],[]
    for v,k in zip(vfs,kfs):
        ave_vx=abs(sc.array(v).T[0]).mean()
        ave_vy=abs(sc.array(v).T[1]).mean()
        ave_vz=abs(sc.array(v).T[2]).mean()
        print '%.3e %.3e %.3e'%(ave_vx,ave_vy,ave_vz)
        vf.extend(v)
        kf.extend(k)
    x,y,z=zip(*sc.array(kf))
    vf=sc.array(vf)
    ave_vx=abs(vf.T[0]).mean()
    ave_vy=abs(vf.T[1]).mean()
    ave_vz=abs(vf.T[2]).mean()
    print '%.3e %.3e %.3e'%(ave_vx,ave_vy,ave_vz)
    absv=sc.array([abs(v).sum() for v in vf])
    fs=ax.scatter(x,y,z,c=absv,cmap=cm.jet)
    ax.set_xlim(-sc.pi, sc.pi)
    ax.set_ylim(-sc.pi, sc.pi)
    ax.set_zlim(-sc.pi, sc.pi)
    plt.colorbar(fs,format='%.2e')
    plt.show()

def plot_vec2(veloc,klist):
    v=[]
    k=[]
    for vv,kk in zip(veloc,klist):
        v0=sc.array([sc.sqrt((abs(v0)*abs(v0)).sum()) for v0 in vv])
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
    plt.colorbar(format='%.2e')
    plt.show()

def plot_FS(uni,klist,ol,eig,X,Y,sw_color):
    """
    This function plot 2D Fermi Surface with/without orbital weight
    argument:
    uni: eigenvectors
    klist: klist of Fermi surface
    ol: orbital list using color plot
    eig: eigenvalues
    X: X axis array
    Y: Y axis array
    sw_color: swtich of color plot
    """
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    if sw_color:
        col=['r','g','b','c','m','y','k','w']
        ncut=8
        for kk,cl,cb in zip(klist,uni,col):
            cl=sc.array(cl)
            c1=((abs(cl[:,ol[0]])*abs(cl[:,ol[0]])).round(4) if isinstance(ol[0],int)
                else (abs(cl[:,ol[0]])*abs(cl[:,ol[0]])).sum(axis=1).round(4))
            c2=((abs(cl[:,ol[1]])*abs(cl[:,ol[1]])).round(4) if isinstance(ol[1],int)
                else (abs(cl[:,ol[1]])*abs(cl[:,ol[1]])).sum(axis=1).round(4))
            c3=((abs(cl[:,ol[2]])*abs(cl[:,ol[2]])).round(4) if isinstance(ol[2],int)
                else (abs(cl[:,ol[2]])*abs(cl[:,ol[2]])).sum(axis=1).round(4))
            clist=sc.array([c1,c2,c3]).T
            if(with_spin):
                v1=((cl[:,no/2:]*cl[:,:no/2].conjugate()).sum(axis=1)
                    +(cl[:,:no/2]*cl[:,no/2:].conjugate()).sum(axis=1)).real
                v2=((cl[:,no/2:]*cl[:,:no/2].conjugate()).sum(axis=1)
                    -(cl[:,:no/2]*cl[:,no/2:].conjugate()).sum(axis=1)).imag
                v1=v1[::ncut].round(4)
                v2=v2[::ncut].round(4)
                k1=kk[::ncut,0]
                k2=kk[::ncut,1]
                plt.quiver(k1,k2,v1,v2,color=cb,angles='xy',scale_units='xy',scale=3.0)
            plt.scatter(kk[:,0],kk[:,1],s=2.0,c=clist)
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

#--------------------------main program-------------------------------
if __name__=="__main__":
    if sw_inp==0: #.input file
        rvec,ndegen,ham_r,no,nr=input_ham.import_out(fname,False)
    elif sw_inp==1: #rvec.txt, ham_r.txt, ndegen.txt files
        rvec,ndegen,ham_r,no,nr=input_ham.import_hop(fname,True,False)
    elif sw_inp==2:
        rvec,ndegen,ham_r,no,nr=input_ham.import_hr(fname,False)
    else: #Hopping.dat file
        rvec,ndegen,ham_r,no,nr,axis=input_ham.import_Hopping(False)

    if sw_calc_mu:
        mu=get_mu(fill,rvec,ham_r,ndegen)
    else:
        try:
            mu
        except NameError:
            mu=get_mu(fill,rvec,ham_r,ndegen)

    if sw_dec_axis:
        rvec1=sc.array([Arot.T.dot(r) for r in rvec])
        rvec=rvec1

    if sw_3dfs:
        if sw_plot_veloc:
            klist,blist=mk_kf(FSmesh,True,3)
            veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
            plot_veloc_FS(veloc,klist)
        else:
            gen_3d_fs_plot(FSmesh)
    else:
        if sw_FS:
            if sw_plot_veloc:
                klist,blist=mk_kf(FSmesh,True,2,kz)
            else:
                klist,X,Y=gen_ksq(FSmesh,kz)
                klist1,blist=mk_kf(FSmesh,True,2,kz)
                ham1=sc.array([[get_ham(k,rvec,ham_r,ndegen) for k in kk] for kk in klist1])
        else:
            klist,spa_length,xticks=mk_klist(k_list,N)
        if sw_plot_veloc:
            if sw_FS:
                veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
            else:
                veloc=sc.array([get_vec(k,rvec,ham_r,ndegen) for k in klist])
                abs_veloc=sc.array([[sc.sqrt((v*v).sum()) for v in vv] for vv in veloc]).T
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
                    uni=sc.array([[sclin.eigh(h)[1][:,b] for h in hh] for hh,b in zip(ham1,blist)])
                    plot_FS(uni,klist1,olist,eig,X,Y,sw_color)
            else:
                eig,uni=gen_eig(ham,mass,mu,True)
                plot_band(eig,spa_length,xticks,uni,olist)

__license__="""Copyright (c) 2018-2019 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
