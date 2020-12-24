#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np

fname='000AsP.input' #hamiltonian file name
sw_inp=0             #input hamiltonian format
"""
sw_inp: switch input hamiltonian's format
0: .input file
1: ham_r.txt, irvec.txt, ndegen.txt
2: {case}_hr.dat file (wannier90 default hopping file)
else: Hopping.dat file (ecalj hopping file)
"""

option=0
"""
option: switch calculation modes
 0: band plot
 1: write Fermi surface at kz (default: kz=0)
 2: write 3D Fermi surface
 3: plot 3D Fermi velocity with Fermi surface
 4: write Fermi velocity with Fermi surface
 5: plot spectrum like band plot
 6: plot spectrum at E=EF
 7: calc conductivity
 8: plot Dos
 9: calc carrier num.
10: calc cycrotron mass
"""

N=200                #kmesh btween symmetry points
FSmesh=40           #kmesh for option in {1,2,3,5,6}
wmesh=200
(emin,emax)=(-3,3)
eta=1.0e-3           #eta for green function
de=1.e-4
kz=np.pi*0.          #kz for option 1,4 and 6
sw_dec_axis=False    #transform Cartesian axis
sw_color=True        #plot band or FS with orbital weight
with_spin=False      #use only with soc hamiltonian
mass=1.0             #effective mass (reduce band width by hand)

sw_calc_mu =True
fill=6.00            #band filling
temp=1.0e-9          #temperature
mu=9.8               #chemical potential

alatt=np.array([3.96*np.sqrt(2.),3.96*np.sqrt(2.),13.02*0.5]) #Bravais lattice parameter a,b,c
Arot=np.array([[ .5, -.5, .5],[ .5, .5, .5],[-.5,-.5, .5]]) #rotation matrix for dec. to primitive vector
#Arot=np.array([[ 1., 0., 0.],[ 0., 1., 0.],[ 0.,0., 1.]]) #rotation matrix for dec. to primitive vector

k_list=[[0.,0.,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]] #coordinate of sym. points
xlabel=['Z','$\Gamma$','X','M','$\Gamma$'] #sym. points name
#orbital number with color plot [R,G,B] if you merge some orbitals input orbital list in elements
olist=[[2,7],[1,3,6,8],[4,9]]

#----------import modules without scipy-------------
import scipy as sc
import scipy.linalg as sclin
import scipy.optimize as scopt
import scipy.constants as scconst
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpi4py import MPI
import input_ham
#----------------define functions-------------------
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

if sw_dec_axis:
    pass
else:
    avec=alatt*Arot
    bvec=sclin.inv(avec).T
    Vuc=sclin.det(avec)

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
    expk=(np.cos(phase)-1j*np.sin(phase))/ndegen
    no,nr=len(ham_r),len(expk)
    ham=(ham_r.reshape(no*no,nr)*expk).sum(axis=1).reshape(no,no)
    if out_phase:
        return ham, expk,no,nr
    else:
        return ham

def gen_klist(mesh,k_sw=True,dim=None):
    if rank==0:
        if k_sw:
            km=np.linspace(0,2*np.pi,mesh,False)
            x,y,z=np.meshgrid(km,km,km)
            klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
        else:
            klist=make_kmesh(mesh,dim,kz,sw=False)
        Nk=len(klist)
        sendbuf=klist.flatten()
        cks=divmod(sendbuf.size//3,size)
        count=np.array([3*(cks[0]+1) if i<cks[1] else 3*cks[0] for i in range(size)])
        displ=np.array([sum(count[:i]) for i in range(size)])
    else:
        Nk=None
        sendbuf=None
        count=np.empty(size,dtype=np.int)
        displ=None
    comm.Bcast(count,root=0)
    Nk=comm.bcast(Nk,root=0)
    recvbuf=np.empty(count[rank],dtype='f8')
    comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
    k_mpi=recvbuf.reshape(count[rank]//3,3)
    return (Nk,count//3,k_mpi)

def get_mu(fill,rvec,ham_r,ndegen,temp,mesh=40):
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

    Nk,count,k_mpi=gen_klist(mesh)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    e_mpi=np.array([sclin.eigvalsh(h) for h in ham]).T
    sendbuf=e_mpi.flatten()
    if rank==0:
        no=sendbuf.size//count[rank]
        count=count*no
        recvbuf=np.empty(count.sum(),dtype='f8')
        displ=np.array([count[:i].sum() for i in range(size)])
    else:
        recvbuf=None
        displ=None
    comm.Bcast(count,root=0)
    comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE],root=0)
    if rank==0:
        eig=recvbuf.reshape(Nk,no)
        f=lambda mu: 2.*fill*mesh**3+(np.tanh(0.5*(eig-mu)/temp)-1.).sum()
        mu=scopt.brentq(f,eig.min(),eig.max())
        #mu=scopt.newton(f,0.5*(eig.min()+eig.max()))
        print('chemical potential = %6.3f'%mu)
    else:
        mu=None
    mu=comm.bcast(mu,root=0)
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
    #ihbar=1.
    len_avec=np.sqrt(np.abs(avec).sum(axis=1))
    ham,expk,no,nr=get_ham(k,rvec,ham_r,ndegen,out_phase=True)
    uni=sclin.eigh(ham)[1]
    vec0=np.array([-1j*ihbar*(ham_r.reshape(no*no,nr)*(r*expk)).sum(axis=1).reshape(no,no)
                    for r in (len_avec*rvec).T])
    vec=np.array([(uni.conjugate().T.dot(v0).dot(uni)).diagonal() for v0 in vec0]).T
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
        eigtmp=np.array([eg[0] for eg in etmp])
        eig=eigtmp.T/mass-mu
        uni=np.array([eg[1] for eg in etmp]).T
        return eig,uni
    else:
        eigtmp=np.array([sclin.eigvalsh(h) for h in ham])
        return (eigtmp.max()/mass-mu),(eigtmp.min()/mass-mu)

def get_col(cl,ol):
    col=(np.abs(cl[ol])**2 if isinstance(ol,int)
         else (np.abs(cl[ol])**2).sum(axis=0)).round(4)
    return col

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
        dkv=np.array(ke)-np.array(ks)
        dkv_length=abs(dkv.dot(bvec)).sum()
        tmp=2.*np.pi*np.linspace(ks,ke,N)
        tmp2=np.linspace(0,dkv_length,N)+maxsplen
        maxsplen=tmp2.max()
        xticks=xticks+[tmp2[0]]
        klist=klist+list(tmp[:-1])
        splen=splen+list(tmp2[:-1])
    klist=klist+[2*np.pi*np.array(k_list[-1])]
    splen=splen+[maxsplen+dkv_length/N]
    xticks=xticks+[splen[-1]]
    return np.array(klist),np.array(splen),xticks

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
    fig=plt.figure()
    ax=plt.axes()
    for e,cl in zip(eig,uni):
        c1=get_col(cl,ol[0])
        c2=get_col(cl,ol[1])
        c3=get_col(cl,ol[2])
        clist=np.array([c1,c2,c3]).T
        plt.scatter(spl,e,s=5,c=clist)
        #for i in range(len(e)):
        #    plt.plot(spl[i:i+2],e[i:i+2],c=clist[i])
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.ylim(emin,emax)
    plt.xlim(0,spl.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

def plot_spectrum(ham,klen,xticks,mu,eta0=5.e-2,de=100,smesh=200):
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
    w_orig=np.linspace(emin*1.1,emax*1.1,de)
    if rank==0:
        sendbuf=w_orig
        cks=divmod(sendbuf.size,size)
        count=np.array([cks[0]+1 if i<cks[1] else cks[0] for i in range(size)])
        displ=np.array([sum(count[:i]) for i in range(size)])
    else:
        sendbuf=None
        count=np.empty(size,dtype=np.int)
        displ=None
    comm.Bcast(count,root=0)
    recvbuf=np.empty(count[rank],dtype='f8')
    comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
    w=recvbuf
    no=len(ham[0])
    nk=len(ham)

    etamax=4.0e0
    #eta=w*0+eta0
    eta=etamax*w*w/min(emax*emax,emin*emin)+eta0
    G=np.array([[-sclin.inv((ww+mu+et*1j)*np.identity(no)-h) for h in ham] for ww,et in zip(w,eta)])
    trG=np.array([[np.trace(gg).imag/(no*no) for gg in g] for g in G])
    sendbuf=trG.flatten()
    w=w_orig
    if rank==0:
        count=count*nk
        recvbuf=np.empty(count.sum(),dtype='f8')
        displ=np.array([count[:i].sum() for i in range(size)])
    else:
        recvbuf=None
        displ=None
    comm.Bcast(count,root=0)
    comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE],root=0)
    if rank==0:
        sp,w=np.meshgrid(klen,w)
        trG=recvbuf.reshape(de,nk)
        plt.hot()
        plt.contourf(sp,w,trG,smesh)
        plt.colorbar()
        for x in xticks[1:-1]:
            plt.axvline(x,ls='-',lw=0.25,color='black')
        plt.xlim(0,klen.max())
        plt.axhline(0.,ls='--',lw=0.25,color='black')
        plt.xticks(xticks,xlabel)
        plt.show()

def make_kmesh(mesh,dim,kz=0,sw=False):
    """
    This function generates square or cube k mesh
    arguments:
    mesh: k-mesh grid size
    kz: kz of plotting FS plane
    """
    km=np.linspace(-np.pi,np.pi,mesh+1,True)
    if dim==2:
        x,y=np.meshgrid(km,km)
        z=y*0.0+kz
    elif dim==3:
        x,y,z=np.meshgrid(km,km,km)
    klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    if sw:
        return klist,x,y
    else:
        return(klist)

def mk_kf(mesh,sw_bnum,dim,rvec,ham_r,ndegen,mu,kz=0):
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
    import skimage.measure as sk
    from mpl_toolkits.mplot3d import axes3d
    Nk,count,k_mpi=gen_klist(mesh,False,dim)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    e_mpi=np.array([sclin.eigvalsh(h) for h in ham])/mass-mu
    sendbuf=e_mpi.flatten()
    if rank==0:
        no=sendbuf.size//count[rank]
        count=count*no
        recvbuf=np.empty(count.sum(),dtype='f8')
        displ=np.array([count[:i].sum() for i in range(size)])
    else:
        recvbuf=None
        displ=None
    comm.Bcast(count,root=0)
    comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE],root=0)
    if rank==0:
        eig=recvbuf.reshape(Nk,no).T
        v2=[]
        if sw_bnum:
            fsband=[]
        for i,e in enumerate(eig):
            if(e.max()*e.min() < 0. ):
                cont=sk.find_contours(e.reshape(mesh+1,mesh+1),0)
                ct0=[]
                for c in cont:
                    ct0.extend(c)
                    ct=(np.array([[c[0],c[1],mesh/2] for c in ct0])-mesh/2)*2*np.pi/mesh
                ct[:,2]=kz
                if sw_bnum:
                    fsband.append(i)
                    v2.append(ct)
                else:
                    v2.extend(ct)
    else:
        v2=None
        fsband=None
    if sw_bnum:
        return v2,fsband
    else:
        return np.array(v2)

def gen_3d_fs_plot(mesh,rvec,ham_r,ndegen,mu,surface_opt=0):
    """
    This function plot 3D Fermi Surface
    argument:
    mesh: k-grid mesh size
    rvec,ham_r,ndegen: model hamiltonian
    mu: chemical potential
    surface_opt: switch of surface color 1:orbital weights, 2:size of velocities
    """
    import skimage.measure as sk
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    import matplotlib.colors as colors
    Nk,count,k_mpi=gen_klist(mesh,False,3)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    e_mpi=np.array([sclin.eigvalsh(h) for h in ham])/mass-mu
    sendbuf=e_mpi.flatten()
    if rank==0:
        no=sendbuf.size//count[rank]
        count=count*no
        recvbuf=np.empty(count.sum(),dtype='f8')
        displ=np.array([count[:i].sum() for i in range(size)])
    else:
        recvbuf=None
        displ=None
    comm.Bcast(count,root=0)
    comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE],root=0)
    if rank==0:
        if surface_opt!=0:
            avev_all=np.zeros(3)
            nk=0
            v_weight=[]
            v_verts=[]
            vc=[]
        clist=['r','g','b','c','m','y','k','w']
        fig=plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        eig=recvbuf.reshape(Nk,no).T
        j=-1
        for i,e in enumerate(eig):
            if(e.max()*e.min() < 0. ):
                j=j+1
                #for old skimage version 
                #verts,faces, _, _=sk.marching_cubes_lewiner(
                #    e.reshape(mesh+1,mesh+1,mesh+1),0,spacing=(2*np.pi/mesh,2*np.pi/mesh,2*np.pi/mesh))
                verts,faces, _, _=sk.marching_cubes(e.reshape(mesh+1,mesh+1,mesh+1),0,
                                                    spacing=(2*np.pi/mesh,2*np.pi/mesh,2*np.pi/mesh))
                surf=sk.mesh_surface_area(verts,faces)
                verts=verts-np.pi
                if surface_opt==0:
                    ax.add_collection3d(Poly3DCollection(verts[faces],facecolor=clist[j%6]))
                else:
                    if surface_opt==2:
                        avev=np.zeros(3)
                    for v in verts[faces]:
                        ave_verts=v.mean(axis=0)
                        if surface_opt==1:
                            htmp=get_ham(ave_verts,rvec,ham_r,ndegen)
                            uni=sclin.eigh(htmp)[1][:,i]
                            cl=np.array(uni)
                            clst=[get_col(cl,ol) for ol in olist]
                            v_weight.append(colors.rgb2hex(clst))
                        else:
                            vtmp=get_vec(ave_verts,rvec,ham_r,ndegen)[i].real
                            avev=avev+np.abs(vtmp)
                            absv=np.abs(vtmp).sum()
                            v_weight.append(absv)
                            vc.append(ave_verts)
                    if(surface_opt==2):
                        print('%.3e %.3e %.3e'%tuple(avev/len(verts[faces])))
                        nk=nk+len(verts[faces])
                        avev_all=avev_all+avev
                    v_verts.extend(verts[faces])
        if(surface_opt==1):
            clist=np.array(v_weight)
            tri=Poly3DCollection(v_verts,facecolors=clist,lw=0)
            ax.add_collection3d(tri)
        if(surface_opt==2):
            vc=np.array(vc)
            v_weight=np.array(v_weight)
            clmax=v_weight.max()
            clmin=v_weight.min()
            clist=(v_weight-clmin)/(clmax-clmin)
            tri=Poly3DCollection(v_verts,facecolors=cm.jet(clist),lw=0)
            ax.add_collection3d(tri)
            fs=ax.scatter(vc[:,0],vc[:,1],vc[:,2],c=v_weight,cmap=cm.jet,s=0.1)
            plt.colorbar(fs,format='%.2e')
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(-np.pi, np.pi)
        plt.tight_layout()
        plt.show()

def plot_vec2(veloc,klist):
    v=[]
    k=[]
    for vv,kk in zip(veloc,klist):
        v0=np.array([np.sqrt((np.abs(v0)**2).sum()) for v0 in vv])
        v.extend(v0)
        k.extend(kk)
    v=np.array(v)
    k=np.array(k)
    plt.scatter(k[:,0],k[:,1],s=1.0,c=v)
    plt.jet()
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.colorbar(format='%.2e')
    plt.show()

def plot_FS(uni,klist,ol,eig,X,Y,sw_color,ncut=8):
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
    def get_col(cl,ol):
        col=(np.abs(cl[:,ol])**2 if isinstance(ol,int)
             else (np.abs(cl[:,ol])**2).sum(axis=1)).round(4)
        return col
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    if sw_color:
        col=['r','g','b','c','m','y','k','w']
        for kk,cl,cb in zip(klist,uni,col):
            cl=np.array(cl)
            c1=get_col(cl,ol[0])
            c2=get_col(cl,ol[1])
            c3=get_col(cl,ol[2])
            clist=np.array([c1,c2,c3]).T
            if(with_spin):
                vud=cl[:,no//2:]*cl[:,:no//2].conjugate()
                vdu=cl[:,:no//2]*cl[:,no//2:].conjugate()
                v1=(vud+vdu).sum(axis=1).real
                v2=(vud-vdu).sum(axis=1).imag
                #v3=(abs(cl[:,:no//2])**2-abs(cl[:,no//2:])**2).sum(axis=1).real
                v1=v1[::ncut].round(4)
                v2=v2[::ncut].round(4)
                #v3=v3[::ncut].round(4)
                k1=kk[::ncut,0]
                k2=kk[::ncut,1]
                plt.quiver(k1,k2,v1,v2,color=cb,angles='xy',scale_units='xy',scale=3.0)
            plt.scatter(kk[:,0],kk[:,1],s=2.0,c=clist)
    else:
        for en in eig:
            if(en.max()*en.min()<0.0):
                plt.contour(X,Y,en.reshape(FSmesh,FSmesh),levels=[0.],color='black')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
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
    no=len(ham[0])
    G=np.array([-sclin.inv((0.+mu+eta*1j)*np.identity(no)-h) for h in ham])
    trG=np.array([np.trace(gg).imag/(no*no) for gg in G]).reshape(FSmesh+1,FSmesh+1)
    #trG=np.array([(gg[4,4]+gg[9,9]).imag/(no*no) for gg in G]).reshape(FSmesh,FSmesh)

    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    ax.set_xticks([-np.pi,0,np.pi])
    ax.set_xticklabels(['-$\pi$','0','$\pi$'])
    ax.set_yticks([-np.pi,0,np.pi])
    ax.set_yticklabels(['-$\pi$','0','$\pi$'])
    cont=ax.contourf(X,Y,trG,smesh,cmap=plt.jet())
    fig.colorbar(cont)
    plt.show()

def get_conductivity(mesh,rvec,ham_r,ndegen,mu,temp):
    """
    this function calculates conductivity at tau==1 from Boltzmann equation in metal
    """
    kb=scconst.physical_constants['Boltzmann constant in eV/K'][0] #temp=kBT[eV], so it need to convert eV>K
    ihbar=1./scconst.physical_constants['Planck constant over 2 pi in eV s'][0]*1.0e-10
    #kb=1.
    eC=scconst.e*1.e10
    Nk,count,k_mpi=gen_klist(mesh)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    dfermi=0.25*(1.-np.tanh(0.5*eig/temp)**2)/temp
    veloc=np.array([get_vec(k,rvec,ham_r,ndegen).real for k in k_mpi])

    l11=np.array([[(vk1*vk2*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])
    l22=kb*np.array([[(vk1*vk2*eig**2*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])
    l12=kb*np.array([[(vk1*vk2*eig*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])
    l11=comm.allreduce(l11,MPI.SUM)
    l22=comm.allreduce(l22,MPI.SUM)
    l12=comm.allreduce(l12,MPI.SUM)

    Seebeck=l12.dot(sclin.inv(l11))/temp
    sigma=eC*l11/(Nk*Vuc)
    kappa=eC*l22/(temp*Nk*Vuc)

    if rank==0:
        print('sigma matrix')
        print(sigma)
        print('Seebeck matrix')
        print(Seebeck)
        print('kappa matrix')
        print(kappa)
        print('lorenz matrix')
        print(kb*kappa/(sigma*temp))

def get_carrier_num(mesh,rvec,ham_r,ndegen,mu):
    Nk,count,k_mpi=gen_klist(mesh)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    for i,en in enumerate(eig):
        num_hole=float(np.where(en>0)[0].size)/Nk
        num_particle=float(np.where(en<=0)[0].size)/Nk

        num_hole=comm.allreduce(num_hole,MPI.SUM)
        num_particle=comm.allreduce(num_particle,MPI.SUM)
        if(rank==0):
            print(i+1,round(num_hole,4),round(num_particle,4))

def plot_dos(mesh,rvec,ham_r,ndegen,mu,no,eta,de=1000):
    Nk,count,k_mpi=gen_klist(mesh)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    eig=np.array([sclin.eigvalsh(h) for h in ham])-mu

    emax=comm.allreduce(eig.max(),MPI.MAX)
    emin=comm.allreduce(eig.min(),MPI.MIN)
    w=np.linspace(emin*1.1,emax*1.1,de)
    dos=np.array([(eta/((ww-eig)**2+eta**2)).sum() for ww in w])/(np.pi*Nk)
    dosef=(eta/(eig**2+eta**2)).sum()/(np.pi*Nk)
    dos=comm.allreduce(dos,MPI.SUM)
    dosef=comm.allreduce(dosef,MPI.SUM)
    if rank==0:
        kb=scconst.physical_constants['Boltzmann constant in eV/K'][0]
        eV2J=scconst.physical_constants['electron volt-joule relationship'][0]
        mol=6.02e23
        gamma=dosef*eV2J*mol*(np.pi*kb)**2/3
        fermi=.5*(1.-np.tanh(.5*w/temp))
        dw=w[1]-w[0]
        print('gamma = %8.2e mJ/(MolK^2)'%(gamma*1e3))
        print(dw*(dos*fermi).sum())
        print(dosef/13.606)
        plt.plot(w,dos*2)
        plt.ylim(0,dos.max()*1.2*2)
        plt.show()

def get_mass(mesh,rvec,ham_r,ndegen,mu,de=3.e-4,meshkz=20):
    import skimage.measure as sk
    if sw_dec_axis:
        al=alatt[:2]
    else:
        al=alatt[:2]
        #al[0]=np.sqrt(alatt[0]**2+alatt[1]**2)*0.5
        #al[1]=al[0]
    #me=1.
    me=scconst.m_e
    #hbar=1.
    hbar=scconst.physical_constants['Planck constant over 2 pi in eV s'][0]*1.e10
    #eV2J=1.
    eV2J=scconst.physical_constants['electron volt-joule relationship'][0]
    Nkh=mesh**2
    ABZ=4.*np.pi**2/(al[0]*al[1])
    def get_k(kx,ky,kz_num):
        if rank==0:
            kz=0.*ky+kz_num
            klist=np.array([kx.ravel(),ky.ravel(),kz.ravel()]).T
            Nk=len(klist)
            sendbuf=klist.flatten()
            cks=divmod(sendbuf.size//3,size)
            count=np.array([3*(cks[0]+1) if i<cks[1] else 3*cks[0] for i in range(size)])
            displ=np.array([sum(count[:i]) for i in range(size)])
        else:
            Nk=None
            sendbuf=None
            count=np.empty(size,dtype=np.int)
            displ=None
        Nk=comm.bcast(Nk,root=0)
        comm.Bcast(count,root=0)
        recvbuf=np.empty(count[rank],dtype='f8')
        comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
        k_mpi=recvbuf.reshape(count[rank]//3,3)
        return k_mpi,count//3,Nk

    def gen_ef_point(eig,con_eig,nb):
        efp=sk.find_contours(eig,con_eig)
        #print(efp)
        if(len(efp)==1):
            return np.array([efp[0]])
        elif(len(efp)==4):
            if(efp[0][0,0]==0. or efp[0][0,1]==0.):
                ef_point=[]
                ef_point.extend(efp[0])
                ef_point.extend(efp[1])
                if(efp[1][-1,1]==efp[3][0,1]):
                    ef_point.extend(efp[3])
                    ef_point.extend(efp[2])
                elif(efp[1][-1,1]==efp[2][0,1]):
                    ef_point.extend(efp[2])
                    ef_point.extend(efp[3])
                return np.array([ef_point])
            else:
                return(np.array(efp))
        elif(len(efp)==7):
            ef_point=[]
            ef_point.extend(efp[0])
            ef_point.extend(efp[1])
            ef_point.extend(efp[6])
            ef_point.extend(efp[5])
            return(np.array([np.array(ef_point),efp[2],efp[3],efp[4]]))
        elif(len(efp)==8):
            ef_point=[]
            ef_point.extend(efp[0])
            ef_point.extend(efp[1])
            ef_point.extend(efp[7])
            ef_point.extend(efp[6])
            return(np.array([np.array(ef_point),efp[2],efp[3],efp[4],efp[5]]))
        else:
            return(np.array(efp))

    k0=np.linspace(-np.pi,np.pi,mesh,False)
    kx,ky=np.meshgrid(k0,k0)
    kz0=np.linspace(-np.pi,np.pi,meshkz,False)
    if False:
        Skz=[]
        for kz_num in kz0:
            k_mpi,count,Nk=get_k(kx,ky,kz_num)
            ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
            eig=np.array([sclin.eigvalsh(h) for h in ham])-mu
            Sb=[]
            for e in eig.T:
                if(e.max()*e.min()<0.):
                    num_hole=float(np.where(e>0)[0].size)
                    num_particle=float(np.where(e<=0)[0].size)
                else:
                    num_hole=0.
                    num_particle=0.
                num_hole=comm.allreduce(num_hole,MPI.SUM)/Nkh
                num_particle=comm.allreduce(num_particle,MPI.SUM)/Nkh
                Sb.append(min(num_hole,num_particle)*ABZ)
            Skz.append(np.array(Sb))
        Skz=np.array(Skz)
        kz_Smax=[]
        S_max=[]
        sband=[]
        kz_Smin=[]
        S_min=[]
        sband2=[]
        for i,sz in enumerate(Skz.T):
            if(sz.max()==0.):
                smax=0
                smin=0
            else:
                sm=np.where(sz==sz.max())[0]
                if rank==0:
                    print('max',sm)
                smax=sm[0]
                sm=np.where(sz==sz[np.where(sz!=0)].min())[0]
                if rank==0:
                    print('min',sm)
                smin=sm[0]
            if(sz[smax]!=0.):
                S_max.append(sz[smax])
                kz_Smax.append(kz0[smax])
                sband.append(i)
            if(sz[smin]!=0.):
                S_min.append(sz[smin])
                kz_Smin.append(kz0[smin])
                sband2.append(i)
        if rank==0:
            print(sband)
            print(sband2)
            print(kz_Smax)
            print(kz_Smin)
    else:
        sband=[]
        sband2=[]
        k_mpi,count,Nk=get_k(kx,ky,kz0[0])
        ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
        eig=np.array([sclin.eigvalsh(h) for h in ham])-mu
        cksf=0
        for i,e in enumerate(eig.T):
            if(e.max()*e.min()<0.):
                ckfs=1
            else:
                ckfs=0
            ckfs=comm.allreduce(ckfs,MPI.SUM)
            if ckfs!=0:
                sband.append(i)
        kz_Smax=kz0[[0]*len(sband)]

        k_mpi,count,Nk=get_k(kx,ky,kz0[10])
        ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
        eig=np.array([sclin.eigvalsh(h) for h in ham])-mu
        for i,e in enumerate(eig.T):
            if(e.max()*e.min()<0.):
                ckfs=1
            else:
                ckfs=0
            ckfs=comm.allreduce(ckfs,MPI.SUM)
            if ckfs!=0:
                sband2.append(i)
        kz_Smin=kz0[[10]*len(sband2)]
    def obtain_mass(Skz,sband,pre_strings):
        #if rank==0:
        #    fig=plt.figure()
        for kz_val,nb in zip(Skz,sband):
            k_mpi,count,Nk=get_k(kx,ky,kz_val)
            ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
            e_mpi=(np.array([sclin.eigvalsh(h) for h in ham])-mu).T[nb]
            sendbuf=e_mpi.flatten()
            if rank==0:
                count=count
                recvbuf=np.empty(count.sum(),dtype='f8')
                displ=np.array([count[:i].sum() for i in range(size)])
            else:
                recvbuf=None
                displ=None
            comm.Bcast(count,root=0)
            comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE],root=0)
            if rank==0:
                #ax=fig.add_subplot(229+nb) #10orb
                #ax=fig.add_subplot(223+nb) #Ba
                #ax=fig.add_subplot(219+nb)
                eig=recvbuf.reshape(mesh,mesh)
                efp=gen_ef_point(eig,de,nb)
                #print(efp)
                S1=[]
                for i,ef_point in enumerate(efp):
                    #if(i==0):
                    #    plt.scatter(ef_point[:,0],ef_point[:,1],s=0.1,c='red')
                    #    for k,(i,j) in enumerate(zip(ef_point,ef_point[1:])):
                    #        col=cm.jet(k/(len(ef_point)-1))
                    #        tri=plt.Polygon(((0.,0.),tuple(i),tuple(j)),facecolor=col,alpha=0.5)
                    #        ax.add_patch(tri)
                    S1.append((np.array([r1[0]*r2[1]-r2[0]*r1[1] for r1,r2 in zip(ef_point[:],ef_point[1:])]).sum()
                        +ef_point[-1,0]*ef_point[0,1]-ef_point[-1,1]*ef_point[0,0])*ABZ*.5/Nkh)
                efp=gen_ef_point(eig,-de,nb)
                #print(efp)
                S2=[]
                for ef_point in efp:
                    #plt.scatter(ef_point[:,0],ef_point[:,1],s=0.1,c='blue')
                    S2.append((np.array([r1[0]*r2[1]-r2[0]*r1[1] for r1,r2 in zip(ef_point,ef_point[1:])]).sum()
                        +ef_point[-1,0]*ef_point[0,1]-ef_point[-1,1]*ef_point[0,0])*ABZ*.5/Nkh)
                mc=hbar**2*abs(np.array(S1)-np.array(S2))*.25/(np.pi*de*me)*eV2J
                for mcc in mc:
                    print(pre_strings,np.round(mcc,4),nb,np.round(kz_val,4))
        #if rank==0:
        #    plt.show()

    def obtain_mass2(Skz,sband,pre_strings):
        for kz_val,nb in zip(Skz,sband):
            k_mpi,count,Nk=get_k(kx,ky,kz_val)
            ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
            eig=(np.array([sclin.eigvalsh(h) for h in ham])-mu).T[nb]
            num_hole=float(np.where((eig-de)>0)[0].size)/Nkh
            num_particle=float(np.where((eig-de)<=0)[0].size)/Nkh
            num_hole=comm.allreduce(num_hole,MPI.SUM)
            num_particle=comm.allreduce(num_particle,MPI.SUM)
            S1=min(num_hole,num_particle)*ABZ
            num_hole=float(np.where((eig+de)>0)[0].size)/Nkh
            num_particle=float(np.where((eig+de)<=0)[0].size)/Nkh
            num_hole=comm.allreduce(num_hole,MPI.SUM)
            num_particle=comm.allreduce(num_particle,MPI.SUM)
            S2=min(num_hole,num_particle)*ABZ
            mc=hbar**2*(S1-S2)/(4.*np.pi*de*me)*eV2J
            if rank==0:
                print(pre_strings,mc,nb)

    obtain_mass(kz_Smax,sband,'m_max')
    obtain_mass(kz_Smin,sband2,'m_min')

def main():
    if rank==0:
        if sw_inp==0: #.input file
            rvec,ndegen,ham_r,no,nr=input_ham.import_out(fname,False)
        elif sw_inp==1: #rvec.txt, ham_r.txt, ndegen.txt files
            rvec,ndegen,ham_r,no,nr=input_ham.import_hop(fname,True,False)
            ham_r=ham_r.flatten()
        elif sw_inp==2:
            rvec,ndegen,ham_r,no,nr=input_ham.import_hr(fname,False)
            ham_r=ham_r.flatten()
        elif sw_inp==3: #Hopping.dat file
            rvec,ndegen,ham_r,no,nr,axis=input_ham.import_Hopping(fname,False,True)
            ham_r=ham_r.flatten()
        else:
            pass
    else:
        no=None
        nr=None
    no=comm.bcast(no,root=0)
    nr=comm.bcast(nr,root=0)
    if rank!=0:
        rvec=np.empty([nr,3],dtype='f8')
        ndegen=np.empty(nr,dtype='f8')
        if sw_inp==0:
            ham_r=np.empty([no,no,nr],dtype='c16')
        else:
            ham_r=np.empty([no*no*nr],dtype='c16')
    comm.Bcast(rvec,root=0)
    comm.Bcast(ndegen,root=0)
    comm.Bcast(ham_r,root=0)
    if sw_inp!=0:
        ham_r=ham_r.reshape(no,no,nr)
    if sw_inp==3:
        if rank!=0:
            axis=np.empty([3,3],dtype='f8')
        comm.Bcast(axis,root=0)
    if sw_calc_mu:
        mu=get_mu(fill,rvec,ham_r,ndegen,temp)
    else:
        try:
            mu
        except NameError:
            mu=get_mu(fill,rvec,ham_r,ndegen)

    if sw_dec_axis:
        rvec1=np.array([Arot.T.dot(r) for r in rvec])
        rvec=rvec1
        rvec[:,2]=rvec[:,2]*2

    if option in (0,1,4,5):
        if option in (0,4):
            klist,spa_length,xticks=mk_klist(k_list,N)
        else: #1,5
            klist,X,Y=make_kmesh(FSmesh,2,kz,sw=True)
        Nk=len(klist)
        if rank==0:
            sendbuf=klist.flatten()
            cks=divmod(sendbuf.size//3,size)
            count=np.array([3*(cks[0]+1) if i<cks[1] else 3*cks[0] for i in range(size)])
            displ=np.array([sum(count[:i]) for i in range(size)])
        else:
            sendbuf=None
            count=np.empty(size,dtype=np.int)
            displ=None
        comm.Bcast(count,root=0)
        recvbuf=np.empty(count[rank],dtype='f8')
        comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
        count=count//3
        k_mpi=recvbuf.reshape(count[rank],3)
        ham_mpi=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
        sendbuf=ham_mpi.flatten()
        if rank==0:
            count=count*no*no
            recvbuf=np.empty(count.sum(),dtype='c16')
            displ=np.array([count[:i].sum() for i in range(size)])
        else:
            recvbuf=None
            displ=None
        comm.Bcast(count,root=0)
        comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE_COMPLEX],root=0)
        if rank==0:
            ham=recvbuf.reshape(Nk,no,no)
            if option in (0,1):
                eig,uni=gen_eig(ham,mass,mu,True)
        else:
            if option==4:
                ham=np.empty([Nk,no,no],dtype='c16')
            else:
                ham=None
            eig=None
            uni=None

    if option==0: #band plot
        if rank==0:
            plot_band(eig,spa_length,xticks,uni,olist)
    elif option==1: #write Fermi surfaces at kz
        klist1,blist=mk_kf(FSmesh,True,2,rvec,ham_r,ndegen,mu,kz)
        if rank==0:
            ham1=np.array([[get_ham(k,rvec,ham_r,ndegen) for k in kk] for kk in klist1])
            uni=np.array([[sclin.eigh(h)[1][:,b] for h in hh] for hh,b in zip(ham1,blist)])
            plot_FS(uni,klist1,olist,eig,X,Y,sw_color)
    elif option==2: #write 3D Fermi surfaces
        if sw_color: #plot orbital weight on 3D Ferrmi surfaces
            gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu,1)
        else:
            gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu)
    elif option==3: #plot size of Fermi velocity on 3D Fermi surfacea
        gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu,2)
    elif option==4: #write Fermi velocity on 2D Fermi surfaces
        klist,blist=mk_kf(FSmesh,True,2,rvec,ham_r,ndegen,mu,kz)
        if rank==0:
            veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
            plot_vec2(veloc,klist)
    elif option==5: #plot spectrum like band plot
        comm.Bcast(ham,root=0)
        plot_spectrum(ham,spa_length,xticks,mu,eta,wmesh)
    elif option==6: #plot spectrum at E=EF
        if rank==0:
            plot_FSsp(ham,mu,X,Y,eta)
    elif option==7: #plot conductivity
        get_conductivity(FSmesh,rvec,ham_r,ndegen,mu,temp)
    elif option==8: #plot dos
        plot_dos(FSmesh,rvec,ham_r,ndegen,mu,no,eta,wmesh)
    elif option==9: #plot carrier number
        get_carrier_num(FSmesh,rvec,ham_r,ndegen,mu)
    elif option==10: 
        get_mass(FSmesh,rvec,ham_r,ndegen,mu,de)
#--------------------------main program-------------------------------
if __name__=="__main__":
    main()
__license__="""Copyright (c) 2018-2019 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
