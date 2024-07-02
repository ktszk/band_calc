#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np

fname='000AsP.input' #hamiltonian file name
sw_inp=0             #setting input hamiltonian format
"""
sw_inp: switch input hamiltonian's format
0: .input file
1: ham_r.txt, irvec.txt, ndegen.txt
2: {case}_hr.dat file (wannier90 default hopping file)
else: Hopping.dat file (ecalj hopping file)
"""
with_spin=False      #use only with soc hamiltonian

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

#k-point settings
N=400                #kmesh btween symmetry points
FSmesh=40            #kmesh for option in {1,2,3,5,6}
wmesh=400            #w-mesh for dos and spectrum

eta=2.0e-2           #eta for green function
sw_dec_axis=False    #transform Cartesian axis
fill=3.00            #band filling
brav=0               #0:sc, 1,2: bc, 3: orthorhombic, 4: monoclinic 5: fcc 6: hexa
sw_calc_mu =True     #switch calc mu or not
temp=3*8.62e-3       #temp if sw_tdep True upper limit of T 100K~8.62e-3eV

sw_color=True        #plot band or FS with orbital weight
sw_unit=True
mu0=10.58            #chemical potential if you want to fix
mass=1.0             #effective mass (reduce band width)
de=1.e-4             #delta for spectrum
kz=np.pi*0.          #kz for option 1 and 6

#parameters for option 0 or 5
sw_T_range=True      #plot figure with pmkbT range around mu

#parameters for option 7
sw_tdep=True         #switch calc. t dep conductivity or not
temp_min=5*8.62e-5   #lower limit of T
tstep=60             #range of temp step
sw_tau=0             #0:constant tau, 1:w dep. tau
sw_mu_const = True
sw_out_Tdep = True
out_name='sigma.txt'
outputs='sigma[0,0]'  #set output various

"""
lattice parameters
alatt: lattice length a,b,c
deg: lattice degree alpha,beta,gamma
if calc rhomb, monocli and tricli,please set deg=[alpha,beta,gamma](degree)
"""

alatt=np.array([3.96*np.sqrt(2.),3.96*np.sqrt(2.),13.02*0.5])
deg=np.array([90.,90.,90.])

"""
olist: list
orbital number with color plot [R,G,B] 
if you merge some orbitals to each color, enter a list of orbital for each element of the olist
"""
olist=[0,[1,2],3]

#----------import modules without scipy-------------
import scipy as sc
import scipy.linalg as sclin
import scipy.optimize as scopt
import scipy.constants as scconst
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpi4py import MPI
import input_ham
#----------------define parameters------------------
comm=MPI.COMM_WORLD
size=comm.Get_size()
rank=comm.Get_rank()

ihbar=1./scconst.physical_constants['Planck constant over 2 pi in eV s'][0]*1.0e-10 if sw_unit else 1.

if brav==0:
    Arot=np.array([[ 1., 0., 0.],[ 0., 1., 0.],[ 0.,0., 1.]]) #rotation matrix for dec. to primitive vector
    k_list=[[0.,0.,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]] #coordinate of sym. points 2
    xlabel=['Z','$\Gamma$','X','Z','$\Gamma$'] #sym. points name  1
    #k_list=[[0.,0.,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,.5,.0],[0.,0.,0.],[.5,.5,0.]] #3 
    #xlabel=['Z','$\Gamma$','X','M','Y','$\Gamma$','M'] #sym. points name 3
elif brav in {1,2}: #bcc
    Arot=np.array([[ 1., 0., 0.],[ 0., 1., 0.],[-.5,-.5, .5]] if brav==1 else
                  [[ .5, -.5, .5],[ .5, .5, .5],[-.5,-.5, .5]])
    k_list=([[0.,0.,.5],[0., 0., 0.],[.5, .5, -.5],[1.,0.,-.5],[0.,0.,0.]] if brav==1 else
            [[.5,.5,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5,-.5],[0.,0.,0.]])
    xlabel=['Z','$\Gamma$','X','M','$\Gamma$']
elif brav==3: #ortho
    Arot=np.array([[ .5, .5, 0.],[-.5, .5, 0.],[ 0.,0., 1.]])
    k_list=[[0.,0.,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,.5,0.],[0.,0.,0.]]
    xlabel=['Z','$\Gamma$','X','M','Y','$\Gamma$']
elif brav==4: #monocli
    Arot=np.array([[ 1., 0., 0.],[ 0., 1., 0.],[np.cos(deg[1]),0., np.sin(deg[1])]])
    k_list=[[0.,0.,.5],[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]]
    xlabel=['Z','$\Gamma$','X','Z','$\Gamma$']
elif brav==5: #fcc
    Arot=np.array([[-.5,0.,.5],[0.,.5,.5],[-.5,.5,0.]])
    k_list=[[0.,0.,0.],[.5, 0., .5],[1., 0., 0.],[.5, .5, .5],[.5,.25,.75],[0.,0.,0.]]
    xlabel=['$\Gamma$','X','$\Gamma$','L','W','$\Gamma$']
elif brav==6: #hexa
    Arot=np.array([[ 1., 0., 0.],[-.5, .5*np.sqrt(3.), 0.],[ 0.,0., 1.]])
    k_list=[[0.,0.,0.],[2./3.,-1./3., 0.],[.5, 0., 0.],[0., 0., 0.],[0.,0.,.5]]
    xlabel=['$\Gamma$','K','M','$\Gamma$','Z']
elif brav==7: #trigonal
    cosg=np.cos(np.pi*deg[2]/180.)
    tx=np.sqrt((1.-cosg)*.5)
    ty=np.sqrt((1.-cosg)/6.)
    tz=np.sqrt((1.+2*cosg)/3.)
    Arot=np.array([[tx,-ty,tz],[0,2*ty,tz],[-tx,-ty,tz]])
    k_list=[[0.,0.,0.],[.5,0.,.5],[.5,0.,0.],[0.,0.,0.],[.5,.5,.5]]
    xlabel=['$\Gamma$','K','M','$\Gamma$','Z']
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
        count=np.empty(size,dtype='i8')
        displ=None
    comm.Bcast(count,root=0)
    Nk=comm.bcast(Nk,root=0)
    recvbuf=np.empty(count[rank],dtype='f8')
    comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
    k_mpi=recvbuf.reshape(count[rank]//3,3)
    return (Nk,count//3,k_mpi)

def get_mu(fill,rvec,ham_r,ndegen,temp=1e-6,mesh=40):
    """
    This function calculates initial chemical potential.
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
    mu=calc_mu(e_mpi,Nk,fill,temp)
    if rank==0:
        print('chemical potential = %6.3f'%mu)
    return mu

def calc_mu(e_mpi,Nk,fill,temp):
    """
    This function obtains chemical potential from energy.
    argments:
    e_mpi: energy of each mpi process.
    Nk: Number of k-points
    fill: band filling (number of particles in band) 
    temp: temperature
    """

    Nall=(2*fill-len(e_mpi))*Nk #calculate (2*filling-Nband)*Nk
    def func(mu): #calculate f(x)=Nall-sum_e tanh((e-mu)/2kbT)
        sum_tanh=np.tanh(0.5*(e_mpi-mu)/temp).sum()
        sum_tanh=comm.allreduce(sum_tanh,MPI.SUM)
        return(Nall+sum_tanh)
    emax=comm.allreduce(e_mpi.max(),MPI.MAX)
    emin=comm.allreduce(e_mpi.min(),MPI.MIN)
    #obtain mu that f(mu)=0
    mu=scopt.brentq(func,emin,emax)
    #mu=scopt.newton(func,0.5*(emin+emax))
    return mu

def get_vec(k,rvec,ham_r,ndegen,avec):
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
    ham,expk,no,nr=get_ham(k,rvec,ham_r,ndegen,out_phase=True)
    uni=sclin.eigh(ham)[1]
    vec0=np.array([-1j*ihbar*(ham_r.reshape(no*no,nr)*(r*expk)).sum(axis=1).reshape(no,no)
                    for r in rvec.T])
    vecb=np.array([(uni.conjugate().T.dot(v0).dot(uni)).diagonal() for v0 in vec0]).T
    vec=np.array([avec.T.dot(vb) for vb in vecb]).real
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

def mk_klist(k_list,N,bvec):
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
        xticks+=[tmp2[0]]
        klist+=tmp[:-1].tolist()
        splen+=tmp2[:-1].tolist()
    klist+=[2*np.pi*np.array(k_list[-1])]
    splen+=[maxsplen+dkv_length/N]
    xticks+=[splen[-1]]
    return np.array(klist),np.array(splen),xticks

def get_BZ_edges(bvec):
    import scipy.spatial as ssp
    r0=range(-1,2)
    x,y,z=np.meshgrid(r0,r0,r0)
    ini_points=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    zero=np.where(abs(ini_points).sum(axis=1)==0)[0]
   
    points=ini_points.dot(bvec)
    voro=ssp.Voronoi(points)
    voro_points=voro.regions[voro.point_region[zero][0]]
    vp=set(voro_points)
    BZ_faces=[]
    for i in voro.ridge_vertices:
        if -1 not in i:
            if len(set(i)&vp)==len(i):
                BZ_faces.append(voro.vertices[i])
    return BZ_faces

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
        #plt.scatter(spl,e,s=5,c=clist)
        for i in range(len(e)):
            plt.plot(spl[i:i+2],e[i:i+2],c=clist[i])
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
    plt.ylim(eig.min()*1.1,eig.max()*1.1)
    plt.xlim(0,spl.max())
    plt.axhline(0.,ls='--',lw=0.25,color='black')
    if sw_T_range:
        plt.axhline(temp,ls='--',lw=0.25,color='black')
        plt.axhline(-temp,ls='--',lw=0.25,color='black')
    plt.xticks(xticks,xlabel)
    plt.show()

def plot_spectrum(ham,klen,xticks,mu,sw_tau,eta0=5.e-2,de=100,smesh=200,etamax=4.0):
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
    if rank==0:
        w_orig=np.linspace(emin*1.1,emax*1.1,de)
        sendbuf=w_orig
        cks=divmod(sendbuf.size,size)
        count=np.array([cks[0]+1 if i<cks[1] else cks[0] for i in range(size)])
        displ=np.array([sum(count[:i]) for i in range(size)])
    else:
        sendbuf=None
        count=np.empty(size,dtype='i8')
        displ=None
    comm.Bcast(count,root=0)
    recvbuf=np.empty(count[rank],dtype='f8')
    comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
    w=recvbuf
    no=len(ham[0])
    nk=len(ham)

    if sw_tau==0:
        eta=w*0+eta0
    elif sw_tau==1:
        eta=etamax*w*w/min(emax*emax,emin*emin)+eta0
    G=np.array([[-sclin.inv((ww+mu+et*1j)*np.identity(no)-h) for h in ham] for ww,et in zip(w,eta)])
    trG=np.array([[np.trace(gg).imag/(no*no) for gg in g] for g in G])
    sendbuf=trG.flatten()
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
        w=w_orig
        sp,w=np.meshgrid(klen,w)
        trG=recvbuf.reshape(de,nk)
        if sw_T_range:
            dfermi=0.25*(1.-np.tanh(0.5*(w)/temp)**2)/temp
            trG=trG*dfermi
        plt.hot()
        plt.contourf(sp,w,trG,smesh)
        plt.colorbar()
        for x in xticks[1:-1]:
            plt.axvline(x,ls='-',lw=0.25,color='white')
        plt.xlim(0,klen.max())
        plt.axhline(0.,ls='--',lw=0.25,color='white')
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

def mk_kf(mesh,rvec,ham_r,ndegen,mu,kz=0):
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
    Nk,count,k_mpi=gen_klist(mesh,False,2)
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
        fsband=[]
        for i,e in enumerate(eig):
            if(e.max()*e.min() < 0. ):
                cont=sk.find_contours(e.reshape(mesh+1,mesh+1),0)
                ct=[[np.array(c.tolist()+[kz]) for c in (cc-mesh/2)*2*np.pi/mesh] for cc in cont]
                fsband.append(i)
                v2.append(ct)
    else:
        v2=None
        fsband=None
    return v2,fsband

def gen_3d_fs_plot(mesh,rvec,ham_r,ndegen,mu,avec,BZ_faces,surface_opt=0):
    """
    This function plot 3D Fermi Surface
    argument:
    mesh: k-grid mesh size
    rvec,ham_r,ndegen: hopping parameters of model hamiltonian
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
                            vtmp=get_vec(ave_verts,rvec,ham_r,ndegen,avec)[i]
                            avev=avev+np.abs(vtmp)
                            absv=np.sqrt((abs(vtmp)**2).sum())
                            #absv=vtmp[2]
                            v_weight.append(absv)
                            vc.append(ave_verts)
                    if(surface_opt==2):
                        aveabsv=np.sqrt(((abs(avev)/len(verts[faces]))**2).sum())
                        print('%.3e %.3e %.3e'%tuple(avev/len(verts[faces]))+' %.3e'%aveabsv)
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
            tri=Poly3DCollection(v_verts,facecolors=cm.jet(clist),edigecolor=(0,0,0,0),lw=0)
            ax.add_collection3d(tri)
            fs=ax.scatter(vc[:,0],vc[:,1],vc[:,2],c=v_weight,cmap=cm.jet,s=0.1)
            plt.colorbar(fs,format='%.2e')
        ax.grid(False)
        ax.set_xlim(-np.pi, np.pi)
        ax.set_ylim(-np.pi, np.pi)
        ax.set_zlim(-np.pi, np.pi)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        plt.tight_layout()
        #for face in BZ_faces:
        #    BZ=Poly3DCollection(face,facecolors=(0,0,0,0),lw=2,edgecolor='k')
        #    ax.add_collection3d(BZ)

        if brav==0:
            BZtops=[[[-np.pi,np.pi],[np.pi,np.pi],[np.pi,np.pi]],[[-np.pi,np.pi],[np.pi,np.pi],[-np.pi,-np.pi]],
                    [[-np.pi,np.pi],[-np.pi,-np.pi],[np.pi,np.pi]],[[-np.pi,np.pi],[-np.pi,-np.pi],[-np.pi,-np.pi]],

                    [[np.pi,np.pi],[-np.pi,np.pi],[np.pi,np.pi]],[[np.pi,np.pi],[-np.pi,np.pi],[-np.pi,-np.pi]],
                    [[-np.pi,-np.pi],[-np.pi,np.pi],[np.pi,np.pi]],[[-np.pi,-np.pi],[-np.pi,np.pi],[-np.pi,-np.pi]],

                    [[np.pi,np.pi],[np.pi,np.pi],[-np.pi,np.pi]],[[np.pi,np.pi],[-np.pi,-np.pi],[-np.pi,np.pi]],
                    [[-np.pi,-np.pi],[np.pi,np.pi],[-np.pi,np.pi]],[[-np.pi,-np.pi],[-np.pi,-np.pi],[-np.pi,np.pi]]]
        elif brav==4:
            da=0.0088435
            dc=0.12914
            BZtops=[[[(1.-da)*np.pi,(1.-da)*np.pi],[-np.pi,np.pi],[np.pi,np.pi]],
                    [[-(1.-da)*np.pi,-(1.-da)*np.pi],[-np.pi,np.pi],[-np.pi,-np.pi]],
                    [[(1-da)*np.pi,(1.-da)*np.pi],[-np.pi,np.pi],[-np.pi,-np.pi]],
                    [[-(1.-da)*np.pi,-(1.-da)*np.pi],[-np.pi,np.pi],[np.pi,np.pi]],

                    [[-(1.-da)*np.pi,(1.-da)*np.pi],[np.pi,np.pi],[np.pi,np.pi]],
                    [[-(1.-da)*np.pi,(1.-da)*np.pi],[np.pi,np.pi],[-np.pi,-np.pi]],
                    [[-(1.-da)*np.pi,(1.-da)*np.pi],[-np.pi,-np.pi],[np.pi,np.pi]],
                    [[-(1.-da)*np.pi,(1.-da)*np.pi],[-np.pi,-np.pi],[-np.pi,-np.pi]],

                    [[(1.+da)*np.pi,(1.-da)*np.pi],[np.pi,np.pi],[-(1.-dc)*np.pi,np.pi]],
                    [[(1.+da)*np.pi,(1.-da)*np.pi],[np.pi,np.pi],[-(1.-dc)*np.pi,-np.pi]],
                    [[(1.+da)*np.pi,(1.-da)*np.pi],[-np.pi,-np.pi],[-(1.-dc)*np.pi,np.pi]],
                    [[(1.+da)*np.pi,(1.-da)*np.pi],[-np.pi,-np.pi],[-(1.-dc)*np.pi,-np.pi]],
                    [[(1.+da)*np.pi,(1.+da)*np.pi],[-np.pi,np.pi],[-(1.-dc)*np.pi,-(1.-dc)*np.pi]],

                    [[-(1.+da)*np.pi,-(1.-da)*np.pi],[np.pi,np.pi],[(1.-dc)*np.pi,-np.pi]],
                    [[-(1.+da)*np.pi,-(1.-da)*np.pi],[np.pi,np.pi],[(1.-dc)*np.pi,np.pi]],
                    [[-(1.+da)*np.pi,-(1.-da)*np.pi],[-np.pi,-np.pi],[(1.-dc)*np.pi,-np.pi]],
                    [[-(1.+da)*np.pi,-(1.-da)*np.pi],[-np.pi,-np.pi],[(1.-dc)*np.pi,np.pi]],
                    [[-(1.+da)*np.pi,-(1.+da)*np.pi],[-np.pi,np.pi],[(1.-dc)*np.pi,(1.-dc)*np.pi]]]
        elif brav in {1,2}:
            cpa=alatt[2]/alatt[0]
            icpa=1./cpa
            ep=icpa**2
            #top plain
            BZtops=[[[-(1.-ep)*np.pi,(1.-ep)*np.pi],[(1-ep)*np.pi,(1-ep)*np.pi],[np.pi,np.pi]],
                    [[-(1.-ep)*np.pi,(1.-ep)*np.pi],[-(1.-ep)*np.pi,-(1.-ep)*np.pi],[np.pi,np.pi]],
                    [[(1.-ep)*np.pi,(1.-ep)*np.pi],[-(1.-ep)*np.pi,(1.-ep)*np.pi],[np.pi,np.pi]],
                    [[-(1.-ep)*np.pi,-(1.-ep)*np.pi],[-(1.-ep)*np.pi,(1.-ep)*np.pi],[np.pi,np.pi]],
                    #bottom plain
                    [[-(1.-ep)*np.pi,(1.-ep)*np.pi],[(1-ep)*np.pi,(1-ep)*np.pi],[-np.pi,-np.pi]],
                    [[-(1.-ep)*np.pi,(1.-ep)*np.pi],[-(1.-ep)*np.pi,-(1.-ep)*np.pi],[-np.pi,-np.pi]],
                    [[(1.-ep)*np.pi,(1.-ep)*np.pi],[-(1.-ep)*np.pi,(1.-ep)*np.pi],[-np.pi,-np.pi]],
                    [[-(1.-ep)*np.pi,-(1.-ep)*np.pi],[-(1.-ep)*np.pi,(1.-ep)*np.pi],[-np.pi,-np.pi]],

                    [[(1.-ep)*np.pi,np.pi],[(1.-ep)*np.pi,np.pi],[np.pi,.5*np.pi]],
                    [[(1.-ep)*np.pi,np.pi],[-(1.-ep)*np.pi,-np.pi],[np.pi,.5*np.pi]],
                    [[-(1.-ep)*np.pi,-np.pi],[(1.-ep)*np.pi,np.pi],[np.pi,.5*np.pi]],
                    [[-(1.-ep)*np.pi,-np.pi],[-(1.-ep)*np.pi,-np.pi],[np.pi,.5*np.pi]],

                    [[(1.-ep)*np.pi,np.pi],[(1.-ep)*np.pi,np.pi],[-np.pi,-.5*np.pi]],
                    [[(1.-ep)*np.pi,np.pi],[-(1.-ep)*np.pi,-np.pi],[-np.pi,-.5*np.pi]],
                    [[-(1.-ep)*np.pi,-np.pi],[(1.-ep)*np.pi,np.pi],[-np.pi,-.5*np.pi]],
                    [[-(1.-ep)*np.pi,-np.pi],[-(1.-ep)*np.pi,-np.pi],[-np.pi,-.5*np.pi]],

                    [[np.pi,(1.-ep)*np.pi],[np.pi,(1.+ep)*np.pi],[.5*np.pi,0.]],
                    [[np.pi,(1.+ep)*np.pi],[np.pi,(1.-ep)*np.pi],[.5*np.pi,0.]],
                    [[np.pi,(1.-ep)*np.pi],[np.pi,(1.+ep)*np.pi],[-.5*np.pi,0.]],
                    [[np.pi,(1.+ep)*np.pi],[np.pi,(1.-ep)*np.pi],[-.5*np.pi,0.]],

                    [[np.pi,(1.-ep)*np.pi],[-np.pi,-(1.+ep)*np.pi],[.5*np.pi,0.]],
                    [[np.pi,(1.+ep)*np.pi],[-np.pi,-(1.-ep)*np.pi],[.5*np.pi,0.]],
                    [[np.pi,(1.-ep)*np.pi],[-np.pi,-(1.+ep)*np.pi],[-.5*np.pi,0.]],
                    [[np.pi,(1.+ep)*np.pi],[-np.pi,-(1.-ep)*np.pi],[-.5*np.pi,0.]],

                    [[-np.pi,-(1.-ep)*np.pi],[np.pi,(1.+ep)*np.pi],[.5*np.pi,0.]],
                    [[-np.pi,-(1.+ep)*np.pi],[np.pi,(1.-ep)*np.pi],[.5*np.pi,0.]],
                    [[-np.pi,-(1.-ep)*np.pi],[np.pi,(1.+ep)*np.pi],[-.5*np.pi,0.]],
                    [[-np.pi,-(1.+ep)*np.pi],[np.pi,(1.-ep)*np.pi],[-.5*np.pi,0.]],

                    [[-np.pi,-(1.-ep)*np.pi],[-np.pi,-(1.+ep)*np.pi],[.5*np.pi,0.]],
                    [[-np.pi,-(1.+ep)*np.pi],[-np.pi,-(1.-ep)*np.pi],[.5*np.pi,0.]],
                    [[-np.pi,-(1.-ep)*np.pi],[-np.pi,-(1.+ep)*np.pi],[-.5*np.pi,0.]],
                    [[-np.pi,-(1.+ep)*np.pi],[-np.pi,-(1.-ep)*np.pi],[-.5*np.pi,0.]],

                    [[(1.-ep)*np.pi,-(1.-ep)*np.pi],[(1.+ep)*np.pi,(1.+ep)*np.pi],[0.,0.]],
                    [[(1.+ep)*np.pi,(1.+ep)*np.pi],[(1.-ep)*np.pi,-(1.-ep)*np.pi],[0.,0.]],
                    [[(1.-ep)*np.pi,-(1.-ep)*np.pi],[-(1.+ep)*np.pi,-(1.+ep)*np.pi],[0.,0.]],
                    [[-(1.+ep)*np.pi,-(1.+ep)*np.pi],[(1.-ep)*np.pi,-(1.-ep)*np.pi],[0.,0.]]]
        if brav in {0,1,2,4}:
            for tops in BZtops:
                ax.plot(tops[0],tops[1],tops[2],ls='-',lw=1.,color='black')

        plt.show()

def plot_vec2(veloc,klist):
    v=[]
    k=[]
    for vl,kl in zip(veloc,klist):
        for vv,kk in zip(vl,kl):
            v0=np.array([np.sqrt((np.abs(v0)**2).sum()) for v0 in vv])
            #for vc,k1,k2 in zip(v0,kk,kk[1:]):
            #    plt.plot([k1[0],k2[0]],[k1[1],k2[1]],c=vc)
            v.extend(v0)
            k.extend(kk)
    v=np.array(v)
    k=np.array(k)
    vmax=v.max()
    vmin=v.min()
    for vl,kl in zip(veloc,klist):
        for vv,kk in zip(vl,kl):
            v0=(np.array([np.sqrt((np.abs(v0)**2).sum()) for v0 in vv])-vmin)/(vmax-vmin)
            for vc1,vc2,k1,k2 in zip(v0,v0[1:],kk,kk[1:]):
                plt.plot([k1[0],k2[0]],[k1[1],k2[1]],c=cm.jet((vc1+vc2)*.5))
    plt.scatter(k[:,0],k[:,1],s=1.0,c=v)
    plt.jet()
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.colorbar(format='%.2e')
    plt.show()

def plot_FS(uni,klist,ol,ncut=8):
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
    col=['r','g','b','c','m','y','k','w']
    for kl,ul,cb in zip(klist,uni,col):
        for kk,cl,in zip(kl,ul):
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
            for k1,k2,clst in zip(kk,kk[1:],clist):
                plt.plot([k1[0],k2[0]],[k1[1],k2[1]],c=clst)
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.show()

def plot_FS_cont(eig,X,Y):
    Nk=len(X)
    fig=plt.figure()
    ax=fig.add_subplot(111,aspect='equal')
    for en in eig:
        if(en.max()*en.min()<0.0):
            plt.contour(X,Y,en.reshape(Nk,Nk),levels=[0.],colors='black')
    plt.xlim(-np.pi,np.pi)
    plt.ylim(-np.pi,np.pi)
    plt.xticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
    plt.yticks([-np.pi,0,np.pi],['-$\pi$','0','$\pi$'])
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

def get_conductivity(sw_tdep,mesh,rvec,ham_r,ndegen,avec,fill,temp_max,temp_min,tstep,sw_tau,idelta=1e-3,tau0=100):
    """
    this function calculates conductivity from Boltzmann equation in metal
    if you set sw_tau we set tau=1fs. else we take tau=1/(1/tau_0+w**2) w:energy
    """
    def calc_Kn(eig,veloc,temp,mu,tau):
        """
        this function obtain Kn
        in here, Kn_ij=sum_k(v_ki*v_kj*(e_k-mu)^n*(-df(e_k)/de)) 
        """
        dfermi=0.25*(1.-np.tanh(0.5*(eig-mu)/temp)**2)/temp #-df(e_k)/de
        K0=np.array([[(vk1*vk2*dfermi*tau).sum() for vk2 in veloc.T] for vk1 in veloc.T])
        K1=np.array([[(vk1*vk2*(eig-mu)*dfermi*tau).sum() for vk2 in veloc.T] for vk1 in veloc.T])
        K2=np.array([[(vk1*vk2*(eig-mu)**2*dfermi*tau).sum() for vk2 in veloc.T] for vk1 in veloc.T])
        K0=comm.allreduce(K0,MPI.SUM)
        K1=comm.allreduce(K1,MPI.SUM)
        K2=comm.allreduce(K2,MPI.SUM)
        return(K0,K1,K2)

    if sw_unit:
        kb=scconst.physical_constants['Boltzmann constant in eV/K'][0] #the unit of temp is kBT[eV], so it need to convert eV>K
        eC=scconst.e #electron charge, it need to convert eV>J (1eV=eCJ)
        tau_u=1.e-15 #unit of tau is sec. default of tau is 1fs
    else:
        kb=1.
        eC=1.
        tau_u=1.
    itau0=1./tau0
    gsp=(1.0 if with_spin else 2.0) #spin weight
    Vuc=sclin.det(avec)*1e-30 #unit is AA^3. Nk*Vuc is Volume of system.

    Nk,count,k_mpi=gen_klist(mesh)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass
    veloc=np.array([get_vec(k,rvec,ham_r,ndegen,avec) for k in k_mpi])/mass

    emin=comm.allreduce(eig.min(),MPI.MIN)
    emax=comm.allreduce(eig.max(),MPI.MAX)
    wlength=np.linspace(emin,emax,300)
    tdf=np.array([[[(v1*v2*tau_u/((w-eig)**2+idelta**2)).sum() for w in wlength]
                   for v1 in veloc.T] for v2 in veloc.T])
    tdf=gsp*comm.allreduce(tdf,MPI.SUM)/Nk
    iNV=1./(Nk*Vuc)
    if rank==0:
        f=open('tdf.dat','w')
        for w,td in zip(wlength,tdf.T):
            f.write('%7.3f '%w)
            for i,d in enumerate(td):
                for dd in d[:i+1]:
                    f.write('%10.3e '%(dd))
            f.write('\n')
        f.close()
        if sw_out_Tdep:
            f=open(out_name,'w')
    if sw_tdep:
        temp0=np.linspace(temp_min,temp_max,tstep)
    else:
        temp0=[temp_max]
    for temp in temp0:
        itemp=1./temp
        mu=(mu0 if sw_mu_const else calc_mu(eig,Nk,fill,temp))
        if sw_tau==0:
            tauw=eig*0+1.
        elif sw_tau==1:
            tauw=1./(itau0+(eig-mu)**2)
        K0,K1,K2=calc_Kn(eig,veloc,temp,mu,tauw)
        sigma=gsp*tau_u*eC*K0*iNV          #sigma=e^2K0 (A/Vm) :1eC is cannceled with eV>J
        #kappa=gsp*tau_u*kb*eC*K2*iNV*itemp #kappa=K2/T (W/Km) :eC(kb) appears with converting eV>J(eV>K)
        kappa=gsp*tau_u*kb*eC*(K2-K1.dot(sclin.inv(K0).dot(K1)))*iNV*itemp #kappa=(K2-K1K0^-1K1)/T
        sigmaS=gsp*tau_u*kb*eC*K1*iNV*itemp #sigmaS=eK1/T (A/mK)
        Seebeck=-kb*sclin.inv(K0).dot(K1)*itemp #S=K0^(-1)K1/eT (V/K) :kb appears with converting eV>K
        Pertier=K1.dot(sclin.inv(K0))           #pi=K1K0^(-1)/e (V:J/C) :eC is cannceled with eV>J
        PF=sigmaS.dot(Seebeck)

        if rank==0:
            '''
            sigma,kappa,sigmaS consistent with boltzwann in cartesian coordinate.
            but S is sign inverted. should we multiply by a minus?
            Lorenz number of free electron is 2.44e-8(WOhmK^-2)
            O(L)~1e-8
            '''
            print('temperature = %4.0d[K]'%int(temp/kb))
            print('mu = %7.3f'%mu)
            print('sigma matrix')
            print(sigma.round(10))
            print('kappa matrix')
            print(kappa.round(10))
            print('sigmaS matrix')
            print(sigmaS.round(10))
            print('Seebeck matrix')
            print(Seebeck.round(10))
            print('Pertier matrix')
            print(Pertier.round(13))
            print('Lorenz matrix')
            print(kb*kappa/(sigma*temp))
            print('Power Factor')
            print(PF.round(10))
            if sw_out_Tdep:
                f.write('%d, %e\n'%(int(temp/kb),eval(outputs)))
    if rank==0:
        if sw_out_Tdep:
            f.close()

def get_carrier_num(mesh,rvec,ham_r,ndegen,mu):
    """
    This function obtains the carrier number from the sum of k-points above or below the chemical potential.
    argments:
    mesh: k-grid mesh size
    rvec,ham_r,ndegen: hopping parameters of model hamiltonian
    mu: chemical potential
    """
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
    """
    This function calculate Dos
    argments:
    mesh: k-grid mesh size
    rvec,ham_r,ndegen: hopping parameters of model hamiltonian
    mu: chemical potential
    no: number of orbitals
    eta: dumping factor
    de: energy mesh (optional: default is 1000)
    """
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
            count=np.empty(size,dtype='i8')
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

def import_hamiltonian(fname,sw_inp):
    if rank==0:
        if sw_inp==0: #.input file
            rvec,ndegen,ham_r,no,nr=input_ham.import_out(fname,False)
        elif sw_inp==1: #rvec.txt, ham_r.txt, ndegen.txt files
            rvec,ndegen,ham_r,no,nr=input_ham.import_hop(fname,True,False)
            ham_r=ham_r.flatten()
        elif sw_inp==2: #from _hr.dat file
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
    return(no,nr,rvec,ham_r,ndegen)

def get_hams(klist,rvec,ham_r,ndegen,no):
    """
    This function obtains hamiltonian Hk
    argments:
    klist: the list of k-points
    rvec,ham_r,ndegen: hopping parameter of model Hamiltonian
    no: the number of orbitals
    """
    Nk=len(klist)
    if rank==0:
        sendbuf=klist.flatten()
        cks=divmod(sendbuf.size//3,size)
        count=np.array([3*(cks[0]+1) if i<cks[1] else 3*cks[0] for i in range(size)])
        displ=np.array([sum(count[:i]) for i in range(size)])
    else:
        sendbuf=None
        count=np.empty(size,dtype='i8')
        displ=None
    comm.Bcast(count,root=0)
    recvbuf=np.empty(count[rank],dtype='f8')
    comm.Scatterv([sendbuf,count,displ,MPI.DOUBLE],recvbuf, root=0)
    count=count//3
    k_mpi=recvbuf.reshape(count[rank],3)
    ham_mpi=np.array([get_ham(k,rvec,ham_r,ndegen) for k in k_mpi])
    sendbuf=ham_mpi.flatten()
    if rank==0:
        count*=no*no
        recvbuf=np.empty(count.sum(),dtype='c16')
        displ=np.array([count[:i].sum() for i in range(size)])
    else:
        recvbuf=None
        displ=None
    comm.Bcast(count,root=0)
    comm.Gatherv(sendbuf,[recvbuf,count,displ,MPI.DOUBLE_COMPLEX],root=0)

    if rank==0:
        ham=recvbuf.reshape(Nk,no,no)
    else:
        ham=np.empty([Nk,no,no],dtype='c16')
    comm.Bcast(ham,root=0)
    return ham

def main():
    no,nr,rvec,ham_r,ndegen=import_hamiltonian(fname,sw_inp)

    if sw_calc_mu:
        mu=get_mu(fill,rvec,ham_r,ndegen,temp)
    else:
        try:
            mu=mu0
        except NameError:
            mu=get_mu(fill,rvec,ham_r,ndegen,temp)
    if option in {2,3,4}:
        bvec=2*np.pi*sclin.inv(alatt*Arot).T
        BZ_faces=get_BZ_edges(bvec)
    if sw_dec_axis:
        rvec1=Arot.T.dot(rvec.T).T
        rvec=rvec1
        if brav in {1,2}:
            rvec[:,2]*=2.
            avec=alatt*np.eye(3)
            avec[:,2]*=.5
        elif brav==5:
            rvec*=2.
            avec=(alatt*np.eye(3))*.5
        else:
            avec=alatt*np.eye(3)
    else:
        avec=alatt*Arot
    bvec=sclin.inv(avec).T
    if rank==0:
        print(avec)
        print(2*np.pi*bvec)
        #arvec=avec.T.dot(rvec.T).T
        #fig=plt.figure()
        #ax=fig.add_subplot(111,projection='3d')
        #ax.scatter(arvec[:,0],arvec[:,1],arvec[:,2])
        #plt.show()
    if option==0: #band plot
        klist,spa_length,xticks=mk_klist(k_list,N,bvec)
        ham=get_hams(klist,rvec,ham_r,ndegen,no)
        if rank==0:
            eig,uni=gen_eig(ham,mass,mu,True)
            plot_band(eig,spa_length,xticks,uni,olist)
    elif option==1: #write Fermi surfaces at kz
        if sw_color:
            klist,blist=mk_kf(FSmesh,rvec,ham_r,ndegen,mu,kz)
            if rank==0:
                ham=[[[get_ham(k,rvec,ham_r,ndegen) for k in klsurf] for klsurf in klb] for klb in klist]
                uni=[[np.array([sclin.eigh(h)[1][:,b] for h in hh]) for hh in hb] for hb,b in zip(ham,blist)]
                plot_FS(uni,klist,olist)
        else:
            klist,X,Y=make_kmesh(FSmesh,2,kz,sw=True)
            ham=get_hams(klist,rvec,ham_r,ndegen,no)
            if rank==0:
                eig,uni=gen_eig(ham,mass,mu,True)
                plot_FS_cont(eig,X,Y)
    elif option==2: #write 3D Fermi surfaces
        if sw_color: #plot orbital weight on 3D Ferrmi surfaces
            gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu,avec,BZ_faces,1)
        else:
            gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu,avec,BZ_faces)
    elif option==3: #plot size of Fermi velocity on 3D Fermi surfacea
        gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu,avec,BZ_faces,2)
    elif option==4: #write Fermi velocity on 2D Fermi surfaces
        klist,blist=mk_kf(FSmesh,rvec,ham_r,ndegen,mu,kz)
        if rank==0:
            veloc=[[np.array([get_vec(k,rvec,ham_r,ndegen,avec)[b] for k in kk]) for kk in kb]
                   for b,kb in zip(blist,klist)]
            plot_vec2(veloc,klist)
    elif option==5: #plot spectrum like band plot
        klist,spa_length,xticks=mk_klist(k_list,N,bvec)
        ham=get_hams(klist,rvec,ham_r,ndegen,no)
        plot_spectrum(ham,spa_length,xticks,mu,sw_tau,eta,wmesh)
    elif option==6: #plot spectrum at E=EF
        if rank==0:
            plot_FSsp(ham,mu,X,Y,eta)
    elif option==7: #plot conductivity
        get_conductivity(sw_tdep,FSmesh,rvec,ham_r,ndegen,avec,fill,temp,temp_min,tstep,sw_tau)
    elif option==8: #plot dos
        plot_dos(FSmesh,rvec,ham_r,ndegen,mu,no,eta,wmesh)
    elif option==9: #plot carrier number
        get_carrier_num(FSmesh,rvec,ham_r,ndegen,mu)
    elif option==10: 
        get_mass(FSmesh,rvec,ham_r,ndegen,mu,de)
#--------------------------main program-------------------------------
if __name__=="__main__":
    main()
__license__="""Copyright (c) 2018-2024 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
