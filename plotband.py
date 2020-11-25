#!/usr/bin/env python
#-*- coding:utf-8 -*-
import numpy as np

#fname='000AsP.input' #hamiltonian file name
fname='Cu' #hamiltonian file name
mu=9.8               #chemical potential
mass=1.0             #effective mass
sw_inp=2             #input hamiltonian format
"""
sw_inp: switch input hamiltonian's format
0: .input file
1: ham_r.txt, irvec.txt, ndegen.txt
2: {case}_hr.dat file (wannier90 default hopping file)
else: Hopping.dat file (ecalj hopping file)
"""

option=7
"""
option: switch calculation modes
0: band plot
1: write Fermi surface at kz=0
2: write 3D Fermi surface
3: write Fermi velocity with Fermi surface
4: plot spectrum like band plot
5: plot spectrum at E=EF
6: plot 3D Fermi velocity with Fermi surface
7: calc conductivity
8: plot Dos
9: calc carrier num.
"""

sw_calc_mu =True
fill=5.50

alatt=np.array([1.,1.,1.]) #Bravais lattice parameter a,b,c
#alatt=np.array([3.96*np.sqrt(2.),3.96*np.sqrt(2.),13.02*0.5]) #Bravais lattice parameter a,b,c
Arot=np.array([[ .5,-.5, .5],[ .5, .5, .5],[-.5,-.5, .5]]) #rotation matrix for dec. to primitive vector
k_list=[[0., 0., 0.],[.5, 0., 0.],[.5, .5, 0.],[0.,0.,0.]] #coordinate of sym. points
xlabel=['$\Gamma$','X','M','$\Gamma$'] #sym. points name

olist=[1,2,3]        #orbital number with color plot [R,G,B] if you merge some orbitals input orbital list in elements
N=80                #kmesh btween symmetry points
FSmesh=80           #kmesh for option in {1,2,3,5,6}
eta=5.0e-3           #eta for green function
sw_dec_axis=False    #transform Cartesian axis
sw_color=True        #plot band or FS with orbital weight
kz=np.pi*0.
with_spin=False #use only with soc hamiltonian

#----------import modules without scipy-------------
import scipy.linalg as sclin
import scipy.optimize as scopt
import scipy.constants as scconst
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import input_ham
#----------------define functions-------------------
if sw_dec_axis:
    pass
else:
    avec=alatt*Arot
    len_avec=np.sqrt((abs(avec)**2).sum(axis=1))
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
    km=np.linspace(0,2*np.pi,mesh,False)
    x,y,z=np.meshgrid(km,km,km)
    klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T
    f=lambda mu: 2.*fill*mesh**3+(np.tanh(0.5*(eig-mu)/temp)-1.).sum()
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
    #ihbar=1.
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
        dkv_length=np.sqrt(((dkv*len_avec)**2).sum())
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
    def get_col(cl,ol):
        col=(np.abs(cl[ol])**2 if isinstance(ol,int)
             else (np.abs(cl[ol])**2).sum(axis=0)).round(4)
        return col
    for e,cl in zip(eig,uni):
        c1=get_col(cl,ol[0])
        c2=get_col(cl,ol[1])
        c3=get_col(cl,ol[2])
        clist=np.array([c1,c2,c3]).T
        plt.scatter(spl,e,s=5,c=clist)
    for x in xticks[1:-1]:
        plt.axvline(x,ls='-',lw=0.25,color='black')
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
    w=np.linspace(emin*1.1,emax*1.1,de)
    no=len(ham[0])
    #eta=w*0+eta0
    etamax=4.0e0
    eta=etamax*w*w/min(emax*emax,emin*emin)+eta0
    G=np.array([[-sclin.inv((ww+mu+et*1j)*np.identity(no)-h) for h in ham] for ww,et in zip(w,eta)])
    trG=np.array([[np.trace(gg).imag/(no*no) for gg in g] for g in G])
    sp,w=np.meshgrid(klen,w)
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
    This function generates square k mesh for 2D spectrum plot
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
        return(klist,x,y)
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
    klist=make_kmesh(mesh,dim,kz)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    v2=[]
    if sw_bnum:
        fsband=[]
    for i,e in enumerate(eig):
        if(e.max()*e.min() < 0. ):
            if dim==2:
                cont=sk.find_contours(e.reshape(mesh+1,mesh+1),0)
                ct0=[]
                for c in cont:
                    ct0.extend(c)
                ct=(np.array([[c[0],c[1],+mesh/2] for c in ct0])-mesh/2)*2*np.pi/mesh
                ct[:,2]=kz
                if sw_bnum:
                    fsband.append(i)
                    v2.append(ct)
                else:
                    v2.extend(ct)
            elif dim==3:
                vertices,faces,normals,values=sk.marching_cubes_lewiner(e.reshape(mesh+1,mesh+1,mesh+1),0)
                if sw_bnum:
                    fsband.append(i)
                    v2.append((vertices-mesh/2)*2*np.pi/mesh)
                    #v3.append(faces)
                else:
                    v2.extend((2*np.pi*(vertices-mesh/2)/mesh)[faces])
    if sw_bnum:
        return v2,fsband
    else:
        return np.array(v2)

def gen_3d_fs_plot(mesh,rvec,ham_r,ndegen,mu):
    """
    This function plot 3D Fermi Surface
    argument:
    mesh: k-grid mesh size
    """
    from mpl_toolkits.mplot3d import axes3d
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    vert=mk_kf(mesh,False,3,rvec,ham_r,ndegen,mu)
    fig=plt.figure()
    ax=fig.add_subplot(111,projection='3d')
    m = Poly3DCollection(vert)
    ax.add_collection3d(m)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
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
        ave_vx=np.abs(np.array(v).T[0]).mean()
        ave_vy=np.abs(np.array(v).T[1]).mean()
        ave_vz=np.abs(np.array(v).T[2]).mean()
        print('%.3e %.3e %.3e'%(ave_vx,ave_vy,ave_vz))
        vf.extend(v)
        kf.extend(k)
    x,y,z=zip(*np.array(kf))
    vf=np.array(vf)
    ave_vx=np.abs(vf.T[0]).mean()
    ave_vy=np.abs(vf.T[1]).mean()
    ave_vz=np.abs(vf.T[2]).mean()
    print('%.3e %.3e %.3e'%(ave_vx,ave_vy,ave_vz))
    absv=np.array([np.abs(v).sum() for v in vf])
    fs=ax.scatter(x,y,z,c=absv,cmap=cm.jet)
    ax.set_xlim(-np.pi, np.pi)
    ax.set_ylim(-np.pi, np.pi)
    ax.set_zlim(-np.pi, np.pi)
    plt.colorbar(fs,format='%.2e')
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
             else (np.abs(cl[:,ol])**2).sum(axis=0)).round(4)
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

def get_conductivity(mesh,rvec,ham_r,ndegen,mu,temp=1.0e-3):
    """
    this function calculates conductivity at tau==1 from Boltzmann equation in metal
    """
    kb=scconst.physical_constants['Boltzmann constant in eV/K'][0] #temp=kBT[eV], so it need to convert eV>K
    #kb=1.
    km=np.linspace(0,2*np.pi,mesh,False)
    x,y,z=np.meshgrid(km,km,km)
    klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    Nk=len(klist)
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    dfermi=0.25*(1.-np.tanh(0.5*eig/temp)**2)/temp
    veloc=np.array([get_vec(k,rvec,ham_r,ndegen).real for k in klist])
    sigma=np.array([[(vk1*vk2*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])/Nk
    l12=kb*np.array([[(vk1*vk2*eig*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])/(temp*Nk))
    kappa=kb*np.array([[(vk1*vk2*eig**2*dfermi).sum() for vk2 in veloc.T] for vk1 in veloc.T])/(temp*Nk)
    Seebeck=l12.dot(sclin.inv(sigma))
    print('sigma matrix')
    print(sigma)
    print('Seebeck matrix')
    print(Seebeck)
    print('kappa matrix')
    print(kappa)
    print('lorenz matrix')
    print(kb*kappa/(sigma*temp))

def get_carrier_num(mesh,rvec,ham_r,ndegen,mu):
    km=np.linspace(0,2*np.pi,mesh,False)
    x,y,z=np.meshgrid(km,km,km)
    klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    Nk=klist.size/3
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=np.array([sclin.eigvalsh(h) for h in ham]).T/mass-mu
    for i,en in enumerate(eig):
        num_hole=float(np.where(en>0)[0].size)/Nk
        num_particle=float(np.where(en<=0)[0].size)/Nk
        print(i+1,round(num_hole,4),round(num_particle,4))

def plot_dos(mesh,rvec,ham_r,ndegen,mu,no,eta,de=200):
    km=np.linspace(0,2*np.pi,mesh,False)
    x,y,z=np.meshgrid(km,km,km)
    klist=np.array([x.ravel(),y.ravel(),z.ravel()]).T
    ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
    eig=np.array([sclin.eigvalsh(h) for h in ham])-mu
    emax=eig.max()
    emin=eig.min()
    w=np.linspace(emin*1.1,emax*1.1,de)
    dos=np.array([(eta/((ww-eig)**2+eta**2)).sum() for ww in w])/len(klist)*eta
    plt.plot(w,dos)
    plt.ylim(0,dos.max()*1.2)
    plt.show()
def main():
    if sw_inp==0: #.input file
        rvec,ndegen,ham_r,no,nr=input_ham.import_out(fname,False)
    elif sw_inp==1: #rvec.txt, ham_r.txt, ndegen.txt files
        rvec,ndegen,ham_r,no,nr=input_ham.import_hop(fname,True,False)
    elif sw_inp==2:
        rvec,ndegen,ham_r,no,nr=input_ham.import_hr(fname,False)
    else: #Hopping.dat file
        rvec,ndegen,ham_r,no,nr,axis=input_ham.import_Hopping(fname,False,True)

    if sw_calc_mu:
        mu=get_mu(fill,rvec,ham_r,ndegen)
    else:
        try:
            mu
        except NameError:
            mu=get_mu(fill,rvec,ham_r,ndegen)

    if sw_dec_axis:
        rvec1=np.array([Arot.T.dot(r) for r in rvec])
        rvec=rvec1

    if option in (0,1,4,5):
        if option in (0,4):
            klist,spa_length,xticks=mk_klist(k_list,N)
        else: #1,5
            klist,X,Y=make_kmesh(FSmesh,2,kz=0,sw=True)
        ham=np.array([get_ham(k,rvec,ham_r,ndegen) for k in klist])
        if option in (0,1):
            eig,uni=gen_eig(ham,mass,mu,True)

    if option==0: #band plot
        plot_band(eig,spa_length,xticks,uni,olist)
    elif option==1: #write Fermi surface at kz=0
        klist1,blist=mk_kf(FSmesh,True,2,rvec,ham_r,ndegen,mu,kz)
        ham1=np.array([[get_ham(k,rvec,ham_r,ndegen) for k in kk] for kk in klist1])
        uni=np.array([[sclin.eigh(h)[1][:,b] for h in hh] for hh,b in zip(ham1,blist)])
        plot_FS(uni,klist1,olist,eig,X,Y,sw_color)
    elif option==2: #write 3D Fermi surface
        gen_3d_fs_plot(FSmesh,rvec,ham_r,ndegen,mu)
    elif option==3: #write Fermi velocity with Fermi surface
        klist,blist=mk_kf(FSmesh,True,2,rvec,ham_r,ndegen,mu,kz)
        veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
        plot_vec2(veloc,klist)
    elif option==4: #plot spectrum like band plot
        plot_spectrum(ham,spa_length,xticks,mu,eta)
    elif option==5: #plot spectrum at E=EF
        plot_FSsp(ham,mu,X,Y,eta)
    elif option==6: #plot 3D Fermi velocity with Fermi surface
        klist,blist=mk_kf(FSmesh,True,3,rvec,ham_r,ndegen,mu)
        veloc=[[get_vec(k,rvec,ham_r,ndegen)[b].real for k in kk] for b,kk in zip(blist,klist)]
        plot_veloc_FS(veloc,klist)
    elif option==7:
        get_conductivity(FSmesh,rvec,ham_r,ndegen,mu,temp=1.0e-3)
    elif option==8:
        plot_dos(FSmesh,rvec,ham_r,ndegen,mu,no,eta)
    elif option==9:
        get_carrier_num(FSmesh,rvec,ham_r,ndegen,mu)
#--------------------------main program-------------------------------
if __name__=="__main__":
    main()
__license__="""Copyright (c) 2018-2019 K. Suzuki
Released under the MIT license
http://opensource.org/licenses/mit-license.php
"""
