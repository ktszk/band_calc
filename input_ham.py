#!/usr/bin/env python
#-*- coding:utf-8 -*-
from __future__ import print_function, division
import numpy as np
"""
fname: input file name
sw_hoplist: switch hopping order if True [nr,no,no] else [no,no,nr] default True
no is number of orbitals. nr is number of hopping matrices. 
sw_ndegen: import_hop only, select lead ndegen from ndegen.txt (True) or not (False) default False
"""
def import_hop(fname,sw_ndegen=False,sw_hoplist=True):
    rvec=np.loadtxt(fname+'/irvec.txt')
    nr=rvec[:,0].size
    tmp=np.array([complex(float(tp[0]),float(tp[1])) for tp in [f.strip(' ()\n').split(',') for f in open(fname+'/ham_r.txt','r')]])
    no=int(np.sqrt(tmp.size/nr))
    ham_r=(tmp.reshape(nr,no,no) if sw_hoplist else np.reshape(tmp,(nr,no*no)).T.reshape(no,no,nr))
    ndegen=(np.loadtxt(fname+'/ndegen.txt') if sw_ndegen else np.ones(nr))
    return(rvec,ndegen,ham_r,no,nr)

def import_out(fname,sw_hoplist=True):
    data=np.loadtxt(fname)
    con=(data[:,:3]==data[0,:3]).prod(axis=1).sum()
    no,nr =int(np.sqrt(con)),data[:,0].size//con
    rvec=np.array(data[:nr,:3])
    ham_r=(data[:,3]+1j*data[:,4]).reshape((nr,no,no) if sw_hoplist else (no,no,nr))
    ndegen=np.ones(nr)
    return(rvec,ndegen,ham_r,no,nr)

def import_hr(name,sw_hoplist=True):
    tmp=[f.split() for f in open('%s_hr.dat'%name,'r')]
    no, nr=int(tmp[1][0]), int(tmp[2][0])
    c2,tmp1=3,[]
    while not len(tmp1)==nr:
        tmp1.extend(tmp[c2])
        c2=c2+1
    ndegen=np.array([int(t) for t in tmp1])
    tmp1=[[float(t) for t in tp] for tp in tmp[c2:]]
    tmp=np.array([complex(tp[5],tp[6]) for tp in tmp1])
    rvec=np.array([tmp1[no*no*i][:3] for i in range(nr)])
    ham_r=(np.reshape(tmp,(nr,no,no)) if sw_hoplist
           else np.reshape(tmp,(nr,no*no)).T.reshape(no,no,nr))
    return(rvec,ndegen,ham_r,no,nr)

def import_Hopping(name,sw_hoplist=True,sw_axis=False):
    tmp=[f.split() for f in open('%s/Hopping.dat'%name,'r')]
    axis=np.array([[float(tp) for tp in tpp] for tpp in tmp[1:4]])
    no,nr=int(tmp[4][0]),int(tmp[4][1])
    ndegen=np.ones(nr,dtype='f8')
    tmp1=np.array([[float(t) for t in tp] for tp in tmp[7+no:]])
    rvec=np.array([tmp1[no*no*i][:3] for i in range(nr)])
    tmp=np.array([complex(tp[8],tp[9]) for tp in tmp1])
    ham_r=(np.reshape(tmp,(nr,no,no)) if sw_hoplist
           else np.reshape(tmp,(nr,no*no)).T.reshape(no,no,nr))
    if sw_axis:
        return(rvec,ndegen,ham_r,no,nr,axis)
    else:
        return(rvec,ndegen,ham_r,no,nr)
