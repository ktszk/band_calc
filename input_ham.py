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
    tmp=[f.split() for f in open(fname+'/irvec.txt','r')]
    rvec=np.array([[float(tt) for tt in tp] for tp in tmp])
    tmp=[f.strip(' ()\n').split(',') for f in open(fname+'/ham_r.txt','r')]
    nr=len(rvec)
    no=int(np.sqrt(len(tmp)/nr))
    tmp1=np.array([complex(float(tp[0]),float(tp[1])) for tp in tmp])
    ham_r=(np.reshape(tmp1,(nr,no,no)) if sw_hoplist else np.reshape(tmp1,(nr,no*no)).T.reshape(no,no,nr))
    ndegen=(np.array([float(f) for f in open(fname+'/ndegen.txt','r')]) if sw_ndegen else np.array([1]*nr))
    return(rvec,ndegen,ham_r,no,nr)

def import_out(fname,sw_hoplist=True):
    tmp=[f.split() for f in open(fname,'r')]
    tmp1=[[float(tp) for tp in tpp] for tpp in tmp]
    tmp=np.array([complex(tp[3],tp[4]) for tp in tmp1])
    r0=tmp1[0][:3]
    con=sum(1 if rr[:3]==r0 else 0  for rr in tmp1)
    no,nr =int(np.sqrt(con)),len(tmp1)//con
    rvec=np.array([tp[:3] for tp in tmp1[:nr]])
    ham_r=(np.reshape(tmp,(no*no,nr)).T.reshape(nr,no,no) if sw_hoplist else np.reshape(tmp,(no,no,nr)))
    ndegen=np.array([1]*nr)
    return(rvec,ndegen,ham_r,no,nr)

def import_hr(name,sw_hoplist=True):
    tmp=[f.split() for f in open('%s_hr.dat'%name,'r')]
    no, nr=int(tmp[1][0]), int(tmp[2][0])
    c1, c2=0,3
    while not c1==nr:
        c1=c1+len(tmp[c2])
        c2=c2+1
    tmp1=[[int(t) for t in tp] for tp in tmp[3:c2]]
    ndegen=[]
    for tp in tmp1:
        ndegen=ndegen+tp
    ndegen=np.array(ndegen)
    tmp1=[[float(t) for t in tp] for tp in tmp[c2:]]
    tmp=np.array([complex(tp[5],tp[6]) for tp in tmp1])
    rvec=np.array([tmp1[no*no*i][:3] for i in range(nr)])
    ham_r=(np.reshape(tmp,(nr,no,no)) if sw_hoplist
           else np.reshape(tmp,(nr,no*no)).T.reshape(no,no,nr))
    return(rvec,ndegen,ham_r,no,nr)

def import_Hopping(sw_hoplist):
    tmp=[f.split() for f in open('Hopping.dat','r')]
    axis=np.array([[float(tp) for tp in tpp] for tpp in tmp[1:4]])
    no,nr=int(tmp[4][0]),int(tmp[4][1])
    ndegen=np.array([1]*nr)
    tmp1=[[float(t) for t in tp] for tp in tmp[7+no:]]
    rvec=np.array([tmp1[no*no*i][:3] for i in range(nr)])
    tmp=np.array([complex(tp[8],tp[9]) for tp in tmp1])
    ham_r=(np.reshape(tmp,(nr,no,no)) if sw_hoplist
           else np.reshape(tmp,(nr,no*no)).T.reshape(no,no,nr))
    return(rvec,ndegen,ham_r,no,nr,axis)
