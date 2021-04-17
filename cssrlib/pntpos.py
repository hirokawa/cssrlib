# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
from gnss import uGNSS,rCST,rSIG,prn2sat,sat2prn,Eph,gpst2time,time2gpst,sat2id,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav
from ephemeris import findeph,eph2pos,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt

MAXITR=10
ELMIN=10
NX=4

def rescode(itr,obs,nav,rs,dts,svh,x):
    nv=0
    n=obs.sat.shape[0]
    rr=x[0:3]
    dtr=x[3]
    pos=ecef2pos(rr)
    v=np.zeros(n)
    H=np.zeros((n,NX))
    az=np.zeros(n)
    el=np.zeros(n)    
    for i in range(n):
        sys,prn=sat2prn(obs.sat[i])
        if np.linalg.norm(rs[i,:])<rCST.RE_WGS84:
            continue
        r,e=geodist(rs[i,:],rr)
        az[i],el[i]=satazel(pos,e)
        if el[i]<np.deg2rad(ELMIN):
            continue
        eph=findeph(nav.eph,obs.t,obs.sat[i])
        P=obs.P[i,0]-eph.tgd*rCST.CLIGHT
        dion=ionmodel(obs.t,pos,az[i],el[i],nav.ion)
        dtrp=tropmodel(obs.t,pos,el[i])
        v[nv]=P-(r+dtr-rCST.CLIGHT*dts[i]+dion+dtrp)
        H[nv,0:3]=-e;H[nv,3]=1
        nv+=1

    return v,H,nv,az,el

def estpos(obs,nav,rs,dts,svh,rr):
    sol=[]
    n=obs.sat.shape[0]

    var=np.zeros(n+4)
    x=np.zeros(NX)
    dx=np.zeros(NX)
    Q=np.zeros((NX,NX))    
    x[0:3]=rr
    
    for itr in range(MAXITR):
        v,H,nv,az,el=rescode(itr,obs,nav,rs,dts,svh,x)
        if itr==0:
            x[3]=np.mean(v)
            v-=x[3]
        dx=np.linalg.lstsq(H,v,rcond=None)[0]
        x+=dx
        if np.linalg.norm(dx)<1e-4:
            break
    return x,az,el



def pntpos(obs,nav,rr):
    n=obs.sat.shape[0]
    rs,vs,dts,svh=satposs(obs,nav)
    sol,az,el=estpos(obs,nav,rs,dts,svh,rr)

    return sol,az,el

if __name__ == '__main__':
    bdir='C:/work/gps/cssrlib/data/'

    xyz_ref=[-3962108.6754,   3381309.5308,   3668678.6346]
    pos_ref=ecef2pos(xyz_ref)

    navfile=bdir+'SEPT0781.21P'
    obsfile=bdir+'SEPT0782s.21O'

    dec = rnxdec()
    nav = Nav()
    dec.decode_nav(navfile,nav)
    nep=3600//30
    t=np.zeros(nep)
    enu=np.zeros((nep,3))
    sol=np.zeros((nep,4))
    dop=np.zeros((nep,4))
    if dec.decode_obsh(obsfile)>=0:
        rr=dec.pos
        pos=ecef2pos(rr)
        for ne in range(nep):
            obs=dec.decode_obs()
            week,tow=time2gpst(obs.t)
            if ne==0:
                t0=obs.t
            t[ne]=(obs.t-t0).total_seconds()
            sol[ne,:],az,el=pntpos(obs,nav,rr)
            dop[ne,:]=dops(az,el)
            enu[ne,:]=ecef2enu(pos_ref,sol[ne,0:3]-xyz_ref)
        dec.fobs.close()
    
    plt.plot(t,enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east','north','up'])
    plt.grid()
    plt.axis([0,3600,-6,6])
    


    
    



                    
