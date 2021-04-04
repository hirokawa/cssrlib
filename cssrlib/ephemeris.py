# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

import numpy as np
from gnss import uGNSS,rCST,sat2prn,Eph
import datetime

MAX_ITER_KEPLER=30
RTOL_KEPLER=1e-13

def findeph(nav,t,sat,iode=-1):
    dt_p=3600*4
    eph=None
    sys,prn=sat2prn(sat)
    for eph_ in nav:
        if eph_.sat!=sat:
            continue
        if sys==uGNSS.GAL and (eph_.code>>9)&1==0: # I/NAV
            continue
        if eph_.iode==iode:
            eph=eph_
            break
        dt=(t-eph_.toe).total_seconds()
        if abs(dt)<dt_p:
            dt_p=abs(dt)
            eph=eph_
    return eph

def eph2pos(t,eph):
    tk=(t-eph.toe).total_seconds()
    sys,prn=sat2prn(eph.sat)
    c_=rCST.CLIGHT
    if sys==uGNSS.GAL:
        mu=rCST.MU_GAL;omge=rCST.OMGE_GAL
    else: # GPS,QZS
        mu=rCST.MU_GPS;omge=rCST.OMGE
    M=eph.M0+(np.sqrt(mu/(eph.A**3))+eph.deln)*tk
    E=M;Ek=0.0
    for n in range(MAX_ITER_KEPLER):
        Ek=E
        E-=(E-eph.e*np.sin(E)-M)/(1.0-eph.e*np.cos(E))
        if np.abs(E-Ek)<RTOL_KEPLER:
            break
    sE=np.sin(E);cE=np.cos(E)
    u=np.arctan2(np.sqrt(1-eph.e**2)*sE,cE-eph.e)+eph.omg
    r=eph.A*(1-eph.e*cE)
    i=eph.i0+eph.idot*tk
    s2u=np.sin(2*u);c2u=np.cos(2*u)
    u+=eph.cus*s2u+eph.cuc*c2u
    r+=eph.crs*s2u+eph.crc*c2u
    i+=eph.cis*s2u+eph.cic*c2u
    x=r*np.cos(u);y=r*np.sin(u);ci=np.cos(i)
    dOmg=eph.OMG0+(eph.OMGd-omge)*tk-omge*eph.toes
    sO=np.sin(dOmg);cO=np.cos(dOmg)
    rs=np.array([x*cO-y*ci*sO,x*sO+y*ci*cO,y*np.sin(i)])

    tk=(t-eph.toc).total_seconds()
    dts=eph.af0+eph.af1*tk+eph.af2*tk**2
    dts-=2.0*np.sqrt(mu*eph.A)*eph.e*sE/(c_**2)
    return rs,dts     

def eph2clk(time,eph):
    t=(time-eph.toc).total_seconds()
    for i in range(2):
      t-=eph.af0+eph.af1*t+eph.af2*t**2
    dts=eph.af0+eph.af1*t+eph.af2*t**2
    return dts

def ephclk(t,eph,sat):
    dts=eph2clk(t,eph)
    return dts

def satposs(obs,nav):
    n=obs.sat.shape[0]
    rs=np.zeros((n,3))
    dts=np.zeros(n)
    svh=np.zeros(n,dtype=int)
    for i in range(n):
        sat=obs.sat[i]
        pr=obs.P[i,0]
        t=obs.t-datetime.timedelta(seconds=pr/rCST.CLIGHT)
        eph=findeph(nav,t,sat)
        if eph is None:
            svh[i]=1;continue
        svh[i]=eph.svh
        dt=eph2clk(t,eph)
        t-=datetime.timedelta(seconds=dt)
        rs[i,:],dts[i]=eph2pos(t,eph)
    return rs,dts,svh

        