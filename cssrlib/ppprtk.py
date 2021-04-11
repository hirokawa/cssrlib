# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
from gnss import rCST,sat2prn,time2gpst,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav
from ephemeris import findeph,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt
from cssrlib.cssrlib import cssr

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
    x=np.zeros(NX)
    dx=np.zeros(NX)  
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
    rs,dts,svh=satposs(obs,nav)
    sol,az,el=estpos(obs,nav,rs,dts,svh,rr)
    return sol,az,el

if __name__ == '__main__':
    bdir='../data/'
    l6file=bdir+'2021078M.l6'
    griddef=bdir+'clas_grid.def'
    
    # based on GSI F5 solution
    xyz_ref=[-3962108.6754,   3381309.5308,   3668678.6346]
    pos_ref=ecef2pos(xyz_ref)

    navfile=bdir+'SEPT0781.21P'
#    obsfile=bdir+'SEPT0782s.21O'
    obsfile=bdir+'SEPT0781.21O'

    cs=cssr()
    cs.monlevel=2
    cs.week=2149
    cs.read_griddef(griddef)

    dec = rnxdec()
    nav = Nav()
    nav=dec.decode_nav(navfile,nav)
    #nep=3600//30
    nep=30
    t=np.zeros(nep)
    enu=np.zeros((nep,3))
    sol=np.zeros((nep,4))
    dop=np.zeros((nep,4))
    if dec.decode_obsh(obsfile)>=0:
        rr=dec.pos
        pos=ecef2pos(rr)
        inet=cs.find_grid_index(pos)
        
        fc=open(bdir+l6file,'rb')
        if not fc:
            print("L6 messsage file cannot open."); exit(-1)
        for ne in range(nep):
            cs.decode_l6msg(fc.read(250),0)
            if cs.fcnt==5:
                cs.decode_cssr(cs.buff,0)            
            obs=dec.decode_obs()
            week,tow=time2gpst(obs.t)

            if ne==0:
                t0=obs.t
            t[ne]=(obs.t-t0).total_seconds()
            sol[ne,:],az,el=pntpos(obs,nav,rr)
            dop[ne,:]=dops(az,el)
            enu[ne,:]=ecef2enu(pos_ref,sol[ne,0:3]-xyz_ref)
            
            if ne>=20:
                inet=cs.find_grid_index(pos)
                dlat,dlon=cs.get_dpos(pos)
                trph,trpw=cs.get_trop(dlat,dlon)
                stec=cs.get_stec(dlat,dlon)
            
        fc.close()
        dec.fobs.close()
    
    plt.plot(t,enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east','north','up'])
    plt.grid()
    plt.axis([0,3600,-6,6])
    


    
    



                    
