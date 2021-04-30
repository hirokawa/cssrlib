# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import gnss as gn
from gnss import rCST,sat2prn,time2gpst,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav,timediff
from ephemeris import findeph,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt
from cssrlib.cssrlib import cssr

from rtk import IB,zdres,ddres,kfupdate,resamb_lambda,valpos,holdamb

MAXITR=10
ELMIN=10
NX=4

def rtkinit(nav,pos0=np.zeros(3)):
    nav.nf=2
    nav.pmode=0 # 0:static, 1:kinematic

    nav.na=3 if nav.pmode==0 else 6
    nav.ratio=0
    nav.thresar=[2]
    nav.nx=nav.na+gn.uGNSS.MAXSAT*nav.nf
    nav.x=np.zeros(nav.nx)
    nav.P=np.zeros((nav.nx,nav.nx))
    nav.xa=np.zeros(nav.na)
    nav.Pa=np.zeros((nav.na,nav.na))
    nav.nfix=nav.neb=0
    nav.eratio=[100,100]
    nav.err=[0,0.003,0.003]
    nav.sig_p0 = 30.0
    nav.sig_v0 = 10.0
    nav.sig_n0 = 30.0
    nav.sig_qp=0.1
    nav.sig_qv=0.01
    #
    nav.x[0:3]=pos0
    di = np.diag_indices(6)
    nav.P[di[0:3]]=nav.sig_p0**2
    nav.q=np.zeros(nav.nx)
    nav.q[0:3]=nav.sig_qp**2    
    if nav.pmode>=1:
        nav.P[di[3:6]]=nav.sig_v0**2
        nav.q[3:6]=nav.sig_qv**2 
    # obs index
    i0={gn.uGNSS.GPS:0,gn.uGNSS.GAL:0,gn.uGNSS.QZS:0}
    i1={gn.uGNSS.GPS:1,gn.uGNSS.GAL:2,gn.uGNSS.QZS:2}
    freq0={gn.uGNSS.GPS:nav.freq[0],gn.uGNSS.GAL:nav.freq[0],gn.uGNSS.QZS:nav.freq[0]}
    freq1={gn.uGNSS.GPS:nav.freq[1],gn.uGNSS.GAL:nav.freq[2],gn.uGNSS.QZS:nav.freq[1]}
    nav.obs_idx=[i0,i1]
    nav.obs_freq=[freq0,freq1]

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

def udstate_ppp(nav,obs):
    tt=1.0

    ns=len(obs.sat)
    sys=[]
    sat=obs.sat
    for sat_i in obs.sat:
        sys_i,prn=gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel
    na=nav.na
    if nav.pmode>=1:
        F=np.eye(na)
        F[0:3,3:6]=np.eye(3)*tt
        nav.x[0:3]+=tt*nav.x[3:6]
        Px=nav.P[0:na,0:na]
        Px=F.T@Px@F
        Px[np.diag_indices(nav.na)]+=nav.q[0:nav.na]*tt
        nav.P[0:na,0:na]=Px
    # bias
    for f in range(nav.nf):
        bias=np.zeros(ns)
        offset=0
        na=0
        for i in range(ns):
            if sys[i] not in nav.gnss_t:
                continue
            j=nav.obs_idx[f][sys[i]]
            freq=nav.obs_freq[f][sys[i]]
            #cp=obs.L[iu[i],j]-obsb.L[ir[i],j]
            #pr=obs.P[iu[i],j]-obsb.P[ir[i],j]
            cp=obs.L[i,j]
            pr=obs.P[i,j]
            bias[i]=cp-pr*freq/gn.rCST.CLIGHT   
            amb=nav.x[IB(sat[i],f,nav.na)]
            if amb!=0.0:
                offset+=bias[i]-amb
                na+=1
        # adjust phase-code coherency
        if na>0:
            db=offset/na
            for i in range(gn.uGNSS.MAXSAT):
                if nav.x[IB(i+1,f,nav.na)]!=0.0:
                    nav.x[IB(i+1,f,nav.na)]+=db
        # initialize ambiguity
        for i in range(ns):
            j=IB(sat[i],f,nav.na)
            if bias[i]==0.0 or nav.x[j]!=0.0:
                continue
            nav.x[j]=bias[i]
            nav.P[j,j]=nav.sig_n0**2
    return 0



def relpos(nav,obs,cs):
    nf=nav.nf
    ns=len(obs.sat)

    rs,vs,dts,svh=satposs(obs,nav,cs)
    #rsb,vsb,dtsb,svhb=satposs(obsb,nav)
    
    # non-differencial residual for base 
    #yr,er,el=zdres(nav,obsb,rsb,dtsb,nav.rb)
    
    #ns,iu,ir=selsat(nav,obs,obsb,el)
        
    #y[ns:,:]=yr[ir,:]
    #e[ns:,]=er[ir,:]
    
    # Kalman filter time propagation
    udstate_ppp(nav,obs)
    
    pos=gn.ecef2pos(nav.x[0:3])
    
    inet=cs.find_grid_index(pos)
    dlat,dlon=cs.get_dpos(pos)
    trph,trpw=cs.get_trop(dlat,dlon)
    stec=cs.get_stec(dlat,dlon)
    
    xa=np.zeros(nav.nx)
    xp=nav.x.copy()
    bias=np.zeros(nav.nx)

    # non-differencial residual for rover 
    y,e,el=zdres(nav,obs,rs,dts,xp[0:3])
    
    
    #y[:ns,:]=yu[iu,:]
    #e[:ns,:]=eu[iu,:]
    #el = el[iu]
    sat=obs.sat
    # DD residual
    v,H,R=ddres(nav,xp,y,e,sat,el)
    Pp=nav.P.copy()
    
    # Kalman filter measurement update
    xp,Pp=kfupdate(xp,Pp,H,v,R)
    
    if True:
        # non-differencial residual for rover after measurement update
        yu,eu,elr=zdres(nav,obs,rs,dts,xp[0:3])
        y[:ns,:]=yu[:,:]
        e[:ns,:]=eu[:,:]
        # reisdual for float solution
        v,H,R=ddres(nav,xp,y,e,sat,el)
        if valpos(nav,v,R):
            nav.x=xp
            nav.P=Pp
    
    nb,xa=resamb_lambda(nav,bias)
    if nb>0:
        yu,eu,elr=zdres(nav,obs,rs,dts,xa[0:3])
        y[:ns,:]=yu[:,:]
        e[:ns,:]=eu[:,:]
        v,H,R=ddres(nav,xa,y,e,sat,el)
        if valpos(nav,v,R):
            holdamb(nav,xa)
    
    return 0

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

if __name__ == '__main__':
    bdir='../data/'
    l6file=bdir+'2021078M.l6'
    griddef=bdir+'clas_grid.def'
    
    # based on GSI F5 solution
    xyz_ref=[-3962108.673,   3381309.574,   3668678.638]
    pos_ref=ecef2pos(xyz_ref)

    bdir='../data/'
    navfile=bdir+'SEPT078M.21P'
    obsfile=bdir+'SEPT078M.21O'

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
        rtkinit(nav,dec.pos)
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
            t[ne]=timediff(obs.t,t0)

            if ne>=20:
                relpos(nav,obs,cs)
            
        fc.close()
        dec.fobs.close()
    
    plt.plot(t,enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east','north','up'])
    plt.grid()
    plt.axis([0,3600,-6,6])
    


    
    



                    
