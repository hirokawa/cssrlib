# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import gnss as gn
from gnss import rCST,sat2prn,time2gpst,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav,timediff,vnorm,antmodel
from ephemeris import findeph,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt
from cssrlib.cssrlib import cssr,sSigGPS,sSigGAL,sSigQZS
from ppp import tidedisp,shapiro,windupcorr
from rtk import IB,ddres,kfupdate,resamb_lambda,valpos,holdamb

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
    nav.eratio=[50,50]
    #nav.err=[0,0.003,0.003]
    nav.err=[100,0.00707,0.00354]
    nav.sig_p0 = 30.0
    nav.sig_v0 = 10.0
    nav.sig_n0 = 30.0
    nav.sig_qp=0.1
    nav.sig_qv=0.01
    nav.tidecorr=True
    nav.phw=np.zeros(gn.uGNSS.MAXSAT)
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
    i1={gn.uGNSS.GPS:1,gn.uGNSS.GAL:2,gn.uGNSS.QZS:1}
    freq0={gn.uGNSS.GPS:nav.freq[0],gn.uGNSS.GAL:nav.freq[0],gn.uGNSS.QZS:nav.freq[0]}
    freq1={gn.uGNSS.GPS:nav.freq[1],gn.uGNSS.GAL:nav.freq[2],gn.uGNSS.QZS:nav.freq[1]}
    nav.obs_idx=[i0,i1]
    nav.obs_freq=[freq0,freq1]
    nav.cs_sig_idx={gn.uGNSS.GPS:[sSigGPS.L1C,sSigGPS.L2W],
                    gn.uGNSS.GAL:[sSigGAL.L1X,sSigGAL.L5X],
                    gn.uGNSS.QZS:[sSigQZS.L1C,sSigQZS.L2X]
                    }
    # antenna type: TRM59800.80     NONE [mm] 0:5:90 [deg]
    nav.ant_pcv=[[+0.00,-0.22,-0.86,-1.87,-3.17,-4.62,-6.03,-7.21,-7.98,
                  -8.26,-8.02,-7.32,-6.20,-4.65,-2.54,+0.37,+4.34,+9.45,+15.42],
                 [+0.00,-0.14,-0.53,-1.13,-1.89,-2.74,-3.62,-4.43,-5.07,
                  -5.40,-5.32,-4.79,-3.84,-2.56,-1.02,+0.84,+3.24,+6.51,+10.84],
                 [+0.00,-0.14,-0.53,-1.13,-1.89,-2.74,-3.62,-4.43,-5.07,
                  -5.40,-5.32,-4.79,-3.84,-2.56,-1.02,+0.84,+3.24,+6.51,+10.84]]
    nav.ant_pco=[+89.51,+117.13,+117.13]

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

def zdres(nav,obs,rs,vs,dts,rr,cs):
    """ non-differencial residual """
    _c=gn.rCST.CLIGHT
    nf=nav.nf
    n=len(obs.P)
    y=np.zeros((n,nf*2))
    el=np.zeros(n)
    e=np.zeros((n,3))
    rr_=rr.copy()
    if nav.tidecorr:
        pos=gn.ecef2pos(rr_)
        disp=tidedisp(gn.gpst2utc(obs.t),pos)
        rr_+=disp
    pos=gn.ecef2pos(rr_)
    
    inet=cs.find_grid_index(pos)
    dlat,dlon=cs.get_dpos(pos)
    trph,trpw=cs.get_trop(dlat,dlon)
    
    trop_hs,trop_wet,z=tropmodel(t,pos)
    trop_hs0,trop_wet0,z=tropmodel(t,[pos[0],pos[1],0])
    r_hs=trop_hs/trop_hs0;r_wet=trop_wet/trop_wet0  
    
    stec=cs.get_stec(dlat,dlon)
    sat_n=cs.decode_local_sat(cs.lc[inet].netmask)
    
    for i in range(n):
        sat=obs.sat[i]
        sys,prn=gn.sat2prn(sat)
        if sys not in nav.gnss_t or sat in nav.excl_sat or sat not in sat_n:
            continue
        idx_n=np.where(cs.sat_n==sat)[0][0]
        kidx=[-1]*nav.nf;nsig=0
        for k,sig in enumerate(cs.sig_n[idx_n]):
            for f in range(nav.nf):
                if sig==nav.cs_sig_idx[sys][f]:
                    kidx[f]=k;nsig+=1               
        if nsig<nav.nf:      
            continue

        r,e[i,:]=gn.geodist(rs[i,:],rr_)
        az,el[i]=gn.satazel(pos,e[i,:])
        if el[i]<nav.elmin:
            continue

        freq=np.zeros(nav.nf)
        lam=np.zeros(nav.nf)
        iono=np.zeros(nav.nf)
        for f in range(nav.nf):
            freq[f]=nav.obs_freq[f][sys]
            lam[f]=gn.rCST.CLIGHT/freq[f]
            iono[f]=40.3e16/(freq[f]*freq[f])*stec[idx_n]
        
        # global/local signal bias
        cbias=np.zeros(nav.nf)
        pbias=np.zeros(nav.nf)
        
        if cs.lc[0].cbias is not None:        
            cbias+=cs.lc[0].cbias[idx_n][kidx]
        if cs.lc[0].pbias is not None:
            pbias+=cs.lc[0].pbias[idx_n][kidx]
        if cs.lc[inet].cbias is not None:
            cbias+=cs.lc[inet].cbias[idx_n][kidx]
        if cs.lc[inet].pbias is not None:
            pbias+=cs.lc[inet].pbias[idx_n][kidx]
        
        # relativity effect
        relatv=shapiro(rs[i,:],rr_)

        # tropospheric delay                
        mapfh,mapfw=gn.tropmapf(obs.t,pos,el[i])
        trop=mapfh*trph*r_hs+mapfw*trpw*r_wet
        
        # phase wind-up effect    
        nav.phw[sat-1]=windupcorr(obs.t,rs[i,:],vs[i,:],rr_,nav.phw[sat-1])
        phw=lam*nav.phw[sat-1]
        
        antr=antmodel(nav,el[i],nav.nf)
        
        prc=trop+relatv+antr+iono+cbias
        cpc=trop+relatv+antr-iono+pbias+phw
        
        r+=-_c*dts[i]
        
        for f in range(nf):
            k=nav.obs_idx[f][sys]
            y[i,f]=obs.L[i,k]*lam[f]-(r+cpc[f])
            y[i,f+nf]=obs.P[i,k]-(r+prc[f])

        if sys==gn.uGNSS.QZS:
            print(k)

    return y,e,el

def relpos(nav,obs,cs):
    rs,vs,dts,svh=satposs(obs,nav,cs)
    
    # Kalman filter time propagation
    udstate_ppp(nav,obs)
        
    xa=np.zeros(nav.nx)
    xp=nav.x.copy()

    # non-differencial residual for rover 
    yu,eu,elu=zdres(nav,obs,rs,vs,dts,xp[0:3],cs)
    iu=np.where(elu>0)[0]
    sat=obs.sat[iu]
    y=yu[iu,:]
    e=eu[iu,:]
    el=elu[iu]
    # DD residual
    v,H,R=ddres(nav,xp,y,e,sat,el)
    Pp=nav.P.copy()
    
    # Kalman filter measurement update
    xp,Pp=kfupdate(xp,Pp,H,v,R)
    
    if True:
        # non-differencial residual for rover after measurement update
        yu,eu,elu=zdres(nav,obs,rs,vs,dts,xp[0:3],cs)
        y=yu[iu,:]
        e=eu[iu,:]
        # reisdual for float solution
        v,H,R=ddres(nav,xp,y,e,sat,el)
        if valpos(nav,v,R):
            nav.x=xp
            nav.P=Pp
    
    nb,xa=resamb_lambda(nav,sat)
    if nb>0:
        yu,eu,elu=zdres(nav,obs,rs,vs,dts,xa[0:3],cs)
        y=yu[iu,:]
        e=eu[iu,:]
        v,H,R=ddres(nav,xa,y,e,sat,el)
        if valpos(nav,v,R):
            holdamb(nav,xa)
    
    return 0

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
    


    
    



                    
