# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import gnss as gn
from gnss import rCST,sat2prn,time2gpst,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav,timediff,vnorm,antmodel,uGNSS
from ephemeris import findeph,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt
from cssrlib.cssrlib import cssr,sSigGPS,sSigGAL,sSigQZS,sCType
from ppp import tidedisp,shapiro,windupcorr
from rtk import IB,ddres,kfupdate,resamb_lambda,valpos,holdamb
from pntpos import pntpos

MAXITR=10
ELMIN=10
NX=4

def logmon(nav,sat,cs,iu=None):
    week,tow=gn.time2gpst(obs.t)
    if iu is None:
        cpc=cs.cpc;prc=cs.prc;osr=cs.osr
    else:
        cpc=cs.cpc[iu,:]
        prc=cs.prc[iu,:]
        osr=cs.osr[iu,:]
    if nav.loglevel>=2:
        n = cpc.shape
        for i in range(n[0]):
            if cpc[i,0]==0 and cpc[i,1]==0:
                continue
            nav.fout.write("%6d\t%3d\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t%2d\t" 
                           % (tow,sat[i],cpc[i,0],cpc[i,1],prc[i,0],prc[i,1],cs.iodssr))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t" 
                           % (osr[i,0],osr[i,1],osr[i,2],osr[i,3],osr[i,4]))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\n" 
                           % (osr[i,5],osr[i,6],osr[i,7],osr[i,8]))    
    return 0

def rtkinit(nav,pos0=np.zeros(3)):
    nav.nf=2
    nav.pmode=1 # 0:static, 1:kinematic

    nav.na=3 if nav.pmode==0 else 6
    nav.ratio=0
    nav.thresar=[2]
    nav.nx=nav.na+gn.uGNSS.MAXSAT*nav.nf
    nav.x=np.zeros(nav.nx)
    nav.P=np.zeros((nav.nx,nav.nx))
    nav.xa=np.zeros(nav.na)
    nav.Pa=np.zeros((nav.na,nav.na))
    nav.nfix=nav.neb=0
    nav.phw=np.zeros(gn.uGNSS.MAXSAT)
    
    # parameter for PPP-RTK
    nav.eratio=[50,50]
    nav.err=[100,0.00707,0.00354]
    nav.sig_p0 = 30.0
    nav.sig_v0 = 10.0
    nav.sig_n0 = 30.0
    nav.sig_qp=0.1
    nav.sig_qv=0.01
    nav.tidecorr=True
    nav.armode = 1 # 1:contunous,2:instantaneous,3:fix-and-hold
    nav.gnss_t=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS]
    
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
                    gn.uGNSS.QZS:[sSigQZS.L1C,sSigQZS.L2X]}
    
    nav.fout=None
    nav.logfile='log.txt'
    if nav.loglevel>=2:
       nav.fout=open(nav.logfile,'w') 
    
    


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
        # cycle slip check by LLI
        for i in range(ns):
            if sys[i] not in nav.gnss_t:
                continue
            j=nav.obs_idx[f][sys[i]]
            if obs.lli[i,j]&1==0:
                continue
            nav.x[IB(sat[i],f,nav.na)]=0
        
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
    week,tow=gn.time2gpst(obs.t)
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
    #sat_n=cs.decode_local_sat(cs.lc[inet].netmask)
    
    cs.cpc=np.zeros((n,nf))
    cs.prc=np.zeros((n,nf))
    cs.osr=np.zeros((n,2*nf+5))
    
    for i in range(n):
        sat=obs.sat[i]
        sys,prn=gn.sat2prn(sat)
        if sys not in nav.gnss_t or sat in nav.excl_sat:
            continue
        if sat not in cs.lc[inet].sat_n:
            continue
        idx_n=np.where(cs.sat_n==sat)[0][0] # global
        idx_l=np.where(cs.lc[inet].sat_n==sat)[0][0] # local
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
            iono[f]=40.3e16/(freq[f]*freq[f])*stec[idx_l]
        iono_ = 40.3e16/(freq[0]*freq[1])*stec[idx_l]
        
        # global/local signal bias
        cbias=np.zeros(nav.nf)
        pbias=np.zeros(nav.nf)
        
        if cs.lc[0].cbias is not None:        
            cbias+=cs.lc[0].cbias[idx_n][kidx]
        if cs.lc[0].pbias is not None:
            pbias+=cs.lc[0].pbias[idx_n][kidx]
        if cs.lc[inet].cbias is not None:
            cbias+=cs.lc[inet].cbias[idx_l][kidx]
        if cs.lc[inet].pbias is not None:
            pbias+=cs.lc[inet].pbias[idx_l][kidx]
            t1=timediff(obs.t,cs.lc[0].t0[sCType.ORBIT])
            t2=timediff(obs.t,cs.lc[inet].t0[sCType.PBIAS])
            if t1>=0 and t1<30 and t2>=30:
                pbias+=nav.dsis[sat]*0
                
        
        # relativity effect
        relatv=shapiro(rs[i,:],rr_)

        # tropospheric delay                
        mapfh,mapfw=gn.tropmapf(obs.t,pos,el[i])
        trop=mapfh*trph*r_hs+mapfw*trpw*r_wet
        
        # phase wind-up effect    
        nav.phw[sat-1]=windupcorr(obs.t,rs[i,:],vs[i,:],rr_,nav.phw[sat-1])
        phw=lam*nav.phw[sat-1]
        
        antr=antmodel(nav,el[i],nav.nf)
        
        # range correction
        prc_c = trop+relatv+antr
        #prc_c += nav.dorb[sat]-nav.dclk[sat]
        cs.prc[i,:]=prc_c+iono+cbias
        cs.cpc[i,:]=prc_c-iono+pbias+phw
        cs.osr[i,:]=[pbias[0],pbias[1],cbias[0],cbias[1],trop,iono_,relatv,nav.dorb[sat],nav.dclk[sat]]
        
        r+=-_c*dts[i]
        
        for f in range(nf):
            k=nav.obs_idx[f][sys]
            y[i,f]=obs.L[i,k]*lam[f]-(r+cs.cpc[i,f])
            y[i,f+nf]=obs.P[i,k]-(r+cs.prc[i,f])

    return y,e,el

def relpos(nav,obs,cs):
    rs,vs,dts,svh=satposs(obs,nav,cs)
    
    # Kalman filter time propagation
    udstate_ppp(nav,obs)
        
    xa=np.zeros(nav.nx)
    xp=nav.x.copy()

    # non-differencial residual for rover 
    yu,eu,elu=zdres(nav,obs,rs,vs,dts,xp[0:3],cs)
    
    iu=np.where(elu>=nav.elmin)[0]
    sat=obs.sat[iu]
    y=yu[iu,:]
    e=eu[iu,:]
    el=elu[iu]
    
    logmon(nav,sat,cs,iu)
    
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
    nav.smode=5 # float
    if nb>0:
        yu,eu,elu=zdres(nav,obs,rs,vs,dts,xa[0:3],cs)
        y=yu[iu,:]
        e=eu[iu,:]
        v,H,R=ddres(nav,xa,y,e,sat,el)
        if valpos(nav,v,R): # R <= Q=H'PH+R  rtk->sol.chisq<max_inno[3] (0.5)
            if nav.armode==3: # fix and hold
                holdamb(nav,xa) # hold fixed ambiguity
            # if rtk->sol.chisq<max_inno[4] (5)
            nav.smode=4 # fix
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
    nep=300
    t=np.zeros(nep)
    tc=np.zeros(nep)
    enu=np.ones((nep,3))*np.nan
    sol=np.zeros((nep,4))
    dop=np.zeros((nep,4))
    smode=np.zeros(nep,dtype=int)
    if dec.decode_obsh(obsfile)>=0:
        rr=dec.pos
        rtkinit(nav,dec.pos)
        pos=ecef2pos(rr)
        inet=cs.find_grid_index(pos)
        
        fc=open(bdir+l6file,'rb')
        if not fc:
            print("L6 messsage file cannot open."); exit(-1)
        for ne in range(nep):
            obs=dec.decode_obs()
            week,tow=time2gpst(obs.t)
            
            cs.decode_l6msg(fc.read(250),0)
            if cs.fcnt==5: # end of sub-frame
                cs.week=week
                cs.decode_cssr(cs.buff,0)            

            if ne==0:
                #nav.sol=1
                #sol,az,el=pntpos(obs,nav,rr)
                #nav.x[0:3]=sol[0:3] # initial estimation
                t0=obs.t
                t0.time=t0.time//30*30
                cs.time=obs.t
                nav.time_p=t0
            t[ne]=timediff(obs.t,t0)
            tc[ne] = timediff(cs.time,t0)
            
            week,tow=time2gpst(obs.t)
    
            cstat=cs.chk_stat()
            
            if tow>=475413:
                tow

            if cstat or tow>=475220:
                # for debug
                #nav.x[0:3]=rr
            
                relpos(nav,obs,cs)
            
            sol=nav.x[0:3]
            enu[ne,:]=gn.ecef2enu(pos_ref,sol-xyz_ref)
            smode[ne]=nav.smode
            
        fc.close()
        dec.fobs.close()
    

    fig_type=1
    ylim=0.2
    
    if fig_type==1:    
        plt.plot(t,enu,'.')
        plt.xticks(np.arange(0,nep+1, step=30))
        plt.ylabel('position error [m]')
        plt.xlabel('time[s]')
        plt.legend(['east','north','up'])
        plt.grid()
        plt.axis([0,ne,-ylim,ylim])    
    elif fig_type==2:
        plt.plot(enu[:,0],enu[:,1],'.')
        plt.xlabel('easting [m]')
        plt.ylabel('northing [m]')
        plt.grid()
        plt.axis([-ylim,ylim,-ylim,ylim])   
    

    if nav.fout is not None:
        nav.fout.close()

    if True:
        gnss_tbl={uGNSS.GPS:1,uGNSS.GAL:8,uGNSS.QZS:16}
        
        fname='C:/work/clas_test_library/util/rnx2rtkp/2021078M.nmea.osr'
        names=['tow','sys','prn','cpc1','cpc2','prc1','prc2','dorb','dclk']
        vr = np.genfromtxt('%s'%(fname),delimiter=',',skip_header=1,usecols=[1,2,3,23,24,26,27,29,30],names=names)
        
        dtype0 = (float,int,float,float,float,float)
        d = np.genfromtxt(nav.logfile)
        tow0=(d[0,0]//30)*30
        t=d[:,0]-tow0
        sat=d[:,1]
        cpc1=d[:,2];cpc2=d[:,3];prc1=d[:,4];prc2=d[:,5]

        sats=np.unique(sat)
        #tofst=0
        tofst=4
        tr=vr['tow']-tow0
        #cpc1_=vr['cpc1']+vr['dorb']-vr['dclk']
        cpc1_=vr['cpc1'];prc1_=vr['prc1'];cpc2_=vr['cpc2'];prc2_=vr['prc2']
        
        plt.figure()
        for sat0 in sats:
            sys,prn=sat2prn(sat0)
            gnss = gnss_tbl[sys]
            #if sat0!=64:
            #    continue
            idx=np.where(sat==sat0)[0]
            plt.plot(t[idx]-tofst,prc1[idx]-prc1[idx[0]],label=gn.sat2id(sat0))
            
            idx1=np.where(np.logical_and(vr['sys']==gnss,vr['prn']==prn))[0]
            if len(idx1)==0:
                continue
            plt.plot(tr[idx1],prc1_[idx1]-prc1_[idx1[0]],'--',label=gn.sat2id(sat0))

        plt.grid()
        plt.xticks(np.arange(0,nep+1, step=30))
        plt.legend()
        plt.xlim([0,nep])
        plt.show()
    
        plt.figure()
        for sat0 in sats:
            sys,prn=sat2prn(sat0)
            gnss = gnss_tbl[sys]
            if sat0!=9:
                continue
            idx=np.where(sat==sat0)[0]
            plt.plot(t[idx]-tofst,cpc1[idx]-cpc1[idx[0]],label=gn.sat2id(sat0))
            
            idx1=np.where(np.logical_and(vr['sys']==gnss,vr['prn']==prn))[0]
            if len(idx1)==0:
                continue
            plt.plot(tr[idx1],cpc1_[idx1]-cpc1_[idx1[0]],'--',label=gn.sat2id(sat0))

        plt.grid()
        plt.xticks(np.arange(0,nep+1, step=30))
        plt.legend()
        plt.xlim([0,nep])
        plt.show()




                    
