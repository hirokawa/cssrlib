# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import gnss as gn
import rinex as rn
from pntpos import pntpos
from ephemeris import findeph,eph2pos,satposs
from ppp import tidedisp
from mlambda import mlambda

def zdres(nav,obs,rs,dts,rr):
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
    for i in range(n):
        sys,prn=gn.sat2prn(obs.sat[i])
        r,e[i,:]=gn.geodist(rs[i,:],rr_)
        az,el[i]=gn.satazel(pos,e[i,:])
        if el[i]<nav.elmin:
            continue
        if obs.sat[i] in nav.excl_sat:
            continue
        r+=-_c*dts[i]
        
        j=2 if sys==gn.uGNSS.GAL else 1
        y[i,0]=obs.L[i,0]*_c/nav.freq[0]-r
        y[i,1]=obs.L[i,j]*_c/nav.freq[j]-r
        y[i,2]=obs.P[i,0]-r
        y[i,3]=obs.P[i,j]-r
    return y,e,el

def ddcov(nb,n,Ri,Rj,nv):
    """ DD measurement error covariance """
    R=np.zeros((nv,nv))
    k=0
    for b in range(n):
        for i in range(nb[b]):
            for j in range(nb[b]):
                R[k+i,k+j]=Ri[k+i]
                if i==j:
                    R[k+i,k+j]+=Rj[k+i]
        k+=nb[b] 
    return R

def sysidx(satlist,sys_ref):
    idx=[]
    for k,sat in enumerate(satlist):
        sys,prn=gn.sat2prn(sat)
        if sys==sys_ref:
            idx.append(k)
    return idx

def IB(s,f):
    idx=6+gn.uGNSS.MAXSAT*f+s-1
    return idx

def varerr(nav,sat,sys,el,f):
    s_el=np.sin(el)
    fact= nav.eratio[f-nav.nf] if f>=nav.nf else 1
    a=fact*nav.err[1]
    b=fact*nav.err[2]
    return 2.0*(a**2+(b/s_el)**2)
    
def ddres(nav,y,e,sat,el):
    """ DD phase/code residual """
    _c=gn.rCST.CLIGHT
    nf=nav.nf
    ns=len(el)
    x=nav.x
#    posu=gn.ecef2pos(x)
#    posr=gn.ecef2pos(nav.rb)
    nb=np.zeros(2*4*2+2,dtype=int)
    Ri=np.zeros(ns*nf*2+2)
    Rj=np.zeros(ns*nf*2+2)
#    im=np.zeros(ns)
    nv=0;b=0
    H=np.zeros((ns*nf*2,nav.nx))
    v=np.zeros(ns*nf*2)
    idx_f=[0,1]
    for m,sys in enumerate(nav.gnss_t):
        idx_f[1]=2 if sys==gn.uGNSS.GAL else 1
        for f in range(0,nf*2):
            if f<nf:
                freq=nav.freq[idx_f[f]]
            # reference satellite
            idx=sysidx(sat,sys)
            i=idx[np.argmax(el[idx])]
            for j in idx:
                if i==j:
                    continue
                ## DD residual
                v[nv]=(y[i,f]-y[i+ns,f])-(y[j,f]-y[j+ns,f])
                H[nv,0:3]=-e[i,:]+e[j,:]
                if f<nf: # carrier
                    idx_i=IB(sat[i],f)
                    idx_j=IB(sat[j],f)
                    lami=_c/freq
                    v[nv]-=lami*(x[idx_i]-x[idx_j])
                    H[nv,idx_i]=lami
                    H[nv,idx_j]=-lami
                    varerr(nav,sat,sys,el,f)
                Ri[nv]=varerr(nav,sat[i],sys,el[i],f)
                Rj[nv]=varerr(nav,sat[j],sys,el[j],f)
                
                nb[b]+=1
                nv+=1
            b+=1
    v=np.resize(v,nv)
    H=np.resize(H,(nv,nav.nx))
    R=ddcov(nb,b,Ri,Rj,nv)

    return v,H,R

def valpos(nav,v,R):
    """ post-file residual test """
    return 0

def ddidx(nav,n):
    """ index for SD to DD transformation matrix D """
    nb=0
    na=nav.na
    ix=[]
    fix=np.zeros((n,nav.nf))
    for m in range(gn.rCST.MAXGNSS):
        k=na
        for f in range(nav.nf):
            for i in range(k,k+n):
                if nav.x[i]==0.0:
                    continue
                fix[i-k,f]=1
            k+=n
            for j in range(k,k+n):
                if i==j or nav.x[j]==0.0:
                    continue
                fix[j-k,j]=1
    return ix

def resamb_lambda(nav,bias,xa):
    nx=nav.nx;na=nav.na
    ix=np.zeros((nx,2),dtype=int)
    nb,ix=ddidx(nav)
    if nb<=0:
        print("no valid DD")
        return -1
    y=np.zeros(nb)
    DP=np.zeros((nb,nx-na))
    b=np.zeros((nb,2))
    db=np.zeros(nb)
    Qb=np.zeros((nb,nb))
    Qab=np.zeros((na,nb))
    
    # y=D*xc, Qb=D*Qc*D', Qab=Qac*D'
    #y=nav.x[ix[0:2:2*nb]]-nav.x[idx[1:2:2*nb+1]]
    #for j in range(nx-na):
    #    DP[:,j]=nav.P[ix[0:2:2*nb],na+j]-nav.P[ix[1:2:2*nb+1],na+j]-
    #for j in range(nb):
    #    Qb[:,j]=DP[]-DP[]
    #for j in range(nb):
    #    Qab[:,j]=nav.P[]-nav.P[]
    # MLAMBDA ILS
    b,s=mlambda(y,Qb)
    stat=False
    if s[0]<=0.0 or s[1]/s[0]>=nav.thresar[0]:
        #nav.xa=x
        nav.Pa=nav.P[0:na,0:na]
        bias=b
        y-=b
        Qb=np.linalg.inv(Qb)
        nav.xa-=Qab@Qb@y
        nav.Pa-=Qab@Qb@Qab.T
        
        # restore SD ambiguity
        #restamb(nav,bias,nb,xa)
        stat=True

    return stat

def kfupdate(x,P,H,v,R):
    PHt=P@H.T
    S=H@PHt+R
    K=PHt@np.linalg.inv(S)
    x+=K@v
    P-=K@H@P
    return x,P

def selsat(nav,obs,obsb,elb):
    idx0=np.where(elb>=nav.elmin)
    idx=np.intersect1d(obs.sat,obsb.sat[idx0],return_indices=True)
    k=len(idx[0])
    iu=idx[1]
    ir=idx0[0][idx[2]]
    return k,iu,ir

def relpos(nav,obs,obsb):
    nf=nav.nf
    if (obs.t-obsb.t).total_seconds()!=0:
        return -1
  
    rsb,vsb,dtsb,svhb=satposs(obsb,nav)
    rs,vs,dts,svh=satposs(obs,nav)
    
    # non-differencial residual for base 
    yr,er,el=zdres(nav,obsb,rsb,dtsb,nav.rb)
    
    ns,iu,ir=selsat(nav,obs,obsb,el)
    
    y = np.zeros((ns*2,nf*2))
    e = np.zeros((ns*2,3))
    
    y[ns:,:]=yr[ir,:]
    e[ns:,]=er[ir,:]
    
    # Kalman filter time propagation
    #udstate(nav,obs,obsb,idx)
    
    xa=np.zeros(nav.nx)
    xp=nav.x.copy()
    bias=np.zeros(nav.nx)

    # non-differencial residual for rover 
    yu,eu,el=zdres(nav,obs,rs,dts,xp[0:3])
    
    y[:ns,:]=yu[iu,:]
    e[:ns,:]=er[iu,:]
    el = el[iu]
    sat=obs.sat[iu]
    # DD residual
    v,H,R=ddres(nav,y,e,sat,el)
    Pp=nav.P.copy()
    
    # Kalman filter measurement update
    xp,Pp=kfupdate(xp,Pp,H,v,R)
    
    if False:
        # non-differencial residual for rover after measurement update
        yr,er,elr=zdres(nav,obs,rs,dts,nav.x)
        # reisdual for float solution
        v,H,R=ddres(nav,y,e,el)
        if valpos(nav,v,R):
            nav.x=xp
            nav.P=Pp
    
    stat=resamb_lambda(nav,bias,xa)
    if stat:
        v,H,R=ddres(nav,y,e,el)
        #if valpos(nav,v,R):
            #holdamb(nav)
    
    return 0
    
    

            
if __name__ == '__main__':
    bdir='../data/'
    navfile=bdir+'SEPT078M.21P'
    obsfile=bdir+'SEPT078M.21O'
    basefile=bdir+'3034078M.21O'
        
    # rover
    dec = rn.rnxdec()
    nav = gn.Nav()
    dec.decode_nav(navfile,nav)
    
    # base
    decb=rn.rnxdec()
    decb.decode_obsh(basefile)
    dec.decode_obsh(obsfile)
 
    nep=600//30
    nep=1
    # GSI 3034 fujisawa
    nav.rb=[-3.9594006311e6,3.3857045326e6,3.6675231107e6]
    
    if True:
        rr=dec.pos
        for ne in range(nep):
            obs=dec.decode_obs()
            obsb=decb.decode_obs()
            
            sol,az,el=pntpos(obs,nav,rr)
            nav.x[0:3]=sol[0:3]
            relpos(nav,obs,obsb)

            
            
            
        dec.fobs.close()
        decb.fobs.close()
    
    
    
    
    