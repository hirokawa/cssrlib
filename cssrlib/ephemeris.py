# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

import numpy as np
from gnss import uGNSS,rCST,sat2prn,Eph,ecef2pos,prn2sat,Nav,timediff,timeadd,epoch2time
#import datetime

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
        dt=timediff(t,eph_.toe)
        if iode<0 and abs(dt)<dt_p:
            dt_p=abs(dt)
            eph=eph_
    return eph

def dtadjust(t1,t2,tw=604800):
    dt=timediff(t1,t2)
    if dt>tw:
        dt-=tw
    elif dt<-tw:
        dt+=tw
    return dt

def eph2pos(t,eph,flg_v=False):
    sys,prn=sat2prn(eph.sat)
    if sys==uGNSS.GAL:
        mu=rCST.MU_GAL;omge=rCST.OMGE_GAL
    else: # GPS,QZS
        mu=rCST.MU_GPS;omge=rCST.OMGE
    dt=dtadjust(t,eph.toe)
    n=np.sqrt(mu/eph.A**3)+eph.deln
    M=eph.M0+n*dt
    E=M
    for iter in range(10):
        Eold=E
        sE=np.sin(E)
        E=M+eph.e*sE
        if abs(Eold-E)<1e-12:
            break
    cE=np.cos(E)
    dtc=dtadjust(t,eph.toc)
    dtrel=-2*np.sqrt(mu)*eph.e*np.sqrt(eph.A)*sE/rCST.CLIGHT**2
    dts=eph.af0+eph.af1*dtc+eph.af2*dtc**2+dtrel
    
    nus=np.sqrt(1-eph.e**2)*sE
    nuc=cE-eph.e;nue=1-eph.e*cE
    
    nu=np.arctan2(nus,nuc)
    phi=nu+eph.omg
    h2=np.array([np.cos(2*phi),np.sin(2*phi)])
    u=phi+np.array([eph.cuc,eph.cus])@h2
    r=eph.A*nue+np.array([eph.crc,eph.crs])@h2
    h=np.array([np.cos(u),np.sin(u)])
    xo=r*h
    
    inc=eph.i0+eph.idot*dt+np.array([eph.cic,eph.cis])@h2
    Omg=eph.OMG0+eph.OMGd*dt-omge*(eph.toes+dt)
    sOmg=np.sin(Omg);cOmg=np.cos(Omg)
    si=np.sin(inc);ci=np.cos(inc)
    p=np.array([cOmg,sOmg,0])    
    q=np.array([-ci*sOmg,ci*cOmg,si])
    rs=xo@np.array([p,q])
    
    if flg_v: # satellite velocity
        qq=np.array([si*sOmg,-si*cOmg,ci])
        Ed=n/nue
        nud=np.sqrt(1-eph.e**2)/nue*Ed
        h2d=2*nud*np.array([-h[1],h[0]])
        ud=nud+np.array([eph.cuc,eph.cus])@h2d
        rd=eph.A*eph.e*sE*Ed+np.array([eph.crc,eph.crs])@h2d
        
        hd=np.array([-h[1],h[0]])
        xod=rd*h+(r*ud)*hd
        incd=eph.idot+np.array([eph.cic,eph.cis])@h2d
        omegd=eph.OMGd-omge
        
        pd=np.array([-p[1],p[0],0])*omegd
        qd=np.array([-q[1],q[0],0])*omegd+qq*incd
        
        vs=xo@np.array([pd,qd])+xod@np.array([p,q])
        return rs,vs,dts
    
    return rs,dts

def eph2clk(time,eph):
    t=timediff(time,eph.toc)
    for i in range(2):
      t-=eph.af0+eph.af1*t+eph.af2*t**2
    dts=eph.af0+eph.af1*t+eph.af2*t**2
    return dts

def ephclk(t,eph,sat):
    dts=eph2clk(t,eph)
    return dts

def satposs(obs,nav,cs=None):
    n=obs.sat.shape[0]
    rs=np.zeros((n,3))
    vs=np.zeros((n,3))
    dts=np.zeros(n)
    svh=np.zeros(n,dtype=int)
    iode=-1
    for i in range(n):
        sat=obs.sat[i]
        sys,prn=sat2prn(sat)
        if sys not in nav.gnss_t:
            continue
        pr=obs.P[i,0]
        t=timeadd(obs.t,-pr/rCST.CLIGHT)
        if cs is not None:
            if sat not in cs.sat_n:
                continue
            idx=cs.sat_n.index(sat)
            iode=cs.lc[0].iode[idx]
            dorb=cs.lc[0].dorb[idx,:]
            dclk=cs.lc[0].dclk[idx]
            dt_=timediff(t,cs.lc[cs.inet_ref].t0[5])
            
        eph=findeph(nav.eph,t,sat,iode)
        if eph is None:
            svh[i]=1;continue
        svh[i]=eph.svh
        dt=eph2clk(t,eph)
        t=timeadd(t,-dt)
        rs[i,:],vs[i,:],dts[i]=eph2pos(t,eph,True)
        if cs is not None: # apply SSR correction
            ea=vs[i,:]/np.linalg.norm(vs[i,:])
            rc=np.cross(rs[i,:],vs[i,:])
            ec=rc/np.linalg.norm(rc)
            er=np.cross(ea,ec)
            dorb_e=-dorb@[er,ea,ec]
            rs[i,:]+=dorb_e
            dts[i]+=dclk/rCST.CLIGHT
    return rs,vs,dts,svh


if __name__ == '__main__':
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
    from rinex import rnxdec 

    bdir='../data/'
    #navfile=bdir+'SEPT078M.21P'
    navfile=bdir+'30340780.21q'
    
    nav = Nav()
    dec = rnxdec()
    nav=dec.decode_nav(navfile,nav)

    n=24*3600//300
    t0=epoch2time([2021,3,19,0,0,0])

    flg_plot=True
    
    if True:
        t=t0
        sat=prn2sat(uGNSS.QZS,194)
        eph=findeph(nav.eph,t,sat)
        rs,vs,dts=eph2pos(t,eph,True)

    if flg_plot:
        lon0=135    
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=lon0,central_latitude=0))
        ax.coastlines(resolution='50m')
        ax.gridlines()
        ax.stock_img()        
        pos=np.zeros((n,3))
    
        for k in range(uGNSS.MAXSAT):
            sat=k+1
            sys,prn=sat2prn(sat)
            if sys!=uGNSS.QZS:
                continue
            for i in range(n):
                t=timeadd(t0,i*300)
                if eph==None:
                    continue
                rs,dts=eph2pos(t,eph)    
                pos[i,:]=ecef2pos(rs)
                pos[i,0]=np.rad2deg(pos[i,0])
                pos[i,1]=np.rad2deg(pos[i,1])
                
            plt.plot(pos[:,1],pos[:,0],'m-',transform=ccrs.Geodetic())
        

    
    
    
    
    

        