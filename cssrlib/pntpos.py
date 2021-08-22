import numpy as np
from gnss import rCST,sat2prn,ecef2pos,geodist,satazel,ionmodel,tropmodel,dops,ecef2enu,Nav,timediff,tropmapf,kfupdate
from ephemeris import findeph,satposs
from rinex import rnxdec    
import matplotlib.pyplot as plt

def varerr(nav,el):
    """ variation of measurement """
    s_el=np.sin(el)
    if s_el<=0.0:
        return 0.0
    a=nav.err[1]
    b=nav.err[2]
    return a**2+(b/s_el)**2

def stdinit():
    nav=Nav()
    nav.na=6
    nav.nx=8
    nav.x=np.zeros(nav.nx)
    sig_p0=100.0*np.ones(3)
    sig_v0=10.0*np.ones(3)
    nav.P=np.diag(np.hstack((sig_p0**2,sig_v0**2,100,100)))
    dt=1
    nav.Phi=np.eye(nav.nx)
    nav.Phi[0:3,3:6]=dt*np.eye(3)
    nav.Phi[6,7]=dt
    nav.elmin=np.deg2rad(10)
    sq=1e-2
    nav.Q=np.diag([0,0,0,sq,sq,sq,0,1e-4])
    nav.err=[0,0.3,0.3]
    return nav

def rescode(obs,nav,rs,dts,svh,x):
    n=obs.sat.shape[0]
    rr=x[0:3]
    dtr=x[nav.na]
    pos=ecef2pos(rr)
    v=np.zeros(n); H=np.zeros((n,nav.nx))
    azv=np.zeros(n); elv=np.zeros(n)
    nv=0 
    for i in range(n):
        if np.linalg.norm(rs[i,:])<rCST.RE_WGS84 or svh[i]>0:
            continue
        r,e=geodist(rs[i,:],rr)
        az,el=satazel(pos,e)
        if el<nav.elmin:
            continue
        eph=findeph(nav.eph,obs.t,obs.sat[i])
        P=obs.P[i,0]-eph.tgd*rCST.CLIGHT
        dion=ionmodel(obs.t,pos,az,el,nav.ion)
        trop_hs,trop_wet,z=tropmodel(obs.t,pos,el)
        mapfh,mapfw=tropmapf(obs.t,pos,el)
        dtrp=mapfh*trop_hs+mapfw*trop_wet
        v[nv]=P-(r+dtr-rCST.CLIGHT*dts[i]+dion+dtrp)
        H[nv,0:3]=-e; H[nv,nav.na]=1
        azv[nv]=az; elv[nv]=el
        nv+=1
    v=v[0:nv]; H=H[0:nv,:]
    azv=azv[0:nv]; elv=elv[0:nv]       
    return v,H,nv,azv,elv
    
def pntpos(obs,nav):
    rs,vs,dts,svh=satposs(obs,nav)
    x=nav.x.copy()
    P=nav.P.copy()   
    x=nav.Phi@x
    P=nav.Phi@P@nav.Phi.T+nav.Q
    v,H,nv,az,el=rescode(obs,nav,rs,dts,svh,x)
    if abs(np.mean(v))>100:
        x[nav.na]=np.mean(v)
        v-=x[nav.na]
    n=len(v)
    R=np.zeros((n,n))  
    for k in range(n):
        R[k,k]=varerr(nav,el[k])
    nav.x,nav.P=kfupdate(x,P,H,v,R)
    return nav,az,el

if __name__ == '__main__':    
    xyz_ref=[-3962108.673,   3381309.574,   3668678.638]
    pos_ref=ecef2pos(xyz_ref)

    navfile='c:/work/gps/cssrlib/data/SEPT078M.21P'
    obsfile='c:/work/gps/cssrlib/data/SEPT078M.21O'

    dec = rnxdec()
    nav = stdinit()
    nav=dec.decode_nav(navfile,nav)
    nep=120
    t=np.zeros(nep)
    enu=np.zeros((nep,3))
    sol=np.zeros((nep,nav.nx))
    dop=np.zeros((nep,4))
    nsat=np.zeros(nep,dtype=int)
    if dec.decode_obsh(obsfile)>=0:
        nav.x[0:3]=dec.pos
        for ne in range(nep):
            obs=dec.decode_obs()
            if ne==0:
                t0=obs.t
            t[ne]=timediff(obs.t,t0)
            nav,az,el=pntpos(obs,nav)
            sol[ne,:]=nav.x
            dop[ne,:]=dops(az,el)
            enu[ne,:]=ecef2enu(pos_ref,sol[ne,0:3]-xyz_ref)
            nsat[ne]=len(el)
        dec.fobs.close()
    
    dmax=3
    plt.figure()
    plt.plot(t,enu)
    plt.ylabel('pos err[m]')
    plt.xlabel('time[s]')
    plt.legend(['east','north','up'])
    plt.grid()
    plt.axis([0,nep,-dmax,dmax])
    plt.show()

    plt.figure()
    plt.plot(t,sol[:,3:6])
    plt.ylabel('vel err[m/s]')
    plt.xlabel('time[s]')
    plt.legend(['x','y','z'])
    plt.grid()
    plt.axis([0,nep,-0.5,0.5])
    plt.show()

    sol[0,7]=np.nan
    plt.figure()
    plt.subplot(211)
    plt.plot(t,sol[:,6]-sol[0,6])
    plt.ylabel('clock bias [m]')
    plt.grid()
    plt.subplot(212)
    plt.plot(t,sol[:,7])
    plt.ylabel('clock drift [m/s]')
    plt.xlabel('time[s]')
    plt.grid()
    plt.show()

    if True:
        plt.figure()
        plt.plot(enu[:,0],enu[:,1])
        plt.xlabel('easting[m]')
        plt.ylabel('northing[m]')
        plt.grid()
        plt.axis([-dmax,dmax,-dmax,dmax])
        plt.show()
        
        plt.figure()
        plt.plot(t,dop[:,1:])
        plt.legend(['pdop','hdop','vdop'])
        plt.grid()
        plt.axis([0,nep,0,2])
        plt.xlabel('time[s]')
        plt.show()


    
    



                    
