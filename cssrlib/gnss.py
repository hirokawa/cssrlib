# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

from enum import IntEnum,Enum
import numpy as np
import datetime

class rCST():
    CLIGHT=299792458.0
    MU_GPS=3.9860050E14
    MU_GAL=3.986004418E14
    GME=3.986004415E+14
    GMS=1.327124E+20
    GMM=4.902801E+12
    OMGE=7.2921151467E-5
    OMGE_GAL=7.2921151467E-5
    RE_WGS84=6378137.0
    FE_WGS84=(1.0/298.257223563)
    AU=149597870691.0
    D2R=0.017453292519943295
    AS2R=D2R/3600.0
    DAY_SEC=86400.0    
    CENTURY_SEC=DAY_SEC*36525.0

class uGNSS(IntEnum):
    GPS=0;SBS=1;GAL=2;BDS=3;QZS=5;GLO=6;GNSSMAX=7
    GPSMAX=32;GALMAX=36;BDSMAX=63;QZSMAX=10;
    GLOMAX=24;SBSMAX=24
    NONE=-1
    MAXSAT=GPSMAX+GLOMAX+GALMAX+BDSMAX+QZSMAX

class uSIG(IntEnum):
    GPS_L1CA=0;GPS_L2W=2;GPS_L2CL=3;GPS_L2CM=4;GPS_L5Q=6
    SBS_L1CA=0
    GAL_E1C=0;GAL_E1B=1;GAL_E5BI=5;GAL_E5BQ=6
    BDS_B1ID1=0;BDS_B1ID2=1;BDS_B2ID1=2;BDS_B2ID2=3
    QZS_L1CA=0;QZS_L1S=1;QZS_L2CM=4;QZS_L2CL=5
    GLO_L1OF=0;GLO_L2OF=2
    NONE=-1
    SIGMAX=7

class rSIG(IntEnum):
    NONE=0;L1C=1;L2L=2;L2W=3;L5Q=4;L7Q=5
    SIGMAX=16

class Obs():
    def __init__(self):
        self.nm=0
        self.t=0
        self.P=[]
        self.L=[]
        self.data=[]
        self.sat=[]

class Eph():
    sat=0;iode=0;iodc=0
    af0=0.0;af1=0.0;af2=0.0
    toc=0;toe=0;tot=0;week=0
    crs=0.0;crc=0.0;cus=0.0;cus=0.0;cis=0.0;cic=0.0
    e=0.0;i0=0.0;A=0.0;deln=0.0;M0=0.0;OMG0=0.0
    OMGd=0.0;omg=0.0;idot=0.0;tgd=0.0;tgd_b=0.0
    sva=0;health=0;fit=0
    toes=0
    
    def __init__(self,sat=0):
        self.sat=sat

class Nav():
    def __init__(self):
        self.eph=[]
        self.ion=np.array([
            [0.1118E-07,-0.7451E-08,-0.5961E-07, 0.1192E-06],
            [0.1167E+06,-0.2294E+06,-0.1311E+06, 0.1049E+07]])

def gpst2time(week,tow):
    t=datetime.datetime(1980,1,6)+datetime.timedelta(weeks=week,seconds=tow)
    return t

def time2gpst(t):
    dt=(t-datetime.datetime(1980,1,6)).total_seconds()
    week=int(dt)//604800
    tow=dt-week*604800
    return week,tow

def prn2sat(sys,prn):
    if sys==uGNSS.GPS:
        sat=prn
    elif sys==uGNSS.GLO:
        sat=prn+uGNSS.GPSMAX
    elif sys==uGNSS.GAL:
        sat=prn+uGNSS.GPSMAX+uGNSS.GLOMAX
    elif sys==uGNSS.BDS:
        sat=prn+uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX
    elif sys==uGNSS.QZS:
        sat=prn-192+uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX            
    else:
        sat=0
    return sat

def sat2prn(sat):
    if sat>uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX:
        prn=sat-(uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX)+192
        sys=uGNSS.QZS
    elif sat>uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX:
        prn=sat-(uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX)
        sys=uGNSS.BDS
    elif sat>uGNSS.GPSMAX+uGNSS.GLOMAX:
        prn=sat-(uGNSS.GPSMAX+uGNSS.GLOMAX)
        sys=uGNSS.GAL
    elif sat>uGNSS.GPSMAX:
        prn=sat-uGNSS.GPSMAX
        sys=uGNSS.GLO
    else:
        prn=sat
        sys=uGNSS.GPS
    return (sys,prn)

def sat2id(sat):
    id=[]
    sys,prn=sat2prn(sat)
    gnss_tbl='GSECIJR'
    if sys==uGNSS.QZS:
        prn-=192
    id='%s%02d' %(gnss_tbl[sys],prn)
    return id

def geodist(rs,rr):
    e=rs-rr
    r=np.linalg.norm(e)
    e=e/r
    r+=rCST.OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/rCST.CLIGHT
    return r,e

# TBD
def dops_h(H):
    Qinv=np.linalg.inv(np.dot(H.T,H))
    dop=np.diag(Qinv)
    hdop=dop[0]+dop[1] # TBD
    vdop=dop[2] # TBD
    pdop=hdop+vdop;gdop=pdop+dop[3]
    dop=np.array([gdop,pdop,hdop,vdop])
    return dop

def dops(az,el,elmin=0):
    nm=az.shape[0]
    H=np.zeros((nm,4))
    n=0
    for i in range(nm):
        if el[i]<elmin:
            continue
        cel=np.cos(el[i]);sel=np.sin(el[i])
        H[n,0]=cel*np.sin(az[i])
        H[n,1]=cel*np.cos(az[i])
        H[n,2]=sel
        H[n,3]=1
        n+=1
    if n<4:
        return None
    Qinv=np.linalg.inv(np.dot(H.T,H))
    dop=np.diag(Qinv)
    hdop=dop[0]+dop[1] # TBD
    vdop=dop[2] # TBD
    pdop=hdop+vdop;gdop=pdop+dop[3]
    dop=np.array([gdop,pdop,hdop,vdop])
    return dop        


def xyz2enu(pos):
    sp=np.sin(pos[0]);cp=np.cos(pos[0]);
    sl=np.sin(pos[1]);cl=np.cos(pos[1]);
    E=np.array([[-sl,cl,0],
                [-sp*cl,-sp*sl,cp],
                [cp*cl,cp*sl,sp]])
    return E

def ecef2pos(r):
    e2=rCST.FE_WGS84*(2-rCST.FE_WGS84)
    r2=r[0]**2+r[1]**2
    v=rCST.RE_WGS84
    z=r[2]
    while True:
        zk=z
        sp=z/np.sqrt(r2+z**2)
        v=rCST.RE_WGS84/np.sqrt(1-e2*sp**2)
        z=r[2]+v*e2*sp
        if np.fabs(z-zk)<1e-4:
            break
    pos=np.array([np.arctan(z/np.sqrt(r2)),
                  np.arctan2(r[1],r[0]),
                  np.sqrt(r2+z**2)-v])
    return pos

def ecef2enu(pos,r):
    E=xyz2enu(pos)
    e=np.dot(E,r)
    return e

def satazel(pos,e):
    enu=ecef2enu(pos,e)
    az=np.arctan2(enu[0],enu[1])
    el=np.arcsin(enu[2])
    return az,el

def ionmodel(t,pos,az,el,ion=None):
    psi=0.0137/(el/np.pi+0.11)-0.022
    phi=pos[0]/np.pi+psi*np.cos(az)
    phi=np.max((-0.416,np.min((0.416,phi))))
    lam=pos[1]/np.pi+psi*np.sin(az)/np.cos(phi*np.pi)
    phi+=0.064*np.cos((lam-1.617)*np.pi)
    week,tow=time2gpst(t)
    tt=43200.0*lam+tow # local time
    tt-=np.floor(tt/86400)*86400
    f=1.0+16.0*np.power(0.53-el/np.pi,3.0) # slant factor
    
    h=[1,phi,phi**2,phi**3]
    amp=np.dot(h,ion[0,:])
    per=np.dot(h,ion[1,:])
    if amp<0:
        amp=0
    if per<72000.0:
        per=72000.0
    x=2.0*np.pi*(tt-50400.0)/per
    if np.abs(x)<1.57:
        v=5e-9+amp*(1.0+x*x*(-0.5+x*x/24.0))
    else:
        v=5e-9
    diono=rCST.CLIGHT*f*v
    return diono

def tropmodel(t,pos,el,humi=0.7):
    hgt=pos[2]
    
    # standard atmosphere
    pres=1013.25*np.power(1-2.2557e-5*hgt,5.2568)
    temp=15.0-6.5e-3*hgt+273.16
    e=6.108*humi*np.exp((17.15*temp-4684.0)/(temp-38.45))
    # saastamoinen
    z=np.pi/2.0-el
    trop_hs=0.0022768*pres/(1.0-0.00266*np.cos(2*pos[0])-0.00028e-3*hgt)
    trop_wet=0.002277*(1255.0/temp+0.05)*e
    return (trop_hs+trop_wet)/np.cos(z)   

if __name__ == '__main__':
    t=gpst2time(2151,554726)
    week,tow=time2gpst(t)
        