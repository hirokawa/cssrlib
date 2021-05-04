# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
import gnss as gn
from math import sin,cos,atan2,asin

def nut_iau1980(t,f):
    nut=np.array([
        [   0,   0,   0,   0,   1, -6798.4, -171996, -174.2, 92025,   8.9],
        [   0,   0,   2,  -2,   2,   182.6,  -13187,   -1.6,  5736,  -3.1],
        [   0,   0,   2,   0,   2,    13.7,   -2274,   -0.2,   977,  -0.5],
        [   0,   0,   0,   0,   2, -3399.2,    2062,    0.2,  -895,   0.5],
        [   0,  -1,   0,   0,   0,  -365.3,   -1426,    3.4,    54,  -0.1],
        [   1,   0,   0,   0,   0,    27.6,     712,    0.1,    -7,   0.0],
        [   0,   1,   2,  -2,   2,   121.7,    -517,    1.2,   224,  -0.6],
        [   0,   0,   2,   0,   1,    13.6,    -386,   -0.4,   200,   0.0],
        [   1,   0,   2,   0,   2,     9.1,    -301,    0.0,   129,  -0.1],
        [   0,  -1,   2,  -2,   2,   365.2,     217,   -0.5,   -95,   0.3],
        [  -1,   0,   0,   2,   0,    31.8,     158,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   1,   177.8,     129,    0.1,   -70,   0.0],
        [  -1,   0,   2,   0,   2,    27.1,     123,    0.0,   -53,   0.0],
        [   1,   0,   0,   0,   1,    27.7,      63,    0.1,   -33,   0.0],
        [   0,   0,   0,   2,   0,    14.8,      63,    0.0,    -2,   0.0],
        [  -1,   0,   2,   2,   2,     9.6,     -59,    0.0,    26,   0.0],
        [  -1,   0,   0,   0,   1,   -27.4,     -58,   -0.1,    32,   0.0],
        [   1,   0,   2,   0,   1,     9.1,     -51,    0.0,    27,   0.0],
        [  -2,   0,   0,   2,   0,  -205.9,     -48,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   1,  1305.5,      46,    0.0,   -24,   0.0],
        [   0,   0,   2,   2,   2,     7.1,     -38,    0.0,    16,   0.0],
        [   2,   0,   2,   0,   2,     6.9,     -31,    0.0,    13,   0.0],
        [   2,   0,   0,   0,   0,    13.8,      29,    0.0,    -1,   0.0],
        [   1,   0,   2,  -2,   2,    23.9,      29,    0.0,   -12,   0.0],
        [   0,   0,   2,   0,   0,    13.6,      26,    0.0,    -1,   0.0],
        [   0,   0,   2,  -2,   0,   173.3,     -22,    0.0,     0,   0.0],
        [  -1,   0,   2,   0,   1,    27.0,      21,    0.0,   -10,   0.0],
        [   0,   2,   0,   0,   0,   182.6,      17,   -0.1,     0,   0.0],
        [   0,   2,   2,  -2,   2,    91.3,     -16,    0.1,     7,   0.0],
        [  -1,   0,   0,   2,   1,    32.0,      16,    0.0,    -8,   0.0],
        [   0,   1,   0,   0,   1,   386.0,     -15,    0.0,     9,   0.0],
        [   1,   0,   0,  -2,   1,   -31.7,     -13,    0.0,     7,   0.0],
        [   0,  -1,   0,   0,   1,  -346.6,     -12,    0.0,     6,   0.0],
        [   2,   0,  -2,   0,   0, -1095.2,      11,    0.0,     0,   0.0],
        [  -1,   0,   2,   2,   1,     9.5,     -10,    0.0,     5,   0.0],
        [   1,   0,   2,   2,   2,     5.6,      -8,    0.0,     3,   0.0],
        [   0,  -1,   2,   0,   2,    14.2,      -7,    0.0,     3,   0.0],
        [   0,   0,   2,   2,   1,     7.1,      -7,    0.0,     3,   0.0],
        [   1,   1,   0,  -2,   0,   -34.8,      -7,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   2,    13.2,       7,    0.0,    -3,   0.0],
        [  -2,   0,   0,   2,   1,  -199.8,      -6,    0.0,     3,   0.0],
        [   0,   0,   0,   2,   1,    14.8,      -6,    0.0,     3,   0.0],
        [   2,   0,   2,  -2,   2,    12.8,       6,    0.0,    -3,   0.0],
        [   1,   0,   0,   2,   0,     9.6,       6,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   1,    23.9,       6,    0.0,    -3,   0.0],
        [   0,   0,   0,  -2,   1,   -14.7,      -5,    0.0,     3,   0.0],
        [   0,  -1,   2,  -2,   1,   346.6,      -5,    0.0,     3,   0.0],
        [   2,   0,   2,   0,   1,     6.9,      -5,    0.0,     3,   0.0],
        [   1,  -1,   0,   0,   0,    29.8,       5,    0.0,     0,   0.0],
        [   1,   0,   0,  -1,   0,   411.8,      -4,    0.0,     0,   0.0],
        [   0,   0,   0,   1,   0,    29.5,      -4,    0.0,     0,   0.0],
        [   0,   1,   0,  -2,   0,   -15.4,      -4,    0.0,     0,   0.0],
        [   1,   0,  -2,   0,   0,   -26.9,       4,    0.0,     0,   0.0],
        [   2,   0,   0,  -2,   1,   212.3,       4,    0.0,    -2,   0.0],
        [   0,   1,   2,  -2,   1,   119.6,       4,    0.0,    -2,   0.0],
        [   1,   1,   0,   0,   0,    25.6,      -3,    0.0,     0,   0.0],
        [   1,  -1,   0,  -1,   0, -3232.9,      -3,    0.0,     0,   0.0],
        [  -1,  -1,   2,   2,   2,     9.8,      -3,    0.0,     1,   0.0],
        [   0,  -1,   2,   2,   2,     7.2,      -3,    0.0,     1,   0.0],
        [   1,  -1,   2,   0,   2,     9.4,      -3,    0.0,     1,   0.0],
        [   3,   0,   2,   0,   2,     5.5,      -3,    0.0,     1,   0.0],
        [  -2,   0,   2,   0,   2,  1615.7,      -3,    0.0,     1,   0.0],
        [   1,   0,   2,   0,   0,     9.1,       3,    0.0,     0,   0.0],
        [  -1,   0,   2,   4,   2,     5.8,      -2,    0.0,     1,   0.0],
        [   1,   0,   0,   0,   2,    27.8,      -2,    0.0,     1,   0.0],
        [  -1,   0,   2,  -2,   1,   -32.6,      -2,    0.0,     1,   0.0],
        [   0,  -2,   2,  -2,   1,  6786.3,      -2,    0.0,     1,   0.0],
        [  -2,   0,   0,   0,   1,   -13.7,      -2,    0.0,     1,   0.0],
        [   2,   0,   0,   0,   1,    13.8,       2,    0.0,    -1,   0.0],
        [   3,   0,   0,   0,   0,     9.2,       2,    0.0,     0,   0.0],
        [   1,   1,   2,   0,   2,     8.9,       2,    0.0,    -1,   0.0],
        [   0,   0,   2,   1,   2,     9.3,       2,    0.0,    -1,   0.0],
        [   1,   0,   0,   2,   1,     9.6,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,   2,   1,     5.6,      -1,    0.0,     1,   0.0],
        [   1,   1,   0,  -2,   1,   -34.7,      -1,    0.0,     0,   0.0],
        [   0,   1,   0,   2,   0,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   1,   2,  -2,   0,   117.5,      -1,    0.0,     0,   0.0],
        [   0,   1,  -2,   2,   0,  -329.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,   2,   0,    23.8,      -1,    0.0,     0,   0.0],
        [   1,   0,  -2,  -2,   0,    -9.5,      -1,    0.0,     0,   0.0],
        [   1,   0,   2,  -2,   0,    32.8,      -1,    0.0,     0,   0.0],
        [   1,   0,   0,  -4,   0,   -10.1,      -1,    0.0,     0,   0.0],
        [   2,   0,   0,  -4,   0,   -15.9,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,   4,   2,     4.8,      -1,    0.0,     0,   0.0],
        [   0,   0,   2,  -1,   2,    25.4,      -1,    0.0,     0,   0.0],
        [  -2,   0,   2,   4,   2,     7.3,      -1,    0.0,     1,   0.0],
        [   2,   0,   2,   2,   2,     4.7,      -1,    0.0,     0,   0.0],
        [   0,  -1,   2,   0,   1,    14.2,      -1,    0.0,     0,   0.0],
        [   0,   0,  -2,   0,   1,   -13.6,      -1,    0.0,     0,   0.0],
        [   0,   0,   4,  -2,   2,    12.7,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   0,   2,   409.2,       1,    0.0,     0,   0.0],
        [   1,   1,   2,  -2,   2,    22.5,       1,    0.0,    -1,   0.0],
        [   3,   0,   2,  -2,   2,     8.7,       1,    0.0,     0,   0.0],
        [  -2,   0,   2,   2,   2,    14.6,       1,    0.0,    -1,   0.0],
        [  -1,   0,   0,   0,   2,   -27.3,       1,    0.0,    -1,   0.0],
        [   0,   0,  -2,   2,   1,  -169.0,       1,    0.0,     0,   0.0],
        [   0,   1,   2,   0,   1,    13.1,       1,    0.0,     0,   0.0],
        [  -1,   0,   4,   0,   2,     9.1,       1,    0.0,     0,   0.0],
        [   2,   1,   0,  -2,   0,   131.7,       1,    0.0,     0,   0.0],
        [   2,   0,   0,   2,   0,     7.1,       1,    0.0,     0,   0.0],
        [   2,   0,   2,  -2,   1,    12.8,       1,    0.0,    -1,   0.0],
        [   2,   0,  -2,   0,   1,  -943.2,       1,    0.0,     0,   0.0],
        [   1,  -1,   0,  -2,   0,   -29.3,       1,    0.0,     0,   0.0],
        [  -1,   0,   0,   1,   1,  -388.3,       1,    0.0,     0,   0.0],
        [  -1,  -1,   0,   2,   1,    35.0,       1,    0.0,     0,   0.0],
        [   0,   1,   0,   1,   0,    27.3,       1,    0.0,     0,   0.0]
        ])
    dpsi=0;deps=0
    for i in range(106):
        ang=nut[i,0:5]@f
        dpsi+=(nut[i,6]+nut[i,7]*t)*sin(ang)
        deps+=(nut[i,8]+nut[i,9]*t)*cos(ang)
    
    dpsi*=gn.rCST.AS2R*1e-4
    deps*=gn.rCST.AS2R*1e-4
    
    return dpsi,deps

def ast_args(t):
    # iau1980 nutation l,l',F,D,OMG [rad]
    fc=np.array([[134.96340251, 1717915923.2178,  31.8792,  0.051635, -0.00024470],
                 [357.52910918,  129596581.0481,  -0.5532,  0.000136, -0.00001149],
                 [ 93.27209062, 1739527262.8478, -12.7512, -0.001037,  0.00000417],
                 [297.85019547, 1602961601.2090,  -6.3706,  0.006593, -0.00003169],
                 [125.04455501,   -6962890.2665,   7.4722,  0.007702, -0.00005939]])
    tt=np.zeros(4)
    f=np.zeros(5)
    tt[0]=t
    for k in range(3):
        tt[k+1]=tt[k]*t
    for k in range(5):
        f[k]=fc[k,0]*3600.0+fc[k,1:]@tt
        f[k]=np.fmod(f[k]*gn.rCST.AS2R,2.0*np.pi)
    return f

def Rx(t):
    c=cos(t);s=sin(t)
    return np.array([[1,0,0],[0,c,s],[0,-s,c]])

def Ry(t):
    c=cos(t);s=sin(t)
    return np.array([[c,0,-s],[0,1,0],[s,0,c]])

def Rz(t):
    c=cos(t);s=sin(t)
    return np.array([[c,s,0],[-s,c,0],[0,0,1]])

def utc2gmst(t,ut1_utc=0):
    """ UTC to GMST """
    ep0=gn.epoch2time([2000,1,1,12,0,0])
    tut=gn.timeadd(t,ut1_utc)
    ep=gn.time2epoch(tut)    
    tut0=gn.epoch2time([ep[0],ep[1],ep[2],0,0,0])
    ut=ep[3]*3600+ep[4]*60+ep[5]
    t1=gn.timediff(tut0,ep0)/gn.rCST.CENTURY_SEC
    t2=t1**2;t3=t2*t1
    gmst0=24110.54841+8640184.812866*t1+0.093104*t2-6.2e-6*t3
    gmst=gmst0+1.002737909350795*ut
    return np.fmod(gmst,gn.rCST.DAY_SEC)*(2.0*np.pi/gn.rCST.DAY_SEC)

def eci2ecef(tgps,erpv):
    tutc=gn.gpst2utc(tgps)
    ep0=gn.epoch2time([2000,1,1,12,0,0])
    dt=gn.timediff(tgps,ep0)
    t=(dt+19+32.184)/gn.rCST.CENTURY_SEC
    t2=t**2;t3=t2*t
    f=ast_args(t)
    
    # iau1976 precession
    ze=(2306.2181*t+0.30188*t2+0.017998*t3)*gn.rCST.AS2R
    th=(2004.3109*t-0.42665*t2-0.041833*t3)*gn.rCST.AS2R
    z =(2306.2181*t+1.09468*t2+0.018203*t3)*gn.rCST.AS2R
    eps=(84381.448-46.8150*t-0.00059*t2+0.001813*t3)*gn.rCST.AS2R
    P=Rz(-z)@Ry(th)@Rz(-ze)
    
    # iau1980 nutation
    dpsi,deps=nut_iau1980(t,f)
    N=Rx(-eps-deps)@Rz(-dpsi)@Rx(eps)

    # Greenwich aparent sidereal time [rad]
    gmst=utc2gmst(tutc,erpv[2])
    gast=gmst+dpsi*cos(eps)
    gast+=(0.00264*sin(f[4])+0.000063*sin(2.0*f[4]))*gn.rCST.AS2R

    W=Ry(-erpv[0])@Rx(-erpv[1])
    U=W@Rz(gast)@N@P
    
    return U,gmst

def sunmoonpos(tutc,erpv=np.zeros(5)):
    """ calculate sun/moon position in ECEF """
    tut=gn.timeadd(tutc,erpv[2])
    ep0=gn.epoch2time([2000,1,1,12,0,0])
    t=gn.timediff(tut,ep0)/gn.rCST.CENTURY_SEC
    f=ast_args(t)
    eps=np.deg2rad(23.439291-0.0130042*t) # Mean Obliquity of the ecliptic
    c_e=cos(eps);s_e=sin(eps)
    
    # Sun position in ECI
    Ms=np.deg2rad(357.5277233+35999.05034*t) # Mean anomaly of the sun 
    # Mean longitude of the Sun (Ecliptic coordinate)
    ls=np.deg2rad(280.460+36000.770*t+1.914666471*sin(Ms)+0.019994643*sin(2.0*Ms))
    # Distance of the Sun from the Earth
    rs=gn.rCST.AU*(1.000140612-0.016708617*cos(Ms)-0.000139589*cos(2.0*Ms))
    c_l=cos(ls);s_l=sin(ls)
    rsun_eci=np.array([rs*c_l,rs*c_e*s_l,rs*s_e*s_l])
    
    lm=218.32+481267.883*t+6.29*sin(f[0])-1.27*sin(f[0]-2.0*f[3])+0.66*sin(2.0*f[3])+0.21*sin(2.0*f[0])-0.19*sin(f[1])-0.11*sin(2.0*f[2])
    pm=5.13*sin(f[2])+0.28*sin(f[0]+f[2])-0.28*sin(f[2]-f[0])-0.17*sin(f[2]-2.0*f[3])
    u=(0.9508+0.0518*cos(f[0])+0.0095*cos(f[0]-2.0*f[3])+0.0078*cos(2.0*f[3])+0.0028*cos(2.0*f[0]))
    rm=gn.rCST.RE_WGS84/sin(np.deg2rad(u))
    c_l=cos(np.deg2rad(lm));s_l=sin(np.deg2rad(lm))
    c_p=cos(np.deg2rad(pm));s_p=sin(np.deg2rad(pm))    
    rmoon_eci=rm*np.array([c_p*c_l,c_e*c_p*s_l-s_e*s_p,s_e*c_p*s_l+c_e*s_p])

    U,gmst=eci2ecef(tutc,erpv)
    rsun=U@rsun_eci
    rmoon=U@rmoon_eci

    return rsun,rmoon,gmst
    
def shapiro(rsat,rrcv):
    """ relativistic shapiro effect """
    rs=np.linalg.norm(rsat)
    rr=np.linalg.norm(rrcv)
    rrs=np.linalg.norm(rsat-rrcv)
    corr=(2*gn.rCST.GME/gn.rCST.CLIGHT**2)*np.log((rs+rr+rrs)/(rs+rr-rrs))
    return corr

def windupcorr(time,rs,vs,rr,phw):
    we=np.array([0,0,gn.rCST.OMGE])
    rsun,rmoon,gmst=sunmoonpos(gn.gpst2utc(time))
    ek=gn.vnorm(rr-rs)
    ezs=gn.vnorm(-rs)
    ess=gn.vnorm(vs+np.cross(we,rs))
    eys=gn.vnorm(np.cross(ezs,ess))
    exs=np.cross(eys,ezs)
    pos=gn.ecef2pos(rr)
    E=gn.xyz2enu(pos)
    exr=E[0,:]
    eyr=E[1,:]
    eks=np.cross(ek,eys)
    ekr=np.cross(ek,eyr)
    ds=exs-ek*(ek@exs)-eks
    dr=exr-ek*(ek@exr)+ekr
    c_p=(ds@dr)/(np.linalg.norm(ds)*np.linalg.norm(dr))
    c_p=max(-1.0,min(1.0,c_p))
    ph=np.arccos(c_p)/(2.0*np.pi)
    drs=np.cross(ds,dr)
    if ek@drs<0.0:
        ph=-ph
    phw=ph+np.floor(phw-ph+0.5) # [cycle]
    return phw
    
def tide_pl(eu,rp,GMp,pos):
    H3=0.293;L3=0.0156
    r=np.linalg.norm(rp)
    ep=rp/r
    K2=GMp/gn.rCST.GME*gn.rCST.RE_WGS84**4/r**3
    K3=K2*gn.rCST.RE_WGS84/r
    latp=asin(ep[2])
    lonp=atan2(ep[1],ep[0])
    c_p=cos(latp)
    c_l=cos(pos[0]);s_l=sin(pos[0])
    
    p=(3.0*s_l**2-1.0)/2.0
    H2=0.6078-0.0006*p
    L2=0.0847+0.0002*p
    a=ep@eu;a2=a**2
    dp=K2*3.0*L2*a
    du=K2*(H2*(1.5*a2-0.5)-3.0*L2*a2)
    
    dp+=K3*L3*(7.5*a2-1.5)
    du+=K3*a*(H3*(2.5*a2-1.5)-L3*(7.5*a2-1.5))
    dlon=pos[1]-lonp
    du+=3.0/4.0*0.0025*K2*sin(2.0*latp)*sin(2.0*pos[0])*sin(dlon)
    du+=3.0/4.0*0.0022*K2*(c_p*c_l)**2*sin(2.0*dlon)
    
    dr=dp*ep+du*eu 
    return dr
    
def solid_tide(rsun,rmoon,pos,E,gmst,flag=True):
    # time domain
    eu=E[2,:]
    dr1=tide_pl(eu,rsun ,gn.rCST.GMS,pos)
    dr2=tide_pl(eu,rmoon,gn.rCST.GMM,pos)
    # frequency domain
    s_2l=sin(2.0*pos[0])
    du=-0.012*s_2l*sin(gmst+pos[1])
    
    dr=dr1+dr2+du*eu
    
    # eliminate permanent tide
    if flag:
        s_l=sin(pos[0])
        du=0.1196*(1.5*s_l**2-0.5)
        dn=0.0247*s_2l
        dr+=du*E[2,:]+dn*E[1,:]
    
    return dr

def tidedisp(tutc,pos,erpv=None):
    if erpv is None:
        erpv=np.zeros(5)
    rs,rm,gmst=sunmoonpos(tutc,erpv)
    E=gn.xyz2enu(pos)
    dr=solid_tide(rs,rm,pos,E,gmst)
    return dr

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from ephemeris import findeph,eph2pos
    from rinex import rnxdec
    
    tgps=gn.epoch2time([2021,3,19,0,0,0])
    
    #tgps=gn.gpst2time(2149,475230)
    pos=np.array([0.61678759,  2.43512138, 64.94054687])
    erpv=np.array([2.1079217879069683e-06, 4.8733853217911866e-07, 
                   -0.044509672541668682, -0.0007141, 0])

    flg_tide=False
    flg_pwup=True
    
    if flg_tide:
        n=86400//300
        t=np.zeros(n)
        dr=np.zeros((n,3))
        for k in range(n):
            tn=gn.timeadd(tgps,k*300)
            t[k]=gn.timediff(tn,tgps)
            dn=tidedisp(gn.gpst2utc(tn),pos,erpv)
            dr[k,:]=gn.ecef2enu(pos,dn)
     
        plt.figure()
        plt.plot(t/3600,dr)
        plt.xlabel('time [h]')
        plt.ylabel('displacement [m]')
        plt.grid()
        plt.axis([0,24,-0.2,0.2])
        plt.legend(('east','north','up'))
      
    if flg_pwup:
        bdir='../data/'
        navfile=bdir+'30340780.21q'
        nav = gn.Nav()
        dec = rnxdec()
        nav=dec.decode_nav(navfile,nav)
        rr=gn.pos2ecef(pos)
        sat=gn.prn2sat(gn.uGNSS.QZS,194)

        n=86400//300
        t=np.zeros(n)
        ph=np.zeros(n)
        d=np.zeros(n)
        phw=0
        for k in range(n):
            tn=gn.timeadd(tgps,k*300)
            eph=findeph(nav.eph,tn,sat)
            rs,vs,dts=eph2pos(tn,eph,True)
            phw=windupcorr(tn,rs,vs,rr,phw)
            t[k]=gn.timediff(tn,tgps)
            ph[k]=phw
            d[k]=shapiro(rs,rr)
        
        plt.figure()
        plt.plot(t/3600,ph,label='phase windup')
        plt.plot(t/3600,d,label='shapiro')
        plt.xlabel('time [h]')
        plt.ylabel('delta range [m]')
        plt.grid()
        plt.axis([0,24,0,0.4])
        plt.legend()        
        