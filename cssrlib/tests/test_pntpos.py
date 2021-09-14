# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

import matplotlib.pyplot as plt
import numpy as np
from cssrlib.rinex import rnxdec
from cssrlib.gnss import ecef2pos,timediff,dops,ecef2enu
from cssrlib.pntpos import stdinit,pntpos

xyz_ref=[-3962108.673,   3381309.574,   3668678.638]
pos_ref=ecef2pos(xyz_ref)

navfile='./data/SEPT078M.21P'
obsfile='./data/SEPT078M.21O'

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


if True:
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


    
    
