# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import numpy as np
from gnss import uGNSS,rCST,rSIG,prn2sat,sat2prn,Eph,gpst2time,time2gpst,sat2id,ecef2pos,geodist,satazel
from ephemeris import findeph,eph2pos
from plot import skyplot,plot_elv
from rinex import rnxdec    

if __name__ == '__main__':
    #f_in='../data/ublox_D9_1.ubx'

    bdir='C:/work/gps/cssrlib/data/'

    navfile=bdir+'SEPT0781.21P'
    obsfile=bdir+'SEPT0782s.21O'

    dec = rnxdec()
    nav=dec.decode_nav(navfile)
     
    t=[]
    nep=3600//30
    elv=np.ones((nep,dec.MAXSAT))*np.nan
    azm=np.ones((nep,dec.MAXSAT))*np.nan
    mag=np.zeros((nep,dec.MAXSAT),dtype=int)
    t=np.zeros(nep)*np.nan
    ne=0
    if dec.decode_obsh(obsfile)>=0:
        rr=dec.pos
        pos=ecef2pos(rr)
        for ne in range(nep):
            obs=dec.decode_obs()
            if ne==0:
                t0=obs.t
            t[ne]=(obs.t-t0).total_seconds()
            for k,sat in enumerate(obs.sat):
                eph=findeph(nav,obs.t,sat)
                rs,dts=eph2pos(obs.t,eph)
                r,e=geodist(rs,rr)
                azm[ne,sat-1],elv[ne,sat-1]=satazel(pos,e)      
                mag[ne,sat-1]=obs.mag[k,0]

        dec.fobs.close()
    
    #plot_elv(t,elv)
    skyplot(azm,elv)
    


    
    



                    
