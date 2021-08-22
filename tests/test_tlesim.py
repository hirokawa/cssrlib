# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.gnss import uGNSS
from cssrlib.tlesim import loadname,loadTLE,tleorb
import numpy as np
from pandas import date_range
from cssrlib.plot import skyplot,plot_elv,plot_nsat
import matplotlib.pyplot as plt
from ephem import Observer

#satlst=loadname('../data/TLE_GNSS_PRN-GEJ.txt')
satlst=loadname('./data/TLE_GNSS_PRN.txt')
sats=loadTLE('./data/gnss.txt',satlst)
st='2021-05-14T00:00:00Z'
ed='2021-05-15T00:00:00Z'    
#st='2021-05-15T06:00:00Z'
#ed='2021-05-15T06:30:00Z' 
period=30 # minutes
elmask = np.deg2rad(15)

dates = date_range(start=st,end=ed,freq='{}T'.format(period))
t0=dates[0]
t=(dates-t0).total_seconds()
obs = Observer()
obs.lat,obs.lon='35.6804','139.769'
    
mode=2 # 1: skyplot, 2: elevation

if mode<3:
    lat,lon,azm,elv,sats_=tleorb(sats,dates,obs)
    idx=np.where(sats_>0)[0]

if mode==1:
    nsat=skyplot(azm[:,idx],elv[:,idx],elmask,sats_[idx])
    plt.show()
elif mode==2:
    nsat=plot_elv(t,elv[:,idx],elmask,sats_[idx])
    plot_nsat(t,nsat)
    #nidx=[uGNSS.GPS] # mean:7.5
    nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS] # mean:16.7
    #nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS,uGNSS.GLO] # mean:22.7
    #nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS,uGNSS.GLO,uGNSS.BDS] # mean:41.3
    nsat_mean=np.mean(np.sum(nsat[:,nidx],1))
    plt.show()
