# -*- coding: utf-8 -*-
"""
Created on Sat May 15 09:36:59 2021

@author: ruihi
"""

from ephem import readtle,Observer
from pandas import date_range
import numpy as np
from gnss import id2sat,uGNSS,sat2prn
from datetime import timedelta

def loadname(file):
    satlist={}
    with open(file,'r') as f:
        for line in f:
            if line[0]=='#':
                continue
            v=line.split()
            if len(v)>=2:
                satlist[v[1]]=v[0]
    return satlist

def loadTLE(tle_file,satlst=None):
    with open(tle_file,'r') as f:
        satlist = []; prn = []
        for l1 in f:
            l2 = f.readline()
            l3 = f.readline()
            norad_id = l2[2:8]
            if satlst is not None:
                if norad_id in satlst:
                    name = satlst[norad_id]
                else:
                    continue
            else:
                name = l1
            sat = readtle(name,l2,l3)
            satlist.append(sat)

    return satlist,prn

def tleorb(sat,dates,obs=None):
    nsat=len(sat);nd=len(dates)
    lat=np.zeros((nd,nsat))
    lon=np.zeros((nd,nsat))
    el=np.zeros((nd,nsat))
    az=np.zeros((nd,nsat))
    sats=np.zeros(nsat,dtype=int)
    for k,sv in enumerate(sat):
        sat_=id2sat(sv.name)
        if sat_<=0:
            continue
        sats[k]=sat_
        for i,t in enumerate(dates):
            sv.compute(t)
            lat[i,k] = sv.sublat;lon[i,k] = sv.sublong
            if obs is not None:
                obs.date=t
                sv.compute(obs)
                el[i,k] = sv.alt;az[i,k] = sv.az
                
    return lat,lon,az,el,sats

if __name__ == '__main__':
    from plot import skyplot,plot_elv,plot_nsat
    import matplotlib.pyplot as plt
    
    #satlst=loadname('../data/TLE_GNSS_PRN-GEJ.txt')
    satlst=loadname('../data/TLE_GNSS_PRN.txt')
    sats,prn=loadTLE('../data/gnss.txt',satlst)
    start_day='2021-05-14T00:00:00Z'
    end_day='2021-05-15T00:00:00Z'    
    #start_day='2021-05-15T06:00:00Z'
    #end_day='2021-05-15T06:00:00Z' 
    period=1 # minutes
    elmask = np.deg2rad(15)
    
    dates = date_range(start=start_day, end=end_day,  freq='{}T'.format(period))
    t0=dates[0]
    t=(dates-t0).total_seconds()
    obs = Observer()
    obs.lat,obs.lon='35.6804','139.769'
    
    lat,lon,azm,elv,sats_=tleorb(sats,dates,obs)
    idx=np.where(sats_>0)[0]
    
    mode=2 # 1: skyplot, 2: elevation
    
    if mode==1:
        nsat=skyplot(azm[:,idx],elv[:,idx],elmask,sats_[idx])
    elif mode==2:
        nsat=plot_elv(t,elv[:,idx],elmask,sats_[idx])
        plot_nsat(t,nsat)
        #nidx=[uGNSS.GPS] # mean:7.5
        nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS] # mean:16.7
        #nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS,uGNSS.GLO] # mean:22.7
        #nidx=[uGNSS.GPS,uGNSS.GAL,uGNSS.QZS,uGNSS.GLO,uGNSS.BDS] # mean:41.3
        nsat_mean=np.mean(np.sum(nsat[:,nidx],1))
    
    
        