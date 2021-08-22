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
        satlist = []
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

    return satlist

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
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.img_tiles as cimgt
    
    #satlst=loadname('../data/TLE_GNSS_PRN-GEJ.txt')
    satlst=loadname('../data/TLE_GNSS_PRN.txt')
    sats=loadTLE('../data/gnss.txt',satlst)
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
        
    mode=3 # 1: skyplot, 2: elevation, 3: hot-map
    
    if mode<3:
        lat,lon,azm,elv,sats_=tleorb(sats,dates,obs)
        idx=np.where(sats_>0)[0]

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

    elif mode==3:
        lat_t = np.arange(-85,90,2); 
        lon_t = np.arange(-180,180,2); 
        elmask = np.deg2rad(10)
        
        nlat=len(lat_t)
        nlon=len(lon_t)
        if True:
            nsat_t = np.zeros((nlat,nlon))
            nep=len(dates)
            for k,lat in enumerate(lat_t):
                for j,lon in enumerate(lon_t):
                    obs.lat='%.4f' % (lat)
                    obs.lon='%.4f' % (lon)
                    lat_,lon_,azm,elv,sats_=tleorb(sats,dates,obs)
                    nsat=np.zeros(nep)
                    idx=np.where(sats_>0)[0]
                    el=elv[:,idx]
                    for i,sat in enumerate(sats_[idx]):
                        if np.all(np.isnan(el[:,i])):
                            continue
                        idx=el[:,i]>elmask
                        nsat[idx]+=1
                    nsat_t[k,j]=np.mean(nsat)
            np.save('nsat_t',nsat_t)
 
        if True:
 
            stamen_terrain = cimgt.Stamen('terrain-background') 
           
 
            fig=plt.figure(figsize=(12,8))
            ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            #ax.add_image(stamen_terrain, 8)
            
            lons, lats = np.meshgrid(lon_t, lat_t)
            nsat_t=np.load('nsat_t.npy')

            
            #ax.add_feature(cfeature.LAND)
            #ax.add_feature(cfeature.COASTLINE)
            #ax.add_feature(cfeature.BORDERS)
            #ax.add_feature(states_provinces, edgecolor='gray')
            #ax.add_image(tiler, 7)
            #ax.gridlines()
            #ax.stock_img()
            cm=plt.cm.jet

            lebels=np.arange(np.floor(np.min(nsat_t))-2,np.ceil(np.max(nsat_t))+2,2)
            cs=ax.contourf(lons,lats, nsat_t,lebels,transform=ccrs.PlateCarree(),cmap=cm)
            fig.colorbar(cs, shrink=0.5)
            ax.coastlines() 
            plt.show()
            
            
            