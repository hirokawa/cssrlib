# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

from enum import IntEnum,Enum
import numpy as np
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from gnss import uGNSS,rCST,prn2sat,sat2prn,Eph,gpst2time,time2gpst,sat2id

# GPS=0;SBS=1;GAL=2;BDS=3;QZS=5;GLO=6

def plot_nsat(t,nsat):
    lbl_t={uGNSS.GPS:'GPS',uGNSS.GAL:'GALILEO',uGNSS.QZS:'QZS',
           uGNSS.GLO:'GLONASS',uGNSS.BDS:'BDS'}
    col_tbl='bygmkrc'
    fig=plt.figure()
    fig,ax=plt.subplots(1,1)
    ns=np.zeros(nsat.shape[0])
    for gnss in lbl_t.keys():
        y=ns+nsat[:,gnss]
        #ax.plot(t,y)
        if gnss>=0:
            ax.fill_between(t/3600,ns,y,label=lbl_t[gnss],facecolor=col_tbl[gnss])
        ns+=nsat[:,gnss]
      
    plt.ylabel('number of satellites')
    plt.xlabel('time [h]')
    plt.legend()
    plt.axis([0,24,0,50])
    plt.show()


def plot_elv(t,elv,elmask=0,satlist=None):
    # elevation plot
    nsat=np.zeros((len(t),uGNSS.GNSSMAX),dtype=int)
    col_tbl='bygmkrc'
    fig=plt.figure('elevation')
    for k,sat in enumerate(satlist):
        if np.all(np.isnan(elv[:,k])):
            continue
        sys,prn=sat2prn(sat)
        idx=elv[:,k]>elmask
        nsat[idx,sys]+=1         
        plt.plot(t/3600,np.rad2deg(elv[:,k]),'-'+col_tbl[sys])
    plt.ylabel('elevation [deg]')
    plt.xlabel('time [h]')
    plt.grid()
    plt.axis([0,24,0,90])
    plt.show()
    return nsat

def skyplot(azm,elv,elmask=0,satlist=None):
    fig=plt.figure('skyplot')
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'])
    ax.set_ylim([0,90])
    ax.set_rgrids(radii=[15,30,45,60,75],labels=['75','60','45','30','15'],fmt='%d')
   
    col_tbl='bygmkrc'

    if satlist is None:
        satlist=range(uGNSS.MAXSAT)

    nsat=0
    for k,sat in enumerate(satlist):
        if np.all(np.isnan(elv[:,k])):
            continue
        sys,prn=sat2prn(sat)
        idx=elv[:,k]>elmask
        if len(elv[idx,k])==0:
            continue
        z=90-np.rad2deg(elv[idx,k])
        theta=azm[idx,k]
        ax.scatter(theta,z,s=5,c=col_tbl[sys])
        ax.text(theta[0],z[0],sat2id(sat),fontsize=8)
        nsat+=1
    return nsat

