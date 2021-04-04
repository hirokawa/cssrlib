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

def plot_elv(t,elv):
    # elevation plot
    fig=plt.figure('elevation')
    plt.plot(t,np.rad2deg(elv))
    plt.ylim([0,90])
    plt.show()

def skyplot(azm,elv):
    fig=plt.figure('skyplot')
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['N','NE','E','SE','S','SW','W','NW'])
    ax.set_ylim([0,90])
    ax.set_rgrids(radii=[15,30,45,60,75],labels=['75','60','45','30','15'],fmt='%d')
   
    col_tbl='gcmrkby'
   
    for k in range(uGNSS.MAXSAT):
        if np.all(np.isnan(elv[:,k])):
            continue
        sys,prn=sat2prn(k+1)
        idx=elv[:,k]>0
        z=90-np.rad2deg(elv[idx,k])
        theta=azm[idx,k]
        ax.scatter(theta,z,s=1,c=col_tbl[sys])
        ax.text(theta[0],z[0],sat2id(k+1),fontsize=8)

