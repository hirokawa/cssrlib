# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

from enum import IntEnum

class uGNSS(IntEnum):
    GPS=0;SBS=1;GAL=2;BDS=3;QZS=5;GLO=6
    GPSMAX=32;GALMAX=36;BDSMAX=63;QZSMAX=10;
    GLOMAX=24;SBSMAX=24

class uSIG(IntEnum):
    GPS_L1CA=0;GPS_L2CL=3;GPS_L2CM=4;SBS_L1CA=0
    GAL_E1C=0;GAL_E1B=1;GAL_E5BI=5;GAL_E5BQ=6
    BDS_B1ID1=0;BDS_B1ID2=1;BDS_B2ID1=2;BDS_B2ID2=3
    QZS_L1CA=0;QZS_L1S=1;QZS_L2CM=4;QZS_L2CL=5
    GLO_L1OF=0;GLO_L2OF=2

class GPSTime():
    """
    GPS time class [week, tow]
    """
    def __init__(self,week=0,tow=0):
        self.week = week
        self.tow = tow
        self.seconds_in_week = 604800  

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

