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

