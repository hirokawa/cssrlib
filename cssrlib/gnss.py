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

