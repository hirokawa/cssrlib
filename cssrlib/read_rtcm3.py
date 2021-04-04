# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import cbitstruct as bs
import numpy as np
import os
import struct as st
from enum import IntEnum
import cssrlib.cssrlib as cs
from nav import ephsat,navmsg
from gnss import uGNSS,uSIG,prn2sat,sat2prn
import crcmod
import crcmod.predefined

crc24=crcmod.crcmod.mkCrcFun(0x1864CFB,rev=False,initCrc=0x000000,xorOut=0x000000)    

class rtcm3dec:
    MAXSAT=uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX
    RXM_SFRBX =0x0213
    RXM_RAWX  =0x0215
    RXM_QZSSL6=0x0273 
    
    def __init__(self,fcnt=-1):
        self.tow=0
        self.week=0
        self.fcnt=fcnt
        self.facility_p=0
        self.buff=bytearray(212*5)
        self.timetag=0
        self.sat_n=0
        self.nw=0
        self.nm=0


    def decode(self,f_in):
        """decode ublox binary message from file """
        blen = os.path.getsize(f_in)
        blmax=min(1000000,blen)
        with open(f_in,'rb') as f:
            msg = f.read(blmax)
            j=0
            while j<blmax:
                if j+6>blmax:
                    break
                if msg[j]==0xd3:
                    v=bs.unpack_from('u10u12',msg,j*8+14)
                    rlen=v[0];msgtype=v[1]
                    cs=crc24(msg[j:j+rlen+3+3])
                    if cs!=0:
                        print('checksum error len=%d' % (rlen))
                        j+=rlen+6
                        continue
                    
                    if msgtype==4073:
                        i=j*8+24
                        cr.decode_cssr(msg,i)
                        self.fcnt=-1
 
                    j+=len+8
                else:
                    j+=1


if __name__ == '__main__':


    f_in='c:/work/log/cssr018m.rtcm3'

    rtcm3 = rtcm3dec()
    cr = cs.cssr()
    rtcm3.decode(f_in)



                    
