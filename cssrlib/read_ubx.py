# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import cbitstruct as bs
import numpy as np
import os
import struct as st

import cssrlib as cs

def checksum(buf,i0,length):
    ck=np.array([0,0])
    ck0=np.array([buf[length+i0],buf[length+i0+1]],dtype=np.uint8)
    for i in range(0,length):
        ck[0]=ck[0]+buf[i+i0]
        ck[1]=ck[1]+ck[0]
    return (np.uint8(ck[0])==ck0[0] and np.uint8(ck[1])==ck0[1])


class ubxdec:
    def __init__(self,fcnt):
        self.fcnt=fcnt
        self.facility_p=0
        self.buff=bytearray(212*5)

    def decode_l6msg(self,msg,ofst):
        fmt = 'u32u8u3u2u2u1u1'
        names = ['preamble','prn','vendor','facility','res','sid','alert']
        i=ofst*8
        l6head = bs.unpack_from_dict(fmt,names,msg,i)
        i=i+49
        if l6head['sid']==1:
            self.fcnt=0
#        if l6head['facility']!=self.facility_p:
#            self.fcnt=-1
#        self.facility_p=l6head['facility']
        if self.fcnt<0:
            return -1
        j=1695*self.fcnt
        for k in range(53):
            sz=32 if k<52 else 31
            fmt='u'+str(sz)
            b=bs.unpack_from(fmt,msg,i)
            i=i+sz
            bs.pack_into(fmt,self.buff,j,b[0])
            j=j+sz
        self.fcnt=self.fcnt+1


    def decode_qzssl6(self,msg,i,len,prn0):
        v=st.unpack_from('BBHIBBHH',msg,i)
        #ver=v[0]
        svid=v[1]+192
        cnr=v[2]/256
        timetag=v[3]*1e-3
        #gd=v[4]
        #berr=v[5]
        chinfo=v[6]>>8
        mt=(chinfo>>2)&0x1 # 0:L6D,1:L6E
        chn=chinfo&0x3
        if svid==prn0:
            self.decode_l6msg(msg,i+14)
            if True:
                print("%.1f %.1f %d %d %d %d" % (timetag,cnr,svid,mt,chn,self.fcnt))
        return 0

    def decode_msg(self,msg,i,prn0):
        cls=msg[i+2]
        id=msg[i+3]
        len=msg[i+5]*256+msg[i+4]
        if cls==0x02 and id==0x73: # RXM-QZSSL6
            self.decode_qzssl6(msg,i+6,len,prn0)
        return 0

f_in='data/ublox_D9_1.ubx'
prn0 = 195

blen = os.path.getsize(f_in)

dec = ubxdec(-1)
cr = cs.cssr()

with open(f_in, 'rb') as f:
    msg = f.read()
    i=0
    while i<blen:
       if (msg[i]==0xb5 and msg[i+1]==0x62) and i+6<=blen:
           len=msg[i+5]*256+msg[i+4]
           cls=msg[i+2]
           id=msg[i+3]           
           if i+len+8<=blen and checksum(msg,i+2,len+4):
               #print("%2x %2x %d" % (cls,id,len))  
               dec.decode_msg(msg,i,prn0)
               if cls==0x02 and id==0x73 and dec.fcnt==5:
                   cr.decode_cssr(dec.buff)
                   print("%d" % dec.fcnt)
                   dec.fcnt=-1
           i=i+len+8

                    
