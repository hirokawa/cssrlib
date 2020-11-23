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
    
class ubxdec:
    MAXSAT=uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX
    RXM_SFRBX =0x0213
    RXM_RAWX  =0x0215
    RXM_QZSSL6=0x0273 
    
    def __init__(self,prn0,fcnt=-1):
        self.tow=0
        self.week=0
        self.fcnt=fcnt
        self.facility_p=0
        self.buff=bytearray(212*5)
        self.timetag=0
        self.prn0=prn0
        self.sat_n=0
        self.nw=0
        self.nm=0
        self.subfrm=[]
        self.eph = [ephsat() for i in range(self.MAXSAT)]
        self.nav=navmsg()   
        for k in range(self.MAXSAT):       
            self.subfrm.append(bytearray(200))

    def checksum(self,buf,i0,length):
        """calculate checksum in ubx message """
        ck=np.array([0,0])
        ck0=np.array([buf[length+i0],buf[length+i0+1]],dtype=np.uint8)
        for i in range(0,length):
            ck[0]=ck[0]+buf[i+i0]
            ck[1]=ck[1]+ck[0]
        return (np.uint8(ck[0])==ck0[0] and np.uint8(ck[1])==ck0[1])

    def decode_l6msg(self,msg,ofst):
        """decode QZS L6 message """
        fmt = 'u32u8u3u2u2u1u1'
        names = ['preamble','prn','vendor','facility','res','sid','alert']
        i=ofst*8
        l6head = bs.unpack_from_dict(fmt,names,msg,i)
        i=i+49
        if l6head['sid']==1:
            self.fcnt=0
        if l6head['facility']!=self.facility_p:
            #print('t=%10.3f facility changed %d => %d' 
            #      % (self.timetag,self.facility_p,l6head['facility']))
            self.fcnt=-1
        self.facility_p=l6head['facility']
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

    def decode_qzssl6(self,msg,i,len):
        """decode RXM-QZSSL6 message """
        v=st.unpack_from('BBHIBBHH',msg,i)
        #ver=v[0]
        svid=v[1]+192
        #cnr=v[2]/256
        self.timetag=v[3]*1e-3   
        #gd=v[4]
        #berr=v[5]
        chinfo=v[6]>>8
        mt=(chinfo>>2)&0x1 # 0:L6D,1:L6E
        #chn=chinfo&0x3
        if svid==self.prn0 and mt==0:
            #print("t=%.2f berr=%d mt=%d chn=%d" % (self.timetag,berr,mt,chn))
            self.decode_l6msg(msg,i+14)
        return 0

    def decode_sfrbx(self,msg,i,len):
        """decode RXM-SFRBX message """
        v=st.unpack_from('BBBBBBBB',msg,i)
        sys=v[0]
        prn=v[1]+192 if sys==uGNSS.QZS else v[1]
        self.sat_n=prn2sat(sys,prn)
        #freq=v[3]
        self.nw=v[4]
        #chn=v[5]
        #ver=v[6]
        i=i+8
        self.buff=st.unpack_from('L'*self.nw,msg,i)
        #print("SFRBX sys=%d prn=%d nw=%d" % (self.sys_n,self.prn_n,self.nw))
        return 0

    def decode_rawx(self,msg,i,len):
        """decode RXM-RAWX message """
        v=st.unpack_from('dHbBBB',msg,i)
        self.tow=v[0]
        self.week=v[1]
        #leaps=v[2]
        nm=v[3]
        #stat=v[4]
        #ver=v[5]
        i=i+16
        self.sat=np.zeros(nm,dtype=int)
        self.sigid=np.zeros((nm,2),dtype=int)
        #self.cno=np.zeros(nm,dtype=int)
        self.obs=np.zeros((nm,2*4))
        j=0
        for k in range(nm):
            v=st.unpack_from('ddfBBBBHBBBBB',msg,i)
            i=i+32
            sys=v[3]
            if sys==uGNSS.GLO or sys==uGNSS.BDS or sys==uGNSS.SBS:
                continue # skip Glonass,BDS,SBAS
            prn=v[4]+192 if sys==uGNSS.QZS else v[4]
            sigid=v[5]
            if sys==uGNSS.GPS:
                ifreq=0 if sigid==uSIG.GPS_L1CA else 1
            if sys==uGNSS.GAL:
                ifreq=0 if sigid==uSIG.GAL_E1C else 1
            if sys==uGNSS.QZS:
                ifreq=0 if sigid==uSIG.QZS_L1CA else 1
            sat=prn2sat(sys,prn)
            idx=np.where(self.sat==sat)[0]
            if np.size(idx)==0:
                self.sat[j]=sat
            else:
                j=idx[0]                
            self.obs[j,ifreq*4]  =v[0] # PR  
            self.obs[j,ifreq*4+1]=v[1] # CP
            self.obs[j,ifreq*4+2]=v[2] # doppler
            self.obs[j,ifreq*4+3]=v[8] # cno
            self.sigid[j,ifreq]=sigid
            if np.size(idx)==0:
                j=j+1
            else:               
                print("RAW t=%.3f sys=%d prn=%3d sig=%d,%d pr=%.3f,%.3f" % (self.tow,sys,prn,self.sigid[j,0],self.sigid[j,1],self.obs[j,0],self.obs[j,4]))            
            #freqid=v[6]
            #locktime=v[7]
            #prstdev=v[9]
            #cpstdev=v[10]
            #dpstdev=v[11]*0.002
            #trkstat=v[12]
        self.nm=j
        return 0

    def decode_msg(self,msg,i):
        """decode ubx messages """
        msgid=msg[i+2]<<8|msg[i+3]
        len=msg[i+5]<<8|msg[i+4]
        if msgid==self.RXM_SFRBX:
            self.decode_sfrbx(msg,i+6,len)
        if msgid==self.RXM_RAWX:
            self.decode_rawx(msg,i+6,len)
        if msgid==self.RXM_QZSSL6:
            self.decode_qzssl6(msg,i+6,len)
        return 0
        
    def decode_nav(self,buff,sat):
        """decode GPS/QZS LNAV navigation message """
        if buff[0]>>24==0x8b or self.nw!=10:
            return -1 # CNAV is not supported
        w = np.zeros(10,dtype=int)
        for i in range(10):
            w[i]=buff[i]>>6
        id=(w[1]>>2)&7
        if id<1 or 5<id:
            return -1
        subfrm=bytearray(30)
        for i in range(10):
            s=(buff[i]>>6) & 0xffffff
            bs.pack_into('u24',subfrm,i*24,s)
        if id<=3: # decode ephemeris
            self.nav.decode_gps_lnav(self.eph,subfrm,sat,id)        
        return 0

    def decode_enav(self,buff,sat):
        """decode Galileo INAV navigation message """        
        subfrm=bytearray(32)
        for i in range(8):  
            bs.pack_into('u32',subfrm,i*32,buff[i])
        v1=bs.unpack_from('u1u1',subfrm,0)
        v2=bs.unpack_from('u1u1',subfrm,16*8)
        part1=v1[0]
        page1=v1[1]
        part2=v2[0]
        page2=v2[1]
        if page1==1 or page2==1: # skip alert
            return 0
        if part1!=0 or part2!=1: # even/odd error
            return -1
        wt=bs.unpack_from('u6',subfrm,2)[0]
        if wt>6:
            return 0
        if wt==2:
            self.eph[sat-1].status=0
        k=wt*16
        i=2
        for j in range(14):
            self.subfrm[sat-1][k]=bs.unpack_from('u8',subfrm,i)[0]
            k=k+1
            i=i+8
        i=16*8
        for j in range(2):
            self.subfrm[sat-1][k]=bs.unpack_from('u8',subfrm,i)[0]
            k=k+1
            i=i+8            
        self.eph[sat-1].status=self.eph[sat-1].status|(1<<wt)
        if self.eph[sat-1].status!=0x7f:
            return 0
        self.nav.decode_gal_inav(self.eph,self.subfrm[sat-1],sat)
        return 0

    def decode_ubx(self,f_in):
        """decode ublox binary message from file """
        blen = os.path.getsize(f_in)
        blmax=min(1000000,blen)
        with open(f_in,'rb') as f:
            msg = f.read(blmax)
            i=0
            while i<blmax:
                if i+6>blmax:
                    break
                if msg[i]==0xb5 and msg[i+1]==0x62:
                    len=msg[i+5]<<8|msg[i+4]
                    msgid=msg[i+2]<<8|msg[i+3]
                    if i+len+8>blmax:
                        break
                    if self.checksum(msg,i+2,len+4)==False:
                        print("checksum error.")
                        i=i+len+8
                        continue
                    self.decode_msg(msg,i) # decode UBX binary format

                    if msgid==self.RXM_QZSSL6 and dec.fcnt==5:
                        cr.decode_cssr(self.buff)
                        self.fcnt=-1
                    if msgid==self.RXM_SFRBX:
                        sys,prn=sat2prn(self.sat_n)
                        if sys==uGNSS.GPS or sys==uGNSS.QZS:
                            self.decode_nav(self.buff,self.sat_n)
                        if sys==uGNSS.GAL:
                            self.decode_enav(self.buff,self.sat_n)
                    i=i+len+8
                else:
                    i=i+1


if __name__ == '__main__':
    #f_in='../data/ublox_D9_1.ubx'
    prn0 = 195

    f_in='../data/logf9p2020228a.ubx'

    dec = ubxdec(prn0)
    cr = cs.cssr()
    dec.decode_ubx(f_in)



                    
