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
import ecdsa
import Crypto
import hmac
import hashlib
import bitstring as bstr
    
class ubxdec:
    MAXSAT=uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX
    RXM_SFRBX =0x0213
    RXM_RAWX  =0x0215
    RXM_QZSSL6=0x0273
    
    __hash_table = {0: hashlib.sha256,1: hashlib.sha3_224,2: hashlib.sha256}
    
    def __init__(self,prn0,fcnt=-1):
        self.monlevel=1
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
        self.kroot=[]
        self.mack=[]
        self.eph = [ephsat() for i in range(self.MAXSAT)]
        self.nav=navmsg() 

        for k in range(self.MAXSAT):       
            self.subfrm.append(bytearray(200))
        for k in range(uGNSS.GALMAX):       
            self.kroot.append(bytearray(15))
            self.mack.append(bytearray(60))
        
        # OSNMA
        self.cid=-1
        self.nb=-1
        self.flg_kroot=[0]*uGNSS.GALMAX
        self.mac_sz_t=[10,11,12,13,14,15,16,17,18,19,20,30,32,0,0,0]
        self.key_sz_t=[80,82,84,86,88,90,92,94,96,98,100,112,128,256,0,0]
        self.dsm_kroot=bytearray(13*16)
        self.flg_dsm=0  
        self.pubk_path='./PubK.pem'

    def checksum(self,buf,i0,length):
        """calculate checksum in ubx message """
        ck=np.array([0,0])
        ck0=np.array([buf[length+i0],buf[length+i0+1]],dtype=np.uint8)
        for i in range(0,length):
            ck[0]=ck[0]+buf[i+i0]
            ck[1]=ck[1]+ck[0]
        return (np.uint8(ck[0])==ck0[0] and np.uint8(ck[1])==ck0[1])

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
            cs.decode_l6msg(msg,i+14)
        return 0

    def decode_sfrbx(self,msg,i,len):
        """decode RXM-SFRBX message """
        v=st.unpack_from('BBBBBBBB',msg,i)
        sys=v[0]
        prn=v[1]+192 if sys==uGNSS.QZS else v[1]
        self.sat_n=prn2sat(sys,prn)
        self.freq=v[3]
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
                if self.monlevel>=3:
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

    def decode_osnma(self,buff,mack,prn):
        self.nma_header=buff[0]
        cid=(self.nma_header>>2)&0x3
        dsm_header=buff[1]
        dsm_id=(dsm_header>>4)&0xf
        dsm_blk_id=dsm_header&0xf
            
        if dsm_blk_id==0:
            self.cid=cid
            i=16
            v=bs.unpack_from('u4u4u2u2u2u2u4u4u1u12u3u48',buff,i)
            self.nb=v[0]+6
            self.pkid=v[1]
            self.cidkr=v[2]
            self.nmack=v[3]
            self.hf=v[4]
            self.mf=v[5]
            self.ks=v[6]
            self.ms=v[7]
            ks=self.key_sz_t[self.ks]
            ms=self.mac_sz_t[self.ms]
            self.mo=v[8]
            self.nmac=(480//self.nmack-ks)//(ms+16)
            self.kroot_wn=v[9]
            self.kroot_dow=v[10]
            self.alp=v[11]
        
        if cid==self.cid and dsm_id<=11: # DSM-KROOT
            for k in range(13):
                self.dsm_kroot[k+dsm_blk_id*13]=buff[k+2]
            self.flg_dsm|=(1<<dsm_blk_id)        
            if self.flg_dsm == (1<<self.nb)-1:
                ks=self.key_sz_t[self.ks]
                dsm_len=13*(self.nb-1)*8-ks
                if dsm_len>1056:
                    self.ds_len=1056
                elif dsm_len>768:
                    self.ds_len=768
                elif dsm_len>512:
                    self.ds_len=512
                else:
                    self.ds_len=448
                blen=self.ds_len//8
                self.ds=bytearray(blen)
                for k in range(blen):
                    self.ds[k]=self.dsm_kroot[13+k]
                dlen=ks//8
                r=ks-dlen*8
                if r>0:
                    dlen+=1
                self.kroot=bytearray(dlen)                    
                for k in range(dlen):
                    self.kroot[k]=self.dsm_kroot[13+blen+k]
                    if r>0 and k==dlen-1:
                        self.kroot[k] &= 0xff<<(8-r)
                
                # verification of kroot with digital signature
                message =bytearray(dlen+11)
                message[0]=self.nma_header
                for k in range(10):
                    message[k+1]=self.dsm_kroot[k+1]
                for k in range(dlen):
                    message[k+11]=self.kroot[k]
                
                hash_func = self.__hash_table[self.hf]
                
                with open(self.pubk_path) as f:
                    vk = ecdsa.VerifyingKey.from_pem(f.read(), hashfunc=hash_func)
                    #result = vk.verify(self.ds, message)
                print(self.flg_dsm)
                
                ms=self.mac_sz_t[self.ms]
                mstr='u'+str(ms)+'u8u4u4'
                i=0
                for k in range(self.nmack):
                    for j in range(self.nmac):
                        v=bs.unpack_from(mstr,mack,i);i+=ms+16
                        print('PRN=%d ADKD=%d IOD=%d' % (v[0],v[1],v[2]))
                        
        
        return 0

    def decode_enav(self,buff,sat):
        """decode Galileo INAV navigation message """
        (sys,prn)=sat2prn(sat)        
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
        if wt>15:
            #if self.monlevel>=1:
            #    print('wt=%d' % wt)
            return 0
        if wt==2:
            self.eph[sat-1].status=0
        k=wt*16
        i=2
        for j in range(14):
            self.subfrm[sat-1][k]=bs.unpack_from('u8',subfrm,i)[0]
            k+=1
            i+=8
        i=16*8+2
        for j in range(2):
            self.subfrm[sat-1][k]=bs.unpack_from('u8',subfrm,i)[0]
            k+=1
            i+=8            
        # OSNMA
        nma=bs.unpack_from('u8u8u8u8',subfrm,i)
        if nma[0]>0:
            towi=int(self.tow+0.5)
            if towi%2==1:               
                idx=towi%30
                if idx==1:
                    idx+=30
                idx=(idx-3)//2
                if idx==0:
                    self.flg_kroot[prn-1]=0
                self.kroot[prn-1][idx]=nma[0]
                self.mack[prn-1][idx*3+0]=nma[1]
                self.mack[prn-1][idx*3+1]=nma[2]
                self.mack[prn-1][idx*3+2]=nma[3]
                self.flg_kroot[prn-1]|=(1<<idx)
                if self.flg_kroot[prn-1]==0x7fff:
                    print("OSNMA %d %3d %02x" % (towi,prn,nma[0]))
                    self.decode_osnma(self.kroot[prn-1],self.mack[prn-1],prn)
                
        self.eph[sat-1].status=self.eph[sat-1].status|(1<<wt)
        if self.eph[sat-1].status!=0x7f:
            return 0
        self.nav.decode_gal_inav(self.eph,self.subfrm[sat-1],sat)
        return 0

    def decode_ubx(self,f_in):
        """decode ublox binary message from file """
        blen = os.path.getsize(f_in)
        blmax=min(10000000,blen)
        with open(f_in,'rb') as f:
            msg = f.read(blmax)
            i=0
            while i<blmax:
                if i+6>blmax:
                    print("size exceeds blmax")
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

    f_in='c:/work/log/f9p044g.ubx'

    dec = ubxdec(prn0)
    cr = cs.cssr()
    dec.decode_ubx(f_in)



                    
