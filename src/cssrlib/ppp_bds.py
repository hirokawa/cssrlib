"""
BDS PPP correction data decoder

[1] BeiDou Navigation Satellite System Signal In Space
Interface Control Document Precise Point Positioning Service Signal PPP-B2b
(Version 1.0) , July 2020

"""

import numpy as np
import bitstruct as bs
from enum import IntEnum
from cssrlib.gnss import uGNSS, prn2sat, sat2prn

class sGNSS(IntEnum):
    """ class to define GNSS """
    GPS = 0
    GLO = 1
    GAL = 2
    BDS = 3
    QZS = 4
    SBS = 5

class bds_decoder():
    sys = []
    sat = []
    sig = []
    nsat_g = {}
    nsat_s = 0
    nsig_s = []
    nsys = 0
    mon_level = 0
    nc = 53
    nsigmax = 8
    nsatmax = 256
    
    msg_t = [1,2,3,4,5,6,7,63]
    
    def __init__(self):
        self.nsat_s = 0
        self.iodp = -1
        
        self.dorb = np.zeros((self.nsatmax,3))
        self.dclk = np.zeros((self.nsatmax))
        self.cb = np.zeros((self.nsatmax,self.nsigmax))
        self.ura = np.zeros((self.nsatmax))
        self.iodn = np.zeros((self.nsatmax),dtype=int)
        self.iodc = np.zeros((self.nsatmax),dtype=int)
        self.mode = np.zeros((self.nsatmax,self.nsigmax),dtype=int)
    
    def sval(self, u, blen=1, scl=1.0):
        if u==-(1<<(blen-1))+1:
            return np.nan
        return u*scl
    
    def decode_mask(self, din, bitlen, ofst=1):
        """ decode n-bit mask with offset """
        v = []
        n = 0
        for k in range(0, bitlen):
            if din & 1 << (bitlen-k-1):
                v.append(k+ofst)
                n += 1
        return (v, n)
    
    def slot2prn(self,slot):
        prn = 0
        sys = uGNSS.NONE
        if slot>=1 and slot<=63:
            prn = slot
            sys = uGNSS.BDS
        elif slot<=100:
            prn = slot-63
            sys = uGNSS.GPS
        elif slot<=137:
            prn = slot-100
            sys = uGNSS.GAL
        elif slot<=174:
            prn = slot-137
            sys = uGNSS.GLO
        return sys, prn
    
    def decode_mt1(self,buff,i):
        """ decode MT1: Satellite Mask """
        
        self.sys = []
        self.sat = np.zeros(self.nsatmax,dtype=int)
        self.sig = []
        self.nsat_s = 0
        self.nsat_g = {}
        self.nsig_s = []
        self.sys_s = []
        
        iodp,mask_bds,mask_gps,mask_gal,mask_glo = bs.unpack_from('u4u63u37u37u37',buff,i)
        i+=4+63+37+37+37
        self.iodp = iodp
        
        if mask_bds!=0:
            prn_,nsat_ = self.decode_mask(mask_bds,63)
            for k in range(nsat_):
                self.sat[k+self.nsat_s] = prn2sat(uGNSS.BDS, prn_[k])
            self.sys_s += [uGNSS.BDS]
            self.nsat_s += nsat_

        if mask_gps!=0:
            prn_,nsat_ = self.decode_mask(mask_gps,37)
            for k in range(nsat_):
                self.sat[k+self.nsat_s] = prn2sat(uGNSS.GPS, prn_[k])            
            self.sys_s += [uGNSS.GPS]
            self.nsat_s += nsat_

        if mask_gal!=0:
            prn_,nsat_ = self.decode_mask(mask_gal,37)
            for k in range(nsat_):
                self.sat[k+self.nsat_s] = prn2sat(uGNSS.GAL, prn_[k]) 
            self.sys_s += [uGNSS.GAL]
            self.nsat_s += nsat_
            
        if mask_glo!=0:
            prn_,nsat_ = self.decode_mask(mask_bds,37)
            for k in range(nsat_):
                self.sat[k+self.nsat_s] = prn2sat(uGNSS.GLO, prn_[k])             
            self.sys_s += [uGNSS.GLO]  
            self.nsat_s += nsat_
            
        return

    def decode_mt2(self,buff,i):
        """ decode MT2: Orbit Correction and URA """
        if self.nsat_s<=0:
            return
        for k in range(6):
            slot,iodn,iodc,rc,ac,cc,urai = bs.unpack_from('u9u10u3s15s13s13u6',buff,i)
            i+=9+10+3+15+13+13+6
            sys, prn = self.slot2prn(slot)
            sat = prn2sat(sys,prn)
            if sat in self.sat:
                idx = np.where(self.sat==sat)[0][0]

                self.iodn[idx] = iodn
                self.iodc[idx] = iodc
                self.dorb[idx,0] = self.sval(rc,15,0.0016)
                self.dorb[idx,1] = self.sval(ac,13,0.0064)
                self.dorb[idx,2] = self.sval(cc,13,0.0064)

                self.ura[idx] = urai  # TBD
                
                if self.mon_level>=1:
                    print("{:2d} {:2d} {:3d} {:1d} {:6.3f} {:6.3f} {:6.3f} {:2d}"
                          .format(sys,prn,iodn,iodc,self.dorb[idx,0],self.dorb[idx,1],self.dorb[idx,2],urai))

        return
    
    def decode_mt3(self,buff,i):
        """ decode MT3: Differential Code Bias """
        if self.nsat_s<=0:
            return
        nsat = bs.unpack_from('u5',buff,i)[0]
        i+=5
        for k in range(nsat):
            slot,nsig = bs.unpack_from('u9u4',buff,i)
            sys, prn = self.slot2prn(slot)
            i+=13
            if self.mon_level>=1:
                print('{:2d} {:2d} {:1d} '.format(sys,prn,nsig),end='')
            for j in range(nsig):
                mode,cb = bs.unpack_from('u4s12',buff,i)
                i+=16
                self.cb[slot,j] = self.sval(cb, 12, 0.017)
                self.mode[slot,j] = mode
                if self.mon_level>=1:
                    print("{:2d} {:8.3f} ".format(mode,self.cb[slot,j]),end='')
            print('')
        return
    
    def decode_mt4(self,buff,i):
        """ decode MT4: Clock """
        iodp,st1 = bs.unpack_from('u4u5',buff,i)
        if iodp!=self.iodp:
            return
        i+=9
        ofst = 23*st1
        print("MT4 iodp={:2d} st1={:1d}".format(iodp,st1))
        for k in range(23):
            j=k+ofst
            iodc,c0 = bs.unpack_from('u3s15',buff,i)
            i+=18
            if j<self.nsat_s:
                sys,prn = sat2prn(self.sat[j])
                self.dclk[j] = self.sval(c0, 15, 0.0016)
                if np.isfinite(self.dclk[j]) and self.mon_level>=1:
                    print("{:2d} {:2d} {:1d} {:8.4f}".format(sys,prn,iodc,self.dclk[j]))
        return
    
    def decode_mt5(self,buff,i):
        return
    
    def decode_mt6(self,buff,i):
        return
    
    def decode_mt7(self,buff,i):
        return
    
    def decode(self,prn,buff):
    
        i=0
        msgid = bs.unpack_from('u6',buff,i)[0]
        i+=6
        if msgid not in self.msg_t:
            print("unknown msgid: {:d}".format(msgid))
            return
        
        tod, res, iodssr = bs.unpack_from('u17u4u2',buff,i)
        i+=17+4+2
        
        if self.mon_level>0:
            print("prn={:2d} mt={:2d} tod={:6d} iodssr={:3d}".format(prn, msgid, tod, iodssr))    
                
        if msgid == 1: # mask block
            self.decode_mt1(buff,i)
        
        elif msgid == 2: # orbit/ura block
            self.decode_mt2(buff,i)

        elif msgid == 3: # cbias block
            self.decode_mt3(buff,i)

        elif msgid == 4: # clock block
            self.decode_mt4(buff,i)        
        
        elif msgid == 5: # ura block
            self.decode_mt5(buff,i)
        
        elif msgid == 6: # clock+orbit block
            self.decode_mt6(buff,i)

        elif msgid == 7: # clock+orbit block
            self.decode_mt7(buff,i)            
        
