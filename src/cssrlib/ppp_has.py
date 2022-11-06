"""
Galileo HAS correction data decoder

[1] Galileo High Accuracy Service Signal-in-Space
  Interface Control Document (HAS SIS ICD), Issue 1.0, May 2022

"""

import numpy as np
import bitstruct as bs
import galois
from enum import IntEnum

class GNSS(IntEnum):
    GPS=0;GAL=2

class has_decoder():
    sys = []
    sat = []
    sig = []
    nsat_g = {}
    nsat_s = 0
    nsig_s = []
    nsys = 0
    mon_level = 0
    nc = 53
    nsigmax = 6
    GF = galois.GF(256)    
    
    def __init__(self):
        self.nsat_s = 0
    
    def decode_has_header(self, buff, i):
        
        if bs.unpack_from('u24',buff,i)[0]==0xaf3bc3:
            return 0, 0, 0, 0, 0
        
        hass,res,mt,mid,ms,pid=bs.unpack_from('u2u2u2u5u5u8',buff,i)
        ms += 1
        return hass, mt, mid, ms, pid
    
    def decode_page(self, idx, has_pages, gMat, ms):
        """ HPVRS decoding for RS(255,32,224) """
        HASmsg = bytearray()
        k = len(idx)
        if k >= ms:
            Wd = self.GF(has_pages[idx,:]) # kx53
            Dinv = np.linalg.inv(self.GF(gMat[idx,:k])) # kxk            
            Md=Dinv@Wd # decoded message (kx53)
            HASmsg = bytearray(np.array(Md).tobytes())
            
        return HASmsg
    
    def sval(self, u, blen=1, scl=1.0):
        if u==-(1<<(blen-1)):
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
    
    def decode(self,buff):
    
        i=0
        toh,flags,res,mask_id,iod_s=bs.unpack_from('u12u6u4u5u5',buff,i)            
        i+=32
        
        if self.mon_level>0:
            print("TOH={:d} flags={:s} mask_id={:d} iod_s={:d}".format(toh,bin(flags),mask_id,iod_s))
                
        if (flags>>5)&1: # mask block
            self.nsys=bs.unpack_from('u4',buff,i)[0]            
            i+=4
            if self.mon_level>0:
                print("mask block ngnss={:2d}".format(self.nsys))
        
            self.sys = []
            self.sat = []
            self.sig = []
            self.nsat_s = 0
            self.nsat_g = {}
            self.nsig_s = []
            self.sys_s = []
            
            j = 0
            for k in range(self.nsys):
                gnss,masksat,masksig,cmaf=bs.unpack_from('u4u40u16u1',buff,i)
                i+=61
                
                sat_,nsat_=self.decode_mask(masksat,40)
                sig_,nsig_=self.decode_mask(masksig,16,ofst=0)
                self.sys_s += [gnss]
                self.sys += nsat_*[gnss]
                self.sat += sat_
                self.nsat_s += nsat_
                self.nsat_g[gnss] = nsat_
        
                if cmaf==1:
                    lcm=nsat_*nsig_
                    cellmask=bs.unpack_from(nsat_*('u'+str(nsig_)),buff,i)
                    for j in range(nsat_):
                        sig__,nsig__=self.decode_mask(cellmask[j],nsig_,ofst=0)
                        sig_v = []
                        for jj in range(nsig__):
                            sig_v += [sig_[sig__[jj]]]
                        self.nsig_s += [nsig__]
                        self.sig += [sig_v]
                    i+=lcm
                else:
                    self.sig += nsat_*[sig_]
                    self.nsig_s += nsat_*[nsig_]
                j+=1
                nm_idx=bs.unpack_from('u3',buff,i)[0]
                i+=3
                if self.mon_level>1:
                    print("gnss-id={:d} nsat={:d} nsig={:d} nm={:d}".format(gnss,nsat_,nsig_,nm_idx))
                    print(sat_)
                    print(sig_)
            i+=6 # reserved
                
        if (flags>>4)&1: # orbit block
            vi=bs.unpack_from('u4',buff,i)[0]
            i+=4
            if self.mon_level>0:
                print("orbit block vi={:d}".format(vi))
            self.dorb = np.zeros((self.nsat_s,3))
            self.iod_ref = np.zeros(self.nsat_s,dtype=int)
            for k in range(self.nsat_s):
                l_iod = 10 if self.sys[k]==GNSS.GAL else 8
                self.iod_ref[k],dr,dit,dct=bs.unpack_from('u'+str(l_iod)+'s13s12s12',buff,i)
                self.dorb[k,0] = self.sval(dr , 13, 0.0025)
                self.dorb[k,1] = self.sval(dit, 12, 0.008)
                self.dorb[k,2] = self.sval(dct, 12, 0.008)
                i+=l_iod+37
                if self.mon_level>1:
                    print("sys={:2d} prn={:3d} iod={:3d} dorb=[{:8.3f},{:8.3f},{:8.3f}]".format(self.sys[k],self.sat[k],self.iod_ref[k],self.dorb[k,0],self.dorb[k,1],self.dorb[k,2]))
        
        
        if (flags>>3)&1: # clock block
            vi=bs.unpack_from('u4',buff,i)[0]
            i+=4
            gm = {}
            if self.mon_level>0:
                print("clock block vi={:d}".format(vi))
            dcm=bs.unpack_from(self.nsys*'u2',buff,i)
            for k in range(self.nsys):
                gm[self.sys_s[k]] = dcm[k]+1
            i+=self.nsys*2
            dcc=bs.unpack_from(self.nsat_s*'s13',buff,i)
            i+=self.nsat_s*13
            
            self.dclk = np.zeros(self.nsat_s)
            for k in range(self.nsat_s):
                self.dclk[k] = self.sval(dcc[k], 13, 0.0025*gm[self.sys[k]])
                if self.mon_level>1:
                    print("clk sys={:2d} prn={:3d} dclk={:8.4f} dcm={:d}"
                          .format(self.sys[k],self.sat[k],self.dclk[k],gm[self.sys[k]]))
        
        if (flags>>2)&1: # clock subset block
            gnss,dcm=bs.unpack_from('u4u2',buff,i)
            if self.nsat_s == 0:
                return
            idx_s = np.where(np.array(self.sys)==gnss)
            i+=6
            mask_s=bs.unpack_from('u'+str(self.nsat_g[gnss]),buff,i)[0]
            i+=self.nsat_g[gnss]
            idx_,nsat_=self.decode_mask(mask_s,self.nsat_g[gnss],ofst=0)
            dcc=bs.unpack_from(nsat_*'s13',buff,i)
            i+=nsat_*13
            for k in range(nsat_):
                j = idx_s[idx_[k]]
                self.dclk[j] = self.sval(dcc[k], 13, 0.0025*(dcm+1))
                if self.mon_level>1:
                    print("dclk sys={:2d} prn={:3d} dclk={:8.4f} dcm={:d}".format(gnss,self.sat[j],self.dclk[j],dcm+1))
        
        if (flags>>1)&1: # code bias block
            vi=bs.unpack_from('u4',buff,i)[0]
            i+=4
            if self.mon_level>0:
                print("cb vi={:d}".format(vi))
            self.cb = np.zeros((self.nsat_s,self.nsigmax))
            for k in range(self.nsat_s):
                print("sys={:2d} sat={:3d} cb=".format(self.sys[k],self.sat[k]), end='')
                for j in range(self.nsig_s[k]):
                    cb_ = bs.unpack_from('s11',buff,i)[0]
                    self.cb[k,j] = self.sval(cb_, 11, 0.02)
                    i+=11
                    if self.mon_level>1:
                        print("{:6.2f}".format(self.cb[k,j]),end='\t')
                print("")
            
        if (flags>>0)&1: # phase bias block
            vi=bs.unpack_from('u4',buff,i)[0]
            i+=4
            if self.mon_level>0:
                print("pb vi={:d}".format(vi))
            self.pb = np.zeros((self.nsat_s,self.nsigmax))
            for k in range(self.nsat_s):
                print("sys={:2d} sat={:3d} pb=".format(self.sys[k],self.sat[k]), end='')
                for j in range(self.nsig_s[k]):
                    pb_,pdi = bs.unpack_from('s11u2',buff,i)
                    self.pb[k,j] = self.sval(pb_, 11, 0.01)
                    i+=13
                    if self.mon_level>1:
                        print("{:6.2f}".format(self.pb[k,j]),end='\t')
                print("")