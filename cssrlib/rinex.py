# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

import datetime
import numpy as np
from gnss import uGNSS,rSIG,rCST,sat2prn,Eph,prn2sat,gpst2time,Obs

class rnxdec:
    MAXSAT=uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX
   
    def __init__(self):
        self.fobs=None
        self.freq_tbl={rSIG.L1C:0,rSIG.L2W:1,rSIG.L2L:1,rSIG.L5Q:2,rSIG.L7Q:1}
        self.gnss_tbl={'G':uGNSS.GPS,'E':uGNSS.GAL,'J':uGNSS.QZS}
        self.sig_tbl={'1C':rSIG.L1C,'2W':rSIG.L2W,'2L':rSIG.L2L,
                 '5Q':rSIG.L5Q,'7Q':rSIG.L7Q}
        self.nf=4

    def decode_nav(self,navfile):
        """decode RINEX Navigation message from file """
        
        nav=[]
        with open(navfile,'rt') as fnav:
            for line in fnav:
                if line[60:73]=='END OF HEADER':
                    break    
                if line[60:80]=='RINEX VERSION / TYPE':
                    self.ver=float(line[4:10])
                    if self.ver<3.02:
                        return -1        
        
            for line in fnav:
                if line[0] not in self.gnss_tbl:
                    continue
                sys=self.gnss_tbl[line[0]]
                prn=int(line[1:3])
                if sys==uGNSS.QZS:
                    prn+=192
                sat=prn2sat(sys,prn)
                eph=Eph(sat)
            
                year=int(line[4:8])
                month=int(line[9:11])
                day=int(line[12:14])
                hour=int(line[15:17])                        
                minute=int(line[18:20])
                sec=int(line[21:23])            
                eph.toc=datetime.datetime(year,month,day,hour,minute,sec)
                
                eph.af0=float(line[23:42])
                eph.af1=float(line[42:61])
                eph.af2=float(line[61:80])        
        
                line=fnav.readline()
                eph.iode=int(float(line[4:23]))
                eph.crs=float(line[23:42])
                eph.deln=float(line[42:61])
                eph.M0=float(line[61:80])         
        
                line=fnav.readline()
                eph.cuc=float(line[4:23])
                eph.e=float(line[23:42])
                eph.cus=float(line[42:61])
                sqrtA=float(line[61:80])
                eph.A=sqrtA**2
                
                line=fnav.readline()
                eph.toes=int(float(line[4:23]))
                eph.cic=float(line[23:42])
                eph.OMG0=float(line[42:61])
                eph.cis=float(line[61:80])    
                
                line=fnav.readline()
                eph.i0=float(line[4:23])
                eph.crc=float(line[23:42])
                eph.omg=float(line[42:61])
                eph.OMGd=float(line[61:80]) 
                
                line=fnav.readline()
                eph.idot=float(line[4:23])
                eph.code=int(float(line[23:42])) # source for GAL
                eph.week=int(float(line[42:61]))
                #if len(line)>=80:
                #    L2_P_data=int(float(line[61:80]) )
                
                line=fnav.readline()
                eph.sva=int(float(line[4:23]))
                eph.svh=int(float(line[23:42]))
                eph.tgd=float(line[42:61])
                if sys==uGNSS.GAL:
                    tgd_b=float(line[61:80])
                    if (eph.code>>9)&1:
                        eph.tgd=tgd_b
                else:
                    eph.iodc=int(float(line[61:80]))
        
                line=fnav.readline()
                tot=int(float(line[4:23]))
                if len(line)>=42:            
                    eph.fit=int(float(line[23:42]))
                
                eph.toe=gpst2time(eph.week,eph.toes)
                eph.tot=gpst2time(eph.week,tot)
        
                nav.append(eph)
    
        return nav

    def decode_obsh(self,obsfile):
        self.sigid=np.ones((uGNSS.GNSSMAX,rSIG.SIGMAX),dtype=int)*rSIG.NONE
        self.typeid=np.ones((uGNSS.GNSSMAX,rSIG.SIGMAX),dtype=int)*rSIG.NONE
        self.nsig=np.zeros((uGNSS.GNSSMAX),dtype=int)        
        self.fobs=open(obsfile,'rt')
        self.pos=np.array([0,0,0])
        for line in self.fobs:
            if line[60:73]=='END OF HEADER':
                break    
            elif line[60:80]=='RINEX VERSION / TYPE':
                self.ver=float(line[4:10])
                if self.ver<3.02:
                    return -1
            elif line[60:79]=='APPROX POSITION XYZ':
                self.pos=np.array([float(line[0:14]),
                          float(line[14:28]),
                          float(line[28:42])])
            elif line[60:79]=='SYS / # / OBS TYPES':
                if line[0] in self.gnss_tbl:
                    sys=self.gnss_tbl[line[0]]
                else:
                    continue
                self.nsig[sys]=int(line[3:6])
                for k in range(self.nsig[sys]):
                    sig=line[7+4*k:10+4*k]
                    if sig[0]!='C' and sig[0]!='L':
                        continue
                    self.typeid[sys][k] = 0 if sig[0]=='C' else 1
                    if sig[1:3] in self.sig_tbl:
                        self.sigid[sys][k]=self.sig_tbl[sig[1:3]]
        return 0


    def decode_obs(self):
        """decode RINEX Observation message from file """
        obs=Obs()
        
        for line in self.fobs:
            if line[0]!='>':
                continue
            nsat=int(line[32:35])
            year=int(line[2:6])
            month=int(line[7:9])
            day=int(line[10:12])
            hour=int(line[13:15])                        
            minute=int(line[16:18])
            sec=float(line[19:29])
            obs.t=datetime.datetime(year,month,day,hour,minute,int(sec),
                                     int((sec-int(sec))*1e6))
            obs.data=np.zeros((nsat,self.nf*4))
            obs.P=np.zeros((nsat,self.nf))
            obs.L=np.zeros((nsat,self.nf))
            obs.mag=np.zeros((nsat,self.nf))
            obs.sat=np.zeros(nsat,dtype=int)
            print("%2d %2d %6.1f %2d" % (hour,minute,sec,nsat))
            for k in range(nsat):
                line = self.fobs.readline()
                if line[0] not in self.gnss_tbl:
                    continue
                sys=self.gnss_tbl[line[0]]
                prn=int(line[1:3])
                if sys==uGNSS.QZS:
                    prn+=192
                obs.sat[k]=prn2sat(sys,prn)
                nsig_max=(len(line)-4)//16
                for i in range(self.nsig[sys]):
                    if i>=nsig_max:
                        break
                    obs_=line[16*i+4:16*i+17].strip()
                    if obs_=='' or self.sigid[sys][i]==0:
                        continue
                    ifreq=self.freq_tbl[self.sigid[sys][i]]
                    if self.typeid[sys][i]==0: # code
                        obs.mag[k,ifreq]=int(line[16*i+18])
                        obs.P[k,ifreq]=float(obs_)
                    elif self.typeid[sys][i]==0: # carrier
                        obs.L[k,ifreq]=float(obs_)
                    obs.data[k,ifreq*self.nf+self.typeid[sys][i]]=float(obs_)

            break
        return obs