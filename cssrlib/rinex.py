# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 20:10:51 2020

@author: ruihi
"""

import datetime
import numpy as np
from gnss import uGNSS,rSIG,rCST,sat2prn,Eph,prn2sat,gpst2time,Obs,Nav

class rnxdec:
    MAXSAT=uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX
   
    def __init__(self):
        self.fobs=None
        self.freq_tbl={rSIG.L1C:0,rSIG.L1X:0,rSIG.L2W:1,rSIG.L2L:1,rSIG.L2X:1,rSIG.L5Q:2,rSIG.L5X:2,rSIG.L7Q:1,rSIG.L7X:1}
        self.gnss_tbl={'G':uGNSS.GPS,'E':uGNSS.GAL,'J':uGNSS.QZS}
        self.sig_tbl={'1C':rSIG.L1C,'1X':rSIG.L1X,'1W':rSIG.L1W,'2W':rSIG.L2W,'2L':rSIG.L2L,'2X':rSIG.L2X,
                 '5Q':rSIG.L5Q,'5X':rSIG.L5X,'7Q':rSIG.L7Q,'7X':rSIG.L7X}
        self.skip_sig_tbl = {uGNSS.GPS:[rSIG.L1X,rSIG.L1W,rSIG.L2L,rSIG.L2X],uGNSS.GAL:[],uGNSS.QZS:[rSIG.L1X]}
        self.nf=4

    def flt(self,u,c=-1):
        if c>=0:
            u=u[19*c+4:19*(c+1)+4]
        return float(u.replace("D", "E"))

    def decode_nav(self,navfile,nav):
        """decode RINEX Navigation message from file """
        
        nav.eph=[]
        with open(navfile,'rt') as fnav:
            for line in fnav:
                if line[60:73]=='END OF HEADER':
                    break    
                elif line[60:80]=='RINEX VERSION / TYPE':
                    self.ver=float(line[4:10])
                    if self.ver<3.02:
                        return -1        
                elif line[60:76]=='IONOSPHERIC CORR':
                    if line[0:4]=='GPSA' or line[0:4]=='QZSA':
                        for k in range(4):
                            nav.ion[0,k]=float(line[5+k*12:5+(k+1)*12])
                    if line[0:4]=='GPSB' or line[0:4]=='QZSB':
                        for k in range(4):
                            nav.ion[1,k]=float(line[5+k*12:5+(k+1)*12])
        
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
                
                eph.af0=self.flt(line,1)
                eph.af1=self.flt(line,2)
                eph.af2=self.flt(line,3)        
        
                line=fnav.readline()
                eph.iode=int(self.flt(line,0))
                eph.crs=self.flt(line,1)
                eph.deln=self.flt(line,2)
                eph.M0=self.flt(line,3)         
        
                line=fnav.readline()
                eph.cuc=self.flt(line,0)
                eph.e=self.flt(line,1)
                eph.cus=self.flt(line,2)
                sqrtA=self.flt(line,3)
                eph.A=sqrtA**2
                
                line=fnav.readline()
                eph.toes=int(self.flt(line,0))
                eph.cic=self.flt(line,1)
                eph.OMG0=self.flt(line,2)
                eph.cis=self.flt(line,3)    
                
                line=fnav.readline()
                eph.i0=self.flt(line,0)
                eph.crc=self.flt(line,1)
                eph.omg=self.flt(line,2)
                eph.OMGd=self.flt(line,3) 
                
                line=fnav.readline()
                eph.idot=self.flt(line,0)
                eph.code=int(self.flt(line,1)) # source for GAL
                eph.week=int(self.flt(line,2))
                #if len(line)>=80:
                #    L2_P_data=int(float(line[61:80]) )
                
                line=fnav.readline()
                eph.sva=int(self.flt(line,0))
                eph.svh=int(self.flt(line,1))
                eph.tgd=float(self.flt(line,2))
                if sys==uGNSS.GAL:
                    tgd_b=float(self.flt(line,3))
                    if (eph.code>>9)&1:
                        eph.tgd=tgd_b
                else:
                    eph.iodc=int(self.flt(line,3))
        
                line=fnav.readline()
                tot=int(self.flt(line,0))
                if len(line)>=42:            
                    eph.fit=int(self.flt(line,1))
                
                eph.toe=gpst2time(eph.week,eph.toes)
                eph.tot=gpst2time(eph.week,tot)
        
                nav.eph.append(eph)
    
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
                    if sig[0]=='C':
                        self.typeid[sys][k] = 0
                    elif sig[0]=='L':
                        self.typeid[sys][k] = 1
                    elif sig[0]=='S':
                        self.typeid[sys][k] = 2
                    else:
                        continue
                    if sig[1:3] in self.sig_tbl:
                        if self.sig_tbl[sig[1:3]] in self.skip_sig_tbl[sys]:
                            continue
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
            obs.S=np.zeros((nsat,self.nf))
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
                        #obs.mag[k,ifreq]=int(line[16*i+18])
                        obs.P[k,ifreq]=float(obs_)
                    elif self.typeid[sys][i]==1: # carrier
                        obs.L[k,ifreq]=float(obs_)
                    elif self.typeid[sys][i]==2: # C/No
                        obs.S[k,ifreq]=float(obs_)
                    obs.data[k,ifreq*self.nf+self.typeid[sys][i]]=float(obs_)

            break
        return obs