# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import cbitstruct as bs
import numpy as np
from enum import IntEnum
from gnss import gpst2time,rCST,ecef2pos,prn2sat,uGNSS

class sGNSS(IntEnum):
    GPS=0;GLO=1;GAL=2;BDS=3
    QZS=4;SBS=5

class sCSSR(IntEnum):
    MASK=1;ORBIT=2;CLOCK=3;CBIAS=4
    PBIAS=5;BIAS=6;URA=7;STEC=8
    GRID=9;SI=10;COMBINED=11;ATMOS=12

class sSigGPS(IntEnum):
    L1C=0;L1P=1;L1W=2;L1S=3;L1L=4
    L1X=5;L2S=6;L2L=7;L2X=8;L2P=9
    L2W=10;L5I=11;L5Q=12;L5X=13

class sSigGLO(IntEnum):
    L1C=0;L1P=1;L2C=2;L2P=3;L4A=4
    L4B=5;L4X=6;L6A=7;L6B=8;L6X=9
    L3I=10;L3Q=11;L3X=12

class sSigGAL(IntEnum):
    L1B=0;L1C=1;L1X=2;L5I=3;L5Q=4
    L5X=5;L7I=6;L7Q=7;L7X=8;L8I=9
    L8Q=10;L8X=11;L6B=12;L6C=13;L6X=14

class sSigBDS(IntEnum):
    L2I=0;L2Q=1;L2X=2;L6I=3;L6Q=4
    L6X=5;L7I=6;L7Q=7;L7X=8;L1D=9
    L1P=10;L1X=11;L5D=12;L5P=13;L5X=14

class sSigQZS(IntEnum):
    L1C=0;L1S=1;L1L=2;L1X=3;L2S=4
    L2L=5;L2X=6;L5I=7;L5Q=8;L5X=9
    L6D=10;L6P=11;L6E=12

class sSigSBS(IntEnum):
    L1CA=0;L5I=1;L5Q=2;L5X=3

class local_corr:
    def __init__(self):
        self.inet=-1
        self.inet_ref=-1
        self.ng=-1
        self.pbias=None
        self.cbias=None
        self.iode=None
        self.dorb=None
        self.dclk=None
        self.stec=None
        self.trph=None
        self.trpw=None
        self.ci=None
        self.ct=None
        self.quality_trp=None
        self.quality_stec=None

class cssr:
    CSSR_MSGTYPE=4073
    MAXNET=32
    stec_sz_t=[4,4,5,7]
    stec_scl_t=[0.04,0.12,0.16,0.24]
    
    def __init__(self):
        self.monlevel=0
        self.week=-1
        self.tow0=-1
        self.iodssr=-1
        self.msgtype=4073
        self.subtype=0
        self.svmask = [-1,-1,-1,-1]
        self.nsat_n=0
        self.sys_n=[]
        self.sat_n=[]
        self.nsig_n=[]
        self.iodssr=-1
        self.nsig_total=0
        self.sig_n=[]
        self.dorb=[]
        self.iode=[]
        self.dclk=[]
        self.ura=[]
        self.cbias=[]
        self.pbias=[]
        self.inet=-1
        self.facility_p=-1
        self.buff=bytearray(250*5)
        self.sinfo=bytearray(160)
        self.grid=None
        self.lc=[]
        for inet in range(self.MAXNET+1):
            self.lc.append(local_corr())
            self.lc[inet].inet=inet

    def sval(self,u,n,scl):
        invalid=-2**(n-1)
        y=np.nan if u==invalid else u*scl
        return y
    
    def isset(self,mask,nbit,k):
        if (mask>>(nbit-k-1))&1:
            return True
        else:
            return False

    def quality_idx(self,cl,val):
        if cl==7 and val==7:
            y=5.4665
        elif cl==0 and val==0: # undefined/unknown
            y=np.nan
        else:
            y=(3**cl*(1+val*0.25)-1)*1e-3 # [m]
        return y

    def gnss2sys(self,gnss):
        tbl={sGNSS.GPS:uGNSS.GPS,sGNSS.GLO:uGNSS.GLO,sGNSS.GAL:uGNSS.GAL,
             sGNSS.BDS:uGNSS.BDS,sGNSS.QZS:uGNSS.QZS,sGNSS.SBS:uGNSS.SBS}
        if gnss not in tbl:
            return -1
        sys=tbl[gnss]        
        return sys

    def decode_local_sat(self,netmask):
        sat=[]
        for k in range(self.nsat_n):
            if not self.isset(netmask,self.nsat_n,k):
                continue
            sat.append(self.sat_n[k])
        return sat

    def decode_mask(self,din,bitlen,ofst=1):
        v=[]
        n=0
        for k in range(0,bitlen):
            if (din & 1<<(bitlen-k-1)):
                v.append(k+ofst)
                n+=1
        return (v,n)

    def decode_head(self,msg,i,st=-1):
        if st==sCSSR.MASK:
            self.tow=bs.unpack_from('u20',msg,i)[0];i+=20
            self.tow0=self.tow
        else:
            dtow=bs.unpack_from('u12',msg,i)[0];i+=12
            if self.tow0>=0:
                self.tow=self.tow0+dtow
        if self.week>=0:
            self.time=gpst2time(self.week,self.tow)
        fmt='u4u1u4';names=['uint','mi','iodssr']
        dfm = bs.unpack_from_dict(fmt,names,msg,i);i+=9
        return (dfm,i)            

    def decode_cssr_mask(self,msg,i):
        """decode MT4073,1 Mask message """
        head,i=self.decode_head(msg,i,sCSSR.MASK)
        dfm = bs.unpack_from_dict('u4',['ngnss'],msg,i)
        self.flg_net=False
        i+=4
        self.iodssr=head['iodssr']
        self.nsat_n=0
        self.nsig_n=[]
        self.sys_n=[]
        self.sat_n=[]
        self.nsig_total=0
        self.sig_n=[]
        self.nsig_max=0

        for gnss in range(0,dfm['ngnss']):
            v=bs.unpack_from_dict('u4u40u16u1',['gnssid','svmask','sigmask','cma'],msg,i) 
            sys=self.gnss2sys(v['gnssid'])
            i+=61
            prn,nsat=self.decode_mask(v['svmask'],40)
            sig,nsig=self.decode_mask(v['sigmask'],16,0)
            self.nsat_n+=nsat
            if v['cma']==1:
                vc=bs.unpack_from(('u'+str(nsig))*nsat,msg,i) 
                i+=nsig*nsat

            self.nsig_max=max(self.nsig_max,nsig)

            for k in range(0,nsat):
                if sys==uGNSS.QZS:
                    prn[k]+=192
                sat=prn2sat(sys,prn[k])
                self.sys_n.append(sys)
                self.sat_n.append(sat)
                if v['cma']==1:
                    sig_s,nsig_s=self.decode_mask(vc[k],nsig,0)
                    sig_n=[sig[i] for i in sig_s]
                    self.nsig_n.append(nsig_s)
                    self.nsig_total=self.nsig_total+nsig_s
                    self.sig_n.append(sig_n)
                else:
                    self.nsig_n.append(nsig)
                    self.nsig_total=self.nsig_total+nsig
                    self.sig_n.append(sig)
        return i    

    def decode_orb_sat(self,msg,i,k,sys,inet=0):
        n=10 if sys==uGNSS.GAL else 8
        v=bs.unpack_from_dict('u'+str(n)+'s15s13s13',['iode','dx','dy','dz'],msg,i) 
        self.lc[inet].iode[k]=v['iode']
        self.lc[inet].dorb[k,0]=self.sval(v['dx'],15,0.0016)
        self.lc[inet].dorb[k,1]=self.sval(v['dy'],13,0.0064)
        self.lc[inet].dorb[k,2]=self.sval(v['dz'],13,0.0064)
        i+=n+41
        return i

    def decode_clk_sat(self,msg,i,k,inet=0):
        v=bs.unpack_from_dict('s15',['dclk'],msg,i)
        self.lc[inet].dclk[k]=self.sval(v['dclk'],15,0.0016)
        i+=15 
        return i

    def decode_cbias_sat(self,msg,i,k,j,inet=0):
        v=bs.unpack_from_dict('s11',['cbias'],msg,i)
        self.lc[inet].cbias[k,j]=self.sval(v['cbias'],11,0.02)
        i+=11 
        return i

    def decode_pbias_sat(self,msg,i,k,j,inet=0):
        v=bs.unpack_from_dict('s15u2',['pbias','di'],msg,i)
        self.lc[inet].pbias[k,j]=self.sval(v['pbias'],15,0.001)
        i+=17 
        return i

    def decode_cssr_orb(self,msg,i,inet=0):
        """decode MT4073,2 Orbit Correction message """
        head,i=self.decode_head(msg,i)
        self.flg_net=False
        if self.iodssr!=head['iodssr']:
            return -1
        self.lc[inet].dorb = np.zeros((self.nsat_n,3))
        self.lc[inet].iode = np.zeros(self.nsat_n,dtype=int)
        for k in range(0,self.nsat_n):
            i=self.decode_orb_sat(msg,i,k,self.sys_n[k],inet)
        return i
    
    def decode_cssr_clk(self,msg,i,inet=0):
        """decode MT4073,3 Clock Correction message """
        head,i=self.decode_head(msg,i)
        self.flg_net=False
        if self.iodssr!=head['iodssr']:
            return -1
        self.lc[inet].dclk = np.zeros(self.nsat_n)
        for k in range(0,self.nsat_n):
            i=self.decode_clk_sat(msg,i,k,inet)
        return i    

    def decode_cssr_cbias(self,msg,i,inet=0):
        """decode MT4073,4 Code Bias Correction message """
        head,i=self.decode_head(msg,i)
        self.flg_net=False
        if self.iodssr!=head['iodssr']:
            return -1      
        self.lc[inet].cbias = np.zeros((self.nsat_n,self.nsig_max))
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                i=self.decode_cbias_sat(msg,i,k,j,inet)       
        return i  

    def decode_cssr_pbias(self,msg,i,inet=0):
        """decode MT4073,5 Phase Bias Correction message """
        head,i=self.decode_head(msg,i)
        self.flg_net=False
        if self.iodssr!=head['iodssr']:
            return -1
        self.lc[inet].pbias = np.zeros((self.nsat_n,self.nsig_max))        
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                i=self.decode_pbias_sat(msg,i,k,j,inet)
        return i  

    def decode_cssr_bias(self,msg,i,inet=0):
        """decode MT4073,6 Bias Correction message """
        nsat=self.nsat_n
        head,i=self.decode_head(msg,i)        
        if self.iodssr!=head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('b1b1b1',['cb','pb','net'],msg,i)
        self.flg_net=dfm['net']
        i+=3 
        if dfm['net']:
            v=bs.unpack_from_dict('u5u'+str(nsat),['inet','svmaskn'],msg,i)
            self.inet=inet=v['inet']
            i+=5+nsat
            
        if dfm['cb']:
            self.lc[inet].cbias = np.zeros((nsat,self.nsig_max))
        if dfm['pb']:
            self.lc[inet].pbias = np.zeros((nsat,self.nsig_max))
        for k in range(nsat):
            if not self.isset(v['svmaskn'],nsat,k):
                continue
            for j in range(self.nsig_n[k]):
                if dfm['cb']:
                    i=self.decode_cbias_sat(msg,i,k,j,inet)
                if dfm['pb']:
                    i=self.decode_pbias_sat(msg,i,k,j,inet)
        return i 

    def decode_cssr_ura(self,msg,i):
        """decode MT4073,7 URA message """
        head,i=self.decode_head(msg,i)  
        if self.iodssr!=head['iodssr']:
            return -1
        self.ura = np.zeros(self.nsat_n)
        for k in range(0,self.nsat_n):
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            self.ura[k]=self.quality_idx(v['class'],v['val'])
            i+=6
        return i 

    def decode_cssr_stec_coeff(self,msg,stype,i):
        ci=np.zeros(6)
        v=bs.unpack_from('s14',msg,i)
        ci[0]=self.sval(v[0],14,0.05)
        i+=14
        if stype>0:
            v=bs.unpack_from('s12s12',msg,i)
            ci[1]=self.sval(v[0],12,0.02)
            ci[2]=self.sval(v[1],12,0.02)
            i+=24                    
        if stype>1:
            v=bs.unpack_from('s10',msg,i)
            ci[3]=self.sval(v[0],10,0.02)
            i+=10
        if stype>2:
            v=bs.unpack_from('s8s8',msg,i)
            ci[4]=self.sval(v[0],8,0.005)
            ci[5]=self.sval(v[1],8,0.005)
            i+=16
        return (ci,i)

    def decode_cssr_stec(self,msg,i):
        """decode MT4073,8 STEC Correction message """
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        self.flg_net=True
        dfm = bs.unpack_from_dict('u2u5u'+str(self.nsat_n),['stype','inet','svmaskn'],msg,i)
        self.inet=inet=dfm['inet']
        self.netmask[inet]=netmask=dfm['svmaskn']
        i+=7+self.nsat_n
        self.lc[inet].stec_quality=np.zeros(self.nsat_n)
        self.lc[inet].ci=np.zeros((self.nsat_n,6))
        for k in range(self.nsat_n):
            if not self.isset(netmask,self.nsat_n,k):
                continue
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            self.lc[inet].stec_quality[k]=self.quality_idx(v['class'],v['val'])
            i+=6
            ci,i=self.decode_cssr_stec_coeff(msg,dfm['stype'],i)
            self.lc[inet].ci[k,:] = ci
        return i 

    def decode_cssr_grid(self,msg,i):
        """decode MT4073,9 Grid Correction message """
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        dfm = bs.unpack_from_dict('u2u1u5u'+str(self.nsat_n)+'u3u3u6',
                ['ttype','range','inet','svmaskn','class','value','ng'],msg,i)
        self.flg_net=True
        self.inet=inet=dfm['inet']
        self.netmask[inet]=netmask=dfm['svmaskn']
        ng=dfm['ng']
        self.lc[inet].ng=ng
        self.lc[inet].trop_quality=self.quality_idx(dfm['class'],dfm['value'])
        i+=20+self.nsat_n
        sz=7 if dfm['range']==0 else 16
        fmt='s'+str(sz)
        self.lc[inet].stec=np.zeros((ng,self.nsat_n))
        self.lc[inet].dtd=np.zeros(ng)
        self.lc[inet].dtw=np.zeros(ng)

        for j in range(0,ng):
            if dfm['ttype']>0:
                vd = bs.unpack_from_dict('s9s8',['dtd','dtw'],msg,i)    
                i+=17
                self.lc[inet].dtd[j]=self.sval(vd['dtd'],9,0.004)+2.3
                self.lc[inet].dtw[j]=self.sval(vd['dtw'],8,0.004)            
            
            for k in range(self.nsat_n):
                if not self.isset(netmask,self.nsat_n,k):
                    continue
                dstec=bs.unpack_from(fmt,msg,i)[0];i+=sz
                self.lc[inet].stec[j,k]=self.sval(dstec,sz,0.04)
        return i

    def parse_sinfo(self):
        # TBD
        return 0

    def decode_cssr_sinfo(self,msg,i):
        """decode MT4073,10 Service Information message """
        dfm = bs.unpack_from_dict('b1u3u2',['mi','cnt','dsize'],msg,i)
        self.flg_net=False
        i+=6
        n=dfm['dsize']+1
        j=n*40*dfm['cnt']
        for k in range(n):
            d=bs.unpack_from('u40',msg,i)[0];i+=40
            bs.pack_info('u40',self.sinfo,j,d);j+=40
        if dfm['mi']==False:
             self.parse_sinfo()
        return i

    def decode_cssr_comb(self,msg,i,inet=0):
        """decode MT4073,11 Orbit,Clock Combined Correction message """
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        dfm = bs.unpack_from_dict('b1b1b1',['orb','clk','net'],msg,i)
        i+=3
        self.flg_net=dfm['net']
        if self.flg_net:
            v=bs.unpack_from_dict('u5u'+str(self.nsat_n),['inet','svmask'],msg,i)
            self.inet=inet=v['inet']
            self.lc[inet].svmask=svmask=v['svmask']
            i+=5+self.nsat_n
       
        if dfm['orb']:
            self.lc[inet].dorb = np.zeros((self.nsat_n,3))
            self.lc[inet].iode = np.zeros(self.nsat_n,dtype=int)    
        if dfm['clk']:
            self.lc[inet].dclk = np.zeros(self.nsat_n)    
        
        for k in range(self.nsat_n):
            if self.flg_net and not self.isset(svmask,self.nsat_n,k):
                continue
            if dfm['orb']:
                i=self.decode_orb_sat(msg,i,k,self.sys_n[k],inet)
            if dfm['clk']:
                i=self.decode_clk_sat(msg,i,k,inet)
        return i

    def decode_cssr_atmos(self,msg,i):
        """decode MT4073,12 Atmospheric Correction message """
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('u2u2u5u6',['trop','stec','inet','ng'],msg,i)
        self.flg_net=True
        self.inet=inet=dfm['inet']
        self.lc[inet].ng=ng=dfm['ng']
        self.lc[inet].flg_trop=dfm['trop']
        self.lc[inet].flg_stec=dfm['stec']
        i+=15
        # trop
        if dfm['trop']>0:
            v=bs.unpack_from_dict('u3u3',['class','value'],msg,i)
            self.lc[inet].trop_quality=self.quality_idx(v['class'],v['value'])
            i+=6
        if dfm['trop']&2: # function
            self.lc[inet].ttype=ttype=bs.unpack_from('u2',msg,i)[0];i+=2
            names=['t00','t01','t10','t11']
            vt=bs.unpack_from_dict('s9',[names[0]],msg,i)
            i+=9
            self.lc[inet].ct=np.zeros(4)
            self.lc[inet].ct[0]=self.sval(vt['t00'],9,0.004)
            if ttype>0:
                vt=bs.unpack_from_dict('s7s7',[names[1:3]],msg,i)
                i+=14 
                self.lc[inet].ct[1]=self.sval(vt['t01'],7,0.002)
                self.lc[inet].ct[2]=self.sval(vt['t10'],7,0.002) 
            if ttype>1:
                vt=bs.unpack_from_dict('s7',[names[3]],msg,i)
                i+=7
                self.lc[inet].ct[3]=self.sval(vt['t11'],7,0.001) 

        if dfm['trop']&1: # residual                          
            vh=bs.unpack_from_dict('u1u4',['sz','ofst'],msg,i)
            i+=5
            trop_ofst=vh['ofst']*0.02
            sz=6 if vh['sz']==0 else 8
            vtr=bs.unpack_from(('s'+str(sz))*ng,msg,i);i+=sz*ng
            self.lc[inet].dtw=np.zeros(ng)
            for k in range(ng):
                self.lc[inet].dtw[k]=self.sval(vtr[k],sz,0.004)+trop_ofst

        # STEC
        netmask=bs.unpack_from('u'+str(self.nsat_n),msg,i)[0];i+=self.nsat_n
        self.lc[inet].netmask=netmask
        #loc,nsat_l=self.decode_mask(netmask,self.nsat_n)
        self.lc[inet].stec_quality=np.zeros(self.nsat_n,dtype=int)
        if dfm['stec']&2>0:
            self.lc[inet].ci=np.zeros((self.nsat_n,6))
            self.lc[inet].stype=np.zeros(self.nsat_n,dtype=int)
        if dfm['stec']&1>0:
            self.lc[inet].dstec=np.zeros((self.nsat_n,ng))
        for k in range(0,self.nsat_n):
            if not self.isset(netmask,self.nsat_n,k):
                continue
            if dfm['stec']>0:
                v=bs.unpack_from_dict('u3u3',['class','value'],msg,i)
                i+=6
                self.lc[inet].stec_quality[k]=self.quality_idx(v['class'],v['value'])
            if dfm['stec']&2>0: # functional term            
                self.lc[inet].stype[k]=bs.unpack_from('u2',msg,i)[0];i+=2
                ci,i=self.decode_cssr_stec_coeff(msg,self.lc[inet].stype[k],i)
                self.lc[inet].ci[k,:]=ci
            
            if dfm['stec']&1>0: # residual term
                sz_idx=bs.unpack_from('u2',msg,i)[0];i+=2
                sz=self.stec_sz_t[sz_idx]
                scl=self.stec_scl_t[sz_idx]
                v=bs.unpack_from(('s'+str(sz))*ng,msg,i);i+=sz*ng
                for j in range(ng):
                    self.lc[inet].dstec[k,j]=self.sval(v[j],sz,scl)
        return i

    def decode_cssr(self,msg,i):
        """decode Compact SSR message """
        df={'msgtype':4073}
        while df['msgtype']==4073:            
            df = bs.unpack_from_dict('u12u4',['msgtype','subtype'],msg,i) 
            i+=16
            if df['msgtype']!=4073:
                return -1
            self.subtype=df['subtype']
            if self.subtype==sCSSR.MASK:
                i=self.decode_cssr_mask(msg,i)
            elif self.subtype==sCSSR.ORBIT: # orbit
                i=self.decode_cssr_orb(msg,i)
            elif self.subtype==sCSSR.CLOCK: # clock
                i=self.decode_cssr_clk(msg,i)
            elif self.subtype==sCSSR.CBIAS: # cbias
                i=self.decode_cssr_cbias(msg,i)
            elif self.subtype==sCSSR.PBIAS: # pbias
                i=self.decode_cssr_pbias(msg,i)
            elif self.subtype==sCSSR.BIAS: # bias
                i=self.decode_cssr_bias(msg,i)            
            elif self.subtype==sCSSR.URA: # ura
                i=self.decode_cssr_ura(msg,i)
            elif self.subtype==sCSSR.STEC: # stec
                i=self.decode_cssr_stec(msg,i)
            elif self.subtype==sCSSR.GRID: # grid
                i=self.decode_cssr_grid(msg,i)                
            elif self.subtype==sCSSR.SI: # service-info
                i=self.decode_cssr_sinfo(msg,i)                
            elif self.subtype==sCSSR.COMBINED: # orb+clk
                i=self.decode_cssr_comb(msg,i)
            elif self.subtype==sCSSR.ATMOS: # atmos
                i=self.decode_cssr_atmos(msg,i)
            if i<=0:
                return 0
            if self.monlevel>=2:
                if self.flg_net:
                    print("tow={:6d} subtype={:2d} inet={:2d}".
                          format(self.tow,self.subtype,self.inet))
                else:
                    print("tow={:6d} subtype={:2d}".format(self.tow,self.subtype))                    

    def decode_l6msg(self,msg,ofst):
        """decode QZS L6 message """
        fmt = 'u32u8u3u2u2u1u1'
        names = ['preamble','prn','vendor','facility','res','sid','alert']
        i=ofst*8
        l6head = bs.unpack_from_dict(fmt,names,msg,i)
        i=i+49
        if l6head['sid']==1:
            self.fcnt=0
        if self.facility_p>=0 and l6head['facility']!=self.facility_p:
            self.fcnt=-1
        self.facility_p=l6head['facility']
        if self.fcnt<0:
            return -1
        j=1695*self.fcnt
        for k in range(53):
            sz=32 if k<52 else 31
            fmt='u'+str(sz)
            b=bs.unpack_from(fmt,msg,i)
            i+=sz
            bs.pack_into(fmt,self.buff,j,b[0])
            j+=sz
        self.fcnt=self.fcnt+1
        
    def read_griddef(self,file):
        """load grid coordinates from file """
        dtype0=[('nid','<i4'),('gid','<i4'),
                ('lat','<f8'),('lon','<f8'),('alt','<f8')]        
        self.grid=np.genfromtxt(file,dtype=dtype0,skip_header=1,skip_footer=2,encoding='utf8')
        
    def find_grid_index(self,pos):
        self.rngmin=5e3
        lat=np.deg2rad(self.grid['lat'])
        lon=np.deg2rad(self.grid['lon'])
        clat=np.cos(pos[0])
        r=np.linalg.norm((lat-pos[0],(lon-pos[1])*clat),axis=0)*rCST.RE_WGS84
        idx=np.argmin(r)
        self.inet_ref=self.grid[idx]['nid']
        if r[idx]<self.rngmin:
            self.ngrid=1
            self.grid_index=self.grid['gid'][idx] 
            self.grid_weight=[1]
        else:    
            idn=self.grid['nid']==self.inet_ref
            rn=r[idn]
            self.ngrid=n=min(len(rn),4)
            idx=np.argsort(rn)[0:n]
            rp=1./rn[idx]
            w=rp/np.sum(rp)
            self.grid_index=self.grid[idn]['gid'][idx]
            self.grid_weight=w
        return self.inet_ref
        
    def get_dpos(self,pos):
        inet=self.inet_ref
        posd=np.rad2deg(pos[0:2])
        grid=self.grid[self.grid['nid']==inet]
        dlat=posd[0]-grid[0]['lat']
        dlon=posd[1]-grid[0]['lon']
        return dlat,dlon
    
    def get_trop(self,dlat=0.0,dlon=0.0):
        inet=self.inet_ref
        trph=0; trpw=0
        if self.lc[inet].flg_trop&2:
            trph=2.3+self.lc[inet].ct@[1,dlat,dlon,dlat*dlon]
        if self.lc[inet].flg_trop&1:
            trpw=self.lc[inet].dtw[self.grid_index]@self.grid_weight
        return trph,trpw
 
    def get_stec(self,dlat=0.0,dlon=0.0):
        inet=self.inet_ref
        stec=np.zeros(self.nsat_n)
        for i in range(self.nsat_n):
            if not self.isset(self.lc[inet].netmask,self.nsat_n,i):
                continue
            if self.lc[inet].flg_stec&2:            
                ci=self.lc[inet].ci[i,:]
                stec[i]=[1,dlat,dlon,dlat*dlon,dlat**2,dlon**2]@ci
            if self.lc[inet].flg_stec&1:   
                dstec=self.lc[inet].dstec[i,self.grid_index]@self.grid_weight              
                stec[i]+=dstec
        return stec
            
        

        
       
        
if __name__ == '__main__':
    bdir='../data/'
    l6file=bdir+'2021078M.l6'
    griddef=bdir+'clas_grid.def'
    xyz=[-3962108.4557,  3381308.8777,  3668678.1749]
    pos=ecef2pos(xyz)
    
    cs=cssr()
    cs.monlevel=2
    cs.week=2149
    
    cs.read_griddef(griddef)
    inet=cs.find_grid_index(pos)
    
    sf=0
    #sfmax=3600//5
    sfmax=30//5
    with open(bdir+l6file,'rb') as f:
        while sf<sfmax:
            msg=f.read(250)
            if not msg:
                break
            cs.decode_l6msg(msg,0)
            if cs.fcnt==5:
                sf+=1
                cs.decode_cssr(cs.buff,0)
            if sf>=6:
                dlat,dlon=cs.get_dpos(pos)
                trph,trpw=cs.get_trop(dlat,dlon)
                stec=cs.get_stec(dlat,dlon)
                
            
                

    
                        
        