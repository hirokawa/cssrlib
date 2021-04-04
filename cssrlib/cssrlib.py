# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import cbitstruct as bs
import numpy as np
from enum import IntEnum

class sGNSS(IntEnum):
    GPS=0;GLO=1;GAL=2;BDS=3
    QZS=4;SBS=5

class sCSSR(IntEnum):
    MASK=1;ORBIT=2;CLOCK=3;CBIAS=4
    PBIAS=5;BIAS=6;URA=7;STEC=8
    GRID=9;SI=10;COMBINED=11;ATMOS=12

class sSigGPS(IntEnum):
    L1CA=0;L1P=1;L1W=2;L1CD=3;L1CP=4
    L1X=5;L2CM=6;L2CL=7;L2X=8;L2P=9
    L2W=10;L5I=11;L5Q=12;L5X=13

class sSigGLO(IntEnum):
    G1CA=0;G1P=1;G2CA=2;G2P=3;G1AD=4
    G1AP=5;G1AX=6;G2AI=7;G2AQ=8;G2AX=9
    G3I=10;G3Q=11;G3X=12

class sSigGAL(IntEnum):
    E1B=0;E1C=1;E1X=2;E5AI=3;E5AQ=4
    E5AX=5;E5BI=6;E5BQ=7;E5BX=8;E5I=9
    E5Q=10;E5X=11

class sSigBDS(IntEnum):
    B1I=0;B1Q=1;B1X=2;B3I=3;B3Q=4
    B3X=5;B2I=6;B2Q=7;B2X=8;

class sSigQZS(IntEnum):
    L1CA=0;L1CD=1;L1CP=2
    L1X=3;L2CM=4;L2CL=5;L2X=6
    L5I=7;L5Q=8;L5X=9

class sSigSBS(IntEnum):
    L1CA=0;L5I=1;L5Q=2;L5X=3

class cssr:
    CSSR_MSGTYPE=4073
    
    def __init__(self):
        self.msgtype=4073
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

    def sval(self,u,n,scl):
        invalid=-2**(n-1)
        y=np.nan if u==invalid else u*scl
        return y

    def quality_idx(self,cl,val):
        if cl==7 and val==7:
            y=5.4665
        elif cl==0 and val==0: # undefined/unknown
            y=np.nan
        else:
            y=(3**cl*(1+val*0.25)-1)*1e-3 # [m]
        return y

    def decode_mask(self,din,bitlen):
        v=[]
        n=0
        for k in range(0,bitlen):
            if (din & 1<<(bitlen-k-1)):
                v.append(k+1)
                n=n+1
        return (v,n)

    def decode_head(self,msg,i,st=-1):
        if st==sCSSR.MASK:
            fmt='u20u4u1u4'
            names=['tow','uint','mi','iodssr']
            blen=29
        else:
            fmt='u12u4u1u4'
            names=['tow_s','uint','mi','iodssr']
            blen=21

        dfm = bs.unpack_from_dict(fmt,names,msg,i)
        return (dfm,i+blen)            

    def decode_cssr_mask(self,msg,i):
        head,i=self.decode_head(msg,i,sCSSR.MASK)
        dfm = bs.unpack_from_dict('u4',['ngnss'],msg,i)
        i=i+4
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
            i=i+61
            sats,nsat=self.decode_mask(v['svmask'],40)
            sig,nsig=self.decode_mask(v['sigmask'],16)
            self.nsat_n=self.nsat_n+nsat
            #print("mask %d %d %d %d" % (gnss,nsat,nsig,v['cma']))
            if v['cma']==1:
                vc=bs.unpack_from(('u'+str(nsig))*nsat,msg,i) 
                i=i+nsig*nsat

            self.nsig_max=max(self.nsig_max,nsig)

            for k in range(0,nsat):
                self.sys_n.append(v['gnssid'])
                self.sat_n.append(sats[k])
                if v['cma']==1:
                    sig_s,nsig_s=self.decode_mask(vc[k],nsig)
                    self.nsig_n.append(nsig_s)
                    self.nsig_total=self.nsig_total+nsig_s
                    self.sig_n.append(sig_s)
                else:
                    self.nsig_n.append(nsig)
                    self.nsig_total=self.nsig_total+nsig
                    self.sig_n.append(sig)
        return i    

    def decode_orb_sat(self,msg,i,k,sys):
        n=10 if sys==sGNSS.GAL else 8
        v=bs.unpack_from_dict('u'+str(n)+'s15s13s13',['iode','dx','dy','dz'],msg,i) 
        self.iode[k]=v['iode']
        self.dorb[k,0]=self.sval(v['dx'],15,0.0016)
        self.dorb[k,1]=self.sval(v['dy'],13,0.0064)
        self.dorb[k,2]=self.sval(v['dz'],13,0.0064)
        i=i+n+15+13+13
        return i

    def decode_clk_sat(self,msg,i,k):
        v=bs.unpack_from_dict('s15',['dclk'],msg,i)
        self.dclk[k]=self.sval(v['dclk'],15,0.0016)
        i=i+15 
        return i

    def decode_cbias_sat(self,msg,i,k,j):
        v=bs.unpack_from_dict('s11',['cbias'],msg,i)
        self.cbias[k,j]=self.sval(v['cbias'],11,0.02)
        i=i+11 
        return i

    def decode_pbias_sat(self,msg,i,k,j):
        v=bs.unpack_from_dict('s15u2',['pbias','di'],msg,i)
        self.pbias[k,j]=self.sval(v['pbias'],15,0.001)
        i=i+17 
        return i

    def decode_cssr_orb(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1
        self.dorb = np.zeros((self.nsat_n,3))
        self.iode = np.zeros(self.nsat_n,dtype=int)
        for k in range(0,self.nsat_n):
            i=self.decode_orb_sat(msg,i,k,self.sys_n[k])
            #print("orb %d %d %.2f %.2f %.2f" % (self.sys_n[k],v['iode'],v['dx'],v['dy'],v['dz']))
        return i
    
    def decode_cssr_clk(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1
        self.dclk = np.zeros(self.nsat_n)
        for k in range(0,self.nsat_n):
            i=self.decode_clk_sat(msg,i,k)
        return i    

    def decode_cssr_cbias(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1      
        self.cbias = np.zeros((self.nsat_n,self.nsig_max))
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                i=self.decode_cbias_sat(msg,i,k,j)       
        return i  

    def decode_cssr_pbias(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1
        self.pbias = np.zeros((self.nsat_n,self.nsig_max))        
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                i=self.decode_pbias_sat(msg,i,k,j)
        return i  

    def decode_cssr_bias(self,msg,i):
        head,i=self.decode_head(msg,i)        
        if self.iodssr!=head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('b1b1b1',['cb','pb','net'],msg,i)
        i=i+3
        if dfm['net']==True:
            v=bs.unpack_from_dict('u5u'+str(self.nsat_n),['inet','svmaskn'],msg,i)
            i=i+5+self.nsat_n
            loc,nsat_l=self.decode_mask(v['svmaskn'],self.nsat_n)
            nsig_l = []
            for k in range(0,nsat_l):
                nsig_l.append(self.nsig_n[loc[k]-1])
        else:
            nsat_l = self.nsat_n
            nsig_l = self.nsig_n
            
        if dfm['cb']==True:
            self.cbias = np.zeros((nsat_l,self.nsig_max))
        if dfm['pb']==True:
            self.pbias = np.zeros((nsat_l,self.nsig_max))
        for k in range(0,nsat_l):
            for j in range(0,nsig_l[k]):
                if dfm['cb']==True:
                    i=self.decode_cbias_sat(msg,i,k,j)
                if dfm['pb']==True:
                    i=self.decode_pbias_sat(msg,i,k,j)
        return i 

    def decode_cssr_ura(self,msg,i):
        head,i=self.decode_head(msg,i)  
        if self.iodssr!=head['iodssr']:
            return -1
        self.ura = np.zeros(self.nsat_n)
        for k in range(0,self.nsat_n):
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            self.ura[k]=self.quality_idx(v['class'],v['val'])
            i=i+6
        return i 

    def decode_cssr_stec_coeff(self,msg,stype,i):
        ci=np.zeros(6)
        v=bs.unpack_from('s14',msg,i)
        ci[0]=self.sval(v[0],14,0.05)
        i=i+14
        if stype>0:
            v=bs.unpack_from('s12s12',msg,i)
            ci[1]=self.sval(v[0],12,0.02)
            ci[2]=self.sval(v[1],12,0.02)
            i=i+24                    
        if stype>1:
            v=bs.unpack_from('s10',msg,i)
            ci[3]=self.sval(v[0],10,0.02)
            i=i+10
        if stype>2:
            v=bs.unpack_from('s8s8',msg,i)
            ci[4]=self.sval(v[0],8,0.005)
            ci[5]=self.sval(v[1],8,0.005)
            i=i+16
        return (ci,i)

    def decode_cssr_stec(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        dfm = bs.unpack_from_dict('u2u5u'+str(self.nsat_n),['stype','inet','svmaskn'],msg,i)
        i=i+7+self.nsat_n
        loc,nsat_l=self.decode_mask(dfm['svmaskn'],self.nsat_n)
        self.stec_quality=np.zeros(nsat_l)
        self.ci=np.zeros((nsat_l,6))
        for k in range(0,nsat_l):
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            self.stec_quality[k]=self.quality_idx(v['class'],v['val'])
            i=i+6
            ci,i=self.decode_cssr_stec_coeff(msg,dfm['stype'],i)
            self.ci[k,:] = ci
        return i 

    def decode_cssr_grid(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        dfm = bs.unpack_from_dict('u2u1u5u'+str(self.nsat_n)+'u3u3u6',
                ['ttype','range','inet','svmaskn','class','value','ng'],msg,i)
        ng=dfm['ng']
        self.trop_quality=self.quality_idx(dfm['class'],dfm['value'])
        i=i+20+self.nsat_n
        loc,nsat_l=self.decode_mask(dfm['svmaskn'],self.nsat_n)
        sz=7 if dfm['range']==0 else 16
        fmt='s'+str(sz)
        if dfm['inet']>12 and ng>1:
            print("inet=%d ng=%d nsat_l=%d i=%d" % (dfm['inet'],ng,nsat_l,i))            
            return 0
        self.stec=np.zeros((ng,nsat_l))
        self.dtd=np.zeros(ng)
        self.dtw=np.zeros(ng)

        for j in range(0,ng):
            if dfm['ttype']>0:
                vd = bs.unpack_from_dict('s9s8',['dtd','dtw'],msg,i)    
                i=i+17
                self.dtd[j]=self.sval(vd['dtd'],9,0.004)+2.3
                self.dtw[j]=self.sval(vd['dtw'],8,0.004)            
            v=bs.unpack_from(fmt*nsat_l,msg,i)
            for k in range(nsat_l):
                self.stec[j,k]=self.sval(v[k],sz,0.04)
            i=i+sz*nsat_l
        return i

    def decode_cssr_sinfo(self,msg,i):
        dfm = bs.unpack_from_dict('u1u3u2',['mi','cnt','dsize'],msg,i)
        i=i+6
        n=dfm['dsize']+1
        #v=bs.unpack_from('u40'*n,msg,i)
        i=i+40*n
        return i

    def decode_cssr_comb(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1        
        dfm = bs.unpack_from_dict('b1b1b1',['orb','clk','net'],msg,i)
        i=i+3

        if dfm['net']==True:
            v=bs.unpack_from_dict('u5u'+str(self.nsat_n),['inet','svmaskn'],msg,i)
            i=i+5+self.nsat_n
            loc,nsat_l=self.decode_mask(v['svmaskn'],self.nsat_n)
            sys_l=[]
            for idx in loc:
                sys_l.append(self.sys_n[idx-1])
        else:
            nsat_l=self.nsat_n
            sys_l=self.sys_n
        
        for k in range(0,nsat_l):
            if dfm['orb']==True:
                i=self.decode_orb_sat(msg,i,k,sys_l[k])
            if dfm['clk']==True:
                i=self.decode_clk_sat(msg,i,k)
               
#            print("comb %d %.2f " % (self.sys_n[k],vc['clk']))
#            if dfm['orb']==True:
#                print("%3d %.2f %.2f %.2f" % (v['iode'],v['dx'],v['dy'],v['dz']))    
        return i

    def decode_cssr_atmos(self,msg,i):
        head,i=self.decode_head(msg,i)
        if self.iodssr!=head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('u2u2u5u6',['trop','stec','inet','ng'],msg,i)
        ng=dfm['ng']
        i=i+15
        # trop
        if dfm['trop']>0:
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            self.trop_quality=self.quality_idx(v['class'],v['value'])
            i=i+6
        if dfm['trop']&2: # function
            vh=bs.unpack_from_dict('u2',['ttype'],msg,i)
            i=i+2
            names=['t00','t01','t10','t11']
            vt=bs.unpack_from_dict('s9',names[0],msg,i)
            i=i+9
            self.ct=np.zeros(4)
            self.ct[0]=self.sval(vt['t00'],9,0.004)
            if vh['ttype']>0:
                vt=bs.unpack_from_dict('s7s7',names[1:3],msg,i)
                i=i+14 
                self.ct[1]=self.sval(vt['t01'],7,0.002)
                self.ct[2]=self.sval(vt['t10'],7,0.002) 
            if vh['ttype']>1:
                vt=bs.unpack_from_dict('s7',names[3],msg,i)
                i=i+7
                self.ct[3]=self.sval(vt['t11'],7,0.001) 

        if dfm['trop']&1: # residual                          
            vh=bs.unpack_from_dict('u1u4',['sz','ofst'],msg,i)
            i=i+5
            trop_ofst=vh[0]*0.02
            sz=6 if vh['sz']==0 else 8
            vtr=bs.unpack_from(('s'+str(sz))*ng,msg,i)
            self.dtw=np.zeros(ng)
            for k in range(ng):
                self.dtw[k]=self.sval(vtr[k],sz,0.004)+trop_ofst

        # STEC
        v=bs.unpack_from_dict('u'+str(self.nsat_n),['svmaskn'],msg,i)
        i=i+self.nsat_n
        loc,nsat_l=self.decode_mask(v['svmaskn'],self.nsat_n)
        self.stec_quality=np.zeros(nsat_l)
        if dfm['stec']&2>0:
            self.ci=np.zeros((nsat_l,6))
        if dfm['stec']&1>0:
            self.dstec=np.zeros((nsat_l,ng))
        for k in range(0,nsat_l):
            if dfm['stec']>0:
                v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
                i=i+6
                self.stec_quality[k]=self.quality_idx(v['class'],v['value'])
            if dfm['stec']&2>0: # function            
                vh=bs.unpack_from_dict('u2',['stype'],msg,i)
                i=i+2
                ci,i=self.decode_cssr_stec_coeff(msg,vh['stype'],i)
                self.ci[k,:]=ci
            
            if dfm['stec']&1>0: # residual
                vs=bs.unpack_from_dict('u2',['sz'],msg,i)
                i=i+2
                sz_t=[4,4,5,7]
                scl_t=[0.04,0.12,0.16,0.24]
                sz=sz_t[vs['sz']]
                scl=scl_t[vs['sz']]
                v=bs.unpack_from(('s'+str(sz))*ng,msg,i)
                i=i+sz*ng
                for j in range(ng):
                    self.dstec[k,j]=self.sval(v[j],sz,scl)
        return i

    def decode_cssr(self,msg,i):
        df={'msgtype':4073}
        while df['msgtype']==4073:            
            df = bs.unpack_from_dict('u12u4',['msgtype','subtype'],msg,i) 
            i=i+16
            if df['msgtype']!=4073:
                return -1
            if df['subtype']==sCSSR.MASK:
                i=self.decode_cssr_mask(msg,i)
            elif df['subtype']==sCSSR.ORBIT: # orbit
                i=self.decode_cssr_orb(msg,i)
            elif df['subtype']==sCSSR.CLOCK: # clock
                i=self.decode_cssr_clk(msg,i)
            elif df['subtype']==sCSSR.CBIAS: # cbias
                i=self.decode_cssr_cbias(msg,i)
            elif df['subtype']==sCSSR.PBIAS: # pbias
                i=self.decode_cssr_pbias(msg,i)
            elif df['subtype']==sCSSR.BIAS: # bias
                i=self.decode_cssr_bias(msg,i)            
            elif df['subtype']==sCSSR.URA: # ura
                i=self.decode_cssr_ura(msg,i)
            elif df['subtype']==sCSSR.STEC: # stec
                i=self.decode_cssr_stec(msg,i)
            elif df['subtype']==sCSSR.GRID: # grid
                i=self.decode_cssr_grid(msg,i)                
            elif df['subtype']==sCSSR.SI: # service-info
                i=self.decode_cssr_sinfo(msg,i)                
            elif df['subtype']==sCSSR.COMBINED: # orb+clk
                i=self.decode_cssr_comb(msg,i)
            elif df['subtype']==sCSSR.ATMOS: # atmos
                i=self.decode_cssr_atmos(msg,i)
            if i<=0:
                return 0


                        
        