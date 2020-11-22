# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 20:03:45 2020

@author: ruihi
"""

import bitstruct as bs
import numpy as np

class cssr:
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
        
    def decode_mask(self,din,bitlen):
        v=[]
        n=0
        for k in range(0,bitlen):
            if (din & 1<<(bitlen-k-1)):
                v.append(k+1)
                n=n+1
        return (v,n)

    def decode_cssr_mask(self,msg,i):
        dfm = bs.unpack_from_dict('u20u4u1u4u4',['tow','uint','mi','iodssr','ngnss'],msg,i)
        i=i+20+4+1+4+4
        self.iodssr=dfm['iodssr']
        self.nsat_n=0
        self.nsig_n=[]
        self.sys_n=[]
        self.sat_n=[]
        self.nsig_total=0
        self.sig_n=[]

        for gnss in range(0,dfm['ngnss']):
            v=bs.unpack_from_dict('u4u40u16u1',['gnssid','svmask','sigmask','cma'],msg,i) 
            i=i+61
            sats,nsat=self.decode_mask(v['svmask'],40)
            sig,nsig=self.decode_mask(v['sigmask'],16)
            self.nsat_n=self.nsat_n+nsat
            print("mask %d %d %d %d" % (gnss,nsat,nsig,v['cma']))
            if v['cma']==1:
                vc=bs.unpack_from(('u'+str(nsig))*nsat,msg,i) 
                i=i+nsig*nsat

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

    def decode_cssr_orb(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4',['tow_s','uint','mi','iodssr'],msg,i)
        i=i+21
        if self.iodssr!=dfm['iodssr']:
            return -1
        self.dorb = np.zeros((self.nsat_n,3))
        self.iode = np.zeros(self.nsat_n,dtype=int)
        for k in range(0,self.nsat_n):
            n=10 if self.sys_n[k]==2 else 8
            v=bs.unpack_from_dict('u'+str(n)+'s15s13s13',['iode','dx','dy','dz'],msg,i) 
            self.iode[k]=v['iode']
            self.dorb[k,:]=[v['dx'],v['dy'],v['dz']]
            i=i+n+15+13+13
            print("orb %d %d %.2f %.2f %.2f" % (self.sys_n[k],v['iode'],v['dx'],v['dy'],v['dz']))
        return i
    
    def decode_cssr_clk(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4',['tow_s','uint','mi','iodssr'],msg,i)
        i=i+21
        if self.iodssr!=dfm['iodssr']:
            return -1
        self.dclk = np.zeros(self.nsat_n)
        for k in range(0,self.nsat_n):
            v=bs.unpack_from_dict('s15',['dclk'],msg,i)
            self.dclk[k]=v['dclk']
            i=i+15        
        return i    

    def decode_cssr_cbias(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4',['tow_s','uint','mi','iodssr'],msg,i)
        i=i+21
        if self.iodssr!=dfm['iodssr']:
            return -1
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                v=bs.unpack_from_dict('s11',['cbias'],msg,i)
                i=i+11         
        return i  

    def decode_cssr_pbias(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4',['tow_s','uint','mi','iodssr'],msg,i)
        i=i+21
        if self.iodssr!=dfm['iodssr']:
            return -1
        for k in range(0,self.nsat_n):
            for j in range(0,self.nsig_n[k]):
                v=bs.unpack_from_dict('s15u2',['pbias','di'],msg,i)
                i=i+17        
        return i  

    def decode_cssr_bias(self,msg,i):        
        dfm = bs.unpack_from_dict('u12u4u1u4u1u1u1',['tow_s','uint','mi','iodssr','cb','pb','net'],msg,i)
        i=i+24
        if self.iodssr!=dfm['iodssr']:
            return -1
        if dfm['net']==1:
            v=bs.unpack_from_dict('u5u'+str(self.nsat_n),['inet','svmaskn'],msg,i)
            i=i+5+self.nsat_n
            loc,nsat_l=self.decode_mask(v['svmaskn'],self.nsat_n)
            nsig_l = []
            for k in range(0,nsat_l):
                nsig_l.append(self.nsig_n[loc[k]-1])
        else:
            nsat_l = self.nsat_n
            nsig_l = self.nsig_n
            
        for k in range(0,nsat_l):
            for j in range(0,nsig_l[k]):
                v=bs.unpack_from_dict('s11s15u2',['cbias','pbias','di'],msg,i)
                i=i+28

        return i 

    def decode_cssr_ura(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4',['tow_s','uint','mi','iodssr'],msg,i)
        i=i+21
        if self.iodssr!=dfm['iodssr']:
            return -1
        for k in range(0,self.nsat_n):
            v=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            i=i+6
        return i 

    def decode_cssr_stec_coeff(self,msg,stype,i):
        if stype==0:
            v=bs.unpack_from_dict('s14',['c00'],msg,i)
            i=i+14
        elif stype==1:
            v=bs.unpack_from_dict('s14s12s12',['c00','c01','c10'],msg,i)
            i=i+38                    
        elif stype==2:
            v=bs.unpack_from_dict('s14s12s12s10',['c00','c01','c10','c11'],msg,i)
            i=i+48
        else:
            v=bs.unpack_from_dict('s14s12s12s10s8s8',['c00','c01','c10','c11','c02','c20'],msg,i)
            i=i+64
        return i

    def decode_cssr_stec(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4u2u5u'+str(self.nsat_n),['tow_s','uint','mi','iodssr','stype','inet','svmaskn'],msg,i)
        i=i+28+self.nsat_n
        if self.iodssr!=dfm['iodssr']:
            return -1
        loc,nsat_l=self.decode_mask(dfm['svmaskn'],self.nsat_n)
        for k in range(0,nsat_l):
            stec_quality=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            i=i+6
            i=self.decode_cssr_stec_coeff(msg,dfm['stype'],i)
        return i 

    def decode_cssr_grid(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4u2u1u5u'+str(self.nsat_n)+'u3u3u6',['tow_s','uint','mi','iodssr','type','range','inet','svmaskn','class','value','ng'],msg,i)
        i=i+41+self.nsat_n
        if self.iodssr!=dfm['iodssr']:
            return -1
        loc,nsat_l=self.decode_mask(dfm['svmaskn'],self.nsat_n)
        blen=7 if dfm['range']==0 else 16
        fmt='s'+str(blen)
        for j in range(0,dfm['ng']):
            vd = bs.unpack_from_dict('s9s8',['dtd','dtw'],msg,i)    
            i=i+17
            v=bs.unpack_from(fmt*nsat_l,msg,i)
            i=i+blen*nsat_l
        return i

    def decode_cssr_sinfo(self,msg,i):
        dfm = bs.unpack_from_dict('u1u3u2',['mi','cnt','dsize'],msg,i)
        i=i+6
        n=dfm['dsize']+1
        v=bs.unpack_from('u40'*n,msg,i)
        i=i+40*n
        return i

    def decode_cssr_comb(self,msg,i):
        dfm = bs.unpack_from_dict('u12u4u1u4u1u1u1',
                ['tow_s','uint','mi','iodssr','orb','clk','net'],msg,i)
        i=i+24
        if self.iodssr!=dfm['iodssr']:
            return -1
        if dfm['net']==1:
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
            if dfm['orb']==1:
                n=10 if sys_l[k]==2 else 8                   
                v=bs.unpack_from_dict('u'+str(n)+'s15s13s13',['iode','dx','dy','dz'],msg,i) 
                i=i+n+15+13+13
            if dfm['clk']==1:                  
                vc=bs.unpack_from_dict('s15',['clk'],msg,i) 
                i=i+15                

            print("comb %d %.2f " % (self.sys_n[k],vc['clk']))
            if dfm['orb']==1:
                print("%3d %.2f %.2f %.2f" % (v['iode'],v['dx'],v['dy'],v['dz']))


    
        return i

    def decode_cssr_atmos(self,msg,i):
        fmt='u12u4u1u4u2u2u5u6'
        dfm = bs.unpack_from_dict(fmt,['tow_s','uint','mi','iodssr','trop','stec','inet','ng'],msg,i)
        i=i+36
        if self.iodssr!=dfm['iodssr']:
            return -1
        # trop
        if dfm['trop']>0:
            trop_quality=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
            i=i+6
        if dfm['trop']&2: # function
            vh=bs.unpack_from_dict('u2',['ttype'],msg,i)
            i=i+2
            if vh['ttype']==0:
                vt=bs.unpack_from_dict('s9',['t00'],msg,i)
                i=i+9
            elif vh['ttype']==1:
                vt=bs.unpack_from_dict('s9s7s7',['t00','t01','t10'],msg,i)
                i=i+23                
            elif vh['ttype']==2:
                vt=bs.unpack_from_dict('s9s7s7s7',['t00','t01','t10','t11'],msg,i)
                i=i+30  

        if dfm['trop']&1: # residual                          
            vh=bs.unpack_from_dict('u1u4',['sz','ofst'],msg,i)
            i=i+5
            sz=6 if vh['sz']==0 else 8
            vtr=bs.unpack_from(('s'+str(sz))*dfm['ng'],msg,i)

        # STEC
        v=bs.unpack_from_dict('u'+str(self.nsat_n),['svmaskn'],msg,i)
        i=i+self.nsat_n
        loc,nsat_l=self.decode_mask(v['svmaskn'],self.nsat_n)
        
        for k in range(0,nsat_l):
            if dfm['stec']>0:
                stec_quality=bs.unpack_from_dict('u3u3',['class','val'],msg,i)
                i=i+6
            if dfm['stec']&2: # function            
                vh=bs.unpack_from_dict('u2',['stype'],msg,i)
                i=i+2
                i=self.decode_cssr_stec_coeff(msg,vh['stype'],i)
            
            if dfm['stec']&1: # residual
                vs=bs.unpack_from_dict('u2',['sz'],msg,i)
                i=i+2
                sz_t=[4,4,5,7]
                sz=sz_t[vs['sz']]
                v=bs.unpack_from(('s'+str(sz))*dfm['ng'],msg,i)
                i=i+sz*dfm['ng']
        return i

    def decode_cssr(self,msg):
        i=0
        df={'msgtype':4073}
        while df['msgtype']==4073:            
            df = bs.unpack_from_dict('u12u4',['msgtype','subtype'],msg,i) 
            i=i+16
            if df['msgtype']!=4073:
                return -1
            if df['subtype']==1:
                i=self.decode_cssr_mask(msg,i)
            elif df['subtype']==2: # orbit
                i=self.decode_cssr_orb(msg,i)
            elif df['subtype']==3: # clock
                i=self.decode_cssr_clk(msg,i)
            elif df['subtype']==4: # cbias
                i=self.decode_cssr_cbias(msg,i)
            elif df['subtype']==5: # pbias
                i=self.decode_cssr_pbias(msg,i)
            elif df['subtype']==6: # bias
                i=self.decode_cssr_bias(msg,i)            
            elif df['subtype']==7: # ura
                i=self.decode_cssr_ura(msg,i)
            elif df['subtype']==8: # stec
                i=self.decode_cssr_stec(msg,i)
            elif df['subtype']==9: # grid
                i=self.decode_cssr_grid(msg,i)                
            elif df['subtype']==10: # service-info
                i=self.decode_cssr_sinfo(msg,i)                
            elif df['subtype']==11: # orb+clk
                i=self.decode_cssr_comb(msg,i)
            elif df['subtype']==12: # atmos
                i=self.decode_cssr_atmos(msg,i)
            if i<=0:
                return 0


                        
        