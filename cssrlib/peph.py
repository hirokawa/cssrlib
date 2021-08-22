# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 18:24:15 2021

@author: ruihi
"""

import gnss as gn
import numpy as np
# id2sat(id)

def read_sp3(file):
    with open(file,'rt') as f:
        line = f.readline()
        ver = line[1]
        sf=np.zeros(6)
        if ver!='c' and ver != 'd':
            return
        if True:
            line = f.readline()
            week = int(line[3:7])
            tow = float(line[8:23])
            dint = float(line[25:38])
            jd = int(line[39:44])
            line = f.readline()
            nsat=int(line[3:6])
            sat=[]
            for i in range(nsat):
                k=i%17
                if i>0 and i%17==0:
                    line = f.readline()
                if line[0:2]!='+ ':
                    break
                sat.append(gn.id2sat(line[9+3*k:12+3*k]))
            while (1):
                line = f.readline()
                if line[0:2]=='++':
                    break
            quality=[]
            for i in range(nsat):
                k=i%17
                if i>0 and i%17==0:
                    line = f.readline()
                if line[0:2]!='++':
                    break
                quality.append(float(line[9+3*k:12+3*k]))
            k=0
            t=[]
            while (1):
                line = f.readline()
                if line[0:2]=='* ':
                    break
                if line[0:2]=='%f':
                    sf[k]=float(line[ 3:13]);k+=1
                    sf[k]=float(line[14:26]);k+=1
            ep=np.zeros(6)
            orb=None
            d=np.zeros((nsat,6))
    
            while (1):
                if line[0:2]!='* ':
                    break                
                ep[0]=float(line[ 3: 7])
                ep[1]=float(line[ 8:10])
                ep[2]=float(line[11:13])
                ep[3]=float(line[14:16])
                ep[4]=float(line[17:19])
                ep[5]=float(line[20:31])
                td=gn.epoch2time(ep)
                week,tow=gn.time2gpst(td)
                t.append(td)
                
                for i in range(nsat):
                    line = f.readline()
                    if line[0]!='P':
                        continue
                    d[i,0] = tow
                    d[i,1] = gn.id2sat(line[1:4])
                    d[i,2] = float(line[4:18])*1e3
                    d[i,3] = float(line[18:32])*1e3
                    d[i,4] = float(line[32:46])*1e3
                    d[i,5] = float(line[46:60])*1e3
                    #print("%3d %10.1f %13.3f %13.3f %13.3f" % (d[i,1],d[i,0],d[i,2],d[i,3],d[i,4]))
                if orb is None:
                    orb=d.copy()
                else:
                    orb=np.vstack((orb,d))
                line = f.readline()
    return t,nsat,orb

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    #sat_t = [156,157,158,162]
    sat_t = [156,157,158] # QZS
    #sat_t = [1,2,3,6] # GPS
    #sat_t = [57,58,59,60] # GAL
    #sat_t = [110,111,112,113]
    week=2160
    #week=2158
    day_t = [0,1,2,3,4]
    bdir='c:/work/log/gbu/'
    
    for day in day_t:
        #sp3file=bdir+'qzf%04d%1d.sp3' % (week,day)
        sp3file=bdir+'COD0MGXFIN_20211360000_01D_05M_ORB.SP3' 
        sp3file1=bdir+'gbu%04d%1d_00.sp3' % (week,day)
        
        t,nsat,orb=read_sp3(sp3file)
        t1,nsat1,orb1=read_sp3(sp3file1)
        
        nsat=len(sat_t)
        nep=len(t)-2
        r = np.zeros((nep,nsat,3))
        tk = np.zeros(nep)
        for k in range(nep):
            week,tow=gn.time2gpst(t[k])
            tk[k]=tow
            d=orb[orb[:,0]==tow]
            d1=orb1[orb1[:,0]==tow]
            for i,sat in enumerate(sat_t):
                v=d[d[:,1]==sat][0]
                v1=d1[d1[:,1]==sat][0]
                r[k,i,:] = v[2:5]-v1[2:5]
                #print("%d %f %f %f %f" % (sat,tow,r[0],r[1],r[2]))
        
        plt.figure(figsize=(12,10))
        tt=tk-tk[0]
        for i in range(nsat):
            id_=gn.sat2id(sat_t[i])
            plt.subplot(2,2,i+1)
            plt.plot(tt/3600,r[:,i,:])
            plt.grid()
            plt.title(id_)
            plt.legend(['x','y','z'])
        plt.savefig('gbu-qzf-%04d%01d.png' % (week,day))
        
        
            