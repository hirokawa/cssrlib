# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 19:57:00 2020

@author: ruihi
"""

import cbitstruct as bs
from gnss import uGNSS

class ephsat():
    sat=0
    iode=-1
    iodc=-1
    sqrtA=0
    week=0
    deln=0
    M0=0
    OMG0=0
    omg=0
    OMGd=0
    idot=0
    i0=0
    e=0
    fit=0
    cic=0
    cis=0
    crc=0
    crs=0
    cuc=0
    cus=0
    flag=-1
    tgd=[0,0]
    sva=-1
    svh=-1
    toc=0
    toe=0
    tr=0
    f2=0
    f1=0
    f0=0
    status=0

class navmsg():
    #def __init(self)__:
        
    def decode_gps_lnav(self,eph,buff,sat,id):
        v=bs.unpack_from_dict('u17u1',['tow','alert'],buff,24)
        if v['alert']==1:
            return 0
        tow=v['tow']*6
        iodc0=eph[sat-1].iodc
        i=48        
        if id==1:
            # frame1
            names=['week','code','sva','svh','iodc0','flag']
            v1a=bs.unpack_from_dict('u10u2u4u6u2u1',names,buff,i)
            i=i+10+2+4+6+2+1+87
            names=['tgd','iodc1','toc','f2','f1','f0']
            v1b=bs.unpack_from_dict('s8u8u16s8s16s22',names,buff,i)
            iodc=(v1a['iodc0']<<8)+v1b['iodc1']
            if eph[sat-1].status&1==0:
                eph[sat-1].status = 1
                eph[sat-1].iodc=iodc
                eph[sat-1].toc=v1b['toc']*16.0
                eph[sat-1].week=v1a['week']
                eph[sat-1].code=v1a['code']
                eph[sat-1].sva=v1a['sva']
                eph[sat-1].svh=v1a['svh']
                eph[sat-1].tgd[0]=0.0 if v1b['tgd']==-128 else v1b['tgd']*4.656612873077393e-10
                eph[sat-1].f0=v1b['f0']*4.656612873077393e-10
                eph[sat-1].f1=v1b['f1']*1.1368683772161603e-13
                eph[sat-1].f2=v1b['f2']*2.7755575615628914e-17
                eph[sat-1].tr=tow
        
        # frame2
        if id==2:
            names=['iode','crs','deln','M0','cuc','e','cus','sqrtA','toe','fit']
            v=bs.unpack_from_dict('u8s16s16s32s16u32s16u32u16u1',names,buff,i)
            if (iodc0&0xff)!=v['iode']:
                return 0
            if eph[sat-1].status&2==0:
                eph[sat-1].iode=v['iode']
                eph[sat-1].crs=v['crs']*0.03125
                eph[sat-1].deln=v['deln']*3.571577341960847e-13
                eph[sat-1].M0=v['M0']*1.462918079267163e-09            
                eph[sat-1].cuc=v['cuc']*1.862645149230957e-09
                eph[sat-1].e=v['e']*1.1641532182693481e-10
                eph[sat-1].cus=v['cus']*1.862645149230957e-09
                eph[sat-1].sqrtA=v['sqrtA']*1.9073486328125e-06
                eph[sat-1].toe=v['toe']*16.0
                eph[sat-1].fit=0.0 if v['fit']==1 else 4.0
                eph[sat-1].status=eph[sat-1].status|2
            
        #frame3
        if id==3:
            names=['cic','OMG0','cis','i0','crc','omg','OMGd','iode','idot']
            v=bs.unpack_from_dict('s16s32s16s32s16s32s24u8s14',names,buff,i)
            if (iodc0&0xff)!=v['iode']:
                return 0
            if eph[sat-1].status&4==0:
                eph[sat-1].cic=v['cic']*1.862645149230957e-09
                eph[sat-1].OMG0=v['OMG0']*1.462918079267163e-09
                eph[sat-1].cis=v['cis']*1.862645149230957e-09
                eph[sat-1].i0=v['i0']*1.462918079267163e-09            
                eph[sat-1].crc=v['crc']*0.03125
                eph[sat-1].omg=v['omg']*1.462918079267163e-09
                eph[sat-1].OMGd=v['OMGd']*3.571577341960847e-13
                eph[sat-1].idot=v['idot']*3.571577341960847e-13
                eph[sat-1].status=eph[sat-1].status|4
   
        if eph[sat-1].status==7:
            eph[sat-1].sat=sat
            print('GPS/QZS LNAV: sat=%d iode=%3d' % (sat,iodc0&0xff))
           
        return 0
            
    def decode_gal_inav(self,eph,buff,sat):
        type=[0,0,0,0,0,0]
        type[0]=bs.unpack_from('u6',buff,0)[0]
        type[1]=bs.unpack_from('u6',buff,128)[0]
        type[2]=bs.unpack_from('u6',buff,128*2)[0]
        type[3]=bs.unpack_from('u6',buff,128*3)[0]
        type[4]=bs.unpack_from('u6',buff,128*4)[0]
        type[5]=bs.unpack_from('u6',buff,128*5)[0]
        
        if type[0]!=0 or type[1]!=1 or type[2]!=2 or type[3]!=3 or type[4]!=4 or type[5]!=5:
            return 0

        iod_nav=[0,0,0,0]
        iod_nav[0]=bs.unpack_from('u10',buff,128+6)[0]
        iod_nav[1]=bs.unpack_from('u10',buff,128*2+6)[0]
        iod_nav[2]=bs.unpack_from('u10',buff,128*3+6)[0]
        iod_nav[3]=bs.unpack_from('u10',buff,128*4+6)[0]
        
        if iod_nav[0]!=iod_nav[1] or iod_nav[0]!=iod_nav[2] or iod_nav[0]!=iod_nav[3]:
            return 0
        
        svid=bs.unpack_from('u6',buff,128*4+16)[0]
        if svid>uGNSS.GALMAX:
            return 0
        i=6
        time_f=bs.unpack_from('u2',buff,i)[0]
        i=i+2+88
        week=bs.unpack_from('u12',buff,i)[0]
        i=i+12
        tow=bs.unpack_from('u20',buff,i)[0]
        
        if time_f!=2:
            return 0
        
        if eph[sat-1].sat==sat and eph[sat-1].iode==iod_nav[0]:
            return 0
        
        i=128+6+10
        v=bs.unpack_from('u14s32u32u32',buff,i)
        eph[sat-1].toe=v[0]*60.0
        eph[sat-1].M0=v[1]*1.462918079267163e-09
        eph[sat-1].e=v[2]*1.1641532182693481e-10
        eph[sat-1].sqrtA=v[3]*1.9073486328125e-06
        i=128*2+6+10
        v=bs.unpack_from('s32s32s32s14',buff,i)        
        eph[sat-1].OMG0=v[0]*1.462918079267163e-09
        eph[sat-1].i0=v[1]*1.462918079267163e-09
        eph[sat-1].omg=v[2]*1.462918079267163e-09
        eph[sat-1].idot=v[3]*3.571577341960847e-13
        i=128*3+6+10
        v=bs.unpack_from('s24s16s16s16s16s16u8',buff,i)        
        eph[sat-1].OMGd=v[0]*3.571577341960847e-13
        eph[sat-1].deln=v[1]*3.571577341960847e-13
        eph[sat-1].cuc=v[2]*1.862645149230957e-09
        eph[sat-1].cus=v[3]*1.862645149230957e-09       
        eph[sat-1].crc=v[4]*0.03125
        eph[sat-1].crs=v[5]*0.03125                      
        eph[sat-1].sva=v[6]       
        i=128*4+6+10
        v=bs.unpack_from('u6s16s16u14s31s21s6',buff,i)        
        svid=v[0]
        eph[sat-1].cic=v[1]*1.862645149230957e-09
        eph[sat-1].cis=v[2]*1.862645149230957e-09
        eph[sat-1].toc=v[3]*60.0       
        eph[sat-1].f0=v[4]*5.820766091346741e-11
        eph[sat-1].f1=v[5]*1.4210854715202004e-14                      
        eph[sat-1].f2=v[6]*1.734723475976807e-18

        i=128*5+6
        #v=bs.unpack_from('u11s11s14u5',buff,i)
        i=i+11+11+14+5
        v=bs.unpack_from('s10s10u2u2u1u1',buff,i)
        eph[sat-1].tgd[0]=v[0]*2.3283064365386963e-10  
        eph[sat-1].tgd[1]=v[1]*2.3283064365386963e-10
        e5b_hs=v[2]
        e1b_hs=v[3]
        e5b_dvs=v[4]
        e1b_dvs=v[5]
        
        eph[sat-1].iode=iod_nav[0]
        eph[sat-1].svh=(e5b_hs<<7)|(e5b_dvs<<6)|(e1b_hs<<1)|e1b_dvs
        
        print('GAL INAV: sat=%2d iodnav=%3d' % (sat,iod_nav[0]))
  
        return 0
        

