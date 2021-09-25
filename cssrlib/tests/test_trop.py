# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

import matplotlib.pyplot as plt
from cssrlib.gnss import Nav, time2gpst, epoch2time, time2epoch, timeadd, \
    antmodel, tropmodel, tropmapf
import numpy as np

igfont = {'family': 'Meiryo'}

t = epoch2time([2021, 3, 19, 12, 0, 0])
ep = time2epoch(t)
week, tow = time2gpst(t)
t1 = timeadd(t, 300)

nav = Nav()

el_t = np.arange(10, 90)
nf = 2
ofst_r = np.zeros((len(el_t), nf))
ofst_b = np.zeros((len(el_t), nf))
for k, el in enumerate(el_t):
    ofst_r[k, :] = antmodel(nav, np.deg2rad(el), nf, 1)
    ofst_b[k, :] = antmodel(nav, np.deg2rad(el), nf, 0)

flg_ant = False
flg_trop = True

plt.figure()
if flg_ant:
    plt.plot(el_t, ofst_b[:, 0]*100, label='Trimble TRM59800.80')
    plt.plot(el_t, ofst_r[:, 0]*100, label='JAVAD RINGANT')
    plt.grid()
    plt.legend()
    plt.xlabel('elevation[deg]')
    plt.ylabel('range correction for antenna offset [cm]')
if flg_trop:
    ep = [2021, 4, 1, 0, 0, 0]
    t = epoch2time(ep)
    el_t = np.arange(0.01, np.pi/2, 0.01)
    n = len(el_t)
    trop = np.zeros(n)
    trop_hs = np.zeros(n)
    trop_wet = np.zeros(n)
    lat_t = [45]
    for lat in lat_t:
        pos = [np.deg2rad(lat), 0, 0]
        for k, el in enumerate(el_t):
            ths, twet, z = tropmodel(t, pos, el)
            mapfh, mapfw = tropmapf(t, pos, el)
            trop_hs[k] = mapfh*ths
            trop_wet[k] = mapfw*twet

        trop = trop_hs+trop_wet
        plt.plot(np.rad2deg(el_t), trop)
        plt.plot(np.rad2deg(el_t), trop_hs)
        plt.plot(np.rad2deg(el_t), trop_wet)

    plt.grid()
    plt.axis([0, 90, 0, 10])
    plt.legend(['全体', '静水圧項', '湿潤項'], prop=igfont)
    plt.xlabel('仰角 [deg]', **igfont)
    plt.ylabel('対流圏遅延 [m]', **igfont)
    plt.show()
