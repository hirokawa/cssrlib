"""
QZSS MADOCA-PPP correction data decoder

[1] Quasi-Zenith Satellite System Interface Specification Multi-GNSS
    Advanced Orbit and Clock Augmentation - Precise Point Positioning
    (IS-QZSS-MDC-003), 2024

"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSRTYPE
from cssrlib.gnss import gpst2time, uGNSS, prn2sat


class areaInfo():
    def __init__(self, sid, latr, lonr, p1=0, p2=0):
        self.sid = sid
        self.latr = latr
        self.lonr = lonr
        if sid == 0:
            self.lats = p1
            self.lons = p2
        elif sid == 1:
            self.rng = p1


class ionoCorr():
    def __init__(self):
        self.t0 = None
        self.qi = []
        self.iodssr = 0
        self.ct = 0
        self.sat = []
        self.c = np.zeros((6))


class cssr_mdc(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.QZS_MADOCA
        self.buff = bytearray(250*10)

        self.pnt = {}
        self.ci = {}

    def decode_mdc_stec_area(self, buff, i=0):
        """ decoder for MT1 - STEC Coverage Message """
        msgtype, subtype = bs.unpack_from('u12u4', buff, i)
        i += 16
        tow, uid, mi, iodssr = bs.unpack_from('u20u4u1u4', buff, i)
        i += 29
        self.tow0 = tow//3600*3600
        reg, alrt, len_, narea = bs.unpack_from('u8u1u16u5', buff, i)
        i += 30
        if reg not in self.pnt:
            self.pnt[reg] = {}

        for k in range(narea):
            area, sid = bs.unpack_from('u5u1', buff, i)
            i += 6

            if sid == 0:
                latr, lonr, lats, lons = bs.unpack_from('s11u12u8u8', buff, i)
                if self.monlevel > 2:
                    print(f"{reg} {area:2d} {sid} {latr*0.1:5.1f} "
                          f"{lonr*0.1:5.1f} {lats*0.1:3.1f} {lons*0.1:3.1f}")
                self.pnt[reg][area] = areaInfo(
                    sid, latr*0.1, lonr*0.1, lats*0.1, lons*0.1)
            else:
                latr, lonr, rng = bs.unpack_from('s15u16u8', buff, i)
                if self.monlevel > 2:
                    print(f"{reg} {area:2d} {sid} {latr*0.01:6.2f} "
                          f"{lonr*0.01:6.2f} {rng*10}")
                self.pnt[reg][area] = areaInfo(
                    sid, latr*0.01, lonr*0.01, rng*10.0)
            i += 39
        return i

    def decode_mdc_stec_corr(self, buff, i=0):
        """ decoder for MT2 - STEC Correction Message """
        msgtype, subtype = bs.unpack_from('u12u4', buff, i)
        i += 16
        dtow, uid, mi, iodssr = bs.unpack_from('u12u4u1u4', buff, i)
        i += 21
        reg, area, ct = bs.unpack_from('u8u5u2', buff, i)
        i += 15

        nsat = bs.unpack_from('u5u5u5u5u5', buff, i)
        i += 25
        # gps, glo, gal, bds, qzss

        sys_t = [uGNSS.GPS, uGNSS.GLO, uGNSS.GAL, uGNSS.BDS, uGNSS.QZS]

        if reg not in self.ci:
            self.ci[reg] = {}
        if area not in self.ci[reg]:
            self.ci[reg][area] = ionoCorr()

        c01 = c10 = c11 = c02 = c20 = 0
        nsat_total = np.sum(nsat)
        c = np.zeros((nsat_total, 6))
        qi = np.zeros((nsat_total), dtype=int)
        sat_ = []
        j = 0
        for gnss in range(5):
            sys = sys_t[gnss]
            for k in range(nsat[gnss]):
                prn, qi[j], c00 = bs.unpack_from('u6u6s14', buff, i)
                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)

                i += 26
                if ct > 0:
                    c01, c10 = bs.unpack_from('s12s12', buff, i)
                    i += 24
                if ct > 1:
                    c11 = bs.unpack_from('s10', buff, i)[0]
                    i += 10
                if ct > 2:
                    c02, c20 = bs.unpack_from('s8s8', buff, i)
                    i += 16

                sat_ += [sat]
                c[j, :] = [c00*0.05, c01*0.02, c10*0.02, c11*0.02,
                           c02*5e-3, c20*5e-3]
                j += 1

            self.ci[reg][area].t0 = gpst2time(self.week, self.tow0+dtow)
            self.ci[reg][area].iodssr = iodssr
            self.ci[reg][area].sat = sat_
            self.ci[reg][area].qi = qi
            self.ci[reg][area].ct = ct
            self.ci[reg][area].c = c
        return i
