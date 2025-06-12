"""
PVS (PPP via SouthPAN) correction data decoder

[1] Service Definition Document for Open Services, SBAS-STN-0001,
    Revision 02, December 2022

[2] Service Definition Document for Data Access Services, SBAS-STN-0002,
    Revision 01, November 2023

For use of SSR corrections see section A 3.9.2 in

[3] EUROCAE, ED-259, Minimum Operational Performance Standards for Galileo -
    Global Positioning System - Satellite-Based Augmentation System Airborne
    Equipment

"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, sCSSRTYPE, prn2sat, sCType
from cssrlib.cssrlib import sat2id
from cssrlib.gnss import uGNSS, rCST, gpst2time, timediff, timeadd, time2gpst
from binascii import unhexlify


def decode_sinca_line(line):
    """ SINCA (SISNeT compression algorithm) decoder  """
    if line[0:4] != '*MSG':
        return None
    v = line.split(',')
    s = v[3]
    week, tow = int(v[1]), int(v[2])
    t = gpst2time(week, tow)
    for key in '|/':
        if key in s:
            while True:
                pos = s.find(key)
                if pos < 0:
                    break
                c = s[pos-1]
                k = 2 if key == '|' else 3
                n = int(s[pos+1:pos+k], 16)
                s = s.replace(s[pos-1:pos+k], c*n, 1)

    sb = unhexlify(s.split('*')[0])
    return t, sb


class cssr_pvs(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.PVS_PPP
        self.nsig_max = 0
        self.sat_n = []

        self.lc[0].dclk = {}
        self.lc[0].ddft = {}
        self.lc[0].dorb = {}
        self.lc[0].dvel = {}
        self.lc[0].iode = {}
        self.lc[0].t0 = {}

        self.tmax = {sCType.CLOCK: 120.0, sCType.ORBIT: 120.0}

        # GPS LNAV, Galileo F/NAV
        self.nav_mode = {uGNSS.GPS: 0, uGNSS.GAL: 1}

    def slot2sat(self, slot):
        sat = 0
        if slot >= 1 and slot <= 32:
            prn = slot
            sat = prn2sat(uGNSS.GPS, prn)
        elif slot >= 38 and slot <= 69:
            prn = slot-37
            sat = prn2sat(uGNSS.GLO, prn)
        elif slot >= 75 and slot <= 110:
            prn = slot-74
            sat = prn2sat(uGNSS.GAL, prn)
        elif slot >= 120 and slot <= 158:
            prn = slot
            sat = prn2sat(uGNSS.SBS, prn)
        elif slot >= 159 and slot <= 195:
            prn = slot-158
            sat = prn2sat(uGNSS.BDS, prn)
        return sat

    def adjust_time_week(self, time, time0):
        dt = timediff(time, time0)
        if dt > rCST.HALFWEEK_SEC:
            time = timeadd(time, -rCST.WEEK_SEC)
        elif dt < -rCST.HALFWEEK_SEC:
            time = timeadd(time,  rCST.WEEK_SEC)
        return time

    def check_validity(self, time):
        for sat in self.sat_n:
            if timediff(time, self.lc[0].t0[sat][sCType.CLOCK]) > \
                    self.tmax[sCType.CLOCK]:
                self.lc[0].dclk[sat] = 0.0
            if timediff(time, self.lc[0].t0[sat][sCType.ORBIT]) > \
                    self.tmax[sCType.ORBIT]:
                self.lc[0].iode[sat] = -1
                self.lc[0].dorb[sat] = np.zeros(3)

    def decode_cssr_orb(self, msg, i, inet=0):
        """ Types 32 clock-ephemeris correction and covariance matrix """
        slot, iodn = bs.unpack_from('u8u10', msg, i)
        i += 18
        dx, dy, dz, db = bs.unpack_from('s11s11s11s12', msg, i)
        i += 45
        dxd, dyd, dzd, dbd, t0 = bs.unpack_from('s8s8s8s9u13', msg, i)
        i += 46

        # i, C = self.decode_cov(msg, i)
        i += 99
        # dfrei, dRcorr = bs.unpack_from('u4u4', msg, i)
        i += 8

        sat = self.slot2sat(slot)

        # correction for PVS is available only if mod(t0/16,2)==1
        if sat == 0 or t0 % 2 == 0:
            return

        if sat not in self.sat_n:
            self.sat_n.append(sat)

        tow = self.tow0 + t0*16.0
        self.time = self.adjust_time_week(
            gpst2time(self.week, tow), self.time0)

        self.lc[0].iode[sat] = iodn
        self.lc[0].dorb[sat] = np.zeros(3)
        self.lc[0].dorb[sat][0] = -self.sval(dx, 11, 0.0625)
        self.lc[0].dorb[sat][1] = -self.sval(dy, 11, 0.0625)
        self.lc[0].dorb[sat][2] = -self.sval(dz, 11, 0.0625)
        self.lc[0].dclk[sat] = self.sval(db, 12, 0.03125)

        self.lc[0].dvel[sat] = np.zeros(3)
        self.lc[0].dvel[sat][0] = -self.sval(dxd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][1] = -self.sval(dyd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][2] = -self.sval(dzd, 8, rCST.P2_11)
        self.lc[0].ddft[sat] = self.sval(dbd, 9, rCST.P2_12)

        self.iodssr = 0
        self.lc[0].cstat |= (1 << sCType.CLOCK) | (1 << sCType.ORBIT)
        self.lc[0].t0[sat] = {
            sCType.CLOCK: self.time0, sCType.ORBIT: self.time0}

        self.iodssr_c[sCType.CLOCK] = self.iodssr
        self.iodssr_c[sCType.ORBIT] = self.iodssr

        if self.monlevel > 0:
            self.fh.write("{:.1f} {:3s} {:3d}\n"
                          .format(tow, sat2id(sat), iodn))

        return i

    def decode_cssr(self, msg, i=0):

        _, mt = bs.unpack_from('u4u6', msg, i)
        i += 10

        if mt == 32:
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)

        if self.monlevel > 3:
            if self.time != -1:
                _, tow = time2gpst(self.time)
            else:
                tow = -1
            self.fh.write("mt={:2d} tow={:6.1f}\n"
                          .format(mt, tow))
