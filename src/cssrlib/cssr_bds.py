"""
BDS PPP correction data decoder

[1] BeiDou Navigation Satellite System Signal In Space Interface Control Document
Precise Point Positioning Service Signal PPP-B2b (Version 1.0), 2020
"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, sCSSRTYPE, sGNSS, prn2sat, sCType
from cssrlib.gnss import bdt2time, bdt2gpst, uGNSS, uSIG, uTYP, rSigRnx


class cssr_bds(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.BDS_PPP
        self.nsig_max = 8
        self.iodp = -1
        self.iodp_p = -1

    def ssig2rsig(self, sys: sGNSS, utyp: uTYP, ssig):
        gps_tbl = {
            0: uSIG.L1C,
            1: uSIG.L1P,
            4: uSIG.L1L,
            5: uSIG.L1X,
            7: uSIG.L2L,
            8: uSIG.L2X,
            11: uSIG.L5I,
            12: uSIG.L5Q,
            13: uSIG.L5X,
        }
        glo_tbl = {
            0: uSIG.L1C,
            1: uSIG.L1P,
            2: uSIG.L2C,
        }

        gal_tbl = {
            1: uSIG.L1B,
            2: uSIG.L1C,
            4: uSIG.L5Q,
            5: uSIG.L5I,
            7: uSIG.L7I,
            8: uSIG.L7Q,
            11: uSIG.L6C,
        }

        bds_tbl = {
            0: uSIG.L2I,
            1: uSIG.L1D,
            2: uSIG.L1P,
            4: uSIG.L5D,
            5: uSIG.L5P,
            7: uSIG.L7D,
            8: uSIG.L7P,
            12: uSIG.L6I,
        }

        usig_tbl_ = {
            uGNSS.GPS: gps_tbl,
            uGNSS.GLO: glo_tbl,
            uGNSS.GAL: gal_tbl,
            uGNSS.BDS: bds_tbl,
        }

        usig_tbl = usig_tbl_[sys]
        return rSigRnx(sys, utyp, usig_tbl[ssig])

    def sval(self, u, n, scl):
        """ calculate signed value based on n-bit int, lsb """
        invalid = -2**(n-1)
        dnu = -(2**(n-1)-1)  # this value seems to be invalid
        y = np.nan if u == invalid or u == dnu else u*scl
        return y

    def slot2prn(self, slot):
        prn = 0
        sys = uGNSS.NONE
        if slot >= 1 and slot <= 63:
            prn = slot
            sys = uGNSS.BDS
        elif slot <= 100:
            prn = slot-63
            sys = uGNSS.GPS
        elif slot <= 137:
            prn = slot-100
            sys = uGNSS.GAL
        elif slot <= 174:
            prn = slot-137
            sys = uGNSS.GLO
        return sys, prn

    def decode_head(self, msg, i, st=-1):
        self.tod, _, iodssr = bs.unpack_from('u17u4u2', msg, i)
        i += 23

        if st == sCSSR.MASK:
            self.iodssr = iodssr

        if self.tow0 >= 0:
            self.tow = self.tow0+self.tod
            if self.week >= 0:
                self.time = bdt2gpst(bdt2time(self.week, self.tow))

        head = {'uint': 0, 'mi': 0, 'iodssr': iodssr}
        return head, i

    def add_gnss(self, mask, blen, gnss):
        prn, nsat = self.decode_mask(mask, blen)
        self.nsat_g[gnss] = nsat
        self.nsat_n += nsat
        if nsat > 0:
            self.ngnss += 1
        sys = self.gnss2sys(gnss)
        for k in range(0, nsat):
            sat = prn2sat(sys, prn[k])
            self.sys_n.append(sys)
            self.sat_n.append(sat)
            self.gnss_n.append(gnss)

    def decode_cssr_mask(self, msg, i):
        """decode MT1 Mask message """
        head, i = self.decode_head(msg, i, sCSSR.MASK)

        self.iodp = bs.unpack_from('u4', msg, i)[0]
        i += 4

        mask_bds, mask_gps, mask_gal, mask_glo = \
            bs.unpack_from('u63u37u37u37', msg, i)
        i += 174

        if self.iodp != self.iodp_p:
            self.sat_n_p = self.sat_n
            self.iodssr_p = self.iodssr
            self.sig_n_p = self.sig_n
            self.iodp_p = self.iodp

            self.nsat_n = 0
            self.nsig_n = []
            self.sys_n = []
            self.gnss_n = []
            self.sat_n = []
            self.nsig_total = 0
            self.sig_n = []
            self.nm_idx = np.zeros(self.SYSMAX, dtype=int)
            self.ngnss = 0
            # self.gnss_idx = np.zeros(self.ngnss, dtype=int)
            self.nsat_g = np.zeros(self.SYSMAX, dtype=int)

            self.add_gnss(mask_bds, 63, sGNSS.BDS)
            self.add_gnss(mask_gps, 37, sGNSS.GPS)
            self.add_gnss(mask_gal, 37, sGNSS.GAL)
            self.add_gnss(mask_glo, 37, sGNSS.GLO)

            inet = 0
            self.lc[inet].dclk = np.ones(self.nsat_n)*np.nan
            self.lc[inet].dorb = np.ones((self.nsat_n, 3))*np.nan
            self.lc[inet].iode = np.zeros(self.nsat_n, dtype=int)
            self.lc[inet].iodc = np.zeros(self.nsat_n, dtype=int)
            self.lc[inet].iodc_c = np.zeros(self.nsat_n, dtype=int)
            self.lc[inet].cbias = np.ones((self.nsat_n, self.nsig_max))*np.nan
            self.nsig_n = np.ones(self.nsat_n, dtype=int)*self.nsig_max
            self.sig_n = -1*np.ones((self.nsat_n, self.nsig_max), dtype=int)
            self.ura = np.zeros(self.nsat_n)

            # fallback for inconsistent clock update
            self.lc[inet].dclk_p = np.ones(self.nsat_n)*np.nan
            self.lc[inet].iodc_c_p = np.zeros(self.nsat_n, dtype=int)

        self.iodssr = head['iodssr']

        self.lc[0].cstat |= (1 << sCType.MASK)
        self.lc[0].t0[sCType.MASK] = self.time
        return i

    def decode_cssr_orb_sat(self, msg, i, inet, sat_n):
        slot, iodn, iodc, dx, dy, dz, ucls, uval = \
            bs.unpack_from('u9u10u3s15s13s13u3u3', msg, i)
        i += 69
        sys, prn = self.slot2prn(slot)
        if sys == uGNSS.NONE:
            return i
        sat = prn2sat(sys, prn)
        if sat not in sat_n:
            return i
        idx = np.where(sat == sat_n)[0][0]

        if (sys == uGNSS.GPS) or (sys == uGNSS.BDS):
            iodn = iodn & 0xff  # IODC -> IODE

        self.lc[inet].iode[idx] = iodn
        self.lc[inet].iodc[idx] = iodc
        self.lc[inet].dorb[idx, 0] = self.sval(dx, 15, self.dorb_scl[0])
        self.lc[inet].dorb[idx, 1] = self.sval(dy, 13, self.dorb_scl[1])
        self.lc[inet].dorb[idx, 2] = self.sval(dz, 13, self.dorb_scl[2])
        self.ura[idx] = self.quality_idx(ucls, uval)
        return i

    def decode_cssr_orb(self, msg, i, inet=0):
        """decode MT2 orbit + URA message """
        head, i = self.decode_head(msg, i)
        sat_n = np.array(self.sat_n)

        if self.iodssr != head['iodssr']:
            return -1

        for k in range(6):
            i = self.decode_cssr_orb_sat(msg, i, inet, sat_n)

        self.iodssr_c[sCType.ORBIT] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.ORBIT)
        self.lc[inet].t0[sCType.ORBIT] = self.time
        i += 19
        return i

    def decode_cssr_cbias(self, msg, i, inet=0):
        """decode MT3 Code Bias Correction message """
        head, i = self.decode_head(msg, i)
        nsat = self.nsat_n
        self.flg_net = False
        if self.iodssr != head['iodssr']:
            return -1

        sat_n = np.array(self.sat_n)
        nsat = bs.unpack_from('u5', msg, i)[0]
        i += 5

        for k in range(nsat):
            slot, nsig = bs.unpack_from('u9u4', msg, i)
            i += 13
            sys, prn = self.slot2prn(slot)
            sat = prn2sat(sys, prn)
            idx = np.where(sat == sat_n)[0]
            if len(idx) == 0:
                continue
            for j in range(0, nsig):
                sig, cb = bs.unpack_from('u4s12', msg, i)
                i += 16
                self.sig_n[idx, j] = sig
                self.lc[inet].cbias[idx, j] = self.sval(cb, 12, self.cb_scl)

        self.iodssr_c[sCType.CBIAS] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CBIAS)
        self.lc[inet].t0[sCType.CBIAS] = self.time
        return i

    def decode_cssr_clk_sat(self, msg, i, inet, idx):
        """ decode clock correction for satellite """
        iodc, dclk = bs.unpack_from('u3s15', msg, i)
        i += 18
        if iodc != self.lc[inet].iodc_c[idx]:
            self.lc[inet].iodc_c_p[idx] = self.lc[inet].iodc_c[idx]
            self.lc[inet].dclk_p[idx] = self.lc[inet].dclk[idx]
        self.lc[inet].iodc_c[idx] = iodc
        # note: the sign of the clock correction reversed
        self.lc[inet].dclk[idx] = -self.sval(dclk, 15, self.dclk_scl)
        return i

    def decode_cssr_clk(self, msg, i, inet=0):
        """decode MT4 Clock Correction message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1

        iodp, st1 = bs.unpack_from('u4u5', msg, i)
        i += 9
        if iodp != self.iodp:
            i += 23*18+10
            return i
        for k in range(23):
            idx = st1*23+k
            if idx < self.nsat_n:
                i = self.decode_cssr_clk_sat(msg, i, inet, idx)

        self.iodssr_c[sCType.CLOCK] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CLOCK)
        self.lc[inet].t0[sCType.CLOCK] = self.time
        i += 10
        return i

    def decode_cssr_ura(self, msg, i):
        """decode MT5 URA message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1

        iodp, st2 = bs.unpack_from('u4u3', msg, i)
        i += 7

        for k in range(70):
            v = bs.unpack_from_dict('u3u3', ['class', 'val'], msg, i)
            i += 6
            idx = st2*70+k
            if idx < self.nsat_n:
                self.ura[idx] = self.quality_idx(v['class'], v['val'])

        self.lc[0].cstat |= (1 << sCType.URA)
        self.lc[0].t0[sCType.URA] = self.time
        return i

    def decode_cssr_comb1(self, msg, i, inet=0):
        """decode MT6 combined message #1 """
        numc, numo = bs.unpack_from('u5u3', msg, i)
        sat_n = np.array(self.sat_n)

        if numc > 0:
            tod, _, iodssr, iodp, slot_s = \
                bs.unpack_from('u17u4u2u4u9', msg, i)
            i += 36
            for k in range(numc):
                idx = slot_s+k
                i = self.decode_cssr_clk_sat(msg, i, inet, idx)

        if numo > 0:
            tod, _, iodssr = bs.unpack_from('u17u4u2', msg, i)
            i += 23
            for k in range(numo):
                i = self.decode_cssr_orb_sat(msg, i, inet, sat_n)

        return i

    def decode_cssr_comb2(self, msg, i, inet=0):
        """decode MT7 combined message #2 """
        numc, numo = bs.unpack_from('u5u3', msg, i)
        sat_n = np.array(self.sat_n)

        if numc > 0:
            tod, _, iodssr = bs.unpack_from('u17u4u2', msg, i)
            i += 23
            for k in range(numc):
                slot = bs.unpack_from('u9', msg, i)
                i += 9
                sys, prn = self.slot2prn(slot)
                sat = prn2sat(sys, prn)
                idx = np.where(sat == sat_n)[0]
                i = self.decode_cssr_clk_sat(msg, i, inet, idx)

        if numo > 0:
            tod, _, iodssr = bs.unpack_from('u17u4u2', msg, i)
            i += 23
            for k in range(numo):
                i = self.decode_cssr_orb_sat(msg, i, inet, sat_n)

        return i

    def decode_cssr(self, msg, i=0):
        mt = bs.unpack_from('u6', msg, i)[0]
        i += 6

        if mt == 1:
            self.subtype = sCSSR.MASK
            i = self.decode_cssr_mask(msg, i)
        elif mt == 2:
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        elif mt == 3:
            self.subtype = sCSSR.CBIAS
            i = self.decode_cssr_cbias(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        elif mt == 4:
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_clk(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        elif mt == 5:
            self.subtype = sCSSR.URA
            i = self.decode_cssr_ura(msg, i)
        elif mt == 6:
            i = self.decode_cssr_comb1(msg, i)
        elif mt == 7:
            i = self.decode_cssr_comb2(msg, i)

        if self.monlevel > 0:
            print(" mt={:2d} tow={:6.1f}".format(mt, self.tow))
