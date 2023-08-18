"""
RTCM 3 decoder

[1] RTCM Standard 10403.4, 2023
[2] IGS SSR Format version 1.00, 2020

"""

import numpy as np
import struct as st
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, sCSSRTYPE, prn2sat, sCType
from cssrlib.gnss import uGNSS, sat2id, gpst2time, timediff, time2str, sat2prn
from cssrlib.gnss import uTYP, uSIG, rSigRnx, bdt2time, bdt2gpst, glo2time
from cssrlib.gnss import time2bdt, gpst2bdt, rCST, time2gpst, utc2gpst, timeadd
from crccheck.crc import Crc24LteA
from enum import IntEnum
from cssrlib.gnss import Eph, Obs, Geph, Seph


class sRTCM(IntEnum):
    """ class to define RTCM message types """
    NRTK_RES = 1001
    MSM = 1002
    ANT_DESC = 1003
    ANT_POS = 1004
    GLO_BIAS = 1005
    GPS_EPH = 1011
    GLO_EPH = 1012
    GAL_EPH = 1013
    BDS_EPH = 1014
    QZS_EPH = 1015
    IRN_EPH = 1016
    SBS_EPH = 1016


class rtcm(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.len = 0
        self.monlevel = 1
        self.sysref = -1
        self.nsig_max = 4

        self.nrtk_r = {}

        self.msm_t = {
            uGNSS.GPS: 1071, uGNSS.GLO: 1081, uGNSS.GAL: 1091,
            uGNSS.SBS: 1101, uGNSS.QZS: 1111, uGNSS.BDS: 1121
        }

        self.ssr_t = {
            uGNSS.GPS: 1057, uGNSS.GLO: 1063, uGNSS.GAL: 1240,
            uGNSS.QZS: 1246, uGNSS.SBS: 1252, uGNSS.BDS: 1258
        }

    def is_msmtype(self, msgtype):
        for sys_ in self.msm_t.keys():
            if msgtype >= self.msm_t[sys_] and msgtype <= self.msm_t[sys_]+6:
                return True
        return False

    def msmtype(self, msgtype):
        sys = uGNSS.NONE
        msm = 0
        for sys_ in self.msm_t.keys():
            if msgtype >= self.msm_t[sys_] and msgtype <= self.msm_t[sys_]+6:
                sys = sys_
                msm = msgtype-self.msm_t[sys_]+1
                break
        return sys, msm

    def ssrtype(self, msgtype):
        sys = uGNSS.NONE
        ssr = 0
        for sys_ in self.ssr_t.keys():
            if msgtype >= self.ssr_t[sys_] and msgtype <= self.ssr_t[sys_]+6:
                sys = sys_
                ssr = msgtype-self.ssr_t[sys_]+1
                break
        return sys, ssr

    def ssig2rsig(self, sys: uGNSS, utyp: uTYP, ssig):
        gps_tbl = {
            0: uSIG.L1C,
            1: uSIG.L1P,
            2: uSIG.L1W,
            5: uSIG.L2C,
            6: uSIG.L2D,
            7: uSIG.L2S,
            8: uSIG.L2L,
            9: uSIG.L2X,
            10: uSIG.L2P,
            11: uSIG.L2W,
            14: uSIG.L5I,
            15: uSIG.L5Q,
            16: uSIG.L5X,
            17: uSIG.L1S,
            18: uSIG.L1L,
            19: uSIG.L1X,
        }
        glo_tbl = {
            0: uSIG.L1C,
            1: uSIG.L1P,
            2: uSIG.L2C,
            3: uSIG.L2P,
            10: uSIG.L3I,
            11: uSIG.L3Q,
            12: uSIG.L3X,
        }

        gal_tbl = {
            0: uSIG.L1A,
            1: uSIG.L1B,
            2: uSIG.L1C,
            3: uSIG.L1X,
            4: uSIG.L1Z,
            5: uSIG.L5I,
            6: uSIG.L5Q,
            7: uSIG.L5X,
            8: uSIG.L7I,
            9: uSIG.L7Q,
            10: uSIG.L7X,
            11: uSIG.L8I,
            12: uSIG.L8Q,
            13: uSIG.L8X,
            14: uSIG.L6A,
            15: uSIG.L6B,
            16: uSIG.L6C,
            17: uSIG.L6X,
            18: uSIG.L6Z,
        }

        bds_tbl = {
            0: uSIG.L2I,
            1: uSIG.L2Q,
            2: uSIG.L2X,
            3: uSIG.L6I,
            4: uSIG.L6Q,
            5: uSIG.L6X,
            6: uSIG.L7I,
            7: uSIG.L7Q,
            8: uSIG.L7X,
            9: uSIG.L1D,
            10: uSIG.L1P,
            11: uSIG.L1X,
            12: uSIG.L5D,
            13: uSIG.L5P,
            14: uSIG.L5X,
        }

        qzs_tbl = {
            0: uSIG.L1C,
            1: uSIG.L1S,
            2: uSIG.L1C,
            3: uSIG.L2S,
            4: uSIG.L2L,
            5: uSIG.L2X,
            6: uSIG.L5I,
            7: uSIG.L5Q,
            8: uSIG.L5X,
            9: uSIG.L6S,
            10: uSIG.L6L,
            11: uSIG.L6X,
            12: uSIG.L1X,
            13: uSIG.L1Z,
            14: uSIG.L5D,
            15: uSIG.L5P,
            16: uSIG.L5Z,
            17: uSIG.L6E,
            18: uSIG.L6Z,
        }

        sbs_tbl = {
            0: uSIG.L1C,
            1: uSIG.L5I,
            2: uSIG.L5Q,
            3: uSIG.L5X,
        }

        usig_tbl_ = {
            uGNSS.GPS: gps_tbl,
            uGNSS.GLO: glo_tbl,
            uGNSS.GAL: gal_tbl,
            uGNSS.BDS: bds_tbl,
            uGNSS.QZS: qzs_tbl,
            uGNSS.SBS: sbs_tbl,
        }

        usig_tbl = usig_tbl_[sys]
        return rSigRnx(sys, utyp, usig_tbl[ssig])

    def msm2rsig(self, sys: uGNSS, utyp: uTYP, ssig):
        gps_tbl = {
            2: uSIG.L1C,
            3: uSIG.L1P,
            4: uSIG.L1W,
            8: uSIG.L2C,
            9: uSIG.L2P,
            10: uSIG.L2W,
            15: uSIG.L2S,
            16: uSIG.L2L,
            17: uSIG.L2X,
            22: uSIG.L5I,
            23: uSIG.L5Q,
            24: uSIG.L5X,
            30: uSIG.L1S,
            31: uSIG.L1L,
            32: uSIG.L1X,
        }
        glo_tbl = {
            2: uSIG.L1C,
            3: uSIG.L1P,
            8: uSIG.L2C,
            9: uSIG.L2P,
            10: uSIG.L4A,
            11: uSIG.L4B,
            12: uSIG.L4X,
            13: uSIG.L6A,
            13: uSIG.L6B,
            14: uSIG.L6X,
            15: uSIG.L3I,
            16: uSIG.L3Q,
            17: uSIG.L3X,
        }

        gal_tbl = {
            2: uSIG.L1C,
            3: uSIG.L1A,
            4: uSIG.L1B,
            5: uSIG.L1X,
            6: uSIG.L1Z,
            8: uSIG.L6C,
            9: uSIG.L6A,
            10: uSIG.L6B,
            11: uSIG.L6X,
            12: uSIG.L6Z,
            14: uSIG.L7I,
            15: uSIG.L7Q,
            16: uSIG.L7X,
            18: uSIG.L8I,
            19: uSIG.L8Q,
            20: uSIG.L8X,
            22: uSIG.L5I,
            23: uSIG.L5Q,
            24: uSIG.L5X,
        }

        bds_tbl = {
            2: uSIG.L2I,
            3: uSIG.L2Q,
            4: uSIG.L2X,
            8: uSIG.L6I,
            9: uSIG.L6Q,
            10: uSIG.L6X,
            14: uSIG.L7I,
            15: uSIG.L7Q,
            16: uSIG.L7X,
            22: uSIG.L5D,
            23: uSIG.L5P,
            24: uSIG.L5X,
            25: uSIG.L7D,
            30: uSIG.L1D,
            31: uSIG.L1P,
            32: uSIG.L1X,
        }

        qzs_tbl = {
            2: uSIG.L1C,
            9: uSIG.L6S,
            10: uSIG.L6L,
            11: uSIG.L6X,
            15: uSIG.L2S,
            16: uSIG.L2L,
            17: uSIG.L2X,
            22: uSIG.L5I,
            23: uSIG.L5Q,
            24: uSIG.L5X,
            30: uSIG.L1S,
            31: uSIG.L1L,
            32: uSIG.L1X,
        }

        sbs_tbl = {
            2: uSIG.L1C,
            22: uSIG.L5I,
            23: uSIG.L5Q,
            24: uSIG.L5X,
        }

        irn_tbl = {
            8: uSIG.L9A,
            22: uSIG.L5A,
        }

        usig_tbl_ = {
            uGNSS.GPS: gps_tbl,
            uGNSS.GLO: glo_tbl,
            uGNSS.GAL: gal_tbl,
            uGNSS.BDS: bds_tbl,
            uGNSS.QZS: qzs_tbl,
            uGNSS.SBS: sbs_tbl,
            uGNSS.IRN: irn_tbl,
        }

        usig_tbl = usig_tbl_[sys]
        return rSigRnx(sys, utyp, usig_tbl[ssig])

    def sys2str(self, sys: uGNSS):
        gnss_t = {uGNSS.GPS: "GPS", uGNSS.GLO: "GLO", uGNSS.GAL: "GAL",
                  uGNSS.BDS: "BDS", uGNSS.QZS: "QZS", uGNSS.SBS: "SBAS",
                  uGNSS.IRN: "NAVIC"}
        if sys not in gnss_t:
            return ""
        return gnss_t[sys]

    def sync(self, buff, k):
        return buff[k] == 0xd3

    def checksum(self, msg, k, maxlen=0):
        len_ = st.unpack_from('>H', msg, k+1)[0] & 0x3ff
        if len_ < 6:
            return False
        if maxlen > 0 and k+len_ >= maxlen:
            return False
        cs = Crc24LteA.calc(msg[k:k+len_+6])
        if self.monlevel > 0 and cs != 0:
            msgtype = (st.unpack_from('>H', msg, k+3)[0] >> 4) & 0xfff
            print(f"checksum error: len={len_} msgtype={msgtype}")
        self.len = len_
        self.dlen = len_+6
        return cs == 0

    def decode_head(self, msg, i, sys):
        if self.msgtype == 4076 or sys != uGNSS.GLO:
            blen = 20
        else:
            blen = 17
        self.tow = bs.unpack_from('u'+str(blen), msg, i)[0]
        self.time = gpst2time(self.week, self.tow)
        i += blen
        uint, mi = bs.unpack_from('u4u1', msg, i)
        i += 5

        if self.subtype in (sCSSR.ORBIT, sCSSR.COMBINED):
            self.datum = bs.unpack_from('u1', msg, i)[0]
            i += 1

        iodssr, pid, sid = bs.unpack_from('u4u16u4', msg, i)
        i += 24

        if self.subtype == sCSSR.PBIAS:
            ci, mw = bs.unpack_from('u1u1', msg, i)
            i += 2

        if self.subtype != sCSSR.VTEC:
            nsat = bs.unpack_from('u6', msg, i)[0]
            i += 6
        else:
            nsat = 0

        v = {'iodssr': iodssr, 'nsat': nsat}
        return i, v

    def decode_sat(self, msg, i, sys=uGNSS.NONE):
        if self.msgtype == 4076:
            blen = 6
        else:
            if sys == uGNSS.GLO:
                blen = 5
            elif sys == uGNSS.QZS:
                blen = 4
            else:
                blen = 6

        prn = bs.unpack_from('u'+str(blen), msg, i)[0]
        i += blen
        sat = prn2sat(sys, prn)
        return i, sat

    def decode_orb_sat(self, msg, i, k, sys=uGNSS.NONE, inet=0):
        if self.msgtype == 4076:
            blen = 8
        else:
            if sys == uGNSS.GAL:
                blen = 10
            elif sys == uGNSS.SBS:
                blen = 24
            else:
                blen = 8
        iode = bs.unpack_from('u'+str(blen), msg, i)[0]
        i += blen
        dx, dy, dz, ddx, ddy, ddz = bs.unpack_from('s22s20s20s21s19s19',
                                                   msg, i)
        i += 121

        self.iode_n[k] = iode
        self.dorb_n[k, 0] = self.sval(dx, 22, 0.1e-3)
        self.dorb_n[k, 1] = self.sval(dy, 20, 0.4e-3)
        self.dorb_n[k, 2] = self.sval(dz, 20, 0.4e-3)
        self.ddorb_n[k, 0] = self.sval(ddx, 21, 1e-6)
        self.ddorb_n[k, 1] = self.sval(ddy, 19, 4e-6)
        self.ddorb_n[k, 2] = self.sval(ddz, 19, 4e-6)

        return i

    def decode_clk_sat(self, msg, i, k, inet=0):
        """ decoder clock correction of cssr """
        dclk, ddclk, dddclk = bs.unpack_from('s22s21s27', msg, i)
        i += 70
        self.dclk_n[k] = self.sval(dclk, 22, 0.1e-3)
        # self.lc[inet].ddclk[k] = self.sval(ddclk, 21, 0.4e-3)
        # self.lc[inet].dddclk[k] = self.sval(dddclk, 27, 4e-6)
        return i

    def decode_hclk_sat(self, msg, i, k, inet=0):
        """ decoder clock correction of cssr """
        dclk, ddclk, dddclk = bs.unpack_from('s22', msg, i)[0]
        i += 22
        self.dclk_n[k] = self.sval(dclk, 22, 0.1e-3)
        return i

    def get_ssr_sys(self, msgtype):
        if msgtype == 4076:
            return self.sysref
        else:
            if msgtype >= 1057 and msgtype < 1063:
                return uGNSS.GPS
            elif msgtype >= 1063 and msgtype < 1069:
                return uGNSS.GLO
            elif msgtype >= 1240 and msgtype < 1246:
                return uGNSS.GAL
            elif msgtype >= 1246 and msgtype < 1252:
                return uGNSS.QZS
            elif msgtype >= 1252 and msgtype < 1258:
                return uGNSS.SBS
            elif msgtype >= 1258 and msgtype < 1264:
                return uGNSS.BDS

    def decode_cssr_orb(self, msg, i, inet=0):
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        self.iode_n = np.zeros(nsat, dtype=int)
        self.dorb_n = np.zeros((nsat, 3))
        self.ddorb_n = np.zeros((nsat, 3))

        if timediff(self.time, self.lc[inet].t0[sCType.ORBIT]) > 0:
            self.nsat_n = 0
            self.sys_n = []
            self.sat_n = []
            self.lc[0].iode = {}
            self.lc[0].dorb = {}

        self.iodssr = v['iodssr']
        sat = []
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_orb_sat(msg, i, k, sys)
            sat.append(sat_)
            self.lc[0].iode[sat_] = self.iode_n[k]
            self.lc[0].dorb[sat_] = self.dorb_n[k, :]

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.lc[0].cstat |= (1 << sCType.ORBIT)
        self.lc[0].t0[sCType.ORBIT] = self.time

        return i

    def decode_cssr_clk(self, msg, i, inet=0):
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0[sCType.CLOCK]) > 0:
            self.lc[0].dclk = {}

        self.dclk_n = np.zeros(nsat)
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_clk_sat(msg, i, k)
            self.lc[0].dclk[sat_] = self.dclk_n[k]

        self.lc[0].cstat |= (1 << sCType.CLOCK)
        self.lc[0].t0[sCType.CLOCK] = self.time

        return i

    def decode_cssr_cbias(self, msg, i, inet=0):
        """decode Code Bias Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0[sCType.CBIAS]) > 0:
            self.sat_b = []
            self.lc[0].cbias = {}

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5
            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
                self.lc[0].cbias[sat_] = {}

            for j in range(nsig):
                sig, cb = bs.unpack_from('u5s14', msg, i)
                i += 19

                self.lc[0].cbias[sat_][sig] = self.sval(cb, 14, 0.01)
                # if self.cssrmode == sCSSRTYPE.GAL_HAS_IDD:
                # work-around for HAS IDD
                #    self.lc[0].cbias[sat_][sig] *= -1.0

        self.lc[0].cstat |= (1 << sCType.CBIAS)
        self.lc[0].t0[sCType.CBIAS] = self.time
        return i

    def decode_cssr_pbias(self, msg, i, inet=0):
        """decode Phase Bias Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']

        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0[sCType.CBIAS]) > 0:
            self.sat_b = []
            self.lc[0].cbias = {}

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5
            yaw, dyaw = bs.unpack_from('u9s8', msg, i)
            yaw *= 1.0/256.0
            dyaw = self.sval(dyaw, 8, 1.0/8192.0)

            i += 17
            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
                self.lc[0].cbias[sat_] = {}

            for j in range(nsig):
                sig, si, wl, ci, pb = bs.unpack_from('u5u1u2u4s20', msg, i)
                i += 32

                self.lc[0].pbias[sat_][sig] = self.sval(pb, 20, 1e-4)
                # if self.cssrmode == sCSSRTYPE.GAL_HAS_IDD:
                # work-around for HAS IDD
                #    self.lc[0].pbias[sat_][sig] *= -1.0

        self.lc[0].cstat |= (1 << sCType.PBIAS)
        self.lc[0].t0[sCType.PBIAS] = self.time
        return i

    def decode_cssr_comb(self, msg, i, inet=0):
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        self.iode_n = np.zeros(nsat, dtype=int)
        self.dorb_n = np.zeros((nsat, 3))
        self.ddorb_n = np.zeros((nsat, 3))
        self.dclk_n = np.zeros(nsat)

        if timediff(self.time, self.lc[inet].t0[sCType.ORBIT]) > 0:
            self.nsat_n = 0
            self.sys_n = []
            self.sat_n = []
            self.lc[0].dclk = {}
            self.lc[0].iode = {}
            self.lc[0].dorb = {}

        self.iodssr = v['iodssr']
        sat = []
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_orb_sat(msg, i, k, sys)
            i = self.decode_clk_sat(msg, i, k)
            sat.append(sat_)
            self.lc[0].iode[sat_] = self.iode_n[k]
            self.lc[0].dorb[sat_] = self.dorb_n[k, :]
            self.lc[0].dclk[sat_] = self.dclk_n[k]

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.lc[0].cstat |= (1 << sCType.ORBIT)
        self.lc[0].t0[sCType.ORBIT] = self.time
        self.lc[0].cstat |= (1 << sCType.CLOCK)
        self.lc[0].t0[sCType.CLOCK] = self.time

        return i

    def decode_cssr_ura(self, msg, i):
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']

        if timediff(self.time, self.lc[0].t0[sCType.URA]) > 0:
            self.ura = np.zeros(self.nsat_n)

        for k in range(nsat):
            i, sat = self.decode_sat(msg, i, sys)
            s = self.sat_n.index(sat)
            cls_, val = bs.unpack_from_dict('u3u3', msg, i)
            self.ura[s] = self.quality_idx(cls_, val)

        self.lc[0].cstat |= (1 << sCType.URA)
        self.lc[0].t0[sCType.URA] = self.time

        return i

    def decode_cssr_hclk(self, msg, i, inet=0):
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0[sCType.HCLOCK]) > 0:
            self.lc[0].dclk = {}

        for k in range(v['nsat']):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_hclk_sat(msg, i, k)
            self.lc[0].dclk[sat_] = self.dclk_n[k]

        self.lc[0].cstat |= (1 << sCType.HCLOCK)
        self.lc[0].t0[sCType.HCLOCK] = self.time

        return i

    def decode_vtec(self, msg, i, inet=0):
        sys = uGNSS.NONE
        i, v = self.decode_head(msg, i, sys)
        # if self.iodssr != v['iodssr']:
        #    return -1

        qi, nlayer = bs.unpack_from('u9u2', msg, i)
        qi *= 0.05
        i += 11

        for k in range(nlayer):
            hl, n, m = bs.unpack_from('u8u4u4', msg, i)
            i += 16
            c = np.zeros((n+1, m+1))
            s = np.zeros((n+1, m+1))
            for j in range(m+1):
                for l_ in range(j, n+1):
                    c_ = bs.unpack_from('s16', msg, i)
                    i += 16
                    c[l_, j] = self.sval(c_, 16, 5e-3)

            for j in range(1, m+1):
                for l_ in range(j, n+1):
                    s_ = bs.unpack_from('s16', msg, i)
                    i += 16
                    s[l_, j] = self.sval(s_, 16, 5e-3)

        self.lc[0].cstat |= (1 << sCType.VTEC)
        self.lc[0].t0[sCType.VTEC] = self.time

        return i

    def decode_igsssr(self, msg, i=0):
        sys_t = {2: uGNSS.GPS,  4: uGNSS.GLO,  6: uGNSS.GAL,
                 8: uGNSS.QZS, 10: uGNSS.BDS, 12: uGNSS.SBS}
        ver, subtype = bs.unpack_from('u3u8', msg, i)
        i += 11

        if self.monlevel > 0 and self.fh is not None:
            self.fh.write("##### IGS SSR subtype: {:d}\n".format(subtype))

        if subtype == 201:
            self.subtype = sCSSR.VTEC
            i = self.decode_vtec(msg, i)
        else:
            self.sysref = sys_t[subtype // 10]
            st = subtype % 10

            if st == 1:
                self.subtype = sCSSR.ORBIT
                i = self.decode_cssr_orb(msg, i)
            elif st == 2:
                self.subtype = sCSSR.CLOCK
                i = self.decode_cssr_clk(msg, i)
            elif st == 3:
                self.subtype = sCSSR.COMBINED
                i = self.decode_cssr_comb(msg, i)
            elif st == 4:
                self.subtype = sCSSR.CLOCK
                i = self.decode_cssr_hclk(msg, i)
            elif st == 5:
                self.subtype = sCSSR.CBIAS
                i = self.decode_cssr_cbias(msg, i)
            elif st == 6:
                self.subtype = sCSSR.PBIAS
                i = self.decode_cssr_pbias(msg, i)
            elif st == 7:
                self.subtype = sCSSR.URA
                i = self.decode_cssr_ura(msg, i)

        return i

    def nrtktype(self, msgtype):
        gnss_t = {1030: uGNSS.GPS, 1031: uGNSS.GLO,
                  36: uGNSS.BDS, 37: uGNSS.GAL, 38: uGNSS.QZS}

        sys = uGNSS.NONE
        nrtk = 0
        if msgtype in gnss_t.keys():
            nrtk = 1
            sys = gnss_t[msgtype]

        return sys, nrtk

    def decode_nrtk_time(self, msg, i):
        sys, nrtk = self.nrtktype(self.msgtype)

        sz = 20 if sys != uGNSS.GLO else 17
        tow = bs.unpack_from('u'+str(sz), msg, i)[0]
        i += sz

        if sys == uGNSS.BDS:
            week, _ = time2bdt(gpst2bdt(gpst2time(self.week, tow)))
            time = bdt2gpst(bdt2time(week, tow))
        elif sys == uGNSS.GLO:
            time = glo2time(self.time, tow)
        else:
            time = gpst2time(self.week, tow)
        return i, sys, time

    def decode_nrtk_residual(self, msg, i=0):
        i, sys, time = self.decode_nrtk_time(msg, i)

        self.refid, self.nrefs, self.nsat = bs.unpack_from('u12u7u5', msg, i)
        i += 24

        for k in range(self.nsat):
            sz = 6 if sys != uGNSS.QZS else 4
            prn = bs.unpack_from('u'+str(sz), msg, i)[0]
            if sys == uGNSS.QZS:
                prn += 192
            sat = prn2sat(sys, prn)
            i += sz
            s0c, s0d, s0h, sic, sid = bs.unpack_from('u8u9u6u10u10', msg, i)
            i += 43
            self.nrtk_r[sat] = np.array([s0c*5e-4, s0d*1e-8, s0h*1e-7,
                                         sic*5e-4, sid*1e-8])

        return i

    def decode_time(self, msg):
        i = 24
        self.msgtype = bs.unpack_from('u12', msg, i)[0]
        i += 12

        sys, msm = self.msmtype(self.msgtype)
        if msm > 0:
            tow_ = bs.unpack_from('u30', msg, i+12)[0]
            time, tow = self.decode_msm_time(sys, self.week, tow_)
            return time

        sys, ssr = self.ssrtype(self.msgtype)
        if ssr > 0 or self.msgtype == 4076:
            sz = 20 if self.msgtype == 4076 or sys != uGNSS.GLO else 17
            self.tow = bs.unpack_from('u'+str(sz), msg, i)[0]
            return gpst2time(self.week, self.tow)

        sys, nrtk = self.nrtktype(self.msgtype)
        if nrtk > 0:
            i, sys, time = self.decode_nrtk_time(msg, i)
            return time

        return False

    def out_log(self, obs=None, eph=None, geph=None, seph=None):
        sys = self.get_ssr_sys(self.msgtype)
        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))

        if self.subtype == sCSSR.CLOCK:
            self.fh.write(" {:s}\t{:s}\n".format("SatID", "dclk [m]"))
            for k, sat_ in enumerate(self.sat_n):
                sys_, _ = sat2prn(sat_)
                if sys != uGNSS.NONE and sys_ != sys:
                    continue
                self.fh.write(" {:s}\t{:5.3f}\n".format(sat2id(sat_),
                                                        self.lc[0].dclk[sat_]))

        elif self.subtype == sCSSR.ORBIT:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "IODE", "Radial[m]",
                                  "Along[m]", "Cross[m]"))
            for k, sat_ in enumerate(self.sat_n):
                sys_, _ = sat2prn(sat_)
                if sys != uGNSS.NONE and sys_ != sys:
                    continue
                self.fh.write(" {:s}\t{:3d}\t{:5.3f}\t{:5.3f}\t{:5.3f}\n".
                              format(sat2id(sat_),
                                     self.lc[0].iode[sat_],
                                     self.lc[0].dorb[sat_][0],
                                     self.lc[0].dorb[sat_][1],
                                     self.lc[0].dorb[sat_][2]))

        elif self.subtype == sCSSR.COMBINED:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "IODE", "Radial[m]",
                                  "Along[m]", "Cross[m]", "dclk[m]"))
            for k, sat_ in enumerate(self.sat_n):
                sys_, _ = sat2prn(sat_)
                if sys != uGNSS.NONE and sys_ != sys:
                    continue
                self.fh.write(
                    " {:s}\t{:3d}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:6.3f}\n".
                    format(sat2id(sat_),
                           self.lc[0].iode[sat_],
                           self.lc[0].dorb[sat_][0],
                           self.lc[0].dorb[sat_][1],
                           self.lc[0].dorb[sat_][2],
                           self.lc[0].dclk[sat_]))

        elif self.subtype == sCSSR.CBIAS:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "Bias[m]", "..."))
            for k, sat_ in enumerate(self.sat_b):
                sys_, _ = sat2prn(sat_)
                if sys != uGNSS.NONE and sys_ != sys:
                    continue
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for j, sig in enumerate(self.lc[0].cbias[sat_].keys()):
                    sig_ = self.ssig2rsig(sys, uTYP.C, sig)
                    self.fh.write(
                        "{:s}\t{:5.2f}\t".format(sig_.str(),
                                                 self.lc[0].cbias[sat_][sig]))
                self.fh.write("\n")

        elif self.subtype == sRTCM.ANT_DESC:
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            self.fh.write(" {:20s}{:s}\n".format("Antenna Descriptor]:",
                                                 self.ant_desc))
            self.fh.write(" {:20s}{:6d}\n".format("Antenna Setup ID:",
                                                  self.ant_id))
            if self.msgtype == 1008 or self.msgtype == 1033:
                self.fh.write(" {:20s}{:s}\n".format("Antenna Serial:",
                                                     self.ant_serial))
            if self.msgtype == 1033:
                self.fh.write(" {:20s}{:s}\n".format("Receiver Type:",
                                                     self.rcv_type))
                self.fh.write(" {:20s}{:s}\n".format("Firmware Version:",
                                                     self.firm_ver))
                self.fh.write(" {:20s}{:s}\n".format("Receiver Serial:",
                                                     self.rcv_serial))

        elif self.subtype == sRTCM.NRTK_RES:
            sys, nrtk = self.nrtktype(self.msgtype)
            self.fh.write(" {:20s}{:6d} ({:s})\n".format("NRTK Residual:",
                                                         self.msgtype,
                                                         self.sys2str(sys)))
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            self.fh.write(" {:20s}{:6.1f}\n".format("GNSS Time of Week [s]:",
                                                    self.tow))
            self.fh.write(" {:20s}{:6d}\n".format("Number of stations:",
                                                  self.nrefs))
            self.fh.write(" {:20s}{:5d}\n".format("Number of satellites:",
                                                  self.nsat))

            self.fh.write(" PRN  s0c[mm] s0d[ppm] s0h[ppm]  sic[mm] sid[ppm]\n")
            sat_ = self.nrtk_r.keys()
            for sat in sat_:
                sys_, _ = sat2prn(sat)
                if sys_ != sys:
                    continue
                v = self.nrtk_r[sat]
                self.fh.write(" {:s}".format(sat2id(sat)))
                self.fh.write("  {:7.1f} {:8.2f} {:8.1f} {:8.1f} {:8.2f}\n"
                              .format(v[0]*1e3, v[1], v[2], v[3]*1e3, v[4]))

        elif self.subtype == sRTCM.ANT_POS:
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            if self.msgtype in (1005, 1006):
                self.fh.write(" {:20s}{:6d}\n".format("Station Indicator:",
                                                      self.sti))
            self.fh.write(" {:20s} {:8.4f} {:8.4f} {:8.4f}\n"
                          .format("Antenna Position [m]:",
                                  self.pos_arp[0], self.pos_arp[1],
                                  self.pos_arp[2]))

        elif self.subtype == sRTCM.GLO_BIAS:
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            self.fh.write(" {:20s} {:8.2f}\n"
                          .format("GLONASS BIAS L1 C/A [m]:",
                                  self.glo_bias[0]))
            self.fh.write(" {:20s} {:8.2f}\n"
                          .format("GLONASS BIAS L1 P   [m]:",
                                  self.glo_bias[1]))
            self.fh.write(" {:20s} {:8.2f}\n"
                          .format("GLONASS BIAS L2 C/A [m]:",
                                  self.glo_bias[2]))
            self.fh.write(" {:20s} {:8.2f}\n"
                          .format("GLONASS BIAS L2 P   [m]:",
                                  self.glo_bias[3]))

        elif self.subtype == sRTCM.MSM:
            sys, msm = self.msmtype(self.msgtype)
            self.fh.write(" {:20s}{:6d} ({:s})\n".format("MSM:", msm,
                                                         self.sys2str(sys)))
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            self.fh.write(" {:20s}{:6.1f}\n".format("GNSS Time of Week [s]:",
                                                    self.tow))
            self.fh.write(" {:20s}{:6d}\n".format("IOD Station:",
                                                  self.iods))
            self.fh.write(" {:20s}{:5d}\n".format("Number of Satellites:",
                                                  self.nsat))
            self.fh.write(" {:20s}{:6d}\n".format("Number of Signals:",
                                                  self.nsig))

        if eph is not None:
            sys, prn = sat2prn(eph.sat)
            self.fh.write(" {:20s}{:6d} ({:s})\n".format("PRN:", prn,
                                                         self.sys2str(sys)))
            self.fh.write(" {:20s}{:6d}\n".format("IODE:", eph.iode))
            self.fh.write(" {:20s}{:6d}\n".format("MODE:", eph.mode))

        if geph is not None:
            sys, prn = sat2prn(geph.sat)
            self.fh.write(" {:20s}{:6d} ({:s})\n".format("PRN:", prn,
                                                         self.sys2str(sys)))
            self.fh.write(" {:20s}{:6d}\n".format("IODE:", geph.iode))
            self.fh.write(" {:20s}{:6d}\n".format("MODE:", geph.mode))

        if seph is not None:
            sys, prn = sat2prn(seph.sat)
            self.fh.write(" {:20s}{:6d} ({:s})\n".format("PRN:", prn,
                                                         self.sys2str(sys)))
            self.fh.write(" {:20s}{:6d}\n".format("IODN:", seph.iodn))
            self.fh.write(" {:20s}{:6d}\n".format("MODE:", seph.mode))

    def decode_msm_time(self, sys, week, t):
        if sys == uGNSS.GLO:
            dow = (t >> 27) & 0x1f
            tow = (t & 0x7ffffff)*1e-3
            time = gpst2time(week, tow+dow*86400.0)
            time = utc2gpst(timeadd(time, -10800.0))
        else:
            tow = t*1e-3
            time = gpst2time(week, tow)
        return time, tow

    def decode_msm(self, msg, i):
        sys, msm = self.msmtype(self.msgtype)

        self.refid, tow_, self.mi, self.iods = \
            bs.unpack_from('u12u30u1u3', msg, i)
        i += 53
        self.time, self.tow = self.decode_msm_time(sys, self.week, tow_)

        csi, eci, si, smi = bs.unpack_from('u2u2u1u3', msg, i)
        i += 8
        svmask, sigmask = bs.unpack_from('u64u32', msg, i)
        i += 96

        ofst = 193 if sys == uGNSS.QZS else 1
        prn, self.nsat = self.decode_mask(svmask, 64, ofst)
        sig, self.nsig = self.decode_mask(sigmask, 32)
        sz = self.nsat*self.nsig

        if sz > 64:
            return -1
        ncell = 0
        self.sig_n = []
        for k in range(self.nsat):
            cellmask = bs.unpack_from('u'+str(self.nsig), msg, i)[0]
            i += self.nsig
            sig_, nsig_ = self.decode_mask(cellmask, self.nsig, 0)
            sig_n = [sig[i] for i in sig_]
            self.sig_n += [sig_n]
            ncell += nsig_

        r = np.zeros(self.nsat)
        pr = np.ones(ncell)*np.nan
        cp = np.ones(ncell)*np.nan
        lock = np.zeros(ncell, dtype=int)
        half = np.zeros(ncell, dtype=int)
        cnr = np.zeros(ncell)
        rrf = np.zeros(ncell)

        ex = np.zeros(self.nsat, dtype=int)
        rr = np.zeros(self.nsat)

        rms = rCST.CLIGHT*1e-3
        P2_10 = 0.0009765625
        P2_24 = 5.960464477539063E-08
        P2_29 = 1.862645149230957E-09
        P2_31 = 4.656612873077393E-10

        # satellite part
        if msm >= 1 and msm <= 3:
            for k in range(self.nsat):
                v = bs.unpack_from('u10', msg, i)[0]
                if v != 2**10-1:
                    r[k] = v*rms
                i += 10
        else:
            for k in range(self.nsat):
                v = bs.unpack_from('u8', msg, i)[0]
                if v != 2**8-1:
                    r[k] = v*rms
                i += 8
            if msm == 5 or msm == 7:
                for k in range(self.nsat):
                    ex[k] = bs.unpack_from('u4', msg, i)[0]
                    i += 4
            for k in range(self.nsat):
                if r[k] != 0.0:
                    r[k] += bs.unpack_from('u10', msg, i)[0]*P2_10*rms
                i += 10
            if msm == 5 or msm == 7:
                for k in range(self.nsat):
                    v = bs.unpack_from('s14', msg, i)[0]
                    i += 14
                    rr[k] = self.sval(v, 14, 1.0)

        # signal part
        if msm != 2:
            sz = 15 if msm < 6 else 20
            scl = P2_24 if msm < 6 else P2_29
            for k in range(ncell):
                pr_ = bs.unpack_from('s'+str(sz), msg, i)[0]
                i += sz
                pr[k] = self.sval(pr_, sz, scl*rms)

        if msm > 1:
            sz = 22 if msm < 6 else 24
            scl = P2_29 if msm < 6 else P2_31
            for k in range(ncell):
                cp_ = bs.unpack_from('s'+str(sz), msg, i)[0]
                i += sz
                cp[k] = self.sval(cp_, sz, scl*rms)

            sz = 4 if msm < 6 else 10
            for k in range(ncell):
                lock[k] = bs.unpack_from('u'+str(sz), msg, i)[0]
                i += sz

            for k in range(ncell):
                half[k] = bs.unpack_from('u1', msg, i)[0]
                i += 1

        if msm > 3:
            sz = 6 if msm < 6 else 10
            scl = 1.0 if msm < 6 else 0.0625
            for k in range(ncell):
                cnr[k] = bs.unpack_from('u'+str(sz), msg, i)[0]*scl
                i += sz

        if msm == 5 or msm == 7:
            for k in range(ncell):
                v = bs.unpack_from('s15', msg, i)[0]
                i += 15
                rrf[k] = self.sval(v, 15, 1e-4)

        obs = Obs()
        obs.t = self.time

        obs.P = np.empty((0, self.nsig), dtype=np.float64)
        obs.L = np.empty((0, self.nsig), dtype=np.float64)
        obs.S = np.empty((0, self.nsig), dtype=np.float64)
        obs.lli = np.empty((0, self.nsig), dtype=np.int32)
        obs.sat = np.empty(0, dtype=np.int32)

        obs.sig = {}
        obs.sig[sys] = {}
        obs.sig[sys][uTYP.C] = []
        obs.sig[sys][uTYP.L] = []
        obs.sig[sys][uTYP.S] = []
        for sig_ in sig:
            obs.sig[sys][uTYP.C].append(self.msm2rsig(sys, uTYP.C, sig_))
            obs.sig[sys][uTYP.L].append(self.msm2rsig(sys, uTYP.L, sig_))
            obs.sig[sys][uTYP.S].append(self.msm2rsig(sys, uTYP.S, sig_))

        ofst = 0
        for k in range(self.nsat):
            sat_ = prn2sat(sys, prn[k])
            nsig_ = len(self.sig_n[k])

            pr_ = np.zeros(self.nsig, dtype=np.float64)
            cp_ = np.zeros(self.nsig, dtype=np.float64)
            ll_ = np.zeros(self.nsig, dtype=np.int32)
            cn_ = np.zeros(self.nsig, dtype=np.float64)

            for j, sig_ in enumerate(self.sig_n[k]):
                idx = sig.index(sig_)
                pr_[idx] = pr[j+ofst]+r[k]
                cp_[idx] = cp[j+ofst]+r[k]
                cn_[idx] = cnr[j+ofst]
                ll_[idx] = lock[j+ofst]
            ofst += nsig_

            obs.P = np.append(obs.P, pr_)
            obs.L = np.append(obs.L, cp_)
            obs.S = np.append(obs.S, cn_)
            obs.lli = np.append(obs.lli, ll_)
            obs.sat = np.append(obs.sat, sat_)

        return i, obs

    def decode_ant_desc(self, msg, i):
        """ Antenna Description Message """
        self.refid, nc = bs.unpack_from('u12u8', msg, i)
        i += 20
        j = 7
        self.ant_desc = msg[j:j+nc].decode()
        i += 8*nc
        self.ant_id = bs.unpack_from('u8', msg, i)[0]
        i += 8
        j += nc+1
        if self.msgtype == 1008 or self.msgtype == 1033:
            nc = bs.unpack_from('u8', msg, i)[0]
            i += 8
            j += 1
            self.ant_serial = msg[j:j+nc].decode()
            i += 8*nc
            j += nc
        if self.msgtype == 1033:
            nc = bs.unpack_from('u8', msg, i)[0]
            i += 8
            j += 1
            self.rcv_type = msg[j:j+nc].decode()
            i += 8*nc
            j += nc
            nc = bs.unpack_from('u8', msg, i)[0]
            i += 8
            j += 1
            self.firm_ver = msg[j:j+nc].decode()
            i += 8*nc
            j += nc
            nc = bs.unpack_from('u8', msg, i)[0]
            i += 8
            j += 1
            self.rcv_serial = msg[j:j+nc].decode()
            i += 8*nc
            j += nc

        return i

    def decode_sta_pos(self, msg, i):
        """ Physical Reference Station Position Message """
        if self.msgtype == 1032:
            self.np_staid = bs.unpack_from('u12', msg, i)[0]
            i += 12
        self.refid = bs.unpack_from('u12', msg, i)[0]
        i += 18
        if self.msgtype == 1005 or self.msgtype == 1006:
            ind = bs.unpack_from('u4', msg, i)[0]
            self.sti = ind & 1  # 0: physical, 1: non-physical
            i += 4
        xp = bs.unpack_from('s38', msg, i)[0]
        i += 38
        if self.msgtype == 1005 or self.msgtype == 1006:
            # oi = bs.unpack_from('u1', msg, i)[0]
            i += 2
        yp = bs.unpack_from('s38', msg, i)[0]
        i += 38
        if self.msgtype == 1005 or self.msgtype == 1006:
            # qci = bs.unpack_from('u2', msg, i)[0]
            i += 2
        zp = bs.unpack_from('s38', msg, i)[0]
        i += 38
        self.pos_arp = np.array([xp*1e-4, yp*1e-4, zp*1e-4])
        if self.msgtype == 1006:
            self.ant_height = bs.unpack_from('u16', msg, i)[0]*1e-4
            i += 16
        return i

    def decode_glo_bias(self, msg, i):
        """ GLONASS Bias Information Message """
        self.refid, cb, _, sigmask = bs.unpack_from('u12u1u3u4', msg, i)
        i += 20
        self.glo_bias = np.zeros(4)
        for k in range(4):
            if (sigmask >> (3-k)) & 1 == 1:
                self.glo_bias[k] = bs.unpack_from('s16', msg, i)[0]*0.02
                i += 16

        return i

    def decode_gps_eph(self, msg, i):
        """ GPS Satellite Ephemeris Message """
        prn, week, sva, code, idot = bs.unpack_from('u6u10s22s16s8', msg, i)
        i += 6+10+4+2+14
        iode, toc, af2, af1, af0 = bs.unpack_from('u8u16s8s16s22', msg, i)
        i += 8+16+8+16+22
        iodc, crs, deln, M0, cuc = bs.unpack_from('u10s16s16s32s16', msg, i)
        i += 10+16+16+32+16
        e, cus, Asq, toe, cic = bs.unpack_from('u32s16u32u16s16', msg, i)
        i += 32+16+32+16+16
        Omg0, cis, i0, crc, omg = bs.unpack_from('s32s16s32s16s32', msg, i)
        i += 32+16+32+16+32
        OMGd, tgd, svh, flag, fit = bs.unpack_from('s24s8u6u1u1', msg, i)
        i += 40

        eph = Eph()
        eph.sat = prn2sat(uGNSS.GPS, prn)
        eph.week = week
        eph.sva = sva
        eph.code = code
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.iode = iode
        toc *= 16.0
        eph.af2 = af0*rCST.P2_55
        eph.af1 = af1*rCST.P2_43
        eph.af0 = af0*rCST.P2_31
        eph.iodc = iodc
        eph.crs = crs*rCST.P2_5
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_29
        eph.e = e*rCST.P2_33
        eph.cus = cus*rCST.P2_29
        sqrtA = Asq*rCST.P2_19
        eph.toes = toe*60.0
        eph.cic = cic*rCST.P2_29
        eph.OMG0 = Omg0*rCST.P2_31*rCST.SC2RAD
        eph.cis = cis*rCST.P2_29
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.crc = crc*rCST.P2_5
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.tgd = tgd*rCST.P2_31
        eph.svh = svh
        eph.flag = flag
        eph.fit = fit

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.ttr = self.time
        eph.A = sqrtA*sqrtA
        eph.mode = 0

        return i, eph

    def decode_glo_eph(self, msg, i):
        """ GLONASS Satellite Ephemeris Message """
        prn, frq, cn, svh, P1 = bs.unpack_from('u6u5u1u1u2', msg, i)
        i += 15
        tk_h, tk_m, tk_s = bs.unpack_from('u5u6u1', msg, i)
        i += 12
        bn, P2, tb = bs.unpack_from('u1u1u7', msg, i)
        i += 9
        xd, x, xdd = bs.unpack_from('s24s27s5', msg, i)
        i += 56
        yd, y, ydd = bs.unpack_from('s24s27s5', msg, i)
        i += 56
        zd, z, zdd = bs.unpack_from('s24s27s5', msg, i)
        i += 56
        P3, gamn, P, ln, taun, dtaun = bs.unpack_from('u1s11u2u1s22s5', msg, i)
        i += 42
        En, P4, Ft, Nt, M, ad, Na, tauc, N4, taugps, ln = \
            bs.unpack_from('u5u1u4u11u2u1u11s32u5s22u1', msg, i)
        i += 102

        geph = Geph()
        geph.frq = frq-7
        tk_s *= 30.0
        geph.vel[0] = xd*rCST.P2_20*1e3
        geph.pos[0] = x*rCST.P2_11*1e3
        geph.acc[0] = xdd*rCST.P2_30*1e3
        geph.vel[1] = xd*rCST.P2_20*1e3
        geph.pos[1] = x*rCST.P2_11*1e3
        geph.acc[1] = xdd*rCST.P2_30*1e3
        geph.vel[2] = xd*rCST.P2_20*1e3
        geph.pos[2] = x*rCST.P2_11*1e3
        geph.acc[2] = xdd*rCST.P2_30*1e3
        geph.gamn = gamn*rCST.P2_40
        geph.taun = taun*rCST.P2_30
        geph.dtaun = dtaun*rCST.P2_30
        geph.age = En

        geph.sat = prn2sat(uGNSS.GLO, prn)
        geph.svh = bn
        geph.iode = tb & 0x7F

        week, tow = time2gpst(self.time)
        tod = tow % 86400.0
        tow -= tod
        tof = tk_h*3600.0+tk_m*60.0+tk_s-10800.0
        if tof < tod-43200.0:
            tof += 86400.0
        elif tof > tod+43200.0:
            tof -= 86400.0
        geph.tof = utc2gpst(gpst2time(week, tow+tof))
        toe = tb*900.0-10800.0
        if toe < tod-43200.0:
            toe += 86400.0
        elif toe > tod+43200.0:
            toe -= 86400.0
        geph.toe = utc2gpst(gpst2time(week, tow+toe))
        geph.mode = 0

        return i, geph

    def decode_gal_eph(self, msg, i):
        """ Galileo Satellite Ephemeris Message """
        prn, week, iodnav, sisa, idot = bs.unpack_from('u6u12u10u8s14', msg, i)
        i += 6+12+10+8+14
        toc, af2, af1, af0 = bs.unpack_from('u14s6s21s31', msg, i)
        i += 14+6+21+31
        crs, deln, M0, cuc = bs.unpack_from('s16s16s32s16', msg, i)
        i += 16+16+32+16
        e, cus, Asq, toe, cic = bs.unpack_from('u32s16u32u14s16', msg, i)
        i += 32+16+32+14+16
        Omg0, cis, i0, crc, omg = bs.unpack_from('s32s16s32s16s32', msg, i)
        i += 32+16+32+16+32
        OMGd, tgd = bs.unpack_from('s24s10', msg, i)
        i += 24+10
        if self.msgtype == 1045:  # F/NAV
            hs, dvs = bs.unpack_from('u2u1', msg, i)
            i += 2+1+7
        else:  # I/NAV
            tgd2, hs, dvs, hs1, dvs1 = bs.unpack_from('s10u2u1u2u1', msg, i)
            i += 18

        eph = Eph()
        eph.sat = prn2sat(uGNSS.GAL, prn)
        eph.week = week
        eph.iode = iodnav
        eph.sva = sisa
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        toc *= 60.0

        eph.af2 = af0*rCST.P2_59
        eph.af1 = af1*rCST.P2_46
        eph.af0 = af0*rCST.P2_34
        eph.crs = crs*rCST.P2_5
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_29
        eph.e = e*rCST.P2_33
        eph.cus = cus*rCST.P2_29
        sqrtA = Asq*rCST.P2_19
        eph.toes = toe*60.0
        eph.cic = cic*rCST.P2_29
        eph.OMG0 = Omg0*rCST.P2_31*rCST.SC2RAD
        eph.cis = cis*rCST.P2_29
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.crc = crc*rCST.P2_5
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.tgd = tgd*rCST.P2_32
        if self.msgtype == 1046:
            eph.tgd_b = tgd2*rCST.P2_32

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.ttr = self.time
        eph.A = sqrtA*sqrtA
        eph.svh = (hs << 7)+(dvs << 6)
        if self.msgtype == 1046:
            eph.svh |= (hs1 << 1)+(dvs1)
        eph.code = (1 << 0)+(1 << 2)+(1 << 9)
        eph.iodc = eph.iode

        eph.mode = 0 if self.msgtype == 1046 else 1

        return i, eph

    def decode_qzs_eph(self, msg, i):
        """ QZS Satellite Ephemeris Message """
        prn, toc, af2, af1, af0 = bs.unpack_from('u4u16s8s16s22', msg, i)
        i += 4+16+8+16+22
        iode, crs, deln, M0, cuc = bs.unpack_from('u8s16s16s32s16', msg, i)
        i += 8+16+16+32+16
        e, cus, Asq, toe, cic = bs.unpack_from('u32s16u32u16s16', msg, i)
        i += 32+16+32+16+16
        Omg0, cis, i0, crc, omg = bs.unpack_from('s32s16s32s16s32', msg, i)
        i += 32+16+32+16+32
        OMGd, idot, l2c, week, sva = bs.unpack_from('s24s14u2u10u4', msg, i)
        i += 24+14+2+10+4
        svh, tgd, iodc, fit = bs.unpack_from('u6s8u10u1', msg, i)
        i += 6+8+10+1

        eph = Eph()
        eph.sat = prn2sat(uGNSS.QZS, prn+192)
        toc *= 16.0
        eph.af2 = af0*rCST.P2_55
        eph.af1 = af1*rCST.P2_43
        eph.af0 = af0*rCST.P2_31
        eph.iode = iode
        eph.crs = crs*rCST.P2_5
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_29
        eph.e = e*rCST.P2_33
        eph.cus = cus*rCST.P2_29
        sqrtA = Asq*rCST.P2_19
        eph.toes = toe*16.0
        eph.cic = cic*rCST.P2_29
        eph.OMG0 = Omg0*rCST.P2_31*rCST.SC2RAD
        eph.cis = cis*rCST.P2_29
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.crc = crc*rCST.P2_5
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.code = l2c

        eph.week = week
        eph.sva = sva
        eph.svh = svh
        eph.tgd = tgd*self.P2_31
        eph.iodc = iodc
        eph.fit = fit

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.ttr = self.time
        eph.A = sqrtA*sqrtA
        eph.flag = 1
        eph.mode = 0

        return i, eph

    def decode_bds_eph(self, msg, i):
        """ BDS Satellite Ephemeris Message """
        prn, week, ura, idot = bs.unpack_from('u6u13u4s14', msg, i)
        i += 6+13+4+14
        aode, toc, af2, af1, af0 = bs.unpack_from('u5u17s11s22s24', msg, i)
        i += 5+17+11+22+24
        aodc, crs, deln, M0, cuc = bs.unpack_from('u5s18s16s32s18', msg, i)
        i += 5+18+16+32+18
        e, cus, Asq, toe, cic = bs.unpack_from('u32s16u32u17s18', msg, i)
        i += 32+16+32+17+18
        Omg0, cis, i0, crc, omg = bs.unpack_from('s32s18s32s18s32', msg, i)
        i += 32+18+32+18+32
        OMGd, tgd1, tgd2, svh = bs.unpack_from('s24s10s10u1', msg, i)
        i += 45

        eph = Eph()
        eph.sat = prn2sat(uGNSS.BDS, prn)
        eph.week = week
        eph.sva = ura
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.iode = aode
        toc *= 8.0
        eph.af2 = af0*rCST.P2_66
        eph.af1 = af1*rCST.P2_50
        eph.af0 = af0*rCST.P2_33
        eph.iodc = aodc
        eph.crs = crs*rCST.P2_6
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_31
        eph.e = e*rCST.P2_33
        eph.cus = cus*rCST.P2_31
        sqrtA = Asq*rCST.P2_19
        eph.toes = toe*8.0
        eph.cic = cic*rCST.P2_31
        eph.OMG0 = Omg0*rCST.P2_31*rCST.SC2RAD
        eph.cis = cis*rCST.P2_31
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.crc = crc*rCST.P2_6
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.tgd = tgd1*1e-10
        eph.tgd_b = tgd2*1e-10
        eph.svh = svh

        eph.toe = bdt2gpst(bdt2time(eph.week, eph.toes))
        eph.toc = bdt2gpst(bdt2time(eph.week, toc))
        eph.ttr = self.time
        eph.A = sqrtA*sqrtA
        eph.mode = 0

        return i, eph

    def decode_irn_eph(self, msg, i):
        """ NavIC Satellite Ephemeris Message """
        prn, week, af0, af1, af2 = bs.unpack_from('u6u10s22s16s8', msg, i)
        i += 6+10+22+16+8
        ura, toc, tgd, deln, iode = bs.unpack_from('u4u16s8s22u8', msg, i)
        i += 4+16+8+22+8+10
        svh, cuc, cus, cic = bs.unpack_from('u2s15s15s15', msg, i)
        i += 32+15
        cis, crc, crs, idot, M0 = bs.unpack_from('s15s15s15s14s32', msg, i)
        i += 59+32
        toe, e, Asq, Omg0, omg = bs.unpack_from('u16u32u32s32s32', msg, i)
        i += 80+64
        OMGd, i0 = bs.unpack_from('s22s32', msg, i)
        i += 22+32+2+2

        eph = Eph()
        eph.sat = prn2sat(uGNSS.IRN, prn)
        eph.week = week
        eph.af0 = af0*rCST.P2_31
        eph.af1 = af1*rCST.P2_43
        eph.af2 = af0*rCST.P2_55
        eph.sva = ura
        toc *= 16.0
        eph.tgd = tgd*rCST.P2_31
        eph.deln = deln*rCST.P2_41*rCST.SC2RAD
        eph.iode = iode
        eph.svh = svh
        eph.cuc = cuc*rCST.P2_28
        eph.cus = cus*rCST.P2_28
        eph.cic = cic*rCST.P2_28
        eph.cis = cis*rCST.P2_28
        eph.crc = crc*0.0625
        eph.crs = crs*0.0625
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.toes = toe*16.0
        eph.e = e*rCST.P2_33
        sqrtA = Asq*rCST.P2_19
        eph.OMG0 = Omg0*rCST.P2_31*rCST.SC2RAD
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_41*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.ttr = self.time
        eph.A = sqrtA*sqrtA
        eph.iodc = eph.iode
        eph.mode = 0

        return i, eph

    def decode_sbs_eph(self, msg, i):
        """ SBAS Satellite Ephemeris Message """
        prn, iodn, toc, ura = bs.unpack_from('u6u8u13u4', msg, i)
        i += 6+8+13+4
        x, y, z = bs.unpack_from('s30s30s25', msg, i)
        i += 30+30+25
        vx, vy, vz = bs.unpack_from('s17s17s18', msg, i)
        i += 17+17+18
        ax, ay, az = bs.unpack_from('s10s10s10', msg, i)
        i += 10+10+10
        af0, af1 = bs.unpack_from('s12s8', msg, i)
        i += 12+8

        seph = Seph()
        seph.sat = prn2sat(uGNSS.SBS, prn+119)
        seph.iode = iodn
        toc *= 16.0
        seph.ura = ura
        seph.pos[0] = x*0.08
        seph.pos[1] = y*0.08
        seph.pos[2] = z*0.4
        seph.vel[0] = vx*0.000625
        seph.vel[1] = vy*0.000625
        seph.vel[2] = vz*0.004
        seph.acc[0] = ax*0.0000125
        seph.acc[1] = ay*0.0000125
        seph.acc[2] = az*0.0000625
        seph.af0 = af0*self.P2_31
        seph.af1 = af1*self.P2_40
        week, tow = time2gpst(self.time)
        tod = (tow//86400)*86400
        seph.t0 = gpst2time(week, toc+tod)
        seph.mode = 0

        return i, seph

    def decode(self, msg):
        i = 24
        self.msgtype = bs.unpack_from('u12', msg, i)[0]
        i += 12
        if self.monlevel > 0 and self.fh is not None:
            self.fh.write("##### RTCM 3.x type:{:04d} msg_size: {:d} data_len: {:d} bytes\n".
                          format(self.msgtype, self.dlen, self.len))

        obs = None
        eph = None
        geph = None
        seph = None

        # Network RTK residual messages
        if self.msgtype in (1057, 1063, 1240, 1246, 1252, 1258):
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)
        elif self.msgtype in (1058, 1064, 1241, 1247, 1253, 1259):
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_clk(msg, i)
        elif self.msgtype in (1059, 1065, 1242, 1248, 1254, 1260):
            self.subtype = sCSSR.CBIAS
            i = self.decode_cssr_cbias(msg, i)
        elif self.msgtype in (1060, 1066, 1243, 1249, 1255, 1261):
            self.subtype = sCSSR.COMBINED
            i = self.decode_cssr_comb(msg, i)
        elif self.msgtype in (1061, 1067, 1244, 1250, 1256, 1262):
            self.subtype = sCSSR.URA
            i = self.decode_cssr_ura(msg, i)
        elif self.msgtype in (1062, 1068, 1245, 1251, 1257, 1263):
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_hclk(msg, i)
        elif self.msgtype == 4076:  # IGS SSR
            i = self.decode_igsssr(msg, i)
        elif self.msgtype in (1030, 1031, 36, 37, 38):
            self.subtype = sRTCM.NRTK_RES
            i = self.decode_nrtk_residual(msg, i)
        elif self.is_msmtype(self.msgtype):
            self.subtype = sRTCM.MSM
            i, obs = self.decode_msm(msg, i)
        elif self.msgtype in (1007, 1008, 1033):
            self.subtype = sRTCM.ANT_DESC
            i = self.decode_ant_desc(msg, i)
        elif self.msgtype in (1005, 1006, 1032):
            self.subtype = sRTCM.ANT_POS
            i = self.decode_sta_pos(msg, i)
        elif self.msgtype == 1230:
            self.subtype = sRTCM.GLO_BIAS
            i = self.decode_glo_bias(msg, i)
        elif self.msgtype == 1019:
            self.subtype = sRTCM.GPS_EPH
            i, eph = self.decode_gps_eph(msg, i)
        elif self.msgtype == 1020:
            self.subtype = sRTCM.GLO_EPH
            i, geph = self.decode_glo_eph(msg, i)
        elif self.msgtype == 1041:
            self.subtype = sRTCM.IRN_EPH
            i, eph = self.decode_irn_eph(msg, i)
        elif self.msgtype == 1042 or self.msgtype == 63:
            self.subtype = sRTCM.BDS_EPH
            i, eph = self.decode_bds_eph(msg, i)
        elif self.msgtype == 1043:
            self.subtype = sRTCM.SBS_EPH
            i, seph = self.decode_sbs_eph(msg, i)
        elif self.msgtype == 1044:
            self.subtype = sRTCM.QZS_EPH
            i, eph = self.decode_qzs_eph(msg, i)
        elif self.msgtype == 1045 or self.msgtype == 1046:
            self.subtype = sRTCM.GAL_EPH
            i, eph = self.decode_gal_eph(msg, i)
        else:
            self.subtype = -1

        if self.monlevel > 0 and self.fh is not None:
            self.out_log(obs, eph, geph, seph)

        return i, obs, eph
