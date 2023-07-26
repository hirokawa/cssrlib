"""
RTCM 3 decoder

[1] RTCM Standard 10403.4, 2023
[2] IGS SSR Format version 1.00, 2020

"""

import numpy as np
import struct as st
import bitstruct.c as bs
from cssrlib.cssrlib import cssr, sCSSR, prn2sat, sCType
from cssrlib.gnss import uGNSS, sat2id, gpst2time, timediff, time2str, sat2prn
from cssrlib.gnss import uTYP, uSIG, rSigRnx
from crccheck.crc import Crc24LteA


class rtcm(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.len = 0
        self.monlevel = 1
        self.sysref = -1
        self.nsig_max = 4

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

        if self.monlevel > 0:
            print(f"tow={self.tow} st={self.subtype} sys={sys} iodssr={iodssr}")

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

    def decode_time(self, msg):
        i = 24
        self.msgtype = bs.unpack_from('u12', msg, i)[0]
        sys = self.get_ssr_sys(self.msgtype)

        if self.msgtype == 4076 or sys != uGNSS.GLO:
            blen = 20
        else:
            blen = 17

        self.tow = bs.unpack_from('u'+str(blen), msg, i)[0]
        self.time = gpst2time(self.week, self.tow)
        return self.time

    def out_log(self):
        sys = self.get_ssr_sys(self.msgtype)
        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))

        if self.subtype == sCSSR.CLOCK:
            for k, sat_ in enumerate(self.sat_n):
                sys_, _ = sat2prn(sat_)
                if sys != uGNSS.NONE and sys_ != sys:
                    continue
                self.fh.write(" {:s}\t{:5.3f}\n".format(sat2id(sat_),
                                                        self.lc[0].dclk[sat_]))

        elif self.subtype == sCSSR.ORBIT:
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

    def decode(self, msg):
        i = 24
        self.msgtype = bs.unpack_from('u12', msg, i)[0]
        i += 12
        if self.monlevel > 0 and self.fh is not None:
            self.fh.write("##### RTCM 3.x type:{:04d} msg_size: {:d} data_len: {:d} bytes\n".
                          format(self.msgtype, self.dlen, self.len))

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

        if self.monlevel > 0 and self.fh is not None:
            self.out_log()

        return i
