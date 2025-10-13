"""
RTCM 3 decoder

[1] RTCM Standard 10403.4 with Amendment 1
    Differential GNSS (Global Navigation Satellite Systems)
    Services - Version 3, 2024
[2] RTCM Standard 13400.00 (Working Draft)
    Integrity for High Accuracy GNSS based Applications
    - Version 0.91, 2025
[3] IGS SSR Format version 1.00, 2020

@author Rui Hirokawa

"""

import numpy as np
import struct as st
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, prn2sat, sCType, cssre
from cssrlib.gnss import uGNSS, sat2id, gpst2time, timediff, time2str, sat2prn
from cssrlib.gnss import uTYP, uSIG, rSigRnx, bdt2time, bdt2gpst, glo2time
from cssrlib.gnss import time2bdt, gpst2bdt, rCST, time2gpst, utc2gpst, timeadd
from cssrlib.gnss import gtime_t, sys2str, sat2prn, sat2id
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

    # test messages
    INTEG_MIN = 2003
    INTEG_EXT = 2004
    INTEG_EXT_SIS = 2005
    INTEG_EXT_AREA = 2006
    INTEG_QUALITY = 2007
    INTEG_CNR = 2008
    INTEG_VMAP = 2009
    INTEG_MMAP = 2010
    INTEG_SSR = 2011
    INTEG_SSR_IONO = 2012
    INTEG_SSR_TROP = 2013


class Integrity():
    """ class for integrity information in SC-134 """
    pid = 0  # provider id DFi027 (0-4095)
    vp = 0  # validity period DFi065 (0-15)
    uri = 0  # update rate interval DFi067 (b16)
    tow = 0
    iod_sys = {}  # issue of GNSS satellite mask DFi010 (b2)
    sts = {}  # constellation integrity status DFi029 (b16)
    mst = {}  # constellation monitoring status DFi030 (b16)
    src = {}

    nid = {}
    flag = {}
    sigmask = {}  # GNSS signal mask DFi033 (b32)
    iod_sig = {}  # issue of GNSS signal mask DFi012 (b2)
    sig_ists = {}  # signal integrity status DFi031 (b32)
    sig_msts = {}  # signal monitoring status DFi032 (b32)

    Paug = 0  # Augmentation System Probability Falut DFi049

    Pa_sp = {}  # Single Satellite PR Message Fault Probability DFi046
    sig_a = {}  # Overbounding stdev of PR Augmentation Meesage DFi036
    Pa_sc = {}  # Single Satellite CP Message Fault Probability DFi048
    ob_p = {}  # Overbounding bias of long-ter PR bias DFi037
    sig_ob_c = {}  # Ovebounding stdev of CP DFi038
    ob_c = {}  # Overbounding bias of long-term CP bias DFi039

    nsys = 0
    nsat = {}
    sys_t = None
    sat = {}
    mask_sys = 0

    mm_param = None

    sys_tbl = {0: uGNSS.GPS, 1: uGNSS.GLO, 2: uGNSS.GAL, 3: uGNSS.BDS,
               4: uGNSS.QZS, 5: uGNSS.IRN}

    # Mean Failure Duration (MFD) of the augmentation system indicator (DFi034)
    mfd_t = [5, 10, 30, 60, 120, 600, 900, 1800, 3600, 7200, 10800, 21600,
             43200, 86400, 604800, -1]
    # overbounding standard deviation indicator (DFi036)
    sig_ob_c_t = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20,
                  0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70, 0.80, 0.90,
                  1.00, 1.25, 1.50, 1.75, 2.00, 2.50, 3.00, 4.00, 5.00, 7.00,
                  -1, -1]
    # overbounding standard deviation of the carrier phase (DFi038)
    sig_ob_p_t = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008,
                  0.009, 0.010, 0.011, 0.012, 0.013, 0.014, 0.015, 0.016,
                  0.017, 0.018, 0.019, 0.020, 0.022, 0.024, 0.026, 0.028,
                  0.030, 0.032, 0.034, 0.036, 0.040, 0.045, 0.050, -1]
    # overbounding standard deviation of the carrier phase (DFi039)
    b_ob_p_t = [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.008, 0.010,
                0.012, 0.014, 0.016, 0.018, 0.020, 0.025, 0.030, -1]
    # overbounding standard deviation of the pseudorange (DFi037)
    b_ob_c_t = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10,
                0.20, 0.40, 0.60, 0.80, 1.0, 3.0, 7.0, -1]
    # duration of augmentation system (MFD) (DFi040)
    mfd_t = [0, 5, 10, 30, 60, 90, 300, 600, 1800, 3600, 7200, 14400, 28800,
             57600, 115200, -1]
    # correlation time of pseudorange error indicator (DFi044)
    tau_c_t = [0, 5, 10, 20, 30, 45, 60, 120, 180, 300, 600, 1200, 3600, 7200,
               14400, -1]
    # correlation time of carrier phase error indicator (DFi045)
    tau_p_t = [0, 2, 5, 10, 20, 30, 45, 60, 120, 180, 240, 300, 600, 1800,
               7200, -1]
    # probability indicator (DFi035, DFi046, DFi047, DFi048, DFi049)
    tau_p_t = [10**-11, 10**-10, 10**-9, 10**-8, 10**-7, 10**-6.2, 10**-6,
               10**-5.2, 10**-5, 10**-4.2, 10**-4, 10**-3.3, 10**-3, 10**-2.3,
               10**-2, -1]
    # TTT_comm indicator DFi050
    # ttt_comm_t = 0.1*(i+1), 127 is not provided
    # stdev indicator of bound on rate of change of PR error DFi120
    sig_rr_t = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2,
                6e-2, 0.10, 0.50, 5.0, 0, -1]
    # stdev indicator of bound on rate of change of phase error DFi121
    sig_rp_t = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 4e-4, 1e-3, 4e-3,
                0.01, 0.04, 0.10, 0.50, 5.0, -1]
    # stdev indicator of bound on rate of change of phase error DFi122
    sig_prre_t = [1e-3, 2e-3, 5e-3, 7.5e-3, 0.010, 0.014, 0.018, 0.025, 0.050,
                  0.10, 0.15, 0.20, 0.50, 1.0, 20.0, -1]
    # stdev indicator of bound on rate of change of phase error DFi123
    sig_dprre_t = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                   0.01, 0.1, 1.0, 10.0, 0, 0, -1]
    # bounding parameter in iono, tropo error indicator DFi129, DFi131
    b_datm_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # bounding pseudorange bias rate error indicator DFi125
    b_rsdsa_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # bounding rate of change in satellite orbit/clock error indicator DFi127
    b_rboc_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # residual iono error stdev for L1 GPS DFi128
    sig_di_t = [5e-3, 1e-2, 2e-2, 5e-2, 0.10, 0.25, 1.0, -1]
    # residual trop error stdev for L1 GPS DFi130
    sig_dt_t = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, -1]
    # bounding inter-constellation bias DFi132
    b_intc_t = [0.00, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, -1]
    # bounding inter-frequency bias
    b_intf_t = [0.00, 0.04, 0.07, 0.10, 0.20, 0.30, 0.50, -1]
    # GMM component expection Table 8-40
    mu_k_t = [0.00, 0.04, 0.07, 0.10, 0.20, 0.30, 0.50, 1, 2, 5, 10, 20, 50,
              80, 100, -1]
    # GMM component stdev Table 8-41
    sig_k_t = [0.0, 0.04, 0.07, 0.10, 0.20, 0.25, 0.30, 0.40, 0.50, 0.70,
               1, 2, 5, 7, 10, -1]

    # Validity Period DFi065
    vp_tbl = [1, 2, 5, 10, 15, 30, 60, 120, 240,
              300, 600, 900, 1800, 3600, 7200, 10800]

    def __init__(self):
        self.sys_r_tbl = {self.sys_tbl[s]: s for s in self.sys_tbl.keys()}
        None


class rtcm(cssr):
    """ class to decode RTCM3 messages """
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.len = 0
        self.monlevel = 1
        self.sysref = -1
        self.nsig_max = 4
        self.lock = {}
        self.mask_pbias = False

        self.nrtk_r = {}

        self.msm_t = {
            uGNSS.GPS: 1071, uGNSS.GLO: 1081, uGNSS.GAL: 1091,
            uGNSS.SBS: 1101, uGNSS.QZS: 1111, uGNSS.BDS: 1121
        }

        self.ssr_t = {
            uGNSS.GPS: 1057, uGNSS.GLO: 1063, uGNSS.GAL: 1240,
            uGNSS.QZS: 1246, uGNSS.SBS: 1252, uGNSS.BDS: 1258
        }

        self.eph_t = {
            uGNSS.GPS: 1019, uGNSS.GLO: 1020, uGNSS.BDS: 1042,
            uGNSS.QZS: 1044, uGNSS.GAL: 1046
        }

        self.integ = Integrity()
        self.test_mode = False  # for interop testing in SC134

    def is_msmtype(self, msgtype):
        """ check if the message type is MSM """
        for sys_ in self.msm_t.keys():
            if msgtype >= self.msm_t[sys_] and msgtype <= self.msm_t[sys_]+6:
                return True
        return False

    def adjustweek(self, week: int, tref: gtime_t):
        """ adjust week number considering reference time """
        week_, _ = time2gpst(tref)
        week_ref = (week_//1024)*1024
        return (week % 1024) + week_ref

    def msmtype(self, msgtype):
        """ get system and msm type from message type """
        sys = uGNSS.NONE
        msm = 0
        for sys_ in self.msm_t.keys():
            if msgtype >= self.msm_t[sys_] and msgtype <= self.msm_t[sys_]+6:
                sys = sys_
                msm = msgtype-self.msm_t[sys_]+1
                break
        return sys, msm

    def ssrtype(self, msgtype):
        """ get system and ssr type from message type """
        sys = uGNSS.NONE
        ssr = 0
        for sys_ in self.ssr_t.keys():
            if msgtype >= self.ssr_t[sys_] and msgtype <= self.ssr_t[sys_]+6:
                sys = sys_
                ssr = msgtype-self.ssr_t[sys_]+1
                break
        return sys, ssr

    def svid2sat(self, sys, svid):
        """ convert svid to sat """
        prn = svid
        if sys == uGNSS.QZS:
            prn += 192
        elif sys == uGNSS.SBS:
            prn += 119
        return prn2sat(sys, prn)

    def ssig2rsig(self, sys: uGNSS, utyp: uTYP, ssig):
        """ convert ssig to rSigRnx """
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
        """ convert ssig to rSigRnx for MSM """
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
        """ convert system enum to string """
        gnss_t = {uGNSS.GPS: "GPS", uGNSS.GLO: "GLO", uGNSS.GAL: "GAL",
                  uGNSS.BDS: "BDS", uGNSS.QZS: "QZS", uGNSS.SBS: "SBAS",
                  uGNSS.IRN: "NAVIC"}
        if sys not in gnss_t:
            return ""
        return gnss_t[sys]

    def sync(self, buff, k):
        """ check if the buffer has a sync pattern """
        return buff[k] == 0xd3

    def checksum(self, msg, k, maxlen=0):
        """ check the checksum of the message """
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
        """ decode the header of ssr message """
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
        """ decode satellite id """
        if self.msgtype == 4076:
            blen = 6
        else:
            if sys == uGNSS.GLO:
                blen = 5
            elif sys == uGNSS.QZS:
                blen = 4
            else:
                blen = 6

        svid = bs.unpack_from('u'+str(blen), msg, i)[0]
        i += blen

        sat = self.svid2sat(sys, svid)
        return i, sat

    def decode_orb_sat(self, msg, i, k, sys=uGNSS.NONE, inet=0):
        """ decode orbit correction of cssr """
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
        """ decoder high-rate clock correction of cssr """
        dclk, ddclk, dddclk = bs.unpack_from('s22', msg, i)[0]
        i += 22
        self.dclk_n[k] = self.sval(dclk, 22, 0.1e-3)
        return i

    def get_ssr_sys(self, msgtype):
        """ get system from ssr message type """
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
            elif msgtype >= 1265 and msgtype < 1271:  # proposed phase bias
                tbl_t = {1265: uGNSS.GPS, 1266: uGNSS.GLO, 1267: uGNSS.GAL,
                         1268: uGNSS.QZS, 1269: uGNSS.SBS, 1270: uGNSS.BDS}
                return tbl_t[msgtype]

    def decode_cssr_orb(self, msg, i, inet=0):
        """ decode RTCM Orbit Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        self.iode_n = np.zeros(nsat, dtype=int)
        self.dorb_n = np.zeros((nsat, 3))
        self.ddorb_n = np.zeros((nsat, 3))

        if timediff(self.time, self.lc[inet].t0s[sCType.ORBIT]) > 0:
            self.nsat_n = 0
            self.sys_n = []
            self.sat_n = []
            self.lc[inet].iode = {}
            self.lc[inet].dorb = {}

        self.iodssr = v['iodssr']
        sat = []
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_orb_sat(msg, i, k, sys)
            sat.append(sat_)
            self.lc[inet].iode[sat_] = self.iode_n[k]
            self.lc[inet].dorb[sat_] = self.dorb_n[k, :]

            self.set_t0(inet, sat_, sCType.ORBIT, self.time)

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.iodssr_c[sCType.ORBIT] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.ORBIT)
        self.lc[inet].t0s[sCType.ORBIT] = self.time

        return i

    def decode_cssr_clk(self, msg, i, inet=0):
        """decode RTCM Clock Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0s[sCType.CLOCK]) > 0:
            self.lc[inet].dclk = {}

        self.dclk_n = np.zeros(nsat)
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_clk_sat(msg, i, k)
            self.lc[inet].dclk[sat_] = self.dclk_n[k]

            self.set_t0(inet, sat_, sCType.CLOCK, self.time)

        self.iodssr_c[sCType.CLOCK] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.CLOCK)
        self.lc[inet].t0s[sCType.CLOCK] = self.time

        return i

    def decode_cssr_cbias(self, msg, i, inet=0):
        """decode RTCM Code Bias Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0s[sCType.CBIAS]) > 0:
            self.sat_b = []
            self.lc[inet].cbias = {}

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            self.set_t0(inet, sat_, sCType.CBIAS, self.time)

            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5
            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
            if sat_ not in self.lc[inet].cbias:
                self.lc[inet].cbias[sat_] = {}

            for j in range(nsig):
                sig, cb = bs.unpack_from('u5s14', msg, i)
                i += 19

                rsig = self.ssig2rsig(sys, uTYP.C, sig).str()
                self.lc[inet].cbias[sat_][rsig] = self.sval(cb, 14, 0.01)

        self.iodssr_c[sCType.CBIAS] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.CBIAS)
        self.lc[inet].t0s[sCType.CBIAS] = self.time

        return i

    def decode_cssr_pbias(self, msg, i, inet=0):
        """decode RTCM Phase Bias Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']

        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0s[sCType.PBIAS]) > 0:
            self.sat_b = []
            self.lc[inet].pbias = {}

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            self.set_t0(inet, sat_, sCType.PBIAS, self.time)

            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5
            yaw, dyaw = bs.unpack_from('u9s8', msg, i)
            yaw *= 1.0/256.0
            dyaw = self.sval(dyaw, 8, 1.0/8192.0)

            i += 17
            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
            if sat_ not in self.lc[inet].pbias:
                self.lc[inet].pbias[sat_] = {}

            for j in range(nsig):
                sig, si, wl, ci, pb = bs.unpack_from('u5u1u2u4s20', msg, i)
                i += 32

                rsig = self.ssig2rsig(sys, uTYP.L, sig).str()
                self.lc[inet].pbias[sat_][rsig] = self.sval(pb, 20, 1e-4)

        self.iodssr_c[sCType.PBIAS] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.PBIAS)
        self.lc[inet].t0s[sCType.PBIAS] = self.time

        return i

    def decode_cssr_comb(self, msg, i, inet=0):
        """ decode RTCM Combined Orbit and Clock Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']
        self.iode_n = np.zeros(nsat, dtype=int)
        self.dorb_n = np.zeros((nsat, 3))
        self.ddorb_n = np.zeros((nsat, 3))
        self.dclk_n = np.zeros(nsat)

        if timediff(self.time, self.lc[inet].t0s[sCType.ORBIT]) > 0:
            self.nsat_n = 0
            self.sys_n = []
            self.sat_n = []
            self.lc[inet].dclk = {}
            self.lc[inet].iode = {}
            self.lc[inet].dorb = {}

        self.iodssr = v['iodssr']
        sat = []
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            self.set_t0(inet, sat_, sCType.ORBIT, self.time)
            self.set_t0(inet, sat_, sCType.CLOCK, self.time)

            i = self.decode_orb_sat(msg, i, k, sys)
            i = self.decode_clk_sat(msg, i, k)
            sat.append(sat_)
            self.lc[inet].iode[sat_] = self.iode_n[k]
            self.lc[inet].dorb[sat_] = self.dorb_n[k, :]
            self.lc[inet].dclk[sat_] = self.dclk_n[k]

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.iodssr_c[sCType.ORBIT] = v['iodssr']
        self.iodssr_c[sCType.CLOCK] = v['iodssr']

        self.lc[0].cstat |= (1 << sCType.ORBIT)
        self.lc[0].cstat |= (1 << sCType.CLOCK)
        self.lc[inet].t0s[sCType.ORBIT] = self.time

        return i

    def decode_cssr_ura(self, msg, i, inet=0):
        """ decode RTCM URA message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']

        if timediff(self.time, self.lc[inet].t0s[sCType.URA]) > 0:
            self.ura = np.zeros(self.nsat_n)

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            s = self.sat_n.index(sat_)
            cls_, val = bs.unpack_from_dict('u3u3', msg, i)
            self.ura[s] = self.quality_idx(cls_, val)
            self.set_t0(inet, sat_, sCType.URA, self.time)

        self.iodssr_c[sCType.URA] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.URA)
        self.lc[inet].t0s[sCType.URA] = self.time

        return i

    def decode_cssr_hclk(self, msg, i, inet=0):
        """ decode RTCM High-rate Clock Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0s[sCType.HCLOCK]) > 0:
            self.lc[inet].dclk = {}

        for k in range(v['nsat']):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_hclk_sat(msg, i, k)
            self.lc[inet].dclk[sat_] = self.dclk_n[k]
            self.set_t0(inet, sat_, sCType.HCLOCK, self.time)

        self.iodssr_c[sCType.HCLOCK] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.HCLOCK)
        self.lc[inet].t0s[sCType.HCLOCK] = self.time

        return i

    def decode_vtec(self, msg, i, inet=0):
        """ decode RTCM VTEC Grid Point Data message """
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

        self.iodssr_c[sCType.VTEC] = v['iodssr']
        self.lc[0].cstat |= (1 << sCType.VTEC)
        self.lc[0].t0[sCType.VTEC] = self.time
        self.set_t0(ctype=sCType.VTEC, t=self.time)

        return i

    def decode_igsssr(self, msg, i=0):
        """ decode IGS SSR message """
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
        """ get system from nrtk message type """
        gnss_t = {1030: uGNSS.GPS, 1031: uGNSS.GLO,
                  1303: uGNSS.BDS, 1304: uGNSS.GAL, 1305: uGNSS.QZS}

        sys = uGNSS.NONE
        nrtk = 0
        if msgtype in gnss_t.keys():
            nrtk = 1
            sys = gnss_t[msgtype]

        return sys, nrtk

    def decode_nrtk_time(self, msg, i):
        """ decode Network RTK Time Message """
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
        """ decode Network RTK Residual Message """
        i, sys, time = self.decode_nrtk_time(msg, i)
        sz = 6 if sys != uGNSS.QZS else 4

        self.refid, self.nrefs, self.nsat = bs.unpack_from('u12u7u5', msg, i)
        i += 24

        for k in range(self.nsat):
            svid = bs.unpack_from('u'+str(sz), msg, i)[0]
            sat = self.svid2sat(sys, svid)
            i += sz
            s0c, s0d, s0h, sic, sid = bs.unpack_from('u8u9u6u10u10', msg, i)
            i += 43
            self.nrtk_r[sat] = np.array([s0c*5e-4, s0d*1e-8, s0h*1e-7,
                                         sic*5e-4, sid*1e-8])

        return i

    def decode_time(self, msg):
        """ decode time from message """
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

    def out_log_ssr_clk(self, sys):
        """ output ssr clock correction to log file """
        self.fh.write(" {:s}\t{:s}\n".format("SatID", "dclk [m]"))
        for k, sat_ in enumerate(self.lc[0].dclk.keys()):
            sys_, _ = sat2prn(sat_)
            if sys_ != sys:
                continue
            self.fh.write(" {:s}\t{:8.4f}\n".format(sat2id(sat_),
                                                    self.lc[0].dclk[sat_]))

    def out_log_ssr_orb(self, sys):
        """ output ssr orbit correction to log file """
        self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                      .format("SatID", "IODE", "Radial[m]",
                              "Along[m]", "Cross[m]"))
        for k, sat_ in enumerate(self.lc[0].dorb.keys()):
            sys_, _ = sat2prn(sat_)
            if sys_ != sys:
                continue
            self.fh.write(" {:s}\t{:3d}\t{:6.3f}\t{:6.3f}\t{:6.3f}\n".
                          format(sat2id(sat_),
                                 self.lc[0].iode[sat_],
                                 self.lc[0].dorb[sat_][0],
                                 self.lc[0].dorb[sat_][1],
                                 self.lc[0].dorb[sat_][2]))

    def out_log(self, obs=None, eph=None, geph=None, seph=None):
        """ output ssr message to log file """
        sys = self.get_ssr_sys(self.msgtype)
        inet = self.inet
        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))

        if self.subtype == sCSSR.CLOCK:
            self.out_log_ssr_clk(sys)

        if self.subtype == sCSSR.ORBIT:
            self.out_log_ssr_orb(sys)

        if self.subtype == sCSSR.COMBINED:
            self.out_log_ssr_clk(sys)
            self.out_log_ssr_orb(sys)

        if self.subtype == sCSSR.CBIAS or self.subtype == sCSSR.BIAS:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "CBias[m]", "..."))
            for k, sat_ in enumerate(self.lc[inet].cbias.keys()):
                sys_, _ = sat2prn(sat_)
                if sys_ != sys:
                    continue
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for sig in self.lc[inet].cbias[sat_].keys():
                    self.fh.write("{:s}\t{:5.2f}\t"
                                  .format(sig, self.lc[inet].cbias[sat_][sig]))
                self.fh.write("\n")

        if self.subtype == sCSSR.PBIAS or self.subtype == sCSSR.BIAS:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "PBias[m]", "..."))
            for k, sat_ in enumerate(self.lc[inet].pbias.keys()):
                sys_, _ = sat2prn(sat_)
                if sys_ != sys:
                    continue
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for sig in self.lc[inet].pbias[sat_].keys():
                    self.fh.write("{:s}\t{:5.2f}\t"
                                  .format(sig, self.lc[inet].pbias[sat_][sig]))
                self.fh.write("\n")

        if self.subtype == sRTCM.ANT_DESC:
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

        if self.subtype == sRTCM.NRTK_RES:
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

            self.fh.write(
                " PRN  s0c[mm] s0d[ppm] s0h[ppm]  sic[mm] sid[ppm]\n")
            sat_ = self.nrtk_r.keys()
            for sat in sat_:
                sys_, _ = sat2prn(sat)
                if sys_ != sys:
                    continue
                v = self.nrtk_r[sat]
                self.fh.write(" {:s}".format(sat2id(sat)))
                self.fh.write("  {:7.1f} {:8.2f} {:8.1f} {:8.1f} {:8.2f}\n"
                              .format(v[0]*1e3, v[1], v[2], v[3]*1e3, v[4]))

        if self.subtype == sRTCM.ANT_POS:
            self.fh.write(" {:20s}{:6d}\n".format("StationID:", self.refid))
            if self.msgtype in (1005, 1006):
                self.fh.write(" {:20s}{:6d}\n".format("Station Indicator:",
                                                      self.sti))
            self.fh.write(" {:20s} {:8.4f} {:8.4f} {:8.4f}\n"
                          .format("Antenna Position [m]:",
                                  self.pos_arp[0], self.pos_arp[1],
                                  self.pos_arp[2]))

        if self.subtype == sRTCM.GLO_BIAS:
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

        if self.subtype == sRTCM.MSM:
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

        if self.subtype in (sRTCM.INTEG_SSR, sRTCM.INTEG_SSR_IONO,
                            sRTCM.INTEG_SSR_TROP):
            self.fh.write(" {:20s}{:6d} TOW={:6d}\n".
                          format("INTEG-SSR:", self.msgtype,
                                 int(self.integ.tow)))
            self.fh.write(" {:20s}{:6d}\n".format("ProviderID:",
                                                  self.integ.pid))

            if self.subtype == sRTCM.INTEG_SSR:
                self.fh.write(" {:20s}{:6.1f}\n".format("Varidity Period:",
                                                        self.integ.vp))
                self.fh.write(" {:20s}{:6.1f}\n".format("Update Interval:",
                                                        self.integ.uri))

            self.fh.write(" {:20s}{:04x}\n".format("Constellation Mask:",
                                                   self.integ.mask_sys))
            self.fh.write(" IOD GNSS Mask: ")
            for sys in self.integ.iod_sys.keys():
                self.fh.write(" {:8s}: {:1d}".format(
                    sys2str(sys), self.integ.iod_sys[sys]))
            self.fh.write("\n")

            for sys in self.integ.flag.keys():
                self.fh.write(" Network ID:{:3d} Integrity Flag: ".format(
                    self.integ.nid[sys]))
                for sat in self.integ.flag[sys]:
                    self.fh.write(" {:3s}:{:2d}".format(
                        sat2id(sat), self.integ.flag[sys][sat]))
                self.fh.write("\n")
            self.fh.flush()

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

        if self.msgtype == 1029:  # Unicode Text
            self.fh.write("{:s}\n".format(self.ustr))

        if self.subtype == sRTCM.INTEG_SSR:
            None

    def decode_msm_time(self, sys, week, t):
        """ decode msm time """
        if sys == uGNSS.GLO:
            dow = (t >> 27) & 0x1f
            tow = (t & 0x7ffffff)*1e-3
            time = gpst2time(week, tow+dow*86400.0)
            time = utc2gpst(timeadd(time, -10800.0))
        elif sys == uGNSS.BDS:
            tow = t*1e-3
            time = bdt2gpst(gpst2time(week, tow))
        else:
            tow = t*1e-3
            time = gpst2time(week, tow)
        return time, tow

    def decode_msm(self, msg, i):
        """ decode MSM message """
        sys, msm = self.msmtype(self.msgtype)

        self.refid, tow_, self.mi, self.iods = bs.unpack_from(
            'u12u30u1u3', msg, i)
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
        obs.time = self.time

        obs.P = np.zeros((self.nsat, self.nsig), dtype=np.float64)
        obs.L = np.zeros((self.nsat, self.nsig), dtype=np.float64)
        obs.S = np.zeros((self.nsat, self.nsig), dtype=np.float64)
        obs.D = np.zeros((self.nsat, self.nsig), dtype=np.float64)
        obs.lli = np.zeros((self.nsat, self.nsig), dtype=np.int32)
        obs.sat = np.zeros(self.nsat, dtype=np.int32)

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
            hf_ = np.zeros(self.nsig, dtype=np.int32)
            cn_ = np.zeros(self.nsig, dtype=np.float64)

            fcn = 0
            if sys == uGNSS.GLO:
                if ex[k] <= 13:
                    fcn = ex[k]-7

            for j, sig_ in enumerate(self.sig_n[k]):
                idx = sig.index(sig_)
                if sys != uGNSS.GLO:
                    freq = obs.sig[sys][uTYP.C][idx].frequency()
                else:
                    freq = obs.sig[sys][uTYP.C][idx].frequency(fcn)

                pr_[idx] = pr[j+ofst]+r[k]
                cp_[idx] = (cp[j+ofst]+r[k])*freq/rCST.CLIGHT
                cn_[idx] = cnr[j+ofst]
                ll_[idx] = lock[j+ofst]
                hf_[idx] = half[j+ofst]

            ofst += nsig_

            obs.P[k, :] = pr_
            obs.L[k, :] = cp_
            obs.S[k, :] = cn_
            obs.sat[k] = sat_

            if sat_ in self.lock:
                for j, ll in enumerate(ll_):
                    ll_p = self.lock[sat_][j]
                    if (ll == 0 & ll_p != 0) | ll < ll_p:
                        obs.lli[k, j] |= 1
                    if hf_[j] > 0:
                        obs.lli[k, j] |= 3

            self.lock[sat_] = ll_

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
        prn, week, sva, code, idot = bs.unpack_from('u6u10u4u2s14', msg, i)
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
        eph.af2 = af2*rCST.P2_55
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
        eph.toes = toe*16.0
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

        eph.week = self.adjustweek(eph.week, self.time)

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.tot = self.time
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
        En, P4, Ft, Nt, M, ad, Na, tauc, N4, taugps, ln = bs.unpack_from(
            'u5u1u4u11u2u1u11s32u5s22u1', msg, i)
        i += 102

        geph = Geph()
        geph.pos = np.zeros(3)
        geph.vel = np.zeros(3)
        geph.acc = np.zeros(3)

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

        # b7-8: M, b6: P4, b5: P3, b4: P2, b2-3: P1, b0-1: P
        geph.status = M << 7 | P4 << 6 | P3 << 5 | P2 << 4 | P1 << 2 | P

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

        eph.af2 = af2*rCST.P2_59
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

        eph.week = self.adjustweek(eph.week, self.time)

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.tot = self.time
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
        svid, toc, af2, af1, af0 = bs.unpack_from('u4u16s8s16s22', msg, i)
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
        eph.sat = prn2sat(uGNSS.QZS, svid+192)
        toc *= 16.0
        eph.af2 = af2*rCST.P2_55
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
        eph.tgd = tgd*rCST.P2_31
        eph.iodc = iodc
        eph.fit = fit

        eph.week = self.adjustweek(eph.week, self.time)

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.tot = self.time
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
        eph.tot = self.time
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

        eph.week = self.adjustweek(eph.week, self.time)

        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = gpst2time(eph.week, toc)
        eph.tot = self.time
        eph.A = sqrtA*sqrtA
        eph.iodc = eph.iode
        eph.mode = 0

        return i, eph

    def decode_sbs_eph(self, msg, i):
        """ SBAS Satellite Ephemeris Message """
        svid, iodn, toc, ura = bs.unpack_from('u6u8u13u4', msg, i)
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
        seph.sat = prn2sat(uGNSS.SBS, svid+119)
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

    def decode_unicode_text(self, msg, i):
        """ RTCM SC-104 Unicode Text Message"""
        refid, mjd, sod, nch, ncp = bs.unpack_from('u12u16u17u7u8', msg, i)
        i += 12+16+17+7+8
        ic = i//8
        self.ustr = bytes(msg[ic:ic+ncp]).decode()

    def decode_integrity_min(self, msg, i):
        """ RTCM SC-134 minimum integrity message (MT3) """

        # augmentation service provider id DFi027
        # GPS Epoch Time (TOW) DFi008
        # Constellation Mask DFi013
        pid, tow, mask_sys = bs.unpack_from('u12u30u16', msg, i)
        i += 58

        # constellation integirity status DFi029
        # constellation monitoring status DFi030
        c_integ, c_status, vp, uri = bs.unpack_from('u16u16u4u16', msg, i)
        i += 52

        self.integ.vp = vp  # Validity Period DFi065 (0-15)
        self.integ.uri = uri  # update rate interval DFi067 (0.1)

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)

        iod_sys = {}
        sts_tbl = {}
        mst_tbl = {}
        src_tbl = {}

        iod_sig = {}
        sigmask = {}
        sig_ists = {}
        sig_msts = {}

        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            # GNSS satellite mask DFi009
            # Issue of GNSS Satellite Mask DFi010
            mask_sat, iod = bs.unpack_from('u64u2', msg, i)
            i += 66
            # Satellite Integrity Status DFi019
            # Satellite Monitoring Status DFi018
            # Satellite Integrity Fault source DFi052
            sts, mst, src = bs.unpack_from('u64u64u3', msg, i)
            i += 131

            svid_t, nsat = self.decode_mask(mask_sat, 64)

            iod_sys[sys] = iod
            sts_tbl[sys] = sts
            mst_tbl[sys] = mst
            src_tbl[sys] = src

            iod_sig[sys] = {}
            sigmask[sys] = {}
            sig_ists[sys] = {}
            sig_msts[sys] = {}
            for svid in svid_t:
                ofst = 192 if sys == uGNSS.QZS else 0
                prn = svid+ofst
                sat = prn2sat(sys, prn)

                # GNSS signal mask DFi033
                sigmask[sys][sat] = bs.unpack_from('u32', msg, i)[0]
                i += 32
                # issue of GNSS signal mask DFi012
                iod_sig[sys][sat] = bs.unpack_from('u2', msg, i)[0]
                i += 2
                # signal integrity status DFi031
                sig_ists[sys][sat] = bs.unpack_from('u32', msg, i)[0]
                i += 32
                # signal monitoring status DFi032
                sig_msts[sys][sat] = bs.unpack_from('u32', msg, i)[0]
                i += 32

        self.integ.pid = pid
        self.integ.tow = tow

        self.integ.iod_sys = iod_sys
        self.integ.sts = sts_tbl
        self.integ.mst = mst_tbl
        self.integ.src = src_tbl

        self.integ.sigmask = sigmask
        self.integ.iod_sig = iod_sig
        self.integ.sig_ists = sig_ists
        self.integ.sig_msts = sig_msts

    def decode_integrity_ext(self, msg, i):
        """ RTCM SC-134 extended integrity message
            service levels and overbounding parameters (MT4) """

        # GPS Epoch Time (TOW) DFi008
        # augmentation service provider id DFi027
        # Augmentation Integrity Level DFi026
        # Service Provider Solution Type DFi004
        tow, pid, ilvl, stype = bs.unpack_from('u30u12u4u4', msg, i)
        i += 50

        self.integ.pid = pid
        self.integ.tow = tow
        self.integ.ilvl = ilvl
        self.intrg.stype = stype

        # GNSS Constellation Mask DFi013
        # Validity Period DFi065 (0-15)
        # update rate interval DFi067 (0.1)
        # Validity Area Type DFi056
        mask_sys, vp, uri, atype = bs.unpack_from('u16u4u16u2', msg, i)
        i += 38

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)
        self.integ.vp = vp  # Validity Period DFi065 (0-15)
        self.integ.uri = uri  # update rate interval DFi067 (0.1)
        self.integ.atype = atype

        # Paug Bundling flag DFi066
        f_paug = bs.unpack_from('u1', msg, i)[0]
        i += 1
        if f_paug == 1:
            # Paug Augmentation System Probability Falut DFi049
            self.integ.Paug = bs.unpack_from('u4', msg, i)[0]
            i += 4

        # Time correlation integrity/continuity flag DFi137
        #  0: DFi040-045 are valid only for integrity monitoring
        #  1: DFi040-045 are valid also for continuity monitoring
        # integrity parameter IOD DFi006
        self.integ.fc, self.integ.iod_p = bs.unpack_from('u1u6', msg, i)
        i += 7

        if atype == 1:  # Validity Area Parameters defined in Table 10.3-5
            # service area IOD DFi005
            # number of area points DFi201
            self.integ.iod_sa, narea = bs.unpack_from('u6u8', msg, i)
            i += 14
            self.integ.narea = narea
            self.integ.pos = np.zeros((narea, 3))

            for k in range(narea):
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                # Area Point - Height DFi204
                lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
                i += 83

                self.integ.pos[k, :] = [lat, lon, alt]

        elif atype == 0:  # Validity Radius Data defined in Table 10.3-6
            # service area IOD DFi005
            self.integ.iod_sa = bs.unpack_from('u6', msg, i)[0]
            i += 6
            # Augmentation Network Computation Point ECEF - X DFi053
            # Augmentation Network Computation Point ECEF - Y DFi054
            # Augmentation Network Computation Point ECEF - Z DFi055
            # Validity Radius DFi057
            x, y, z, vr = bs.unpack_from('s34s34s34u20', msg, i)
            i += 122

            self.integ.pnt_vr = [x, y, z, vr]

        self.integ.Pa_sp = {}
        self.integ.sig_a = {}
        self.integ.Pa_sc = {}
        self.integ.ob_p = {}
        self.integ.sig_ob_c = {}
        self.integ.ob_c = {}

        self.integ.mfd_s = {}
        self.integ.mfd_c = {}

        self.integ.tau_p = {}
        self.integ.tau_c = {}
        self.integ.tau_cp = {}
        self.integ.tau_cc = {}

        self.integ.uri = {}

        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            # Mean failure duration (MFD) of a single satellite fault DFi034
            # Mean failure duration (MFD) of a constellation fault DFi040
            mfd_s, mfd_c = bs.unpack_from('u4u4', msg, i)
            i += 8

            # Correlation Time of Pseudorange Augmentation Message Error DFi044
            # Correlation Time of Carrier Phase Augmentation Message Error
            #  DFi045
            # Multiple Satellite Pseudorange Augmentation Message Fault
            # Probability DFi047
            # Multiple Satellite Carrier Phase Augmentation Message Fault
            # Probability DFi035
            tau_p, tau_c, tau_cp, tau_cc = bs.unpack_from('u4u4u4u4', msg, i)
            i += 16

            # Update rate interval DFi067
            # GNSS satellite mask DFi009
            uri, mask_sat = bs.unpack_from('u16u64', msg, i)
            i += 80

            self.integ.mfd_s[sys] = mfd_s
            self.integ.mfd_c[sys] = mfd_c

            self.integ.tau_p[sys] = tau_p
            self.integ.tau_c[sys] = tau_c
            self.integ.tau_cp[sys] = tau_cp
            self.integ.tau_cc[sys] = tau_cc

            self.integ.uri[sys] = uri

            svid_t, nsat = self.decode_mask(mask_sat, 64)

            for svid in svid_t:
                ofst = 192 if sys == uGNSS.QZS else 0
                prn = svid+ofst
                sat = prn2sat(sys, prn)
                # Single Satellite Pseudorange Augmentation Message Fault
                # Probability DFi046
                # Overbounding Standard Deviation of the Pseudorange
                # Augmentation Message Error under Fault-Free Scenario DFi036
                # Single Satellite Carrier Phase Augmentation Message
                # Fault Probability DFi048
                # Overbounding Bias of the Long-Term Pseudorange Augmentation
                # Message Bias Error under Fault-Free Scenario DFi037
                # Overbounding Standard Deviation of the Carrier Phase
                # Augmentation Message Error under Fault-Free Scenario DFi038
                # Overbounding Bias of the Long-Term Carrier Phase
                # Augmentation Message Bias Error under Fault-Free DFi039
                Pa_sp, sig_a, Pa_sc, ob_p, sig_ob_c, ob_c = bs.unpack_from(
                    'u4u5u4u4u5u4', msg, i)
                i += 26

                self.integ.Pa_sp[sys][sat] = Pa_sp
                self.integ.sig_a[sys][sat] = sig_a
                self.integ.Pa_sc[sys][sat] = Pa_sc
                self.integ.ob_p[sys][sat] = ob_p
                self.integ.sig_ob_c[sys][sat] = sig_ob_c
                self.integ.ob_c[sys][sat] = ob_c

    def decode_integrity_ext_sis_local(self, msg, i):
        """ RTCM SC-134 extended integrity message
            signal in space integrity and local error parameters (MT5) """

        # integrity parameter IOD DFi006
        # augmentation service provider id DFi027
        # GPS Epoch Time (TOW) DFi008
        iod_ip, pid, tow = bs.unpack_from('u6u12u30', msg, i)
        i += 48

        # GNSS Constellation Mask DFi013
        # Validity Period DFi065 (0-15)
        # update rate interval DFi067 (0.1)
        # Validity Area Type DFi056
        # Time correlation integrity/continuity flag DFi137
        mask_sys, vp, uri, atype, fc = bs.unpack_from('u16u4u16u2u1', msg, i)
        i += 39

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)
        self.integ.vp = vp  # Validity Period DFi065 (0-15)
        self.integ.uri = uri  # update rate interval DFi067 (0.1)

        if atype == 1:  # Validity Area Parameters defined in Table 10.3-5
            # service area IOD DFi005
            # number of area points DFi201
            self.integ.iod_sa, narea = bs.unpack_from('u6u8', msg, i)
            i += 14
            self.integ.pos = np.zeros((narea, 3))

            for k in range(narea):
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                # Area Point - Height DFi204
                lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
                i += 83

                self.integ.pos[k, :] = [lat, lon, alt]

        elif atype == 0:  # Validity Radius Data defined in Table 10.3-6
            # service area IOD DFi005
            self.integ.iod_sa = bs.unpack_from('u6', msg, i)[0]
            i += 6
            # Augmentation Network Computation Point ECEF - X DFi053
            # Augmentation Network Computation Point ECEF - Y DFi054
            # Augmentation Network Computation Point ECEF - Z DFi055
            # Validity Radius DFi057
            x, y, z, vr = bs.unpack_from('s34s34s34u20', msg, i)
            i += 122

            self.integ.pnt_vr = [x, y, z, vr]

        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            # GNSS satellite mask DFi009
            # issue of GNSS satellite mask DFi010
            mask_sat, iod_mask = bs.unpack_from('u64u2', msg, i)
            i += 66

            svid_t, nsat = self.decode_mask(mask_sat, 64)

            self.integ.idx_c[sys] = {}
            self.integ.idx_p[sys] = {}
            self.integ.cbr[sys] = {}
            self.integ.sig_dp[sys] = {}
            self.integ.idx_dp[sys] = {}

            self.integ.oc[sys] = {}
            self.integ.sig_ion[sys] = {}
            self.integ.idx_ion[sys] = {}
            self.integ.sig_trp[sys] = {}
            self.integ.idx_trp[sys] = {}

            for svid in svid_t:
                ofst = 192 if sys == uGNSS.QZS else 0
                prn = svid+ofst
                sat = prn2sat(sys, prn)

                # Pseudorange error rate integrity index DFi120
                # Phase error rate integrity index DFi121
                # Nominal pseudorange bias rate of change DFi125
                # Bounding sigma on phase-range-rate error DFi122
                # Phase-range-rate error rate integrity index DFi123
                idx_c, idx_p, cbr, sig_dp, idx_dp = bs.unpack_from(
                    'u4u4u3u4u4', msg, i)
                i += 19

                # Orbit and clock error rate integrity parameter DFi127
                # Residual ionospheric error standard deviation DFi128
                # Index of bounding parameter on change in iono error DFi129
                # Residual tropospheric error standard deviation DFi130
                # Index of bounding parameter on change in tropo error DFi131
                oc, sig_ion, idx_ion, sig_trp, idx_trp = bs.unpack_from(
                    'u3u3u3u3u3', msg, i)
                i += 15

                self.integ.idx_c[sys][sat] = idx_c
                self.integ.idx_p[sys][sat] = idx_p
                self.integ.cbr[sys][sat] = cbr
                self.integ.sig_dp[sys][sat] = sig_dp
                self.integ.idx_dp[sys][sat] = idx_dp

                self.integ.oc[sys][sat] = oc
                self.integ.sig_ion[sys][sat] = sig_ion
                self.integ.idx_ion[sys][sat] = idx_ion
                self.integ.sig_trp[sys][sat] = sig_trp
                self.integ.idx_trp[sys][sat] = idx_trp

    def decode_integrity_ext_service_area(self, msg, i):
        """ RTCM SC-134 Extended Service Area Parameters (MT6) """

        # integrity parameter IOD DFi006
        # augmentation service provider id DFi027
        # GPS Epoch Time (TOW) DFi008
        # augmentation integrity level DFi026
        iod_sa, tow, pid, alvl = bs.unpack_from('u6u30u12u4', msg, i)
        i += 52

        # GNSS Constellation Mask DFi013
        # Validity Period DFi065 (0-15)
        # update rate interval DFi067 (0.1)
        # Validity Area Type DFi056
        mask_sys, vp, uri, atype = bs.unpack_from('u16u4u16u2', msg, i)
        i += 38

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)

        # Augmentation IR Degradation Factor DFi140
        # Augmentation TTD Degradation Factor DFi141
        # Extended Area Time Parameter Degradation Factor DFi138
        # Extended Area Spatial Parameter Degradation Factor DFi139
        ir_d, ttd_d, t_d, s_d = bs.unpack_from('u2u2u3u3', msg, i)
        i += 10

        self.integ.iod_sa = iod_sa
        self.integ.tow = tow
        self.integ.pid = pid
        self.integ.alvl = alvl
        self.integ.vp = vp
        self.integ.uri = uri
        self.integ.ir_d = ir_d
        self.integ.ttd_d = ttd_d
        self.integ.t_d = t_d
        self.integ.s_d = s_d

        if atype == 1:  # Validity Area Parameters defined in Table 10.3-5
            # service area IOD DFi005
            # number of area points DFi201
            self.integ.iod_sa, narea = bs.unpack_from('u6u8', msg, i)
            i += 14
            self.integ.pos = np.zeros((narea, 3))
            for k in range(narea):
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                # Area Point - Height DFi204
                lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
                i += 83
                self.integ.pos[k, :] = [lat, lon, alt]

        elif atype == 0:  # Validity Radius Data defined in Table 10.3-6
            # service area IOD DFi005
            self.integ.iod_sa = bs.unpack_from('u6', msg, i)[0]
            i += 6
            # Augmentation Network Computation Point ECEF - X DFi053
            # Augmentation Network Computation Point ECEF - Y DFi054
            # Augmentation Network Computation Point ECEF - Z DFi055
            # Validity Radius DFi057
            x, y, z, vr = bs.unpack_from('s34s34s34u20', msg, i)
            i += 122
            self.integ.pnt_vr = [x, y, z, vr]

    def decode_integrity_quality(self, msg, i):
        """ RTCM SC-134 Message Quality Indicator (MT7) """

        # GPS Epoch Time (TOW) DFi008
        # augmentation service provider id DFi027
        # Validity Period DFi065 (0-15)

        tow, pid, vp = bs.unpack_from('u30u12u4', msg, i)
        i += 46

        # update rate interval DFi067 (0.1)
        # network id DFi071
        # quality indicator mask DFi061
        uri, nid, mask_q = bs.unpack_from('u16u8u8', msg, i)
        i += 32

        self.integ.tow = tow
        self.integ.pid = pid
        self.integ.vp = vp
        self.integ.uri = uri
        self.integ.nid = nid

        idx_q, np = self.decode_mask(mask_q, 8, ofst=0)
        self.integ.qi = np.zeros(np)
        for k in range(np):
            self.integ.qi[k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

    def decode_integrity_cnr_acg(self, msg, i):
        """ RTCM SC-134 CNR/ACG Signal In Space Monitoring Message (MT8) """

        # augmentation service provider id DFi027
        # GPS Epoch Time (TOW) DFi008
        # number of reference stations DFi072
        # update rate interval DFi067 (0.1)
        pid, tow, nst, uri = bs.unpack_from('u12u30u12u16', msg, i)
        i += 12+30+12+16

        self.integ.tow = tow
        self.integ.pid = pid
        self.integ.uri = uri

        self.integ.pos = np.zeros((nst, 3))
        self.integ.iod_sat = np.zeros(nst, dtype=int)

        for k in range(nst):
            # Area Point - Lat DFi202
            # Area Point - Lon DFi203
            # Area Point - Height DFi204
            lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
            i += 34+35+14

            self.integ.pos[k, :] = [lat, lon, alt]

            # Constellation Mask DFi013
            # GNSS Satellite Mask DFi009
            # Issue of GNSS Satellite Mask DFi010
            mask_c, mask_s, iod_sat = bs.unpack_from('u16u64u2', msg, i)
            i += 82

            self.integ.iod_sat[k] = iod_sat

            sys_t, nsys = self.decode_mask(mask_c, 16, ofst=0)
            svid_t, nsat = self.decode_mask(mask_s, 64)

            for sys in sys_t:
                self.integ.band[sys] = {}
                self.integ.cnr[sys] = {}
                self.integ.agc[sys] = {}

                for svid in svid_t:
                    ofst = 192 if sys == uGNSS.QZS else 0
                    prn = svid+ofst
                    sat = prn2sat(sys, prn)

                    # GNSS Signal Mask DFi033
                    # Issue of GNSS Signal Mask DFi012
                    mask_sig, iod_sig = bs.unpack_from('u32u2', msg, i)
                    i += 34
                    sig_t, nsig = self.decode_mask(mask_sig, 32, ofst=0)

                    self.integ.band[sys][sat] = np.zeros(nsig)
                    self.integ.cnr[sys][sat] = np.zeros(nsig)
                    self.integ.agc[sys][sat] = np.zeros(nsig)

                    for sig in sig_t:
                        # Frequency band ID DFi135
                        band = bs.unpack_from('u5', msg, i)[0]
                        i += 5
                        self.integ.band[sys][sat][sig] = band

                    for sig in sig_t:
                        # CNR carrier to noise ratio DFi133
                        cnr = bs.unpack_from('u8', msg, i)[0]
                        i += 8
                        self.integ.cnr[sys][sat][sig] = cnr

                    for sig in sig_t:
                        # AGC, Automatic Gain Control DFi134
                        agc = bs.unpack_from('u8', msg, i)[0]
                        i += 8
                        self.integ.agc[sys][sat][sig] = agc

    def decode_integrity_vmap(self, msg, i):
        """ RTCM SC-134 Satellite Visibility Map Message (MT9) """

        if self.test_mode:  # for test
            wg, st, rev = bs.unpack_from('u4u8u8', msg, i)
            i += 20
            self.subtype = st
            self.ver = rev
            self.wg = wg-1

        # GPS Epoch Time (TOW) DFi008
        # number of area points DFi201
        # Number of Azimuth Slices DFi205
        # Message Continuation Flag DFi021
        tow, narea, naz, mcf = bs.unpack_from('u30u8u6u1', msg, i)
        i += 45
        self.integ.pos = np.zeros((narea, 3))

        tow *= 1e-3

        for k in range(narea):
            # Area Point - Lat DFi202
            # Area Point - Lon DFi203
            # Area Point - Height DFi204
            lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
            i += 83

            self.integ.pos[k, :] = [lat*1.1e-8, lon*1.1e-8, alt]

        self.integ.azel = np.zeros((naz, 2))

        # Azimuth DFi206
        # Elevation Mask DFi208
        az = 0
        for k in range(naz):
            daz, mask_el = bs.unpack_from('u9u7', msg, i)
            i += 16
            az += daz
            self.integ.azel[k, :] = [az, mask_el]

        return i

    def decode_integrity_mmap(self, msg, i):
        """ RTCM SC-134 Multipath Map Message (MT10) """

        if self.test_mode:  # for test
            wg, st, rev = bs.unpack_from('u4u8u8', msg, i)
            i += 20
            self.subtype = st
            self.ver = rev
            self.wg = wg-1

        # GPS Epoch Time (TOW) DFi008
        # number of area points DFi201
        # multipath model ID DFi209: 0:GMM,1:MBM,2:JM
        tow, narea, mm_id, mcf = bs.unpack_from('u30u8u3u1', msg, i)
        i += 42
        tow *= 1e-3
        self.integ.pos = np.zeros((narea, 3))
        self.integ.mm_id = mm_id

        if mm_id == 0:

            for k in range(narea):
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                # Area Point - Height DFi204
                lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
                i += 83

                self.integ.pos[k, :] = [lat*1.1e-8, lon*1.1e-8, alt]

        elif mm_id == 1 or mm_id == 2:
            sigmask = np.array(narea, dtype=np.int32)
            for k in range(narea):
                # GNSS Signal Modulation Mask - DFi214
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                # Area Point - Height DFi204
                sigmask[k], lat, lon, alt = bs.unpack_from(
                    'u8s34s35s14', msg, i)
                i += 91

                self.integ.pos[k, :] = [lat*1.1e-8, lon*1.1e-8, alt]

        if mm_id == 0:
            # Number of GMM components DFi210
            ngmm = bs.unpack_from('u2', msg, i)[0]
            i += 2

            self.integ.mm_param = np.zeros((ngmm, 3))

            for k in range(ngmm):
                # GMM component probability DFi211
                # GMM component expectation DFi212
                # GMM component standard deviation DFi213
                prob, exp_, std_ = bs.unpack_from('u4u4u4', msg, i)
                i += 12

                prob *= 0.0625
                exp = self.integ.mu_k_t[exp_]
                std = self.integ.sig_k_t[std_]
                self.integ.mm_param[k, :] = [prob, exp, std]

        elif mm_id == 1 or mm_id == 2:
            self.integ.mm_param = {}
            n = 3 if mm_id == 1 else 4

            for k in range(narea):
                sig_t, nsig = self.decode_mask(sigmask[k], 8, ofst=0)
                prm = np.zeros((nsig, n))

                for j in range(nsig):
                    # Multipath parameter a DFi215
                    # Multipath parameter b DFi216
                    # Multipath parameter c DFi217
                    a, b, c = bs.unpack_from('u4s5s5', msg, i)
                    i += 14

                    a = self.integ.sig_k_t[a]
                    prm[j, 0:3] = [a, b*0.25, c*0.0625]

                    if mm_id == 2:
                        # Multipath parameter d DFi218
                        d = bs.unpack_from('u8', msg, i)[0]
                        i += 8
                        prm[j, 3] = d*0.3515625

                self.integ.mm_param[k] = prm

        return i

    def decode_integrity_ssr(self, msg, i):
        """ RTCM SC-134 SSR integrity message (MT11,12,13) """
        pid, tow, mask_sys = bs.unpack_from('u12u30u16', msg, i)
        i += 58
        tow *= 1e-3

        # update rate interval DFi067

        # mask_sys:: DFi013 0:GPS,1:GLO,2:GAL,3:BDS,4:QZS,5:IRN

        if self.msgtype == 11:
            vp, uri = bs.unpack_from('u4u16', msg, i)
            i += 20
            # Validity Period DFi065 (0-15)
            self.integ.vp = self.integ.vp_tbl[vp]
            self.integ.uri = uri*0.1  # update rate interval DFi067 (0.1)

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)
        iod_sys = {}
        for k in range(nsys):
            sys = self.integ.sys_tbl[sys_t[k]]
            iod_sys[sys] = bs.unpack_from('u2', msg, i)[0]
            i += 2

        flag_t = {}
        nid = {}
        for sys_ in sys_t:
            nid_, mask_sat = bs.unpack_from('u8u64', msg, i)
            i += 72
            svid_t, nsat = self.decode_mask(mask_sat, 64)
            sys = self.integ.sys_tbl[sys_]
            nid[sys] = nid_
            flag_t[sys] = {}
            for svid in svid_t:
                ofst = 192 if sys == uGNSS.QZS else 0
                prn = svid+ofst
                sat = prn2sat(sys, prn)
                flag_t[sys][sat] = bs.unpack_from('u2', msg, i)[0]
                i += 2

        self.integ.mask_sys = mask_sys
        self.integ.pid = pid
        self.integ.tow = tow
        self.integ.iod_sys = iod_sys
        self.integ.nid = nid
        self.integ.flag = flag_t

    def decode(self, msg, subtype=None):
        """ decode RTCM messages """
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

        self.subtype = subtype

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
        elif self.msgtype in (1265, 1266, 1267, 1268, 1269, 1270):
            if not self.mask_pbias:
                self.subtype = sCSSR.PBIAS
                i = self.decode_cssr_pbias(msg, i)
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
        elif self.msgtype in (1030, 1031, 1303, 1304, 1305):
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
        elif self.msgtype == 1029:
            self.decode_unicode_text(msg, i)

        # test messages for SC-134
        elif self.msgtype == 3:  # minimum integrity
            self.subtype = sRTCM.INTEG_MIN
            self.decode_integrity_min(msg, i)
        elif self.msgtype == 4:  # extended integrity
            self.subtype = sRTCM.INTEG_EXT
            self.decode_integrity_ext(msg, i)
        elif self.msgtype == 5:  # sis integrity/local error
            self.subtype = sRTCM.INTEG_EXT_SIS
            self.decode_integrity_ext_sis_local(msg, i)
        elif self.msgtype == 6:  # extended sercice area
            self.subtype = sRTCM.INTEG_EXT_AREA
            self.decode_integrity_ext_service_area(msg, i)
        elif self.msgtype == 7:  # quality indicator
            self.subtype = sRTCM.INTEG_QUALITY
            self.decode_integrity_quality(msg, i)
        elif self.msgtype == 8:  # CNR/ACG SIS Monitoring
            self.subtype = sRTCM.INTEG_CNR
            self.decode_integrity_cnr_acg(msg, i)
        elif self.msgtype == 9:  # satellite visibility map
            self.subtype = sRTCM.INTEG_VMAP
            self.decode_integrity_vmap(msg, i)
        elif self.msgtype == 10:  # multipath map
            self.subtype = sRTCM.INTEG_MMAP
            self.decode_integrity_mmap(msg, i)
        elif self.msgtype == 11:  # SSR integrity
            self.subtype = sRTCM.INTEG_SSR
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 12:  # SSR integrity Iono
            self.subtype = sRTCM.INTEG_SSR_IONO
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 13:  # SSR integrity Trop
            self.subtype = sRTCM.INTEG_SSR_TROP
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 54:  # SSR integrity test msg
            if self.subtype == 9:
                self.decode_integrity_vmap(msg, i)
            elif self.subtype == 10:
                self.decode_integrity_mmap(msg, i)
        else:
            self.subtype = -1

        if self.monlevel > 0 and self.fh is not None:
            self.out_log(obs, eph, geph, seph)

        return i, obs, eph, geph, seph


class rtcme(cssre):
    """ class for RTCM message encoder """

    def __init__(self):
        super().__init__()
        self.integ = Integrity()

    def set_sync(self, msg, k):
        msg[k] = 0xd3

    def set_len(self, msg, k, len_):
        st.pack_into('>H', msg, k+1, len_ & 0x3ff)
        self.len = len_
        self.dlen = len_+6
        return len_

    def set_checksum(self, msg, k, maxlen=0):
        if self.len < 6:
            return False
        if maxlen > 0 and k+self.len >= maxlen:
            return False
        cs = Crc24LteA.calc(msg[k:k+self.len+3])
        st.pack_into('>H', msg, k+self.len+3, cs >> 8)
        st.pack_into('>B', msg, k+self.len+5, cs & 0xff)
        return cs

    def set_body(self, msg, buff, k, len_):
        msg[k+3:k+len_] = buff[3:len_]

    def encode_integrity_ssr(self, msg, i):
        """ RTCM SC-134 SSR integrity message (MT11,12,13) """

        sys_t = self.integ.iod_sys.keys()
        gnss_t = []
        for sys in sys_t:
            gnss_t.append(self.integ.sys_r_tbl[sys])

        mask_sys = self.encode_mask(gnss_t, 16, ofst=0)

        bs.pack_into('u12', msg, i, self.integ.pid)  # provider id DFi027
        i += 12
        bs.pack_into('u30', msg, i, self.integ.tow)  # tow DFi008
        i += 30
        bs.pack_into('u16', msg, i, mask_sys)  # GNSS constellation mask DFi013
        i += 16

        # mask_sys:: DFi013 0:GPS,1:GLO,2:GAL,3:BDS,4:QZS,5:IRN

        if self.msgtype == 11:
            # validity period DFi065
            # update rate interval DFi067
            bs.pack_into('u4u16', msg, i, self.integ.vp, self.integ.uri)
            i += 20

        for sys in sys_t:
            # issue of GNSS satellite mask DFi010
            bs.pack_into('u2', msg, i, self.integ.iod_sys[sys])
            i += 2

        for sys in sys_t:
            flag = self.integ.flag[sys]
            sat_t = flag.keys()
            svid_t = []
            for sat in sat_t:
                sys, prn = sat2prn(sat)
                ofst = 192 if sys == uGNSS.QZS else 0
                svid_t.append(prn-ofst)

            mask_sat = self.encode_mask(svid_t, 64)
            nid_ = self.integ.nid[sys]
            # network id DFi071
            # GNSS satellite mask DFi009
            bs.pack_into('u8u64', msg, i, nid_, mask_sat)
            i += 72

            for f in flag.values():
                # integrity flag DFi068
                bs.pack_into('u2', msg, i, f)
                i += 2

        return i

    def encode(self, msg):
        """ encode RTCM messages """
        i = 24
        bs.pack_into('u12', msg, i, self.msgtype)
        i += 12

        if self.msgtype in (11, 12, 13):  # SSR integrity (test)
            i = self.encode_integrity_ssr(msg, i)

        return i
