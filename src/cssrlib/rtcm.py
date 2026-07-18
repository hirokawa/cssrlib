"""
RTCM 3 decoder

[1] RTCM Standard 10403.4 with Amendment 1
    Differential GNSS (Global Navigation Satellite Systems)
    Services - Version 3, 2024
[2] RTCM Standard 13400.00 (Working Draft)
    Integrity for High Accuracy GNSS based Applications
    - Version 0.91, 2025
[3] IGS SSR Format version 1.00, 2020
[4] RTCM SSR Standard, Draft version 0.9.5 Amendment 2, 2026

@author Rui Hirokawa

"""

import numpy as np
import struct as st
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, prn2sat, sCType, cssre
from cssrlib.gnss import uGNSS, sat2id, gpst2time, timediff, time2str, sat2prn
from cssrlib.gnss import uTYP, uSIG, rSigRnx, bdt2time, bdt2gpst, glo2time
from cssrlib.gnss import time2bdt, gpst2bdt, rCST, time2gpst, utc2gpst, timeadd
from cssrlib.gnss import gtime_t, sys2str, gpst2utc, sys2char
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
    SBS_EPH = 1017

    # SC-134 test messages
    INTEG_MIN = 2000
    INTEG_EXT = 2005
    INTEG_EXT_SIS = 2006
    INTEG_PRI_AREA = 2007
    INTEG_EXT_AREA = 2008
    INTEG_QUALITY = 2051
    INTEG_CNR = 2091
    INTEG_VMAP = 2071
    INTEG_MMAP = 2072
    INTEG_SSR = 2011
    INTEG_SSR_IONO = 2012
    INTEG_SSR_TROP = 2013

    # RTCM SSR test messages
    SSR_META = 60
    SSR_GRID = 61
    SSR_SATANT = 80
    SSR_TROP = 95
    SSR_STEC = 96
    SSR_HCLK = 46
    SSR_CBIAS = 68
    SSR_PBIAS = 85
    SSR_PBIAS_EX = 90


class satAntCorr():
    """ class for satellite antenna corrections """

    def __init__(self, nadc, naddc):
        self.nadc = nadc
        self.naddc = naddc


class Integrity():
    """ class for integrity information in SC-134 """

    sys_tbl = {0: uGNSS.GPS, 1: uGNSS.GLO, 2: uGNSS.GAL, 3: uGNSS.BDS,
               4: uGNSS.QZS, 5: uGNSS.IRN}

    # Mean Failure Duration (MFD) of the augmentation system indicator (DFi034)
    mfd_s_t = [5, 10, 30, 60, 120, 600, 900, 1800, 3600, 7200, 10800, 21600,
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
    mfd_c_t = [0, 5, 10, 30, 60, 90, 300, 600, 1800, 3600, 7200, 14400, 28800,
             57600, 115200, -1]
    # correlation time of pseudorange error indicator (DFi044)
    tau_c_t = [0, 5, 10, 20, 30, 45, 60, 120, 180, 300, 600, 1200, 3600, 7200,
               14400, -1]
    # correlation time of carrier phase error indicator (DFi045)
    tau_p_t = [0, 2, 5, 10, 20, 30, 45, 60, 120, 180, 240, 300, 600, 1800,
               7200, -1]
    # probability indicator (DFi035, DFi046, DFi047, DFi048, DFi049)
    P_t = [10**-11, 10**-10, 10**-9, 10**-8, 10**-7, 10**-6.2, 10**-6,
           10**-5.2, 10**-5, 10**-4.2, 10**-4, 10**-3.3, 10**-3, 10**-2.3,
           10**-2, -1]
    # TTT_comm indicator DFi050
    # ttt_comm_t = 0.1*(i+1), 127 is not provided
    # Bound on rate of change of PR error standard deviation DFi120
    sig_cd_t = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3, 6e-3, 1e-2, 3e-2,
                6e-2, 0.10, 0.50, 5.0, 0, -1]
    # Bound on rate of change in phase error standard deviation DFi121
    sig_pd_t = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 4e-4, 1e-3, 4e-3,
                0.01, 0.04, 0.10, 0.50, 5.0, -1]
    # Bounding sigma on phase-range-rate error DFi122
    sig_b_pd_t = [1e-3, 2e-3, 5e-3, 7.5e-3, 0.010, 0.014, 0.018, 0.025, 0.050,
                 0.10, 0.15, 0.20, 0.50, 1.0, 20.0, -1]
    # Phase-range-rate error rate integrity index DFi123
    b_pd_t = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3,
                0.01, 0.1, 1.0, 10.0, 0, 0, -1]
    # bounding parameter in iono, tropo error indicator DFi129, DFi131
    b_datm_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # bounding pseudorange bias rate error indicator DFi125
    b_rsdsa_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # bounding rate of change in satellite orbit/clock error indicator DFi127
    ocr_t = [5e-4, 1e-3, 2e-3, 4e-3, 0.01, 0.02, 0.05, -1]
    # residual iono error stdev for L1 GPS DFi128
    sig_ion_t = [5e-3, 1e-2, 2e-2, 5e-2, 0.10, 0.25, 1.0, -1]
    # residual trop error stdev for L1 GPS DFi130
    sig_trp_t = [0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.125, -1]
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
    # Time Parameter Degradation Factor DFi138
    df_t = [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 1, -1]
    # Extended Area Spatial Parameter Degradation Factor DFi139
    exs_df_t = [1.0, 1.1, 1.3, 1.6, 2.0, 3.0, 5.0, -1]
    # Augmentation IR Degradation Factor DFi140
    f_ir_t = [1.0, 10, 1e2, 1e3]
    # Augmentation TTD Degradation Factor DFi141
    f_ttd_t = [1.0,1.25,2.0,4.0]
    # GNSS Signal Modulation Mask Table 8-45
    mod_t = ['BPSK(1)', 'BPSK(5)', 'BPSK(10)', 'BOC(1,1)', 'CBOC(6,1,1/11)',
             'AltBOC(15,10)', '', '']

    atype_t = {0:"Validity Radius", 1:"Area Points"}
    #
    mm_model_t = ['GMM', 'Mats Brenner', 'Jahn']
    
    # DFi052 Satellite Integrity Fault Source
    fault_src_t = ['satellite orbit','satellite clock','atmospheric','other']

    # Table 8 16 GNSS Frequency Indexes
    frq_idx = {
        uGNSS.GPS: ['L1','L2', 'L5'],
        uGNSS.GLO: ['L1','L2', 'L3'],
        uGNSS.GAL: ['E1','E5a', 'E5b', 'E5 AltBOC', 'E6'],
        uGNSS.BDS: ['B1I','B2I', 'B3I','B1C','B2a','B2b'],
        uGNSS.QZS: ['L1','L2', 'L5','L6'],
        uGNSS.IRN: ['L1','L5', 'S'],
    }

    # Validity Period DFi065
    vp_tbl = [1, 2, 5, 10, 15, 30, 60, 120, 240,
              300, 600, 900, 1800, 3600, 7200, 10800]

    mt_integ_t = {2000: "Minimum Integrity", 
                  2005: "Extended Integrity (Service Levels and Overbounding)", 
                  2006: "Extended Integrity (SIS and Local Error)", 
                  2007: "Primary Service Area",
                  2008: "Extended Service Area",
                  2011: "SIS SSR Integrity",
                  2051: "Quality Indicator",
                  2091: "AGC SIS Monitoring",
                  2071: "Satellite Visibility Map",
                  2072: "Multipath Map"}

    def __init__(self):
        self.sys_r_tbl = {self.sys_tbl[s]: s for s in self.sys_tbl.keys()}
        self.vp_r_tbl = {s: k for k, s in enumerate(self.vp_tbl)}

        self.pid = 0  # augmentation provider id DFi027 (0-4095)
        self.vp = 0  # validity period DFi065 (0-15)
        self.udi = 0  # update rate interval DFi067 (b16)

        # placeholder for RTCM SSR
        self.pidssr = 0  # SSR provider ID
        self.sidssr = 0  # SSR solution type
        self.iodssr = 0  # SSR iod
        self.inet = 0

        self.tow = 0
        self.iod_sys = {}  # issue of GNSS satellite mask DFi010 (b2)
        self.sts = {}  # constellation integrity status DFi029 (b16)
        self.mst = {}  # constellation monitoring status DFi030 (b16)
        self.src = {}

        self.nid = {}
        self.flag = {}
        self.freq = {}  # GNSS Frequency mask (b8)
        self.iod_freq = {}  # issue of GNSS signal mask DFi012 (b2)
        self.ists = {}  # signal integrity status DFi031 (b32)
        self.msts = {}  # signal monitoring status DFi032 (b32)

        self.pos_v = None # service area (polygon/circle)

        self.Paug = 0  # Augmentation System Probability Falut DFi049

        self.tau_c = {}
        self.sig_c = {}
        self.tau_p = {}
        self.sig_p = {}
        self.Pa_cc = {}
        self.Pa_cp = {}
        self.Pa_sc = {}  # Single Satellite Code Message Fault Probability DFi048
        self.Pa_sp = {}  # Single Satellite Phase Message Fault Probability DFi046        
        self.sig_ob_c = {}  # Ovebounding stdev of Code DFi038
        self.sig_ob_p = {}  # Ovebounding stdev of Phase DFi038
        self.b_ob_c = {}  # Overbounding bias of long-term CP bias DFi039
        self.b_ob_p = {}  # Overbounding bias of long-ter PR bias DFi037
          
        self.iod_m = {} # IOD GNSS Satellite mask DFi010

        self.sig_cd = {} # Bound on rate of change of PR error stdev DFi120
        self.sig_pd = {} # Bound on rate of change in phase error stdev DFi121

        self.cbr = {} # Nominal pseudorange bias rate of change DFi125
        self.sig_dp = {} # Bounding sigma on phase-range-rate error DFi122
        self.idx_dp = {} # Phase-range-rate error rate integrity index DFi123

        self.ocr = {} # Orbit and clock error rate integrity parameter DFi127
        self.sig_ion = {} # Residual ionospheric error stdev DFi128
        self.idx_ion = {} # Index of bounding parameter on change in iono error DFi129
        self.sig_trp = {} # Residual tropospheric error stdev DFi130
        self.idx_trp = {} # Index of bounding parameter on change in tropo error  DFi131

        self.cnr = {}
        self.agc_t = {}
        self.agc = {}

        self.nsys = 0
        self.nsat = {}
        self.sys_t = None
        self.sat = {}
        self.mask_sys = 0

        self.mm_param = None

        self.sat_e = [] # excluded satellites by SSR integrity flag

    def cnr_lvl(self, idx):
        """ get index to C/N level [dB-Hz] """
        cnr = -1
        if idx>0 and idx<20:
            cnr = idx-1
        elif idx<245:
            cnr = 19+(idx-20)*0.2
        else:
            cnr = 64+(idx-245)
        
        return cnr

    def agc_lvl(self, idx):
        """ get index to AGC level [dB] """
        agc = 'N/A'
        if idx>0:
            agc = idx-128
        return agc
    
    def decode_sol_t(self, mask):
        """ decode Service Provider Solution Type"""
        sol_t = ['ARAIM','SBAS','GBAS','DGNSS','RTK','NRTK/VRS','PPP','PPP-AR',
                 'PPP-RTK','','','','','','','other']
        s = ""
        for k in range(15):
            if (mask>>(15-k-1)) & 1:
                s += sol_t[k]
        return s

class rtcmUtil:
    """ class to define common parameters and utilities for RTCM """

    msm_t = {
        uGNSS.GPS: 1071, uGNSS.GLO: 1081, uGNSS.GAL: 1091,
        uGNSS.SBS: 1101, uGNSS.QZS: 1111, uGNSS.BDS: 1121
    }

    ssr_t = {
        uGNSS.GPS: 1057, uGNSS.GLO: 1063, uGNSS.GAL: 1240,
        uGNSS.QZS: 1246, uGNSS.SBS: 1252, uGNSS.BDS: 1258
    }

    eph_t = {
        uGNSS.GPS: 1019, uGNSS.GLO: 1020, uGNSS.BDS: 1042,
        uGNSS.QZS: 1044, uGNSS.GAL: 1046
    }

    sc_t = {sCSSR.ORBIT: sCType.ORBIT, sCSSR.CLOCK: sCType.CLOCK,
            sCSSR.MASK: sCType.MASK, sCSSR.CBIAS: sCType.CBIAS,
            sCSSR.PBIAS: sCType.PBIAS, sCSSR.URA: sCType.URA,
            sCSSR.GRID: sCType.TROP, sCSSR.STEC: sCType.STEC,
            sCSSR.COMBINED: sCType.OC,
            sRTCM.SSR_PBIAS: sCType.PBIAS,
            sRTCM.SSR_TROP: sCType.TROP,
            sRTCM.SSR_STEC: sCType.STEC,
            }

    ssrtype_t = {
        1057: (uGNSS.GPS, sCType.ORBIT),
        1058: (uGNSS.GPS, sCType.CLOCK),
        1059: (uGNSS.GPS, sCType.CBIAS),
        1060: (uGNSS.GPS, sCType.OC),
        1061: (uGNSS.GPS, sCType.URA),
        1062: (uGNSS.GPS, sCType.HCLOCK),
        1063: (uGNSS.GLO, sCType.ORBIT),
        1064: (uGNSS.GLO, sCType.CLOCK),
        1065: (uGNSS.GLO, sCType.CBIAS),
        1066: (uGNSS.GLO, sCType.OC),
        1067: (uGNSS.GLO, sCType.URA),
        1068: (uGNSS.GLO, sCType.HCLOCK),
        1240: (uGNSS.GAL, sCType.ORBIT),
        1241: (uGNSS.GAL, sCType.CLOCK),
        1242: (uGNSS.GAL, sCType.CBIAS),
        1243: (uGNSS.GAL, sCType.OC),
        1244: (uGNSS.GAL, sCType.URA),
        1245: (uGNSS.GAL, sCType.HCLOCK),
        1264: (uGNSS.NONE, sCType.VTEC),
        1265: (uGNSS.GPS, sCType.PBIAS),
        1266: (uGNSS.GLO, sCType.PBIAS),
        1267: (uGNSS.GAL, sCType.PBIAS),
        1268: (uGNSS.QZS, sCType.PBIAS),
        1269: (uGNSS.SBS, sCType.PBIAS),
        1270: (uGNSS.BDS, sCType.PBIAS),
        41: (uGNSS.GLO, sCType.ORBIT),
        42: (uGNSS.GLO, sCType.CLOCK),
        43: (uGNSS.GLO, sCType.CBIAS),
        44: (uGNSS.GLO, sCType.OC),
        45: (uGNSS.GLO, sCType.URA),
        46: (uGNSS.GLO, sCType.HCLOCK),
        60: (uGNSS.NONE, sCType.META),
        61: (uGNSS.NONE, sCType.GRID),
        62: (uGNSS.GAL, sCType.ORBIT),
        63: (uGNSS.BDS, sCType.ORBIT),
        64: (uGNSS.QZS, sCType.ORBIT),
        65: (uGNSS.GAL, sCType.CLOCK),
        66: (uGNSS.BDS, sCType.CLOCK),
        67: (uGNSS.QZS, sCType.CLOCK),
        68: (uGNSS.GAL, sCType.CBIAS),
        69: (uGNSS.BDS, sCType.CBIAS),
        70: (uGNSS.QZS, sCType.CBIAS),
        71: (uGNSS.GAL, sCType.OC),
        72: (uGNSS.BDS, sCType.OC),
        73: (uGNSS.QZS, sCType.OC),
        74: (uGNSS.GAL, sCType.URA),
        75: (uGNSS.BDS, sCType.URA),
        76: (uGNSS.QZS, sCType.URA),
        77: (uGNSS.GAL, sCType.HCLOCK),
        78: (uGNSS.BDS, sCType.HCLOCK),
        79: (uGNSS.QZS, sCType.HCLOCK),
        80: (uGNSS.GPS, sCType.SATANT),
        81: (uGNSS.GLO, sCType.SATANT),
        82: (uGNSS.GAL, sCType.SATANT),
        83: (uGNSS.BDS, sCType.SATANT),
        84: (uGNSS.QZS, sCType.SATANT),
        85: (uGNSS.GPS, sCType.PBIAS),
        86: (uGNSS.GLO, sCType.PBIAS),
        87: (uGNSS.GAL, sCType.PBIAS),
        88: (uGNSS.BDS, sCType.PBIAS),
        89: (uGNSS.QZS, sCType.PBIAS),
        90: (uGNSS.GPS, sCType.PBIAS_EX),
        91: (uGNSS.GLO, sCType.PBIAS_EX),
        92: (uGNSS.GAL, sCType.PBIAS_EX),
        93: (uGNSS.BDS, sCType.PBIAS_EX),
        94: (uGNSS.QZS, sCType.PBIAS_EX),
        95: (uGNSS.NONE, sCType.TROP),
        96: (uGNSS.GPS, sCType.STEC),
        97: (uGNSS.GLO, sCType.STEC),
        98: (uGNSS.GAL, sCType.STEC),
        99: (uGNSS.BDS, sCType.STEC),
        100: (uGNSS.QZS, sCType.STEC),
    }

    ssrt_t = {
        sCType.ORBIT: "ORBIT",
        sCType.CLOCK: "CLOCK",
        sCType.OC: "ORBIT/CLOCK",
        sCType.URA: "URA",
        sCType.CBIAS: "CODE-BIAS",
        sCType.PBIAS: "PHASE-BIAS",
        sCType.META: "META",
        sCType.HCLOCK: "HIGH-RATE CLOCK",
        sCType.PBIAS_EX: "PHASE-BIAS EXTENDED",
        sCType.SATANT: "SATELLITE ANTENNA",
        sCType.TROP: "TROP",
        sCType.STEC: "STEC",
        sCType.VTEC: "VTEC",
        sCType.GRID: "GRID",
        }
    
    igsssrt_t = {
        sCSSR.VTEC: "VTEC",
        sCSSR.ORBIT: "ORBIT",
        sCSSR.CLOCK: "CLOCK",
        sCSSR.COMBINED: "ORBIT/CLOCK",
        sCSSR.HCLOCK: "HIGH-RATE CLOCK",
        sCSSR.CBIAS: "CODE-BIAS",
        sCSSR.PBIAS: "PHASE-BIAS",        
        sCSSR.URA: "URA",        
        }
    
    def mt2sct(self, s):
        """ SSR Message type to SCType """
        return self.ssrtype_t[s]

    def sct2mt(self, sct: sCType, sys: uGNSS = uGNSS.NONE):
        """ SCType to SSR Message type """
        for k, v in self.ssrtype_t.items():
            if v[0] == sys and v[1] == sct:
                return k
        return -1

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
            elif msgtype == 1264:
                return uGNSS.NONE
            elif msgtype >= 1265 and msgtype < 1271:  # proposed phase bias
                tbl_t = {1265: uGNSS.GPS, 1266: uGNSS.GLO, 1267: uGNSS.GAL,
                         1268: uGNSS.QZS, 1269: uGNSS.SBS, 1270: uGNSS.BDS}
                return tbl_t[msgtype]
            # RTCM SSR test messages
            elif msgtype >= 41 and msgtype < 47:  # OBC
                return uGNSS.GLO
            elif msgtype in [62, 65, 68, 71, 74, 77]:  # OBC
                return uGNSS.GAL
            elif msgtype in [63, 66, 69, 72, 75, 78]:  # OBC
                return uGNSS.BDS
            elif msgtype in [64, 67, 70, 73, 76, 79]:  # OBC
                return uGNSS.QZS
            elif msgtype >= 80 and msgtype < 85:  # proposed satellite antenna
                tbl_t = {80: uGNSS.GPS, 81: uGNSS.GLO, 82: uGNSS.GAL,
                         83: uGNSS.BDS, 84: uGNSS.QZS}
                return tbl_t[msgtype]
            elif msgtype >= 85 and msgtype < 90:  # proposed phase bias
                tbl_t = {85: uGNSS.GPS, 86: uGNSS.GLO, 87: uGNSS.GAL,
                         88: uGNSS.BDS, 89: uGNSS.QZS}
                return tbl_t[msgtype]
            elif msgtype >= 90 and msgtype < 95:  # proposed phase bias
                tbl_t = {90: uGNSS.GPS, 91: uGNSS.GLO, 92: uGNSS.GAL,
                         93: uGNSS.BDS, 94: uGNSS.QZS}
                return tbl_t[msgtype]
            elif msgtype >= 96 and msgtype < 101:  # regional iono
                tbl_t = {96: uGNSS.GPS, 97: uGNSS.GLO, 98: uGNSS.GAL,
                         99: uGNSS.BDS, 100: uGNSS.QZS}
            elif msgtype in [60, 61, 95]:  # METADATA, grid definition, trop
                return uGNSS.NONE
            elif msgtype in [11, 12, 13]:  # RTCM SC-134 test messages
                return uGNSS.NONE
            else:
                print(f"definition of {msgtype} is missing")
                # return uGNSS.NONE

            return tbl_t[msgtype]

    def rsig2code(self, rsig):
        """ convert rSigRnx to RTCM SSR code """
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
            4: uSIG.L4A,
            5: uSIG.L4B,
            6: uSIG.L6A,
            7: uSIG.L6B,
            10: uSIG.L3I,
            11: uSIG.L3Q,
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
            15: uSIG.L1A,
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
            17: uSIG.L6E,
            19: uSIG.L1E,  # TBD
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

        usig_tbl = usig_tbl_[rsig.sys]
        dic = {usig_tbl[s]: s for s in usig_tbl}

        return dic[rsig.sig]

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
            3: uSIG.L1E,
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

    def is_ssrtype(self, msgtype, tstmsg=False):
        """ check if the message type is MSM """
        for sys_ in self.ssr_t.keys():
            if msgtype >= self.ssr_t[sys_] and msgtype <= self.ssr_t[sys_]+6:
                return True
            if msgtype >= 1264 and msgtype <= 1270:  # VTEC, PBIAS (obsoleted)
                return True
            if tstmsg and msgtype >= 60 and msgtype <= 100:  # SSR test message
                return True
        return False

    def is_msmtype(self, msgtype):
        """ check if the message type is MSM """
        for sys_ in self.msm_t.keys():
            if msgtype >= self.msm_t[sys_] and msgtype <= self.msm_t[sys_]+6:
                return True
        return False

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

    def ssrtype(self, msgtype):
        """ get system and ssr type from message type """
        sys = uGNSS.NONE
        ssr = 0
        for sys_ in self.ssr_t.keys():
            if msgtype >= self.ssr_t[sys_] and msgtype < self.ssr_t[sys_]+6:
                sys = sys_
                ssr = msgtype-self.ssr_t[sys_]+1
                break
        return sys, ssr

    def sat2svid(self, sat):
        """ convert sat number to svid """
        sys, svid = sat2prn(sat)
        if sys == uGNSS.QZS:
            svid -= uGNSS.MINPRNQZS-1
        elif sys == uGNSS.SBS:
            svid -= uGNSS.MINPRNSBS-1
        return svid

    def svid2sat(self, sys, svid):
        """ convert svid to sat """
        prn = svid
        if sys == uGNSS.QZS:
            prn += uGNSS.MINPRNQZS-1
        elif sys == uGNSS.SBS:
            prn += uGNSS.MINPRNSBS-1
        return prn2sat(sys, prn)

    def adjustweek(self, week: int, tref: gtime_t):
        """ adjust week number considering reference time """
        week_, _ = time2gpst(tref)
        week_ref = (week_//1024)*1024
        return (week % 1024) + week_ref

    def sys2str(self, sys: uGNSS):
        """ convert system enum to string """
        gnss_t = {uGNSS.GPS: "GPS", uGNSS.GLO: "GLO", uGNSS.GAL: "GAL",
                  uGNSS.BDS: "BDS", uGNSS.QZS: "QZS", uGNSS.SBS: "SBAS",
                  uGNSS.IRN: "NAVIC"}
        if sys not in gnss_t:
            return ""
        return gnss_t[sys]


class rtcm(cssr, rtcmUtil):
    """ class to decode RTCM3 messages """

    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.len = 0
        self.monlevel = 1
        self.sysref = -1
        self.nsig_max = 4
        self.lock = {}
        self.mask_pbias = False

        self.pid = 0  # SSR Provider ID
        self.sid = 0  # SSR Solution Type
        self.inet = 0
        self.mi = False

        self.nrtk_r = {}

        self.glo_bias = None  # GLONASS receiver bias
        self.pos_arp = None  # receiver position
        self.ant_desc = "unknown"
        self.ant_id = ""
        self.ant_serial = ""
        self.rcv_type = ""
        self.firm_ver = "unknown"
        self.rcv_serial = ""

        self.antc = {}
        self.integ = Integrity()
        self.test_mode = False  # for interop testing in SC134
        
        self.mt_skip = [] # skip decode for the specified message types
        
        self.workaround_sc134 = False

    def ssig2rsig(self, sys: uGNSS, utyp=None, ssig=None):
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
            4: uSIG.L4A,
            5: uSIG.L4B,
            6: uSIG.L6A,
            7: uSIG.L6B,
            10: uSIG.L3I,
            11: uSIG.L3Q,
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
            15: uSIG.L1A,
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
            17: uSIG.L6E,
            19: uSIG.L1E,  # TBD
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
        
        if ssig is None:
            return len(usig_tbl)
        
        if ssig not in usig_tbl:
            print(f"sys={sys} sig={ssig} undefined.")
            return rSigRnx()
        return rSigRnx(sys, utyp, usig_tbl[ssig])

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

    def decode_head(self, msg, i, sys=uGNSS.NONE):
        """ decode the header of ssr message """
        
        if sys==uGNSS.GLO and self.msgtype != 4076:
            self.tod = bs.unpack_from('u17', msg, i)[0]     
            i += 17
            self.time = utc2gpst(glo2time(self.time, self.tod))
            if self.workaround_sc134:
                self.time = timeadd(self.time, -36) # work-around
        else:
            self.tow = bs.unpack_from('u20', msg, i)[0]   
            i += 20               
            self.time = gpst2time(self.week, self.tow)

        if self.subtype not in [sRTCM.SSR_PBIAS_EX]:
            udi, mi = bs.unpack_from('u4b1', msg, i)
            i += 5
            v = {'udi': udi, 'mi': mi}

        if self.subtype in (sCSSR.ORBIT, sCSSR.COMBINED):
            self.datum = bs.unpack_from('u1', msg, i)[0]
            i += 1

        iodssr, pid, sid = bs.unpack_from('u4u16u4', msg, i)
        i += 24

        v['iodssr'] = iodssr
        self.pid = pid
        self.sid = sid

        if self.subtype == sCSSR.PBIAS:
            ci, mw = bs.unpack_from('u1u1', msg, i)
            self.iexpb = 0
            i += 2
        elif self.subtype == sRTCM.SSR_PBIAS:
            # Satellite Yaw Information Indicator DF486
            # Extended Phase Bias Property ID DF+2
            v['iyaw'], self.iexpb = bs.unpack_from('b1u4', msg, i)
            i += 5
            v['iexpb'] = self.iexpb

        if self.subtype not in [sCSSR.VTEC, sRTCM.SSR_PBIAS_EX, sRTCM.SSR_STEC,
                                sRTCM.SSR_TROP]:
            nsat = bs.unpack_from('u6', msg, i)[0]
            v['nsat'] = nsat
            i += 6

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
        self.ddclk_n[k] = self.sval(ddclk, 21, 0.4e-3)
        self.dddclk_n[k] = self.sval(dddclk, 27, 4e-6)
        return i

    def decode_hclk_sat(self, msg, i, k, inet=0):
        """ decoder high-rate clock correction of cssr """
        hclk = bs.unpack_from('s22', msg, i)[0]
        i += 22
        self.dclk_n[k] = self.sval(hclk, 22, 0.1e-3)
        return i

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
            self.lc[inet].ddorb = {}

        self.iodssr = v['iodssr']
        self.udi[sCType.ORBIT] = v['udi']
        self.mi = v['mi']
        sat = []
        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_orb_sat(msg, i, k, sys)
            sat.append(sat_)
            self.lc[inet].iode[sat_] = self.iode_n[k]
            self.lc[inet].dorb[sat_] = self.dorb_n[k, :]
            self.lc[inet].ddorb[sat_] = self.ddorb_n[k, :]

            self.set_t0(inet, sat_, sCType.ORBIT, self.time)

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.iodssr_c[sCType.ORBIT] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.ORBIT)
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
            self.lc[inet].ddclk = {}
            self.lc[inet].dddclk = {}

        self.dclk_n = np.zeros(nsat)
        self.ddclk_n = np.zeros(nsat)
        self.dddclk_n = np.zeros(nsat)

        self.iodssr = v['iodssr']
        self.udi[sCType.CLOCK] = v['udi']
        self.mi = v['mi']

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_clk_sat(msg, i, k)
            self.lc[inet].dclk[sat_] = self.dclk_n[k]
            self.lc[inet].ddclk[sat_] = self.ddclk_n[k]
            self.lc[inet].dddclk[sat_] = self.dddclk_n[k]

            self.set_t0(inet, sat_, sCType.CLOCK, self.time)

        self.iodssr_c[sCType.CLOCK] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CLOCK)
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

        self.iodssr = v['iodssr']
        self.udi[sCType.CBIAS] = v['udi']
        self.mi = v['mi']

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            self.set_t0(inet, sat_, sCType.CBIAS, self.time)

            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5
            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
            if sat_ not in self.lc[inet].cbias:
                self.lc[inet].cbias[sat_] = {}

            if nsig > self.ssig2rsig(sys):
                break

            for j in range(nsig):
                sig, cb = bs.unpack_from('u5s14', msg, i)
                i += 19

                rsig = self.ssig2rsig(sys, uTYP.C, sig).str()
                self.lc[inet].cbias[sat_][rsig] = self.sval(cb, 14, 0.01)

        self.iodssr_c[sCType.CBIAS] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CBIAS)
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
            self.lc[inet].si = {}
            self.lc[inet].di = {}
            self.lc[inet].wl = {}

        self.iodssr = v['iodssr']
        self.udi[sCType.PBIAS] = v['udi']
        self.mi = v['mi']
        self.iyaw = v['iyaw'] if 'iyaw' in v.keys() else True

        inet = self.inet

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            self.set_t0(inet, sat_, sCType.PBIAS, self.time)

            nsig = bs.unpack_from('u5', msg, i)[0]
            i += 5

            if self.subtype != sRTCM.SSR_PBIAS or \
                    (self.subtype == sRTCM.SSR_PBIAS and v['iyaw']):
                yaw, dyaw = bs.unpack_from('u9s8', msg, i)
                yaw *= 1.0/256.0
                dyaw = self.sval(dyaw, 8, 1.0/8192.0)
                i += 17
                self.lc[inet].yaw[sat_] = yaw
                self.lc[inet].dyaw[sat_] = dyaw

            if sat_ not in self.sat_b:
                self.sat_b.append(sat_)
            if sat_ not in self.lc[inet].pbias:
                self.lc[inet].pbias[sat_] = {}
                self.lc[inet].si[sat_] = {}
                self.lc[inet].di[sat_] = {}
                self.lc[inet].wl[sat_] = {}

            if nsig > self.ssig2rsig(sys):
                break

            for j in range(nsig):
                # tracking mode: DF461
                # signal integer indicator: DF483
                sig, si = bs.unpack_from('u5b1', msg, i)
                i += 6

                rsig = self.ssig2rsig(sys, uTYP.L, sig).str()
                if self.subtype != sRTCM.SSR_PBIAS:  # IGS-SSR
                    wl = bs.unpack_from('u2', msg, i)[0]
                    i += 2
                    self.lc[inet].wl[sat_][rsig] = wl

                # signal discontinuity indicator: DF485
                # phase bias: DF482
                di, pb = bs.unpack_from('u4s20', msg, i)
                i += 24

                self.lc[inet].pbias[sat_][rsig] = self.sval(pb, 20, 1e-4)
                self.lc[inet].si[sat_][rsig] = si
                self.lc[inet].di[sat_][rsig] = di

        self.iodssr_c[sCType.PBIAS] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.PBIAS)
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
        self.ddclk_n = np.zeros(nsat)
        self.dddclk_n = np.zeros(nsat)

        if timediff(self.time, self.lc[inet].t0s[sCType.ORBIT]) > 0:
            self.nsat_n = 0
            self.sys_n = []
            self.sat_n = []
            self.lc[inet].dclk = {}
            self.lc[inet].ddclk = {}
            self.lc[inet].dddclk = {}
            self.lc[inet].iode = {}
            self.lc[inet].dorb = {}
            self.lc[inet].ddorb = {}

        self.iodssr = v['iodssr']
        self.udi[sCType.ORBIT] = v['udi']
        self.udi[sCType.CLOCK] = v['udi']
        self.mi = v['mi']

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
            self.lc[inet].ddorb[sat_] = self.ddorb_n[k, :]
            self.lc[inet].dclk[sat_] = self.dclk_n[k]
            self.lc[inet].ddclk[sat_] = self.ddclk_n[k]
            self.lc[inet].dddclk[sat_] = self.dddclk_n[k]

        self.nsat_n += nsat
        self.sys_n += [sys]*nsat
        self.sat_n += sat

        self.iodssr_c[sCType.ORBIT] = v['iodssr']
        self.iodssr_c[sCType.CLOCK] = v['iodssr']

        self.lc[inet].cstat |= (1 << sCType.ORBIT)
        self.lc[inet].cstat |= (1 << sCType.CLOCK)
        self.lc[inet].t0s[sCType.ORBIT] = self.time

        return i

    def decode_cssr_ura(self, msg, i, inet=0):
        """ decode RTCM URA message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        nsat = v['nsat']

        if timediff(self.time, self.lc[inet].t0s[sCType.URA]) > 0:
            self.lc[inet].ura = {}

        self.iodssr = v['iodssr']
        self.udi[sCType.URA] = v['udi']
        self.mi = v['mi']

        for k in range(nsat):
            i, sat_ = self.decode_sat(msg, i, sys)
            cls_, val = bs.unpack_from('u3u3', msg, i)
            i += 6
            self.lc[inet].ura[sat_] = self.quality_idx(cls_, val)
            self.set_t0(inet, sat_, sCType.URA, self.time)

        self.iodssr_c[sCType.URA] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.URA)
        self.lc[inet].t0s[sCType.URA] = self.time

        return i

    def decode_cssr_hclk(self, msg, i, inet=0):
        """ decode RTCM High-rate Clock Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        # if self.iodssr != v['iodssr']:
        #    return -1

        if timediff(self.time, self.lc[inet].t0s[sCType.HCLOCK]) > 0:
            self.lc[inet].hclk = {}

        self.iodssr = v['iodssr']
        self.udi[sCType.HCLOCK] = v['udi']
        self.mi = v['mi']

        for k in range(v['nsat']):
            i, sat_ = self.decode_sat(msg, i, sys)
            i = self.decode_hclk_sat(msg, i, k)
            self.lc[inet].hclk[sat_] = self.dclk_n[k]
            self.set_t0(inet, sat_, sCType.HCLOCK, self.time)

        self.iodssr_c[sCType.HCLOCK] = v['iodssr']
        self.lc[inet].cstat |= (1 << sCType.HCLOCK)
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
        self.lc[inet].cstat |= (1 << sCType.VTEC)
        self.lc[inet].t0[sCType.VTEC] = self.time
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
                self.subtype = sCSSR.HCLOCK
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
            time = utc2gpst(glo2time(self.time, tow))
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
            tow = bs.unpack_from('u'+str(sz), msg, i)[0]
            if sys == uGNSS.GLO:
                t = utc2gpst(glo2time(self.time, tow))
                if self.workaround_sc134:
                    t = timeadd(t, -36) # work-around on sc134
                return t
            else:
                self.tow = tow
                return gpst2time(self.week, self.tow)

        sys, nrtk = self.nrtktype(self.msgtype)
        if nrtk > 0:
            i, sys, time = self.decode_nrtk_time(msg, i)
            return time

        return False

    def out_log_ssr_clk(self, sys, flg_drift=True):
        """ output ssr clock correction to log file """
        self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                      f"SolID:{self.sid}\n")
        self.fh.write(" {:s}\t{:s}".format("SatID", "dclk [m]"))

        if flg_drift:
            self.fh.write("\t{:s}\t{:s}".format(
                "ddclk [mm/s]", "dddclk [mm/s^2]"))
        self.fh.write("\n")

        for k, sat_ in enumerate(self.lc[0].dclk.keys()):
            sys_, _ = sat2prn(sat_)
            if sys_ != sys:
                continue
            self.fh.write(" {:s}\t{:8.4f}".format(sat2id(sat_),
                                                  self.lc[0].dclk[sat_]))
            if flg_drift:
                self.fh.write("\t{:8.4f}\t{:8.4f}".format(
                    self.lc[0].ddclk[sat_], self.lc[0].dddclk[sat_]))
            self.fh.write("\n")

    def out_log_ssr_orb(self, sys, flg_vel=True):
        """ output ssr orbit correction to log file """
        self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                      f"SolID:{self.sid}\n")
        self.fh.write(f" Reference Datum: {self.datum}\n")
        self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}"
                      .format("SatID", "IODE", "R[m]", "A[m]", "C[m]"))
        if flg_vel:
            self.fh.write("\t{:s}\t{:s}\t{:s}".format("dR[mm/s]",
                                                      "dA[mm/s]", "dC[mm/s]"))
        self.fh.write("\n")
        for k, sat_ in enumerate(self.lc[0].dorb.keys()):
            sys_, _ = sat2prn(sat_)
            if sys_ != sys:
                continue
            self.fh.write(" {:s}\t{:3d}\t{:7.4f}\t{:7.4f}\t{:7.4f}".
                          format(sat2id(sat_),
                                 self.lc[0].iode[sat_],
                                 self.lc[0].dorb[sat_][0],
                                 self.lc[0].dorb[sat_][1],
                                 self.lc[0].dorb[sat_][2]))
            if flg_vel:
                self.fh.write("\t{:7.4f}\t{:7.4f}\t{:7.4f}".
                              format(self.lc[0].ddorb[sat_][0]*1e3,
                                     self.lc[0].ddorb[sat_][1]*1e3,
                                     self.lc[0].ddorb[sat_][2]*1e3))
            self.fh.write("\n")

    def out_log_ssr_ura(self, sys):
        """ output ssr URA to log file """
        self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                      f"SolID:{self.sid}\n")
        self.fh.write(" {:s}\t{:s}\n".format("SatID", "ura [m]"))

        for k, sat_ in enumerate(self.lc[0].ura):
            sys_, _ = sat2prn(sat_)
            if sys_ != sys:
                continue
            self.fh.write(" {:s}\t{:8.4f}\n".format(sat2id(sat_),
                                                    self.lc[0].ura[sat_]))

    def out_log_integ_head(self):
        self.fh.write(f" ASP Id: {self.integ.pid}\n")
        self.fh.write(f" Validity Period [s]: {self.integ.vp:4.1f}\n")
        self.fh.write(f" Update Interval [s]: {self.integ.udi:4.1f}\n")
        self.fh.write(f" Service Area ID: {self.integ.aid:d}\n")        

    def out_log_area_point(self):
        if self.integ.atype == 1: # lat/lon
            self.fh.write(f" Number of Area Points: {self.integ.npnt}\n")                
            self.fh.write(" Area Point (lat, lon):\n")
            for pos in self.integ.pos_v:
                self.fh.write(f" {pos[0]:15.9f} {pos[1]:15.9f}\n")
        else: # lat/lon/radius
            pos = self.integ.pos_v
            self.fh.write(" Area Point (lat, lon, radius): ") 
            self.fh.write(f" {pos[0]:15.9f} {pos[1]:15.9f} {pos[2]:10.0f}\n")

    def out_log(self, obs=None, eph=None, geph=None, seph=None):
        """ output ssr message to log file """
        sys = -1
        inet = self.inet
        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))

        if self.is_ssrtype(self.msgtype, True):
            sys = self.get_ssr_sys(self.msgtype)
            sys, type_ = self.ssrtype_t[self.msgtype]
            self.fh.write(f" Message Type: RTCM-SSR {self.ssrt_t[type_]} {sys2str(sys)}\n")

        if self.msgtype == 4076: # IGS-SSR
            self.fh.write(f" Message Type: IGS-SSR {self.igsssrt_t[self.subtype]} {sys2str(self.sysref)}\n")

        if sys > 0 and self.subtype not in [None, sRTCM.SSR_META, sRTCM.SSR_GRID]:
            j = self.sc_t[self.subtype]
            self.fh.write(f" Update Interval: {self.udint_t[self.udi[j]]}[s]")
            self.fh.write(f" MultiMsg: {self.mi}\n")

        if self.subtype == sCSSR.CLOCK:
            self.out_log_ssr_clk(sys)

        if self.subtype == sCSSR.ORBIT:
            self.out_log_ssr_orb(sys)

        if self.subtype == sCSSR.COMBINED:
            self.out_log_ssr_clk(sys)
            self.out_log_ssr_orb(sys)

        if self.subtype == sCSSR.URA:
            self.out_log_ssr_ura(sys)

        if self.subtype == sRTCM.SSR_META:
            self.fh.write(f" ProvID:{self.pid} SolID:{self.sid} " +
                          f"NumEnt:{self.nm}\n")

        if self.subtype == sRTCM.SSR_TROP:
            self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                          f"SolID:{self.sid} GridID:{self.gid}\n")
            self.fh.write(f" Residiaul Model Indicator: {self.rmi}\n")

            ah, bh, ch = self.lc[inet].maph
            aw, bw, cw = self.lc[inet].mapw
            ct = self.lc[inet].ct
            self.fh.write(f" a(h/w): {ah:7.4f}  {aw:7.4f}\n")
            self.fh.write(f" b(h/w): {bh:7.4f}  {bw:7.4f}\n")
            self.fh.write(f" c(h/w): {ch:7.4f}  {cw:7.4f}\n")
            self.fh.write(f" c00(h/w)[m]: {ct[0, 0]:7.4f}"
                          + f" {ct[1, 0]:7.4f}\n")
            self.fh.write(f" c10(h/w)[mm/deg]: {ct[0, 2]*1e3:7.4f}"
                          + f" {ct[1, 2]*1e3:7.4f}\n")
            self.fh.write(f" c01(h/w)[mm/deg]: {ct[0, 1]*1e3:7.4f}"
                          + f" {ct[1, 1]*1e3:7.4f}\n")

            if self.rmi:
                dth = self.lc[inet].dth
                dtw = self.lc[inet].dtw
                ng = self.lc[inet].ng
                ofst = self.lc[inet].ofst
                for k in range(ofst, ofst+ng):
                    self.fh.write(" res(d/w)[m]")
                    if dth is not None:
                        self.fh.write(f" {dth[k]:7.4f}")
                    else:
                        self.fh.write("      ")
                    if dtw is not None:
                        self.fh.write(f" {dtw[k]:7.4f}")
                    else:
                        self.fh.write("      ")
                    self.fh.write("\n")

        if self.subtype == sRTCM.SSR_STEC:
            self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                          f"SolID:{self.sid} GridID:{self.gid}\n")
            self.fh.write(f" Polynomial[TECU]: {self.pmi} (Grad:{self.pgi})\n")
            if self.pmi:
                for sat in self.lc[inet].sat_n:
                    self.fh.write(f"  {sat2id(sat):s}")
                    ci = self.lc[inet].ci[sat]
                    self.fh.write(f" c00 {ci[0]:8.4f}")
                    if self.lc[inet].stype[sat] > 0:
                        self.fh.write(f" c01 {ci[1]:7.4f} c10 {ci[2]:7.4f}")
                    self.fh.write("\n")
            self.fh.write(f" Residual[TECU]: {self.rmi}\n")
            if self.rmi:
                for sat in self.lc[inet].sat_n:
                    self.fh.write(f"  {sat2id(sat):s}")
                    dstec = self.lc[inet].dstec[sat]
                    ng = self.lc[inet].ng
                    ofst = self.lc[inet].ofst
                    for k in range(ofst, ofst+ng):
                        self.fh.write(f" {dstec[k]:7.4f}")
                    self.fh.write("\n")

        if self.subtype == sRTCM.SSR_GRID:
            self.fh.write(f" ProvID:{self.pid} GridID:{self.gid}" +
                          f" GridType:{self.gtype} Ofst:{self.ofst}\n")
            self.fh.write(" RefPos:{:6.3f} {:6.3f} {:6.3f}\n"
                          .format(self.pos0[0], self.pos0[1], self.pos0[2]))
            ng = self.dpos.shape[0]
            for k in range(ng):
                self.fh.write(" dpos: {:6.3f} {:6.3f} {:9.3f}\n".format(
                    self.dpos[k][0], self.dpos[k][1], self.dpos[k][2]))

        if self.subtype in [sCSSR.CBIAS, sCSSR.BIAS, sRTCM.SSR_CBIAS]:
            self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                          f"SolID:{self.sid}\n")
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "CBias[m]", "..."))
            for k, sat_ in enumerate(self.lc[0].cbias.keys()):
                sys_, _ = sat2prn(sat_)
                if sys_ != sys:
                    continue
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for sig in self.lc[0].cbias[sat_].keys():
                    self.fh.write("{:s}\t{:5.2f}\t"
                                  .format(sig, self.lc[0].cbias[sat_][sig]))
                self.fh.write("\n")

        if self.subtype in [sCSSR.PBIAS, sCSSR.BIAS, sRTCM.SSR_PBIAS]:
            self.fh.write(f" IODSSR:{self.iodssr} ProvID:{self.pid} " +
                          f"SolID: {self.sid}\n")
            self.fh.write(f" ExtPB ID: {self.iexpb} Yaw: {self.iyaw}\n")

            self.fh.write(" {:s}".format("SatID"))
            if self.iyaw:
                self.fh.write("\t{:s}\t{:s}".format("Yaw", "dYaw"))

            self.fh.write("\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SigID", "PBias[m]", "i", "cnt", "..."))
            for k, sat_ in enumerate(self.lc[inet].pbias.keys()):
                sys_, _ = sat2prn(sat_)
                if sys_ != sys:
                    continue
                self.fh.write(" {:s}\t".format(sat2id(sat_)))

                if self.iyaw:
                    self.fh.write("{:7.4f}\t{:7.4f}\t"
                                  .format(self.lc[inet].yaw[sat_],
                                          self.lc[inet].dyaw[sat_]))

                for sig in self.lc[inet].pbias[sat_].keys():
                    self.fh.write("{:s}\t{:7.4f}\t{:1d}\t{:2d}\t"
                                  .format(sig, self.lc[inet].pbias[sat_][sig],
                                          self.lc[inet].si[sat_][sig],
                                          self.lc[inet].di[sat_][sig]))
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

        if self.msgtype == 54 or self.msgtype in self.integ.mt_integ_t:  # RTCM SC134 integrity messages
            if self.msgtype == 54:
                self.fh.write(f" Message Type: {self.msgtype}\n")
            else:
                self.fh.write(f" Message Type: {self.integ.mt_integ_t[self.msgtype]} ({self.msgtype})\n")

            if self.subtype == sRTCM.INTEG_MIN: # MT2000
                self.out_log_integ_head()

                self.fh.write(" Issue of Sat Mask: ")
                for sys in self.integ.iod_sys:
                    self.fh.write(f" {sys2char(sys)}:{self.integ.iod_sys[sys]}")                
                self.fh.write("\n") 
                    
                self.fh.write(" Integrity Status:\n")
                for sys in self.integ.sts:
                    self.fh.write(f"  {sys2char(sys)}:"
                        f" {format(self.integ.sts[sys], '064b')}\n")
                self.fh.write(" Monitoring Status:\n")
                for sys in self.integ.mst:
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat in self.integ.freq[sys]:
                        self.fh.write(f" {sat2id(sat)}")
                    self.fh.write("\n") 
                        
                self.fh.write(" Integrity Fault Source:\n")
                for sys in self.integ.src:
                    msg = self.integ.fault_src_t[self.integ.src[sys]]
                    self.fh.write(f"  {sys2char(sys)}: {msg}\n")                    

                self.fh.write(" Frequency mask:\n")
                for sys in self.integ.freq:
                    self.fh.write(f"  {sys2char(sys)}:")
                    s = self.integ.frq_idx[sys]
                    for sat in self.integ.freq[sys]:
                        f = self.integ.freq[sys][sat]
                        self.fh.write(f" {sat2id(sat)}") 
                        for k in f:
                            self.fh.write(f" {s[k]}")  
                    self.fh.write("\n")  
                        
                self.fh.write(" IOD freq mask:\n")
                for sys in self.integ.iod_freq:
                    self.fh.write(f"  {sys2char(sys)}:")       
                    for sat in self.integ.iod_freq[sys]:
                        self.fh.write(
                            f" {sat2id(sat)} {self.integ.iod_freq[sys][sat]}")                           
                    self.fh.write("\n") 
                    
                self.fh.write(" Integrity Status:\n")
                for sys in self.integ.ists:
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat in self.integ.ists[sys]: 
                        s = format(self.integ.ists[sys][sat], '08b')
                        self.fh.write(f" {sat2id(sat)} {s}")
                    self.fh.write("\n")                     
                    
                self.fh.write(" Monitoring Status:\n")
                for sys in self.integ.msts:
                    self.fh.write(f"  {sys2char(sys)}:")    
                    for sat in self.integ.msts[sys]:
                        s = format(self.integ.msts[sys][sat], '08b')
                        self.fh.write(f" {sat2id(sat)} {s}")    
                    self.fh.write("\n") 
                    
            elif self.subtype == sRTCM.INTEG_EXT:  # MT2005
                self.out_log_integ_head()
            
                self.fh.write(f" Integrity Level: {self.integ.ilvl}\n")
                self.fh.write(f" TTT: {self.integ.ttt}\n")
                self.fh.write(f" Solution Type: {self.integ.decode_sol_t(self.integ.stype)}\n")

                self.fh.write(f" Paug: {self.integ.Paug}\n")
                self.fh.write(f" IOD param: {self.integ.iod_p}\n")
                self.fh.write(f" Integrity/Continuity Flag: {self.integ.fc}\n")

                self.fh.write(" Mean Failure Duration (MFD) of a Single Satellite Fault: ")
                for sys, s in self.integ.mfd_s.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.mfd_s_t[s]}")
                self.fh.write("\n")   
                self.fh.write(" Mean Failure Duration (MFD) of a Constellation Fault: ")
                for sys, s in self.integ.mfd_c.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.mfd_c_t[s]}")
                self.fh.write("\n")                 
                self.fh.write(" Correlation Time of Pseudorange Augmentation Message Error: ")
                for sys, s in self.integ.tau_c.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.tau_c_t[s]}")
                self.fh.write("\n") 
                self.fh.write(" Pseudorange stdev: ")
                for sys, s in self.integ.sig_c.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.sig_ob_c_t[s]}")
                self.fh.write("\n") 
                self.fh.write(" Correlation Time of Carrier Phase Augmentation Message Error: ")
                for sys, s in self.integ.tau_p.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.tau_p_t[s]}")
                self.fh.write("\n") 
                self.fh.write(" Carrier Phase stdev: ")
                for sys, s in self.integ.sig_p.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.sig_ob_p_t[s]}")
                self.fh.write("\n") 
                self.fh.write(" Multiple PR Augmentation Message Fault Probability: ")
                for sys, s in self.integ.Pa_cc.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.P_t[s]}") 
                self.fh.write("\n") 
                self.fh.write(" Multiple CP Augmentation Message Fault Probability: ")
                for sys, s in self.integ.Pa_cp.items():
                    self.fh.write(f"  {sys2char(sys)}: {self.integ.P_t[s]}")
                self.fh.write("\n") 

                self.fh.write(" Single PR Augmentation Message Fault Probability:\n")
                for sys, s in self.integ.Pa_sc.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.P_t[v]}")
                    self.fh.write("\n")
                self.fh.write(" Overbounding stdev of PR Augmentation Message Error under Fault-Free Scenario:\n")
                for sys, s in self.integ.sig_ob_c.items():
                    self.fh.write(f"  {sys2char(sys)}:")  
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_ob_c_t[v]}")
                    self.fh.write("\n")
                self.fh.write(" Single CP Augmentation Message Fault Probability:\n")
                for sys, s in self.integ.Pa_sp.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.P_t[v]}")
                    self.fh.write("\n")
                self.fh.write(" Overbounding Bias of the Long-Term PR Augmentation Message Bias Error under Fault-Free Scenario:\n")
                for sys, s in self.integ.b_ob_c.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_ob_c_t[v]}")
                    self.fh.write("\n") 
                self.fh.write(" Overbounding stdev of CP Augmentation Message Error under Fault-Free Scenario:\n")
                for sys, s in self.integ.sig_ob_p.items():
                    self.fh.write(f"  {sys2char(sys)}:")  
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_ob_p_t[v]}")
                    self.fh.write("\n")           
                self.fh.write(" Overbounding Bias of the Long-Term CP Augmentation Message Bias Error under Fault-Free Scenario:\n")
                for sys, s in self.integ.b_ob_p.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_ob_p_t[v]}")
                    self.fh.write("\n") 

            elif self.subtype == sRTCM.INTEG_EXT_SIS:  # MT2006
                self.out_log_integ_head()
                
                self.fh.write(f" IOD param: {self.integ.iod_ip}\n")
                self.fh.write(f" Time Correlation Integrity/Continuity Flag: {self.integ.fc}\n")
                
                b = self.integ.bias
                self.fh.write(f" Bounding inter-constellation bia: {self.integ.b_intc_t[b[0]]}\n")
                self.fh.write(f" Bounding inter-frequency bias: {self.integ.b_intf_t[b[1]]}\n")

                self.fh.write(" Issue of GNSS Satellite Mask:\n")
                for sys, s in self.integ.iod_m.items():
                    self.fh.write(f"  {sys2char(sys)}: {s}")                 
                self.fh.write("\n") 
                
                self.fh.write(" Bound on rate of change of PR error stdev:\n")
                for sys, s in self.integ.sig_cd.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_cd_t[v]}")
                    self.fh.write("\n") 

                self.fh.write(" Bound on rate of change in phase error stdev:\n")
                for sys, s in self.integ.sig_pd.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_pd_t[v]}")
                    self.fh.write("\n")                 

                self.fh.write(" Nominal pseudorange bias rate of change:\n")
                for sys, s in self.integ.cbr.items():
                    self.fh.write(f"  {sys2char(sys)}:")   
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_rsdsa_t[v]}")
                    self.fh.write("\n") 
                
                self.fh.write(" Bounding sigma on phase-range-rate error:\n")
                for sys, s in self.integ.sig_dp.items():
                    self.fh.write(f"  {sys2char(sys)}:")       
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_b_pd_t[v]}")
                    self.fh.write("\n") 

                self.fh.write(" Phase-range-rate error rate integrity index:\n")
                for sys, s in self.integ.idx_dp.items():
                    self.fh.write(f"  {sys2char(sys)}:")   
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_pd_t[v]}")
                    self.fh.write("\n") 

                self.fh.write(" Orbit/clock error rate integrity:\n")
                for sys, s in self.integ.ocr.items():
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.ocr_t[v]}")
                    self.fh.write("\n") 
                
                self.fh.write(" Residual ionospheric error stdev:\n")
                for sys, s in self.integ.sig_ion.items():
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_ion_t[v]}")
                    self.fh.write("\n") 
                
                self.fh.write(" Index of bounding parameter on change in iono error:\n")
                for sys, s in self.integ.idx_ion.items():
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_datm_t[v]}")
                    self.fh.write("\n") 
                
                self.fh.write(" Residual tropospheric error stdev:\n")
                for sys, s in self.integ.sig_trp.items():
                    self.fh.write(f"  {sys2char(sys)}:")
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.sig_trp_t[v]}")
                    self.fh.write("\n")
                
                self.fh.write(" Index of bounding parameter on change in tropo error:\n")
                for sys, s in self.integ.idx_trp.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f"  {sat2id(sat)} {self.integ.b_datm_t[v]}")
                    self.fh.write("\n") 
                
            elif self.subtype == sRTCM.INTEG_PRI_AREA:  # MT2007
                self.out_log_integ_head()

                self.fh.write(f" IOD area: {self.integ.iod_sa}\n")                
                self.fh.write(f" Service Area Type: {self.integ.atype_t[self.integ.atype]}\n")

                self.fh.write(f" Continuation Flag: {self.integ.cf}\n")
                self.fh.write(f" Seq: {self.integ.seq}\n")

                self.fh.write(f" Time Parameter Degradation Factor: {self.integ.df_t[self.integ.f_t]}\n")
                
                self.out_log_area_point()

            elif self.subtype == sRTCM.INTEG_EXT_AREA:  # MT2008
                self.out_log_integ_head()              
                
                self.fh.write(f" IOD area: {self.integ.iod_sa}\n")
                self.fh.write(f" Service Area Type: {self.integ.atype_t[self.integ.atype]}\n")
 
                self.fh.write(f" Continuation Flag: {self.integ.cf}\n")
                self.fh.write(f" Seq: {self.integ.seq}\n")

                self.fh.write(f" Augmentation IR Degradation Factor: {self.integ.f_ir_t[self.integ.f_ir]}\n")
                self.fh.write(f" Augmentation TTD Degradation Factor: {self.integ.f_ttd_t[self.integ.f_ttd]}\n")
                self.fh.write(f" Time Parameter Degradation Factor: {self.integ.df_t[self.integ.f_t]}\n")
                self.fh.write(f" Extended Area Spatial Parameter Degradation Factor: {self.integ.exs_df_t[self.integ.f_exs]}\n")

                self.out_log_area_point()
        
            elif self.subtype == sRTCM.INTEG_QUALITY:  # MT2051
                self.out_log_integ_head()
                
                self.fh.write(f" Network ID: {self.integ.nid}\n")
                self.fh.write(f" Quality Indicator [m]: {self.integ.qi}\n")

            elif self.subtype == sRTCM.INTEG_CNR:  # MT2091
                self.out_log_integ_head()
                
                self.fh.write(" CNR [dB-Hz]:\n")
                for sys, s in self.integ.cnr.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f" {sat2id(sat)}")
                        for sig, u in v.items():
                            self.fh.write(f" {self.integ.cnr_lvl(u)}")                            

                    self.fh.write("\n") 
                    
                self.fh.write(" AGC type:\n")
                for sys, s in self.integ.agc_t.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f" {sat2id(sat)}")
                        for sig, u in v.items():
                            self.fh.write(f" {u}")
                    self.fh.write("\n") 
                    
                self.fh.write(" AGC [dB]:\n")
                for sys, s in self.integ.agc.items():
                    self.fh.write(f"  {sys2char(sys)}:") 
                    for sat, v in s.items():
                        self.fh.write(f" {sat2id(sat)}")
                        for sig, u in v.items():
                            self.fh.write(f" {self.integ.agc_lvl(u)}")
                    self.fh.write("\n") 

            elif self.subtype == sRTCM.INTEG_VMAP:  # MT2071
                self.out_log_integ_head()
                
                self.fh.write(f" Number of Area Points: {self.integ.narea:}\n")
                self.fh.write(f" Continuation Flag: {self.integ.cf}\n")
                self.fh.write(f" Seq: {self.integ.seq}\n")
                
                self.fh.write(" Boundary Points (lat, long, alt, naz):\n")
                for k in range(self.integ.narea):
                    self.fh.write(" {:2d}\t{:12.9f}\t{:12.9f}\t{:4.0f}\t{:2d}\n".format
                                  (k+1, self.integ.pos[k, 0], self.integ.pos[k, 1],
                                   self.integ.pos[k, 2], self.integ.naz[k]))

                    self.fh.write(" Visibility mask (az, el) [deg]:\n")
                    for j in range(self.integ.naz[k]):
                        self.fh.write("  {:3.0f}\t{:3.0f}\n".format
                                      (self.integ.azel[k][j][0], self.integ.azel[k][j][1]))

            elif self.subtype == sRTCM.INTEG_MMAP:  # MT2072
                self.out_log_integ_head()
                
                model = self.integ.mm_model_t[self.integ.mm_id]
                self.fh.write(f" TOW (s): {self.integ.tow:9.3f}\n")
                
                self.fh.write(f" Number of Area Points: {self.integ.narea:}\n")
                self.fh.write(f" Multipath Model ID: {model}\n")
                self.fh.write(f" Continuation Flag: {self.integ.cf}\n")
                self.fh.write(f" Seq: {self.integ.seq}\n")
                
                self.fh.write(" Boundary Points (lat, long, alt), np:\n")
                for k in range(self.integ.narea):
                    self.fh.write(" {:2d}\t{:12.9f}\t{:12.9f}\t{:4.0f}\t{:2d}\n".format
                                  (k+1, self.integ.pos[k, 0], self.integ.pos[k, 1],
                                   self.integ.pos[k, 2], self.integ.np[k]))

                if self.integ.mm_id == 0:

                    self.fh.write(" Components (Prob, Exp, stdev):\n")
                    for k in range(self.integ.narea):
                        for j in range(self.integ.np[k]):
                            self.fh.write(" {:2d}\t{:1d}\t{:6.4f}\t{:7.2f}\t{:7.2f}\n".format
                                          (k+1, j+1, self.integ.mm_param[k][j][0],
                                           self.integ.mm_param[k][j][1],
                                           self.integ.mm_param[k][j][2]))
                elif self.integ.mm_id in [1, 2]:
                    n = 3 if self.integ.mm_id == 1 else 4
                    self.fh.write(
                        " Multipath Parameters (Modulation,a,b,c(,d)) :\n")
                    for k in range(self.integ.narea):
                        for j in range(self.integ.np[k]):
                            s = int(self.integ.mm_param[k][j][0])
                            self.fh.write(" {:2d}\t{:1d}\t{:10s}".format
                                          (k+1, j+1, self.integ.mod_t[s]))
                            for i in range(n):
                                self.fh.write("\t{:9.6f}".format
                                              (self.integ.mm_param[k][j][i+1]))
                            self.fh.write("\n")

            self.fh.flush()

        if self.subtype in (sRTCM.INTEG_SSR, sRTCM.INTEG_SSR_IONO,
                            sRTCM.INTEG_SSR_TROP):
            self.out_log_integ_head()            

            self.fh.write(f" SSR Provider ID: {self.integ.pidssr}\n")
            self.fh.write(f" SSR Solution ID: {self.integ.sidssr}\n")
            self.fh.write(f" IOD SSR: {self.integ.iodssr}\n")

            self.fh.write(" {:20s}{:04x}\n".format("Constellation Mask:",
                                                   self.integ.mask_sys))
            self.fh.write(" IOD GNSS Mask: ")
            for sys in self.integ.iod_sys.keys():
                self.fh.write(" {:8s}: {:1d}".format(
                    sys2str(sys), self.integ.iod_sys[sys]))
            self.fh.write("\n")

            for sys in self.integ.flag.keys():
                self.fh.write(" NIntegrity Flag: ")
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

        if sys == uGNSS.QZS:
            ofst = 193
        elif sys == uGNSS.SBS:
            ofst = 120
        else:
            ofst = 1

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
                    r[k] += bs.unpack_from('u10', msg, i)[0]*rCST.P2_10*rms
                i += 10
            if msm == 5 or msm == 7:
                for k in range(self.nsat):
                    v = bs.unpack_from('s14', msg, i)[0]
                    i += 14
                    rr[k] = self.sval(v, 14, 1.0)

        # signal part
        if msm != 2:
            sz = 15 if msm < 6 else 20
            scl = rCST.P2_24 if msm < 6 else rCST.P2_29
            for k in range(ncell):
                pr_ = bs.unpack_from('s'+str(sz), msg, i)[0]
                i += sz
                pr[k] = self.sval(pr_, sz, scl*rms)

        if msm > 1:
            sz = 22 if msm < 6 else 24
            scl = rCST.P2_29 if msm < 6 else rCST.P2_31
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
                    if sig[j] not in self.sig_n:
                        continue
                    idx = sig.index(sig[j])
                    ll_p = self.lock[sat_][j]
                    if (ll == 0 & ll_p != 0) | ll < ll_p:
                        obs.lli[k, idx] |= 1
                    if hf_[j] > 0:
                        obs.lli[k, idx] |= 3

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

    def decode_ssr_metadata(self, msg, i):
        """ decode SSR Metadata Message """
        # nm: nomber of ssr model/correction entries
        iodssr, pid, sid, nm = bs.unpack_from('u4u16u4u5', msg, i)
        i += 29

        self.iodssr = iodssr
        self.pid = pid
        self.sid = sid
        self.nm = nm  # number of entries

        if nm == 0:  # no metadata
            i += 7
            return i

        mpi = 0
        # Model/Correction Part

        # Model/Correction Type Indicator: DF+64
        # 1:satellite antenna PCV, 2:satellite antenna GDV
        # 3:solid earth tides, 4:ocean loading, 5:pole tides, 6:relatively
        # 7:GNSS BE reference

        # Model/Correction Application Indicator: DF+65 0:not applied,1:applied
        # Non-Default Model/Correction Indicator: DF+66 0:default,1:non default
        mt, ap, mi = bs.unpack_from('u5u1b1', msg, i)
        i += 7
        if mi:
            # non-default model/correction id DF+67
            # mci = bs.unpack_from('u3', msg, i)[0]
            i += 3
        iodmi = bs.unpack_from('b1', msg, i)[0]  # DF+68
        i += 1
        if iodmi:
            # DF+69 model/correction data IOD
            # iodm = bs.unpack_from('u6', msg, i)[0]
            i += 6
        if mi:
            # DF+070 1:additional model parameter present
            mpi = bs.unpack_from('b1', msg, i)[0]
            i += 1
        if mpi:
            # number of additional model parameter bits DF+071
            nb = bs.unpack_from('u8', msg, i)[0]
            i += 8
        if mi and mpi:
            # aditional model parameter DF+072
            # data = bs.unpack_from('u'+str(nb), msg, i)[0]
            i += nb

        if mt == 7:  # GNSS BE Reference
            gnss, be = bs.unpack_from('u4u4', msg, i)
            i += 8

        return i

    def decode_ssr_satant(self, msg, i):
        """ decode GNSS Satellite Antenna Message (E80,81,82,83,84) """
        sys = self.get_ssr_sys(self.msgtype)

        # Provider ID
        # Satellite Antenna IOD (DF+10)
        # Phase Center Information Infdicator (DF+11)
        # Group Delay Informatyion Indicator (DF+12)
        # Nadir Angle Dependent Corrections Indicator (DF+13)
        pid, iods, pci, gdi, ndi = bs.unpack_from('u16u6b1b1b1', msg, i)
        i += 25
        if ndi:
            # Maximum off-nadir angle (DF+14)
            # Nadir angle dependent corrections range extension (DF+15)
            emax, nde = bs.unpack_from('u5u4', msg, i)
            i += 9
        # Satellite Set Indicator (DF+16): set if True, else for single
        ssi, satmask = bs.unpack_from('b1u64', msg, i)
        i += 65

        svids, nsat = self.decode_mask(satmask, 64, 1)
        for svid in svids:
            sat = self.svid2sat(sys, svid)
            if sat not in self.antc:
                self.antc[sat] = {}
            # Frequency Set Indicator (DF+17) Set if True else Single
            # GNSS Frequency Mask (DF+18)
            fsi, freqmask = bs.unpack_from('b1u6', msg, i)
            i += 7
            freq_, nf = self.decode_mask(freqmask, 6, 0)
            for freq in freq_:  # Frequency part
                # Nadir correction indicator (DF+19)
                nci = bs.unpack_from('b1', msg, i)[0]
                i += 1
                if nci:  # nadir corrections
                    nadc_ = bs.unpack_from('s12', msg, i)[0]  # DF+20
                    i += 12
                    nadc = self.sval(nadc_, 12, 1e-3)  # [m]
                else:
                    nadc = None
                if ndi:  # nadir angle dependent corrections (PCVs) (DF+21)
                    naddc = np.zeros(nde+1)
                    for k in range(nde+1):
                        naddc_ = bs.unpack_from('s'+str(3+nde), msg, i)[0]
                        i += 3+nde
                        naddc[k] = self.sval(naddc_, 3+nde, 1e-3)  # [m]
                else:
                    naddc = None
                self.antc[sat][freq] = satAntCorr(nadc, naddc)
        return i

    def decode_ssr_grid(self, msg, i):
        """ decode SSR Grid Definition Message (E61) """
        self.pid, self.mi, gid, gtype = bs.unpack_from('u16b1u10u3', msg, i)
        i += 30

        self.gid = gid
        self.gtype = gtype
        self.inet = gid

        # """load grid coordinates from file """
        dtype0 = [('nid', 'i4'), ('gid', 'i4'),
                  ('lat', 'f8'), ('lon', 'f8'), ('alt', 'f8')]

        if gtype == 0:  # grid type 0
            lat0, lon0, alt0, ofst, npnt = bs.unpack_from(
                's18s19u10u12u8', msg, i)
            i += 67

            lat = self.sval(lat0, 18, 1e-3)
            lon = self.sval(lon0, 19, 1e-3)
            alt = self.sval(alt0, 10, 12.5)-1000.0
            self.grid = np.array([(gid, ofst, lat, lon, alt)], dtype=dtype0)
            self.dpos = np.zeros((npnt, 3))
            self.pos0 = np.array([lat, lon, alt])
            self.ofst = ofst

            for k in range(ofst, ofst+npnt):
                dlat, dlon, dalt = bs.unpack_from('s13s14s9', msg, i)
                i += 36
                dlat_s = self.sval(dlat, 13, 1e-3)
                dlon_s = self.sval(dlon, 14, 1e-3)
                dalt_s = self.sval(dalt, 9, 12.5)
                lat += dlat_s
                lon += dlon_s
                alt += dalt_s
                d = np.array([(gid, k+1, lat, lon, alt)], dtype=dtype0)
                self.grid = np.append(self.grid, d)
                self.dpos[k-ofst, :] = (dlat_s, dlon_s, dalt_s)

        elif gtype == 1:  # grid type 1
            lat0, lon0, ofst, npnt = bs.unpack_from('s18s19u12u8', msg, i)
            i += 57

            lat = self.sval(lat0, 18, 1e-3)
            lon = self.sval(lon0, 19, 1e-3)

            self.grid = np.array([(gid, ofst, lat, lon, 0)], dtype=dtype0)
            self.dpos = np.zeros((npnt, 3))
            self.pos0 = np.array([lat, lon, 0])
            self.ofst = ofst

            for k in range(npnt):
                dlat, dlon = bs.unpack_from('s13s14', msg, i)
                i += 27
                dlat_s = self.sval(dlat, 13, 1e-3)
                dlon_s = self.sval(dlon, 14, 1e-3)
                lat += dlat_s
                lon += dlon_s
                d = np.array([(gid, k, lat, lon, 0)], dtype=dtype0)
                self.grid = np.append(self.grid, d)
                self.dpos[k-ofst, :] = (dlat_s, dlon_s, 0)

        elif gtype == 2:  # grid type 2
            lat0_, lon0_, nrow, ncol, dlat_, dlon_, gma = bs.unpack_from(
                's18s19u6u6u9u10b1', msg, i)
            i += 69
            lat0 = self.sval(lat0_, 18, 1e-3)
            lon0 = self.sval(lon0_, 19, 1e-3)
            dlat = self.sval(dlat_, 9, 0.01)
            dlon = self.sval(dlon_, 10, 0.01)
            self.grid = np.array()

            if gma:
                latmask = bs.unpack_from('u'+str(nrow), msg, i)[0]
                i += nrow
                lonmask = bs.unpack_from('u'+str(ncol), msg, i)[0]
                i += ncol

                ilat, nlat = self.decode_mask(latmask, nrow, 0)
                ilon, nlon = self.decode_mask(lonmask, ncol, 0)
                n = nlat*nlon
                egpmask = bs.unpack_from('u'+str(n), msg, i)[0]
                i += n
                iex, nex = self.decode_mask(egpmask, n, 0)
            else:
                iex = []

            ii = ofst
            self.grid = np.array([], dtype=dtype0)
            for j in range(ncol):
                for k in range(nrow):
                    lat = lat0 + dlat*j
                    lon = lon0 + dlon*k
                    idx = nrow*j+k
                    if gma and idx not in iex:
                        continue
                    d = np.array([(gid, ii, lat, lon, 0)], dtype=dtype0)
                    self.grid = np.append(self.grid, d)
                    ii += 1

        return i

    def decode_ssr_pbias_ex(self, msg, i, inet=0):
        """ decode SSR Extended Satellite Phase Bias Message """

        """
        - linked with Phase bias messages with epoch time for given GNSS and
            id (DF+002).
        - widelane signal group indicator (DF484) shall be repeated for
            every satellite and signal in corresponding phase bias message.
        - the message shall be sent before corresponding phase bias message.

        - DF484:
            00 - no widelane group information for this signal
            10 - signal belongs to group one of widelane combinations
                 with integer property

        """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)

        # Extended Phase Bias Property ID
        iexpb = bs.unpack_from('u4', msg, i)[0]  # DF+002
        i += 4
        if iexpb != self.iexpb:
            return -1

        self.lc[inet].wl = {}
        for sat_ in self.lc[inet].pbias.keys():
            self.lc[inet].wl[sat_] = {}
            for rsig in self.lc[inet].pbias[sat_].keys():
                wl = bs.unpack_from('u2', msg, i)[0]
                i += 2
                self.lc[inet].wl[sat_][rsig] = wl

        return i

    def decode_ssr_trop(self, msg, i, inet=0):
        """ decode SSR Tropspheric Correction Message """
        i, v = self.decode_head(msg, i)
        # Grid ID: DF+022
        # Residual model indicator DF+009
        gid, rmi = bs.unpack_from('u10b1', msg, i)
        i += 11
        inet = gid

        # atmospheric model part
        dah, dbh, dch, daw, dbw, dcw = bs.unpack_from(
            's11s9s9s13s6s5', msg, i)
        i += 53
        ct00h, ct10h, ct01h = bs.unpack_from('s13s15s15', msg, i)
        i += 43
        ct00w, ct10w, ct01w = bs.unpack_from('s13s15s15', msg, i)
        i += 43

        # mapping function for  hydrostatic/wet term

        ah = self.sval(dah, 11, 2.5e-7)+0.00118
        bh = self.sval(dbh, 9, 5e-6)+0.00298
        ch = self.sval(dch, 9, 2e-4)+0.0682
        aw = self.sval(daw, 13, 1e-6)+0.000104
        bw = self.sval(dbw, 6, 2.5e-5)+0.0015
        cw = self.sval(dcw, 5, 2e-3)+0.048

        ct = np.zeros((2, 4))

        # hydrostatic
        ct[0, 0] = self.sval(ct00h, 13, 0.1e-3)+2.3  # offset c00 [m]
        # south-north gradient c01 [m/deg]
        ct[0, 1] = self.sval(ct01h, 15, 0.01e-3)
        # west-east gradient c10 [m/deg]
        ct[0, 2] = self.sval(ct10h, 15, 0.01e-3)

        # wet
        ct[1, 0] = self.sval(ct00w, 13, 0.1e-3)+0.252  # offset c00 [m]
        # south-north gradient c01    [m/deg]
        ct[1, 1] = self.sval(ct01w, 15, 0.01e-3)
        # west-east gradient c10 [m/deg]
        ct[1, 2] = self.sval(ct10w, 15, 0.01e-3)

        if rmi:  # residual information part
            ofst, ng, n, m = bs.unpack_from('u12u12u4u4', msg, i)
            i += 32

            rh = np.zeros(ofst+ng) if n > 0 else None
            rw = np.zeros(ofst+ng) if m > 0 else None

            for k in range(ofst, ofst+ng):
                if n > 0:
                    rh_ = bs.unpack_from(f's{n}', msg, i)[0]
                    i += n
                    # hydrostatic grid point residual [m]
                    rh[k] = self.sval(rh_, n, 0.1e-3)
                if m > 0:
                    rw_ = bs.unpack_from(f's{m}', msg, i)[0]
                    i += m
                    # wet grid point residual [m]
                    rw[k] = self.sval(rw_, m, 0.1e-3)

        # functional term parameters
        self.lc[inet].ct = ct
        # mapping function parameters
        self.lc[inet].maph = np.array([ah, bh, ch])
        self.lc[inet].mapw = np.array([aw, bw, cw])
        self.lc[inet].dth = rh
        self.lc[inet].dtw = rw

        self.lc[inet].ng = ng
        self.lc[inet].ofst = ofst

        self.iodssr = v['iodssr']
        self.rmi = rmi
        return i

    def decode_ssr_iono(self, msg, i, inet=0):
        """ decode SSR Regional Ionospheric Correction message (E96,E97,E98,E99,E100) """
        sys = self.get_ssr_sys(self.msgtype)
        i, v = self.decode_head(msg, i, sys)
        # Grid ID DF+22
        # Polynomial Model Indicator DF+96
        # Residual Model Indicator DF+099
        # Satellite mask DF394
        self.gid, self.pmi, self.rmi, satmask = bs.unpack_from(
            'u10b1b1u64', msg, i)
        i += 76
        # Zenith-mapped STEC Polynomial Gradient Indicator DF+60
        svid, nsat = self.decode_mask(satmask, 64, 1)
        sat = np.zeros(nsat, dtype=int)
        for k in range(nsat):
            # sat[k] = self.svid2sat(sys, svid[k], True)
            sat[k] = self.svid2sat(sys, svid[k])

        inet = self.gid
        self.lc[inet].sat_n = sat

        if self.pmi:  # polynomial model information part
            self.pgi = bs.unpack_from('b1', msg, i)[0]
            i += 1
            sz = 3 if self.pgi else 1
            self.lc[inet].ci = {}
            self.lc[inet].stype = {}
            for k in range(nsat):
                sat_ = sat[k]
                self.lc[inet].ci[sat_] = np.zeros(sz)
                c00 = bs.unpack_from('s17', msg, i)[0]
                i += 17
                self.lc[inet].ci[sat_][0] = self.sval(c00, 17, 0.01)
                if self.pgi:  # polynomial grandient indicator is true
                    self.lc[inet].stype[sat_] = 1

                    c01, c10 = bs.unpack_from('s18s18', msg, i)
                    i += 36
                    self.lc[inet].ci[sat_][1] = self.sval(c01, 18, 1e-3)
                    self.lc[inet].ci[sat_][2] = self.sval(c10, 18, 1e-3)
                else:
                    self.lc[inet].stype[sat_] = 0

        if self.rmi:  # residual model information part
            ofst, ng, rlen = bs.unpack_from('u12u12u5', msg, i)
            i += 29
            if rlen > 0:
                # Zenith-mapped STEC Grid Point Residual
                # Resolution Scale Factor Exponent
                sf = bs.unpack_from('u3', msg, i)[0]  # DF+073
                i += 3
                scl = 1e-3*(2**sf)
            else:
                scl = 1.0

            self.lc[inet].dstec = {}
            fmt = 's'+str(rlen)

            for k in range(nsat):
                sat_ = sat[k]
                self.lc[inet].dstec[sat_] = np.zeros(ofst+ng)
                for j in range(ofst, ofst+ng):
                    res = bs.unpack_from(fmt, msg, i)[0]  # DF+062
                    i += rlen
                    self.lc[inet].dstec[sat_][j] = self.sval(res, rlen, scl)

            self.lc[inet].ng = ng
            self.lc[inet].ofst = ofst

        return i

    def decode_integ_head(self, msg, i):
        """ decode common headder of SC-134 messages """
        tow, pid, vp, udi = bs.unpack_from('u30u12u4u16', msg, i)
        i += 62

        self.integ.tow = tow*1e-3
        self.integ.pid = pid # Augmentation Service Provider Id
        self.integ.vp = self.integ.vp_tbl[vp] # Validity Period DFi065 (0-15)
        self.integ.udi = udi*0.1  # update rate interval DFi067 (0.1) 
 
        self.tow = self.integ.tow        
        self.time = gpst2time(self.week, self.tow)       
 
        if self.msgtype != 2000:
            aid = bs.unpack_from('u10', msg, i)[0]
            i += 10
            self.integ.aid = aid  # service area id DFi075
        
        return i

    def decode_integrity_min(self, msg, i):
        """ RTCM SC-134 minimum integrity message (MT3,2000) """

        i = self.decode_integ_head(msg, i)
        
        # Constellation Mask DFi013
        # constellation integirity status DFi029
        # constellation monitoring status DFi030
        mask_sys, c_integ, c_status = bs.unpack_from('u16u16u16', msg, i)
        i += 48

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)

        iod_sys = {}
        sts_tbl = {}
        mst_tbl = {}
        src_tbl = {}

        iod_freq = {}
        freq = {}
        ists = {}
        msts = {}

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

            ofst = 193 if sys == uGNSS.QZS else 1
            prn_m, _ = self.decode_mask(mst, 64, ofst)

            iod_sys[sys] = iod
            sts_tbl[sys] = sts
            mst_tbl[sys] = prn_m
            src_tbl[sys] = src

            iod_freq[sys] = {}
            freq[sys] = {}
            ists[sys] = {}
            msts[sys] = {}
            for svid in svid_t:
                sat = self.svid2sat(sys, svid)

                # GNSS frequency mask DFi022
                f, _ = self.decode_mask(bs.unpack_from('u8', msg, i)[0], 8, 0)
                freq[sys][sat] = f
                i += 8
                # issue of GNSS frequency mask DFi023
                iod_freq[sys][sat] = bs.unpack_from('u2', msg, i)[0]
                i += 2
                # frequency integrity status DFi024
                ists[sys][sat] = bs.unpack_from('u8', msg, i)[0]
                i += 8
                # frequency monitoring status DFi025
                msts[sys][sat] = bs.unpack_from('u8', msg, i)[0]
                i += 8

            self.integ.iod_sys = iod_sys
            self.integ.sts = sts_tbl
            self.integ.mst = mst_tbl
            self.integ.src = src_tbl

            self.integ.freq = freq
            self.integ.iod_freq = iod_freq
            self.integ.ists = ists
            self.integ.msts = msts
        return i

    def decode_integrity_ext(self, msg, i):
        """ RTCM SC-134 extended integrity message
        service levels and overbounding parameters (MT4,2005) """

        i = self.decode_integ_head(msg, i)
        
        # Augmentation Integrity Level DFi026
        # TTTcomm DFi050
        # Service Provider Solution Type DFi004
        ilvl, ttt, stype = bs.unpack_from('u8u7u15', msg, i)
        i += 82

        self.integ.ilvl = ilvl
        self.integ.ttt = (ttt+1)*0.1
        self.integ.stype = stype

        # GNSS Constellation Mask DFi013
        mask_sys = bs.unpack_from('u16', msg, i)[0]
        i += 16

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)

        # Paug Bundling flag DFi066
        f_paug = bs.unpack_from('b1', msg, i)[0]
        i += 1
        if f_paug:
            # Paug Augmentation System Probability Falut DFi049
            self.integ.Paug = bs.unpack_from('u4', msg, i)[0]
            i += 4

        # Time correlation integrity/continuity flag DFi137
        #  0: DFi040-045 are valid only for integrity monitoring
        #  1: DFi040-045 are valid also for continuity monitoring
        # integrity parameter IOD DFi006
        self.integ.fc, self.integ.iod_p = bs.unpack_from('b1u6', msg, i)
        i += 7

        # constellation specific part

        self.integ.Pa_sc = {}
        self.integ.Pa_sp = {}
        self.integ.sig_ob_c = {}        
        self.integ.sig_ob_p = {}
        self.integ.b_ob_c = {}
        self.integ.b_ob_p = {}

        self.integ.mfd_s = {}
        self.integ.mfd_c = {}

        self.integ.tau_p = {}
        self.integ.tau_c = {}
        self.integ.sig_p = {}
        self.integ.sig_c = {}
        self.integ.tau_cp = {}
        self.integ.tau_cc = {}

        self.integ.Pa_cp = {}
        self.integ.Pa_cc = {}

        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            # Mean failure duration (MFD) of a single satellite fault DFi034
            # Mean failure duration (MFD) of a constellation fault DFi040
            mfd_s, mfd_c = bs.unpack_from('u4u4', msg, i)
            i += 8

            # Correlation Time of Pseudorange Augmentation Message Error DFi044
            # Gauss-Markov stdev of pseudorange DFi069
            # Correlation Time of Carrier Phase Augmentation Message Error
            #  DFi045
            # Gauss-Markov stdev of carrier-phase DFi070
            # Multiple Satellite Pseudorange Augmentation Message Fault
            # Probability DFi047
            # Multiple Satellite Carrier Phase Augmentation Message Fault
            # Probability DFi035
            tau_c, sig_c, tau_p, sig_p, Pa_cc, Pa_cp = bs.unpack_from(
                'u4u5u4u5u4u4', msg, i)
            i += 26

            # GNSS satellite mask DFi009
            mask_sat = bs.unpack_from('u64', msg, i)[0]
            i += 64

            self.integ.mfd_s[sys] = mfd_s
            self.integ.mfd_c[sys] = mfd_c

            self.integ.tau_p[sys] = tau_p
            self.integ.tau_c[sys] = tau_c

            self.integ.sig_p[sys] = sig_p
            self.integ.sig_c[sys] = sig_c

            self.integ.Pa_cp[sys] = Pa_cp
            self.integ.Pa_cc[sys] = Pa_cc

            self.integ.Pa_sc[sys] = {}
            self.integ.Pa_sp[sys] = {}
            self.integ.sig_ob_c[sys] = {}            
            self.integ.sig_ob_p[sys] = {}
            self.integ.b_ob_c[sys] = {}
            self.integ.b_ob_p[sys] = {}

            svid_t, _ = self.decode_mask(mask_sat, 64)

            # satellite-specific part
            for svid in svid_t:
                sat = self.svid2sat(sys, svid)
                
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
                Pa_sc, sig_ob_c, Pa_sp, b_ob_c, sig_ob_p, b_ob_p = \
                    bs.unpack_from('u4u5u4u4u5u4', msg, i)
                i += 26

                self.integ.Pa_sc[sys][sat] = Pa_sc
                self.integ.Pa_sp[sys][sat] = Pa_sp                
                self.integ.sig_ob_c[sys][sat] = sig_ob_c
                self.integ.sig_ob_p[sys][sat] = sig_ob_p
                self.integ.b_ob_c[sys][sat] = b_ob_c                
                self.integ.b_ob_p[sys][sat] = b_ob_p

        return i

    def decode_integrity_ext_sis_local(self, msg, i):
        """ RTCM SC-134 extended integrity message
        signal in space integrity and local error parameters (MT5, 2006) """

        i = self.decode_integ_head(msg, i)

        # integrity parameter IOD DFi006
        # GNSS Constellation Mask DFi013
        # Time correlation integrity/continuity flag DFi137
        iod_ip, mask_sys, fc = bs.unpack_from('u6u16u1', msg, i)[0]
        i += 23

        self.integ.iod_ip = iod_ip
        self.integ.fc = fc

        # Bounding inter-constellation bias error parameter DFi132
        # Bounding inter-frequency bias error parameter DFi136
        bias_ic, bias_if = bs.unpack_from('u3u3', msg, i)
        i += 6

        self.integ.bias = [bias_ic, bias_if]

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)

        # constellation-specific part
        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            # GNSS satellite mask DFi009
            # issue of GNSS satellite mask DFi010
            mask_sat, iod_mask = bs.unpack_from('u64u2', msg, i)
            i += 66

            self.integ.iod_m[sys] = iod_mask
            svid_t, _ = self.decode_mask(mask_sat, 64)

            self.integ.sig_pd[sys] = {}
            self.integ.sig_cd[sys] = {}
            self.integ.cbr[sys] = {}
            self.integ.sig_dp[sys] = {}
            self.integ.idx_dp[sys] = {}

            self.integ.ocr[sys] = {}
            self.integ.sig_ion[sys] = {}
            self.integ.idx_ion[sys] = {}
            self.integ.sig_trp[sys] = {}
            self.integ.idx_trp[sys] = {}

            for svid in svid_t:
                sat = self.svid2sat(sys, svid)

                # Bound on rate of change of PR error stdev DFi120
                # Bound on rate of change in phase error stdev DFi121
                # Nominal pseudorange bias rate of change DFi125
                # Bounding sigma on phase-range-rate error DFi122
                # Phase-range-rate error rate integrity index DFi123
                sig_pd, sig_cd, cbr, sig_dp, idx_dp = bs.unpack_from(
                    'u4u4u3u4u4', msg, i)
                i += 19

                # Orbit and clock error rate integrity parameter DFi127
                # Residual ionospheric error standard deviation DFi128
                # Index of bounding parameter on change in iono error DFi129
                # Residual tropospheric error standard deviation DFi130
                # Index of bounding parameter on change in tropo error DFi131
                ocr, sig_ion, idx_ion, sig_trp, idx_trp = bs.unpack_from(
                    'u3u3u3u3u3', msg, i)
                i += 15

                self.integ.sig_pd[sys][sat] = sig_pd
                self.integ.sig_cd[sys][sat] = sig_cd
                self.integ.cbr[sys][sat] = cbr
                self.integ.sig_dp[sys][sat] = sig_dp
                self.integ.idx_dp[sys][sat] = idx_dp

                self.integ.ocr[sys][sat] = ocr
                self.integ.sig_ion[sys][sat] = sig_ion
                self.integ.idx_ion[sys][sat] = idx_ion
                self.integ.sig_trp[sys][sat] = sig_trp
                self.integ.idx_trp[sys][sat] = idx_trp
        return i

    def decode_integrity_service_area(self, msg, i, ext=False):
        """ decode SC-134 Primary/Extended Service Area Parameters (MT2007/2008) """
        
        i = self.decode_integ_head(msg, i)
        
        atype, iod_sa, cf = bs.unpack_from('u2u6u1', msg, i)
        i += 9

        self.integ.atype = atype # service area type DFi056
        self.integ.iod_sa = iod_sa 
        self.integ.cf = cf # service area parameter continuation flag DFi041

        if not ext:
            # Time Parameter Degradation Factor (DFi138)
            self.integ.f_t = bs.unpack_from('u3', msg, i)[0]
            i += 3

        # multiple message sequence number DFi079
        self.integ.seq = bs.unpack_from('u5', msg, i)[0]
        i += 5
        
        if ext: # Extended Service Area
            f_ir, f_ttd, f_t, f_exs = bs.unpack_from('u2u2u3u3', msg, i)
            i += 10
    
            self.integ.f_ir = f_ir   # Augmentation IR Degradation Factor DFi140
            self.integ.f_ttd = f_ttd # Augmentation TTD Degradation Factor DFi141
            self.integ.f_t = f_t     # Time Parameter Degradation Factor DFi138
            self.integ.f_exs = f_exs # Extended Area Spatial Parameter Degradation Factor DFi139

        if atype == 1:  # Validity Area Parameters
            # number of area points DFi201
            npnt = bs.unpack_from('u8', msg, i)[0]
            i += 8
            self.integ.npnt = npnt
            self.integ.pos_v = np.zeros((npnt, 3))
            for k in range(npnt):
                # Area Point - Lat DFi202
                # Area Point - Lon DFi203
                lat_i, lon_i = bs.unpack_from('s34s35', msg, i)
                i += 69
                lat = self.sval(lat_i, 34, 1.1e-8)
                lon = self.sval(lon_i, 35, 1.1e-8)
                self.integ.pos_v[k, :] = [lat, lon, 0]

        elif atype == 0:  # Validity Radius Data
            # Area Point - Lat DFi202
            # Area Point - Lon DFi203
            # Validity Radius DF057
            lat_i, lon_i, r = bs.unpack_from('s34s35u20', msg, i)
            i += 89
            lat = self.sval(lat_i, 34, 1.1e-8)
            lon = self.sval(lon_i, 35, 1.1e-8)
            self.integ.pos_v = [lat, lon, r]
        return i

    def decode_integrity_quality(self, msg, i):
        """ RTCM SC-134 Message Quality Indicator (MT7, 2051) """

        i = self.decode_integ_head(msg, i)

        # network id DFi071
        # quality indicator mask DFi061
        nid, mask_q = bs.unpack_from('u8u8', msg, i)
        i += 16

        self.integ.nid = nid

        idx_q, np_ = self.decode_mask(mask_q, 8, ofst=0)
        self.integ.qi = np.zeros(np_)
        for k in range(np_):
            self.integ.qi[k] = bs.unpack_from('u8', msg, i)[0]*0.1
            i += 8
        return i

    def decode_integrity_cnr_acg(self, msg, i):
        """ RTCM SC-134 CNR/ACG SIS Monitoring Message (MT8, 2091) """

        i = self.decode_integ_head(msg, i)

        # constellation mask DFi013
        mask_c = bs.unpack_from('u16', msg, i)[0]
        i += 16

        # constellation specific part
        sys_t, nsys = self.decode_mask(mask_c, 16, ofst=0)

        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]

            self.integ.cnr[sys] = {}
            self.integ.agc_t[sys] = {}
            self.integ.agc[sys] = {}

            mask_s, iod_sm = bs.unpack_from('u64u2', msg, i)
            i += 66

            svid_t, _ = self.decode_mask(mask_s, 64)

            for svid in svid_t:
                sat = self.svid2sat(sys, svid)

                self.integ.cnr[sys][sat] = {}
                self.integ.agc_t[sys][sat] = {}
                self.integ.agc[sys][sat] = {}

                mask_f, iod_fm = bs.unpack_from('u8u2', msg, i)
                i += 10
                sig_t, nsig = self.decode_mask(mask_f, 8, ofst=0)

                for sig in sig_t:
                    # CNR carrier to noise ratio DFi133
                    cnr, agc_t, agc = bs.unpack_from('u8u1u8', msg, i)
                    i += 17
                    self.integ.cnr[sys][sat][sig] = cnr
                    self.integ.agc_t[sys][sat][sig] = agc_t
                    self.integ.agc[sys][sat][sig] = agc

        return i

    def decode_integrity_vmap(self, msg, i):
        """ RTCM SC-134 Satellite Visibility Map Message (MT9, 2071) """

        i = self.decode_integ_head(msg, i)

        # number of area points DFi201
        # Message Continuation Flag DFi021
        # Multiple Message Sequence Number DFi079
        narea, cf, seq = bs.unpack_from('u8u1u5', msg, i)
        i += 14

        self.integ.narea = narea
        self.integ.cf = cf
        self.integ.seq = seq
        
        self.integ.pos = np.zeros((narea, 3))
        self.integ.naz = np.zeros(narea, dtype=int)
        self.integ.azel = {}

        for k in range(narea):
            # Area Point - Lat DFi202
            # Area Point - Lon DFi203
            # Area Point - Height DFi204
            # Number of Azimuth Slices DFi205
            lat, lon, alt, naz = bs.unpack_from('s34s35s14u6', msg, i)
            i += 89

            self.integ.pos[k, :] = [lat*1.1e-8, lon*1.1e-8, alt]
            self.integ.naz[k] = naz

            azel = np.zeros((naz, 2))

            # Azimuth DFi206
            # Elevation Mask DFi208
            az = 0
            for j in range(naz):
                daz, mask_el = bs.unpack_from('u9u7', msg, i)
                i += 16
                az += daz
                azel[j, :] = [az, mask_el]

            self.integ.azel[k] = azel

        return i

    def decode_integrity_mmap(self, msg, i):
        """ RTCM SC-134 Multipath Map Message (MT10, 2072) """

        i = self.decode_integ_head(msg, i)

        # number of area points DFi201
        # multipath model ID DFi209: 0:GMM,1:MBM,2:JM
        narea, mm_id, self.integ.cf, self.integ.seq = bs.unpack_from(
            'u8u3u1u5', msg, i)
        i += 17

        self.integ.narea = narea
        self.integ.pos = np.zeros((narea, 3))
        self.integ.mm_id = mm_id

        self.integ.np = np.zeros(narea, dtype=np.int32)
        self.integ.mm_param = {}
        n = 4 if mm_id == 1 else 5

        for k in range(narea):
            if mm_id in [1, 2]:
                mask_s = bs.unpack_from('u8', msg, i)[0]
                i += 8
            # Area Point - Lat DFi202
            # Area Point - Lon DFi203
            # Area Point - Height DFi204
            lat, lon, alt = bs.unpack_from('s34s35s14', msg, i)
            i += 83
            self.integ.pos[k, :] = [lat*1.1e-8, lon*1.1e-8, alt]
            # i += 26  # ?

            if mm_id == 0:
                # Number of GMM components DFi210
                ngmm = bs.unpack_from('u2', msg, i)[0]
                i += 2
                self.integ.np[k] = ngmm
                prm = np.zeros((ngmm, 3))

                for j in range(ngmm):
                    # GMM component probability DFi211
                    # GMM component expectation DFi212
                    # GMM component standard deviation DFi213
                    prob, exp_, std_ = bs.unpack_from('u4u4u4', msg, i)
                    i += 12

                    prob *= 0.0625
                    exp = self.integ.mu_k_t[exp_]
                    std = self.integ.sig_k_t[std_]
                    prm[j, :] = [prob, exp, std]

                self.integ.mm_param[k] = prm

            elif mm_id in [1, 2]:
                # mm_id=1: Mats Brenner Model Data
                sig_t, nsig = self.decode_mask(mask_s, 8, ofst=0)
                self.integ.np[k] = nsig
                prm = np.zeros((nsig, n))

                for j in range(nsig):
                    # Multipath parameter a DFi215
                    # Multipath parameter b DFi216
                    # Multipath parameter c DFi217
                    a, b, c = bs.unpack_from('u4s5s5', msg, i)
                    i += 14

                    prm[j, 0] = sig_t[j]
                    a = self.integ.sig_k_t[a]
                    prm[j, 1:4] = [a, b*0.25, c*0.0625]

                    if mm_id == 2:
                        # Multipath parameter d DFi218
                        d = bs.unpack_from('u8', msg, i)[0]
                        i += 8
                        prm[j, 4] = d*0.3515625

                self.integ.mm_param[k] = prm

        return i

    def decode_integrity_ssr(self, msg, i):
        """ RTCM SC-134 SSR integrity message (MT2011) """
        i = self.decode_integ_head(msg, i)
        
        # SSR provider ID, solution type, iod
        self.integ.pidssr, self.integ.sidssr, self.integ.iodssr = \
            bs.unpack_from('u16u4u4', msg, i)
        i += 24

        # GNSS Constellation Mask (DFi013)
        # 0:GPS,1:GLO,2:GAL,3:BDS,4:QZS,5:IRN
        mask_sys = bs.unpack_from('u16', msg, i)[0]
        i += 16

        sys_t, nsys = self.decode_mask(mask_sys, 16, ofst=0)
        iod_sys = {}

        flag_t = {}
        sat_e = []
        for sys_ in sys_t:
            sys = self.integ.sys_tbl[sys_]
            mask_sat, iod_sys[sys] = bs.unpack_from('u64u2', msg, i)
            i += 66
            svid_t, _ = self.decode_mask(mask_sat, 64)
            sys = self.integ.sys_tbl[sys_]
            flag_t[sys] = {}
            for svid in svid_t:
                sat = self.svid2sat(sys, svid)
                flag_t[sys][sat] = bs.unpack_from('u2', msg, i)[0]
                i += 2
                if flag_t[sys][sat] == 1: # Do Not USe:
                    sat_e.append(sat)
                    
        self.integ.sat_e = sat_e
        self.integ.iod_sys = iod_sys
        self.integ.flag = flag_t

    def decode_sc134_test_header(self, msg, i):
        # MT51-56 has subtype to extent message type space
        # DFi028 uint8 as in Table 9-35

        # Message Type     DF002  uint12
        # Working Group    DFi020 uint4
        # Sub-Message      DFi028 uint8
        # Message Revision DFi00x uint4 => uint8?
        wg, self.subtype, self.ver = bs.unpack_from('u4u8u4', msg, i)
        self.wg = wg-1
        i += 16
        return i

    def decode(self, msg, subtype=None, scanmode=False):
        """ decode RTCM messages """
        i = 24
        self.msgtype = bs.unpack_from('u12', msg, i)[0]
        i += 12
        if self.monlevel > 0 and self.fh is not None:
            self.fh.write("##### RTCM 3.x type:{:04d} msg_size: {:d} bytes\n".
                          format(self.msgtype, self.dlen))

        obs = None
        eph = None
        geph = None
        seph = None

        if scanmode:
            if self.msgtype in (1007, 1008, 1033):
                self.subtype = sRTCM.ANT_DESC
                i = self.decode_ant_desc(msg, i)
            elif self.msgtype in (1005, 1006, 1032):
                self.subtype = sRTCM.ANT_POS
                i = self.decode_sta_pos(msg, i)
            elif self.msgtype == 1230:
                self.subtype = sRTCM.GLO_BIAS
                i = self.decode_glo_bias(msg, i)
            return i, obs, eph, geph, seph

        self.subtype = subtype

        if self.msgtype in self.mt_skip:
            return i

        # SSR messages
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
        elif self.msgtype in (1030, 1031, 1303, 1304, 1305):  # Network RTK
            self.subtype = sRTCM.NRTK_RES
            i = self.decode_nrtk_residual(msg, i)
        elif self.is_msmtype(self.msgtype):  # MSM
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

        # test messages for RTCM SSR
        elif self.msgtype == 60:
            self.subtype = sRTCM.SSR_META
            i = self.decode_ssr_metadata(msg, i)
        elif self.msgtype == 61:
            self.subtype = sRTCM.SSR_GRID
            i = self.decode_ssr_grid(msg, i)
        elif self.msgtype in [1057, 41, 62, 63, 64]:
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)
        elif self.msgtype in [1058, 42, 65, 66, 67]:
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_clk(msg, i)
        elif self.msgtype in [1060, 44, 71, 72, 73]:
            self.subtype = sCSSR.COMBINED
            i = self.decode_cssr_comb(msg, i)
        elif self.msgtype in [1061, 45, 74, 75, 76]:
            self.subtype = sCSSR.URA
            i = self.decode_cssr_ura(msg, i)
        elif self.msgtype in [1062, 46, 77, 78, 79]:
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_hclk(msg, i)
        elif self.msgtype in [1059, 43, 68, 69, 70]:
            self.subtype = sCSSR.CBIAS
            i = self.decode_cssr_cbias(msg, i)
        elif self.msgtype >= 80 and self.msgtype < 85:
            self.subtype = sRTCM.SSR_SATANT
            i = self.decode_ssr_satant(msg, i)
        elif self.msgtype >= 85 and self.msgtype < 90:
            if not self.mask_pbias:
                self.subtype = sRTCM.SSR_PBIAS
                i = self.decode_cssr_pbias(msg, i)
        elif self.msgtype >= 90 and self.msgtype < 95:
            self.subtype = sRTCM.SSR_PBIAS_EX
            i = self.decode_ssr_pbias_ex(msg, i)
        elif self.msgtype == 95:
            self.subtype = sRTCM.SSR_TROP
            i = self.decode_ssr_trop(msg, i)
        elif self.msgtype >= 96 and self.msgtype < 101:
            self.subtype = sRTCM.SSR_STEC
            i = self.decode_ssr_iono(msg, i)

        # test messages for SC-134
        elif self.msgtype in [3, 2000]:  # minimum integrity
            self.subtype = sRTCM.INTEG_MIN
            self.decode_integrity_min(msg, i)
        elif self.msgtype in [4, 2005]:  # extended integrity
            self.subtype = sRTCM.INTEG_EXT
            self.decode_integrity_ext(msg, i)
        elif self.msgtype in [5, 2006]:  # sis integrity/local error
            self.subtype = sRTCM.INTEG_EXT_SIS
            self.decode_integrity_ext_sis_local(msg, i)
        elif self.msgtype in [6, 2007]:  # primary sercice area
            self.subtype = sRTCM.INTEG_PRI_AREA
            self.decode_integrity_service_area(msg, i)
        elif self.msgtype == 2008:  # extended sercice area
            self.subtype = sRTCM.INTEG_EXT_AREA
            self.decode_integrity_service_area(msg, i, True)
        elif self.msgtype in [7, 2051]:  # quality indicator
            self.subtype = sRTCM.INTEG_QUALITY
            self.decode_integrity_quality(msg, i)
        elif self.msgtype in [8, 2091]:  # CNR/ACG SIS Monitoring
            self.subtype = sRTCM.INTEG_CNR
            self.decode_integrity_cnr_acg(msg, i)
        elif self.msgtype == 2071:  # satellite visibility map
            self.subtype = sRTCM.INTEG_VMAP
            self.decode_integrity_vmap(msg, i)
        elif self.msgtype == 2072:  # multipath map
            self.subtype = sRTCM.INTEG_MMAP
            self.decode_integrity_mmap(msg, i)
        elif self.msgtype in [11, 2011]:  # SSR integrity
            self.subtype = sRTCM.INTEG_SSR
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 12:  # SSR integrity Iono
            self.subtype = sRTCM.INTEG_SSR_IONO
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 13:  # SSR integrity Trop
            self.subtype = sRTCM.INTEG_SSR_TROP
            self.decode_integrity_ssr(msg, i)
        elif self.msgtype == 54:  # SSR integrity test msg
            i = self.decode_sc134_test_header(msg, i)
            if self.subtype == 9:
                self.subtype = sRTCM.INTEG_VMAP
                self.decode_integrity_vmap(msg, i)
            elif self.subtype == 10:
                self.subtype = sRTCM.INTEG_MMAP
                self.decode_integrity_mmap(msg, i)
            else:
                self.subtype = -1

        if self.monlevel > 0 and self.fh is not None:
            self.out_log(obs, eph, geph, seph)

        return i, obs, eph, geph, seph


class rtcme(cssre, rtcmUtil):
    """ class for RTCM message encoder """

    def __init__(self):
        super().__init__()
        self.integ = Integrity()

        self.pid = 0  # SSR Provider ID
        self.sid = 0  # SSR Solution Type

        self.mi = False  # multiple message bit
        self.iods = 0  # issue of data station
        self.refid = 0  # reference station ID

        self.csi = 0  # clock streering indicator
        self.eci = 0  # external clock indicator
        self.si = 0  # smoothing indicator
        self.smi = 0  # smoothing interval

        self.gtype = 0
        self.ofst = 0
        self.nm = 0  # number of metadata model
        self.iyaw = False
        self.iexpb = 0
        self.nlen = 0  # Hydrostatic Grid Point Residual Length DF+050
        self.mlen = 14  # Wet Grid Point Residual Length DF+051

        # RTCM SSR grid parameters
        self.ncol = 0
        self.nrow = 0
        self.dlat = 0
        self.dlon = 0
        self.gma = False
        self.latmask = 0
        self.lonmask = 0
        self.egpmask = 0

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

    def round_i(self, v):
        return int(v+0.5)

    def ival(self, y, nbit, scl):
        """ float to int conversion with scaling """
        if np.isnan(y):
            u = -(1 << (nbit-1))  # -lim: data is not available
        else:
            u = round(y/scl)

        return u

    def encode_msm_time(self, sys, time):
        """ decode msm time """
        if sys == uGNSS.GLO:
            time = timeadd(gpst2utc(time), 10800.0)
            week, tow = time2gpst(time)
            dow = int(tow/86400.0)  # day of week
            tod = self.round_i((tow - dow*86400.0)*1e3)  # time of day [ms]
            epoch = ((dow & 0x1f) << 27) | (tod & 0x7ffffff)

        elif sys == uGNSS.BDS:
            week, tow = time2gpst(gpst2bdt(time))
            epoch = self.round_i(tow*1e3)
        else:
            week, tow = time2gpst(time)
            epoch = self.round_i(tow*1e3)

        return week, epoch

    def encode_msm(self, msg, obs, i):
        """ encode MSM message """
        sys, msm = self.msmtype(self.msgtype)

        # self.time, self.tow = self.encode_msm_time(sys, self.week, tow_)
        self.week, tow_ = self.encode_msm_time(sys, self.time)

        ofst = 193 if sys == uGNSS.QZS else 1

        svmask = self.encode_mask(obs.sat, 64, ofst)
        sigmask = self.encode_mask(obs.sig, 32)
        self.nsat = len(obs.sat)
        self.nsig = len(obs.sig)

        # augmentation service provider id DFi027
        bs.pack_into('u12u30b1u3', msg, i, self.refid,
                     tow_, self.mi, self.iods)
        i += 53

        bs.pack_into('u2u2u1u3', msg, i, self.csi,
                     self.eci, self.si, self.smi)
        i += 8
        bs.pack_into('u64u32', msg, i, svmask, sigmask)
        i += 96

        sz = self.nsat*self.nsig
        if sz > 64:
            return -1
        ncell = 0
        for k in range(self.nsat):
            sz_ = len(self.sig_n[k])
            cellmask = self.encode_mask(self.sig_n[k], self.nsig)
            bs.pack_into(f'u{self.nsig}', msg, i, cellmask)
            i += self.nsig
            ncell += sz_

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

        nobs, nsig = obs.P.shape()

        for i in range(nobs):
            for j in range(nsig):
                k = 0  # TBD
                # freq = obs.sig.frequency()
                r[k] = self.round_i(
                    obs.P[i, j]/(rms*rCST.P2_10))*rms*rCST.P2_10

                # satellite part
                if msm >= 1 and msm <= 3:
                    for k in range(self.nsat):
                        v = int(r[k]/rms)
                        bs.pack_into('u10', msg, i, v)
                        i += 10
                else:
                    for k in range(self.nsat):
                        v = int(r[k]/rms)
                        bs.pack_into('u8', msg, i, v)
                        i += 8

                if msm == 5 or msm == 7:
                    for k in range(self.nsat):
                        bs.pack_into('u4', msg, i, ex[k])
                        i += 4

                    for k in range(self.nsat):
                        if r[k] != 0.0:
                            v = int(r[k]/(rCST.P2_10*rms))
                            bs.pack_into('u10', msg, i, v)
                            i += 10

                if msm == 5 or msm == 7:
                    for k in range(self.nsat):
                        v = int(rr[k])
                        bs.pack_into('s14', msg, i, v)
                        i += 14

                # signal part
                if msm != 2:
                    sz = 15 if msm < 6 else 20
                    scl = rCST.P2_24 if msm < 6 else rCST.P2_29
                    for k in range(ncell):
                        pr_ = int(pr[k]/(scl*rms))
                        bs.pack_into('s'+str(sz), msg, i, pr_)
                        i += sz

                if msm > 1:
                    sz = 22 if msm < 6 else 24
                    scl = rCST.P2_29 if msm < 6 else rCST.P2_31
                    for k in range(ncell):
                        cp_ = int(cp[k]/(scl*rms))
                        bs.pack_into('s'+str(sz), msg, i, cp_)
                        i += sz

                    sz = 4 if msm < 6 else 10
                    for k in range(ncell):
                        bs.pack_into('u'+str(sz), msg, i, lock[k])
                        i += sz

                    for k in range(ncell):
                        bs.pack_into('u1', msg, i, half[k])
                        i += 1

                if msm > 3:
                    sz = 6 if msm < 6 else 10
                    scl = 1.0 if msm < 6 else 0.0625
                    for k in range(ncell):
                        v = int(cnr[k]/scl)
                        bs.pack_into('u'+str(sz), msg, i, v)
                        i += sz

                if msm == 5 or msm == 7:
                    for k in range(ncell):
                        bs.pack_into('s15', msg, i, rrf[k]*1e4)
                        i += 15

        return i

    def encode_integrity_ssr(self, msg, i):
        """ RTCM SC-134 SSR integrity message (MT11,12,13) """

        sys_t = self.integ.iod_sys.keys()
        gnss_t = []
        for sys in sys_t:
            gnss_t.append(self.integ.sys_r_tbl[sys])

        # mask_sys:: DFi013 0:GPS,1:GLO,2:GAL,3:BDS,4:QZS,5:IRN
        mask_sys = self.encode_mask(gnss_t, 16, ofst=0)

        # augmentation service provider id DFi027
        bs.pack_into('u12', msg, i, self.integ.pid)
        i += 12

        # SSR provider id, soluition type, iod
        bs.pack_into('u16u4u4', msg, i, self.pid, self.sid, self.iodssr)
        i += 24

        bs.pack_into('u30', msg, i, self.integ.tow*1e3)  # tow DFi008
        i += 30
        bs.pack_into('u16', msg, i, mask_sys)  # GNSS constellation mask DFi013
        i += 16

        # validity period DFi065
        # update rate interval DFi067
        vp = self.integ.vp_r_tbl[int(self.integ.vp)]
        bs.pack_into('u4u16', msg, i, vp, self.integ.udi*10)
        i += 20

        for sys in sys_t:
            flag = self.integ.flag[sys]
            sat_t = flag.keys()
            svid_t = []
            for sat in sat_t:
                sys_, prn = sat2prn(sat)
                ofst = 192 if sys_ == uGNSS.QZS else 0
                svid_t.append(prn-ofst)

            # GNSS satellite mask DFi009
            mask_sat = self.encode_mask(svid_t, 64)
            bs.pack_into('u64u2', msg, i, mask_sat, self.integ.iod_sys[sys])
            i += 66

            for f in flag.values():
                # integrity flag DFi068
                bs.pack_into('u2', msg, i, f)
                i += 2

        return i

    def encode_ssr_metadata(self, msg, i):
        """ encode SSR meta-information message (MT60) """

        # nm: nomber of ssr model/correction entries
        bs.pack_into('u4u16u4u5', msg, i, self.iodssr, self.pid,
                     self.sid, self.nm)
        i += 29

        if self.nm == 0:  # no metadata
            i += 7
            return i

        # TBD
        return i

    def encode_ssr_grid(self, msg, i):
        """ encode SSR grid definition message (MT61) """

        bs.pack_into('u16b1u10u3', msg, i, self.pid,
                     self.mi, self.gid, self.gtype)
        i += 30

        self.inet = self.gid
        grid = self.grid[self.grid['nid'] == self.inet]
        self.ng = grid.shape[0]

        if self.gtype == 0:  # grid type 0
            gid, ofst, lat, lon, alt = grid[0]

            lat_i = int(lat/1e-3)
            lon_i = int(lon/1e-3)
            alt_i = int((alt+1000)/12.5)

            # Latitude of Reference Point DF+024
            # Longitude of Reference Point DF+025
            # Ellipsoidal Height of Reference Point DF+026
            # Number of Relative Points: DF+027
            bs.pack_into('s18s19u10u12u8', msg, i, lat_i, lon_i, alt_i,
                         self.ofst, self.ng-1)
            i += 67

            for k in range(1, self.ng):
                gid, ofst, lat_, lon_, alt_ = grid[k]
                dlat = lat_-lat
                dlon = lon_-lon
                dalt = alt_-alt
                lat = lat_
                lon = lon_
                alt = alt_

                dlat_i = int(dlat/1e-3)
                dlon_i = int(dlon/1e-3)
                dalt_i = int(dalt/12.5)

                bs.pack_into('s13s14s9', msg, i, dlat_i, dlon_i, dalt_i)
                i += 36

        elif self.gtype == 1:  # grid type 1
            gid, ofst, lat, lon, alt = grid[0]

            lat_i = int(lat/1e-3)
            lon_i = int(lon/1e-3)

            bs.pack_into('s18s19u12u8', msg, i, lat_i, lon_i,
                         self.ofst, self.ng-1)
            i += 57

            for k in range(1, self.ng):
                gid, ofst, lat_, lon_, _ = grid[k]
                dlat = lat_-lat
                dlon = lon_-lon
                lat = lat_
                lon = lon_

                dlat_i = int(dlat/1e-3)
                dlon_i = int(dlon/1e-3)
                bs.pack_into('s13s14', msg, i, dlat_i, dlon_i)
                i += 27

        elif self.gtype == 2:  # grid type 2
            gid, ofst, lat0, lon0, alt = grid[0]

            lat_i = int(lat0/1e-3)
            lon_i = int(lon0/1e-3)
            dlat_i = int(self.dlat/0.01)
            dlon_i = int(self.dlon/0.01)

            bs.pack_into('s18s19u6u6u9u10b1', msg, i, lat_i, lon_i,
                         self.nrow, self.ncol, dlat_i,  dlon_i, self.gma)
            i += 69

            if self.gma:
                bs.pack_into('u'+str(self.nrow), msg, i, self.latmask)
                i += self.nrow
                bs.pack_into('u'+str(self.ncol), msg, i, self.lonmask)
                i += self.ncol

                ilat, nlat = self.decode_mask(self.latmask, self.nrow, 0)
                ilon, nlon = self.decode_mask(self.lonmask, self.ncol, 0)
                n = nlat*nlon
                bs.pack_into('u'+str(n), msg, i, self.egpmask)
                i += n
                # iex, nex = self.decode_mask(egpmask, n, 0)

        return i

    def encode_head(self, msg, i, sys=uGNSS.NONE):
        """ encode the header of ssr message """
        if self.msgtype == 4076 or sys != uGNSS.GLO:
            blen = 20
        else:
            blen = 17
        bs.pack_into('u'+str(blen), msg, i, self.tow)
        i += blen

        if self.subtype not in [sRTCM.SSR_PBIAS_EX]:
            bs.pack_into('u4b1', msg, i, self.udi[self.sc_t[self.subtype]],
                         self.mi)
            i += 5

        if self.subtype in (sCSSR.ORBIT, sCSSR.COMBINED):
            bs.pack_into('u1', msg, i, self.datum)
            i += 1

        bs.pack_into('u4u16u4', msg, i, self.iodssr, self.pid, self.sid)
        i += 24

        if self.subtype == sCSSR.PBIAS:
            bs.pack_into('u1u1', msg, i, self.ci, self.mw)
            i += 2
        elif self.subtype == sRTCM.SSR_PBIAS:
            # Satellite Yaw Information Indicator DF486
            # Extended Phase Bias Property ID DF+2
            bs.pack_into('b1u4', msg, i, self.iyaw, self.iexpb)
            i += 5

        if self.subtype not in [sCSSR.VTEC, sRTCM.SSR_PBIAS, sRTCM.SSR_STEC,
                                sRTCM.SSR_TROP]:
            nsat = 0
            for sat in self.sat_n:
                sys_, _ = sat2prn(sat)
                if sys_ != sys:
                    continue
                nsat += 1
            self.nsat_n = nsat

            bs.pack_into('u6', msg, i, nsat)
            i += 6

        return i

    def encode_sat(self, msg, i, sat):
        """ encode satellite id """

        sys, prn = sat2prn(sat)
        svid = prn

        if self.msgtype == 4076:
            blen = 6
        else:
            if sys == uGNSS.QZS:
                blen = 4
                svid = prn-192
            else:
                blen = 6

        bs.pack_into('u'+str(blen), msg, i, svid)
        i += blen

        return i

    def encode_orb_sat(self, msg, i, sat, inet=0):
        """ encode orbit correction of cssr """
        sys, _ = sat2prn(sat)

        if self.msgtype == 4076:
            blen = 8
        else:
            if sys == uGNSS.GAL:
                blen = 10
            elif sys == uGNSS.SBS:
                blen = 24
            else:
                blen = 8
        bs.pack_into('u'+str(blen), msg, i, self.lc[inet].iode[sat])
        i += blen

        dx = self.ival(self.lc[inet].dorb[sat][0], 22, 0.1e-3)
        dy = self.ival(self.lc[inet].dorb[sat][1], 20, 0.4e-3)
        dz = self.ival(self.lc[inet].dorb[sat][2], 20, 0.4e-3)

        bs.pack_into('s22s20s20', msg, i, dx, dy, dz)
        i += 62

        if self.lc[inet].ddorb is not None:
            ddx = self.ival(self.lc[inet].ddorb[sat][0], 21, 1e-6)
            ddy = self.ival(self.lc[inet].ddorb[sat][1], 19, 4e-6)
            ddz = self.ival(self.lc[inet].ddorb[sat][2], 19, 4e-6)
        else:
            ddx, ddy, ddz = 0, 0, 0

        bs.pack_into('s21s19s19', msg, i, ddx, ddy, ddz)
        i += 59

        return i

    def encode_clk_sat(self, msg, i, sat, inet=0):
        """ encoder clock correction of cssr """

        dclk = self.ival(self.lc[inet].dclk[sat], 22, 0.1e-3)

        if self.lc[inet].ddclk is not None:
            ddclk = self.ival(self.lc[inet].ddclk[sat], 21, 0.4e-3)
            dddclk = self.ival(self.lc[inet].dddclk[sat], 27, 4e-6)
        else:
            ddclk, dddclk = 0, 0

        bs.pack_into('s22s21s27', msg, i, dclk, ddclk, dddclk)
        i += 70
        return i

    def encode_hclk_sat(self, msg, i, sat, inet=0):
        """ encoder high-rate clock correction of cssr """
        if self.lc[inet].hclk is not None:
            hclk = self.ival(self.lc[inet].hclk[sat], 22, 0.1e-3)
        else:
            hclk = 0

        bs.pack_into('s22', msg, i, hclk)
        i += 22
        return i

    def encode_cssr_orb(self, msg, i, inet=0):
        """ encode RTCM SSR Orbit Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys)

        for sat in self.sat_n:
            sys_, _ = sat2prn(sat)
            if sys_ != sys:
                continue
            i = self.encode_sat(msg, i, sat)
            i = self.encode_orb_sat(msg, i, sat)

        return i

    def encode_cssr_clk(self, msg, i, inet=0):
        """encode RTCM SSR Clock Correction message """
        sys = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys)

        for sat in self.sat_n:
            sys_, _ = sat2prn(sat)
            if sys_ != sys:
                continue
            i = self.encode_sat(msg, i, sat)
            i = self.encode_clk_sat(msg, i, sat)

        return i

    def encode_cssr_ura(self, msg, i, inet=0):
        """ encode RTCM SSR URA message """
        sys = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys)
        ura = self.lc[inet].ura

        for sat in ura:
            sys_, _ = sat2prn(sat)
            if sys_ != sys:
                continue
            i = self.encode_sat(msg, i, sat)
            if np.isnan(ura[sat]):
                cls_, val_ = 0, 0
            else:
                cls_, val_ = self.quality2qi(ura[sat])
            bs.pack_into('u3u3', msg, i, cls_, val_)
            i += 6

        return i

    def encode_cssr_cbias(self, msg, i, inet=0):
        """ encode RTCM SSR code bias message """
        sys = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys)

        for sat in self.lc[inet].cbias:
            sys_, _ = sat2prn(sat)
            if sys_ != sys:
                continue
            i = self.encode_sat(msg, i, sat)
            cbias = self.lc[inet].cbias[sat]
            nsig = len(cbias)
            bs.pack_into('u5', msg, i, nsig)
            i += 5

            for rsig_ in cbias:
                cb = int(cbias[rsig_]/0.01)
                code = self.rsig2code(rsig_)
                bs.pack_into('u5s14', msg, i, code, cb)
                i += 19

        return i

    def encode_cssr_pbias(self, msg, i, inet=0):
        """ encode RTCM SSR phase bias message """
        sys = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys)

        inet = self.inet

        nsat = 0
        for sat in self.lc[inet].pbias:
            sys_, _ = sat2prn(sat)
            if sys_ == sys:
                nsat += 1
        bs.pack_into('u6', msg, i, nsat)
        i += 6

        for sat in self.lc[inet].pbias:
            sys_, _ = sat2prn(sat)
            if sys_ != sys:
                continue
            i = self.encode_sat(msg, i, sat)
            pbias = self.lc[inet].pbias[sat]
            nsig = len(pbias)
            bs.pack_into('u5', msg, i, nsig)
            i += 5

            if sat in self.lc[inet].yaw:
                yaw = int(self.lc[inet].yaw[sat]*256.0)
                dyaw = int(self.lc[inet].dyaw[sat]*8192.0)
                bs.pack_into('u9s8', msg, i, yaw, dyaw)
                i += 17

            for rsig_ in pbias:

                pb = self.ival(pbias[rsig_], 20, 1e-4)
                code = self.rsig2code(rsig_)
                if sat in self.lc[inet].si:
                    si = self.lc[inet].si[sat][rsig_]
                else:
                    si = 0
                di = self.lc[inet].di[sat][rsig_]

                bs.pack_into('u5b1', msg, i, code, si)
                i += 6

                if self.subtype != sRTCM.SSR_PBIAS:  # IGS-SSR
                    wl = self.lc[inet].wl[sat][rsig_]
                    bs.pack_into('u2', msg, i, wl)
                    i += 2

                bs.pack_into('u4s20', msg, i, di, pb)
                i += 24

        return i

    def encode_ssr_trop(self, msg, i, inet=0):
        """ encode SSR Tropspheric Correction Message """
        i = self.encode_head(msg, i)
        inet = self.gid
        rmi = self.lc[inet].flg_trop & 1
        # Grid ID: DF+022
        # Residual model indicator DF+009
        bs.pack_into('u10b1', msg, i, self.gid, rmi)
        i += 11

        # mapping function parameters
        ah, bh, ch = self.lc[inet].maph
        aw, bw, cw = self.lc[inet].mapw

        dah = self.ival((ah-0.00118), 11, 2.5e-7)
        dbh = self.ival((bh-0.00298), 9, 5e-6)
        dch = self.ival((ch-0.0682), 9, 2e-4)
        daw = self.ival((aw-0.000104), 13, 1e-6)
        dbw = self.ival((bw-0.0015), 6, 2.5e-5)
        dcw = self.ival((cw-0.048), 5, 2e-3)

        # atmospheric model part
        bs.pack_into(
            's11s9s9s13s6s5', msg, i, dah, dbh, dch, daw, dbw, dcw)
        i += 53

        # functional term parameters
        ct = self.lc[inet].ct

        # hydrostatic
        ct00h = self.ival((ct[0, 0]-2.3), 13, 0.1e-3)
        ct01h = self.ival(ct[0, 1], 15, 0.01e-3)
        ct10h = self.ival(ct[0, 2], 15, 0.01e-3)
        # wet
        ct00w = self.ival((ct[1, 0]-0.252), 13, 0.1e-3)
        ct01w = self.ival(ct[1, 1], 15, 0.01e-3)
        ct10w = self.ival(ct[1, 2], 15, 0.01e-3)

        bs.pack_into('s13s15s15', msg, i, ct00h, ct10h, ct01h)
        i += 43
        bs.pack_into('s13s15s15', msg, i, ct00w, ct10w, ct01w)
        i += 43

        n = self.nlen
        m = self.mlen

        if n > 0:
            rh = self.lc[inet].dth
        if m > 0:
            rw = self.lc[inet].dtw

        if rmi:  # residual information part
            ng = self.lc[inet].ng
            ofst = self.lc[inet].ofst

            bs.pack_into('u12u12u4u4', msg, i, ofst, ng, n, m)
            i += 32

            for k in range(ofst, ofst+ng):
                if n > 0:
                    rh_ = self.ival(rh[k], n, 0.1e-3)
                    bs.pack_into(f's{n}', msg, i, rh_)
                    i += n
                if m > 0:
                    rw_ = self.ival(rw[k], m, 0.1e-3)
                    bs.pack_into(f's{m}', msg, i, rw_)
                    i += m

        return i

    def encode_ssr_iono(self, msg, i, inet=0):
        """ encode SSR Regional Ionospheric Correction message (E96,E97,E98,E99,E100) """
        sys_m = self.get_ssr_sys(self.msgtype)
        i = self.encode_head(msg, i, sys_m)

        inet = self.inet

        pmi = (self.lc[inet].flg_stec >> 1) & 1
        rmi = self.lc[inet].flg_stec & 1

        # Grid ID DF+22
        # Polynomial Model Indicator DF+96
        # Residual Model Indicator
        bs.pack_into('u10b1b1', msg, i, self.gid, pmi, rmi)
        i += 12

        inet = self.gid

        svid = []
        sat = []
        stype = 0
        for sat_ in self.lc[inet].ci:
            sys, _ = sat2prn(sat_)
            if sys != sys_m:
                continue
            sat.append(sat_)
            svid.append(self.sat2svid(sat_))
            if stype < self.lc[inet].stype[sat_]:
                stype = self.lc[inet].stype[sat_]

        satmask = self.encode_mask(svid, 64)
        bs.pack_into('u64', msg, i, satmask)
        i += 64

        # Zenith-mapped STEC Polynomial Gradient Indicator DF+60
        if pmi:  # polynomial model information part
            pgi = stype > 0
            bs.pack_into('b1', msg, i, pgi)
            i += 1

            for sat_ in sat:
                ci = self.lc[inet].ci[sat_]

                c00i = self.ival(ci[0], 17, 0.01)
                bs.pack_into('s17', msg, i, c00i)
                i += 17
                if pgi:
                    c01i = self.ival(ci[1], 18, 1e-3)
                    c10i = self.ival(ci[2], 18, 1e-3)
                    bs.pack_into('s18s18', msg, i, c01i, c10i)
                    i += 36

        if rmi:  # residual model information part
            ng = self.lc[inet].ng
            ofst = self.lc[inet].ofst
            rlen = 18  # TBD
            scl = 2e-3  # TBD

            grid = self.grid[self.grid['nid'] == self.inet]
            lat0 = grid[0]['lat']
            lon0 = grid[0]['lon']

            bs.pack_into('u12u12u5', msg, i, ofst, ng, rlen)
            i += 29
            if rlen > 0:
                # Zenith-mapped STEC Grid Point Residual
                # Resolution Scale Factor Exponent
                sf = np.log(scl/1e-3)/np.log(2)
                bs.pack_into('u3', msg, i, sf)  # DF+073
                i += 3

            fmt = 's'+str(rlen)
            for sat_ in sat:
                ci = self.lc[inet].ci[sat_]
                for j in range(ofst, ofst+ng):
                    dstec = self.lc[inet].dstec[sat_][j]
                    if np.abs(ci[3]) > 0:
                        dlat = grid[j]['lat']-lat0
                        dlon = grid[j]['lon']-lon0
                        dstec += [dlat*dlon, dlat**2, dlon**2]@ci[3:]

                    res = int(dstec/scl)
                    bs.pack_into(fmt, msg, i, res)
                    i += rlen

        return i

    def encode(self, msg, obs=None):
        """ encode RTCM messages """
        i = 24
        bs.pack_into('u12', msg, i, self.msgtype)
        i += 12

        if self.is_msmtype(self.msgtype):
            self.subtype = sRTCM.MSM
            i = self.encode_msm(msg, obs, i)

        elif self.msgtype in (11, 12, 13):  # SSR integrity (test)
            i = self.encode_integrity_ssr(msg, i)

        elif self.msgtype == 60:
            self.subtype = sRTCM.SSR_META
            i = self.encode_ssr_metadata(msg, i)
        elif self.msgtype == 61:
            self.subtype = sRTCM.SSR_GRID
            i = self.encode_ssr_grid(msg, i)
        elif self.msgtype in [1057, 41, 62, 63, 64]:
            self.subtype = sCSSR.ORBIT
            i = self.encode_cssr_orb(msg, i)
        elif self.msgtype in [1058, 42, 65, 66, 67]:
            self.subtype = sCSSR.CLOCK
            i = self.encode_cssr_clk(msg, i)
        elif self.msgtype in [1060, 44, 71, 72, 73]:
            self.subtype = sCSSR.COMBINED
            i = self.encode_cssr_comb(msg, i)
        elif self.msgtype in [1061, 45, 74, 75, 76]:
            self.subtype = sCSSR.URA
            i = self.encode_cssr_ura(msg, i)
        elif self.msgtype in [1062, 46, 77, 78, 79]:
            self.subtype = sCSSR.CLOCK
            i = self.encode_cssr_hclk(msg, i)
        elif self.msgtype in [1059, 43, 68, 69, 70]:
            self.subtype = sCSSR.CBIAS
            i = self.encode_cssr_cbias(msg, i)
        elif self.msgtype >= 80 and self.msgtype < 85:
            self.subtype = sRTCM.SSR_SATANT
            # i = self.encode_ssr_satant(msg, i)
        elif self.msgtype >= 85 and self.msgtype < 90:
            self.subtype = sRTCM.SSR_PBIAS
            i = self.encode_cssr_pbias(msg, i)
        elif self.msgtype >= 90 and self.msgtype < 95:
            self.subtype = sRTCM.SSR_PBIAS_EX
            # i = self.encode_ssr_pbias_ex(msg, i)
        elif self.msgtype == 95:
            self.subtype = sRTCM.SSR_TROP
            i = self.encode_ssr_trop(msg, i)
        elif self.msgtype >= 96 and self.msgtype < 101:
            self.subtype = sRTCM.SSR_STEC
            i = self.encode_ssr_iono(msg, i)

        return i
