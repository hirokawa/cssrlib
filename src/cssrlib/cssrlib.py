"""
module for Compact SSR processing

[1] Specification of Compact SSR Messages for Satellite Based Augmentation 
    Service, v08, DOI: 10.13140/RG.2.2.10749.49129, 2019
    
@author Rui Hirokawa

"""

import bitstruct as bs
import copy
import numpy as np
from enum import IntEnum
from cssrlib.gnss import gpst2time, rCST, prn2sat, uGNSS, gtime_t, rSigRnx
from cssrlib.gnss import uSIG, uTYP, sat2prn, time2str, sat2id, timediff
from cssrlib.gnss import copy_buff


class sCSSRTYPE(IntEnum):
    QZS_CLAS = 0     # QZS CLAS PPP-RTK
    QZS_MADOCA = 1   # MADOCA-PPP
    GAL_HAS_SIS = 2  # Galileo HAS Signal-In-Space
    GAL_HAS_IDD = 3  # Galileo HAS Internet Data Distribution
    BDS_PPP = 4      # BDS PPP
    IGS_SSR = 5
    RTCM3_SSR = 6
    PVS_PPP = 7      # PPP via SouthPAN
    SBAS_L1 = 8      # L1 SBAS
    SBAS_L5 = 9      # L5 SBAS (DFMC)
    DGPS = 10        # DGPS (QZSS SLAS)
    STDPOS = 11


class sGNSS(IntEnum):
    """ class to define GNSS """
    GPS = 0
    GLO = 1
    GAL = 2
    BDS = 3
    QZS = 4
    SBS = 5
    IRN = 6
    BDS3 = 7


class sCSSR(IntEnum):
    """ class to define Compact SSR message types """
    MASK = 1
    ORBIT = 2
    CLOCK = 3
    CBIAS = 4
    PBIAS = 5
    BIAS = 6
    URA = 7
    STEC = 8
    GRID = 9
    SI = 10
    COMBINED = 11
    ATMOS = 12
    AUTH = 13
    VTEC = 16


class sCType(IntEnum):
    """ class to define correction message types """
    MASK = 0
    ORBIT = 1
    CLOCK = 2
    CBIAS = 3
    PBIAS = 4
    STEC = 5
    TROP = 6
    URA = 7
    AUTH = 8
    HCLOCK = 9
    VTEC = 10
    MAX = 11


class sSigGPS(IntEnum):
    """ class to define GPS signals """
    L1C = 0
    L1P = 1
    L1W = 2
    L1S = 3
    L1L = 4
    L1X = 5
    L2S = 6
    L2L = 7
    L2X = 8
    L2P = 9
    L2W = 10
    L5I = 11
    L5Q = 12
    L5X = 13


class sSigGLO(IntEnum):
    """ class to define GLONASS signals """
    L1C = 0
    L1P = 1
    L2C = 2
    L2P = 3
    L4A = 4
    L4B = 5
    L4X = 6
    L6A = 7
    L6B = 8
    L6X = 9
    L3I = 10
    L3Q = 11
    L3X = 12


class sSigGAL(IntEnum):
    """ class to define Galileo signals  """
    L1B = 0
    L1C = 1
    L1X = 2
    L5I = 3
    L5Q = 4
    L5X = 5
    L7I = 6
    L7Q = 7
    L7X = 8
    L8I = 9
    L8Q = 10
    L8X = 11
    L6B = 12
    L6C = 13
    L6X = 14


class sSigBDS(IntEnum):
    """ class to define BDS signals """
    L2I = 0
    L2Q = 1
    L2X = 2
    L6I = 3
    L6Q = 4
    L6X = 5
    L7I = 6
    L7Q = 7
    L7X = 8
    L1D = 9
    L1P = 10
    L1X = 11
    L5D = 12
    L5P = 13
    L5X = 14


class sSigQZS(IntEnum):
    """ class to define QZSS signals """
    L1C = 0
    L1S = 1
    L1L = 2
    L1X = 3
    L2S = 4
    L2L = 5
    L2X = 6
    L5I = 7
    L5Q = 8
    L5X = 9
    L6D = 10
    L6P = 11
    L6E = 12
    L1E = 13


class sSigSBS(IntEnum):
    """ class to define SBAS signals """
    L1C = 0
    L5I = 1
    L5Q = 2
    L5X = 3


class sSigIRN(IntEnum):
    """ class to define NavIC signals """
    L1D = 0
    L1P = 1
    L1X = 2
    L5X = 3
    L9A = 6


def sgnss2sys(sys: sGNSS):
    ugnss_tbl = {
        sGNSS.GPS: uGNSS.GPS,
        sGNSS.GLO: uGNSS.GLO,
        sGNSS.GAL: uGNSS.GAL,
        sGNSS.BDS: uGNSS.BDS,
        sGNSS.QZS: uGNSS.QZS,
        sGNSS.SBS: uGNSS.SBS,
        sGNSS.BDS3: uGNSS.BDS,
    }
    return ugnss_tbl[sys]


def sys2sgnss(sys: uGNSS):
    sgnss_tbl = {
        uGNSS.GPS: sGNSS.GPS,
        uGNSS.GLO: sGNSS.GLO,
        uGNSS.GAL: sGNSS.GAL,
        uGNSS.BDS: sGNSS.BDS,
        uGNSS.QZS: sGNSS.QZS,
        uGNSS.SBS: sGNSS.SBS,
    }
    return sgnss_tbl[sys]


class local_corr:
    """ class for local corrections """

    def __init__(self):
        self.inet = -1
        self.inet_ref = -1
        self.ng = -1
        self.pbias = {}
        self.cbias = {}
        self.iode = None
        self.dorb = None
        self.dclk = None
        self.hclk = None
        self.stec = None
        self.trph = None
        self.trpw = None
        self.ci = None
        self.ct = None
        self.quality_trp = None
        self.quality_stec = None
        self.sat_n = []
        self.t0 = {}
        self.cstat = 0            # status for receiving CSSR message
        self.t0s = {}
        sc_t = [sCType.CLOCK, sCType.ORBIT, sCType.CBIAS, sCType.PBIAS,
                sCType.HCLOCK]
        for sc in sc_t:
            self.t0s[sc] = gtime_t()


class cssr:
    """ class to process Compact SSR messages """
    CSSR_MSGTYPE = 4073
    MAXNET = 32
    SYSMAX = 16
    stec_sz_t = [4, 4, 5, 7]
    stec_scl_t = [0.04, 0.12, 0.16, 0.24]

    def __init__(self, foutname=None):
        """ constructor of cssr """
        self.cssrmode = sCSSRTYPE.QZS_CLAS
        self.monlevel = 0
        self.week = -1
        self.tow0 = -1
        self.iodssr = -1
        self.mask_id = -1
        self.mask_id_clk = -1
        self.msgtype = 4073
        self.subtype = 0
        self.svmask = [-1, -1, -1, -1]
        self.nsat_n = 0
        self.sys_n = []
        self.sat_n = []
        self.nsig_n = []
        self.nsig_total = 0
        self.sig_n = []
        self.dorb = []
        self.iode = []
        self.dclk = []
        self.sat_n_p = []
        self.dorb_d = []
        self.dclk_d = []
        self.ura = []
        self.cbias = []
        self.pbias = []
        self.inet = -1
        self.facility = -1
        self.pattern = -1
        self.sid = -1
        self.cstat = 0
        self.local_pbias = True  # for QZS CLAS
        self.buff = bytearray(250*5)
        self.buff_p = None
        self.sinfo = bytearray(160)
        self.grid = None
        self.prc = None
        self.cpc = None
        self.tow = 0
        self.lc = []
        self.fcnt = -1
        self.flg_net = False
        self.time = gtime_t()
        self.nsig_max = 0
        self.ngrid = 0
        self.grid_index = []
        self.grid_weight = []
        self.rngmin = 0
        self.inet_ref = -1
        self.netmask = np.zeros(self.MAXNET+1, dtype=np.dtype('u8'))
        for inet in range(self.MAXNET+1):
            self.lc.append(local_corr())
            self.lc[inet].inet = inet
            self.lc[inet].flg_trop = 0
            self.lc[inet].flg_stec = 0
            self.lc[inet].nsat_n = 0
            self.lc[inet].t0 = {}

        self.dorb_scl = [0.0016, 0.0064, 0.0064]
        self.dclk_scl = 0.0016
        self.dorb_blen = [15, 13, 13]
        self.dclk_blen = 15
        self.cb_blen = 11
        self.cb_scl = 0.02
        self.pb_blen = 15
        self.pb_scl = 0.001  # m
        self.iodssr_p = -1
        self.iodssr_c = np.ones(16, dtype=np.int32)*-1
        self.sig_n_p = []

        # maximum validty time for correction
        self.tmax = {sCType.CLOCK: 30.0, sCType.ORBIT: 120.0,
                     sCType.CBIAS: 120.0, sCType.PBIAS: 120.0,
                     sCType.TROP: 120.0, sCType.STEC: 120.0}

        # default navigation message mode: 0:LNAV/INAV, 1: CNAV/CNAV1
        self.nav_mode = {uGNSS.GPS: 0, uGNSS.QZS: 0,
                         uGNSS.GAL: 0, uGNSS.BDS: 1, uGNSS.GLO: 0}

        self.fh = None
        if foutname is not None:
            self.fh = open(foutname, "w")

    def ssig2rsig(self, sys: sGNSS, utyp: uTYP, ssig):
        gps_tbl = {
            sSigGPS.L1C: uSIG.L1C,
            sSigGPS.L1P: uSIG.L1P,
            sSigGPS.L1W: uSIG.L1W,
            sSigGPS.L1S: uSIG.L1S,
            sSigGPS.L1L: uSIG.L1L,
            sSigGPS.L1X: uSIG.L1X,
            sSigGPS.L2S: uSIG.L2S,
            sSigGPS.L2L: uSIG.L2L,
            sSigGPS.L2X: uSIG.L2X,
            sSigGPS.L2P: uSIG.L2P,
            sSigGPS.L2W: uSIG.L2W,
            sSigGPS.L5I: uSIG.L5I,
            sSigGPS.L5Q: uSIG.L5Q,
            sSigGPS.L5X: uSIG.L5X,
        }
        glo_tbl = {
            sSigGLO.L1C: uSIG.L1C,
            sSigGLO.L1P: uSIG.L1P,
            sSigGLO.L2C: uSIG.L2C,
            sSigGLO.L2P: uSIG.L2P,
            sSigGLO.L4A: uSIG.L4A,
            sSigGLO.L4B: uSIG.L4B,
            sSigGLO.L4X: uSIG.L4X,
            sSigGLO.L6A: uSIG.L6A,
            sSigGLO.L6B: uSIG.L6B,
            sSigGLO.L6X: uSIG.L6X,
            sSigGLO.L3I: uSIG.L3I,
            sSigGLO.L3Q: uSIG.L3Q,
            sSigGLO.L3X: uSIG.L3X,
        }

        gal_tbl = {
            sSigGAL.L1B: uSIG.L1B,
            sSigGAL.L1C: uSIG.L1C,
            sSigGAL.L1X: uSIG.L1X,
            sSigGAL.L5I: uSIG.L5I,
            sSigGAL.L5Q: uSIG.L5Q,
            sSigGAL.L5X: uSIG.L5X,
            sSigGAL.L7I: uSIG.L7I,
            sSigGAL.L7Q: uSIG.L7Q,
            sSigGAL.L7X: uSIG.L7X,
            sSigGAL.L8I: uSIG.L8I,
            sSigGAL.L8Q: uSIG.L8Q,
            sSigGAL.L8X: uSIG.L8X,
            sSigGAL.L6B: uSIG.L6B,
            sSigGAL.L6C: uSIG.L6C,
            sSigGAL.L6X: uSIG.L6X,
        }

        bds_tbl = {
            sSigBDS.L2I: uSIG.L2I,
            sSigBDS.L2Q: uSIG.L2Q,
            sSigBDS.L2X: uSIG.L2X,
            sSigBDS.L6I: uSIG.L6I,
            sSigBDS.L6Q: uSIG.L6Q,
            sSigBDS.L6X: uSIG.L6X,
            sSigBDS.L7I: uSIG.L7I,
            sSigBDS.L7Q: uSIG.L7Q,
            sSigBDS.L7X: uSIG.L7X,
            sSigBDS.L1D: uSIG.L1D,
            sSigBDS.L1P: uSIG.L1P,
            sSigBDS.L1X: uSIG.L1X,
            sSigBDS.L5D: uSIG.L5D,
            sSigBDS.L5P: uSIG.L5P,
            sSigBDS.L5X: uSIG.L5X,
        }

        qzs_tbl = {
            sSigQZS.L1C: uSIG.L1C,
            sSigQZS.L1S: uSIG.L1S,
            sSigQZS.L1L: uSIG.L1C,
            sSigQZS.L1X: uSIG.L1X,
            sSigQZS.L2S: uSIG.L2S,
            sSigQZS.L2L: uSIG.L2L,
            sSigQZS.L2X: uSIG.L2X,
            sSigQZS.L5I: uSIG.L5I,
            sSigQZS.L5Q: uSIG.L5Q,
            sSigQZS.L5X: uSIG.L5X,
            sSigQZS.L6D: uSIG.L6D,
            sSigQZS.L6P: uSIG.L6P,
            sSigQZS.L6E: uSIG.L6E,
            sSigQZS.L1E: uSIG.L1E,
        }

        sbs_tbl = {
            sSigSBS.L1C: uSIG.L1C,
            sSigSBS.L5I: uSIG.L5I,
            sSigSBS.L5Q: uSIG.L5Q,
            sSigSBS.L5X: uSIG.L5X,
        }

        irn_tbl = {
            sSigIRN.L1D: uSIG.L1D,
            sSigIRN.L1P: uSIG.L1P,
            sSigIRN.L1X: uSIG.L1X,
            sSigIRN.L5X: uSIG.L5X,
            sSigIRN.L9A: uSIG.L9A,
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

    def sval(self, u, n, scl):
        """ calculate signed value based on n-bit int, lsb """
        invalid = -2**(n-1)
        y = np.nan if u == invalid else u*scl
        return y

    def isset(self, mask, nbit, k):
        """ check if k-th bit in nbit mask is set """
        if (mask >> (nbit-k-1)) & 1:
            return True
        return False

    def set_t0(self, inet=0, sat=0, ctype=0, t=gtime_t()):
        """ set reference time for correction to check validity time """
        sc_t = [sCType.CLOCK, sCType.ORBIT, sCType.CBIAS, sCType.PBIAS,
                sCType.HCLOCK]

        if sat not in self.lc[inet].t0:
            self.lc[inet].t0[sat] = {}
            for sc in sc_t:
                self.lc[inet].t0[sat][sc] = gtime_t()

        self.lc[inet].t0[sat][ctype] = t

    def quality_idx(self, cl, val):
        """ calculate quality index """
        if cl == 7 and val == 7:
            y = 5.4665
        elif cl == 0 and val == 0:  # undefined/unknown
            y = np.nan
        else:
            y = (3**cl*(1+val*0.25)-1)*1e-3  # [m]
        return y

    def gnss2sys(self, gnss: sGNSS):
        """ convert from sGNSS to sys """
        tbl = {sGNSS.GPS: uGNSS.GPS, sGNSS.GLO: uGNSS.GLO,
               sGNSS.GAL: uGNSS.GAL, sGNSS.BDS: uGNSS.BDS,
               sGNSS.QZS: uGNSS.QZS, sGNSS.SBS: uGNSS.SBS,
               sGNSS.BDS3: uGNSS.BDS}
        if gnss not in tbl:
            return -1
        sys = tbl[gnss]
        return sys

    def decode_local_sat(self, netmask):
        """ decode netmask, and return list of local sats """
        sat = []
        for k in range(self.nsat_n):
            if not self.isset(netmask, self.nsat_n, k):
                continue
            sat.append(self.sat_n[k])
        return sat

    def decode_mask(self, din, bitlen, ofst=1):
        """ decode n-bit mask with offset """
        v = []
        n = 0
        for k in range(0, bitlen):
            if din & 1 << (bitlen-k-1):
                v.append(k+ofst)
                n += 1
        return (v, n)

    def decode_head(self, msg, i, st=-1):
        """ decode header of cssr message """
        if st == sCSSR.MASK:
            self.tow = bs.unpack_from('u20', msg, i)[0]
            i += 20
            self.tow0 = self.tow//3600*3600
        else:
            dtow = bs.unpack_from('u12', msg, i)[0]
            i += 12
            if self.tow >= 0:
                self.tow = self.tow0+dtow
        if self.week >= 0:
            self.time = gpst2time(self.week, self.tow)
        fmt = 'u4u1u4'
        names = ['uint', 'mi', 'iodssr']
        dfm = bs.unpack_from_dict(fmt, names, msg, i)
        i += 9
        return (dfm, i)

    def decode_cssr_mask(self, msg, i):
        """decode MT4073,1 Mask message """
        head, i = self.decode_head(msg, i, sCSSR.MASK)
        self.ngnss = bs.unpack_from('u4', msg, i)[0]
        self.flg_net = False
        i += 4

        if self.iodssr != head['iodssr']:
            self.sat_n_p = self.sat_n
            self.iodssr_p = self.iodssr
            self.sig_n_p = self.sig_n
        self.iodssr = head['iodssr']

        self.nsat_n = 0
        self.nsig_n = []
        self.sys_n = []
        self.gnss_n = []
        self.sat_n = []
        self.nsig_total = 0
        self.sig_n = {}
        self.nsig_max = 0
        self.nm_idx = np.zeros(self.SYSMAX, dtype=int)

        self.dcm = np.ones(self.SYSMAX)  # delta clock multipliers for HAS
        self.gnss_idx = np.zeros(self.ngnss, dtype=uGNSS)
        self.nsat_g = np.zeros(self.SYSMAX, dtype=int)

        for j in range(self.ngnss):
            gnss, svmask, sigmask, cma = bs.unpack_from('u4u40u16u1', msg, i)
            sys = self.gnss_idx[j] = self.gnss2sys(gnss)
            i += 61
            prn, nsat = self.decode_mask(svmask, 40)
            sig, nsig = self.decode_mask(sigmask, 16, 0)
            self.nsat_g[gnss] = nsat
            self.nsat_n += nsat
            if cma == 1:
                vc = bs.unpack_from(('u'+str(nsig))*nsat, msg, i)
                i += nsig*nsat

            self.nsig_max = max(self.nsig_max, nsig)

            sig_r = [self.ssig2rsig(sys, uTYP.L, s) for s in sig]

            for k in range(0, nsat):
                if gnss == sGNSS.QZS:
                    prn[k] += 192
                elif gnss == sGNSS.BDS3:
                    prn[k] += 18
                sat = prn2sat(sys, prn[k])
                self.sys_n.append(sys)
                self.sat_n.append(sat)
                self.gnss_n.append(gnss)
                if cma == 1:
                    sig_s, nsig_s = self.decode_mask(vc[k], nsig, 0)
                    sig_n = [sig_r[i] for i in sig_s]
                    self.nsig_n.append(nsig_s)
                    self.nsig_total += nsig_s
                    self.sig_n[sat] = sig_n
                else:
                    self.nsig_n.append(nsig)
                    self.nsig_total += nsig
                    self.sig_n[sat] = sig_r

            if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS only
                self.nm_idx[gnss] = bs.unpack_from('u3', msg, i)[0]
                i += 3

        if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS only
            i += 6

        self.lc[0].cstat |= (1 << sCType.MASK)
        self.set_t0(ctype=sCType.MASK, t=self.time)
        return i

    def decode_orb_sat(self, msg, i, sat, sys=uGNSS.NONE, inet=0):
        """ decoder orbit correction of cssr """
        n = 10 if sys == uGNSS.GAL else 8
        fmt = 'u{:d}s{:d}s{:d}s{:d}'\
            .format(n, self.dorb_blen[0], self.dorb_blen[1], self.dorb_blen[2])
        iode, dx, dy, dz = bs.unpack_from(fmt, msg, i)

        dorb = np.zeros(3)
        dorb[0] = self.sval(dx, self.dorb_blen[0], self.dorb_scl[0])
        dorb[1] = self.sval(dy, self.dorb_blen[1], self.dorb_scl[1])
        dorb[2] = self.sval(dz, self.dorb_blen[2], self.dorb_scl[2])

        self.lc[inet].iode[sat] = iode
        self.lc[inet].dorb[sat] = dorb

        if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS SIS
            self.lc[inet].dorb[sat] *= -1.0

        i += n + self.dorb_blen[0]+self.dorb_blen[1]+self.dorb_blen[2]
        return i

    def decode_clk_sat(self, msg, i, sat, inet=0):
        """ decoder clock correction of cssr """
        dclk = bs.unpack_from('s'+str(self.dclk_blen), msg, i)[0]

        self.lc[inet].dclk[sat] = self.sval(
            dclk, self.dclk_blen, self.dclk_scl)

        if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS SIS
            sys, _ = sat2prn(sat)
            self.lc[inet].dclk[sat] *= self.dcm[sys]

        i += self.dclk_blen
        return i

    def decode_cbias_sat(self, msg, i, sat, rsig, inet=0):
        """ decoder code bias correction of cssr """
        cb = bs.unpack_from('s'+str(self.cb_blen), msg, i)[0]
        self.lc[inet].cbias[sat][rsig] = self.sval(
            cb, self.cb_blen, self.cb_scl)
        i += self.cb_blen
        return i

    def decode_pbias_sat(self, msg, i, sat, rsig, inet=0):
        """ decoder phase bias correction of cssr """
        pb, di = bs.unpack_from('s'+str(self.pb_blen) + 'u2', msg, i)
        self.lc[inet].pbias[sat][rsig] = self.sval(
            pb, self.pb_blen, self.pb_scl)
        self.lc[inet].di[sat][rsig] = di
        i += self.pb_blen + 2
        return i

    def decode_cssr_orb(self, msg, i, inet=0):
        """decode MT4073,2 Orbit Correction message """
        head, i = self.decode_head(msg, i)
        self.flg_net = False
        self.lc[inet].dorb = {}
        self.lc[inet].iode = {}

        for k in range(self.nsat_n):
            i = self.decode_orb_sat(msg, i, self.sat_n[k], self.sys_n[k], inet)
            self.set_t0(inet, self.sat_n[k], sCType.ORBIT, self.time)

        self.iodssr_c[sCType.ORBIT] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.ORBIT)

        return i

    def decode_cssr_clk(self, msg, i, inet=0):
        """decode MT4073,3 Clock Correction message """
        head, i = self.decode_head(msg, i)
        self.flg_net = False
        # if self.iodssr != head['iodssr']:
        #    return -1

        if (self.lc[0].cstat & (1 << sCType.MASK)) != (1 << sCType.MASK):
            return -1

        if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS only
            self.dcm = {}
            for k in range(self.ngnss):
                self.dcm[self.gnss_idx[k]] = bs.unpack_from('u2', msg, i)[
                    0]+1.0
                i += 2

        self.lc[inet].dclk = {}
        for k in range(self.nsat_n):
            i = self.decode_clk_sat(msg, i, self.sat_n[k], inet)
            self.set_t0(inet, self.sat_n[k], sCType.CLOCK, self.time)

        if self.cssrmode == sCSSRTYPE.GAL_HAS_SIS:  # HAS only
            self.sat_n_p = self.sat_n
        self.iodssr_c[sCType.CLOCK] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CLOCK)

        return i

    def decode_cssr_cbias(self, msg, i, inet=0):
        """decode MT4073,4 Code Bias Correction message """
        head, i = self.decode_head(msg, i)
        nsat = self.nsat_n
        self.flg_net = False
        self.lc[inet].cbias = {}
        for k in range(nsat):
            sat = self.sat_n[k]
            self.lc[inet].cbias[sat] = {}
            for j in range(len(self.sig_n[sat])):
                rsig = self.sig_n[sat][j].toTyp(uTYP.C)
                i = self.decode_cbias_sat(msg, i, sat, rsig, inet)
                self.set_t0(inet, self.sat_n[k], sCType.CBIAS, self.time)

        self.iodssr_c[sCType.CBIAS] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.CBIAS)

        return i

    def decode_cssr_pbias(self, msg, i, inet=0):
        """decode MT4073,5 Phase Bias Correction message """
        head, i = self.decode_head(msg, i)
        nsat = self.nsat_n
        self.flg_net = False
        self.lc[inet].pbias = {}
        self.lc[inet].di = {}
        for k in range(nsat):
            sat = self.sat_n[k]
            self.lc[inet].pbias[sat] = {}
            self.lc[inet].di[sat] = {}
            for j in range(len(self.sig_n[sat])):
                rsig = self.sig_n[sat][j].toTyp(uTYP.L)
                i = self.decode_pbias_sat(msg, i, sat, rsig, inet)
                self.set_t0(inet, self.sat_n[k], sCType.PBIAS, self.time)

        self.iodssr_c[sCType.PBIAS] = head['iodssr']
        self.lc[inet].cstat |= (1 << sCType.PBIAS)

        return i

    def decode_cssr_bias(self, msg, i, inet=0):
        """decode MT4073,6 Bias Correction message """
        nsat = self.nsat_n
        head, i = self.decode_head(msg, i)
        # if self.iodssr != head['iodssr']:
        #    return -1
        dfm = bs.unpack_from_dict('b1b1b1', ['cb', 'pb', 'net'], msg, i)
        self.flg_net = dfm['net']
        i += 3
        if dfm['net']:
            v = bs.unpack_from_dict('u5u'+str(nsat),
                                    ['inet', 'svmaskn'], msg, i)
            self.inet = inet = v['inet']
            i += 5+nsat
            self.lc[inet].sat_n = self.decode_local_sat(v['svmaskn'])
            self.lc[inet].nsat_n = nsat = len(self.lc[inet].sat_n)

        if dfm['cb']:
            self.lc[inet].cbias = {}
        if dfm['pb']:
            self.lc[inet].pbias = {}
            self.lc[inet].di = {}
        ki = 0
        for k in range(self.nsat_n):
            if not self.isset(v['svmaskn'], self.nsat_n, k):
                continue
            sat = self.sat_n[k]
            if dfm['cb']:
                self.lc[inet].cbias[sat] = {}
            if dfm['pb']:
                self.lc[inet].pbias[sat] = {}
                self.lc[inet].di[sat] = {}
            for j in range(self.nsig_n[k]):
                if dfm['cb']:
                    rsig = self.sig_n[sat][j].toTyp(uTYP.C)
                    i = self.decode_cbias_sat(msg, i, sat, rsig, inet)
                    self.set_t0(inet, self.sat_n[k], sCType.CBIAS, self.time)

                if dfm['pb']:
                    rsig = self.sig_n[sat][j].toTyp(uTYP.L)
                    i = self.decode_pbias_sat(msg, i, sat, rsig, inet)
                    self.set_t0(inet, self.sat_n[k], sCType.PBIAS, self.time)

            ki += 1

        if dfm['cb']:
            self.iodssr_c[sCType.CBIAS] = head['iodssr']
            self.lc[inet].cstat |= (1 << sCType.CBIAS)
        if dfm['pb']:
            self.iodssr_c[sCType.PBIAS] = head['iodssr']
            self.lc[inet].cstat |= (1 << sCType.PBIAS)

        return i

    def decode_cssr_ura(self, msg, i):
        """decode MT4073,7 URA message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1
        self.ura = {}
        for k in range(self.nsat_n):
            sat = self.sat_n[k]
            v = bs.unpack_from_dict('u3u3', ['class', 'val'], msg, i)
            self.ura[sat] = self.quality_idx(v['class'], v['val'])
            i += 6
        self.lc[0].cstat |= (1 << sCType.URA)
        self.set_t0(ctype=sCType.URA, t=self.time)

        return i

    def decode_cssr_stec_coeff(self, msg, stype, i):
        """ decode coefficient of STEC correction """
        ci = np.zeros(6)
        v = bs.unpack_from('s14', msg, i)
        ci[0] = self.sval(v[0], 14, 0.05)
        i += 14
        if stype > 0:
            v = bs.unpack_from('s12s12', msg, i)
            ci[1] = self.sval(v[0], 12, 0.02)
            ci[2] = self.sval(v[1], 12, 0.02)
            i += 24
        if stype > 1:
            v = bs.unpack_from('s10', msg, i)
            ci[3] = self.sval(v[0], 10, 0.02)
            i += 10
        if stype > 2:
            v = bs.unpack_from('s8s8', msg, i)
            ci[4] = self.sval(v[0], 8, 0.005)
            ci[5] = self.sval(v[1], 8, 0.005)
            i += 16
        return (ci, i)

    def decode_cssr_stec(self, msg, i):
        """decode MT4073,8 STEC Correction message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1
        self.flg_net = True
        dfm = bs.unpack_from_dict('u2u5u'+str(self.nsat_n),
                                  ['stype', 'inet', 'svmaskn'], msg, i)
        inet = dfm['inet']
        self.netmask[inet] = netmask = dfm['svmaskn']
        self.lc[inet].sat_n = self.decode_local_sat(netmask)
        self.lc[inet].nsat_n = nsat = len(self.lc[inet].sat_n)
        i += 7+self.nsat_n
        self.lc[inet].stec_quality = {}
        self.lc[inet].ci = {}
        for k in range(nsat):
            sat = self.lc[inet].sat_n[k]
            v = bs.unpack_from_dict('u3u3', ['class', 'val'], msg, i)
            self.lc[inet].stec_quality[sat] = self.quality_idx(v['class'],
                                                               v['val'])
            i += 6
            ci, i = self.decode_cssr_stec_coeff(msg, dfm['stype'], i)
            self.lc[inet].ci[sat] = ci
            self.set_t0(inet, sat, sCType.STEC, self.time)

        self.lc[inet].cstat |= (1 << sCType.STEC)

        return i

    def decode_cssr_grid(self, msg, i):
        """decode MT4073,9 Grid Correction message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('u2u1u5u'+str(self.nsat_n)+'u3u3u6',
                                  ['ttype', 'range', 'inet', 'svmaskn',
                                  'class', 'value', 'ng'], msg, i)
        self.flg_net = True
        inet = dfm['inet']
        self.netmask[inet] = netmask = dfm['svmaskn']
        self.lc[inet].sat_n = self.decode_local_sat(netmask)
        self.lc[inet].nsat_n = nsat = len(self.lc[inet].sat_n)
        ng = dfm['ng']
        self.lc[inet].ng = ng
        self.lc[inet].trop_quality = self.quality_idx(dfm['class'],
                                                      dfm['value'])
        i += 20+self.nsat_n
        sz = 7 if dfm['range'] == 0 else 16
        fmt = 's'+str(sz)
        self.lc[inet].stec = np.zeros(ng, dtype=list)
        self.lc[inet].dtd = np.zeros(ng)
        self.lc[inet].dtw = np.zeros(ng)

        for j in range(ng):
            self.lc[inet].stec[j] = {}
            if dfm['ttype'] > 0:
                vd = bs.unpack_from_dict('s9s8', ['dtd', 'dtw'], msg, i)
                i += 17
                self.lc[inet].dtd[j] = self.sval(vd['dtd'], 9, 0.004)+2.3
                self.lc[inet].dtw[j] = self.sval(vd['dtw'], 8, 0.004)

            for k in range(nsat):
                sat = self.lc[inet].sat_n[k]
                dstec = bs.unpack_from(fmt, msg, i)[0]
                i += sz
                self.lc[inet].stec[j][sat] = self.sval(dstec, sz, 0.04)
        self.lc[inet].cstat |= (1 << sCType.TROP)
        for k in range(nsat):
            self.set_t0(inet, self.lc[inet].sat_n[k], sCType.TROP, self.time)

        return i

    def parse_sinfo(self):
        """decode content of service info """
        # TBD
        return 0

    def decode_cssr_sinfo(self, msg, i):
        """decode MT4073,10 Service Information message """
        dfm = bs.unpack_from_dict('b1u3u2', ['mi', 'cnt', 'dsize'], msg, i)
        self.flg_net = False
        i += 6
        n = dfm['dsize']+1
        j = n*40*dfm['cnt']
        for _ in range(n):
            d = bs.unpack_from('u40', msg, i)[0]
            i += 40
            bs.pack_into('u40', self. sinfo, j, d)
            j += 40
        if dfm['mi'] is False:
            self.parse_sinfo()
        return i

    def decode_cssr_comb(self, msg, i, inet=0):
        """decode MT4073,11 Orbit,Clock Combined Correction message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('b1b1b1', ['orb', 'clk', 'net'], msg, i)
        i += 3
        self.flg_net = dfm['net']
        if self.flg_net:
            v = bs.unpack_from_dict('u5u'+str(self.nsat_n),
                                    ['inet', 'svmask'], msg, i)
            self.inet = inet = v['inet']
            self.lc[inet].svmask = svmask = v['svmask']
            i += 5+self.nsat_n

        if dfm['orb']:
            self.lc[inet].dorb = {}
            self.lc[inet].iode = {}
        if dfm['clk']:
            self.lc[inet].dclk = {}

        for k in range(self.nsat_n):
            if self.flg_net and not self.isset(svmask, self.nsat_n, k):
                continue
            sat = self.sat_n[k]
            if dfm['orb']:
                i = self.decode_orb_sat(msg, i, sat, self.sys_n[k], inet)
                self.set_t0(inet, sat, sCType.ORBIT, self.time)

            if dfm['clk']:
                i = self.decode_clk_sat(msg, i, sat, inet)
                self.set_t0(inet, sat, sCType.CLOCK, self.time)

        if dfm['clk']:
            # self.iodssr_c[sCType.CLOCK] = head['iodssr']
            self.lc[inet].cstat |= (1 << sCType.CLOCK)

        if dfm['orb']:
            # self.iodssr_c[sCType.ORBIT] = head['iodssr']
            self.lc[inet].cstat |= (1 << sCType.ORBIT)

        return i

    def decode_cssr_atmos(self, msg, i):
        """decode MT4073,12 Atmospheric Correction message """
        head, i = self.decode_head(msg, i)
        if self.iodssr != head['iodssr']:
            return -1
        dfm = bs.unpack_from_dict('u2u2u5u6', ['trop', 'stec', 'inet', 'ng'],
                                  msg, i)
        self.flg_net = True
        inet = dfm['inet']
        self.lc[inet].ng = ng = dfm['ng']
        self.lc[inet].flg_trop = dfm['trop']
        self.lc[inet].flg_stec = dfm['stec']
        i += 15
        # trop
        if dfm['trop'] > 0:
            v = bs.unpack_from_dict('u3u3', ['class', 'value'], msg, i)
            self.lc[inet].trop_quality = self.quality_idx(v['class'],
                                                          v['value'])
            i += 6
        if dfm['trop'] & 2:  # functional term
            self.lc[inet].ttype = ttype = bs.unpack_from('u2', msg, i)[0]
            i += 2
            vt = bs.unpack_from_dict('s9', ['t00'], msg, i)
            i += 9
            self.lc[inet].ct = np.zeros(4)
            self.lc[inet].ct[0] = self.sval(vt['t00'], 9, 0.004)
            if ttype > 0:
                vt = bs.unpack_from_dict('s7s7', ['t01', 't10'], msg, i)
                i += 14
                self.lc[inet].ct[1] = self.sval(vt['t01'], 7, 0.002)
                self.lc[inet].ct[2] = self.sval(vt['t10'], 7, 0.002)
            if ttype > 1:
                vt = bs.unpack_from_dict('s7', ['t11'], msg, i)
                i += 7
                self.lc[inet].ct[3] = self.sval(vt['t11'], 7, 0.001)

        if dfm['trop'] & 1:  # residual term
            vh = bs.unpack_from_dict('u1u4', ['sz', 'ofst'], msg, i)
            i += 5
            trop_ofst = vh['ofst']*0.02
            sz = 6 if vh['sz'] == 0 else 8
            vtr = bs.unpack_from(('s'+str(sz))*ng, msg, i)
            i += sz*ng
            self.lc[inet].dtw = np.zeros(ng)
            for k in range(ng):
                self.lc[inet].dtw[k] = self.sval(vtr[k], sz, 0.004)+trop_ofst

        # STEC
        netmask = bs.unpack_from('u'+str(self.nsat_n), msg, i)[0]
        i += self.nsat_n
        self.lc[inet].netmask = netmask
        self.lc[inet].sat_n = self.decode_local_sat(netmask)
        self.lc[inet].nsat_n = nsat = len(self.lc[inet].sat_n)
        self.lc[inet].stec_quality = {}
        if dfm['stec'] & 2 > 0:  # functional term is available
            self.lc[inet].ci = {}
            self.lc[inet].stype = {}
        if dfm['stec'] & 1 > 0:  # residual term is available
            self.lc[inet].dstec = {}
            self.lc[inet].s_sz = {}

        for k in range(nsat):
            sat = self.lc[inet].sat_n[k]
            self.set_t0(inet, sat, sCType.STEC, self.time)

            if dfm['stec'] > 0:
                v = bs.unpack_from_dict('u3u3', ['class', 'value'], msg, i)
                i += 6
                self.lc[inet].stec_quality[sat] = self.quality_idx(v['class'],
                                                                   v['value'])
            if dfm['stec'] & 2 > 0:  # functional term
                self.lc[inet].stype[sat] = bs.unpack_from('u2', msg, i)[0]
                i += 2
                ci, i = self.decode_cssr_stec_coeff(msg,
                                                    self.lc[inet].stype[sat],
                                                    i)
                self.lc[inet].ci[sat] = ci

            if dfm['stec'] & 1 > 0:  # residual term
                sz_idx = bs.unpack_from('u2', msg, i)[0]
                i += 2
                self.lc[inet].s_sz[sat] = sz_idx
                sz = self.stec_sz_t[sz_idx]
                scl = self.stec_scl_t[sz_idx]
                v = bs.unpack_from(('s'+str(sz))*ng, msg, i)
                i += sz*ng
                self.lc[inet].dstec[sat] = np.zeros(ng)
                for j in range(ng):
                    self.lc[inet].dstec[sat][j] = self.sval(v[j], sz, scl)

        if dfm['trop'] > 0:
            self.lc[inet].cstat |= (1 << sCType.TROP)
            self.set_t0(inet, 0, sCType.TROP, self.time)

        if dfm['stec'] > 0:
            self.lc[inet].cstat |= (1 << sCType.STEC)

        return i

    def decode_cssr(self, msg, i=0):
        """decode Compact SSR message """
        df = {'msgtype': 4073}
        while df['msgtype'] == 4073:
            df = bs.unpack_from_dict('u12u4', ['msgtype', 'subtype'], msg, i)
            i += 16
            if df['msgtype'] != 4073:
                return -1
            self.subtype = df['subtype']
            if self.subtype == sCSSR.MASK:
                i = self.decode_cssr_mask(msg, i)
            elif self.subtype == sCSSR.ORBIT:  # orbit
                i = self.decode_cssr_orb(msg, i)
            elif self.subtype == sCSSR.CLOCK:  # clock
                i = self.decode_cssr_clk(msg, i)
            elif self.subtype == sCSSR.CBIAS:  # cbias
                i = self.decode_cssr_cbias(msg, i)
            elif self.subtype == sCSSR.PBIAS:  # pbias
                i = self.decode_cssr_pbias(msg, i)
            elif self.subtype == sCSSR.BIAS:  # bias
                i = self.decode_cssr_bias(msg, i)
            elif self.subtype == sCSSR.URA:  # ura
                i = self.decode_cssr_ura(msg, i)
            elif self.subtype == sCSSR.STEC:  # stec
                i = self.decode_cssr_stec(msg, i)
            elif self.subtype == sCSSR.GRID:  # grid
                i = self.decode_cssr_grid(msg, i)
            elif self.subtype == sCSSR.SI:  # service-info
                i = self.decode_cssr_sinfo(msg, i)
            elif self.subtype == sCSSR.COMBINED:  # orb+clk
                i = self.decode_cssr_comb(msg, i)
            elif self.subtype == sCSSR.ATMOS:  # atmos
                i = self.decode_cssr_atmos(msg, i)
            if i <= 0:
                return 0
            if self.monlevel >= 2:
                if self.flg_net:
                    print("tow={:6d} subtype={:2d} inet={:2d}".
                          format(self.tow, self.subtype, self.inet))
                else:
                    print("tow={:6.0f} subtype={:2d}".format(self.tow,
                                                             self.subtype))
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()

    def chk_stat(self):
        """ check status for received messages """
        cs_global = self.lc[0].cstat
        cs_local = self.lc[self.inet_ref].cstat

        if (cs_global & 0x0f) != 0x0f:  # mask,orb,clk,cbias
            return False
        if (cs_local & 0x60) != 0x60:  # stec,trop
            return False
        if self.local_pbias and (cs_local & 0x10) != 0x10:  # pbias(loc)
            return False
        return True

    def check_validity(self, time):
        """ check validity time of correction """
        for sat in self.sat_n:
            if timediff(time, self.lc[0].t0[sat][sCType.CLOCK]) > \
                    self.tmax[sCType.CLOCK]:
                self.lc[0].dclk[sat] = 0.0
            if timediff(time, self.lc[0].t0[sat][sCType.ORBIT]) > \
                    self.tmax[sCType.ORBIT]:
                self.lc[0].iode[sat] = -1
                self.lc[0].dorb[sat] = np.zeros(3)

    def out_log(self):

        if self.time == -1:
            return

        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))
        if (self.lc[0].cstat & (1 << sCType.MASK)) != (1 << sCType.MASK):
            return

        if self.subtype == sCSSR.CLOCK:
            self.fh.write(" {:s}\t{:s}\n".format("SatID", "dclk [m]"))
            for sat_ in self.lc[0].dclk.keys():
                if np.isnan(self.lc[0].dclk[sat_]):
                    continue
                self.fh.write(" {:s}\t{:8.4f}\n"
                              .format(sat2id(sat_),
                                      self.lc[0].dclk[sat_]))

        elif self.subtype == sCSSR.ORBIT:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "IODE", "Radial[m]",
                                  "Along[m]", "Cross[m]"))
            for sat_ in self.lc[0].dorb.keys():
                if np.isnan(self.lc[0].dorb[sat_][0]):
                    continue
                self.fh.write(" {:s}\t{:3d}\t{:6.3f}\t{:6.3f}\t{:6.3f}\n"
                              .format(sat2id(sat_),
                                      self.lc[0].iode[sat_],
                                      self.lc[0].dorb[sat_][0],
                                      self.lc[0].dorb[sat_][1],
                                      self.lc[0].dorb[sat_][2]))

        elif self.subtype == sCSSR.COMBINED:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "IODE", "Radial[m]",
                                  "Along[m]", "Cross[m]", "dclk[m]"))
            for sat_ in self.lc[0].dorb.keys():
                if np.isnan(self.lc[0].dorb[sat_][0]) or \
                   np.isnan(self.lc[0].dclk[sat_]):
                    continue
                self.fh.write(
                    " {:s}\t{:3d}\t{:6.3f}\t{:6.3f}\t{:6.3f}\t{:8.4f}\n"
                    .format(sat2id(sat_),
                            self.lc[0].iode[sat_],
                            self.lc[0].dorb[sat_][0],
                            self.lc[0].dorb[sat_][1],
                            self.lc[0].dorb[sat_][2],
                            self.lc[0].dclk[sat_]))

        elif self.subtype == sCSSR.CBIAS:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "Code Bias[m]", "..."))
            for sat_ in self.lc[0].cbias.keys():
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for sig_ in self.lc[0].cbias[sat_].keys():
                    self.fh.write("{:s}\t{:5.2f}\t"
                                  .format(sig_.str(),
                                          self.lc[0].cbias[sat_][sig_]))
                self.fh.write("\n")

        elif self.subtype == sCSSR.PBIAS:
            self.fh.write(" {:s}\t{:s}\t{:s}\t{:s}\n"
                          .format("SatID", "SigID", "Phase Bias[m]", "..."))
            for sat_ in self.lc[0].pbias.keys():
                self.fh.write(" {:s}\t".format(sat2id(sat_)))
                for sig_ in self.lc[0].pbias[sat_].keys():
                    self.fh.write("{:s}\t{:5.2f}\t"
                                  .format(sig_.str(),
                                          self.lc[0].pbias[sat_][sig_]))
                self.fh.write("\n")

        self.fh.flush()

    def read_griddef(self, file):
        """load grid coordinates from file """
        dtype0 = [('nid', '<i4'), ('gid', '<i4'),
                  ('lat', '<f8'), ('lon', '<f8'), ('alt', '<f8')]
        self.grid = np.genfromtxt(file, dtype=dtype0, skip_header=1,
                                  skip_footer=0, encoding='utf8')

    def find_grid_index(self, pos):
        """ find index/weight of surounding grid   """
        self.rngmin = 5e3
        clat = np.cos(pos[0])
        dlat = np.deg2rad(self.grid['lat'])-pos[0]
        dlon = (np.deg2rad(self.grid['lon'])-pos[1])*clat

        r = np.linalg.norm((dlat, dlon), axis=0)*rCST.RE_WGS84
        idx = np.argmin(r)
        self.inet_ref = self.grid[idx]['nid']
        if r[idx] < self.rngmin:
            self.ngrid = 1
            self.grid_index = self.grid['gid'][idx]
            self.grid_weight = [1]
        else:
            idn = self.grid['nid'] == self.inet_ref
            rn = r[idn]
            self.ngrid = n = min(len(rn), 4)
            idx = np.argsort(rn)

            if n < 4:
                idx_n = idx[0:n]
            else:
                idx_n = idx[0:3]
                # select 4th grid between v21 and v31
                v = np.vstack((dlat[idn][idx], dlon[idn][idx]))
                vp = -v[:, 0]  # relative rover position
                vn = v[:, 1:].copy()  # relative grid position
                vn[0, :] = vn[0, :] + vp[0]
                vn[1, :] = vn[1, :] + vp[1]
                vn1 = np.array((-vn[:, 0][1], vn[:, 0][0]))  # normal of v21
                vn2 = np.array((-vn[:, 1][1], vn[:, 1][0]))  # normal of v31
                s1 = vn1@vp
                s2 = vn2@vp
                for k, i in enumerate(idx[3:]):
                    if s1*(vn1@vn[:, k+2]) >= 0 and s2*(vn2@vn[:, k+2]) >= 0:
                        idx_n = np.append(idx_n, i)
                        break

            rp = 1./rn[idx_n]
            w = rp/np.sum(rp)
            self.grid_index = self.grid[idn]['gid'][idx_n]
            self.grid_weight = w
        return self.inet_ref

    def get_dpos(self, pos):
        """ calculate position offset from reference """
        inet = self.inet_ref
        posd = np.rad2deg(pos[0:2])
        grid = self.grid[self.grid['nid'] == inet]
        dlat = posd[0]-grid[0]['lat']
        dlon = posd[1]-grid[0]['lon']
        return dlat, dlon

    def get_trop(self, dlat=0.0, dlon=0.0):
        """ calculate trop delay correction by interporation """
        inet = self.inet_ref
        trph = 0
        trpw = 0
        if self.lc[inet].flg_trop & 2:
            trph = 2.3+self.lc[inet].ct@[1, dlat, dlon, dlat*dlon]
        if self.lc[inet].flg_trop & 1:
            trpw = self.lc[inet].dtw[self.grid_index-1]@self.grid_weight
        return trph, trpw

    def get_stec(self, dlat=0.0, dlon=0.0):
        """ calculate STEC correction by interporation """
        inet = self.inet_ref
        nsat = self.lc[inet].nsat_n
        stec = np.zeros(nsat)
        for i in range(nsat):
            sat = self.lc[inet].sat_n[i]
            if self.lc[inet].flg_stec & 2:
                ci = self.lc[inet].ci[sat]
                stec[i] = [1, dlat, dlon, dlat*dlon, dlat**2, dlon**2]@ci
            if self.lc[inet].flg_stec & 1:
                dstec = self.lc[inet].dstec[sat][self.grid_index-1] \
                    @ self.grid_weight
                stec[i] += dstec
        return stec

    #
    # QZS CLAS specific function
    #
    def decode_l6msg(self, msg, ofst):
        """decode QZS L6 message """
        fmt = 'u32u8u3u2u2u1u1'
        names = ['preamble', 'prn', 'vendor', 'facility', 'pt', 'sid',
                 'alert']
        facility_t = [[0, 2, 1, 3], [2, 0, 3, 1]]  # Table 4.1.2-2
        i = ofst*8
        l6head = bs.unpack_from_dict(fmt, names, msg, i)
        if l6head['preamble'] != 0x1acffc1d:
            return -1
        i += 49
        if l6head['sid'] == 1:
            self.fcnt = 0
            self.buff_p = copy.copy(self.buff)
            self.buff = bytearray(250*5)

        if l6head['vendor'] == 5:  # CLAS
            facility = facility_t[l6head['pt']][l6head['facility']]
        else:
            facility = l6head['facility']
        if self.facility >= 0 and facility != self.facility:
            self.fcnt = -1
        facility_p = self.facility
        self.facility = facility
        self.pattern = l6head['pt']
        self.sid = l6head['sid']
        if self.fcnt < 0:
            print(f"facility changed: {facility_p} -> {facility}")
            return -1

        copy_buff(msg, self.buff, i, 1695*self.fcnt, 1695)
        self.fcnt = self.fcnt + 1


class cssre():
    """ Class to encode the Compact SSR messages (experimental) """
    monlevel = 0
    dlen = 0
    msgtype = 0

    def __init__(self):
        super().__init__()
        self.len = 0
        self.dlen = 0
        self.msgtype = 0

    def encode_mask(self, v, bitlen, ofst=1):
        """ encode n-bit mask with offset """
        d = 0
        for p in v:
            k = p-ofst
            d |= 1 << (bitlen-k-1)

        return d
