"""
module for GNSS processing
"""

from copy import deepcopy
from enum import IntEnum
from math import floor, sin, cos, sqrt, asin, atan2, fabs
import numpy as np
from datetime import datetime, timezone

gpst0 = [1980, 1, 6, 0, 0, 0]  # GPS system time reference
gst0 = [1999, 8, 22, 0, 0, 0]  # Galileo system time reference
bdt0 = [2006, 1, 1, 0, 0, 0]  # BeiDou system time reference


class rCST():
    """ class for constants """
    CLIGHT = 299792458.0
    MU_GPS = 3.9860050E14
    MU_GLO = 3.9860044E14
    MU_GAL = 3.986004418E14
    MU_BDS = 3.986004418E14
    J2_GLO = 1.0826257E-3
    GME = 3.986004415E+14
    GMS = 1.327124E+20
    GMM = 4.902801E+12
    OMGE = 7.2921151467E-5
    OMGE_GLO = 7.292115E-5
    OMGE_GAL = 7.2921151467E-5
    OMGE_BDS = 7.2921150E-5
    RE_WGS84 = 6378137.0
    FE_WGS84 = (1.0/298.257223563)
    AU = 149597870691.0
    D2R = 0.017453292519943295
    R2D = 57.29577951308232
    AS2R = D2R/3600.0
    DAY_SEC = 86400.0
    CENTURY_SEC = DAY_SEC*36525.0

    FREQ_G1 = 1575.42e6      # [Hz] GPS L1
    FREQ_G2 = 1227.60e6      # [Hz] GPS L2
    FREQ_G5 = 1176.45e6      # [Hz] GPS L5

    FREQ_R1 = 1602.000e6     # [Hz] GLO G1   FDMA center frequency
    FREQ_R1k = 562.500e3     # [Hz] GLO G1   FDMA frequency separation
    FREQ_R2 = 1246.000e6     # [Hz] GLO G2   FDMA center frequency
    FREQ_R2k = 437.500e3     # [Hz] GLO G2   FDMA frequency separation
    FREQ_R1a = 1600.995e6    # [Hz] GLO G1a
    FREQ_R2a = 1248.065e6    # [Hz] GLO G2a
    FREQ_R3 = 1202.025e6     # [Hz] GLO G3

    FREQ_E1 = 1575.42e6      # [Hz] GAL E1
    FREQ_E5a = 1176.450e6    # [Hz] GAL E5a
    FREQ_E5b = 1207.140e6    # [Hz] GAL E5b
    FREQ_E5 = 1191.795e6     # [Hz] GAL E5
    FREQ_E6 = 1278.750e6     # [Hz] GAL E6

    FREQ_C1 = 1575.42e6      # [Hz] BDS B1
    FREQ_C12 = 1561.098e6    # [Hz] BDS B1-2
    FREQ_C2a = 1176.450e6    # [Hz] BDS B2a
    FREQ_C2b = 1207.140e6    # [Hz] BDS B2b
    FREQ_C2 = 1191.795e6     # [Hz] BDS B2
    FREQ_C3 = 1268.520e6     # [Hz] BDS B3

    FREQ_J1 = 1575.42e6      # [Hz] QZS L1

    FREQ_J2 = 1227.60e6      # [Hz] QZS L2
    FREQ_J5 = 1176.45e6      # [Hz] QZS L5
    FREQ_J6 = 1278.75e6      # [Hz] QZS LEX

    FREQ_S1 = 1575.42e6      # [Hz] SBS L1
    FREQ_S5 = 1176.45e6      # [Hz] SBS L5

    FREQ_I5 = 1191.795e6     # [Hz] IRS L5
    FREQ_IS = 2492.028e6     # [Hz] IRS S

    COS_5 = 0.9961946980917456
    SIN_5 = -0.0871557427476582

    P2_5 = 0.03125
    P2_6 = 0.015625
    P2_8 = 0.00390625
    P2_10 = 9.765625000000000E-04
    P2_11 = 4.882812500000000E-04
    P2_12 = 2.441406250000000E-04
    P2_13 = 1.220703125000000E-04
    P2_14 = 6.103515625000000E-05
    P2_15 = 3.051757812500000E-05
    P2_16 = 1.525878906250000E-05
    P2_17 = 7.629394531250000E-06
    P2_19 = 1.907348632812500E-06
    P2_20 = 9.536743164062500E-07
    P2_21 = 4.768371582031250E-07
    P2_28 = 3.725290298461914E-09
    P2_29 = 1.862645149230957E-09
    P2_30 = 9.313225746154785E-10
    P2_31 = 4.656612873077393E-10
    P2_32 = 2.328306436538696E-10
    P2_33 = 1.164153218269348E-10
    P2_34 = 5.820766091346740E-11
    P2_35 = 2.910383045673370E-11
    P2_37 = 7.275957614183426E-12
    P2_38 = 3.637978807091713E-12
    P2_39 = 1.818989403545856E-12
    P2_40 = 9.094947017729280E-13
    P2_41 = 4.547473508864641E-13
    P2_43 = 1.136868377216160E-13
    P2_44 = 5.684341886080802E-14
    P2_46 = 1.421085471520200E-14
    P2_48 = 3.552713678800501E-15
    P2_49 = 1.776356839400251E-15
    P2_50 = 8.881784197001252E-16
    P2_51 = 4.440892098500626E-16
    P2_55 = 2.775557561562891E-17
    P2_57 = 6.938893903907228E-18
    P2_59 = 1.734723475976810E-18
    P2_60 = 8.673617379884035E-19
    P2_66 = 1.355252715606880E-20
    P2_68 = 3.388131789017201E-21
    SC2RAD = 3.1415926535898


class uGNSS(IntEnum):
    """ class for GNS systems """

    NONE = -1

    GPS = 0
    GAL = 1
    QZS = 2
    BDS = 3
    GLO = 4
    SBS = 5
    IRN = 6

    GNSSMAX = 7

    GPSMAX = 32
    GALMAX = 36
    QZSMAX = 10
    BDSMAX = 63
    GLOMAX = 27
    SBSMAX = 39
    IRNMAX = 10

    GPSMIN = 0
    GALMIN = GPSMIN+GPSMAX
    QZSMIN = GALMIN+GALMAX
    BDSMIN = QZSMIN+QZSMAX
    GLOMIN = BDSMIN+BDSMAX
    SBSMIN = GLOMIN+GLOMAX
    IRNMIN = SBSMIN+SBSMAX

    MAXSAT = GPSMAX+GALMAX+QZSMAX+BDSMAX+GLOMAX+SBSMAX+IRNMAX


class uTYP(IntEnum):
    """ class for observation types"""

    NONE = -1

    C = 1
    L = 2
    D = 3
    S = 4


class uSIG(IntEnum):
    """ class for signal band and attribute """

    NONE = -1

    # GPS   L1  1575.42 MHz
    # GLO   G1  1602+k*9/16 MHz
    # GAL   E1  1575.42 MHz
    # SBAS  L1  1575.42 MHz
    # QZSS  L1  1575.42 MHz
    # BDS-3 B1  1575.42 MHz
    L1 = 100
    L1A = 101
    L1B = 102
    L1C = 103
    L1D = 104
    L1E = 105
    L1L = 112
    L1M = 113
    L1N = 114
    L1P = 116
    L1S = 119
    L1W = 123
    L1X = 124
    L1Y = 125
    L1Z = 126

    # GPS   L2  1227.60  MHz
    # GLO   G2  1246+k*7/16 MHz
    # QZS   L2  1227.60  MHz
    # BDS   B1  1561.098 MHz
    L2 = 200
    L2C = 203
    L2D = 204
    L2I = 209
    L2L = 212
    L2M = 213
    L2N = 214
    L2P = 216
    L2Q = 217
    L2S = 219
    L2W = 223
    L2X = 224
    L2Y = 225

    # GLO G3 1202.025 MHz
    L3 = 300
    L3I = 309
    L3Q = 317
    L3X = 324

    # GLO G1a 1600.995 MHz
    L4 = 400
    L4A = 401
    L4B = 402
    L4X = 424

    # GPS   L5  1176.45 MHz
    # GAL   E5  1176.45 MHz
    # SBS   L5  1176.45 MHz
    # QZS   L5  1176.45 MHz
    # BDS-3 B2a 1176.45 MHz
    # IRN   L5  1176.45 MHz
    L5 = 500
    L5A = 501
    L5B = 502
    L5C = 503
    L5D = 504
    L5I = 509
    L5P = 516
    L5Q = 517
    L5X = 524
    L5Z = 526

    # GLO   G2a 1248.06 MHz
    # GAL   E6  1278.75 MHz
    # QZS   L6  1278.75 MHz
    # BDS   B3  1278.75 MHz
    L6 = 600
    L6A = 601
    L6B = 602
    L6C = 603
    L6D = 604
    L6E = 605
    L6I = 609
    L6L = 612
    L6P = 616
    L6Q = 617
    L6S = 619
    L6X = 624
    L6Z = 626

    # GAL   E5b 1207.14 MHz
    # BDS-2 B2  1207.14 MHz
    # BDS-3 B2b 1207.14 MHz
    L7 = 700
    L7D = 704
    L7I = 709
    L7P = 716
    L7Q = 717
    L7X = 724
    L7Z = 726

    # GAL  E5a+b 1191.795 MHz
    # BDS  B2a+b 1191.795 MHz
    L8 = 800
    L8D = 804
    L8I = 809
    L8P = 816
    L8Q = 817
    L8X = 824

    # IRN  S    2492.028 MHz
    L9 = 900
    L9A = 901
    L9B = 902
    L9C = 903
    L9X = 924


class rSigRnx():

    def __init__(self, *args, **kwargs):
        """ Constructor """

        self.sys = uGNSS.NONE
        self.typ = uTYP.NONE
        self.sig = uSIG.NONE

        # Empty
        if len(args) == 0:

            self.sys = uGNSS.NONE
            self.typ = uTYP.NONE
            self.sig = uSIG.NONE

        # Four char string e.g. GC1W
        elif len(args) == 1:

            [x] = args
            if isinstance(x, str) and 3 <= len(x) <= 4:
                tmp = rSigRnx()
                tmp.str2sig(char2sys(x[0]), x[1:])
                self.sys = tmp.sys
                self.typ = tmp.typ
                self.sig = tmp.sig
            else:
                raise ValueError

        # System and three char string e.g. GPS, C1W
        elif len(args) == 2:

            sys, sig = args
            if isinstance(sys, uGNSS) and isinstance(sig, str) and \
                    2 <= len(sig) <= 3:
                tmp = rSigRnx()
                tmp.str2sig(sys, sig)
                self.sys = tmp.sys
                self.typ = tmp.typ
                self.sig = tmp.sig
            else:
                raise ValueError

        # System, type and signal
        elif len(args) == 3:

            sys, typ, sig = args
            if isinstance(sys, uGNSS) and \
                    isinstance(typ, uTYP) and \
                    isinstance(sig, uSIG):
                self.sys = sys
                self.typ = typ
                self.sig = sig
            else:
                raise ValueError

        else:

            raise ValueError

    def __repr__(self) -> str:
        """ string representation """
        return sys2char(self.sys)+self.str()

    def __eq__(self, other):
        """ equality operator """
        return self.sys == other.sys and \
            self.typ == other.typ and \
            self.sig == other.sig

    def __hash__(self):
        """ hash operator """
        return hash((self.sys, self.typ, self.sig))

    def toTyp(self, typ):
        """ Replace signal type """
        if isinstance(typ, uTYP):
            return rSigRnx(self.sys, typ, self.sig)
        else:
            raise ValueError

    def toAtt(self, att=""):
        """ Replace signal attribute """
        if isinstance(att, str):
            return rSigRnx(self.sys, self.str()[0:2]+att)
        else:
            raise ValueError

    def isGPS_PY(self):
        """
        Check if signal is GPS P(Y) tracking
        """
        return self.sys == uGNSS.GPS and \
            (self.sig == uSIG.L1W or self.sig == uSIG.L2W)

    def str2sig(self, sys, s):
        """ string to signal code conversion """

        if isinstance(sys, uGNSS) and isinstance(s, str):
            self.sys = sys
        else:
            raise ValueError

        s = s.strip()
        if len(s) < 2:
            raise ValueError

        if s[0] == 'C':
            self.typ = uTYP.C
        elif s[0] == 'L':
            self.typ = uTYP.L
        elif s[0] == 'D':
            self.typ = uTYP.D
        elif s[0] == 'S':
            self.typ = uTYP.S
        else:
            raise ValueError

        # Convert frequency ID
        #
        sig = int(s[1])*100

        # Check for valid tracking attribute
        #
        if len(s) == 3:
            if sys == uGNSS.GPS:
                if (s[1] == '1' and s[2] not in 'CSLXPWYM') or \
                   (s[2] == '2' and s[2] not in 'CDSLXPWYMN') or \
                   (s[2] == '5' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.GLO:
                if (s[1] == '1' and s[2] not in 'CP') or \
                   (s[1] == '2' and s[2] not in 'CP') or \
                   (s[1] == '3' and s[2] not in 'IQX') or \
                   (s[1] == '4' and s[2] not in 'ABX') or \
                   (s[1] == '6' and s[2] not in 'ABX'):
                    raise ValueError
            elif sys == uGNSS.GAL:
                if (s[1] == '1' and s[2] not in 'ABCXZ') or \
                   (s[1] == '5' and s[2] not in 'IQX') or \
                   (s[1] == '6' and s[2] not in 'ABCXZ') or \
                   (s[1] == '7' and s[2] not in 'IQX') or \
                   (s[1] == '8' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.SBS:
                if (s[1] == '1' and s[2] not in 'C') or \
                   (s[1] == '5' and s[2] not in 'IQX'):
                    raise ValueError
            elif sys == uGNSS.QZS:
                if (s[1] == '1' and s[2] not in 'CESLXZB') or \
                   (s[1] == '2' and s[2] not in 'SLX') or \
                   (s[1] == '5' and s[2] not in 'IQXDPZ') or \
                   (s[1] == '6' and s[2] not in 'SLXEZ'):
                    raise ValueError
            elif sys == uGNSS.BDS:
                if (s[1] == '2' and s[2] not in 'IQX') or \
                   (s[1] == '1' and s[2] not in 'DPXSLZ') or \
                   (s[1] == '5' and s[2] not in 'DPX') or \
                   (s[1] == '7' and s[2] not in 'IQXDPZ') or \
                   (s[1] == '8' and s[2] not in 'DPX') or \
                   (s[1] == '6' and s[2] not in 'IQXDPZ'):
                    raise ValueError
            elif sys == uGNSS.IRN:
                if (s[1] == '5' and s[2] not in 'ABCX') or \
                   (s[1] == '9' and s[2] not in 'ABCX'):
                    raise ValueError

            sig += ord(s[2]) - ord('A') + 1

        self.sig = uSIG(sig)

    def str(self):
        """ signal code to string conversion """

        s = ''

        if self.typ == uTYP.C:
            s += 'C'
        elif self.typ == uTYP.L:
            s += 'L'
        elif self.typ == uTYP.D:
            s += 'D'
        elif self.typ == uTYP.S:
            s += 'S'
        else:
            return '???'

        s += '{}'.format(int(self.sig / 100))

        if self.sig % 100 == 0:
            s += ' '
        else:
            s += '{}'.format(chr(self.sig % 100+ord('A')-1))

        return s

    def band(self):
        """
        Retrieve signal band
        """
        return uSIG((self.sig//100)*100)

    def frequency(self, k=None):
        """ frequency in Hz """

        if self.sys == uGNSS.GPS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_G1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_G2
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_G5
            else:
                return None
        elif self.sys == uGNSS.GLO:
            if int(self.sig / 100) == 1 and k is not None:
                return rCST.FREQ_R1 + k * rCST.FREQ_R1k
            elif int(self.sig / 100) == 2 and k is not None:
                return rCST.FREQ_R2 + k * rCST.FREQ_R2k
            elif int(self.sig / 100) == 3:
                return rCST.FREQ_R3
            elif int(self.sig / 100) == 4:
                return rCST.FREQ_R1a
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_R2a
            else:
                return None
        elif self.sys == uGNSS.GAL:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_E1
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_E5a
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_E6
            elif int(self.sig / 100) == 7:
                return rCST.FREQ_E5b
            elif int(self.sig / 100) == 8:
                return rCST.FREQ_E5
            else:
                return None
        elif self.sys == uGNSS.BDS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_C1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_C12
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_C2a
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_C3
            elif int(self.sig / 100) == 7:
                return rCST.FREQ_C2b
            elif int(self.sig / 100) == 8:
                return rCST.FREQ_C2
            else:
                return None
        if self.sys == uGNSS.QZS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_J1
            elif int(self.sig / 100) == 2:
                return rCST.FREQ_J2
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_J5
            elif int(self.sig / 100) == 6:
                return rCST.FREQ_J6
            else:
                return None
        if self.sys == uGNSS.SBS:
            if int(self.sig / 100) == 1:
                return rCST.FREQ_S1
            elif int(self.sig / 100) == 5:
                return rCST.FREQ_S5
        elif self.sys == uGNSS.IRN:
            if int(self.sig / 100) == 5:
                return rCST.FREQ_I5
            elif int(self.sig / 100) == 9:
                return rCST.FREQ_IS
        else:
            return None

    def wavelength(self, k=None):
        """ wavelength in [m] """

        frq = self.frequency(k)
        return rCST.CLIGHT/frq if frq is not None else None


class gtime_t():
    """ class to define the time """

    def __init__(self, time=0, sec=0.0):
        self.time = time
        self.sec = sec

    def __gt__(self, other):
        return self.time > other.time or \
            (self.time == other.time and self.sec > other.sec)


class Obs():
    """ class to define the observation """

    def __init__(self):
        self.t = gtime_t()
        self.P = []
        self.L = []
        self.S = []
        self.lli = []
        self.sat = []
        self.sig = {}


class Eph():
    """ class to define ephemeris """
    sat = 0
    iode = 0
    iodc = 0
    af0 = 0.0
    af1 = 0.0
    af2 = 0.0
    toc = 0
    toe = 0
    tot = 0
    top = 0
    week = 0
    crs = 0.0
    crc = 0.0
    cus = 0.0
    cus = 0.0
    cis = 0.0
    cic = 0.0
    e = 0.0
    i0 = 0.0
    A = 0.0
    Adot = 0.0
    deln = 0.0
    delnd = 0.0
    M0 = 0.0
    OMG0 = 0.0
    OMGd = 0.0
    omg = 0.0
    idot = 0.0
    tgd = 0.0
    tgd_b = 0.0
    tgd_c = 0.0
    sva = 0
    svh = 0
    fit = 0
    toes = 0
    tops = 0
    l2p = 0
    sattype = 0
    sismai = 0
    code = 0
    urai = np.zeros(3)
    sisai = np.zeros(4)
    isc = np.zeros(6)
    integ = 0
    # 0:LNAV,INAV,D1/D2, 1:CNAV/CNAV1/FNAV, 2: CNAV2, 3: CNAV3, 4:FDMA, 5:SBAS
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class Geph():
    """ class to define GLONASS ephemeris """
    sat = 0
    iode = 0  # IODE: 0-6bit of tb field
    frq = 0
    svh = 0
    sva = 0
    age = 0.0
    toe = gtime_t()
    tof = gtime_t()
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    taun = 0.0         # SV clock bias [s]
    gamn = 0.0         # relative frq bias
    dtaun = 0.0        # delta between L1 and L2 [s]
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class Seph():
    """ class to define SBAS ephemeris """
    sat = 0
    iodn = 0
    t0 = gtime_t()
    tof = gtime_t()
    svh = 0
    sva = 0
    pos = np.zeros(3)
    vel = np.zeros(3)
    acc = np.zeros(3)
    af0 = 0.0
    af1 = 0.0
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class Nav():
    """ class to define the navigation message """

    def __init__(self, nf=2):
        self.eph = []
        self.peph = []
        self.ion = np.array([
            [0.1118E-07, -0.7451E-08, -0.5961E-07, 0.1192E-06],
            [0.1167E+06, -0.2294E+06, -0.1311E+06, 0.1049E+07]])
        self.ion_gim = np.zeros(9)
        self.ion_region = 0  # 0: wide-area, 1: Japan-aera (QZSS only)
        self.sto = np.zeros(3)
        self.sto_prm = np.zeros(4, dtype=int)
        self.eop = np.zeros(9)
        self.elmin = np.deg2rad(15.0)
        self.tidecorr = False
        self.nf = nf
        self.ne = 0
        self.nc = 0
        self.excl_sat = []  # Excluded satellites
        self.rb = [0, 0, 0]  # base station position in ECEF [m]
        self.smode = 0  # position mode 0:NONE,1:std,2:DGPS,4:fix,5:float
        self.pmode = 1  # 0: static, 1: kinematic
        self.ephopt = 2  # ephemeris option 0: BRDC, 1: SBAS, 2: SSR-APC,
        #                  3: SSR-CG, 4: PREC

        self.monlevel = 1
        self.cnr_min = 25
        self.cnr_min_gpy = 15
        self.maxout = 5  # maximum outage [epochs]

        self.sat_ant = None
        self.rcv_ant = None
        self.rcv_ant_b = None

        # SSR correction placeholder
        self.dorb = np.zeros(uGNSS.MAXSAT)
        self.dclk = np.zeros(uGNSS.MAXSAT)
        self.dsis = np.zeros(uGNSS.MAXSAT)
        self.sis = np.zeros(uGNSS.MAXSAT)

        # Satellite observation status
        self.fix = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        self.edt = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        # Measurement outage indicator
        self.outc = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)
        # Carrier-phase processed indicator
        self.vsat = np.zeros((uGNSS.MAXSAT, self.nf), dtype=int)

        self.tt = 0
        self.t = gtime_t()


def epoch2time(ep):
    """ calculate time from epoch """
    doy = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]
    time = gtime_t()
    year = int(ep[0])
    mon = int(ep[1])
    day = int(ep[2])

    if year < 1970 or year > 2099 or mon < 1 or mon > 12:
        return time
    days = (year-1970)*365+(year-1969)//4+doy[mon-1]+day-2
    if year % 4 == 0 and mon >= 3:
        days += 1
    sec = int(ep[5])
    time.time = days*86400+int(ep[3])*3600+int(ep[4])*60+sec
    time.sec = ep[5]-sec
    return time


leaps_ = [[2017, 1, 1, 0, 0, 0, -18],
          [2015, 7, 1, 0, 0, 0, -17],
          [2012, 7, 1, 0, 0, 0, -16],
          [2009, 1, 1, 0, 0, 0, -15],
          [2006, 1, 1, 0, 0, 0, -14],
          [1999, 1, 1, 0, 0, 0, -13],
          [1997, 7, 1, 0, 0, 0, -12],
          [1996, 1, 1, 0, 0, 0, -11],
          [1994, 7, 1, 0, 0, 0, -10],
          [1993, 7, 1, 0, 0, 0, -9],
          [1992, 7, 1, 0, 0, 0, -8],
          [1991, 1, 1, 0, 0, 0, -7],
          [1990, 1, 1, 0, 0, 0, -6],
          [1988, 1, 1, 0, 0, 0, -5],
          [1985, 7, 1, 0, 0, 0, -4],
          [1983, 7, 1, 0, 0, 0, -3],
          [1982, 7, 1, 0, 0, 0, -2],
          [1981, 7, 1, 0, 0, 0, -1]]


def timeget():
    """ return current time in UTC """
    now = datetime.now(timezone.utc)
    ep = np.array([now.year, now.month, now.day, now.hour, now.minute,
                   now.second])
    return epoch2time(ep)


def glo2time(tref: gtime_t, tod):
    time = timeadd(gpst2utc(tref), 10800.0)
    week, tow = time2gpst(time)
    tod_p = tow % 86400.0
    tow -= tod_p
    if tod < tod_p-43200.0:
        tod += 86400.0
    elif tod > tod_p+43200.0:
        tod -= 86400.0
    time = gpst2time(week, tow+tod)
    return utc2gpst(timeadd(time, -10800.0))


def gpst2utc(t: gtime_t):
    for i in range(len(leaps_)):
        tu = timeadd(t, leaps_[i][6])
        if timediff(tu, epoch2time(leaps_[i])) >= 0.0:
            return tu
    return t


def utc2gpst(t: gtime_t):
    for i in range(len(leaps_)):
        if timediff(t, epoch2time(leaps_[i])) >= 0.0:
            return timeadd(t, -leaps_[i][6])
    return t


def timeadd(t: gtime_t, sec: float):
    """ return time added with sec """
    tr = deepcopy(t)
    tr.sec += sec
    tt = floor(tr.sec)
    tr.time += int(tt)
    tr.sec -= tt
    return tr


def timediff(t1: gtime_t, t2: gtime_t):
    """ return time difference """
    dt = t1.time-t2.time
    dt += t1.sec-t2.sec
    return dt


def gpst2time(week, tow):
    """ convert to time from gps-time """
    t = epoch2time(gpst0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def time2gpst(t: gtime_t):
    """ convert to gps-time from time """
    t0 = epoch2time(gpst0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def gst2time(week, tow):
    """ convert to time from galileo system time """
    t = epoch2time(gst0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def time2gst(t: gtime_t):
    """ convert to galileo system time from time """
    t0 = epoch2time(gst0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def bdt2time(week, tow):
    """ convert to time from BeiDou system time """
    t = epoch2time(bdt0)
    if tow < -1e9 or tow > 1e9:
        tow = 0.0
    t.time += 86400*7*week+int(tow)
    t.sec = tow-int(tow)
    return t


def bdt2gpst(t: gtime_t):
    """ convert from BeiDou system time to GPS time  """
    return timeadd(t, 14.0)


def gpst2bdt(t: gtime_t):
    """ convert to GPS time from BeiDou system time """
    return timeadd(t, -14.0)


def time2bdt(t: gtime_t):
    """ convert to BeiDou system time from time """
    t0 = epoch2time(bdt0)
    sec = t.time-t0.time
    week = int(sec/(86400*7))
    tow = sec-week*86400*7+t.sec
    return week, tow


def time2epoch(t):
    """ convert time to epoch """
    mday = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31, 31, 28, 31, 30, 31,
            30, 31, 31, 30, 31, 30, 31, 31, 29, 31, 30, 31, 30, 31, 31, 30, 31,
            30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    days = int(t.time/86400)
    sec = int(t.time-days*86400)
    day = days % 1461
    for mon in range(48):
        if day >= mday[mon]:
            day -= mday[mon]
        else:
            break
    ep = [0, 0, 0, 0, 0, 0]
    ep[0] = 1970+days//1461*4+mon//12
    ep[1] = mon % 12+1
    ep[2] = day+1
    ep[3] = sec//3600
    ep[4] = sec % 3600//60
    ep[5] = sec % 60+t.sec
    return ep


def time2doy(t):
    """ convert time to epoch """
    ep = time2epoch(t)
    ep[1] = ep[2] = 1.0
    ep[3] = ep[4] = ep[5] = 0.0
    return timediff(t, epoch2time(ep))/86400+1


def str2time(s, i, n):
    if i < 0 or len(s) < i:
        return -1
    ep = np.array([float(x) for x in s[i:i+n].split()])
    if len(ep) < 6:
        return -1
    if ep[0] < 100.0:
        ep[0] += 2000.0 if ep[0] < 80.0 else 1900.0
    return epoch2time(ep)


def time2str(t):
    e = time2epoch(t)
    return "{:04d}-{:02d}-{:02d} {:02d}:{:02d}:{:02d}"\
        .format(e[0], e[1], e[2], e[3], e[4], int(e[5]))


def prn2sat(sys, prn):
    """ convert sys+prn to sat """
    if sys == uGNSS.GPS:
        sat = prn
    elif sys == uGNSS.GAL:
        sat = prn+uGNSS.GALMIN
    elif sys == uGNSS.QZS:
        sat = prn-192+uGNSS.QZSMIN
    elif sys == uGNSS.GLO:
        sat = prn+uGNSS.GLOMIN
    elif sys == uGNSS.BDS:
        sat = prn+uGNSS.BDSMIN
    elif sys == uGNSS.SBS:
        sat = prn-119+uGNSS.SBSMIN
    elif sys == uGNSS.IRN:
        sat = prn+uGNSS.IRNMIN
    else:
        sat = 0
    return sat


def sat2prn(sat):
    """ convert sat to sys+prn """
    if sat > uGNSS.MAXSAT:
        prn = 0
        sys = uGNSS.NONE
    elif sat > uGNSS.IRNMIN:
        prn = sat-uGNSS.IRNMIN
        sys = uGNSS.IRN
    elif sat > uGNSS.SBSMIN:
        prn = sat+119-uGNSS.SBSMIN
        sys = uGNSS.SBS
    elif sat > uGNSS.GLOMIN:
        prn = sat-uGNSS.GLOMIN
        sys = uGNSS.GLO
    elif sat > uGNSS.BDSMIN:
        prn = sat-uGNSS.BDSMIN
        sys = uGNSS.BDS
    elif sat > uGNSS.QZSMIN:
        prn = sat+192-uGNSS.QZSMIN
        sys = uGNSS.QZS
    elif sat > uGNSS.GALMIN:
        prn = sat-uGNSS.GALMIN
        sys = uGNSS.GAL
    else:
        prn = sat
        sys = uGNSS.GPS
    return (sys, prn)


def sat2id(sat):
    """ convert satellite number to id """
    sys, prn = sat2prn(sat)
    gnss_tbl = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E', uGNSS.BDS: 'C',
                uGNSS.QZS: 'J', uGNSS.SBS: 'S', uGNSS.IRN: 'I'}
    if sys not in gnss_tbl:
        return -1
    if sys == uGNSS.QZS:
        prn -= 192
    elif sys == uGNSS.SBS:
        prn -= 100
    return '%s%02d' % (gnss_tbl[sys], prn)


def id2sat(id_):
    """ convert id to satellite number """
    sys = char2sys(id_[0])
    if sys == uGNSS.NONE:
        return -1

    prn = int(id_[1:3])
    if sys == uGNSS.QZS:
        prn += 192
    elif sys == uGNSS.SBS:
        prn += 100
    sat = prn2sat(sys, prn)
    return sat


def char2sys(c):
    """ convert character to GNSS """
    gnss_tbl = {'G': uGNSS.GPS, 'R': uGNSS.GLO, 'E': uGNSS.GAL, 'C': uGNSS.BDS,
                'J': uGNSS.QZS, 'S': uGNSS.SBS, 'I': uGNSS.IRN}

    if c not in gnss_tbl:
        return uGNSS.NONE
    else:
        return gnss_tbl[c]


def sys2char(sys):
    """ convert gnss to character """
    gnss_tbl = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E', uGNSS.BDS: 'C',
                uGNSS.QZS: 'J', uGNSS.SBS: 'S', uGNSS.IRN: 'I'}

    if sys not in gnss_tbl:
        return "?"
    else:
        return gnss_tbl[sys]


def sys2str(sys):
    """ convert gnss to string """
    gnss_tbl = {uGNSS.GPS: 'GPS', uGNSS.GLO: 'GLONASS',
                uGNSS.GAL: 'GALILEO', uGNSS.BDS: 'BEIDOU',
                uGNSS.QZS: 'QZSS', uGNSS.SBS: 'SBAS', uGNSS.IRN: 'IRNSS'}

    if sys not in gnss_tbl:
        return "???"
    else:
        return gnss_tbl[sys]


def vnorm(r):
    """ calculate norm of a vector """
    return r/np.linalg.norm(r)


def geodist(rs, rr):
    """ calculate geometric distance """
    e = rs-rr
    r = np.linalg.norm(e)
    e = e/r
    r += rCST.OMGE*(rs[0]*rr[1]-rs[1]*rr[0])/rCST.CLIGHT
    return r, e


def kfupdate(x, P, H, v, R):
    """ Kalman filter measurement update """
    # select subset of states and covariance
    ix = []
    for i in range(len(x)):
        if P[i, i] > 0.0:
            ix.append(i)
    x_ = x[ix]
    P_ = P[ix, :][:, ix]
    # measurement update
    H_ = H[:, ix]
    PHt = P_@H_.T
    S = H_@PHt+R
    K = PHt@np.linalg.inv(S)
    x_ += K@v
    P_ -= K@H_@P_
    # restore states and covariance
    x[ix] = x_
    sP = P[ix, :]
    sP[:, ix] = P_
    P[ix, :] = sP
    return x, P, S


def dops_h(H):
    """ calculate DOP from H """
    Qinv = np.linalg.inv(np.dot(H.T, H))
    dop = np.diag(Qinv)
    hdop = dop[0]+dop[1]  # TBD
    vdop = dop[2]  # TBD
    pdop = hdop+vdop
    gdop = pdop+dop[3]
    dop = np.array([gdop, pdop, hdop, vdop])
    return dop


def dops(az, el, elmin=0):
    """ calculate DOP from az/el """
    nm = az.shape[0]
    H = np.zeros((nm, 4))
    n = 0
    for i in range(nm):
        if el[i] < elmin:
            continue
        cel = cos(el[i])
        sel = sin(el[i])
        H[n, 0] = cel*sin(az[i])
        H[n, 1] = cel*cos(az[i])
        H[n, 2] = sel
        H[n, 3] = 1
        n += 1
    if n < 4:
        return None
    Qinv = np.linalg.inv(np.dot(H.T, H))
    dop = np.diag(Qinv)
    hdop = dop[0]+dop[1]  # TBD
    vdop = dop[2]  # TBD
    pdop = hdop+vdop
    gdop = pdop+dop[3]
    dop = np.array([gdop, pdop, hdop, vdop])
    return dop


def xyz2enu(pos):
    """ return ECEF to ENU conversion matrix from LLH """
    sp = sin(pos[0])
    cp = cos(pos[0])
    sl = sin(pos[1])
    cl = cos(pos[1])
    E = np.array([[-sl, cl, 0],
                  [-sp*cl, -sp*sl, cp],
                  [cp*cl, cp*sl, sp]])
    return E


def enu2xyz(pos):
    """ return ENU to ECEF conversion matrix from LLH """
    return np.array(np.matrix(xyz2enu(pos)).I)


def ecef2pos(r):
    """  ECEF to LLH position conversion """
    e2 = rCST.FE_WGS84*(2-rCST.FE_WGS84)
    r2 = r[0]**2+r[1]**2
    v = rCST.RE_WGS84
    z = r[2]
    cnt = 0
    while cnt < 1000:
        zk = z
        sp = z/np.sqrt(r2+z**2)
        v = rCST.RE_WGS84/np.sqrt(1-e2*sp**2)
        z = r[2]+v*e2*sp
        if np.fabs(z-zk) < 1e-4:
            break
        cnt += 1
    pos = np.array([np.arctan(z/np.sqrt(r2)),
                    np.arctan2(r[1], r[0]),
                    np.sqrt(r2+z**2)-v])
    return pos


def pos2ecef(pos, isdeg: bool = False):
    """ LLH (rad/deg) to ECEF position conversion  """
    if isdeg:
        pos[0] *= np.pi/180.0
        pos[1] *= np.pi/180.0
    s_p = sin(pos[0])
    c_p = cos(pos[0])
    s_l = sin(pos[1])
    c_l = cos(pos[1])
    e2 = rCST.FE_WGS84*(2.0-rCST.FE_WGS84)
    v = rCST.RE_WGS84/sqrt(1.0-e2*s_p**2)
    r = np.array([(v+pos[2])*c_p*c_l,
                  (v+pos[2])*c_p*s_l,
                  (v*(1.0-e2)+pos[2])*s_p])
    return r


def ecef2enu(pos, r):
    """ relative ECEF to ENU conversion """
    E = xyz2enu(pos)
    e = E@r
    return e


def deg2dms(deg):
    """ convert from deg to dms """
    if deg < 0.0:
        sign = -1
    else:
        sign = 1
    a = fabs(deg)
    dms = np.zeros(3)
    dms[0] = floor(a)
    a = (a-dms[0])*60.0
    dms[1] = floor(a)
    a = (a-dms[1])*60.0
    dms[2] = a
    dms[0] *= sign
    return dms


def satazel(pos, e):
    """ calculate az/el from LOS vector in ECEF (e) """
    enu = ecef2enu(pos, e)
    az = atan2(enu[0], enu[1])
    el = asin(enu[2])
    return az, el


def ionmodel(t, pos, az, el, ion=None):
    """ klobuchar model of ionosphere delay estimation """
    psi = 0.0137/(el/np.pi+0.11)-0.022
    phi = pos[0]/np.pi+psi*cos(az)
    phi = np.max((-0.416, np.min((0.416, phi))))
    lam = pos[1]/np.pi+psi*sin(az)/cos(phi*np.pi)
    phi += 0.064*cos((lam-1.617)*np.pi)
    _, tow = time2gpst(t)
    tt = 43200.0*lam+tow  # local time
    tt -= np.floor(tt/86400)*86400
    f = 1.0+16.0*np.power(0.53-el/np.pi, 3.0)  # slant factor

    h = [1, phi, phi**2, phi**3]
    amp = np.dot(h, ion[0, :])
    per = np.dot(h, ion[1, :])
    amp = max(amp, 0)
    per = max(per, 72000.0)
    x = 2.0*np.pi*(tt-50400.0)/per
    if np.abs(x) < 1.57:
        v = 5e-9+amp*(1.0+x*x*(-0.5+x*x/24.0))
    else:
        v = 5e-9
    diono = rCST.CLIGHT*f*v
    return diono


def interpc(coef, lat):
    """ linear interpolation (lat step=15) """
    i = int(lat/15.0)
    if i < 1:
        return coef[:, 0]
    if i > 4:
        return coef[:, 4]
    d = lat/15.0-i
    return coef[:, i-1]*(1.0-d)+coef[:, i]*d


class uTropoModel(IntEnum):
    """
    Enumeration for tropo model selection
    """

    NONE = -1
    SAAST = 0
    HOPF = 1


def tropmapf(t, pos, el, model=uTropoModel.SAAST):
    """
    Tropo mapping function
    """

    if model == uTropoModel.SAAST:
        return tropmapfNiell(t, pos, el)
    elif model == uTropoModel.HOPF:
        return tropmapfHpf(el)
    else:
        return 0


def tropmodel(t, pos, el=np.pi/2, humi=0.7, model=uTropoModel.SAAST):
    """
    Tropospheric delay model
    """

    if model == uTropoModel.SAAST:
        return tropmodelSaast(t, pos, el, humi)
    elif model == uTropoModel.HOPF:
        return tropmodelHpf()
    else:
        return 0


def meteo(hgt, humi):
    """ standard atmosphere model """
    pres = 1013.25*np.power(1-2.2557e-5*hgt, 5.2568)
    temp = 15.0-6.5e-3*hgt+273.16
    e = 6.108*humi*np.exp((17.15*temp-4684.0)/(temp-38.45))
    return pres, temp, e


def mapf(el, a, b, c):
    """ simple tropospheric mapping function """
    sinel = np.sin(el)
    return (1.0+a/(1.0+b/(1.0+c)))/(sinel+(a/(sinel+b/(sinel+c))))


def tropmapfNiell(t, pos, el):
    """ tropospheric mapping function by Niell (NMF) """
    if pos[2] < -1e3 or pos[2] > 20e3 or el <= 0.0:
        return 0.0, 0.0
    coef = np.array([
        [1.2769934E-3, 1.2683230E-3, 1.2465397E-3, 1.2196049E-3, 1.2045996E-3],
        [2.9153695E-3, 2.9152299E-3, 2.9288445E-3, 2.9022565E-3, 2.9024912E-3],
        [62.610505E-3, 62.837393E-3, 63.721774E-3, 63.824265E-3, 64.258455E-3],
        [0.0000000E-0, 1.2709626E-5, 2.6523662E-5, 3.4000452E-5, 4.1202191E-5],
        [0.0000000E-0, 2.1414979E-5, 3.0160779E-5, 7.2562722E-5, 11.723375E-5],
        [0.0000000E-0, 9.0128400E-5, 4.3497037E-5, 84.795348E-5, 170.37206E-5],
        [5.8021897E-4, 5.6794847E-4, 5.8118019E-4, 5.9727542E-4, 6.1641693E-4],
        [1.4275268E-3, 1.5138625E-3, 1.4572752E-3, 1.5007428E-3, 1.7599082E-3],
        [4.3472961E-2, 4.6729510E-2, 4.3908931E-2, 4.4626982E-2, 5.4736038E-2],
    ])
    aht = [2.53E-5, 5.49E-3, 1.14E-3]
    lat = np.rad2deg(pos[0])
    hgt = pos[2]
    y = (time2doy(t)-28.0)/365.25
    if lat < 0.0:
        y += 0.5
    cosy = np.cos(2.0*np.pi*y)
    lat = np.abs(lat)
    c = interpc(coef, lat)
    ah = c[0:3]-c[3:6]*cosy
    aw = c[6:9]
    dm = (1.0/np.sin(el)-mapf(el, aht[0], aht[1], aht[2]))*hgt*1e-3
    mapfh = mapf(el, ah[0], ah[1], ah[2])+dm
    mapfw = mapf(el, aw[0], aw[1], aw[2])
    return mapfh, mapfw


def tropmodelSaast(t, pos, el=np.pi/2, humi=0.7):
    """ saastamoinen tropospheric delay model """
    hgt = pos[2]
    # standard atmosphere
    pres, temp, e = meteo(hgt, humi)
    # saastamoinen
    z = np.pi/2.0-el
    trop_hs = 0.0022768*pres/(1.0-0.00266*np.cos(2*pos[0])-0.00028e-3*hgt)
    trop_wet = 0.002277*(1255.0/temp+0.05)*e
    return trop_hs, trop_wet, z


def meteoHpf():
    """
    Hopfield atmosphere model
    """
    pres = 1010.0
    temp = 291.1
    e = 10.4
    return pres, temp, e


def tropmapfHpf(el):
    """
    Tropospheric mapping functions for Hopfield model
    """

    mapfh = 1.0/(np.sin(np.sqrt(el**2+(np.pi/72.0)**2)))
    mapfw = 1.0/(np.sin(np.sqrt(el**2+(np.pi/120.0)**2)))

    return mapfh, mapfw,


def tropmodelHpf():
    """
    Hopfield tropospheric delay model
    """
    # standard atmosphere for Hopfield model
    pres, temp, e = meteoHpf()
    # Hopfield
    trop_dry = (77.6e-6 * (-613.3768/temp+148.98)*pres)/5.0
    trop_wet = (77.6e-6 * 11000.0 * 4810.0 * e/temp**2)/5.0

    return trop_dry, trop_wet, None
