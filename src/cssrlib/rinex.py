"""
module for RINEX 3.0x processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.gnss import bdt2gpst, time2bdt
from cssrlib.gnss import gpst2time, bdt2time, epoch2time, timediff, gtime_t
from cssrlib.gnss import prn2sat, char2sys, timeget, utc2gpst, time2epoch
from cssrlib.gnss import Eph, Obs, sat2id, sat2prn, gpst2bdt, time2gpst


class pclk_t:
    def __init__(self, time=None):
        if time is not None:
            self.time = time
        else:
            self.time = gtime_t()
        self.clk = np.zeros(uGNSS.MAXSAT)
        self.std = np.zeros(uGNSS.MAXSAT)


class rnxdec:
    """ class for RINEX decoder """

    def __init__(self):

        self.ver = -1.0
        self.fobs = None

        # signal code mapping from RINEX header to columns in data section
        self.sig_map = {}
        # signal selection for internal data structure
        self.sig_tab = {}
        self.nsig = {uTYP.C: 0, uTYP.L: 0, uTYP.D: 0, uTYP.S: 0}

        self.pos = np.array([0, 0, 0])
        self.ecc = np.array([0, 0, 0])
        self.rcv = None
        self.ant = None
        self.ts = None
        self.te = None
        # 0:LNAV,INAV,D1/D2, 1:CNAV/CNAV1/FNAV, 2: CNAV2, 3: CNAV3,
        # 4:FDMA, 5:SBAS
        self.mode_nav = 0

    def setSignals(self, sigList):
        """ define the signal list for each constellation """

        for sig in sigList:
            if sig.sys not in self.sig_tab:
                self.sig_tab.update({sig.sys: {}})
            if sig.typ not in self.sig_tab[sig.sys]:
                self.sig_tab[sig.sys].update({sig.typ: []})
            if sig not in self.sig_tab[sig.sys][sig.typ]:
                self.sig_tab[sig.sys][sig.typ].append(sig)
            else:
                raise ValueError("duplicate signal {} specified!".format(sig))

        for _, sigs in self.sig_tab.items():
            for typ, sig in sigs.items():
                self.nsig[typ] = max((self.nsig[typ], len(sig)))

    def getSignals(self, sys, typ):
        """ retrieve signal list for constellation and obs type """
        if sys in self.sig_tab.keys() and typ in self.sig_tab[sys].keys():
            return self.sig_tab[sys][typ]
        else:
            return []

    def autoSubstituteSignals(self):
        """
        Automatically substitute signal tracking attribute based on
        available signals
        """
        for sys, tmp in self.sig_tab.items():
            for typ, sigs in tmp.items():
                for i, sig in enumerate(sigs):

                    # Skip unavailable systems or available signals
                    #
                    if sys not in self.sig_map.keys():
                        continue
                    if sig in self.sig_map[sys].values():
                        continue

                    # Not found try to replace
                    #
                    if sys == uGNSS.GPS and sig.str()[1] in '12':
                        atts = 'CW' if sig.str()[2] in 'CW' else 'SLX'
                    elif sys == uGNSS.GPS and sig.str()[1] in '5':
                        atts = 'IQX'
                    elif sys == uGNSS.GAL and sig.str()[1] in '578':
                        atts = 'IQX'
                    elif sys == uGNSS.GAL and sig.str()[1] in '16':
                        atts = 'BCX'
                    elif sys == uGNSS.QZS and sig.str()[1] in '126':
                        atts = 'SLX'
                    elif sys == uGNSS.QZS and sig.str()[1] in '5':
                        atts = 'IQX'
                    elif sys == uGNSS.BDS and sig.str()[1] in '157':
                        atts = 'PX'
                    else:
                        atts = []

                    for a in atts:
                        if sig.toAtt(a) in self.sig_map[sys].values():
                            self.sig_tab[sys][typ][i] = sig.toAtt(a)

    def flt(self, u, c=-1):
        if c >= 0:
            u = u[19*c+4:19*(c+1)+4]
        if u.isspace():
            return 0.0
        return float(u.replace("D", "E"))

    def decode_time(self, s, ofst=0, slen=2):
        year = int(s[ofst+0:ofst+4])
        month = int(s[ofst+5:ofst+7])
        day = int(s[ofst+8:ofst+10])
        hour = int(s[ofst+11:ofst+13])
        minute = int(s[ofst+14:ofst+16])
        sec = float(s[ofst+17:ofst+slen+17])
        t = epoch2time([year, month, day, hour, minute, sec])
        return t

    def decode_nav(self, navfile, nav):
        """
        Decode RINEX Navigation message from file

        NOTE: system time epochs are converted into GPST on reading!

        """

        nav.eph = []
        with open(navfile, 'rt') as fnav:
            for line in fnav:
                if line[60:73] == 'END OF HEADER':
                    break
                elif line[60:80] == 'RINEX VERSION / TYPE':
                    self.ver = float(line[4:10])
                    if self.ver < 3.02:
                        return -1
                elif line[60:76] == 'IONOSPHERIC CORR':
                    if line[0:4] == 'GPSA' or line[0:4] == 'QZSA':
                        for k in range(4):
                            nav.ion[0, k] = self.flt(line[5+k*12:5+(k+1)*12])
                    if line[0:4] == 'GPSB' or line[0:4] == 'QZSB':
                        for k in range(4):
                            nav.ion[1, k] = self.flt(line[5+k*12:5+(k+1)*12])

            for line in fnav:

                if self.ver >= 4.0:

                    if line[0:5] == '> STO':  # system time offset (TBD)
                        ofst_src = {'GP': uGNSS.GPS, 'GL': uGNSS.GLO,
                                    'GA': uGNSS.GAL, 'BD': uGNSS.BDS,
                                    'QZ': uGNSS.QZS, 'IR': uGNSS.IRN,
                                    'SB': uGNSS.SBS, 'UT': uGNSS.NONE}
                        sys = char2sys(line[6])
                        itype = line[10:14]
                        line = fnav.readline()
                        ttm = self.decode_time(line, 4)
                        mode = line[24:28]
                        if mode[0:2] in ofst_src and mode[2:4] in ofst_src:
                            nav.sto_prm[0] = ofst_src[mode[0:2]]
                            nav.sto_prm[1] = ofst_src[mode[2:4]]

                        line = fnav.readline()
                        ttm = self.flt(line, 0)
                        for k in range(3):
                            nav.sto[k] = self.flt(line, k+1)
                        continue

                    elif line[0:5] == '> EOP':  # earth orientation param
                        sys = char2sys(line[6])
                        itype = line[10:14]
                        line = fnav.readline()
                        ttm = self.decode_time(line, 4)
                        for k in range(3):
                            nav.eop[k] = self.flt(line, k+1)
                        line = fnav.readline()
                        for k in range(3):
                            nav.eop[k+3] = self.flt(line, k+1)
                        line = fnav.readline()
                        ttm = self.flt(line, 0)
                        for k in range(3):
                            nav.eop[k+6] = self.flt(line, k+1)
                        continue

                    elif line[0:5] == '> ION':  # iono (TBD)
                        sys = char2sys(line[6])
                        itype = line[10:14]
                        line = fnav.readline()
                        ttm = self.decode_time(line, 4)
                        if sys == uGNSS.GAL and itype == 'IFNV':  # Nequick-G
                            for k in range(3):
                                nav.ion[0, k] = self.flt(line, k+1)
                            line = fnav.readline()
                            nav.ion[0, 3] = int(self.flt(line, 0))
                        elif sys == uGNSS.BDS and itype == 'CNVX':  # BDGIM
                            ttm = self.decode_time(line, 4)
                            self.ion_gim = np.zeros(9)
                            for k in range(3):
                                nav.ion_gim[k] = self.flt(line, k+1)
                            line = fnav.readline()
                            for k in range(4):
                                nav.ion_gim[k+3] = self.flt(line, k)
                            line = fnav.readline()
                            for k in range(2):
                                nav.ion_gim[k+7] = self.flt(line, k)
                        else:  # Klobucher (LNAV, D1D2, CNVX)
                            self.ion_gim = np.zeros(9)
                            for k in range(3):
                                nav.ion[0, k] = self.flt(line, k+1)
                            line = fnav.readline()
                            nav.ion[0, 3] = self.flt(line, 0)
                            for k in range(3):
                                nav.ion[1, k] = self.flt(line, k+1)
                            line = fnav.readline()
                            nav.ion[1, 3] = self.flt(line, 0)
                            if len(line) >= 42:
                                nav.ion_region = int(self.flt(line, 1))
                        continue

                    elif line[0:5] == '> EPH':
                        sys = char2sys(line[6])
                        self.mode_nav = 0  # LNAV, D1/D2, INAV
                        m = line[10:14]
                        if m == 'CNAV' or m == 'CNV1' or m == 'FNAV':
                            self.mode_nav = 1
                        elif m == 'CNV2':
                            self.mode_nav = 2
                        elif m == 'CNV3':
                            self.mode_nav = 3
                        elif m == 'FDMA':
                            self.mode_nav = 4
                        elif m == 'SBAS':
                            self.mode_nav = 5
                        line = fnav.readline()

                # Process ephemeris information
                #
                sys = char2sys(line[0])

                # Skip undesired constellations
                #
                if sys not in (uGNSS.GPS, uGNSS.GAL, uGNSS.QZS, uGNSS.BDS):
                    continue

                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)
                eph = Eph(sat)

                eph.mode = self.mode_nav

                eph.toc = self.decode_time(line, 4)
                eph.af0 = self.flt(line, 1)
                eph.af1 = self.flt(line, 2)
                eph.af2 = self.flt(line, 3)

                line = fnav.readline()  # line #1

                if self.mode_nav > 0:
                    eph.Adot = self.flt(line, 0)
                else:
                    eph.iode = int(self.flt(line, 0))
                eph.crs = self.flt(line, 1)
                eph.deln = self.flt(line, 2)
                eph.M0 = self.flt(line, 3)

                line = fnav.readline()  # line #2
                eph.cuc = self.flt(line, 0)
                eph.e = self.flt(line, 1)
                eph.cus = self.flt(line, 2)
                sqrtA = self.flt(line, 3)
                eph.A = sqrtA**2

                line = fnav.readline()  # line #3
                eph.toes = int(self.flt(line, 0))
                eph.cic = self.flt(line, 1)
                eph.OMG0 = self.flt(line, 2)
                eph.cis = self.flt(line, 3)

                line = fnav.readline()  # line #4
                eph.i0 = self.flt(line, 0)
                eph.crc = self.flt(line, 1)
                eph.omg = self.flt(line, 2)
                eph.OMGd = self.flt(line, 3)

                line = fnav.readline()  # line #5
                eph.idot = self.flt(line, 0)
                if self.mode_nav > 0:
                    eph.delnd = self.flt(line, 1)
                    if sys == uGNSS.BDS:
                        eph.sattype = int(self.flt(line, 2))
                        eph.tops = int(self.flt(line, 3))
                    else:
                        eph.urai[0] = int(self.flt(line, 2))
                        eph.urai[1] = int(self.flt(line, 3))
                else:
                    eph.code = int(self.flt(line, 1))  # source for GAL
                    eph.week = int(self.flt(line, 2))

                line = fnav.readline()  # line #6
                if sys == uGNSS.BDS and self.mode_nav > 0:
                    eph.sisai[0] = int(self.flt(line, 0))  # oe
                    eph.sisai[1] = int(self.flt(line, 1))  # ocb
                    eph.sisai[2] = int(self.flt(line, 2))  # oc1
                    eph.sisai[3] = int(self.flt(line, 3))  # oc2
                else:
                    eph.sva = int(self.flt(line, 0))
                    eph.svh = int(self.flt(line, 1))
                    eph.tgd = float(self.flt(line, 2))
                    if sys == uGNSS.GPS or sys == uGNSS.QZS:
                        if self.mode_nav == 0:
                            eph.iodc = int(self.flt(line, 3))
                        else:
                            eph.urai[2] = int(self.flt(line, 3))
                    elif sys == uGNSS.GAL:
                        tgd_b = float(self.flt(line, 3))
                        if (eph.code >> 9) & 1:
                            eph.tgd = tgd_b
                        else:
                            eph.iodc = int(self.flt(line, 3))
                    elif sys == uGNSS.BDS:
                        eph.tgd_b = float(self.flt(line, 3))  # tgd2 B2/B3

                if self.mode_nav < 3:
                    line = fnav.readline()  # line #7
                    if sys == uGNSS.BDS:
                        if self.mode_nav == 0:
                            tot = self.flt(line, 0)
                            eph.iodc = int(self.flt(line, 1))
                        else:
                            if self.mode_nav == 1:
                                eph.isc[0] = float(self.flt(line, 0))  # B1Cd
                            elif self.mode_nav == 2:
                                eph.isc[1] = float(self.flt(line, 1))  # B2ad

                            eph.tgd = float(self.flt(line, 2))    # tgd_B1Cp
                            eph.tgd_b = float(self.flt(line, 3))  # tgd_B2ap

                    else:
                        if self.mode_nav > 0 and sys != uGNSS.GAL:
                            eph.isc[0] = self.flt(line, 0)
                            eph.isc[1] = self.flt(line, 1)
                            eph.isc[2] = self.flt(line, 2)
                            eph.isc[3] = self.flt(line, 3)
                            line = fnav.readline()

                        if self.mode_nav == 2:
                            eph.isc[4] = self.flt(line, 0)
                            eph.isc[5] = self.flt(line, 1)
                            line = fnav.readline()

                        tot = int(self.flt(line, 0))
                        if self.mode_nav > 0:
                            eph.week = int(self.flt(line, 1))
                        elif len(line) >= 42:
                            eph.fit = int(self.flt(line, 1))

                if sys == uGNSS.BDS and self.mode_nav > 0:
                    line = fnav.readline()  # line #8
                    eph.sismai = int(self.flt(line, 0))
                    eph.svh = int(self.flt(line, 1))
                    eph.integ = int(self.flt(line, 2))
                    if self.mode_nav < 3:
                        eph.iodc = int(self.flt(line, 3))
                    else:
                        eph.tgd_b = float(self.flt(line, 3))

                    line = fnav.readline()  # line #9
                    tot = int(self.flt(line, 0))
                    if self.mode_nav < 3:
                        eph.iode = int(self.flt(line, 3))

                if sys == uGNSS.BDS:
                    if self.mode_nav > 0:
                        eph.week, _ = time2bdt(eph.toc)
                    eph.toc = bdt2gpst(eph.toc)
                    eph.toe = bdt2gpst(bdt2time(eph.week, eph.toes))
                    eph.tot = bdt2gpst(bdt2time(eph.week, tot))
                else:
                    eph.toe = gpst2time(eph.week, eph.toes)
                    eph.tot = gpst2time(eph.week, tot)

                nav.eph.append(eph)

        return nav

    def decode_clk(self, clkfile, nav):
        """decode Clock-RINEX data from file """

        # Offset for Clock-RINEX v3.x data section
        #
        offs = None

        nav.pclk = []
        fnav = open(clkfile, 'rt')

        # Read header section
        #
        for line in fnav:

            if 'RINEX VERSION / TYPE' in line:
                ver = float(line[0:20])
                offs = 0 if ver < 3.04 else 5

            if 'END OF HEADER' in line:
                break

        # Read data section
        #
        for line in fnav:

            if line[0:2] != 'AS':
                continue

            sys = char2sys(line[3])
            prn = int(line[4:7])
            if sys == uGNSS.QZS:
                prn += 192
            sat = prn2sat(sys, prn)

            t = self.decode_time(line, offs+8, 9)

            if nav.nc <= 0 or abs(timediff(nav.pclk[-1].time, t)) > 1e-9:
                nav.nc += 1
                pclk = pclk_t()
                pclk.time = t
                nav.pclk.append(pclk)

            nrec = int(line[offs+35:offs+37])
            clk = float(line[offs+40:offs+59])
            std = float(line[offs+61:offs+80]) if nrec >= 2 else 0.0

            nav.pclk[nav.nc-1].clk[sat-1] = clk
            nav.pclk[nav.nc-1].std[sat-1] = std

        return nav

    # TODO: decode GLONASS FCN lines
    def decode_obsh(self, obsfile):
        self.fobs = open(obsfile, 'rt')
        for line in self.fobs:
            if line[60:73] == 'END OF HEADER':
                break
            if line[60:80] == 'RINEX VERSION / TYPE':
                self.ver = float(line[4:10])
                if self.ver < 3.02:
                    return -1
            elif 'REC # / TYPE / VERS' in line:
                self.rcv = line[20:40].upper()
            elif 'ANT # / TYPE' in line:
                self.ant = line[20:40].upper()
            elif line[60:79] == 'APPROX POSITION XYZ':
                self.pos = np.array([float(line[0:14]),
                                     float(line[14:28]),
                                     float(line[28:42])])
            elif 'ANTENNA: DELTA H/E/N' in line[60:]:
                self.ecc = np.array([float(line[14:28]),  # East
                                     float(line[28:42]),  # North
                                     float(line[0:14])])  # Up
            elif line[60:79] == 'SYS / # / OBS TYPES':

                gns = char2sys(line[0])
                nsig = int(line[3:6])

                # Extract string list of signal codes
                #
                sigs = line[7:60].split()
                for _ in range(int(nsig/14)):
                    line2 = self.fobs.readline()
                    sigs += line2[7:60].split()

                # Convert to RINEX signal code and store in map
                #
                for i, sig in enumerate(sigs):
                    rnxSig = rSigRnx(gns, sig)
                    if gns not in self.sig_map:
                        self.sig_map.update({gns: {}})
                    self.sig_map[gns].update({i: rnxSig})
            elif 'TIME OF FIRST OBS' in line[60:]:
                self.ts = epoch2time([float(v) for v in line[0:44].split()])
            elif 'TIME OF LAST OBS' in line[60:]:
                self.te = epoch2time([float(v) for v in line[0:44].split()])

        return 0

    def decode_obs(self):
        """ decode RINEX Observation message from file """

        obs = Obs()

        for line in self.fobs:

            if line[0] != '>':
                continue

            nsat = int(line[32:35])

            year = int(line[2:6])
            month = int(line[7:9])
            day = int(line[10:12])
            hour = int(line[13:15])
            minute = int(line[16:18])
            sec = float(line[19:29])
            obs.t = epoch2time([year, month, day, hour, minute, sec])

            # Initialize data structures
            #
            obs.P = np.empty((0, self.nsig[uTYP.C]), dtype=np.float64)
            obs.L = np.empty((0, self.nsig[uTYP.L]), dtype=np.float64)
            obs.S = np.empty((0, self.nsig[uTYP.S]), dtype=np.float64)
            obs.lli = np.empty((0, self.nsig[uTYP.L]), dtype=np.int32)
            obs.sat = np.empty(0, dtype=np.int32)
            obs.sig = self.sig_tab

            for _ in range(nsat):

                line = self.fobs.readline()
                sys = char2sys(line[0])

                # Skip constellation not contained in RINEX header
                #
                if sys not in self.sig_map.keys():
                    continue

                # Skip undesired constellations
                #
                if sys not in self.sig_tab:
                    continue

                # Convert to satellite ID
                #
                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                elif sys == uGNSS.SBS:
                    prn += 100
                sat = prn2sat(sys, prn)

                pr = np.zeros(len(self.getSignals(sys, uTYP.C)),
                              dtype=np.float64)
                cp = np.zeros(len(self.getSignals(sys, uTYP.L)),
                              dtype=np.float64)
                ll = np.zeros(len(self.getSignals(sys, uTYP.L)),
                              dtype=np.int32)
                cn = np.zeros(len(self.getSignals(sys, uTYP.S)),
                              dtype=np.float64)

                for i, sig in self.sig_map[sys].items():

                    # Skip undesired signals
                    #
                    if sig.typ not in self.sig_tab[sys] or \
                            sig not in self.sig_tab[sys][sig.typ]:
                        continue

                    # Get string representation of measurement value
                    #
                    sval = line[16*i+3:16*i+17].strip()
                    slli = line[16*i+17] if len(line) > 16*i+17 else ''

                    # Convert from string to numerical value
                    #
                    val = 0.0 if not sval else float(sval)
                    lli = 1 if slli == '1' else 0

                    # Signal index in data structure
                    #
                    j = self.sig_tab[sys][sig.typ].index(sig)

                    if sig.typ == uTYP.C:
                        pr[j] = val
                    elif sig.typ == uTYP.L:
                        cp[j] = val
                        ll[j] = lli
                    elif sig.typ == uTYP.S:
                        cn[j] = val
                    else:
                        continue

                # Store prn and data
                #
                obs.P = np.append(obs.P, pr)
                obs.L = np.append(obs.L, cp)
                obs.S = np.append(obs.S, cn)
                obs.lli = np.append(obs.lli, ll)
                obs.sat = np.append(obs.sat, sat)

            obs.P = obs.P.reshape(len(obs.sat), self.nsig[uTYP.C])
            obs.L = obs.L.reshape(len(obs.sat), self.nsig[uTYP.L])
            obs.S = obs.S.reshape(len(obs.sat), self.nsig[uTYP.S])
            obs.lli = obs.lli.reshape(len(obs.sat), self.nsig[uTYP.L])

            break

        return obs


def sync_obs(dec, decb, dt_th=0.1):
    """ sync observation between rover and base """
    obs = dec.decode_obs()
    obsb = decb.decode_obs()
    while True:
        dt = timediff(obs.t, obsb.t)
        if np.abs(dt) <= dt_th:
            break
        if dt > dt_th:
            obsb = decb.decode_obs()
        elif dt < dt_th:
            obs = dec.decode_obs()
    return obs, obsb


class rnxenc:
    """ class for RINEX encoder """

    def __init__(self, sig_tab=None):
        self.ver = -1.0
        self.fobs = None
        self.fnav = None
        self.rnx_obs_header_sent = False
        self.sig_tab = sig_tab

        self.prog = "cssrlib"
        self.runby = "Unknown"
        self.agency = "Unknown"
        self.observer = "Unknown"
        self.rec = "Unknown"
        self.rectype = "Unkown"
        self.recver = ""
        self.ant = "Unknown"
        self.anttype = "Unknown"
        self.pos = np.zeros(3)
        self.dant = np.zeros(3)

        self.rec_eph = {}

    def rnx_nav_header(self, fh=None, ver=4.00):
        tutc = timeget()
        tgps = utc2gpst(tutc)
        leaps = timediff(tgps, tutc)

        ep = time2epoch(tutc)
        s = "{:4d}{:02d}{:02d} {:02d}{:02d}{:02d} {:3s}". \
            format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], "UTC")

        fh.write("{:9.2f}           {:19s} {:19s} {:20s}\n".
                 format(ver, "NAVIGATION DATA", "M", "RINEX VERSION / TYPE"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".
                 format(self.prog, self.runby, s, "PGM / RUN BY / DATE"))
        fh.write("{:6d}{:6s}{:6s}{:6s}{:3s}{:33s}{:20s}\n".
                 format(leaps, "", "", "", "", "", "LEAP SECONDS"))
        fh.write("{:60s}{:20s}\n".
                 format("", "END OF HEADER"))

    def rnx_obs_header(self, ts: gtime_t, fh=None, ver=4.00):

        if self.rnx_obs_header_sent:
            return
        self.rnx_obs_header_sent = True

        sys_t = {uGNSS.GPS: 'G', uGNSS.GLO: 'R', uGNSS.GAL: 'E',
                 uGNSS.QZS: 'J', uGNSS.BDS: 'C', uGNSS.IRN: 'I',
                 uGNSS.SBS: 'S'}

        tutc = timeget()
        tgps = utc2gpst(tutc)
        leaps = timediff(tgps, tutc)

        ep = time2epoch(tutc)
        s = "{:4d}{:02d}{:02d} {:02d}{:02d}{:02d} {:3s}". \
            format(ep[0], ep[1], ep[2], ep[3], ep[4], ep[5], "UTC")

        fh.write("{:9.2f}           {:19s} {:19s} {:20s}\n".
                 format(ver, "OBSERVATION DATA", "M", "RINEX VERSION / TYPE"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".
                 format(self.prog, self.runby, s, "PGM / RUN BY / DATE"))

        fh.write("{:60s}{:20s}\n".format("Unknown", "MARKER NAME"))
        fh.write("{:20s}{:40s}{:20s}\n".format(
            self.observer, self.agency, "OBSERVER / AGENCY"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".format(
            self.rec, self.rectype, self.recver, "REC # / TYPE / VERS"))
        fh.write("{:20s}{:20s}{:20s}{:20s}\n".format(
            self.ant, self.anttype, "", "ANT # / TYPE"))
        fh.write("{:14.4f}{:14.4f}{:14.4f}{:18s}{:20s}\n".format(
            self.pos[0], self.pos[1], self.pos[2], "", "APPROX POSITION XYZ"))
        fh.write("{:14.4f}{:14.4f}{:14.4f}{:18s}{:20s}\n".format(
            self.dant[0], self.dant[1], self.dant[2], "",
            "ANTENNA: DELTA H/E/N"))

        for sys in self.sig_tab:
            pr = self.sig_tab[sys][uTYP.C]
            cp = self.sig_tab[sys][uTYP.L]
            dp = self.sig_tab[sys][uTYP.D]
            cn = self.sig_tab[sys][uTYP.S]

            nsig = len(pr)+len(cp)+len(dp)+len(cn)

            fh.write("{:1s}  {:3d}".format(sys_t[sys], nsig))

            n = 0
            for k, _ in enumerate(pr):
                fh.write(" {:3s}".format(pr[k].str()))
                n += 1
                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))

                fh.write(" {:3s}".format(cp[k].str()))
                n += 1
                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))

                fh.write(" {:3s}".format(dp[k].str()))
                n += 1
                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))

                fh.write(" {:3s}".format(cn[k].str()))
                n += 1

                if n == 13:
                    fh.write("  {:20s}\n{:6s}".format(
                        "SYS / # / OBS TYPES", ""))
                elif n >= nsig-1:
                    fh.write("  {:s}".format("    "*(13-(nsig % 13))))
                    fh.write("{:20s}".format("SYS / # / OBS TYPES \n"))

        # TBD
        ep = time2epoch(ts)
        fh.write(" {:5d} {:5d} {:5d} {:5d} {:5d}{:13.7f}".
                 format(int(ep[0]), int(ep[1]), int(ep[2]), int(ep[3]),
                        int(ep[4]), ep[5]))
        fh.write("{:5s}{:3s}{:9s}{:20s}\n".
                 format("", "GPS", "", "TIME OF FIRST OBS"))

        fh.write("{:6d}{:6s}{:6s}{:6s}{:3s}{:33s}{:20s}\n".
                 format(leaps, "", "", "", "", "", "LEAP SECONDS"))
        fh.write("{:60s}{:20s}\n".
                 format("", "END OF HEADER"))
        fh.flush()

    def sval(self, v: float):
        if v == 0.0:
            s = "{:14s}".format("")
        else:
            s = "{:14.3f}".format(v)
        return s

    def rnx_obs_body(self, obs=None, fh=None):

        ep = time2epoch(obs.time)
        nsat = len(obs.sat)
        nsig = obs.P.shape[1]
        fh.write("> {:4d} {:02d} {:02d} {:02d} {:02d} {:010.7f}".
                 format(int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), ep[5]))
        fh.write("  {:1d}{:3d}\n".format(0, nsat))

        for k in range(nsat):
            fh.write("{:3s}".format(sat2id(obs.sat[k])))
            sys, _ = sat2prn(obs.sat[k])
            for i in range(nsig):
                ssi = min(max(int(obs.S[k][i]/6), 1), 9)
                lli = obs.lli[k][i]
                fh.write("{:14s}{:2s}".format(
                    self.sval(obs.P[k][i]), ""))
                fh.write("{:14s}".format(self.sval(obs.L[k][i])))
                if obs.L[k][i] == 0.0:
                    fh.write("{:2s}".format(""))
                else:
                    fh.write("{:1d}{:1d}".format(lli, ssi))

                fh.write("{:14s}{:2s}".format(
                    self.sval(obs.D[k][i]), ""))
                fh.write("{:14s}{:2s}".format(self.sval(obs.S[k][i]), ""))
            fh.write("\n")

    def rnx_nav_body(self, eph=None, fh=None):
        if eph.sat in self.rec_eph.keys():
            if eph.iode in self.rec_eph[eph.sat].keys():
                return
        else:
            self.rec_eph[eph.sat] = {}
        self.rec_eph[eph.sat][eph.iode] = eph.toes

        id_ = sat2id(eph.sat)
        sys, prn = sat2prn(eph.sat)

        if sys == uGNSS.BDS and (eph.iodc & 0xff) != eph.iode:
            return

        if sys == uGNSS.BDS:
            ep = time2epoch(gpst2bdt(eph.toc))
            week, tot_ = time2bdt(eph.tot)
        else:
            ep = time2epoch(eph.toc)
            week, tot_ = time2gpst(eph.tot)

        if sys == uGNSS.BDS and eph.mode == 1:  # B-CNAV1
            lbl = "CNV1"
            v1 = eph.Adot
        elif (sys == uGNSS.GPS or sys == uGNSS.QZS) and eph.mode == 0:  # LNAV
            lbl = "LNAV"
            v1 = float(eph.iode)
        elif sys == uGNSS.GAL and eph.mode == 0:  # INAV
            lbl = "INAV"
            v1 = float(eph.iode)
        else:
            return

        fh.write("> {:2s} {:3s} {:2s}\n".format("EPH", id_, lbl))
        fh.write("{:3s} {:4d} {:02d} {:02d} {:02d} {:02d} {:02d}".
                 format(id_, int(ep[0]), int(ep[1]), int(ep[2]),
                        int(ep[3]), int(ep[4]), int(ep[5])))
        fh.write("{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.af0, eph.af1, eph.af2))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(v1, eph.crs, eph.deln, eph.M0))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.cuc, eph.e, eph.cus, np.sqrt(eph.A)))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.toes, eph.cic, eph.OMG0, eph.cis))
        fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                 format(eph.i0, eph.crc, eph.omg, eph.OMGd))

        if sys == uGNSS.BDS and eph.mode == 1:  # B-CNAV1
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(eph.idot, eph.delnd, eph.sattype, eph.top))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(eph.sisai[0], eph.sisai[1], eph.sisai[2],
                            eph.sisai[3]))
            fh.write("    {:19.12E}{:19s}{:19.12E}{:19.12E}\n".
                     format(eph.isc[0], "", eph.tgd, eph.tgd_b))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(eph.sismai), float(eph.svh),
                            float(eph.integ), float(eph.iodc)))
            fh.write("    {:19.12E}{:19s}{:19s}{:19.12E}\n".
                     format(tot_, "", "", float(eph.iode)))

        if (sys == uGNSS.GPS or sys == uGNSS.QZS) and eph.mode == 0:  # LNAV
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(eph.idot, float(eph.code), float(eph.week),
                            float(eph.l2p)))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(eph.sva), float(eph.svh), eph.tgd,
                            float(eph.iodc)))
            fh.write("    {:19.12E}{:19.12E}{:19s}{:19s}\n".
                     format(tot_, float(eph.fit), "", ""))

        if sys == uGNSS.GAL and eph.mode == 0:  # INAV
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19s}\n".
                     format(eph.idot, float(eph.code), float(eph.week), ""))
            fh.write("    {:19.12E}{:19.12E}{:19.12E}{:19.12E}\n".
                     format(float(eph.sva), float(eph.svh), eph.tgd,
                            float(eph.tgd_b)))
            fh.write("    {:19.12E}{:19s}{:19s}{:19s}\n".
                     format(tot_, "", "", ""))
