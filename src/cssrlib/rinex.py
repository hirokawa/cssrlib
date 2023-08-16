"""
module for RINEX 3.0x processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.gnss import bdt2gpst, time2bdt
from cssrlib.gnss import gpst2time, bdt2time, epoch2time, timediff, gtime_t
from cssrlib.gnss import prn2sat, char2sys
from cssrlib.gnss import Eph, Obs


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
                        atts = 'SLX'
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
