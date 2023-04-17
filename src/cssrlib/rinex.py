"""
module for RINEX 3.0x processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, uTYP, rSigRnx
from cssrlib.gnss import gpst2time, epoch2time, timediff, gtime_t
from cssrlib.gnss import prn2sat, char2gns
from cssrlib.gnss import Eph, Obs
from _curses import nl


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
    MAXSAT = uGNSS.GPSMAX+uGNSS.GLOMAX+uGNSS.GALMAX+uGNSS.BDSMAX+uGNSS.QZSMAX

    def __init__(self):

        self.ver = -1.0
        self.fobs = None
        self.nf = 3  # TODO: create dynamically!!

        self.sig_map = {}  # signal code mapping to columns in data section
        self.sig_tab = {}  # signal selection for internal data structure
        self.nsig = {uTYP.C: 0, uTYP.L: 0, uTYP.D: 0, uTYP.S: 0}

        self.pos = np.array([0, 0, 0])
        self.rcv = None
        self.ant = None

    def setSignals(self, sigList):
        """ define the signal list for each constellation """

        for sig in sigList:
            if sig.gns not in self.sig_tab:
                self.sig_tab.update({sig.gns: {}})
            if sig.typ not in self.sig_tab[sig.gns]:
                self.sig_tab[sig.gns].update({sig.typ: []})
            if sig not in self.sig_tab[sig.gns][sig.typ]:
                self.sig_tab[sig.gns][sig.typ].append(sig)

        for _, sigs in self.sig_tab.items():
            for typ, sig in sigs.items():
                self.nsig[typ] = max((self.nsig[typ], len(sig)))

    def getSignals(self, gns, typ):
        """ retrieve signal list for constellation and obs type """
        if gns in self.sig_tab.keys() and typ in self.sig_tab[gns].keys():
            return self.sig_tab[gns][typ]
        else:
            return []

    def flt(self, u, c=-1):
        if c >= 0:
            u = u[19*c+4:19*(c+1)+4]
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
        """decode RINEX Navigation message from file """
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

                sys = char2gns(line[0])
                # Skip undesired constellations
                #
                if sys not in self.sig_tab:
                    continue

                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)
                eph = Eph(sat)

                eph.toc = self.decode_time(line, 4)
                eph.af0 = self.flt(line, 1)
                eph.af1 = self.flt(line, 2)
                eph.af2 = self.flt(line, 3)

                line = fnav.readline()
                eph.iode = int(self.flt(line, 0))
                eph.crs = self.flt(line, 1)
                eph.deln = self.flt(line, 2)
                eph.M0 = self.flt(line, 3)

                line = fnav.readline()
                eph.cuc = self.flt(line, 0)
                eph.e = self.flt(line, 1)
                eph.cus = self.flt(line, 2)
                sqrtA = self.flt(line, 3)
                eph.A = sqrtA**2

                line = fnav.readline()
                eph.toes = int(self.flt(line, 0))
                eph.cic = self.flt(line, 1)
                eph.OMG0 = self.flt(line, 2)
                eph.cis = self.flt(line, 3)

                line = fnav.readline()
                eph.i0 = self.flt(line, 0)
                eph.crc = self.flt(line, 1)
                eph.omg = self.flt(line, 2)
                eph.OMGd = self.flt(line, 3)

                line = fnav.readline()
                eph.idot = self.flt(line, 0)
                eph.code = int(self.flt(line, 1))  # source for GAL
                eph.week = int(self.flt(line, 2))

                line = fnav.readline()
                eph.sva = int(self.flt(line, 0))
                eph.svh = int(self.flt(line, 1))
                eph.tgd = float(self.flt(line, 2))
                if sys == uGNSS.GAL:
                    tgd_b = float(self.flt(line, 3))
                    if (eph.code >> 9) & 1:
                        eph.tgd = tgd_b
                else:
                    eph.iodc = int(self.flt(line, 3))

                line = fnav.readline()
                tot = int(self.flt(line, 0))
                if len(line) >= 42:
                    eph.fit = int(self.flt(line, 1))

                eph.toe = gpst2time(eph.week, eph.toes)
                eph.tot = gpst2time(eph.week, tot)
                nav.eph.append(eph)

        return nav

    def decode_clk(self, clkfile, nav):
        """decode RINEX Navigation message from file """
        nav.pclk = []
        with open(clkfile, 'rt') as fnav:
            for line in fnav:
                if line[0:2] != 'AS':
                    continue
                if line[3] not in self.gnss_tbl:
                    continue
                sys = self.gnss_tbl[line[3]]
                prn = int(line[4:7])
                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)
                t = self.decode_time(line, 8, 9)
                if nav.nc <= 0 or abs(timediff(nav.pclk[-1].time, t)) > 1e-9:
                    nav.nc += 1
                    pclk = pclk_t()
                    pclk.time = t
                    nav.pclk.append(pclk)

                nrec = int(line[35:37])
                clk = float(line[40:59])
                std = float(line[61:80]) if nrec >= 2 else 0.0
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
            elif line[60:79] == 'SYS / # / OBS TYPES':

                gns = char2gns(line[0])
                nsig = int(line[3:6])

                # Extract string list of signal codes
                #
                sigs = line[7:60].split()
                if nsig >= 14:
                    line2 = self.fobs.readline()
                    sigs += line2[7:60].split()

                # Convert to RINEX signal code and store in map
                #
                for i, sig in enumerate(sigs):
                    rnxSig = rSigRnx()
                    rnxSig.str2sig(gns, sig)
                    if gns not in self.sig_map:
                        self.sig_map.update({gns: {}})
                    self.sig_map[gns].update({i: rnxSig})

        return 0

    def decode_obs(self):
        """decode RINEX Observation message from file """
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

            obs.P = np.empty((0, self.nsig[uTYP.C]), dtype=np.float64)
            obs.L = np.empty((0, self.nsig[uTYP.L]), dtype=np.float64)
            obs.S = np.empty((0, self.nsig[uTYP.S]), dtype=np.float64)
            obs.lli = np.empty((0, self.nsig[uTYP.L]), dtype=np.int)
            obs.sat = np.empty(0, dtype=np.int)

            for _ in range(nsat):

                line = self.fobs.readline()
                sys = char2gns(line[0])

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
                sat = prn2sat(sys, prn)

                pr = np.zeros(len(self.getSignals(sys, uTYP.C)),
                              dtype=np.float64)
                cp = np.zeros(len(self.getSignals(sys, uTYP.L)),
                              dtype=np.float64)
                ll = np.zeros(len(self.getSignals(sys, uTYP.L)),
                              dtype=np.int)
                cn = np.zeros(len(self.getSignals(sys, uTYP.S)),
                              dtype=np.float64)

                for i, sig in self.sig_map[sys].items():

                    # Skip undesired signals
                    #
                    if sig.typ not in self.sig_tab[sys] or \
                            sig not in self.sig_tab[sys][sig.typ]:
                        continue

                    j = self.sig_tab[sys][sig.typ].index(sig)
                    obs_ = line[16*i+4:16*i+17].strip()

                    if sig.typ == uTYP.C:
                        pr[j] = float(obs_)
                    elif sig.typ == uTYP.L:
                        cp[j] = float(obs_)
                        if line[16*i+17] == '1':
                            ll[j] = 1
                    elif sig.typ == uTYP.S:
                        cn[j] = float(obs_)
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
