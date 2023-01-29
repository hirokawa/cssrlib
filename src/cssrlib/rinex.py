"""
module for RINEX 3.0x processing
"""

import numpy as np
from cssrlib.gnss import uGNSS, rSIG, Eph, prn2sat, gpst2time, Obs, \
    epoch2time, timediff, gtime_t

class pclk_t:
    def __init__(self, time = None):
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
        self.freq_tbl = {rSIG.L1C: 0, rSIG.L1X: 0, rSIG.L2W: 1, rSIG.L2L: 1,
                         rSIG.L2X: 1, rSIG.L5Q: 2, rSIG.L5X: 2, rSIG.L7Q: 1,
                         rSIG.L7X: 1}
        self.gnss_tbl = {'G': uGNSS.GPS, 'E': uGNSS.GAL, 'J': uGNSS.QZS}
        self.sig_tbl = {'1C': rSIG.L1C, '1X': rSIG.L1X, '1W': rSIG.L1W,
                        '2W': rSIG.L2W, '2L': rSIG.L2L, '2X': rSIG.L2X,
                        '5Q': rSIG.L5Q, '5X': rSIG.L5X, '7Q': rSIG.L7Q,
                        '7X': rSIG.L7X}
        self.skip_sig_tbl = {uGNSS.GPS: [rSIG.L1X, rSIG.L1W, rSIG.L2L,
                                         rSIG.L2X], uGNSS.GAL: [],
                             uGNSS.QZS: [rSIG.L1X]}
        self.nf = 4
        self.sigid = np.ones((uGNSS.GNSSMAX, rSIG.SIGMAX*3),
                             dtype=int)*rSIG.NONE
        self.typeid = np.ones((uGNSS.GNSSMAX, rSIG.SIGMAX*3),
                              dtype=int)*rSIG.NONE
        self.nsig = np.zeros((uGNSS.GNSSMAX), dtype=int)
        self.pos = np.array([0, 0, 0])

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
                if line[0] not in self.gnss_tbl:
                    continue
                sys = self.gnss_tbl[line[0]]
                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)
                eph = Eph(sat)

                eph.toc = self.decode_time(line,4)
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
                t = self.decode_time(line,8,9)
                if nav.nc<=0 or abs(timediff(nav.pclk[-1].time, t))>1e-9:
                    nav.nc+=1
                    pclk = pclk_t()
                    pclk.time = t
                    nav.pclk.append(pclk)
                    
                nrec = int(line[35:37])
                clk = float(line[40:59])
                std = float(line[61:80]) if nrec>=2 else 0.0
                nav.pclk[nav.nc-1].clk[sat-1] = clk
                nav.pclk[nav.nc-1].std[sat-1] = std
              
        return nav


    def decode_obsh(self, obsfile):
        self.fobs = open(obsfile, 'rt')
        for line in self.fobs:
            if line[60:73] == 'END OF HEADER':
                break
            if line[60:80] == 'RINEX VERSION / TYPE':
                self.ver = float(line[4:10])
                if self.ver < 3.02:
                    return -1
            elif line[60:79] == 'APPROX POSITION XYZ':
                self.pos = np.array([float(line[0:14]),
                                     float(line[14:28]),
                                     float(line[28:42])])
            elif line[60:79] == 'SYS / # / OBS TYPES':
                if line[0] in self.gnss_tbl:
                    sys = self.gnss_tbl[line[0]]
                else:
                    continue
                self.nsig[sys] = int(line[3:6])
                s = line[7:7+4*13]
                if self.nsig[sys] >= 14:
                    line2 = self.fobs.readline()
                    s += line2[7:7+4*13]

                for k in range(self.nsig[sys]):
                    sig = s[4*k:3+4*k]
                    if sig[0] == 'C':
                        self.typeid[sys][k] = 0
                    elif sig[0] == 'L':
                        self.typeid[sys][k] = 1
                    elif sig[0] == 'S':
                        self.typeid[sys][k] = 2
                    elif sig[0] == 'D':
                        self.typeid[sys][k] = 3
                    else:
                        continue
                    if sig[1:3] in self.sig_tbl:
                        if self.sig_tbl[sig[1:3]] in self.skip_sig_tbl[sys]:
                            continue
                        self.sigid[sys][k] = self.sig_tbl[sig[1:3]]
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
            obs.data = np.zeros((nsat, self.nf*4))
            obs.P = np.zeros((nsat, self.nf))
            obs.L = np.zeros((nsat, self.nf))
            obs.S = np.zeros((nsat, self.nf))
            obs.lli = np.zeros((nsat, self.nf), dtype=int)
            obs.mag = np.zeros((nsat, self.nf))
            obs.sat = np.zeros(nsat, dtype=int)
            for k in range(nsat):
                line = self.fobs.readline()
                if line[0] not in self.gnss_tbl:
                    continue
                sys = self.gnss_tbl[line[0]]
                prn = int(line[1:3])
                if sys == uGNSS.QZS:
                    prn += 192
                obs.sat[k] = prn2sat(sys, prn)
                nsig_max = (len(line)-4+2)//16
                for i in range(self.nsig[sys]):
                    if i >= nsig_max:
                        break
                    obs_ = line[16*i+4:16*i+17].strip()
                    if obs_ == '' or self.sigid[sys][i] == 0:
                        continue
                    ifreq = self.freq_tbl[self.sigid[sys][i]]
                    if self.typeid[sys][i] == 0:  # code
                        obs.P[k, ifreq] = float(obs_)
                    elif self.typeid[sys][i] == 1:  # carrier
                        obs.L[k, ifreq] = float(obs_)
                        if line[16*i+17] == '1':
                            obs.lli[k, ifreq] = 1
                    elif self.typeid[sys][i] == 2:  # C/No
                        obs.S[k, ifreq] = float(obs_)
                    obs.data[k, ifreq*self.nf+self.typeid[sys][i]] = \
                        float(obs_)

            break
        return obs


def sync_obs(dec, decb, dt_th=0.1):
    """ sync obseverbation beteen rover and base """
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
