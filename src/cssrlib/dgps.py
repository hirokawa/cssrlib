"""
DGPS (QZSS SLAS) correction data decoder

[1] Quasi-Zenith Satellite System Interface Specification
    Sub-meter Level Augmentation Service (IS-QZSS-L1S-006), 2023

"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSRTYPE, prn2sat, sCType
from cssrlib.gnss import uGNSS, rCST, timediff, time2str, \
    ecef2pos


def vardgps(el, nav):
    """ measurement error varianvce for DGPS [1] """
    c_el = np.cos(el)
    pos = ecef2pos(nav.x[0:3])
    Fpp = 1/np.sqrt(1-(rCST.RE_WGS84*c_el/(rCST.RE_WGS84+pos[2]))**2)
    s_iono = Fpp*0.004*(nav.baseline+2*100*0.07)
    s_mp = 0.13+0.53*np.exp(-el/0.1745)
    v_air = 0.11**2 + s_mp**2
    v_pr = (0.16+0.107*np.exp(-el/0.271))**2/1+0.08**2
    v_sig = v_pr + v_air + s_iono**2
    return v_sig


class dgpsDec(cssr):
    MAX_ST = 64
    MAX_SAT = 23

    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.DGPS
        self.nsig_max = 0
        self.sat_n = []

        self.sat = []

        self.lc[0].dclk = {}
        self.lc[0].ddft = {}
        self.lc[0].dorb = {}
        self.lc[0].dvel = {}
        self.lc[0].iode = {}
        self.lc[0].t0 = {}

        self.tmax = {sCType.CLOCK: 120.0, sCType.ORBIT: 120.0}

        # GPS/QZS LNAV, Galileo F/NAV
        self.nav_mode = {uGNSS.GPS: 0, uGNSS.GAL: 1, uGNSS.QZS: 0,
                         uGNSS.GLO: 0, uGNSS.BDS: 0}

        self.iodp = -1
        self.iodi = -1
        self.iodm = -1
        self.iodg = -1

        self.iodssr = -1
        self.iodp_c = {}

        self.baselen = 0  # baseline length [km]

        self.posr = np.zeros((self.MAX_ST, 3))
        self.posr[0, :] = [43.150, 141.220,  50]
        self.posr[1, :] = [38.270, 140.740, 200]
        self.posr[3, :] = [36.580, 140.550, 150]
        self.posr[5, :] = [36.400, 136.410,  50]
        self.posr[6, :] = [34.710, 135.040, 200]
        self.posr[7, :] = [34.350, 132.450,  50]
        self.posr[8, :] = [33.600, 130.230,  50]
        self.posr[9, :] = [30.550, 130.940, 100]
        self.posr[10, :] = [28.420, 129.690,  50]
        self.posr[11, :] = [26.150, 127.690, 100]
        self.posr[12, :] = [24.730, 125.350, 100]
        self.posr[13, :] = [24.370, 124.130, 100]
        self.posr[14, :] = [27.090, 142.190, 100]

        self.mlen = {uGNSS.GPS: 64, uGNSS.QZS: 9, uGNSS.GLO: 36, uGNSS.GAL: 36,
                     uGNSS.BDS: 36}
        self.prn_ofst = {uGNSS.GPS: 0, uGNSS.QZS: 192, uGNSS.GLO: 0,
                         uGNSS.GAL: 0, uGNSS.BDS: 0}

    def sval(self, u, n, scl):
        """ calculate signed value based on n-bit int, lsb """
        invalid = 2**(n-1)-1
        y = np.nan if u == invalid else u*scl
        return y

    def check_validity(self, time):
        for sat in self.sat_n:
            if timediff(time, self.lc[0].t0[sat][sCType.CLOCK]) > \
                    self.tmax[sCType.CLOCK]:
                self.lc[0].dclk[sat] = 0.0
            if timediff(time, self.lc[0].t0[sat][sCType.ORBIT]) > \
                    self.tmax[sCType.ORBIT]:
                self.lc[0].iode[sat] = -1
                self.lc[0].dorb[sat] = np.zeros(3)

    def decode_mt43(self, msg, i):
        """ decode DC report by JMA (TBD) """
        # i = self.dcr.decode_jma(msg)
        return i

    def decode_mt44(self, msg, i):
        """ decode DC report by other organization (TBD) """
        # i = self.dcr.decode_ews(msg)
        return i

    def decode_mt47(self, msg, i):
        """ decode location of reference station """
        i = 14
        for k in range(5):
            code, lat, lon, alt = bs.unpack_from('u6s15s15u6', msg, i)
            if code == 63:
                continue
            self.posr[code, 0] = lat*0.005
            self.posr[code, 1] = lon*0.005+115
            self.posr[code, 2] = alt*50.0-100

            i += 42
        return 0

    def decode_mt48(self, msg, i):
        """ decode satellite mask """
        self.iodssr_p = self.iodp
        self.sat_n_p = self.sat_n

        i = 14
        self.nsat = 0
        v = bs.unpack_from('u2u64u9u36u36u36', msg, i)

        self.sat_n = []
        self.iodp = v[0]
        sys_t = [uGNSS.GPS, uGNSS.QZS, uGNSS.GLO, uGNSS.GAL, uGNSS.BDS]
        for k, sys in enumerate(sys_t):
            prn, nsat_s = self.decode_mask(v[k+1], self.mlen[sys])
            for j in range(nsat_s):
                self.sat_n.append(prn2sat(sys, prn[j]+self.prn_ofst[sys]))
            self.nsat += nsat_s

        self.iodssr = self.iodp

        self.lc[0].cstat |= (1 << sCType.MASK)
        self.set_t0(ctype=sCType.MASK, t=self.time)

        self.sat = self.sat_n

        return i

    def decode_mt49(self, msg, i):
        """ decode issue of data """
        iodp = bs.unpack_from('u2', msg, 223)[0]
        if iodp != self.iodp:
            return -1
        i = 14
        iodi, mask = bs.unpack_from('u2u23', msg, i)
        i += 25

        self.iodn = {}

        self.iodi = iodi  # issue of data
        for k in range(23):
            if (mask >> (23-k-1)) & 1 == 0 or k >= self.nsat:
                continue
            iodn = bs.unpack_from('u8', msg, i)[0]
            i += 8

            sat = self.sat[k]
            self.iodn[sat] = iodn

        return i

    def decode_mt50(self, msg, i):
        """ decode DGPS correction """
        iodp, iodi, code, h, mask_s = bs.unpack_from('u2u2u6u1u23', msg, i)
        i += 34
        if iodp != self.iodp or iodi != self.iodi or h == 1:
            return i

        nsat = 0
        inet = code+1

        self.lc[inet].iode = {}
        self.lc[inet].dclk = {}
        self.lc[inet].dorb = {}
        self.lc[inet].sat_n = []

        for k in range(23):
            if k >= self.nsat:
                continue
            sat = self.sat[k]
            self.lc[inet].dorb[sat] = np.zeros(3)

            if (mask_s >> (23-k-1)) & 1 == 0:
                # self.lc[inet].iode[sat] = -1
                # self.lc[inet].dclk[sat] = 0.0
                continue

            prc = bs.unpack_from('s12', msg, i)[0]*0.04
            i += 12

            self.lc[inet].sat_n.append(sat)
            self.lc[inet].iode[sat] = self.iodn[sat]
            self.lc[inet].dclk[sat] = prc

            self.set_t0(ctype=sCType.CLOCK, sat=sat, inet=inet, t=self.time)
            self.set_t0(ctype=sCType.ORBIT, sat=sat, inet=inet, t=self.time)

            nsat += 1
            if nsat > 14:
                break

        self.iodp_c[inet] = iodp

        self.lc[0].cstat |= (1 << sCType.CLOCK)
        self.lc[0].cstat |= (1 << sCType.ORBIT)

        return i

    def decode_mt51(self, msg, i):
        """ decode satellite health """
        v = bs.unpack_from('u64u9u36u36u36', msg, i)
        i += 181
        self.svh[uGNSS.GPS] = v[0]
        self.svh[uGNSS.QZS] = v[1]
        self.svh[uGNSS.GLO] = v[2]
        self.svh[uGNSS.GAL] = v[3]
        self.svh[uGNSS.BDS] = v[4]
        return i

    def decode_cssr(self, msg, i=0):
        """ decode DGPS messages """

        _, mt = bs.unpack_from('u8u6', msg, i)
        i += 14

        self.msgtype = mt

        if mt == 43:  # DC report (JMA)
            i = self.decode_mt43(msg, i)
        elif mt == 44:  # DC report (other source)
            i = self.decode_mt44(msg, i)
        elif mt == 47:  #
            i = self.decode_mt47(msg, i)
        elif mt == 48:  #
            i = self.decode_mt48(msg, i)
        elif mt == 49:  #
            i = self.decode_mt49(msg, i)
        elif mt == 50:  #
            i = self.decode_mt50(msg, i)
        elif mt == 51:  #
            i = self.decode_mt51(msg, i)
        elif mt == 63:  # Null
            None

        if self.monlevel > 3:
            self.fh.write("{:s} mt={:2d} \n".format(time2str(self.time), mt))

    def get_station(self, r):
        """ get the nearest referece station """
        pos = ecef2pos(r)
        c_lat = np.cos(pos[0])
        idx = np.where(self.posr[:, 0]**2+self.posr[:, 1]**2 > 0)[0]
        if len(idx) == 0:
            return -1, -1
        dpos = np.deg2rad(self.posr[idx, 0:2])-pos[0:2]
        rng = np.sqrt(dpos[:, 0]**2+(dpos[:, 1]*c_lat)**2)*rCST.RE_WGS84
        kmin = np.argmin(rng)
        refid = idx[kmin]
        return refid, rng[kmin]

    def set_dgps_corr(self, rr):
        """ get DGPS correction """
        if self.iodp < 0 or self.iodi < 0:
            return -1
        refid, rmin = self.get_station(rr)
        baselen = rmin*1e-3  # baseline length [km]
        self.inet = refid+1
        if len(self.lc[self.inet].sat_n) > 0:
            self.sat_n = self.lc[self.inet].sat_n
            self.iodssr_c[sCType.ORBIT] = self.iodp_c[self.inet]
            self.iodssr_c[sCType.CLOCK] = self.iodp_c[self.inet]
            self.lc[0].t0 = self.lc[self.inet].t0
            self.lc[0].iode = self.lc[self.inet].iode
            self.lc[0].dclk = self.lc[self.inet].dclk
            self.lc[0].dorb = self.lc[self.inet].dorb
        return baselen
