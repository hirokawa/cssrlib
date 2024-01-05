"""
SBAS correction data decoder

"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSR, sCSSRTYPE, prn2sat, sCType
from cssrlib.cssrlib import sat2id
from cssrlib.gnss import uGNSS, rCST, timediff, \
    time2gpst, Seph, Eph, Alm, tod2tow, gtime_t, ionppp

MAXBAND = 11


class Salm():
    """ class to define SBAS almanac """
    sat = 0
    t0 = gtime_t()
    svh = 0
    pos = np.zeros(3)
    vel = np.zeros(3)
    mode = 0

    def __init__(self, sat=0):
        self.sat = sat


class SBASService():
    """ structure SBAS service information for MT27 """
    iods = -1
    nmsg = 0
    snum = 0
    nreg = 0
    priority = 0
    dUDRE_in = 0
    dUDRE_out = 0
    lat1 = np.zeros(5)
    lon1 = np.zeros(5)
    lat2 = np.zeros(5)
    lon2 = np.zeros(5)
    shape = 0


class paramOBAD():
    tid = 0
    dt_mt32 = 0
    dt_mt39 = 0
    Cer = 0
    Ccov = 0
    Icorr = np.zeros(6)
    Ccorr = np.zeros(6)
    Rcorr = np.zeros(6)
    dfrei_tbl = np.zeros(16)


def searchIGP(t, posp, cs):
    """ search IGP with pierce-point position """
    x = y = 0.0
    igp = {}

    latp = np.zeros(4)
    lonp = np.zeros(4)

    lat, lon = posp[0]*rCST.R2D, posp[1]*rCST.R2D

    if lon >= 180.0:
        lon -= 360.0

    if lat >= -55.0 and lat < 55.0:
        latp[0] = lat//5*5
        latp[1] = latp[0]+5
        lonp[0] = lonp[1] = lon//5*5
        lonp[2] = lonp[3] = lonp[0]+5
        x = (lon-lonp[0])/5.0
        y = (lat-latp[0])/5.0
    else:
        latp[0] = (lat-5.0)//10*10+5
        latp[1] = latp[0] + 10
        lonp[0] = lonp[1] = lon//10*10
        lonp[2] = lonp[3] = lonp[0]+10
        x = (lon-lonp[0])/10.0
        y = (lat-latp[0])/10.0
        if lat >= 75.0 and lat < 85.0:
            lonp[1] = lon//90*90
            lonp[3] = lonp[1]+90
        elif lat >= -85.0 and lat < -75.0:
            lonp[0] = (lon-50.0)//90*90+40
            lonp[2] = lonp[0] + 90
        elif lat >= 85.0:
            lonp[0:4] = lon//90*90
        elif lat < -85.0:
            lonp[0:4] = (lon-50)//90*90+40

    for i in range(4):
        if lonp[i] == 180:
            lonp[i] = -180
    for band in range(MAXBAND):
        for k, gp in enumerate(cs.igp_t[band]):
            for i in range(4):
                if gp[0] == latp[i] and gp[1] == lonp[i] and \
                        cs.givei[band][k] > 0:
                    igp[i] = [cs.vtec[band][k], cs.givei[band][k]]
                    break
            if 0 in igp and 1 in igp and 2 in igp and 3 in igp:
                break

    return igp, x, y


def ionoSBAS(t, pos, az, el, cs):
    """ ionospheric delay for SBAS """
    re, hion = 6378.1363, 350.0
    diono = 0.0
    var = 0.0

    err = False
    givei_t = [0.0084, 0.0333, 0.0749, 0.1331, 0.2079, 0.2994, 0.4075, 0.5322,
               0.6735, 0.8315, 1.1974, 1.8709, 3.326, 20.787, 187.0826]

    idx_t = [[1, 2, 0], [0, 3, 2], [0, 3, 1], [1, 2, 3]]

    if pos[2] < -100.0 or el <= 0.0:
        return None

    # ipp (ionospheric pierce point) position and slant factor
    sf, posp = ionppp(pos, az, el, re, hion)

    igp, x, y = searchIGP(t, posp, cs)
    x1, y1 = 1.0-x, 1.0-y

    if 0 in igp and 1 in igp and 2 in igp and 3 in igp:
        w = [x1*y1, x1*y, x*y1, x*y]
    else:  # triangle case
        w = [0, 0, 0, 0]
        v = [[y, x], [x1, y], [y1, x], [x1, y1]]
        err = True
        for k, idx in enumerate(idx_t):
            if idx[0] in igp and idx[1] in igp and idx[2] in igp:
                w[idx[0]] = v[k][0]
                w[idx[1]] = v[k][1]
                w[idx[2]] = 1.0-w[idx[0]]-w[idx[1]]
                err = True if w[idx[2]] < 0.0 else False
                break

    if err:
        return diono, var

    for i in range(4):
        if i not in igp:
            continue
        dt = timediff(t, cs.t0_igp)
        diono += w[i]*igp[i][0]
        var += w[i]*givei_t[igp[i][1]]*9e-8*abs(dt)

    diono *= sf
    var *= sf**2
    return diono, var


class sbasDec(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.PVS_PPP
        self.nsig_max = 0
        self.sat_n = []

        self.lc[0].dclk = {}
        self.lc[0].ddft = {}
        self.lc[0].dorb = {}
        self.lc[0].dvel = {}
        self.lc[0].iode = {}
        self.lc[0].t0 = {}

        self.tmax = {sCType.CLOCK: 120.0, sCType.ORBIT: 120.0}

        # GPS LNAV, Galileo F/NAV
        self.nav_mode = {uGNSS.GPS: 0, uGNSS.GAL: 1}

        self.iodp = -1
        self.iodi = -1
        self.iodm = -1
        self.iodg = -1

        # iono
        self.igp_idx = {}
        self.vtec = {}
        self.givei = {}
        self.t0_igp = gtime_t()

        # intrgrity information
        self.cov = {}
        self.fc = {}
        self.udrei = {}
        self.dRcorr = {}
        self.ai = {}
        self.deg_prm = np.zeros(16)

        self.obad = paramOBAD()

        # ephemeris, almanac
        self.sat_ref = -1
        self.Seph = None
        self.Eph = None
        self.Alm = {}

        self.sinfo = SBASService()

        self.givei_t = [0.0084, 0.0333, 0.0749, 0.1331, 0.2079, 0.2994, 0.4075,
                        0.5322, 0.6735, 0.8315, 1.1974, 1.8709, 3.3260, 20.787,
                        187.0826, 0]

        self.ura_t = [2, 2.8, 4, 5.7, 8, 11.3, 16,
                      32, 64, 128, 256, 512, 1024, 2048, 4096, 0]

        # UDREI_i
        self.udrei_t = [0.0520, 0.0924, 0.1444, 0.2830, 0.4678, 0.8315, 1.2992,
                        1.8709, 2.5465, 3.3260, 5.1968, 20.7870, 230.9661,
                        2078.695, 0, 0]

        self.scl_dfrei_t = [0.0625, 0.125, 0.125, 0.125, 0.125,
                            0.25, 0.25, 0.25, 0.25, 0.25, 0.5, 0.5, 1, 3, 6]

        # dUDRE indicator
        self.dudrei_t = [1, 1.1, 1.25, 1.5, 2, 3,
                         4, 5, 6, 8, 10, 20, 30, 40, 50, 100]

        # fast correction degradation factor
        self.ai_t = [0.0, 0.05, 0.09, 0.12, 0.15, 0.20, 0.30,
                     0.45, 0.60, 0.90, 1.50, 2.10, 2.70, 3.30, 4.60, 5.80]

        self.init_sbas_gp()

    def init_sbas_gp(self):
        """ SBAS IGP coordinates definition """
        self.igp_t = {}

        p0 = np.r_[np.array([-85, -75, -65]),
                   np.arange(-55, 60, 5),
                   np.array([65, 75, 85])]

        for band in range(0, 9):  # for band 0-8
            lon0 = -180+40*band
            ngp = 200 if band == 8 else 201
            self.igp_t[band] = np.zeros((ngp, 2))
            s = band % 2
            kr = band - s
            i0 = 0
            for k in range(8):
                if k % 2 == 0:
                    if k == kr:
                        n = 28
                        if s == 0:
                            self.igp_t[band][i0:i0+n, 0] = p0[1:]
                        else:
                            self.igp_t[band][i0:i0+n, 0] = p0[0:28]
                    else:
                        n = 27
                        self.igp_t[band][i0:i0+n, 0] = p0[1:28]
                else:
                    n = 23
                    self.igp_t[band][i0:i0+n, 0] = p0[3:26]
                self.igp_t[band][i0:i0+n, 1] = lon0 + k*5
                i0 += n

        for band in range(9, 11):  # for band 9-10
            self.igp_t[band] = np.zeros((192, 2))
            lat0 = 60 if band == 9 else -60
            i0 = 0
            for k in range(5):
                lat = lat0+k*np.sign(lat0)*5
                if k == 0:
                    n = 72
                    self.igp_t[band][i0:i0+n, 1] = np.arange(-180, 180, 5)
                elif k == 4:
                    n = 12
                    lon0 = -180 if lat0 > 0 else -170
                    self.igp_t[band][i0:i0+n, 1] = np.arange(lon0, 180, 30)
                else:
                    n = 36
                    self.igp_t[band][i0:i0+n, 1] = np.arange(-180, 180, 10)

                self.igp_t[band][i0:i0+n, 0] = lat
                i0 += n

    def sval(self, u, n, scl):
        """ calculate signed value based on n-bit int, lsb """
        invalid = 2**(n-1)-1
        y = np.nan if u == invalid else u*scl
        return y

    def slot2sat(self, slot):
        """ convert from satellite slot to satellite number """
        sat = 0
        if slot >= 1 and slot <= 32:
            prn = slot
            sat = prn2sat(uGNSS.GPS, prn)
        elif slot >= 38 and slot <= 69:
            prn = slot-37
            sat = prn2sat(uGNSS.GLO, prn)
        elif slot >= 75 and slot <= 110:
            prn = slot-74
            sat = prn2sat(uGNSS.GAL, prn)
        elif slot >= 120 and slot <= 158:
            prn = slot
            sat = prn2sat(uGNSS.SBS, prn)
        elif slot >= 159 and slot <= 195:
            prn = slot-158
            sat = prn2sat(uGNSS.BDS, prn)
        return sat

    def check_validity(self, time):
        for sat in self.sat_n:
            if timediff(time, self.lc[0].t0[sat][sCType.CLOCK]) > \
                    self.tmax[sCType.CLOCK]:
                self.lc[0].dclk[sat] = 0.0
            if timediff(time, self.lc[0].t0[sat][sCType.ORBIT]) > \
                    self.tmax[sCType.ORBIT]:
                self.lc[0].iode[sat] = -1
                self.lc[0].dorb[sat] = np.zeros(3)

    def decode_sbas_mask(self, msg, i):
        """ Type 1 PRN mask message """
        self.sat = []
        mask_gps, mask_glo = bs.unpack_from('u37u24', msg, i)
        i += 61

        prn, nsat = self.decode_mask(mask_gps, 37)
        for k in range(nsat):
            self.sat.append(prn2sat(uGNSS.GPS, prn[k]))

        prn, nsat = self.decode_mask(mask_glo, 24)
        for k in range(nsat):
            self.sat.append(prn2sat(uGNSS.GLO, prn[k]))

        i += 58
        mask_sbs = bs.unpack_from('u39', msg, i)[0]
        i += 91

        prn, nsat = self.decode_mask(mask_sbs, 39, 120)
        for k in range(nsat):
            self.sat.append(prn2sat(uGNSS.SBS, prn[k]))

        self.iodp = bs.unpack_from('u2', msg, i)[0]
        self.iodssr = self.iodp
        i += 2
        return i

    def decode_sbas_fast_corr(self, msg, i):
        """ Types 2 to 5 fast correction message """
        type_ = 0 if self.msgtype == 0 else self.msgtype-2

        iodf, iodp = bs.unpack_from('u2u2', msg, i)
        i += 4

        if iodp != self.iodp:
            i += 208
            return i

        for k in range(13):
            fc = bs.unpack_from('s12', msg, i)[0]
            i += 12
            j = type_*13+k
            if j >= len(self.sat):
                break
            self.fc[self.sat[j]] = self.sval(fc, 12, 0.125)
        for k in range(13):
            udrei = bs.unpack_from('u4', msg, i)[0]
            i += 4
            j = type_*13+k
            if j >= len(self.sat):
                break
            self.udrei[self.sat[j]] = udrei
            # self.lc[0].t0[self.sat[j]][sCType.HCLOCK] = self.time0
        return i

    def decode_sbas_integrity(self, msg, i):
        """ Type 6 integrity message """
        iodf2, iodf3, iodf4, iodf5 = bs.unpack_from('u2u2u2u2', msg, i)
        i += 8
        for k in range(51):
            if k >= len(self.sat):
                break
            self.udrei[self.sat[k]] = bs.unpack_from('u4', msg, i)[0]
            i += 4
        return i

    def decode_sbas_fast_degradation(self, msg, i):
        """ Type 7 fast correction degradation factor message """
        latency, iodp = bs.unpack_from('u4u2', msg, i)
        i += 8
        if iodp != self.iodp:
            i += 204
            return i
        for k in range(51):
            if k >= len(self.sat):
                break
            self.ai[self.sat[k]] = bs.unpack_from('u4', msg, i)[0]
            i += 4
        return i

    def decode_sbas_ranging(self, msg, i):
        """ Type 9 ranging function message """
        i += 8
        t0, ura, xg, yg, zg = bs.unpack_from('u13u4s30s30s25', msg, i)
        i += 102
        vxg, vyg, vzg = bs.unpack_from('s17s17s18', msg, i)
        i += 52
        axg, ayg, azg = bs.unpack_from('s10s10s10', msg, i)
        i += 30
        af0, af1 = bs.unpack_from('s12s8', msg, i)
        i += 20

        if self.sat_ref <= 0:
            return i

        seph = Seph(self.sat_ref)
        seph.t0 = tod2tow(t0*16.0, self.time0)
        seph.iodn = 0
        seph.svh = 0
        seph.sva = self.ura_t[ura]
        seph.mode = 0
        seph.pos[0] = self.sval(xg, 30, 0.08)
        seph.pos[1] = self.sval(yg, 30, 0.08)
        seph.pos[2] = self.sval(zg, 25, 0.4)
        seph.vel[0] = self.sval(vxg, 17, 6.25e-4)
        seph.vel[1] = self.sval(vyg, 17, 6.25e-4)
        seph.vel[2] = self.sval(vzg, 18, 4.0e-3)
        seph.acc[0] = self.sval(axg, 10, 1.25e-5)
        seph.acc[1] = self.sval(ayg, 10, 1.25e-5)
        seph.acc[2] = self.sval(azg, 10, 6.25e-5)
        seph.af0 = self.sval(af0, 12, rCST.P2_31)
        seph.af1 = self.sval(af1,  8, rCST.P2_40)

        self.Seph = seph

        return i

    def decode_sbas_degradation_param(self, msg, i):
        """ Type 10 degradation parameter message """
        Brrc, Cltc_lsb, Cltc_v1, Iltc_v1, Cltc_v0, Iltc_v0 = bs.unpack_from(
            'u10u10u10u9u10u9', msg, i)
        i += 58
        Cgeo_lsb, Cgeo_v, Igeo, Cer, Ciono_s, Iiono, Ciono_r = bs.unpack_from(
            'u10u10u9u6u10u9u10', msg, i)
        i += 64
        RSS_udre, RSS_iono, Ccov = bs.unpack_from('u1u1u7', msg, i)
        i += 90

        self.deg_prm = np.array(
            [Brrc*2e-3, Cltc_lsb*2e-3, Cltc_v1*5e-4, Iltc_v1*1.0,
             Cltc_v0*2e-3, Iltc_v0*1.0, Cgeo_lsb*5e-4, Cgeo_v*5e-5,
             Igeo*1.0, Cer*0.5, Ciono_s*1e-3, Iiono*1.0,
             Ciono_r*5e-6, RSS_udre, RSS_iono, Ccov*0.1])

        return i

    def decode_sbas_utc(self, msg, i):
        """ Type 12 SBAS network time/UTC message """
        A1snt, A0snt, t0t, wnt, dtls, wnlsf, dn, dtlsf = bs.unpack_from(
            's24s32u8u8s8u8u8s8', msg, i)
        i += 104
        utcid, tow, wn, gloid, daglo = bs.unpack_from(
            'u3u20u10u1s24', msg, i)
        i += 108
        return i

    def decode_sbas_almanac(self, msg, i):
        """ Type 17 GEO almanac message """

        ta = bs.unpack_from('u11', msg, 201)[0]*64.0
        t0 = tod2tow(ta, self.time0)

        for k in range(3):
            i += 2
            prn, svh, x, y, z, vx, vy, vz = bs.unpack_from(
                'u8u8s15s15s9s3s3s4', msg, i)
            i += 65

            if prn == 0:
                continue

            sat = prn2sat(uGNSS.SBS, prn)
            alm = Salm(sat)

            alm.svh = svh
            alm.pos[0] = self.sval(x, 15, 2600.0)
            alm.pos[1] = self.sval(y, 15, 2600.0)
            alm.pos[2] = self.sval(z, 9, 26000.0)

            alm.vel[0] = self.sval(vx, 3, 10.0)
            alm.vel[1] = self.sval(vy, 3, 10.0)
            alm.vel[2] = self.sval(vz, 4, 60.0)

            alm.t0 = t0

            self.Alm[sat] = alm

        i += 11
        return i

    def decode_sbas_igpmask(self, msg, i):
        """ Type 18 IGP mask message """
        nb, band, iodi = bs.unpack_from('u4u4u2', msg, i)
        i += 10
        idx = []
        for k in range(201):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                idx.append(k)
        i += 201+1
        self.igp_idx[band] = idx
        self.iodi = iodi
        ngp = len(idx)

        if band not in self.vtec:
            self.vtec[band] = np.zeros(ngp)

        if band not in self.givei:
            self.givei[band] = np.zeros(ngp, dtype=int)

        return i

    def add_sbas_corr(self, slot, iodn, dx, dy, dz, db,
                      dxd=0, dyd=0, dzd=0, dbd=0):
        """ add SBAS long-term correction """
        sat = self.slot2sat(slot)
        if sat not in self.sat_n:
            self.sat_n.append(sat)

        self.lc[0].iode[sat] = iodn
        self.lc[0].dorb[sat] = np.zeros(3)
        self.lc[0].dorb[sat][0] = self.sval(dx, 11, 0.125)
        self.lc[0].dorb[sat][1] = self.sval(dy, 11, 0.125)
        self.lc[0].dorb[sat][2] = self.sval(dz, 11, 0.125)

        self.lc[0].dclk[sat] = self.sval(db, 11, rCST.P2_31*rCST.CLIGHT)
        self.lc[0].dvel[sat] = np.zeros(3)
        self.lc[0].dvel[sat][0] = self.sval(dxd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][1] = self.sval(dyd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][2] = self.sval(dzd, 8, rCST.P2_11)

        self.lc[0].ddft[sat] = self.sval(db, 8, rCST.P2_39*rCST.CLIGHT)

        self.lc[0].cstat |= (1 << sCType.CLOCK) | (1 << sCType.ORBIT)
        self.lc[0].t0[sat] = {
            sCType.CLOCK: self.time0, sCType.ORBIT: self.time0}

        return sat

    def decode_sbas_lcorr_half(self, msg, i):
        """ Type 25 long-term satellite error correction half message """
        vc = bs.unpack_from('u1', msg, i)[0]
        i += 1

        i0 = 117 if vc == 0 else 105
        iodp = bs.unpack_from('u2', msg, i0)[0]
        if iodp != self.iodp:
            return i

        if vc == 0:
            for k in range(2):
                slot, iodn, dx, dy, dz, db = bs.unpack_from(
                    'u6u8s9s9s9s10', msg, i)
                i += 51
                if slot > 0:
                    self.add_sbas_corr(slot, iodn, dx, dy, dz, db)

            iodp = bs.unpack_from('u2', msg, i)[0]
            i += 3
        else:
            slot, iodn, dx, dy, dz, db, dxd, dyd, dzd, dbd = bs.unpack_from(
                'u6u8s11s11s11s11s8s8s8s8', msg, i)
            i += 90
            self.add_sbas_corr(slot, iodn, dx, dy, dz, db, dxd, dyd, dzd, dbd)

            toa, iodp = bs.unpack_from('u13u2', msg, i)
            i += 15

        self.iodssr_c[sCType.CLOCK] = iodp
        self.iodssr_c[sCType.ORBIT] = iodp

        return i

    def decode_sbas_long_corr(self, msg, i):
        """ Type 25 long-term satellite error correction """
        for k in range(2):
            i = self.decode_sbas_lcorr_half(msg, i)
        return i

    def decode_sbas_iono(self, msg, i):
        """ Type 26 Ionospheric delay message """
        iodi = bs.unpack_from('u2', msg, 217)[0]
        if iodi != self.iodi:
            i += 212
            return i
        band, bid = bs.unpack_from('u4u4', msg, i)
        nigp = len(self.vtec[band])
        i += 8
        for k in range(15):
            vd, givei = bs.unpack_from('u9u4', msg, i)
            i += 13
            j = bid*15+k
            if j >= nigp:
                break

            self.vtec[band][j] = vd*0.125 if vd < 0x1ff else np.nan
            self.givei[band][j] = givei
        i += 9

        self.t0_igp = self.time0

        return i

    def decode_sbas_service(self, msg, i):
        """ Type 27 SBAS service message """
        iods, ns, sm, nreg, code = bs.unpack_from('u3u3u3u3u2', msg, i)
        i += 14
        dudre1, dudre2 = bs.unpack_from('u4u4', msg, i)
        i += 8

        self.sinfo.iods = iods
        self.sinfo.nmsg = ns
        self.sinfo.snum = sm
        self.sinfo.nreg = nreg
        self.sinfo.priority = code
        self.sinfo.dUDRE_in = dudre1
        self.sinfo.dUDRE_out = dudre2

        for k in range(5):
            lat1, lon1, lat2, lon2 = bs.unpack_from('s8s9s8s9', msg, i)
            i += 34
            self.sinfo.lat1[k] = lat1
            self.sinfo.lon1[k] = lon1
            self.sinfo.lat2[k] = lat2
            self.sinfo.lon2[k] = lon2

        self.sinfo.shape = bs.unpack_from('u1', msg, i)[0]
        i += 16

        return i

    def decode_cov(self, msg, i):
        """ covarience for SBAS """
        scale, E11, E22, E33, E44, E12, E13, E14, E23, E24, E34 = \
            bs.unpack_from('u3u9u9u9u9s10s10s10s10s10s10', msg, i)
        i += 99
        R = np.array([[E11, E12, E13, E14],
                      [0, E22, E23, E23],
                      [0,   0, E33, E34],
                      [0,   0,   0, E44]])*(2**(scale-5))
        C = R.T@R
        return i, C

    def decode_sbas_cov(self, msg, i):
        """ Type 28 clock-ephemeris covariance matrix """
        iodp = bs.unpack_from('u2', msg, i)[0]
        i += 2
        if iodp != self.iodp:
            i += 210
            return i

        for k in range(2):
            slot = bs.unpack_from('u6', msg, i)[0]
            i += 6
            i, C = self.decode_cov(msg, i)
            sat = self.sat[slot]
            self.cov[sat] = C

        return i

    """ DFMC functions """

    def decode_dfmc_mask(self, msg, i):
        """ Type 31 PRN mask message """
        self.sat = []
        for k in range(32):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                self.sat.append(prn2sat(uGNSS.GPS, k+1))
        i += 37
        for k in range(32):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                self.sat.append(prn2sat(uGNSS.GLO, k+1))
        i += 37
        for k in range(36):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                self.sat.append(prn2sat(uGNSS.GAL, k+1))
        i += 37
        i += 8  # skip spare

        for k in range(39):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                self.sat.append(prn2sat(uGNSS.SBS, k+120))
        i += 39
        for k in range(37):
            if bs.unpack_from('u1', msg, i+k)[0] == 1:
                self.sat.append(prn2sat(uGNSS.BDS, k+1))
        i += 37
        i += 19  # skip spare
        self.iodm = bs.unpack_from('u2', msg, i)[0]
        self.iodssr = self.iodm
        i += 2
        return i

    def decode_dfmc_integrity_mt34(self, msg, i):
        """ Types 34 integrity message """
        iodm = bs.unpack_from('u2', msg, 214)[0]
        if iodm != self.iodm:
            i += 216
            return i

        idx = []
        for k in range(92):
            dfreci = bs.unpack_from('u2', msg, i)[0]
            i += 2
            if dfreci == 1:
                idx += [k]
            elif dfreci == 2:
                if self.dfrei[k] < 15:
                    self.dfrei[k] += 1
            elif dfreci == 3:
                self.dfrei[k] = 15

        dfrei = np.zeros(7, dtype=int)
        for k in range(7):
            dfrei[k] = bs.unpack_from('u4', msg, i)[0]
            i += 4
        j = 0
        for k in idx:
            self.udrei[k] = dfrei[j]
            j += 1
            if j > 7:
                break

        i += 4
        return i

    def decode_dfmc_integrity_mt35(self, msg, i):
        """ Types 35 integrity message """
        iodm = bs.unpack_from('u2', msg, 214)[0]
        if iodm != self.iodm:
            i += 216
            return i

        for k in range(53):
            self.udrei[k] = bs.unpack_from('u4', msg, i)[0]
            i += 4
        i += 4
        return i

    def decode_dfmc_integrity_mt36(self, msg, i):
        """ Types 36 integrity message """
        iodm = bs.unpack_from('u2', msg, 214)[0]
        if iodm != self.iodm:
            i += 216
            return i

        for k in range(39):
            self.udrei[k+53] = bs.unpack_from('u4', msg, i)[0]
            i += 4
        i += 60
        return i

    def decode_dfmc_obad(self, msg, i):
        """ Types 37 OBAD parameter and DFREI scale table """
        ivalid_mt32, ivalid_mt39, Cer, Ccov = \
            bs.unpack_from('u6u6u6u7', msg, i)
        i += 25

        self.obad.dt_mt32 = ivalid_mt32*6.0
        self.obad.dt_mt39 = ivalid_mt39*6.0
        self.obad.Cer = Cer*0.5
        self.obad.Ccov = Ccov*0.1

        for k in range(6):
            Icorr, Ccorr, Rcorr = bs.unpack_from('u5u8u8', msg, i)
            i += 21
            self.obad.Icorr[k] = Icorr*6.0
            self.obad.Ccorr[k] = Ccorr*0.01
            self.obad.Rcorr[k] = Rcorr*0.2

        for k in range(15):
            scl = self.scl_dfrei_t[k]
            self.obad.dfrei_tbl[k] = bs.unpack_from('u4', msg, i)[0]*scl
            i += 4

        self.obad.tid = bs.unpack_from('u3', msg, i)[0]
        i += 5
        return i

    def decode_dfmc_eph_mt39(self, msg, i):
        """ Keplerian param. part I MT39 """
        slotd, iodg, provider = bs.unpack_from('u6u2u5', msg, i)
        i += 13
        # Keplerian param. part I
        cuc, cus, idot, omg, Omg0, M0 = bs.unpack_from(
            's19s19s22s34s34s34', msg, i)
        i += 162
        af0, af1 = bs.unpack_from('s25s16', msg, i)
        i += 41
        self.iodg = iodg

        sat = self.slot2sat(slotd)
        eph = Eph(sat)
        eph.cuc = self.sval(cuc, 19, rCST.SC2RAD*rCST.P2_19*1e-4)
        eph.cus = self.sval(cus, 19, rCST.SC2RAD*rCST.P2_19*1e-4)
        eph.idot = self.sval(idot, 22, 7/6*rCST.SC2RAD*rCST.P2_21*1e-6)
        eph.omg = self.sval(omg, 34, rCST.SC2RAD*rCST.P2_33)
        eph.Omg0 = self.sval(Omg0, 34, rCST.SC2RAD*rCST.P2_33)
        eph.M0 = self.sval(M0, 34, rCST.SC2RAD*rCST.P2_33)

        eph.af0 = self.sval(af0, 25, 0.02/rCST.CLIGHT)  # [m] -> [s]
        eph.af1 = self.sval(af1, 16, 4e-5/rCST.CLIGHT)  # [m/s] -> [s/s]

        self.Eph = eph

        return i

    def decode_dfmc_eph_mt40(self, msg, i):
        """ Keplerian param. part II MT40 """
        iodg = bs.unpack_from('u2', msg, i)[0]
        i += 2
        if iodg != self.iodg:
            i += 115
            return i

        i0, e, a, te = bs.unpack_from('u33u30u31u13', msg, i)
        i += 107

        eph = self.Eph
        eph.i0 = i0*rCST.SC2RAD*rCST.P2_33
        eph.e = e*rCST.P2_30
        eph.a = a*0.02

        eph.toe = tod2tow(te*16.0, self.time0)
        eph.toc = eph.toe

        i, C = self.decode_cov(msg, i)
        dfrei, dRcorr = bs.unpack_from('u4u4', msg, i)
        i += 8

        self.cov[eph.sat] = C
        self.udrei[eph.sat] = dfrei
        self.dRcorr[eph.sat] = dRcorr

        return i

    def decode_dfmc_alm(self, msg, i):
        """ DFMC SBAS almanac MT47 """
        for k in range(2):
            slotd, provider, bi = bs.unpack_from('u6u5u1', msg, i)
            i += 12
            if slotd == 0:
                i += 94
                continue
            sat = prn2sat(uGNSS.SBS, slotd+119)

            a, e, i0, omg, OMG0, OMGd, M0, ta = \
                bs.unpack_from('u16u8u13s14s14s8s15u6', msg, i)
            i += 94

            alm = Alm(sat)
            alm.A = a*650.0
            alm.e = e*rCST.P2_8
            alm.i0 = i0*rCST.SC2RAD*rCST.P2_13
            alm.omg = omg*rCST.SC2RAD*rCST.P2_13
            alm.OMG0 = OMG0*rCST.SC2RAD*rCST.P2_13
            alm.OMDd = OMGd*1e-9
            alm.M0 = M0*rCST.SC2RAD*rCST.P2_14
            alm.toa = tod2tow(ta*1800.0, self.time0)
            self.Alm[sat] = alm

        # wnro_c = bs.unpack_from('u4', msg, i)[0]
        i += 4
        return i

    def add_dfmc_corr(self, slot, iodn, dx, dy, dz, db,
                      dxd=0, dyd=0, dzd=0, dbd=0):
        sat = self.slot2sat(slot)
        if sat not in self.sat_n:
            self.sat_n.append(sat)

        self.lc[0].iode[sat] = iodn
        self.lc[0].dorb[sat] = np.zeros(3)
        self.lc[0].dorb[sat][0] = self.sval(dx, 11, 0.0625)
        self.lc[0].dorb[sat][1] = self.sval(dy, 11, 0.0625)
        self.lc[0].dorb[sat][2] = self.sval(dz, 11, 0.0625)

        self.lc[0].dclk[sat] = self.sval(db, 12, 0.03125)

        self.lc[0].dvel[sat] = np.zeros(3)
        self.lc[0].dvel[sat][0] = self.sval(dxd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][1] = self.sval(dyd, 8, rCST.P2_11)
        self.lc[0].dvel[sat][2] = self.sval(dzd, 8, rCST.P2_11)

        self.lc[0].ddft[sat] = self.sval(dbd, 9, rCST.P2_12)

        self.lc[0].cstat |= (1 << sCType.CLOCK) | (1 << sCType.ORBIT)
        self.lc[0].t0[sat] = {
            sCType.CLOCK: self.time0, sCType.ORBIT: self.time0}

        return sat

    def decode_cssr_orb(self, msg, i, inet=0):
        """ Types 32 clock-ephemeris correction and covariance matrix """
        slot, iodn = bs.unpack_from('u8u10', msg, i)
        i += 18

        if slot == 0:  # for DFMC SBAS only
            return

        dx, dy, dz, db = bs.unpack_from('s11s11s11s12', msg, i)
        i += 45
        dxd, dyd, dzd, dbd, t0 = bs.unpack_from('s8s8s8s9u13', msg, i)
        i += 46

        i, C = self.decode_cov(msg, i)

        dfrei, dRcorr = bs.unpack_from('u4u4', msg, i)
        i += 8

        sat = self.add_dfmc_corr(
            slot, iodn, dx, dy, dz, db, dxd, dyd, dzd, dbd)

        self.cov[sat] = C
        self.udrei[sat] = dfrei
        self.dRcorr[sat] = dRcorr

        self.time = tod2tow(t0*16.0, self.time0)

        self.iodssr = 0

        self.iodssr_c[sCType.CLOCK] = self.iodssr
        self.iodssr_c[sCType.ORBIT] = self.iodssr

        if self.monlevel > 0:
            week, tow = time2gpst(self.time)
            self.fh.write("{:.1f} {:3s} {:3d}\n"
                          .format(tow, sat2id(sat), iodn))

        return i

    def decode_cssr(self, msg, i=0, src=0, prn=0):
        """ decode SBAS/DFMC SBAS messages """

        if src == 0:  # L1 SBAS
            _, mt = bs.unpack_from('u8u6', msg, i)
            i += 14
        else:  # L5 DFMC SBAS
            _, mt = bs.unpack_from('u4u6', msg, i)
            i += 10

        self.msgtype = mt
        # self.src = src
        if prn > 0:
            self.sat_ref = prn2sat(uGNSS.SBS, prn)

        if mt == 1:  # PRN mask
            i = self.decode_sbas_mask(msg, i)
        elif mt in (2, 3, 4, 5):  # fast correction
            i = self.decode_sbas_fast_corr(msg, i)
        elif mt == 6:  # integrity
            i = self.decode_sbas_integrity(msg, i)
        elif mt == 7:  # fast correction degradation factor
            i = self.decode_sbas_fast_degradation(msg, i)
        elif mt == 9:  # ranging
            i = self.decode_sbas_ranging(msg, i)
        elif mt == 10:  # degradation parameter
            i = self.decode_sbas_degradation_param(msg, i)
        elif mt == 12:  # UTC
            i = self.decode_sbas_utc(msg, i)
        elif mt == 17:  # GEO almanac
            i = self.decode_sbas_almanac(msg, i)
        elif mt == 18:  # IGP mask
            i = self.decode_sbas_igpmask(msg, i)
        elif mt == 24:  # mixed fast/long correction
            i = self.decode_sbas_mixed_corr(msg, i)
        elif mt == 25:  # long-term correction
            i = self.decode_sbas_long_corr(msg, i)
        elif mt == 26:  # iono-delay message
            i = self.decode_sbas_iono(msg, i)
        elif mt == 27:  # SBAS service
            i = self.decode_sbas_service(msg, i)
        elif mt == 28:  # clock ephemeris covariance matrix
            i = self.decode_sbas_cov(msg, i)

        # L5 SBAS
        elif mt == 31:
            i = self.decode_dfmc_mask(msg, i)
        elif mt == 32:
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)
        elif mt == 34:
            i = self.decode_dfmc_integrity_mt34(msg, i)
        elif mt == 35:
            i = self.decode_dfmc_integrity_mt35(msg, i)
        elif mt == 36:
            i = self.decode_dfmc_integrity_mt36(msg, i)
        elif mt == 37:  # OBAD (Old But Acive Data) parameters
            i = self.decode_dfmc_obad(msg, i)
        elif mt == 39:  # SBAS ephemeris
            i = self.decode_dfmc_eph_mt39(msg, i)
        elif mt == 40:  # SBAS ephemeris
            i = self.decode_dfmc_eph_mt40(msg, i)
        elif mt == 42:  # GNSS Time offsets
            None
        elif mt == 47:  # SBAS almanacs
            i = self.decode_dfmc_alm(msg, i)
        elif mt == 62:  # internal test message
            None
        elif mt == 63:  # Null
            None

        if self.monlevel > 3:
            if self.time != -1:
                _, tow = time2gpst(self.time)
            else:
                tow = -1
            self.fh.write("mt={:2d} tow={:6.1f}\n"
                          .format(mt, tow))
