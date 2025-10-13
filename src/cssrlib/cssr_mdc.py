"""
QZSS MADOCA-PPP correction data decoder

[1] Quasi-Zenith Satellite System Interface Specification Multi-GNSS
    Advanced Orbit and Clock Augmentation - Precise Point Positioning
    (IS-QZSS-MDC-004), 2025

"""

import numpy as np
import bitstruct as bs
from cssrlib.cssrlib import cssr, sCSSRTYPE, sCSSR, local_corr, sCType
from cssrlib.gnss import gpst2time, time2str, uGNSS, prn2sat, rCST, sat2id


class areaInfo():
    def __init__(self, sid, latr, lonr, p1=0, p2=0):
        self.sid = sid
        self.latr = latr
        self.lonr = lonr
        if sid == 0:
            self.lats = p1
            self.lons = p2
        elif sid == 1:
            self.rng = p1


class ionoCorr():
    def __init__(self):
        self.t0 = None
        self.qi = []
        self.iodssr = 0
        self.ct = 0
        self.sat = []
        self.c = np.zeros((6))


class cssr_mdc(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)

        self.cssrmode = sCSSRTYPE.QZS_MADOCA
        self.buff = bytearray(250*10)

        self.pnt = {}
        self.ci = {}

        self.area = -1
        self.reg = -1
        self.alrt = 0
        self.narea_t = {1: 8, 2: 16, 3: 5, 4: 1, 5: 8}
        self.MAXNET = np.sum(list(self.narea_t.values()))

        for inet in range(self.MAXNET+1):
            self.lc.append(local_corr())
            self.lc[inet].inet = inet
            self.lc[inet].flg_trop = 0
            self.lc[inet].flg_stec = 0
            self.lc[inet].nsat_n = 0
            self.lc[inet].t0 = {}

    def find_grid_index(self, pos):
        """ find region/area   """

        latd, lond = np.rad2deg(pos[0:2])
        self.reg = self.area = -1

        for reg in self.pnt.keys():
            for area in self.pnt[reg].keys():
                p = self.pnt[reg][area]
                if p.sid == 0:  # rectangle shape:
                    if p.latr-p.lats <= latd and latd <= p.latr+p.lats and \
                            p.lonr-p.lons <= lond and lond <= p.lonr+p.lons:
                        self.reg = reg
                        self.area = area
                        break
                elif p.sid == 1:  # circle
                    dn = (latd - p.latr)*rCST.D2R*rCST.RE_WGS84
                    de = (lond - p.lonr)*rCST.D2R * \
                        rCST.RE_WGS84*np.cos(p.lonr*rCST.D2R)
                    r = np.sqrt(dn**2+de**2)
                    if r <= p.rng:
                        self.reg = reg
                        self.area = area
                        break

        inet = self.get_inet(self.reg, self.area)
        return inet

    def get_dpos(self, pos):
        """ calculate position offset from reference """
        if self.reg < 0 or self.area < 0:
            print("get_dpos: region, area not defined.")
            return 0, 0

        latd, lond = np.rad2deg(pos[0:2])
        p = self.pnt[self.reg][self.area]
        dlat = latd - p.latr
        dlon = lond - p.lonr

        return dlat, dlon

    def get_stec(self, dlat=0.0, dlon=0.0):
        """ calculate STEC correction by interporation """
        if self.inet < 0:
            print("get_stec: region, area not defined.")
            return np.zeros(self.nsat_n)

        p = self.lc[self.inet]
        # if p.inet_ref != self.iodssr:
        #    return 0.0
        stec = np.zeros(p.nsat_n)
        v = np.array([1, dlat, dlon, dlat*dlon, dlat**2, dlon**2])

        for i, sat in enumerate(p.sat_n):
            stec[i] = v@p.ci[sat]

        return stec

    def decode_mdc_stec_area(self, buff, i=0):
        """ decoder for MT1 - STEC Coverage Message """
        tow, uid, mi, iodssr = bs.unpack_from('u20u4u1u4', buff, i)
        i += 29
        self.tow0 = tow//3600*3600
        reg, alrt, len_, narea = bs.unpack_from('u8u1u16u5', buff, i)
        i += 30
        self.reg = reg
        self.alrt = alrt

        if reg not in self.pnt:
            self.pnt[reg] = {}

        for k in range(narea):
            area, sid = bs.unpack_from('u5u1', buff, i)
            i += 6

            if sid == 0:  # rectangle shape
                latr, lonr, lats, lons = bs.unpack_from('s11u12u8u8', buff, i)
                if self.monlevel >= 2:
                    print(f"{reg} {area:2d} {sid} {latr*0.1:5.1f} "
                          f"{lonr*0.1:5.1f} {lats*0.1:3.1f} {lons*0.1:3.1f}")
                self.pnt[reg][area] = areaInfo(
                    sid, latr*0.1, lonr*0.1, lats*0.1, lons*0.1)
            else:  # circle range
                latr, lonr, rng = bs.unpack_from('s15u16u8', buff, i)
                if self.monlevel >= 2:
                    print(f"{reg} {area:2d} {sid} {latr*0.01:6.2f} "
                          f"{lonr*0.01:6.2f} {rng*10}")
                self.pnt[reg][area] = areaInfo(
                    sid, latr*0.01, lonr*0.01, rng*10.0)
            i += 39
        return i

    def get_inet(self, reg, area):
        """ region, area to inet conversion """

        if reg < 0 or area < 0:
            return -1

        inet = 0
        for r in self.narea_t.keys():
            if r >= reg:
                break
            inet += self.narea_t[r]

        inet += area

        return inet

    def decode_mdc_stec_corr(self, buff, i=0):
        """ decoder for MT2 - STEC Correction Message """
        dtow, uid, mi, iodssr = bs.unpack_from('u12u4u1u4', buff, i)
        i += 21
        reg, area, stype_ = bs.unpack_from('u8u5u2', buff, i)
        i += 15

        self.reg = reg
        self.area = area

        nsat = bs.unpack_from('u5u5u5u5u5', buff, i)
        i += 25
        # gps, glo, gal, bds, qzss
        sys_t = [uGNSS.GPS, uGNSS.GLO, uGNSS.GAL, uGNSS.BDS, uGNSS.QZS]

        inet = self.get_inet(reg, area)

        self.lc[inet].nsat_n = np.sum(nsat)
        ci = {}
        qi = {}
        stype = {}
        sat_ = []
        j = 0
        for gnss in range(5):
            sys = sys_t[gnss]
            for k in range(nsat[gnss]):
                c01 = c10 = c11 = c02 = c20 = 0.0

                prn, qi_, c00 = bs.unpack_from('u6u6s14', buff, i)
                i += 26

                if sys == uGNSS.QZS:
                    prn += 192
                sat = prn2sat(sys, prn)
                qi[sat] = qi_
                stype[sat] = stype_

                if stype_ > 0:
                    c01, c10 = bs.unpack_from('s12s12', buff, i)
                    i += 24
                if stype_ > 1:
                    c11 = bs.unpack_from('s10', buff, i)[0]
                    i += 10
                if stype_ > 2:
                    c02, c20 = bs.unpack_from('s8s8', buff, i)
                    i += 16

                sat_ += [sat]
                ci[sat] = np.array(
                    [c00*0.05, c01*0.02, c10*0.02, c11*0.02,
                     c02*5e-3, c20*5e-3])
                j += 1

        self.lc[inet].stype = stype
        self.lc[inet].sat_n = sat_
        self.lc[inet].ci = ci
        self.lc[inet].stec_quality = qi

        t0 = gpst2time(self.week, self.tow0+dtow)
        self.set_t0(inet, 0, sCType.STEC, t0)
        self.lc[inet].cstat |= (1 << sCType.STEC)

        self.lc[inet].inet_ref = iodssr

        return i

    def out_log(self):
        # if self.msgtype not in (1, 2):
        #    return super(cssr_mdc, self).out_log()

        sz_t = [1, 3, 4, 6]

        if self.time == -1:
            return

        self.fh.write("{:4d}\t{:s}\n".format(self.msgtype,
                                             time2str(self.time)))

        if self.msgtype == 1:

            self.fh.write(f"Reg\tArea\tsid\t"
                          f"latr\tlonr\tlats\tlons\n")

            for area in self.pnt[self.reg].keys():
                p = self.pnt[self.reg][area]
                self.fh.write(f"{self.reg}\t{area:2d}\t{p.sid}\t"
                              f"{p.latr:3.1f}\t{p.lonr:4.1f}\t")

                if p.sid == 0:
                    self.fh.write(f"{p.lats:3.1f}\t{p.lons:3.1f}\n")
                else:
                    self.fh.write(f"{p.rng:3.1f}\n")

        elif self.msgtype == 2:
            inet = self.get_inet(self.reg, self.area)

            self.fh.write(f"Reg:{self.reg}\tArea:{self.area:2d}\n")
            self.fh.write("Sat,stype,c00,c01,c10,c11,c20,c02\n")

            for sat in self.lc[inet].sat_n:
                stype = self.lc[inet].stype[sat]
                ci = self.lc[inet].ci[sat]
                self.fh.write(f"{sat2id(sat)}\t{stype}")
                for k in range(sz_t[stype]):
                    self.fh.write(f"\t{ci[k]:6.2f}")
                self.fh.write("\n")
        self.fh.flush()

    def decode_cssr(self, msg, i=0):
        """decode Compact SSR message with MADOCA-PPP extension """
        df = {'msgtype': 4073}
        while df['msgtype'] in [1, 2, 4073]:
            df = bs.unpack_from_dict('u12u4', ['msgtype', 'subtype'], msg, i)
            i += 16
            if df['msgtype'] not in [1, 2, 4073]:
                return -1

            self.msgtype = df['msgtype']
            self.subtype = df['subtype']
            if df['msgtype'] == 4073:  # Compact SSR
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
            elif df['msgtype'] in [1, 2] and df['subtype'] == 0:
                if df['msgtype'] == 1:  # MADOCA MT1
                    i = self.decode_mdc_stec_area(msg, i)
                elif df['msgtype'] == 2:  # MADOCA MT2
                    i = self.decode_mdc_stec_corr(msg, i)

            if i <= 0:
                return 0
            if self.monlevel >= 2:
                print(f"tow={int(self.tow):6d} msgtype={df['msgtype']:4d} "
                      f"subtype={self.subtype:2d} inet={self.inet:2d}")

            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
