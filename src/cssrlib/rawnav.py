"""
Raw Navigation Message decoder

[1-1] NAVSTAR GPS Space Segment/Navigation User Segment Interfaces
    (IS-GPS-200) Rev.N, 2022
[1-2] NAVSTAR GPS Space Segment/Navigation User Segment L5 Interfaces
    (IS-GPS-705) Rev.J, 2022
[1-3] NAVSTAR GPS Space Segment/Navigation User Segment L1C Interfaces
    (IS-GPS-800) Rev.J, 2022
[2] Quasi-Zenith Satellite System Interface Specification
    Satellite Positioning, Navigation and Timing Service (IS-QZSS-PNT-005),
    2022
[3] Galileo Open Service Signal-in-Space Interface Control Document
    (OS SIS ICD) Issue 2.1, 2023
[4] BeiDou Navigation Satellite System Signal In Space Interface Control
    Document: Open Service Signal B1C (Version 1.0), 2017
[5] GLONASS Interface Control Document (Edition 5.1), 2008
[6] Indian Regional NAvigation Satellite System Signal in Space ICD
    for Standard Positioning Service (Version 1.1), 2017

"""

import numpy as np
import bitstruct.c as bs
from cssrlib.gnss import gpst2time, gst2time, bdt2time, rCST
from cssrlib.gnss import prn2sat, uGNSS, sat2prn, bdt2gpst, utc2gpst
from cssrlib.gnss import time2gpst, gpst2utc
from cssrlib.gnss import Eph, Geph, uTYP
from cssrlib.rinex import rnxenc, rSigRnx


class RawNav():
    def __init__(self, opt=None, prefix=''):
        self.gps_lnav = {}
        for k in range(uGNSS.GPSMAX):
            self.gps_lnav[k] = bytearray(200)

        self.qzs_lnav = {}
        for k in range(uGNSS.QZSMAX):
            self.qzs_lnav[k] = bytearray(200)

        self.gps_cnav = {}
        for k in range(uGNSS.GPSMAX):
            self.gps_cnav[k] = bytearray(200)

        self.qzs_cnav = {}
        for k in range(uGNSS.QZSMAX):
            self.qzs_cnav[k] = bytearray(200)

        self.gal_inav = {}
        for k in range(uGNSS.GALMAX):
            self.gal_inav[k] = bytearray(200)

        self.gal_fnav = {}
        for k in range(uGNSS.GALMAX):
            self.gal_fnav[k] = bytearray(200)

        self.bds_d12 = {}
        for k in range(uGNSS.BDSMAX):
            self.bds_d12[k] = bytearray(200)

        self.bds_cnv2 = {}
        for k in range(uGNSS.BDSMAX):
            self.bds_cnv2[k] = bytearray(200)

        self.bds_cnv3 = {}
        for k in range(uGNSS.BDSMAX):
            self.bds_cnv3[k] = bytearray(200)

        self.glo_ca = {}
        for k in range(uGNSS.GLOMAX):
            self.glo_ca[k] = bytearray(200)

        self.irn_nav = {}
        for k in range(uGNSS.IRNMAX):
            self.irn_nav[k] = bytearray(200)

    def u2s(self, u, blen):
        if u & (1 << (blen-1)):
            u = u-(1 << blen)
        return u

    def getbitu2(self, buff, p1, l1, p2, l2):
        return (bs.unpack_from('u'+str(l1), buff, p1)[0] << l2) + \
            bs.unpack_from('u'+str(l2), buff, p2)[0]

    def getbits2(self, buff, p1, l1, p2, l2):
        s = bs.unpack_from('u1', buff, p1)[0]
        if s:
            return (bs.unpack_from('s'+str(l1), buff, p1)[0] << l2) + \
                bs.unpack_from('u'+str(l2), buff, p2)[0]
        else:
            return self.getbitu2(buff, p1, l1, p2, l2)

    def getbitu3(self, buff, p1, l1, p2, l2, p3, l3):
        return (bs.unpack_from('u'+str(l1), buff, p1)[0] << (l2+l3)) + \
            (bs.unpack_from('u'+str(l2), buff, p2)[0] << l3) + \
            bs.unpack_from('u'+str(l3), buff, p3)[0]

    def getbits3(self, buff, p1, l1, p2, l2, p3, l3):
        s = bs.unpack_from('u1', buff, p1)[0]
        if s:
            return (bs.unpack_from('s'+str(l1), buff, p1)[0] << (l2+l3)) + \
                (bs.unpack_from('u'+str(l2), buff, p2)[0] << l3) + \
                bs.unpack_from('u'+str(l3), buff, p3)[0]
        else:
            return self.getbitu3(buff, p1, l1, p2, l2, p3, l3)

    def getbitg(self, buff, pos, len_):
        s, v = bs.unpack_from('u1u'+str(len_-1), buff, pos)
        return -v if s else v

    def urai2sva(self, urai, sys=uGNSS.GPS):
        """ GPS/QZSS SV accuracy [1] 20.3.3.3.1.3, [2] Tab. 5.4.3-2 """
        urai_t = [2.40, 3.40, 4.85, 6.85, 9.65, 13.65, 24.00, 48.00, 96.00,
                  192.00, 384.00, 768.00, 1536.00, 3072.00, 6144.00, -1.0]

        return urai_t[urai]

    def sisa2sva(self, sisa):
        """ Galileo SISA Index Values [3] Tab. 89 """
        sva = np.nan
        if sisa < 50:
            sva = sisa*0.01
        elif sisa < 75:
            sva = sisa*0.02-0.5
        elif sisa < 100:
            sva = sisa*0.04-2.0
        elif sisa < 126:
            sva = sisa*0.16-14.0
        elif sisa == 255:  # NAPA
            sva = -1.0

        return sva

    def decode_gal_inav(self, week, time, sat, type_, msg):
        """ Galileo I/NAV message decoder [3] """
        sys, prn = sat2prn(sat)
        eph = Eph(sat)
        # 1:even/odd, 1:page, 112:data1, 6:tail, 1:even/odd, 1:page, 16:page2
        # 64:field1, 24:crc, 8:field2, 6:tail
        # for E1B: field1: reserved1(40) + SAR(22)+spare(2), field2=SSP
        # for E5b: field1: reserved1. field2 = reserved2
        msg = bytes(msg)
        odd, page = bs.unpack_from('u1u1', msg, 0)
        sid, iodnav = bs.unpack_from('u6u10', msg, 2)

        if sid > 5:
            return None

        buff = self.gal_inav[prn-1]
        i = 2
        for k in range(14):
            buff[sid*16+k] = bs.unpack_from('u8', msg, i)[0]
            i += 8
        i += 8
        for k in range(2):
            buff[sid*16+14+k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

        buff = bytes(buff)
        sid1, iodnav1 = bs.unpack_from('u6u10', buff, 128)
        sid2, iodnav2 = bs.unpack_from('u6u10', buff, 128*2)
        sid3, iodnav3 = bs.unpack_from('u6u10', buff, 128*3)
        sid4, iodnav4 = bs.unpack_from('u6u10', buff, 128*4)
        sid5 = bs.unpack_from('u6', buff, 128*5)[0]

        if sid1 != 1 or sid2 != 2 or sid3 != 3 or sid4 != 4 or sid5 != 5:
            return None
        if iodnav1 != iodnav2 or iodnav1 != iodnav3 or iodnav1 != iodnav4:
            return None

        week_gst, tow = bs.unpack_from('u12u20', buff, 128*5+73)
        eph.week = week_gst + 1024

        toe, M0, e, sqrtA = bs.unpack_from('u14s32u32u32', buff, 128+16)
        OMG0, i0, omg, idot = bs.unpack_from('s32s32s32s14', buff, 128*2+16)
        OMGd, deln, cuc, cus, crc, crs, sisa = bs.unpack_from(
            's24s16s16s16s16s16u8', buff, 128*3+16)
        svid, cic, cis, toc, af0, af1, af2 = bs.unpack_from(
            'u6s16s16u14s31s21s6', buff, 128*4+16)
        ai0, ai1, ai2, reg, bgda, bgdb, e5b_hs, e1b_hs, e5b_dvs, e1b_dvs \
            = bs.unpack_from('s11s11s14u5s10s10u2u2u1u1', buff, 128*5+6)

        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.e = e*rCST.P2_33
        sqrtA *= rCST.P2_19
        eph.A = sqrtA**2
        eph.OMG0 = OMG0*rCST.P2_31*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_29
        eph.cus = cus*rCST.P2_29
        eph.crc = crc*rCST.P2_5
        eph.crs = crs*rCST.P2_5
        eph.cic = cic*rCST.P2_29
        eph.cis = cis*rCST.P2_29
        eph.toes = toe*60.0
        toc = toc*60.0
        eph.af0 = af0*rCST.P2_34
        eph.af1 = af1*rCST.P2_46
        eph.af2 = af2*rCST.P2_59
        eph.iode = iodnav1
        eph.iodc = iodnav1

        eph.sva = self.sisa2sva(sisa)

        if type_ == 0:
            eph.code = 1
        elif type_ == 1:
            eph.code = 2
        elif type_ == 2:
            eph.code = 4

        if type_ == 0 or type_ == 2:  # INAV E1B, E5B
            eph.code |= (1 << 9)  # toc/af0-2, SISA are for E5b, E1
        elif type_ == 1:  # FNAV E1B, R5A
            eph.code |= (1 << 8)  # toc/af0-2, SISA are for E5a, E1

        eph.toe = gst2time(week_gst, eph.toes)
        eph.toc = gst2time(week_gst, toc)
        eph.tot = gst2time(week_gst, tow)

        eph.tgd = bgda*rCST.P2_32    # TGD (E1, E5a)
        eph.tgd_b = bgdb*rCST.P2_32  # TGD (E1, E5b)

        # e5b_hs, e1b_hs, e5b_dvs, e1b_dvs
        eph.svh = (e5b_hs << 7) | (e5b_dvs << 6) | (e1b_hs << 1) | (e1b_dvs)

        eph.mode = 0
        return eph

    def decode_gal_fnav(self, week, time, sat, type_, msg):
        """ Galileo F/NAV message decoder [3] """
        sys, prn = sat2prn(sat)
        eph = Eph(sat)
        # 244bit => type(6) + svid(6) + iodnav(10) + body () + crc(24)+tail(6)
        msg = bytes(msg)
        sid, svid, iodnav = bs.unpack_from('u6u6u10', msg, 0)

        if sid > 4:
            return None

        buff = self.gal_fnav[prn-1]
        i = 0
        for k in range(31):  # copy 244bits
            buff[(sid-1)*31+k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

        buff = bytes(buff)
        sid1, svid1, iodnav1 = bs.unpack_from('u6u6u10', buff, 0)
        sid2, iodnav2 = bs.unpack_from('u6u10', buff, 248*1)
        sid3, iodnav3 = bs.unpack_from('u6u10', buff, 248*2)
        sid4, iodnav4 = bs.unpack_from('u6u10', buff, 248*3)

        if svid != svid1 or sid1 != 1 or sid2 != 2 or sid3 != 3 or sid4 != 4:
            return None
        if iodnav1 != iodnav2 or iodnav1 != iodnav3 or iodnav1 != iodnav4:
            return None

        # clock correction
        toc, af0, af1, af2 = bs.unpack_from('u14s31s21s6', buff, 22)
        # iono correction
        sisa, ai0, ai1, ai2, reg, bgda, e5a_hs, week_gst, tow, e5a_dvs \
            = bs.unpack_from('u8s11s11s14u5s10u2u12u20u1', buff, 94)
        M0, OMGd, e, sqrtA, OMG0, idot = bs.unpack_from('s32s24u32u32s32s14',
                                                        buff, 248+16)
        i0, omg, deln, cuc, cus, crc, crs, toe = bs.unpack_from(
            's32s32s16s16s16s16s16u14', buff, 248*2+16)

        cic, cis = bs.unpack_from('s16s16', buff, 248*3+16)

        eph.week = week_gst + 1024

        eph.M0 = M0*rCST.P2_31*rCST.SC2RAD
        eph.deln = deln*rCST.P2_43*rCST.SC2RAD
        eph.e = e*rCST.P2_33
        sqrtA *= rCST.P2_19
        eph.A = sqrtA**2
        eph.OMG0 = OMG0*rCST.P2_31*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
        eph.idot = idot*rCST.P2_43*rCST.SC2RAD
        eph.cuc = cuc*rCST.P2_29
        eph.cus = cus*rCST.P2_29
        eph.crc = crc*rCST.P2_5
        eph.crs = crs*rCST.P2_5
        eph.cic = cic*rCST.P2_29
        eph.cis = cis*rCST.P2_29
        eph.toes = toe*60.0
        toc = toc*60.0
        eph.af0 = af0*rCST.P2_34
        eph.af1 = af1*rCST.P2_46
        eph.af2 = af2*rCST.P2_59
        eph.iode = iodnav1
        eph.iodc = iodnav1

        eph.sva = self.sisa2sva(sisa)

        if type_ == 0:
            eph.code = 1
        elif type_ == 1:
            eph.code = 2
        elif type_ == 2:
            eph.code = 4

        if type_ == 0 or type_ == 2:  # INAV E1B, E5B
            eph.code |= (1 << 9)  # toc/af0-2, SISA are for E5b, E1
        elif type_ == 1:  # FNAV E1B, R5A
            eph.code |= (1 << 8)  # toc/af0-2, SISA are for E5a, E1

        eph.toe = gst2time(week_gst, eph.toes)
        eph.toc = gst2time(week_gst, toc)
        eph.tot = gst2time(week_gst, tow)

        eph.tgd = bgda*rCST.P2_32    # TGD (E1, E5a)
        eph.tgd_b = 0.0
        eph.svh = (e5a_hs << 4) | (e5a_dvs << 3)

        eph.mode = 1  # F/NAV
        return eph

    def decode_gps_lnav(self, week, time, sat, msg):
        """ GPS/QZSS LNAV Message decoder [1], [2] """

        sys, prn = sat2prn(sat)
        eph = Eph(sat)

        preamble = bs.unpack_from('u8', msg, 2)[0]
        if preamble != 0x8b:
            return None

        if sys == uGNSS.GPS:
            buff = self.gps_lnav[prn-1]
        elif sys == uGNSS.QZS:
            buff = self.qzs_lnav[prn-193]
        else:
            return None

        sid = bs.unpack_from('u3', msg, 53)[0]
        buff[(sid-1)*40:(sid-1)*40+40] = msg

        id1 = bs.unpack_from('u3', buff, 53)[0]
        id2 = bs.unpack_from('u3', buff, 320+53)[0]
        id3 = bs.unpack_from('u3', buff, 320*2+53)[0]

        if id1 == 1 and id2 == 2 and id3 == 3:
            tow, alert, asf = bs.unpack_from('u17u1u1', buff, 32+2)
            tow *= 6.0
            # subfram #1
            wn, code, urai, svh, iodc_ = bs.unpack_from('u10u2u4u6u2', buff,
                                                        32*2+2)
            l2p = bs.unpack_from('u1', buff, 32*3+2)[0]
            tgd = bs.unpack_from('s8', buff, 32*6+2+16)[0]
            iodc, toc = bs.unpack_from('u8u16', buff, 32*7+2)
            af2, af1 = bs.unpack_from('s8s16', buff, 32*8+2)
            af0 = bs.unpack_from('s22', buff, 32*9+2)[0]
            iodc |= (iodc_ << 8)

            # subframe #2
            i0 = 320
            iode, crs = bs.unpack_from('u8s16', buff, i0+32*2+2)
            deln, M0_ = bs.unpack_from('s16u8', buff, i0+32*3+2)
            M0 = bs.unpack_from('u24', buff, i0+32*4+2)[0]
            M0 = bs.unpack('s32', bs.pack('u8u24', M0_, M0))[0]
            cuc, e_ = bs.unpack_from('s16u8', buff, i0+32*5+2)
            e = bs.unpack_from('u24', buff, i0+32*6+2)[0]
            e = bs.unpack('s32', bs.pack('u8u24', e_, e))[0]
            cus, sa_ = bs.unpack_from('s16u8', buff, i0+32*7+2)
            sa = bs.unpack_from('u24', buff, i0+32*8+2)[0]
            sqrtA = bs.unpack('u32', bs.pack('u8u24', sa_, sa))[0]
            toe, fit, aodo = bs.unpack_from('u16u1u5', buff, i0+32*9+2)

            # subframe #3
            i0 = 320*2
            cic, OMG0_ = bs.unpack_from('s16u8', buff, i0+32*2+2)
            OMG0 = bs.unpack_from('u24', buff, i0+32*3+2)[0]
            OMG0 = bs.unpack('s32', bs.pack('u8u24', OMG0_, OMG0))[0]
            cis, inc0_ = bs.unpack_from('s16u8', buff, i0+32*4+2)
            inc0 = bs.unpack_from('u24', buff, i0+32*5+2)[0]
            inc0 = bs.unpack('s32', bs.pack('u8u24', inc0_, inc0))[0]
            crc, omg_ = bs.unpack_from('s16u8', buff, i0+32*6+2)
            omg = bs.unpack_from('u24', buff, i0+32*7+2)[0]
            omg = bs.unpack('s32', bs.pack('u8u24', omg_, omg))[0]
            OMGd = bs.unpack_from('s24', buff, i0+32*8+2)[0]
            iode, idot = bs.unpack_from('u8s14', buff, i0+32*9+2)

            eph.week = (week // 1024)*1024 + wn
            eph.l2p = l2p
            eph.code = code
            eph.svh = svh
            eph.sva = self.urai2sva(urai)
            eph.tgd = tgd*rCST.P2_31
            eph.iodc = iodc
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
            sqrtA *= rCST.P2_19
            eph.A = sqrtA**2
            eph.toes = toe*16.0
            eph.cic = cic*rCST.P2_29
            eph.OMG0 = OMG0*rCST.P2_31*rCST.SC2RAD
            eph.cis = cis*rCST.P2_29
            eph.i0 = inc0*rCST.P2_31*rCST.SC2RAD
            eph.crc = crc*rCST.P2_5
            eph.omg = omg*rCST.P2_31*rCST.SC2RAD
            eph.OMGd = OMGd*rCST.P2_43*rCST.SC2RAD
            eph.idot = idot*rCST.P2_43*rCST.SC2RAD
            eph.mode = 0  # LNAV
            eph.toc = gpst2time(eph.week, toc)
            eph.toe = gpst2time(eph.week, eph.toes)
            eph.tot = bdt2time(eph.week, tow)
        else:
            return None

        return eph

    def decode_gps_cnav_gd(self, msg, eph, i):
        """ GPS/QZSS CNAV group delay """
        isc1, isc2, isc3, isc4 = \
            bs.unpack_from('s13s13s13s13', msg, i)
        i += 52
        eph.isc[0:4] = np.array([isc1, isc2, isc3, isc4])*rCST.P2_35
        return i

    def decode_gps_cnav_iono(self, msg, i):
        """ GPS/QZSS CNAV iono delay """
        a0, a1, a2, a3, b0, b1, b2, b3 = \
            bs.unpack_from('s8s8s8s8s8s8s8s8', msg, i)
        i += 64
        alp = np.array([a0*rCST.P2_30, a1*rCST.P2_27,
                        a2*rCST.P2_24, a3*rCST.P2_24])
        bet = np.array([b0*(2**11), b1*(2**14), b2*(2**16), b3*(2**16)])
        return i

    def decode_gps_cnav_utc(self, msg, i):
        """ GPS/QZSS CNAV UTC parameters """
        a0, a1, a2, dtls, tot, wnot, wnlsf, dn, dtlsf = \
            bs.unpack_from('s16s13s7s8u16u13u13u4s8', msg, i)
        i += 98
        return i

    def decode_gps_cnav(self, week, time, sat, msg):
        """ GPS/QZSS CNAV Message decoder """

        sys, prn = sat2prn(sat)
        eph = Eph(sat)

        if sys == uGNSS.GPS:
            buff = self.gps_cnav[prn-1]
        elif sys == uGNSS.QZS:
            buff = self.qzs_cnav[prn-193]
        else:
            return None

        pre, prn_, sid = bs.unpack_from('u8u6u6', msg, 0)

        if sys == uGNSS.QZS:
            prn_ += 192

        if pre != 0x8b or prn != prn_:
            return None

        if sid not in (10, 11, 12, 15, 60, 61) and (sid < 30 or sid > 37):
            return None

        if sid in (10, 11):  # ephemeris
            buff[(sid-10)*38:(sid-10)*38+38] = msg
        elif (sid >= 30 and sid <= 37) or \
             (sys == uGNSS.QZS and sid == 61):  # clock
            buff[2*38:2*38+38] = msg
        elif sid == 12:  # QZSS reduced almanac
            None
        elif sid == 15:  # Text
            None

        id1 = bs.unpack_from('u6', buff, 14)[0]
        id2 = bs.unpack_from('u6', buff, 304+14)[0]
        id3 = bs.unpack_from('u6', buff, 304*2+14)[0]

        toe1 = bs.unpack_from('u11', buff, 70)[0]
        toe2 = bs.unpack_from('u11', buff, 304+38)[0]
        toc = bs.unpack_from('u11', buff, 304*2+60)[0]

        if id1 != 10 or id2 != 11 or id3 != 30:
            if sys == uGNSS.QZS and id3 == 61:
                None
            else:
                return None

        if toe1 != toe2 or toe1 != toc:
            return None

        i = 20
        tow, alert = bs.unpack_from('u17u1', buff, i)
        i += 18
        tow *= 6.0

        # type 10
        wn, svh, top, ura_ed = bs.unpack_from('u13u3u11s5', buff, i)
        i += 32
        toe, dA, Adot, deln, delnd, M0, e, omg, isf, esf = bs.unpack_from(
            'u11s26s25s17s23s33u33s33u1u1', buff, i)

        # type 11
        i = 304+38
        toe, OMG0, i0, OMGd, idot, cis, cic, crs, crc, cus, cuc = \
            bs.unpack_from('u11s33s33s17s15s16s16s24s24s21s21', buff, i)

        # type 3x or type 61 (QZS)
        i = 304*2+38
        top, ura0, ura1, ura2, toc, af0, af1, af2 = \
            bs.unpack_from('u11s5u3u3u11s26s20s10', buff, i)

        i = 304*2+127

        eph.isc = np.zeros(6)
        sid = id3

        if sid in (30, 61):  # clock, iono, group delay
            tgd = bs.unpack_from('s13', buff, i)[0]
            i += 13
            eph.tgd = tgd*rCST.P2_35
            i = self.decode_gps_cnav_gd(buff, eph, i)
            i = self.decode_gps_cnav_iono(buff, i)
            wn_op = bs.unpack_from('u8', buff, i)[0]
            i += 8

        elif sid == 31:  # clock, reduced almanac
            wn_alm, toa = bs.unpack_from('u13u8', buff, i)
            i += 21
            for k in range(4):
                # TBD
                i += 31
        elif sid == 32:  # clock, EOP
            None
            # TBD
        elif sid == 33:  # clock, UTC
            None
            # TBD
        elif sid == 35:  # clock, QZSS/GNSS time offset
            None
            # TBD
        elif sid == 37:  # clock, midi almanac
            None
            # TBD

        eph.week = wn
        eph.code = 1
        eph.svh = svh

        # clock
        toc *= 300.0
        eph.af2 = af2*rCST.P2_60
        eph.af1 = af1*rCST.P2_48
        eph.af0 = af0*rCST.P2_35
        eph.urai = np.array([ura0, ura1, ura2, ura_ed], dtype=int)
        # eph.sva = self.urai2sva(urai)

        # ephemeris
        eph.tops = top*300.0
        A0 = 26559710.0 if sys == uGNSS.GPS else 42164200.0
        eph.A = A0 + dA*rCST.P2_9
        eph.Adot = Adot*rCST.P2_21
        eph.deln = deln*rCST.P2_44*rCST.SC2RAD
        eph.delnd = delnd*rCST.P2_57*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_32*rCST.SC2RAD
        eph.e = e*rCST.P2_34
        eph.omg = omg*rCST.P2_32*rCST.SC2RAD
        eph.integ = isf
        # eph.esf = esf

        eph.wn_op = wn//256*256 + wn_op

        eph.toes = toe*300.0
        eph.OMG0 = OMG0*rCST.P2_32*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_32*rCST.SC2RAD
        eph.OMGd = (OMGd*rCST.P2_44 - 2.6e-9)*rCST.SC2RAD
        eph.idot = idot*rCST.P2_44*rCST.SC2RAD

        eph.cis = cis*rCST.P2_30
        eph.cic = cic*rCST.P2_30
        eph.crs = crs*rCST.P2_8
        eph.crc = crc*rCST.P2_8
        eph.cus = cus*rCST.P2_30
        eph.cuc = cuc*rCST.P2_30

        eph.mode = 1  # CNAV
        eph.toc = gpst2time(eph.week, toc)
        eph.toe = gpst2time(eph.week, eph.toes)
        eph.tot = bdt2time(eph.week, tow)
        eph.top = gpst2time(eph.wn_op, eph.tops)

        return eph

    def decode_gps_cnav2(self, week, time, sat, msg):
        """ GPS/QZSS CNAV/2 Message decoder """

        sys, prn = sat2prn(sat)
        eph = Eph(sat)

        # subframe 1 (52 syms)
        toi = bs.unpack_from('u9', msg, 0)[0]

        # subframe 2 (600)
        i = 52
        wn, itow, top, svh, ura_ed, toe = \
            bs.unpack_from('u13u8u11u1s5u11', msg, i)
        i += 49

        dA, Adot, deln, delnd, M0, e, omg, OMG0, i0, OMGd, idot = \
            bs.unpack_from('s26s25s17s23s33u33s33s33s33s17s15', msg, i)
        i += 288

        cis, cic, crs, crc, cus, cuc = \
            bs.unpack_from('s16s16s24s24s21s21', msg, i)
        i += 122

        ura0, ura1, ura2, af0, af1, af2, tgd, isc1, isc2, isf, wn_op, esf = \
            bs.unpack_from('s5u3u3s26s20s10s13s13s13u1u8u1', msg, i)
        i += 116

        tow = itow*7200+toi*18

        # subframe 3 (274)
        i = 52+1200
        prn, page = bs.unpack_from('u8u6', msg, i)
        i += 14

        eph.isc = np.zeros(6)

        if page in (1, 61):  # UTC, iono
            i = self.decode_gps_cnav_utc(msg, i)
            i = self.decode_gps_cnav_iono(msg, i)
            i = self.decode_gps_cnav_gd(msg, eph, i)
        elif page == 2:  # GGTO, EOP
            None
        elif page == 3:  # Reduced almanac
            None
        elif page == 4:  # Midi almanac
            None
        elif page == 6:  # Text
            None

        if page not in (1, 61):
            return None

        eph.week = wn
        eph.code = 2
        eph.svh = svh

        # clock
        eph.af2 = af2*rCST.P2_60
        eph.af1 = af1*rCST.P2_48
        eph.af0 = af0*rCST.P2_35
        eph.urai = np.array([ura0, ura1, ura2, ura_ed], dtype=int)
        # eph.sva = self.urai2sva(urai)

        # group-delay
        eph.tgd = tgd*rCST.P2_35  # L1CA
        eph.isc[4] = isc2*rCST.P2_35  # L1CD
        eph.isc[5] = isc1*rCST.P2_35  # L1CP

        # ephemeris
        eph.tops = top*300.0
        A0 = 26559710.0 if sys == uGNSS.GPS else 42164200.0
        eph.A = A0 + dA*rCST.P2_9
        eph.Adot = Adot*rCST.P2_21
        eph.deln = deln*rCST.P2_44*rCST.SC2RAD
        eph.delnd = delnd*rCST.P2_57*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_32*rCST.SC2RAD
        eph.e = e*rCST.P2_34
        eph.omg = omg*rCST.P2_32*rCST.SC2RAD
        eph.integ = isf
        # eph.esf = esf

        eph.wn_op = wn//256*256 + wn_op

        eph.toes = toe*300.0
        eph.OMG0 = OMG0*rCST.P2_32*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_32*rCST.SC2RAD
        eph.OMGd = (OMGd*rCST.P2_44 - 2.6e-9)*rCST.SC2RAD
        eph.idot = idot*rCST.P2_44*rCST.SC2RAD

        eph.cis = cis*rCST.P2_30
        eph.cic = cic*rCST.P2_30
        eph.crs = crs*rCST.P2_8
        eph.crc = crc*rCST.P2_8
        eph.cus = cus*rCST.P2_30
        eph.cuc = cuc*rCST.P2_30

        eph.mode = 2  # CNAV/2
        eph.toe = gpst2time(eph.week, eph.toes)
        eph.toc = eph.toe
        eph.tot = bdt2time(eph.week, tow)
        eph.top = gpst2time(eph.wn_op, eph.tops)

        return eph

    def decode_bds_cnav_eph1(self, msg, eph, i):
        """ BDS CNAV message decoder for Ephemeris I """
        toe, sattype, dA, Adot, deln, delnd, M0, ecc, omg = \
            bs.unpack_from('u11u2s26s25s17s23s33u33s33', msg, i)
        i += 203
        eph.toes = toe*300.0
        eph.sattype = sattype
        eph.Adot = Adot*rCST.P2_21
        eph.deln = deln*rCST.P2_44*rCST.SC2RAD
        eph.delnd = delnd*rCST.P2_57*rCST.SC2RAD
        eph.M0 = M0*rCST.P2_32*rCST.SC2RAD
        eph.e = ecc*rCST.P2_34
        eph.omg = omg*rCST.P2_32*rCST.SC2RAD

        Aref = 27906100.0 if eph.sattype == 3 else 42162200.0
        eph.A = Aref + dA*2**-9
        eph.toe = bdt2gpst(bdt2time(eph.week, eph.toes))
        return i

    def decode_bds_cnav_eph2(self, msg, eph, i):
        """ BDS CNAV message decoder for Ephemeris II """
        OMG0, i0, OMGd, idot, cis, cic, crs, crc, cus, cuc = \
            bs.unpack_from('s33s33s19s15s16s16s24s24s21s21', msg, i)
        i += 222
        eph.OMG0 = OMG0*rCST.P2_32*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_32*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_44*rCST.SC2RAD
        eph.idot = idot*rCST.P2_44*rCST.SC2RAD
        eph.cis = cis*rCST.P2_30
        eph.cic = cic*rCST.P2_30
        eph.crs = crs*rCST.P2_8
        eph.crc = crc*rCST.P2_8
        eph.cus = cus*rCST.P2_30
        eph.cuc = cuc*rCST.P2_30

        return i

    def decode_bds_cnav_clk(self, msg, eph, i):
        """ BDS CNAV message decoder for Clock """
        toc, a0, a1, a2 = \
            bs.unpack_from('u11s25s22s11', msg, i)
        i += 69
        tocs = toc*300.0
        eph.af0 = a0*rCST.P2_34
        eph.af1 = a1*rCST.P2_50
        eph.af2 = a2*rCST.P2_66
        eph.toc = bdt2gpst(bdt2time(eph.week, tocs))
        return i

    def decode_bds_cnav_iono(self, msg, i):
        """ BDS CNAV message decoder for Ionoephere parameters """
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = \
            bs.unpack_from('u10s8u8u8u8s8s8s8s8', msg, i)
        i += 74
        iono = np.array([a1, a2, a3, a4, -a5, a6, a7, a8, a9])*0.125
        return i

    def decode_bds_cnav_utc(self, msg, i):
        """ BDS CNAV message decoder for UTC parameters """
        a0, a1, a2, dtls, tot, wn_ot, wn_lsf, dn, dt_lsf = \
            bs.unpack_from('s16s13s7s8u16u13u13u3s8', msg, i)
        i += 97
        return i

    def decode_bds_cnav_ralm(self, msg, i):
        """ BDS CNAV message decoder for reduced almanacs """
        prn, sattype, dela, OMG0, Phi0, svh = \
            bs.unpack_from('u6u2s8s7s7u8', msg, i)
        i += 38
        return i

    def decode_bds_cnav_eop(self, msg, i):
        """ BDS CNAV message decoder for Earth Orientation Parameters """
        teop, x, xd, y, yd, dut1, dut1d = \
            bs.unpack_from('u16s21s15s21s15s31s19', msg, i)
        i += 138
        return i

    def decode_bds_cnav_ggto(self, msg, i):
        """ BDS CNAV message decoder for GGTO """
        gnss, wn, t0, a0, a1, a2 = \
            bs.unpack_from('u3u13u16s16s13s7', msg, i)
        i += 38
        return i

    def decode_bds_cnav_malm(self, msg, i):
        """ BDS CNAV message decoder for Midium Almanacs """
        prn, sattype, wn, toa, ecc, inc, sqrta, OMG0, OMGd, omg, M0, \
            af0, af1, svh = bs.unpack_from(
                'u6u2u13u8u11s11u17s16s11s16s16s11s10u8', msg, i)
        i += 156
        return i

    def decode_bds_cnav_sisai(self, msg, eph, i):
        """ BDS CNAV message decoder for SISAI """
        top, eph.sisai[1], eph.sisai[2], eph.sisai[3] = bs.unpack_from(
            'u11s5s3s3', msg, i)
        eph.tops = top*300.0
        i += 22
        return i

    def decode_bds_b1c(self, week, time_, prn, msg):
        """ BDS B1C (B-CNAV1 message decoder) [4] """

        eph = Eph()
        eph.sat = prn2sat(uGNSS.BDS, prn)
        eph.sisai = np.zeros(4, dtype=int)
        eph.isc = np.zeros(6)
        # data2: 600b, errCorr2: 8b, data3: 264b, soh: 8b
        # decode Subframe 1
        soh = bs.unpack_from('u8', msg, 600+8+264)[0]*18.0
        # decode Subframe 2
        i = 0
        eph.week, how, eph.iodc, eph.iode = bs.unpack_from(
            'u13u8u10u8', msg, i)
        i += 39
        tow = how*3600+soh
        eph.tot = bdt2time(eph.week, tow)

        i = self.decode_bds_cnav_eph1(msg, eph, i)
        i = self.decode_bds_cnav_eph2(msg, eph, i)
        i = self.decode_bds_cnav_clk(msg, eph, i)
        tgd_b2ap, isc_b1cd, tgd_b1cp = bs.unpack_from('s12s12s12', msg, i)
        eph.tgd = tgd_b1cp*rCST.P2_34
        eph.isc[0] = isc_b1cd*rCST.P2_34
        eph.tgd_b = tgd_b2ap*rCST.P2_34
        i += 36+7+24

        # decode Subframe 3
        i = 608
        page, eph.svh, eph.integ, eph.sismai = bs.unpack_from('u6u2u3u4',
                                                              msg, i)
        i += 15
        if page == 1:  # Iono, BDT-UTC
            eph.sisai[0] = bs.unpack_from('u5', msg, i)[0]
            i += 5
            i = self.decode_bds_cnav_sisai(msg, eph, i)
        elif page == 2:  # Reduced almanac
            i = self.decode_bds_cnav_sisai(msg, eph, i)
            i += 22
        elif page == 3:  # EOP, BGTO
            eph.sisai[0] = bs.unpack_from('u5', msg, i)[0]
            i += 5
        elif page == 4:  # Midi almanac
            i = self.decode_bds_cnav_sisai(msg, eph, i)
            i += 22

        # i += 264-15
        eph.mode = 1  # B-CNAV1

        return eph

    def decode_bds_b2a(self, week, time_, sat, msg):
        """ BDS B2a (B-CNAV2 message decoder) [4] """

        sys, prn = sat2prn(sat)
        sid, sow = bs.unpack_from('u6u18', msg, 6)

        if sid not in (10, 11, 30, 34, 40):
            return None

        msg_t = {10: 0, 11: 1, 30: 2, 34: 3, 40: 4}
        mid = msg_t[sid]
        buff = self.bds_cnv2[prn-1]
        buff[mid*40:mid*40+40] = msg

        id1, sow1 = bs.unpack_from('u6u18', buff, 6)
        id2, sow2 = bs.unpack_from('u6u18', buff, 320+6)
        id3, sow3 = bs.unpack_from('u6u18', buff, 320*2+6)
        id4, sow4 = bs.unpack_from('u6u18', buff, 320*3+6)
        id5, sow5 = bs.unpack_from('u6u18', buff, 320*4+6)

        if id1 != 10 or id2 != 11 or id3 != 30 or id4 != 34 or id5 != 40:
            return None

        if sow2 != sow1+1 or sow3 != sow2+1:
            return None

        sow1 *= 3.0
        eph = Eph(sat)
        eph.sisai = np.zeros(4, dtype=int)
        eph.isc = np.zeros(6)

        if id4 == 34:
            i = 320*3+42
            i = self.decode_bds_cnav_sisai(buff, eph, i)

        if id5 == 40:
            i = 320*4+42
            eph.sisai[0] = bs.unpack_from('u5', buff, i)[0]

        # decode MT10
        i = 30
        eph.week, ib2a, eph.sismai, ib1c, eph.iode = bs.unpack_from(
            'u13u3u4u3u8', buff, i)
        i += 31
        i = self.decode_bds_cnav_eph1(buff, eph, i)
        eph.integ = (ib2a << 3)+ib1c

        # decode MT11
        i = 320+30
        hs = bs.unpack_from('u2', buff, i)[0]
        i += 12
        i = self.decode_bds_cnav_eph2(buff, eph, i)

        # decode MT30
        i = 320*2+42
        i = self.decode_bds_cnav_clk(buff, eph, i)

        eph.iodc, tgd_b2ap, isc_b2ad = bs.unpack_from('u10s12s12', buff, i)
        i += 34
        i = self.decode_bds_cnav_iono(buff, i)
        tgd_b1cp = bs.unpack_from('u12', buff, i)[0]

        eph.tgd = tgd_b1cp*rCST.P2_34
        eph.isc[1] = isc_b2ad*rCST.P2_34
        eph.tgd_b = tgd_b2ap*rCST.P2_34

        eph.svh = hs
        eph.tot = bdt2time(eph.week, sow1)
        eph.mode = 2  # B-CNAV2

        return eph

    def decode_bds_b2b(self, week, time_, sat, msg):
        """ BDS B2b (B-CNAV3 message decoder) [4] """

        sys, prn = sat2prn(sat)
        sid, sow = bs.unpack_from('u6u20', msg, 12)

        if sid not in (10, 30, 40):
            return None

        msg_t = {10: 0, 30: 1, 40: 2}
        mid = msg_t[sid]
        buff = self.bds_cnv3[prn-1]
        buff[mid*64:mid*64+64] = msg

        id1, sow1 = bs.unpack_from('u6u20', buff, 12)
        id2, sow2 = bs.unpack_from('u6u20', buff, 512+12)
        id3, sow3 = bs.unpack_from('u6u20', buff, 512*2+12)

        if id1 != 10 or id2 != 30:
            return None

        if sow2 != sow1+1:
            return None

        eph = Eph(sat)
        eph.sisai = np.zeros(4, dtype=int)
        eph.isc = np.zeros(6)

        # decode MT10
        i = 12+30
        i = self.decode_bds_cnav_eph1(buff, eph, i)
        i = self.decode_bds_cnav_eph2(buff, eph, i)

        eph.integ, eph.sismai = bs.unpack_from('u3u4', buff, i)
        i += 7

        # decode MT30
        i = 512+12+26
        eph.week = bs.unpack_from('u13', buff, i)[0]
        i += 13+4
        i = self.decode_bds_cnav_clk(buff, eph, i)
        eph.tgd = bs.unpack_from('s12', buff, i)[0]*rCST.P2_34  # tgd-B2bI
        i += 12
        i = self.decode_bds_cnav_iono(buff, i)
        i = self.decode_bds_cnav_utc(buff, i)
        i = self.decode_bds_cnav_eop(buff, i)
        i = self.decode_bds_cnav_sisai(buff, eph, i)
        eph.sisai[0], eph.svh = bs.unpack_from('u5u2', buff, i)
        i += 7

        eph.tot = bdt2time(eph.week, sow1)
        eph.mode = 3  # B-CNAV3

        return eph

    def decode_bds_d1(self, week, time, sat, msg):
        """ BDS D1 Message decoder """

        sys, prn = sat2prn(sat)

        i = 0
        pre, _, sid, sow1 = bs.unpack_from('u11u4u3u8', msg, i)
        i += 30
        if pre != 0x712:
            return None

        buff = self.bds_d12[prn-1]
        buff[(sid-1)*40:(sid-1)*40+40] = msg

        id1 = bs.unpack_from('u3', buff, 15)[0]
        id2 = bs.unpack_from('u3', buff, 320+15)[0]
        id3 = bs.unpack_from('u3', buff, 320*2+15)[0]

        sow1 = self.getbitu2(buff, 18, 8, 30, 12)
        sow2 = self.getbitu2(buff, 320+18, 8, 320+30, 12)
        sow3 = self.getbitu2(buff, 320*2+18, 8, 320*2+30, 12)

        if id1 != 1 or id2 != 2 or id3 != 3:
            return None

        if sow2 != sow1+6 or sow3 != sow2+6:
            return None

        eph = Eph(sat)

        # subframe 1
        i = 0
        eph.svh, eph.iodc, urai = bs.unpack_from('u1u5u4', buff, i+42)
        eph.week = bs.unpack_from('u13', buff, i+60)[0]
        toc = self.getbitu2(buff, i+73, 9, i+90, 8)*8.0
        eph.tgd = bs.unpack_from('s10', buff, i+98)[0]*1e-10
        eph.tgd_b = self.getbits2(buff, i+108, 4, i+120, 6)*1e-10
        eph.af2 = bs.unpack_from('s11', buff, i+214)[0]*rCST.P2_66
        eph.af0 = self.getbits2(buff, i+225, 7, i+240, 17)*rCST.P2_33
        eph.af1 = self.getbits2(buff, i+257, 5, i+270, 17)*rCST.P2_50
        eph.iode = bs.unpack_from('u5', buff, i+287)[0]

        # subframe 2
        i = 320
        eph.deln = self.getbits2(buff, i+42, 10, i+60, 6) * \
            rCST.P2_43*rCST.SC2RAD
        eph.cuc = self.getbits2(buff, i+66, 16, i+90, 2)*rCST.P2_31
        eph.M0 = self.getbits2(buff, i+92, 20, i+120, 12) * \
            rCST.P2_31*rCST.SC2RAD
        eph.e = self.getbitu2(buff, i+132, 10, i+150, 22)*rCST.P2_33
        eph.cus = bs.unpack_from('s18', buff, i+180)[0]*rCST.P2_31
        eph.crc = self.getbits2(buff, i+198, 4, i+210, 14)*rCST.P2_6
        eph.crs = self.getbits2(buff, i+224, 8, i+240, 10)*rCST.P2_6
        sqrtA = self.getbitu2(buff, i+250, 12, i+270, 20)*rCST.P2_19
        eph.A = sqrtA**2
        toe1 = bs.unpack_from('s2', buff, i+290)[0]

        # subframe 3
        i = 320*2
        toe2 = self.getbitu2(buff, i+42, 10, i+60, 5)
        eph.i0 = self.getbits2(buff, i+65, 17, i+90, 15)*rCST.P2_31*rCST.SC2RAD
        eph.cic = self.getbits2(buff, i+105, 7, i+120, 11)*rCST.P2_31
        eph.OMGd = self.getbits2(buff, i+131, 11, i+150, 13) * \
            rCST.P2_43*rCST.SC2RAD
        eph.cis = self.getbits2(buff, i+163, 9, i+180, 9)*rCST.P2_31
        eph.idot = self.getbits2(buff, i+189, 13, i+210, 1) * \
            rCST.P2_43*rCST.SC2RAD
        eph.OMG0 = self.getbits2(buff, i+211, 21, i+240, 11) * \
            rCST.P2_31*rCST.SC2RAD
        eph.omg = self.getbits2(buff, i+251, 11, i+270, 21) * \
            rCST.P2_31*rCST.SC2RAD

        eph.sva = self.urai2sva(urai)

        eph.toes = ((toe1 << 15)+toe2)*8.0
        eph.tot = bdt2gpst(bdt2time(eph.week, sow1))
        if eph.toes > sow1+302400.0:
            eph.week += 1
        elif eph.toes < sow1-302400.0:
            eph.week -= 1
        eph.toc = bdt2time(eph.week, toc)
        eph.toe = bdt2time(eph.week, eph.toes)

        eph.flag = 1  # IGSO/MEO
        eph.mode = 0
        return eph

    def decode_bds_d2(self, week, time, sat, msg):
        """ BDS D2 Message decoder """

        sys, prn = sat2prn(sat)

        pre, _, frame, sow1, _, sow2, page = bs.unpack_from(
            'u11u4u3u8u4u12u4', msg, 0)

        if pre != 0x712:
            return None

        if frame == 1 and (page >= 1 and page <= 10):
            buff = self.bds_d12[prn-1]
            buff[(page-1)*20:(page-1)*20+20] = msg
        else:
            return None

        sow_ = np.zeros(10, dtype=int)

        for k in range(10):
            if k == 1:
                continue
            page = bs.unpack_from('u4', buff, 160*k+42)[0]
            if page != k+1:
                return None
            sow_[k] = self.getbitu2(buff, 160*k+18, 8, 160*k+30, 12)
            if k == 2:
                sow_[k] != sow_[k-2]+6
            elif k > 0 and sow_[k] != sow_[k-1]+3:
                return None

        eph = Eph(sat)

        # page 1
        i = 0
        eph.svh, eph.iodc = bs.unpack_from('u1u5', buff, i+46)
        urai, eph.week = bs.unpack_from('u4u13', buff, i+60)
        toc = self.getbitu2(buff, i+77, 5, i+90, 12)*8.0
        eph.tgd = bs.unpack_from('s10', buff, i+102)[0]*1e-10
        eph.tgd_b = bs.unpack_from('s10', buff, i+120)[0]*1e-10

        # page 3
        i = 160*2
        eph.af0 = self.getbits2(buff, i+100, 12, i+120, 12)*rCST.P2_33
        f1p3 = bs.unpack_from('s4', buff, i+132)[0]

        # page 4
        i = 160*3
        f1p4 = self.getbitu2(buff, i+46, 6, i+60, 12)
        eph.af2 = self.getbits2(buff, i+72, 10, i+90, 1)*rCST.P2_66
        eph.iode = bs.unpack_from('u5', buff, i+91)[0]
        eph.deln = bs.unpack_from('s16', buff, i+96)[0]*rCST.P2_43*rCST.SC2RAD
        cucp4 = bs.unpack_from('s14', buff, i+120)[0]

        # page 5
        i = 160*4
        cucp5 = bs.unpack_from('u4', buff, i+46)[0]
        eph.M0 = self.getbits3(buff, i+50, 2, i+60, 12, i+90, 8) * \
            rCST.P2_31*rCST.SC2RAD
        eph.cus = self.getbits2(buff, i+98, 14, i+120, 4)*rCST.P2_31
        ep5 = bs.unpack_from('s10', buff, i+124)[0]

        # page 6
        i = 160*5
        ep6 = self.getbitu2(buff, i+46, 6, i+60, 16)
        sqrtA = self.getbitu3(buff, i+76, 6, i+90, 22, 120, 4)*rCST.P2_19
        eph.A = sqrtA**2
        cicp6 = bs.unpack_from('s10', buff, i+124)[0]

        # page 7
        i = 160*6
        cicp7 = self.getbitu2(buff, i+46, 6, i+60, 2)
        eph.cis = bs.unpack_from('s18', buff, i+62)[0]*rCST.P2_31
        eph.toes = self.getbitu2(buff, i+80, 2, i+90, 15)*8.0
        i0p7 = self.getbits2(buff, i+105, 7, i+120, 14)

        # page 8
        i = 160*7
        i0p8 = self.getbitu2(buff, i+46, 6, i+60, 5)
        eph.crc = self.getbits2(buff, i+65, 17, i+90, 1)*rCST.P2_6
        eph.crs = bs.unpack_from('s18', buff, i+91)[0]*rCST.P2_6
        OMGdp8 = self.getbits2(buff, i+109, 3, i+120, 16)

        # page 9
        i = 160*8
        OMGdp9 = bs.unpack_from('u5', buff, i+46)[0]
        eph.OMG0 = self.getbits3(buff, i+51, 1, i+60, 22, i+90, 9) * \
            rCST.P2_31*rCST.SC2RAD
        omgp9 = self.getbits2(buff, i+99, 13, i+120, 14)

        # page 10
        i = 160*9
        omgp10 = bs.unpack_from('u5', buff, i+46)[0]
        eph.idot = self.getbits2(buff, i+51, 1, i+60, 13) * \
            rCST.P2_43*rCST.SC2RAD

        eph.sva = self.urai2sva(urai)

        eph.af1 = ((f1p3 << 18)+f1p4)*rCST.P2_50
        eph.cuc = ((cucp4 << 4)+cucp5)*rCST.P2_31
        eph.e = ((ep5 << 22)+ep6)*rCST.P2_33
        eph.cic = ((cicp6 << 8)+cicp7)*rCST.P2_31
        eph.i0 = ((i0p7 << 11)+i0p8)*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = ((OMGdp8 << 5)+OMGdp9)*rCST.P2_43*rCST.SC2RAD
        eph.omg = ((omgp9 << 5)+omgp10)*rCST.P2_31*rCST.SC2RAD

        eph.tot = bdt2gpst(bdt2time(eph.week, sow_[0]))
        if eph.toes > sow_[0]+302400.0:
            eph.week += 1
        elif eph.toes < sow_[0]-302400.0:
            eph.week -= 1
        eph.toc = bdt2time(eph.week, toc)
        eph.toe = bdt2time(eph.week, eph.toes)

        eph.flag = 2  # GEO
        eph.mode = 0
        return eph

    def decode_irn_lnav(self, week, time, sat, msg):
        """ NavIC (IRNSS) navigation message decoder """

        sys, prn = sat2prn(sat)

        i = 0
        tlm, towc, alert, autonav, sid = \
            bs.unpack_from('u8u17u1u1u2', msg, i)
        i += 30

        buff = self.irn_nav[prn-1]
        buff[sid*40:sid*40+40] = msg

        if sid > 2:
            return None

        id1 = bs.unpack_from('u2', buff, 27)[0]
        id2 = bs.unpack_from('u2', buff, 320+27)[0]

        tow1 = bs.unpack_from('u17', buff, 8)[0]
        tow2 = bs.unpack_from('u17', buff, 320+8)[0]

        if id1 != 0 or id2 != 1:
            return None

        if tow2 != tow1+1:
            return None

        eph = Eph(sat)

        # subframe 1
        i = 30
        wk, af0, af1, af2, urai = bs.unpack_from('u10s22s16s8u4', buff, i)
        i += 60
        toc, tgd, deln, iode = bs.unpack_from('u16s8s22u8', buff, i)
        i += 64
        svh, cuc, cus, cic, cis, crc, crs, idot = bs.unpack_from(
            'u2s15s15s15s15s15s15s14', buff, i)
        i += 106

        # subframe 2
        i = 320+30
        M0, toe, e, sA, OMG0, omg, OMGd, i0 = bs.unpack_from(
            's32u16u32u32s32s32s22s32', buff, i)
        i += 230

        tow1 *= 12.0

        eph.week = wk+(week-wk+512)//1024*1024
        eph.af0 = af0*rCST.P2_31
        eph.af1 = af1*rCST.P2_43
        eph.af2 = af2*rCST.P2_55

        eph.urai = self.urai2sva(urai)
        toc *= 16.0
        eph.tgd = tgd*rCST.P2_31
        eph.deln = deln*rCST.P2_41*rCST.SC2RAD
        eph.iode = eph.iodc = iode
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
        eph.A = (sA*rCST.P2_19)**2
        eph.OMG0 = OMG0*rCST.P2_31*rCST.SC2RAD
        eph.omg = omg*rCST.P2_31*rCST.SC2RAD
        eph.OMGd = OMGd*rCST.P2_41*rCST.SC2RAD
        eph.i0 = i0*rCST.P2_31*rCST.SC2RAD

        eph.toc = gpst2time(eph.week, toc)
        eph.toe = gpst2time(eph.week, eph.toes)

        if eph.toes > tow1+302400.0:
            week += 1
        elif eph.toes < tow1-302400.0:
            week -= 1
        eph.tot = gpst2time(week, tow1)

        eph.mode = 0
        return eph

    def decode_glo_fdma(self, week, time, sat, msg, freq=0):
        """ Glonass FDMA navigation message decoder """

        sys, prn = sat2prn(sat)

        sid = bs.unpack_from('u4', msg, 1)[0]

        if sid == 0 or sid > 4:
            return None

        buff = self.glo_ca[prn-1]

        if sid == 1:  # when tk is updated, clear buffer
            tk = bs.unpack_from('u12', msg, 9)[0]
            tk_ = bs.unpack_from('u12', buff, 9)[0]
            if tk != tk_:
                for sid_ in range(4):
                    buff[sid_*12:sid_*12+12] = bytearray(12)

        buff[(sid-1)*12:(sid-1)*12+12] = msg

        id1 = bs.unpack_from('u4', buff, 1)[0]
        id2 = bs.unpack_from('u4', buff, 96+1)[0]
        id3 = bs.unpack_from('u4', buff, 96*2+1)[0]
        id4 = bs.unpack_from('u4', buff, 96*3+1)[0]

        if id1 != 1 or id2 != 2 or id3 != 3 or id4 != 4:
            return None

        geph = Geph(sat)

        # frame 1
        i = 7
        P1, tk_h, tk_m, tk_s = bs.unpack_from('u2u5u6u1', buff, i)
        i += 14
        geph.vel[0] = self.getbitg(buff, i, 24)*rCST.P2_20*1e3
        i += 24
        geph.acc[0] = self.getbitg(buff, i, 5)*rCST.P2_30*1e3
        i += 5
        geph.pos[0] = self.getbitg(buff, i, 27)*rCST.P2_11*1e3
        i += 27

        # frame 2
        i = 96+5
        Bn, P2, tb = bs.unpack_from('u3u1u7', buff, i)
        i += 11+5
        geph.vel[1] = self.getbitg(buff, i, 24)*rCST.P2_20*1e3
        i += 24
        geph.acc[1] = self.getbitg(buff, i, 5)*rCST.P2_30*1e3
        i += 5
        geph.pos[1] = self.getbitg(buff, i, 27)*rCST.P2_11*1e3
        i += 27

        # frame 3
        i = 96*2+5
        P3 = bs.unpack_from('u1', buff, i)[0]
        i += 1
        geph.gamn = self.getbitg(buff, i, 11)*rCST.P2_40
        i += 11+1
        P, ln = bs.unpack_from('u2u1', buff, i)
        i += 3
        geph.vel[2] = self.getbitg(buff, i, 24)*rCST.P2_20*1e3
        i += 24
        geph.acc[2] = self.getbitg(buff, i, 5)*rCST.P2_30*1e3
        i += 5
        geph.pos[2] = self.getbitg(buff, i, 27)*rCST.P2_11*1e3
        i += 27

        # frame 4
        i = 96*3+5
        geph.taun = self.getbitg(buff, i, 22)*rCST.P2_30
        i += 22
        geph.dtaun = self.getbitg(buff, i, 5)*rCST.P2_30
        i += 5
        En, _, P4, FT, _, NT, slot, M = bs.unpack_from(
            'u5u14u1u4u3u11u5u2', buff, i)

        geph.sat = prn2sat(uGNSS.GLO, slot)

        geph.svh = Bn
        geph.age = En
        geph.sva = FT
        geph.frq = freq-8

        geph.iode = tb

        geph.flag = (M << 7)+(P4 << 6)+(P3 << 5)+(P2 << 4)+(P1 << 2)+P

        # week, tow = time2gpst(gpst2utc(geph.tof))
        tod = time % 86400
        time -= tod
        tk = tk_h*3600.0+tk_m*60.0+tk_s*30.0
        tof = tk-10800.0
        if tof < tod-43200.0:
            tof += 86400.0
        elif tof > tod+43200.0:
            tof -= 86400.0
        geph.tof = utc2gpst(gpst2time(week, time+tof))
        toe = tb*900.0-10800.0
        if toe < tod-43200.0:
            toe += 86400.0
        elif toe > tod+43200.0:
            toe -= 86400.0
        geph.toe = utc2gpst(gpst2time(week, time+toe))
        geph.toes = toe
        geph.mode = 0
        return geph


class rcvOpt():
    flg_qzslnav = False
    flg_gpslnav = False
    flg_qzscnav = False
    flg_gpscnav = False
    flg_qzscnav2 = False
    flg_gpscnav2 = False
    flg_qzsl6 = False
    flg_gale6 = False
    flg_galinav = False
    flg_galfnav = False
    flg_bdsb1cc = False
    flg_bdsb2a = False
    flg_bdsb2b = False
    flg_bdsd12 = False
    flg_gloca = False
    flg_irnnav = False
    flg_sbas = False
    flg_rnxnav = False
    flg_rnxobs = False


class rcvDec():
    """ template class for receiver message decoder """

    monlevel = 0
    flg_qzslnav = False
    flg_gpslnav = False
    flg_qzscnav = False
    flg_gpscnav = False
    flg_qzscnav2 = False
    flg_gpscnav2 = False
    flg_qzsl6 = False
    flg_gale6 = False
    flg_galinav = False
    flg_galfnav = False
    flg_bdsb1c = False
    flg_bdsb2a = False
    flg_bdsb2b = False
    flg_bdsd12 = False
    flg_gloca = False
    flg_irnnav = False
    flg_sbas = False
    flg_rnxnav = False
    flg_rnxobs = False

    fh_qzslnav = None
    fh_gpslnav = None
    fh_qzscnav = None
    fh_gpscnav = None
    fh_qzscnav2 = None
    fh_gpscnav2 = None
    fh_qzsl6 = None
    fh_gale6 = None
    fh_galinav = None
    fh_galfnav = None
    fh_bdsb1c = None
    fh_bdsb2b = None
    fh_sbas = None
    fh_rnxnav = None
    fh_rnxobs = None

    mode_galinav = 0  # 0: RawNav, 1: Decoded

    rn = None  # placeholder for Raw Navigation message decoder
    re = None  # placeholder for RINEX encoder

    sig_tab = {}

    def init_sig_tab(self, gnss_t='GEJ'):
        """ initialize signal table for RINEX output """
        sig_tab = {}

        if 'G' in gnss_t:
            sig_tab[uGNSS.GPS] = {
                uTYP.C: [rSigRnx('GC1C'), rSigRnx('GC2W'), rSigRnx('GC2L'),
                         rSigRnx('GC5Q')],
                uTYP.L: [rSigRnx('GL1C'), rSigRnx('GL2W'), rSigRnx('GL2L'),
                         rSigRnx('GL5Q')],
                uTYP.D: [rSigRnx('GD1C'), rSigRnx('GD2W'), rSigRnx('GD2L'),
                         rSigRnx('GD5Q')],
                uTYP.S: [rSigRnx('GS1C'), rSigRnx('GS2W'), rSigRnx('GS2L'),
                         rSigRnx('GS5Q')],
            }

        if 'R' in gnss_t:
            sig_tab[uGNSS.GLO] = {
                uTYP.C: [rSigRnx('RC1C'), rSigRnx('RC2C'), rSigRnx('RC2P'),
                         rSigRnx('RC3Q')],
                uTYP.L: [rSigRnx('RL1C'), rSigRnx('RL2C'), rSigRnx('RL2P'),
                         rSigRnx('RL3Q')],
                uTYP.D: [rSigRnx('RD1C'), rSigRnx('RD2C'), rSigRnx('RD2P'),
                         rSigRnx('RD3Q')],
                uTYP.S: [rSigRnx('RS1C'), rSigRnx('RS2C'), rSigRnx('RS2P'),
                         rSigRnx('RS3Q')],
            }

        if 'E' in gnss_t:
            sig_tab[uGNSS.GAL] = {
                uTYP.C: [rSigRnx('EC1C'), rSigRnx('EC5Q'), rSigRnx('EC7Q'),
                         rSigRnx('EC8Q'), rSigRnx('EC6C')],
                uTYP.L: [rSigRnx('EL1C'), rSigRnx('EL5Q'), rSigRnx('EL7Q'),
                         rSigRnx('EL8Q'), rSigRnx('EL6C')],
                uTYP.D: [rSigRnx('ED1C'), rSigRnx('ED5Q'), rSigRnx('ED7Q'),
                         rSigRnx('ED8Q'), rSigRnx('ED6C')],
                uTYP.S: [rSigRnx('ES1C'), rSigRnx('ES5Q'), rSigRnx('GS7Q'),
                         rSigRnx('ES8Q'), rSigRnx('ES6C')],
            }

        if 'C' in gnss_t:
            sig_tab[uGNSS.BDS] = {
                uTYP.C: [rSigRnx('CC1P'), rSigRnx('CC2I'), rSigRnx('CC5P'),
                         rSigRnx('CC6I'), rSigRnx('CC7D'), rSigRnx('CC7I')],
                uTYP.L: [rSigRnx('CL1P'), rSigRnx('CL2I'), rSigRnx('CL5P'),
                         rSigRnx('CL6I'), rSigRnx('CL7D'), rSigRnx('CL7I')],
                uTYP.D: [rSigRnx('CD1P'), rSigRnx('CD2I'), rSigRnx('CD5P'),
                         rSigRnx('CD6I'), rSigRnx('CD7D'), rSigRnx('CD7I')],
                uTYP.S: [rSigRnx('CS1P'), rSigRnx('CS2I'), rSigRnx('CS5P'),
                         rSigRnx('CS6I'), rSigRnx('CS7D'), rSigRnx('CS7I')],
            }

        if 'J' in gnss_t:
            sig_tab[uGNSS.QZS] = {
                uTYP.C: [rSigRnx('JC1C'), rSigRnx('JC2L'), rSigRnx('JC5Q')],
                uTYP.L: [rSigRnx('JL1C'), rSigRnx('JL2L'), rSigRnx('JL5Q')],
                uTYP.D: [rSigRnx('JD1C'), rSigRnx('JD2L'), rSigRnx('JD5Q')],
                uTYP.S: [rSigRnx('JS1C'), rSigRnx('JS2L'), rSigRnx('JS5Q')],
            }

        if 'S' in gnss_t:
            sig_tab[uGNSS.SBS] = {
                uTYP.C: [rSigRnx('SC1C'), rSigRnx('SC5I')],
                uTYP.L: [rSigRnx('SL1C'), rSigRnx('SL5I')],
                uTYP.D: [rSigRnx('SD1C'), rSigRnx('SD5I')],
                uTYP.S: [rSigRnx('SS1C'), rSigRnx('SS5I')],
            }

        if 'I' in gnss_t:
            sig_tab[uGNSS.IRN] = {
                uTYP.C: [rSigRnx('IC5A')], uTYP.L: [rSigRnx('IL5A')],
                uTYP.D: [rSigRnx('ID5A')], uTYP.S: [rSigRnx('IS5A')],
            }

        return sig_tab

    def __init__(self, opt=None, prefix='', gnss_t='GECJ'):
        self.sig_tab = self.init_sig_tab(gnss_t)
        if opt is not None:
            self.init_param(opt, prefix)
        self.nsig = {uTYP.C: 0, uTYP.L: 0, uTYP.D: 0, uTYP.S: 0}

    def init_param(self, opt: rcvOpt, prefix=''):
        if opt.flg_rnxnav or opt.flg_rnxobs:
            self.re = rnxenc(sig_tab=self.sig_tab)

        self.rn = RawNav()

        if opt.flg_qzslnav:
            self.flg_qzslnav = True
            self.file_qzslnav = "qzslnav.txt"
            self.fh_qzslnav = open(prefix+self.file_qzslnav, mode='w')
        if opt.flg_gpslnav:
            self.flg_gpslnav = True
            self.file_gpslnav = "gpslnav.txt"
            self.fh_gpslnav = open(prefix+self.file_gpslnav, mode='w')
        if opt.flg_qzscnav:
            self.flg_qzscnav = True
            self.file_qzscnav = "qzscnav.txt"
            self.fh_qzscnav = open(prefix+self.file_qzscnav, mode='w')
        if opt.flg_gpscnav:
            self.flg_gpscnav = True
            self.file_gpscnav = "gpscnav.txt"
            self.fh_gpscnav = open(prefix+self.file_gpscnav, mode='w')
        if opt.flg_qzscnav2:
            self.flg_qzscnav2 = True
            self.file_qzscnav2 = "qzscnav2.txt"
            self.fh_qzscnav2 = open(prefix+self.file_qzscnav2, mode='w')
        if opt.flg_gpscnav2:
            self.flg_gpscnav2 = True
            self.file_gpscnav2 = "gpscnav2.txt"
            self.fh_gpscnav2 = open(prefix+self.file_gpscnav2, mode='w')
        if opt.flg_qzsl6:
            self.flg_qzsl6 = True
            self.file_qzsl6 = "qzsl6.txt"
            self.fh_qzsl6 = open(prefix+self.file_qzsl6, mode='w')
        if opt.flg_gale6:
            self.flg_gale6 = True
            self.file_gale6 = "gale6.txt"
            self.fh_gale6 = open(prefix+self.file_gale6, mode='w')
        if opt.flg_galinav:
            self.flg_galinav = True
            self.file_galinav = "galinav.txt"
            self.fh_galinav = open(prefix+self.file_galinav, mode='w')
        if opt.flg_galfnav:
            self.flg_galfnav = True
        if opt.flg_bdsb1c:
            self.flg_bdsb1c = True
        if opt.flg_bdsb2a:
            self.flg_bdsb2a = True
        if opt.flg_bdsb2b:
            self.flg_bdsb2b = True
            self.file_bdsb2b = "bdsb2b.txt"
            self.fh_bdsb2b = open(prefix+self.file_bdsb2b, mode='w')
        if opt.flg_bdsd12:
            self.flg_bdsd12 = True
        if opt.flg_gloca:
            self.flg_gloca = True
        if opt.flg_irnnav:
            self.flg_irnnav = True
        if opt.flg_sbas:
            self.flg_sbas = True
            self.file_sbas = "sbas.txt"
            self.fh_sbas = open(prefix+self.file_sbas, mode='w')
        if opt.flg_rnxnav:
            self.flg_rnxnav = True
            self.file_rnxnav = "rnx.nav"
            self.fh_rnxnav = open(prefix+self.file_rnxnav, mode='w')
            self.re.rnx_nav_header(self.fh_rnxnav)
        if opt.flg_rnxobs:
            self.flg_rnxobs = True
            self.file_rnxobs = "rnx.obs"
            self.fh_rnxobs = open(prefix+self.file_rnxobs, mode='w')
            # self.re.rnx_obs_header(self.fh_rnxobs)

    def file_close(self):
        if self.fh_qzsl6 is not None:
            self.fh_qzsl6.close()

        if self.fh_qzslnav is not None:
            self.fh_qzslnav.close()

        if self.fh_gpslnav is not None:
            self.fh_gpslnav.close()

        if self.fh_qzscnav is not None:
            self.fh_qzscnav.close()

        if self.fh_gpscnav is not None:
            self.fh_gpscnav.close()

        if self.fh_qzscnav2 is not None:
            self.fh_qzscnav2.close()

        if self.fh_gpscnav2 is not None:
            self.fh_gpscnav2.close()

        if self.fh_gale6 is not None:
            self.fh_gale6.close()

        if self.fh_galinav is not None:
            self.fh_galinav.close()

        if self.fh_galfnav is not None:
            self.fh_galfnav.close()

        if self.fh_bdsb1c is not None:
            self.fh_bdsb1c.close()

        if self.fh_bdsb2b is not None:
            self.fh_bdsb2b.close()

        if self.fh_sbas is not None:
            self.fh_sbas.close()

        if self.fh_rnxnav is not None:
            self.fh_rnxnav.close()

        if self.fh_rnxobs is not None:
            self.fh_rnxobs.close()
