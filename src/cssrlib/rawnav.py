"""
Raw Navigation Message decoder
"""

import numpy as np
import struct as st
import bitstruct.c as bs
from cssrlib.gnss import epoch2time, time2gpst, gpst2time, gst2time, time2bdt, bdt2time, rCST
from cssrlib.gnss import prn2sat, uGNSS, time2epoch, sat2prn, sat2id, bdt2gpst
from cssrlib.gnss import Eph, uTYP
from cssrlib.rinex import rnxenc


class RawNav():
    def __init__(self, opt=None, prefix=''):
        self.gps_lnav = {}
        for k in range(uGNSS.GPSMAX):
            self.gps_lnav[k] = bytearray(200)

        self.qzs_lnav = {}
        for k in range(uGNSS.QZSMAX):
            self.qzs_lnav[k] = bytearray(200)

        self.gal_inav = {}
        for k in range(uGNSS.GALMAX):
            self.gal_inav[k] = bytearray(200)

    def urai2sva(self, urai):
        urai_t = [2.0, 2.8, 4.0, 5.7, 8.0, 11.3, 16,
                  32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
        return urai_t[urai]

    def sisa2sva(self, sisa):
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

    def decode_gps_lnav(self, week, time, sat, msg):

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

    def decode_bds_cnav_eph1(self, msg, eph, i):
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
        toc, a0, a1, a2 = \
            bs.unpack_from('u11s25s22s11', msg, i)
        i += 69
        tocs = toc*300.0
        eph.af0 = a0*rCST.P2_34
        eph.af1 = a1*rCST.P2_50
        eph.af2 = a2*rCST.P2_66
        eph.toc = bdt2gpst(bdt2time(eph.week, tocs))
        return i

    def decode_bds_cnav_sisa(self, msg, i):
        top, sisai_ocb, sisai_oc1, sisai_oc2 = \
            bs.unpack_from('u11u5u3u3', msg, i)
        i += 22
        return i

    def decode_bds_cnav_iono(self, msg, i):
        a1, a2, a3, a4, a5, a6, a7, a8, a9 = \
            bs.unpack_from('u10s8u8u8u8s8s8s8s8', msg, i)
        i += 74
        iono = np.array([a1, a2, a3, a4, -a5, a6, a7, a8, a9])*0.125
        return i

    def decode_bds_cnav_utc(self, msg, i):
        a0, a1, a2, dtls, tot, wn_ot, wn_lsf, dn, dt_lsf = \
            bs.unpack_from('s16s13s7s8u16u13u13u3s8', msg, i)
        i += 97
        return i

    def decode_bds_cnav_ralm(self, msg, i):
        prn, sattype, dela, OMG0, Phi0, svh = \
            bs.unpack_from('u6u2s8s7s7u8', msg, i)
        i += 38
        return i

    def decode_bds_cnav_eop(self, msg, i):
        teop, x, xd, y, yd, dut1, dut1d = \
            bs.unpack_from('u16s21s15s21s15s31s19', msg, i)
        i += 138
        return i

    def decode_bds_cnav_ggto(self, msg, i):
        gnss, wn, t0, a0, a1, a2 = \
            bs.unpack_from('u3u13u16s16s13s7', msg, i)
        i += 38
        return i

    def decode_bds_cnav_malm(self, msg, i):
        prn, sattype, wn, toa, ecc, inc, sqrta, OMG0, OMGd, omg, M0, \
            af0, af1, svh = bs.unpack_from(
                'u6u2u13u8u11s11u17s16s11s16s16s11s10u8', msg, i)
        i += 156
        return i

    def decode_bds_cnav_sisai(self, msg, eph, i):
        top, eph.sisai[1], eph.sisai[2], eph.sisai[3] = bs.unpack_from(
            'u11u5u3u3', msg, i)
        eph.top = top*300.0
        i += 22
        return i

    def decode_bds_b1c(self, week, time_, prn, msg):

        eph = Eph()
        eph.sat = prn2sat(uGNSS.BDS, prn)
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
        eph.tgd = tgd_b1cp*2**-34
        eph.isc[0] = isc_b1cd*2**-34
        eph.tgd_b = tgd_b2ap*2**-34
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


class rcvOpt():
    flg_qzslnav = False
    flg_gpslnav = False
    flg_qzsl6 = False
    flg_gale6 = False
    flg_galinav = False
    flg_bdsb1cc = False
    flg_bdsb2b = False
    flg_sbas = False
    flg_rnxnav = False
    flg_rnxobs = False


class rcvDec():
    """ template class for receiver message decoder """

    monlevel = 0
    flg_qzslnav = False
    flg_gpslnav = False
    flg_qzsl6 = False
    flg_gale6 = False
    flg_galinav = False
    flg_bdsb1c = False
    flg_bdsb2b = False
    flg_sbas = False
    flg_rnxnav = False
    flg_rnxobs = False

    fh_qzslnav = None
    fh_gpslnav = None
    fh_qzsl6 = None
    fh_gale6 = None
    fh_galinav = None
    fh_bdsb1c = None
    fh_bdsb2b = None
    fh_sbas = None
    fh_rnxnav = None
    fh_rnxobs = None

    mode_galinav = 0  # 0: RawNav, 1: Decoded

    rn = None  # placeholder for Raw Navigation message decoder
    re = None  # placeholder for RINEX encoder

    def __init__(self, opt=None):
        if opt is not None:
            self.init_param(opt)
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
        if opt.flg_qzsl6:
            self.flg_qzsl6 = True
            self.file_qzsl6 = "qzsl6.txt"
            self.fh_qzsl6 = open(prefix+self.file_qzsl6, mode='w')
        if opt.flg_gale6:
            self.flg_gale6 = True
            self.file_gale6 = "gale6.txt"
            self.fh_gale6 = open(prefix+self.file_gale6, mode='w')
        if opt.flg_gale6:
            self.flg_galinav = True
            self.file_galinav = "galinav.txt"
            self.fh_galinav = open(prefix+self.file_galinav, mode='w')
        if opt.flg_bdsb1c:
            self.flg_bdsb1c = True
            # self.file_bdsb1c = "bdsb1c.nav"
            # self.fh_bdsb1c = open(prefix+self.file_bdsb1c, mode='w')
        if opt.flg_bdsb2b:
            self.flg_bdsb2b = True
            self.file_bdsb2b = "bdsb2b.txt"
            self.fh_bdsb2b = open(prefix+self.file_bdsb2b, mode='w')
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

        if self.fh_gale6 is not None:
            self.fh_gale6.close()

        if self.fh_galinav is not None:
            self.fh_galinav.close()

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
