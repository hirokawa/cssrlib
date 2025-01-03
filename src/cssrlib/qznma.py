"""
QZSS Navigation Message Authentication (QZNMA)

[1] Quasi-Zenith Satellite System Interface Specification
    Satellite Positioning, Navigation and Timing Service (IS-QZSS-PNT-006),
    July, 2024

Note:
    to use the package for QZSNMA, the user needs to
    install the public keys provided by QSS.

@author: Rui Hirokawa
"""

from binascii import unhexlify, hexlify
import numpy as np
import bitstruct.c as bs
from cssrlib.gnss import uGNSS, prn2sat, sat2prn, copy_buff
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.x509 import load_pem_x509_certificate
from cryptography.exceptions import InvalidSignature
from enum import IntEnum
import copy

dtype_ = [('wn', 'int'), ('tow', 'float'), ('prn', 'int'),
          ('type', 'int'), ('len', 'int'), ('nav', 'S512')]


class uNavId(IntEnum):
    """ class for navigation message types"""
    # mt 1: GPS LNAV, 2: GPS CNAV, 3: GPS CNAV2
    #    4: Galileo F/NAV, 5: Galileo I/NAV

    NONE = -1

    GPS_LNAV = 1
    GPS_CNAV = 2
    GPS_CNAV2 = 3
    GAL_FNAV = 4
    GAL_INAV = 5


class uCert(IntEnum):
    X509_CRT = 1
    PEM = 2
    DER = 3


class NavMsg():
    """ class to store the navigation message """
    sys = 0
    navid = 0
    iodn = 0
    iodc = 0
    toe = 0
    toc = 0
    tow = 0
    msg = None

    def __init__(self, sys, navid):
        self.msg = bytearray(120)
        self.sys = sys
        self.navid = navid


class NavParam():
    """ class to store the parameters for authentication """
    nid = 0
    rtow = 0
    svid = 0
    mt = 0
    iode1 = 0
    iode2 = 0
    iodc = 0
    keyid = 0
    ds = None
    salt = None

    def __init__(self, keyid, ds, salt, mt=0, nid=0, rtow=0, svid=0,
                 iode1=0, iode2=0, iodc=0):

        self.nid = nid
        self.mt = mt
        self.rtow = rtow
        self.svid = svid
        self.iode1 = iode1
        self.iode2 = iode2
        self.iodc = iodc
        self.keyid = keyid
        self.ds = ds
        self.salt = salt


def load_pubkey(pubk_path):
    """ load public key information in crt/pem/der format  """
    ext = pubk_path.split('.')[-1]
    if ext == 'crt':
        pk_fmt = uCert.X509_CRT
        mode = 'rt'
    elif ext == 'pem':
        pk_fmt = uCert.PEM
        mode = 'rt'
    elif ext == 'der':
        pk_fmt = uCert.DER
        mode = 'rb'
    else:
        return None

    with open(pubk_path, mode) as f:
        pubk = f.read()
        if pk_fmt == uCert.X509_CRT:
            pk = load_pem_x509_certificate(pubk.encode()).public_key()
        elif pk_fmt == uCert.PEM:
            pk = serialization.load_pem_public_key(pubk.encode())
        elif pk_fmt == uCert.DER:
            pk = serialization.load_der_public_key(pubk)

    return pk


def raw2der(ds):
    """ convert digital signature in raw-format to der-format """
    lds = len(ds)
    ln = lds//2

    r = int.from_bytes(ds[:ln], byteorder='big')
    s = int.from_bytes(ds[ln:], byteorder='big')
    der = utils.encode_dss_signature(r, s)
    return bytes(der)


class qznma():
    """ class for QZNMA processing """

    def __init__(self):
        self.navmsg = {}
        self.pk = None
        self.rds = {}
        self.flag_e = {}
        self.mask = {}
        self.tow = -1
        self.tow_p = {}
        self.buff = {}
        self.vstatus = {}
        self.vcnt_min = 1
        self.nsat = {uGNSS.GPS: 0, uGNSS.GAL: 0, uGNSS.QZS: 0}
        self.monlevel = 0
        self.sat_t = []
        self.npr = {}

        self.navmode = {uGNSS.GPS: uNavId.GPS_LNAV,
                        uGNSS.GAL: uNavId.GAL_INAV,
                        uGNSS.QZS: uNavId.GPS_LNAV}

        self.cnav_mt_t = {10: 1, 11: 2, 30: 3, 31: 3, 32: 3, 33: 3,
                          35: 3, 36: 3, 37: 3, 61: 3}
        self.pubk_bdir = '../data/pubkey/qznma'

    def load_navmsg_lnav(self, navfile):
        """ load GPS/QZSS LNAV navigation messages """
        v = np.genfromtxt(navfile, dtype=dtype_)
        prn_ = np.unique(v['prn'])

        for prn in prn_:
            sat = prn2sat(uGNSS.GPS, prn)
            if sat not in self.navmsg.keys():
                self.navmsg[sat] = {}

            buff = bytearray(120)
            vi = v[v['prn'] == prn]
            for msg_ in vi['nav']:
                msg = unhexlify(msg_)

                sid = bs.unpack_from('u3', msg, 53)[0]
                if sid > 3:
                    continue

                buff[(sid-1)*40:(sid-1)*40+40] = msg[0:40]
                navmsg = self.chk_gps_lnav(uGNSS.GPS, buff)

                if navmsg is not None:
                    iodn = navmsg.iodn
                    if iodn not in self.navmsg[sat].keys():
                        self.navmsg[sat][iodn] = {}
                        self.navmsg[sat][iodn][uNavId.GPS_LNAV] = navmsg

    def load_navmsg_cnav(self, navfile):
        """ load GPS/QZSS CNAV navigation messages """
        v = np.genfromtxt(navfile, dtype=dtype_)
        prn_ = np.unique(v['prn'])

        for prn in prn_:
            sat = prn2sat(uGNSS.GPS, prn)
            if sat not in self.navmsg.keys():
                self.navmsg[sat] = {}

            buff = bytearray(114)
            vi = v[v['prn'] == prn]
            for msg_ in vi['nav']:
                msg = unhexlify(msg_)

                prn_, mt = bs.unpack_from('u6u6', msg, 8)
                if mt not in self.cnav_mt_t.keys():
                    continue

                sid = self.cnav_mt_t[mt]

                if sid > 3:
                    continue

                buff[(sid-1)*38:(sid-1)*38+38] = msg[0:38]

                navmsg = self.chk_gps_cnav(uGNSS.GPS, buff)

                if navmsg is not None:
                    iodn = navmsg.iodn
                    if iodn not in self.navmsg[sat].keys():
                        self.navmsg[sat][iodn] = {}
                        self.navmsg[sat][iodn][uNavId.GPS_CNAV] = navmsg

    def load_navmsg_fnav(self, navfile):
        """ load Galileo F/NAV navigation messages """
        v = np.genfromtxt(navfile, dtype=dtype_)
        prn_ = np.unique(v['prn'])

        for prn in prn_:
            sat = prn2sat(uGNSS.GAL, prn)
            if sat not in self.navmsg.keys():
                self.navmsg[sat] = {}

            buff = bytearray(124)
            vi = v[v['prn'] == prn]
            for msg_ in vi['nav']:
                msg = unhexlify(msg_)

                sid = bs.unpack_from('u6', msg, 0)[0]
                if sid > 4:
                    continue

                for k in range(31):  # copy 244bits
                    buff[(sid-1)*31+k] = msg[k]

                sid1, svid1, iodnav1, toc = bs.unpack_from(
                    'u6u6u10u14', buff, 0)
                sid2, iodnav2 = bs.unpack_from('u6u10', buff, 248*1)
                sid3, iodnav3 = bs.unpack_from('u6u10', buff, 248*2)
                sid4, iodnav4 = bs.unpack_from('u6u10', buff, 248*3)

                if sid1 != 1 or sid2 != 2 or sid3 != 3 or sid4 != 4:
                    continue
                if iodnav1 != iodnav2 or iodnav1 != iodnav3 or \
                        iodnav1 != iodnav4:
                    continue

                if iodnav1 not in self.navmsg[sat].keys():
                    self.navmsg[sat][iodnav1] = {}

                if uNavId.GAL_FNAV in self.navmsg[sat][iodnav1].keys():
                    continue

                toe = bs.unpack_from('u14', buff, 248*2+160)[0]

                tow1 = bs.unpack_from('u20', buff, 248*0+167)[0]
                # tow2 = bs.unpack_from('u20', buff, 248*1+194)[0]
                # tow3 = bs.unpack_from('u20', buff, 248*2+186)[0]
                # tow4 = bs.unpack_from('u20', buff, 248*3+189)[0]

                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV] = \
                    NavMsg(uGNSS.GAL, uNavId.GAL_FNAV)
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].tow = tow1
                # [tow1, tow2, tow3, tow4]
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].toe = toe
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].toc = toc
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].iodn = iodnav1
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].iodc = 0
                self.navmsg[sat][iodnav1][uNavId.GAL_FNAV].msg = \
                    copy.copy(buff)

    def load_navmsg_inav(self, navfile):
        """ load Galileo I/NAV navigation messages """
        v = np.genfromtxt(navfile, dtype=dtype_)
        prn_ = np.unique(v['prn'])

        for prn in prn_:
            sat = prn2sat(uGNSS.GAL, prn)
            if sat not in self.navmsg.keys():
                self.navmsg[sat] = {}

            buff = bytearray(80)
            vi = v[(v['prn'] == prn) & (v['type'] == 0)]  # E1 only
            tow_t = [0, 0, 0, 0, 0]
            for j, msg_ in enumerate(vi['nav']):
                tow_ = int(vi['tow'][j])

                msg = unhexlify(msg_)

                sid, iodnav = bs.unpack_from('u6u10', msg, 2)

                if sid == 0 or sid > 5:
                    continue

                j = (sid-1)*16*8
                copy_buff(msg, buff, 2, j, 112)
                copy_buff(msg, buff, 122, j+112, 16)

                sid1, iodnav1 = bs.unpack_from('u6u10', buff, 0)
                sid2, iodnav2 = bs.unpack_from('u6u10', buff, 128*1)
                sid3, iodnav3 = bs.unpack_from('u6u10', buff, 128*2)
                sid4, iodnav4 = bs.unpack_from('u6u10', buff, 128*3)
                sid5 = bs.unpack_from('u6', buff, 128*4)[0]
                tow = bs.unpack_from('u20', buff, 128*4+85)[0]
                toe = bs.unpack_from('u14', buff, 16)[0]
                toc = bs.unpack_from('u14', buff, 128*3+54)[0]

                tow_t[sid-1] = tow_

                if sid != 5 or sid1 != 1 or sid2 != 2 or sid3 != 3 or \
                        sid4 != 4 or sid5 != 5:
                    continue
                if iodnav1 != iodnav2 or iodnav1 != iodnav3 or \
                        iodnav1 != iodnav4:
                    continue

                if iodnav1 not in self.navmsg[sat].keys():
                    self.navmsg[sat][iodnav1] = {}

                if uNavId.GAL_INAV in self.navmsg[sat][iodnav1].keys():
                    continue

                self.navmsg[sat][iodnav1][uNavId.GAL_INAV] = \
                    NavMsg(uGNSS.GAL, uNavId.GAL_INAV)
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].iodn = iodnav1
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].iodc = 0
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].tow = tow
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].toe = toe
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].toc = toc
                self.navmsg[sat][iodnav1][uNavId.GAL_INAV].msg = \
                    copy.copy(buff)

    def lnav_to_mnav(self, msg, sys=uGNSS.QZS):
        """ genarate MNAV(900b) from LNAV SF1,2,3 """
        mnav = bytearray(113)

        # RAND
        if sys == uGNSS.QZS:
            mask_t = [[0xfffffc, 0xffff80, 0xffffff, 0xffffff,
                      0xffffff, 0xffffff, 0xffffff, 0xffffff,
                      0xffffff, 0xfffffc],
                      [0xfffffc, 0xffff80, 0xffffff, 0xffffff,
                       0xffffff, 0xffffff, 0xffffff, 0xffffff,
                       0xffffff, 0xfffffc],
                      [0xfffffc, 0xffff80, 0xffffff, 0xffffff,
                       0xffffff, 0xffffff, 0xffffff, 0xffffff,
                       0xffffff, 0xfffffc]]
        else:  # GPS
            mask_t = [[0xff0003, 0xfffffc, 0xffffff, 0xffffff,
                      0xffffff, 0xffffff, 0xffffff, 0xffffff,
                      0xffffff, 0xfffffc],
                      [0xff0003, 0xfffffc, 0xffffff, 0xffffff,
                       0xffffff, 0xffffff, 0xffffff, 0xffffff,
                       0xffffff, 0xffff80],
                      [0xff0003, 0xfffffc, 0xffffff, 0xffffff,
                       0xffffff, 0xffffff, 0xffffff, 0xffffff,
                       0xffffff, 0xfffffc]]

        for sid in range(3):
            for k in range(10):
                d = bs.unpack_from('u30', msg, sid*320+32*k+2)[0] & \
                    (mask_t[sid][k] << 6)
                bs.pack_into('u30', mnav, sid*300+30*k, d)

        return mnav

    def cnav2_to_mnav(self, toi, msg, sys=uGNSS.QZS):
        """ genarate MNAV(600b) from CNAV2 SF2 """
        mnav = bytearray(77)

        mask_t = [0xffffffff, 0x7fffffff, 0xffffffff, 0xffffffff,
                  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                  0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                  0xffffffff, 0xfffffbff]

        bs.pack_into('u9', mnav, 0, toi)

        i = 52
        for k in range(18):
            d = bs.unpack_from('u32', msg, i)[0] & mask_t[k]
            i += 32
            bs.pack_into('u32', mnav, 9+32*k, d)

        return mnav

    def cnav_to_mnav(self, msg, sys=uGNSS.QZS):
        """ genarate MNAV(900b) from CNAV SF1,2,3 """
        mnav = bytearray(113)

        # RAND
        if sys == uGNSS.QZS:
            mask_t = [
                [0xff03ffff, 0xfbffffff, 0xffffffff, 0xffffffff,
                 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xfffef000],
                [0xff03ffff, 0xfbffffff, 0xffffffff, 0xffffffff,
                 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xfffff000],
                [0xff03ffff, 0xfbffffff, 0xffffffff, 0xfffffffe,
                 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                 0x00000000]]

        else:  # GPS
            mask_t = [
                [0xfffc0fff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xfffff000],
                [0xfffc0fff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                 0xfffff000],
                [0xfffc0fff, 0xffffffff, 0xffffffff, 0xfffffffe,
                 0x00000000, 0x00000000, 0x00000000, 0x00000000,
                 0x00000000]]

        for sid in range(3):
            for k in range(9):
                d = bs.unpack_from('u32', msg, sid*304 +
                                   32*k)[0] & mask_t[sid][k]
                bs.pack_into('u32', mnav, sid*300+32*k, d)

        return mnav

    def fnav_to_mnav(self, msg):
        """ genarate MNAV(976b) from Galileo F/NAV message 1,2,3,4 """
        mnav = bytearray(122)

        # RAND
        mask_t = [0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
                  0xffffffff, 0xffffffff, 0xfffffc00, 0x0003f]

        for sid in range(4):
            for k in range(8):
                fmt = 'u32' if k < 7 else 'u20'
                d = bs.unpack_from(fmt, msg, sid*248+32*k)[0] & mask_t[k]
                bs.pack_into(fmt, mnav, sid*244+32*k, d)

        return mnav

    def inav_to_mnav(self, msg):
        """ genarate MNAV(640b) from Galileo I/NAV message 1,2,3,4,5 """
        return copy.copy(msg)

    def svid2sat(self, svid):
        """ convert svid in QZNMA to sat """
        if svid <= 63:
            sat = prn2sat(uGNSS.GPS, svid)
        elif svid <= 127:
            sat = prn2sat(uGNSS.GAL, svid-64)
        elif svid <= 191:
            sat = prn2sat(uGNSS.SBS, svid)
        elif svid <= 202:
            sat = prn2sat(uGNSS.QZS, svid)

        return sat

    def decode_gnss_rds(self, tow, msg, i):
        """ generate RDS for GNSS  """
        # nid = 0: QZNMA
        # rtow:
        #  GPS LNAV/CNAV: tow(17), CNAV2: toi(9),itow(8)
        # svid: GPS:1-63, Galileo:65-127, SBAS:129-191, QZS:193-202
        # mt 1: GPS LNAV, 2: GPS CNAV, 3: GPS CNAV2
        #    4: Galileo F/NAV, 5: Galileo I/NAV
        nid, rtow, svid, mt = bs.unpack_from('u4u20u8u4', msg, i)
        i += 36
        # iode1 GPS LNAV: iode of SF2, GPS CNAV/CNAV2 toe of MT10/SF2
        #       Galileo F/NAV,I/NAV IODnav
        # iode2 GPS LNAV: iode of SF3, GPS CNAV toe of MT11, other: zero
        # iodc GPS LNAV iodc, CNAV toc of MT30-37, other: zero
        iode1, iode2, iodc, keyid = bs.unpack_from('u11u11u11u8', msg, i)
        i += 41
        ds = bytearray(64)

        for k in range(16):
            d = bs.unpack_from('u32', msg, i)[0]
            i += 32
            bs.pack_into('u32', ds, k*32, d)

        salt = bs.unpack_from('u16', msg, i)[0]
        i += 16

        sat = self.svid2sat(svid)
        sys, prn = sat2prn(sat)

        if svid == 0 or sat not in self.navmsg.keys() or \
                iode1 not in self.navmsg[sat].keys() or \
                mt not in self.navmsg[sat][iode1].keys():
            return None, None

        npr = NavParam(keyid, ds, salt, mt, nid,
                       rtow, svid, iode1, iode2, iodc)
        tow_ = self.navmsg[sat][iode1][mt].tow

        if self.monlevel > 1:
            print(f"tow={tow} rtow={rtow} tow_i={tow_} svid={svid:3d} " +
                  f"mt={mt} iode={iode1:4d} iode2={iode2:4d} iodc={iodc:4d} " +
                  f"key={keyid}")

        if mt == uNavId.GPS_LNAV:

            mnav = self.lnav_to_mnav(
                self.navmsg[sat][iode1][mt].msg, sys=uGNSS.GPS)

            bs.pack_into('u17', mnav, 300*0+30, rtow)
            bs.pack_into('u17', mnav, 300*1+30, rtow+1)
            bs.pack_into('u17', mnav, 300*2+30, rtow+2)

        elif mt == uNavId.GPS_CNAV:

            mnav = self.cnav_to_mnav(
                self.navmsg[sat][iode1][mt].msg, sys=uGNSS.GPS)

            bs.pack_into('u17', mnav, 300*0+20, rtow)
            bs.pack_into('u17', mnav, 300*1+20, rtow+1)
            bs.pack_into('u17', mnav, 300*2+20, rtow+2)

        elif mt == uNavId.GPS_CNAV2:

            mnav = self.cnav2_to_mnav(
                self.navmsg[sat][iode1][mt].msg, sys=uGNSS.GPS)

        elif mt == uNavId.GAL_FNAV:

            mnav = self.fnav_to_mnav(self.navmsg[sat][iode1][mt].msg)

            bs.pack_into('u20', mnav, 244*0+167, rtow)
            bs.pack_into('u20', mnav, 244*1+194, rtow+10)
            bs.pack_into('u20', mnav, 244*2+186, rtow+20)
            bs.pack_into('u20', mnav, 244*3+189, rtow+30)

        elif mt == uNavId.GAL_INAV:

            mnav = self.inav_to_mnav(self.navmsg[sat][iode1][mt].msg)
            bs.pack_into('u20', mnav, 128*4+85, rtow)

        return npr, mnav

    def verify_gnss_nav(self, npr, mnav):
        """ verifify the navigation message in MNAV using DS """
        if npr.mt == uNavId.GPS_LNAV:
            blen, mlen = 125, 900
        elif npr.mt == uNavId.GPS_CNAV:
            blen, mlen = 125, 900
        elif npr.mt == uNavId.GPS_CNAV2:
            blen, mlen = 89, 609
        elif npr.mt == uNavId.GAL_FNAV:
            blen, mlen = 134, 976
        elif npr.mt == uNavId.GAL_INAV:
            blen, mlen = 92, 640

        rand_ = bytearray(blen)

        bs.pack_into('u4u20u8u4', rand_, 0, npr.nid,
                     npr.rtow, npr.svid, npr.mt)
        bs.pack_into('u11u11u11u8', rand_, 36, npr.iode1,
                     npr.iode2, npr.iodc, npr.keyid)
        copy_buff(mnav, rand_, 0, 77, mlen)
        bs.pack_into('u16', rand_, 77+mlen, npr.salt)

        sat = self.svid2sat(npr.svid)
        sys, prn = sat2prn(sat)

        if self.pk is None:
            pubk_path = self.pubk_bdir + f"/{npr.keyid:03d}.der"
            self.pk = load_pubkey(pubk_path)
            if self.pk is None:
                if self.monlevel > 0:
                    print(f"loading public key {npr.keyid} was faild.")
                return False

        ds_der = raw2der(npr.ds)
        status = False
        try:
            self.pk.verify(ds_der, bytes(rand_), ec.ECDSA(hashes.SHA256()))
            status = True
        except InvalidSignature:
            status = False

        if self.monlevel > 0:
            s = f"tow={self.tow:6d} mt={npr.mt} sys={sys} prn={prn:2d}"
            if status:
                print(f'#{s} signature OK.')
            else:
                print(f'{s} signature NG.')

        if status:
            if sat not in self.vstatus.keys():
                self.vstatus[sat] = {}

            if npr.iode1 not in self.vstatus[sat]:
                self.vstatus[sat][npr.iode1] = 1
            else:
                self.vstatus[sat][npr.iode1] += 1

            if sys not in self.nsat.keys():
                self.nsat[sys] = 0

            if sat not in self.sat_t and self.navmode[sys] == npr.mt:
                self.nsat[sys] += 1
                self.sat_t.append(sat)

        return status

    def gen_rds(self, sat, mode, msg):
        """ prepare RDS from navigation message and parameters  """

        if mode == uNavId.GPS_LNAV:  # 540bits (180bitsx3)
            sid, d = bs.unpack_from('u2u14', msg, 32*2+2+8)
            if sid > 0:
                i0 = (sid-1)*180
                bs.pack_into('u14', self.rds[sat], i0, d)

                for k in range(7):
                    d = bs.unpack_from('u24', msg, 32*(k+3)+2)[0]
                    bs.pack_into('u24', self.rds[sat], i0+14+24*k, d)

                self.mask[sat] |= (1 << (sid-1))

        elif mode == uNavId.GPS_CNAV:  # 708bits (236bits*3)
            sid = bs.unpack_from('u2', msg, 38)[0]
            if sid > 0:
                copy_buff(msg, self.rds[sat], 40, (sid-1)*236, 236)
                self.mask[sat] |= (1 << (sid-1))

        elif mode == uNavId.GPS_CNAV2:  # 702bits (234bits*3)
            sid = bs.unpack_from('u2', msg, 1266)[0]
            if sid > 0:
                copy_buff(msg, self.rds[sat], 1268, (sid-1)*234, 234)
                self.mask[sat] |= (1 << (sid-1))

        # key ID (8bits), DS (512bits), SALT (16bits)
        if self.mask[sat] == 7:
            keyid = self.rds[sat][0]
            ds = self.rds[sat][1:65]
            salt = (self.rds[sat][65] << 8) | self.rds[sat][66]
            npr = NavParam(keyid, ds, salt, mode)
        else:
            npr = None

        return npr

    def chk_gps_lnav(self, sys, buff):
        """ check the integrity of GPS LNAV """
        id1 = bs.unpack_from('u3', buff, 53)[0]
        id2 = bs.unpack_from('u3', buff, 320+53)[0]
        id3 = bs.unpack_from('u3', buff, 320*2+53)[0]

        if id1 != 1 or id2 != 2 or id3 != 3:
            return None

        # SF1
        iodc_ = bs.unpack_from('u2', buff, 32*2+2+22)[0]
        iodc = bs.unpack_from('u8', buff, 32*7+2)[0]
        iodc |= (iodc_ << 8)

        # SF2
        iode1 = bs.unpack_from('u8', buff, 320+32*2+2)[0]
        # SF3
        iode2 = bs.unpack_from('u8', buff, 320*2+32*9+2)[0]

        if iode1 != iode2 or iode1 != (iodc & 0xff):
            return None

        tow = bs.unpack_from('u17', buff, 320*0+32+2)[0]*6
        toc = bs.unpack_from('u16', buff, 320*0+32*7+8+2)[0]
        toe = bs.unpack_from('u16', buff, 320*0+32*9+2)[0]

        navmsg = NavMsg(sys, uNavId.GPS_LNAV)
        navmsg.tow = tow
        navmsg.toe = toe
        navmsg.toc = toc
        navmsg.iodn = iode1
        navmsg.iodc = iodc
        navmsg.msg = copy.copy(buff)

        return navmsg

    def chk_gps_cnav(self, sys, buff):
        """ check the integrity of GPS CNAV """
        id1 = bs.unpack_from('u6', buff, 14)[0]
        id2 = bs.unpack_from('u6', buff, 304+14)[0]
        id3 = bs.unpack_from('u6', buff, 304*2+14)[0]

        if id1 != 10 or id2 != 11 or id3 not in self.cnav_mt_t.keys():
            return None

        tow1 = bs.unpack_from('u17', buff, 20)[0]*6
        toe1 = bs.unpack_from('u11', buff, 70)[0]

        # type 11
        toe2 = bs.unpack_from('u11', buff, 304+38)[0]
        # tow2 = bs.unpack_from('u17', buff, 304+20)[0]*6

        # MT 3x or 61
        toc = bs.unpack_from('u11', buff, 304*2+60)[0]
        # tow3 = bs.unpack_from('u17', buff, 304*2+20)[0]*6

        if toe1 != toe2 or toe1 != toc:
            return None

        # if (tow2 != tow1+12 or tow3 != tow2+12) and \
        #        (tow2 != tow1+6 or tow3 != tow2+6):
        #    return None

        navmsg = NavMsg(sys, uNavId.GPS_CNAV)
        navmsg.tow = tow1
        navmsg.toe = toe1
        navmsg.toc = toc
        navmsg.iodn = toe1
        navmsg.iodc = toc
        navmsg.msg = copy.copy(buff)

        return navmsg

    def msg2nav(self, sat, i0, msg, mode):
        """ prepare navigation message (LNAV/CNAV) from raw nav message """
        blen = 40 if mode == uNavId.GPS_LNAV else 38

        if sat not in self.buff.keys():
            self.buff[sat] = bytearray(120)
            self.navmsg[sat] = {}

        self.buff[sat][(i0-1)*blen:(i0-1)*blen+blen] = msg[0:blen]
        self.flag_e[sat] |= (1 << (i0-1))

        sys, _ = sat2prn(sat)

        if mode == uNavId.GPS_LNAV:
            navmsg = self.chk_gps_lnav(sys, self.buff[sat])
        elif mode == uNavId.GPS_CNAV:
            navmsg = self.chk_gps_cnav(sys, self.buff[sat])
        else:
            navmsg = None

        if navmsg is not None:
            iodn = navmsg.iodn
            if iodn not in self.navmsg[sat].keys():
                self.navmsg[sat][iodn] = {}
                self.navmsg[sat][iodn][mode] = navmsg

    def verify_qzss_nav(self, sat, npr, msg, mode):
        """ verify the navigation messages for QZSS LNAV/CNAV/CNAV2 """
        if mode == uNavId.GPS_LNAV:
            blen, mlen = 116, 900
        elif mode == uNavId.GPS_CNAV:
            blen, mlen = 137, 900
        elif mode == uNavId.GPS_CNAV2:
            blen, mlen = 100, 609

        k1 = (mlen+7)//8

        rand_ = bytearray(blen)
        rand_[0] = npr.keyid
        rand_[1:k1+1] = msg
        bs.pack_into('u16', rand_, 8+mlen, npr.salt)

        if self.pk is None:
            pubk_path = self.pubk_bdir + f"/{npr.keyid:03d}.der"
            self.pk = load_pubkey(pubk_path)
            if self.pk is None:
                if self.monlevel > 0:
                    print(f"loading public key {npr.keyid} was faild.")
                return False

        status = False
        try:
            self.pk.verify(raw2der(npr.ds), bytes(rand_),
                           ec.ECDSA(hashes.SHA256()))
            status = True
        except InvalidSignature:
            status = False

        if status:
            if sat not in self.vstatus.keys():
                self.vstatus[sat] = {}
            if npr.iode1 not in self.vstatus[sat]:
                self.vstatus[sat][npr.iode1] = 1
            else:
                self.vstatus[sat][npr.iode1] += 1

            if sat not in self.sat_t:
                self.nsat[uGNSS.QZS] += 1
                self.sat_t.append(sat)

        return status

    def count_valid_sat(self):
        """ count number of authenticated satellite """
        nsat = 0
        for sat in self.vstatus.keys():
            for iodn in self.vstatus[sat].keys():
                if self.vstatus[sat][iodn] >= self.vcnt_min:
                    nsat += 1
                    break

        return nsat

    def count_tracked_sat(self, tow):
        """ count number of tracked satellite """
        nsat = 0
        for sat in self.navmsg.keys():
            for iodn in self.navmsg[sat].keys():
                status = False
                for nmode in self.navmsg[sat][iodn].keys():
                    if self.navmsg[sat][iodn][nmode].tow <= tow:
                        nsat += 1
                        status = True
                        break
                if status:
                    break

        return nsat

    def decode(self, tow, msg=None, msg_n=None, sat=0,
               navmode=uNavId.GPS_LNAV):
        """ decode QZNMA message and authenticate """
        self.tow = int(tow)

        if msg is not None:

            if sat not in self.flag_e.keys():
                self.flag_e[sat] = 0
                self.tow_p[sat] = -1
                self.mask[sat] = 0
                self.npr[sat] = None
                self.rds[sat] = bytearray(89)

            if navmode == uNavId.GPS_LNAV:  # LNAV
                sid = bs.unpack_from('u3', msg, 53)[0]
                tow_ = bs.unpack_from('u17', msg, 32+2)[0]*6
                _, prn = sat2prn(sat)
                ki = (tow_-self.tow_p[sat])//6

                if sid == 4 or sid == 5:
                    data_id, svid = bs.unpack_from('u2u6', msg, 32*2+2)

                    if svid == 60:
                        self.npr[sat] = self.gen_rds(sat, uNavId.GPS_LNAV, msg)

                        if self.mask[sat] == 7:
                            self.tow_p[sat], self.mask[sat], self.flag_e[sat] \
                                = tow_, 0, 0

                elif ki in (1, 2, 3):
                    self.msg2nav(sat, ki, msg, uNavId.GPS_LNAV)

                if self.flag_e[sat] == 7:
                    self.flag_e[sat] = 0
                    mnav = self.lnav_to_mnav(self.buff[sat])
                    status = self.verify_qzss_nav(
                        sat, self.npr[sat], mnav, uNavId.GPS_LNAV)
                    if self.monlevel > 0:
                        if status:
                            print(f"# LNAV {tow_}, {prn} signature OK.")
                        else:
                            print(f"LNAV {tow_}, {prn} signature NG.")

            elif navmode == uNavId.GPS_CNAV:  # CNAV
                prn_, msgid, tow_, alert = bs.unpack_from('u6u6u17u1', msg, 8)
                prn = prn_ + 192
                tow_ *= 6
                ki = (tow_-self.tow_p[sat])//6

                if msgid == 60:
                    self.npr[sat] = self.gen_rds(sat, uNavId.GPS_CNAV, msg)

                    if self.mask[sat] == 7:
                        self.tow_p[sat], self.mask[sat], self.flag_e[sat] = \
                            tow_, 0, 0

                elif ki in (1, 2, 3):
                    # copy MT10,11,3x after RDS
                    self.msg2nav(sat, ki, msg, uNavId.GPS_CNAV)

                if self.flag_e[sat] == 7:
                    self.flag_e[sat] = 0
                    mnav = self.cnav_to_mnav(self.buff[sat])
                    status = self.verify_qzss_nav(
                        sat, self.npr[sat], mnav, uNavId.GPS_CNAV)
                    if self.monlevel > 0:
                        if status:
                            print(f"# CNAV {tow_},{prn:3d} signature OK.")
                        else:
                            print(f"CNAV {tow_},{prn:3d} signature NG.")

            elif navmode == uNavId.GPS_CNAV2:  # CNAV2
                toi = bs.unpack_from('u9', msg, 0)[0]  # SF 1 (52 syms)
                itow = bs.unpack_from('u8', msg, 65)[0]  # SF 2 (600)
                tow_ = itow*7200+toi*18
                prn, page = bs.unpack_from('u8u6', msg, 1252)  # SF 3 (274)

                if page == 60:
                    self.npr[sat] = self.gen_rds(sat, uNavId.GPS_CNAV2, msg)

                    if self.mask[sat] == 7:
                        self.tow_p[sat], self.mask[sat] = tow_, 0

                elif tow_ == self.tow_p[sat]+18:
                    mnav = self.cnav2_to_mnav(toi, msg)
                    status = self.verify_qzss_nav(
                        sat, self.npr[sat], mnav, uNavId.GPS_CNAV2)
                    if self.monlevel > 0:
                        if status:
                            print(f"# CNAV2 {tow_},{prn:3d} signature OK.")
                        else:
                            print(f"CNAV2 {tow_},{prn:3d} signature NG.")

        if msg_n is not None:  # L6
            mid, alrt = bs.unpack_from('u8u1', msg_n, 40)
            vid = (mid >> 5) & 0x7  # vendor ID

            if vid == 3:  # QZNMA
                npr1, mnav1 = self.decode_gnss_rds(tow, msg_n, 49)
                npr2, mnav2 = self.decode_gnss_rds(tow, msg_n, 49+605)
                if npr1 is not None:
                    status = self.verify_gnss_nav(npr1, mnav1)
                if npr2 is not None:
                    status = self.verify_gnss_nav(npr2, mnav2)

        return True
