"""
Galileo OSNMA

[1] Galileo Open Service Navigation Message Authentication (OSNMA)
    Signal-in-Space Interface Control Document (SIS ICD), October, 2023.

Note:
    to use the package for OSNMA, the user needs to
    install the public keys provided by EUSPA.

@author Rui Hirokawa

"""

import numpy as np
import bitstruct.c as bs
from cryptography.hazmat.primitives import hashes, hmac, cmac, serialization
from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.exceptions import InvalidSignature
from binascii import unhexlify, hexlify
from enum import IntEnum
import xml.etree.ElementTree as et
from cssrlib.gnss import gpst2time, time2gst
import copy


class uOSNMA(IntEnum):
    """ class for OSNMA constants """
    ROOTKEY_LOADED = 1
    ROOTKEY_VERIFIED = 2
    KEYCHAIN_VERIFIED = 4
    UTC_VERIFIED = 8
    POS_AUTH = 16
    PKR_UPDATED = 32


class pubkey():
    """ class to store public key """
    pkid = -1
    pk = None
    pkt = None

    def __init__(self, pkid, pkt=0, pk=None):
        self.pkid = pkid
        self.pkt = pkt
        self.pk = pk


class taginfo():
    """ class to store tag """
    gst_sf = bytearray(4)
    prn = -1
    adkd = -1
    cnt = 0
    tag = None
    navmsg = None
    iodnav = -1

    def __init__(self, gst_sf, prn, adkd, cop, tag, cnt, navmsg=None):
        if navmsg is False:
            return None
        self.gst_sf = gst_sf
        self.prn = prn
        self.adkd = adkd
        self.cop = cop
        self.tag = tag
        self.cnt = cnt
        if navmsg is not None:
            self.navmsg = navmsg
            if (adkd == 0 or adkd == 12):
                self.iodnav = bs.unpack_from('u10', navmsg, 0)[0]


class osnma():
    """ class for OSNMA """
    GALMAX = 36
    klen_t = [96, 104, 112, 120, 128, 160, 192, 224, 256]
    npk_len_t = [0, 264, 0, 536, 0]
    tag_len_t = [0, 0, 0, 0, 0, 20, 24, 28, 32, 40, 0, 0, 0, 0, 0, 0]
    hash_table = {0: hashes.SHA256, 2: hashes.SHA3_256}
    mode_t = {27: [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
              28: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
                   1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
              31: [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
              33: [1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
              34: [1, 2, 1, 2, 1, 0, 1, 2, 0, 1, 0, 0],
              35: [1, 2, 1, 2, 1, 2, 1, 2, 2, 1, 2, 2],
              36: [1, 2, 1, 2, 1, 1, 2, 0, 1, 0],
              37: [1, 0, 1, 0, 1, 1, 0, 0, 1, 0],
              38: [1, 2, 1, 2, 1, 1, 2, 2, 1, 2],
              39: [1, 2, 1, 2, 1, 2, 0, 1],
              40: [1, 0, 1, 1, 1, 0, 0, 0],
              41: [1, 2, 1, 2, 1, 2, 2, 1],
              }  # 0:cross-auth, 1:self-auth, 2:flex

    adkd_t = {27: [0, 0, 0, 0, 12, 0, 0, 0, 0, 4, 12, 0],
              28: [0, 0, 0, 0, 0, 0, 0, 12, 0, 0,
                   0, 0, 0, 0, 0, 0, 4, 12, 0, 0],
              31: [0, 0, 0, 12, 0, 0, 0, 0, 12, 4],
              33: [0, 0, 4, 0, 12, 0, 0, 0, 0, 12, 0, 12],
              34: [0, 0, 4, 0, 12, 0, 0, 0, 0, 12, 0, 12],
              35: [0, 0, 4, 0, 12, 0, 0, 0, 0, 12, 0,  0],
              36: [0, 0, 4, 0, 12, 0, 0, 0, 12, 12],
              37: [0, 0, 4, 0, 12, 0, 0, 0, 12,  0],
              38: [0, 0, 4, 0, 12, 0, 0, 0, 12,  0],
              39: [0, 0, 4, 0, 0, 0, 0, 12],
              40: [0, 0, 4, 12, 0, 0, 0, 12],
              41: [0, 0, 4, 0, 0, 0, 0, 12],
              }  # 0: NAV or unknown, 4: UTC, 12: NAV(SLOW-MAC)

    status = 0
    cid0 = -1
    dsm = {}
    flg_dsm = {}
    nb = {}
    did0 = -1

    # OSNMA parameters
    pkid = -1
    cidkr = -1
    hf = -1
    mf = -1
    ks = -1
    ts = -1
    maclt = -1
    wn = -1  # reference week number of root key
    towh = -1  # reference hour of week of root key
    tofst = -60  # time offset for time authentication
    alp = bytearray(6)  # seed
    ds = bytearray(64)  # digital signature
    kroot = bytearray(16)  # root key
    key = bytearray(16)  # key used for authentication
    key_p = bytearray(16)  # previous key
    key_c = bytearray(16)  # current key

    gst_sf_c = bytearray(4)

    gst_tow = -1  # current subframe time
    gst_tow_p = -1  # previous subframe time
    gst_sf_p = bytearray(4)  # gst for previous sub-frame

    nt = 0  # number of tags in MACK
    hk = []
    mack = []
    tag = bytearray(42)
    mack_p = bytearray(60*GALMAX)  # previous MACK
    mack_c = bytearray(60*GALMAX)  # current MACK
    tag_list = []

    vcnt_min = 2
    iodnav = np.zeros(GALMAX+1, dtype=int)
    vcnt = np.zeros(GALMAX+1, dtype=int)
    vstatus = np.zeros(GALMAX+1, dtype=bool)

    subfrm_n = None
    subfrm_p = None
    subfrm = None

    # Public Key in PEM received from GSC OSNMA server
    # note : EC_PARAMETER section should be removed.
    # pubk_path = '../data/OSNMA_PublicKey_20210920133026_s.pem'
    pubk_path = None
    # Merkle tree root (received from GSC OSNMA server)
    # mt_path = '../data/pubkey/osnma/OSNMA_MerkleTree_20210920133026.xml'
    mt_path = '../data/pubkey/osnma/OSNMA_MerkleTree_20240115100000_newPKID_1.xml'
    pk_list = []

    flg_slowmac = False

    def raw2der(self, ds):
        """ convert digital signature from raw format to der format """
        lds = len(ds)
        ln = lds//2
        r = int.from_bytes(ds[:ln], byteorder='big')
        s = int.from_bytes(ds[ln:], byteorder='big')
        der = utils.encode_dss_signature(r, s)

        return bytes(der)

    def difftime(self, t1, t2):
        """ difference of time between t1 and t2 """
        wn1, tow1 = bs.unpack_from('u12u20', t1, 0)
        wn2, tow2 = bs.unpack_from('u12u20', t2, 0)

        dt = (wn1-wn2)*604800 + (tow1-tow2)
        return dt

    def pubkey_decompress(self, pkt, pnt):
        """ decompress public-key """
        if pkt == 1:
            curve = ec.SECP256R1()
        elif pkt == 3:
            curve = ec.SECP521R1()
        else:
            curve = None

        if curve is None:
            return False
        pk = ec.EllipticCurvePublicKey.from_encoded_point(curve, pnt)
        return pk

    def load_mt(self, file):
        """ load markov tree from xml file """
        mt = et.parse(file)
        root = mt.getroot()
        h = root.find('body').find('MerkleTree')
        # hash_ = h.find('HashFunction').text
        for h_pk in h.findall('PublicKey'):
            pkid_ = int(h_pk.find('PKID').text)
            pnt_ = unhexlify(h_pk.find('point').text)
            pkt_s = h_pk.find('PKType').text
            if 'ECDSA P-256' in pkt_s:
                pkt_ = 1
            elif 'ECDSA P-521' in pkt_s:
                pkt_ = 3
            elif 'OAM' in pkt_s:
                pkt_ = 4
            else:
                pkt_ = 0

            pk_ = self.pubkey_decompress(pkt_, pnt_)
            self.pk_list[pkid_].pk = pk_
        for h_tn in h.findall('TreeNode'):
            j_ = int(h_tn.find('j').text)
            i_ = int(h_tn.find('i').text)
            x_ = unhexlify(h_tn.find('x_ji').text)
            if j_ == 4 and i_ == 0:  # root of mt
                self.root_mt = x_
        return True

    def __init__(self):
        self.monlevel = 1       # debug monitor level
        self.cnt = np.zeros(self.GALMAX, dtype=int)
        for prn in range(self.GALMAX):
            self.hk.append(bytearray(15))
            self.mack.append(bytearray(60))
        self.pk_list = []
        for k in range(16):
            self.pk_list.append(pubkey(k))
        self.load_mt(self.mt_path)

        self.flg_dsm = {}

        self.subfrm_n = bytearray(16*10*self.GALMAX)
        self.subfrm_p = bytearray(16*10*self.GALMAX)
        self.subfrm = bytearray(16*10*self.GALMAX)

        self.prn_ref = -1

    def process_hash(self, msg):
        """ calculate hash """
        digest = hashes.Hash(self.hash_table[self.hf]())
        digest.update(msg)
        h = digest.finalize()
        return h

    def set_gst_sf(self, gst_wn, gst_tow):
        """ set Galileo sub-frame time """
        gst_sf = bs.pack('u12u20', gst_wn, gst_tow//30*30)
        return gst_sf

    def verify_root_key(self):
        """ verify root key """
        did = self.did0
        if not self.status & uOSNMA.ROOTKEY_LOADED:  # root key loaded
            return False
        lk_b = self.klen_t[self.ks]//8
        msg = bytearray([self.nma_header]) + self.dsm[did][1:13+lk_b]

        result = False
        hash_func = self.hash_table[self.hf]
        ds_der = self.raw2der(self.ds)
        if self.pubk_path is None:
            pk = self.pk_list[self.pkid].pk
        else:
            with open(self.pubk_path) as f:
                pubk = f.read()
                pk = serialization.load_pem_public_key(pubk.encode())
        try:
            pk.verify(ds_der, bytes(msg), ec.ECDSA(hash_func()))
            result = True
        except InvalidSignature:
            print('signature NG.')
            return False
        if result:
            self.status |= uOSNMA.ROOTKEY_VERIFIED  # root key verified
        return result

    def verify_pdk(self, p_dk, did):
        """ verify P_DK """
        lk_b = self.klen_t[self.ks]//8
        msg = bytearray([self.nma_header]) + self.dsm[did][1:13+lk_b] + self.ds
        h = self.process_hash(msg)
        l_pdk = len(p_dk)
        return h[0:l_pdk] == p_dk

    def verify_pdp(self, m0, p_dp):
        """ verify P_DP """
        msg = self.root_mt + m0
        h = self.process_hash(msg)
        l_pdp = len(p_dp)
        return h[0:l_pdp] == p_dp

    def update_key_chain(self, key, gst_wn, gst_tow0, ki=1):
        """ return next key from key chanin """
        # k_l=trunc(lk,hash(k_l+1||GST_SF_l||alp))
        klen_b = self.klen_t[self.ks]//8
        for k in range(ki):
            gst_tow = gst_tow0 - 30*(k+1)
            if gst_tow < 0:
                gst_tow += 86400*7
                gst_wn -= 1
            gst = bs.pack('u12u20', gst_wn, gst_tow)
            msg = key + gst + self.alp
            key = self.process_hash(msg)[0:klen_b]
        return key

    def verify_key_chain(self, key, gst_wn, gst_tow0, ki=1):
        """ verify key chaning """
        if not self.status & uOSNMA.ROOTKEY_VERIFIED:  # root-key verified
            return False
        if self.status & uOSNMA.KEYCHAIN_VERIFIED:
            key = self.update_key_chain(key, gst_wn, gst_tow0, 1)
            result = (key == self.key_p)
        else:
            key = self.update_key_chain(key, gst_wn, gst_tow0, ki)
            result = (key == self.kroot)
        if result:
            self.status |= uOSNMA.KEYCHAIN_VERIFIED  # key-chain verified
        return result

    def decode_dsm_kroot(self, did):
        """ decode DSM-KROOT """
        v = bs.unpack_from('u4u4u2u2u2u2u4u4u8u4u12u8', self.dsm[did], 0)
        nb = v[0]+6   # number of blocks
        self.pkid = v[1]  # Public Key ID
        self.cidkr = v[2]  # KROOT Chain ID
        self.hf = v[4]  # hash function 0:SHA-256,2:SHA3-256
        self.mf = v[5]  # mac function 0:HMAC-SHA-256,1:CMAC-AES
        self.ks = v[6]  # key size 0:96,1:104,2:112,3:120,4:128,
        #          5:160,6:192,7:224,8:256
        self.ts = v[7]  # tag length 5:20,6:24,7:28,8:32,9:40
        self.maclt = v[8]  # MAC lookup table
        self.wn = v[10]  # KROOT week number, tow[h]
        self.towh = v[11]
        self.alp = self.dsm[did][7:13]  # random pattern alpha
        l_dk = nb*104
        l_ds = 512  # P-256/SHA-256
        l_k = self.klen_t[self.ks]
        l_pdk = l_dk-104-l_k-l_ds
        if l_pdk < 0:
            return False

        i = 13+l_k//8
        self.kroot = self.dsm[did][13:i]
        self.ds = self.dsm[did][i:i+l_ds//8]
        i += l_ds//8
        p_dk = self.dsm[did][i:i+(l_pdk+7)//8]
        if not self.verify_pdk(p_dk, did):
            print("p_dk verification error.")
            return False
        self.status |= uOSNMA.ROOTKEY_LOADED  # KROOT loaded
        self.did0 = did
        return True

    def decode_dsm_pkr(self, did):
        """ decode DSM-PKR """
        nb, mid = bs.unpack_from('u4u4', self.dsm[did], 0)
        if nb < 7 or nb > 10:
            return False
        itn = self.dsm[did][1:1+128]  # 32*4
        npkt, npkid = bs.unpack_from('u4u4', self.dsm[did], 1024+8)
        # new public key type 1:ECDSA P-256, 3: ECDSA P-521, 4: OAM
        if npkt > 4 or npkt == 0:
            return False
        l_dp = (nb+6)*104
        if npkt == 4:
            l_npk = 104*(nb+6)-1040
        else:
            l_npk = self.npk_len_t[npkt]
        i0 = 130+l_npk//8
        npk = self.dsm[130:i0]
        l_pdp = l_dp - 1040 - l_npk
        if l_pdp < 0:
            return False
        p_dp = self.dsm[did][i0:i0+l_pdp//8]

        m0 = bytearray([self.dsm[did][129]])+npk  # NPKT||NPKID||NPK
        if not self.verify_pdp(m0, p_dp):  # verify P_DP
            return False

        h = self.process_hash(m0)
        for k in range(4):
            itn_b = itn[k*32:(k+1)*32]
            if mid % 2 == 0:
                msg = h+itn_b
            else:
                msg = itn_b+h
            h = self.process_hash(msg)
            mid >>= 1

        result = (h == self.root_mt)
        self.npkid = npkid
        if result:
            pk_ = self.pubkey_decompress(npkt, npk)
            self.pk_list[npkid].pk = pk_
            self.status |= uOSNMA.PKR_UPDATED  # PKR updated
        return result

    def decode_hk(self, hk, prn):
        """ decode HKROOT message """
        self.nma_header = hk[0]
        nmas, cid, cpks, _ = bs.unpack_from('u2u2u3u1', hk, 0)
        did, bid = bs.unpack_from('u4u4', hk, 8)
        if nmas != 1 and nmas != 2:
            return False
        if cpks != 1:  # nominal only
            return False

        if did not in self.flg_dsm.keys():
            self.flg_dsm[did] = 0
            self.dsm[did] = bytearray(250)
            self.nb[did] = 0

        if self.cid0 < 0:
            self.cid0 = cid

        if cid != self.cid0:
            self.cid0 = cid
            self.flg_dsm[did] = 0

        self.dsm[did][bid*13:bid*13+13] = hk[2:]
        self.flg_dsm[did] |= 1 << bid

        if bid == 0:
            nb_ = (hk[2] >> 4) & 0xf
            self.nb[did] = nb_ + 6  # number of blocks

        result = False
        if self.monlevel > 1:
            print(f"flg_dsm[did={did}]={self.flg_dsm[did]:2x} "
                  f"nb={self.nb[did]:2d} bid={bid} prn={prn}")
        if did in self.nb.keys() and \
                self.flg_dsm[did] == (1 << self.nb[did])-1:
            if did <= 11:  # DSM-KROOT
                result = self.decode_dsm_kroot(did)
            else:  # DSM-PKR
                result = self.decode_dsm_pkr(did)

        if result:
            self.flg_dsm[did] = 0
        return result

    def decode_tags_info(self, k):
        """ decode Tags&Info message """
        if not self.status & uOSNMA.ROOTKEY_LOADED:
            return False
        lt = self.tag_len_t[self.ts]
        lt_b = lt//8
        ltag_b = lt_b+2
        i0 = k*ltag_b
        tag_k = self.tag[i0:i0+ltag_b]
        if k == 0:
            tag0 = tag_k[0:lt_b]
            macseq = tag_k[lt_b:lt_b+2]  # MACSEQ(12)+res(4)
            cop = macseq[1] & 0xf
            macseq[1] &= 0xf0
            return tag0, macseq, cop
        else:
            tag = tag_k[0:lt_b]
            tag_info = tag_k[lt_b:lt_b+2]
            # 1-36 Galileo SV, 255: Galileo constellation
            prn_d = tag_info[0]  # sv transmitting the data to be authenticated
            # ADKD Authentication Data and Key Delay
            #   0:I/NAV ephemeris,clock,status
            #   4:I/NAV timing
            #  12: slow MAC (5min)
            adkd = (tag_info[1] >> 4) & 0xf
            cop = tag_info[1] & 0xf  # Data cut-off point
            return tag, prn_d, adkd, cop
        return False

    def verify_maclt(self):
        """ check consistency of Tag sequence """
        if self.maclt not in self.mode_t:
            return False
        mode_t = self.mode_t[self.maclt]
        adkd_t = self.adkd_t[self.maclt]
        if self.nt*2 != len(adkd_t):
            return False
        gst_wn, gst_tow_p = bs.unpack_from('u12u20', self.gst_sf_p, 0)
        ofst = 0 if gst_tow_p % 60 == 0 else 6
        for k in range(self.nt-1):
            tag, prn_d, adkd, cop = self.decode_tags_info(k+1)
            i = k + ofst + 1
            if mode_t[i] == 1 and prn_d != self.prn_a:  # self-auth
                return False
            if mode_t[i] != 2 and adkd != adkd_t[i]:  # non-flex
                return False
        return True

    def process_mac(self, msg):
        """ calculate crypt-MAC for message using key """
        if self.mf == 0:  # HMAC-SHA-256
            hm = hmac.HMAC(self.key, hashes.SHA256())
        elif self.mf == 1:  # CMAC-AES
            hm = cmac.CMAC(algorithms.AES(self.key))
        else:
            return False
        hm.update(msg)
        return hm.finalize()

    def verify_macseq(self):
        """ verify MACSEQ """
        msg = bytearray([self.prn_a])+self.gst_sf_p  # Eq.22

        if self.maclt not in self.mode_t:
            return False
        mode_t = self.mode_t[self.maclt]
        gst_wn, gst_tow_p = bs.unpack_from('u12u20', self.gst_sf_p, 0)
        ofst = 0 if gst_tow_p % 60 == 0 else 6

        ltag_b = self.tag_len_t[self.ts]//8+2
        for k in range(self.nt):  # add FLEX taginfo
            if mode_t[k+ofst] != 2:
                continue
            i0 = k*ltag_b
            msg += self.tag[i0+ltag_b-2:i0+ltag_b]

        macseq_c_ = self.process_mac(msg)
        tag0, macseq_, cop = self.decode_tags_info(0)
        macseq_c = bs.unpack_from('u12', macseq_c_, 0)[0]
        macseq = bs.unpack_from('u12', macseq_, 0)[0]
        return macseq == macseq_c

    def save_mack(self, mack, prn):
        """ store MACK section (480bits=60bytes) """
        lm_b = 60
        i0 = (prn-1)*lm_b
        self.mack_p[i0:i0+lm_b] = self.mack_c[i0:i0+lm_b]
        self.mack_c[i0:i0+lm_b] = mack

    def decode_mack(self, prn):
        """ decode MACK message """
        if not self.status & uOSNMA.ROOTKEY_LOADED:
            return False
        lt = self.tag_len_t[self.ts]
        lk = self.klen_t[self.ks]
        self.nt = (480-lk)//(lt+16)  # number of tags
        ltag_b = (lt+16)//8
        lm_b = 60
        i0 = (prn-1)*lm_b
        mack_p = self.mack_p[i0:i0+lm_b]  # previous MACK
        mack_c = self.mack_c[i0:i0+lm_b]  # current MACK
        i0 = ltag_b*self.nt
        self.tag = mack_p[0:i0]
        self.key_p = mack_p[i0:i0+lk//8]
        self.key_c = mack_c[i0:i0+lk//8]
        return True

    def decode_nav(self, df):
        """ decode navigation message """
        p1, t1, mt = bs.unpack_from('u1u1u6', df, 0)
        p2, t2 = bs.unpack_from('u1u1', df, 120)
        if p1 != 0 or p2 != 1 or t1 == 1 or t2 == 1:
            return False
        if mt > 0 and mt <= 10:
            j = (mt-1)*16*8
            for k in range(14):
                b = bs.unpack_from('u8', df, 2+8*k)[0]
                bs.pack_into('u8', self.subfrm, j, b)
                j += 8
            for k in range(2):
                b = bs.unpack_from('u8', df, 122+8*k)[0]
                bs.pack_into('u8', self.subfrm, j, b)
                j += 8
        return True

    def load_inav(self, msg):
        """ load I/NAV navigation message """

        # even page (120bit) + odd page (120bit)
        # even page even/odd(1b)+page type(1b)+data1(112b)+tail(6)
        # odd  page even/odd(1b)+page type(1b)+data2(16b)+osnma(40b)
        #           sar(22b)+spare(2b)+crc(24b)+ssp(8b)+tail(6b)
        nav = bytearray(16)
        nma_b = bytearray(5)

        even, pt1 = bs.unpack_from('u1u1', msg, 0)
        odd, pt2 = bs.unpack_from('u1u1', msg, 120)

        if even != 0 or odd != 1 or pt1 != 0 or pt2 != 0:
            print("I/NAV page format error.")

        i = 2
        for k in range(14):
            nav[k] = bs.unpack_from('u8', msg, i)[0]
            i += 8
        i += 8
        for k in range(2):
            nav[14+k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

        for k in range(5):
            nma_b[k] = bs.unpack_from('u8', msg, i)[0]
            i += 8

        return nav, nma_b

    def load_nav(self, nav, prn, tow):
        """ load navigation message into subframe buffer """
        mt = bs.unpack_from('u6', nav, 0)[0]
        j0 = (prn-1)*160

        if tow % 30 == 1:
            self.subfrm[j0:j0+160] = self.subfrm_p[j0:j0+160]
            self.subfrm_p[j0:j0+160] = self.subfrm_n[j0:j0+160]
            # self.subfrm_n[j0:j0+160] = [0]*160

        if mt > 0 and mt <= 10:
            j = j0+(mt-1)*16
            self.subfrm_n[j:j+16] = nav

        if self.monlevel > 0 and prn == self.prn_ref:
            if mt >= 1 and mt <= 4:
                iodnav = bs.unpack_from('u10', nav, 6)[0]
                print(f"tow={tow} prn={prn:2d} mt={mt:2d} iodnav={iodnav}")
            elif mt == 5:
                tow_ = bs.unpack_from('u20', nav, 85)[0]
                print(f"tow={tow} prn={prn:2d} mt={mt:2d} tow={tow_}")
            elif mt == 6:
                tow_ = bs.unpack_from('u20', nav, 105)[0]
                print(f"tow={tow} prn={prn:2d} mt={mt:2d} tow={tow_}")
            else:
                print(f"tow={tow} prn={prn:2d} mt={mt:2d}")

        return True

    def gen_navmsg(self, prn):
        """ generate nav message for nma """
        if prn < 1 or prn > self.GALMAX:
            return None
        j0 = 160*8*(prn-1)
        for k in range(5):
            mt = bs.unpack_from('u6', self.subfrm, j0+k*16*8)[0]
            if mt != k+1:
                return None

        iodnav1 = bs.unpack_from('u10', self.subfrm, j0+6+0*16*8)[0]
        for k in range(1, 4):
            iodnav_ = bs.unpack_from('u10', self.subfrm, j0+6+k*16*8)[0]
            if iodnav_ != iodnav1:
                if self.monlevel > 0:
                    print("error: iodnav mismatch mt=%d %d %d" %
                          (k+1, iodnav1, iodnav_))
                return None

        if self.monlevel > 0:
            svid = bs.unpack_from('u6', self.subfrm, j0+16+3*16*8)[0]
            wn, tow = bs.unpack_from('u12u20', self.subfrm, j0+73+4*16*8)
            print(f" svid={svid:2d} iodnav={
                  iodnav1:2d} gst_wn={wn:4d} gst_tow={tow:6d}")

        msg = bytearray(69)
        # 549b MT1 120b, MT2 120b, MT3 122b, MT4 120b, MT5 67b
        j = 0
        for mt in range(5):
            for k in range(15):
                b = bs.unpack_from('u8', self.subfrm, j0+6+k*8+mt*16*8)[0]
                bs.pack_into('u8', msg, j, b)
                j += 8
                if mt == 4 and k >= 7:
                    b = bs.unpack_from('u3', self.subfrm, j0+70+4*16*8)[0]
                    bs.pack_into('u3', msg, j, b)
                    j += 3
                    break
            if mt == 2:
                b = bs.unpack_from('u2', self.subfrm, j0+126+2*16*8)[0]
                bs.pack_into('u2', msg, j, b)
                j += 2
        return msg

    def gen_utcmsg(self):
        """ generate utc message for nma """
        j0 = 160*(self.prn_a-1)
        i0 = 5*16+j0
        mt6 = self.subfrm[i0:i0+16]
        i0 = 9*16+j0
        mt10 = self.subfrm[i0:i0+16]

        t1 = bs.unpack_from('u6',  mt6, 0)[0]  # MT6
        t2 = bs.unpack_from('u6', mt10, 0)[0]  # MT10
        if t1 != 6 or t2 != 10:
            return None
        if self.monlevel > 0:
            tow = bs.unpack_from('u20', mt6, 105)[0]
            print(" utc gst_tow=%6d" % (tow))

        # 141b MT6 99b, MT10 42b
        msg = bytearray(18)
        j = 0
        for k in range(13):
            b = bs.unpack_from('u8', mt6, 6+k*8)[0]
            bs.pack_into('u8', msg, j, b)
            j += 8
        j = 99

        a0g, a1g, t0g, wn0g = bs.unpack_from('s16s12u8u6', mt10, 86)
        bs.pack_into('s16s12u8u6', msg, j, a0g, a1g, t0g, wn0g)

        return msg

    def gen_msg(self, adkd, prn_d, gst_sf, ctr, msg):
        """ generate message for verification of NMA """
        mlen = 141 if adkd == 4 else 549
        rem_ = mlen-mlen//8*8  # number of remaining bit
        j = 0
        if adkd == 0 and self.prn_a == prn_d:
            mlen += 8+42
            mlen_b = (mlen+7)//8
            m = bytearray(mlen_b)
            bs.pack_into('u8', m, j, self.prn_a)
            j += 8
        else:
            mlen += 16+42
            mlen_b = (mlen+7)//8
            m = bytearray(mlen_b)
            if prn_d == 0xff:
                prn_d = self.prn_a
            bs.pack_into('u8u8', m, j, prn_d, self.prn_a)
            j += 16
        bs.pack_into('r32u8u2', m, j, gst_sf, ctr, self.nma_header >> 6)
        j += 42
        for k in range(len(msg)):
            b = msg[k]
            if rem_ != 0 and k == len(msg)-1:
                bs.pack_into('u'+str(rem_), m, j, b >> (8-rem_))
                j += rem_
            else:
                bs.pack_into('u8', m, j, b)
                j += 8
        return m

    def verify_navmsg(self, tag_):
        """ verify nav/utc message """
        prn_d = tag_.prn
        adkd = tag_.adkd
        msg = tag_.navmsg

        if prn_d == -1 or not msg:
            return False

        m = self.gen_msg(adkd, prn_d, tag_.gst_sf, tag_.cnt, msg)

        tag_c = self.process_mac(m)
        lt_b = self.tag_len_t[self.ts]//8
        result = (tag_c[:lt_b] == tag_.tag)
        return result

    def chk_nav(self, result, tag_):
        """ judge authentication status of message """

        if not result:
            return False

        if tag_.adkd == 4:
            i = self.GALMAX
        else:
            i = tag_.prn-1

        if self.iodnav[i] != tag_.iodnav:
            self.vcnt[i] = 1
        else:
            self.vcnt[i] += 1
        self.iodnav[i] = tag_.iodnav

        if self.vcnt[i] >= self.vcnt_min:
            self.vstatus[i] = True
        else:
            self.vstatus[i] = False
        return self.vstatus[i]

    def decode(self, nma_b, wn, tow, prn):
        """ decode OSNMA message
        Parameters
        ----------
        nma_b: bytearray(5)
               NMA binary message (40bits) in I/NAV
        wn: int
            GPS-Week number
        tow: int
            GPS time-of-week
        prn: int
            PRN number
        """
        status = False
        ki = (tow % 30)//2
        if ki == 0:  # reset counter
            self.cnt[prn-1] = 0
            self.hk[prn-1] = bytearray(15)
            self.mack[prn-1][0] = 0
        elif ki < 0:
            return

        # convert GPS time to Galileo standard time (GST)
        gst_wn, gst_tow = time2gst(gpst2time(wn, tow))

        self.gst_tow = (gst_tow//30)*30  # current subframe-time
        # store sub-frame for NMA
        self.hk[prn-1][ki] = nma_b[0]              # HK-ROOT message
        self.mack[prn-1][ki*4:ki*4+4] = nma_b[1:5]  # MACK message
        self.cnt[prn-1] |= (1 << ki)
        if self.cnt[prn-1] == 0x7fff:  # all(0-14) message loaded
            self.save_mack(self.mack[prn-1], prn)  # store MACK
            # decode HK-ROOT messages
            result = self.decode_hk(self.hk[prn-1], prn)
            if self.monlevel > 0:
                s = "wn=%4d tow=%d gst_tow=%6d prn=%d did=%d" % \
                    (wn, tow, self.gst_tow, prn, self.did0)
                if result:
                    print("decode_hk succeeded %s" % (s))

            if self.status & uOSNMA.ROOTKEY_LOADED and \
                    not (self.status & uOSNMA.ROOTKEY_VERIFIED):
                result = self.verify_root_key()
                if self.monlevel > 0:
                    s = "wn=%4d tow=%d gst_tow=%6d prn=%d" % \
                        (wn, tow, self.gst_tow, prn)
                    if result:
                        print("root-key verified %s" % (s))
                    else:
                        print("root-key not verified %s" % (s))

            if self.decode_mack(prn):
                self.gst_sf = self.set_gst_sf(gst_wn, self.gst_tow)
                # key-chain verification
                # skip if key-chain is already verified at t=gst_sf
                # if self.gst_sf != self.gst_sf_c and \
                #        self.status & uOSNMA.KEYCHAIN_VERIFIED:
                #    self.status ^= uOSNMA.KEYCHAIN_VERIFIED
                if self.status & uOSNMA.ROOTKEY_VERIFIED and \
                        self.gst_sf != self.gst_sf_c:  # root-key verified

                    if self.status & uOSNMA.KEYCHAIN_VERIFIED:
                        ki = 1
                    else:
                        ki = ((gst_wn-self.wn)*86400*7 +
                              (self.gst_tow-self.towh*3600))//30+1
                    result = self.verify_key_chain(
                        self.key_c, gst_wn, self.gst_tow, ki)
                    if result:
                        self.gst_sf_c = self.gst_sf
                        self.key = self.key_c
                    if self.monlevel > 0:
                        s = "wn=%4d tow=%6d ki=%d prn=%d gst_tow=%d" % \
                            (wn, tow, ki, prn, self.gst_tow)
                        if result:
                            print("Key chain verified     %s" % (s))

                        else:
                            print("Key chain not verified %s" % (s))

                if self.status & uOSNMA.KEYCHAIN_VERIFIED:
                    # 6.5 MAC Look-up Table Verification
                    self.gst_tow_p = self.gst_tow-30
                    self.gst_sf_p = self.set_gst_sf(gst_wn, self.gst_tow_p)
                    result = self.verify_maclt()
                    if self.monlevel > 0:
                        s = "on %4d/%6d gst_tow=%d prn=%d" % \
                            (wn, tow, self.gst_tow_p, prn)
                        if result:
                            print("MAC Look-up Table verified %s" % (s))
                        else:
                            print("MAC Look-up Table not verified %s" % (s))
                    if not result:
                        return False
                    # 6.6 MACSEQ Verification
                    # prn_a|gst_sf
                    result = self.verify_macseq()
                    if result and self.monlevel > 0:
                        print("MACSEQ Verified on %4d/%6d prn=%2d" %
                              (wn, tow, prn))
                    else:
                        print("MACSEQ not verified on %4d/%6d prn=%2d" %
                              (wn, tow, prn))
                        return False

                    # 6.7 Tag Verification
                    for k in range(self.nt):
                        ctr = k+1
                        if k == 0:  # self-tag
                            prn_d = self.prn_a
                            adkd = 0
                            tag, macseq, cop = self.decode_tags_info(0)
                        else:
                            tag, prn_d, adkd, cop = self.decode_tags_info(k)
                            if adkd == 12 and self.flg_slowmac:
                                # delayed tag loading
                                navmsg = self.gen_navmsg(prn_d)
                                tag_ = taginfo(self.gst_sf_p, prn_d, adkd, cop,
                                               tag, ctr, navmsg)
                                self.tag_list.append(tag_)

                        if adkd == 12:
                            print(f"{ctr} prn_d={prn_d} adkd={adkd} slow-MAC "
                                  "is skipped")
                            continue
                        elif adkd == 0:
                            navmsg = self.gen_navmsg(prn_d)
                        elif adkd == 4:
                            navmsg = self.gen_utcmsg()
                        else:
                            navmsg = None

                        if navmsg is None:
                            print(f"{ctr} prn_d={prn_d} adkd={adkd} navmsg is "
                                  "not available.")
                            continue

                        tag_ = taginfo(self.gst_sf_p, prn_d,
                                       adkd, cop, tag, ctr, navmsg)
                        result = self.verify_navmsg(tag_)
                        status = self.chk_nav(result, tag_)
                        if status and adkd == 4:
                            self.status |= uOSNMA.UTC_VERIFIED
                        if self.monlevel > 0:
                            if result:
                                print(f"# {ctr} prn_d={prn_d} adkd={
                                      adkd} tag verified")
                            else:
                                print(f"{ctr} prn_d={prn_d} adkd={
                                      adkd} tag not verified")
                    # slow MAC
                    if self.flg_slowmac:
                        for tag_ in self.tag_list:
                            dt = self.difftime(self.gst_sf, tag_.gst_sf)
                            if dt == 300:
                                result = self.verify_navmsg(tag_)
                                if self.monlevel > 0 and result:
                                    print("SLOW-MAC# %d prn_d=%2d adkd=%2d tag verified"
                                          % (tag_.cnt, tag_.prn, tag_.adkd))
                            elif dt > 300:
                                tag_ = None

            self.cnt[prn-1] = 0
        return status
