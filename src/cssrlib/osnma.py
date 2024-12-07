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
from binascii import unhexlify
from enum import IntEnum
import xml.etree.ElementTree as et


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

    def __init__(self, gst_sf, prn, adkd, tag, cnt, navmsg=None):
        if navmsg is False:
            return None
        self.gst_sf = gst_sf
        self.prn = prn
        self.adkd = adkd
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
    self_t = {27: [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0],
              28: [1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0],
              31: [1, 0, 0, 1, 0, 1, 0, 0, 1, 1],
              33: [1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]}  # 0:self,1:cross
    adkd_t = {27: [0, 0, 0, 0, 12, 0, 0, 0, 0, 4, 12, 0],
              28: [0, 0, 0, 0, 0, 0, 0, 12, 0, 0,
                   0, 0, 0, 0, 0, 0, 4, 12, 0, 0],
              31: [0, 0, 0, 12, 0, 0, 0, 0, 12, 4],
              33: [0, 0, 4, 0, 12, 0, 0, 0, 0, 12, 0, 12]}

    status = 0
    cid0 = -1
    dsm = bytearray(512)
    flg_dsm = 0
    nb = 0
    did0 = -1
    pkid = -1
    cidkr = -1
    hf = -1
    mf = -1
    ks = -1
    ts = -1
    maclt = -1
    wn = -1
    towh = -1
    tofst = -60
    alp = bytearray(6)
    ds = bytearray(64)
    pdk = bytearray(16)
    kroot = bytearray(16)
    key = bytearray(16)
    key_p = bytearray(16)
    key_c = bytearray(16)
    gst_sf_c = bytearray(4)

    hk = []
    mack = []
    tag = bytearray(42)
    mack_p = bytearray(60*GALMAX)  # previous MACK
    mack_c = bytearray(60*GALMAX)  # current MACK
    subfrm = bytearray(16*10*GALMAX)
    tag_list = []

    vcnt_min = 2
    iodnav = np.zeros(GALMAX+1, dtype=int)
    vcnt = np.zeros(GALMAX+1, dtype=int)
    vstatus = np.zeros(GALMAX+1, dtype=bool)

    # Public Key received from GSC OSNMA server
    # note : EC_PARAMETER section should be removed.
    # pubk_path = '../data/OSNMA_PublicKey_20210920133026_s.pem'
    pubk_path = None
    # Merkle tree root (received from GSC OSNMA server)
    mt_path = '../data/OSNMA_MerkleTree_20210920133026.xml'
    pk_list = []

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

    def process_hash(self, msg):
        """ calculate hash """
        digest = hashes.Hash(self.hash_table[self.hf]())
        digest.update(msg)
        h = digest.finalize()
        return h

    def set_gst_sf(self, gst_wn, gst_tow):
        """ set Galileo sub-frame time """
        self.gst_sf = bs.pack('u12u20', gst_wn, gst_tow//30*30)
        return True

    def verify_root_key(self):
        """ verify root key """
        if not self.status & uOSNMA.ROOTKEY_LOADED:  # root key loaded
            return False
        lk_b = self.klen_t[self.ks]//8
        msg = bytearray([self.nma_header]) + self.dsm[1:13+lk_b]

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

    def verify_pdk(self, p_dk):
        """ verify P_DK """
        lk_b = self.klen_t[self.ks]//8
        msg = bytearray([self.nma_header]) + self.dsm[1:13+lk_b] + self.ds
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

    def decode_dsm_kroot(self):
        """ decode DSM-KROOT """
        v = bs.unpack_from('u4u4u2u2u2u2u4u4u8u4u12u8', self.dsm, 0)
        self.nb = v[0]+6   # number of blocks
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
        self.alp = self.dsm[7:13]  # random pattern alpha
        l_dk = self.nb*104
        l_ds = 512  # P-256/SHA-256
        l_k = self.klen_t[self.ks]
        l_pdk = l_dk-104-l_k-l_ds
        if l_pdk < 0:
            return False

        i = 13+l_k//8
        self.kroot = self.dsm[13:i]
        self.ds = self.dsm[i:i+l_ds//8]
        i += l_ds//8
        p_dk = self.dsm[i:i+(l_pdk+7)//8]
        if not self.verify_pdk(p_dk):
            print("p_dk verification error.")
            return False
        self.status |= uOSNMA.ROOTKEY_LOADED  # KROOT loaded
        return True

    def decode_dsm_pkr(self):
        """ decode DSM-PKR """
        nb, mid = bs.unpack_from('u4u4', self.dsm, 0)
        if nb < 7 or nb > 10:
            return False
        itn = self.dsm[1:1+128]  # 32*4
        npkt, npkid = bs.unpack_from('u4u4', self.dsm, 1024+8)
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
        p_dp = self.dsm[i0:i0+l_pdp//8]

        m0 = bytearray([self.dsm[129]])+npk  # NPKT||NPKID||NPK
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

    def decode_hk(self, hk):
        """ decode HKROOT message """
        self.nma_header = hk[0]
        nmas, cid, cpks, _ = bs.unpack_from('u2u2u3u1', hk, 0)
        did, bid = bs.unpack_from('u4u4', hk, 8)
        if nmas != 1 and nmas != 2:
            return False
        if cpks != 1:  # nominal only
            return False
        if self.cid0 < 0:
            self.cid0 = cid
            self.flg_dsm = 0
            self.nb = -1
        if cid != self.cid0:
            self.cid0 = cid
            self.did0 = -1
            self.flg_dsm = 0

        if self.did0 < 0:
            self.did0 = did
        if did != self.did0:
            self.did0 = did
            self.flg_dsm = 0

        self.dsm[bid*13:bid*13+13] = hk[2:]
        self.flg_dsm |= 1 << bid

        if bid == 0:
            nb_ = (hk[2] >> 4) & 0xf
            self.nb = nb_ + 6  # number of blocks

        result = False
        if self.nb > 0 and self.flg_dsm == (1 << self.nb)-1:
            if did <= 11:  # DSM-KROOT
                result = self.decode_dsm_kroot()
            else:  # DSM-PKR
                result = self.decode_dsm_pkr()

        if result:
            self.flg_dsm = 0
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
            macseq[1] &= 0xf0
            return tag0, macseq
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
            return tag, prn_d, adkd
        return False

    def verify_maclt(self):
        """ verify Tag sequence """
        if self.maclt not in self.self_t:
            return False
        self_t = self.self_t[self.maclt]
        adkd_t = self.adkd_t[self.maclt]
        if self.nt*2 != len(adkd_t):
            return False
        gst_wn, gst_tow = bs.unpack_from('u12u20', self.gst_sf, 0)
        ofst = 0 if gst_tow % 60 == 0 else 6
        for k in range(self.nt-1):
            tag, prn_d, adkd = self.decode_tags_info(k+1)
            i = k + ofst + 1
            if adkd != adkd_t[i] or (self_t[i] == 1 and prn_d != self.prn_a):
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
        msg = bytearray([self.prn_a])+self.gst_sf
        macseq_c_ = self.process_mac(msg)
        tag0, macseq_ = self.decode_tags_info(0)
        macseq_c = bs.unpack_from('u12', macseq_c_, 0)
        macseq = bs.unpack_from('u12', macseq_, 0)
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
        i0 = 60*(prn-1)
        mack_p = self.mack_p[i0:i0+60]  # previous MACK
        mack_c = self.mack_c[i0:i0+60]  # current MACK
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

    def load_nav(self, nav, prn):
        """ load navigation message into subframe buffer """
        mt = bs.unpack_from('u6', nav, 0)[0]
        if mt > 0 and mt <= 10:
            j = (mt-1)*16*8+160*8*(prn-1)
            for k in range(16):
                bs.pack_into('u8', self.subfrm, j, nav[k])
                j += 8
        return True

    def gen_navmsg(self, prn):
        """ generate nav message for nma """
        if prn < 1 or prn > self.GALMAX:
            return False
        j0 = 160*8*(prn-1)
        for k in range(5):
            mt = bs.unpack_from('u6', self.subfrm, j0+k*16*8)[0]
            if mt != k+1:
                return False

        iodnav1 = bs.unpack_from('u10', self.subfrm, j0+6+0*16*8)[0]
        for k in range(1, 4):
            iodnav_ = bs.unpack_from('u10', self.subfrm, j0+6+k*16*8)[0]
            if iodnav_ != iodnav1:
                if self.monlevel > 0:
                    print("error: iodnav mismatch mt=%d %d %d" %
                          (k+1, iodnav1, iodnav_))
                return False

        if self.monlevel > 0:
            svid = bs.unpack_from('u6', self.subfrm, j0+16+3*16*8)[0]
            wn, tow = bs.unpack_from('u12u20', self.subfrm, j0+73+4*16*8)
            print(" svid=%2d iodnav=%2d wn=%4d tow=%6d" %
                  (svid, iodnav1, wn, tow))

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
            return False
        if self.monlevel > 0:
            tow = bs.unpack_from('u20', mt6, 105)[0]
            bs.pack_into('u20', mt6, 105, tow+self.tofst)  # <- tow-=60
            print(" utc tow=%6d" % (tow))

        # 161b MT6 119b, MT10 42b
        msg = bytearray(21)
        j = 0
        for k in range(15):
            b = bs.unpack_from('u8', mt6, 6+k*8)[0]
            bs.pack_into('u8', msg, j, b)
            j += 8
        j = 119
        for k in range(5):
            b = bs.unpack_from('u8', mt10, k*8+86)[0]
            bs.pack_into('u8', msg, j, b)
            j += 8

        b = bs.unpack_from('u2', mt10, 40+86)[0]
        bs.pack_into('u2', msg, j, b)
        j += 2

        return msg

    def verify_navmsg(self, tag_):
        """ verify nav/utc message """
        prn_d = tag_.prn
        adkd = tag_.adkd

        if prn_d == -1:
            return False

        if adkd == 0 or adkd == 12:
            msg = tag_.navmsg
            if not msg:
                return False
        if adkd == 4:
            msg = self.gen_utcmsg()
            if not msg:
                return False
        mlen = 161 if adkd == 4 else 549
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
        bs.pack_into('r32u8u2', m, j, tag_.gst_sf,
                     tag_.cnt, self.nma_header >> 6)
        j += 42
        for k in range(len(msg)):
            b = msg[k]
            if k == len(msg)-1:
                if adkd == 4:
                    bs.pack_into('u1', m, j, b >> 7)
                    j += 1
                else:
                    bs.pack_into('u5', m, j, b >> 3)
                    j += 5
            else:
                bs.pack_into('u8', m, j, b)
                j += 8

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
        """ decode OSNMA message """
        status = False
        k = (tow % 30)//2
        if k == 0:  # reset counter
            self.cnt[prn-1] = 0
            self.hk[prn-1] = bytearray(15)
            self.mack[prn-1][0] = 0

        self.gst = tow
        self.gst_tow = (tow//30)*30
        # store sub-frame for NMA
        self.hk[prn-1][k] = nma_b[0]              # HK-ROOT message
        self.mack[prn-1][k*4:k*4+4] = nma_b[1:5]  # MACK message
        self.cnt[prn-1] |= (1 << k)
        if self.cnt[prn-1] == 0x7fff:  # all(0-14) message loaded
            self.save_mack(self.mack[prn-1], prn)  # store MACK
            # decode HK-ROOT messages
            result = self.decode_hk(self.hk[prn-1])
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
                self.set_gst_sf(wn, self.gst_tow)
                # key-chain verification
                # skip if key-chain is already verified at t=gst_sf
                if self.gst_sf != self.gst_sf_c:
                    self.status ^= uOSNMA.KEYCHAIN_VERIFIED
                if self.status & uOSNMA.ROOTKEY_VERIFIED and \
                        self.gst_sf != self.gst_sf_c:  # root-key verified
                    ki = ((wn-self.wn)*86400*7 +
                          (self.gst_tow-self.towh*3600))//30+1
                    result = self.verify_key_chain(
                        self.key_c, wn, self.gst_tow, ki)
                    if result:
                        self.gst_sf_c = self.gst_sf
                        self.key = self.key_c
                    if self.monlevel > 0:
                        s = "wn=%4d tow=%6d ki=%d prn=%d gst_tow=%d" % \
                            (wn, tow, ki, prn, self.gst_tow)
                        if result:
                            print("key chain verified     %s" % (s))

                        else:
                            print("key chain not verified %s" % (s))

                if self.status & uOSNMA.KEYCHAIN_VERIFIED:
                    # A1.6.2 Tag Sequence Verification
                    tow0 = self.gst_tow-30
                    self.set_gst_sf(wn, tow0)
                    result = self.verify_maclt()
                    if self.monlevel > 0:
                        s = "on %4d/%6d gst_tow=%d prn=%d" % \
                            (wn, tow, self.gst_tow-30, prn)
                        if result:
                            print("Tag Sequence verified %s" % (s))
                        else:
                            print("Tag Sequence not verified %s" % (s))
                    if not result:
                        return False
                    # A1.6.4 MACSEQ verification
                    # prn_a|gst_sf
                    result = self.verify_macseq()
                    if result and self.monlevel > 0:
                        print("MACSEQ Verified on %4d/%6d prn=%2d" %
                              (wn, tow, prn))
                    else:
                        print("MACSEQ not verified on %4d/%6d prn=%2d" %
                              (wn, tow, prn))
                        return False

                    # A1.6.5 Tags verification
                    for k in range(self.nt):
                        cnt = k+1
                        if k == 0:  # self-tag
                            prn_d = self.prn_a
                            adkd = 0
                            tag, macseq = self.decode_tags_info(0)
                        else:
                            tag, prn_d, adkd = self.decode_tags_info(k)
                            if adkd == 12:  # delayed tag loading
                                navmsg = self.gen_navmsg(prn_d)
                                tag_ = taginfo(
                                    self.gst_sf, prn_d, adkd, tag, cnt, navmsg)
                                self.tag_list.append(tag_)

                        if adkd == 12:
                            continue
                        if adkd == 0:
                            navmsg = self.gen_navmsg(prn_d)
                        else:
                            navmsg = None
                        tag_ = taginfo(self.gst_sf, prn_d,
                                       adkd, tag, cnt, navmsg)
                        result = self.verify_navmsg(tag_)
                        status = self.chk_nav(result, tag_)
                        if status and adkd == 4:
                            self.status |= uOSNMA.UTC_VERIFIED
                        if self.monlevel > 0:
                            if result:
                                print("# %d prn_d=%2d adkd=%2d tag verified"
                                      % (cnt, prn_d, adkd))
                            else:
                                print("%d prn_d=%2d adkd=%2d tag not verified"
                                      % (cnt, prn_d, adkd))
                    # slow MAC
                    for tag_ in self.tag_list:
                        dt = self.difftime(self.gst_sf, tag_.gst_sf)
                        if dt == 300:
                            result = self.verify_navmsg(tag_)
                            if self.monlevel > 0 and result:
                                print("# %d prn_d=%2d adkd=%2d tag verified"
                                      % (tag_.cnt, tag_.prn, tag_.adkd))
                        elif dt > 300:
                            tag_ = None

            self.cnt[prn-1] = 0
        return status
