"""
Galileo HAS correction data decoder

[1] Galileo High Accuracy Service Signal-in-Space
  Interface Control Document (HAS SIS ICD), Issue 1.0, May 2022

"""

import numpy as np
import bitstruct as bs
import galois
from cssrlib.cssrlib import cssr, sCSSR, sCSSRTYPE
from cssrlib.gnss import gpst2time


class cssr_has(cssr):
    def __init__(self, foutname=None):
        super().__init__(foutname)
        self.MAXNET = 1
        self.cssrmode = sCSSRTYPE.GAL_HAS_SIS
        self.dorb_scl = [0.0025, 0.0080, 0.0080]
        self.dclk_scl = 0.0025
        self.dorb_blen = [13, 12, 12]
        self.dclk_blen = 13
        self.cb_blen = 11
        self.cb_scl = 0.02
        self.pb_blen = 11
        self.pb_scl = 0.01  # cycles

        self.GF = galois.GF(256)

    def sval(self, u, n, scl):
        """ calculate signed value based on n-bit int, lsb """
        invalid = -2**(n-1)
        dnu = 2**(n-1)-1
        y = np.nan if u == invalid or u == dnu else u*scl
        return y

    def decode_cssr_clk_sub(self, msg, i=0):
        head, i = self.decode_head(msg, i)
        self.flg_net = False
        if self.iodssr != head['iodssr']:
            return -1

        nclk_sub = bs.unpack_from('u4', msg, i)[0]
        i += 4
        for j in range(nclk_sub):
            gnss, dcm = bs.unpack_from('u4u2', msg, i)
            i += 6
            idx_s = np.where(np.array(self.gnss_n) == gnss)
            mask_s = bs.unpack_from('u'+str(self.nsat_g[gnss]), msg, i)[0]
            i += self.nsat_g[gnss]
            idx, nclk = self.decode_mask(mask_s, self.nsat_g[gnss])
            for k in range(nclk):
                dclk = bs.unpack_from('s'+str(self.dclk_blen), msg, i) \
                    * self.dclk_scl*(dcm+1)
                self.lc[0].dclk[idx_s[idx[k]]] = dclk
                i += self.dclk_blen
        return i

    def decode_head(self, msg, i, st=-1):

        if st == sCSSR.MASK:
            ui = 0
        else:
            ui = bs.unpack_from('u4', msg, i)[0]
            i += 4

        if self.tow0 >= 0:
            self.tow = self.tow0+self.toh
            if self.week >= 0:
                self.time = gpst2time(self.week, self.tow)

        head = {'uint': ui, 'mi': 0, 'iodssr': self.iodssr}
        return head, i

    def decode_cssr(self, msg, i=0):
        if self.msgtype != 1:  # only MT=1 defined
            print(f"invalid MT={self.msgtype}")
            return False
        # time of hour
        # flags: mask,orbit,clock,clock subset,cbias,pbias,mask_id,iodset_id
        self.toh, flags, res, mask_id, self.iodssr = \
            bs.unpack_from('u12u6u4u5u5', msg, i)
        i += 32

        if self.monlevel > 0 and self.fh is not None:
            self.fh.write("##### Galileo HAS SSR: TOH{:6d} flags={:12s} mask_id={:2d} iod_s={:1d}\n"
                          .format(self.toh, bin(flags), mask_id, self.iodssr))

        if self.toh >= 3600:
            print(f"invalid TOH={self.toh}")
            return False

        if (flags >> 5) & 1:  # mask block
            self.mask_id = mask_id
            self.subtype = sCSSR.MASK
            i = self.decode_cssr_mask(msg, i)
        if (flags >> 4) & 1:  # orbit block
            self.subtype = sCSSR.ORBIT
            i = self.decode_cssr_orb(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        if (flags >> 3) & 1:  # clock block
            self.mask_id_clk = mask_id
            self.subtype = sCSSR.CLOCK
            i = self.decode_cssr_clk(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        if (flags >> 2) & 1:  # clock subset block
            i = self.decode_cssr_clk_sub(msg, i)
        if (flags >> 1) & 1:  # code bias block
            self.subtype = sCSSR.CBIAS
            i = self.decode_cssr_cbias(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()
        if (flags >> 0) & 1:  # phase bias block
            self.subtype = sCSSR.PBIAS
            i = self.decode_cssr_pbias(msg, i)
            if self.monlevel > 0 and self.fh is not None:
                self.out_log()

    def decode_has_header(self, buff, i):
        if bs.unpack_from('u24', buff, i)[0] == 0xaf3bc3:
            return 0, 0, 0, 0, 0

        hass, res, mt, mid, ms, pid = bs.unpack_from('u2u2u2u5u5u8', buff, i)
        ms += 1
        return hass, mt, mid, ms, pid

    def decode_has_page(self, idx, has_pages, gMat, ms):
        """ HPVRS decoding for RS(255,32,224) """
        HASmsg = bytes()
        k = len(idx)
        if k >= ms:
            Wd = self.GF(has_pages[idx, :])  # kx53
            Dinv = np.linalg.inv(self.GF(gMat[idx, :k]))  # kxk
            Md = Dinv@Wd  # decoded message (kx53)
            HASmsg = np.array(Md).tobytes()

        return HASmsg
