"""
 test of RINEX decoder
"""

from cssrlib.rinex import rnxdec
from cssrlib.gnss import uTYP, rSigRnx
from cssrlib.gnss import sat2id, sat2prn

from os.path import expanduser

obsfile = '~/GNSS_OBS/IGS/DAILY/2021/078/CHOF00JPN_S_20210780000_01D_30S_MO.rnx'
#obsfile = '../data/SEPT078M.21O'

sigs = [rSigRnx("GC1C"), rSigRnx("EC1X"),
        rSigRnx("GC2W"), rSigRnx("EC5X"),
        rSigRnx("GL1C"), rSigRnx("EL1X"),
        rSigRnx("GL2W"), rSigRnx("EL5X")]

dec = rnxdec()
dec.setSignals(sigs)

nep = 1
if dec.decode_obsh(expanduser(obsfile)) >= 0:

    for ne in range(nep):

        obs = dec.decode_obs()

        for i, sat in enumerate(obs.sat):

            txt = "{} ".format(sat2id(sat))

            sys, _ = sat2prn(sat)
            sigs = dec.getSignals(sys, uTYP.C)
            for j, sig in enumerate(sigs):
                txt += "{} {:13.3f}  ".format(sig.str(), obs.P[i, j])
            sigs = dec.getSignals(sys, uTYP.L)
            for j, sig in enumerate(sigs):
                txt += "{} {:13.3f}  ".format(sig.str(), obs.L[i, j])

            print(txt)

    print()

    dec.fobs.close()
