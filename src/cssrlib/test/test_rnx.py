"""
 test of RINEX decoder
"""

from datetime import datetime

from cssrlib.rinex import rnxdec
from cssrlib.gnss import uTYP, rSigRnx
from cssrlib.gnss import sat2id, sat2prn

obsfile = '../data/SEPT078M1.21O'

sigs = [rSigRnx("GC1C"), rSigRnx("EC1X"),
        rSigRnx("GC2W"), rSigRnx("EC5X"),
        rSigRnx("GL1C"), rSigRnx("EL1X"),
        rSigRnx("GL2W"), rSigRnx("EL5X"),
        rSigRnx("GS1C"), rSigRnx("ES1X"),
        rSigRnx("GS2W"), rSigRnx("ES5X")]

dec = rnxdec()
dec.setSignals(sigs)

nep = 2
if dec.decode_obsh(obsfile) >= 0:

    for ne in range(nep):

        obs = dec.decode_obs()

        print("{:%Y-%m-%d %T}".format(datetime.utcfromtimestamp(obs.t.time)))

        for i, sat in enumerate(obs.sat):

            txt = "{} ".format(sat2id(sat))

            sys, _ = sat2prn(sat)
            sigs = obs.sig[sys][uTYP.C]
            for j, sig in enumerate(sigs):
                txt += "{} {:13.3f}  ".format(sig.str(), obs.P[i, j])

            sigs = obs.sig[sys][uTYP.L]
            for j, sig in enumerate(sigs):
                txt += "{} {:13.3f}  ".format(sig.str(), obs.L[i, j])

            sigs = obs.sig[sys][uTYP.S]
            for j, sig in enumerate(sigs):
                txt += "{} {:7.3f}  ".format(sig.str(), obs.S[i, j])

            print(txt)

        print()

    dec.fobs.close()
