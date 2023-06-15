"""
 test of RINEX decoder
"""

from datetime import datetime

from cssrlib.rinex import rnxdec
from cssrlib.gnss import uTYP, rSigRnx
from cssrlib.gnss import sat2id, sat2prn

obsfile = '../data/SEPT078M1.21O'

sigs = [rSigRnx("GC1C"), rSigRnx("GC2W"),
        rSigRnx("GL1C"), rSigRnx("GL2W"),
        rSigRnx("GS1C"), rSigRnx("GS2W"),
        rSigRnx("EC1X"), rSigRnx("EC5X"),
        rSigRnx("EL1X"), rSigRnx("EL5X"),
        rSigRnx("ES1X"), rSigRnx("ES5X"),
        rSigRnx("JC1C"), rSigRnx("JC2S"),
        rSigRnx("JL1C"), rSigRnx("JL2S"),
        rSigRnx("JS1C"), rSigRnx("JS2S")]


dec = rnxdec()
dec.setSignals(sigs)

nep = 2
if dec.decode_obsh(obsfile) >= 0:

    dec.autoSubstituteSignals()

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
