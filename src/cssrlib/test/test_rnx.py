"""
 test of RINEX decoder
"""

from cssrlib.rinex import rnxdec
from cssrlib.gnss import rSigRnx
from cssrlib.gnss import sat2id, time2str

from os.path import expanduser

obsfile = '~/GNSS_OBS/IGS/DAILY/2021/078/CHOF00JPN_S_20210780000_01D_30S_MO.rnx'
#obsfile = '../data/SEPT078M1.21O'

sigs = [rSigRnx("GC1C"), rSigRnx("EC1X"),
        rSigRnx("GC2W"), rSigRnx("EC5X"),
        rSigRnx("GL1C"), rSigRnx("EL1X"),
        rSigRnx("GL2W"), rSigRnx("EL5X"),
        rSigRnx("GS1C"), rSigRnx("ES1X"),
        rSigRnx("GS2W"), rSigRnx("ES5X")]

dec = rnxdec()
dec.setSignals(sigs)

nep = 2
if dec.decode_obsh(expanduser(obsfile)) >= 0:

    for ne in range(nep):

        obs = dec.decode_obs()

        print("Epoch {}".format(time2str(obs.t)))

        for i, sat in enumerate(obs.sat):

            txt = "{} ".format(sat2id(sat))

            for sig, val in obs.P[sat].items():

                if val is None:
                    txt += "{} {:13s}  ".format(sig.str(), "")
                else:
                    txt += "{} {:13.3f}  ".format(sig.str(), val)

            for sig, val in obs.L[sat].items():

                if val is None:
                    txt += "{} {:13s}  ".format(sig.str(), "")
                else:
                    txt += "{} {:13.3f}  ".format(sig.str(), val)

            for sig, val in obs.S[sat].items():

                if val is None:
                    txt += "{} {:7s}  ".format(sig.str(), "")
                else:
                    txt += "{} {:7.3f}  ".format(sig.str(), val)

            print(txt)

        print()

    dec.fobs.close()
