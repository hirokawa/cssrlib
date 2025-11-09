"""
Utility functions for cssrlib

@author: ruihi
"""

from sys import stdout
from copy import deepcopy
import matplotlib.pyplot as plt
import numpy as np
from cssrlib.gnss import time2str, sys2str, ecef2pos, ecef2enu, timediff
from cssrlib.gnss import gtime_t, char2sys, rSigRnx, uGNSS, uTYP, uSIG
from cssrlib.peph import atxdec, searchpcv
from cssrlib.plot import plot_enu


class process:
    """ class to process the positioning """

    def __init__(self, nav=None, rnx=None, config=None, nep=0, xyz_ref=[]):
        self.nav = nav
        self.rnx = rnx
        self.config = config
        self.nep = nep
        self.t0 = gtime_t()

        if len(xyz_ref) > 0:
            self.xyz_ref = xyz_ref
            self.pos_ref = ecef2pos(self.xyz_ref)

        if nep > 0:
            # Initialize data structures for results
            #
            self.t = np.zeros(nep)
            self.enu = np.ones((nep, 3))*np.nan
            self.ztd = np.zeros((nep, 1))
            self.smode = np.zeros(nep, dtype=int)

    def init_time(self, t):
        """ initialize time """
        self.nav.t = deepcopy(t)
        self.t0 = deepcopy(t)
        self.t0.time = self.t0.time//30*30
        self.nav.time_p = self.t0

    def init_sig(self, sig_t):
        """ convert gnss/signals table to signals """
        # sig_t = {'G': ['1C', '5Q'], 'J': ['1C', '5Q']}
        sigs = []
        nf = 0
        for gnss, sig_g in sig_t.items():
            nf = len(sig_g)
            for sig in sig_g:
                sig_ = rSigRnx(gnss+'C'+sig)
                sigs.append(sig_)
                sig_ = rSigRnx(gnss+'L'+sig)
                sigs.append(sig_)
                sig_ = rSigRnx(gnss+'S'+sig)
                sigs.append(sig_)

        return sigs, nf

    def prepare_signal(self, obsfile=None):
        """ prepare signals """

        self.rnx.autoSubstituteSignals()  # auto-substitute signals

        # Initialize navigation parameters
        #
        self.nav.elmin = np.deg2rad(self.config['elmin'])
        self.nav.csmooth = self.config['nav']['csmooth']
        self.nav.pmode = self.config['nav']['pmode']

        # Logging level
        #
        self.nav.monlevel = self.config['nav']['monlevel']

        self.nav.glo_ch = self.rnx.glo_ch

        # Get equipment information
        #
        if obsfile is not None:
            self.nav.fout.write("FileName: {}\n".format(obsfile))

        self.nav.fout.write("Start   : {}\n".format(time2str(self.rnx.ts)))
        if self.rnx.te is not None:
            self.nav.fout.write("End     : {}\n".format(time2str(self.rnx.te)))
        self.nav.fout.write("Receiver: {}\n".format(self.rnx.rcv))
        self.nav.fout.write("Antenna : {}\n".format(self.rnx.ant))
        self.nav.fout.write("\n")

        if 'UNKNOWN' in self.rnx.ant or self.rnx.ant.strip() == "":
            self.nav.fout.write(
                "ERROR: missing antenna type in RINEX OBS header!\n")

        # Load ANTEX data for satellites and stations
        #
        atx = atxdec()
        atx.readpcv(self.config['atxfile'])

        # Set PCO/PCV information
        #
        self.nav.sat_ant = atx.pcvs
        self.nav.rcv_ant = searchpcv(atx.pcvr, self.rnx.ant,  self.rnx.ts)
        if self.nav.rcv_ant is None:
            self.nav.fout.write("ERROR: missing antenna type <{}> in ANTEX file!\n"
                                .format(self.rnx.ant))

        # Print available signals
        #
        self.nav.fout.write("Available signals\n")
        for sys, sigs in self.rnx.sig_map.items():
            txt = "{:7s} {}\n".format(sys2str(sys), ' '.
                                      join([sig.str() for sig in sigs.values()]))
            self.nav.fout.write(txt)
        self.nav.fout.write("\n")

        self.nav.fout.write("Selected signals\n")
        for sys, tmp in self.rnx.sig_tab.items():
            txt = "{:7s} ".format(sys2str(sys))
            for _, sigs in tmp.items():
                txt += "{} ".format(' '.join([sig.str() for sig in sigs]))
            self.nav.fout.write(txt+"\n")
        self.nav.fout.write("\n")

    def save_output(self, t, ne, ppp=None):
        """ output result """

        self.t[ne] = timediff(self.nav.t, self.t0)/86400.0

        sol = self.nav.xa[0:3] if self.nav.smode == 4 else self.nav.x[0:3]
        self.enu[ne, :] = ecef2enu(self.pos_ref, sol-self.xyz_ref)
        self.smode[ne] = self.nav.smode

        if ppp is not None:
            self.ztd[ne] = self.nav.xa[ppp.IT(self.nav.na)] \
                if self.nav.smode == 4 else self.nav.x[ppp.IT(self.nav.na)]

        enu = self.enu[ne, :]
        smode = self.smode[ne]

        pos_h = np.sqrt(enu[0]**2+enu[1]**2)

        self.nav.fout.write("{} {:14.4f} {:14.4f} {:14.4f} "
                            "ENU {:7.3f} {:7.3f} {:7.3f}, 2D {:6.3f}, mode {:1d}\n"
                            .format(time2str(t),
                                    sol[0], sol[1], sol[2],
                                    enu[0], enu[1], enu[2], pos_h, smode))

        # Log to standard output
        #
        stdout.write('\r {} ENU {:7.3f} {:7.3f} {:7.3f}, '
                     '2D {:6.3f}, mode {:1d}'
                     .format(time2str(t),
                             enu[0], enu[1], enu[2], pos_h, smode))

    def close(self):
        """ finish processing """

        # Send line-break to stdout
        #
        stdout.write('\n')

        # Close RINEX observation file
        #
        self.rnx.fobs.close()

        # Close output file
        #
        if self.nav.fout is not None:
            self.nav.fout.close()

    def plot(self, ttl='test', fig_type=1, ylim=1, ylim_v=2, gfmt='png',
             dpi=300):
        """ plot utility """

        if fig_type == 1:
            plot_enu(self.t, self.enu, self.smode, ylim=ylim, ylim_v=ylim_v)
        elif fig_type == 2:
            plot_enu(self.t, self.enu, self.smode, figtype=fig_type, ylim=ylim)

        plotFileFormat = gfmt
        plotFileName = '.'.join((ttl, plotFileFormat))

        plt.savefig(plotFileName, format=plotFileFormat,
                    bbox_inches='tight', dpi=dpi)
        # plt.show()
