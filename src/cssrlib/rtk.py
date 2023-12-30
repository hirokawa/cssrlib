"""
module for RTK positioning

"""

from cssrlib.pppssr import pppos
import numpy as np
from copy import deepcopy
from cssrlib.ephemeris import satposs


class rtkpos(pppos):
    """ class for RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        """ initialize variables for PPP-RTK """

        # trop, iono from cssr
        # phase windup model is local/regional
        super().__init__(nav=nav, pos0=pos0, logfile=logfile,
                         trop_opt=0, iono_opt=0, phw_opt=0)

        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma
        self.nav.sig_p0 = 30.0  # [m]
        self.nav.thresar = 2.0  # AR acceptance threshold
        self.nav.armode = 1     # AR is enabled

    def base_process(self, obs, obsb, rs, dts, svh):
        """ processing for base station in RTK """
        rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)
        yr, er, elr = self.zdres(
            obsb, None, None, rsb, vsb, dtsb, self.nav.rb, 0)

        # Editing observations (base/rover)
        sat_ed_r = self.qcedit(obsb, rsb, dtsb, svhb, rr=self.nav.rb)
        sat_ed_u = self.qcedit(obs, rs, dts, svh)

        # define common satellite between base and rover
        sat_ed = np.intersect1d(sat_ed_u, sat_ed_r, True)
        ir = np.intersect1d(obsb.sat, sat_ed, True, True)[1]
        iu = np.intersect1d(obs.sat, sat_ed, True, True)[1]
        ns = len(iu)

        y = np.zeros((ns*2, self.nav.nf*2))
        e = np.zeros((ns*2, 3))

        y[ns:, :] = yr[ir, :]
        e[ns:, :] = er[ir, :]

        obs_ = deepcopy(obs)
        obs_.sat = obs.sat[iu]
        obs_.L = obs.L[iu, :]-obsb.L[ir, :]
        obs_.P = obs.P[iu, :]-obsb.P[ir, :]

        return y, e, iu, obs_
