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

    def selsat(self, obs, obsb, elb):
        """ select common satellite between rover and base station """
        # exclude satellite with missing observation and cycle slip for rover
        idx_u = []
        for k, _ in enumerate(obs.sat):
            if obs.P[k, 0] == 0.0 or obs.P[k, 1] == 0.0 or \
               obs.L[k, 0] == 0.0 or obs.L[k, 1] == 0.0 or \
               obs.lli[k, 0] > 0 or obs.lli[k, 1] > 0:
                continue
            idx_u.append(k)

        # exclude satellite with missing observation and cycle slip for base
        idx_r = []
        for k, _ in enumerate(obsb.sat):
            if obsb.P[k, 0] == 0.0 or obsb.P[k, 1] == 0.0 or \
               obsb.L[k, 0] == 0.0 or obsb.L[k, 1] == 0.0 or \
               obsb.lli[k, 0] > 0 or obsb.lli[k, 1] > 0 or \
               elb[k] < self.nav.elmin:
                continue
            idx_r.append(k)

        idx = np.intersect1d(
            obs.sat[idx_u], obsb.sat[idx_r], return_indices=True)
        k = len(idx[0])
        iu = np.array(idx_u)[idx[1]]
        ir = np.array(idx_r)[idx[2]]
        return k, iu, ir

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        RTK positioning
        """

        # Skip empty epochs
        #
        if len(obs.sat) == 0:
            return

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        if nsat < 6:
            print(" too few satellites < 6: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        if obsb is not None:
            rsb, vsb, dtsb, svhb, _ = satposs(obsb, self.nav)
            yr, er, elr = self.zdres(
                obsb, cs, bsx, rsb, vsb, dtsb, self.nav.rb, 0)
            ns, iu, ir = self.selsat(obs, obsb, elr)

            y = np.zeros((ns*2, self.nav.nf*2))
            e = np.zeros((ns*2, 3))

            y[ns:, :] = yr[ir, :]
            e[ns:, :] = er[ir, :]

            obs_ = deepcopy(obs)
            obs_.sat = obs.sat[iu]
            obs_.L = obs.L[iu, :]-obsb.L[ir, :]
            obs_.P = obs.P[iu, :]-obsb.P[ir, :]
        else:
            # Select satellites having passed quality control
            #
            # index of valid sats in obs.sat
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            ns = len(iu)
            y = np.zeros((ns, self.nav.nf*2))
            e = np.zeros((ns, 3))

            obs_ = obs

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        self.udstate(obs_)

        xa = np.zeros(self.nav.nx)
        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])

        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 6 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 6:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 5
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        # Non-differential residuals after measurement update
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])
        y = yu[iu, :]
        e = eu[iu, :]
        ny = y.shape[0]
        if ny < 6:
            return -1

        # Residuals for float solution
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        if self.valpos(v, R):
            self.nav.x = xp
            self.nav.P = Pp
            self.nav.ns = 0
            for i in range(ns):
                j = sat[i]-1
                for f in range(self.nav.nf):
                    if self.nav.vsat[j, f] == 0:
                        continue
                    self.nav.outc[j, f] = 0
                    if f == 0:
                        self.nav.ns += 1
        else:
            self.nav.smode = 0

        self.nav.smode = 5  # 4: fixed ambiguities, 5: float ambiguities

        if self.nav.armode > 0:
            nb, xa = self.resamb_lambda(sat)
            if nb > 0:
                # Use position with fixed ambiguities xa
                yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xa[0:3])
                y = yu[iu, :]
                e = eu[iu, :]
                v, H, R = self.sdres(obs, xa, y, e, sat, el)
                # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
                if self.valpos(v, R):
                    if self.nav.armode == 3:     # fix and hold
                        self.holdamb(xa)    # hold fixed ambiguity
                    self.nav.smode = 4           # fix

        # Store epoch for solution
        #
        self.nav.t = obs.t

        return 0
