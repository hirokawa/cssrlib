"""
module for PPP-RTK positioning
"""

import numpy as np
import cssrlib.gnss as gn
from cssrlib.gnss import tropmodel, uGNSS, uTropoModel, timediff, sat2prn
from cssrlib.gnss import rCST, uTYP, sat2id, time2str, ecef2pos
from cssrlib.gnss import gpst2utc, rSigRnx, geodist, tropmapf
from cssrlib.gnss import satazel
from cssrlib.ephemeris import satposs
from cssrlib.cssrlib import sCType
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.ppp import tidedisp, shapiro, windupcorr
from cssrlib.rtk import ddres
from cssrlib.pppssr import pppos
from cssrlib.cssrlib import sCSSRTYPE as sc


class ppprtkpos(pppos):
    """ class for PPP-RTK processing """

    nav = None

    def __init__(self, nav, pos0=np.zeros(3), logfile=None):
        """ initialize variables for PPP """

        self.nav = nav

        self.nav.pmode = 1  # kinematic

        # Number of frequencies (actually signals!)
        #
        self.nav.ephopt = 2  # SSR-APC

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Position (+ optional velocity)
        #
        self.nav.na = 3 if nav.pmode == 0 else 6
        self.nav.nq = 3 if nav.pmode == 0 else 6

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na+uGNSS.MAXSAT*self.nav.nf

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.phw = np.zeros(uGNSS.MAXSAT)
        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Parameters for PPP
        #
        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*50  # [-] factor
        self.nav.err = [0, 0.01, 0.005]/np.sqrt(2)  # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 30.0  # [m]
        self.nav.sig_v0 = 1.0     # [m/s]
        self.nav.sig_n0 = 30.0    # [cyc]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 100.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]

        self.nav.tidecorr = True
        self.nav.thresar = 2.0  # AR acceptance threshold
        # 0:float-ppp,1:continuous,2:instantaneous,3:fix-and-hold
        self.nav.armode = 1
        self.nav.elmaskar = np.deg2rad(20.0)  # elevation mask for AR
        self.nav.elmin = np.deg2rad(10.0)

        # Initial state vector
        #
        self.nav.x[0:3] = pos0
        if self.nav.pmode >= 1:  # kinematic
            self.nav.x[3:6] = 0.0  # velocity

        # Diagonal elements of covariance matrix
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True

        dP[0:3] = self.nav.sig_p0**2
        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            dP[3:6] = self.nav.sig_v0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2
        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2

        # Logging level
        #
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def find_bias(self, cs, sigref, sat, inet=0):
        nf = len(sigref)
        v = np.zeros(nf)

        if nf == 0:
            return v

        ctype = sigref[0].typ
        if ctype == uTYP.C:
            if cs.lc[inet].cbias is None or \
                    sat not in cs.lc[inet].cbias.keys():
                return v
            sigc = cs.lc[inet].cbias[sat]
        else:
            if cs.lc[inet].pbias is None or \
                    sat not in cs.lc[inet].pbias.keys():
                return v
            sigc = cs.lc[inet].pbias[sat]

        # work-around for Galileo HAS: L2P -> L2W
        if cs.cssrmode in [sc.GAL_HAS_SIS, sc.GAL_HAS_IDD]:
            if ctype == uTYP.C and rSigRnx('GC2P') in sigc.keys():
                sigc[rSigRnx('GC2W')] = sigc[rSigRnx('GC2P')]
            if ctype == uTYP.L and rSigRnx('GL2P') in sigc.keys():
                sigc[rSigRnx('GL2W')] = sigc[rSigRnx('GL2P')]

        for k, sig in enumerate(sigref):
            if sig in sigc.keys():
                v[k] = sigc[sig]
            elif sig.toAtt('X') in sigc.keys():
                v[k] = sigc[sig.toAtt('X')]
        return v

    def udstate(self, obs):
        """ time propagation of states and initialize """

        tt = timediff(obs.t, self.nav.t)

        ns = len(obs.sat)
        sys = []
        sat = obs.sat
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel
        na = self.nav.na
        Phi = np.eye(self.nav.na)
        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
        self.nav.P[0:na, 0:na] = Phi@self.nav.P[0:na, 0:na]@Phi.T

        # Process noise
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*tt

        # Update Kalman filter state elements
        #
        for f in range(self.nav.nf):

            # Reset phase-ambiguity if instantaneous AR
            # or expire obs outage counter
            #
            for i in range(gn.uGNSS.MAXSAT):

                sat_ = i+1
                sys_i, _ = sat2prn(sat_)

                self.nav.outc[i, f] += 1
                reset = (self.nav.outc[i, f] >
                         self.nav.maxout or np.any(self.nav.edt[i, :] > 0))
                if sys_i not in obs.sig.keys():
                    continue

                # Reset ambiguity estimate
                #
                j = self.IB(sat_, f, self.nav.na)
                if reset and self.nav.x[j] != 0.0:
                    self.initx(0.0, 0.0, j)
                    self.nav.outc[i, f] = 0

                    if self.nav.monlevel > 0:
                        self.nav.fout.write(
                            "{}  {} - reset ambiguity  {}\n"
                            .format(time2str(obs.t), sat2id(sat_),
                                    obs.sig[sys_i][uTYP.L][f]))

            bias = np.zeros(ns)

            for i in range(ns):
                # Do not initialize invalid observations
                #
                if np.any(self.nav.edt[sat[i]-1, :] > 0):
                    continue

                # Get pseudorange and carrier-phase observation of signal f
                #
                sig = obs.sig[sys[i]][uTYP.L][f]

                if sys[i] == uGNSS.GLO:
                    fi = sig.frequency(self.nav.glo_ch[sat[i]])
                else:
                    fi = sig.frequency()

                lam = rCST.CLIGHT/fi

                cp = obs.L[i, f]
                pr = obs.P[i, f]
                if cp == 0.0 or pr == 0.0 or lam is None:
                    continue

                bias[i] = cp - pr/lam

            # initialize ambiguity
            #
            for i in range(ns):

                sys_i, _ = sat2prn(sat[i])

                j = self.IB(sat[i], f, self.nav.na)
                if bias[i] != 0.0 and self.nav.x[j] == 0.0:
                    self.initx(bias[i], self.nav.sig_n0**2, j)

                    if self.nav.monlevel > 0:
                        sig = obs.sig[sys_i][uTYP.L][f]
                        self.nav.fout.write(
                            "{}  {} - init  ambiguity  {} {:12.3f}\n"
                            .format(time2str(obs.t), sat2id(sat[i]),
                                    sig, bias[i]))

        return 0

    def zdres(self, obs, cs, bsx, rs, vs, dts, rr):
        """ non-differential residual """

        _c = rCST.CLIGHT

        nf = self.nav.nf
        n = len(obs.P)
        y = np.zeros((n, nf*2))
        el = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()

        # Solid Earth tide corrections
        #
        # TODO: add solid earth tide displacements
        #
        if self.nav.tidecorr:
            pos = ecef2pos(rr_)
            disp = tidedisp(gpst2utc(obs.t), pos)
        else:
            disp = np.zeros(3)
        rr_ += disp

        inet = cs.find_grid_index(pos)
        dlat, dlon = cs.get_dpos(pos)
        trph, trpw = cs.get_trop(dlat, dlon)

        trop_hs, trop_wet, _ = tropmodel(obs.t, pos,
                                         model=self.nav.trpModel)
        trop_hs0, trop_wet0, _ = tropmodel(obs.t, [pos[0], pos[1], 0],
                                           model=self.nav.trpModel)
        r_hs = trop_hs/trop_hs0
        r_wet = trop_wet/trop_wet0

        stec = cs.get_stec(dlat, dlon)

        cpc = np.zeros((n, nf))
        prc = np.zeros((n, nf))

        for i in range(n):

            sat = obs.sat[i]
            sys, _ = gn.sat2prn(sat)

            # Skip edited observations
            #
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            if sat not in cs.lc[inet].sat_n:
                continue

            # Pseudorange, carrier-phase and C/N0 signals
            #
            sigsPR = obs.sig[sys][uTYP.C]
            sigsCP = obs.sig[sys][uTYP.L]

            # Wavelength
            #
            if sys == uGNSS.GLO:
                lam = np.array([s.wavelength(self.nav.glo_ch[sat])
                                for s in sigsCP])
            else:
                lam = np.array([s.wavelength() for s in sigsCP])

            cbias = np.zeros(self.nav.nf)
            pbias = np.zeros(self.nav.nf)

            idx_l = cs.lc[inet].sat_n.index(sat)

            frq = np.array([s.frequency() for s in sigsCP])
            lam = np.array([s.wavelength() for s in sigsCP])
            iono = np.array([40.3e16/(f*f)*stec[idx_l] for f in frq])

            if cs.lc[0].cstat & (1 << sCType.CBIAS) == (1 << sCType.CBIAS):
                cbias += self.find_bias(cs, sigsPR, sat)
            if inet > 0 and cs.lc[inet].cstat & (1 << sCType.CBIAS) == \
                    (1 << sCType.CBIAS):
                cbias += self.find_bias(cs, sigsPR, sat, inet)

            if cs.lc[0].cstat & (1 << sCType.PBIAS) == (1 << sCType.PBIAS):
                pbias += self.find_bias(cs, sigsCP, sat)
            if inet > 0 and cs.lc[inet].cstat & (1 << sCType.PBIAS) == \
                    (1 << sCType.PBIAS):
                pbias += self.find_bias(cs, sigsCP, sat, inet)

            # Check for invalid biases
            #
            if np.isnan(cbias).any() or np.isnan(pbias).any():
                if self.nav.monlevel > 3:
                    print("skip invalid cbias/pbias for sat={:d}".format(sat))
                continue

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            r, e[i, :] = geodist(rs[i, :], rr_)
            _, el[i] = satazel(pos, e[i, :])
            if el[i] < self.nav.elmin:
                continue

            # Shapiro relativistic effect
            #
            relatv = shapiro(rs[i, :], rr_)

            # Tropospheric delay mapping functions
            #
            mapfh, mapfw = tropmapf(obs.t, pos, el[i],
                                    model=self.nav.trpModel)

            # Tropospheric delay
            #
            trop = mapfh*trph*r_hs+mapfw*trpw*r_wet

            # Phase wind-up effect
            #
            self.nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :], rr_,
                                             self.nav.phw[sat-1])
            phw = lam*self.nav.phw[sat-1]

            antrPR = antModelRx(self.nav, pos, e[i, :], sigsPR)
            antrCP = antModelRx(self.nav, pos, e[i, :], sigsCP)

            antsPR = [0.0 for _ in sigsPR]
            antsCP = [0.0 for _ in sigsCP]

            # Range correction
            #
            prc[i, :] = trop + antrPR + antsPR + iono + cbias
            cpc[i, :] = trop + antrCP + antsCP - iono + pbias + phw

            r += relatv - _c*dts[i]

            for f in range(nf):
                y[i, f] = obs.L[i, f]*lam[f]-(r+cpc[i, f])
                y[i, f+nf] = obs.P[i, f]-(r+prc[i, f])

        return y, e, el

    def process(self, obs, cs=None, orb=None, bsx=None):
        """
        PPP-RTK positioning
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

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs)

        xa = np.zeros(self.nav.nx)
        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp[0:3])

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        iu = np.where(np.isin(obs.sat, sat_ed))[0]
        sat = obs.sat[iu]
        y = yu[iu, :]
        e = eu[iu, :]
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
        # v, H, R = self.sdres(obs, xp, y, e, sat, el)
        v, H, R = ddres(self.nav, obs, xp, y, e, sat, el)
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
        # v, H, R = self.sdres(obs, xp, y, e, sat, el)
        v, H, R = ddres(self.nav, obs, xp, y, e, sat, el)
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
                # v, H, R = self.sdres(obs, xa, y, e, sat, el)
                v, H, R = ddres(self.nav, obs, xa, y, e, sat, el)
                # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
                if self.valpos(v, R):
                    if self.nav.armode == 3:     # fix and hold
                        self.holdamb(xa)    # hold fixed ambiguity
                    self.nav.smode = 4           # fix

        # Store epoch for solution
        #
        self.nav.t = obs.t

        return 0
