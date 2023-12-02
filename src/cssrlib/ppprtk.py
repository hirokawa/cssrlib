"""
module for PPP-RTK positioning
"""

import numpy as np
import cssrlib.gnss as gn
from cssrlib.gnss import tropmodel, uGNSS, uTropoModel, timediff, sat2prn
from cssrlib.gnss import rCST, uTYP, sat2id, time2str, ecef2pos, uSIG
from cssrlib.gnss import gpst2utc, rSigRnx, geodist, tropmapf
from cssrlib.gnss import satazel
from cssrlib.ephemeris import satposs
from cssrlib.cssrlib import sSigGPS, sSigGAL, sSigQZS, sCType
from cssrlib.peph import antModelRx, antModelTx
from cssrlib.ppp import tidedisp, shapiro, windupcorr
from cssrlib.rtk import IB, ddres, resamb_lambda, valpos, holdamb, initx
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
        self.nav.sig_ztd0 = 0.25  # [m]
        self.nav.sig_ion0 = 10.0  # [m]
        self.nav.sig_n0 = 30.0    # [cyc]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 100.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]
        self.nav.sig_qztd = 0.1/np.sqrt(3600)      # [m/sqrt(s)]
        self.nav.sig_qion = 10.0/np.sqrt(1)        # [m/s/sqrt(s)]

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

        if nf == 0 or sat not in cs.lc[inet].cbias.keys():
            return v

        ctype = sigref[0].typ
        if ctype == uTYP.C:
            sigc = cs.lc[inet].cbias[sat]
        else:
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

            idx_l = np.where(np.array(cs.lc[inet].sat_n) == sat)[0][0]  # local

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

            frq = np.array([s.frequency() for s in sigsCP])
            lam = np.array([s.wavelength() for s in sigsCP])
            iono = np.array([40.3e16/(f*f)*stec[idx_l] for f in frq])

            if cs.lc[0].cstat & (1 << sCType.CBIAS) == (1 << sCType.CBIAS):
                cbias += self.find_bias(cs, sigsPR, sat)
                if inet > 0:
                    cbias += self.find_bias(cs, sigsPR, sat, inet)

            if cs.lc[0].cstat & (1 << sCType.PBIAS) == (1 << sCType.PBIAS):
                pbias += self.find_bias(cs, sigsCP, sat)
                if inet > 0:
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


MAXITR = 10
ELMIN = 10
NX = 4


def logmon(nav, t, sat, cs, iu=None):
    """ log variables for monitoring """
    _, tow = gn.time2gpst(t)
    if iu is None:
        cpc = cs.cpc
        prc = cs.prc
        osr = cs.osr
    else:
        cpc = cs.cpc[iu, :]
        prc = cs.prc[iu, :]
        osr = cs.osr[iu, :]
    if nav.loglevel >= 2:
        n = cpc.shape
        for i in range(n[0]):
            if cpc[i, 0] == 0 and cpc[i, 1] == 0:
                continue
            sys, prn = gn.sat2prn(sat[i])
            pb = osr[i, 0:2]
            cb = osr[i, 2:4]
            antr = osr[i, 4:6]
            phw = osr[i, 6:8]
            trop = osr[i, 8]
            iono = osr[i, 9]
            relatv = osr[i, 10]
            dorb = osr[i, 11]
            dclk = osr[i, 12]
            # tow  sys  prn  pb1  pb2  cb1 cb2 trop iono  antr1  antr2  relatv
            # wup1  wup2  CPC1  CPC2  PRC1  PRC2  dorb  dclk
            nav.fout.write("%6d\t%2d\t%3d\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t"
                           % (tow, sys, prn, pb[0], pb[1], cb[0], cb[1]))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\t"
                           % (trop, iono, antr[0], antr[1]))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\t%8.3f\t"
                           % (relatv, phw[0], phw[1], cpc[i, 0], cpc[i, 1]))
            nav.fout.write("%8.3f\t%8.3f\t%8.3f\t%8.3f\n"
                           % (prc[i, 0], prc[i, 1], dorb, dclk))
    return 0


def rtkinit(nav, pos0=np.zeros(3), logfile=None):
    """ initialize variables for RTK """

    nav.pmode = 1  # 0:static, 1:kinematic
    nav.monlevel = 1

    nav.na = 3 if nav.pmode == 0 else 6
    nav.nq = 3 if nav.pmode == 0 else 6

    nav.thresar = 2.0
    nav.nx = nav.na+gn.uGNSS.MAXSAT*nav.nf
    nav.x = np.zeros(nav.nx)
    nav.P = np.zeros((nav.nx, nav.nx))
    nav.xa = np.zeros(nav.na)
    nav.Pa = np.zeros((nav.na, nav.na))

    nav.phw = np.zeros(gn.uGNSS.MAXSAT)
    nav.el = np.zeros(gn.uGNSS.MAXSAT)

    # parameter for PPP-RTK
    nav.eratio = [50, 50]
    nav.err = [0, 0.01, 0.005]/np.sqrt(2)
    nav.sig_p0 = 30.0
    nav.sig_v0 = 1.0
    nav.sig_n0 = 30.0
    nav.sig_qp = 0.01
    nav.sig_qv = 1.0
    nav.tidecorr = True
    nav.armode = 1  # 1:continuous,2:instantaneous,3:fix-and-hold
    nav.elmaskar = np.deg2rad(20)  # elevation mask for AR
    nav.x[0:3] = pos0

    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:3] = nav.sig_p0**2
    nav.q = np.zeros(nav.nq)
    if nav.pmode >= 1:  # kinematic
        nav.x[3:6] = 0.0
        dP[3:6] = nav.sig_v0**2
        nav.q[0:3] = nav.sig_qp**2
        nav.q[3:6] = nav.sig_qv**2
    else:
        nav.q[0:3] = nav.sig_qp**2

    nav.cs_sig_idx = {uGNSS.GPS: [rSigRnx('GL1C'), rSigRnx('GL2W')],
                      uGNSS.GAL: [rSigRnx('EL1X'), rSigRnx('EL5X')],
                      uGNSS.QZS: [rSigRnx('JL1C'), rSigRnx('JL2X')]}

    # Logging level
    #
    nav.fout = None
    if logfile is None:
        nav.monlevel = 0
    else:
        nav.fout = open(logfile, 'w')

    if nav.monlevel >= 2:
        nav.fout = open(logfile, 'w')
        nav.fout.write(
            "# tow\tsys\tprn\tpb1\tpb2\tcb1\tcb2\ttrop\tiono\tantr1\tantr2")
        nav.fout.write(
            "# \trelatv\twup1\twup2\tCPC1\tCPC2\tPRC1\tPRC2\tdorb\tdclk\n")


def udstate(nav, obs, cs):
    """ time propagation of states and initialize """
    tt = gn.timediff(obs.t, nav.t)

    ns = len(obs.sat)
    sys = []
    sat = obs.sat
    for sat_i in obs.sat:
        sys_i, _ = gn.sat2prn(sat_i)
        sys.append(sys_i)

    # pos,vel
    na = nav.na
    Phi = np.eye(nav.na)
    if nav.na > 3:
        nav.x[0:3] += nav.x[3:6]*tt
        Phi[0:3, 3:6] = np.eye(3)*tt
    nav.P[0:na, 0:na] = Phi@nav.P[0:na, 0:na]@Phi.T
    dP = np.diag(nav.P)
    dP.flags['WRITEABLE'] = True
    dP[0:nav.nq] += nav.q[0:nav.nq]*tt

    # bias
    for f in range(nav.nf):
        # reset phase-bias if instantaneous AR or
        # expire obs outage counter
        for i in range(gn.uGNSS.MAXSAT):
            sat_ = i+1
            nav.outc[i, f] += 1
            reset = (nav.outc[i, f] > nav.maxout)
            sys_i, _ = gn.sat2prn(sat_)
            if sys_i not in obs.sig.keys():
                continue
            j = IB(sat_, f, nav.na)
            if reset and nav.x[j] != 0.0:
                initx(nav, 0.0, 0.0, j)
                nav.outc[i, f] = 0
        # cycle slip check by LLI
        for i in range(ns):
            if sys[i] not in obs.sig.keys():
                continue
            if obs.lli[i, f] & 1 == 0:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))
        # reset bias if correction is not available
        for i in range(ns):
            if sat[i] in cs.sat_n:
                continue
            initx(nav, 0.0, 0.0, IB(sat[i], f, nav.na))
        # bias
        bias = np.zeros(ns)
        offset = 0
        na = 0
        for i in range(ns):
            if sys[i] not in obs.sig.keys():
                continue

            lam = obs.sig[sys[i]][gn.uTYP.L][f].wavelength()

            cp = obs.L[i, f]
            pr = obs.P[i, f]
            if cp == 0.0 or pr == 0.0 or lam is None:
                continue
            bias[i] = cp-pr/lam
            amb = nav.x[IB(sat[i], f, nav.na)]
            if amb != 0.0:
                offset += bias[i]-amb
                na += 1
        # adjust phase-code coherency
        if na > 0:
            db = offset/na
            for i in range(gn.uGNSS.MAXSAT):
                if nav.x[IB(i+1, f, nav.na)] != 0.0:
                    nav.x[IB(i+1, f, nav.na)] += db
        # initialize ambiguity
        for i in range(ns):
            j = IB(sat[i], f, nav.na)
            if bias[i] == 0.0 or nav.x[j] != 0.0:
                continue
            initx(nav, bias[i], nav.sig_n0**2, j)
    return 0


def zdres(nav, obs, rs, vs, dts, svh, rr, cs):
    """ non-differential residual """
    _c = gn.rCST.CLIGHT
    nf = nav.nf
    n = len(obs.P)
    y = np.zeros((n, nf*2))
    el = np.zeros(n)
    e = np.zeros((n, 3))
    rr_ = rr.copy()
    if nav.tidecorr:
        pos = gn.ecef2pos(rr_)
        disp = tidedisp(gn.gpst2utc(obs.t), pos)
        rr_ += disp
    pos = gn.ecef2pos(rr_)

    inet = cs.find_grid_index(pos)
    dlat, dlon = cs.get_dpos(pos)
    trph, trpw = cs.get_trop(dlat, dlon)

    trop_hs, trop_wet, _ = tropmodel(obs.t, pos)
    trop_hs0, trop_wet0, _ = tropmodel(obs.t, [pos[0], pos[1], 0])
    r_hs = trop_hs/trop_hs0
    r_wet = trop_wet/trop_wet0

    stec = cs.get_stec(dlat, dlon)

    cs.cpc = np.zeros((n, nf))
    cs.prc = np.zeros((n, nf))
    cs.osr = np.zeros((n, 4*nf+5))

    for i in range(n):
        sat = obs.sat[i]
        sys, _ = gn.sat2prn(sat)
        if svh[i] > 0 or sys not in obs.sig.keys() or sat in nav.excl_sat:
            continue
        if sat not in cs.lc[inet].sat_n:
            continue

        sigsPR = obs.sig[sys][uTYP.C]
        sigsCP = obs.sig[sys][uTYP.L]

        idx_l = np.where(np.array(cs.lc[inet].sat_n) == sat)[0][0]  # local
        kidx = [-1]*nav.nf
        nsig = 0

        for k, sig in enumerate(sigsPR):
            if sig in cs.lc[0].cbias[sat]:
                kidx[k] = sig
                nsig += 1
            elif sig.toAtt('X') in cs.lc[0].cbias[sat]:
                kidx[k] = sig.toAtt('X')
                nsig += 1

        if nsig < nav.nf:
            continue
        # check for measurement consistency
        flg_m = True
        for f in range(nav.nf):
            if obs.P[i, f] == 0.0 or obs.L[i, f] == 0.0 \
                    or obs.lli[i, f] == 1:
                flg_m = False
        if obs.S[i, 0] < nav.cnr_min:
            flg_m = False
        if flg_m is False:
            continue

        r, e[i, :] = gn.geodist(rs[i, :], rr_)
        _, el[i] = gn.satazel(pos, e[i, :])
        if el[i] < nav.elmin:
            continue

        frq = [s.frequency() for s in sigsPR]
        ionoPR = np.array([40.3e16/(f*f)*stec[idx_l] for f in frq])

        frq = np.array([s.frequency() for s in sigsCP])
        lam = np.array([s.wavelength() for s in sigsCP])
        ionoCP = np.array([40.3e16/(f*f)*stec[idx_l] for f in frq])

        iono_ = 40.3e16/(frq[0]*frq[1])*stec[idx_l]

        # global/local signal bias
        cbias = np.zeros(nav.nf)
        pbias = np.zeros(nav.nf)
        # t1 = timediff(obs.t, cs.lc[0].t0[sCType.ORBIT])
        # t2 = timediff(obs.t, cs.lc[inet].t0[sCType.PBIAS])

        if cs.lc[0].cbias is not None:
            for k in range(nav.nf):
                cbias[k] += cs.lc[0].cbias[sat][kidx[k]]
        if cs.lc[0].pbias is not None:
            for k in range(nav.nf):
                pbias[k] += cs.lc[0].pbias[sat][kidx[k].toTyp(uTYP.L)]
        if cs.lc[inet].cbias is not None:
            for k in range(nav.nf):
                cbias[k] += cs.lc[inet].cbias[sat][kidx[k]]
        if cs.lc[inet].pbias is not None:
            for k in range(nav.nf):
                pbias[k] += cs.lc[inet].pbias[sat][kidx[k].toTyp(uTYP.L)]
            # if t1 >= 0 and t1 < 30 and t2 >= 30:
            #     pbias += nav.dsis[sat]

        # relativity effect
        relatv = shapiro(rs[i, :], rr_)

        # tropospheric delay
        mapfh, mapfw = gn.tropmapf(obs.t, pos, el[i])
        trop = mapfh*trph*r_hs+mapfw*trpw*r_wet

        # phase wind-up effect
        nav.phw[sat-1] = windupcorr(obs.t, rs[i, :], vs[i, :], rr_,
                                    nav.phw[sat-1])
        phw = lam*nav.phw[sat-1]

        antrPR = antModelRx(nav, pos, e[i, :], sigsPR)
        antrCP = antModelRx(nav, pos, e[i, :], sigsCP)

        # range correction
        prc_c = trop+relatv

        # prc_c += nav.dorb[sat]-nav.dclk[sat]
        cs.prc[i, :] = prc_c+antrPR+ionoPR+cbias
        cs.cpc[i, :] = prc_c+antrCP-ionoCP+pbias+phw
        cs.osr[i, :] = [pbias[0], pbias[1], cbias[0], cbias[1],
                        antrPR[0], antrPR[1], phw[0], phw[1],
                        trop, iono_, relatv, nav.dorb[sat], nav.dclk[sat]]
        r += -_c*dts[i]

        for f in range(nf):
            y[i, f] = obs.L[i, f]*lam[f]-(r+cs.cpc[i, f])
            y[i, f+nf] = obs.P[i, f]-(r+cs.prc[i, f])

    return y, e, el


def kfupdate(x, P, H, v, R):
    """
    Kalman filter measurement update.

    Parameters:
    x (ndarray): State estimate vector
    P (ndarray): State covariance matrix
    H (ndarray): Observation model matrix
    v (ndarray): Innovation vector (residual between measurement and prediction)
    R (ndarray): Measurement noise covariance

    Returns:
    x (ndarray): Updated state estimate vector
    P (ndarray): Updated state covariance matrix
    S (ndarray): Innovation covariance matrix
    """

    PHt = P@H.T
    S = H@PHt+R
    K = PHt@np.linalg.inv(S)
    x += K@v
    # P = P - K@H@P
    IKH = np.eye(P.shape[0])-K@H
    P = IKH@P@IKH.T + K@R@K.T  # Joseph stabilized version

    return x, P, S


def ppprtkpos(nav, obs, cs):
    """ PPP-RTK positioning """
    rs, vs, dts, svh, _ = satposs(obs, nav, cs)
    # Kalman filter time propagation
    udstate(nav, obs, cs)
    # xa = np.zeros(nav.nx)
    xp = nav.x.copy()

    # non-differential residuals for rover
    yu, eu, elu = zdres(nav, obs, rs, vs, dts, svh, xp[0:3], cs)

    iu = np.where(elu >= nav.elmin)[0]
    sat = obs.sat[iu]
    y = yu[iu, :]
    e = eu[iu, :]
    el = elu[iu]
    nav.sat = sat
    nav.el[sat-1] = el
    nav.y = y
    ns = len(sat)

    if nav.monlevel > 1:
        logmon(nav, obs.t, sat, cs, iu)

    ny = y.shape[0]
    if ny < 6:
        nav.P[np.diag_indices(3)] = 1.0
        nav.smode = 5
        return -1

    # DD residual
    v, H, R = ddres(nav, obs, xp, y, e, sat, el)
    Pp = nav.P.copy()
    # Kalman filter measurement update
    xp, Pp, _ = kfupdate(xp, Pp, H, v, R)

    # non-differential residual for rover after measurement update
    yu, eu, elu = zdres(nav, obs, rs, vs, dts, svh, xp[0:3], cs)
    y = yu[iu, :]
    e = eu[iu, :]
    ny = y.shape[0]
    nav.y2 = y
    if ny < 6:
        return -1

    # residual for float solution
    v, H, R = ddres(nav, obs, xp, y, e, sat, el)
    if valpos(nav, v, R):
        nav.x = xp
        nav.P = Pp
        nav.ns = 0
        for i in range(ns):
            j = sat[i]-1
            for f in range(nav.nf):
                if nav.vsat[j, f] == 0:
                    continue
                nav.outc[j, f] = 0
                if f == 0:
                    nav.ns += 1
    else:
        nav.smode = 0

    nb, xa = resamb_lambda(nav, sat)
    nav.smode = 5  # float
    if nb > 0:
        yu, eu, elu = zdres(nav, obs, rs, vs, dts, svh, xa[0:3], cs)
        y = yu[iu, :]
        e = eu[iu, :]
        v, H, R = ddres(nav, obs, xa, y, e, sat, el)
        if valpos(nav, v, R):  # R <= Q=H'PH+R  chisq<max_inno[3] (0.5)
            if nav.armode == 3:  # fix and hold
                holdamb(nav, xa)  # hold fixed ambiguity
            nav.smode = 4  # fix
    nav.t = obs.t
    return 0
