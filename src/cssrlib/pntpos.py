"""
module for standalone positioning
"""
import numpy as np
from cssrlib.gnss import rCST, ecef2pos, geodist, satazel, \
    tropmodel, tropmapf, sat2prn, uGNSS, uTropoModel, uIonoModel, \
    timediff, time2gpst, dops, csmooth
from cssrlib.cssrlib import sCSSRTYPE, sCType
from cssrlib.ephemeris import satposs
from cssrlib.pppssr import pppos
from cssrlib.sbas import ionoSBAS
from cssrlib.dgps import vardgps
from math import sin, cos


def ionKlobuchar(t, pos, az, el, ion=None):
    """ klobuchar model of ionosphere delay estimation """
    psi = 0.0137/(el/np.pi+0.11)-0.022
    phi = pos[0]/np.pi+psi*cos(az)
    phi = np.max((-0.416, np.min((0.416, phi))))
    lam = pos[1]/np.pi+psi*sin(az)/cos(phi*np.pi)
    phi += 0.064*cos((lam-1.617)*np.pi)
    _, tow = time2gpst(t)
    tt = 43200.0*lam+tow  # local time
    tt -= tt//86400*86400
    sf = 1.0+16.0*(0.53-el/np.pi)**3  # slant factor

    h = [1, phi, phi**2, phi**3]
    amp = max(h@ion[0, :], 0)
    per = max(h@ion[1, :], 72000.0)
    x = 2.0*np.pi*(tt-50400.0)/per  # local 14h
    if np.abs(x) < 1.57:
        v = 5e-9+amp*(1.0+x*x*(-0.5+x*x/24.0))
    else:
        v = 5e-9
    diono = rCST.CLIGHT*sf*v  # iono delay at L1 [m]
    return diono


def ionmodel(t, pos, az, el, nav=None, model=uIonoModel.KLOBUCHAR, cs=None):
    """ ionosphere delay estimation """

    if model == uIonoModel.KLOBUCHAR:
        diono = ionKlobuchar(t, pos, az, el, nav.ion)
    elif model == uIonoModel.SBAS:
        if cs is None or cs.iodi < 0:
            diono = ionKlobuchar(t, pos, az, el, nav.ion)
            return diono
        diono, var = ionoSBAS(t, pos, az, el, cs)
        if diono == 0.0:
            diono = ionKlobuchar(t, pos, az, el, nav.ion)

    return diono  # iono delay at L1 [m]


class stdpos(pppos):
    def __init__(self, nav, pos0=np.zeros(3),
                 logfile=None, trop_opt=0, iono_opt=0, phw_opt=0):

        self.nav = nav

        self.nav.nf = 1
        self.nav.csmooth = True

        # Select tropospheric model
        #
        self.nav.trpModel = uTropoModel.SAAST

        # Select iono model
        #
        self.ionoModel = uIonoModel.KLOBUCHAR

        # 0: use trop-model, 1: estimate, 2: use cssr correction
        self.nav.trop_opt = trop_opt

        # 0: use iono-model, 1: estimate, 2: use cssr correction
        self.nav.iono_opt = iono_opt

        self.nav.na = (3 if self.nav.pmode == 0 else 6)
        self.nav.nq = (3 if self.nav.pmode == 0 else 6) + 1

        # State vector dimensions (including slant iono delay and ambiguities)
        #
        self.nav.nx = self.nav.na + 1

        self.nav.x = np.zeros(self.nav.nx)
        self.nav.P = np.zeros((self.nav.nx, self.nav.nx))

        self.nav.xa = np.zeros(self.nav.na)
        self.nav.Pa = np.zeros((self.nav.na, self.nav.na))

        self.nav.el = np.zeros(uGNSS.MAXSAT)

        # Observation noise parameters
        #
        self.nav.eratio = np.ones(self.nav.nf)*100  # [-] factor
        self.nav.err = [0, 0.000, 0.003]       # [m] sigma

        # Initial sigma for state covariance
        #
        self.nav.sig_p0 = 100.0   # [m]
        self.nav.sig_v0 = 1.0     # [m/s]

        self.nav.sig_cb0 = 100.0  # [m]
        # self.nav.sig_cd0 = 1.0    # [m/s]

        # Process noise sigma
        #
        if self.nav.pmode == 0:
            self.nav.sig_qp = 1.0/np.sqrt(1)     # [m/sqrt(s)]
            self.nav.sig_qv = None
        else:
            self.nav.sig_qp = 0.01/np.sqrt(1)      # [m/sqrt(s)]
            self.nav.sig_qv = 1.0/np.sqrt(1)       # [m/s/sqrt(s)]

        self.nav.sig_qcb = 1.0
        # self.nav.sig_qcd = 0.1

        self.nav.elmin = np.deg2rad(10.0)

        self.dop = None

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
        dP[self.nav.na] = self.nav.sig_cb0**2
        # dP[self.nav.na+1] = self.nav.sig_cd0**2

        # Process noise
        #
        self.nav.q = np.zeros(self.nav.nq)
        self.nav.q[0:3] = self.nav.sig_qp**2

        # Velocity
        if self.nav.pmode >= 1:  # kinematic
            self.nav.q[3:6] = self.nav.sig_qv**2

        self.nav.q[self.nav.na] = self.nav.sig_qcb**2
        # self.nav.q[self.nav.na+1] = self.nav.sig_qcd**2

        # Logging level
        #
        self.nav.fout = None
        if logfile is None:
            self.nav.monlevel = 0
        else:
            self.nav.fout = open(logfile, 'w')

    def varerr(self, nav, el, f):
        """ variation of measurement """
        if nav.smode == 2:  # DGPS
            v_sig = vardgps(el, nav)
        else:
            s_el = max(np.sin(el), 0.1*rCST.D2R)
            fact = nav.eratio[f]
            a = fact*nav.err[1]
            b = fact*nav.err[2]
            v_sig = a**2+(b/s_el)**2
        return v_sig

    def udstate(self, obs):
        """ time propagation of states and initialize """

        tt = timediff(obs.t, self.nav.t)

        sys = []
        for sat_i in obs.sat:
            sys_i, _ = sat2prn(sat_i)
            sys.append(sys_i)

        # pos,vel,ztd,ion,amb
        #
        nx = self.nav.nx
        Phi = np.eye(nx)

        if self.nav.pmode > 0:
            self.nav.x[0:3] += self.nav.x[3:6]*tt
            self.nav.x[6] += self.nav.x[7]*tt
            Phi[0:3, 3:6] = np.eye(3)*tt
            Phi[6, 7] = tt

        self.nav.P[0:nx, 0:nx] = Phi@self.nav.P[0:nx, 0:nx]@Phi.T

        # Process noise
        #
        dP = np.diag(self.nav.P)
        dP.flags['WRITEABLE'] = True
        dP[0:self.nav.nq] += self.nav.q[0:self.nav.nq]*tt

        return 0

    def zdres(self, obs, cs, bsx, rs, vs, dts, x, rtype=1):
        """ non-differential residual """

        _c = rCST.CLIGHT

        nf = self.nav.nf
        n = len(obs.P)
        rr = x[0:3]
        dtr = x[3]
        y = np.zeros((n, nf))
        el = np.zeros(n)
        az = np.zeros(n)
        e = np.zeros((n, 3))
        rr_ = rr.copy()

        # Geodetic position
        #
        pos = ecef2pos(rr_)

        if self.nav.trop_opt == 0:  # use tropo model
            # Zenith tropospheric dry and wet delays at user position
            #
            trop_hs, trop_wet, _ = tropmodel(obs.t, pos,
                                             model=self.nav.trpModel)

        for i in range(n):

            sat = obs.sat[i]
            sys, _ = sat2prn(sat)

            # Skip edited observations
            #
            if np.any(self.nav.edt[sat-1, :] > 0):
                continue

            # Geometric distance corrected for Earth rotation
            # during flight time
            #
            r, e[i, :] = geodist(rs[i, :], rr_)
            az[i], el[i] = satazel(pos, e[i, :])
            if el[i] < self.nav.elmin:
                continue

            if self.nav.trop_opt == 0:  # use model
                # Tropospheric delay mapping functions
                #
                mapfh, mapfw = tropmapf(obs.t, pos, el[i],
                                        model=self.nav.trpModel)

                # Tropospheric delay
                #
                trop = mapfh*trop_hs + mapfw*trop_wet
            else:
                trop = 0.0

            if self.nav.iono_opt == 0:  # use model
                # Ionospheric delay
                iono = ionmodel(obs.t, pos, az[i], el[i], self.nav,
                                model=self.ionoModel, cs=cs)
            else:
                iono = 0.0

            r += dtr - _c*dts[i]

            if self.nav.csmooth:  # carrier smoothing for pseudo-range
                csmooth(obs)
            y[i, 0] = obs.P[i, 0]-(r+trop + iono)

        return y, e, az, el

    def sdres(self, obs, x, y, e, sat, el):
        """
        SD phase/code residuals

        Parameters
        ----------

        obs : Obs()
            Data structure with observations
        x   :
            State vector elements
        y   :
            Un-differenced corrected observations
        e   :
            Line-of-sight vectors
        sat : np.array of int
            List of satellites
        el  : np.array of float values
            Elevation angles

        Returns
        -------
        v   : np.array of float values
            Residuals of single-difference measurements
        H   : np.array of float values
            Jacobian matrix with partial derivatives of state variables
        R   : np.array of float values
            Covariance matrix of single-difference measurements
        """

        nf = self.nav.nf  # number of frequencies (or signals)
        ns = len(el)  # number of satellites
        nc = len(obs.sig.keys())  # number of constellations

        nb = np.zeros(nc*nf, dtype=int)
        Rj = np.zeros(ns*nf)

        nv = 0
        b = 0

        H = np.zeros((ns*nf, self.nav.nx))
        v = np.zeros(ns*nf)

        # Loop over constellations
        #
        for sys in obs.sig.keys():

            # Loop over twice the number of frequencies
            #
            for f in range(0, nf):
                # Select satellites from one constellation only
                #
                idx = self.sysidx(sat, sys)

                if len(idx) == 0:
                    continue

                # Loop over satellites
                #
                for j in idx:

                    # Skip edited observations
                    #
                    if np.any(self.nav.edt[sat[j]-1, :] > 0):
                        continue

                    v[nv] = y[j, f]

                    # SD line-of-sight vectors
                    #
                    H[nv, 0:3] = -e[j, :]
                    H[nv, 3] = 1.0

                    Rj[nv] = self.varerr(self.nav, el[j], f)

                    nb[b] += 1  # counter for single-differences per signal
                    nv += 1  # counter for single-difference observations

                b += 1  # counter for signal (pseudrange+carrier-phase)

        v = np.resize(v, nv)
        H = np.resize(H, (nv, self.nav.nx))
        R = self.ddcov(nb, b, Rj, nv)

        return v, H, R

    def ddcov(self, nb, n, Rj, nv):
        """ DD measurement error covariance """
        R = np.zeros((nv, nv))
        k = 0
        for b in range(n):
            for j in range(nb[b]):
                R[k+j, k+j] = Rj[k+j]

            k += nb[b]
        return R

    def process(self, obs, cs=None, orb=None, bsx=None, obsb=None):
        """
        standalone positioning
        """
        if len(obs.sat) == 0:
            return

        if cs is not None and cs.cssrmode == sCSSRTYPE.DGPS:
            self.nav.smode = 2  # DGPS
            self.nav.baseline = cs.set_dgps_corr(self.nav.x[0:3])

        # GNSS satellite positions, velocities and clock offsets
        # for all satellite in RINEX observations
        #
        rs, vs, dts, svh, nsat = satposs(obs, self.nav, cs=cs, orb=orb)

        if nsat < 4:
            print(" too few satellites < 4: nsat={:d}".format(nsat))
            return

        # Editing of observations
        #
        sat_ed = self.qcedit(obs, rs, dts, svh)

        if obsb is None:  # standalone
            # Select satellites having passed quality control
            #
            # index of valid sats in obs.sat
            iu = np.where(np.isin(obs.sat, sat_ed))[0]
            ns = len(iu)
            y = np.zeros((ns, self.nav.nf))
            e = np.zeros((ns, 3))

            obs_ = obs
        else:  # DGPS
            y, e, iu, obs_ = self.base_process(obs, obsb, rs, dts, svh)
            ns = len(iu)

        if ns < 4:
            print(" too few satellites < 4: ns={:d}".format(ns))
            return

        # Kalman filter time propagation, initialization of ambiguities
        # and iono
        #
        self.udstate(obs_)

        xp = self.nav.x.copy()

        # Non-differential residuals
        #
        yu, eu, azu, elu = self.zdres(obs, cs, bsx, rs, vs, dts, xp)

        # Select satellites having passed quality control
        #
        # index of valid sats in obs.sat
        sat = obs.sat[iu]
        y[:ns, :] = yu[iu, :]
        e[:ns, :] = eu[iu, :]
        az = azu[iu]
        el = elu[iu]

        # Store reduced satellite list
        # NOTE: where are working on a reduced list of observations
        # from here on
        #
        self.nav.sat = sat
        self.nav.el[sat-1] = el  # needed in rtk.ddidx()
        self.nav.y = y
        ns = len(sat)

        # Check if observations of at least 4 satellites are left over
        # after editing
        #
        ny = y.shape[0]
        if ny < 4:
            self.nav.P[np.diag_indices(3)] = 1.0
            self.nav.smode = 1
            return -1

        # SD residuals
        #
        v, H, R = self.sdres(obs, xp, y, e, sat, el)
        Pp = self.nav.P.copy()

        if abs(np.mean(v)) > 100.0:  # clock bias initialize/reset
            xp[self.nav.na] = np.mean(v)
            v -= xp[self.nav.na]

        # Kalman filter measurement update
        #
        xp, Pp, _ = self.kfupdate(xp, Pp, H, v, R)

        self.nav.x = xp
        self.nav.P = Pp

        self.nav.smode = 1 if cs is None else 2  # standalone positioning
        # self.nav.smode = 1  # 4: fixed ambiguities, 5: float ambiguities

        # Store epoch for solution
        #
        self.nav.t = obs.t
        self.dop = dops(az, el)

        return 0
