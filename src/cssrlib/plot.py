"""
module for plotting
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from cssrlib.gnss import uGNSS, sat2prn, sat2id
from cssrlib.gnss import rCST


def plot_nsat(t, nsat):
    """ plot number of satellites """
    lbl_t = {uGNSS.GPS: 'GPS', uGNSS.GAL: 'Galileo', uGNSS.QZS: 'QZSS',
             uGNSS.GLO: 'GLONASS', uGNSS.BDS: 'BeiDou'}
    col_tbl = 'bygmkrc'
    _, ax = plt.subplots(1, 1)
    ns = np.zeros(nsat.shape[0])
    for gnss in lbl_t.keys():
        y = ns+nsat[:, gnss]
        if gnss >= 0:
            ax.fill_between(t/3600, ns, y, label=lbl_t[gnss],
                            facecolor=col_tbl[gnss])
        ns += nsat[:, gnss]

    plt.ylabel('number of satellites')
    plt.xlabel('time [h]')
    plt.legend()
    plt.axis([0, 24, 0, 50])
    plt.show()


def plot_elv(t, elv, elmask=0, satlist=None):
    """ elevation plot """
    if satlist is None:
        satlist = range(1, uGNSS.MAXSAT)
    nsat = np.zeros((len(t), uGNSS.GNSSMAX), dtype=int)
    col_tbl = 'bygmkrc'
    plt.figure('elevation')
    for k, sat in enumerate(satlist):
        if np.all(np.isnan(elv[:, k])):
            continue
        sys, _ = sat2prn(sat)
        idx = elv[:, k] > elmask
        nsat[idx, sys] += 1
        plt.plot(t/60, np.rad2deg(elv[:, k]), '-'+col_tbl[sys])

    tmax = t[-1]//60+1
    plt.ylabel('Elevation Angle [deg]')
    plt.xlabel('Time [min]')
    plt.grid()
    plt.axis([0, tmax, 0, 90])
    plt.show()
    return nsat


def skyplot(azm, elv, elmask=0, satlist=None):
    """ plot skyplot """
    fig = plt.figure('skyplot')
    ax = fig.add_subplot(projection='polar')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ticks_loc = ax.get_xticks().tolist()
    ax.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
    ax.set_xticklabels(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
    ax.set_ylim([0, 90])
    ax.set_rgrids(radii=[15, 30, 45, 60, 75],
                  labels=['75', '60', '45', '30', '15'], fmt='%d')
    col_tbl = 'bygmkrc'

    if satlist is None:
        satlist = range(1, uGNSS.MAXSAT)

    nsat = 0
    for k, sat in enumerate(satlist):
        if np.all(np.isnan(elv[:, k])):
            continue
        sys, _ = sat2prn(sat)
        idx = elv[:, k] > elmask
        if len(elv[idx, k]) == 0:
            continue
        z = 90-np.rad2deg(elv[idx, k])
        theta = azm[idx, k]
        ax.scatter(theta, z, s=5, c=col_tbl[sys])
        ax.text(theta[0], z[0], sat2id(sat), fontsize=8)
        nsat += 1
    plt.show()
    return nsat


def draw_rect(ax, ccrs, p, col, alpha=0.5):
    """ draw rectangle centered on [latr, lonr] """

    lonr_ = [p.lonr-p.lons, p.lonr+p.lons, p.lonr +
             p.lons, p.lonr-p.lons, p.lonr-p.lons]
    latr_ = [p.latr+p.lats, p.latr+p.lats, p.latr -
             p.lats, p.latr-p.lats, p.latr+p.lats]

    ax.fill(lonr_, latr_, color=col, alpha=alpha,
            transform=ccrs)


def draw_circle(ax, ccrs, p, col, nc=10, alpha=0.5):
    """ draw circle centered on [latr, lonr] with radius rng """
    r_ = p.rng/(rCST.RE_WGS84*1e-3)*rCST.R2D

    the_ = np.linspace(0, 2*np.pi, nc)

    lonr_ = p.lonr + r_/np.cos(p.latr*rCST.D2R)*np.cos(the_)
    latr_ = p.latr + r_*np.sin(the_)

    ax.fill(lonr_, latr_, color=col, alpha=alpha,
            transform=ccrs)
