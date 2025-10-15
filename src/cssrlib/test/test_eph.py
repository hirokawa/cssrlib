import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
from cssrlib.rinex import rnxdec
from cssrlib.gnss import Nav, epoch2time, prn2sat, uGNSS, sat2prn, \
    timeadd, ecef2pos
from cssrlib.ephemeris import findeph, eph2pos


def test_eph(navfile: str) -> None:
    nav = Nav()
    dec = rnxdec()
    nav = dec.decode_nav(navfile, nav)

    n = 24*3600//300
    t0 = epoch2time([2021, 3, 19, 0, 0, 0])

    flg_plot = True

    if True:
        t = t0
        sat = prn2sat(uGNSS.QZS, 194)
        eph = findeph(nav.eph, t, sat)
        rs, vs, dts = eph2pos(t, eph, True)

    if flg_plot:
        lon0 = 135
        plt.figure(figsize=(6, 6))
        ax = plt.axes(projection=ccrs.Orthographic(central_longitude=lon0,
                                                central_latitude=0))
        ax.coastlines(resolution='50m')
        ax.gridlines()
        ax.stock_img()
        pos = np.zeros((n, 3))

        for k in range(uGNSS.MAXSAT):
            sat = k+1
            sys, prn = sat2prn(sat)
            if sys != uGNSS.QZS:
                continue
            for i in range(n):
                t = timeadd(t0, i*300)
                if eph is None:
                    continue
                rs, dts = eph2pos(t, eph)
                pos[i, :] = ecef2pos(rs)
                pos[i, 0] = np.rad2deg(pos[i, 0])
                pos[i, 1] = np.rad2deg(pos[i, 1])

            plt.plot(pos[:, 1], pos[:, 0], 'm-', transform=ccrs.Geodetic())
        plt.show()


if __name__ == "__main__":
    # Uncompressed RINEX navigation file
    print("Test with uncompressed RINEX navigation file")
    navfile = '../data/30340780.21q'
    test_eph(navfile)

    # Test with compressed RINEX navigation file
    print("Test with compressed RINEX navigation file")

    # Compress navfile with gzip beforehand
    navfile_ga = navfile + '.gz'
    import gzip
    with open(navfile, 'rb') as f_in:
        with gzip.open(navfile_ga, 'wb') as f_out:
            f_out.writelines(f_in)

    # Run test
    test_eph(navfile_ga)

    # Clean up
    import os
    os.remove(navfile_ga)