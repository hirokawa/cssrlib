CSSRlib - Toolkit for PPP-RTK/RTK in Python using Compact SSR
=============

What is CSSRlib?
----------------

CSSRlib is a open toolkit in Python for high accuracy GNSS positioning. It supports SSR (State-Space Representation) based positioning for PPP (Precise Point Positioning) or PPP-RTK (Realtime Kinematic), while also supporting RTK. The goal of the CSSRlib toolkit is to provide an easy-to-understand open implementation to learn PPP/PPP-RTK positioning provided by satellite-based open PPP/PPP-RTK services such as QZSS CLAS, Galileo HAS, and BeiDou 3 PPP. It also supports ground based open service by IGS. The code is based on RTKlib.

It supports the following open format:

- The observation data and ephemeris in RINEX
- The PPP/PPP-RTK correction data in RTCM SSR, IGS SSR, Compact SSR, ...
- The precise orbit/clock data in SP3 format.

Click this button for a quick demo in Google Colab

<a target="_blank" href="https://colab.research.google.com/github/hirokawa/cssrlib/blob/main/tutorials/cssrlib.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>


Prerequisites
-------------
Additional python packages are required as prerequisites and can be installed via the following command

```
pip install -r requirements.txt
```

If the installation of `cartopy` fails, try installing `libgeos++-dev` first.

```
sudo apt-get install libgeos++-dev
```

*NOTE*: the module `pysolid` is used for the computation of solid Earth tides. It contains a hard-coded leap second table with an expiration date, which is set to the next possible injection date of a leap second at the time of the last update. The table is frequently updated by the package maintainers. The following warning is issed when the expiration date is exceeded:

> Mild Warning -- time crossed leap second table boundaries.  Boundary edge value used instead

If you encounter this warning when executing CSSRlib scripts, it can most likely be fixed by updating `pysolid` to the most recent version using

```
pip install -upgrade pysolid
```

Install
-------

You can install the official version of CSSRlib using pip

```
pip install cssrlib
```

If you want to install the development version from this repository, first clone or download the sources and then run

```
pip install .
```

in the root directory, where the ``setup.cfg`` file is located.

Testing
-------

Run orbit plot sample.

```
python test_eph.py
```

Run RTK sample.

```
 python test_rtk.py
```

Other samples with dataset are also available in a separate repository [`cssrlib-data`](https://github.com/hirokawa/cssrlib-data) including:

- Galileo-HAS PPP (SIS/IDD) processing example
- BDS-PPP processing example
- QZSS MADOCA-PPP processing example
- QZSS CLAS PPP-RTK processing example
- IGS (SP3+BIAS) PPP-AR processing example
- PPP via SouthPAN (PVS) processing example
- JPL GDGPS (GPSHAS) processing example
- Binary message parser for Septentrio receiver (Mosaic/PolaRX5)


