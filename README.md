*************************
CSSRlib - Toolkit for PPP-RTK/RTK in Python using Compact SSR
*************************

# What is CSSRlib? 
CSSRlib is a free and open source library to demonstrate RTK and PPP/PPP-RTK positioning 
using RINEX and/or correction data formatted in Compact SSR.
CSSRlib is developped based on RTKlib.

Prerequisites
=============
Additional python packages are required as prerequisites and can be installed via the following commands

```
pip install bitstruct
pip install cbitstruct
pip install galois
pip install cartopy
```

If the installation of `cartopy` fails, try installing `libgeos++-dev` first.

```
sudo apt-get install libgeos++-dev
```

Install
=======
You can install with `pip install cssrlib`

Testing
=======

Run orbit plot sample :code:`python test_eph.py`

Run RTK sample :code:`python test_rtk.py`

