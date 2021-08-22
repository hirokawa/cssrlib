# -*- coding: utf-8 -*-
"""
Created on Sun Aug 22 21:01:49 2021

@author: ruihi
"""

from cssrlib.cssrlib import cssr
from cssrlib.gnss import ecef2pos

bdir='./data/'
l6file=bdir+'2021078M.l6'
griddef=bdir+'clas_grid.def'
xyz=[-3962108.4557,  3381308.8777,  3668678.1749]
pos=ecef2pos(xyz)

cs=cssr()
cs.monlevel=2
cs.week=2149

cs.read_griddef(griddef)
inet=cs.find_grid_index(pos)

sf=0
#sfmax=3600//5
sfmax=30//5
with open(bdir+l6file,'rb') as f:
    while sf<sfmax:
        msg=f.read(250)
        if not msg:
            break
        cs.decode_l6msg(msg,0)
        if cs.fcnt==5:
            sf+=1
            cs.decode_cssr(cs.buff,0)
        if sf>=6:
            dlat,dlon=cs.get_dpos(pos)
            trph,trpw=cs.get_trop(dlat,dlon)
            stec=cs.get_stec(dlat,dlon)

            


                    
    