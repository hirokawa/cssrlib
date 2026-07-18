"""
module for PPP-RTK positioning
"""

import numpy as np
from cssrlib.pppssr import pppos


class ppprtkpos(pppos):
    """ class for PPP-RTK processing """

    def __init__(self, nav, pos0=np.zeros(3), logfile=None, config=None):
        """ initialize variables for PPP-RTK """

        # trop, iono from cssr
        # phase windup model is local/regional
        super().__init__(nav=nav, pos0=pos0, logfile=logfile, config=config)
        
        self.nav.eratio = config['nav']['eratio']
        self.nav.err = config['nav']['err']
        self.nav.sig_p0 = config['nav']['sig_p0']
        self.nav.thresar = config['nav']['thresar']  # AR acceptance threshold
        self.nav.armode = config['nav']['armode']     # AR is enabled
        self.nav.elmaskar = np.deg2rad(config['nav']['elmaskar'])  # elevation mask for AR
        self.nav.elmin = np.deg2rad(config['nav']['elmin'])