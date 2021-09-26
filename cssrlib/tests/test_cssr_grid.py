import numpy as np
from cssrlib.cssrlib import cssr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

griddef = '../data/clas_grid.def'
area = [122, 149, 21, 47]

cs = cssr()
cs.read_griddef(griddef)

fig = plt.figure(figsize=(12, 12))

ax = plt.axes(projection=ccrs.Orthographic(
    central_longitude=135, central_latitude=45))
ax.coastlines(resolution='10m')
ax.add_feature(cfeature.LAND)
ax.add_feature(cfeature.COASTLINE)

ax.gridlines()

for nid in np.unique(cs.grid['nid']):
    idx = np.where(cs.grid['nid'] == nid)[0]
    lat = cs.grid['lat'][idx]
    lon = cs.grid['lon'][idx]
    plt.plot(lon, lat, '.', transform=ccrs.Geodetic(), label=str(nid))
plt.legend(fancybox=True)
plt.show()
