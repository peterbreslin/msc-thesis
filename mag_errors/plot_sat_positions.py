datadir = "/Users/peter/Projects/starlink_data/"

target = f"20221023{camid}"
camid = target[-3:]
date = target[:8]

import bringreduce.configuration as cfg
cfg.initialize(target)


import h5py
import ephem
import numpy as np
import pandas as pd
import astropy.stats as aps
import astropy.io.fits as pf
import matplotlib.pyplot as plt

starcat  = pf.getdata(cfg.starcat)
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)

lstseq = '48506263'
data, header = pf.getdata(f'{datadir}raw_images/{lstseq}{camid}.fits.gz', header=True)
mean, median, sigma = aps.sigma_clipped_stats(np.abs(data))
data[data >= 0.9*65535] = median

lx = header['X0']
nx_image = header['XSIZE']
ux = lx + nx_image
ly = header['Y0']
ny_image = header['YSIZE']
uy = ly + ny_image
lst = header['LST']
JD = header['JD']

passages = pd.read_pickle(f'{datadir}vmags_subset/passages_subset/passages_{target}.p')


"""
edate = ephem.Date('2000/01/01 12:00:00.0')
obs = ephem.Observer()
obs.lat = siteinfo['lat']*np.pi/180
obs.long = siteinfo['lon']*np.pi/180
obs.elev = siteinfo['height']
obs.epoch = edate
moon = ephem.Moon(obs)

fast = h5py.File(f"{datadir}fast_20221023{camid}.hdf5", "r")
astro = np.where((fast['astrometry/lstseq'][:] // 50) == (int(lstseq) // 50))[0][0]
order = fast['astrometry/x_wcs2pix'][astro].shape[0]-1
lst = fast['station/lst'][np.where(fast['station/lstseq'][:]==(fast['astrometry/lstseq'][astro]))[0][0]]
nx = data[lstseq]['nx']
ny = data[lstseq]['ny']

wcspars = { 'crval' : fast['astrometry/crval'][astro].copy(),
            'crpix' : fast['astrometry/crpix'][astro].copy(),
            'cdelt' : [0.02148591731740587,0.02148591731740587],
            'pc'    : fast['astrometry/pc'][astro].copy(),
            'lst'   : lst }

polpars = { 'x_wcs2pix' : fast['astrometry/x_wcs2pix'][astro].copy(),
            'y_wcs2pix' : fast['astrometry/y_wcs2pix'][astro].copy(),
            'x_pix2wcs' : fast['astrometry/x_pix2wcs'][astro].copy(),
            'y_pix2wcs' : fast['astrometry/y_pix2wcs'][astro].copy(),
            'nx'    : nx,
            'ny'    : ny,
            'order' : order }

astrofns = astrometry.Astrometry(wcspars, polpars)
"""



plt.figure()
plt.imshow(np.fabs(data[ly:uy,lx:ux]), interpolation="none", vmin=1000, vmax=2000.)
for sat in list(passages[JD].keys()):
    plt.scatter(np.float64(passages[JD][sat]['start']['x0']), np.float64(passages[JD][sat]['start']['y0']), linewidths = 3, facecolors='None', s=50, edgecolors='green')
    plt.scatter(np.float64(passages[JD][sat]['end']['x0']), np.float64(passages[JD][sat]['end']['y0']), linewidths = 3, facecolors='None', edgecolors='r', s=50)
    plt.annotate(sat, (np.float64(passages[JD][sat]['end']['x0']), np.float64(passages[JD][sat]['end']['y0'])), xytext=(8, 0), va='bottom',textcoords='offset points')
plt.show()

