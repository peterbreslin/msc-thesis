import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="name of target directory", type=str)
args = parser.parse_args()
if args.dir is None:
	sys.exit("Error: no target directory provided. Provide target directory with -d or --dir")
target = args.dir
camid = target[-3:]
date = target[:8]

import os
import cv2
import h5py
import numpy as np
import pandas as pd
import astropy.stats as aps
import scipy.ndimage as scn
import astropy.io.fits as pf
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from astropy.coordinates import EarthLocation, get_moon

import logging
logging.captureWarnings(True)

rootdir = '/net/beulakerwijde/data1/breslin/'
datadir = rootdir + 'data/subtracted/{target}/'

sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
from mascommon import mmm
import bringreduce.configuration as cfg
cfg.initialize(rootdir, target, user='breslin')

import bringreduce.mascara_astrometry as astrometry
starcat = pf.getdata(cfg.starcat)

import bringreduce.bringio as bringio
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)

if not os.path.exists(f"{rootdir}/my_code/affected_area/"):
	os.makedirs(f"{rootdir}/my_code/affected_area/")


edge = 30
minlinelength = 15
masksizes = [2, 4, 8, 16]



# ----------------------------------------------------------------------------------------------------------------------



def createmask(masksize = 4):
	lingrid = np.linspace(0, masksize*2, masksize*2+1)-masksize
	xx, yy = np.meshgrid(lingrid,lingrid)
	rr = np.sqrt(xx**2+yy**2)
	mask = np.zeros((masksize*2+1,masksize*2+1))
	mask[np.where(rr <= (masksize+0.1))] = 1
	return mask        


def createring(radius = 200, dr=.1):
	# if dr < 1, then the ring width is a fraction of the ring radius. Otherwise ringwidth is set by value of dr. 
	if dr < 1:
		masksize = int((1+dr)*radius)
	else:
		masksize = radius+dr
	lingrid = np.linspace(0, masksize*2, masksize*2+1)-masksize
	xx, yy = np.meshgrid(lingrid,lingrid)
	rr = np.sqrt(xx**2+yy**2)
	mask = np.zeros((masksize*2+1,masksize*2+1))
	mask[np.where((rr>=radius)&(rr <= (masksize+0.1)))] = 1
	return mask 


def LineSegments(fast, deltaimage, nx, ny, starcat, astrometry):

	mean, median, sigma = aps.sigma_clipped_stats(np.abs(deltaimage))
	curastro = np.where((fast['astrometry/lstseq'][:] // 50) == (curlstseq // 50))[0][0]
	order = fast['astrometry/x_wcs2pix'][curastro].shape[0]-1
	astrolst = fast['station/lst'][np.where(fast['station/lstseq'][:] == (
		fast['astrometry/lstseq'][curastro]))[0][0]]
	astrolstseq = fast['station/lstseq'][np.where(fast['station/lstseq'][:] == (
		fast['astrometry/lstseq'][curastro]))[0][0]]

	wcspars = { 'crval' : fast['astrometry/crval'][curastro].copy(),
				'crpix' : fast['astrometry/crpix'][curastro].copy(),
				'cdelt' : [0.02148591731740587,0.02148591731740587],
				'pc'    : fast['astrometry/pc'][curastro].copy(),
				'lst'   : astrolst}
	polpars = { 'x_wcs2pix' : fast['astrometry/x_wcs2pix'][curastro].copy(),
				'y_wcs2pix' : fast['astrometry/y_wcs2pix'][curastro].copy(),
				'x_pix2wcs' : fast['astrometry/x_pix2wcs'][curastro].copy(),
				'y_pix2wcs' : fast['astrometry/y_pix2wcs'][curastro].copy(),
				'nx'    : nx,
				'ny'    : ny,
				'order' : order}

	astro = astrometry.Astrometry(wcspars, polpars)
		
	ra, dec = starcat['_raj2000'], starcat['_dej2000']
	stars_x0, stars_y0, stars_err0 = astro.world2pix(midlst,ra,dec,jd=midJD)
	select = np.where(
		(stars_x0 > edge) & 
		(stars_y0 > edge) & 
		(stars_x0 < (nx-edge)) & 
		(stars_y0 < (ny-edge)) & 
		(starcat['vmag'][stars_err0] <= 9.0))[0]
	xxx, yyy = np.round(stars_x0[select]).astype(int), np.round(stars_y0[select]).astype(int)
	vmag = starcat['vmag'][select]
				
	badpixelareas = {'SAE':[[50, 150, 600, 950]], 'AUE':[[0,300,1750, 2720],[3400, 4072, 0, 350]], 
					 'LSE':[], 'LSS':[], 'LSW':[[0,350,2350,2720]], 'LSN':[], 'LSC':[]}
	image_id = str(curlstseq)
	MooninImage = False
	
	return astro, vmag, deltaimage, masksizes, xxx, yyy, stars_x0, stars_y0, stars_err0



def ImageReduction(astro, siteinfo, JD, moonmargin=400):
	
	mean, median, sigma = aps.sigma_clipped_stats(np.abs(deltaimage))

	nostarimage = deltaimage.copy()
	mask = [createmask(ms) for ms in masksizes]

	#The size of the mask depends on the brightness of the star
	for ccc in range(len(yyy)):
		curmask = -1
		if ((vmag[ccc] < 6) & (vmag[ccc] >= 4.5)):
			curmask = 0
		if ((vmag[ccc] < 4.5) & (vmag[ccc] >= 2.5)):
			curmask = 2
		elif (vmag[ccc] < 2.5):
			curmask = 3
		if curmask >=0:
			nostarimage[yyy[ccc]-masksizes[curmask]:yyy[ccc]+masksizes[curmask]+1,
						xxx[ccc]-masksizes[curmask]:xxx[ccc]+masksizes[curmask]+1] = nostarimage[
				yyy[ccc]-masksizes[curmask]:yyy[ccc]+masksizes[curmask]+1,
						xxx[ccc]-masksizes[curmask]:xxx[ccc]+masksizes[curmask]+1]*(
				1-mask[curmask])+ median*mask[curmask]


	#Check if the Moon is in FoV of camera
	observatory = EarthLocation(lat=siteinfo['lat']*u.deg, 
								lon=siteinfo['lon']*u.deg, 
								height=siteinfo['height']*u.m)       
	gcrs_coords = get_moon(Time(JD, format='jd'), location=observatory)

	moonx_astropy, moony_astropy, moon_mask = astro.world2pix(
		midlst, gcrs_coords.ra.value[0], gcrs_coords.dec.value[0], jd=midJD, margin=-moonmargin)

	if moon_mask: 
		MooninImage = True
		moonx, moony = np.round(moonx_astropy).astype(int)[0], np.round(moony_astropy).astype(int)[0]
		ring_median = 1.e6
		while(ring_median>(median+0.5*sigma)):

			ring = createring(radius = moonmargin)
			ringsize = int(1.1*moonmargin)
			indices = np.where(ring[max(0, ringsize-moony):min(2*ringsize+1, ringsize+(ny-moony)), 
									max(0, ringsize-moonx):min(2*ringsize+1, ringsize+(nx-moonx))]==1)
			ring_mean, ring_median, ring_sigma = aps.sigma_clipped_stats(
				np.abs(nostarimage[max(moony-ringsize,0):min(ny, moony+ringsize+1),
								   max(moonx-ringsize, 0):min(nx, moonx+ringsize+1)])[indices])

			moonmargin += 1

		print("Moon is masked with a radius of", moonmargin)
		moonmaskradius = moonmargin
		moonmask = createmask(moonmargin)

		nostarimage[max(moony-moonmargin,0):min(ny, moony+moonmargin+1),
					max(moonx-moonmargin, 0):min(nx, moonx+moonmargin+1)] = nostarimage[max(moony-moonmargin,
					0):min(ny, moony+moonmargin+1),
						max(moonx-moonmargin, 0):min(nx, moonx+moonmargin+1)]*(
			1-moonmask[max(0, moonmargin-moony):min(2*moonmargin+1, moonmargin+(ny-moony)), 
					   max(0, moonmargin-moonx):min(2*moonmargin+1, moonmargin+(nx-moonx))]) + median*moonmask[
			max(0, moonmargin-moony):min(2*moonmargin+1, moonmargin+(ny-moony)), 
			max(0, moonmargin-moonx):min(2*moonmargin+1, moonmargin+(nx-moonx))]


		n_pixels = int(round(np.sqrt((moonx - int(nx/2))**2.+ (moony - int(ny/2))**2.)))
		x_values = np.rint(np.linspace(int(nx/2), moonx, n_pixels)).astype(int)
		y_values = np.rint(np.linspace(int(ny/2), moony, n_pixels)).astype(int)
		linemask = np.zeros((deltaimage.shape[0], deltaimage.shape[1]))
		linemask[y_values, x_values]=1

		#broaden the line segment by 20 pixels in each direction
		linemask = scn.filters.convolve(linemask,np.ones((40,40)))
		linemask[np.where(linemask >= 1)] = 1
		nostarimage[np.where(linemask==1)]=median

	maskedstarimage = nostarimage.copy()
	nostarimage[-edge:,:] = 0
	nostarimage[:edge,:] = 0
	nostarimage[:,-edge:] = 0
	nostarimage[:,:edge] = 0

	maskim = nostarimage.copy()*0
	maskim2 = nostarimage.copy()*0
	maskim2[np.abs(nostarimage) > mean+2.5*sigma] = 1
	out = cv2.dilate(np.uint8(maskim2),np.ones((3,3),np.uint8))
	maskim[(np.abs(nostarimage) > mean+1.0*sigma)] = 1
	maskim = maskim*out
	reducedimage = np.uint8(np.clip(scn.filters.convolve(maskim,np.ones((3,3))),6,9)-6)
	
	return maskedstarimage, nostarimage, reducedimage



# ----------------------------------------------------------------------------------------------------------------------



def areas_affected():

	fast = h5py.File(f"/net/beulakerwijde/data1/bring/testdata/lc/{target}/lightcurves/fast_{target}.hdf5", "r")
	vmags = pd.read_pickle(f'{datadir}vmags_*.p')
	results = {}

	for lstseq in vmags.keys():
		curlstseq = int(lstseq)

		data, header = pf.getdata(f'{datadir}diffimages/diff_{curlstseq}{camid}.fits.gz', header=True)
		nx = header['XSIZE']
		ny = header['YSIZE']
		JD0 = header['JD0']
		midJD = header['MIDJD']
		midlst = header['MIDLST']

		image_area = nx * ny

		astro, vmag, deltaimage, masksizes, xxx, yyy, stars_x0, stars_y0, stars_err0 = LineSegments(
			fast, data, nx, ny, starcat, astrometry)

		maskedstarimage, nostarimage, reducedimage = ImageReduction(astro, siteinfo, JD0)

		satnums = list(vmags[lstseq])[2:]
		if len(satnums) == 0:
			continue

		sat_count = 0
		areas = []
		for satnum in satnums:
			sat_count += 1
			x = vmags[lstseq][satnum]['FOTOS']['x']
			y = vmags[lstseq][satnum]['FOTOS']['y']
			d = vmags[lstseq][satnum]['FOTOS']['length']

			x_min, x_max = x[0], x[1]
			y_min, y_max = y[0], y[1]

			x_values = np.rint(np.linspace(x_min, x_max, d)).astype(int)
			y_values = np.rint(np.linspace(y_min, y_max, d)).astype(int)

			emptymask = np.zeros((deltaimage.shape[0], deltaimage.shape[1]))
			linemask = emptymask.copy()
			linemask[y_values, x_values]=1

			annulus = scn.filters.convolve(linemask, np.ones((41,41))) #size of outer mask region
			annulus[np.where(annulus >= 1)] = 1
			affected_area = np.sum(annulus == 1)
			areas.append(affected_area)


		results[lstseq] = {'count':sat_count, 'img_area':image_area, 'affected_area':np.sum(areas), 'areas':areas,
		'fraction':round(np.sum(areas)/(image_area), 4)}


	pickle.dump(results, open(f"{rootdir}/my_code/affected_area/affected_area_{camid}.p", "wb" ))


