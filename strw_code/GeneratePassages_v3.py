# Changing this routine such that saved dictionaries are not indexed by JD - Peter Breslin 2023

import sys
sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dir", help="name of target directory", type=str)
parser.add_argument("-u", "--user", help="name of user", type=str, default="breslin")
parser.add_argument("-r", "--rootdir", help="name of root directory", type=str, default="/net/beulakerwijde/data1/")
args = parser.parse_args()

if args.dir is None:
	sys.exit("Error: no target directory provided. Provide with -d or --dir")

rootdir = args.rootdir
target = args.dir
camid = target[-3:]
date = target[:8]
user = args.user

import bringreduce.configuration as cfg
cfg.initialize(rootdir,target,user)

import os
import glob
import h5py
import time
import ephem
import subtract
import numpy as np
import pickle as pickle
import astropy.io.fits as pf
import bringreduce.bringio as bringio
import bringreduce.mascara_astrometry as astrometry

if not os.path.exists(f"{rootdir}{user}/my_code/diffimage_passages"):
	os.makedirs(f"{rootdir}{user}/my_code/diffimage_passages")

# Setting-up 
number_of_stacked_images = 2
starcat = pf.getdata(cfg.starcat)
filelist = np.sort(glob.glob(f"{rootdir}bring/testdata/{target}/*.fits.gz"))
files = [filename for filename in filelist if not any(s in filename for s in ['dark','bias','flat'])]
lstseq = np.array([np.int64(fff[-19:-11]) for fff in files])

# Create list such that only every 50th LST remains
sequences = 50*np.unique(lstseq // 50)
longsets  = [np.where(50*np.floor(lstseq//50) == sss)[0] for sss in sequences]

# Number of science frames in long set (typically 25 for bRing and 50 for MASCARA)
nlong = [len(sss) for sss in longsets]

# Reduced fast curves for the astrometric solution  
fast = h5py.File(f"{rootdir}bring/testdata/lc/{target}/lightcurves/fast_{target}.hdf5", "r")
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)

# Setting observer ---> use Skyfield?
obs = ephem.Observer()
edate = ephem.Date('2000/01/01 12:00:00.0')
obs.lat = siteinfo['lat']*np.pi/180
obs.long = siteinfo['lon']*np.pi/180
obs.elev = siteinfo['height']
obs.epoch = edate    

# Output dictionary
passages = {}

# Collecting TLEs
satdir = f"/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/{date}/"
satfiles = f"{satdir}3leComplete.txt"
with open(satfiles) as f:
	all_tles = f.readlines()
	f.close() 
	
# Split TLE list into individual lists for each TLE
all_tles = [i.strip() for i in all_tles]
tles = [all_tles[x:x+3] for x in range(0, len(all_tles), 3)]


def diffimages_fn(sss):
	t0 = time.time()
	midpoint = nlong[sss]//2

	# Take the middle image from set of 25
	header_mid = pf.getheader(files[longsets[sss][midpoint]])
	curlstseq = lstseq[longsets[sss][0]]
	midlstseq = lstseq[longsets[sss][midpoint]]

	midlst = header_mid['LST']
	midJD  = header_mid['JD']                                                        
	lx = header_mid['X0']
	nx = header_mid['XSIZE']
	ux = lx + nx
	ly = header_mid['Y0']
	ny = header_mid['YSIZE']
	uy = ly + ny
	
	# Obtain the astrometric solution from already reduced fast curves
	# Why use the astrometric solution of the first image here and not the middle one, since the middle one is used later on?
	curastro = np.where((fast['astrometry/lstseq'][:] // 50) == (curlstseq // 50))[0][0]
	order = fast['astrometry/x_wcs2pix'][curastro].shape[0]-1
	astrolst = fast['station/lst'][np.where(fast['station/lstseq'][:] == (fast['astrometry/lstseq'][curastro]))[0][0]]
	astrolstseq = fast['station/lstseq'][np.where(fast['station/lstseq'][:] == (fast['astrometry/lstseq'][curastro]))[0][0]]
	
	wcspars = {'crval' : fast['astrometry/crval'][curastro].copy(),
				'crpix' : fast['astrometry/crpix'][curastro].copy(),
				'cdelt' : [0.02148591731740587,0.02148591731740587],
				'pc'    : fast['astrometry/pc'][curastro].copy(),
				'lst'   : astrolst}
	polpars = {'x_wcs2pix' : fast['astrometry/x_wcs2pix'][curastro].copy(),
				'y_wcs2pix' : fast['astrometry/y_wcs2pix'][curastro].copy(),
				'x_pix2wcs' : fast['astrometry/x_pix2wcs'][curastro].copy(),
				'y_pix2wcs' : fast['astrometry/y_pix2wcs'][curastro].copy(),
				'nx'    : nx,
				'ny'    : ny,
				'order' : order}
	
	astro = astrometry.Astrometry(wcspars, polpars)  # THIS USES curlstseq SO WHAT I DID BELOW COULD BE WRONG
	
	t1 = time.time()
	print("Setting up: ", str(t1-t0))
	print(f"Adding {nlong[sss]} images")
	JDs = []
	lst_array = []
	exp_array = []    
	lstseq_array = []
	
	for iii in range(nlong[sss]):
		header0 = pf.getheader(files[longsets[sss][iii]])
		curlst = header0['LST']
		curexp = header0['EXPTIME']
		JDs.append(header0['JD'])
		lstseq_array.append(header0['LSTSEQ'])
		lst_array.append(curlst)
		exp_array.append(curexp)
	
	print('Finished adding images')

	lst_array = np.array(lst_array)
	JDs = np.array(JDs)
	exp_array = np.array(exp_array)
	lstseq_array = np.array(lstseq_array)

		
	for j, tle in enumerate(tles):
		sat = ephem.readtle(tle[0][2:], tle[1], tle[2])
		
		for i, (jd, LSTSEQ) in enumerate(zip(JDs, lstseq_array)):

			seq = '{:08d}'.format(LSTSEQ)
			if seq not in list(passages.keys()):
				passages[seq] = {}

			# pyephem.date is the number of days since 1899 December 31 12:00 UT (2415020 JD) 
			obs.date = jd - 2415020-0.5*exp_array[i]/86400
			sat.compute(obs)

			# Satellite needs to be above the horizon in the first place
			try:
				sat.alt
			except RuntimeError as rte:
				print(rte, 'UTC (', ephem.hours(obs.sidereal_time()), 'local sidereal time) for', tle[0])
			else:
				if sat.alt > 0:

					"""
					Now using a separate dictionary altogether: diffimage_passages. For each diffimage (i.e. LSTSEQ), we
					will store the pixel positions of the positive and negative portion of the line segment. This means
					we must store the following LSTSEQ so as to encode info about the second image (NOTE: first image is
					substracted from the second, hence the negative segment is from the first image).

					First image (negative segment):
						- start: obs.date = jd - 2415020-0.5*exp_array[i]/86400 
						- end:   obs.date = jd - 2415020+0.5*exp_array[i]/86400 

					Second image (positive segment):
						- start: obs.date = jd - 2415020+0.5*exp_array[i]/86400 
						- end:   obs.date = jd - 2415020+1.5*exp_array[i]/86400 

					"""

					# Start (start-point of first i.e. negative segment))
					ra0, dec0 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
					x0, y0, mask0 = astro.world2pix(midlst, ra0, dec0, jd=midJD)

					# Mid (middle-point between negative and positive segment)
					obs.date = jd - 2415020+0.5*exp_array[i]/86400
					sat.compute(obs)
					ra1, dec1 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
					x1, y1, mask1 = astro.world2pix(midlst, ra1, dec1, jd=midJD)

					# End (end-point of next i.e. positive segment)
					obs.date = jd - 2415020+1.5*exp_array[i]/86400 
					sat.compute(obs)
					ra2, dec2 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
					x2, y2, mask2 = astro.world2pix(midlst, ra2, dec2, jd=midJD)

					if mask0 and mask1 and mask2: 

						# LSTSEQ for second (i.e. positive) part of line segment - could be wrong (think astro solution)
						pos_lstseq = int(seq) + 1
						pos_seq = '{:08d}'.format(pos_lstseq)                   

						satnum = tle[1][2:8]
						passages[seq][satnum] = {'negative':{}, 'positive':{}}
						passages[seq][satnum]['negative'] = {'start':{}, 'end':{}, 'lstseq':seq}
						passages[seq][satnum]['positive'] = {'start':{}, 'end':{}, 'lstseq':pos_seq}

						# FIRST SEGMENT
						passages[seq][satnum]['negative']['start']['jd']  = jd-0.5*exp_array[i]/86400
						passages[seq][satnum]['negative']['start']['lst'] = lst_array[i]-0.5*exp_array[i]/3600
						passages[seq][satnum]['negative']['start']['x'] = x0[0]
						passages[seq][satnum]['negative']['start']['y'] = y0[0]
						passages[seq][satnum]['negative']['end']['jd']  = jd+0.5*exp_array[i]/86400
						passages[seq][satnum]['negative']['end']['lst'] = lst_array[i]+0.5*exp_array[i]/3600
						passages[seq][satnum]['negative']['end']['x'] = x1[0]
						passages[seq][satnum]['negative']['end']['y'] = y1[0]

						# SECOND SEGMENT
						passages[seq][satnum]['positive']['start']['jd']  = jd+0.5*exp_array[i]/86400
						passages[seq][satnum]['positive']['start']['lst'] = lst_array[i]+0.5*exp_array[i]/3600
						passages[seq][satnum]['positive']['start']['x'] = x1[0]
						passages[seq][satnum]['positive']['start']['y'] = y1[0]
						passages[seq][satnum]['positive']['end']['jd']  = jd+1.5*exp_array[i]/86400
						passages[seq][satnum]['positive']['end']['lst'] = lst_array[i]+1.5*exp_array[i]/3600
						passages[seq][satnum]['positive']['end']['x'] = x2[0]
						passages[seq][satnum]['positive']['end']['y'] = y2[0]


	pickle.dump(passages, open(f"{rootdir}{user}/my_code/diffimage_passages/diffimage_passages_{target}.p", "wb" ))


for sss in range(len(longsets[0:10])):
	diffimages_fn(sss)
	break
