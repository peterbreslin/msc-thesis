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
import tracemalloc
import pickle as pickle
import astropy.io.fits as pf
import bringreduce.bringio as bringio
import bringreduce.mascara_astrometry as astrometry

tracemalloc.start()

if not os.path.exists(f"{rootdir}{user}/data/subtracted/{target}"):
	os.makedirs(f"{rootdir}{user}/data/subtracted/{target}")
	print(f"Created {rootdir}{user}/data/subtracted/{target}")

elif os.path.exists(f"{rootdir}{user}/data/subtracted/{target}/passages_{target}.p"):

	choice = input("Warning! Dictionaries have already been created in this directory! Do you want to continue? [y/n] ").lower()

	if choice == "y":
		pass
	else:
		sys.exit()


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
print('siteinfo', siteinfo)

# Setting observer ---> use Skyfield?
obs = ephem.Observer()
edate = ephem.Date('2000/01/01 12:00:00.0')
obs.lat = siteinfo['lat']*np.pi/180
obs.long = siteinfo['lon']*np.pi/180
obs.elev = siteinfo['height']
obs.epoch = edate    

# Output dictionaries
passages = {}
passed_satellites = {}

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
	current, peak = tracemalloc.get_traced_memory()
	print(f"Get midimage: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")  
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
	
	astro = astrometry.Astrometry(wcspars, polpars)

	current, peak = tracemalloc.get_traced_memory()
	print(f"Astrometric solution: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")  
	
	t1 = time.time()
	print("Setting up: " str(t1-t0))
	print(f"Adding {nlong[sss]} images")
	JDs = []
	lst_array = []
	exp_array = []    
	lstseq_array = []
	
	sub = subtract.Subtractor(wcspars)
	for iii in range(nlong[sss]):
		data0, header0 = pf.getdata(files[longsets[sss][iii]], header=True)
		curlst = header0['LST']
		curexp = header0['EXPTIME']
		JDs.append(header0['JD'])
		lstseq_array.append(header0['LSTSEQ'])
		lst_array.append(curlst)
		exp_array.append(curexp)
		sub.add_imageS(curlst, data0[ly:uy,lx:ux], curexp, midlst, astro, method="Course")
	
		current, peak = tracemalloc.get_traced_memory()
		print(f"Add images: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")   
	print('Finished adding images')

	lst_array = np.array(lst_array)
	JDs = np.array(JDs)
	exp_array = np.array(exp_array)
	lstseq_array = np.array(lstseq_array)
	
		
	for tle in tles:
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

					ra0, dec0 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
					x0, y0, mask0 = astro.world2pix(midlst, ra0, dec0, jd=midJD)
					obs.date = jd - 2415020+0.5*exp_array[i]/86400

					sat.compute(obs)
					ra1, dec1 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
					x1, y1, mask1 = astro.world2pix(midlst, ra1, dec1, jd=midJD)

					if mask0 and mask1:                    
						if abs(lst_array[i]+0.5*exp_array[i]/3600 - obs.sidereal_time()*24./(2.*np.pi))*3600. > 2.:
							sys.exit("LST of image differs by more than 2 secs from PyEphem LST (Check e.g. coords of location)")
						

						# I think this overwrites each satellite!!!! Probably why index by JD..

						satnum = tle[1][2:8]
						passages[seq][satnum] = {'start':{}, 'end':{}, 'JD': jd}

						passages[seq][satnum]['start']['jd']  = jd-0.5*exp_array[i]/86400
						passages[seq][satnum]['start']['lst'] = lst_array[i]-0.5*exp_array[i]/3600
						passages[seq][satnum]['start']['ra']  = ra0
						passages[seq][satnum]['start']['dec'] = dec0
						passages[seq][satnum]['start']['x0']  = x0[0]
						passages[seq][satnum]['start']['y0']  = y0[0]

						passages[seq][satnum]['end']['jd']  = jd+0.5*exp_array[i]/86400
						passages[seq][satnum]['end']['lst'] = lst_array[i]+0.5*exp_array[i]/3600
						passages[seq][satnum]['end']['ra']  = ra1
						passages[seq][satnum]['end']['dec'] = dec1
						passages[seq][satnum]['end']['x0']  = x1[0]
						passages[seq][satnum]['end']['y0']  = y1[0]                            
						

						if satnum not in list(passed_satellites.keys()):                       
							passed_satellites[satnum] = {}
							passed_satellites[satnum]['line0'] = tle[0][2:] 
							passed_satellites[satnum]['line1'] = tle[1]
							passed_satellites[satnum]['line2'] = tle[2]
							passed_satellites[satnum]['passages'] = {1:{'jd':jd, 'lst':lst_array[i]}} 
						else:                           
							n_overflight = max(passed_satellites[satnum]['passages'].keys())+1 
							passed_satellites[satnum]['passages'][n_overflight] = {'jd':jd, 'lst':lst_array[i]}
	 
		
	current, peak = tracemalloc.get_traced_memory()
	print(f"Satellite positions: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
	pickle.dump(passages, open(f"{rootdir}{user}/data/subtracted/{target}/passages_{target}.p", "wb" ))
	pickle.dump(passed_satellites, open(f"{rootdir}{user}/data/subtracted/{target}/passed_satellites_{target}.p", "wb"))
	


	
	t2 = time.time()
	print('Beginning difference of two images')
	print("Collecting images: " + str(t2-t1))

	subimage, submean, submedian, substd, subheader = sub.get_image(midlst)
	current, peak = tracemalloc.get_traced_memory()
	print(f"Collecting images: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

	t3 = time.time()
	print("Creating delta-images: " + str(t3-t2))
	
	for iii in range(nlong[sss]-1):
		print("Processing " + str(iii))
		
		curlst0 = lst_array[iii]
		curlst1 = lst_array[iii+1]
		JD0 = JDs[iii]
		JD1 = JDs[iii+1]
		exp0 = exp_array[iii]
		exp1 = exp_array[iii+1]
		curlstseq = lstseq_array[iii]
		
		print("Creating header")
		diffheader = pf.Header()
		diffheader['LSTSEQ'] = (curlstseq, 'LSTSEQ of image')
		diffheader['LST0'] = (curlst0, 'LST of first (subtracted) image')
		diffheader['LST1'] = (curlst1, 'LST of second image')
		diffheader['JD0']  = (JD0, 'JD of first (subtracted) image')
		diffheader['JD1']  = (JD1, 'JD of second image')
		diffheader['EXP0'] = (exp0, 'Exposure of first (subtracted) image')
		diffheader['EXP1'] = (exp1, 'Exposure of second image')
		diffheader['MIDLST'] = (midlst, 'LST to which images are shifted')
		diffheader['MIDJD'] = (midJD, 'JD to which images are shifted')
		diffheader['XSIZE'] = nx
		diffheader['YSIZE'] = ny
		diffheader['X0'] = lx 
		diffheader['Y0'] = ly
		
		filename = 'diff_{:08d}{}.fits.gz'.format(curlstseq, camid)
		filename = os.path.join(f"{rootdir}{user}/data/subtracted/{target}", filename)
	
		hdu = pf.PrimaryHDU(data=subimage[:,:,iii], header=diffheader)
		print("Saving file")
		hdu.writeto(filename, overwrite=True)
		current, peak = tracemalloc.get_traced_memory()
		print(f"Saving images: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

			  

current, peak = tracemalloc.get_traced_memory()
print(len(longsets))
print(f"Start: Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")  

for sss in range(len(longsets[0:5])):
	diffimages_fn(sss)
