"""
This is a script to estimate whether or not a Starlink satellite is expected to cross the MASCARA 
FOV for a given image.

The script works to reduce the number of difference image files (i.e. the images obtained from 
CreateDiffImages.py) such that the reduced image pool should have at least one Starlink satellite
present in each image. 

For each image (or more specifically, for each LSTSEQ), it is determined if there are Starlink 
satellites overhead MASCARA at the time of observation. For each Starlink deemed present, we then 
estimate if it will be visible. This is achieved by checking its altitude and illumination. 

If a Starlink checks these boxes, we then determine what camera (LSC, LSN, LSS, LSE, LSW) it is 
expected to have crossed. For now, this outputs a text file with a list of the difference image 
file names (LSTSEQ + camid) expected to have a Starlink trail present.

"""

import os
import sys
import h5py
import glob
import time
import argparse
import numpy as np
import pickle as pickle
import astropy.io.fits as pf
from astropy.time import Time
from skyfield.api import load, wgs84, EarthSatellite


#---------------------------------------------------------------------------------------------------


# Date of observation required (e.g. -d '20221023')
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", type=str, default='20221023', help="the date of observation")
args = parser.parse_args()
date = args.date
subtracted = f"/net/beulakerwijde/data1/breslin/data/subtracted/"

sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
import bringreduce.mascara_astrometry as astrometry


#---------------------------------------------------------------------------------------------------

# ========================  Defining some Functions used in the main loop ======================== #

#--------------------------------------------------------------------------------------------------- 


def reduce_tles(target):
	""" 
	1) Reduces TLE list of all Earth orbiting satellites for a given date to Starlink TLEs only. 
	2) Further reduces this list to those passing overhead a given camera.

	Input:  
	-   target: date + camid (e.g '202210LSC') 

	Output: 
	-   list of reduced, camera dependent, Starlink TLEs

	"""

	# Load TLEs for all passages
	satfiles = f"/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/{date}/3leComplete.txt"
	with open(satfiles) as f:
		all_tles = f.readlines()
		f.close()   

	# Split TLE list into individual lists for each TLE
	all_tles = [i.strip() for i in all_tles]
	tles = [all_tles[x:x+3] for x in range(0, len(all_tles), 3)]

	# Reduce TLEs to Starlink only
	starlink_tles = []
	for tle in tles:
		if "STARLINK" in tle[0]:
			starlink_tles.append(tle)

	# Obtain satellite passages over given camera
	passed_sats = pickle.load(open(f"{subtracted}{target}/passed_satellites_{target}.p", "rb"))

	# Find any Starlink TLEs in the passages
	idx = []
	flatlist = np.asarray(starlink_tles).flatten()
	for key in passed_sats.keys():
		line1 = passed_sats[key]['line1'].strip()
		i = np.where(flatlist == line1)[0] 
		if i.size > 0:
			idx.append(i[0] - 1) #appending the name of the starlink sat

	# Indices for the flattened TLE list --> divide by 3 to get indices for the original list
	orig_idx = [int(x/3) for x in idx]
	passed_tles = [starlink_tles[i] for i in orig_idx]

	# Remove '0' label from the first line of each TLE (makes things easier later on)
	for tle in passed_tles:
		tle[0] = tle[0][2:]
	
	return passed_tles


#---------------------------------------------------------------------------------------------------


def passage_check(lstseq, tles, passages, JD0, JD1):
	"""
	Checks if any Starlinks passed the camera at a given time

	Input:
	-   tles:     reduced list of Starlink TLEs (output of the reduce_tles() function)
	-   passages: pickle file of the satellite passages for a given camera over a given date    

	Ouput:
	-   List of indices for the TLE list that give the Starlink TLEs passing overhead the 
		particular camera at the particular time 

	"""

	tle_satnums = []
	for tle in tles:
		tle_satnums.append(tle[1].split()[1])
		
	satnums = []
	sats = passages[lstseq].keys()
	for sat in sats:
		jd0 = passages[lstseq][sat]['start']['jd']
		jd1 = passages[lstseq][sat]['end']['jd']

		if jd0 >= (JD0 - 7/86400) and jd0 <= (JD0 + 7/86400):
			if jd1 >= (JD1 - 7/86400) and jd1 <= (JD1 + 7/86400):
				satnums.append(sat)  
			
	# Cross-reference
	idx_reduced = []
	for sat in satnums:
		if sat in tle_satnums:
			idx = tle_satnums.index(sat)
			idx_reduced.append(idx)

	return idx_reduced


#---------------------------------------------------------------------------------------------------


def image_info(target):
	"""
	Collects the relevant information from each image header and stores it in a dictionary.

	Input:  
	- target: date + camid (e.g. '20221023LSC')

	Output: 
	- dictionary containing the date / camera dependent header information for all images

	"""
	
	images = np.sort(glob.glob(f'{subtracted}{target}/diff_*.fits.gz')[0:5])

	imginfo = {}
	for img in images:
		header = pf.getheader(img)
		lstseq = img[-19:-11]
		imginfo[lstseq] = {}
		imginfo[lstseq]['JD0'] = header['JD0']
		imginfo[lstseq]['JD1'] = header['JD1']
		imginfo[lstseq]['midLST'] = header['MIDLST']
		imginfo[lstseq]['midJD'] = header['MIDJD']
		imginfo[lstseq]['nx'] = header['XSIZE']
		imginfo[lstseq]['ny'] = header['YSIZE']
	return imginfo


#---------------------------------------------------------------------------------------------------

# ========  Set-up: there are a number of things we don't want to repeat in the main loop ======== #

#--------------------------------------------------------------------------------------------------- 


""" 
Collecting information FOR EACH CAMERA that can be done outside the main loop (saves some time, 
particularly for collecting the image info as this is quite expensive for many images).

"""

t0 = time.time()
ts = load.timescale()

imginfo = []
all_tles = []
fastfiles = []
all_passages = []
camids = ['LSC', 'LSN', 'LSS', 'LSE', 'LSW']
pools = [{}] * len(camids)

print('Setting-up')
for camid in camids:
	print(camid)
	imginfo.append(image_info(target=f'{date}{camid}'))
	all_tles.append(reduce_tles(target=f"{date}{camid}"))
	fastdir = f"/net/beulakerwijde/data1/bring/testdata/lc/{date}{camid}"
	fastfiles.append(h5py.File(f"{fastdir}/lightcurves/fast_{date}{camid}.hdf5", "r"))
	all_passages.append(pickle.load(open(f"{subtracted}{date}{camid}/passages_{date}{camid}.p", "rb")))

# Get La Silla info (will be the same for each camid)
cfg = '/net/beulakerwijde/data1/breslin/code/fotos-python3/bringfiles/siteinfo.dat'
dtype = [('sitename', '|U20'), ('lat', 'float32'), ('lon', 'float32'), ('height', 'float32'), ('ID', '|U2')]
siteinfo = np.genfromtxt(cfg, dtype=dtype)   
mask = siteinfo['ID'] == 'LS'
site = siteinfo[mask]

# DE421 planetary ephemeris (to get position of Sun)
eph = load('de421.bsp')

# Define observer
mascara = wgs84.latlon(latitude_degrees=site[0][1], longitude_degrees=site[0][2], elevation_m=site[0][3])
observer = mascara + eph['earth']


#---------------------------------------------------------------------------------------------------

# ================== Main Loop - loops over (1) each LSTSEQ and (2) each camera ================== #

#--------------------------------------------------------------------------------------------------- 


print(f'\nSet-up runtime: {time.time() - t0}')
t0 = time.time()

def create_selection_pool():

	for seq in fastfiles[0]['station']['lstseq'][:]: #LSTSEQs will be the same for each camid
		lstseq = str(seq)

		""" 
		For every LSTSEQ, we will check the 'crossing quality' for EACH of the 5 cameras.
		Hence, we loop over each camera such that each iteration includes the camera 
		dependent image info (header), camid, fastcurve, tles, and passages.

		"""

		for header, camid, fast, tles, passages, pool in zip(
			imginfo, camids, fastfiles, all_tles, all_passages, pools):

			# This is required if we are not searching through ALL images
			if (lstseq not in header.keys()) or (lstseq not in passages.keys()):
				continue
	   
			# Set time of observation
			JD0 = header[lstseq]['JD0']
			JD1 = header[lstseq]['JD1']
			mid = (JD0+JD1)/2
			t = Time(mid, format='jd') 

			# Obtain passages overhead MASCARA
			idx_reduced = passage_check(lstseq, tles, passages, JD0, JD1)
			if len(idx_reduced) == 0:
				print(f'{lstseq}{camid}: No starlinks passing MASCARA')
				continue
				
			# Sun must be low enough below the horizon, otherwise data is not good enough
			sun_pos = observer.at(ts.from_astropy(t)).observe(eph['sun'])
			sun_alt, sun_az, sun_dist = sun_pos.apparent().altaz()
				
			if sun_alt.degrees > -18.:
				print("Sun is less than 18 degrees below the horizon")
				continue

			print(f'There are {len(idx_reduced)} Starlinks passing overhead {camid} for LSTSEQ={lstseq}')

			# Attaining the astrometric solution (depends on FAST file and LSTseq)
			astro = np.where((fast['astrometry/lstseq'][:] // 50) == (int(lstseq) // 50))[0][0]
			order = fast['astrometry/x_wcs2pix'][astro].shape[0]-1
			lst = fast['station/lst'][np.where(fast['station/lstseq'][:] == (fast['astrometry/lstseq'][astro]))[0][0]]
			nx = header[lstseq]['nx']
			ny = header[lstseq]['ny']

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
			midJD  = header[lstseq]['midJD']
			midLST = header[lstseq]['midLST']

			# We now check the quality of each Starlink
			for idx in idx_reduced:
				line1 = tles[idx][0]
				line2 = tles[idx][1]
				line3 = tles[idx][2] 
				sat = EarthSatellite(line2, line3, line1, ts)

				diff = sat - mascara
				topocentric = diff.at(ts.from_astropy(t))
				alt, az, dist = topocentric.altaz()
				
				# Criteria check: if satellite is sufficiently above the horizon
				if alt.degrees >= 20.:

					# Criteria check: if satellite is illuminated
					if sat.at(ts.tt_jd(JD0)).is_sunlit(eph) | sat.at(ts.tt_jd(JD1)).is_sunlit(eph):

						# Now check is the Starlink is in the camera FOV
						ra, dec, radec_dist = topocentric.radec() 
						radeg = ra._degrees
						dedeg = dec._degrees
						x, y, mask = astrofns.world2pix(midLST, radeg, dedeg, jd=midJD)

						if mask:

							if lstseq not in pool:
								pool[lstseq] = {}
								pool[lstseq] = [line2[2:8]]

							if line2[2:8] not in pool[lstseq]:
								pool[lstseq].append(line2[2:8])
			
			print('\n')

		
	for (fast, camid, pool) in zip(fastfiles, camids, pools):
		fast.close()
		if len(pool) > 0:
			print('Saving')
			print(f"{camid}: {len(pool.keys())} good images")
			pickle.dump(pool, open(f'selection_pool/pool_{camid}.p', 'wb'))


create_selection_pool()
print(f'Runtime: {time.time() - t0}')
