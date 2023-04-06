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
import pandas as pd
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
    passed_sats = pd.read_pickle(f"{subtracted}{target}/passed_satellites_{target}.p")

    # Find any Starlink TLEs in the passages
    idx = []
    flatlist = np.asarray(starlink_tles).flatten()
    for key in passed_sats.keys():
        line1 = passed_sats[key]['TLE line1'].strip()
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


def passage_check(jd0, jd1, midJD, tles, passages):
    """
    Checks if any Starlinks passed the camera at a given time (i.e. the midJD of the image)

    Input:
    -   midJD:    JD to which images are shifted (we use this as the date of observation)
    -   tles:     reduced list of Starlink TLEs (output of the reduce_tles() function)
    -   passages: pickle file of the satellite passages for a given camera over a given date    

    Ouput:
    -   List of indices for the TLE list that give the Starlink TLEs passing overhead the 
        particular camera at the particular time 

    """
        
    #idx_reduced = []
    #sats = passages[midJD].keys() #these are the satellite numbers 
    
    # Make list of satellite numbers from the TLEs
    #satnums = []
    #for tle in tles:
    #    satnums.append(tle[1].split()[1])

    ## Now cross-referencing
    #for i, sat in enumerate(sats):
    #    if sat in satnums:
    #        idx = satnums.index(sat)
    #        idx_reduced.append(idx)
    
    idx_reduced = []
    sats_mid = passages[midJD].keys()
    sats_jd0 = passages[jd0].keys()
    sats_jd1 = passages[jd1].keys()

    satnums = []
    for tle in tles:
        satnums.append(tle[1].split()[1])
        
    common_sats = set(sats_mid).intersection(sats_jd0, sats_jd1)

    # Cross-reference
    for sat in common_sats:
        if sat in satnums:
            idx = satnums.index(sat)
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
    
    images = np.sort(glob.glob(f'{subtracted}{target}/diff_*.fits.gz')[700:750])

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


def image_timerange(imginfo, timescale):
    """
    Creates a timerange spanned by the date / camera dependent collection of images

    Input:
    -   imginfo: the output of the image_info() function

    Output:
    -   Skyfield time instance of a timerange divided into steps of ~3 minutes

    """

    dates = []
    for lst in list(imginfo):
        dates.append(imginfo[lst]['midJD'])

    oldest = min(dates)
    newest = max(dates)
    t_old = Time(oldest, format='jd') #astropy gives closer result to Pyephem than Skyfield..
    t_new = Time(newest, format='jd')
    timerange = np.linspace(t_old, t_new, 150, endpoint=True) # every ~3mins

    return timescale.from_astropy(timerange)


#---------------------------------------------------------------------------------------------------


def check_illumination(imginfo, midJD, timerange, sat):
    """
    Computes whether or not the given satellite is illuminated by the Sun for a given time

    Input:
    -   imginfo:     output of the image_info() function
    -   midJD:       date of observation for image
    -   timerange:   output of image_timerange() function
    -   sat:         Skyfield EarthSatellite instance of a Starlink satellite

    Output:
    -   illuminated: a Boolean specifying whether or not the satellite is illuminated at the midJD

    """

    # Check when satellite is sunlit  
    sunlit = sat.at(timerange).is_sunlit(eph) #returns a list of bools for each time

    # Obtain the indices of the first and last True element for each sequence of True elements 
    sunlit_idx = []
    start_idx = None
    for i, elem in enumerate(sunlit):
        if elem:
            # Checking if the start of a sequence of True elements
            if start_idx is None:
                start_idx = i
        else:
            # Checking if a sequence has just ended
            if start_idx is not None:
                sunlit_idx.append((start_idx, i-1)) #i-1 since at first False after True sequence
                start_idx = None

    # If the last element is True, we need to append its index as well
    if start_idx is not None:
        sunlit_idx.append((start_idx, len(sunlit)-1))
    
    # Getting the times from the indices
    sunlit_times = []
    for idx in sunlit_idx:
        sunlit_times.append([timerange.tt[idx[0]], timerange.tt[idx[1]]])

    # Check if midJD is within each period and flag it if so
    illuminated = False
    for sunlitrng in sunlit_times:
        if midJD >= (sunlitrng[0] - 1/86400) and midJD <= (sunlitrng[1] + 1/86400): #bumper of 1 sec
            illuminated = True
            if illuminated:
                break

    return illuminated


#---------------------------------------------------------------------------------------------------

# ========  Set-up: there are a number of things we don't want to repeat in the main loop ======== #

#--------------------------------------------------------------------------------------------------- 


""" 
Collecting the TLEs, passages, image info, reduced Fast curves, and image timeranges FOR EACH CAMERA
are operations that can be done outside the main loop (saves some time, particularly for collecting 
the image info as this is quite expensive for many images)

"""

ts = load.timescale()
t0 = time.time()
tles = []
passgs = []
imginfo = []
fastfiles = []
timeranges = []
camids = ['LSC', 'LSN', 'LSS', 'LSE', 'LSW']
pools = [{}] * len(camids)

print('Setting-up')
for camid in camids:
    print(camid)
    d = image_info(target=f'{date}{camid}')
    imginfo.append(d)
    timeranges.append(image_timerange(d, ts))
    tles.append(reduce_tles(target=f"{date}{camid}"))
    passgs.append(pd.read_pickle(f"{subtracted}{date}{camid}/passages_{date}{camid}.p"))
    fastdir = f"/net/beulakerwijde/data1/bring/testdata/lc/{date}{camid}"
    fastfiles.append(h5py.File(f"{fastdir}/lightcurves/fast_{date}{camid}.hdf5", "r"))

# Get La Silla info (will be the same for each camid)
cfg = '/net/beulakerwijde/data1/breslin/code/fotos-python3/bringfiles/siteinfo.dat'
dtype = [('sitename', '|U20'), ('lat', 'float32'), ('lon', 'float32'), ('height', 'float32'), ('ID', '|U2')]
siteinfo = np.genfromtxt(cfg, dtype=dtype)   
mask = siteinfo['ID'] == 'LS'
site = siteinfo[mask]

# Define observer
mascara = wgs84.latlon(latitude_degrees=site[0][1], longitude_degrees=site[0][2], elevation_m=site[0][3])

# DE421 planetary ephemeris (to get position of Sun)
eph = load('de421.bsp')


#---------------------------------------------------------------------------------------------------

# ================== Main Loop - loops over (1) each LSTSEQ and (2) each camera ================== #

#--------------------------------------------------------------------------------------------------- 


print(f'\nSet-up runtime: {time.time() - t0}')
print('Beginning main loop\n')
t0 = time.time()

for seq in fastfiles[0]['station']['lstseq'][:]: #LSTSEQs will be the same for each camid
    lstseq = str(seq)

    """ 
    For every LSTSEQ, we will check the 'crossing quality' for EACH of the five cameras.
    Hence, we now loop over each camera such that each iteration includes the camera dependent 
    image info, camid, fast curve, tles, and passages.

    """

    for data, timerange, camid, fast, starlinks, passages, pool in zip(
        imginfo, timeranges, camids, fastfiles, tles, passgs, pools):

        # This is required if we are not searching through ALL images
        if lstseq not in data.keys():
            continue
   
        # Set time of observation
        jd0 = data[lstseq]['JD0']
        jd1 = data[lstseq]['JD1']
        midJD = data[lstseq]['midJD']
        t = Time(midJD, format='jd') #astropy instance found to be closer to Pyephem than Skyfield!

        # Obtain any passages overhead MASCARA
        idx_reduced = passage_check(jd0, jd1, midJD, starlinks, passages)
        if len(idx_reduced) == 0:
            print(f'{lstseq}: No starlinks passing MASCARA')
            continue
            
        # Sun must be low enough below the horizon, otherwise data is not good enough
        observer = mascara + eph['earth']
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
        midLST = data[lstseq]['midLST']

        # We now check the quality of each Starlink
        for idx in idx_reduced:
            line1 = starlinks[idx][0]
            line2 = starlinks[idx][1]
            line3 = starlinks[idx][2] 
            sat = EarthSatellite(line2, line3, line1, ts)

            diff = sat - mascara
            topocentric = diff.at(ts.from_astropy(t))
            alt, az, dist = topocentric.altaz()
            
            # Criteria check: if satellite is sufficiently above the horizon
            if alt.degrees >= 20:

                # Criteria check: if satellite is illuminated
                illuminated = check_illumination(data, midJD, timerange, sat)
                if illuminated:
                    print('Illuminated!')
                    ra, dec, radec_dist = topocentric.radec() 
                    radeg = ra._degrees
                    dedeg = dec._degrees

                    # Now check is the Starlink is in the camera FOV
                    x, y, mask = astrofns.world2pix(midLST, radeg, dedeg, jd=midJD)

                    if mask:

                        if lstseq not in pool:
                            pool[lstseq] = {}

                        if jd0 not in pool[lstseq]:
                            pool[lstseq][jd0] = {}
                            pool[lstseq][jd0] = [line2[2:8]]
                        else:
                            if line2[2:8] not in pool[lstseq][jd0]:
                                pool[lstseq][jd0].append(line2[2:8])

                        if jd1 not in pool[lstseq]:
                            pool[lstseq][jd1] = {}
                            pool[lstseq][jd1] = [line2[2:8]]
                        else:    
                            if line2[2:8] not in pool[lstseq][jd1]:
                                pool[lstseq][jd1].append(line2[2:8])

                        if midJD not in pool[lstseq]:
                            pool[lstseq][midJD] = {}
                            pool[lstseq][midJD] = [line2[2:8]]
                        else:    
                            if line2[2:8] not in pool[lstseq][midJD]:
                                pool[lstseq][midJD].append(line2[2:8])

                        if midJD == jd0:
                            print(lstseq, 'midJD == jd0')

                        if midJD == jd1:
                            print(lstseq, 'midJD == jd1')

                        if jd1 == jd0:
                            print(lstseq, 'jd1 == jd0')
        
        print('\n')

for fast in fastfiles:
    fast.close()

for (pool, camid) in zip(pools, camids):
    
    if len(pool) > 0:
        pickle.dump(pool, open(f'selection_pool/pool_{camid}.p', 'wb'))

print(f'Loop time: {time.time() - t0}')
