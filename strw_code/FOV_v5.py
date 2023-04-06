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
    passed_sats = pickle.load(open(f"{subtracted}{target}/passed_satellites_{target}.p","rb" ))

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


def image_headers(camid):

    images = np.sort(glob.glob(f'{subtracted}{date}{camid}/*.fits.gz'))[0:50] 
    
    for img in images:    
        header = pf.getheader(img)
        lstseq = img[-19:-11] 
        imgdata[lstseq] = {}
        imgdata[lstseq]['JD0'] = header['JD0']
        imgdata[lstseq]['JD1'] = header['JD1']
        imgdata[lstseq]['midLST'] = header['MIDLST']
        imgdata[lstseq]['midJD'] = header['MIDJD']
        imgdata[lstseq]['nx'] = header['XSIZE']
        imgdata[lstseq]['ny'] = header['YSIZE']

    return imgdata


#---------------------------------------------------------------------------------------------------


def passage_check(midJD, tles, passages):
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
        
    idx_reduced = []
    sats = passages[midJD].keys() #these are the satellite numbers 
    
    # Make list of satellite numbers from the TLEs
    satnums = []
    for tle in tles:
        satnums.append(tle[1].split()[1])

    # Now cross-referencing
    for i, sat in enumerate(sats):
        if sat in satnums:
            idx = satnums.index(sat)
            idx_reduced.append(idx)
  
    return idx_reduced


#---------------------------------------------------------------------------------------------------

# ========  Set-up: there are a number of things we don't want to repeat in the main loop ======== #

#--------------------------------------------------------------------------------------------------- 


""" 
Collecting the TLEs, passages, image info, reduced Fast curves, and image timeranges FOR EACH CAMERA
are operations that can be done outside the main loop (saves some time, particularly for collecting 
the image info as this is quite expensive for many images)

"""

t0 = time.time()
ts = load.timescale()

tles = []
passgs = []
headers = []
fastfiles = []
camids = ['LSC', 'LSN', 'LSS', 'LSE', 'LSW']
pools = [{}] * len(camids)

print('Setting-up')
for camid in camids:
    tles.append(reduce_tles(target=f"{date}{camid}"))
    headers.append(image_headers(camid))
    passgs.append(pickle.load(open(f"{subtracted}{date}{camid}/passages_{date}{camid}.p", "rb" )))
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

for header, camid, fast, starlinks, passages, pool, in zip(
    headers, camids, fastfiles, tles, passgs, pools):
        
    for lstseq in header.keys():
        midJD = header[lstseq]['midJD']

        # Set time of image
        ts = load.timescale()
        t = Time(midJD, format='jd')

        # Obtain passages
        idx_reduced = passage_check(midJD, starlinks, passages)

        if idx_reduced is None:
            print(f'{lstseq}: No starlinks passing MASCARA')
            continue
        print(f'There are {len(idx_reduced)} Starlinks passing overhead {camid} for LSTSEQ={lstseq}')

        # Sun must be low enough below the horizon, otherwise data is not good enough
        observer = mascara + eph['earth']
        sun_pos = observer.at(ts.from_astropy(t)).observe(eph['sun'])
        sun_alt, sun_az, sun_dist = sun_pos.apparent().altaz()
            
        if sun_alt.degrees > -18.:
            print("Sun is less than 18 degrees below the horizon")
            continue

        print(f'There are {len(idx_reduced)} Starlinks passing overhead {camid} for LSTSEQ={lstseq}')

        # Attaining the astrometric solution (depends on FAST file and LSTSEQ)
        astro = np.where((fast['astrometry/lstseq'][:] // 50) == (int(lstseq) // 50))[0][0]
        order = fast['astrometry/x_wcs2pix'][astro].shape[0]-1
        lst = fast['station/lst'][np.where(fast['station/lstseq'][:]==(fast['astrometry/lstseq'][astro]))[0][0]]
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

        jd0 = header[lstseq]['JD0']
        jd1 = header[lstseq]['JD1']
        midLST = header[lstseq]['midLST']
        astrofns = astrometry.Astrometry(wcspars, polpars)

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
                if sat.at(ts.from_astropy(t)).is_sunlit(eph):
                    ra, dec, radec_dist = topocentric.radec() 
                    radeg = ra._degrees
                    dedeg = dec._degrees

                    # Now check is the Starlink is in the camera FOV
                    x, y, mask = astrofns.world2pix(midLST, radeg, dedeg, jd=midJD)

                    if mask:

                        if lstseq not in pool:
                            pool[lstseq] = {}
                            pool[lstseq][jd0] = {}
                            pool[lstseq][jd1] = {}
                            pool[lstseq][midJD] = {}
                            pool[lstseq][jd0] = [line2[2:8]]
                            pool[lstseq][jd1] = [line2[2:8]]
                            pool[lstseq][midJD] = [line2[2:8]]
 
                        else:

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

        print('\n')

for fast in fastfiles:
    fast.close()

for (pool, camid) in zip(pools, camids):
    
    if len(pool) > 0:
        pickle.dump(pool, open(f'selection_pool/pool_{camid}.p', 'wb'))

print(f'Loop time: {time.time() - t0}')
