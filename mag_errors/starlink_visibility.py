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


10/05 --> working with passages that are already reduced to Starlinks only

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
user_path = "/net/beulakerwijde/data1/breslin/my_code/"

sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
import bringreduce.mascara_astrometry as astrometry


if not os.path.exists(f"/net/beulakerwijde/data1/breslin/my_code/selection_pool"):
    os.makedirs(f"/net/beulakerwijde/data1/breslin/my_code/selection_pool")


#---------------------------------------------------------------------------------------------------

# ========================  Defining some Functions used in the main loop ======================== #

#--------------------------------------------------------------------------------------------------- 


def reduce_tles(target):
    """ 
    1) Reduces TLE list of all Earth orbiting satellites for a given date to Starlink TLEs only. 
    2) Further reduces this list to those passing overhead a given camera.

    NOTE: already have this info in the passages file but it is quicker this way

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
    passed_sats = pickle.load(open(f"{user_path}passages_full_track/passed_satellites_{target}.p", "rb"))
    satnums = list(passed_sats)

    # Reduce all TLEs to Starlinks passing over the given camera by comparing the satellite numbers
    idx = []
    for i, tle in enumerate(starlink_tles):
        satnum = tle[1].split()[1]
        if satnum in satnums:
            idx.append(i)

    # These TLEs are now only those Starlinks that are passing the camera over the given date
    reduced_tles = [starlink_tles[i] for i in idx]

    # Remove '0' label from the first line of each TLE (makes things handier later on)
    for tle in reduced_tles:
        tle[0] = tle[0][2:]
    
    return reduced_tles


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

        if jd0 >= (JD0 - 5/86400) and jd0 <= (JD0 + 5/86400):
            if jd1 >= (JD1 - 5/86400) and jd1 <= (JD1 + 5/86400):
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
    
    images = np.sort(glob.glob(f'{subtracted}{target}/diff_*.fits.gz'))

    imginfo = {}
    for img in images:
        header = pf.getheader(img)
        lstseq = img[-19:-11]
        imginfo[lstseq] = {}
        imginfo[lstseq]['nx'] = header['XSIZE']
        imginfo[lstseq]['ny'] = header['YSIZE']
        imginfo[lstseq]['JD0'] = header['JD0']
        imginfo[lstseq]['JD1'] = header['JD1']
        imginfo[lstseq]['midJD'] = header['MIDJD']
        imginfo[lstseq]['midLST'] = header['MIDLST']
    return imginfo


#---------------------------------------------------------------------------------------------------


def image_timerange(header, ts):
    
    dates = []
    for lstseq in header.keys():
        dates.append(header[lstseq]['JD0'])
        dates.append(header[lstseq]['JD1'])
    
    # Adding a bumper of 1 second
    t0 = min(dates) - 1/86400
    t1 = max(dates) + 1/86400

    # Calculating the number of seconds between t0 and t1
    delta_t = (t1 - t0) * 86400  # convert to seconds
    
    # Converting to astropy time instances (more accurate?)
    t0 = Time(t0, format='jd')
    t1 = Time(t1, format='jd')

    # Generating a range of dates separated by 15 seconds (~2 exposures)
    timerange = np.linspace(t0, t1, num=int(delta_t/15)+1) # +1 to include endpoints

    return ts.from_astropy(timerange)


#---------------------------------------------------------------------------------------------------


def check_illumination(header, JD0, JD1, timerange, sat):

    # Check when satellite is illuminated 
    sunlit = sat.at(timerange).is_sunlit(eph)
    
    """
    The sunlit array is an array of True and False statements, where True corresponds to the satellite being
    sunlit. We want to now find the indices of each sunlit period so that we can correlate these to the 
    corresponding times of illumination. A sunlit period would be a sequence of True statements, where the first
    marks the beginning and the last marking the end. There may be multiple sunlit periods within each timerange.
    
    The timerange is a list of times separated by 15 seconds (~2 exposures). The start and end times are from the
    oldest and newest image in the collection of images used for each camid (i.e. from the header info).
        
    """

    # Obtain the indices of the first and last True element for each sequence of True elements 
    idx_sunlit = []
    idx_start  = None
    for i, elem in enumerate(sunlit):
        if elem:
            # Checking if the start of a sequence of True elements
            if idx_start is None:
                idx_start = i
        else:
            # Checking if a sequence has just ended
            if idx_start is not None:
                idx_sunlit.append((idx_start, i-1)) # -1 because we're at the first False element after True sequence
                idx_start = None

    # If the last element is True, we need to append its index as well
    if idx_start is not None:
        idx_sunlit.append((idx_start, len(sunlit)-1))
    
    # Correlating the indices to the correpsonding times
    sunlit_range = []
    for idx in idx_sunlit:
        sunlit_range.append([timerange.tt[idx[0]], timerange.tt[idx[1]]])

    # Check if the midpoint time is within each period and flag it if so
    illuminated = False
    for t in sunlit_range:
        if JD0 >= (t[0] - 1/86400) and JD1 <= (t[1] + 1/86400): #adding a bumper of one second
            illuminated = True
            if illuminated:
                break
                
    return illuminated    


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
timeranges = []
all_passages = []
camids = ['LSC', 'LSN', 'LSS', 'LSE', 'LSW']

pools = []
for i in range(len(camids)):
    pools.append({})

print('Setting-up')
for i, camid in enumerate(camids):
    print(camid)
    target = date + camid
    imginfo.append(image_info(target=f"{target}"))
    all_tles.append(reduce_tles(target=f"{target}"))
    timeranges.append(image_timerange(imginfo[i], ts))
    fastdir = f"/net/beulakerwijde/data1/bring/testdata/lc/{target}"
    fastfiles.append(h5py.File(f"{fastdir}/lightcurves/fast_{target}.hdf5", "r"))
    all_passages.append(pickle.load(open(f"{user_path}passages_full_track/passages_{target}.p", "rb")))

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


""" 
For every LSTSEQ, we will check the 'crossing quality' for EACH of the five cameras.
Hence, we now loop over each camera such that each iteration includes the camera dependent 
image info, camid, fast curve, tles, and passages.

"""

print(f'\nSet-up runtime: {time.time() - t0}')
t0 = time.time()

for camid, header in zip(camids, imginfo):
    print(f'{camid}: checking {len(header)} images')

for seq in fastfiles[0]['station']['lstseq'][:]: #LSTSEQs will be the same for each camid
    lstseq = str(seq)
    
    for header, camid, timerange, fast, tles, passages, pool in zip(
        imginfo, camids, timeranges, fastfiles, all_tles, all_passages, pools):
       
        # This is required if we are not searching through ALL images
        if (lstseq not in header.keys()) or (lstseq not in passages.keys()):
            continue
        
        # Set time of observation
        JD0 = header[lstseq]['JD0']
        JD1 = header[lstseq]['JD1']
        mid = (JD0+JD1)/2.
        t = Time(mid, format='jd') 

        # Obtain passages overhead MASCARA
        idx_reduced = passage_check(lstseq, tles, passages, JD0, JD1)
        if len(idx_reduced) == 0:
            #print(f'{lstseq}{camid}: No starlinks passing MASCARA')
            continue
            
        # Sun must be low enough below the horizon, otherwise data is not good enough
        sun_pos = observer.at(ts.from_astropy(t)).observe(eph['sun'])
        sun_alt, sun_az, sun_dist = sun_pos.apparent().altaz()
            
        if sun_alt.degrees > -18.:
            print("Sun is less than 18 degrees below the horizon")
            continue

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
                illumination = check_illumination(header, JD0, JD1, timerange, sat)
                    
                if illumination:
                    # Now check is the Starlink is in the camera FOV
                    ra, dec, radec_dist = topocentric.radec() 
                    radeg = ra._degrees
                    dedeg = dec._degrees
                    x, y, mask = astrofns.world2pix(midLST, radeg, dedeg, jd=midJD)

                    if mask:

                        #if idx==idx_reduced[0]:
                        #    print(f'{camid} {lstseq} in FOV')
                        
                        if lstseq not in pool:
                            pool[lstseq] = {}
                            pool[lstseq] = [line2[2:8]]

                        if line2[2:8] not in pool[lstseq]:
                            pool[lstseq].append(line2[2:8])
        

for (fast, camid, pool) in zip(fastfiles, camids, pools):        
    fast.close()
    if len(pool) != 0:
        print(f"{camid}: Saving {len(pool)} images")
        pickle.dump(pool, open(f'{user_path}selection_pool/pool_{camid}.p', 'wb'))


print(f'Runtime: {time.time() - t0}')
