import os
import sys
import h5py
import glob
import time
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as pf
from astropy.time import Time
from skyfield.api import load, wgs84, EarthSatellite


#-----------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument("-d", "--date", type=str, required=True, help="the date of observation")
args = parser.parse_args()
date = args.date
subtracted = f"/net/beulakerwijde/data1/breslin/data/subtracted/"

sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
import bringreduce.mascara_astrometry as astrometry


#-----------------------------------------------------------------------------------------------------------------------


def reduce_tles(target):

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

    # Obtain satellite passages
    passed_sats = pd.read_pickle(f"{subtracted}{target}/passed_satellites_{target}.p")

    # Find any Starlink TLEs in the passages
    idx = []
    flatlist = np.asarray(starlink_tles).flatten()
    for key in passed_sats.keys():
        line1 = passed_sats[key]['TLE line1'].strip()
        i = np.where(flatlist == line1)[0] 
        if i.size > 0:
            idx.append(i[0] - 1) #appending the name of the starlink sat

    # Now have indices for the flattened Starlink TLE list --> divide by 3 to get indices for the original list
    orig_idx = [int(x/3) for x in idx]
    passed_tles = [starlink_tles[i] for i in orig_idx]

    # Remove 0 labeling of first line of TLE because that's the proper format
    for tle in passed_tles:
        tle[0] = tle[0][2:]
    
    return passed_tles


#-----------------------------------------------------------------------------------------------------------------------


def passage_check(midJD, tles, passages):
        
    idx_reduced = []
    sats = passages[midJD].keys()
    
    satnums = []
    for tle in tles:
        satnums.append(tle[1].split()[1])

    # Cross-reference
    for i, sat in enumerate(sats):
        if sat in satnums:
            idx = satnums.index(sat)
            idx_reduced.append(idx)
  
    return idx_reduced


#-----------------------------------------------------------------------------------------------------------------------


def image_info(target):
    
    images = glob.glob(f'{subtracted}{target}/diff_*.fits.gz')

    imginfo = {}
    for img in images[1000:1050]:
        header = pf.getheader(img) 
        lstseq = img[-19:-11]
        midlst = header['MIDLST']
        midJD = header['MIDJD']
        nx = header['XSIZE']    
        ny = header['YSIZE']

        imginfo[lstseq] = {}
        imginfo[lstseq]['midLST'] = midlst
        imginfo[lstseq]['midJD']  = midJD
        imginfo[lstseq]['nx'] = nx
        imginfo[lstseq]['ny'] = ny
    
    return imginfo


#-----------------------------------------------------------------------------------------------------------------------


def image_timerange(imginfo):
    ts = load.timescale()
    dates = []
    for lst in list(imginfo):
        dates.append(imginfo[lst]['midJD'])

    oldest = min(dates)
    newest = max(dates)
    t_old = Time(oldest, format='jd')
    t_new = Time(newest, format='jd')
    timerange = np.linspace(t_old, t_new, 150, endpoint=True) # every ~3mins

    return ts.from_astropy(timerange)


#-----------------------------------------------------------------------------------------------------------------------


def check_illumination(imginfo, midJD, timerange, sat):

    # Check when satellite is illuminated 
    timerange = image_timerange(imginfo)
    sunlit = sat.at(timerange).is_sunlit(eph)

    # Obtain the indices of the first and last True element for each sequence of True elements 
    sunlit_idx = []
    start_idx = None
    for i, elem in enumerate(sunlit):
        if elem:
            # Checking if the start of a sequence of True elements
            if start_idx is None:
                start_idx = i
        else:
            # Checking if a sequence is just ended
            if start_idx is not None:
                sunlit_idx.append((start_idx, i-1)) # -1 because we're at the first False element after True sequence
                start_idx = None

    # If the last element is True, we need to append its index as well
    if start_idx is not None:
        sunlit_idx.append((start_idx, len(sunlit)-1))
    
    # Getting the times from the indices
    sunlit_times = []
    for idx in sunlit_idx:
        sunlit_times.append([timerange[idx[0]].to_astropy().value, timerange[idx[1]].to_astropy().value])
        
    #sunlit_times.append([timerange.tt[idx[0]], timerange.tt[idx[1]]])

    # Check if midJD is within each period and flag it if so
    illuminated = False
    for sunlitrng in sunlit_times:
        if midJD >= (sunlitrng[0] - 1/86400) and midJD <= (sunlitrng[1] + 1/86400): # adding a bumper of one second
            illuminated = True
            if illuminated:
                break

    return illuminated


#-----------------------------------------------------------------------------------------------------------------------


### THERE ARE A NUMBER OF THINGS WE DON'T WANT TO REPEAT IN THE MAIN LOOP
t0 = time.time()

tles = []
passgs = []
imginfo = []
fastfiles = []
timeranges = []
camids = ['LSC', 'LSN', 'LSS', 'LSE', 'LSW']

print('Setting-up')
for camid in camids:
    d = image_info(target=f'{date}{camid}')
    imginfo.append(d)
    timeranges.append(image_timerange(d))
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


#-----------------------------------------------------------------------------------------------------------------------

print(f'\nSet-up runtime: {time.time() - t0}')
print('Beginning main loop\n')
t0 = time.time()
ts = load.timescale()

# Want to check the date for each camid before moving on to the next date
# The LST sequences will be the same for each camid

result = []
for seq in fastfiles[0]['station']['lstseq'][:]:  
    lstseq = str(seq)
    for data, timerange, camid, fast, starlinks, passages in zip(imginfo, timeranges, camids, fastfiles, tles, passgs):

        # Add this if not searching through ALL images
        if lstseq not in data.keys():
            continue
   
        # Set time of image
        midJD = data[lstseq]['midJD']
        t = Time(midJD, format='jd')

        # Obtain passages
        idx_reduced = passage_check(midJD, starlinks, passages)
        if idx_reduced is None:
            print(f'{lstseq}: No starlinks passing MASCARA')
            continue
            
        else:
            # Sun must be low enough below the horizon, otherwise data is not good enough
            observer = mascara + eph['earth']
            sun_pos = observer.at(ts.from_astropy(t)).observe(eph['sun'])
            sun_alt, sun_az, sun_dist = sun_pos.apparent().altaz()
                
            if sun_alt.degrees > -18.:
                print("Sun is less than 18 degrees below the horizon")
                pass

            else:

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

                for idx in idx_reduced:
                    line1 = starlinks[idx][0]
                    line2 = starlinks[idx][1]
                    line3 = starlinks[idx][2] 
                    sat = EarthSatellite(line2, line3, line1, ts)

                    diff = sat - mascara
                    topocentric = diff.at(ts.from_astropy(t))
                    alt, az, dist = topocentric.altaz()
                    
                    # Criteria check: if satellite is >20 degrees above the horizon
                    if alt.degrees >= 20:

                        # Criteria check: if satellite is illuminated
                        illuminated = check_illumination(data, midJD, timerange, sat)
                        if illuminated:
                            print('Illuminated!')
                            ra, dec, radec_dist = topocentric.radec() 
                            radeg = ra._degrees
                            dedeg = dec._degrees
                            x, y, mask = astrofns.world2pix(midLST, radeg, dedeg, jd=midJD)

                            if mask:
                                print(f"At least one Starlink should be in {camid} FOV")
                                image = lstseq + camid + ' ' + line1
                                result.append(image)
                                break

                print('\n')

for fast in fastfiles:
    fast.close()

if len(result) > 0:
    np.savetxt('/net/beulakerwijde/data1/breslin/selection_pool.txt', result, fmt='%s')

print(f'Loop time: {time.time() - t0}')
