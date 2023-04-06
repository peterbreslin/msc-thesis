import os
import sys
import h5py
import glob
import time
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as pf
import matplotlib.pyplot as plt
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
    tlefile = f"/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/{date}/3leComplete.txt"
    with open(tlefile) as f:
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
    for satnum in passed_sats.keys():
        line1 = passed_sats[satnum]['TLE line1'].strip()
        i = np.where(flatlist == line1)[0] 
        if i.size > 0:
            idx.append(i[0] - 1) #negative 1 to get to line0 i.e. the  name of the starlink sat

    # Now have indices for the flattened Starlink TLE list --> divide by 3 to get indices for the original list
    orig_idx = [int(x/3) for x in idx]
    passed_tles = [starlink_tles[i] for i in orig_idx]

    # Remove 0 labeling of first line of TLE because that's the proper format
    for tle in passed_tles:
        tle[0] = tle[0][2:]
    
    return passed_tles


#-----------------------------------------------------------------------------------------------------------------------


def passage_check(midJD, tles, passages):
    
    idx_reduced = None
    sats = passages[midJD].keys()
    
    satnums = []
    for tle in tles:
        satnums.append(tle[1].split()[1])

    # Cross-reference
    for i, sat in enumerate(sats):
        if sat in satnums:
            idx = satnums.index(sat)
            idx_reduced.append(idx)

    # else: date of image does not correspond to any Starlinks passing overhead
        
    return idx_reduced


#-----------------------------------------------------------------------------------------------------------------------


def image_data(target):
    
    images = glob.glob(f'{subtracted}{target}/diff_*.fits.gz')

    imgdata = {}
    for img in images[700:751]:
        data, header = pf.getdata(img, header=True) 
        lstseq = img[-19:-11]
        midlst = header['MIDLST']
        midJD = header['MIDJD']
        nx = header['XSIZE']    
        ny = header['YSIZE']

        imgdata[lstseq] = {}
        imgdata[lstseq]['midLST'] = midlst
        imgdata[lstseq]['midJD']  = midJD
        imgdata[lstseq]['nx'] = nx
        imgdata[lstseq]['ny'] = ny
    
    return imgdata


#-----------------------------------------------------------------------------------------------------------------------


def image_timerange(imgdata):
    
    dates = []
    for lst in list(imgdata):
        dates.append(imgdata[lst]['midJD'])
    
    oldest = min(dates)
    newest = max(dates)

    # Converting to utc for convenience
    ts = load.timescale()
    newest = ts.tt_jd(newest).utc_strftime()
    oldest = ts.tt_jd(oldest).utc_strftime()

    beg = pd.to_datetime(oldest)
    end = pd.to_datetime(newest)
    rng = pd.date_range(beg, end, freq='0.05H').to_pydatetime().tolist() #every 3 minutes
    timerange = ts.from_datetimes(rng)
    
    return timerange


#-----------------------------------------------------------------------------------------------------------------------


def check_illumination(imgdata, midJD, timerange, sat):

    # Check when satellite is illuminated 
    timerange = image_timerange(imgdata)
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
        sunlit_times.append([timerange.tt[idx[0]], timerange.tt[idx[1]]])

    # Check if midJD is within each period and flag it if so
    illuminated = False
    for sunlitrng in sunlit_times:
        if midJD >= (sunlitrng[0] - 1/86400) and midJD <= (sunlitrng[1] + 1/86400): #adding a bumper of second
            illuminated = True
            if illuminated:
                break

    return illuminated


#-----------------------------------------------------------------------------------------------------------------------


### THERE ARE A NUMBER OF THINGS WE DON'T WANT TO REPEAT IN THE MAIN LOOP

t0 = time.time()

# Opening image header
print('Loading image data')
LSC = image_data(target=f"{date}LSC")
LSN = image_data(target=f"{date}LSN")
LSS = image_data(target=f"{date}LSS")
LSE = image_data(target=f"{date}LSE")
LSW = image_data(target=f"{date}LSW")

# Get image timeranges
print('Determining timeranges')
trC = image_timerange(LSC)
trN = image_timerange(LSN)
trS = image_timerange(LSS)
trE = image_timerange(LSE)
trW = image_timerange(LSW)

# Opening the reduced fast curves
print('Loading fast curves')
fastdir = f"/net/beulakerwijde/data1/bring/testdata/lc/{date}"
FC = h5py.File(f"{fastdir}LSC/lightcurves/fast_{date}LSC.hdf5", "r")    
FN = h5py.File(f"{fastdir}LSN/lightcurves/fast_{date}LSN.hdf5", "r")    
FS = h5py.File(f"{fastdir}LSS/lightcurves/fast_{date}LSS.hdf5", "r")    
FE = h5py.File(f"{fastdir}LSE/lightcurves/fast_{date}LSE.hdf5", "r")    
FW = h5py.File(f"{fastdir}LSW/lightcurves/fast_{date}LSW.hdf5", "r")  

# Get passages
print('Loading passages')
psgsC = pd.read_pickle(f"{subtracted}{date}LSC/passages_{date}LSC.p")
psgsN = pd.read_pickle(f"{subtracted}{date}LSN/passages_{date}LSN.p")
psgsS = pd.read_pickle(f"{subtracted}{date}LSS/passages_{date}LSS.p")
psgsE = pd.read_pickle(f"{subtracted}{date}LSE/passages_{date}LSE.p")
psgsW = pd.read_pickle(f"{subtracted}{date}LSW/passages_{date}LSW.p")

# Get starlink TLEs
print('Loading TLEs')
tlesC = reduce_tles(target=f"{date}LSC")
tlesN = reduce_tles(target=f"{date}LSN")
tlesS = reduce_tles(target=f"{date}LSS")
tlesE = reduce_tles(target=f"{date}LSE")
tlesW = reduce_tles(target=f"{date}LSW")

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

print(f'Set-up time: {time.time() - t0}')
t0 = time.time()
print('Beginning main loop\n')

# Want to check the date for each camid before moving on to the next date
# The LST sequences will be the same for each camid

result = []
for seq in FN['station']['lstseq'][:]:  
    lstseq = str(seq)
    for data, timerange, camid, fast, starlinks, passages in zip(
        [LSC, LSN, LSS, LSE, LSW],
        [trC, trN, trS, trE, trW],
        ['LSC', 'LSN', 'LSS', 'LSE', 'LSW'],
        [FC, FN, FS, FE, FW],
        [tlesC, tlesN, tlesS, tlesE, tlesW],
        [psgsC, psgsN, psgsS, psgsE, psgsW]
        ):

        # Add this if not searching through ALL images
        if lstseq not in data.keys():
            continue
   
        # Set time of image
        midJD = data[lstseq]['midJD']
        ts = load.timescale()
        t  = ts.tt_jd(midJD)

        # Obtain passages
        idx_reduced = passage_check(midJD, starlinks, passages)
        
        if idx_reduced is None:
            print(f'{lstseq}: No starlinks passing MASCARA')
            continue
            
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
                topocentric = diff.at(t)
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

FC.close()
FN.close()
FS.close()
FE.close()
FW.close()

if len(result) > 0:
    np.savetxt('/net/beulakerwijde/data1/breslin/selection_pool.txt', result, fmt='%s')

print(f'Loop time: {time.time() - t0}')
#-----------------------------------------------------------------------------------------------------------------------
