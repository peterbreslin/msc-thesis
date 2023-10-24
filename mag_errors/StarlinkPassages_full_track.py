# Generates the satellite passages over a given date and CAMID
# Saved dictionaries are not indexed by JD
# Compiled passages are for the full segment (i.e. start and end position of the track)
# I.e. beginning of negative segment (LSTSEQ1), end of positive segment (LSTSEQ2)
# NOTE: This reduces TLEs to Starlinks only, so only Starlink passages are generated!


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

if not os.path.exists(f"{rootdir}{user}/my_code/passages_full_track"):
    os.makedirs(f"{rootdir}{user}/my_code/passages_full_track")

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


# Reduce TLEs to Starlink only
starlink_tles = []
for tle in tles:
    if "STARLINK" in tle[0]:
        starlink_tles.append(tle)


def diffimages_fn(sss):
    done = False
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
    
    astro = astrometry.Astrometry(wcspars, polpars)
    
    t1 = time.time()
    print("Setting up:")
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
    if 48506747 in lstseq_array:
        done = True

    print('Generating passages')    
    for j, tle in enumerate(starlink_tles):
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


                    # Want to include both the positive and negative segment of the track positions
                    # Need to have positions from LSTSEQ[i] (negative) and LSTSEQ[i+1] (positive)

                    # So we will change: 
                    # old: obs.date = jd - 2415020+0.5*exp_array[i]/86400 
                    # new: obs.date = jd - 2415020+1.5*exp_array[i]/86400


                    obs.date = jd - 2415020+1.5*exp_array[i]/86400 #now looking at end-point of next segment 
                    sat.compute(obs)

                    ra1, dec1 = sat.a_ra * 180 / np.pi, sat.a_dec * 180 / np.pi
                    x1, y1, mask1 = astro.world2pix(midlst, ra1, dec1, jd=midJD)

                    if mask0 and mask1:                    

                        satnum = tle[1][2:8]
                        passages[seq][satnum] = {'start':{}, 'end':{}, 'JD': jd}
                        passages[seq][satnum]['start']['jd']  = jd-0.5*exp_array[i]/86400
                        passages[seq][satnum]['start']['lst'] = lst_array[i]-0.5*exp_array[i]/3600
                        passages[seq][satnum]['start']['ra']  = ra0
                        passages[seq][satnum]['start']['dec'] = dec0
                        passages[seq][satnum]['start']['x']  = x0[0]
                        passages[seq][satnum]['start']['y']  = y0[0]
                        passages[seq][satnum]['end']['jd']  = jd+1.5*exp_array[i]/86400
                        passages[seq][satnum]['end']['lst'] = lst_array[i]+1.5*exp_array[i]/3600
                        passages[seq][satnum]['end']['ra']  = ra1
                        passages[seq][satnum]['end']['dec'] = dec1
                        passages[seq][satnum]['end']['x']  = x1[0]
                        passages[seq][satnum]['end']['y']  = y1[0]
    

                        if satnum not in list(passed_satellites):
                            passed_satellites[satnum] = {}
                            passed_satellites[satnum]['line0'] = tle[0][2:]
                            passed_satellites[satnum]['line1'] = tle[1]
                            passed_satellites[satnum]['line2'] = tle[2]
                            passed_satellites[satnum]['passages'] = {1:{'jd':jd, 'lst':lst_array[i]}}
                        else:
                            n_overflight = max(passed_satellites[satnum]['passages'].keys())+1
                            passed_satellites[satnum]['passages'][n_overflight] = {'jd':jd, 'lst':lst_array[i]}

    print('Saving')
    pickle.dump(passages, open(f"{rootdir}{user}/my_code/passages_full_track/passages_{target}.p", "wb" ))
    pickle.dump(passed_satellites, open(f"{rootdir}{user}/my_code/passages_full_track/passed_satellites_{target}.p", "wb"))
    if done:
        print('Limit reached (467)')
        sys.exit()

for k, sss in enumerate(range(len(longsets))):
    print(f'Block {k+1}')
    diffimages_fn(sss)
    print('\n')
