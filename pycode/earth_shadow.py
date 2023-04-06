import os
import glob
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as pf
from scipy import stats as st
from skyfield.api import load, wgs84, EarthSatellite, N, S, E, W


#-----------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
# parser.add_argument('-n', '--number', type=str, required=True, help='how many images to search')
parser.add_argument('-c', '--camera', type=str, required=True, help='which camera (LSC/LSS/LSN/LSW/LSE)')
args = parser.parse_args()
camid = args.camera
data = '/net/beulakerwijde/data1/breslin/data/subtracted/'


#-----------------------------------------------------------------------------------------------------------------------


# Specify the directory to search
directory = f'{data}20221023{camid}'

# Search for pickle files
files = []
for filename in os.listdir(directory):
    if filename.endswith(".p"):
        files.append(os.path.join(directory, filename))

passages = pd.read_pickle(files[0])
passed_sats = pd.read_pickle(files[1])

# Load TLEs for all passages
with open("/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/20221023/3leComplete.txt") as f:
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

# Obtain satellite passes
keys = list(passed_sats)

# Find any Starlink TLEs in the passes
idx = []
starlinks = np.asarray(starlink_tles).flatten()
for key in keys:
    mascara_tle1 = passed_sats[key]['TLE line1'].strip()
    i = np.where(starlinks == mascara_tle1)[0] #this is not going to be fast for big lists...
    if i.size > 0:
        idx.append(i[0] - 1) #appending the name of the starlink sat
        
# Now have indices for the flattened Starlink TLE list --> divide by 3 to get indices for the original list
orig_idx = [int(x/3) for x in idx]
slk_mas_tles = res_list = [starlink_tles[i] for i in orig_idx]

# Remove 0 labeling of first line of TLE because that's the proper format
for tle in slk_mas_tles:
    tle[0] = tle[0][2:]

print(f'Number of satellites recorded for this day: {len(all_tles)}')
print(f'Number of them that were Starlinks: {len(starlink_tles)}')
print(f'Number of Starlinks that passed MASCARA: {len(slk_mas_tles)}')


#-----------------------------------------------------------------------------------------------------------------------


# Load the DE421 planetary ephemeris to get positions of Sun and Earth
eph = load('de421.bsp')
sun = eph['sun']
earth = eph['earth']

# Location of MASCARA
mascara = wgs84.latlon(latitude_degrees=29.26111*S, longitude_degrees=70.73139*W, elevation_m=2400)

# Set the observer to be MASCARA at ESO's La Silla Observatory
observer = earth + mascara

# Convert our TLEs to Skyfield EarthSatellites
print('Creating EarthSatellites')
sats = []
for tle in slk_mas_tles:
    sats.append(EarthSatellite(tle[1], tle[2], tle[0]))

# Specify our time range
n = 1
day = 24
ts = load.timescale()
timerange = ts.utc(2022, 10, day, 0, range(0, 24*60, n))

# We now begin to create a list of timeranges for which a satellite is sunlit
starlink_times = {}

print('Finding sunlit times')
for sat in sats:
    # Check when satellite is sunlit 
    sunlit = sat.at(timerange).is_sunlit(eph)

    # Obtain the indices of the first and last TRUE element for each sequence of TRUE elements 
    x = 0
    idx = []
    for i, n in enumerate(sunlit):
        if n and x==0:
            idx.append(i)
            x=1
        if not n and x==1:
            idx.append(i-1)
            x=0
        if n and i==len(sunlit)-1:
            idx.append(i)

    # Obtain times corresponding to the inidces found 
    sunlit_ = [timerange.tt[i] for i in idx]

    # Now split into separate list --> now have the start and end time for each sunlit period
    values = [sunlit_[i:i + 2] for i in range(0, len(sunlit_), 2)]
    starlink_times[sat.name] = values


#-----------------------------------------------------------------------------------------------------------------------


# Some satellites have 16 time elements, some have 15..
N = 16
vals = []
cur1 = []
cur2 = []
keys = list(starlink_times)

print('Avergaing sunlit times')   
for j in range(N):
    for sat in keys:
        if len(starlink_times[sat]) != N:
            continue
        else:
            cur1.append(starlink_times[sat][j][0])
            cur2.append(starlink_times[sat][j][1])
    vals.append([np.mean(cur1), np.mean(cur2)])
    #vals.append([st.mode(cur1, keepdims=False)[0], st.mode(cur2, keepdims=False)[0]])
    cur1 = []
    cur2 = []


#-----------------------------------------------------------------------------------------------------------------------


print(f'Collecting images from 20221023{camid}')
images = glob.glob(f'{directory}/diff_*.fits.gz')
diffimg_with_starlink = []

print('Beginning search')
for img in images[100:150]:
    print(f'Checking image {img[-19:-11]}')
    _, header = pf.getdata(img, header=True)
    midJD = ts.tt_jd(header[12]).tt

    # Check if midJD is within any of the ranges
    for timerange in vals:
        if ts.tt_jd(midJD).tt >= timerange[0] and ts.tt_jd(midJD).tt <= timerange[1]:
            diffimg_with_starlink.append(img[-19:-11])

print(f'Number of images expected to have Starlink trails visible: {len(diffimg_with_starlink)}')
print(f'Image numbers: {diffimg_with_starlink}')
np.savetxt('/net/beulakerwijde/data1/breslin/diffimg_with_starlink.txt', diffimg_with_starlink, fmt='%s')



# Work is progress below
#-----------------------------------------------------------------------------------------------------------------------

# Check when satellite is below horiozon:
sat = sats[100]
t, events = sat.find_events(mascara, t0, t1, altitude_degrees=30)
above = t[1::3]
below = t[2::3]
periods = list(zip(above, below))

# stopping as number of periods will be very different per satellite..


#-----------------------------------------------------------------------------------------------------------------------


# Twillight times
import datetime as dt
from pytz import timezone
from skyfield import almanac

zone = timezone('Chile/Continental')
f = almanac.dark_twilight_day(eph, mascara)
times, events = almanac.find_discrete(t0, t1, f)

previous_e = f(t0).item()
for t, e in zip(times, events):
    tstr = str(t.astimezone(zone))[:16]
    if previous_e < e:
        print(tstr, ' ', almanac.TWILIGHTS[e], 'starts')
    else:
        print(tstr, ' ', almanac.TWILIGHTS[previous_e], 'ends')
    previous_e = e


#-----------------------------------------------------------------------------------------------------------------------

