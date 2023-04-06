import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
import astropy.io.fits as pf
from skyfield.api import load, wgs84, EarthSatellite, S, W


#-----------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('-n', '--number', type=str, required=True, help='the number of images to search')
parser.add_argument('-f', '--folder', type=str, required=True, help='image folder: date + camera (LSC/LSS/LSN/LSW/LSE)')
args = parser.parse_args()
n = args.number
folder = args.folder
camid  = folder[-1:-4]
data = '/net/beulakerwijde/data1/breslin/data/subtracted/'
directory = f'{data}{folder}' #specify the directory to search


#-----------------------------------------------------------------------------------------------------------------------


### FIND STARLINK TLEs

file = f'{directory}/passed_satellites_{folder}.p'
passed_sats = pd.read_pickle(file)

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


### DETERMINE TIMERANGE OF IMAGES

t0 = time.time()
ts = load.timescale()
print(f'Collecting images from {folder}')
images = glob.glob(f'{directory}/diff_*.fits.gz')

dates = [] 
names = []

# Loop over each image
print('Determining timerange')
for img in images[0:n]:
    _, header = pf.getdata(img, header=True)
    midJD = ts.tt_jd(header[12]).tt
    names.append(img)
    dates.append(midJD)

# pool = list(zip(names, dates)) #this will be a 2D array of [image, date] --> to be used later (3)
oldest = min(dates)
newest = max(dates)

# Converting to utc for convenience
newest = ts.tt_jd(newest).utc_strftime()
oldest = ts.tt_jd(oldest).utc_strftime()
print('Oldest date:', oldest)
print('Newest date:', newest)

beg = pd.to_datetime(oldest)
end = pd.to_datetime(newest)
rng = pd.date_range(beg, end, freq='0.05H').to_pydatetime().tolist() #every 3 minutes=
timerange = ts.from_datetimes(rng)
print(f'Runtime: {str(time.time() - t0)}')


#-----------------------------------------------------------------------------------------------------------------------


### DETERMINE SUNLIT PERIODS

t0 = time.time()

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


### FINDING IF EACH SUNLIT STARLINK IS WITHIN THE IMAGE DATE

# We have a dictionary where each key is a Starlink ID. 
# In each key, we have a list of sunlit periods where each element is a list of the start and end time. 
# Must now loop through every Starlink ID and check if any of the sunlit periods are within the midJD of the image.


img_with_sunlit_starlink = []
print('Checking when each sunlit Starlink is within image')
t0 = time.time()

##from itertools import product
##for i, (name, sat) in enumerate(product(names, list(starlink_times)):

# Only want to know if satellite is sunlit at ANY time in this image
# Boolean statement ensures an image name isn't added multiple times

for i, name in enumerate(names):
    X = True

    for sat in list(starlink_times):
        for sunlitrng in starlink_times[sat]:
            if dates[i] >= sunlitrng[0] and dates[i] <= sunlitrng[1]:
                img_with_sunlit_starlink.append(name)
                X = False
                break

        if not X:
            break


    # YUCK, but really don't think there's a smarter way..

print(f'Triple nested For Loop for {N} images took {time.time() - t0}')
print(f'Number of images expected to have Starlink trails visible: {len(img_with_sunlit_starlink)}')


#-----------------------------------------------------------------------------------------------------------------------