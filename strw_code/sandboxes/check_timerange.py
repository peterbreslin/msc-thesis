import os
import glob
import time
import argparse
import numpy as np
import pandas as pd
import datetime as dt
import astropy.io.fits as pf
from skyfield.api import load


#-----------------------------------------------------------------------------------------------------------------------


parser = argparse.ArgumentParser()
parser.add_argument('-i', '--folder', type=str, required=True, help='folder: date + which camera (LSC/LSS/LSN/LSW/LSE)')
args = parser.parse_args()
folder = args.folder
camid  = folder[-1:-4]
data = '/net/beulakerwijde/data1/breslin/data/subtracted/'
directory = f'{data}{folder}' #specify the directory to search


#-----------------------------------------------------------------------------------------------------------------------


t0 = time.time()
ts = load.timescale()

print(f'Collecting images from {folder}')
images = glob.glob(f'{directory}/diff_*.fits.gz')
dates = []
names = []

# Loop over each image
print('Determining timerange')
for img in images[0:10]:
    _, header = pf.getdata(img, header=True)
    midJD = ts.tt_jd(header[12]).tt
    names.append(img)
    dates.append(midJD)

pool = zip(names, dates) #this will be a 2D array of [image, date] --> to be used later
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


