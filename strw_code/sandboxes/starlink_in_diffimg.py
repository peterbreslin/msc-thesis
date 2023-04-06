import argparse
import glob
import numpy as np
import pandas as pd
import astropy.io.fits as pf


# select all starlinks in passages file for each day (i.e. first + second day of diff of 2 images)


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camid', type=str, required=True, help='which camera (LSC/LSS/LSN/LSW/LSE)')
args = parser.parse_args()
camid = args.camid
rootdir = '/net/beulakerwijde/data1/breslin/data/subtracted/20221023'

# get TLEs
with open('/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/20221023/3leComplete.txt') as f:
    all_tles = f.readlines()
    f.close()
all_tles = [i.strip() for i in all_tles]

# split tles list into individual lists for each tle
tles = [all_tles[x:x+3] for x in range(0, len(all_tles), 3)]

# reduce tles to starlink only
starlink_tles = []
for tle in tles:
    if "STARLINK" in tle[0]:
        starlink_tles.append(tle)

# get list of satellite numbers
starlinks_no = []
for sat in starlink_tles:
    starlinks_no.append(sat[1][2:8])

# passages
#passed_sats = f'{rootdir}{camid}passed_satellites_20221023{camid}.p'
passages = pd.read_pickle(f'{rootdir}{camid}/passages_20221023{camid}.p')

print(f'Collecting images from 20221023{camid}')
images = glob.glob(f'{rootdir}{camid}/diff_*.fits.gz')
diffimg_with_starlink = []

print('Beginning search')
for img in images[10:15]:
    _, header = pf.getdata(img, header=True) 
    JD0 = header[7]
    #JD1 = header[8]
    
    for sat in starlinks_no:
        if sat in list(passages[JD0]): # | (sat in list(passages[JD1])):
           diffimg_with_starlink.append(img[-19:-11])
            
print(len(diffimg_with_starlink))            
print(diffimg_with_starlink)
