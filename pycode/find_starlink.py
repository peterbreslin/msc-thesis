import os
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', type=str, required=True, help='which camera (LSC/LSS/LSN/LSW/LSE)')
args = parser.parse_args()
camera = args.camera

data = "/net/beulakerwijde/data1/breslin/data/subtracted/"

# specify the directory to search
directory = "{}20221023{}".format{data, camera}

# search for pickle files
files = []
for filename in os.listdir(directory):
    if filename.endswith(".p"):
        files.append(os.path.join(directory, filename))
        
passages = pd.read_pickle(files[0])
elements = pd.read_pickle(files[1])

# open text file of TLEs
with open("/net/mascara0/data3/stuik/LaPalma/inputdata/satellites/20221023/") as f:
    all_tles = f.readlines()
    f.close()

# the keys are the satellite classification    
keys = list(passages.keys())
all_tles = [i.strip() for i in all_tles]

# split tles list into individual lists for each tle
tles = [all_tles[x:x+3] for x in range(0, len(all_tles), 3)]

# let's reduce the tles to starlink only for now
starlink_tles = []
for tle in tles:
    if "STARLINK" in tle[0]:
        starlink_tles.append(tle)

# now we want to find any starlink tles in the mascara passages
idx = []
starlinks = np.asarray(starlink_tles).flatten()
for key in keys:
    mascara_tle1 = passages[key]['TLE line1'].strip()
    i = np.where(starlinks == mascara_tle1)[0] #this is not going to be fast for big lists...
    if i.size > 0:
        idx.append(i[0] - 1) #appending the name of the starlink sat
