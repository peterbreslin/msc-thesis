# Version 2: computes the vmag for the negative and positive part of the segment, using a single LSTSEQ (from negative)
# Uses the version of the passages file that was created with start, middle, and end positions for each satellite
# 	-> i.e. from GeneratePassages_v2.py

import sys
sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("-u", "--user", help="name of user", type=str, default="breslin")
parser.add_argument("-d", "--dir", help="target directory", type=str)
parser.add_argument("-r", "--rootdir", help="root directory", type=str, default="/net/beulakerwijde/data1/")
args = parser.parse_args()

if args.dir is None:
	sys.exit("Error: no target directory provided. Provide target directory with -d or --dir")

rootdir = args.rootdir
target = args.dir
camid = target[-3:]
date = target[:8]
user = args.user

from STD_breslin import LineSegments
from MatchKnownPositions_breslin import CheckKnownSatellites

import bringreduce.configuration as cfg
cfg.initialize(rootdir,target,user)

import os
import glob
import time
import ephem
import numpy as np
import pickle as pickle
import astropy.io.fits as pf
from itertools import islice
import bringreduce.bringio as bringio
import bringreduce.mascara_astrometry as astrometry

starcat  = pf.getdata(cfg.starcat)
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)


if not os.path.exists(f"{rootdir}{user}/my_code/vmags"):
	os.makedirs(f"{rootdir}{user}/my_code/vmags")


# -------------------------------------------------------------------------------------------------------------------- #


# all_passages = pickle.load(open(f'{rootdir}{user}/data/subtracted/{target}/passages_{target}.p',"rb" ))
all_passages = pickle.load(open(f'{rootdir}{user}/my_code/new_passages/new_passages_{target}.p',"rb" )) #test
pool = pickle.load(open(f'{rootdir}{user}/my_code/selection_pool/pool_{camid}.p',"rb"))

passages = {}
for lstseq, data in all_passages.items():
	if lstseq in pool:
		passages[lstseq] = {}
		for satnum in pool[lstseq]:
			if satnum in data:
				passages[lstseq][satnum] = data[satnum]

# Keys of passages (i.e. LSTSEQs) will define the images we use
lstseqs = list(passages)

# Dictionary to store the visual magnitudes of the tracks
vmag_dict = {}

for i, curlstseq in enumerate(lstseqs):
	t0 = time.time()
	print(f'{curlstseq}: image {i} of {len(lstseqs)}')

	# Find the satellites for the current LSTSEQ (i.e. image)
	MatchSatellites = CheckKnownSatellites(passages, user, target, rootdir)   
	
	data, header = pf.getdata(f'{rootdir}{user}/data/subtracted/{target}/diff_{curlstseq}{camid}.fits.gz', header=True)
	JD0 = header['JD0']
	lst0 = header['LST0']
	JD1 = header['JD1']
	lst1 = header['LST1']
	midlst = header['MIDLST']
	midJD = header['MIDJD']
	nx = header['XSIZE']
	ny = header['YSIZE']
	exp0 = header['EXP0']
	exp1 = header['EXP1']

	lstseq = int(curlstseq) #just for LineSegements and determine_satvmag routines

	# Initiate the Satellite Track Determination module    
	readdata = LineSegments(data, nx, ny, starcat, astrometry, user, target, rootdir, lstseq, midlst, midJD, 
		plots=False, verbal=False)

	# Mask the stars and reduce the data, such that the contrast of the satellite tracks is enhanced
	readdata.ImageReduction(siteinfo, JD0)



	vmag_dict[curlstseq] = {}
	for satnum in list(passages[curlstseq]):

		vmag_dict[curlstseq][satnum] = {}

		# 'Start' of track
		x0 = passages[curlstseq][satnum]['start']['x']
		y0 = passages[curlstseq][satnum]['start']['y']

		# 'Middle' of track (end of first segment, start of second segment)
		x1 = passages[curlstseq][satnum]['mid']['x']
		y1 = passages[curlstseq][satnum]['mid']['y']

		# 'End' of track
		x2 = passages[curlstseq][satnum]['end']['x']
		y2 = passages[curlstseq][satnum]['end']['y']

		# The first image is substracted from the second
		negative = [[x0, x1], [y0, y1]]
		positive = [[x1, x2], [y1, y2]]

		for segment, positions in zip(['positive', 'negative'], [positive, negative]):

			x_min = min(positions[0])
			x_max = max(positions[0])
			y_min = min(positions[1])
			y_max = max(positions[1])

			# Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
			sat_vmag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, lstseq)

			vmag_dict[curlstseq][satnum][segment] = {'start':{}, 'end':{}, 'vmag':sat_vmag}
			vmag_dict[curlstseq][satnum][segment]['start']['x'] = positions[0][0]
			vmag_dict[curlstseq][satnum][segment]['start']['y'] = positions[1][0]
			vmag_dict[curlstseq][satnum][segment]['end']['x'] = positions[0][1]
			vmag_dict[curlstseq][satnum][segment]['end']['y'] = positions[1][1]


pickle.dump(vmag_dict, open(f"{rootdir}{user}/my_code/vmags/vmags_{target}.p", "wb" ))
