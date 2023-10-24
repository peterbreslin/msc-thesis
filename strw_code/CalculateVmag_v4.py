# Version 4: computes the vmag for the full line with both positive and negative segments
# Uses the version of the passages file that was created with a start, middle and end position of the entire segment
# 	-> i.e. from GeneratePassages_v2.py
# Finds positions from Hough Transform / Ransac that are closest to TLE positions and uses those for vmag measurement

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


# -------------------------------------------------------------------------------------------------------------------- #


# all_passages = pickle.load(open(f'{rootdir}{user}/data/subtracted/{target}/passages_{target}.p',"rb" ))
# all_passages = pickle.load(open(f"{rootdir}{user}/my_code/vmag_tests/passages_vmag_tests/passages_full_segment.p","rb"))
all_passages = pickle.load(open(f"{rootdir}{user}/my_code/vmag_tests/passages_vmag_tests/passages_pos_neg.p","rb"))
pool = pickle.load(open(f'{rootdir}{user}/my_code/selection_pool/pool_{camid}.p',"rb"))

passages = {}
for lstseq, data in all_passages.items():
	if lstseq in pool:
		passages[lstseq] = {}
		for satnum in pool[lstseq]:
			if satnum in data:
				passages[lstseq][satnum] = data[satnum]

# Keys of passages (i.e. LSTSEQs) will define the images we use (same as the keys of the pool)
lstseqs = list(pool)


# Dictionary to store the visual magnitudes of the tracks
vmag_dict = {}
for i, curlstseq in enumerate(lstseqs):
	if i==1:
		break
	t0 = time.time()
	print(f'{curlstseq}: image {i} of {len(lstseqs)}')

	# Find the satellites for the current LSTSEQ (i.e. image)
	MatchSatellites = CheckKnownSatellites(passages[curlstseq], user, target, rootdir)   
	
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

	curlstseq = int(curlstseq)
	t1 = time.time()
	print("Setting up: "+str(t1-t0))

	# Initiate the Satellite Track Determination module    
	readdata = LineSegments(data, nx, ny, starcat, astrometry, user, target, rootdir, curlstseq, midlst, midJD, 
		plots=False, verbal=False)

	# Mask the stars and reduce the data, such that the contrast of the satellite tracks is enhanced
	readdata.ImageReduction(siteinfo, JD0)

	# Find all line segments in the reduced image with the Hough transform
	readdata.HoughTransform()


	if np.any(readdata.alllines != None):

		# The remove_badpixel_lines is particularly for MASCARA data because it has a bad area on the detector
		readdata.remove_badpixel_lines()

		# Sometimes Hough transform fits multiple lines to single track; this routine merges those lines to a single one
		if readdata.alllines.ndim>1:
			print("Found ", len(readdata.alllines), " lines with Hough transform")
			readdata.merge_lines(threshold=75)

			# The connect_lines routine is similar to the merge_lines, but here we look if line segments are in line 
			# with each other and can be combined into a single line
			if readdata.mergedlines.ndim>1:
				readdata.connect_lines(threshold=75)
				print("Reduced to ", len(readdata.cleanedlines), " lines")

		else:
			print("Found ", readdata.alllines.ndim, " line with Hough transform")

			# Set the cleanedlines manually and add an extra dimension for the upcoming endpoint determination routines
			readdata.cleanedlines = np.array([readdata.alllines])



		"""
		
		New Idea: reduce all the lines by finding the (x,y) positions closest to the start- and end-points determined
		by the TLEs


		1st idea: use x0, y0, x1, y1 = readdata.cleanedlines[line_id] and correlate to TLE positions
				  --> this didn't work too well

		2nd idea: use l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id]) and correlate to TLE
				  such that the determine_endpoints routine is not used
				  --> must do for positive and negative segment and define lst0 and lst1


		"""


		lx_list, ly_list = [], []

		# Loop over the remaining lines to determine the start- and end-points of the satellite tracks
		for line_id in range(len(readdata.cleanedlines)):

			# (x0, y0) and (x1, y1) are the start- and end-points of the line segments from the Hough transform, 
			# NOT necessarily the start- and end-point of the track
			x0, y0, x1, y1 = readdata.cleanedlines[line_id]

			# We determine the 'orientation' of a satellite track with the RANSAC routine
			l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id])
			lx_list.append(l_x)
			ly_list.append(l_y)


		# Remove booleans if any
		lx_list = [lx for lx in lx_list if not isinstance(lx, bool)]
		ly_list = [ly for ly in ly_list if not isinstance(ly, bool)]


		# Collasping all x and y values into single lists 
		x0_vals = [val[0] for val in lx_list]
		x1_vals = [val[1] for val in lx_list]
		y0_vals = [val[0] for val in ly_list]
		y1_vals = [val[1] for val in ly_list]

		tmp_lstseq = str(curlstseq)
		vmag_dict[tmp_lstseq] = {}


		# NOTE: removed readdata.determine_endpoints, MatchSatellites.match_start_endpoints


		for satnum in list(passages[tmp_lstseq]):
			print(satnum)
			vmag_dict[tmp_lstseq][satnum] = {}

			x_neg, y_neg = passages[tmp_lstseq][satnum]['start']['x'], passages[tmp_lstseq][satnum]['start']['y']
			x_mid, y_mid = passages[tmp_lstseq][satnum]['mid']['x'], passages[tmp_lstseq][satnum]['mid']['y']
			x_pos, y_pos = passages[tmp_lstseq][satnum]['end']['x'], passages[tmp_lstseq][satnum]['end']['y']

			# Closest pixel coords from TLE 
			neg = [x0_vals[np.abs(np.array(x0_vals) - x_neg).argmin()], 
				   y0_vals[np.abs(np.array(y0_vals) - y_neg).argmin()]]

			mid = [x0_vals[np.abs(np.array(x0_vals) - x_mid).argmin()],
				   y0_vals[np.abs(np.array(y0_vals) - y_mid).argmin()]]

			pos = [x0_vals[np.abs(np.array(x0_vals) - x_pos).argmin()],
				   y0_vals[np.abs(np.array(y0_vals) - y_pos).argmin()]]


			for coord, segment in zip([[neg,mid],[mid,pos]], ['negative', 'positive']):
				print(f'Measuring the {segment} segment')

				x_min, x_max = min(coord[0][0], coord[1][0]), max(coord[0][0], coord[1][0])
				y_min, y_max = min(coord[0][1], coord[1][1]), max(coord[0][1], coord[1][1])

				# Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
				sat_vmag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)

				vmag_dict[tmp_lstseq][satnum][segment] = {'start':{}, 'end':{}, 'vmag':sat_vmag}
				vmag_dict[tmp_lstseq][satnum][segment]['start']['x'] = coord[0][0]
				vmag_dict[tmp_lstseq][satnum][segment]['start']['y'] = coord[0][1]
				vmag_dict[tmp_lstseq][satnum][segment]['end']['x'] = coord[1][0]
				vmag_dict[tmp_lstseq][satnum][segment]['end']['y'] = coord[1][1]


pickle.dump(vmag_dict, open(f"{rootdir}{user}/my_code/vmags_v4.p", "wb" ))

