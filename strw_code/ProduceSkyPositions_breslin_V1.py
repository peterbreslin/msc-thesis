# Uses the version of the passages file that was created with a start, mid and end position of the entire segment
# 	-> i.e. from GeneratePassages_v2.py
# Masks out the image except for the region containing each starlink --> hopefully will help with merge_lines routine
# NOTE: changed the merge threshold i.e. merge_lines(threshold=75)

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

import matplotlib.pyplot as plt

starcat  = pf.getdata(cfg.starcat)
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)


""" 
I have re-written some parts of this code such that it cycles through a list of images (LSTSEQs) and
only finds a subset of satellites in that image. Hence, upon every iteration (of an image), the 
passages file is reduced to only the satellites given for the given image.

UPDATED - for the new way in how we create the passage files

Reduce passages file to selection pool:

 - all_passages = all passages for target
 - pool = selection pool (LSTSEQs with satnums)
 - passages = reduced passages to LSTSEQs in pool and such that each LSTSEQ has only found satnums

"""

# -------------------------------------------------------------------------------------------------------------------- #


# all_passages = pickle.load(open(f'{rootdir}{user}/data/subtracted/{target}/passages_{target}.p',"rb" ))

# NOTE: full segment file might cause ruckus in the match known positions code --> potentially could get rid of that tho
all_passages = pickle.load(open(f"{rootdir}{user}/my_code/vmag_tests/passages_vmag_tests/passages_full_segment.p","rb"))
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

for i, lstseq in enumerate(lstseqs):
	if i==1:
		sys.exit()

	t0 = time.time()
	print(f'{lstseq}: image {i} of {len(lstseqs)}')  
	
	data, header = pf.getdata(f'{rootdir}{user}/data/subtracted/{target}/diff_{lstseq}{camid}.fits.gz', header=True)
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


	curlstseq = int(lstseq)

	for satnum in list(passages[lstseq]):
		print(satnum)

		# Define the line segment
		x1, x2 = passages[lstseq][satnum]['start']['x'], passages[lstseq][satnum]['start']['y']
		y1, y2 = passages[lstseq][satnum]['end']['x'], passages[lstseq][satnum]['end']['y']

		# Create a mask for a rectangular portion
		mask = np.zeros_like(data)
		xmin, xmax = sorted([x1, x2])
		ymin, ymax = sorted([y1, y2])
		mask[int(ymin)-100:int(ymax)+100, int(xmin)-100:int(xmax)+100] = 1 #padded by 100 pixels

		# Apply the mask to the image
		masked_data = data * mask

		# Initiate the Satellite Track Determination module    
		readdata = LineSegments(masked_data, nx, ny, starcat, astrometry, user, target, rootdir, curlstseq, midlst, 
			midJD, plots=False, verbal=False)

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
					readdata.connect_lines()
					print("Reduced to ", len(readdata.cleanedlines), " lines")

			else:
				print("Found ", readdata.alllines.ndim, " line with Hough transform")

				# Set the cleanedlines manually and add an extra dimension for the upcoming endpoint determination routines
				readdata.cleanedlines = np.array([readdata.alllines])


			lines_x = []
			lines_y = []
			# Loop over the remaining lines to determine the start- and end-points of the satellite tracks
			for i, line_id in enumerate(range(len(readdata.cleanedlines))):

				# (x0, y0) and (x1, y1) are the start- and end-points of the line segments from the Hough transform, 
				# NOT necessarily the start- and end-point of the track
				x0, y0, x1, y1 = readdata.cleanedlines[line_id]

				# We determine the 'orientation' of a satellite track with the RANSAC routine
				l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id])

				# If RANSAC routine is able to fit a line through the track, we determine its start- and end-points
				if l_x is not False:
					print("Determining endpoints...")

					# The determine_endpoints routine determines the start- and endpoints on the original subtracted image 
					# It also returns the lst of the original image in which the satellite track was observed
					x_min, y_min, x_max, y_max, lst = readdata.determine_endpoints(lst0, lst1, 
						readdata.cleanedlines[line_id], l_x, l_y)

					lines_x.append([x_min,x_max])
					lines_y.append([y_min,y_max])


			plt.figure()
			plt.imshow(np.abs(readdata.maskedstarimage), interpolation='none', vmin=0, vmax=1000, cmap='jet')
			for i in range(len(lines_x)):
				x0 = lines_x[i][0]
				x1 = lines_x[i][1]
				y0 = lines_y[i][0]
				y1 = lines_y[i][1]
				plt.plot(np.array([x0, x1]),np.array([y0,y1]), ls='-', lw=2)
			plt.savefig(f'masked_data{satnum}.png', bbox_inches='tight', dpi=150, facecolor='w')


			"""

					if not np.isnan(x_min):

						# Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
						sat_vmag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)

						print(satnum, sat_vmag)

						# if lst == lst0:
						#    MatchSatellites.match_start_endpoints(JD0, x_min, y_min, x_max, y_max, sat_vmag, lst, exp0, 
						#     model, readdata.astro, midlst, midJD, str(curlstseq), keyword='negative')

						# # Same routine as above, but for the lst and JD of the other image (from which the image with lst0 
						# # and JD0 is subtracted)
						# elif lst == lst1:
						#     MatchSatellites.match_start_endpoints(JD1, x_min, y_min, x_max, y_max, sat_vmag, lst, exp1, 
						#         model, readdata.astro, midlst, midJD, str(curlstseq), keyword='positive')

			"""



