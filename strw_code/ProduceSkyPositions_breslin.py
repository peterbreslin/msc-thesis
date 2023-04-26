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


all_passages = pickle.load(open(f'{rootdir}{user}/data/subtracted/{target}/passages_{target}.p',"rb" ))
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

for i, curlstseq in enumerate(lstseqs):
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
            readdata.merge_lines()

            # The connect_lines routine is similar to the merge_lines, but here we look if line segments are in line 
            # with each other and can be combined into a single line
            if readdata.mergedlines.ndim>1:
                readdata.connect_lines()
                print("Reduced to ", len(readdata.cleanedlines), " lines")

        else:
            print("Found ", readdata.alllines.ndim, " line with Hough transform")

            # Set the cleanedlines manually and add an extra dimension for the upcoming endpoint determination routines
            readdata.cleanedlines = np.array([readdata.alllines])


        # Loop over the remaining lines to determine the start- and end-points of the satellite tracks
        for line_id in range(len(readdata.cleanedlines)):

            # (x0, y0) and (x1, y1) are the start- and end-points of the line segments from the Hough transform, 
            # NOT necessarily the start- and end-point of the track
            x0, y0, x1, y1 = readdata.cleanedlines[line_id]
            t_R_start = time.time()

            # We determine the 'orientation' of a satellite track with the RANSAC routine
            l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id])
            print("RANSAC took: ", time.time()-t_R_start)

            # If RANSAC routine is able to fit a line through the track, we determine its start- and end-points
            if l_x is not False:
                print("Determining endpoints...")

                # The determine_endpoints routine determines the start- and endpoints on the original subtracted image 
                # It also returns the lst of the original image in which the satellite track was observed
                x_min, y_min, x_max, y_max, lst = readdata.determine_endpoints(lst0, lst1, 
                    readdata.cleanedlines[line_id], l_x, l_y)

                if not np.isnan(x_min):

                    # Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
                    sat_vmag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)

                    if lst == lst0:
                       MatchSatellites.match_start_endpoints(JD0, x_min, y_min, x_max, y_max, sat_vmag, lst, exp0, 
                        model, readdata.astro, midlst, midJD, str(curlstseq), keyword='negative')

                    # Same routine as above, but for the lst and JD of the other image (from which the image with lst0 
                    # and JD0 is subtracted)
                    elif lst == lst1:
                        MatchSatellites.match_start_endpoints(JD1, x_min, y_min, x_max, y_max, sat_vmag, lst, exp1, 
                            model, readdata.astro, midlst, midJD, str(curlstseq), keyword='positive')

