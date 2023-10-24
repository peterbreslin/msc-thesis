# Reduces lines to one single track!
# Include error propagation

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

import bringreduce.configuration as cfg
cfg.initialize(rootdir,target,user)

import os
import glob
import time
import ephem
import numpy as np
import pandas as pd
import pickle as pickle
import astropy.io.fits as pf
from itertools import islice
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from STD_breslin_flux import LineSegments
import bringreduce.bringio as bringio
import bringreduce.mascara_astrometry as astrometry
starcat  = pf.getdata(cfg.starcat)
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)

if not os.path.exists(f"{rootdir}{user}/data/subtracted/{target}/vmag_images"):
    os.makedirs(f"{rootdir}{user}/data/subtracted/{target}/vmag_images")


# -------------------------------------------------------------------------------------------------------------------- #

t0 = time.time()
subtracted = f'{rootdir}{user}/data/subtracted/'
all_passages = pd.read_pickle(f"{subtracted}{target}/passages_20221023{camid}.p")
pool = pd.read_pickle(f'{subtracted}{target}/pool_{camid}.p')

passages = {}
for seq, data in all_passages.items():
    if seq in pool:
        passages[seq] = {}
        for satnumber in pool[seq]:
            if satnumber in data:
                passages[seq][satnumber] = data[satnumber]


vmag_dict = {}
df = pd.read_pickle(f"{rootdir}{user}/my_code/starlink_names.p")

# Keys of passages (i.e. LSTSEQs) will define the images we use (same as the keys of the pool)
lstseqs = list(pool)
for i, lstseq in enumerate(lstseqs, start=1):
    print(f'{lstseq}: image {i} of {len(lstseqs)}')

    try:
        data, header = pf.getdata(f'{subtracted}{target}/diffimages/diff_{lstseq}{camid}.fits.gz', header=True)
    except:
        print(f'No difference image for {lstseq}')
        continue

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
                readdata.connect_lines()
                print("Reduced to ", len(readdata.cleanedlines), " lines")

        else:
            print("Found ", readdata.alllines.ndim, " line with Hough transform")

            # Set the cleanedlines manually and add an extra dimension for the upcoming endpoint determination routines
            readdata.cleanedlines = np.array([readdata.alllines])

       
        if len(readdata.alllines) < 1:
            print('Hough Transform could not find any lines')
            continue

        # Before determine_endpoints, we check to see if the line is long enough
        # I.e. if the segments do not make up more than half the line, the determine_endpoints routine will fail
        # Maybe instead of re-jigging everything (since would have to define the ellipse now), just add condition
        # after the determine_endpoints routine

        Dx = []
        Dy = []
        # Loop over the remaining lines to determine the start- and end-points of the satellite tracks
        for i, line_id in enumerate(range(len(readdata.cleanedlines))):
            
            # We determine the 'orientation' of a satellite track with the RANSAC routine
            l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id])

            # If RANSAC routine is able to fit a line through the track, we determine its start- and end-points
            if l_x is not False:
                #print("Determining endpoints...")

                # The determine_endpoints routine determines the start- and endpoints on the original subtracted image 
                # It also returns the lst of the original image in which the satellite track was observed
                x_min, y_min, x_max, y_max, lst = readdata.determine_endpoints(lst0, lst1, 
                    readdata.cleanedlines[line_id], l_x, l_y)
                 
                Dx.append([x_min, x_max])
                Dy.append([y_min, y_max])
        
        vmag_dict[lstseq] = {'JD0':JD0, 'JD1':JD1}
        fig, ax = plt.subplots(figsize=[10,6])
        #ax.imshow(np.abs(readdata.maskedstarimage), interpolation='none', vmin=0, vmax=1000, cmap='jet')
        ax.imshow(data, vmin=-15, vmax=100, cmap='terrain')

        for i, sat in enumerate(list(passages[lstseq])):
            print('Looking for ' + sat)
            x1, y1 = passages[lstseq][sat]['start']['x'], passages[lstseq][sat]['start']['y']
            x2, y2 = passages[lstseq][sat]['end']['x'], passages[lstseq][sat]['end']['y']
            ax.scatter([x1,x2], [y1,y2], s=10, c='orange', zorder=1)

            # Center points
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            # Angle of the track
            delx = x2-x1
            dely = y2-y1
            theta_rad = np.arctan2(dely, delx)
            theta_deg = np.degrees(theta_rad)
            

            # Length of the track
            tle_length = int(round(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))) #this is based off TLE coords

            # In some cases, e.g. when the line is at the opposite angle, the major and minor axes end up being swapped.
            # This resuls in an incorrect ellipse orientation!
            # Based on the angle of the line, we can determine whether the major axis or the minor axis should be longer.
            
            if theta_deg % 180 == 0:
                major_axis = int(tle_length/4)
                minor_axis = tle_length + int(tle_length/3)
            else:
                major_axis = tle_length + int(tle_length/3)
                minor_axis = int(tle_length/4)


            e = Ellipse((cx,cy), major_axis, minor_axis, angle=theta_deg, ls='--', color='w', lw=0.5, fill=False) 
            ax.add_patch(e)

            # Lines from dtermine_endpoints
            x1_segment, x2_segment = [], []
            y1_segment, y2_segment = [], []
            
            cols = ['m', 'r', 'orange', 'y', 'w', 'g', 'b', 'k', 'pink', 'c', 'brown', 'lime', 'gray', 'blueviolet', 'gold']
            for x, y in zip(Dx, Dy):
                dx1, dx2 = x[0], x[1]
                dy1, dy2 = y[0], y[1]
                
                # Check to see if line bounded by ellipse (need to transform the coord system!)
                if e.contains_point(ax.transData.transform([dx1,dy1])) and e.contains_point(ax.transData.transform([dx2,dy2])):
                    
                    # All (x,y)-coords from Ransac within ellipse
                    x1_segment.append(dx1)
                    y1_segment.append(dy1)
                    x2_segment.append(dx2)
                    y2_segment.append(dy2)    
            
            if len(x1_segment) == 0:
                print('Routine failed; could not determine end-points')
                continue
 

            
            # Track orientation - this is already determined by the determine_endpoints routine but we do it again our way since 
            # we want to collapse the segments into one continuous line (min and max points will therefore be different)
            # This caused quite the headache to do, but the below routine works (although not very elegant)

            if x1_segment[0] < x2_segment[0]:
                # Satellite (maybe*) moving to the right 
                xmin, xmax = min(x1_segment), max(x2_segment)
            else:
                # Satellite (maybe*) moving to the left 
                xmin, xmax = max(x1_segment), min(x2_segment) 

            if y1_segment[0] < y2_segment[0]:
                # Satellite (maybe*) moving downwards
                ymin, ymax = min(y1_segment), max(y2_segment)
            else:
                # Satellite (maybe*) moving upwards
                ymin, ymax = max(y1_segment), min(y2_segment)


            # * Since we're unsure whether the min or max points correspond to the start of the negative segment or the end of 
            # the positive segment, these coords could be reversed. So, we determine this by checking which point is closest to 
            # the negative or positive segment by comparing the distance to the TLE coord:


            # Distance from (x_min, y_min) to start of TLE track (i.e. start of negative segment):
            d1 = int(round(np.sqrt((x1 - xmin)**2. + (y1 - ymin)**2.)))
            d2 = int(round(np.sqrt((x1 - xmax)**2. + (y1 - ymax)**2.)))
            if d1 < d2:
                # hence min corresponds to start of negative segment, max to end of positive segment
                x_min, x_max = xmin, xmax
                y_min, y_max = ymin, ymax
            else:
                x_min, x_max = xmax, xmin
                y_min, y_max = ymax, ymin

            # Just to clarify: 
            # min coords = entry points (LSTSEQ1, negative segment)
            # max coords = exit points (LSTSEQ2, positive segment)


            # Length of track as determined by the line segment
            fotos_length = int(round(np.sqrt((x_max - x_min)**2. + (y_max - y_min)**2.)))

            """
            
            Some instances where lines are still found for tracks that are ver much not complete i.e. lines being extracted
            that are very short relative to the true length of the track. We will add a condition such that, if the line
            segment (fotos_length) is less than 1/3 the size of the true segment (tle_length), then we'll skip it and not 
            do the brightness magnitude calculation for it!
            
            """

            if fotos_length < int(round(tle_length/3)):
                print('Determined line segment too short')
                continue

            # Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
            sat_vmag, sat_flux, sigma_B, popt = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)
            
            # Adding vmag to dictionary
            sat_name = df.loc[df['num'] == sat, 'name'].values[0]
            vmag_dict[lstseq][sat] = {'name':sat_name, 'vmag':sat_vmag, 'flux':sat_flux, 'sigma_B':sigma_B, 'popt':popt, 'FOTOS':{}, 'TLE':{}}

            # Recording the (x,y) positions and track length as determined by FOTOS
            vmag_dict[lstseq][sat]['FOTOS']['x'] = [round(x_min), round(x_max)]
            vmag_dict[lstseq][sat]['FOTOS']['y'] = [round(y_min), round(y_max)]
            vmag_dict[lstseq][sat]['FOTOS']['length'] = fotos_length

            # Recording the (x,y) positions and track length as determined by the TLE
            vmag_dict[lstseq][sat]['TLE']['x'] = [x1, x2]
            vmag_dict[lstseq][sat]['TLE']['y'] = [y1, y2]
            vmag_dict[lstseq][sat]['TLE']['length'] = tle_length

            print('Plotting')
            ax.plot([x_min, x_max], [y_min, y_max], zorder=2, c=cols[i], lw=2, ls=':', label=r"{}: m$_v$ = {}".format(
                sat_name, round(sat_vmag, 2)))
      
    ax.legend(framealpha=0.5)
    plt.savefig(f'{subtracted}{target}/vmag_images/vmags_{lstseq}{camid}.png', bbox_inches='tight', dpi=150, facecolor='w')
print('Saving dictionary')
pickle.dump(vmag_dict, open(f"{subtracted}{target}/vmags_{camid}.p", "wb"))
print(f'Runtime: {(time.time()-t0)/60} minutes')
