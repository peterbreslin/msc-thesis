rootdir = "/Users/peter/Projects/master-thesis/mag_errors/"
datadir = "/Users/peter/Projects/starlink_data/"
target = "20221023LSC"
camid = target[-3:]
date = target[:8]

import bringreduce.configuration as cfg
cfg.initialize(target)

import os
import sys
import glob
import time
import ephem
import numpy as np
import pickle as pickle
import astropy.io.fits as pf
from itertools import islice
from STD_breslin import LineSegments
import bringreduce.bringio as bringio
import bringreduce.mascara_astrometry as astrometry

starcat  = pf.getdata(cfg.starcat)
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse


if not os.path.exists(f"{rootdir}/error_tests/"):
    os.makedirs(f"{rootdir}/error_tests/")


# -------------------------------------------------------------------------------------------------------------------- #


all_passages = pickle.load(open(f"{datadir}vmags_subset/passages_subset/passages_20221023{camid}.p","rb"))
pool = pickle.load(open(f'{datadir}vmags_subset/selection_pool_subset/pool_{camid}.p',"rb"))

passages = {}
for lstseq, data in all_passages.items():
    if lstseq in pool:
        passages[lstseq] = {}
        for satnum in pool[lstseq]:
            if satnum in data:
                passages[lstseq][satnum] = data[satnum]


vmag_dict = {}
df = pd.read_pickle(f'{rootdir}starlink_names.p')
lstseqs = ['48506274', '48506275', '48506276'] 
for i, lstseq in enumerate(lstseqs):
    t0 = time.time()
    print(f'{lstseq}: image {i} of {len(lstseqs)}')

    try:
        data, header = pf.getdata(f'{datadir}test_data/diff_images/{camid}/diff_{lstseq}{camid}.fits.gz', header=True)
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
    t1 = time.time()

    # Initiate the Satellite Track Determination module    
    readdata = LineSegments(data, nx, ny, starcat, astrometry, target, curlstseq, midlst, midJD, 
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
                print("Determining endpoints...")

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
            
            cols = ['m', 'r', 'orange', 'y', 'w', 'g', 'b']
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
            
            # Routine to find the visual mag of the track based on the mag and pixel values of surrounding stars
            sat_vmag, delta_mag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)
            # print(sat, sat_vmag)
            
            # Adding vmag to dictionary
            satname = df.loc[df['num'] == sat, 'name'].values[0]
            vmag_dict[lstseq][sat] = {'name':satname, 'vmag':sat_vmag, 'delta':delta_mag, 'FOTOS':{}, 'TLE':{}}

            # Recording the (x,y) positions and track length as determined by FOTOS
            vmag_dict[lstseq][sat]['FOTOS']['x'] = [round(x_min), round(x_max)]
            vmag_dict[lstseq][sat]['FOTOS']['y'] = [round(y_min), round(y_max)]
            vmag_dict[lstseq][sat]['FOTOS']['length'] = fotos_length

            # Recording the (x,y) positions and track length as determined by the TLE
            vmag_dict[lstseq][sat]['TLE']['x'] = [x1, x2]
            vmag_dict[lstseq][sat]['TLE']['y'] = [y1, y2]
            vmag_dict[lstseq][sat]['TLE']['length'] = tle_length

            print('Plotting')
            ax.plot([x_min, x_max], [y_min, y_max], zorder=2, c=cols[i], lw=2, ls=':', label=r"{}: m = {} $\pm$ {}".format(
                satname, round(sat_vmag, 2), round(delta_mag, 2)))
      
    ax.legend(framealpha=0.5)
    plt.savefig(f'{rootdir}error_tests/vmags_{lstseq}.png', bbox_inches='tight', dpi=150, facecolor='w')
print('Saving dictionary')
pickle.dump(vmag_dict, open(f"{rootdir}error_tests/vmags_{camid}.p", "wb" ))

