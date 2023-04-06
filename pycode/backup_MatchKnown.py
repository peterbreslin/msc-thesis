import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-u", "--user", help="name of user", type=str, default="breslin")
parser.add_argument("-d", "--dir", help="target directory", type=str)
parser.add_argument("-r", "--rootdir", help="root directory", type=str, default="/net/beulakerwijde/data1/")
# parser.add_argument("-i", "--image", help="name of target image", type=int, default=None)
# parser.add_argument("-p", "--pool", help="selection pool", type=str, default=None)
# parser.add_argument("-l", "--lstseqs", help="list of lstseqs", type=str, default=None)

args = parser.parse_args()
if args.dir is None:
    sys.exit("Error: no target directory provided. Provide target directory with -d or --dir")
user = args.user
rootdir = args.rootdir
target = args.dir
camid = target[-3:]
date = target[:8]

import bringreduce.configuration as cfg
cfg.initialize(rootdir,target,user)
import glob
import astropy.io.fits as pf
import numpy as np
if camid[:2] == 'LS':
    import bringreduce.mascara_astrometry as astrometry
else:
    import bringreduce.bring_astrometry as astrometry
import time
import ephem
import bringreduce.bringio as bringio
from STD_breslin import LineSegments
from MatchKnownPositions_breslin import CheckKnownSatellites
from itertools import islice
import pickle as pickle
import os
import tracemalloc
tracemalloc.start()


# ------------------------------------------------------------------------------------------------ #


starcat = pf.getdata(cfg.starcat)
filelist = np.sort(glob.glob(rootdir+user+'/data/subtracted/'+target+'/*.fits.gz'))
files = [filename for filename in filelist if not any(s in filename for s in ['dark','bias','flat'])]
lstseq = np.array([np.int64(fff[-19:-11]) for fff in files])

edate = ephem.Date('2000/01/01 12:00:00.0')
siteinfo = bringio.read_siteinfo(cfg.siteinfo, camid)
obs = ephem.Observer()
obs.lat = siteinfo['lat']*np.pi/180
obs.long = siteinfo['lon']*np.pi/180
obs.elev = siteinfo['height']
obs.epoch = edate
sun = ephem.Sun(obs)


# ------------------------------------------------------------------------------------------------ #


# Reduce passages file to selection pool: (THIS IS CAMID DEPENDENT!)
# P = all passages for target
# X = selection pool (midJDs with satnums)
# Y = P reduced to midJDs found in X
# Z = Y reduced such that each midJD has only the satnums found in X

P = pickle.load(open(f'{rootdir}{user}/data/subtracted/{target}/passages_{target}.p',"rb" ))
X = pickle.load(open(f'{rootdir}{user}/my_code/selection_pool/pool_{camid}.p',"rb"))

# Firstly reduce midJDs ( SAME THING: Y = {JD: data for JD, data in P.items() if JD in list(X)} )
Y = {}
for JD, data in P.items():
    if JD in X:
        Y[JD] = data

# Now reduce satnums in each midJD
Z = {}
for JD in Y:
    Z[JD] = {}
    for satnum in X[JD]:
        if satnum in Y[JD]:
            Z[JD][satnum] = Y[JD][satnum]

# Now link to LSTSEQ
# Associated list of image LSTSEQs (this will define what images we use)
images = np.load(f'{rootdir}{user}/my_code/selection_pool/lstseqs_{camid}.npy', dtype=int) 

for img in images:

    header = pf.getheader(img)
    midJD = header[12]
    passages = {} #passages to use upon each iteration (i.e. for each image)

    if midJD in Z:
        passages[midJD] = Z[midJD]   
        MatchSatellites = CheckKnownSatellites(passages, user, target, rootdir)   

    else:
        print('midJD not in image')
        continue



# Now pass the reduced passages into MatchKownPositions code
# MatchSatellites = CheckKnownSatellites(Z, user, target, rootdir)  


# ------------------------------------------------------------------------------------------------ #


def findtracks(i):
    #The time.time() commands are just to time the duration of the Python operations.
    t0 = time.time()
    curlstseq = lstseq[i]
    print(i, ' of ', len(lstseq))
    print(curlstseq)

    data, header = pf.getdata(files[i], header=True)

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

    #PyEphem dates are encoded as the “Dublin Julian Day”      
    obs.date = JD0 - 2415020
    sun.compute(obs)

    #Sun must be low enough below the horizon (lower than 18 degrees), otherwise data is not good enough
    if np.degrees(sun.alt) > -18.:

        print("Sun is less than 18 degrees below the horizon")
        pass

     else:


        t1 = time.time()
        print("Setting up: "+str(t1-t0))


        #Initiate the Satellite Track Determination module    
        readdata = LineSegments(data, nx, ny, starcat, astrometry, user, target, rootdir, curlstseq, midlst, midJD, plots=False, verbal=False)
        #Mask the stars and reduce the data, such that the contrast of the satellite tracks is enhanced
        readdata.ImageReduction(siteinfo, JD0)

        #Find all line segments in the reduced image with the Hough transform
        readdata.HoughTransform()

        if np.any(readdata.alllines != None):

            #The remove_badpixel_lines is particularly for MASCARA data, because it has a bad area on the detector
            readdata.remove_badpixel_lines()

            #Sometimes the Hough transform fits multiple lines to a single satellite track. This routine merges those lines to a single one.
            if readdata.alllines.ndim>1:
                print("Found ", len(readdata.alllines), " lines with Hough transform")
                readdata.merge_lines()
                #The connect_lines routine is similar to the merge_lines, but here we look if line segments are in line with each other and can be combined into a single line
                if readdata.mergedlines.ndim>1:
                    readdata.connect_lines()
                    print("Reduced to ", len(readdata.cleanedlines), " lines")

            else:
                print("Found ", readdata.alllines.ndim, " line with Hough transform")
                #Set the cleanedlines manually and add an extra dimension for the upcoming endpoint determination routines
                readdata.cleanedlines = np.array([readdata.alllines])



            #loop over the remaining lines to determine the start- and endpoints of the satellite tracks
            for line_id in range(len(readdata.cleanedlines)):

                #x0,y0,x1,y1 are the start- and endpoints of the line segments resulting from the Hough transform, NOT necessarily the start- and endpoint of the satellite track
                x0,y0,x1,y1 = readdata.cleanedlines[line_id]


                t_R_start = time.time()
                #We determine the 'orientation' of a satellite track with the RANSAC routine
                l_x, l_y, model = readdata.RANSAC_linefit(readdata.cleanedlines[line_id])
                print("RANSAC took: ", time.time()-t_R_start)



                #If the RANSAC routine is able to fit a line through the satellite track, we determine its start- and endpoints
                if l_x is not False:
                    print("Determining endpoints...")
                    #The determine_endpoints routine determines the start- and endpoints on the original subtracted image.
                    #It also returns the lst of the original image in which the satellite track was observed
                    x_min, y_min, x_max, y_max, lst = readdata.determine_endpoints(lst0, lst1, readdata.cleanedlines[line_id], l_x, l_y)


                    if not np.isnan(x_min):

                        #Routine to determine the visual magnitude of the satellite track, based on the magnitude and pixel values of surrounding stars
                        sat_vmag = readdata.determine_satvmag(x_min, y_min, x_max, y_max, curlstseq)

                        if lst == lst0:

                            MatchSatellites.match_start_endpoints(JD0, x_min, y_min, x_max, y_max, sat_vmag, lst, exp0, model, readdata.astro, midlst, midJD, curlstseq, keyword='negative')

                        #Same routine as above, but for the lst and JD of the other image (from which the image with lst0 and JD0 is subtracted)
                        elif lst == lst1:

                            MatchSatellites.match_start_endpoints(JD1, x_min, y_min, x_max, y_max, sat_vmag, lst, exp1, model, readdata.astro, midlst, midJD, curlstseq, keyword='positive')





t_start = time.time()
for curlstseq in images:
    print(curlstseq)
    print(lstseq)
    print(np.where(lstseq==curlstseq))
    findtracks(np.where(lstseq==curlstseq)[0][0])

    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")

    print("Taking a total of ", (time.time()-t_start)/3600., " hours")

tracemalloc.stop()
