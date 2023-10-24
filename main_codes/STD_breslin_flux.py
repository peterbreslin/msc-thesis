"""
This is the Satellite Track Determination module. 
It reduces the images such that line segments can be find and determines the endpoints of those line segments.

"""

import sys
sys.path.append("/net/beulakerwijde/data1/breslin/code/fotos-python3/")
from mascommon import mmm

import os
import cv2
import h5py
import matplotlib
import numpy as np
import astropy.stats as aps
import scipy.ndimage as scn
from astropy.time import Time
from astropy import units as u
import matplotlib.pyplot as plt
from sklearn import linear_model
from scipy.optimize import curve_fit
from astropy.coordinates import EarthLocation, get_moon


class LineSegments(object):
    def __init__(self, deltaimage, nx, ny, starcat, astrometry, user, target, rootdir, curlstseq, midlst, midJD, 
        minlinelength=15, edge=30, masksizes=[2,4,8,16], plots=False, verbal=False):
       
        self.deltaimage = deltaimage.copy()
        self.mean, self.median, self.sigma = aps.sigma_clipped_stats(np.abs(self.deltaimage))
        self.edge = edge
        self.nx = nx
        self.ny = ny
        self.user = user
        self.target = target
        self.rootdir = rootdir
        self.plots = plots
        self.verbal = verbal
        self.camid = target[-3:]
        self.midlst = midlst
        self.midJD = midJD
        self.minLineLength = minlinelength # minimum length of line segments to be found
        self.fast = h5py.File(f'/data1/bring/testdata/lc/{target}/lightcurves/fast_{target}.hdf5','r')
        self.starcat = starcat
        curastro = np.where((self.fast['astrometry/lstseq'][()] // 50) == (curlstseq // 50))[0][0]
        order = self.fast['astrometry/x_wcs2pix'][curastro].shape[0]-1
        astrolst = self.fast['station/lst'][np.where(
            self.fast['station/lstseq'][()] == (self.fast['astrometry/lstseq'][curastro]))[0][0]]
        astrolstseq = self.fast['station/lstseq'][np.where(
            self.fast['station/lstseq'][()] == (self.fast['astrometry/lstseq'][curastro]))[0][0]]
        
        wcspars = {'crval' : self.fast['astrometry/crval'][curastro].copy(),
                    'crpix' : self.fast['astrometry/crpix'][curastro].copy(),
                    'cdelt' : [0.02148591731740587,0.02148591731740587],
                    'pc'    : self.fast['astrometry/pc'][curastro].copy(),
                    'lst'   : astrolst}

        polpars = {'x_wcs2pix' : self.fast['astrometry/x_wcs2pix'][curastro].copy(),
                    'y_wcs2pix' : self.fast['astrometry/y_wcs2pix'][curastro].copy(),
                    'x_pix2wcs' : self.fast['astrometry/x_pix2wcs'][curastro].copy(),
                    'y_pix2wcs' : self.fast['astrometry/y_pix2wcs'][curastro].copy(),
                    'nx'    : nx,
                    'ny'    : ny,
                    'order' : order}
        
        self.astro = astrometry.Astrometry(wcspars, polpars)

        # Define radii of masks, used in ImageReduction() to mask stars with the median pixel value
        self.masksizes = masksizes
                
        # Convert the sky positions of stars to pixelpositions (only for stars that are more than 30 pixels from the 
        # edge of the image and brighter than 9th magnitude)
        ra, dec = self.starcat['_raj2000'], self.starcat['_dej2000']
        self.stars_x0, self.stars_y0, self.stars_err0 = self.astro.world2pix(self.midlst,ra,dec,jd=self.midJD)
        select = np.where(
            (self.stars_x0 > edge) & (self.stars_y0 > self.edge) & (self.stars_x0 < (self.nx-self.edge)) & 
            (self.stars_y0 < (self.ny-self.edge)) & (self.starcat['vmag'][self.stars_err0] <= 9.0)
            )[0]
        self.xxx, self.yyy = np.round(self.stars_x0[select]).astype(int), np.round(self.stars_y0[select]).astype(int)
        self.vmag = self.starcat['vmag'][select]
                
        
        """ 
        - SAE: Remove all lines with x-value between 50-150 pxls and y-value between 600-950 pxls (bad area of detector)
        - AUE: Has 2 bad pixel areas on the detector. Remove all lines that have x-value between 0-300, and y-value 
            between 1750-2720 (a line of trees), and also with x-value between 3400-4072, and y-value between 0-350
        - LSE, LSS, LSN, LSC : no bad pixel areas found for 20200102
        - LSW: is looking at a tower with a weather station
        - x-value corresponds to horizontally plotted (axis length of 4080)
        - y-value corresponds to vertically plotted (axis length of 2720)

        """

        self.badpixelareas = {
            'SAE':[[50, 150, 600, 950]], 
            'AUE':[[0,300,1750, 2720],[3400, 4072, 0, 350]], 
            'LSE':[], 
            'LSS':[], 
            'LSW':[[0,350,2350,2720]], 
            'LSN':[], 
            'LSC':[]
            }

        #if not os.path.exists(f'{rootdir}{user}/data/subtracted/{target}/linesegments'):
        #    os.makedirs(f'{rootdir}{user}/data/subtracted/{target}/linesegments')

        self.image_id = str(curlstseq)
        self.MooninImage = False # Boolean that is False by default and set to True when the Moon is in the image
                

    def ImageReduction(self, siteinfo, JD, moonmargin=400):

        nostarimage = self.deltaimage.copy()

        # Masking stars as there is noise in pixels - with very bright stars noise is not subtracted, can be substantial
        mask = [self.createmask(ms) for ms in self.masksizes]

        # The size of the mask depends on the brightness of the star
        for ccc in range(len(self.yyy)):
            curmask = -1

            if ((self.vmag[ccc] < 6) & (self.vmag[ccc] >= 4.5)):
                curmask = 0
            if ((self.vmag[ccc] < 4.5) & (self.vmag[ccc] >= 2.5)):
                curmask = 2
            elif (self.vmag[ccc] < 2.5):
                curmask = 3

            if curmask >=0:
                nostarimage[
                    self.yyy[ccc]-self.masksizes[curmask]:self.yyy[ccc]+self.masksizes[curmask]+1,
                    self.xxx[ccc]-self.masksizes[curmask]:self.xxx[ccc]+self.masksizes[curmask]+1] = \
                nostarimage[
                    self.yyy[ccc]-self.masksizes[curmask]:self.yyy[ccc]+self.masksizes[curmask]+1,
                    self.xxx[ccc]-self.masksizes[curmask]:self.xxx[ccc]+self.masksizes[curmask]+1]*(
                        1-mask[curmask])+self.median*mask[curmask]


        # Check if the Moon is in FoV of camera
        observatory = EarthLocation(lat=siteinfo['lat']*u.deg, lon=siteinfo['lon']*u.deg, height=siteinfo['height']*u.m)        
        gcrs_coords = get_moon(Time(JD, format='jd'), location=observatory)
                
        moonx_astropy, moony_astropy, moon_mask = self.astro.world2pix(
            self.midlst, gcrs_coords.ra.value[0], gcrs_coords.dec.value[0], jd=self.midJD, margin=-moonmargin)
        
        # Mask moon if it's in the FoV
        if moon_mask: 
            
            self.MooninImage = True
            
            # Round the position to integer pixel values
            self.moonx, self.moony = np.round(moonx_astropy).astype(int)[0], np.round(moony_astropy).astype(int)[0]
            
            # Arbitrarily large initial value for the ring_median
            ring_median = 1.e6
            
            # Adaptively determine size of the moonmask by checking the median pixel value in the ring surrounding it.
            # Making sure the median pixel value in the surrounding ring < median pixel value + 1 sigma of whole image. 
            while(ring_median>(self.median+0.5*self.sigma)):
                
                ring = self.createring(radius = moonmargin)
                ringsize = int(1.1*moonmargin)

                indices = np.where(ring[
                    max(0, ringsize-self.moony):min(2*ringsize+1, ringsize+(self.ny-self.moony)), 
                    max(0, ringsize-self.moonx):min(2*ringsize+1, ringsize+(self.nx-self.moonx))
                    ] == 1)

                ring_mean, ring_median, ring_sigma = aps.sigma_clipped_stats(np.abs(nostarimage[
                    max(self.moony-ringsize,0):min(self.ny, self.moony+ringsize+1),
                    max(self.moonx-ringsize, 0):min(self.nx, self.moonx+ringsize+1)])[indices])
                
                moonmargin += 1
                
            print("Moon is masked with a radius of", moonmargin)
            self.moonmaskradius = moonmargin
            moonmask = self.createmask(moonmargin)
            
            nostarimage[max(self.moony-moonmargin,0):min(self.ny, self.moony+moonmargin+1),
                max(self.moonx-moonmargin, 0):min(self.nx, self.moonx+moonmargin+1)
                ] = nostarimage[
                    max(self.moony-moonmargin,0):min(self.ny, self.moony+moonmargin+1),
                    max(self.moonx-moonmargin, 0):min(self.nx, self.moonx+moonmargin+1)
                    ] * (1-moonmask[
                        max(0, moonmargin-self.moony):min(2*moonmargin+1, moonmargin+(self.ny-self.moony)),
                        max(0, moonmargin-self.moonx):min(2*moonmargin+1, moonmargin+(self.nx-self.moonx))
                        ]) + self.median*moonmask[
                            max(0, moonmargin-self.moony):min(2*moonmargin+1, moonmargin+(self.ny-self.moony)), 
                            max(0, moonmargin-self.moonx):min(2*moonmargin+1, moonmargin+(self.nx-self.moonx))
                            ]
        
            
            # Also mask the line that joins the center of the moon to the center of the image, because there are bright 
            # reflection artefacts on this line

            # Length of the line segment
            n_pixels = int(round(np.sqrt((self.moonx - int(self.nx/2))**2.+ (self.moony - int(self.ny/2))**2.)))
            
            # Assuming the line segment is linear
            x_values = np.rint(np.linspace(int(self.nx/2), self.moonx, n_pixels)).astype(int)
            y_values = np.rint(np.linspace(int(self.ny/2), self.moony, n_pixels)).astype(int)
                    
            linemask = np.zeros((self.deltaimage.shape[0], self.deltaimage.shape[1]))
            linemask[y_values, x_values] = 1
            
            # Broaden the line segment by 20 pixels in each direction
            linemask = scn.filters.convolve(linemask,np.ones((40,40)))
            linemask[np.where(linemask >= 1)] = 1
            nostarimage[np.where(linemask==1)] = self.median


        self.maskedstarimage = nostarimage.copy()
        
        nostarimage[-self.edge:,:] = 0
        nostarimage[:self.edge,:] = 0
        nostarimage[:,-self.edge:] = 0
        nostarimage[:,:self.edge] = 0
                
        # The sequence of operations below has shown to give good results during FOTOS. 
        # However they could be improved such that the contrast between satellite tracks and the background is enhanced.

        maskim = nostarimage.copy()*0
        maskim2 = nostarimage.copy()*0
        maskim2[np.abs(nostarimage) > self.mean+2.5*self.sigma] = 1
        out = cv2.dilate(np.uint8(maskim2),np.ones((3,3),np.uint8))
        maskim[(np.abs(nostarimage) > self.mean+1.0*self.sigma)] = 1
        maskim = maskim*out

        # The Houghlines routine only works with 8-bits images
        self.reducedimage = np.uint8(np.clip(scn.filters.convolve(maskim,np.ones((3,3))),6,9)-6)
        

    def createmask(self, masksize = 4):
        lingrid = np.linspace(0, masksize*2, masksize*2+1)-masksize
        xx, yy = np.meshgrid(lingrid,lingrid)
        rr = np.sqrt(xx**2+yy**2)
        mask = np.zeros((masksize*2+1,masksize*2+1))
        mask[np.where(rr <= (masksize+0.1))] = 1
        return mask        


    def createring(self, radius = 200, dr=.1):
        # if dr < 1, then the ring width is a fraction of the ring radius. Otherwise ringwidth is set by value of dr. 
        if dr < 1:
            masksize = int((1+dr)*radius)
        else:
            masksize = radius+dr
        lingrid = np.linspace(0, masksize*2, masksize*2+1)-masksize
        xx, yy = np.meshgrid(lingrid,lingrid)
        rr = np.sqrt(xx**2+yy**2)
        mask = np.zeros((masksize*2+1,masksize*2+1))
        mask[np.where((rr>=radius)&(rr <= (masksize+0.1)))] = 1
        return mask 

        
        
    def HoughTransform(self, dr=1, dtheta=np.pi/720, threshold=10, minLineLength=None, maxLineGap=3):
        
        if minLineLength is None:
            minLineLength = self.minLineLength
            
        # All line segments found with the Hough transform
        self.alllines = cv2.HoughLinesP(self.reducedimage, dr, dtheta, threshold=threshold, minLineLength=minLineLength,
            maxLineGap=maxLineGap)
        
        
        
    def remove_badpixel_lines(self):
    
        lines = self.alllines.copy()

        # Needs to be reshaped because the CV Hough transform routine adds another dimension to the array
        lines = lines.reshape(lines.shape[0],lines.shape[2])
        if self.verbal:
            print("Discarding lines in bad pixel areas for ", self.camid)
            print('lines in remove_badpixel_lines:', lines)
        
        if self.camid in self.badpixelareas.keys():
            for badarea in np.arange(len(self.badpixelareas[self.camid])):

                if self.verbal:
                    print("Discarding in bad pixel area ", self.badpixelareas[self.camid][badarea])

                # [:,0] is x0, [:,1] is y0, [:,2] is x1, [:,3] is y1
                indices = (np.logical_and(self.badpixelareas[self.camid][badarea][0]<=lines[:,0], 
                    lines[:,0]<=self.badpixelareas[self.camid][badarea][1])) * (
                        np.logical_and(self.badpixelareas[self.camid][badarea][0]<=lines[:,2], 
                        lines[:,2]<=self.badpixelareas[self.camid][badarea][1])
                        ) * (
                            np.logical_and(self.badpixelareas[self.camid][badarea][2]<=lines[:,1], 
                            lines[:,1]<=self.badpixelareas[self.camid][badarea][3])
                            ) * (
                                np.logical_and(self.badpixelareas[self.camid][badarea][2]<=lines[:,3], 
                                lines[:,3]<=self.badpixelareas[self.camid][badarea][3])
                                )

                if self.verbal:
                    print('indices', indices)

                lines = lines[np.logical_not(indices)]

                if self.verbal:
                    print('lines', lines)

        self.alllines = lines
    
    
    def merge_lines(self, threshold=25):
        # This routine narrows down the amount of lines fitted to a single track. If the start- and end-point of 
        # different lines are within the threshold number of pixels from eachother, they are combined into a single 
        # line with the outermost start- and end-points.
    
        lines_array = self.alllines.copy()
    
        # Need to create dictionaries for unique key value pairs. 
        # Matched/merged lines can be removed from the copy of the dictionary.
        lines = {}
        copied_lines = {}
        
        # Sort found lines by length, such that shortest lines are merged with other lines first. 
        # If the shortest lines are merged later, their start- and end-points may be further than the threshold value 
        # from the remaining lines.
        length_of_lines = []
        
        for i in np.arange(len(lines_array)):
            length_of_lines.append(
                np.sqrt((lines_array[i][0]-lines_array[i][2])**2. + (lines_array[i][1]-lines_array[i][3])**2.)
                )
        
        sorted_indices = np.argsort(np.array(length_of_lines))
        
        for i, v in enumerate(lines_array[sorted_indices]):
            lines[i] = v.copy()
            copied_lines[i] = v.copy()
        
        cleaned_lines = []
        
        for line_id in list(lines.keys()):
            if line_id in list(copied_lines.keys()):  
            
                xstart = lines[line_id][0]
                xend = lines[line_id][2]
                ystart =lines[line_id][1]
                yend = lines[line_id][3]

                to_be_removed = []
                for ind in list(copied_lines.keys()):
                
                    copiedline = copied_lines[ind]
                    difference = lines[line_id] - copiedline

                    # Check if start and endpoint are within threshold value         
                    if all(abs(x) < threshold for x in difference):
                        x0, y0, x1, y1 = copiedline

                        # Set outermost x-value, depends on direction of line
                        if x0 - x1 < 0:
                        
                            if x0 < xstart:
                                xstart = x0
                            
                            if x1 > xend:
                                xend = x1
                                                        
                        else:
                            
                            if x0 > xstart:
                                xstart = x0
                            if x1 < xend:
                                xend = x1
                                
                        # Set outermost y-value, depends on slope of line
                        if y0 - y1 < 0:
                        
                            if y0 < ystart:
                                ystart = y0
                            if y1 >yend:
                                yend = y1
                            
                        else:
                            if y0 > ystart:
                                ystart = y0
                            if y1 < yend:
                                yend = y1
                        
                        to_be_removed.append(ind)
                    
                if to_be_removed:
                    for k in to_be_removed:
                        if k in copied_lines:
                            del copied_lines[k]
                        
               
                cleaned_lines.append([xstart, ystart, xend, yend])         
                
        self.mergedlines = np.array(cleaned_lines)
    
    
    
    def connect_lines(self, threshold='adaptive'):
        # This routine connects lines when their start- and end-point are within the threshold of pixels from eachother
        # Similar to merge_lines
        
        lines_array = self.mergedlines.copy()
        
        lines = {}
        copied_lines = {}
        
        for i, v in enumerate(lines_array):
            lines[i] = v.copy()
            copied_lines[i] = v.copy()
        
        connected_lines = []
        
        for line_id in list(lines.keys()):
            if line_id in list(copied_lines.keys()):  
            
                xstart = lines[line_id][0]
                xend = lines[line_id][2]
                ystart =lines[line_id][1]
                yend = lines[line_id][3]
                
                to_be_removed = []
                if threshold == 'adaptive':
                    line_length = np.sqrt((xstart-xend)**2. + (ystart-yend)**2.)
                    thresh = 0.25*line_length
                else:
                    thresh = threshold
                
                for ind in list(copied_lines.keys()):
                
                    copiedline = copied_lines[ind]       
                    difference_start_end = lines[line_id] - np.roll(copiedline,2)
                    difference_end_end = lines[line_id] - copiedline

                    if (all(abs(x) < thresh for x in difference_start_end[0:2])) or (
                        all(abs(x)< thresh for x in difference_start_end[2:4])) or (
                        all(abs(x)< thresh for x in difference_end_end[0:2])) or (
                        all(abs(x)< thresh for x in difference_end_end[2:4])):

                        x0, y0, x1, y1 = copiedline
    
                        # Set outermost x-value, depends on direction of line
                        if x0 - x1 < 0:
                        
                            if x0 < xstart:
                                xstart = x0
                            
                            if x1 > xend:
                                xend = x1

                        else:
                            
                            if x0 > xstart:
                                xstart = x0
                            if x1 < xend:
                                xend = x1
                                
                        # Set outermost y-value, depends on slope of line
                        if y0 - y1 < 0:
                        
                            if y0 < ystart:
                                ystart = y0
                            if y1 >yend:
                                yend = y1
                            
                        else:
                            if y0 > ystart:
                                ystart = y0
                            if y1 < yend:
                                yend = y1
                        
                        to_be_removed.append(ind)
                    
                if to_be_removed:
                    for k in to_be_removed:
                        if k in copied_lines:
                            del copied_lines[k]
                        
                                  
                connected_lines.append([xstart, ystart, xend, yend])         
                
        self.cleanedlines = np.array(connected_lines)
    
    
    def RANSAC_linefit(self, Hough_start_endpoints, max_line_segment = 200):
        # Hough_start_endpoints is the (reduced) output of the Hough transform of the form [x0, y0, x1, y1]
        # Image is the whole image from which a subset will be selected based on the position of the Hough line estimate
        # max_line_segment is the maximum length in pixels a satellite in low-earth orbit can have
        
        Hough_line_length = np.sqrt((Hough_start_endpoints[0] - Hough_start_endpoints[2])**2.+ (
            Hough_start_endpoints[1] - Hough_start_endpoints[3])**2.) 
        
        # Extend the estimated line such that margins are included in the RANSAC method (in case found line segment 
        # is shorter than actual satellite track)
        line_extension = max(max_line_segment - Hough_line_length,0)

        # Extend line in the x- and y-direction according to delta x and delta y in line segment from Hough transform
        line_extension_y = line_extension / np.sqrt(
            1. + (Hough_start_endpoints[0] - Hough_start_endpoints[2])**2 /
                 (Hough_start_endpoints[1] - Hough_start_endpoints[3])**2
        )

        # Area for RANSAC needs a min width + height otherwise too small and almost vertical or horizontal line segment
        # May not be fitted properly by the RANSAC method
        line_extension_x = max(np.sqrt(line_extension**2.-line_extension_y**2.), 25)
        line_extension_y = max(line_extension_y, 25)
            
        # The corners of the selected subset of the image
        xmin = int(min(Hough_start_endpoints[0], Hough_start_endpoints[2]) - line_extension_x)
        xmax = int(round(max(Hough_start_endpoints[0], Hough_start_endpoints[2]) + line_extension_x))    
        ymin = int(min(Hough_start_endpoints[1], Hough_start_endpoints[3]) - line_extension_y)
        ymax = int(round(max(Hough_start_endpoints[1], Hough_start_endpoints[3]) + line_extension_y))
 
        if self.verbal:
            print(xmin, xmax, ymin, ymax)        
        
        # We still need to think of a way to handle satellite tracks at the edge of the image
        if (xmin <= self.edge) or (ymin <= self.edge) or (xmax >= (self.nx-self.edge)) or (ymax >= (self.ny-self.edge)):
            print("WARNING!!! LINE AT THE EDGE OF THE IMAGE!")
            return False, False, False
           
        elif self.MooninImage is True:
            if (
                np.sqrt((xmin - self.moonx)**2. + (ymin - self.moony)**2.) <= self.moonmaskradius
                or np.sqrt((xmax - self.moonx)**2. + (ymax - self.moony)**2.) <= self.moonmaskradius
            ):
                print("WARNING!!! LINE AT THE EDGE OF THE MOON!")
                return False, False, False

        # x- and y-coordinate are reversed in image
        sub_image = self.reducedimage[ymin:ymax, xmin:xmax].copy()
        
        indices = np.where(sub_image > 0)
        X = indices[1] + xmin
        y = indices[0] + ymin
        
        try:
            # Consistency of RANSAC regression over multiple runs can be increased by increasing the number of 
            # max_trials (100 by default) but of course, increasing max_trials slows down the regression. 
            model_ransac = linear_model.RANSACRegressor(residual_threshold=2, max_trials=1000,min_samples=5)
            model_ransac.fit(X[:,np.newaxis], y)
            
            # Predict data of estimated models
            line_X = np.arange(xmin, xmax)
            line_y_ransac = model_ransac.predict(line_X[:, np.newaxis])
            

            return line_X, line_y_ransac, model_ransac
        
        # In case the RANSAC routine fails to fit a line
        except ValueError:
        
            return False, False, False
        
    
    def determine_endpoints(self, lst_negative, lst_positive, connected_line, RANSAC_x, RANSAC_y):
        # delta image is the difference between two consecutive images, lst_positive is the local sidereal time 
        # corresponding to the second image and lst_negative is the local sidereal time corresponding to the first image 
        # (which is subtracted from the first and therefore corresponds to negative pixel values), connected_line is the 
        # begin- and endpoint of a line found by the Hough transform routine RANSAC_lx and RANSAC_ly are the x, y 
        # coordinates of the RANSAC fit to the line segment. 
        
        x0, y0, x1, y1 = connected_line
        
        # The total number of pixels is equal to the total length of the RANSAC line segment, otherwise the resolution 
        # in the x-direction may be too small for a line segment that is (close to) vertical 
        # (in which case RANSAC_x contains only a few pixels and RANSAC_y by definition has the same length as RANSAC_x)

        n_pixels = int(round(np.sqrt((RANSAC_x[-1] - RANSAC_x[0])**2.+ (RANSAC_y[0] - RANSAC_y[-1])**2.)))
        pixels = np.arange(n_pixels)
        
        if self.verbal:
            print(x0, y0, x1, y1)
        
        xmin = min(x0,x1)
        xmax = max(x0,x1)

        if self.verbal:
            print(("Ransac_x = ", RANSAC_x))
            print(("Ransac_y = ", RANSAC_y))

        x_values = np.linspace(RANSAC_x[0], RANSAC_x[-1], n_pixels)
        y_values = np.linspace(RANSAC_y[0], RANSAC_y[-1], n_pixels)
        
        if self.verbal:        
            print(("x_values = ", x_values))
            print(("y_values = ", y_values))
        
              
        # Maybe we should use the mean and standard deviation in the local subset of the image, which provides better 
        # results in case of e.g. twilight where the intensity in pixels can vary substantially over the whole image.     
        mean, median, sigma = aps.sigma_clipped_stats(np.fabs(self.deltaimage[
            int(round(min(RANSAC_y))):int(round(max(RANSAC_y))), int(round(min(RANSAC_x))):int(round(max(RANSAC_x)))]))
        if self.verbal:
            print(self.mean, mean)
            print(self.median, median)
            print(self.sigma, sigma)
        
        # x- and y-coordinate are reversed, so must be given as such to vstack. x- and y-coordinates for satellites
        # are reversed accordingly due to the astro.world2pix
        zi = scn.map_coordinates(self.deltaimage, np.vstack((y_values,x_values)), mode='nearest')#, order=5)
                
        # We distinguish between +ve and -ve pixel values such that we can determine the timestamp of a line-element
        zi_positive = zi.copy()
        zi_negative = zi.copy()
        zi_positive[zi_positive<0] = 0
        zi_negative[zi_negative>0] = 0
        zi_negative = np.fabs(zi_negative)
  
        # We start our routine at the pixel with the maximum value on the line found by the Hough transform as that 
        # pixel should certainly be part of the line segment
        startpixel_index = np.min(np.where(abs(x_values-xmin)<1))
        endpixel_index = np.max(np.where(abs(x_values-xmax)<1))
        if self.verbal:        
            print(('startpixel_index = ', startpixel_index))
            print(('endpixel_index = ', endpixel_index))

        if startpixel_index == endpixel_index:
            # No need to look further for vertical line. Routine will loop over zi for all indices so it doesn't matter 
            # if the line found by the Hough transfer is vertical (or horizontal)
            startpixel = startpixel_index.copy()
        else:
            # Start where the intensity peaks between startpixel_index and endpixel_index
            startpixel = np.argmax(np.fabs(zi)[startpixel_index:endpixel_index]) + startpixel_index
        
        # First we look for the line seqment in the positive pixel values
        if np.sum(zi_positive[startpixel_index:endpixel_index]) / (
            endpixel_index - startpixel_index) > (self.mean + self.sigma):

            """
            We define a merit function (=0 at the start- and endpoint) and moving out from our start- and endpoints we 
            add the difference between the pixel value and the mean pixel value plus 1 standard deviation to this merit 
            function. This way if there's a gap in a line segment, we can extend the line segment to pixels beyond the 
            gap, if there value contributes significantly. For example, a gap pixel that has the mean pixel value 
            contributes -sigma to the merit function. A following pixel has to be at least mu + 2*sigma, otherwise the 
            merit function remains negative. When we have iterated over all pixels, we take the pixel at each end where 
            the merit function is maximum.  

            """
              
            meritfunction_endpoint = []
            meritfunction_endpoint.append(0.)
            merit = 0.

            # Starting from endpixel_index + 1
            for pi in np.delete(np.arange(n_pixels-startpixel)+1,-1):
                ind = startpixel + pi
                
                merit += zi_positive[ind] - (mean+sigma)
                
                meritfunction_endpoint.append(merit)
                
            # argmax returns the index of the element with the highest value
            end = startpixel+np.argmax(meritfunction_endpoint)            
    
            meritfunction_startpoint = []  
            meritfunction_startpoint.append(0.) 
            merit = 0.

            #Starting from startpixel_index -1
            for pi in np.delete(np.arange(startpixel)+1, -1):
                ind = startpixel - pi
                
                merit += zi_positive[ind] - (mean+sigma)
                
                meritfunction_startpoint.append(merit)
            
            
            start = startpixel-np.argmax(meritfunction_startpoint)
            if self.plots is True:
                pass    
                            
            linesegmentlength = np.sqrt((x_values[start] - x_values[end])**2.+ (y_values[start] - y_values[end])**2.)     
                
            if self.verbal:            
                print("Positive line end coordinates: (",
                      x_values[start], ",", y_values[start], "), (",
                      x_values[end], ",", y_values[end], ")")
                print("Length =", linesegmentlength, "pixels")
            
            if linesegmentlength < self.minLineLength:
                if self.verbal:
                    print("Line segment is shorter than minimal lenght of", self.minLineLength, "pixels")
                return np.nan, np.nan, np.nan, np.nan, np.nan 
            
            else:            
            
                return int(round(x_values[start])), int(round(y_values[start])), int(round(x_values[end])), \
                    int(round(y_values[end])), lst_positive
    
    
    
        elif np.sum(zi_negative[startpixel_index:endpixel_index]) / (
            endpixel_index - startpixel_index) > (self.mean + self.sigma):            
           
            meritfunction_startpoint = []            
            meritfunction_endpoint = []
            meritfunction_endpoint.append(0.)
            meritfunction_startpoint.append(0.)
            merit = 0.

            for pi in np.delete(np.arange(n_pixels-startpixel)+1,-1):
                ind = startpixel + pi
                
                merit += zi_negative[ind] - (mean+sigma)
                
                meritfunction_endpoint.append(merit)
                
            end = startpixel+np.argmax(meritfunction_endpoint)            
            merit = 0.

            for pi in np.delete(np.arange(startpixel)+1, -1):
                ind = startpixel - pi
                
                merit += zi_negative[ind] - (mean+sigma)
                
                meritfunction_startpoint.append(merit)
                         
            start = startpixel-np.argmax(meritfunction_startpoint)
          
            if self.plots is True:
                pass   
                
            linesegmentlength = np.sqrt((x_values[start] - x_values[end])**2.+ (y_values[start] - y_values[end])**2.)     


            if self.verbal:            
                print("Negative line end coordinates: (", 
                        x_values[start], ",", 
                        y_values[start], "), (", 
                        x_values[end], ",", 
                        y_values[end], ")")
                print("Length =", linesegmentlength, "pixels")
                
            if linesegmentlength < self.minLineLength:
                if self.verbal:
                    print("Line segment is shorter than minimal lenght of", self.minLineLength, "pixels")
                return np.nan, np.nan, np.nan, np.nan, np.nan 
            
            else:                              
    
                return int(round(x_values[start])), int(round(y_values[start])), int(round(x_values[end])), \
                int(round(y_values[end])), lst_negative
    
        else:
            if self.verbal:
                print("Line segment not found in delta image")
            
            return np.nan, np.nan, np.nan, np.nan, np.nan        
        
        

    def determine_satvmag(self, x_min, y_min, x_max, y_max, curlstseq):
        # Determine the apparent magnitude of a satellite line segments from the measured flux and the relation between 
        # magnitude and flux of the surrounding stars
        
        # Length of the line segment
        n_pixels = int(round(np.sqrt((x_max - x_min)**2.+ (y_max - y_min)**2.)))
        
        # Assuming the line segment is linear
        x_values = np.rint(np.linspace(x_min, x_max, n_pixels)).astype(int)
        y_values = np.rint(np.linspace(y_min, y_max, n_pixels)).astype(int)
                
        emptymask = np.zeros((self.deltaimage.shape[0], self.deltaimage.shape[1]))

        linemask = emptymask.copy()
        linemask[y_values, x_values]=1

        # The inner_skymask is used to 'cut a hole' in the outer_skymask, such that we can select a donutshaped patch of 
        # the sky to determine the typical background pixel value
        
        # Broaden the line segment by 6.5 pixels in each direction
        inner_skymask = scn.filters.convolve(linemask,np.ones((11,11)))
        inner_skymask[np.where(inner_skymask >= 1)] = 1

        # Broaden the line segment by 21 pixels in each direction
        outer_skymask = scn.filters.convolve(linemask,np.ones((41,41)))
        outer_skymask[np.where(inner_skymask == 1)] = 0
        outer_skymask[np.where(outer_skymask >= 1)] = 1

        # To determine the satellite flux, we broaden the line segment by 4 pixels in each direction
        sat_mask = scn.filters.convolve(linemask,np.ones((8,8)))
        sat_mask[np.where(sat_mask >= 1)] = 1
        
        # We use the image in which the stars are masked to determine the typical sky background
        skybuf = self.maskedstarimage[np.where(outer_skymask==1)]
        skymod, skysig, skyskew = mmm(skybuf, minsky=0)
        
        # We also used the masked image for the flux of the satellite track. In case the satellite passes in front of a 
        # star we do not want to erroneously count the flux of the star towards the satellite. 
        sattrack = self.maskedstarimage[np.where(sat_mask==1)]
        
        # Total satellite flux, uncorrected for background
        uncorrected_sat_flux = np.sum(np.abs(sattrack)) / 2
        
        # In case this was the negative line segment in the difference image
        uncorrected_sat_flux *= np.sign(uncorrected_sat_flux)
        
        # Sum over all the pixel and subtract the typical sky background value from each pixelvalue
        satflux = np.log10(uncorrected_sat_flux - len(sattrack)*skymod)
        
        # We select surrounding stars that lie within circle with r=2*line length of center of the line segment (a,b)
        a,b = int(round((x_min+x_max)/2.)), int(round((y_min+y_max)/2.))
        yy, xx = np.meshgrid(np.linspace(
            0, emptymask.shape[0]-1, emptymask.shape[0]), np.linspace(0, emptymask.shape[1]-1, emptymask.shape[1]))

        starsmask = emptymask.copy()
        starsmask[np.where((xx-a)**2.+(yy-b)**2. <= (4*n_pixels*n_pixels))[1], 
        np.where((xx-a)**2.+(yy-b)**2. <= (4*n_pixels*n_pixels))[0]]=1

        select_stars = np.where(np.in1d(np.round(self.stars_x0).astype(int), 
            np.argwhere(starsmask==1)[:,1]) & np.in1d(np.round(self.stars_y0).astype(int), 
            np.argwhere(starsmask==1)[:,0]) & (self.starcat['vmag'][self.stars_err0] <= 9.0))[0]
        
        if self.plots is True:
            pass


        # The fluxes of the surrounding stars have been determined as part of standard MASCARA/BRING data reduction. 
        # We read in the files that contain these values and use ascc values to identify the surrounding stars.
        vmag_list  = self.starcat['vmag'][select_stars]
        ascc_list  = self.starcat['ascc'][select_stars]
        
        starsflux = []
        starsvmag = []       
        for vm, ascc in enumerate(ascc_list):
            
            try:            
                lc = self.fast['lightcurves/'+str(ascc)]
                # select the 50 data points that are in the same 5 minutes sequence to allow some averaging
                select = np.where((lc['lstseq'] % 13500)%270 == (curlstseq %13500) % 270) 
                if (4 <= vmag_list[vm] <= 6): 
                    if (not np.isnan(np.median(lc['flux1'][select]))and(not np.isinf(np.median(lc['flux1'][select])))):
                        # Apparently the determined flux can be Not A Number
                        starsflux.append(np.log10(np.median(lc['flux1'][select])))
                        starsvmag.append(vmag_list[vm])

            except KeyError:
                #Apparently the flux is not determined for all stars in the image
                continue
        
        
        starsflux = np.array(starsflux)
        starsvmag = np.array(starsvmag)

        # Use linear fitting to derive a relation from which the satellite magnitude can be determined
        # coeffs = np.polyfit(starsflux, starsvmag,1)

        try:
            popt, pcov = curve_fit(self.fitmagfunc, starsflux, starsvmag)            
            satvmag = self.fitmagfunc(satflux, *popt)#coeffs[0]*satflux + coeffs[1]
            if self.verbal:
                print("Satellite magnitude = ", satvmag)

            
        except (TypeError, ValueError):
            
            satvmag = np.nan
       

        # Uncertainty due to reference stars
        sigma_B = np.sqrt(pcov[0][0])
        opt_params = popt[0]

        return satvmag, 10**satflux, sigma_B, opt_params


    def fitmagfunc(self, x, b):
        # magnitude = -2.5 * np.log10(flux) + constant (from background noise)
        return -2.5 * x + b    


