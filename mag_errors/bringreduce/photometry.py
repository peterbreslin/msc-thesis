# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 12:02:13 2016

@author: talens
"""

import numpy as np

import bottleneck as bt
 
import logging

class Photometry(object):
    
    def __init__(self, aper, sky, phpadu=.62, badpixval=63000.):
        """ 
        Class for repeatedly performing aperture photometry.
            
        Parameters
        ----------
            aper : (naper,) array_like
                Radii of the photometric apertures.
            sky : (2,) array_like
                Inner and outer radius of the sky annulus.
            phpadu: float, optional
                Gain of the CCD detector.
            badpixval: float, optional
                Bad pixel value.
            
        """
        
        self.aper = aper
        self.sky = sky 
        self.phpadu = phpadu
        self.badpixval = badpixval
        
        nhalf = np.amax(sky)
        nhalf = np.ceil(nhalf)
        nbox = 2*nhalf + 1         
        
        self.naper = len(aper)
        self.nhalf = int(nhalf)
        
        tmp = np.arange(nbox) - nhalf
        self.xx, self.yy = np.meshgrid(tmp, tmp)                
        
        return
    
    def _aper_mask(self, x0, y0):
       
        rad = np.sqrt((self.xx - x0)**2 + (self.yy - y0)**2)

        return rad - .5   
    
    def get_phot(self, image, x, y):
        """
        Perform aperture photometry.

        Parameters
        ----------
            image : (ny, nx) array_like
                The image to perform the photometry on.
            x : (nstars,) array_like 
                The x-positions of the stars in the image.
            y : (nstars,) array_like
                The y-positions of the stars in the image.
                
        Returns
        -------
            flux : (nstars, naper) ndarray
                Array containing the measured fluxes.
            eflux : (nstars, naper) ndarray
                Array containing the measurement errors on the fluxes.
            sky : (nstars,) ndarray
                Array containing the sky values.
            esky : (nstars,) ndarray
                Array containing the std on the sky values.
            peak : (nstars,) ndarray
                The highest pixel value in the apertures.
            flag : (nstars,) ndarray
                Integer value indicating the quality of the photometry as
                decribed below.
        
        Notes
        -----
        The flags take the following values.
        0   : All went well.
        1   : The star was too close to the edge of the image no photometry was performed.
        2   : There was a pixel in the sky annulus with a value greater than badpixval.
        4   : The sky value was negative.
        8   : The peak value was greater than badpixval.
        16  : One of the flux measurements was negative.
        
        """
        
        log = logging.getLogger('bringreduce').getChild('photometry')
        
        ny, nx = image.shape
        nstars = len(x)        
        
        log.info('Performing aperture photometry on {} stars in {} apertures.'.format(nstars, self.naper))        
        
        # Seperate the coordinates in pixel center and deviation.
        xi, yi = np.around(x), np.around(y)
        dx, dy = x - xi, y - yi        
        
        # Initialize arrays for results
        flux = np.zeros((nstars, self.naper)) 
        eflux = np.zeros((nstars, self.naper))
        sky = np.zeros((nstars,))
        esky = np.zeros((nstars,))
        peak = np.zeros((nstars,))
        flag = np.zeros((nstars,), dtype='int')

        for i in range(nstars):
            
            # Extract sub-image.
            lx = int(xi[i] - self.nhalf)
            ux = int(xi[i] + self.nhalf + 1)
            ly = int(yi[i] - self.nhalf)
            uy = int(yi[i] + self.nhalf + 1)            
            
            # Check if the sub-image is inside the CCD.
            if (lx < 0) | (ux > nx) | (ly < 0) | (uy > ny):
                flag[i] += 1   
                continue
            
            subim = image[ly:uy,lx:ux]
            
            # Get the radii of the sub-image pixels.
            rad = self._aper_mask(dx[i], dy[i])
            
            # Get sky annulus.
            mask = ((rad + .5) >= self.sky[0]) & ((rad + .5) <= self.sky[1])
            skydonut = subim[mask]

            # Check for bad pixels in the annulus.
            if (bt.nanmax(skydonut) > self.badpixval):
                flag[i] += 2

            # Compute sky value.
            skymed = bt.median(skydonut)
            skymean = bt.nanmean(skydonut)
            skymod = 3.*skymed - 2.*skymean
            skystd = bt.nanstd(skydonut)
             
            # Check for negative sky values.
            if (skymod < 0):
                flag[i] += 4
             
            skyvar = skystd**2 # Sky variance  
            sigsq = skyvar/bt.nansum(mask) 
            
            sky[i] = skymod
            esky[i] = skystd
             
            # Compute the peak value.
            mask = (rad < np.amax(self.aper))
            peak[i] = bt.nanmax(subim[mask])             
             
            # Check for bad pixels in the apertures.
            if (peak[i] > self.badpixval):
                flag[i] += 8
             
            for j in range(self.naper):
                
                area = np.pi*(self.aper[j])**2                
                
                # Get aperture.
                mask = (rad < self.aper[j])
                
                aval = subim[mask]                
                arad = rad[mask]
                
                # Fraction of each pixel to count.
                fractn = (self.aper[j] - arad)
                
                idx1, = np.where(fractn >= 1.)
                idx2, = np.where(fractn < 1.) 
                
                fcounts = len(idx1)
                fractn[idx1] = 1.          
                
                factor = (area - fcounts)/bt.nansum(fractn[idx2])
                fractn[idx2] = fractn[idx2]*factor
            
                # Flux measurement.
                flux[i,j] = bt.nansum(aval*fractn) - skymod*area	
                
                # Error on the flux measurement.
                error1 = area*skystd**2
                error2 = flux[i,j]/self.phpadu
                error3 = sigsq*area**2
                
                eflux[i,j] = np.sqrt(error1 + error2 + error3)
            
            # Check for negative fluxes.
            if np.any(flux[i] < 0):
                flag[i] += 16                
                
        return flux, eflux, sky, esky, peak, flag
