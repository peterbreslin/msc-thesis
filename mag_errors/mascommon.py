import astropy.convolution as aco
import numpy as np
import pandas as pd
import astropy.io.fits as pf


def mmmm(sky, minsky=20, maxiter=50, highbad=None):
    # GJ version of mmm
    # Remove bad values and sort.
    sky = sky[np.isfinite(sky)]
    sky = np.sort(sky)

    if len(sky) < minsky:
        return np.nan, -1., 0.

    # Determine window for robust computations.
    skymid = np.median(sky)

    cut = min([skymid - sky[0], sky[-1] - skymid])
    if highbad is not None:
        cut = min([cut, highbad - skymid])

    cut1 = skymid - cut
    cut2 = skymid + cut

    idx1 = np.searchsorted(sky, cut1)
    idx2 = np.searchsorted(sky, cut2)

    if (idx2 - idx1) < minsky:
        return np.nan, -1., 0.

    # Get statistics.
    skymed = np.median(sky[idx1:idx2])
    skymn = np.mean(sky[idx1:idx2])
    sigma = np.std(sky[idx1:idx2])

    if (skymed < skymn):
        skymod = 3.*skymed - 2.*skymn
    else:
        skymod = skymn

        # Iteratively refine.
    old = 0
    clamp = 1
    idx1_old = idx1
    idx2_old = idx2
    for niter in range(maxiter):

        # Determine window for robust computations.
        r = np.log10(idx2 - idx1)
        r = max([ 2., (-0.1042*r + 1.1695)*r + 0.8895])

        cut = r*sigma + 0.5*np.abs(skymn - skymod)
        cut1 = skymod - cut
        cut2 = skymod + cut

        idx1 = np.searchsorted(sky, cut1)
        idx2 = np.searchsorted(sky, cut2)

        if (idx2 - idx1) < minsky:
            return np.nan, -1., 0.

        skymn = np.mean(sky[idx1:idx2])
        sigma = np.std(sky[idx1:idx2])

        # Use the mean of the central 20% as the median.
        center = (idx1 + idx2 - 1)/2.
        side = round(0.2*(idx2 - idx1))/2.

        j = np.ceil(center - side).astype('int')
        k = np.floor(center + side + 1).astype('int')

        skymed = np.mean(sky[j:k])

        # Update the mode.
        if (skymed < skymn):
            dmod = 3.*skymed - 2.*skymn - skymod
        else:
            dmod = skymn - skymod

        if (dmod*old < 0):
            clamp = 0.5*clamp

        skymod = skymod + clamp*dmod
        old = dmod

        if (idx1 == idx1_old) & (idx2 == idx2_old):
            break

        idx1_old = idx1
        idx2_old = idx2

    return skymod, sigma, idx2 - idx1



def mmm( sky_vector, 
         highbad = False,
         debug = False,
         readnoise = False,
         nsky = False,
         printerror = True,
         integer = "discrete",
         mxiter = 50,
         minsky = 20,
         nan=True):
    """Estimate the sky background in a stellar contaminated field.

    MMM assumes that contaminated sky pixel values overwhelmingly display 
    POSITIVE departures from the true value.  Adapted from DAOPHOT 
    routine of the same name.
    
    CALLING SEQUENCE:
         skymod,sigma,skew = mmm.mmm( sky, highbad= , readnoise=, debug=, 
                                      minsky=, nsky=, integer=)
    
    INPUTS:
         sky - Array or Vector containing sky values.  This version of
                MMM does not require SKY to be sorted beforehand.
    
    RETURNS:
         skymod - Scalar giving estimated mode of the sky values
         sigma -  Scalar giving standard deviation of the peak in the sky
                   histogram.  If for some reason it is impossible to derive
                   skymod, then SIGMA = -1.0
         skew -   Scalar giving skewness of the peak in the sky histogram
    
         If no output variables are supplied or if "debug" is set
         then the values of skymod, sigma and skew will be printed.
    
    OPTIONAL KEYWORD INPUTS:
         highbad - scalar value of the (lowest) "bad" pixel level (e.g. cosmic 
                    rays or saturated pixels) If not supplied, then there is 
                    assumed to be no high bad pixels.
         minsky - Integer giving mininum number of sky values to be used.   MMM
                    will return an error if fewer sky elements are supplied.
                    Default = 20.
         maxiter - integer giving maximum number of iterations allowed,default=50
         readnoise - Scalar giving the read noise (or minimum noise for any 
                     pixel).  Normally, MMM determines the (robust) median by 
                    averaging the central 20% of the sky values.  In some cases
                    where the noise is low, and pixel values are quantized a
                    larger fraction may be needed.  By supplying the optional
                    read noise parameter, MMM is better able to adjust the
                    fraction of pixels used to determine the median.                
         integer - Set this keyword if the  input SKY vector only contains
                    discrete integer values.  This keyword is only needed if the
                    SKY vector is of type float or double precision, but contains 
                    only discrete integer values.  (Prior to July 2004, the
                    equivalent of /INTEGER was set for all data types)
         debug -   If this keyword is set and non-zero, then additional 
                    information is displayed at the terminal.

    OPTIONAL OUTPUT KEYWORD:
         nsky - Integer scalar giving the number of pixels actually used for the
                 sky computation (after outliers have been removed).

    NOTES:
         (1) Program assumes that low "bad" pixels (e.g. bad CCD columns) have
              already been deleted from the SKY vector.
         (2) MMM was updated in June 2004 to better match more recent versions
              of DAOPHOT.
         (3) Does not work well in the limit of low Poisson integer counts
         (4) MMM may fail for strongly skewed distributions.

    METHOD:
         The algorithm used by MMM consists of roughly two parts:
           (1) The average and sigma of the sky pixels is computed.  These values
                are used to eliminate outliers, i.e. values with a low probability
                given a Gaussian with specified average and sigma.  The average
                and sigma are then recomputed and the process repeated up to 20
                iterations.
           (2) The amount of contamination by stars is estimated by comparing the 
                mean and median of the remaining sky pixels.  If the mean is larger
                than the median then the true sky value is estimated by
                3*median - 2*mean
             
     REVISION HISTORY:
           Adapted to IDL from 1986 version of DAOPHOT in STSDAS        W. Landsman, STX           Feb,      1987
           Added HIGHBAD keyword                                        W. Landsman                January,  1991
           Fixed occasional problem with integer inputs                 W. Landsman                Feb,      1994
           Avoid possible 16 bit integer overflow                       W. Landsman                November, 2001
           Added READNOISE, NSKY keywords,  new median computation      W. Landsman                June,     2004
           Added INTEGER keyword                                        W. Landsman                July,     2004
           Improve numerical precision                                  W. Landsman                October,  2004
           Fewer aborts on strange input sky histograms                 W. Landsman                October,  2005
           Added /SILENT keyword                                                                   November, 2005
           Fix too many /CON keywords to MESSAGE                        W.L.                       December, 2005
           Fix bug introduced June 2004 removing outliers               N. Cunningham/W. Landsman  January,  2006
            when READNOISE not set
           Make sure that MESSAGE never aborts                          W. Landsman                January,  2008
           Add mxiter keyword and change default to 50                  W. Landsman                August,   2011
           Added MINSKY keyword                                         W.L.                       December, 2011
           Converted to Python                                          D. Jones                   January,  2014
    """
    try:
        if nan: sky_vector = sky_vector[np.where(sky_vector == sky_vector)]
        nsky = len( sky_vector )            #Get number of sky elements 
    
        if nsky < minsky:
            sigma=-1.0 ;  skew = 0.0; skymod = np.nan
            if printerror: print(('ERROR -Input vector must contain at least '+str(minsky)+' elements'))
            return(skymod,sigma,skew)
    
        nlast = nsky-1                        #Subscript of last pixel in SKY array
        if debug:
            print(('Processing '+str(nsky) + ' element array'))
#        sz_sky = np.shape(sky_vector)
    
        sky = np.sort(sky_vector)    #Sort SKY in ascending values
    
        skymid = 0.5*sky[(nsky-1)/2] + 0.5*sky[nsky/2]  #Median value of all sky values
        
        cut1 = np.min( [skymid-sky[0],sky[nsky-1] - skymid] ) 
        if highbad: 
            cut1[np.where(cut1 > highbad - skymid)[0]] = highbad - skymid
        cut2 = skymid + cut1
        cut1 = skymid - cut1
    
        # Select the pixels between Cut1 and Cut2
    
        good = np.where( (sky <= cut2) & (sky >= cut1))[0]
        Ngood = len(good)
    
        if ( Ngood == 0 ):
            sigma=-1.0 ;  skew = 0.0; skymod = 0.0   
            if printerror: print(('ERROR - No sky values fall within ' + str(cut1) + \
    	   ' and ' + str(cut2)))
            return(skymod,sigma,skew)
    
        delta = sky[good] - skymid  #Subtract median to improve arithmetic accuracy
        sum = np.sum(delta.astype('float64'))
        sumsq = np.sum(delta.astype('float64')**2)
    
        maximm = np.max( good) ; minimm = np.min(good)  # Highest value accepted at upper end of vector
        minimm = minimm -1               #Highest value reject at lower end of vector
    
        # Compute mean and sigma (from the first pass).
    
        skymed = 0.5*sky[(minimm+maximm+1)/2] + 0.5*sky[(minimm+maximm)/2 + 1] #median 
        skymn = sum/(maximm-minimm)                            #mean       
        sigma = np.sqrt(sumsq/(maximm-minimm)-skymn**2)             #sigma          
        skymn = skymn + skymid         #Add median which was subtracted off earlier 
    
        #    If mean is less than the mode, then the contamination is slight, and the
        #    mean value is what we really want.
    #    skymod =  (skymed < skymn) ? 3.*skymed - 2.*skymn : skymn
        if skymed < skymn:
            skymod = 3.*skymed - 2.*skymn
        else: skymod = skymn
    
        # Rejection and recomputation loop:
    
        niter = 0
        clamp = 1
        old = 0
    # START_LOOP:
        redo = True
        while redo:
            niter = niter + 1                     
            if ( niter > mxiter ):
                sigma=-1.0 ;  skew = 0.0   
                if printerror: print(('ERROR - Too many ('+str(mxiter) + ') iterations,' + \
                        ' unable to compute sky'))
    #            import pdb; pdb.set_trace()
                return(skymod,sigma,skew)
    
            if ( maximm-minimm < minsky ):    #Error? 
    
                sigma = -1.0 ;  skew = 0.0   
                if printerror: print(('ERROR - Too few ('+str(maximm-minimm) +  \
                        ') valid sky elements, unable to compute sky'))
                return(skymod,sigma,skew)
    
            # Compute Chauvenet rejection criterion.
    
            r = np.log10( float( maximm-minimm ) )      
            r = np.max( [ 2., ( -0.1042*r + 1.1695)*r + 0.8895 ] )
    
            # Compute rejection limits (symmetric about the current mode).
    
            cut = r*sigma + 0.5*np.abs(skymn-skymod)   
        #    if integer: cut = cut > 1.5 
            cut1 = skymod - cut   ;    cut2 = skymod + cut
    
            # 
            # Recompute mean and sigma by adding and/or subtracting sky values
            # at both ends of the interval of acceptable values.
        
            redo = False
            newmin = minimm             
            if sky[newmin+1] >= cut1: tst_min = 1      #Is minimm+1 above current CUT?
            else: tst_min = 0
            if (newmin == -1) and tst_min: done = 1    #Are we at first pixel of SKY?
            else: done = 0
            if not done:
                if newmin > 0: skyind = newmin
                else: skyind = 0
                if (sky[skyind] < cut1) and tst_min: done = 1
            if not done:
                istep = 1 - 2*int(tst_min)
                while not done:
                    newmin = newmin + istep
                    if (newmin == -1) | (newmin == nlast): done = 1
                    if not done:
                        if (sky[newmin] <= cut1) and (sky[newmin+1] >= cut1): done = 1
            
                if tst_min:  delta = sky[newmin+1:minimm+1] - skymid
                else: delta = sky[minimm+1:newmin+1] - skymid
                sum = sum - istep*np.sum(delta)
                sumsq = sumsq - istep*np.sum(delta**2)
                redo = True
                minimm = newmin
    
            newmax = maximm
            if sky[maximm] <= cut2: tst_max = 1           #Is current maximum below upper cut?
            else: tst_max = 0
            if (maximm == nlast) and tst_max: done = 1
            else: done = 0                  #Are we at last pixel of SKY array?
            if not done:
                if maximm+1 < nlast: skyind = maximm+1
                else: skyind = nlast
                if ( tst_max ) and (sky[skyind] > cut2): done = 1 
            if not done: # keep incrementing newmax
                istep = -1 + 2*int(tst_max)         #Increment up or down?
                while not done:
                    newmax = newmax + istep
                    if (newmax == nlast) or (newmax == -1): done = 1
                    if not done:
                        if ( sky[newmax] <= cut2 ) and ( sky[newmax+1] >= cut2 ): done = 1
    
                if tst_max: 
                    delta = sky[maximm+1:newmax+1] - skymid
                else:
                    delta = sky[newmax+1:maximm+1] - skymid
                sum = sum + istep*np.sum(delta)
                sumsq = sumsq + istep*np.sum(delta**2)
                redo = True
                maximm = newmax
    
            #       
            # Compute mean and sigma (from this pass).
            #
            nsky = maximm - minimm
            if ( nsky < minsky ): # error?
                sigma = -1.0 ;  skew = 0.0   
                if printerror: print('ERROR - Outlier rejection left too few sky elements')
                return(skymod,sigma,skew)
    
            skymn = sum/nsky
            var = sumsq/nsky - skymn**2
            if var < 0: var = 0
            sigma = float( np.sqrt( var ))
            skymn = skymn + skymid 
    
            #  Determine a more robust median by averaging the central 20% of pixels.
            #  Estimate the median using the mean of the central 20 percent of sky
            #  values.   Be careful to include a perfectly symmetric sample of pixels about
            #  the median, whether the total number is even or odd within the acceptance
            #  interval
        
            center = (minimm + 1 + maximm)/2.
            side = np.round(0.2*(maximm-minimm))/2.  + 0.25
            j = np.round(center-side).astype(int)
            k = np.round(center+side).astype(int)
    
            #  In case  the data has a large number of of the same (quantized) 
            #  intensity, expand the range until both limiting values differ from the 
            #  central value by at least 0.25 times the read noise.
    
            if readnoise:
                L = round(center-0.25).astype(int)
                M = round(center+0.25).astype(int)
                R = 0.25*readnoise
                while ((j > 0) and (k < nsky-1) and \
                        ( ((sky[L] - sky[j]) < R) or ((sky[k] - sky[M]) < R))):
                    j -= 1
                    k += 1
    
            skymed = np.sum(sky[j:k+1])/(k-j+1)
    
            #  If the mean is less than the median, then the problem of contamination
            #  is slight, and the mean is what we really want.
    
            if skymed < skymn : 
                dmod = 3.*skymed-2.*skymn-skymod 
            else: dmod = skymn - skymod
    
            # prevent oscillations by clamping down if sky adjustments are changing sign
            if dmod*old < 0: clamp = 0.5*clamp
            skymod = skymod + clamp*dmod 
            old = dmod     
    
    #   if redo then goto, START_LOOP
    
    #       
        skew = float( (skymn-skymod)/max([1.,sigma]) )
        nsky = maximm - minimm 
    
        if debug:
            print(('% MMM: Number of unrejected sky elements: ', str(nsky,2), \
                    '    Number of iterations: ',  str(niter)))
            print(('% MMM: Mode, Sigma, Skew of sky vector:', skymod, sigma, skew   ))
    except IndexError:
        sigma=-1.0 ;  skymod = 0.0   ; skew = 0.0
        return(skymod,sigma,skew)
    return(skymod,sigma,skew)

def ipeak(image, xc, yc, apr):
    lxyc = len(xc)
    peakval = np.zeros(lxyc)
    maxapr = np.max(apr)
    for iii in range(lxyc):
        peakval[iii] = np.max(image[np.uint16(np.floor(yc[iii]-maxapr)):np.uint16(np.ceil(yc[iii]+maxapr+1)),np.uint16(np.floor(xc[iii]-maxapr)):np.uint16(np.ceil(xc[iii]+maxapr+1))])
    return peakval
            
def iaper(image, xc, yc, apr, skyradii, phpadu, meanback=False, clipsig=3, maxiter=5, converge_num=0.02, minsky=20):
    maxsky = 10000
    if (image.ndim != 2):
        print('ERROR - Image array (first parameter) must be 2 dimensional')
        exit()
    nrow, ncol = image.shape
    if len(skyradii) != 2 :
        print('skyradii must contain exactly 2 elements')
        exit()
    else:
        skyrad = skyradii
    naper = len(apr)
    if type(xc) == type(np.array([0,1])):
        nstars = len(xc)
    else:
        nstars = 1
    mags = np.zeros((naper, nstars))
    errap = np.zeros((naper, nstars))
    sky = np.zeros(nstars)
    skyerr = np.zeros(nstars)
    area = np.pi*apr**2.
    rinsq = (max(skyrad[0], 0.))**2.
    routsq = skyrad[1]**2.
    lx = np.where(np.trunc(xc-skyrad[1])>0, np.trunc(xc-skyrad[1]), 0)
    ux = np.where(np.trunc(xc+skyrad[1])<(ncol-1), np.trunc(xc+skyrad[1]), ncol-1)
    nx = ux-lx+1
    ly = np.where(np.trunc(yc-skyrad[1])>0, np.trunc(yc-skyrad[1]), 0)
    uy = np.where(np.trunc(yc+skyrad[1])<(nrow-1), np.trunc(yc+skyrad[1]), nrow-1)
    ny = uy-ly+1
    dx = xc-lx
    dy = yc-ly
    edge = np.amin(np.array([[dx-0.5], [nx+0.5-dx], [dy-0.5], [ny+0.5-dy]]), axis=0)
    edge = np.squeeze(edge)
    badstar = ((xc < 0.5) | (xc>ncol-1.5) | (yc<0.5) | (yc> nrow-1.5))
    nbad = np.sum(badstar)
    if nbad > 0:
        print('WARNING - %i star positions outside image'%nbad)
    badval = np.nan
    baderr = np.nan
    for i in range(nstars):
        apmag = np.repeat(badval, naper)
        magerr = np.repeat(baderr, naper)
        skymod = 0.
        skysig = 0.
        if badstar[i]:
            sky[i] = skymod
            skyerr[i] = skysig
            mags[:,i] = apmag
            errap[:,i] = magerr
            continue
        error1 = 1*apmag
        error2 = 1*apmag
        error3 = 1*apmag
        rotbuf = image[ly[i]:uy[i]+1,lx[i]:ux[i]+1]
        dxsq = (np.arange(nx[i])-dx[i])**2.
        rsq = np.zeros((ny[i], nx[i]))
# 9,5 seconds in for loop
        for ii in range(ny[i].astype('int')):
            rsq[ii,:] = dxsq + (ii-dy[i])**2.
        r = np.sqrt(rsq)-0.5
        skypix = (rsq >= rinsq) & (rsq <= routsq)
        nsky = np.sum(skypix)
        nsky = min(nsky, maxsky)
        if nsky < minsky:
            sky[i] = skymod
            skyerr[i] = skysig
            mags[:,i] = apmag
            errap[:,i] = magerr
            continue
        skybuf = np.ravel(rotbuf[skypix])[:nsky]
#23.6 seconds in mmm
        skymod, skysig, skyskew = mmm(skybuf, minsky=minsky)
        skyvar = skysig**2.
        sigsq = skyvar/nsky
        if (skysig < 0.0): 
            sky[i] = skymod
            skyerr[i] = skysig
            mags[:,i] = apmag
            errap[:,i] = magerr
            continue
        skysig = min(skysig, 999.99)
# 5.6 seconds in for loop
        for k in range(naper):
            if edge[i] >= apr[k]:
                thisap = r<apr[k]
                thisapd = rotbuf[thisap]
                thisapr = r[thisap]
                fractn = np.where(apr[k]-thisapr<1.0, apr[k]-thisapr, 1.0)
                fractn = np.where(fractn>0.0, fractn, 0.0)
                full = fractn == 1.0
                nfull = np.sum(full)
                gfract = fractn != 1.0
                factor = (area[k]-nfull)/np.sum(fractn[gfract])
                fractn[gfract] = fractn[gfract]*factor
                apmag[k] = np.sum(thisapd*fractn)
# 1.8 seconds in lines including if statement         
        g = np.isfinite(apmag)
        ng = np.sum(g)
        if ng > 0:
            apmag[g] = apmag[g]-skymod*area[g]
            error1[g] = area[g]*skyvar
            error2[g] = np.where(apmag[g]>0, apmag[g], 0)/phpadu
            error3[g] = sigsq*area[g]**2.
            magerr[g] = np.sqrt(error1[g]+error2[g]+error3[g])
        sky[i] = skymod
        skyerr[i] = skysig
        mags[:,i] = apmag
        errap[:,i] = magerr
    return mags, errap, sky, skyerr

def outliers(data,sigma=5.0,limit=100):
    ndata = len(data)
    selected = np.zeros(ndata,dtype='uint8')+1
    nselected = ndata
    nprev = 0
    counter = 0
    while (nselected != nprev) & (counter <= limit):
        nprev = nselected
        wselected = np.where(selected == 1)[0]
        meandata = np.mean(data[wselected])
        rmsdata = np.std(data[wselected],ddof=1)
        newselected = np.where(np.abs(data-meandata) > (sigma*rmsdata))[0]
        if len(newselected) > 0: selected[newselected] = 0
        nselected = np.sum(selected) 
        counter = counter+1
    return np.where(selected == 1)[0]

def GetCentroids(data,xin,yin,binsize,brightselect=40,faintselect=-1,debug=0,smooth=None):
    maxiter = 20
    if faintselect == -1: faintselect = np.power(binsize,2)-40
    if smooth != None: gauss = aco.Gaussian1DKernel(stddev=3.0)
    xsize = data.shape[1]
    ysize = data.shape[0]
    nx = len(xin)
    xarr = (((np.arange(binsize*binsize)).reshape(binsize,binsize) % binsize)-binsize/2.0).reshape(np.power(binsize,2))
    yarr = (((np.arange(binsize*binsize)).reshape(binsize,binsize) / binsize)-binsize/2.0).reshape(np.power(binsize,2))
    x = np.zeros((nx))
    y = np.zeros((nx))
    x2 = np.zeros((nx))
    y2 = np.zeros((nx))
    bg = np.zeros((nx))
    ebg = np.zeros((nx))
    nbg = np.zeros((nx),dtype='uint16')
    flux = np.zeros((nx))
    flag = np.zeros((nx),dtype='uint16')
    if debug > 0:
        pxyarr  = (xarr+yarr)/np.sqrt(2)
        mxyarr  = (xarr-yarr)/np.sqrt(2)
        pxy = np.zeros((nx))
        mxy = np.zeros((nx))
        pxy2 = np.zeros((nx))
        mxy2 = np.zeros((nx))
        fg = np.zeros((nx))
        efg = np.zeros((nx))
        nfg = np.zeros((nx))
        iters = np.zeros((nx))
#   1 Saturated pixel(s) in aperture radius
#   2 Saturated pixel(s) in background
#   4 Uneven sky background distribution (for instance a binary pair, or laser or airplane or UFO)
#   8 Reserved (moon)
#  16 Reserved
#  32 Background flux is less or equal to 0
#  64 Flux is less or equal to 0
# 128 Flux is not finite/no idea what happened her
    for nnn in range(nx):
        if ((np.round(xin[nnn]) <= binsize/2) | (np.round(yin[nnn]) <= binsize/2) | (np.round(xin[nnn]) >= xsize-binsize/2-1) | (np.round(yin[nnn]) >= ysize-binsize/2-1)):
            bg[nnn] = -1 
            ebg[nnn] = -1 
            nbg[nnn] = 0 
            flux[nnn] = -1 
            x2[nnn] = binsize
            y2[nnn] = binsize
            if debug > 0:
                pxy[nnn] = 0.0
                mxy[nnn] = 0.0
                pxy2[nnn] = binsize
                mxy2[nnn] = binsize
            x[nnn] = np.round(xin[nnn])
            y[nnn] = np.round(yin[nnn])
            flag[nnn] = 255
        else:
            niter = 0
            prevx = 0
            prevy = 0
            while ((prevx != np.round(x[nnn])) | (prevy != np.round(y[nnn])) | (niter == 0)) & (niter < maxiter) & (flag[nnn] != 255): 
                if (((np.round(xin[nnn])+np.round(x[nnn])) <= binsize/2) | ((np.round(yin[nnn])+np.round(y[nnn])) <= binsize/2) | 
                    ((np.round(xin[nnn])+np.round(x[nnn])) >= xsize-binsize/2-1) | ((np.round(yin[nnn])+np.round(y[nnn])) >= ysize-binsize/2-1)):
                    bg[nnn] = -1 
                    ebg[nnn] = -1 
                    nbg[nnn] = 0 
                    flux[nnn] = -1 
                    x2[nnn] = binsize
                    y2[nnn] = binsize
                    if debug > 0:
                        pxy[nnn] = 0.0
                        mxy[nnn] = 0.0
                        pxy2[nnn] = binsize
                        mxy2[nnn] = binsize
                    x[nnn] = np.round(xin[nnn])
                    y[nnn] = np.round(yin[nnn])
                    flag[nnn] = 255
                    niter = niter+1
                else:
                    prevx = np.round(x[nnn])
                    prevy = np.round(y[nnn])
                    binimage = data[np.round(yin[nnn])+np.round(y[nnn])-binsize/2:np.round(yin[nnn])+np.round(y[nnn])+binsize/2,np.round(xin[nnn])+np.round(x[nnn])-binsize/2:np.round(xin[nnn])+np.round(x[nnn])+binsize/2].reshape(np.power(binsize,2))
                    if smooth != None: binimage = aco.convolve(binimage, gauss, boundary='extend')
                    select = (np.argsort(binimage)[::-1])[0:brightselect-1]
                    deselect = (np.argsort(binimage)[::-1])[np.power(binsize,2)-faintselect:np.power(binsize,2)-1]
                    outl = outliers(binimage[deselect],sigma=3.0) 
                    bg[nnn] = np.mean(binimage[deselect[outl]])
                    binimage = binimage-bg[nnn]
                    totalsum = np.sum(binimage[select])
                    if totalsum != 0: 
                        x[nnn] = np.sum((xarr*binimage)[select])/totalsum+np.round(x[nnn])
                        y[nnn] = np.sum((yarr*binimage)[select])/totalsum+np.round(y[nnn])
                    else: 
                        bg[nnn] = -1 
                        ebg[nnn] = -1 
                        nbg[nnn] = 0 
                        flux[nnn] = -1 
                        x2[nnn] = binsize
                        y2[nnn] = binsize
                        if debug > 0:
                            pxy[nnn] = 0.0
                            mxy[nnn] = 0.0
                            pxy2[nnn] = binsize
                            mxy2[nnn] = binsize
                        x[nnn] = np.round(xin[nnn])
                        y[nnn] = np.round(yin[nnn])
                        flag[nnn] = 255
                    niter = niter+1
            if flag[nnn] != 255:
                ebg[nnn] = np.std(binimage[deselect[outl]],ddof=1)
                nbg[nnn] = len(outl)
                flux[nnn] = np.sum(binimage) 
                if debug > 0: 
                    outlf = outliers(binimage[select],sigma=3.0) 
                    fg[nnn] = np.mean(binimage[select[outlf]])
                    efg[nnn] = np.std(binimage[select[outlf]],ddof=1)
                    nfg[nnn] = len(outlf)
                    iters[nnn] = niter
                    pxy[nnn] = np.sum((pxyarr*binimage)[select])/np.sum(binimage[select])
                    mxy[nnn] = np.sum((mxyarr*binimage)[select])/np.sum(binimage[select])
                    pxy2[nnn] = np.sum((np.power((pxyarr-pxy[nnn]),2)*binimage)[select])/np.sum(binimage[select])
                    mxy2[nnn] = np.sum((np.power((mxyarr-mxy[nnn]),2)*binimage)[select])/np.sum(binimage[select])
                x2[nnn] = np.sum((np.power((xarr-x[nnn]),2)*binimage)[select])/np.sum(binimage[select])
                y2[nnn] = np.sum((np.power((yarr-y[nnn]),2)*binimage)[select])/np.sum(binimage[select])
                x[nnn] = x[nnn]+np.round(xin[nnn])
                y[nnn] = y[nnn]+np.round(yin[nnn])
                if len(np.where(binimage[deselect] > 60000)[0]) > 0: flag[nnn] = flag[nnn]+1
                if len(np.where(binimage[select] > 60000)[0]) > 0  : flag[nnn] = flag[nnn]+2
                if (bg[nnn] <= 0)                                  : flag[nnn] = flag[nnn]+32
                if (flux[nnn] <= 0)                                : flag[nnn] = flag[nnn]+64
                if (np.isfinite(flux[nnn]) != True)                : flag[nnn] = flag[nnn]+128
    if debug > 0:
        result = pd.DataFrame([x,y,bg,ebg,nbg,flux,x2,y2,pxy,mxy,pxy2,mxy2,flag,fg,efg,nfg],
            index=['x','y','bg','ebg','nbg','flux','x2','y2','pxy','mxy','pxy2','mxy2','flag','fg','efg','nfg']).T
    else:
        result = pd.DataFrame([x,y,bg,ebg,nbg,flux,x2,y2,flag],
            index=['x','y','bg','ebg','nbg','flux','x2','y2','flag']).T
    return result

def saveCompFits(filename, data, header):
    ''' to save an image in a compressed fits.
       Inputs: - path, the path for saving the images
               - name, the name of the image
               - data, the data to be saved
               - header, the corresponding header
        The data is saved in a CompImageHDU, using a RICE compression type.
        If the data is not in UINT16 format, it will be converted before
        the compression.
    '''
    ny, nx = data.shape
    header.set('BITPIX', 16, comment='array data type', before=0)
    header.set('NAXIS', 2, comment='number of array dimensions', after='BITPIX')
    header.set('NAXIS1', nx, after='NAXIS')
    header.set('NAXIS2', ny, after='NAXIS1')
    f = pf.CompImageHDU(data=data, header=header, compression_type='RICE_1', \
                        name='COMP_IMAGE', uint=True)
    f.verify(option='fix')
    f.writeto(filename, clobber=True)
    return


def histeq(im,nbr_bins=256):
   #get image histogram
   imhist,bins = np.histogram(im.flatten(),nbr_bins,normed=True)
   cdf = imhist.cumsum() #cumulative distribution function
   cdf = 255 * cdf / cdf[-1] #normalize
   #use linear interpolation of cdf to find new pixel values
   im2 = np.interp(im.flatten(),bins[:-1],cdf)
   return im2.reshape(im.shape), cdf
