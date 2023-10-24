# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 16:19:48 2016

@author: talens
"""

import numpy as np

def flux2mag(flux, eflux=None, m0=25.):
    
    mag = m0 - 2.5*np.log10(flux)
    
    if eflux is not None:
        emag = 2.5/np.log(10.)*np.abs(eflux/flux)
        return mag, emag
    
    return mag
    
def sigma_clip(values, sigma=5., maxiter=5):
    
    mask = np.ones(len(values), dtype='bool')
    
    for i in range(maxiter):
        
        m0 = np.nanmean(values[mask])
        m1 = np.nanstd(values[mask])
        
        mask = (np.abs(values - m0) < sigma*m1)
        
    return m0, m1
    
def clip_rectangle(x, y, nx, ny, margin):
    
    maskx = (x > margin) & (x < (nx - margin))
    masky = (y > margin) & (y < (ny - margin))
    
    return maskx & masky
    