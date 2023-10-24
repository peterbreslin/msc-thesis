#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

class PolarGrid(object):
    """ Create a polar coordinate grid.
    
    Attributes:
        nx (int): The resolution of the polargrid along the ra-axis.
        ny (int): The resolution of the polargrid along the dec-axis.
        xedges (float): The binedges of the grid along the ra-axis.
        yedges (float): The binedges of the grid along the dec-axis.
        npix (int): The total number of pixels on the grid.
    
    """
    
    def __init__(self, nx, ny):
        """ Initialize the coordinate grid.
        
        Args:
            nx (int): The resolution of the polargrid along the ra-axis.
            ny (int): The resolution of the polargrid along the dec-axis.
            
        """
        
        self.nx = nx
        self.ny = ny
        
        self.xedges = np.linspace(0, 360, self.nx+1)
        self.yedges = np.linspace(-90, 90, self.ny+1)
        
        self.npix = (nx + 2)*(ny + 2)
        
        return
    
    def _ra2idx(self, ra):
        
        idx = np.searchsorted(self.xedges, ra, 'right')
        
        return idx
        
    def _dec2idx(self, dec):
        
        idx = np.searchsorted(self.yedges, dec, 'right')
        
        return idx
    
    def radec2idx(self, ra, dec):
        """ Convert ra, dec coordinates to indices on the grid.
        
        Args:
            ra (float): An array of ra coordinates.
            dec (float): An array of dec coordinates.
            
        Returns:
            idx1 (int): First index of the cell, corresponds to ra.  
            idx2 (int): Second index of the cell, corresponds to dec. 
            
        """
        
        idx1 = self._ra2idx(ra)
        idx2 = self._dec2idx(dec)
        
        return idx1, idx2
            
    def _idx2ra(self, idx):
        
        ra = np.full(self.nx+2, fill_value=np.nan)
        ra[1:-1] = (self.xedges[:-1] + self.xedges[1:])/2.
        
        return ra[idx]
        
    def _idx2dec(self, idx):
        
        dec = np.full(self.ny+2, fill_value=np.nan)
        dec[1:-1] = (self.yedges[:-1] + self.yedges[1:])/2.
        
        return dec[idx]
        
    def idx2radec(self, idx1, idx2):
        """ Convert indices on the grid to gridcell coordinates.
        
        Args:
            idx1 (int): An array of cell indices along the ra-axis.
            idx2 (int): An array of cell indices along the dec-axis.
            
        Returns:
            ra (float): The ra coordinates of the gridcells.
            dec (float): The dec coordinates of the gridcells.
            
        """
        
        ra = self._idx2ra(idx1)
        dec = self._idx2dec(idx2)
        
        return ra, dec
        
    def values2grid(self, idx1, idx2, values, fill_value=0):
        """ Given grid indices and values return an array.
        
        Args:
            idx1 (int): An array of indices along the ra-axis.
            idx2 (int): An array of indices along the dec-axis.
            values (float): An array of values corresponding to idx1, idx2. 
            fill_value (float): Value to fill unspecified cells with.
            
        Returns:
            array (float): An array corresponding to the grid.
        
        """
        
        array = np.full((self.nx+2, self.ny+2), fill_value=fill_value)
        array[idx1, idx2] = values
        
        return array


class PolarEAGrid(object):
    """ Create a polar coordinate grid.
    
    Attributes:
        nrings (int): The number of declination rings between the polar caps.
        yedges (float): The binedges of the grid along the dec-axis.
        ncells (int): The number of grid cells in each ring.
        npix (int): The total number of pixels on the grid.
    
    """
    
    def __init__(self, nrings):
        """ Initialize the coordinate grid.
        
        Args:
            nrings (int): The number of declination rings between the polar caps.
            
        """
        
        self.nrings = nrings
        
        # The edges of the declination rings.
        yedges = np.linspace(-90. + 90./(self.nrings + 1), 90. - 90./(self.nrings + 1), self.nrings + 1)
        self.yedges = np.hstack([-90., yedges, 90.])
        
        # The number of cells in each ring.
        idx = np.arange(self.nrings)
        ncells = np.cos(np.pi*(idx + .5)/(self.nrings + 1)) - np.cos(np.pi*(idx + 1.5)/(self.nrings + 1))
        ncells = ncells/(1 - np.cos(np.pi*.5/(self.nrings + 1)))
        ncells = np.rint(ncells).astype('int')
        self.ncells = np.hstack([1, ncells, 1])
        
        # The total number of pixels on the grid.
        self.npix = np.sum(self.ncells)
        
        return
        
    def radec2idx(self, ra, dec):
        """ Convert ra, dec coordinates to indices on the grid.
        
        Args:
            ra (float): An array of ra coordinates.
            dec (float): An array of dec coordinates.
            
        Returns:
            ring (int): The declination ring the datapoint belongs to.
            cell (int): The cell on the ring the datapoint belongs to.
            idx (int): The pixel on the full grid the datapoint belongs to.
            
        """
        
        # Find the ring each datapoint belongs to.
        ring = np.searchsorted(self.yedges, dec, 'right')
        
        # Clean up the out-of-bound indices.
        ring[ring == 0] = 1
        ring[ring == (self.nrings + 3)] = self.nrings + 2
        ring = ring - 1 

        cell = np.zeros(len(ra), dtype='int')
        for i in range(self.nrings + 2):

            select = (ring == i)
            
            # Find the cell along the ring the datapoint belongs to.
            xedges = np.linspace(0, 360, self.ncells[i] + 1)
            cell_ = np.searchsorted(xedges, ra[select], 'right')

            # Clean up the out-of-bounds indices.
            cell_[cell_ == 0] = 1
            cell_[cell_ == (self.ncells[i] + 1)] = self.ncells[i]
            cell_ = cell_ - 1        
            
            cell[select] = cell_  
            
        # Create a single index.
        idx = (np.cumsum(self.ncells) - self.ncells)[ring] + cell
        
        return ring, cell, idx
        
    def idx2radec(self, idx):
        
        print('Warning: this function has not yet been written.')
        
        return 0., 0.
        
    def values2grid(self, idx, values, fill_value=0):
        """ Given grid indices and values return an array.
        
        Args:
            idx (int): An array of indices.
            values (float): An array of values corresponding to idx. 
            fill_value (float): Value to fill unspecified cells with.
            
        Returns:
            array (float): An array corresponding to the grid.
        
        """
        
        array = np.full(self.npix, fill_value=fill_value)
        array[idx] = values
        
        return array


class HealpixGrid(object):
    """ Create a Healpix coordinate grid.
    
    Attributes:
        nside (int): Determines the resolution of the grid for further
            information refer to the healpy documentation.
        npix (int): The total number of pixels on the grid.
    
    """
    
    def __init__(self, nside):
        """ Initialize the coordinate grid.
        
        Args:
            nside (int): Determines the resolution of the grid for further
            information refer to the healpy documentation.
            
        """

        import healpy
        self.healpy = healpy
        
        self.nside = nside
        self.npix = self.healpy.nside2npix(self.nside)
        
        return
    
    def radec2idx(self, ra, dec):
        """ Convert ra, dec coordinates to indices on the grid.
        
        Args:
            ra (float): An array of ra coordinates.
            dec (float): An array of dec coordinates.
            
        Returns:
            idx (int): The index of the gridcell to which each ra, dec pair
                belongs.
        
        """

        idx = self.healpy.ang2pix(self.nside, (90. - dec)*np.pi/180., ra*np.pi/180.)
        
        return idx
            
    def idx2radec(self, idx):
        """ Converts gridcell indices the the ra, dec of the cell.
        
        Args:
            idx (int): Gridcell indices.
            
        Returns:
            ra (float): The ra coordinates of each of the cells.
            dec (float): The dec coordinates of each of the cells.
            
        """
    
        theta, phi = self.healpy.pix2ang(self.nside, idx)
        dec = 90. - theta*180./np.pi
        ra = phi*180./np.pi
            
        return ra, dec
        
    def values2grid(self, idx, values, fill_value=0):
        """ Given grid indices and values return an array.
        
        Args:
            idx (int): An array of gridcell indices.
            values (float): An array of values corresponding to idx. 
            fill_value (float): Value to fill unspecified cells with.
            
        Returns:
            array (float): An array corresponding to the grid.
        
        """
        
        array = np.full(self.npix, fill_value=fill_value)
        array[idx] = values
        
        return array

    
class CartesianGrid(object):
    """ Create a cartesian coordinate grid.
    
    Attributes:
        nx (int): The resolution of the grid along the x-axis.
        ny (int): The resolution of the grid along the y-axis.
        xedges (float): The binedges of the grid along the x-axis.
        yedges (float): The binedges of the grid along the y-axis.
        npix (int): The total number of pixels on the grid.
    
    """
    
    def __init__(self, nx, ny, Lx=4008, Ly=2672):
        """ Initialize the coordinate grid.
        
        Args:
            nx (int): The resolution of the grid along the x-axis.
            ny (int): The resolution of the grid along the y-axis.
            Lx (float): The maximum x-coordinate default 4008.
            Ly (float): The maximum y-coordinate default 2672.
            
        """
        
        self.nx = nx
        self.ny = ny
        
        self.xedges = np.linspace(0, Lx, self.nx+1)
        self.yedges = np.linspace(0, Ly, self.ny+1)
        
        self.npix = (nx + 2)*(ny + 2)
        
        return
        
    def _x2idx(self, x):
        
        idx = np.searchsorted(self.xedges, x, 'right')
        
        return idx

    def _y2idx(self, y):
        
        idx = np.searchsorted(self.yedges, y, 'right')
        
        return idx
        
    def xy2idx(self, x, y):
        """ Convert x, y coordinates the the indices of gridcells.
        
        Args:
            x (float): The x-coordinates.
            y (float): The y-coordinates.
            
        Returns:
            idx1 (int): The indices corresponding to the x-coordinates.
            idx2 (int): The indices corresponding to the y-coordinates.
            
        """
        
        idx1 = self._x2idx(x)
        idx2 = self._y2idx(y)
        
        return idx1, idx2
        
    def _idx2x(self, idx):
        
        x = np.full(self.nx+2, fill_value=np.nan)
        x[1:-1] = (self.bins1[:-1] + self.bins1[1:])/2.
        
        return x[idx]
        
    def _idx2y(self, idx):
        
        y = np.full(self.ny+2, fill_value=np.nan)
        y[1:-1] = (self.bins2[:-1] + self.bins2[1:])/2.
        
        return y[idx]
    
    def idx2xy(self, idx1, idx2):
        """ Convert the indices of the gridcells to x, y coordinates.
        
        Args:
            idx1 (float): The indices corresponding to the x-coordinates.
            idx2 (float): The indices corresponding to the y-coordinates.
            
        Returns:
            x (float): The central x-coordinates of the gridcells.
            y (float): The central y-coordinates of the gridcells.
        
        """
        
        x = self._idx2x(idx1)
        y = self._idx2y(idx2)
        
        return x, y    
    
    def values2grid(self, idx1, idx2, values, fill_value=0):
        """ Given grid indices and values return an array.
        
        Args:
            idx1 (int): An array of indices along the x-axis.
            idx2 (int): An array of indices along the y-axis.
            values (float): An array of values corresponding to idx1, idx2. 
            fill_value (float): Value to fill unspecified cells with.
            
        Returns:
            array (float): An array corresponding to the grid.
        
        """
        
        array = np.full((self.nx+2, self.ny+2), fill_value=fill_value)
        array[idx1, idx2] = values
        
        return array
