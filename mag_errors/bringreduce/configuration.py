import os

darktables = {}
astromaster = {}
systable = {}

def initialize(targetdir):
    global confdir
    global starcat
    global siteinfo
    global targets

    confdir = "/Users/peter/Projects/starlink_data/fotos-python3/bringfiles"
    
    camid = targetdir[-3:]
    darktables[camid] = [os.path.join(confdir, 'darktable'+camid+'long.dat'),
                      os.path.join(confdir, 'darktable'+camid+'short.dat')]
    
    astromaster[camid] = os.path.join(confdir, 'astromaster'+camid+'.hdf5')
    systable[camid] = os.path.join(confdir, 'systable'+camid+'.dat')


    ###############################################################################
    ### Locations of various files.
    ###############################################################################
    
    starcat = os.path.join(confdir, 'bringcat20191231.fits') # Location of the stellar catalogue.
    siteinfo = os.path.join(confdir, 'siteinfo.dat') # Location of the siteinfo file.
    targets = os.path.join(confdir, 'targets.dat') # Location of the targets file.