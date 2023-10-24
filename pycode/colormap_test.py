import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt

rootdir = '/Users/peter/Projects/starlink_data/'

lstseq = '48510542'
diff = pf.getdata(f'{rootdir}test_data/diff_images/LSC/diff_{lstseq}LSC.fits.gz')
mask = pf.getdata(f'{rootdir}noise_reduction_demo/{lstseq}_starsmasked.fits.gz')

# List of available colormaps
colormaps = plt.colormaps()

# Function to create a new plot with the next colormap
def create_next_plot(idx, colormap):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[12,8], sharey=True)
    ax1.imshow(diff, vmin=-15, vmax=100., cmap=colormap)
    ax2.imshow(mask, vmin=-15, vmax=100., cmap=colormap)

    ax1.set_title(f'{idx}: {colormap}')
    ax1.axis('off')
    ax2.set_title(f'{idx}: {colormap}')
    ax2.axis('off')

    plt.tight_layout()
    plt.show()

for idx, colormap in enumerate(colormaps):
    create_next_plot(idx, colormap)