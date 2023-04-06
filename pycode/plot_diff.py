import glob
import astropy.io.fits as pf
import matplotlib.pyplot as plt

img = glob.glob("../test_data/*.fits.gz")[0]

data, header = pf.getdata(img, header=True)

plt.figure(figsize=[6,4])
plt.imshow(data, vmin=-10, vmax=10)
plt.show()