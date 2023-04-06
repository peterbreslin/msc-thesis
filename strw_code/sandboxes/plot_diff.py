import os
import numpy as np
import astropy.io.fits as pf
import matplotlib.pyplot as plt

import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--camera', type=str,required=True, help='which camera (LSC/LSS/LSN/LSW/LSE)')
parser.add_argument('-i', '--image', type=str, required=False, help='specify what image to use')
parser.add_argument('--random', action='store_true', help='choose for a random image')
args = parser.parse_args()
camera = args.camera
image  = args.image
random = args.random

# 48510733LSC
# 48511058LSC
# 48506343LSN
path = f'/net/beulakerwijde/data1/breslin/data/subtracted/20221023{camera}'


if random:
    print(f'Choosing a random image from 20221023{camera}')
    files = glob.glob(path+'/*.fits.gz')
    i = np.random.randint(0, len(files))
    img = files[i]
else:
    img = f'{path}/diff_{image}{camera}.fits.gz'


print(f'Image: {img[-19:-8]}')
data, header = pf.getdata(img, header=True)


def new_image(event):
    if event.key == 'n':
        ax.cla()
        data = get_image(args)
        ax.imshow(data, vmin=-10, vmax=10)


#fig, ax = plt.subplots(1, 1, figsize=[12,8])
#fig.canvas.mpl_connect('key_press_event', new_image)

plt.figure(figsize=[12,8])
plt.imshow(data, vmin=-10, vmax=10)
plt.show()
