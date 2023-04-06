import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import load, wgs84, EarthSatellite, S, W

url = 'https://celestrak.org/NORAD/elements/supplemental/starlink.txt'
sats = load.tle_file(url)
sat = sats[0]

R_earth = 6371.
ax = plt.figure().add_subplot(projection='3d')
ax.xaxis._axinfo['grid'].update({'linewidth':0.1})
ax.yaxis._axinfo['grid'].update({'linewidth':0.1})
ax.zaxis._axinfo['grid'].update({'linewidth':0.1})

ts = load.timescale()
hrs = np.arange(0, 3, 0.01)
t = ts.utc(2022, 10, 23, hrs)
position = sat.at(t).position.km

img = plt.imread('../blue_marble_lowres.jpeg')
theta = np.linspace(0, np.pi, img.shape[0])
phi = np.linspace(0, 2*np.pi, img.shape[1])
count = 180
theta_inds = np.linspace(0, img.shape[0] - 1, count).round().astype(int)
phi_inds = np.linspace(0, img.shape[1] - 1, count).round().astype(int)
theta = theta[theta_inds]
phi = phi[phi_inds]
img = img[np.ix_(theta_inds, phi_inds)]
theta, phi = np.meshgrid(theta, phi)

x = R_earth * np.sin(theta) * np.cos(phi)
y = R_earth * np.sin(theta) * np.sin(phi)
z = R_earth * np.cos(theta)
ax.plot_surface(x.T, y.T, z.T, facecolors=img/255, cstride=1, rstride=1, zorder=0)

x, y, z = position
ax.plot(x, y, z, c='C3', zorder=1)

# Overplot MASCARA
lasilla = wgs84.latlon(latitude_degrees=29.26111*S, longitude_degrees=70.73139*W, elevation_m=2400)
m_lon = lasilla.longitude.radians
m_lat = lasilla.latitude.radians
x = R_earth * np.sin(m_lat) * np.cos(m_lon)
y = R_earth * np.sin(m_lat) * np.sin(m_lon)
z = R_earth * np.cos(m_lat)
ax.scatter(x, y, z, marker='*', s=100, c='C1', zorder=2)

plt.show()