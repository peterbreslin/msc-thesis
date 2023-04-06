import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skyfield.api import load, wgs84, EarthSatellite, S, W

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--hours', help='Number of hours for orbit', type=float, default=1.5)
args = parser.parse_args()
n = args.hours

url = 'https://celestrak.org/NORAD/elements/supplemental/starlink.txt'
sats = load.tle_file(url)
sat = sats[0]


R_earth = 6371.
ax = plt.figure().add_subplot(projection='3d')
ax.xaxis._axinfo['grid'].update({'linewidth':0.1})
ax.yaxis._axinfo['grid'].update({'linewidth':0.1})
ax.zaxis._axinfo['grid'].update({'linewidth':0.1})

ts = load.timescale()
hrs = np.arange(0, n, 0.01)
t = ts.utc(2022, 10, 23, hrs)
position = sat.at(t).position.km

## Spherical coords --> (r, theta, phi) ... create meshgrid of n by m points (u=theta, v=phi)
u, v = np.mgrid[0:2*np.pi:40j, 0:np.pi:20j]

## Cartesian coords of each point on the surface of the sphere
x = R_earth * np.cos(u) * np.sin(v)
y = R_earth * np.sin(u) * np.sin(v)
z = R_earth * np.cos(v)
ax.plot_surface(x, y, z, color='C0', alpha=0.1)
ax.plot_wireframe(x, y, z, rstride=5, cstride=5, color='C0', alpha=0.1)

## Create a plane going through the Equator
X, Y = np.meshgrid(np.linspace(-R_earth, R_earth, 10), np.linspace(-R_earth, R_earth, 10))
Z = np.zeros_like(X)
ax.plot_surface(X, Y, Z, color='C9', alpha=0.3)

## Create a plane going through the axis of rotation of Earth 
adj = R_earth * np.cos(np.radians(66.5)) #90-23.5
X, Y = np.meshgrid(np.linspace(-R_earth, R_earth, 10), np.linspace(-adj, adj, 10))
Z = Y * np.tan(np.radians(66.5)) 
ax.plot_surface(X, Y, Z, color='C4', alpha=0.3)

## Satellite track
x, y, z = position
ax.plot(x, y, z, c='C3', ls='--')
ax.scatter(x[0], y[0], z[0], s=25, c='C2')
ax.scatter(x[-1], y[-1], z[-1], s=25, c='C3')

## Overplot MASCARA
mas_lon = -70.73139 
mas_lat = -29.26111 
mas_alt = 2.4
theta = np.pi/2 - np.radians(mas_lat)
phi = np.radians(mas_lon)
r = R_earth + mas_alt
x = r * np.sin(theta) * np.cos(phi)
y = r * np.sin(theta) * np.sin(phi)
z = r * np.cos(theta)
ax.scatter(x, y, z, marker='*', s=100, c='C1')

ax.set_axis_off()
plt.show()