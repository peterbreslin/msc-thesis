import cartopy.crs as ccrs
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

plt.figure(figsize=(6, 6))


def decorate_axes(ax):
    ax.set_global()
    ax.coastlines()


def animate(i):
    lon = i

    ax = plt.gca()
    ax.remove()

    ax = plt.axes([0, 0, 1, 1], projection=ccrs.Orthographic(
        central_latitude=0, central_longitude=lon))
    decorate_axes(ax)


ani = animation.FuncAnimation(
    plt.gcf(), animate,
    frames=np.linspace(0, 360, 40),
    interval=125, repeat=False)

plt.show()