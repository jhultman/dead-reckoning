import numpy as np
import matplotlib.pyplot as plt
from dead_reckoning import *


def configure_axis(axis, title):
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(axis='both', which='both', length=0)
    axis.set_title(title)
    axis.set_xlabel('East')
    axis.set_ylabel('North')


def plot_navigation(x1, y1, x2, y2, img):
    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    titles = ['GPS tracking', 'Yaw attitude', 'Google Maps']

    axarr[0].plot(x1, y1)
    axarr[1].plot(x2, y2)
    axarr[2].imshow(img, aspect='auto')

    for i in range(3):
        configure_axis(axarr[i], titles[i])

    fig.tight_layout()
    plt.savefig('../images/dead-reckoning.png')


def main():
    c0 = 40.075 * 1e6;  # Circumference of the Earth (m)
    v0 = 0.97           # Measured walking speed (m/s)
    dt = 0.02           # Interval between yaw samples (s)

    fnames = ('../data/lati-longi.csv', '../data/yaw.csv')
    lati, longi, yaw = load_data(*fnames)

    x1, y1 = gen_xy_from_lati_longi(lati, longi, c0)
    x1, y1 = rotate(x1, y1, -pi / 9)

    x2, y2 = gen_xy_from_yaw(yaw, dt, v0)
    x2, y2 = shift_and_scale(x1, y1, x2, y2)

    img = imread('../data/upenn.png')
    plot_navigation(x1, y1, x2, y2, img)


if __name__ == '__main__':
    main()
