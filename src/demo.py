import os.path as osp
import numpy as np
import matplotlib.pyplot as plt
from dead_reckoning import *


def configure_axis(axis, title):
    axis.set_xticklabels([])
    axis.set_yticklabels([])
    axis.tick_params(axis='both', which='both', length=0)
    axis.set_xlabel('East')
    axis.set_ylabel('North')
    axis.set_title(title)


def plot_navigation(x0, y0, x1, y1, groundtruth, img):
    x2, y2 = groundtruth.T

    fig, axarr = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
    titles = ['GPS', 'Magnetometer', 'Google Maps']

    axarr[0].plot(x0, y0, c='black')
    axarr[1].plot(x1, y1, c='black')
    axarr[2].plot(x2, y2, c='orange', linestyle='dashed')
    axarr[2].imshow(img, aspect='auto')

    for i in range(3):
        configure_axis(axarr[i], titles[i])

    fig.tight_layout()
    plt.savefig('../images/dead-reckoning.png')


def demo():
    circ = 40.075 * 1e6  # Circumference of the Earth (m)
    vel = 0.97           # Measured walking speed (m/s)
    dt = 0.02            # Interval between yaw samples (s)

    datadir = osp.join('..', 'data')
    fnames = ['lati-longi', 'yaw', 'groundtruth']
    fpaths = [osp.join(datadir, f'{fname}.csv') for fname in fnames]
    lati, longi, yaw, groundtruth = load_data(*fpaths)

    x0, y0 = gen_xy_from_lati_longi(lati, longi, circ)
    x0, y0 = rotate(x0, y0, -pi / 9)

    x1, y1 = gen_xy_from_yaw(yaw, dt, vel)
    x1, y1 = shift_and_scale(x0, y0, x1, y1)

    img = imread('../data/upenn.png')
    plot_navigation(x0, y0, x1, y1, groundtruth, img)


def main():
    demo()

if __name__ == '__main__':
    main()
