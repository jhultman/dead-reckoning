import numpy as np
from numpy import cos, sin, pi
from imageio import imread


def load_data(lati_longi_fname, yaw_fname, groundtruth_fname):
    load = lambda f: np.loadtxt(f, delimiter=',')
    lati_longi = load(lati_longi_fname)
    yaw = load(yaw_fname)
    groundtruth = load(groundtruth_fname)
    lati, longi = lati_longi.T
    return lati, longi, yaw, groundtruth


def gen_xy_from_lati_longi(lati, longi, c0):
    lati_adj = lati - lati[0];
    longi_adj = longi - longi[0];   

    x = -longi_adj * (c0 * cos(lati) / 360)
    y = +lati_adj * (c0 / 360)
    return x, y


def rotate(x, y, theta):
    c, s = cos(theta), sin(theta)
    rotation = [[c, -s], [s, c]]
    
    xy = np.vstack((x.T, y.T))
    x, y = np.dot(rotation, xy)
    return x, y


def gen_xy_from_yaw(yaw, dt, v0):
    psi0 = -pi / 2 + yaw[0]
    psiR = yaw

    dxdt = v0 * sin(psi0 - psiR)
    dydt = v0 * cos(psi0 - psiR)

    x = (dt * dxdt).cumsum()
    y = (dt * dydt).cumsum()
    return x, y


def shift_and_scale(x0, y0, x1, y1):
    """
    Ensure (x0, y0) and (x1, y1) 
    have same mean and range.
    """
    ux = x1.mean() - x0.mean()
    uy = y1.mean() - y0.mean()

    alpha_x = (x0.max() - x0.min()) / (x1.max() - x1.min())
    alpha_y = (y0.max() - y0.min()) / (y1.max() - y1.min())

    x1 = alpha_x * (x1 - ux)
    y1 = alpha_y * (y1 - uy)
    return x1, y1
