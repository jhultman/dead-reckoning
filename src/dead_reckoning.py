import numpy as np
from scipy.misc import imread
from numpy import cos, sin, pi


def load_data(lati_longi_fname, yaw_fname):
    lati_longi = np.loadtxt(lati_longi_fname, delimiter=',')
    yaw = np.loadtxt(yaw_fname, delimiter=',')
    lati = lati_longi[:, 0]
    longi = lati_longi[:, 1]
    return lati, longi, yaw


def gen_xy_from_lati_longi(lati, longi, c0):
    lati_adj = lati - lati[0];
    longi_adj = longi - longi[0];   

    x = -(c0 * cos(lati) / 360) * longi_adj
    y = (c0 / 360) * lati_adj
    return x, y


def rotate(x, y, theta):
    c, s = cos(theta), sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    
    xy = np.vstack((x.transpose(), y.transpose()))
    xy = np.dot(rotation, xy)
    return xy[0, :], xy[1, :]


def gen_xy_from_yaw(yaw, dt, v0):
    psiR = yaw
    psi0 = -pi / 2 + yaw[0]

    dxdt = v0 * sin(psi0 - psiR)
    dydt = v0 * cos(psi0 - psiR)

    x = np.cumsum(dt * dxdt)
    y = np.cumsum(dt * dydt)
    return x, y


def shift_and_scale(x1, y1, x2, y2):
    sx = np.mean(x2) - np.mean(x1)
    sy = np.mean(y2) - np.mean(y1)

    alpha_x = (np.max(x2) - np.min(x2)) / (np.max(x1) - np.min(x1))
    alpha_y = (np.max(y2) - np.min(y2)) / (np.max(y1) - np.min(y1))

    x2 = (x2 - sx) / alpha_x
    y2 = (y2 - sy) / alpha_y
    return x2, y2
