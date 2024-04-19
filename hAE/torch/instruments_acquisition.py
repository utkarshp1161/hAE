import numpy as np
import Pyro5.api


def spectrum_calc(x, y):
    uri = "PYRO:array.server@10.46.217.242:9093"
    array_server = Pyro5.api.Proxy(uri)
    array_server.set_beam_pos(x, y)
    array_server.acquire_camera()
    array_list, shape, dtype = array_server.get_eels()
    array = np.array(array_list, dtype=dtype).reshape(shape)
    return array

def spectrum_calc_dummy(x, y):
    uri = "PYRO:array.server@10.46.217.242:9092"
    array_server = Pyro5.api.Proxy(uri)
    array_list, shape, dtype = array_server.get_eels(dummy = True)
    array = np.array(array_list, dtype=dtype).reshape(shape)
    return array

def spectrum_calc_grid(specim,x, y):
    print("microscope spectroscopy called at x,y", x, y)
    spectrum = specim[y, x, :]
    return spectrum


def spectrum_calc_grid_no_specim(x, y):
    print("microscope spectroscopy called at x,y", x, y)
    # if x and y in 1st quadrant of 32*32
    if 15< x < 32 and 15 < y < 32:
        spectrum = np.ones(32)
    else:
        spectrum = np.zeros(32)
    return spectrum

import numpy as np
import matplotlib.pyplot as plt

def detect_bright_region(image):
    # Calculate the gradient in the X and Y directions
    gx = np.gradient(image, axis=1)  # Gradient in X direction
    gy = np.gradient(image, axis=0)  # Gradient in Y direction
    g = np.sqrt(gx**2 + gy**2)
    g_normalized = (g / g.max()) * 255
    spots = g_normalized.astype(np.uint8)
    threhold_eels = spots.mean()*4
    spots[spots<threhold_eels]=0
    spots[spots>0]= 1.0
    
    return spots



