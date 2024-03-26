import numpy as np

def normalize(x):
    return (x-np.min(x))/np.ptp(x)

def calculate_peaks(targets, features, indices, e_axis, band):
    """
    args :
        targets : (num_patches, 439)
        features : (num_patches, patch_size, patch_size)
        indices : (num_patches, 2) - kind of where the patch is located in actual image 
        e_axis : (439,)
        band : list with 2 enteries - [lower_bound, upper_bound]
    Returns :  
    """
    e1, e2 = np.abs(band[0] - e_axis).argmin(), np.abs(band[1] - e_axis).argmin()# argmin returns the index of the min value
    """e1 and e2 are the indices of the lower and upper bounds of the band in the e_axis"""
    # look between e1 and e2 in the targets and find the max value --> 
    # peaks 1: 45 to 65 indices out of 439
    # peaks 2: 115 to 135 indices out of 439
    # peak 3: 155 to 175 indices out of 439
    peaks_all_scalar1, features_all, indices_all = [], [], []
    for i, t in enumerate(targets):# walking through all the patches and related spectrum
        peak = np.max(t[e1:e2])# for that patch, find the max intensity in the band with 
        peaks_all_scalar1.append(np.array([peak]))
        features_all.append(features[i])
        indices_all.append(indices[i])

    peaks_all_scalar1 = normalize(np.concatenate(peaks_all_scalar1))
    features_all = np.array(features_all)
    indices_all = np.array(indices_all)

    return peaks_all_scalar1, features_all, indices_all




