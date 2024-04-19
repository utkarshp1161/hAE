#@title
# DKL specific functions
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
def plot_dkl(full_img, window_size, y,
             dkl_mean, dkl_var):

    plt.figure(figsize=(15,15))
    gs = gridspec.GridSpec(1,2,width_ratios = [1,1])
    ax01,ax02 = plt.subplot(gs[0,0]), plt.subplot(gs[0,1])

    s1, s2 = full_img.shape[0] - window_size+1, full_img.shape[1] - window_size+1#48,63

    if y.ndim == 1:   # Scalar DKL
        #original    = y.reshape(s1,s2)#(48, 63)- 3024 = 48*63
        predicted   = dkl_mean.reshape(s1,s2)# (48,63)
        uncertainty = dkl_var.reshape(s1,s2)# (48,63)

    elif y.ndim == 2: # Vector DKL
        #original    = np.swapaxes(y.reshape(-1,s1,s2).T, 0, 1)
        predicted   = np.swapaxes(dkl_mean.reshape(-1,s1,s2).T, 0, 1)
        uncertainty = np.swapaxes(dkl_var.reshape(-1,s1,s2).T, 0, 1)

    #ax00.imshow(original)
    ax01.imshow(predicted)
    ax02.imshow(uncertainty)

    #ax00.set_title("original")
    ax01.set_title("mean (prediction)")
    ax02.set_title("variance (uncertainty)")

    # if y.ndim == 2: # we can show individual spectra. Otherwise, we've already scalarized it

    #     rng = np.random.default_rng()
    #     xx_ = rng.choice(s1, 3, replace = False)
    #     yy_ = rng.choice(s2, 3, replace = False)
    #     XY = np.asarray([xx_,yy_]).T

    #     x1,y1 = XY[0]
    #     x2,y2 = XY[1]
    #     x3,y3 = XY[2]

    #     ax00.scatter(x1,y1, c='dodgerblue', ec='k', s=100)
    #     ax00.scatter(x2,y2, c='darkorange', ec='k', s=100)
    #     ax00.scatter(x3,y3, c='crimson', ec='k', s=100)

    #     plt.figure(figsize=(10,5))
    #     ax = plt.gca()

    #     ax.plot(original[y1,x1], c='dodgerblue', lw = 2, ls ='--')
    #     ax.plot(original[y2,x2], c='darkorange', lw = 2, ls ='--')
    #     ax.plot(original[y3,x3], c='crimson', lw = 2, ls ='--')

    #     ax.plot(predicted[y1,x1], c='dodgerblue', lw = 2)
    #     ax.plot(predicted[y2,x2], c='darkorange', lw = 2)
    #     ax.plot(predicted[y3,x3], c='crimson', lw = 2)

    #     ax.axhline(0, c='k')
    #     ax.set_xlabel("Energy loss (eV)", fontsize = 22)
    #     ax.tick_params(length=5, width=3, labelsize = 18)


# Plot and save the image with highlighted patch
def plot_highlighted_patch_during_seed(rf, full_img, features, spectrum, indices, next_point_idx , window_size, step, seed_or_dkl = 'seed'):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    y, x = indices[next_point_idx]
    ax1.plot(spectrum)
    ax2.imshow(features[next_point_idx].reshape(-1, window_size), cmap='gray')# dkl has flattened the shape
    ax3.imshow(full_img, cmap='gray')
    ax3.add_patch(Rectangle([indices[next_point_idx][1] - window_size / 2, indices[next_point_idx][0] - window_size / 2], window_size, window_size, fill=False, ec='r'))
    plt.savefig(f"{rf}{seed_or_dkl}_{step}.png")
    plt.close()

def plot_highlighted_patch_during_dkl(full_img, features, spectrum, indices, next_point_idx , window_size, step, seed_or_dkl = 'seed'):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    y, x = indices[next_point_idx]
    ax1.plot(spectrum)
    ax2.imshow(features[next_point_idx].reshape(-1, window_size), cmap='gray')# dkl has flattened the shape
    ax3.imshow(full_img, cmap='gray')
    ax3.add_patch(Rectangle([indices[next_point_idx][1] - window_size / 2, indices[next_point_idx][0] - window_size / 2], window_size, window_size, fill=False, ec='r'))
    plt.savefig(f"{seed_or_dkl}_{step}.png")
    plt.close()


















