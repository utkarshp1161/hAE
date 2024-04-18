import os
import glob
import h5py
from copy import deepcopy as dc

from scipy import ndimage
from sklearn import decomposition

import pyroved as pv

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from atomai.utils import get_coord_grid, extract_patches_and_spectra

import cv2

import atomai as aoi
from atomai import utils

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import logging
from datetime import datetime
import yaml
import argparse
import pyTEMlib.file_tools as ft

# Initialize logging
def init_logging(save_dir, config):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"PF_Logging to {save_dir}log_{current_time}.log")
    filename = f"{save_dir}PF_log_{current_time}.log"
    logging.basicConfig(filename=filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    # Log the configuration values
    logging.info("Configuration values:")
    for section, settings in config.items():
        logging.info(f"Section: {section}")
        for key, value in settings.items():
            logging.info(f"{key}: {value}")
    
    
# Load configuration settings
def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
      
# Load and preprocess data
def load_and_preprocess_data(NPs, key):
    specim = NPs[f'{key}']['spectrum image']
    eax = NPs[f'{key}']['energy axis']
    pxsizenm = NPs[f'{key}']['scale']
    img = NPs[f'{key}']['image']
    return specim, eax, pxsizenm, img
  
# Extract patches and spectra
def extract_data(specim, img, coordinates, window_size, spectralavg):
    return extract_patches_and_spectra(specim, img, coordinates=coordinates,
                                       window_size=window_size, avg_pool=spectralavg)
      
def normalize_data(features, targets):
    norm_ = lambda x: (x - x.min()) / x.ptp()
    return norm_(features), norm_(targets)

# Plot and save the image with highlighted patch
def plot_highlighted_patch(full_img, features, targets, indices, k, window_size, save_dir):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    ax1.plot(targets[k])
    ax2.imshow(features[k], cmap='gray')
    ax3.imshow(full_img, cmap='gray')
    ax3.add_patch(Rectangle([indices[k][1] - window_size / 2, indices[k][0] - window_size / 2], window_size, window_size, fill=False, ec='r'))
    plt.savefig(save_dir + "2_portion_f_img.png")
    plt.close()

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process the path to the config file.')
    parser.add_argument('--config', type=str, required=True, help='Path to the config file.')
    return parser.parse_args()
# Main function



args = parse_arguments()
config_path = args.config
config = load_config(config_path)

window_size = config['settings']['window_size']
ws = window_size
spectralavg = config['settings']['spectralavg']
save_dir = config['settings']['save_dir']
data_path = config['settings']['data_path']

dataset = ft.open_file(data_path)
image = dataset['Channel_000']
spectra = dataset['Channel_002']
image_for_dkl = dataset['Channel_001']
specim = np.array(spectra)
img = np.array(image_for_dkl).T

init_logging(save_dir=save_dir, config=config)
logging.info(f"Directory {save_dir} already exists.")

#data_path = "/hAE_data/Plasmonic_sets_7222021_fixed.npy"
cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'None')
logging.info(f"CUDA Visible Devices: {cuda_visible_devices}")

full_img = img
coordinates = get_coord_grid(img, step=1, return_dict=False)
patches, spectra, coords = extract_patches_and_spectra(specim, img, coordinates=coordinates,
                                    window_size=window_size, avg_pool=spectralavg)

full_img    = np.copy(img) # (55, 70)
indices     = np.copy(coords)# (3024, 2)
features    = np.copy(patches)# (3024, 8, 8)
targets     = np.copy(spectra)# (3024, 439)
n, d1, d2 = features.shape # (3024, 8, 8)
X = features.reshape(n, d1*d2) # (3024, 64)
logging.info("learning curve analysis--------------------------------------------------")
logging.info("Plot preditive uncertainty (of each model) as a function of automated experiment step. Shown is the average uncertainty for all patches, and the distribution of uncertainties.")
rf = save_dir + "torch/"
exploration_steps = len(glob.glob(save_dir + "/torch/EI/random_band1/record*")) # This should be 20
acq_funcs = ['EI', 'MU', 'UCB'] 

# Create array to save results of live and complete DKL models
# save results from EI acuqisition funciton

var_compare = np.empty((3, 3, exploration_steps))  # uncertainty
var_dev_compare = np.empty((3, 3, exploration_steps)) # uncertainty variation
nextpoint_uncertainty_compare = np.empty((3, 3, exploration_steps)) # predicted next point uncertainty
nextpoint_measure_compare = np.empty((3, 3, exploration_steps)) # next point ground truth
nextpoint_pred_compare = np.empty((3, 3, exploration_steps)) # next point prediction

for acq_idx in range(3):
    os.chdir (rf + acq_funcs[acq_idx] + '/mature_full_record')#********************************************

    for i in range (exploration_steps):
        # load results of DKL
        rec = np.load("record{}.npz".format(i))
        var_compare[acq_idx, :, i] = rec['var'].mean(), rec['var0'].mean(), rec['var1'].mean()# mean of variance
        var_dev_compare[acq_idx, :, i] = np.std(rec['var']), np.std(rec['var0']), np.std(rec['var1'])

        nextpoint_uncertainty_compare[acq_idx, :, i] = rec['var'].max(), rec['var0'].max(), rec['var1'].max()# max of variance
        nextpoint_measure_compare[acq_idx, :, i] = rec['y_test'][rec['var'].argmax()], rec['y_test'][rec['var0'].argmax()], rec['y_test'][rec['var1'].argmax()]
        nextpoint_pred_compare[acq_idx, :, i] = rec['mean'][rec['var'].argmax()], rec['mean0'][rec['var0'].argmax()], rec['mean1'][rec['var1'].argmax()]
        
s = np.arange(exploration_steps) #step

fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (18, 4), dpi = 200)

logging.info("EI acquisition function")
#Plot EI
ax1.set_title("Acquisition function EI")
ax1.plot(s, var_compare[0, 0,], c= 'black', label = "Live")
ax1.fill_between(s, var_compare[0, 0,]-var_dev_compare[0, 0,],
                var_compare[0, 0,]+var_dev_compare[0, 0,], color='black', alpha=0.2)
ax1.plot(s, var_compare[0, 1,], c = 'blue', label = "Final")
ax1.fill_between(s, var_compare[0, 1,]-var_dev_compare[0, 1,],
                var_compare[0, 1,]+var_dev_compare[0, 1,],  color='blue', alpha=0.2)
ax1.plot(s, var_compare[0, 2,], c = 'orange', label = "Complete")
ax1.fill_between(s, var_compare[0, 2,]-var_dev_compare[0, 2,],
                var_compare[0, 2,]+var_dev_compare[0, 2,], color='orange', alpha=0.2)
ax1.set_ylabel("DKL Uncertainty")
ax1.set_xlabel('Step')
ax1.legend()


logging.info("MU acquisition function")
#Plot MU
ax2.set_title("Acquisition function MU")
ax2.plot(s, var_compare[1, 0,], c= 'black', label = "Live")
ax2.fill_between(s, var_compare[1, 0,]-var_dev_compare[1, 0,],
                var_compare[1, 0,]+var_dev_compare[1, 0,], color='black', alpha=0.2)
ax2.plot(s, var_compare[1, 1,], c = 'blue', label = "Final")
ax2.fill_between(s, var_compare[1, 1,]-var_dev_compare[1, 1,],
                var_compare[1, 1,]+var_dev_compare[1, 1,],  color='blue', alpha=0.2)
ax2.plot(s, var_compare[1, 2,], c = 'orange', label = "Complete")
ax2.fill_between(s, var_compare[1, 2,]-var_dev_compare[1, 2,],
                var_compare[1, 2,]+var_dev_compare[1, 2,], color='orange', alpha=0.2)
ax2.set_ylabel("DKL Uncertainty")
ax2.set_xlabel('Step')
ax2.legend()


logging.info("UCB acquisition function")
#Plot UCB
ax3.set_title("Acquisition function UCB")
ax3.plot(s, var_compare[2, 0,], c= 'black', label = "Live")
ax3.fill_between(s, var_compare[2, 0,]-var_dev_compare[2, 0,],
                var_compare[2, 0,]+var_dev_compare[2, 0,], color='black', alpha=0.2)
ax3.plot(s, var_compare[2, 1,], c = 'blue', label = "Final")
ax3.fill_between(s, var_compare[2, 1,]-var_dev_compare[2, 1,],
                var_compare[2, 1,]+var_dev_compare[2, 1,],  color='blue', alpha=0.2)
ax3.plot(s, var_compare[2, 2,], c = 'orange', label = "Complete")
ax3.fill_between(s, var_compare[2, 2,]-var_dev_compare[2, 2,],
                var_compare[2, 2,]+var_dev_compare[2, 2,], color='orange', alpha=0.2)
ax3.set_ylabel("DKL Uncertainty")
ax3.set_xlabel('Step')
ax3.legend()
plt.savefig(save_dir + "pf_1_learning_curve.png")

plt.close()

logging.info("Now, let's plot the maximum uncertainty for each model as a function of step. In principle, we can also calculate the acquisition functions.")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 4), dpi = 200)

# Plot EI
ax1.set_title("Acquisition function EI")
ax1.plot(s, nextpoint_uncertainty_compare[0, 0,], c= 'black', label = "Live")
ax1.plot(s, nextpoint_uncertainty_compare[0, 1,], c = 'blue', label = "Final")
ax1.plot(s, nextpoint_uncertainty_compare[0, 2,], c = 'orange', label = "Complete")
ax1.set_ylabel("DKL Uncertainty"); ax1.set_xlabel('Step'); ax1.legend()

# Plot MU
ax2.set_title("Acquisition function MU")
ax2.plot(s, nextpoint_uncertainty_compare[1, 0,], c= 'black', label = "Live")
ax2.plot(s, nextpoint_uncertainty_compare[1, 1,], c = 'blue', label = "Final")
ax2.plot(s, nextpoint_uncertainty_compare[1, 2,], c = 'orange', label = "Complete")
ax2.set_ylabel("DKL Uncertainty"); ax2.set_xlabel('Step'); ax2.legend()

# Plot UCB
ax3.set_title("Acquisition function UCB")
ax3.plot(s, nextpoint_uncertainty_compare[2, 0,], c= 'black', label = "Live")
ax3.plot(s, nextpoint_uncertainty_compare[2, 1,], c = 'blue', label = "Final")
ax3.plot(s, nextpoint_uncertainty_compare[2, 2,], c = 'orange', label = "Complete")
ax3.set_ylabel("DKL Uncertainty"); ax3.set_xlabel('Step'); ax3.legend()

plt.savefig(save_dir + "pf_2.png")
plt.close()

logging.info("Next measurement value from each model as a function of step. For now, the next point is simply defined as maximum uncertainty (we can also use EI or UCB function; or next point from experiment)")



fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 4), dpi = 200)

# Plot EI
ax1.set_title("Acquisition function EI")
ax1.plot(s, nextpoint_measure_compare[0, 0,], c= 'black', label = "Live")
ax1.plot(s, nextpoint_measure_compare[0, 1,], c = 'blue', label = "Final")
ax1.plot(s, nextpoint_measure_compare[0, 2,], c = 'orange', label = "Complete")
ax1.set_ylabel("Next point ground truth"); ax1.set_xlabel('Step'); ax1.legend()

# Plot MU
ax2.set_title("Acquisition function MU")
ax2.plot(s, nextpoint_measure_compare[1, 0,], c= 'black', label = "Live")
ax2.plot(s, nextpoint_measure_compare[1, 1,], c = 'blue', label = "Final")
ax2.plot(s, nextpoint_measure_compare[1, 2,], c = 'orange', label = "Complete")
ax2.set_ylabel("Next point ground truth"); ax2.set_xlabel('Step'); ax2.legend()

# Plot UCB
ax3.set_title("Acquisition function UCB")
ax3.plot(s, nextpoint_measure_compare[2, 0,], c= 'black', label = "Live")
ax3.plot(s, nextpoint_measure_compare[2, 1,], c = 'blue', label = "Final")
ax3.plot(s, nextpoint_measure_compare[2, 2,], c = 'orange', label = "Complete")
ax3.set_ylabel("Next point ground truth"); ax3.set_xlabel('Step'); ax3.legend()

plt.savefig(save_dir + "pf_3.png")
plt.close()

logging.info("Next point prediction from each model as a function of step. For now, the next point is simply defined as maximum uncertainty (we can use EI or UCB function, or next point from experiment)")

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (18, 4), dpi = 200)

# Plot EI
ax1.set_title("Acquisition function EI")
ax1.plot(s, nextpoint_pred_compare[0, 0,], c= 'black', label = "Live")
ax1.plot(s, nextpoint_pred_compare[0, 1,], c = 'blue', label = "Final")
ax1.plot(s, nextpoint_pred_compare[0, 2,], c = 'orange', label = "Complete")
ax1.set_ylabel("Next point prediction"); ax1.set_xlabel('Step'); ax1.legend()

# Plot MU
ax2.set_title("Acquisition function MU")
ax2.plot(s, nextpoint_pred_compare[1, 0,], c= 'black', label = "Live")
ax2.plot(s, nextpoint_pred_compare[1, 1,], c = 'blue', label = "Final")
ax2.plot(s, nextpoint_pred_compare[1, 2,], c = 'orange', label = "Complete")
ax2.set_ylabel("Next point prediction"); ax2.set_xlabel('Step'); ax2.legend()

# Plot UCB
ax3.set_title("Acquisition function UCB")
ax3.plot(s, nextpoint_pred_compare[2, 0,], c= 'black', label = "Live")
ax3.plot(s, nextpoint_pred_compare[2, 1,], c = 'blue', label = "Final")
ax3.plot(s, nextpoint_pred_compare[2, 2,], c = 'orange', label = "Complete")
ax3.set_ylabel("Next point prediction"); ax3.set_xlabel('Step'); ax3.legend()

plt.savefig(save_dir + "pf_4.png")
plt.close()

logging.info("learning curve analysis over--------------------------------------------------")

logging.info("regret analysis--------------------------------------------------")

save_mature_full = '/mature_full_record'

# Create array to save results of real time and matured DKL models
# save results from three acuqisition funciton
reg = np.empty((3, 2, exploration_steps))  # save the difference of predictions between real time DKL and trained DKL at each step; as well as deviation

for acq_idx in range (3):
    os.chdir (rf + acq_funcs[acq_idx] + save_mature_full)
    for i in range (exploration_steps):
        # load results
        rec = np.load("record{}.npz".format(i))
        mean_d = rec['mean']-rec['mean0']
        reg[acq_idx,:,i] = mean_d.mean(), np.std(mean_d)

fig, ax = plt.subplots(figsize = (6, 4), dpi = 200)

ax.plot(s, reg[0,0,], c= 'black', label = "EI")
ax.fill_between(s, reg[0,0,]-reg[0,1,], reg[0,0,]+reg[0,1,],
                color='black', alpha=0.2)

ax.plot(s, reg[1,0,], c= 'blue', label = "MU")
ax.fill_between(s, reg[1,0,]-reg[1,1,], reg[1,1,]+reg[1,1,],
                color='blue', alpha=0.2)

ax.plot(s, reg[2,0,], c= 'orange', label = "UCB")
ax.fill_between(s, reg[2,0,]-reg[2,1,], reg[2,0,]+reg[2,1,],
                color='orange', alpha=0.2)

ax.set_ylabel("Regret")
ax.set_xlabel('Step')
plt.legend()
plt.savefig(save_dir + "pf_5.png")
plt.close()

logging.info("regret analysis over--------------------------------------------------")
logging.info("real space trajectory analysis--------------------------------------------------")
save_explore = '/explore_dkl_record'

traj_EI = np.zeros((exploration_steps, 2))
traj_MU = np.zeros((exploration_steps, 2))
traj_UCB = np.zeros((exploration_steps, 2))

for i in range (exploration_steps):
    os.chdir (rf + acq_funcs[0] + save_explore)
    rec = np.load("record{}.npz".format(i))
    nextpoint = rec ['next_point']
    traj_EI [i, ] = nextpoint

    try:
        os.chdir (rf + acq_funcs[1] + save_explore)
        rec = np.load("record{}.npz".format(i))
        nextpoint = rec ['next_point']
        traj_MU [i, ] = nextpoint
    except FileNotFoundError:
        pass

    os.chdir (rf + acq_funcs[2] + save_explore)
    rec = np.load("record{}.npz".format(i))
    nextpoint = rec ['next_point']
    traj_UCB [i, ] = nextpoint

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi = 300)

cm = 'gray'
shrink = 0.7

ax1.imshow(full_img, interpolation='nearest', origin = "lower", cmap=cm)
im1 = ax1.scatter(traj_EI[:,1], traj_EI[:,0], c = np.arange(exploration_steps), cmap = 'bwr')
fig.colorbar(im1, ax=ax1, shrink = shrink, label = "Step")
ax1.set_title("DKL-EI trajectory")
ax1.axis('off')

ax2.imshow(full_img, interpolation='nearest', origin = "lower", cmap=cm)
im2 = ax2.scatter(traj_MU[:,1], traj_MU[:,0], c = np.arange(exploration_steps), cmap = 'bwr')
fig.colorbar(im2, ax=ax2, shrink = shrink, label = "Step")
ax2.set_title("DKL-MU trajectory")
ax2.axis('off')

ax3.imshow(full_img, interpolation='nearest', origin = "lower", cmap=cm)
im3 = ax3.scatter(traj_UCB[:,1], traj_UCB[:,0], c = np.arange(exploration_steps), cmap = 'bwr')
fig.colorbar(im3, ax=ax3, shrink = shrink, label = "Step")
ax3.set_title("DKL-UCB trajectory")
ax3.axis('off')

plt.savefig(save_dir + "pf_6.png")
plt.close()

logging.info("real space trajectory analysis over--------------------------------------------------")

logging.info("feature discovery--------------------------------------------------")
logging.info("Here, we explore what are the characteristic structural elements of the points explored during the AE. To do it, we use the VAE (or invariant VAE) analysis of the image patches in the **experimental trace**.")

logging.info("EI acquisition function")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Device: {device}")

# Load DKL sampled data
os.chdir(rf + acq_funcs[0] + save_explore)
rec = np.load("final_traindata.npz")
X_train = rec['X_train']

# Initialize VAE train loader
vae_EI_data = torch.tensor(X_train.reshape(-1, ws, ws)).float()
vae_EI_loader = pv.utils.init_dataloader(vae_EI_data, batch_size=32)

np.random.seed(1)
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for ax in axes.flat:
    i = np.random.randint(len(vae_EI_data))
    ax.imshow(vae_EI_data[i].cpu(), interpolation='nearest')

plt.savefig(save_dir + "pf_7.png")
plt.close()

tensor = vae_EI_data

logging.info("Tensor properties:")
logging.info("Data type:", tensor.dtype)
logging.info("Device:", tensor.device)
logging.info("Shape:", tensor.shape)
logging.info("Number of dimensions:", tensor.dim())
logging.info("Total number of elements:", tensor.numel())

# train VAE using the constructed stack of subimages
logging.info("Training VAE using the constructed stack of subimages")
logging.info("EI acquisition function")
in_dim = (ws, ws)
rvae_EI = pv.models.iVAE(in_dim, latent_dim=2, invariances='r',
                    sampler_d='gaussian', decoder_sig=0.01, seed=42)

# Initialize SVI trainer
trainer = pv.trainers.SVItrainer(rvae_EI)

#####
# was having issue in logging loss: edited pyroved:
#vi /nfs/home/upratius/.conda/envs/utk_new/lib/python3.10/site-packages/pyroved/trainers/svi.py
# to last line :--> print_statistics --> return the output as well--> so got he strin gto log
# Train for n epochs:
for e in range(200):
    trainer.step(vae_EI_loader, scale_factor=1)
    output = trainer.print_statistics()
    logging.info(output)
    

z_mean_EI, z_sd_EI = rvae_EI.encode(vae_EI_data)

plt.figure(figsize=(5, 5), dpi = 100)
plt.scatter(z_mean_EI[:, -1], z_mean_EI[:, -2], s=10, c='blue')
plt.xlabel("$z_2$", fontsize=14)
plt.ylabel("$z_1$", fontsize=14)
plt.savefig(save_dir + "pf_8.png")
plt.close()

rvae_EI.manifold2d(6,
                # z_coord = [z_mean_EI[:, -1].min(), z_mean_EI[:, -1].max(), z_mean_EI[:, -2].min(), z_mean_EI[:, -2].max()],
                cmap = 'plasma', dpi = 100);

plt.savefig(save_dir + "pf_9.png")
plt.close()

logging.info("MU acquisition function")

# Load DKL sampled data
os.chdir(rf + acq_funcs[1] + save_explore)
rec = np.load("final_traindata.npz")
X_train = rec['X_train']

# Initialize VAE train loader
vae_MU_data = torch.tensor(X_train.reshape(-1, ws, ws)).float()
vae_MU_loader = pv.utils.init_dataloader(vae_MU_data, batch_size=64)

np.random.seed(1)
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for ax in axes.flat:
    i = np.random.randint(len(vae_MU_data))
    ax.imshow(vae_MU_data[i].cpu(), interpolation='nearest')

plt.savefig(save_dir + "pf_10.png")
plt.close()


# train VAE using the constructed stack of subimages
in_dim = (ws, ws)
rvae_MU = pv.models.iVAE(in_dim, latent_dim=2, invariances='r',
                    sampler_d='gaussian', decoder_sig=0.01, seed=42)
# Initialize SVI trainer
trainer = pv.trainers.SVItrainer(rvae_MU)

# Train for n epochs:
for e in range(200):
    trainer.step(vae_MU_loader, scale_factor=1)
    output = trainer.print_statistics()
    logging.info(output)

z_mean_MU, z_sd_MU = rvae_MU.encode(vae_MU_data)

plt.figure(figsize=(5, 5), dpi = 100)
plt.scatter(z_mean_MU[:, -1], z_mean_MU[:, -2], s=10, c='blue')
plt.xlabel("$z_2$", fontsize=14)
plt.ylabel("$z_1$", fontsize=14)
plt.savefig(save_dir + "pf_11.png")
plt.close()

rvae_MU.manifold2d(6,
                # z_coord = [z_mean_MU[:, -1].min(), z_mean_MU[:, -1].max(), z_mean_MU[:, -2].min(), z_mean_MU[:, -2].max()],
                cmap = 'plasma', dpi = 100);

plt.savefig(save_dir + "pf_12.png")
plt.close()

logging.info("UCB acquisition function")
# Load DKL sampled data
os.chdir(rf + acq_funcs[2] + save_explore)
rec = np.load("final_traindata.npz")
X_train = rec['X_train']

# Initialize VAE train loader
vae_UCB_data = torch.tensor(X_train.reshape(-1, ws, ws)).float()
vae_UCB_loader = pv.utils.init_dataloader(vae_UCB_data, batch_size=64)

np.random.seed(1)
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for ax in axes.flat:
    i = np.random.randint(len(vae_UCB_data))
    ax.imshow(vae_UCB_data[i].cpu(), interpolation='nearest')
plt.savefig(save_dir + "pf_13.png")
plt.close()

# train VAE using the constructed stack of subimages
in_dim = (ws, ws)
rvae_UCB = pv.models.iVAE(in_dim, latent_dim=2, invariances='r',
                    sampler_d='gaussian', decoder_sig=0.01, seed=42)
# Initialize SVI trainer
trainer = pv.trainers.SVItrainer(rvae_UCB)

# Train for n epochs:
for e in range(200):
    trainer.step(vae_UCB_loader, scale_factor=1)
    output = trainer.print_statistics()
    logging.info(output)

z_mean_UCB, z_sd_UCB = rvae_MU.encode(vae_UCB_data)

plt.figure(figsize=(5, 5), dpi = 100)
plt.scatter(z_mean_UCB[:, -1], z_mean_UCB[:, -2], s=10, c='blue')
plt.xlabel("$z_2$", fontsize=14)
plt.ylabel("$z_1$", fontsize=14)
plt.savefig(save_dir + "pf_14.png")
plt.close()

rvae_UCB.manifold2d(6,
                    # z_coord = [z_mean_UCB[:, -1].min(), z_mean_UCB[:, -1].max(), z_mean_UCB[:, -2].min(), z_mean_UCB[:, -2].max()],
                    cmap = 'plasma', dpi = 100);
plt.savefig(save_dir + "pf_15.png")
plt.close()

logging.info("feature discovery over--------------------------------------------------")
logging.info("latent trajectory analysis--------------------------------------------------")
logging.info("Here, we analyze the feature space (i.e. latent space) of the full set of image patches, and plot the AE trajectories in this latent space.")

# Initialize VAE train loader
vae_all_data = torch.tensor(X.reshape(-1, ws, ws)).float()
vae_all_loader = pv.utils.init_dataloader(vae_all_data, batch_size=64)

np.random.seed(1)
fig, axes = plt.subplots(5, 5, figsize=(8, 8),
                        subplot_kw={'xticks':[], 'yticks':[]},
                        gridspec_kw=dict(hspace=0.1, wspace=0.1))
for ax in axes.flat:
    i = np.random.randint(len(vae_all_data))
    ax.imshow(vae_all_data[i].cpu(), interpolation='nearest')
    
plt.savefig(save_dir + "pf_16.png")
plt.close()

# train VAE using the constructed stack of subimages
in_dim = (ws, ws)
rvae_all = pv.models.iVAE(in_dim, latent_dim=2, invariances='r',
                    sampler_d='gaussian', decoder_sig=0.01, seed=42)
# Initialize SVI trainer
trainer = pv.trainers.SVItrainer(rvae_all)

# Train for n epochs:
for e in range(200):
    trainer.step(vae_all_loader, scale_factor=1)
    output = trainer.print_statistics() 
    logging.info(output)
    
z_mean_all, z_sd_all = rvae_all.encode(vae_all_data)

plt.figure(figsize=(5, 5), dpi = 100)
plt.title('all data')
plt.scatter(z_mean_all[:, -1], z_mean_all[:, -2], s=10, c='blue')
plt.xlabel("$z_2$", fontsize=14)
plt.ylabel("$z_1$", fontsize=14)
plt.savefig(save_dir + "pf_17.png")
plt.close()

rvae_all.manifold2d(20,
                    #z_coord = [z_mean_all[:, -1].min(), z_mean_all[:, -1].max(), z_mean_all[:, -2].min(), z_mean_all[:, -2].max()],
                    cmap = 'plasma', dpi = 300);
plt.savefig(save_dir + "pf_18.png")
plt.close()

indices_all = np.copy(indices)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5), dpi = 300)
ax1.imshow(full_img, origin = "lower", cmap = 'gray')
im1=ax1.scatter(indices_all[:,1], indices_all[:,0], s=1, c = z_mean_all[:,-2], cmap = 'jet')
ax1.axis(False)
cbar1 = fig.colorbar(im1, ax=ax1, shrink=.7)
cbar1.set_label("$z_1$", fontsize=16)

ax2.imshow(full_img, origin = "lower", cmap = 'gray')
im2=ax2.scatter(indices_all[:,1], indices_all[:,0], s=1, c = z_mean_all[:,-1], cmap = 'jet')
ax2.axis(False)
cbar2 = fig.colorbar(im2, ax=ax2, shrink=.7)
cbar2.set_label("$z_2$", fontsize=16)

im3=ax3.imshow(full_img, origin = "lower", cmap = 'gray')
im3=ax3.scatter(indices_all[:,1], indices_all[:,0], s=1, c = z_mean_all[:,-3], cmap = 'jet')
ax3.axis(False)
cbar3 = fig.colorbar(im3, ax=ax3, shrink=.7)
cbar3.set_label("$z_0$", fontsize=16)
plt.savefig(save_dir + "pf_19.png")
plt.close()

logging.info("Plot samplings (from different acquition functions) distribution in the latent space of all patches")
        
    
import pandas as pd
import seaborn as sns

z1 = np.asarray(z_mean_all, dtype = np.float32);
z2 = np.asarray(z_mean_EI, dtype = np.float32)
d = np.append(z1, z2, axis = 0)

s = pd.DataFrame({"z0": d[:,0], "z1": d[:,1], "z2": d[:,2]})

#, "Data": 'All Data'})
#s['Data'][len(z1):] = "DKL Train Data"

z_mean_all_EI, z_sd_all_EI = rvae_all.encode(vae_EI_data)
z_mean_all_MU, z_sd_all_MU = rvae_all.encode(vae_MU_data)
z_mean_all_UCB, z_sd_all_UCB = rvae_all.encode(vae_UCB_data)
# import pdb;--> did for saving vae model
# pdb.set_trace()
f, ax = plt.subplots(figsize = (6,6), dpi = 200)

data = z1
s = pd.DataFrame({"z0": data[:,0], "z1": data[:,1], "z2": data[:,2]})
ax = sns.kdeplot(data=s, x = "z2", y = "z1", fill=True, thresh=0.01, levels=100, cmap="Greens")
# ax.set_ylim(-5, 5)
# ax.set_xlim(-5, 5)

#ax.scatter(z_mean_all[:, -1], z_mean_all[:, -2], c = 'gray', s=0.5, label = "Full patches")
ax.scatter(z_mean_all_EI[:, -1], z_mean_all_EI[:, -2], c=np.arange(len(z_mean_all_EI)), marker='o', s=20, label="DKL-EI sampling", cmap='bwr')
ax.scatter(z_mean_all_MU[:, -1], z_mean_all_MU[:, -2], c=np.arange(len(z_mean_all_MU)), marker='*', s=20, label="DKL-MU sampling", cmap='bwr')
ax.scatter(z_mean_all_UCB[:, -1], z_mean_all_UCB[:, -2], c=np.arange(len(z_mean_all_UCB)), marker='^', s=20, label="DKL-UCB sampling", cmap='bwr')
ax.set_xlabel("$z_2$", fontsize=14)
ax.set_ylabel("$z_1$", fontsize=14)


ax.legend()
plt.savefig(save_dir + "pf_20.png")
plt.close()
logging.info("done")
