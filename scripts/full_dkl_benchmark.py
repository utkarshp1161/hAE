import os
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
import numpy as np
import matplotlib.pyplot as plt
import atomai as aoi
from atomai.utils import get_coord_grid, extract_patches_and_spectra
import logging
from datetime import datetime
import yaml
import argparse
import hAE.torch
from hAE.torch.scalarizer_eels import calculate_peaks#, calculate_peaks_and_fit_gaussians_with_lmfit, calculate_peaks_and_fit_gaussians_with_lmfit_integral
from  hAE.torch.plot_utils import plot_dkl
from  hAE.torch.all_dkl import dkl_explore
from  hAE.torch.all_dkl import dkl_counterfactual
from  hAE.torch.all_dkl import dkl_random
from  hAE.torch.all_dkl import dkl_mature_full



# Initialize logging
def init_logging(save_dir):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Logging to {save_dir}log_{current_time}.log")
    filename = f"{save_dir}log_{current_time}.log"
    logging.basicConfig(filename=filename, level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', 
                        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info("This is a test log message.")
    
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

def main():

  args = parse_arguments()
  config_path = args.config
  config = load_config(config_path)
  
  window_size = config['settings']['window_size']
  spectralavg = config['settings']['spectralavg']
  save_dir = config['settings']['save_dir']
  scalarizer_index = config['settings']['scalarizer_index']



  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
      print(f"Directory {save_dir} created.")

  else:
      print(f"Directory {save_dir} already exists.")
      
  init_logging(save_dir=save_dir)
  logging.info(f"Directory {save_dir} already exists.")

  data_path = "/hAE_data/Plasmonic_sets_7222021_fixed.npy"
  cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'None')
  logging.info(f"CUDA Visible Devices: {cuda_visible_devices}")
  NPs = np.load(data_path, allow_pickle=True).tolist()
  key = 3
  specim, eax, pxsizenm, img = load_and_preprocess_data(NPs, key)

  plt.imshow(img)
  plt.savefig(save_dir + "1_img.png")

  coordinates = get_coord_grid(img, step=1, return_dict=False)
  patches, spectra, coords = extract_data(specim, img, coordinates, window_size, spectralavg)
  # features, targets = normalize_data(features, targets)
  # ... [Rest of the code for DKL embedding and other processing] ...
  ws = window_size # For convenience later...


  # Not necessary, just placing into 'standard' nomenclature

  full_img    = np.copy(img) # (55, 70)
  indices     = np.copy(coords)# (3024, 2)
  features    = np.copy(patches)# (3024, 8, 8)
  targets     = np.copy(spectra)# (3024, 439)
  scale       = np.copy(pxsizenm) # array(5.12597961)

  e_axis      = np.copy(eax[::spectralavg])

  while e_axis.shape[0] > spectra[0].shape[0]:
      e_axis = e_axis[:-1] # # (439,)
  
  k = 350 # randomly select one of the 3024 patches
  
  plot_highlighted_patch(full_img, features, targets, indices, k, ws, save_dir)
  

  logging.info("Reached scalarizer----------------------------------")
  #------------------------------------------------
  #sacalrizer



  band = [0.3, 0.4]
  peaks_all_scalar1, features_all, indices_all = calculate_peaks(targets, features, indices, e_axis, band)
  #[(3024,), (3024, 8, 8), (3024, 2)]
  plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar1, s = 90)
  plt.savefig(save_dir + "3_peaks_all_scalar1.png")
  plt.close()


  ## Edge mode
  band = [0.65, 0.75]
  peaks_all_scalar2, features_all, indices_all = calculate_peaks(targets, features, indices, e_axis, band)
  plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar2, s = 90)
  plt.savefig(save_dir + "4_peaks_all_scalar2.png")
  plt.close()


  ## Bulk / face mode
  band = [0.85, 0.95]
  peaks_all_scalar3, features_all, indices_all = calculate_peaks(targets, features, indices, e_axis, band)
  plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar3, s = 90)
  plt.savefig(save_dir + "5_peaks_all_scalar3.png")
  plt.close()


  # plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar3, s = 90)

  dklgp = None  # Reset any model, may help clear memory
  
  if scalarizer_index == 1:
    y = np.copy(peaks_all_scalar1)
    logging.info("scalarizer_index 1")
  elif scalarizer_index == 2:
    y = np.copy(peaks_all_scalar2)
    logging.info("scalarizer_index 2")
  elif scalarizer_index == 3:
    y = np.copy(peaks_all_scalar3)
    logging.info("scalarizer_index 3")
  else: 
    print("Invalid scalarizer index")
    logging.info("Invalid scalarizer index")
    return
  n, d1, d2 = features.shape # (3024, 8, 8)
  X = features.reshape(n, d1*d2) # (3024, 64)



  logging.info("Reached dkl full data----------------------------------")
  #---------------------------------------------------------------------------------------
  # dkl on full data

  ## 2d embediing
  data_dim = X.shape[-1] # 3024 patches

  dklgp = aoi.models.dklGPR(data_dim, embedim=2, precision="double")
  dklgp.fit(X, y, training_cycles=200, lr=1e-2) # cycles 200





  mean, var = dklgp.predict(X, batch_size=len(X))
  plot_dkl(img, window_size, y, mean, var)
  plt.savefig(save_dir+ "6_dkl2d_full_data.png")
  plt.close()
  """window_size = 8, [i.shape for i in [full_img, y, dkl_mean, dkl_var]]
  [(55, 70), (3024,), (3024,), (3024,)]"""

  s1,s2 = img.shape[0] - window_size + 1, img.shape[1] - window_size + 1

  embeded = dklgp.embed(X)# (3024, 2) -- its just the output of the encoder
  embeded = embeded / embeded.max()

  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

  cm = 'RdBu'
  shrink = 0.7

  im1 = ax1.imshow(embeded[:,0].reshape(s1,s2), interpolation='nearest', origin = "lower", cmap=cm)
  fig.colorbar(im1, ax=ax1, shrink = shrink)
  ax1.set_title("Embedded 1")
  ax1.axis('off')

  im2 = ax2.imshow(embeded[:,1].reshape(s1,s2), interpolation='nearest', origin = "lower", cmap=cm)
  fig.colorbar(im2, ax=ax2, shrink = shrink)
  ax2.set_title("Embedded 2")
  ax2.axis('off')
  plt.savefig(save_dir + "7_dkl2d_full_data_embeded.png")
  plt.close()




  fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi = 300)
  ax1.imshow(img, origin = "lower", cmap = 'gray')
  im1=ax1.scatter(indices_all[:,1], indices_all[:,0], s=1, c = embeded[:,0].reshape(s1, s2), cmap = 'jet')
  ax1.axis(False)
  cbar1 = fig.colorbar(im1, ax=ax1, shrink=.7)
  cbar1.set_label("$z_1$", fontsize=16)

  ax2.imshow(img, origin = "lower", cmap = 'gray')
  im2=ax2.scatter(indices_all[:,1], indices_all[:,0], s=1, c = embeded[:,1].reshape(s1, s2), cmap = 'jet')
  ax2.axis(False)
  cbar2 = fig.colorbar(im2, ax=ax2, shrink=.7)
  cbar2.set_label("$z_2$", fontsize=16)
  plt.savefig(save_dir + "8_dkl2d_full_data_embeded_scatter.png")
  plt.close()




  # 3D embedding

  data_dim = X.shape[-1]

  dklgp4 = aoi.models.dklGPR(data_dim, embedim=3, precision="double")
  dklgp4.fit(X, y, training_cycles=200, lr=1e-2, batch_size = 3)# batch_size default is 32

  mean4, var4 = dklgp4.predict(X, batch_size=len(X)) # (3024,), (3024,)
  plot_dkl(img, window_size, y, mean4, var4)
  plt.savefig(save_dir + "9_dkl3d_full_data.png")
  plt.close()

  s1,s2 = img.shape[0] - window_size + 1, img.shape[1] - window_size + 1# 48, 63

  embeded4 = dklgp4.embed(X)# (3024, 3)
  embeded4 = embeded4 / embeded4.max()

  fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

  cm = 'RdBu'
  shrink = 0.7

  im1 = ax1.imshow(embeded4[:,0].reshape(s1,s2), interpolation='nearest', origin = "lower", cmap=cm)
  fig.colorbar(im1, ax=ax1, shrink = shrink)
  ax1.set_title("Embedded 1")
  ax1.axis('off')

  im2 = ax2.imshow(embeded4[:,1].reshape(s1,s2), interpolation='nearest', origin = "lower", cmap=cm)
  fig.colorbar(im2, ax=ax2, shrink = shrink)
  ax2.set_title("Embedded 2")
  ax2.axis('off')

  im3 = ax3.imshow(embeded4[:,2].reshape(s1,s2), interpolation='nearest', origin = "lower", cmap=cm)
  fig.colorbar(im3, ax=ax3, shrink = shrink)
  ax3.set_title("Embedded 3")
  ax3.axis('off')
  plt.savefig(save_dir + "10_dkl3d_full_data_embeded.png")
  plt.close()


  logging.info("Reached active exp----------------------------------")
  #--------------------------------------------------------------------------------------------------
  # active exp
  ## Some globals


  exploration_steps = config['settings']['exploration_steps']
  num_cycles        = config['settings']['num_cycles']

  xi = config['settings']['xi'] # balances exploration and explitation for EI and qEI
  cm = 'viridis'
  shrink = 0.7

  ts = 0.995  # (0.9995 for key 3) test data size: # Adjust ts to be as small as possible without running into error
  rs = 42   #random state determines initial train data, 42 because answer to everything

  acq = 3  # 0 uses EI, 1 explores max uncertainty point, 2 uses UCB

  rf = '/nfs/home/upratius/scratch/projects/edited_AutomatedExperiment_Summer2023/utkarsh_eels/' + save_dir  #root folder
  if not os.path.exists(rf):
      os.mkdir(rf)

  acq_funcs = ['EI', 'MU', 'UCB'] 

  if acq == 3:             # when acq = 3, run all acq_funcs
    if not os.path.exists(rf + acq_funcs[0]):
      os.mkdir(rf + acq_funcs[0])
    if not os.path.exists(rf + acq_funcs[1]):
      os.mkdir(rf + acq_funcs[1])
    if not os.path.exists(rf + acq_funcs[2]):
      os.mkdir(rf + acq_funcs[2])
  elif acq < 3:
    if not os.path.exists(rf + acq_funcs[acq]):  # change root folder name when you change acquistion funciton
      os.mkdir(rf + acq_funcs[acq])
      

  logging.info("Reached dkl explore----------------------------------")
  #--------------------------------------------------------------------------------------
  # dkl explore

  ## band1


  # importlib.reload(plot_utils)

  save_explore =  "/explore_dkl_record/"
  logging.info("Reached dkl explore band 1----------------------------------")
  # on band 1

  y_targ = peaks_all_scalar1    #training output;-- 3024
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min()) # bring b/w 0 and 1
  y = y_targ.reshape(-1)# 

  # run DKL exploration with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_explore(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps, acq, acq_idx, ws,
                img, window_size, xi, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_explore(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_explore, num_cycles)
      print("==============acq_index")
      logging.info("acq_index")
      
  # plot hist(y)
  # plot_hist(

  logging.info("Reached dkl counterfactual----------------------------------")
  #---------------------------------------------------------------------------------------------
  #dkl counterfactual



  ##band 2
  logging.info("Reached dkl counterfactual band 2----------------------------------")

  # Set band 2 as target property

  y_targ = peaks_all_scalar2    #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_counter1 = "/counter_band2/"

  # run DKL counterfactual with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_counterfactual(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_counter1, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_counterfactual(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_counter1, save_explore, num_cycles)
      
  ## band 3
  logging.info("Reached dkl counterfactual band 3----------------------------------")
  y_targ = peaks_all_scalar3    #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_counter2 = "/counter_band3/"



  # run DKL counterfactual with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_counterfactual(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_counter2, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_counterfactual(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_counter2, save_explore, num_cycles)


  logging.info("Reached dkl random---------------------------------")
  


  #--------------------------------------------------------------------------------------------
  #dkl random grid

  logging.info("Reached dkl random band 1---------------------------------")
  # band 1 as target property
  y_targ = peaks_all_scalar1    #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_random1 = "/random_band1/"

  # run DKL random with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random1, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random1, num_cycles)


  logging.info("Reached dkl random band 2---------------------------------")
  # Band 2 as target property
  y_targ = peaks_all_scalar2    #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_random2 = "/random_band2/"

  # run DKL random with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random2, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random2, num_cycles)
      
  # band 3 as target property
  logging.info("Reached dkl random band 3---------------------------------")
  # Band #3 as target property
  y_targ = peaks_all_scalar3    #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_random3 = "/random_band3/"

  # run DKL random with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random3, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_random3, num_cycles)

  logging.info("Reached dkl complete - mature_full---------------------------------")
  #-----------------------------------------------------------------------------------------------
  # dkl complete -- mature full



  logging.info("Reached dkl complete - mature_full band 1---------------------------------")
  # Band 1

  y_targ = peaks_all_scalar1   #training output;
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  y = y_targ.reshape(-1)

  save_mature_full = "/mature_full_record/"

  # run real-time DKL, mature DKL, and full DKL with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_mature_full(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_mature_full, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_mature_full(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, save_mature_full, save_explore, num_cycles)

  logging.info("done")
  
if __name__ == '__main__':
    main()

    













