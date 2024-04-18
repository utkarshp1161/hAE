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
from hAE.torch.scalarizer_eels import calculate_peaks, calculate_peaksTF_grid#, calculate_peaks_and_fit_gaussians_with_lmfit, calculate_peaks_and_fit_gaussians_with_lmfit_integral
from  hAE.torch.plot_utils import plot_dkl
from  hAE.torch.all_dkl import dkl_explore
from  hAE.torch.all_dkl import dkl_counterfactual
from  hAE.torch.all_dkl import dkl_random
from  hAE.torch.all_dkl import dkl_mature_full
import pyTEMlib.file_tools as ft



# Initialize logging
def init_logging(save_dir, config):
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"Logging to {save_dir}log_{current_time}.log")
    filename = f"{save_dir}log_{current_time}.log"
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

def main():
  args = parse_arguments()
  config_path = args.config
  config = load_config(config_path)
  
  window_size = config['settings']['window_size']
  spectralavg = config['settings']['spectralavg']
  save_dir = config['settings']['save_dir']
  data_path = config['settings']['data_path']

  dataset = ft.open_file(data_path)
  image = dataset['Channel_000']
  spectra = dataset['Channel_002']
  image_for_dkl = dataset['Channel_001']
  specim = np.array(spectra)
  img = np.array(image_for_dkl).T


  if not os.path.exists(save_dir):
      os.makedirs(save_dir)
      print(f"Directory {save_dir} created.")

  else:
      print(f"Directory {save_dir} already exists.")
      
  init_logging(save_dir=save_dir, config=config)
  logging.info(f"Directory {save_dir} already exists.")

  #data_path = "/hAE_data/Plasmonic_sets_7222021_fixed.npy"
  cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', 'None')
  logging.info(f"CUDA Visible Devices: {cuda_visible_devices}")


#   plt.imshow(img)
#   plt.savefig(save_dir + "1_img.png")

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



  
  k = 350 # randomly select one of the 3024 patches
  
  plot_highlighted_patch(full_img, features, targets, indices, k, ws, save_dir)
  

  logging.info("Reached scalarizer----------------------------------")
  #------------------------------------------------
  #sacalrizer

  band = [300, 500]# carbon peak
  e_axis = range(len(targets[k]))
  peaks_all_scalar1, features_all, indices_all = calculate_peaksTF_grid(targets, features, indices, e_axis, band)
  #[(3024,), (3024, 8, 8), (3024, 2)]
  plt.scatter(indices_all[:, 0], indices_all[:, 1], c=peaks_all_scalar1, s = 90)
  plt.savefig(save_dir + "3_peaks_all_scalar1.png")
  plt.close()




  # ## Edge mode
  # band = [0.65, 0.75]
  # peaks_all_scalar2, features_all, indices_all = calculate_peaks(targets, features, indices, e_axis, band)
  # plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar2, s = 90)
  # plt.savefig(save_dir + "4_peaks_all_scalar2.png")
  # plt.close()


  # ## Bulk / face mode
  # band = [0.85, 0.95]
  # peaks_all_scalar3, features_all, indices_all = calculate_peaks(targets, features, indices, e_axis, band)
  # plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar3, s = 90)
  # plt.savefig(save_dir + "5_peaks_all_scalar3.png")
  # plt.close()


  # plt.scatter(indices_all[:, 1], indices_all[:, 0], c=peaks_all_scalar3, s = 90)

  dklgp = None  # Reset any model, may help clear memory

  y = np.copy(peaks_all_scalar1)# using first scalarizer
  n, d1, d2 = features.shape # (3024, 8, 8)
  X = features.reshape(n, d1*d2) # (3024, 64)



  logging.info("not doing dkl full data----------------------------------")
#   #---------------------------------------------------------------------------------------

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

  rf = save_dir + "torch/"  #root folder
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



  save_explore =  "/explore_dkl_record/"
  logging.info("Reached dkl explore band 1----------------------------------")
  # on band 1

  y_targ = peaks_all_scalar1    #training output;-- 3024
  y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min()) # bring b/w 0 and 1
  y = y_targ.reshape(-1)# 
  
  beta = config['settings']["beta"] # 0.2
  # run DKL exploration with different acquisition functions
  if acq < 3:   # if user select an individual acquistion function
    acq_idx = acq
    dkl_explore(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps, acq, acq_idx, ws,
                img, window_size, xi, beta, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_explore(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, beta, save_explore, num_cycles)
      print("==============acq_index")
      logging.info("acq_index")
      
  # plot hist(y)
  # plot_hist(

  # logging.info("Reached dkl counterfactual----------------------------------")
  # #---------------------------------------------------------------------------------------------
  # #dkl counterfactual



  # ##band 2
  # logging.info("Reached dkl counterfactual band 2----------------------------------")

  # # Set band 2 as target property

  # y_targ = peaks_all_scalar2    #training output;
  # y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  # y = y_targ.reshape(-1)

  # save_counter1 = "/counter_band2/"

  # # run DKL counterfactual with different acquisition functions
  # if acq < 3:   # if user select an individual acquistion function
  #   acq_idx = acq
  #   dkl_counterfactual(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_counter1, save_explore, num_cycles)

  # elif acq == 3:  # if user selected to run all acquistion functions
  #   for i in range (3):
  #     acq_idx = i
  #     dkl_counterfactual(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_counter1, save_explore, num_cycles)
      
  # ## band 3
  # logging.info("Reached dkl counterfactual band 3----------------------------------")
  # y_targ = peaks_all_scalar3    #training output;
  # y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  # y = y_targ.reshape(-1)

  # save_counter2 = "/counter_band3/"



  # # run DKL counterfactual with different acquisition functions
  # if acq < 3:   # if user select an individual acquistion function
  #   acq_idx = acq
  #   dkl_counterfactual(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_counter2, save_explore, num_cycles)

  # elif acq == 3:  # if user selected to run all acquistion functions
  #   for i in range (3):
  #     acq_idx = i
  #     dkl_counterfactual(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_counter2, save_explore, num_cycles)


  # logging.info("Reached dkl random---------------------------------")



  # #--------------------------------------------------------------------------------------------
  # #dkl random grid

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
                img, window_size, xi, beta, save_random1, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_random(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, beta,  save_random1, num_cycles)


  # logging.info("Reached dkl random band 2---------------------------------")
  # # Band 2 as target property
  # y_targ = peaks_all_scalar2    #training output;
  # y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  # y = y_targ.reshape(-1)

  # save_random2 = "/random_band2/"

  # # run DKL random with different acquisition functions
  # if acq < 3:   # if user select an individual acquistion function
  #   acq_idx = acq
  #   dkl_random(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_random2, num_cycles)

  # elif acq == 3:  # if user selected to run all acquistion functions
  #   for i in range (3):
  #     acq_idx = i
  #     dkl_random(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_random2, num_cycles)
      
  # # band 3 as target property
  # logging.info("Reached dkl random band 3---------------------------------")
  # # Band #3 as target property
  # y_targ = peaks_all_scalar3    #training output;
  # y_targ = (y_targ-y_targ.min())/(y_targ.max()-y_targ.min())
  # y = y_targ.reshape(-1)

  # save_random3 = "/random_band3/"

  # # run DKL random with different acquisition functions
  # if acq < 3:   # if user select an individual acquistion function
  #   acq_idx = acq
  #   dkl_random(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_random3, num_cycles)

  # elif acq == 3:  # if user selected to run all acquistion functions
  #   for i in range (3):
  #     acq_idx = i
  #     dkl_random(X, y, indices_all, ts, 
  #               rs, rf, acq_funcs, 
  #               exploration_steps,acq, acq_idx, ws,
  #               img, window_size, xi, beta, save_random3, num_cycles)

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
                img, window_size, xi, beta, save_mature_full, save_explore, num_cycles)

  elif acq == 3:  # if user selected to run all acquistion functions
    for i in range (3):
      acq_idx = i
      dkl_mature_full(X, y, indices_all, ts, 
                rs, rf, acq_funcs, 
                exploration_steps,acq, acq_idx, ws,
                img, window_size, xi, beta, save_mature_full, save_explore, num_cycles)

  logging.info("done")
  
if __name__ == '__main__':
    main()

    













