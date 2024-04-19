# Define a function to run dkl analysis with random sampling
from sklearn.model_selection import train_test_split
import numpy as np
import os
import numpy as np
import matplotlib.pyplot as plt
import atomai as aoi
import os
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle
from atomai import utils
from sklearn.model_selection import train_test_split
import importlib
import hAE.torch
from  hAE.torch.plot_utils import plot_dkl, plot_highlighted_patch_during_dkl, plot_highlighted_patch_during_seed
from hAE.torch.scalarizer_eels import calculate_peaks, calculate_peaksTF_grid, scalarazier_TF#, calculate_peaks_and_fit_gaussians_with_lmfit, calculate_peaks_and_fit_gaussians_with_lmfit_integral
from hAE.torch.instruments_acquisition import spectrum_calc_grid, spectrum_calc_grid_no_specim, detect_bright_region, spectrum_calc_dummy
from hAE.torch.update_data import sel_next_point, update_train_test_data_instrument
import Pyro5.api



def dkl_explore(X, indices_all, ts, rs, rf, acq_funcs, exploration_steps,
                acq, acq_idx, ws, img, window_size, xi, beta, save_explore, num_cycles=200):
  # Here X_train and y_train are our measured image patches and spectra information,
  # whereas X_test and y_test are the "unknown" ones. The indices_train are grid coordinates of the measured points,
  # whereas the indices_test are the grid coordinates of the remaining available points on the grid
  """

  """
  (X_train, X_test,
   indices_train, indices_test) = train_test_split(
       X, indices_all, test_size=ts, shuffle=True, random_state=rs)# ts is test size which has 99.5% of data
  print('X_train shape: '); print(X_train.shape)
  data_dim = X_train.shape[-1]# 64
  seed_points = len(X_train)# 3
  spots = detect_bright_region(img)
  y_train_unnor = np.zeros(seed_points)# 3
  # experiment on seed points:
  for i in range(seed_points):
    x_coord, y_coord = indices_train[i]
    #spectrum = spectrum_calc_grid(specim ,int(x_coord), int(y_coord))
    spectrum = spectrum_calc_dummy(int(x_coord), int(y_coord))
    y_train_unnor[i] = spectrum.sum()#scalarazier_TF(spectrum, 0, 31)
    #y_train_unnor[i] = spots[int(x_coord), int(y_coord)]
    print("scalrizer val",y_train_unnor[i])
    plot_highlighted_patch_during_seed(rf, img, X_train,spots[0,:], indices_train, i , window_size, i, "seed")# careful with y_train what you send
  
  y_train = (y_train_unnor-y_train_unnor.min())/(y_train_unnor.max()-y_train_unnor.min())# normalizing y_train

    
    
  
  if not os.path.exists(rf + acq_funcs[acq_idx] + save_explore):
    print(rf + acq_funcs[acq_idx] + save_explore)
    os.makedirs(rf + acq_funcs[acq_idx] + save_explore, exist_ok=True)
  os.chdir(rf + acq_funcs[acq_idx] + save_explore) # go to this directory
  np.savez("initial_traindata.npz", X_train=X_train, X_test=X_test, y_train=y_train,
            indices_train=indices_train, indices_test=indices_test)




  for e in range(exploration_steps):# out of 15
    print("{}/{}".format(e+1, exploration_steps))
    # Update GP posterior
    dklgp = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
    dklgp.fit(X_train, y_train, training_cycles=num_cycles)

    mean, var = dklgp.predict(X, batch_size=len(X))# same shape as len(y_all) = np.concatenate((y_train, y_test))
    #mean, var-- shape - (3024,), (3024,) # same shape as y
    # plot_dkl_result(y, mean, var, indices=indices_train,
    #                 ws=ws, scatter_location=True)

    plot_dkl(img, window_size,y_train, mean, var)
    
    plt.savefig('dkl_explore_{}.png'.format(e))
    plt.close()
    # Compute acquisition function
    next_point_idx = sel_next_point(dklgp, X_train, X_test, xi, beta, acq_idx)# got where to look
    # next_point_idx_1: UCB_0.2
    # next_point_idx_2: UCB_0.5
    # next_point_idx_3: EI
    # now we have{3 points}
    next_point  = indices_test[next_point_idx]# select the index of thae next patch
    
    x_coord, y_coord = next_point
    #spectrum = spectrum_calc_grid(specim ,int(x_coord), int(y_coord))
    #spectrum = spectrum_calc_grid_no_specim(int(x_coord), int(y_coord))
    # Do "measurement"
    #measured_point = scalarazier_TF(spectrum)
    #measured_point = spots[int(x_coord), int(y_coord)]
    spectrum = spectrum_calc_dummy(int(x_coord), int(y_coord))
    measured_point = spectrum.sum()#scalarazier_TF(spectrum, 0, 31)
    print("measured_point",measured_point)
    # Update train and test datasets
    # Plot current result

    # Update train and test datasets 
    """adds 1 more data point to train data at every iteration"""
    (X_train, X_test, y_train, indices_train,
     indices_test) = update_train_test_data_instrument(X_train, X_test, y_train,
                                            indices_train, indices_test,
                                            next_point_idx, measured_point, next_point)
    y_train = (y_train-y_train.min())/(y_train.max()-y_train.min())# normalizing y_train
    plot_highlighted_patch_during_dkl(img, X_train, spots[0,:], indices_train, -1 , window_size, e, "dkl")# -1 is a hack to make it work, basicall recentally added point is the last in the array
    # Save result
    np.savez("record{}.npz".format(e),
             mean=mean, var=var, next_point_idx=next_point_idx,
             next_point=next_point, measured_point=measured_point)
  # Save final traindata
  np.savez("final_traindata.npz", X_train=X_train, X_test=X_test,
           y_train=y_train,
           indices_train=indices_train, indices_test=indices_test)
  


def dkl_counterfactual(X, y, indices_all, ts, rs, rf, acq_funcs, exploration_steps,
                acq, acq_idx, ws, img, window_size, xi, beta, save_counter, save_explore, num_cycles=200):

  # Initialize train/test datasets
  (X_train, X_test, y_train, y_test,
   indices_train, indices_test) = train_test_split(
       X, y, indices_all, test_size=ts, shuffle=True, random_state=rs)
    

  # Set path for saving result
  if not os.path.exists(rf + acq_funcs[acq_idx] + save_counter):
    os.mkdir(rf + acq_funcs[acq_idx] + save_counter)

  # Save initial train data
  os.chdir(rf + acq_funcs[acq_idx] + save_counter)
  np.savez("initial_traindata.npz", X_train = X_train, X_test = X_test,
           y_train = y_train, y_test = y_test,
           indices_train = indices_train, indices_test = indices_test)
  data_dim = X_train.shape[-1]

  # Start DKL analysis

  for e in range(exploration_steps):
    print("{}/{}".format(e+1, exploration_steps))
    # update GP posterior
    dklgp = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
    dklgp.fit(X_train, y_train, training_cycles=num_cycles, lr = 0.01)

    mean, var = dklgp.predict(X, batch_size=len(X))
    # plot_dkl_result(y, mean, var, indices=indices_train,
    #                 ws = ws, scatter_location = True)
    plot_dkl(img, window_size, y, mean, var)
    plt.savefig('dkl_counterfactual_{}.png'.format(e))
    plt.close()

    #########################################################
    #################### Critical step ######################
    #########################################################

    # Load measurement locations from Step 1
    os.chdir(rf + acq_funcs[acq_idx] + save_explore)
    rec = np.load("record{}.npz".format(e))
    next_point_idx = rec['next_point_idx']
    next_point = indices_test[next_point_idx]
    #########################################################
    #########################################################

    # Do "measurement"
    measured_point = y_test[next_point_idx]

    # Update train and test datasets
    (X_train, X_test, y_train, y_test, indices_train,
     indices_test) = update_train_test_data(X_train, X_test, y_train, y_test,
                                            indices_train, indices_test,
                                            next_point_idx, measured_point, next_point)

    # Save result
    os.chdir(rf + acq_funcs[acq_idx] + save_counter)
    np.savez("record{}.npz".format(e),
             mean = mean, var = var, next_point_idx = next_point_idx,
             next_point = next_point, measured_point = measured_point)

  # Save final traindata
  os.chdir(rf + acq_funcs[acq_idx] + save_counter)
  np.savez("final_traindata.npz", X_train = X_train, X_test = X_test,
           y_train = y_train, y_test = y_test,
           indices_train = indices_train, indices_test = indices_test)


    
    
def dkl_random(X, y, indices_all, ts, rs, rf, acq_funcs, exploration_steps,
                acq, acq_idx, ws, img, window_size, xi, beta, save_random, num_cycles=200):
  # Initialize train/test datasets
  (X_train, X_test, y_train, y_test,
   indices_train, indices_test) = train_test_split(
       X, y, indices_all, test_size=ts, shuffle=True, random_state=rs)
  data_dim = X_train.shape[-1]

  # Set path for saving result
  if not os.path.exists(rf + acq_funcs[acq_idx] + save_random):
    os.mkdir(rf + acq_funcs[acq_idx] + save_random)

  # Save initial train data
  os.chdir(rf + acq_funcs[acq_idx] + save_random)
  np.savez("initial_traindata.npz", X_train = X_train, X_test = X_test,
           y_train = y_train, y_test = y_test,
           indices_train = indices_train, indices_test = indices_test)

  # Start DKL analysis
  for e in range(exploration_steps):
    print("{}/{}".format(e+1, exploration_steps))
    # update GP posterior
    dklgp = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
    dklgp.fit(X_train, y_train, training_cycles=num_cycles, lr = 0.01)

    mean, var = dklgp.predict(X, batch_size=len(X))
    # plot_dkl_result(y, mean, var, indices=indices_train,
    #                 ws = ws, scatter_location = True)
    plot_dkl(img, window_size, y, mean, var)
    plt.savefig('dkl_random_{}.png'.format(e))
    plt.close()

    #########################################################
    #################### Critical step ######################
    #########################################################
    next_point_idx = np.random.randint(len(X_test))
    next_point = indices_test[next_point_idx]
    #########################################################
    #########################################################

    # Do "measurement"
    measured_point = y_test[next_point_idx]

    # Update train and test datasets
    (X_train, X_test, y_train, y_test, indices_train,
     indices_test) = update_train_test_data(X_train, X_test, y_train, y_test,
                                            indices_train, indices_test,
                                            next_point_idx, measured_point, next_point)

    # Save result
    os.chdir(rf + acq_funcs[acq_idx] + save_random)
    np.savez("record{}.npz".format(e),
             mean = mean, var = var, next_point_idx = next_point_idx,
             next_point = next_point, measured_point = measured_point)

  # Save final traindata
  os.chdir(rf + acq_funcs[acq_idx] + save_random)
  np.savez("final_traindata.npz", X_train = X_train, X_test = X_test,
           y_train = y_train, y_test = y_test,
           indices_train = indices_train, indices_test = indices_test)




# Define a function to run dkl counterfactual with different scalarizer
def dkl_mature_full(X, y, indices_all, ts, rs, rf, acq_funcs, exploration_steps,
                acq, acq_idx, ws, img, window_size, xi, beta, save_mature_full, save_explore, num_cycles=200):

  # Load last step train dataset
  os.chdir(rf + acq_funcs[acq_idx] + save_explore)
  rec = np.load("final_traindata.npz")
  X_train = rec["X_train"]
  y_train = rec["y_train"]
  data_dim = X_train.shape[-1]

  # Train a matured DKL model with the last step train dataset
  # dklgp0 will be the matured DKL model
  dklgp0 = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
  dklgp0.fit(X_train, y_train, training_cycles=num_cycles)

  # Train a full-informed DKL model with all data
  # dklgp1 will be the full-informed DKL model
  dklgp1 = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
  dklgp1.fit(X, y, training_cycles=num_cycles)

  # Initialize train/test datasets
  (X_train, X_test, y_train, y_test,
   indices_train, indices_test) = train_test_split(
       X, y, indices_all, test_size=ts, shuffle=True, random_state=rs)

  # Set path for saving result
  if not os.path.exists(rf + acq_funcs[acq_idx] + save_mature_full):
    os.mkdir(rf + acq_funcs[acq_idx] + save_mature_full)

  # Start DKL analysis
  for e in range(exploration_steps):
    print("{}/{}".format(e+1, exploration_steps))
    #update GP posterior
    dklgp = aoi.models.dklGPR(data_dim, embedim=2, precision="single")
    dklgp.fit(X_train, y_train, training_cycles=num_cycles, lr = 0.01)

    mean, var = dklgp.predict(X_test, batch_size=len(X_test))  #real-time DKL model
    mean0, var0 = dklgp0.predict(X_test, batch_size=len(X_test)) #matured DKL model
    mean1, var1 = dklgp1.predict(X_test, batch_size=len(X_test)) #full-informed DKL model

    # Load measurement location from Step 1
    os.chdir(rf + acq_funcs[acq_idx] + save_explore)
    rec = np.load("record{}.npz".format(e))
    next_point_idx = rec['next_point_idx']
    next_point = indices_test[next_point_idx]

    # Do "measurement"
    measured_point = y_test[next_point_idx]

        # Save result
    os.chdir(rf + acq_funcs[acq_idx] + save_mature_full)
    np.savez("record{}.npz".format(e), y_test = y_test,
             mean = mean, var = var,
             mean0 = mean0, var0 = var0,
             mean1 = mean1, var1 = var1)

    # Update train and test datasets
    (X_train, X_test, y_train, y_test, indices_train,
     indices_test) = update_train_test_data(X_train, X_test, y_train, y_test,
                                            indices_train, indices_test,
                                            next_point_idx, measured_point, next_point = next_point)
    



