from ECEI import ECEI
import numpy as np

no_chan_missing = np.loadtxt('/p/d3d_ecei/shot_lists/no_channels_missing_list.txt')
some_chan_missing = np.loadtxt('/p/d3d_ecei/shot_lists/some_channels_missing_list.txt')
already_dloaded = np.loadtxt('/p/d3d_ecei/signal_data/100kHz/missing_shot_info/no_channels_missing_list_10-23.txt')

shots_of_interest = np.append(no_chan_missing, some_chan_missing)
shots_of_interest = np.sort(shots_of_interest).tolist()
shots = np.setdiff1d(shots_of_interest, already_dloaded)
num_shots = len(shots)//2

E = ECEI(server = 'localhost:8002')
for i in range(10):
    E.Acquire_Shots_D3D(shots[:num_shots], save_path = '/p/d3d_ecei/signal_data/100kHz/',\
                max_cores = 4, verbose = False, d_sample = 10, try_again = True)