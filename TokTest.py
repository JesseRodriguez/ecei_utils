#!/opt/toksearch/builds/latest/22_10_20.08_07.461/envs/toksearch_py3.7/bin/python

from ECEI import ECEI
import numpy as np

clear = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_clear_since_2016.txt').astype(int)
disrupt = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_disrupt_since_2016.txt').astype(int)
already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
no_data = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/all_channels_missing_list.txt').astype(int)

shots_of_interest = np.append(already_dloaded, some_miss)
shots = np.sort(shots_of_interest).tolist()
#shots = np.setdiff1d(shots_of_interest, np.append(already_dloaded,no_data))

E = ECEI()#server = 'localhost:8002')
#print(len(shots))
#E.Acquire_Shots_D3D(shots, save_path = '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#                    d_sample = 10, tksrch = True)
#E.Generate_Quality_Report_Parallel("11-08-2023-EV", disrupt, shots,\
#        '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#        "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/")
E.Clean_Missing_Signals("/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/",\
         '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/')
