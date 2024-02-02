#!/opt/toksearch/builds/latest/22_10_20.08_07.461/envs/toksearch_py3.7/bin/python

from ECEI import ECEI
import numpy as np

#clear = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_clear_since_2016.txt').astype(int)
#disrupt = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_disrupt_since_2016.txt').astype(int)
clean = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/ecei_clean_conservative_sublist.txt').astype(int)
#already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
#some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
#no_data = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/all_channels_missing_list.txt').astype(int)
#removed = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/AllChannelsMissing_removed.txt').astype(int)

#shots_of_interest = np.append(clear[:,0], disrupt[:,0])
#shots = np.sort(shots_of_interest).tolist()
#shots = np.arange(195001,200000,1)
#shots = np.setdiff1d(shots_of_interest, np.append(already_dloaded,no_data))

E = ECEI(side = 'LFS')#server = 'localhost:8002')
#print(len(shots))
E.Acquire_Shots_D3D(shots, save_path = '/mnt/beegfs/users/rodriguezj/signal_data/500kHz/',\
                    d_sample = 2, tksrch = True)
#E.Generate_Missing_Report_Parallel("01-29-2024-PM",\
#        '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#        "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/")
#already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
#some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
#shots_of_interest_ = np.append(already_dloaded, some_miss)
#shots_ = np.sort(shots_of_interest_).tolist()
#shots_ = np.sort(np.setdiff1d(no_data, removed))
#E.Generate_Quality_Report_Parallel("01-28-2023-PM", disrupt, shots_,\
#        '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#        "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/")
#E.Clean_Missing_Signals("/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/",\
#         '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/')
#E.Clean_Missing_Signal_List('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#            "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/", no_data)
