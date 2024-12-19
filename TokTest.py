#!/opt/toksearch/builds/release/1.6/1.6.1/envs/toksearch_py3.7/bin/python

from ECEI import ECEI
import numpy as np

#clear = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_clear_since_2016.txt').astype(int)
#disrupt = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/d3d_disrupt_since_2016.txt').astype(int)
clean = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/Current_labeled_dset_highest_tol.txt').astype(int)
#already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
#some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
#no_data = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/all_channels_missing_list.txt').astype(int)
#removed = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/AllChannelsMissing_removed.txt').astype(int)
#ips_dir = '/mnt/beegfs/users/rodriguezj/signal_data/ipspr15v'
t_end = np.loadtxt('/mnt/beegfs/users/rodriguezj/shot_lists/t_end.txt')

#shots_of_interest = np.append(clear[:,0], disrupt[:,0])
#shots = np.sort(shots_of_interest).tolist()
#shots = np.arange(195001,200000,1)
#shots = np.setdiff1d(shots_of_interest, np.append(already_dloaded,no_data))
#shots = clean[189:1200,0]
#t_disrupt = clean[189:1200,:]
shots = clean[189:290,0]
t_disrupt = clean[189:290,:]

E = ECEI(side = 'LFS')#server = 'localhost:8002')
print(len(shots))
E.Acquire_Shots_D3D(shots, save_path = '/mnt/beegfs/users/rodriguezj/signal_data/ECEI_felipe/',\
                    d_sample = 1, tksrch = True, rm_spikes = True, felipe_format = True,\
                    t_end = t_end, t_disrupt = t_disrupt, verbose = False)
#E.Get_t_end(
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
