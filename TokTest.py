#!/opt/toksearch/builds/latest/22_10_20.08_07.461/envs/toksearch_py3.7/bin/python

from ECEI import ECEI
import numpy as np

clear = np.loadtxt('/eagle/fusiondl_aesp/shot_lists/d3d_clear_since_2016.txt').astype(int)
disrupt = np.loadtxt('/eagle/fusiondl_aesp/shot_lists/d3d_disrupt_since_2016.txt').astype(int)
already_dloaded = np.loadtxt('/eagle/fusiondl_aesp/signal_data/d3d/ECEI_HFS_100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
some_miss = np.loadtxt('/eagle/fusiondl_aesp/signal_data/d3d/ECEI_HFS_100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
no_data = np.loadtxt('/eagle/fusiondl_aesp/signal_data/d3d/ECEI_100kHz/missing_shot_info/all_channels_missing_list.txt').astype(int)
#removed = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/AllChannelsMissing_removed.txt').astype(int)
read_error = np.loadtxt('/eagle/fusiondl_aesp/signal_data/d3d/ECEI_1kHz/missing_shot_info/read_error_list.txt').astype(int)

shots_of_interest = np.append(already_dloaded, some_miss)
shots = np.sort(shots_of_interest).tolist()
#shots = np.setdiff1d(shots_of_interest, np.append(already_dloaded,no_data))

E = ECEI(side = 'HFS')#server = 'localhost:8002')
#print(len(shots))
#E.Acquire_Shots_D3D(shots, save_path = '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#                    d_sample = 10, tksrch = True)
#E.Generate_Missing_Report_Parallel("11-28-2023-EV",\
#        '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#        "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/")
#already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
#some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
#shots_of_interest_ = np.append(already_dloaded, some_miss)
#shots_ = np.sort(shots_of_interest_).tolist()
#shots_ = np.sort(np.setdiff1d(no_data, removed))
E.Generate_Quality_Report_Parallel("12-29-2023-PM", disrupt, shots,\
        '/eagle/fusiondl_aesp/signal_data/d3d/ECEI_HFS_1kHz/',\
        "/eagle/fusiondl_aesp/signal_data/d3d/ECEI_HFS_1kHz/missing_shot_info/")
#E.Clean_Missing_Signals("/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/",\
#         '/mnt/beegfs/users/rodriguezj/signal_data/100kHz/')
#E.Clean_Missing_Signal_List('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/',\
#            "/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/", no_data)
#E.Remove_List('/eagle/fusiondl_aesp/signal_data/d3d/ECEI_1kHz/', read_error)
