#!/opt/toksearch/builds/latest/22_10_20.08_07.461/envs/toksearch_py3.7/bin/python

"""
Use the functions in this file to fetch non-ECEI data using toksearch on DIII-D's 
Saga cluster
Jesse A Rodriguez, 01/30/2024
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import time
import sys
import os
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import h5py
import scipy.signal
import math
try:
    import toksearch as ts
    tksrch = True
except ImportError:
    tksrch = False
    pass
try:
    import MDSplus as MDS
except ImportError:
    pass

################################################################################
## Utility Functions and Globals
################################################################################
etemp_profile = "ZIPFIT01/PROFILES.ETEMPFIT"
edens_profile = "ZIPFIT01/PROFILES.EDENSFIT"
itemp_profile = "ZIPFIT01/PROFILES.ITEMPFIT"
q95 = "EFIT01/RESULTS.AEQDSK.Q95"
ip = "ipspr15V"
iptarget = "ipsiptargt"
iperr = "ipeecoil"
li = "efsli"
lm = 'dusbradial'
dens = 'dssdenest'
energy = 'efswmhd'
p_in = 'bmspinj'
pradcore = r'\bol_l15_p'
pradedge = r'\bol_l03_p'
betan = 'efsbetan'
torquein = 'bmstinj'
tmamp1 = 'nssampn1l'
tmamp2 = 'nssampn2l'
tmfreq1 = 'nssfrqn1l'
tmfreq2 = 'nssfrqn2l'
ipdirect = "iptdirect"
pt_data_sigs = [iptarget, iperr, li, lm, dens, energy, p_in, pradcore, pradedge,\
        betan, torquein, tmamp1, tmamp2, tmfreq1, tmfreq2, ipdirect]
causal_shifts = {"ZIPFIT01/PROFILES.ETEMPFIT": 10,\
        "ZIPFIT01/PROFILES.EDENSFIT": 10,\
        "ZIPFIT01/PROFILES.ITEMPFIT": 10,\
        "EFIT01/RESULTS.AEQDSK.Q95": 10}

def Download_Shot_List_toksearch(shots, signal_list, savepath, verbose = False): 
    # Initialize the toksearch pipeline
    pipe = ts.Pipeline(shots)

    #make sure signal directories exist
    for signal in signal_list:
        os.makedirs(savepath+signal+'/', exist_ok=True)

        # Fetch signals for these 32 channels
        try:
            if len(signal.split('/')) == 1:
                pipe.fetch(signal, ts.PtDataSignal(signal))
            else:
                sig = signal.split('/')[1]
                tree = signal.split('/')[0]
                pipe.fetch(sig, ts.MdsSignal(sig, tree, location='remote://atlas.gat.com'))
        except Exception as e:
            print(f"An error occurred: {e}")

    # Function to process and write to txt
    @pipe.map
    def process_and_save(rec):
        # Get the shot ID from the record
        shot_id = rec['shot']
        if np.random.uniform() < 1/100:
            print(f"Working on shot {shot_id}. This job runs from {shots[0]}-{shots[len(shots)-1]}.")

        for signal in signal_list:
            file_path = savepath+signal+f'/{shot_id}.txt'
            try:
                data = np.asarray(rec[signal]['data'])
                time = np.asarray(rec[signal]['times'])*1e-3 #D3D Data is reported in ms
                if signal in causal_shifts.keys():
                    time = time + causal_shifts[signal]*1e-3
            except Exception as e:
                if verbose:
                    print(f"An error occurred in shot {shot_id}: {e}")

            # Save channel-specific data
            if rec[signal] is None:
                array = np.array([[0,0]])
                if verbose:
                    print('shot', shot_id, signal, "No data")
            else:
                array = np.column_stack((time.T, data.T))

            np.savetxt(file_path, array)
    
    # Discard data from pipeline
    pipe.keep([])

    # Fetch data, limiting to 10GB per shot as per collaborator's advice
    #results = list(pipe.compute_serial())
    #results = list(pipe.compute_spark())
    pipe.compute_ray(memory_per_shot=int(1.1*(10e9)))


if __name__ == '__main__':
    already_dloaded = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/no_channels_missing_list.txt').astype(int)
    some_miss = np.loadtxt('/mnt/beegfs/users/rodriguezj/signal_data/100kHz/missing_shot_info/some_channels_missing_list.txt').astype(int)
    shots_of_interest = np.append(already_dloaded, some_miss)
    shotlist = np.sort(shots_of_interest).tolist()
    #np.savetxt('/mnt/beegfs/users/rodriguezj/shot_lists/ECEI_unlabeled.txt', shotlist, fmt = '%i')
    Download_Shot_List_toksearch(shotlist, [pradcore, pradedge], '/mnt/beegfs/users/rodriguezj/signal_data/')
