"""
The module composed in this file is designed to handle the processing/handling
and incorporation of electron cyclotron emission imaging data into the FRNN
disruption prediction software suite. It contains snippets from the rest of
the FRNN codebase, and therefore is partially redundant.
Jesse A Rodriguez, 06/28/2021
"""

import numpy as np
import matplotlib as mpl
mpl.rcParams['figure.dpi']=100
import matplotlib.pylab as plt
plt.rc('font', family='tahoma')
font = 18
plt.rc('xtick', labelsize=font)
plt.rc('ytick', labelsize=font)
import time
import sys
import os
import multiprocessing as mp
import MDSplus as MDS
from functools import partial

###############################################################################
## Utility Functions and Globals
###############################################################################
c = 299792458


def Fetch_ECEI_d3d(channel_path, shot_number, c = None, verbose = False):
    """
    Basic fetch ecei data function.

    Args:
        channel_path: str, path to save .txt file (channel folder, format LFSxxxx)
        shot_number: int, DIII-D shot number
        c: MDSplus.Connection object. None by default
        verbose: bool, suppress print statements
    """
    channel = channel_path
    shot = str(int(shot_number))
    mds_fail_pd = False
    mds_fail_pd2 = False
    mds_fail_p = False
    mds_fail_t = False

    #ptdata2 method (seems to be most reliable)
    try:
        x_pd2 = c.get('dim_of(_s = ptdata2('+channel+','+shot+'))')
        y_pd2 = c.get('_s = ptdata2('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_pd2 = True
        pass
    if not mds_fail_pd2:
        if x_pd2.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_pd2, y_pd2, None, True
    
    #psuedo method
    try:
        x_p = c.get('dim_of(_s = psuedo('+channel+','+shot+'))')
        y_p = c.get('_s = psuedo('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_p = True
        pass
    if not mds_fail_p:
        if x_p.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_p, y_p, None, True
            
    #ptdata method
    try:
        x_pd = c.get('dim_of(_s = ptdata('+channel+','+shot+'))')
        y_pd = c.get('_s = ptdata('+channel+','+shot+')')
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_pd = True
        pass
    if not mds_fail_pd:
        if x_pd.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_pd, y_pd, None, True

    #tree method
    try:
        c.openTree(channel, shot)
        x_t = c.get('dim_of(_s = '+shot+')').data()
        y_t = c.get('_s = '+shot).data()
    except Exception as e:
        if verbose:
            print(e)
        mds_fail_t = True
        pass
    if not mds_fail_t:
        if x_t.shape[0] > 1:
            print('Data exists for shot '+shot+' in channel '+channel[-5:-1]+'.')
            return x_t, y_t, None, True

    print('Data DOES NOT exist for shot '+shot+' in channel '+channel[-5:-1]+'.')
    return None, None, None, False


def Download_Shot(shot_num_queue, c, channel_paths, sentinel = -1, verbose = False):
    """
    Accepts a multiprocessor queue of shot numbers and downloads/saves data for
    a single shot off the front of the queue.

    Args:
        shot_num_queue: muliprocessing queue object containing shot numbers
        c: MDSplus.Connection object
        channel_paths: list containing savepaths to channel folders
        sentinel: sentinel value; -1 by default. Serves as the mechanism for
                  terminating the parallel program.
        verbose: bool, suppress print statements
    """
    missing_shots = 0
    while True:
        shot_num = shot_num_queue.get()
        if shot_num == sentinel:
            break
        shot_complete = True
        for channel_path in channel_paths:
            save_path = channel_path+'/{}.txt'.format(int(shot_num))

            success = False
            if os.path.isfile(save_path):
                if os.path.getsize(save_path) > 0:
                    success = True
                else:
                    print('Channel {}, shot {} '.format(channel_path[-5:-1],\
                           int(shot_num)),'was downloaded incorrectly (empty file). \
                           Redownloading.')

            if not success:
                try:
                    try:
                        time, data, mapping, success = Fetch_ECEI_d3d(\
                                                channel_path[-9:], shot_num, c,\
                                                verbose)
                    except Exception as e:
                        print(e)
                        sys.stdout.flush()
                        print('Channel {}, shot {} missing, all mds commands \
                               failed.'.format(channel_path[-5:-1], shot_num))
                        success = False

                    if success:
                        data_two_column = np.vstack((time, data)).transpose()
                        np.savetxt(save_path, data_two_column, fmt='%.5e')
                    else:
                        np.savetxt(save_path[:-10]+'missing_'+save_path[-10:],\
                                   np.array([-1.0]), fmt='%.5e')

                except BaseException:
                    print('Could not save channel {}, shot {}.'.format(\
                           channel_path[-5:-1], shot_num))
                    print('Warning: Incomplete!!!')
                    raise
            else:
                print('Channel {}, shot {} '.format(channel_path[-5:-1],\
                       int(shot_num)),'has already been downloaded.')
            sys.stdout.flush()
            if not success:
                missing_shots += 1

    print('Finished with {} channel signals missing.'.format(missing_shots))
    return
                         

def Download_Shot_List(shot_numbers, channel_paths, max_cores = 8,\
                       server = 'atlas.gat.com', verbose = False):
    """
    Accepts list of shots and downloads them in parallel

    Args:
        shot_numbers: list of integer shot numbers
        channel_paths: list of channel save path folders
        max_cores: int, max number of cores for parallelization
        server: MDSplus server, str. D3D server by default
        verbose: bool, suppress print statements
    """
    sentinel = -1
    fn = partial(Download_Shot, channel_paths = channel_paths,\
                 sentinel = sentinel, verbose = verbose)
    num_cores = min(mp.cpu_count(), max_cores)
    queue = mp.Queue()
    assert len(shot_numbers) < 32000
    for shot_num in shot_numbers:
        queue.put(shot_num)
    for i in range(num_cores):
        queue.put(sentinel)

    connections = [MDS.Connection(server) for _ in range(num_cores)]
    processes = [mp.Process(target=fn, args = (queue, connections[i]))\
                 for i in range(num_cores)]
    print('Running in parallel on {} processes.'.format(num_cores))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


def Count_Missing(shot_list, channel_paths, missing_path):
    """
    Accepts a list of all channel paths and produces an up-to-date list of all
    missing data and places it in missing_path

    Args:
        shot_list: 1-D numpy array of DIII-D shot numbers
        channel_paths: list of channel paths
        missing_path: folder for missing shot reports
    """
    min_shot = np.argmin(shot_list)
    max_shot = np.argmax(shot_list)
    report = open(missing_path+'/missing_report_'+str(int(shot_list[min_shot]))+'-'+\
                  str(int(shot_list[max_shot]))+'.txt', mode = 'w',\
                  encoding='utf-8')
    report.write('Missing channel signals for download from shot {} to shot {}:\n'.\
                  format(shot_list[min_shot], shot_list[max_shot]))
    for channel_path in channel_paths:
        for filename in os.listdir(channel_path):
            if filename.startswith('missing'):
                report.write('Channel '+channel_path[-5:-1]+', shot #'+filename[8:-4]+'\n')

    report.close()


###############################################################################
## ECEI Class
###############################################################################
class ECEI:
    def __init__(self):
        """
        Initialize ECEI object

        Args:
        """
        self.ecei_channels = []
        for i in range(20):
            for j in range(8):
                self.ecei_channels.append('"LFS{:02d}{:02d}"'.format(i+1,j+1))

    ###########################################################################
    ## Data Processing
    ###########################################################################
    def Proc(self):
        """
        Processing function

        Args:
        """
        return 0

    ###########################################################################
    ## Visualization
    ###########################################################################
    def Viz(self):
        """
        Visualization function
        
        Args:
        """
        return 0

    ###########################################################################
    ## Data Acquisition
    ###########################################################################
    def Acquire_Shots_D3D(self, shot_numbers, save_path = os.getcwd(),\
                          max_cores = 8, verbose = False):
        """
        Accepts a list of shot numbers and downloads the data, saving them into
        folders corresponding to the individual channels. Returns nothing.

        Args:
            shot_numbers: 1-D numpy array of integers, DIII-D shot numbers
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress most print statements
        """
        # Construct channel save paths and create them if needed.
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            channel_path = os.path.join(save_path, self.ecei_channels[i])
            channel_paths.append(channel_path)
            if not os.path.exists(channel_path):
                os.mkdir(channel_path)
        #Missing shots directory
        missing_path = os.path.join(save_path, 'missing_shot_info')
        if not os.path.exists(missing_path):
            os.mkdir(missing_path)

        try:
            c = MDS.Connection('atlas.gat.com')
        except Exception as e:
            print(e)
            return False

        Download_Shot_List(shot_numbers, channel_paths, max_cores = max_cores,\
                           server = 'atlas.gat.com', verbose = verbose)

        Count_Missing(shot_numbers, channel_paths, missing_path)

        return


    def Acquire_Shot_Sequence_D3D(self, shots, shot_1, clear_file, disrupt_file,\
                                  save_path = os.getcwd(), max_cores = 8,\
                                  verbose = False):
        """
        Accepts a desired number of non-disruptive shots, then downloads all
        shots in our labelled database up to the last non-disruptive shot.
        Returns nothing.

        Args:
            shots: int, number of non-disruptive shots you want to download
            shot_1: int, the shot number you want to start with
            clear_file: The path to the clear shot list
            disrupt_file: The path to the disruptive shot list
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress some exception info
        """
        clear_shots = np.loadtxt(clear_file)
        disrupt_shots = np.loadtxt(disrupt_file)

        first_c = False
        first_d = False
        i = 0
        while not first_c:
            if clear_shots[i,0] >= shot_1:
                start_c = i
                first_c = True
            i += 1
        i = 0
        while not first_d:
            if disrupt_shots[i,0] >= shot_1:
                start_d = i
                first_d = True
            i += 1

        shot_list = np.array([clear_shots[start_c,0]])
        for i in range(shots-1):
            shot_list = np.append(shot_list, [clear_shots[i+start_c+1,0]])

        last = False
        i = start_d
        while not last:
            if disrupt_shots[i,0] <= clear_shots[start_c+shots-1,0]:
                end_d = i
                last = True
            i += 1

        for i in range(end_d - start_d + 1):
            shot_list = np.append(shot_list, [disrupt_shots[i+start_d,0]])

        self.Acquire_Shots_D3D(shot_list, save_path, max_cores, verbose)

        return

    def Clean_Channel_Dirs(self, save_path = os.getcwd()):
        """
        Removes all signal files in the channel directories. If the directories
        don't exist, they are created.
        """
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            channel_path = os.path.join(save_path, self.ecei_channels[i])
            channel_paths.append(channel_path)
            if not os.path.exists(channel_path):
                os.mkdir(channel_path)

        for channel_path in channel_paths:
            for signal_file in os.listdir(channel_path):
                signal = os.path.join(channel_path, signal_file)
                os.remove(signal)
