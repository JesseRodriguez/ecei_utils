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


def Fetch_ECEI_d3d(channel_path, shot_number, c = None):
    """
    Basic fetch ecei data function, to be fed to Machine object.

    Args:
        channel_path: str, path to save .txt file (channel folder)
        shot_number: int, DIII-D shot number
        c: MDSplus.Connection object. None by default
    """
    channel = channel_path
    shot = str(shot_number)

    try:
        found = False
        x = c.get('dim_of(_s = ptdata2('+channel+','+shot+'))')
        y = c.get('_s = ptdata2('+channel+','+shot+')')
        if x.shape[0] > 0:
            found = True
            print('Data exists for shot #:', shot)

    except Exception as e:
        print(e)
        return None, None, None, False

    if found:
        return x, y, None, True
    else:
        return None, None, None, False


def Download_Shot(shot_num_queue, c, channel_paths, sentinel = -1):
    """
    Accepts a multiprocessor queue of shot numbers and downloads/saves data for
    a single shot off the front of the queue.

    Args:
        shot_num_queue: muliprocessing queue object containing shot numbers
        c: MDSplus.Connection object
        channel_paths: list containing savepaths to channel folders
        sentinel: sentinel value; -1 by default. Serves as the mechanism for
                  terminating the parallel program.
    """
    missing_shots = 0
    while True:
        shot_num = shot_num_queue.get()
        if shot_num == sentinel:
            break
        shot_complete = True
        for channel_path in channel_paths:
            save_path = channel_path+'/{}.txt'.format(shot_num)

            success = False
            if os.path.isfile(save_path):
                if os.path.getsize(save_path) > 0:
                    print('-', end = '')
                    success = True
                else:
                    print('Channel {}, shot {} '.format(channel_path[-5:-1],\
                           shot_num),'was downloaded incorrectly (empty file). \
                           Redownloading.')

            if not success:
                try:

                    try:
                        time, data, mapping, success = Fetch_ECEI_d3d(\
                                                channel_path[-9:], shot_num, c)
                        if not success:
                            print('No success channel {}, shot {} '.format(\
                                   channel_path[-5:-1], shot_num))
                    except Exception as e:
                        print(e)
                        sys.stdout.flush()
                        print('Channel {}, shot {} missing.'.format(\
                               channel_path[-5:-1], shot_num))
                        success = False

                    if success:
                        data_two_column = np.vstack((time, data)).transpose()
                        np.savetxt(save_path, data_two_column, fmt='%.5e')
                    else:
                        np.savetxt(save_path, np.array([-1.0]), fmt='%.5e')
                    print('.', end='')

                except BaseException:
                    print('Could not save channel {}, shot {}.'.format(\
                           channel_path[-5:-1], shot_num))
                    print('Warning: Incomplete!!!')
                    raise
            sys.stdout.flush()
            if not success:
                missing_shots += 1
                print('Shot {} missing a signal in channel {}.'.format(\
                       shot_num, channel_path[-5:-1]))

    print('Finished with {} channel signals missing.'.format(missing_shots))
    return
                         

def Download_Shot_List(shot_numbers, channel_paths, max_cores = 8, server = 'atlas.gat.com'):
    """
    Accepts list of shots and downloads them in parallel

    Args:
        shot_numbers: list of integer shot numbers
        channel_paths: list of channel save path folders
        max_cores: int, max number of cores for parallelization
        server: MDSplus server, str. D3D server by default
    """
    sentinel = -1
    fn = partial(Download_Shot, channel_paths = channel_paths,\
                 sentinel = sentinel)
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
    def Acquire_Shots_D3D(self, shot_numbers, save_path = os.getcwd(), max_cores = 8):
        """
        Accepts a list of shot numbers and downloads the data, saving them into
        folders corresponding to the individual channels. Returns boolean 
        indicating success.

        Args:
            shot_numbers: list of integers, DIII-D shot numbers
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
        """
        # Construct channel save paths and create them if needed.
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            channel_path = os.path.join(save_path, self.ecei_channels[i])
            channel_paths.append(channel_path)
            if not os.path.exists(channel_path):
                os.mkdir(channel_path)

        try:
            c = MDS.Connection('atlas.gat.com')
        except Exception as e:
            print(e)
            return False

        Download_Shot_List(shot_numbers, channel_paths, max_cores = max_cores,\
                           server = 'atlas.gat.com')

        return True
