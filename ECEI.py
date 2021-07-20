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
import h5py
import scipy.signal
import math

###############################################################################
## Utility Functions and Globals
###############################################################################
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


def Download_Shot(shot_num_queue, c, channel_paths, sentinel = -1,\
                  verbose = False, d_sample = 1):
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
        d_sample: int, downsample factor, MUST BE IN FORM 10^y
    """
    missing_shots = 0
    while True:
        shot_num = shot_num_queue.get()
        if shot_num == sentinel:
            break
        shot_complete = True
        for channel_path in channel_paths:
            save_path = channel_path[:-9]+'{}.hdf5'.format(int(shot_num))
            channel = channel_path[-9:]

            success = False
            if os.path.isfile(save_path):
                if os.path.getsize(save_path) > 0:
                    f = h5py.File(save_path, 'r')
                    for key in f.keys():
                        if key == channel:
                            success = True
                    f.close()
                else:
                    print('Shot {} '.format(int(shot_num)),'was downloaded \
                           incorrectly (empty file). Redownloading.')

            else:
                f = h5py.File(save_path, 'w')
                f.close()

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
                        if d_sample > 1:
                            n = int(math.log10(d_sample))
                            for _ in range(n):
                                data_two_column = scipy.signal.decimate(data_two_column,\
                                                  10, axis = 0)
                        f = h5py.File(save_path, 'r+')
                        for key in f.keys():
                            if key.startswith('missing'):
                                if key[8:] == channel:
                                    del f[key]
                        dset = f.create_dataset(channel, data = data_two_column)
                        f.close()
                    else:
                        f = h5py.File(save_path, 'r+')
                        dsetk = 'missing_'+channel
                        already = False
                        for key in f.keys():
                            if key == dsetk:
                                f.close()
                                already = True
                        if not already:
                            dset = f.create_dataset(dsetk, data = np.array([-1.0]))
                            f.close()

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
                       server = 'atlas.gat.com', verbose = False,\
                       d_sample = 1):
    """
    Accepts list of shots and downloads them in parallel

    Args:
        shot_numbers: list of integer shot numbers
        channel_paths: list of channel save path folders
        max_cores: int, max number of cores for parallelization
        server: MDSplus server, str. D3D server by default
        verbose: bool, suppress print statements
        d_sample: int, downsample factor, MUST BE IN FORM 10^y
    """
    sentinel = -1
    fn = partial(Download_Shot, channel_paths = channel_paths,\
                 sentinel = sentinel, verbose = verbose,\
                 d_sample = d_sample)
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


def Count_Missing(shot_list, shot_path, missing_path):
    """
    Accepts a list of all channel paths and produces an up-to-date list of all
    missing data and places it in missing_path

    Args:
        shot_list: 1-D numpy array of DIII-D shot numbers
        shot_path: path to folder containing shot files
        missing_path: folder for missing shot reports
    """
    min_shot = np.argmin(shot_list)
    max_shot = np.argmax(shot_list)
    num_shots = np.shape(shot_list)[0]*160
    num_shots_miss = 0
    print("Generating report for {} shots between shots {} and {}".format(\
           np.shape(shot_list)[0], int(shot_list[min_shot]),\
           int(shot_list[max_shot])))
    report = open(missing_path+'/missing_report_'+str(int(shot_list[min_shot]))+'-'+\
                  str(int(shot_list[max_shot]))+'.txt', mode = 'w',\
                  encoding='utf-8')
    report.write('Missing channel signals for download from shot {} to shot {}:\n'.\
                  format(int(shot_list[min_shot]), int(shot_list[max_shot])))
    for filename in os.listdir(shot_path):
        if filename.endswith('hdf5'):
            if int(filename[:-5]) >= shot_list[min_shot] and int(filename[:-5]) <=\
                                                             shot_list[max_shot]:
                f = h5py.File(shot_path+'/'+filename, 'r')
                count = 0
                for key in f.keys():
                    if key.startswith('missing'):
                        count += 1
                        report.write('Channel '+key[-5:-1]+', shot #'+filename[:-5]+'\n')
                        num_shots_miss +=1
                if count > 160:
                    print('Shot #'+filename[:-5]+' has more than 160 channels missing!')

    report.write('Missing channel signals for {} out of {} signals.'.\
                  format(num_shots_miss, num_shots))
    report.close()

    return (num_shots_miss, num_shots)


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
                self.ecei_channels.append('"LFS{:02d}{:02d}"'.format(i+3,j+1))

    ###########################################################################
    ## Data Processing
    ###########################################################################
    def Viz(self):
        """
        Visualization function
        
        Args:
        """
        return 0


    ###########################################################################
    ## Visualization
    ###########################################################################
    def Generate_Txt(self, shot, channel, save_dir = os.getcwd()):
        """
        Get a .txt file out for reading signal data

        Args:
            shot: int, shot number
            channel: str, format "XXYY", 03<=XX<=22, 01<=YY<=08, designates
                     channel
            save_dir: str, directory where shot files are stored
        """
        shot_s = str(shot)
        for filename in os.listdir(save_dir):
            if filename.startswith(shot_s):
                f = hdf5.File(save_dir+'/'+filename, 'r')
                data = f.get('"LFS'+channel+'"')
                np.savetxt(save_dir+'/'+shot_s+'_chan'+channel+'.txt', data)

        return

    
    def Generate_Txt_Interactive(self, save_dir = os.getcwd()):
        """
        Get a .txt file out for reading signal data, accepts input from command
        line.

        Args:
            save_dir: str, directory where shot files are stored
        """
        shot = int(input("Which shot? Enter an integer.\n"))
        channel = input("Which channel? format 'XXYY', 03<=XX<=22, 01<=YY<=08.\n")

        self.Generate_Txt(shot, channel, save_dir)

        return


    ###########################################################################
    ## Data Acquisition
    ###########################################################################
    def Acquire_Shots_D3D(self, shot_numbers, save_path = os.getcwd(),\
                          max_cores = 8, verbose = False, chan_lowlim = 3,\
                          chan_uplim = 22, d_sample = 1):
        """
        Accepts a list of shot numbers and downloads the data, saving them into
        folders corresponding to the individual channels. Returns nothing.

        Args:
            shot_numbers: 1-D numpy array of integers, DIII-D shot numbers
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress most print statements
            chan_lowlim: int, lower limit of subset of channels to download
            chan_uplim: int, upper limit of subset of channels to download
            d_sample: int, downsample factor, MUST BE IN FORM 10^y
        """
        t_b = time.time()
        # Construct channel save paths and create them if needed.
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            XX = int(self.ecei_channels[i][-5:-3])
            if XX >= chan_lowlim and XX <= chan_uplim:
                channel_path = os.path.join(save_path, self.ecei_channels[i])
                channel_paths.append(channel_path)
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
                           server = 'atlas.gat.com', verbose = verbose,\
                           d_sample = d_sample)

        missed = Count_Missing(shot_numbers, save_path, missing_path)

        t_e = time.time()
        T = t_e-t_b

        print("Downloaded {} out of {} signals in {} seconds.".format(missed[1]-missed[0],\
               missed[1], T))

        return


    def Acquire_Shot_Sequence_D3D(self, shots, shot_1, clear_file, disrupt_file,\
                                  save_path = os.getcwd(), max_cores = 8,\
                                  verbose = False, chan_lowlim = 3, chan_uplim = 22,\
                                  d_sample = 1):
        """
        Accepts a desired number of non-disruptive shots, then downloads all
        shots in our labelled database up to the last non-disruptive shot.
        Returns nothing. Shots are saved in hdf5 format, downsampling is done
        BEFORE saving. Each channel is labelled within its own dataset in the
        hdf5 file, where the label is the channel name/MDS point name, e.g.
        '"LFSXXYY"'.

        Args:
            shots: int, number of non-disruptive shots you want to download
            shot_1: int, the shot number you want to start with
            clear_file: The path to the clear shot list
            disrupt_file: The path to the disruptive shot list
            save_path: location where the channel folders will be stored,
                       current directory by default
            max_cores: int, max # of cores to carry out download tasks
            verbose: bool, suppress some exception info
            chan_lowlim: int, lower limit of subset of channels to download
            chan_uplim: int, upper limit of subset of channels to download
            d_sample: int, downsample factor, MUST BE IN FORM 10^y
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

        if start_c + shots > clear_shots.shape[0]-1:
            shots = clear_shots.shape[0] - start_c - 1

        shot_list = np.array([clear_shots[start_c,0]])
        for i in range(shots-1):
            shot_list = np.append(shot_list, [clear_shots[i+start_c+1,0]])

        last = False
        no_disrupt = False
        i = start_d
        while not last:
            if disrupt_shots[i,0] >= clear_shots[start_c+shots-1,0]:
                end_d = i
                last = True
            i += 1
            if i >= disrupt_shots.shape[0]:
                no_disrupt = True
                last = True

        if not no_disrupt:
            for i in range(end_d - start_d + 1):
                shot_list = np.append(shot_list, [disrupt_shots[i+start_d,0]])

        self.Acquire_Shots_D3D(shot_list, save_path, max_cores, verbose,\
                               chan_lowlim, chan_uplim, d_sample)

        return


    def Generate_Missing_Report(self, shots, shot_1, clear_file, disrupt_file,\
                                save_path = os.getcwd()):
        """
        Accept a start shot and a number of clear shots and generate a missing
        shot report for all shots in that range.

        Args:
            shots: int, number of non-disruptive shots you want to download
            shot_1: int, the shot number you want to start with
            clear_file: The path to the clear shot list
            disrupt_file: The path to the disruptive shot list
            save_path: location where the shot hdf5 files will be stored,
                       current directory by default
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

        if start_c + shots > clear_shots.shape[0]-1:
            shots = clear_shots.shape[0] - start_c - 1

        shot_list = np.array([clear_shots[start_c,0]])
        for i in range(shots-1):
            shot_list = np.append(shot_list, [clear_shots[i+start_c+1,0]])

        last = False
        no_disrupt = False
        i = start_d
        while not last:
            if disrupt_shots[i,0] >= clear_shots[start_c+shots-1,0]:
                end_d = i
                last = True
            i += 1
            if i >= disrupt_shots.shape[0]:
                no_disrupt = True
                last = True

        if not no_disrupt:
            for i in range(end_d - start_d + 1):
                shot_list = np.append(shot_list, [disrupt_shots[i+start_d,0]])
            
        channel_paths = []
        for i in range(len(self.ecei_channels)):
            channel_path = os.path.join(save_path, self.ecei_channels[i])
            channel_paths.append(channel_path)
        #Missing shots directory
        missing_path = os.path.join(save_path, 'missing_shot_info')
        if not os.path.exists(missing_path):
            os.mkdir(missing_path)

        missed = Count_Missing(shot_list, save_path, missing_path)

        return


    def Clean_Signals(self, save_path = os.getcwd()):
        """
        Removes all signal files in the save_path directory.
        """
        check = input("WARNING: this function will delete ALL signal files in the "+\
                "designated save path. Type 'yes' to continue, anything else to cancel.\n")
        if check == 'yes':
            for filename in os.listdir(save_path):
                if filename.endswith('hdf5'):
                    shot = os.path.join(save_path, filename)
                    os.remove(shot)


    def Clean_Missing_Signals(self, save_path = os.getcwd()):
        """
        Removes all signal files with all channels missing in the save_path directory.
        """
        check = input("WARNING: this function will delete ALL signal files in the "+\
                      "designated save path which have all channel signals missing. "+\
                      "Type 'yes' to continue, anything else to cancel.\n")
        if check == 'yes':
            for filename in os.listdir(save_path):
                if filename.endswith('hdf5'):
                    shot = os.path.join(save_path, filename)
                    f = h5py.File(shot, 'r')
                    count = 0
                    for key in f.keys():
                        if key.startswith('missing'):
                            count += 1
                    if count == 160:
                        f.close()
                        if os.path.getsize(shot) <= 342289:
                            os.remove(shot)
                    else:
                        f.close()


    def Clean_Missing_Signal(self, shot_file):
        """
        Removes a single shot file if it has all channel signals missing.
        """
        shot = os.path.join(os.getcwd(), shot_file)
        f = h5py.File(shot, 'r')
        count = 0
        for key in f.keys():
            if key.startswith('missing'):
                count += 1
        if count == 160:
            f.close()
            check = input("You are about to delete "+shot+". Are "+\
                          "you sure about that?\n")
            if check == 'yes':
                os.remove(shot)
        else:
            f.close()


    def Clean_Missing_Signal_List(self, shots):
        """
        Removes shot files in a list if they have all channel signals missing.
        """
        for s in shots:
            shot_file = str(s)+".hdf5"
            shot = os.path.join(os.getcwd(), shot_file)
            f = h5py.File(shot, 'r')
            count = 0
            for key in f.keys():
                if key.startswith('missing'):
                    count += 1
            if count == 160:
                f.close()
                check = input("You are about to delete "+shot+". Are "+\
                              "you sure about that?\n")
                if check == 'yes':
                    os.remove(shot)
            else:
                f.close()
