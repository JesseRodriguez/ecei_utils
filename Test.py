from ECEI import ECEI
import os
import MDSplus as MDS

clear_path = os.path.join(os.getcwd(), 'd3d_clear_since_2016.txt')
disrupt_path = os.path.join(os.getcwd(), 'd3d_disrupt_since_2016.txt')

E = ECEI()
E.Clean_Channel_Dirs()
E.Acquire_Shot_Sequence_D3D(2, 154055, clear_path, disrupt_path, max_cores = 3)