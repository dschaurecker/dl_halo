import numpy as np
import os
from nbodykit.source.catalog import ArrayCatalog
import time
from argparse import ArgumentParser
from torch.multiprocessing import spawn # can also use standard python multiprocessing
import torch

def main():
    node_worker()

def node_worker():
    nodes = 0
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
    cpus_per_node = int(os.environ['SLURM_CPUS_PER_TASK'])
    world_size = nodes * cpus_per_node

    node = int(os.environ['SLURM_NODEID'])
    
    #make parser
    parser = ArgumentParser(description='create statistics')

    parser.add_argument('--counts', action='store_true', help='eval stat for counts fields')
    parser.add_argument('--density', action='store_true', help='eval stat for density fields')
    parser.add_argument('--overdensity', action='store_true', help='eval stat for overdensity fields')
    parser.add_argument('--run-name', type=str, default='no_name', help='enter name of run to test')
    parser.add_argument('--pixside', type=int, default=8192, help='number of pixels of whole sim along one dim')
    parser.add_argument('--cubesz', type=int, default=64, help='number of pixels of training cube along one dim')
    parser.add_argument('--state-num', type=int, default=1, help='number of state to use, eg. 600 for state_600.pt')

    args = parser.parse_args()

    run = args.run_name
    statenum = args.state_num

    pixside = args.pixside
    testcube_sz = int(pixside/2)
    cubesz = args.cubesz
    box_sz = 75./2 #for testing region only
    
    if node != 0: time.sleep(5)

    print(run)

    direc = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/'

    #create dirs
    if not os.path.exists(direc):
        os.makedirs(direc)

    direc_data = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/data/'

    if not os.path.exists(direc_data):
        os.makedirs(direc_data)

    direc_plots = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(statenum)+'/plots/'

    if not os.path.exists(direc_plots):
        os.makedirs(direc_plots)

    run_path = '/scratch/ds6311/mywork/scripts/test/'+run+'/state_'+str(statenum)+'/'
    
    if not os.path.exists(direc_data+'cpu_cats/'):
        os.makedirs(direc_data+'cpu_cats/')

    print('current node', node)
    print('cpus per node', cpus_per_node)
    print('world_size', world_size)

    spawn(cpu_worker, args=(node, cpus_per_node, world_size, args, run_path, direc_data, pixside, testcube_sz, cubesz, box_sz), nprocs=cpus_per_node)


def cpu_worker(local_rank, node, cpus_per_node, world_size, args, run_path, direc_data, pixside, testcube_sz, cubesz, box_sz):

    device = torch.device('cpu')

    rank = cpus_per_node * node + local_rank
    
    ################# FUNCTIONS

    def myexp(x):
        return np.exp(x) - 1e-7
    def mylog(x):
        return np.log(1e-7 + x)
    def mylog1p(x):
        return np.log1p(1e-7 + x)
    def PoissonSample(cub):
        floor = np.floor(cub)
        diff = cub - floor
        rand = np.random.rand(cub.shape[0], cub.shape[1], cub.shape[2], cub.shape[3])
        #print(diff[0][0,510,0])
        return np.where(rand < diff, floor + 1, floor).astype('int')
    def unmakeDensity(cub):
        dm_particle_mass1 = 7.5e6 #sunmasses, 7.5e6 for Ill-1-Dark, 6.0e7 for Ill-2-Dark
        dm_particle_mass2 = 6e7 #sunmasses, 7.5e6 for Ill-1-Dark, 6.0e7 for Ill-2-Dark
        l_box = 75 #Mpc/h
        why = np.copy(cub)
        if why.shape[0] == 3:
            why[0] /= dm_particle_mass2/(l_box/pixside)**3 #convert to density Ill2
            why[1] /= dm_particle_mass1/(l_box/pixside)**3 #convert to density Ill1
            why[2] /= dm_particle_mass1/(l_box/pixside)**3 #convert to density Ill1
        else:
            print('Weird input (not shape input/output/target)')
            why /= dm_particle_mass1/(l_box/pixside)**3 #convert to density
        return why
    def unmakeOverdensity(cub): #input overdens, output number count
        mean_dens = 107174542222. #from Illustris 1 AND 2 (they have the same)
        dens = np.copy(cub)
        dens = dens * mean_dens + mean_dens
        return unmakeDensity(dens)
    def getDensity(cub, field): #number count input
        l_box = 75 #Mpc/h
        if field == 0:
            return cub * 6e7/(l_box/pixside)**3  #Illustris 2
        else:
            return cub * 7.5e6/(l_box/pixside)**3  #Illustris 1
    def getOverdensity(cub, field): #count input
        dens = getDensity(cub, field) 
        mean_dens = np.mean(dens)
        return (dens - mean_dens) / mean_dens
    def getOverdensityfromDensity(cub, field): #count input
        dens = cub
        mean_dens = np.mean(dens)
        return (dens - mean_dens) / mean_dens


    ########################################################
    
    np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning) #ignore deprication warnings
    
    
    data = [np.empty(0, dtype=[('Position', ('f8', 3))]), np.empty(0, dtype=[('Position', ('f8', 3))]), np.empty(0, dtype=[('Position', ('f8', 3))])]
    
    list_cubes_with_paths = []
    list_subfolders_with_paths_0 = [f.path for f in os.scandir(run_path) if f.is_dir()] #scan test cubes (might just be one)
    for direc in list_subfolders_with_paths_0:
        list_subfolders_with_paths_1 = [f.path for f in os.scandir(direc) if f.is_dir()] #scan subfolders 0-9
        for direc2 in list_subfolders_with_paths_1:
            list_cubes_with_paths.extend([f.path for f in os.scandir(direc2) if f.is_file()]) #scan files

    # split workload among ranks
    names = np.array_split(np.array(list_cubes_with_paths), world_size)[rank].tolist()
                
    for (cubecounter, cubefile) in enumerate(names):
        cubepath = cubefile[:-4] #remove .npz from name
        name = cubepath.split('/')[-1]
        elem = name.split('_')
        x_start = int(elem[1])
        y_start = int(elem[2])
        z_start = int(elem[3])
        cub = np.load(cubefile)
        cube = np.zeros((3,cubesz,cubesz,cubesz))
        
        cube[0] = cub['input'][0,0]
        cube[1] = cub['output'][0,0]
        cube[2] = cub['target'][0,0]
        if args.density: cube = unmakeDensity(cube)
        if args.overdensity: cube = unmakeOverdensity(cube)
        cube = cube.clip(min=0)
        cube = PoissonSample(cube)
                
        n_part = [np.sum(cube[0,:,:,:]), np.sum(cube[1,:,:,:]), np.sum(cube[2,:,:,:])]

        data_cube = [np.empty(n_part[0], dtype=[('Position', ('f8', 3))]), np.empty(n_part[1], dtype=[('Position', ('f8', 3))]), np.empty(n_part[2], dtype=[('Position', ('f8', 3))])]
        for i in range(3):
            data_cube[i]['Position'] = np.random.rand(n_part[i],3)
        
        #python loops are slow but fast enough for this
        """
            Go to each pixel and place a particle for each number count randomly inside this pixel's volume.
            Add each position to the catalog and create one catalog for each rank.
            All catalogs will later get unified in the next step, eval_cat.py.
        """
        for i in range(3): #loop for input, output, target
            part_counter = 0
            for n in range(cubesz):
                for m in range(cubesz):
                    for l in range(cubesz):
                        if cube[i,n,m,l] != 0:
                            x_pos = x_start + n #in pixels
                            y_pos = y_start + m
                            z_pos = z_start + l
                            data_cube[i]['Position'][part_counter:part_counter+cube[i,n,m,l],:] += [x_pos,y_pos,z_pos]
                            data_cube[i]['Position'][part_counter:part_counter+cube[i,n,m,l],:] *= (box_sz/testcube_sz)
                            part_counter += cube[i,n,m,l]

        for i in range(3):
            data[i] = np.concatenate((data[i],data_cube[i]))

    
    np.save(direc_data+'cpu_cats/cpu_'+str(rank)+'.npy', np.array(data))
    print('rank', rank, 'finished.')
    
if __name__ == '__main__':
    main()