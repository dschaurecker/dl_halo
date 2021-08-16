from nbodykit.lab import *
from nbodykit import setup_logging
import time
import numpy as np
from nbodykit.source.catalog import BigFileCatalog, ArrayCatalog
from nbodykit.algorithms.paircount_tpcf.tpcf import SimulationBox2PCF
from nbodykit import CurrentMPIComm
from nbodykit import use_mpi
import illustris_python as il

########################################################################
#params
box_sz = 75./2
cube_sz = 1024

run = '2021_May_16_64061__pod_vgan_test_2048_64_loop' #name of run
state = 599 #state to load
sub = 0.2 #use only a random 20% subset of the catalog

bins = np.arange(box_sz/cube_sz, 2, box_sz/cube_sz/2) #search radius up to 2 Mpc/h

########################################################################

setup_logging("info")

comm = CurrentMPIComm.get()

use_mpi(comm=comm)

if comm.rank == 0:
    print('run: ', run)
    print('state: ', state, flush=True)

direc = '/scratch/ds6311/mywork/statistics/'+run+'/state_'+str(state)+'/'
direc_data = direc+'data/'

#illustris
basePath3='/scratch/ds6311/Illustris-3/dm_only'
basePath2='/scratch/ds6311/Illustris-2/dm_only'
#load illustris dm pos:
particles3 = il.snapshot.loadSubset(basePath3,135,'dm',['Coordinates']) / 1e3 #in Mpc/h
particles2 = il.snapshot.loadSubset(basePath2,135,'dm',['Coordinates']) / 1e3 #in Mpc/h
#select testing cube portion
particles3 = particles3[(particles3[:,0] >= box_sz) & (particles3[:,1] >= box_sz) & (particles3[:,2] >= box_sz)]
particles2 = particles2[(particles2[:,0] >= box_sz) & (particles2[:,1] >= box_sz) & (particles2[:,2] >= box_sz)]

subnum = [int(particles3.shape[0]*sub), int(particles2.shape[0]*sub)]

partss = [np.empty(subnum[0], dtype=[('Position', ('f8', 3))]), np.empty(subnum[1], dtype=[('Position', ('f8', 3))])]

#get sub cats from illustris
sub_idx = [0,0]
sub_idx[0] = np.random.randint(0, particles3.shape[0], size=subnum[0])
sub_idx[1] = np.random.randint(0, particles2.shape[0], size=subnum[1])

partss[0]['Position'] = particles3[sub_idx[0]]
partss[1]['Position'] = particles2[sub_idx[1]]

parts3 = ArrayCatalog(partss[0])
parts2 = ArrayCatalog(partss[1])

parts = [parts3, parts2]

del partss, particles3, particles2, sub_idx

#cats
f0 = BigFileCatalog(direc_data+'/Cat_Input_sub20', comm=comm)
f1 = BigFileCatalog(direc_data+'/Cat_Output_sub20', comm=comm)
f2 = BigFileCatalog(direc_data+'/Cat_Target_sub20', comm=comm)

n_inn = f0.size
rand0 = {}
rand0['Position'] = np.random.uniform(0, box_sz, size=(n_inn*10,3))
n_out = f1.size
rand1 = {}
rand1['Position'] = np.random.uniform(0, box_sz, size=(n_out*10,3))
n_tgt = f2.size
rand2 = {}
rand2['Position'] = np.random.uniform(0, box_sz, size=(n_tgt*10,3))

n_ill3 = parts[0].size
rand_ill3 = {}
rand_ill3['Position'] = np.random.uniform(0, box_sz, size=(n_ill3*10,3))
n_ill2 = parts[1].size
rand_ill2 = {}
rand_ill2['Position'] = np.random.uniform(0, box_sz, size=(n_ill2*10,3))

cat_rand0 = ArrayCatalog(rand0, comm=comm)
cat_rand1= ArrayCatalog(rand1, comm=comm)
cat_rand2 = ArrayCatalog(rand2, comm=comm)
cat_rand_ill3 = ArrayCatalog(rand_ill3, comm=comm)
cat_rand_ill2 = ArrayCatalog(rand_ill2, comm=comm)

#calc

if comm.rank == 0: print('f0 2pcf')
cf0 = SimulationBox2PCF('1d', data1=f0, randoms1=cat_rand0, edges=bins, BoxSize=box_sz, periodic=False, show_progress=True)
if comm.rank == 0: 
    np.save(direc_data+'2pcf_input_r', cf0.corr['r'])
    np.save(direc_data+'2pcf_input_corr', cf0.corr['corr'])
if comm.rank == 0: print('f1 2pcf')
cf1 = SimulationBox2PCF('1d', data1=f1, randoms1=cat_rand1, edges=bins, BoxSize=box_sz, periodic=False, show_progress=True)
if comm.rank == 0: 
    np.save(direc_data+'2pcf_output_r', cf1.corr['r'])
    np.save(direc_data+'2pcf_output_corr', cf1.corr['corr'])
if comm.rank == 0: print('f2 2pcf')
cf2 = SimulationBox2PCF('1d', data1=f2, randoms1=cat_rand2, edges=bins, BoxSize=box_sz, periodic=False, show_progress=True)
if comm.rank == 0: 
    np.save(direc_data+'2pcf_target_r', cf2.corr['r'])
    np.save(direc_data+'2pcf_target_corr', cf2.corr['corr'])

if comm.rank == 0: print('ill3 2pcf')
cf0_ill = SimulationBox2PCF('1d', data1=f0_ill, randoms1=cat_rand_ill3, edges=bins, BoxSize=box_sz, periodic=False, show_progress=True)
if comm.rank == 0: 
    np.save(direc_data+'2pcf_ill3_r', cf0_ill.corr['r'])
    np.save(direc_data+'2pcf_ill3_corr', cf0_ill.corr['corr'])
if comm.rank == 0: print('ill2 2pcf')
cf1_ill = SimulationBox2PCF('1d', data1=f1_ill, randoms1=cat_rand_ill2, edges=bins, BoxSize=box_sz, periodic=False, show_progress=True)
if comm.rank == 0: 
    np.save(direc_data+'2pcf_ill2_r', cf1_ill.corr['r'])
    np.save(direc_data+'2pcf_ill2_corr', cf1_ill.corr['corr'])