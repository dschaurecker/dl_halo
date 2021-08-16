import sys
from pprint import pprint
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .data import FieldDataset, FieldDataset_zoom
from . import models
from .models import narrow_cast
from .utils import import_attr, load_model_state_dict

import time
from datetime import datetime
from torch.multiprocessing import spawn
import os


def node_worker(args):
    if 'SLURM_STEP_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_STEP_NUM_NODES'])
    elif 'SLURM_JOB_NUM_NODES' in os.environ:
        args.nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    else:
        raise KeyError('missing node counts in slurm env')
        
    args.cpus_per_node = int(os.environ['SLURM_CPUS_PER_TASK'])
    args.world_size = args.nodes * args.cpus_per_node

    node = int(os.environ['SLURM_NODEID'])
    
    run_name = args.train_run_name #name of training run that's loaded
    
    #make dir for saving the epoch states
    if not os.path.exists('test'):
        os.mkdir('test')
    now = datetime.now()
    save_to = run_name + '_' + str(os.environ['SLURM_JOB_NAME']) + '/' + 'state_' + str(args.state_num) + '/'
    #save_to = 'name_you_want' #set manually
    
    if node != 0: time.sleep(5) #wait 5 seconds for node 0 to create folders
    
    folder_path = './test/'+save_to
    in_path = args.in_folder #eg. /scratch/ds6311/Illustris-3/dm_only_2048_counts_arrays_subcubes_128/cic/cube7_1024_1024_1024/*/*'
    
    #actually use glob import here
    eigth = in_path.split('/')[-3]
    
    subpath = folder_path+eigth+'/'
    
    save_to = subpath
    
    spawn(cpu_worker, args=(node, args, save_to, run_name), nprocs=args.cpus_per_node)


def cpu_worker(local_rank, node, args, save_to, run_name):

    device = torch.device('cpu')

    torch.set_num_threads(1)

    rank = args.cpus_per_node * node + local_rank
    
    #create folders
    if rank != 0:
        time.sleep(5) #wait 5 seconds for rank 0 to create all folders
    else:
        #create 10 testcube subfolders
        for i in range(10):
            if not os.path.exists(save_to+str(i)+'/'):
                os.makedirs(save_to+str(i)+'/')
        
        #print for slurm output:
        print('Run Name: ', run_name)
        print('save_to:', save_to)
        print()
    
        pprint(vars(args))
        sys.stdout.flush()
    
    test_in_pattern = [args.in_folder]
    test_tgt_pattern = [args.tgt_folder]

    test_dataset = FieldDataset_zoom(
        in_patterns=test_in_pattern,
        tgt_patterns=test_tgt_pattern,
        in_norms=args.in_norms,
        tgt_norms=args.tgt_norms,
        callback_at=args.callback_at,
        augment=False,
        aug_shift=None,
        aug_add=None,
        aug_mul=None,
        crop=args.crop,
        crop_start=args.crop_start,
        crop_stop=args.crop_stop,
        crop_step=args.crop_step,
        pad=args.pad,
        scale_factor=args.scale_factor,
    )
    
    n_data = len(test_dataset)
    rank_indices = np.array_split(np.arange(n_data), args.world_size)[rank].tolist()
    smaller_test_set = Subset(test_dataset, rank_indices)
    
    test_loader = DataLoader(
        smaller_test_set,
        batch_size=args.batches,
        shuffle=False,
        num_workers=args.loader_workers,
    )

    in_chan, out_chan = test_dataset.in_chan, test_dataset.tgt_chan

    model = import_attr(args.model, models, callback_at=args.callback_at)
    model = model(sum(in_chan), sum(out_chan)) #, scale_factor=args.scale_factor
    try:
        criterion = import_attr(args.criterion, torch.nn, callback_at=args.callback_at)
    except:
        criterion = getattr(losses, args.criterion)
    criterion = criterion()

    device = torch.device('cpu')
    state = torch.load(args.load_state, map_location=device)
    load_model_state_dict(model, state['model'], strict=args.load_state_strict)
    del state

    model.eval()

    with torch.no_grad():
        for i, (input, target, in_fpath, tgt_fpath) in enumerate(test_loader):
            #get filename of test_cube (batchsize == 1)
            sample_fname_tot = in_fpath[0][0].split('/')[-1] #get name of 1st in_field
            sample_fname = sample_fname_tot[:-4] #remove .npy ending
            
            sample_subfolder = in_fpath[0][0].split('/')[-2] #a number between [0,9]
            
            """#check if cube already exists, if yes: skip testing cube
            if sample_fname_tot in np.array(f):
                continue"""
            
            output = model(input)
            
            if i == 0 and rank == 0:
                print('input shape :', input.shape)
                print('output shape :', output.shape)
                print('target shape :', target.shape)

            input, output, target = narrow_cast(input, output, target)            
            
            #undo norms from data before saving
            if args.in_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.in_norms, np.cumsum(in_chan)):
                    norm(input[:, start:stop], undo=True)
                    start = stop
            if args.tgt_norms is not None:
                start = 0
                for norm, stop in zip(test_dataset.tgt_norms, np.cumsum(out_chan)):
                    norm(output[:, start:stop], undo=True)
                    norm(target[:, start:stop], undo=True)
                    start = stop
         
                    
            #save cube to 0-9 subfolder
            pathh = save_to+sample_subfolder+'/'
            np.savez(pathh+'{}.npz'.format(sample_fname), input=input.numpy(),
                            output=output.numpy(), target=target.numpy())
            

