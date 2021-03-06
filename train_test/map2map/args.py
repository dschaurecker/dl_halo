import os
import argparse
import json
import warnings

from .train import ckpt_link


def get_args():
    """Parse arguments and set runtime defaults.
    """
    parser = argparse.ArgumentParser(
        description='Transform field(s) to field(s)')

    subparsers = parser.add_subparsers(title='modes', dest='mode', required=True)
    train_parser = subparsers.add_parser(
        'train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    test_parser = subparsers.add_parser(
        'test',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    add_train_args(train_parser)
    add_test_args(test_parser)


    args = parser.parse_args()

    if args.mode == 'train':
        set_train_args(args)
    elif args.mode == 'test':
        set_test_args(args)
    
    return args


def add_common_args(parser):
    parser.add_argument('--in-norms', type=str_list, help='comma-sep. list '
            'of input normalization functions')
    parser.add_argument('--tgt-norms', type=str_list, help='comma-sep. list '
            'of target normalization functions')
    parser.add_argument('--crop', type=int,
            help='size to crop the input and target data. Default is the '
            'field size')
    parser.add_argument('--crop-start', type=int,
            help='starting point of the first crop. Default is the origin')
    parser.add_argument('--crop-stop', type=int,
            help='stopping point of the last crop. Default is the opposite '
            'corner to the origin')
    parser.add_argument('--crop-step', type=int,
            help='spacing between crops. Default is the crop size')
    parser.add_argument('--pad', default=0, type=int,
            help='size to pad the input data beyond the crop size, assuming '
            'periodic boundary condition')
    parser.add_argument('--scale-factor', default=1, type=int,
            help='upsampling factor for super-resolution, in which case '
            'crop and pad are sizes of the input resolution')

    parser.add_argument('--model', type=str, required=True,
            help='(generator) model')
    parser.add_argument('--criterion', default='MSELoss', type=str,
            help='loss function')
    parser.add_argument('--load-state', default=ckpt_link, type=str,
            help='path to load the states of model, optimizer, rng, etc. '
            'Default is the checkpoint. '
            'Start from scratch in case of empty string or missing checkpoint')
    parser.add_argument('--load-state-non-strict', action='store_false',
            help='allow incompatible keys when loading model states',
            dest='load_state_strict')

    parser.add_argument('--batches', type=int, required=True,
            help='mini-batch size, per GPU in training or in total in testing')
    parser.add_argument('--loader-workers', default=-8, type=int,
            help='number of subprocesses per data loader. '
            '0 to disable multiprocessing; '
            'negative number to multiply by the batch size')

    parser.add_argument('--callback-at', type=lambda s: os.path.abspath(s),
            help='directory of custorm code defining callbacks for models, '
            'norms, criteria, and optimizers. Disabled if not set. '
            'This is appended to the default locations, '
            'thus has the lowest priority.')


def add_train_args(parser):
    add_common_args(parser)

    parser.add_argument('--train-in-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training input data')
    parser.add_argument('--train-tgt-patterns', type=str_list, required=True,
            help='comma-sep. list of glob patterns for training target data')
    parser.add_argument('--val-in-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation input data')
    parser.add_argument('--val-tgt-patterns', type=str_list,
            help='comma-sep. list of glob patterns for validation target data')
    parser.add_argument('--augment', action='store_true',
            help='enable data augmentation of axis flipping and permutation')
    parser.add_argument('--aug-shift', type=int,
            help='data augmentation by shifting [0, aug_shift) pixels, '
            'useful for models that treat neighboring pixels differently, '
            'e.g. with strided convolutions')
    parser.add_argument('--aug-add', type=float,
            help='additive data augmentation, (normal) std, '
            'same factor for all fields')
    parser.add_argument('--aug-mul', type=float,
            help='multiplicative data augmentation, (log-normal) std, '
            'same factor for all fields')

    parser.add_argument('--adv-model', type=str,
            help='adversary (discriminator) model, disabled by default')
    parser.add_argument('--adv-model-spectral-norm', action='store_true',
            help='enable spectral normalization on the adversary model')
    parser.add_argument('--adv-criterion', default='BCEWithLogitsLoss', type=str,
            help='adversarial loss function')
    parser.add_argument('--adv-r1-reg-interval', default=0, type=int,
            help='interval (batches) between R1 regularization. '
            'Disabled if non-positive; '
            'lazy regularization if greater than 1 (not every batch)')
    parser.add_argument('--adv-iter-ratio', default=1, type=int,
            help='discriminator iterations per generator update')
    parser.add_argument('--adv-wgan-gp-interval', default=0, type=int,
            help='interval (batches) between WGAN gradient penalty. '
            'Disabled if non-positive; '
            'lazy regularization if greater than 1 (not every batch)')
    parser.add_argument('--cgan', action='store_true',
            help='enable conditional GAN')
    parser.add_argument('--adv-start', default=0, type=int,
            help='epoch to start adversarial training')
    parser.add_argument('--adv-label-smoothing', default=1, type=float,
            help='label of real samples for the adversary model, '
            'e.g. 0.9 for label smoothing and 1 to disable')
    parser.add_argument('--instance-noise', default=0, type=float,
            help='noise added to the adversary inputs to stabilize training')
    parser.add_argument('--instance-noise-batches', default=1e4, type=float,
            help='noise annealing duration')
    parser.add_argument('--instance-noise-mode', default=None, type=str,
            help='mode either max, std or None. How noise should be rescaled')
    parser.add_argument('--instance-noise-percent', default=0.02, type=float,
            help='percent of max/std target value that is used to rescale noise')

    parser.add_argument('--optimizer', default='Adam', type=str,
            help='optimization algorithm')
    parser.add_argument('--lr', type=float, required=True,
            help='initial learning rate')
    parser.add_argument('--optimizer-args', default='{}', type=str,
            help='optimizer arguments in addition to the learning rate, '
            'e.g. --optimizer-args \'{"betas": [0., 0.9]}\'')
    parser.add_argument('--adv-lr', type=float,
            help='initial adversary learning rate, default to --lr')
    parser.add_argument('--adv-optimizer-args', type=str,
            help='adversary optimizer arguments, default to --optimizer-args')
    parser.add_argument('--reduce-lr-on-plateau', action='store_true',
            help='Enable ReduceLROnPlateau learning rate scheduler')
    parser.add_argument('--scheduler-args', default='{"verbose": true}',
            type=json.loads,
            help='arguments for the ReduceLROnPlateau scheduler')
    parser.add_argument('--init-weight-std', type=float,
            help='weight initialization std')
    parser.add_argument('--epochs', default=128, type=int,
            help='total number of epochs to run')
    parser.add_argument('--seed', default=42, type=int,
            help='seed for initializing training')

    parser.add_argument('--div-data', action='store_true',
            help='enable data division among GPUs for better page caching. '
            'Data division is shuffled every epoch. '
            'Only relevant if there are multiple crops in each field')
    parser.add_argument('--div-shuffle-dist', default=1, type=float,
            help='distance to further shuffle cropped samples relative to '
            'their fields, to be used with --div-data. '
            'Only relevant if there are multiple crops in each file. '
            'The order of each sample is randomly displaced by this value. '
            'Setting it to 0 turn off this randomization, and setting it to N '
            'limits the shuffling within a distance of N files. '
            'Change this to balance cache locality and stochasticity')
    parser.add_argument('--dist-backend', default='nccl', type=str,
            choices=['gloo', 'nccl'], help='distributed backend')
    parser.add_argument('--dist-address', default='', type=str,
                        help='dist addr for connecting ranks and nodes.')
    parser.add_argument('--log-interval', default=100, type=int,
            help='interval (batches) between logging training loss')
    
    parser.add_argument('--percentile', default=1., type=float,
            help='determine how many percent of the hole training set should be loaded for each epoch.'
                       'percentile*int must be equals 1 for simplicity.'
                       'eg. 0.25: every 4 epochs the same training subset will be used'
                       'for training so the subset is not '
                       'randomized at each epoch')
    
    
    
    parser.add_argument('--intype', default=3, type=int,
            help='type of input simulation, eg Illustris-2, or Illustris-3')
    parser.add_argument('--boxsize', default=1024, type=int,
            help='number of voxels per side that the entire simulation is divided into, eg. 1024, 2048 or 8192')

    parser.add_argument('--incr-adv-lr', default=1., type=float,
            help='increase adv lr by percentage eg. 1.1 evry 10 epochs')
    
    parser.add_argument('--randnumber', default=0, type=int, required=False,
            help='random number to be determined in slurmscript to identify run')

def add_test_args(parser):
    add_common_args(parser)

    parser.add_argument('--test-in-patterns', type=str_list, required=False,
            help='comma-sep. list of glob patterns for test input data')
    parser.add_argument('--test-tgt-patterns', type=str_list, required=False,
            help='comma-sep. list of glob patterns for test target data')
    parser.add_argument('--train-run-name', type=str, default='no_name',
            help='enter name for output')
    parser.add_argument('--cube-name', type=str, default='no_name',
            help='enter name for output')
    parser.add_argument('--in-folder', type=str, default='no_name',
            help='enter name of in-cube testing folder')
    parser.add_argument('--tgt-folder', type=str, default='no_name',
            help='enter name of tgt-cube testing folder')
    parser.add_argument('--state-num', type=int, required=True,
            help='which trained state to load, eg.: 100')

    
    

def str_list(s):
    return s.split(',')


def set_common_args(args):
    if args.loader_workers < 0:
        args.loader_workers *= - args.batches


def set_train_args(args):
    set_common_args(args)

    args.val = args.val_in_patterns is not None and \
            args.val_tgt_patterns is not None

    args.adv = args.adv_model is not None

    if args.adv:
        if args.adv_lr is None:
            args.adv_lr = args.lr
        if args.adv_optimizer_args is None:
            args.adv_optimizer_args = args.optimizer_args

    if args.cgan and not args.adv:
        args.cgan =False
        warnings.warn('Disabling cgan given adversary is disabled',
                      RuntimeWarning)


def set_test_args(args):
    set_common_args(args)
