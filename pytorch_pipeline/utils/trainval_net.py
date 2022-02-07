# --------------------------------------------------------
# Pytorch FPN
# Licensed under The MIT License [see LICENSE for details]
# Written by Jianwei Yang, based on code from faster R-CNN
# --------------------------------------------------------

# import _init_paths  # since no intent to build the cell_is as a site package and thus to use it as a whole
import os.path

import numpy as np
import argparse
import pprint
import pdb
import time

import torch
from torch.autograd import Variable
import torch.nn as nn

from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import sampler, roibatchLoader
from model.utils.config import cfg, cfg_from_file, cfg_from_list
#from model.utils.net_utils import adjust_learning_rate, save_checkpoint
from utils.data_loader_tests import print_out_info_of_dl

# "tensorboardX" is an alternative to the official one "torch.utils.tensorboard", which BTY still requires installation
from tensorboardX import SummaryWriter
from model.utils.summary import *
import pdb


try:
    xrange  # Python 2
except NameError:
    xrange = range  # Python 3


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Train an instance segmentation network')
    # exp_name is a pos. argument that can be replaced by the user-defined name
    parser.add_argument('exp_name', type=str, default=None, help='experiment name')
    parser.add_argument('--isAnchor_based', default=True, dest='isAnchor_based',
                        help='choose anchor-based or anchor-free')
    parser.add_argument('--dataset', dest='dataset',
                        help='training dataset',
                        default='livecell', type=str)
    parser.add_argument('--net', dest='net',
                        help='detnet59, etc',
                        default='detnet59', type=str)
    parser.add_argument('--start_epoch', dest='start_epoch',
                        help='starting epoch',
                        default=1, type=int)
    parser.add_argument('--epochs', dest='max_epochs',
                        help='number of epochs to train',
                        default=20, type=int)
    parser.add_argument('--disp_interval', dest='disp_interval',
                        help='number of iterations to display',
                        default=100, type=int)
    parser.add_argument('--checkpoint_interval', dest='checkpoint_interval',
                        help='number of iterations to display',
                        default=10000, type=int)

    parser.add_argument('--save_dir', dest='save_dir',
                        help='directory to save models', default="../results/models", )
    parser.add_argument('--nw', dest='num_workers',
                        help='number of worker to load data',
                        default=1, type=int)
    parser.add_argument('--cuda', dest='cuda',
                        help='whether use CUDA',
                        action='store_true')
    parser.add_argument('--mGPUs', dest='mGPUs',
                        help='whether use multiple GPUs',
                        action='store_true')
    parser.add_argument('--lscale', dest='lscale',
                        help='whether use large scale',
                        action='store_true')
    parser.add_argument('--bs', dest='batch_size',
                        help='batch_size',
                        default=1, type=int)
    parser.add_argument('--vbs', dest='val_batch_size',
                        help='validation_batch_size',
                        default=1, type=int)
    parser.add_argument('--cag', dest='class_agnostic',
                        help='whether perform class_agnostic bbox regression',
                        action='store_true')

    # config optimization
    parser.add_argument('--o', dest='optimizer',
                        help='training optimizer',
                        default="sgd", type=str)
    parser.add_argument('--lr', dest='lr',
                        help='starting learning rate',
                        default=0.001, type=float)
    parser.add_argument('--lr_decay_step', dest='lr_decay_step',
                        help='step to do learning rate decay, unit is epoch',
                        default=5, type=int)
    parser.add_argument('--lr_decay_gamma', dest='lr_decay_gamma',
                        help='learning rate decay ratio',
                        default=0.1, type=float)

    # set training session
    parser.add_argument('--s', dest='session',
                        help='training session',
                        default=1, type=int)

    # resume trained model
    parser.add_argument('--r', dest='resume',
                        help='resume checkpoint or not',
                        default=False, type=bool)
    parser.add_argument('--checksession', dest='checksession',
                        help='checksession to load model',
                        default=1, type=int)
    parser.add_argument('--checkepoch', dest='checkepoch',
                        help='checkepoch to load model',
                        default=1, type=int)
    parser.add_argument('--checkpoint', dest='checkpoint',
                        help='checkpoint to load model',
                        default=0, type=int)
    # log and diaplay
    parser.add_argument('--use_tfboard', dest='use_tfboard',
                        help='whether use tensorflow tensorboard',
                        default=True, type=bool)
    parser.add_argument('--cascade', help='whether use cascade structure', action='store_true')

    args = parser.parse_args()
    return args


def _print(str, logger=None):
    print(str)
    if logger is None:
        return
    logger.info(str)


if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)

    if args.use_tfboard:
        # from model.utils.logger import Logger
        # # Set the logger
        # logger = Logger('./logs')
        writer = SummaryWriter(comment=args.exp_name)

    # to switch to the given dataset
    if args.dataset == "coco":
        args.imdb_name = "coco_2014_train+coco_2014_valminusminival"
        args.imdbval_name = "coco_2017_minival"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "livecell":
        args.imdb_name = "livecell_coco_train"
        args.imdbval_name = "livecell_coco_val"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']
    elif args.dataset == "competition":
        args.imdb_name = "competition_train"
        args.imdbval_name = "competition_val"
        args.set_cfgs = ['FPN_ANCHOR_SCALES', '[32, 64, 128, 256, 512]', 'FPN_FEAT_STRIDES', '[4, 8, 16, 32, 64]',
                         'MAX_NUM_GT_BOXES', '20']

    args.cfg_file = "../cfgs/{}_ls.yml".format(args.net) if args.lscale else "../cfgs/{}.yml".format(args.net)

    # add items defined above to the configuring variable cfg
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    # print('trainval', cfg.POOLING_MODE)
    # print('fpn', get_cfg().POOLING_MODE)

    print('Using config:')
    pprint.pprint(cfg)
    # logging.info(cfg)
    np.random.seed(cfg.RNG_SEED)

    # torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available() and not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    if args.isAnchor_based:
        # train set
        # -- Note: Use validation set and disable the flipped to enable faster loading.
        # TODO (XU) to enable more image augmentations in the future, the load_image_gt() function can be referenced from:
        # https://github.com/matterport/Mask_RCNN/blob/3deaec5d902d16e1daf56b62d5971d428dc920bc/mrcnn/model.py
        cfg.TRAIN.USE_FLIPPED = True
        cfg.USE_GPU_NMS = args.cuda

        imdb, roidb, ratio_list, ratio_index = combined_roidb(args.imdb_name)
        train_size = len(roidb)
        _print('{:d} roidb entries'.format(train_size))
        single_batch_sampler = sampler(train_size, args.batch_size)
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            single_batch_sampler, args.batch_size, drop_last=False
        )
        # In the early code, the args.batch_size did not work as expected, i.e. the single sample as batch was executed
        # no matter how much args.batch_size takes. Hence, the batch_sampler is inserted here for dataloader.
        dataset = roibatchLoader(imdb, roidb, ratio_list, ratio_index, args.batch_size,
                                 imdb.num_classes, training=True)
        dataloader = torch.utils.data.DataLoader(dataset,
                                                 batch_sampler=batch_sampler,
                                                 num_workers=args.num_workers)
        #!! SHAPE of each: (2, 3, h, w), (2,3), (2, 20, 5), (2, h, w, 20), (2,), (2,) if arg.batch_size is set as 2!


        # validation set
        cfg.TRAIN.USE_FLIPPED = False
        print('Start combining roidbs...')
        tic_cb = time.time()
        val_imdb, val_roidb, val_ratio_list, val_ratio_index = combined_roidb(args.imdbval_name)
        print(('Done by combining (t={:0.2f}s)'.format(time.time() - tic_cb)))
        val_size = len(val_roidb)
        _print('{:d} validation roidb entries'.format(val_size))
        val_single_batch_sampler = sampler(val_size, args.val_batch_size)
        val_batch_sampler = torch.utils.data.sampler.BatchSampler(
            val_single_batch_sampler, args.val_batch_size, drop_last=False
        )
        # -- Note: training mode also applies to validation ds since num_gpu <=1, but
        #       - no data augmentation is applied
        #       - only one (not two or more) image per batch (per gpu if any) for validation
        #       - shuffle is not adopted when distributed computing is off
        print('Start buidling val dataset object...')
        tic_ds = time.time()
        val_dataset = roibatchLoader(val_imdb, val_roidb, val_ratio_list, val_ratio_index,
                                     args.val_batch_size, val_imdb.num_classes, training=True)
        print(('Done by val ds (t={:0.2f}s)'.format(time.time() - tic_ds)))
        print('Start buidling val dataloader object...')
        tic_dl = time.time()
        val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                     batch_sampler=val_batch_sampler,
                                                     num_workers=args.num_workers)
        print(('Done by val dl (t={:0.2f}s)'.format(time.time() - tic_dl)))

        # print out the results of dataset and dataloader for validation
        print_out_info_of_dl(val_dataset, val_dataloader, mode='val')

    else:
        # set up the data preprocessing for anchor-free model
        pass

    if args.exp_name is not None:
        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset + '/' + args.exp_name
    else:
        output_dir = args.save_dir + "/" + args.net + "/" + args.dataset
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)