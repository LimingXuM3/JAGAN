import os
import sys
import pprint
import random
import time
import tqdm
import logging
import argparse
import numpy as np
import os.path as path
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist

import losses
import model
import datasets
import lib.utils as utils
from lib.utils import AverageMeter
from optimizer.optimizer import Optimizer
from evaluation.evaler import Evaler
from scorer.scorer import Scorer
from lib.config import cfg, cfg_from_file

class Tester(object):
    def __init__(self, args):
        super(Tester, self).__init__()
        self.args = args
        self.device = torch.device("cuda")

        self.setup_logging()
        self.setup_network()
        self.evaler = Evaler(args,
            eval_ids = cfg.DATA_LOADER.TEST_ID,
            gv_feat = cfg.DATA_LOADER.TEST_GV_FEAT,
            att_feats = cfg.DATA_LOADER.TEST_ATT_FEATS,
            eval_annfile = cfg.INFERENCE.TEST_ANNFILE
        )

    def setup_logging(self):
        self.logger = logging.getLogger(cfg.LOGGER_NAME)
        self.logger.setLevel(logging.INFO)
        
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter("[%(levelname)s: %(asctime)s] %(message)s")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        if not os.path.exists(cfg.ROOT_DIR):
            os.makedirs(cfg.ROOT_DIR)
        
        fh = logging.FileHandler(os.path.join(cfg.ROOT_DIR, cfg.LOGGER_NAME + '.txt'))
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        self.logger.addHandler(fh)

    def setup_network(self):
        generator=model.create(cfg.MODEL.TYPE)
        self.generator = generator.to(self.device)
        checkpoint_path = args.storage + '/ckpts/' + args.dataset + '/gen/' + args.checkpoint_filename
        if path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.generator.load_state_dict(checkpoint['gen_state_dict'])
            self.generator.to(self.device)
        
        
    def eval(self, epoch):
        res = self.evaler(self.generator, 'test_' + str(epoch))
        self.logger.info('######## Epoch ' + str(epoch) + ' ########')
        self.logger.info(str(res))
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image Captioning')
    parser.add_argument('--folder', dest='folder', type=str, default='./')
    #parser.add_argument("--resume", type=int, default=-1)
    parser.add_argument('--checkpoint-filename', type=str,default="TRAIN_PG_GEN_ROLLOUT_0_G-STEPS_1_D-STEPS_1_35_0_36.pth")
    parser.add_argument('--storage', type=str, default='.')
    parser.add_argument('--dataset', type=str, default='LGK')
    parser.add_argument("--result-folder", type=str, default='./stats/LGK/gen/pg/result')
    #parser.add_argument('--dataset', type=str, default='CXR')
    args = parser.parse_args()
    #print('Called with args:')
    #print(args)

    cfg_from_file(os.path.join(args.folder, 'configpg.yml'))
    cfg.ROOT_DIR = args.folder

    tester = Tester(args)
    tester.eval(50)
